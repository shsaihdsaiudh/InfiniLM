#include "../csrc/models/deepseek_v4/deepseek_v4_hyper_connection.hpp"

#include <infinicore/device.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using infinicore::DataType;
using infinicore::Device;
using infinicore::Shape;
using infinicore::Tensor;
using infinilm::models::deepseek_v4::DeepseekV4HyperConnection;
using infinilm::models::deepseek_v4::DeepseekV4HyperHead;

Tensor to_device(std::vector<float> &values,
                 const Shape &shape,
                 const Device &device,
                 const DataType &dtype = DataType::F32) {
    auto cpu = Tensor::from_blob(
        values.data(), shape, DataType::F32, Device::cpu());
    auto result = cpu->to(device);
    if (dtype != DataType::F32) {
        auto cast = Tensor::empty(shape, dtype, device);
        infinicore::op::cast_(cast, result);
        return cast;
    }
    return result;
}

std::vector<float> to_host(const Tensor &tensor) {
    auto source = tensor;
    if (source->dtype() != DataType::F32) {
        auto cast = Tensor::empty(
            source->shape(), DataType::F32, source->device());
        infinicore::op::cast_(cast, source);
        source = cast;
    }
    auto cpu = source->to(Device::cpu())->contiguous();
    std::vector<float> values(cpu->numel());
    std::memcpy(values.data(), cpu->data(), cpu->nbytes());
    return values;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float max_abs_diff(const std::vector<float> &lhs,
                   const std::vector<float> &rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("comparison size mismatch");
    }
    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result = std::max(result, std::abs(lhs[i] - rhs[i]));
    }
    return result;
}

struct ReferenceOutput {
    std::vector<float> post;
    std::vector<float> comb;
    std::vector<float> collapsed;
};

ReferenceOutput reference_hyper_connection(
    const std::vector<float> &hidden,
    const std::vector<float> &fn,
    const std::vector<float> &base,
    const std::vector<float> &scale,
    size_t tokens,
    size_t hc,
    size_t hidden_size,
    size_t sinkhorn_iters,
    float eps,
    float rms_eps) {
    const size_t flat_size = hc * hidden_size;
    const size_t mix_size = (2 + hc) * hc;
    ReferenceOutput output{
        std::vector<float>(tokens * hc),
        std::vector<float>(tokens * hc * hc),
        std::vector<float>(tokens * hidden_size)};

    for (size_t token = 0; token < tokens; ++token) {
        float square_sum = 0.0f;
        for (size_t i = 0; i < flat_size; ++i) {
            const float value = hidden[token * flat_size + i];
            square_sum += value * value;
        }
        const float inv_rms = 1.0f
                            / std::sqrt(square_sum / flat_size + rms_eps);
        std::vector<float> mix(mix_size, 0.0f);
        for (size_t out = 0; out < mix_size; ++out) {
            for (size_t in = 0; in < flat_size; ++in) {
                mix[out] += fn[out * flat_size + in]
                          * hidden[token * flat_size + in] * inv_rms;
            }
        }

        std::vector<float> pre(hc);
        for (size_t stream = 0; stream < hc; ++stream) {
            pre[stream] = sigmoid(
                mix[stream] * scale[0] + base[stream]) + eps;
            output.post[token * hc + stream] = 2.0f * sigmoid(
                mix[hc + stream] * scale[1] + base[hc + stream]);
        }

        auto *comb = output.comb.data() + token * hc * hc;
        for (size_t row = 0; row < hc; ++row) {
            float max_logit = -INFINITY;
            for (size_t col = 0; col < hc; ++col) {
                const size_t index = row * hc + col;
                comb[index] = mix[2 * hc + index] * scale[2]
                            + base[2 * hc + index];
                max_logit = std::max(max_logit, comb[index]);
            }
            float denominator = 0.0f;
            for (size_t col = 0; col < hc; ++col) {
                const size_t index = row * hc + col;
                comb[index] = std::exp(comb[index] - max_logit);
                denominator += comb[index];
            }
            for (size_t col = 0; col < hc; ++col) {
                comb[row * hc + col] = comb[row * hc + col] / denominator + eps;
            }
        }

        auto normalize_rows = [&]() {
            for (size_t row = 0; row < hc; ++row) {
                float denominator = eps;
                for (size_t col = 0; col < hc; ++col) {
                    denominator += comb[row * hc + col];
                }
                for (size_t col = 0; col < hc; ++col) {
                    comb[row * hc + col] /= denominator;
                }
            }
        };
        auto normalize_columns = [&]() {
            for (size_t col = 0; col < hc; ++col) {
                float denominator = eps;
                for (size_t row = 0; row < hc; ++row) {
                    denominator += comb[row * hc + col];
                }
                for (size_t row = 0; row < hc; ++row) {
                    comb[row * hc + col] /= denominator;
                }
            }
        };

        normalize_columns();
        for (size_t iteration = 1; iteration < sinkhorn_iters; ++iteration) {
            normalize_rows();
            normalize_columns();
        }

        for (size_t feature = 0; feature < hidden_size; ++feature) {
            float value = 0.0f;
            for (size_t stream = 0; stream < hc; ++stream) {
                value += pre[stream]
                       * hidden[token * flat_size + stream * hidden_size + feature];
            }
            output.collapsed[token * hidden_size + feature] = value;
        }
    }
    return output;
}

std::vector<float> reference_apply(
    const std::vector<float> &hidden,
    const std::vector<float> &sublayer,
    const ReferenceOutput &mapping,
    size_t tokens,
    size_t hc,
    size_t hidden_size) {
    std::vector<float> output(hidden.size());
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t target = 0; target < hc; ++target) {
            for (size_t feature = 0; feature < hidden_size; ++feature) {
                float mixed = 0.0f;
                for (size_t source = 0; source < hc; ++source) {
                    mixed += mapping.comb[
                                 token * hc * hc + source * hc + target]
                           * hidden[token * hc * hidden_size
                                    + source * hidden_size + feature];
                }
                output[token * hc * hidden_size + target * hidden_size + feature]
                    = mapping.post[token * hc + target]
                    * sublayer[token * hidden_size + feature]
                    + mixed;
            }
        }
    }
    return output;
}

std::vector<float> reference_hyper_head(
    const std::vector<float> &hidden,
    const std::vector<float> &fn,
    const std::vector<float> &base,
    float scale,
    size_t tokens,
    size_t hc,
    size_t hidden_size,
    float eps,
    float rms_eps) {
    const size_t flat_size = hc * hidden_size;
    std::vector<float> output(tokens * hidden_size);
    for (size_t token = 0; token < tokens; ++token) {
        float square_sum = 0.0f;
        for (size_t i = 0; i < flat_size; ++i) {
            const float value = hidden[token * flat_size + i];
            square_sum += value * value;
        }
        const float inv_rms = 1.0f
                            / std::sqrt(square_sum / flat_size + rms_eps);
        std::vector<float> pre(hc);
        for (size_t stream = 0; stream < hc; ++stream) {
            float mix = 0.0f;
            for (size_t i = 0; i < flat_size; ++i) {
                mix += fn[stream * flat_size + i]
                     * hidden[token * flat_size + i] * inv_rms;
            }
            pre[stream] = sigmoid(mix * scale + base[stream]) + eps;
        }
        for (size_t feature = 0; feature < hidden_size; ++feature) {
            for (size_t stream = 0; stream < hc; ++stream) {
                output[token * hidden_size + feature]
                    += pre[stream]
                     * hidden[token * flat_size + stream * hidden_size + feature];
            }
        }
    }
    return output;
}

void require_close(const std::string &name,
                   const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   float tolerance) {
    const float diff = max_abs_diff(actual, expected);
    std::cout << name << " max|diff|=" << diff << '\n';
    if (diff > tolerance) {
        throw std::runtime_error(name + " exceeded tolerance");
    }
}

} // namespace

int main() {
    constexpr size_t tokens = 2;
    constexpr size_t hc = 2;
    constexpr size_t hidden_size = 4;
    constexpr size_t flat_size = hc * hidden_size;
    constexpr size_t mix_size = (2 + hc) * hc;
    constexpr size_t sinkhorn_iters = 5;
    constexpr float eps = 1e-6f;
    constexpr float rms_eps = 1e-5f;
    const Device device(Device::Type::NVIDIA, 0);

    std::vector<float> hidden(tokens * flat_size);
    std::vector<float> fn(mix_size * flat_size);
    std::vector<float> base(mix_size);
    std::vector<float> scale{0.7f, -0.4f, 1.1f};
    std::vector<float> sublayer(tokens * hidden_size);
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = 0.13f * static_cast<float>(i) - 0.61f;
    }
    for (size_t i = 0; i < fn.size(); ++i) {
        fn[i] = std::sin(static_cast<float>(i + 1)) * 0.09f;
    }
    for (size_t i = 0; i < base.size(); ++i) {
        base[i] = std::cos(static_cast<float>(i + 2)) * 0.12f;
    }
    for (size_t i = 0; i < sublayer.size(); ++i) {
        sublayer[i] = 0.17f * static_cast<float>(i) - 0.28f;
    }

    const auto reference = reference_hyper_connection(
        hidden, fn, base, scale, tokens, hc, hidden_size,
        sinkhorn_iters, eps, rms_eps);
    const auto expected_apply = reference_apply(
        hidden, sublayer, reference, tokens, hc, hidden_size);

    DeepseekV4HyperConnection connection(
        hidden_size, hc, sinkhorn_iters, eps, rms_eps, device);
    connection.load_parameter("fn", to_device(fn, {mix_size, flat_size}, device));
    connection.load_parameter("base", to_device(base, {mix_size}, device));
    connection.load_parameter("scale", to_device(scale, {3}, device));

    auto hidden_device = to_device(
        hidden, {1, tokens, hc, hidden_size}, device);
    auto sublayer_device = to_device(
        sublayer, {1, tokens, hidden_size}, device);
    const auto native = connection.forward(hidden_device);
    require_close("post", to_host(native.post), reference.post, 1e-4f);
    require_close("comb", to_host(native.comb), reference.comb, 1e-4f);
    require_close(
        "collapsed", to_host(native.collapsed), reference.collapsed, 1e-4f);
    require_close(
        "apply",
        to_host(connection.apply(
            hidden_device, sublayer_device, native.post, native.comb)),
        expected_apply,
        1e-4f);

    std::vector<float> head_fn(hc * flat_size);
    std::vector<float> head_base(hc);
    std::vector<float> head_scale{0.8f};
    for (size_t i = 0; i < head_fn.size(); ++i) {
        head_fn[i] = std::cos(static_cast<float>(i + 1)) * 0.07f;
    }
    for (size_t i = 0; i < head_base.size(); ++i) {
        head_base[i] = 0.03f * static_cast<float>(i + 1);
    }
    const auto expected_head = reference_hyper_head(
        hidden, head_fn, head_base, head_scale[0], tokens, hc, hidden_size,
        eps, rms_eps);

    DeepseekV4HyperHead head(hidden_size, hc, eps, rms_eps, device);
    head.load_parameter("hc_fn", to_device(head_fn, {hc, flat_size}, device));
    head.load_parameter("hc_base", to_device(head_base, {hc}, device));
    head.load_parameter("hc_scale", to_device(head_scale, {1}, device));
    require_close("head", to_host(head.forward(hidden_device)), expected_head, 1e-4f);

    if (std::getenv("INFINILM_MHC_TEST_BF16") != nullptr) {
        auto hidden_bf16 = to_device(
            hidden, {1, tokens, hc, hidden_size}, device, DataType::BF16);
        auto sublayer_bf16 = to_device(
            sublayer, {1, tokens, hidden_size}, device, DataType::BF16);
        const auto native_bf16 = connection.forward(hidden_bf16);
        require_close(
            "collapsed_bf16",
            to_host(native_bf16.collapsed),
            reference.collapsed,
            5e-3f);
        require_close(
            "apply_bf16",
            to_host(connection.apply(
                hidden_bf16,
                sublayer_bf16,
                native_bf16.post,
                native_bf16.comb)),
            expected_apply,
            5e-3f);
        require_close(
            "head_bf16",
            to_host(head.forward(hidden_bf16)),
            expected_head,
            5e-3f);
    } else {
        std::cout << "BF16 path skipped; set INFINILM_MHC_TEST_BF16=1 "
                     "with an ATen-enabled InfiniCore build\n";
    }

    std::cout << "DeepSeek-V4 native mHC smoke passed\n";
    return 0;
}
