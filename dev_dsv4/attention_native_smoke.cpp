#include "../csrc/models/deepseek_v4/deepseek_v4_attention.hpp"

#include <infinicore/device.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using infinicore::DataType;
using infinicore::Device;
using infinicore::Shape;
using infinicore::Tensor;
using infinilm::models::deepseek_v4::DeepseekV4Attention;

Tensor to_device(std::vector<float> &values,
                 const Shape &shape,
                 const Device &device,
                 const DataType &dtype = DataType::F32) {
    auto cpu = Tensor::from_blob(
        values.data(), shape, DataType::F32, Device::cpu());
    auto result = cpu->to(device);
    if (dtype == DataType::F32) {
        return result;
    }
    auto cast = Tensor::empty(shape, dtype, device);
    infinicore::op::cast_(cast, result);
    return cast;
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

std::vector<float> linear(const std::vector<float> &input,
                          const std::vector<float> &weight,
                          size_t rows,
                          size_t in_features,
                          size_t out_features) {
    std::vector<float> output(rows * out_features, 0.0f);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t out = 0; out < out_features; ++out) {
            for (size_t in = 0; in < in_features; ++in) {
                output[row * out_features + out]
                    += input[row * in_features + in]
                     * weight[out * in_features + in];
            }
        }
    }
    return output;
}

void rms_norm(std::vector<float> &values,
              const std::vector<float> *weight,
              size_t rows,
              size_t width,
              float eps) {
    for (size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        for (size_t column = 0; column < width; ++column) {
            const float value = values[row * width + column];
            square_sum += value * value;
        }
        const float inv_rms = 1.0f
                            / std::sqrt(square_sum / width + eps);
        for (size_t column = 0; column < width; ++column) {
            values[row * width + column] *= inv_rms;
            if (weight != nullptr) {
                values[row * width + column] *= (*weight)[column];
            }
        }
    }
}

void partial_rope(std::vector<float> &values,
                  const std::vector<float> &cos,
                  const std::vector<float> &sin,
                  size_t sequence_length,
                  size_t num_heads,
                  size_t head_dim,
                  size_t rope_dim,
                  bool conjugate) {
    const size_t nope_dim = head_dim - rope_dim;
    const size_t pairs = rope_dim / 2;
    const float sign = conjugate ? -1.0f : 1.0f;
    for (size_t token = 0; token < sequence_length; ++token) {
        for (size_t head = 0; head < num_heads; ++head) {
            const size_t base = (token * num_heads + head) * head_dim;
            for (size_t pair = 0; pair < pairs; ++pair) {
                const size_t offset = base + nope_dim + pair * 2;
                const float x = values[offset];
                const float y = values[offset + 1];
                const float c = cos[token * pairs + pair];
                const float s = sign * sin[token * pairs + pair];
                values[offset] = x * c - y * s;
                values[offset + 1] = y * c + x * s;
            }
        }
    }
}

struct ReferenceOutput {
    std::vector<float> output;
    std::vector<float> attention_weights;
    std::vector<float> query;
    std::vector<float> kv;
};

ReferenceOutput reference_attention(
    const std::vector<float> &hidden,
    const std::vector<float> &cos,
    const std::vector<float> &sin,
    const std::vector<float> &q_a_weight,
    const std::vector<float> &q_a_norm_weight,
    const std::vector<float> &q_b_weight,
    const std::vector<float> &kv_weight,
    const std::vector<float> &kv_norm_weight,
    const std::vector<float> &o_a_weight,
    const std::vector<float> &o_b_weight,
    const std::vector<float> &sinks,
    size_t sequence_length,
    size_t hidden_size,
    size_t q_lora_rank,
    size_t num_heads,
    size_t head_dim,
    size_t rope_dim,
    size_t o_groups,
    size_t o_lora_rank,
    float eps) {
    auto q_residual = linear(
        hidden, q_a_weight, sequence_length, hidden_size, q_lora_rank);
    rms_norm(
        q_residual,
        &q_a_norm_weight,
        sequence_length,
        q_lora_rank,
        eps);
    auto query = linear(
        q_residual,
        q_b_weight,
        sequence_length,
        q_lora_rank,
        num_heads * head_dim);
    rms_norm(
        query, nullptr, sequence_length * num_heads, head_dim, eps);
    partial_rope(
        query,
        cos,
        sin,
        sequence_length,
        num_heads,
        head_dim,
        rope_dim,
        false);

    auto kv = linear(
        hidden, kv_weight, sequence_length, hidden_size, head_dim);
    rms_norm(kv, &kv_norm_weight, sequence_length, head_dim, eps);
    partial_rope(
        kv,
        cos,
        sin,
        sequence_length,
        1,
        head_dim,
        rope_dim,
        false);

    std::vector<float> attention_weights(
        num_heads * sequence_length * sequence_length, 0.0f);
    std::vector<float> attention_output(
        sequence_length * num_heads * head_dim, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t query_token = 0;
             query_token < sequence_length;
             ++query_token) {
            std::vector<float> logits(query_token + 2, sinks[head]);
            for (size_t key_token = 0;
                 key_token <= query_token;
                 ++key_token) {
                float score = 0.0f;
                for (size_t feature = 0; feature < head_dim; ++feature) {
                    score += query[
                                 (query_token * num_heads + head) * head_dim
                                 + feature]
                           * kv[key_token * head_dim + feature];
                }
                logits[key_token] = score * scale;
            }
            float maximum = *std::max_element(logits.begin(), logits.end());
            float denominator = 0.0f;
            for (float &logit : logits) {
                logit = std::exp(logit - maximum);
                denominator += logit;
            }
            for (size_t key_token = 0;
                 key_token <= query_token;
                 ++key_token) {
                const float probability = logits[key_token] / denominator;
                attention_weights[
                    (head * sequence_length + query_token)
                    * sequence_length + key_token] = probability;
                for (size_t feature = 0; feature < head_dim; ++feature) {
                    attention_output[
                        (query_token * num_heads + head) * head_dim + feature]
                        += probability * kv[key_token * head_dim + feature];
                }
            }
        }
    }

    partial_rope(
        attention_output,
        cos,
        sin,
        sequence_length,
        num_heads,
        head_dim,
        rope_dim,
        true);

    const size_t group_input = num_heads * head_dim / o_groups;
    std::vector<float> grouped(
        sequence_length * o_groups * o_lora_rank, 0.0f);
    for (size_t token = 0; token < sequence_length; ++token) {
        for (size_t group = 0; group < o_groups; ++group) {
            for (size_t out = 0; out < o_lora_rank; ++out) {
                for (size_t in = 0; in < group_input; ++in) {
                    grouped[(token * o_groups + group) * o_lora_rank + out]
                        += attention_output[
                               token * num_heads * head_dim
                               + group * group_input + in]
                         * o_a_weight[
                               (group * o_lora_rank + out) * group_input + in];
                }
            }
        }
    }
    return {
        linear(
            grouped,
            o_b_weight,
            sequence_length,
            o_groups * o_lora_rank,
            hidden_size),
        attention_weights,
        query,
        kv};
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

void require_close(const std::string &name,
                   const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   float tolerance) {
    const float difference = max_abs_diff(actual, expected);
    std::cout << name << " max|diff|=" << difference << '\n';
    if (difference > tolerance) {
        for (size_t i = 0; i < actual.size(); ++i) {
            std::cout << "  [" << i << "] actual=" << actual[i]
                      << " expected=" << expected[i] << '\n';
        }
        throw std::runtime_error(name + " exceeded tolerance");
    }
}

void fill_wave(std::vector<float> &values, float scale, float phase) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(static_cast<float>(i + 1) * phase) * scale;
    }
}

} // namespace

int main() {
    constexpr size_t sequence_length = 3;
    constexpr size_t hidden_size = 8;
    constexpr size_t q_lora_rank = 4;
    constexpr size_t num_heads = 2;
    constexpr size_t head_dim = 4;
    constexpr size_t rope_dim = 2;
    constexpr size_t o_groups = 2;
    constexpr size_t o_lora_rank = 3;
    constexpr float eps = 1e-5f;
    constexpr size_t group_input = num_heads * head_dim / o_groups;
    const Device device(Device::Type::NVIDIA, 0);

    std::vector<float> hidden(sequence_length * hidden_size);
    std::vector<float> cos(sequence_length * rope_dim / 2);
    std::vector<float> sin(sequence_length * rope_dim / 2);
    std::vector<float> q_a_weight(q_lora_rank * hidden_size);
    std::vector<float> q_a_norm_weight(q_lora_rank);
    std::vector<float> q_b_weight(num_heads * head_dim * q_lora_rank);
    std::vector<float> kv_weight(head_dim * hidden_size);
    std::vector<float> kv_norm_weight(head_dim);
    std::vector<float> o_a_weight(o_groups * o_lora_rank * group_input);
    std::vector<float> o_b_weight(hidden_size * o_groups * o_lora_rank);
    std::vector<float> sinks(num_heads);

    fill_wave(hidden, 0.7f, 0.23f);
    fill_wave(q_a_weight, 0.31f, 0.17f);
    fill_wave(q_b_weight, 0.27f, 0.11f);
    fill_wave(kv_weight, 0.29f, 0.19f);
    fill_wave(o_a_weight, 0.25f, 0.13f);
    fill_wave(o_b_weight, 0.21f, 0.07f);
    for (size_t i = 0; i < q_a_norm_weight.size(); ++i) {
        q_a_norm_weight[i] = 0.9f + 0.07f * static_cast<float>(i);
    }
    for (size_t i = 0; i < kv_norm_weight.size(); ++i) {
        kv_norm_weight[i] = 1.1f - 0.05f * static_cast<float>(i);
    }
    for (size_t token = 0; token < sequence_length; ++token) {
        const float angle = 0.31f * static_cast<float>(token + 1);
        cos[token] = std::cos(angle);
        sin[token] = std::sin(angle);
    }
    sinks = {-0.35f, 0.42f};

    const auto reference = reference_attention(
        hidden,
        cos,
        sin,
        q_a_weight,
        q_a_norm_weight,
        q_b_weight,
        kv_weight,
        kv_norm_weight,
        o_a_weight,
        o_b_weight,
        sinks,
        sequence_length,
        hidden_size,
        q_lora_rank,
        num_heads,
        head_dim,
        rope_dim,
        o_groups,
        o_lora_rank,
        eps);

    auto run = [&](const DataType &dtype,
                   const std::string &suffix,
                   float projection_tolerance,
                   float output_tolerance,
                   float weight_tolerance) {
        DeepseekV4Attention attention(
            hidden_size,
            q_lora_rank,
            num_heads,
            head_dim,
            rope_dim,
            o_groups,
            o_lora_rank,
            eps,
            dtype,
            device);
        std::unordered_map<std::string, Tensor> parameters{
            {"q_a_proj.weight", to_device(q_a_weight, {q_lora_rank, hidden_size}, device, dtype)},
            {"q_a_norm.weight", to_device(q_a_norm_weight, {q_lora_rank}, device, dtype)},
            {"q_b_proj.weight", to_device(q_b_weight, {num_heads * head_dim, q_lora_rank}, device, dtype)},
            {"kv_proj.weight", to_device(kv_weight, {head_dim, hidden_size}, device, dtype)},
            {"kv_norm.weight", to_device(kv_norm_weight, {head_dim}, device, dtype)},
            {"o_a_proj.weight", to_device(o_a_weight, {o_groups * o_lora_rank, group_input}, device, dtype)},
            {"o_b_proj.weight", to_device(o_b_weight, {hidden_size, o_groups * o_lora_rank}, device, dtype)},
            {"sinks", to_device(sinks, {num_heads}, device, dtype)},
        };
        attention.load_parameters_no_sync(parameters, true);
        if (attention.state_dict_keys().size() != parameters.size()) {
            throw std::runtime_error("DeepSeek-V4 attention state-dict key mismatch");
        }
        auto hidden_device = to_device(
            hidden, {1, sequence_length, hidden_size}, device, dtype);
        auto cos_device = to_device(
            cos, {1, sequence_length, rope_dim / 2}, device);
        auto sin_device = to_device(
            sin, {1, sequence_length, rope_dim / 2}, device);
        auto native = attention.forward(hidden_device, cos_device, sin_device);
        auto projections = attention.project_qkv(
            hidden_device, cos_device, sin_device);
        require_close(
            "query" + suffix,
            to_host(projections.query),
            reference.query,
            projection_tolerance);
        require_close(
            "kv" + suffix,
            to_host(projections.kv),
            reference.kv,
            projection_tolerance);
        require_close(
            "attention_weights" + suffix,
            to_host(native.attention_weights),
            reference.attention_weights,
            weight_tolerance);
        require_close(
            "attention_output" + suffix,
            to_host(native.output),
            reference.output,
            output_tolerance);
    };

    run(DataType::F32, "_f32", 5e-3f, 5e-3f, 2e-3f);
    run(DataType::BF16, "_bf16", 6e-2f, 3e-2f, 1e-2f);
    std::cout << "DeepSeek-V4 native dense attention smoke passed\n";
    return 0;
}
