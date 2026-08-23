#include "../csrc/models/deepseek_v4/deepseek_v4_moe.hpp"

#include <infinicore/device.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using infinicore::DataType;
using infinicore::Device;
using infinicore::Shape;
using infinicore::Tensor;
using infinilm::models::deepseek_v4::DeepseekV4SparseMoeBlock;

Tensor to_device(std::vector<float> &values,
                 const Shape &shape,
                 const Device &device,
                 const DataType &dtype) {
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

Tensor i64_to_device(std::vector<int64_t> &values,
                     const Shape &shape,
                     const Device &device) {
    return Tensor::from_blob(
               values.data(), shape, DataType::I64, Device::cpu())
        ->to(device);
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
    std::vector<float> result(cpu->numel());
    std::memcpy(result.data(), cpu->data(), cpu->nbytes());
    return result;
}

std::vector<float> linear(const std::vector<float> &input,
                          const float *weight,
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

float sqrt_softplus(float value) {
    return std::sqrt(
        std::max(value, 0.0f) + std::log1p(std::exp(-std::abs(value))));
}

std::vector<float> mlp(const std::vector<float> &input,
                       const float *gate_weight,
                       const float *up_weight,
                       const float *down_weight,
                       size_t rows,
                       size_t hidden_size,
                       size_t intermediate_size,
                       float limit) {
    auto gate = linear(
        input, gate_weight, rows, hidden_size, intermediate_size);
    auto up = linear(
        input, up_weight, rows, hidden_size, intermediate_size);
    std::vector<float> activated(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        const float bounded_gate = std::min(gate[i], limit);
        const float bounded_up = std::clamp(up[i], -limit, limit);
        activated[i] = bounded_gate / (1.0f + std::exp(-bounded_gate))
                     * bounded_up;
    }
    return linear(
        activated,
        down_weight,
        rows,
        intermediate_size,
        hidden_size);
}

struct ReferenceRouting {
    std::vector<int32_t> indices;
    std::vector<float> weights;
};

ReferenceRouting route(const std::vector<float> &hidden,
                       const std::vector<float> &router_weight,
                       const std::vector<float> &correction,
                       const std::vector<int64_t> *tid2eid,
                       const std::vector<int64_t> *input_ids,
                       size_t tokens,
                       size_t hidden_size,
                       size_t num_experts,
                       size_t top_k,
                       float routed_scale) {
    const auto logits = linear(
        hidden,
        router_weight.data(),
        tokens,
        hidden_size,
        num_experts);
    std::vector<float> scores(logits.size());
    std::transform(
        logits.begin(), logits.end(), scores.begin(), sqrt_softplus);
    ReferenceRouting result{
        std::vector<int32_t>(tokens * top_k),
        std::vector<float>(tokens * top_k)};
    std::vector<size_t> order(num_experts);
    for (size_t token = 0; token < tokens; ++token) {
        if (tid2eid != nullptr) {
            for (size_t rank = 0; rank < top_k; ++rank) {
                result.indices[token * top_k + rank] =
                    static_cast<int32_t>(
                        tid2eid->at(
                            input_ids->at(token) * top_k + rank));
            }
        } else {
            for (size_t expert = 0; expert < num_experts; ++expert) {
                order[expert] = expert;
            }
            std::partial_sort(
                order.begin(), order.begin() + top_k, order.end(),
                [&](size_t lhs, size_t rhs) {
                    return scores[token * num_experts + lhs]
                               + correction[lhs]
                         > scores[token * num_experts + rhs]
                               + correction[rhs];
                });
            for (size_t rank = 0; rank < top_k; ++rank) {
                result.indices[token * top_k + rank] =
                    static_cast<int32_t>(order[rank]);
            }
        }
        float denominator = 1e-20f;
        for (size_t rank = 0; rank < top_k; ++rank) {
            denominator += scores[
                token * num_experts
                + result.indices[token * top_k + rank]];
        }
        for (size_t rank = 0; rank < top_k; ++rank) {
            result.weights[token * top_k + rank] =
                scores[token * num_experts
                       + result.indices[token * top_k + rank]]
                / denominator * routed_scale;
        }
    }
    return result;
}

std::vector<float> moe_reference(
    const std::vector<float> &hidden,
    const ReferenceRouting &routing,
    const std::vector<float> &gate_up,
    const std::vector<float> &down,
    const std::vector<float> &shared_gate,
    const std::vector<float> &shared_up,
    const std::vector<float> &shared_down,
    size_t tokens,
    size_t hidden_size,
    size_t intermediate_size,
    size_t top_k,
    float limit) {
    std::vector<float> output(tokens * hidden_size, 0.0f);
    for (size_t token = 0; token < tokens; ++token) {
        std::vector<float> token_input(
            hidden.begin() + token * hidden_size,
            hidden.begin() + (token + 1) * hidden_size);
        for (size_t rank = 0; rank < top_k; ++rank) {
            const size_t expert = static_cast<size_t>(
                routing.indices[token * top_k + rank]);
            const float *expert_gate = gate_up.data()
                + expert * 2 * intermediate_size * hidden_size;
            const float *expert_up =
                expert_gate + intermediate_size * hidden_size;
            const float *expert_down = down.data()
                + expert * hidden_size * intermediate_size;
            const auto contribution = mlp(
                token_input,
                expert_gate,
                expert_up,
                expert_down,
                1,
                hidden_size,
                intermediate_size,
                limit);
            for (size_t feature = 0; feature < hidden_size; ++feature) {
                output[token * hidden_size + feature]
                    += contribution[feature]
                     * routing.weights[token * top_k + rank];
            }
        }
    }
    const auto shared = mlp(
        hidden,
        shared_gate.data(),
        shared_up.data(),
        shared_down.data(),
        tokens,
        hidden_size,
        intermediate_size,
        limit);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] += shared[i];
    }
    return output;
}

void fill_wave(std::vector<float> &values, float scale, float phase) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(static_cast<float>(i + 1) * phase) * scale;
    }
}

void require_close(const std::string &name,
                   const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   float tolerance) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + " comparison size mismatch");
    }
    float difference = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        difference = std::max(
            difference, std::abs(actual[i] - expected[i]));
    }
    std::cout << name << " max|diff|=" << difference << '\n';
    if (difference > tolerance) {
        throw std::runtime_error(name + " exceeded tolerance");
    }
}

std::shared_ptr<infinilm::config::ModelConfig> make_config(
    bool hash,
    const DataType &dtype) {
    nlohmann::json config{
        {"model_type", "deepseek_v4"},
        {"hidden_size", 4},
        {"vocab_size", 8},
        {"num_experts", 3},
        {"num_experts_per_tok", 2},
        {"moe_intermediate_size", 3},
        {"n_shared_experts", 1},
        {"routed_scaling_factor", 1.5},
        {"swiglu_limit", 0.35},
        {"expert_dtype", "dense"},
        {"mlp_layer_types", nlohmann::json::array(
             {hash ? "hash_moe" : "moe"})},
        {"dtype", dtype == DataType::F32 ? "float32" : "bfloat16"},
    };
    return std::make_shared<infinilm::config::ModelConfig>(config);
}

} // namespace

int main() {
    constexpr size_t tokens = 3;
    constexpr size_t hidden_size = 4;
    constexpr size_t num_experts = 3;
    constexpr size_t top_k = 2;
    constexpr size_t intermediate_size = 3;
    constexpr float routed_scale = 1.5f;
    constexpr float limit = 0.35f;
    const Device device(Device::Type::NVIDIA, 0);

    std::vector<float> hidden(tokens * hidden_size);
    std::vector<float> router_weight(num_experts * hidden_size);
    std::vector<float> correction{0.31f, -0.27f, 0.14f};
    std::vector<float> gate_up(
        num_experts * 2 * intermediate_size * hidden_size);
    std::vector<float> down(
        num_experts * hidden_size * intermediate_size);
    std::vector<float> shared_gate(intermediate_size * hidden_size);
    std::vector<float> shared_up(intermediate_size * hidden_size);
    std::vector<float> shared_down(hidden_size * intermediate_size);
    std::vector<int64_t> input_ids{1, 4, 7};
    std::vector<int64_t> tid2eid(8 * top_k);
    fill_wave(hidden, 1.4f, 0.29f);
    fill_wave(router_weight, 0.47f, 0.21f);
    fill_wave(gate_up, 0.58f, 0.17f);
    fill_wave(down, 0.41f, 0.13f);
    fill_wave(shared_gate, 0.52f, 0.23f);
    fill_wave(shared_up, 0.49f, 0.31f);
    fill_wave(shared_down, 0.38f, 0.19f);
    for (size_t token = 0; token < 8; ++token) {
        tid2eid[token * top_k] = static_cast<int64_t>(token % num_experts);
        tid2eid[token * top_k + 1] =
            static_cast<int64_t>((token + 2) % num_experts);
    }

    const auto standard_routing = route(
        hidden,
        router_weight,
        correction,
        nullptr,
        nullptr,
        tokens,
        hidden_size,
        num_experts,
        top_k,
        routed_scale);
    const auto hash_routing = route(
        hidden,
        router_weight,
        {},
        &tid2eid,
        &input_ids,
        tokens,
        hidden_size,
        num_experts,
        top_k,
        routed_scale);
    const auto standard_reference = moe_reference(
        hidden,
        standard_routing,
        gate_up,
        down,
        shared_gate,
        shared_up,
        shared_down,
        tokens,
        hidden_size,
        intermediate_size,
        top_k,
        limit);
    const auto hash_reference = moe_reference(
        hidden,
        hash_routing,
        gate_up,
        down,
        shared_gate,
        shared_up,
        shared_down,
        tokens,
        hidden_size,
        intermediate_size,
        top_k,
        limit);

    auto run = [&](const DataType &dtype,
                   const std::string &suffix,
                   float tolerance) {
        auto hidden_device = to_device(
            hidden, {1, tokens, hidden_size}, device, dtype);
        auto ids_device = i64_to_device(
            input_ids, {1, tokens}, device);
        auto run_block = [&](bool hash,
                             const std::vector<float> &expected,
                             const std::string &name) {
            DeepseekV4SparseMoeBlock block(
                make_config(hash, dtype), 0, device);
            std::unordered_map<std::string, Tensor> parameters{
                {"gate.weight", to_device(router_weight, {num_experts, hidden_size}, device, dtype)},
                {"experts.gate_up_proj", to_device(gate_up, {num_experts, 2 * intermediate_size, hidden_size}, device, dtype)},
                {"experts.down_proj", to_device(down, {num_experts, hidden_size, intermediate_size}, device, dtype)},
                {"shared_experts.gate_proj.weight", to_device(shared_gate, {intermediate_size, hidden_size}, device, dtype)},
                {"shared_experts.up_proj.weight", to_device(shared_up, {intermediate_size, hidden_size}, device, dtype)},
                {"shared_experts.down_proj.weight", to_device(shared_down, {hidden_size, intermediate_size}, device, dtype)},
            };
            if (hash) {
                parameters.emplace(
                    "gate.tid2eid",
                    i64_to_device(tid2eid, {8, top_k}, device));
            } else {
                parameters.emplace(
                    "gate.e_score_correction_bias",
                    to_device(correction, {num_experts}, device, DataType::F32));
            }
            block.load_parameters_no_sync(parameters, true);
            if (block.state_dict_keys().size() != parameters.size()) {
                throw std::runtime_error(
                    "DeepSeek-V4 MoE state-dict key mismatch");
            }
            auto output = block.forward(
                hidden_device,
                hash ? std::optional<Tensor>(ids_device) : std::nullopt);
            require_close(
                name + suffix, to_host(output), expected, tolerance);
        };
        run_block(false, standard_reference, "moe");
        run_block(true, hash_reference, "hash_moe");
    };

    run(DataType::F32, "_f32", 5e-4f);
    run(DataType::BF16, "_bf16", 3e-2f);
    std::cout << "DeepSeek-V4 native MoE smoke passed\n";
    return 0;
}
