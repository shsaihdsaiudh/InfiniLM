#include "../csrc/models/deepseek_v4/deepseek_v4_moe.hpp"
#include "../csrc/models/deepseek_v4/deepseek_v4_config.hpp"

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
using infinilm::models::deepseek_v4::DeepseekV4Experts;
using infinilm::models::deepseek_v4::DeepseekV4SparseMoeBlock;
using infinilm::models::deepseek_v4::create_deepseek_v4_model_config;

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

Tensor i32_to_device(std::vector<int32_t> &values,
                     const Shape &shape,
                     const Device &device) {
    return Tensor::from_blob(
               values.data(), shape, DataType::I32, Device::cpu())
        ->to(device);
}

Tensor u8_slice_to_device(std::vector<uint8_t> &values,
                          size_t offset,
                          const Shape &shape,
                          const Device &device) {
    return Tensor::from_blob(
               values.data() + offset,
               shape,
               DataType::U8,
               Device::cpu())
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

float decode_e2m1(uint8_t code) {
    static constexpr float magnitudes[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = magnitudes[code & 0x7];
    return code & 0x8 ? -magnitude : magnitude;
}

float packed_dot(const float *input,
                 const uint8_t *packed,
                 const uint8_t *scales,
                 size_t features) {
    float result = 0.0f;
    for (size_t packed_index = 0;
         packed_index < features / 2;
         ++packed_index) {
        const int exponent =
            static_cast<int>(scales[packed_index / 16]) - 127;
        const uint8_t value = packed[packed_index];
        result += input[2 * packed_index]
                * std::ldexp(decode_e2m1(value & 0xf), exponent);
        result += input[2 * packed_index + 1]
                * std::ldexp(decode_e2m1(value >> 4), exponent);
    }
    return result;
}

std::vector<float> packed_moe_reference(
    const std::vector<float> &hidden,
    const std::vector<int32_t> &selected,
    const std::vector<float> &routing,
    const std::vector<uint8_t> &w13,
    const std::vector<uint8_t> &w13_scale,
    const std::vector<uint8_t> &w2,
    const std::vector<uint8_t> &w2_scale,
    size_t tokens,
    size_t hidden_size,
    size_t intermediate_size,
    size_t top_k,
    float limit,
    bool &clamp_exercised) {
    const size_t w13_packed_row = hidden_size / 2;
    const size_t w13_scale_row = hidden_size / 32;
    const size_t w2_packed_row = intermediate_size / 2;
    const size_t w2_scale_row = intermediate_size / 32;
    std::vector<float> output(tokens * hidden_size, 0.0f);
    std::vector<float> activated(intermediate_size);
    for (size_t token = 0; token < tokens; ++token) {
        const float *token_input = hidden.data() + token * hidden_size;
        for (size_t rank = 0; rank < top_k; ++rank) {
            const size_t route = token * top_k + rank;
            const size_t expert = static_cast<size_t>(selected[route]);
            for (size_t i = 0; i < intermediate_size; ++i) {
                const size_t gate_row =
                    expert * 2 * intermediate_size + i;
                const size_t up_row = gate_row + intermediate_size;
                const float gate = packed_dot(
                    token_input,
                    w13.data() + gate_row * w13_packed_row,
                    w13_scale.data() + gate_row * w13_scale_row,
                    hidden_size);
                const float up = packed_dot(
                    token_input,
                    w13.data() + up_row * w13_packed_row,
                    w13_scale.data() + up_row * w13_scale_row,
                    hidden_size);
                clamp_exercised = clamp_exercised
                    || gate > limit || std::abs(up) > limit;
                const float bounded_gate = std::min(gate, limit);
                const float bounded_up = std::clamp(up, -limit, limit);
                activated[i] =
                    bounded_gate / (1.0f + std::exp(-bounded_gate))
                    * bounded_up;
            }
            for (size_t h = 0; h < hidden_size; ++h) {
                const size_t row = expert * hidden_size + h;
                output[token * hidden_size + h] += routing[route]
                    * packed_dot(
                        activated.data(),
                        w2.data() + row * w2_packed_row,
                        w2_scale.data() + row * w2_scale_row,
                        intermediate_size);
            }
        }
    }
    return output;
}

void run_packed_fp4_smoke(const Device &device) {
    constexpr size_t tokens = 3;
    constexpr size_t hidden_size = 64;
    constexpr size_t intermediate_size = 64;
    constexpr size_t num_experts = 4;
    constexpr size_t top_k = 2;
    constexpr float limit = 10.0f;
    const size_t w13_expert_bytes =
        2 * intermediate_size * hidden_size / 2;
    const size_t w13_expert_scales =
        2 * intermediate_size * hidden_size / 32;
    const size_t w2_expert_bytes =
        hidden_size * intermediate_size / 2;
    const size_t w2_expert_scales =
        hidden_size * intermediate_size / 32;

    std::vector<float> hidden(tokens * hidden_size);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t i = 0; i < hidden_size; ++i) {
            hidden[token * hidden_size + i] =
                static_cast<float>((i % 11) + 1) * 0.02f
                * static_cast<float>(token + 1);
        }
    }
    std::vector<int32_t> selected{0, 1, 2, 3, 1, 3};
    std::vector<float> routing{0.7f, 0.3f, 0.4f, 0.6f, 0.25f, 0.75f};
    std::vector<uint8_t> w13(num_experts * w13_expert_bytes);
    std::vector<uint8_t> w13_scale(
        num_experts * w13_expert_scales, 127);
    std::vector<uint8_t> w2(num_experts * w2_expert_bytes);
    std::vector<uint8_t> w2_scale(
        num_experts * w2_expert_scales, 123);
    for (size_t expert = 0; expert < num_experts; ++expert) {
        const uint8_t gate_code = expert % 2 == 0 ? 0x66 : 0x55;
        std::fill_n(
            w13.begin() + expert * w13_expert_bytes,
            intermediate_size * hidden_size / 2,
            gate_code);
        std::fill_n(
            w13.begin() + expert * w13_expert_bytes
                + intermediate_size * hidden_size / 2,
            intermediate_size * hidden_size / 2,
            static_cast<uint8_t>(expert < 2 ? 0x66 : 0xee));
    }
    for (size_t i = 0; i < w2.size(); ++i) {
        const uint8_t low = static_cast<uint8_t>(1 + (i % 6));
        const uint8_t high = static_cast<uint8_t>(
            1 + ((i + 3) % 6) + ((i / 7) % 3 == 0 ? 8 : 0));
        w2[i] = static_cast<uint8_t>((high << 4) | low);
    }

    bool clamp_exercised = false;
    const auto expected = packed_moe_reference(
        hidden,
        selected,
        routing,
        w13,
        w13_scale,
        w2,
        w2_scale,
        tokens,
        hidden_size,
        intermediate_size,
        top_k,
        limit,
        clamp_exercised);
    if (!clamp_exercised) {
        throw std::runtime_error(
            "packed FP4 smoke did not exercise the SwiGLU limit");
    }

    DeepseekV4Experts experts(
        num_experts,
        hidden_size,
        intermediate_size,
        limit,
        true,
        DataType::F32,
        device);
    std::unordered_map<std::string, Tensor> parameters;
    for (size_t expert = 0; expert < num_experts; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        const size_t w13_byte_offset = expert * w13_expert_bytes;
        const size_t w13_scale_offset = expert * w13_expert_scales;
        parameters.emplace(
            prefix + "w1.weight_packed",
            u8_slice_to_device(
                w13,
                w13_byte_offset,
                {intermediate_size, hidden_size / 2},
                device));
        parameters.emplace(
            prefix + "w1.weight_scale",
            u8_slice_to_device(
                w13_scale,
                w13_scale_offset,
                {intermediate_size, hidden_size / 32},
                device));
        parameters.emplace(
            prefix + "w3.weight_packed",
            u8_slice_to_device(
                w13,
                w13_byte_offset + intermediate_size * hidden_size / 2,
                {intermediate_size, hidden_size / 2},
                device));
        parameters.emplace(
            prefix + "w3.weight_scale",
            u8_slice_to_device(
                w13_scale,
                w13_scale_offset + intermediate_size * hidden_size / 32,
                {intermediate_size, hidden_size / 32},
                device));
        parameters.emplace(
            prefix + "w2.weight_packed",
            u8_slice_to_device(
                w2,
                expert * w2_expert_bytes,
                {hidden_size, intermediate_size / 2},
                device));
        parameters.emplace(
            prefix + "w2.weight_scale",
            u8_slice_to_device(
                w2_scale,
                expert * w2_expert_scales,
                {hidden_size, intermediate_size / 32},
                device));
    }
    experts.load_parameters_no_sync(parameters, true);
    if (experts.state_dict_keys().size() != parameters.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 packed expert state-dict key mismatch");
    }
    auto hidden_device = to_device(
        hidden, {tokens, hidden_size}, device, DataType::F32);
    auto selected_device = i32_to_device(
        selected, {tokens, top_k}, device);
    auto routing_device = to_device(
        routing, {tokens, top_k}, device, DataType::F32);
    require_close(
        "packed_fp4_f32",
        to_host(experts.forward(
            hidden_device, selected_device, routing_device)),
        expected,
        2e-3f);
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

void run_official_config_smoke() {
    std::vector<size_t> compress_ratios{0, 0};
    for (size_t layer = 2; layer < 43; ++layer) {
        compress_ratios.push_back(layer % 2 == 0 ? 4 : 128);
    }
    compress_ratios.insert(compress_ratios.end(), {0, 0, 0});
    nlohmann::json released_config{
        {"model_type", "deepseek_v4"},
        {"torch_dtype", "bfloat16"},
        {"num_hidden_layers", 43},
        {"head_dim", 512},
        {"qk_rope_head_dim", 64},
        {"n_routed_experts", 256},
        {"num_experts_per_tok", 6},
        {"num_hash_layers", 3},
        {"hc_mult", 4},
        {"hc_sinkhorn_iters", 20},
        {"compress_ratios", compress_ratios},
        {"expert_dtype", "fp4"},
    };
    auto normalized = create_deepseek_v4_model_config(
        std::make_shared<infinilm::config::ModelConfig>(released_config));
    const auto &config = normalized->get_config_json();
    const auto &layer_types = config.at("layer_types");
    const auto &mlp_layer_types = config.at("mlp_layer_types");
    if (layer_types.size() != 43
        || layer_types.at(0) != "sliding_attention"
        || layer_types.at(2) != "compressed_sparse_attention"
        || layer_types.at(3) != "heavily_compressed_attention"
        || config.at("compress_rates").at("compressed_sparse_attention") != 4
        || config.at("compress_rates").at("heavily_compressed_attention") != 128
        || mlp_layer_types.at(0) != "hash_moe"
        || mlp_layer_types.at(2) != "hash_moe"
        || mlp_layer_types.at(3) != "moe"
        || config.at("num_experts") != 256
        || config.at("expert_dtype") != "fp4"
        || config.at("dtype") != "bfloat16") {
        throw std::runtime_error(
            "released DeepSeek-V4 config normalization mismatch");
    }
    std::cout << "released_config normalization passed\n";
}

} // namespace

int main() {
    run_official_config_smoke();
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
    run_packed_fp4_smoke(device);
    std::cout << "DeepSeek-V4 native MoE smoke passed\n";
    return 0;
}
