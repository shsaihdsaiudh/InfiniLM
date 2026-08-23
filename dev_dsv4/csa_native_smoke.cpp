#include "../csrc/models/deepseek_v4/deepseek_v4_csa_compressor.hpp"

#include <infinicore/device.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
using infinilm::models::deepseek_v4::DeepseekV4Attention;
using infinilm::models::deepseek_v4::DeepseekV4CSACompressor;
using infinilm::models::deepseek_v4::DeepseekV4CSAState;

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

Tensor positions_to_device(std::vector<int64_t> &values,
                           const Shape &shape,
                           const Device &device) {
    auto cpu = Tensor::from_blob(
        values.data(), shape, DataType::I64, Device::cpu());
    return cpu->to(device);
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

std::vector<int32_t> to_host_i32(const Tensor &tensor) {
    if (tensor->dtype() != DataType::I32) {
        throw std::runtime_error("expected an I32 tensor");
    }
    auto cpu = tensor->to(Device::cpu())->contiguous();
    std::vector<int32_t> values(cpu->numel());
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
              const std::vector<float> &weight,
              size_t rows,
              size_t width,
              float eps) {
    for (size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        for (size_t column = 0; column < width; ++column) {
            const float value = values[row * width + column];
            square_sum += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(square_sum / width + eps);
        for (size_t column = 0; column < width; ++column) {
            values[row * width + column]
                *= inv_rms * weight[column];
        }
    }
}

void partial_rope(std::vector<float> &values,
                  const std::vector<float> &cos,
                  const std::vector<float> &sin,
                  size_t sequence_length,
                  size_t num_heads,
                  size_t head_dim,
                  size_t rope_dim) {
    const size_t nope_dim = head_dim - rope_dim;
    const size_t pairs = rope_dim / 2;
    for (size_t token = 0; token < sequence_length; ++token) {
        for (size_t head = 0; head < num_heads; ++head) {
            for (size_t pair = 0; pair < pairs; ++pair) {
                const size_t offset =
                    (token * num_heads + head) * head_dim
                    + nope_dim + pair * 2;
                const float x = values[offset];
                const float y = values[offset + 1];
                const float c = cos[token * pairs + pair];
                const float s = sin[token * pairs + pair];
                values[offset] = x * c - y * s;
                values[offset + 1] = y * c + x * s;
            }
        }
    }
}

std::vector<float> reference_compress(
    const std::vector<float> &hidden,
    const std::vector<float> &kv_weight,
    const std::vector<float> &gate_weight,
    const std::vector<float> &position_bias,
    const std::vector<float> &norm_weight,
    const std::vector<float> &cos,
    const std::vector<float> &sin,
    size_t sequence_length,
    size_t hidden_size,
    size_t head_dim,
    size_t rope_dim,
    size_t compress_rate,
    float eps) {
    const size_t entries = sequence_length / compress_rate;
    const size_t projected_dim = 2 * head_dim;
    const auto kv = linear(
        hidden, kv_weight, sequence_length, hidden_size, projected_dim);
    const auto gate = linear(
        hidden, gate_weight, sequence_length, hidden_size, projected_dim);
    std::vector<float> compressed(entries * head_dim, 0.0f);
    for (size_t entry = 0; entry < entries; ++entry) {
        for (size_t feature = 0; feature < head_dim; ++feature) {
            std::vector<float> logits;
            std::vector<float> source;
            if (entry > 0) {
                for (size_t slot = 0; slot < compress_rate; ++slot) {
                    const size_t token =
                        (entry - 1) * compress_rate + slot;
                    source.push_back(kv[token * projected_dim + feature]);
                    logits.push_back(
                        gate[token * projected_dim + feature]
                        + position_bias[slot * projected_dim + feature]);
                }
            }
            for (size_t slot = 0; slot < compress_rate; ++slot) {
                const size_t token = entry * compress_rate + slot;
                source.push_back(
                    kv[token * projected_dim + head_dim + feature]);
                logits.push_back(
                    gate[token * projected_dim + head_dim + feature]
                    + position_bias[
                        slot * projected_dim + head_dim + feature]);
            }
            const float maximum =
                *std::max_element(logits.begin(), logits.end());
            float denominator = 0.0f;
            for (float &logit : logits) {
                logit = std::exp(logit - maximum);
                denominator += logit;
            }
            for (size_t slot = 0; slot < logits.size(); ++slot) {
                compressed[entry * head_dim + feature]
                    += logits[slot] / denominator * source[slot];
            }
        }
    }
    rms_norm(compressed, norm_weight, entries, head_dim, eps);
    partial_rope(
        compressed, cos, sin, entries, 1, head_dim, rope_dim);
    return compressed;
}

std::vector<int32_t> reference_topk_one(
    const std::vector<float> &hidden,
    const std::vector<float> &q_residual,
    const std::vector<float> &compressed_index_kv,
    const std::vector<float> &q_weight,
    const std::vector<float> &score_weight,
    const std::vector<float> &query_cos,
    const std::vector<float> &query_sin,
    size_t sequence_length,
    size_t hidden_size,
    size_t q_lora_rank,
    size_t num_heads,
    size_t head_dim,
    size_t rope_dim,
    size_t compress_rate) {
    const size_t entries = sequence_length / compress_rate;
    auto query = linear(
        q_residual,
        q_weight,
        sequence_length,
        q_lora_rank,
        num_heads * head_dim);
    partial_rope(
        query,
        query_cos,
        query_sin,
        sequence_length,
        num_heads,
        head_dim,
        rope_dim);
    const auto weights = linear(
        hidden,
        score_weight,
        sequence_length,
        hidden_size,
        num_heads);
    std::vector<int32_t> result(sequence_length, -1);
    for (size_t token = 0; token < sequence_length; ++token) {
        const size_t visible = std::min(
            entries, (token + 1) / compress_rate);
        float best_score = -std::numeric_limits<float>::infinity();
        for (size_t entry = 0; entry < visible; ++entry) {
            float score = 0.0f;
            for (size_t head = 0; head < num_heads; ++head) {
                float dot = 0.0f;
                for (size_t feature = 0; feature < head_dim; ++feature) {
                    dot += query[(token * num_heads + head) * head_dim
                                 + feature]
                         * compressed_index_kv[entry * head_dim + feature];
                }
                score += weights[token * num_heads + head]
                       / std::sqrt(static_cast<float>(num_heads))
                       * std::max(dot, 0.0f)
                       / std::sqrt(static_cast<float>(head_dim));
            }
            if (score > best_score) {
                best_score = score;
                result[token] = static_cast<int32_t>(entry);
            }
        }
    }
    return result;
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
    constexpr size_t sequence_length = 5;
    constexpr size_t hidden_size = 6;
    constexpr size_t q_lora_rank = 3;
    constexpr size_t attention_heads = 2;
    constexpr size_t attention_head_dim = 4;
    constexpr size_t rope_dim = 2;
    constexpr size_t compress_rate = 2;
    constexpr size_t entries = sequence_length / compress_rate;
    constexpr size_t index_heads = 2;
    constexpr size_t index_head_dim = 4;
    constexpr size_t index_topk = 1;
    constexpr size_t o_groups = 2;
    constexpr size_t o_lora_rank = 2;
    constexpr size_t sliding_window = 2;
    constexpr size_t group_input =
        attention_heads * attention_head_dim / o_groups;
    constexpr float eps = 1e-5f;
    const Device device(Device::Type::NVIDIA, 0);

    std::vector<float> hidden(sequence_length * hidden_size);
    std::vector<float> q_a_weight(q_lora_rank * hidden_size);
    std::vector<float> q_a_norm(q_lora_rank);
    std::vector<float> q_b_weight(
        attention_heads * attention_head_dim * q_lora_rank);
    std::vector<float> sliding_kv_weight(attention_head_dim * hidden_size);
    std::vector<float> sliding_kv_norm(attention_head_dim);
    std::vector<float> o_a_weight(
        o_groups * o_lora_rank * group_input);
    std::vector<float> o_b_weight(
        hidden_size * o_groups * o_lora_rank);
    std::vector<float> sinks(attention_heads);

    std::vector<float> csa_kv_weight(
        2 * attention_head_dim * hidden_size);
    std::vector<float> csa_gate_weight(
        2 * attention_head_dim * hidden_size);
    std::vector<float> csa_position_bias(
        compress_rate * 2 * attention_head_dim);
    std::vector<float> csa_norm(attention_head_dim);
    std::vector<float> index_kv_weight(
        2 * index_head_dim * hidden_size);
    std::vector<float> index_gate_weight(
        2 * index_head_dim * hidden_size);
    std::vector<float> index_position_bias(
        compress_rate * 2 * index_head_dim);
    std::vector<float> index_norm(index_head_dim);
    std::vector<float> index_q_weight(
        index_heads * index_head_dim * q_lora_rank);
    std::vector<float> score_weight(index_heads * hidden_size);
    std::vector<float> query_cos(sequence_length * rope_dim / 2);
    std::vector<float> query_sin(sequence_length * rope_dim / 2);
    std::vector<float> compressed_cos(entries * rope_dim / 2);
    std::vector<float> compressed_sin(entries * rope_dim / 2);
    std::vector<int64_t> positions(sequence_length);

    fill_wave(hidden, 0.61f, 0.17f);
    fill_wave(q_a_weight, 0.22f, 0.13f);
    fill_wave(q_b_weight, 0.19f, 0.11f);
    fill_wave(sliding_kv_weight, 0.25f, 0.07f);
    fill_wave(o_a_weight, 0.20f, 0.29f);
    fill_wave(o_b_weight, 0.18f, 0.09f);
    fill_wave(csa_kv_weight, 0.27f, 0.19f);
    fill_wave(csa_gate_weight, 0.23f, 0.31f);
    fill_wave(csa_position_bias, 0.16f, 0.23f);
    fill_wave(index_kv_weight, 0.26f, 0.15f);
    fill_wave(index_gate_weight, 0.21f, 0.27f);
    fill_wave(index_position_bias, 0.14f, 0.35f);
    fill_wave(index_q_weight, 0.24f, 0.21f);
    fill_wave(score_weight, 0.30f, 0.33f);
    for (size_t i = 0; i < q_a_norm.size(); ++i) {
        q_a_norm[i] = 0.94f + 0.05f * static_cast<float>(i);
    }
    for (size_t i = 0; i < attention_head_dim; ++i) {
        sliding_kv_norm[i] = 1.07f - 0.04f * static_cast<float>(i);
        csa_norm[i] = 0.91f + 0.06f * static_cast<float>(i);
        index_norm[i] = 1.02f - 0.03f * static_cast<float>(i);
    }
    sinks = {-0.27f, 0.36f};
    for (size_t token = 0; token < sequence_length; ++token) {
        positions[token] = static_cast<int64_t>(token);
        const float angle = 0.21f * static_cast<float>(token + 1);
        query_cos[token] = std::cos(angle);
        query_sin[token] = std::sin(angle);
    }
    for (size_t entry = 0; entry < entries; ++entry) {
        const size_t position = entry * compress_rate;
        const float angle = 0.21f * static_cast<float>(position + 1);
        compressed_cos[entry] = std::cos(angle);
        compressed_sin[entry] = std::sin(angle);
    }

    auto q_residual = linear(
        hidden,
        q_a_weight,
        sequence_length,
        hidden_size,
        q_lora_rank);
    rms_norm(
        q_residual,
        q_a_norm,
        sequence_length,
        q_lora_rank,
        eps);
    const auto reference_csa = reference_compress(
        hidden,
        csa_kv_weight,
        csa_gate_weight,
        csa_position_bias,
        csa_norm,
        compressed_cos,
        compressed_sin,
        sequence_length,
        hidden_size,
        attention_head_dim,
        rope_dim,
        compress_rate,
        eps);
    const auto reference_index = reference_compress(
        hidden,
        index_kv_weight,
        index_gate_weight,
        index_position_bias,
        index_norm,
        compressed_cos,
        compressed_sin,
        sequence_length,
        hidden_size,
        index_head_dim,
        rope_dim,
        compress_rate,
        eps);
    const auto reference_indices = reference_topk_one(
        hidden,
        q_residual,
        reference_index,
        index_q_weight,
        score_weight,
        query_cos,
        query_sin,
        sequence_length,
        hidden_size,
        q_lora_rank,
        index_heads,
        index_head_dim,
        rope_dim,
        compress_rate);

    auto run = [&](const DataType &dtype,
                   const std::string &suffix,
                   float tolerance) {
        DeepseekV4CSACompressor compressor(
            hidden_size,
            q_lora_rank,
            attention_head_dim,
            rope_dim,
            compress_rate,
            index_heads,
            index_head_dim,
            index_topk,
            eps,
            dtype,
            device);
        std::unordered_map<std::string, Tensor> compressor_parameters{
            {"kv_proj.weight", to_device(csa_kv_weight, {2 * attention_head_dim, hidden_size}, device, dtype)},
            {"gate_proj.weight", to_device(csa_gate_weight, {2 * attention_head_dim, hidden_size}, device, dtype)},
            {"position_bias", to_device(csa_position_bias, {compress_rate, 2 * attention_head_dim}, device, dtype)},
            {"kv_norm.weight", to_device(csa_norm, {attention_head_dim}, device, dtype)},
            {"indexer.kv_proj.weight", to_device(index_kv_weight, {2 * index_head_dim, hidden_size}, device, dtype)},
            {"indexer.gate_proj.weight", to_device(index_gate_weight, {2 * index_head_dim, hidden_size}, device, dtype)},
            {"indexer.position_bias", to_device(index_position_bias, {compress_rate, 2 * index_head_dim}, device, dtype)},
            {"indexer.kv_norm.weight", to_device(index_norm, {index_head_dim}, device, dtype)},
            {"indexer.q_b_proj.weight", to_device(index_q_weight, {index_heads * index_head_dim, q_lora_rank}, device, dtype)},
            {"indexer.scorer.weights_proj.weight", to_device(score_weight, {index_heads, hidden_size}, device, dtype)},
        };
        compressor.load_parameters_no_sync(compressor_parameters, true);
        if (compressor.state_dict_keys().size()
            != compressor_parameters.size()) {
            throw std::runtime_error(
                "DeepSeek-V4 CSA state-dict key mismatch");
        }

        auto hidden_device = to_device(
            hidden, {1, sequence_length, hidden_size}, device, dtype);
        auto q_residual_device = to_device(
            q_residual, {1, sequence_length, q_lora_rank}, device, dtype);
        auto query_cos_device = to_device(
            query_cos, {1, sequence_length, rope_dim / 2}, device);
        auto query_sin_device = to_device(
            query_sin, {1, sequence_length, rope_dim / 2}, device);
        auto compressed_cos_device = to_device(
            compressed_cos, {1, entries, rope_dim / 2}, device);
        auto compressed_sin_device = to_device(
            compressed_sin, {1, entries, rope_dim / 2}, device);
        auto positions_device = positions_to_device(
            positions, {1, sequence_length}, device);
        auto full = compressor.forward(
            hidden_device,
            q_residual_device,
            query_cos_device,
            query_sin_device,
            compressed_cos_device,
            compressed_sin_device,
            positions_device,
            nullptr);
        require_close(
            "csa_stateless" + suffix,
            to_host(full.compressed_kv),
            reference_csa,
            tolerance);
        const auto actual_indices = to_host_i32(full.topk_indices);
        if (dtype == DataType::F32
            && actual_indices != reference_indices) {
            throw std::runtime_error(
                "DeepSeek-V4 CSA top-k indices mismatch");
        }
        for (size_t token = 0; token < sequence_length; ++token) {
            const size_t visible = std::min(
                entries, (token + 1) / compress_rate);
            if ((visible == 0 && actual_indices[token] != -1)
                || (visible != 0
                    && (actual_indices[token] < 0
                        || static_cast<size_t>(actual_indices[token])
                               >= visible))) {
                throw std::runtime_error(
                    "DeepSeek-V4 CSA top-k causality mismatch");
            }
        }
        const auto bias = to_host(full.block_bias);
        for (size_t token = 0; token < sequence_length; ++token) {
            for (size_t entry = 0; entry < entries; ++entry) {
                const bool selected =
                    actual_indices[token] == static_cast<int32_t>(entry);
                const float value = bias[token * entries + entry];
                if ((selected && value != 0.0f)
                    || (!selected
                        && (!std::isinf(value) || value > 0.0f))) {
                    throw std::runtime_error(
                        "DeepSeek-V4 CSA block bias mismatch");
                }
            }
        }

        DeepseekV4CSAState state;
        const std::vector<size_t> chunks{1, 2, 2};
        size_t token_offset = 0;
        for (size_t chunk_length : chunks) {
            const size_t buffered = state.compressor.buffer_kv
                ? state.compressor.buffer_kv->size(1)
                : 0;
            const size_t new_entries =
                (buffered + chunk_length) / compress_rate;
            const size_t compressed_offset =
                state.compressor.entry_count;
            compressor.forward(
                hidden_device
                    ->narrow({{1, token_offset, chunk_length}})
                    ->contiguous(),
                q_residual_device
                    ->narrow({{1, token_offset, chunk_length}})
                    ->contiguous(),
                query_cos_device
                    ->narrow({{1, token_offset, chunk_length}})
                    ->contiguous(),
                query_sin_device
                    ->narrow({{1, token_offset, chunk_length}})
                    ->contiguous(),
                compressed_cos_device
                    ->narrow({{1, compressed_offset, new_entries}})
                    ->contiguous(),
                compressed_sin_device
                    ->narrow({{1, compressed_offset, new_entries}})
                    ->contiguous(),
                positions_device
                    ->narrow({{1, token_offset, chunk_length}})
                    ->contiguous(),
                &state);
            token_offset += chunk_length;
        }
        require_close(
            "csa_incremental" + suffix,
            to_host(state.compressor.compressed_kv),
            to_host(full.compressed_kv),
            tolerance);
        require_close(
            "indexer_incremental" + suffix,
            to_host(state.indexer.compressed_kv),
            reference_index,
            tolerance);
        if (state.compressor.entry_count != entries
            || state.indexer.entry_count != entries
            || !state.compressor.buffer_kv
            || state.compressor.buffer_kv->size(1)
                   != sequence_length % compress_rate
            || !state.compressor.overlap_kv
            || state.compressor.overlap_kv->size(1) != compress_rate) {
            throw std::runtime_error(
                "DeepSeek-V4 CSA incremental state mismatch");
        }

        DeepseekV4Attention attention(
            hidden_size,
            q_lora_rank,
            attention_heads,
            attention_head_dim,
            rope_dim,
            o_groups,
            o_lora_rank,
            eps,
            dtype,
            device,
            0,
            compress_rate,
            index_heads,
            index_head_dim,
            index_topk);
        std::unordered_map<std::string, Tensor> parameters{
            {"q_a_proj.weight", to_device(q_a_weight, {q_lora_rank, hidden_size}, device, dtype)},
            {"q_a_norm.weight", to_device(q_a_norm, {q_lora_rank}, device, dtype)},
            {"q_b_proj.weight", to_device(q_b_weight, {attention_heads * attention_head_dim, q_lora_rank}, device, dtype)},
            {"kv_proj.weight", to_device(sliding_kv_weight, {attention_head_dim, hidden_size}, device, dtype)},
            {"kv_norm.weight", to_device(sliding_kv_norm, {attention_head_dim}, device, dtype)},
            {"o_a_proj.weight", to_device(o_a_weight, {o_groups * o_lora_rank, group_input}, device, dtype)},
            {"o_b_proj.weight", to_device(o_b_weight, {hidden_size, o_groups * o_lora_rank}, device, dtype)},
            {"sinks", to_device(sinks, {attention_heads}, device, dtype)},
        };
        for (const auto &[name, tensor] : compressor_parameters) {
            parameters.emplace("compressor." + name, tensor);
        }
        attention.load_parameters_no_sync(parameters, true);
        if (attention.state_dict_keys().size() != parameters.size()) {
            throw std::runtime_error(
                "DeepSeek-V4 integrated CSA state-dict key mismatch");
        }
        auto prefill = attention.forward_csa(
            hidden_device,
            query_cos_device,
            query_sin_device,
            compressed_cos_device,
            compressed_sin_device,
            positions_device,
            std::nullopt,
            nullptr,
            sliding_window);

        DeepseekV4CSAState decode_state;
        std::optional<Tensor> sliding_cache;
        std::vector<Tensor> token_outputs;
        for (size_t token = 0; token < sequence_length; ++token) {
            const size_t buffered = decode_state.compressor.buffer_kv
                ? decode_state.compressor.buffer_kv->size(1)
                : 0;
            const size_t new_entries = (buffered + 1) / compress_rate;
            const size_t compressed_offset =
                decode_state.compressor.entry_count;
            auto step = attention.forward_csa(
                hidden_device->narrow({{1, token, 1}})->contiguous(),
                query_cos_device->narrow({{1, token, 1}})->contiguous(),
                query_sin_device->narrow({{1, token, 1}})->contiguous(),
                compressed_cos_device
                    ->narrow({{1, compressed_offset, new_entries}})
                    ->contiguous(),
                compressed_sin_device
                    ->narrow({{1, compressed_offset, new_entries}})
                    ->contiguous(),
                positions_device->narrow({{1, token, 1}})->contiguous(),
                sliding_cache,
                &decode_state,
                sliding_window);
            sliding_cache = step.kv_cache;
            token_outputs.push_back(step.output);
        }
        auto decoded = infinicore::op::cat(token_outputs, 1);
        require_close(
            "csa_prefill_decode" + suffix,
            to_host(decoded),
            to_host(prefill.output),
            tolerance);
    };

    run(DataType::F32, "_f32", 5e-3f);
    run(DataType::BF16, "_bf16", 5e-2f);
    std::cout << "DeepSeek-V4 native CSA/Indexer smoke passed\n";
    return 0;
}
