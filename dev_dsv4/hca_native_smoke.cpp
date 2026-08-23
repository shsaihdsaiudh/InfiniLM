#include "../csrc/models/deepseek_v4/deepseek_v4_hca_compressor.hpp"

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
using infinilm::models::deepseek_v4::DeepseekV4HCACompressor;
using infinilm::models::deepseek_v4::DeepseekV4HCAState;

Tensor to_device(std::vector<float> &values,
                 const Shape &shape,
                 const Device &device,
                 const DataType &dtype = DataType::F32) {
    if (values.empty()) {
        return Tensor::empty(shape, dtype, device);
    }
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
        const float inv_rms = 1.0f
                            / std::sqrt(square_sum / width + eps);
        for (size_t column = 0; column < width; ++column) {
            values[row * width + column]
                *= inv_rms * weight[column];
        }
    }
}

void partial_rope(std::vector<float> &values,
                  const std::vector<float> &cos,
                  const std::vector<float> &sin,
                  size_t entries,
                  size_t head_dim,
                  size_t rope_dim) {
    const size_t nope_dim = head_dim - rope_dim;
    const size_t pairs = rope_dim / 2;
    for (size_t entry = 0; entry < entries; ++entry) {
        for (size_t pair = 0; pair < pairs; ++pair) {
            const size_t offset = entry * head_dim + nope_dim + pair * 2;
            const float x = values[offset];
            const float y = values[offset + 1];
            const float c = cos[entry * pairs + pair];
            const float s = sin[entry * pairs + pair];
            values[offset] = x * c - y * s;
            values[offset + 1] = y * c + x * s;
        }
    }
}

std::vector<float> reference_hca(
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
    auto kv = linear(
        hidden, kv_weight, sequence_length, hidden_size, head_dim);
    auto gate = linear(
        hidden, gate_weight, sequence_length, hidden_size, head_dim);
    std::vector<float> compressed(entries * head_dim, 0.0f);
    for (size_t entry = 0; entry < entries; ++entry) {
        for (size_t feature = 0; feature < head_dim; ++feature) {
            float maximum = -INFINITY;
            for (size_t slot = 0; slot < compress_rate; ++slot) {
                maximum = std::max(
                    maximum,
                    gate[(entry * compress_rate + slot) * head_dim + feature]
                        + position_bias[slot * head_dim + feature]);
            }
            float denominator = 0.0f;
            std::vector<float> weights(compress_rate);
            for (size_t slot = 0; slot < compress_rate; ++slot) {
                weights[slot] = std::exp(
                    gate[(entry * compress_rate + slot) * head_dim + feature]
                    + position_bias[slot * head_dim + feature] - maximum);
                denominator += weights[slot];
            }
            for (size_t slot = 0; slot < compress_rate; ++slot) {
                compressed[entry * head_dim + feature]
                    += weights[slot] / denominator
                     * kv[(entry * compress_rate + slot) * head_dim + feature];
            }
        }
    }
    rms_norm(compressed, norm_weight, entries, head_dim, eps);
    partial_rope(compressed, cos, sin, entries, head_dim, rope_dim);
    return compressed;
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
    constexpr size_t head_dim = 4;
    constexpr size_t rope_dim = 2;
    constexpr size_t compress_rate = 2;
    constexpr size_t entries = sequence_length / compress_rate;
    constexpr float eps = 1e-5f;
    const Device device(Device::Type::NVIDIA, 0);

    std::vector<float> hidden(sequence_length * hidden_size);
    std::vector<float> kv_weight(head_dim * hidden_size);
    std::vector<float> gate_weight(head_dim * hidden_size);
    std::vector<float> position_bias(compress_rate * head_dim);
    std::vector<float> norm_weight(head_dim);
    std::vector<float> cos(entries * rope_dim / 2);
    std::vector<float> sin(entries * rope_dim / 2);
    std::vector<int64_t> positions(sequence_length);
    fill_wave(hidden, 0.67f, 0.19f);
    fill_wave(kv_weight, 0.28f, 0.13f);
    fill_wave(gate_weight, 0.24f, 0.23f);
    fill_wave(position_bias, 0.17f, 0.31f);
    for (size_t i = 0; i < norm_weight.size(); ++i) {
        norm_weight[i] = 0.91f + 0.06f * static_cast<float>(i);
    }
    for (size_t entry = 0; entry < entries; ++entry) {
        const size_t position = entry * compress_rate;
        const float angle = 0.21f * static_cast<float>(position + 1);
        cos[entry] = std::cos(angle);
        sin[entry] = std::sin(angle);
    }
    for (size_t token = 0; token < sequence_length; ++token) {
        positions[token] = static_cast<int64_t>(token);
    }
    const auto reference = reference_hca(
        hidden,
        kv_weight,
        gate_weight,
        position_bias,
        norm_weight,
        cos,
        sin,
        sequence_length,
        hidden_size,
        head_dim,
        rope_dim,
        compress_rate,
        eps);

    auto run = [&](const DataType &dtype,
                   const std::string &suffix,
                   float tolerance) {
        DeepseekV4HCACompressor compressor(
            hidden_size,
            head_dim,
            rope_dim,
            compress_rate,
            eps,
            dtype,
            device);
        std::unordered_map<std::string, Tensor> parameters{
            {"kv_proj.weight", to_device(kv_weight, {head_dim, hidden_size}, device, dtype)},
            {"gate_proj.weight", to_device(gate_weight, {head_dim, hidden_size}, device, dtype)},
            {"kv_norm.weight", to_device(norm_weight, {head_dim}, device, dtype)},
            {"position_bias", to_device(position_bias, {compress_rate, head_dim}, device, dtype)},
        };
        compressor.load_parameters_no_sync(parameters, true);
        if (compressor.state_dict_keys().size() != parameters.size()) {
            throw std::runtime_error("DeepSeek-V4 HCA state-dict key mismatch");
        }

        auto hidden_device = to_device(
            hidden, {1, sequence_length, hidden_size}, device, dtype);
        auto cos_device = to_device(
            cos, {1, entries, rope_dim / 2}, device);
        auto sin_device = to_device(
            sin, {1, entries, rope_dim / 2}, device);
        auto positions_device = positions_to_device(
            positions, {1, sequence_length}, device);
        auto full = compressor.forward(
            hidden_device,
            cos_device,
            sin_device,
            positions_device,
            nullptr);
        require_close(
            "hca_stateless" + suffix,
            to_host(full.new_compressed_kv),
            reference,
            tolerance);
        if (full.compressed_kv->shape()
            != Shape{1, 1, entries, head_dim}) {
            throw std::runtime_error("DeepSeek-V4 HCA stateless output shape mismatch");
        }
        if (!full.block_bias.has_value()) {
            throw std::runtime_error("DeepSeek-V4 HCA block bias is missing");
        }
        std::vector<float> expected_bias(
            sequence_length * entries, 0.0f);
        for (size_t token = 0; token < sequence_length; ++token) {
            const size_t threshold = (token + 1) / compress_rate;
            for (size_t entry = threshold; entry < entries; ++entry) {
                expected_bias[token * entries + entry] = -INFINITY;
            }
        }
        const auto actual_bias = to_host(full.block_bias.value());
        for (size_t i = 0; i < actual_bias.size(); ++i) {
            if (std::isinf(expected_bias[i])) {
                if (!std::isinf(actual_bias[i]) || actual_bias[i] > 0.0f) {
                    throw std::runtime_error(
                        "DeepSeek-V4 HCA block bias masking mismatch");
                }
            } else if (actual_bias[i] != expected_bias[i]) {
                throw std::runtime_error(
                    "DeepSeek-V4 HCA block bias value mismatch");
            }
        }

        DeepseekV4HCAState state;
        const std::vector<size_t> chunk_lengths{1, 2, 2};
        size_t token_offset = 0;
        size_t rope_offset = 0;
        for (size_t chunk_length : chunk_lengths) {
            const size_t buffered = state.buffer_kv
                ? state.buffer_kv->size(1)
                : 0;
            const size_t new_entries =
                (buffered + chunk_length) / compress_rate;
            auto hidden_chunk = hidden_device
                                    ->narrow({{1, token_offset, chunk_length}})
                                    ->contiguous();
            auto cos_chunk = cos_device
                                 ->narrow({{1, rope_offset, new_entries}})
                                 ->contiguous();
            auto sin_chunk = sin_device
                                 ->narrow({{1, rope_offset, new_entries}})
                                 ->contiguous();
            auto positions_chunk = positions_device
                                       ->narrow(
                                           {{1, token_offset, chunk_length}})
                                       ->contiguous();
            auto incremental = compressor.forward(
                hidden_chunk,
                cos_chunk,
                sin_chunk,
                positions_chunk,
                &state);
            token_offset += chunk_length;
            rope_offset += new_entries;
            if (incremental.new_compressed_kv->size(1) != new_entries) {
                throw std::runtime_error(
                    "DeepSeek-V4 HCA emitted the wrong number of entries");
            }
        }
        if (state.entry_count != entries || !state.buffer_kv
            || state.buffer_kv->size(1) != sequence_length % compress_rate
            || !state.buffer_gate
            || state.buffer_gate->size(1) != sequence_length % compress_rate) {
            throw std::runtime_error("DeepSeek-V4 HCA cache state mismatch");
        }
        require_close(
            "hca_incremental" + suffix,
            to_host(state.compressed_kv),
            to_host(full.new_compressed_kv),
            tolerance);
    };

    run(DataType::F32, "_f32", 5e-3f);
    run(DataType::BF16, "_bf16", 4e-2f);
    std::cout << "DeepSeek-V4 native HCA compressor smoke passed\n";
    return 0;
}
