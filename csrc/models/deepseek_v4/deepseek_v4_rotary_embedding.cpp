#include "deepseek_v4_rotary_embedding.hpp"

#include <infinicore/ops/cast.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

infinicore::Tensor cast_to(const infinicore::Tensor &input,
                           const infinicore::DataType &dtype) {
    if (input->dtype() == dtype) {
        return input;
    }
    auto output = infinicore::Tensor::empty(
        input->shape(), dtype, input->device());
    infinicore::op::cast_(output, input);
    return output;
}

std::vector<int64_t> positions_to_host(
    const infinicore::Tensor &position_ids) {
    if (!position_ids || position_ids->ndim() != 2) {
        throw std::runtime_error(
            "DeepSeek-V4 RoPE position_ids must be [batch, sequence]");
    }
    auto cpu = position_ids->to(infinicore::Device::cpu())->contiguous();
    std::vector<int64_t> positions(cpu->numel());
    if (cpu->dtype() == infinicore::DataType::I64) {
        std::memcpy(positions.data(), cpu->data(), cpu->nbytes());
    } else if (cpu->dtype() == infinicore::DataType::I32) {
        const auto *source =
            reinterpret_cast<const int32_t *>(cpu->data());
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = source[i];
        }
    } else {
        throw std::runtime_error(
            "DeepSeek-V4 RoPE position_ids must use I32 or I64");
    }
    for (const int64_t position : positions) {
        if (position < 0) {
            throw std::runtime_error(
                "DeepSeek-V4 RoPE positions must be non-negative");
        }
    }
    return positions;
}

std::pair<infinicore::Tensor, infinicore::Tensor> build_rotary(
    const std::vector<int64_t> &positions,
    size_t batch_size,
    size_t sequence_length,
    size_t rope_dim,
    double theta,
    const infinicore::DataType &dtype,
    const infinicore::Device &device,
    const std::optional<DeepseekV4YarnScaling> &yarn) {
    if (rope_dim == 0 || rope_dim % 2 != 0 || theta <= 0.0
        || positions.size() != batch_size * sequence_length) {
        throw std::runtime_error(
            "DeepSeek-V4 RoPE configuration or position count is invalid");
    }
    const size_t pair_count = rope_dim / 2;
    std::vector<double> inverse_frequencies(pair_count);
    for (size_t pair = 0; pair < pair_count; ++pair) {
        const double position_frequency =
            std::pow(theta,
                     2.0 * static_cast<double>(pair)
                         / static_cast<double>(rope_dim));
        inverse_frequencies[pair] = 1.0 / position_frequency;
    }
    double attention_factor = 1.0;
    if (yarn.has_value()) {
        const auto &scaling = yarn.value();
        if (scaling.factor <= 0.0 || scaling.beta_fast <= 0.0
            || scaling.beta_slow <= 0.0
            || scaling.original_max_position_embeddings == 0
            || scaling.attention_factor <= 0.0) {
            throw std::runtime_error(
                "DeepSeek-V4 YaRN configuration is invalid");
        }
        const auto correction_dimension = [&](double rotations) {
            const double pi = std::acos(-1.0);
            return static_cast<double>(rope_dim)
                 * std::log(
                       static_cast<double>(
                           scaling.original_max_position_embeddings)
                       / (rotations * 2.0 * pi))
                 / (2.0 * std::log(theta));
        };
        double low = correction_dimension(scaling.beta_fast);
        double high = correction_dimension(scaling.beta_slow);
        if (scaling.truncate) {
            low = std::floor(low);
            high = std::ceil(high);
        }
        low = std::max(low, 0.0);
        high = std::min(
            high, static_cast<double>(rope_dim - 1));
        if (low == high) {
            high += 0.001;
        }
        for (size_t pair = 0; pair < pair_count; ++pair) {
            const double ramp = std::clamp(
                (static_cast<double>(pair) - low) / (high - low),
                0.0,
                1.0);
            const double extrapolated = inverse_frequencies[pair];
            const double interpolated =
                inverse_frequencies[pair] / scaling.factor;
            inverse_frequencies[pair] =
                interpolated * ramp + extrapolated * (1.0 - ramp);
        }
        attention_factor = scaling.attention_factor;
    }
    std::vector<float> cos_values(positions.size() * pair_count);
    std::vector<float> sin_values(positions.size() * pair_count);
    for (size_t token = 0; token < positions.size(); ++token) {
        for (size_t pair = 0; pair < pair_count; ++pair) {
            const double angle =
                static_cast<double>(positions[token])
                * inverse_frequencies[pair];
            cos_values[token * pair_count + pair] =
                static_cast<float>(std::cos(angle) * attention_factor);
            sin_values[token * pair_count + pair] =
                static_cast<float>(std::sin(angle) * attention_factor);
        }
    }
    const infinicore::Shape shape{batch_size, sequence_length, pair_count};
    auto cos_cpu = infinicore::Tensor::from_blob(
        cos_values.data(), shape, infinicore::DataType::F32,
        infinicore::Device::cpu());
    auto sin_cpu = infinicore::Tensor::from_blob(
        sin_values.data(), shape, infinicore::DataType::F32,
        infinicore::Device::cpu());
    return {
        cast_to(cos_cpu->to(device), dtype),
        cast_to(sin_cpu->to(device), dtype)};
}

} // namespace

std::pair<infinicore::Tensor, infinicore::Tensor>
deepseek_v4_rotary_embedding(
    const infinicore::Tensor &position_ids,
    size_t rope_dim,
    double theta,
    const infinicore::DataType &dtype,
    const infinicore::Device &device,
    const std::optional<DeepseekV4YarnScaling> &yarn) {
    return build_rotary(
        positions_to_host(position_ids),
        position_ids->size(0),
        position_ids->size(1),
        rope_dim,
        theta,
        dtype,
        device,
        yarn);
}

std::pair<infinicore::Tensor, infinicore::Tensor>
deepseek_v4_compressed_rotary_embedding(
    size_t batch_size,
    size_t entry_count,
    size_t first_entry,
    size_t compress_rate,
    size_t rope_dim,
    double theta,
    const infinicore::DataType &dtype,
    const infinicore::Device &device,
    const std::optional<DeepseekV4YarnScaling> &yarn) {
    if (compress_rate == 0) {
        throw std::runtime_error(
            "DeepSeek-V4 compressed RoPE rate must be non-zero");
    }
    std::vector<int64_t> positions(batch_size * entry_count);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t entry = 0; entry < entry_count; ++entry) {
            positions[batch * entry_count + entry] =
                static_cast<int64_t>(
                    (first_entry + entry) * compress_rate);
        }
    }
    return build_rotary(
        positions,
        batch_size,
        entry_count,
        rope_dim,
        theta,
        dtype,
        device,
        yarn);
}

} // namespace infinilm::models::deepseek_v4
