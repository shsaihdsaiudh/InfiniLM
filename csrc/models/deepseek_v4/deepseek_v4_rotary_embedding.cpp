#include "deepseek_v4_rotary_embedding.hpp"

#include <infinicore/ops/cast.hpp>

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
    const infinicore::Device &device) {
    if (rope_dim == 0 || rope_dim % 2 != 0 || theta <= 0.0
        || positions.size() != batch_size * sequence_length) {
        throw std::runtime_error(
            "DeepSeek-V4 RoPE configuration or position count is invalid");
    }
    const size_t pair_count = rope_dim / 2;
    std::vector<float> cos_values(positions.size() * pair_count);
    std::vector<float> sin_values(positions.size() * pair_count);
    for (size_t token = 0; token < positions.size(); ++token) {
        for (size_t pair = 0; pair < pair_count; ++pair) {
            const double inverse_frequency =
                std::pow(theta,
                         -2.0 * static_cast<double>(pair)
                             / static_cast<double>(rope_dim));
            const double angle =
                static_cast<double>(positions[token]) * inverse_frequency;
            cos_values[token * pair_count + pair] =
                static_cast<float>(std::cos(angle));
            sin_values[token * pair_count + pair] =
                static_cast<float>(std::sin(angle));
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
    const infinicore::Device &device) {
    return build_rotary(
        positions_to_host(position_ids),
        position_ids->size(0),
        position_ids->size(1),
        rope_dim,
        theta,
        dtype,
        device);
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
    const infinicore::Device &device) {
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
        device);
}

} // namespace infinilm::models::deepseek_v4
