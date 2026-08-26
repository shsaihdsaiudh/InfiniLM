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

// cast_to：转 dtype；相同则直接返回。
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

// positions_to_host：把 position_ids（可能是 I32/I64、GPU 上）拷到 CPU 的 int64 向量。
// 因为后面的角度计算要在 CPU 上循环生成 cos/sin 表。
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

// build_rotary：核心函数——根据位置号生成 RoPE 的 cos/sin 表。
// 输出形状 [batch, sequence, rope_dim/2]（每个位置 rope_dim/2 对 cos/sin）。
// 如果传了 yarn 缩放，会对频率做 YaRN 校正（长距离插值 + 短距离外推的混合）。
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
    // ① 计算逆频率：inv_freq[pair] = 1 / theta^(2*pair/rope_dim)
    //    这是 RoPE 的标准频率公式（theta 默认 10000，压缩层 160000）。
    std::vector<double> inverse_frequencies(pair_count);
    for (size_t pair = 0; pair < pair_count; ++pair) {
        const double position_frequency =
            std::pow(theta,
                     2.0 * static_cast<double>(pair)
                         / static_cast<double>(rope_dim));
        inverse_frequencies[pair] = 1.0 / position_frequency;
    }
    double attention_factor = 1.0;
    // ② YaRN 缩放：如果启用，对频率做"线性插值 + 斜坡外推"混合。
    //    核心思想：短距离（低 pair）用原频率（外推），长距离（高 pair）用缩放后的
    //    频率（除以 factor，即插值），中间用 ramp 平滑过渡。
    //    这是 V4 压缩分支为了支持超长上下文用的（factor=16）。
    if (yarn.has_value()) {
        const auto &scaling = yarn.value();
        if (scaling.factor <= 0.0 || scaling.beta_fast <= 0.0
            || scaling.beta_slow <= 0.0
            || scaling.original_max_position_embeddings == 0
            || scaling.attention_factor <= 0.0) {
            throw std::runtime_error(
                "DeepSeek-V4 YaRN configuration is invalid");
        }
        // 把 beta_fast/beta_slow（转数）换算成"维度位置"（哪个 pair 开始插值）
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
        // 对每个 pair：ramp 在 [low, high] 间从 0→1，
        // 频率 = 插值频率×ramp + 外推频率×(1-ramp)
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
    // ③ 生成 cos/sin 表：angle = position * inv_freq[pair]，
    //    cos_value = cos(angle) * attention_factor
    //    形状 [batch, sequence, pair_count]，每个位置 pair_count 对 cos/sin。
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

// 主 RoPE：根据 token 的 position_ids 生成 cos/sin 表（滑窗层用，theta=10000）。
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

// 压缩分支的 RoPE（CSA/HCA 层用）：压缩条目的位置号不是 token 号，
// 而是"第几个压缩条目 × 压缩率"。所以位置 = (first_entry + entry) * compress_rate，
// 其中 first_entry 是已累积的条目数（增量推理续算），compress_rate 是压缩率。
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
    // 构造压缩条目的位置号：第 entry 个条目对应的 token 位置是 (first_entry+entry)*compress_rate
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
