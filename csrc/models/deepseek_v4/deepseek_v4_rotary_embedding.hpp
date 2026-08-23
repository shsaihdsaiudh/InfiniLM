#pragma once

#include <infinicore/tensor.hpp>

#include <optional>
#include <utility>

namespace infinilm::models::deepseek_v4 {

struct DeepseekV4YarnScaling {
    double factor;
    double beta_fast;
    double beta_slow;
    size_t original_max_position_embeddings;
    double attention_factor{1.0};
    bool truncate{true};
};

// Builds the interleaved partial-RoPE values consumed by DeepSeek-V4 attention.
// The returned tensors have shape [batch, sequence, rope_dim / 2].
std::pair<infinicore::Tensor, infinicore::Tensor>
deepseek_v4_rotary_embedding(
    const infinicore::Tensor &position_ids,
    size_t rope_dim,
    double theta,
    const infinicore::DataType &dtype,
    const infinicore::Device &device,
    const std::optional<DeepseekV4YarnScaling> &yarn = std::nullopt);

// Compressed entries are positioned at
// (first_entry + i) * compress_rate in the official implementation.
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
    const std::optional<DeepseekV4YarnScaling> &yarn = std::nullopt);

} // namespace infinilm::models::deepseek_v4
