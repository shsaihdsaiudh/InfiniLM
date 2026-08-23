#pragma once

#include <infinicore/tensor.hpp>

#include <utility>

namespace infinilm::models::deepseek_v4 {

// Builds the interleaved partial-RoPE values consumed by DeepSeek-V4 attention.
// The returned tensors have shape [batch, sequence, rope_dim / 2].
std::pair<infinicore::Tensor, infinicore::Tensor>
deepseek_v4_rotary_embedding(
    const infinicore::Tensor &position_ids,
    size_t rope_dim,
    double theta,
    const infinicore::DataType &dtype,
    const infinicore::Device &device);

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
    const infinicore::Device &device);

} // namespace infinilm::models::deepseek_v4
