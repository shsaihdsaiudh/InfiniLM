#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../global_state/global_state.hpp"
#include "flash_attn.hpp"
#include "infinicore/tensor.hpp"
#include "paged_attn.hpp"
#include "static_attn.hpp"
#include <memory>
#include <variant>

namespace infinilm::layers::attention {

class AttentionLayer;

namespace backends {

/**
 * @brief Hybrid attention: prefill (and mixed batches) go to FlashAttention-2
 * varlen; pure decode steps reuse FA's paged cache update (BSHD layout) but
 * run the paged-attention decode kernel, which reads the BSHD cache via
 * strides and is faster than FA2's kvcache path at short/medium contexts.
 */
class HybridAttentionImpl {
public:
    HybridAttentionImpl(size_t num_heads,
                        size_t head_size,
                        float scale,
                        size_t num_kv_heads,
                        size_t layer_idx);

    infinicore::Tensor forward(const AttentionLayer &layer,
                               const infinicore::Tensor &query,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value,
                               infinicore::Tensor &kv_cache,
                               const infinilm::global_state::AttentionMetadata &attn_metadata) const;

private:
    std::shared_ptr<FlashAttentionImpl> flash_;
    size_t num_heads_;
    size_t head_size_;
    float scale_;
};

} // namespace backends

using AttentionImpl = std::variant<std::shared_ptr<backends::StaticAttentionImpl>, std::shared_ptr<backends::PagedAttentionImpl>, std::shared_ptr<backends::FlashAttentionImpl>, std::shared_ptr<backends::HybridAttentionImpl>>;

/**
 * @brief Attention layer.
 * This class takes query, key, and value tensors as input.
 * The input tensors can either contain prompt tokens or generation tokens.
 *
 * The class does the following:
 * - Update the KV cache.
 * - Perform (multi-head/multi-query/grouped-query) attention.
 * - Return the output tensor.
 */
class AttentionLayer {
public:
    AttentionLayer(size_t num_heads,
                   size_t head_size,
                   float scale,
                   size_t num_kv_heads,
                   size_t layer_idx,
                   infinicore::Tensor k_scale,
                   infinicore::Tensor v_scale,
                   ::infinilm::backends::AttentionBackend attention_backend);

    infinicore::Tensor forward(infinicore::Tensor &query,
                               infinicore::Tensor &key,
                               infinicore::Tensor &value) const;

    inline infinicore::Tensor get_k_scale() const { return k_scale_; }
    inline infinicore::Tensor get_v_scale() const { return v_scale_; }

private:
    infinicore::Tensor k_scale_;
    infinicore::Tensor v_scale_;
    size_t layer_idx_;
    AttentionImpl attn_backend_impl_;
    ::infinilm::backends::AttentionBackend attn_backend_;
};
} // namespace infinilm::layers::attention
