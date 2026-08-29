#include "attention_layer.hpp"
#include "infinicore/ops.hpp"

#include <cstdlib>
#include <limits>

namespace infinilm::layers::attention {

AttentionLayer::AttentionLayer(size_t num_heads,
                               size_t head_size,
                               float scale,
                               size_t num_kv_heads,
                               size_t layer_idx,
                               infinicore::Tensor k_scale,
                               infinicore::Tensor v_scale,
                               ::infinilm::backends::AttentionBackend attn_backend) : k_scale_(k_scale), v_scale_(v_scale), layer_idx_(layer_idx), attn_backend_(attn_backend) {
    switch (attn_backend) {
    case ::infinilm::backends::AttentionBackend::STATIC_ATTN:
        attn_backend_impl_ = std::make_shared<backends::StaticAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::PAGED_ATTN:
        attn_backend_impl_ = std::make_shared<backends::PagedAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::FLASH_ATTN:
        attn_backend_impl_ = std::make_shared<backends::FlashAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::HYBRID:
        attn_backend_impl_ = std::make_shared<backends::HybridAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    default:
        throw std::runtime_error("infinilm::layers::attention::AttentionLayer: unsupported attention backend");
    }
}

infinicore::Tensor AttentionLayer::forward(infinicore::Tensor &query,
                                           infinicore::Tensor &key,
                                           infinicore::Tensor &value) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];

    return std::visit(
        [&](auto &impl_ptr) -> infinicore::Tensor {
            return impl_ptr->forward(*this, query, key, value, kv_cache, attn_metadata);
        },
        attn_backend_impl_);
}

} // namespace infinilm::layers::attention

namespace infinilm::layers::attention::backends {

size_t decode_fa_ctx_threshold() {
    static const size_t threshold = []() {
        if (const char *env = std::getenv("INFINILM_DECODE_CTX_THRESHOLD")) {
            const long long v = std::strtoll(env, nullptr, 10);
            return v > 0 ? static_cast<size_t>(v) : std::numeric_limits<size_t>::max();
        }
        const auto &model_config = infinilm::global_state::get_infinilm_config().model_config;
        if (model_config == nullptr) {
            return std::numeric_limits<size_t>::max();
        }
        const size_t hidden = model_config->get_or<size_t>("hidden_size", 0);
        const size_t layers = model_config->get_or<size_t>("num_hidden_layers", 0);
        const size_t kv_heads = model_config->get_or<size_t>("num_key_value_heads", 0);
        // Measured on RTX 5090, bf16, single-request decode (dev_perf
        // --ctx-sweep, v12): splitkv wins below, FA kvcache wins above.
        if (layers == 28 && kv_heads == 8) {
            if (hidden == 1024) { // Qwen3-0.6B: splitkv +9% @3.2k, tie @3.9k
                return static_cast<size_t>(3400);
            }
            if (hidden == 2048) { // Qwen3-1.7B: splitkv +2% @1.3k, -7% @1.9k
                return static_cast<size_t>(1500);
            }
        }
        return std::numeric_limits<size_t>::max();
    }();
    return threshold;
}

HybridAttentionImpl::HybridAttentionImpl(size_t num_heads,
                                         size_t head_size,
                                         float scale,
                                         size_t num_kv_heads,
                                         size_t layer_idx)
    : flash_(std::make_shared<FlashAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx)),
      num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale) {}

infinicore::Tensor HybridAttentionImpl::forward(const AttentionLayer &layer,
                                                const infinicore::Tensor &query,
                                                const infinicore::Tensor &key,
                                                const infinicore::Tensor &value,
                                                infinicore::Tensor &kv_cache,
                                                const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    // Same is_prefill test as the individual impls: in flattened paged mode,
    // pure decode steps have exactly one query token per sequence.
    const size_t seq_len = query->shape()[0];
    const bool is_prefill = (seq_len != attn_metadata.total_sequence_lengths.value()->shape()[0]);
    if (is_prefill) {
        return flash_->forward(layer, query, key, value, kv_cache, attn_metadata);
    }
    // Decode: at long contexts FA's kvcache kernel overtakes the paged
    // splitkv kernel (measured crossover is model-geometry dependent).
    // flash_->forward re-derives the decode case and runs mha_kvcache_ on
    // the same BSHD cache, so routing costs no data movement.
    if (attn_metadata.max_sequence_length > decode_fa_ctx_threshold()) {
        return flash_->forward(layer, query, key, value, kv_cache, attn_metadata);
    }
    // Decode: reuse FA's cache update (BSHD layout + permuted paged_caching_),
    // then run the stride-agnostic paged-attention decode kernel directly.
    auto [k_total, v_total] = flash_->do_kv_cache_update(layer, key, value, kv_cache, attn_metadata.slot_mapping.value());
    const size_t value_head_dim = value->size(value->ndim() - 1);
    auto attn_output = infinicore::Tensor::empty({seq_len, num_heads_, value_head_dim}, query->dtype(), query->device());
    // The paged kernel expects shape [num_blocks, num_kv_heads, block_size, head_dim]
    // (BHSD) but is fully strided, so a permuted view of the BSHD cache works
    // without any data movement.
    infinicore::op::paged_attention_(
        attn_output,
        query,
        k_total->permute({0, 2, 1, 3}),
        v_total->permute({0, 2, 1, 3}),
        attn_metadata.block_tables.value(),
        attn_metadata.total_sequence_lengths.value(),
        std::nullopt,
        scale_);
    return attn_output->view({1, seq_len, num_heads_ * value_head_dim});
}

} // namespace infinilm::layers::attention::backends
