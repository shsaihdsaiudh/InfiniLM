#pragma once

#include "../../config/model_config.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <optional>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4HCACompressor;
struct DeepseekV4HCAState;
class DeepseekV4CSACompressor;
struct DeepseekV4CSAState;

class DeepseekV4Linear : public infinicore::nn::Module {
public:
    DeepseekV4Linear(size_t in_features,
                     size_t out_features,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input) const;

protected:
    INFINICORE_NN_PARAMETER(weight);

private:
    size_t in_features_;
    size_t out_features_;
};

class DeepseekV4RMSNorm : public infinicore::nn::Module {
public:
    DeepseekV4RMSNorm(size_t hidden_size,
                      double eps,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input) const;

protected:
    INFINICORE_NN_PARAMETER(weight);

private:
    size_t hidden_size_;
    double eps_;
    infinicore::Tensor norm_weight_f32_;
};

class DeepseekV4GroupedLinear : public infinicore::nn::Module {
public:
    DeepseekV4GroupedLinear(size_t in_features_per_group,
                            size_t out_features_per_group,
                            size_t num_groups,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input) const;

protected:
    INFINICORE_NN_PARAMETER(weight);

private:
    size_t in_features_per_group_;
    size_t out_features_per_group_;
    size_t num_groups_;
};

struct DeepseekV4AttentionOutput {
    infinicore::Tensor output;
    infinicore::Tensor attention_weights;
};

struct DeepseekV4AttentionProjections {
    // Weighted query LoRA residual consumed by the CSA Lightning Indexer.
    infinicore::Tensor q_residual;
    // Query is [batch, sequence, heads, head_dim]. KV is the shared single
    // head in [batch, sequence, 1, head_dim]. Both already include RoPE.
    infinicore::Tensor query;
    infinicore::Tensor kv;
};

struct DeepseekV4SlidingAttentionOutput {
    infinicore::Tensor output;
    infinicore::Tensor attention_weights;
    // RoPE-applied shared KV entries retained for the next call. Its sequence
    // dimension is capped at sliding_window - 1, matching Transformers'
    // DynamicSlidingWindowLayer semantics.
    infinicore::Tensor kv_cache;
};

// DeepSeek-V4 attention core shared by dense, sliding-window and HCA paths.
// Cache ownership stays with the caller so prefill and decode can use the same
// projection, partial-RoPE, sink and grouped-output implementation.
class DeepseekV4Attention : public infinicore::nn::Module {
public:
    DeepseekV4Attention(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device);

    DeepseekV4Attention(size_t hidden_size,
                        size_t q_lora_rank,
                        size_t num_attention_heads,
                        size_t head_dim,
                        size_t rope_head_dim,
                        size_t o_groups,
                        size_t o_lora_rank,
                        double rms_norm_eps,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device,
                        size_t hca_compress_rate = 0,
                        size_t csa_compress_rate = 0,
                        size_t index_num_heads = 0,
                        size_t index_head_dim = 0,
                        size_t index_topk = 0);

    // hidden_states: [batch, sequence, hidden_size]
    // cos/sin:       [batch, sequence, rope_head_dim / 2]
    DeepseekV4AttentionOutput
    forward(const infinicore::Tensor &hidden_states,
            const infinicore::Tensor &cos,
            const infinicore::Tensor &sin) const;

    DeepseekV4AttentionProjections
    project_qkv(const infinicore::Tensor &hidden_states,
                const infinicore::Tensor &cos,
                const infinicore::Tensor &sin) const;

    DeepseekV4SlidingAttentionOutput
    forward_sliding(const infinicore::Tensor &hidden_states,
                    const infinicore::Tensor &cos,
                    const infinicore::Tensor &sin,
                    const std::optional<infinicore::Tensor> &past_kv,
                    size_t sliding_window) const;

    DeepseekV4SlidingAttentionOutput
    forward_hca(const infinicore::Tensor &hidden_states,
                const infinicore::Tensor &query_cos,
                const infinicore::Tensor &query_sin,
                const infinicore::Tensor &compressed_cos,
                const infinicore::Tensor &compressed_sin,
                const infinicore::Tensor &position_ids,
                const std::optional<infinicore::Tensor> &past_sliding_kv,
                DeepseekV4HCAState *hca_state,
                size_t sliding_window) const;

    DeepseekV4SlidingAttentionOutput
    forward_csa(const infinicore::Tensor &hidden_states,
                const infinicore::Tensor &query_cos,
                const infinicore::Tensor &query_sin,
                const infinicore::Tensor &compressed_cos,
                const infinicore::Tensor &compressed_sin,
                const infinicore::Tensor &position_ids,
                const std::optional<infinicore::Tensor> &past_sliding_kv,
                DeepseekV4CSAState *csa_state,
                size_t sliding_window) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Linear, q_a_proj);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, q_a_norm);
    INFINICORE_NN_MODULE(DeepseekV4Linear, q_b_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, kv_proj);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    INFINICORE_NN_MODULE(DeepseekV4GroupedLinear, o_a_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, o_b_proj);
    INFINICORE_NN_PARAMETER(sinks);
    std::shared_ptr<DeepseekV4HCACompressor> hca_compressor_;
    std::shared_ptr<DeepseekV4CSACompressor> csa_compressor_;

private:
    infinicore::Tensor unweighted_rms_norm_(
        const infinicore::Tensor &input) const;
    infinicore::Tensor apply_partial_rope_(
        const infinicore::Tensor &input,
        const infinicore::Tensor &cos,
        const infinicore::Tensor &sin,
        bool conjugate) const;
    infinicore::Tensor causal_bias_(
        size_t query_length,
        size_t kv_length,
        size_t past_length,
        size_t sliding_window,
        const infinicore::DataType &dtype,
        const infinicore::Device &device) const;
    DeepseekV4AttentionOutput attention_from_projections_(
        const infinicore::Tensor &query,
        const infinicore::Tensor &kv,
        const infinicore::Tensor &cos,
        const infinicore::Tensor &sin,
        size_t past_length,
        size_t sliding_window,
        const std::optional<infinicore::Tensor> &attention_bias =
            std::nullopt) const;

    size_t hidden_size_;
    size_t q_lora_rank_;
    size_t num_attention_heads_;
    size_t head_dim_;
    size_t rope_head_dim_;
    size_t o_groups_;
    size_t o_lora_rank_;
    double rms_norm_eps_;
    infinicore::Tensor q_b_norm_weight_f32_;
};

} // namespace infinilm::models::deepseek_v4
