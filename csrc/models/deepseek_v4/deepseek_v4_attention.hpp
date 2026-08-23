#pragma once

#include "../../config/model_config.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <optional>

namespace infinilm::models::deepseek_v4 {

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

// Minimal, stateless DeepSeek-V4 attention core. It intentionally excludes the
// sliding/CSA/HCA caches so the projection, partial-RoPE, sink and grouped
// output math can be validated independently first.
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
                        const infinicore::Device &device);

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

protected:
    INFINICORE_NN_MODULE(DeepseekV4Linear, q_a_proj);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, q_a_norm);
    INFINICORE_NN_MODULE(DeepseekV4Linear, q_b_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, kv_proj);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    INFINICORE_NN_MODULE(DeepseekV4GroupedLinear, o_a_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, o_b_proj);
    INFINICORE_NN_PARAMETER(sinks);

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
        size_t sliding_window) const;

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
