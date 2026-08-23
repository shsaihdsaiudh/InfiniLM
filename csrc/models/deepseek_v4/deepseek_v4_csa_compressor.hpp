#pragma once

#include "deepseek_v4_attention.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::deepseek_v4 {

struct DeepseekV4CSAStreamState {
    infinicore::Tensor buffer_kv;
    infinicore::Tensor buffer_gate;
    infinicore::Tensor overlap_kv;
    infinicore::Tensor overlap_gate;
    infinicore::Tensor compressed_kv;
    size_t entry_count{0};
};

struct DeepseekV4CSAState {
    DeepseekV4CSAStreamState compressor;
    DeepseekV4CSAStreamState indexer;
};

struct DeepseekV4CSAOutput {
    // [batch, 1, total_compressed_entries, attention_head_dim].
    infinicore::Tensor compressed_kv;
    // [batch, query_sequence, min(index_topk, compressed_entries)], I32.
    // Invalid causal selections use -1.
    infinicore::Tensor topk_indices;
    // [batch, 1, query_sequence, total_compressed_entries].
    infinicore::Tensor block_bias;
};

class DeepseekV4IndexerScorer : public infinicore::nn::Module {
public:
    DeepseekV4IndexerScorer(size_t hidden_size,
                            size_t num_heads,
                            size_t head_dim,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinicore::Tensor &query,
        const infinicore::Tensor &compressed_kv,
        const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Linear, weights_proj);

private:
    size_t num_heads_;
    size_t head_dim_;
};

class DeepseekV4Indexer : public infinicore::nn::Module {
public:
    DeepseekV4Indexer(size_t hidden_size,
                      size_t q_lora_rank,
                      size_t num_heads,
                      size_t head_dim,
                      size_t rope_head_dim,
                      size_t compress_rate,
                      size_t index_topk,
                      double rms_norm_eps,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    struct Output {
        infinicore::Tensor topk_indices;
        infinicore::Tensor compressed_kv;
    };

    Output forward(const infinicore::Tensor &hidden_states,
                   const infinicore::Tensor &q_residual,
                   const infinicore::Tensor &query_cos,
                   const infinicore::Tensor &query_sin,
                   const infinicore::Tensor &compressed_cos,
                   const infinicore::Tensor &compressed_sin,
                   const infinicore::Tensor &position_ids,
                   DeepseekV4CSAStreamState *state) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Linear, kv_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, gate_proj);
    INFINICORE_NN_PARAMETER(position_bias);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    INFINICORE_NN_MODULE(DeepseekV4Linear, q_b_proj);
    INFINICORE_NN_MODULE(DeepseekV4IndexerScorer, scorer);

private:
    size_t hidden_size_;
    size_t q_lora_rank_;
    size_t num_heads_;
    size_t head_dim_;
    size_t rope_head_dim_;
    size_t compress_rate_;
    size_t index_topk_;
};

class DeepseekV4CSACompressor : public infinicore::nn::Module {
public:
    DeepseekV4CSACompressor(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    DeepseekV4CSACompressor(size_t hidden_size,
                            size_t q_lora_rank,
                            size_t attention_head_dim,
                            size_t rope_head_dim,
                            size_t compress_rate,
                            size_t index_num_heads,
                            size_t index_head_dim,
                            size_t index_topk,
                            double rms_norm_eps,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    DeepseekV4CSAOutput
    forward(const infinicore::Tensor &hidden_states,
            const infinicore::Tensor &q_residual,
            const infinicore::Tensor &query_cos,
            const infinicore::Tensor &query_sin,
            const infinicore::Tensor &compressed_cos,
            const infinicore::Tensor &compressed_sin,
            const infinicore::Tensor &position_ids,
            DeepseekV4CSAState *state = nullptr) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Linear, kv_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, gate_proj);
    INFINICORE_NN_PARAMETER(position_bias);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    INFINICORE_NN_MODULE(DeepseekV4Indexer, indexer);

private:
    size_t hidden_size_;
    size_t q_lora_rank_;
    size_t attention_head_dim_;
    size_t rope_head_dim_;
    size_t compress_rate_;
    size_t index_topk_;
};

} // namespace infinilm::models::deepseek_v4
