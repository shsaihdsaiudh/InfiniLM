#pragma once

#include "deepseek_v4_attention.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <optional>

namespace infinilm::models::deepseek_v4 {

struct DeepseekV4HCAState {
    infinicore::Tensor buffer_kv;
    infinicore::Tensor buffer_gate;
    infinicore::Tensor compressed_kv;
    size_t entry_count{0};
};

struct DeepseekV4HCAOutput {
    // All emitted long-range entries, matching the official cache return.
    // Shape: [batch, 1, total_compressed_entries, head_dim].
    infinicore::Tensor compressed_kv;
    // Entries emitted by this call before they are appended to the history.
    // Shape: [batch, new_compressed_entries, head_dim].
    infinicore::Tensor new_compressed_kv;
    // [batch, 1, query_sequence, total_compressed_entries]. Absent for
    // single-token decode or when no compressed entry exists.
    std::optional<infinicore::Tensor> block_bias;
};

// DeepSeek-V4 heavily-compressed attention compressor. RoPE cos/sin are
// supplied by the caller for the newly closed windows; their positions are
// state.entry_count * compress_rate + window_index * compress_rate.
class DeepseekV4HCACompressor : public infinicore::nn::Module {
public:
    DeepseekV4HCACompressor(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    DeepseekV4HCACompressor(size_t hidden_size,
                            size_t head_dim,
                            size_t rope_head_dim,
                            size_t compress_rate,
                            double rms_norm_eps,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    // hidden_states: [batch, sequence, hidden_size]
    // cos/sin: [batch, newly_closed_windows, rope_head_dim / 2]
    // state == nullptr performs the official stateless behavior: only complete
    // windows from this call are emitted and the remainder is discarded.
    DeepseekV4HCAOutput
    forward(const infinicore::Tensor &hidden_states,
            const infinicore::Tensor &cos,
            const infinicore::Tensor &sin,
            const infinicore::Tensor &position_ids,
            DeepseekV4HCAState *state = nullptr) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Linear, kv_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, gate_proj);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, kv_norm);
    INFINICORE_NN_PARAMETER(position_bias);

private:
    infinicore::Tensor apply_partial_rope_(
        const infinicore::Tensor &input,
        const infinicore::Tensor &cos,
        const infinicore::Tensor &sin) const;
    std::optional<infinicore::Tensor> make_block_bias_(
        const infinicore::Tensor &position_ids,
        size_t compressed_length,
        const infinicore::DataType &dtype,
        const infinicore::Device &device) const;

    size_t hidden_size_;
    size_t head_dim_;
    size_t rope_head_dim_;
    size_t compress_rate_;
};

} // namespace infinilm::models::deepseek_v4
