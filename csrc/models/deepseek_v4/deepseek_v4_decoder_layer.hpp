#pragma once

#include "deepseek_v4_attention.hpp"
#include "deepseek_v4_csa_compressor.hpp"
#include "deepseek_v4_hca_compressor.hpp"
#include "deepseek_v4_hyper_connection.hpp"
#include "deepseek_v4_moe.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <optional>
#include <string>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4DecoderLayer : public infinicore::nn::Module {
public:
    DeepseekV4DecoderLayer(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinicore::Tensor &hidden_streams,
        const infinicore::Tensor &position_ids,
        const infinicore::Tensor &query_cos,
        const infinicore::Tensor &query_sin,
        const std::optional<infinicore::Tensor> &input_ids) const;

    void reset_runtime_state() const override;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Attention, self_attn);
    INFINICORE_NN_MODULE(DeepseekV4SparseMoeBlock, mlp);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(DeepseekV4HyperConnection, attn_hc);
    INFINICORE_NN_MODULE(DeepseekV4HyperConnection, ffn_hc);

private:
    std::pair<infinicore::Tensor, infinicore::Tensor>
    compressed_rotary_(size_t batch_size, size_t sequence_length) const;

    size_t layer_idx_;
    std::string layer_type_;
    size_t sliding_window_;
    size_t compress_rate_{0};
    size_t rope_dim_;
    double compress_rope_theta_;
    infinicore::DataType dtype_;
    infinicore::Device device_;

    mutable std::optional<infinicore::Tensor> sliding_kv_;
    mutable DeepseekV4HCAState hca_state_;
    mutable DeepseekV4CSAState csa_state_;
};

} // namespace infinilm::models::deepseek_v4
