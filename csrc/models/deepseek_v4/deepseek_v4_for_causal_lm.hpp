#pragma once

#include "deepseek_v4_decoder_layer.hpp"
#include "deepseek_v4_hyper_connection.hpp"

#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"

#include <infinicore/nn/embedding.hpp>
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Model : public infinicore::nn::Module {
public:
    DeepseekV4Model(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinilm::InfinilmModel::Input &input) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DeepseekV4DecoderLayer, layers);
    INFINICORE_NN_MODULE(DeepseekV4RMSNorm, norm);
    INFINICORE_NN_MODULE(DeepseekV4HyperHead, hc_head);

private:
    infinicore::Tensor normalize_position_ids_(
        const infinicore::Tensor &position_ids,
        size_t batch_size,
        size_t sequence_length) const;

    size_t hidden_size_;
    size_t hc_mult_;
    size_t rope_dim_;
    double rope_theta_;
    infinicore::DataType dtype_;
    infinicore::Device device_;
};

class DeepseekV4ForCausalLM : public infinilm::InfinilmModel {
public:
    DeepseekV4ForCausalLM(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Model, model);
    INFINICORE_NN_MODULE(
        infinilm::layers::linear::ReplicatedLinear, lm_head);
};

} // namespace infinilm::models::deepseek_v4
