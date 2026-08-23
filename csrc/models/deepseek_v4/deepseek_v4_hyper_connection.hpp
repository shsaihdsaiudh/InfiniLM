#pragma once

#include "../../config/model_config.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::deepseek_v4 {

struct DeepseekV4HyperConnectionOutput {
    infinicore::Tensor post;
    infinicore::Tensor comb;
    infinicore::Tensor collapsed;
};

class DeepseekV4HyperConnection : public infinicore::nn::Module {
public:
    DeepseekV4HyperConnection(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    DeepseekV4HyperConnection(size_t hidden_size,
                              size_t hc_mult,
                              size_t sinkhorn_iters,
                              double hc_eps,
                              double rms_norm_eps,
                              const infinicore::Device &device);

    DeepseekV4HyperConnectionOutput
    forward(const infinicore::Tensor &hidden_streams) const;

    infinicore::Tensor apply(const infinicore::Tensor &hidden_streams,
                             const infinicore::Tensor &sublayer_output,
                             const infinicore::Tensor &post,
                             const infinicore::Tensor &comb) const;

protected:
    INFINICORE_NN_PARAMETER(fn);
    INFINICORE_NN_PARAMETER(base);
    INFINICORE_NN_PARAMETER(scale);

private:
    size_t hidden_size_;
    size_t hc_mult_;
    size_t sinkhorn_iters_;
    double hc_eps_;
    double rms_norm_eps_;
    infinicore::Tensor norm_weight_;
    infinicore::Tensor epsilon_;
};

class DeepseekV4HyperHead : public infinicore::nn::Module {
public:
    DeepseekV4HyperHead(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    DeepseekV4HyperHead(size_t hidden_size,
                        size_t hc_mult,
                        double hc_eps,
                        double rms_norm_eps,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_streams) const;

protected:
    INFINICORE_NN_PARAMETER(hc_fn);
    INFINICORE_NN_PARAMETER(hc_base);
    INFINICORE_NN_PARAMETER(hc_scale);

private:
    size_t hidden_size_;
    size_t hc_mult_;
    double hc_eps_;
    double rms_norm_eps_;
    infinicore::Tensor norm_weight_;
    infinicore::Tensor epsilon_;
};

} // namespace infinilm::models::deepseek_v4
