#pragma once

#include "deepseek_v4_attention.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <optional>
#include <string>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MLP : public infinicore::nn::Module {
public:
    DeepseekV4MLP(size_t hidden_size,
                  size_t intermediate_size,
                  float swiglu_limit,
                  const infinicore::DataType &dtype,
                  const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Linear, gate_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, up_proj);
    INFINICORE_NN_MODULE(DeepseekV4Linear, down_proj);

private:
    size_t hidden_size_;
    size_t intermediate_size_;
    float swiglu_limit_;
};

struct DeepseekV4RouterOutput {
    infinicore::Tensor routing_weights;
    infinicore::Tensor selected_experts;
    infinicore::Tensor logits;
};

class DeepseekV4Router : public infinicore::nn::Module {
public:
    DeepseekV4Router(size_t hidden_size,
                     size_t num_experts,
                     size_t experts_per_token,
                     size_t vocab_size,
                     bool is_hash,
                     float routed_scaling_factor,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);

    DeepseekV4RouterOutput forward(
        const infinicore::Tensor &hidden_states,
        const std::optional<infinicore::Tensor> &input_ids) const;

protected:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(e_score_correction_bias);
    INFINICORE_NN_PARAMETER(tid2eid);

private:
    size_t hidden_size_;
    size_t num_experts_;
    size_t experts_per_token_;
    size_t vocab_size_;
    bool is_hash_;
    float routed_scaling_factor_;
};

class DeepseekV4Experts : public infinicore::nn::Module {
public:
    DeepseekV4Experts(size_t num_experts,
                      size_t hidden_size,
                      size_t intermediate_size,
                      float swiglu_limit,
                      bool use_packed_fp4,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinicore::Tensor &hidden_states,
        const infinicore::Tensor &selected_experts,
        const infinicore::Tensor &routing_weights) const;

protected:
    // Dense correctness path, matching the Transformers tiny state dict.
    INFINICORE_NN_PARAMETER(gate_up_proj);
    INFINICORE_NN_PARAMETER(down_proj);

private:
    void register_packed_parameters_(
        const infinicore::Device &device);

    size_t num_experts_;
    size_t hidden_size_;
    size_t intermediate_size_;
    float swiglu_limit_;
    bool use_packed_fp4_;
    infinicore::Tensor packed_w13_;
    infinicore::Tensor w13_scale_;
    infinicore::Tensor packed_w2_;
    infinicore::Tensor w2_scale_;
};

class DeepseekV4SparseMoeBlock : public infinicore::nn::Module {
public:
    DeepseekV4SparseMoeBlock(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinicore::Tensor &hidden_states,
        const std::optional<infinicore::Tensor> &input_ids =
            std::nullopt) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV4Router, gate);
    INFINICORE_NN_MODULE(DeepseekV4Experts, experts);
    INFINICORE_NN_MODULE(DeepseekV4MLP, shared_experts);
};

} // namespace infinilm::models::deepseek_v4
