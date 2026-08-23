#include "deepseek_v4_moe.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/fmin.hpp>
#include <infinicore/ops/fused_moe_mxfp4.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/silu.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

infinicore::Tensor cast_to(const infinicore::Tensor &input,
                           const infinicore::DataType &dtype) {
    if (input->dtype() == dtype) {
        return input;
    }
    auto output = infinicore::Tensor::empty(
        input->shape(), dtype, input->device());
    infinicore::op::cast_(output, input);
    return output;
}

infinicore::Tensor upper_bound(const infinicore::Tensor &input,
                               float limit) {
    auto bound = infinicore::Tensor::ones(
        input->shape(), input->dtype(), input->device());
    infinicore::op::mul_scalar_(bound, bound, limit);
    return infinicore::op::fmin(input, bound);
}

infinicore::Tensor lower_bound(const infinicore::Tensor &input,
                               float limit) {
    auto negated = infinicore::op::mul_scalar(input, -1.0);
    return infinicore::op::mul_scalar(
        upper_bound(negated, -limit), -1.0);
}

infinicore::Tensor deepseek_v4_swiglu(
    const infinicore::Tensor &gate,
    const infinicore::Tensor &up,
    float limit) {
    auto bounded_gate = upper_bound(gate, limit);
    auto bounded_up = lower_bound(upper_bound(up, limit), -limit);
    return infinicore::op::mul(
        infinicore::op::silu(bounded_gate), bounded_up);
}

std::vector<float> tensor_to_f32(const infinicore::Tensor &tensor) {
    auto source = cast_to(tensor, infinicore::DataType::F32)
                      ->to(infinicore::Device::cpu())
                      ->contiguous();
    std::vector<float> values(source->numel());
    std::memcpy(values.data(), source->data(), source->nbytes());
    return values;
}

std::vector<int64_t> tensor_to_i64(const infinicore::Tensor &tensor) {
    auto source = tensor->to(infinicore::Device::cpu())->contiguous();
    std::vector<int64_t> values(source->numel());
    if (source->dtype() == infinicore::DataType::I64) {
        std::memcpy(values.data(), source->data(), source->nbytes());
    } else if (source->dtype() == infinicore::DataType::I32) {
        const auto *input =
            reinterpret_cast<const int32_t *>(source->data());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = input[i];
        }
    } else {
        throw std::runtime_error(
            "DeepSeek-V4 router indices must use I32 or I64");
    }
    return values;
}

float sqrt_softplus(float value) {
    const float softplus =
        std::max(value, 0.0f) + std::log1p(std::exp(-std::abs(value)));
    return std::sqrt(softplus);
}

} // namespace

DeepseekV4MLP::DeepseekV4MLP(
    size_t hidden_size,
    size_t intermediate_size,
    float swiglu_limit,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      swiglu_limit_(swiglu_limit) {
    if (hidden_size_ == 0 || intermediate_size_ == 0
        || swiglu_limit_ <= 0.0f) {
        throw std::runtime_error(
            "DeepSeek-V4 MLP configuration is invalid");
    }
    INFINICORE_NN_MODULE_INIT(
        gate_proj, hidden_size_, intermediate_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        up_proj, hidden_size_, intermediate_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        down_proj, intermediate_size_, hidden_size_, dtype, device);
}

infinicore::Tensor DeepseekV4MLP::forward(
    const infinicore::Tensor &hidden_states) const {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size_) {
        throw std::runtime_error(
            "DeepSeek-V4 MLP expects [batch, sequence, hidden_size]");
    }
    auto gate = gate_proj_->forward(hidden_states);
    auto up = up_proj_->forward(hidden_states);
    return down_proj_->forward(
        deepseek_v4_swiglu(gate, up, swiglu_limit_));
}

DeepseekV4Router::DeepseekV4Router(
    size_t hidden_size,
    size_t num_experts,
    size_t experts_per_token,
    size_t vocab_size,
    bool is_hash,
    float routed_scaling_factor,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      num_experts_(num_experts),
      experts_per_token_(experts_per_token),
      vocab_size_(vocab_size),
      is_hash_(is_hash),
      routed_scaling_factor_(routed_scaling_factor) {
    if (hidden_size_ == 0 || num_experts_ == 0
        || experts_per_token_ == 0
        || experts_per_token_ > num_experts_
        || vocab_size_ == 0 || routed_scaling_factor_ <= 0.0f) {
        throw std::runtime_error(
            "DeepSeek-V4 router configuration is invalid");
    }
    INFINICORE_NN_PARAMETER_INIT(
        weight,
        ({num_experts_, hidden_size_}, dtype, device));
    if (is_hash_) {
        INFINICORE_NN_PARAMETER_INIT(
            tid2eid,
            ({vocab_size_, experts_per_token_},
             infinicore::DataType::I64,
             device));
    } else {
        INFINICORE_NN_PARAMETER_INIT(
            e_score_correction_bias,
            ({num_experts_}, infinicore::DataType::F32, device));
    }
}

DeepseekV4RouterOutput DeepseekV4Router::forward(
    const infinicore::Tensor &hidden_states,
    const std::optional<infinicore::Tensor> &input_ids) const {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size_) {
        throw std::runtime_error(
            "DeepSeek-V4 router expects [batch, sequence, hidden_size]");
    }
    const size_t batch_size = hidden_states->size(0);
    const size_t sequence_length = hidden_states->size(1);
    const size_t tokens = batch_size * sequence_length;
    auto logits = infinicore::op::linear(
        hidden_states->view({tokens, hidden_size_}),
        static_cast<infinicore::Tensor>(weight_),
        std::nullopt);
    const auto logits_host = tensor_to_f32(logits);
    std::vector<float> scores(logits_host.size());
    std::transform(
        logits_host.begin(), logits_host.end(), scores.begin(),
        sqrt_softplus);
    std::vector<int64_t> hash_table;
    std::vector<int64_t> token_ids;
    std::vector<float> correction;
    if (is_hash_) {
        if (!input_ids.has_value() || !input_ids.value()
            || input_ids.value()->numel() != tokens) {
            throw std::runtime_error(
                "DeepSeek-V4 hash router requires one input id per token");
        }
        hash_table = tensor_to_i64(
            static_cast<infinicore::Tensor>(tid2eid_));
        token_ids = tensor_to_i64(input_ids.value());
    } else {
        correction = tensor_to_f32(
            static_cast<infinicore::Tensor>(e_score_correction_bias_));
    }

    std::vector<int32_t> selected(tokens * experts_per_token_);
    std::vector<float> routing(tokens * experts_per_token_);
    std::vector<size_t> expert_order(num_experts_);
    std::iota(expert_order.begin(), expert_order.end(), 0);
    for (size_t token = 0; token < tokens; ++token) {
        if (is_hash_) {
            const int64_t token_id = token_ids[token];
            if (token_id < 0
                || static_cast<size_t>(token_id) >= vocab_size_) {
                throw std::runtime_error(
                    "DeepSeek-V4 hash router input id is out of range");
            }
            for (size_t rank = 0; rank < experts_per_token_; ++rank) {
                const int64_t expert = hash_table[
                    static_cast<size_t>(token_id) * experts_per_token_
                    + rank];
                if (expert < 0
                    || static_cast<size_t>(expert) >= num_experts_) {
                    throw std::runtime_error(
                        "DeepSeek-V4 tid2eid contains an invalid expert");
                }
                selected[token * experts_per_token_ + rank] =
                    static_cast<int32_t>(expert);
            }
        } else {
            std::partial_sort(
                expert_order.begin(),
                expert_order.begin() + experts_per_token_,
                expert_order.end(),
                [&](size_t lhs, size_t rhs) {
                    return scores[token * num_experts_ + lhs]
                               + correction[lhs]
                         > scores[token * num_experts_ + rhs]
                               + correction[rhs];
                });
            for (size_t rank = 0; rank < experts_per_token_; ++rank) {
                selected[token * experts_per_token_ + rank] =
                    static_cast<int32_t>(expert_order[rank]);
            }
        }
        float denominator = 1e-20f;
        for (size_t rank = 0; rank < experts_per_token_; ++rank) {
            const auto expert = static_cast<size_t>(
                selected[token * experts_per_token_ + rank]);
            denominator += scores[token * num_experts_ + expert];
        }
        for (size_t rank = 0; rank < experts_per_token_; ++rank) {
            const auto expert = static_cast<size_t>(
                selected[token * experts_per_token_ + rank]);
            routing[token * experts_per_token_ + rank] =
                scores[token * num_experts_ + expert]
                / denominator * routed_scaling_factor_;
        }
    }

    auto selected_cpu = infinicore::Tensor::from_blob(
        selected.data(),
        {tokens, experts_per_token_},
        infinicore::DataType::I32,
        infinicore::Device::cpu());
    auto routing_cpu = infinicore::Tensor::from_blob(
        routing.data(),
        {tokens, experts_per_token_},
        infinicore::DataType::F32,
        infinicore::Device::cpu());
    return {
        routing_cpu->to(hidden_states->device()),
        selected_cpu->to(hidden_states->device()),
        logits};
}

DeepseekV4Experts::DeepseekV4Experts(
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size,
    float swiglu_limit,
    bool use_packed_fp4,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : num_experts_(num_experts),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      swiglu_limit_(swiglu_limit),
      use_packed_fp4_(use_packed_fp4) {
    if (num_experts_ == 0 || hidden_size_ == 0
        || intermediate_size_ == 0 || swiglu_limit_ <= 0.0f) {
        throw std::runtime_error(
            "DeepSeek-V4 experts configuration is invalid");
    }
    if (use_packed_fp4_) {
        register_packed_parameters_(device);
    } else {
        INFINICORE_NN_PARAMETER_INIT(
            gate_up_proj,
            ({num_experts_, 2 * intermediate_size_, hidden_size_},
             dtype,
             device));
        INFINICORE_NN_PARAMETER_INIT(
            down_proj,
            ({num_experts_, hidden_size_, intermediate_size_},
             dtype,
             device));
    }
}

void DeepseekV4Experts::register_packed_parameters_(
    const infinicore::Device &device) {
    if (hidden_size_ % 32 != 0 || intermediate_size_ % 32 != 0) {
        throw std::runtime_error(
            "DeepSeek-V4 MXFP4 dimensions must be divisible by 32");
    }
    packed_w13_ = infinicore::Tensor::empty(
        {num_experts_, 2 * intermediate_size_, hidden_size_ / 2},
        infinicore::DataType::U8,
        device);
    w13_scale_ = infinicore::Tensor::empty(
        {num_experts_, 2 * intermediate_size_, hidden_size_ / 32},
        infinicore::DataType::U8,
        device);
    packed_w2_ = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, intermediate_size_ / 2},
        infinicore::DataType::U8,
        device);
    w2_scale_ = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, intermediate_size_ / 32},
        infinicore::DataType::U8,
        device);
    auto register_slice = [&](const std::string &name,
                              const infinicore::Tensor &storage,
                              size_t expert,
                              size_t row_start,
                              size_t row_count) {
        auto view = storage
                        ->narrow({{0, expert, 1},
                                  {1, row_start, row_count}})
                        ->squeeze(0);
        this->register_parameter(
            name, infinicore::nn::Parameter(view));
    };
    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        register_slice(
            prefix + "w1.weight_packed",
            packed_w13_,
            expert,
            0,
            intermediate_size_);
        register_slice(
            prefix + "w1.weight_scale",
            w13_scale_,
            expert,
            0,
            intermediate_size_);
        register_slice(
            prefix + "w3.weight_packed",
            packed_w13_,
            expert,
            intermediate_size_,
            intermediate_size_);
        register_slice(
            prefix + "w3.weight_scale",
            w13_scale_,
            expert,
            intermediate_size_,
            intermediate_size_);
        register_slice(
            prefix + "w2.weight_packed",
            packed_w2_,
            expert,
            0,
            hidden_size_);
        register_slice(
            prefix + "w2.weight_scale",
            w2_scale_,
            expert,
            0,
            hidden_size_);
    }
}

infinicore::Tensor DeepseekV4Experts::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &selected_experts,
    const infinicore::Tensor &routing_weights) const {
    if (!hidden_states || hidden_states->ndim() != 2
        || hidden_states->size(1) != hidden_size_
        || !selected_experts || selected_experts->ndim() != 2
        || !routing_weights || routing_weights->ndim() != 2
        || selected_experts->shape() != routing_weights->shape()
        || selected_experts->size(0) != hidden_states->size(0)) {
        throw std::runtime_error(
            "DeepSeek-V4 experts input shape mismatch");
    }
    if (use_packed_fp4_) {
        // The current InfiniCore MXFP4 SwiGLU kernel is structurally correct,
        // but still needs the V4 gate/up clamp fused into its activation path.
        return infinicore::op::fused_moe_mxfp4(
            hidden_states,
            selected_experts,
            routing_weights,
            packed_w13_,
            w13_scale_,
            packed_w2_,
            w2_scale_,
            infinicore::op::FusedMoeActivation::Swiglu);
    }

    const auto selected = tensor_to_i64(selected_experts);
    const auto routing = tensor_to_f32(routing_weights);
    const size_t tokens = hidden_states->size(0);
    const size_t top_k = selected_experts->size(1);
    std::vector<infinicore::Tensor> token_outputs;
    token_outputs.reserve(tokens);
    for (size_t token = 0; token < tokens; ++token) {
        auto token_input = hidden_states
                               ->narrow({{0, token, 1}})
                               ->contiguous();
        auto token_output = infinicore::Tensor::zeros(
            {1, hidden_size_},
            hidden_states->dtype(),
            hidden_states->device());
        for (size_t rank = 0; rank < top_k; ++rank) {
            const int64_t expert = selected[token * top_k + rank];
            if (expert < 0
                || static_cast<size_t>(expert) >= num_experts_) {
                throw std::runtime_error(
                    "DeepSeek-V4 selected expert is out of range");
            }
            auto gate_up_weight =
                static_cast<infinicore::Tensor>(gate_up_proj_)
                    ->narrow({{0, static_cast<size_t>(expert), 1}})
                    ->squeeze(0)
                    ->contiguous();
            auto gate_up = infinicore::op::linear(
                token_input, gate_up_weight, std::nullopt);
            auto gate = gate_up
                            ->narrow({{1, 0, intermediate_size_}})
                            ->contiguous();
            auto up = gate_up
                          ->narrow(
                              {{1, intermediate_size_, intermediate_size_}})
                          ->contiguous();
            auto activated = deepseek_v4_swiglu(
                gate, up, swiglu_limit_);
            auto down_weight =
                static_cast<infinicore::Tensor>(down_proj_)
                    ->narrow({{0, static_cast<size_t>(expert), 1}})
                    ->squeeze(0)
                    ->contiguous();
            auto contribution = infinicore::op::linear(
                activated, down_weight, std::nullopt);
            contribution = infinicore::op::mul_scalar(
                contribution, routing[token * top_k + rank]);
            token_output = infinicore::op::add(
                token_output, contribution);
        }
        token_outputs.push_back(token_output);
    }
    return infinicore::op::cat(token_outputs, 0);
}

DeepseekV4SparseMoeBlock::DeepseekV4SparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    const auto &config = model_config->get_config_json();
    const auto &mlp_layer_types = config.at("mlp_layer_types");
    if (layer_idx >= mlp_layer_types.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 MoE layer index is out of range");
    }
    const bool is_hash =
        mlp_layer_types.at(layer_idx).get<std::string>() == "hash_moe";
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size =
        model_config->get<size_t>("moe_intermediate_size");
    const size_t num_experts = model_config->get<size_t>("num_experts");
    const size_t experts_per_token =
        model_config->get<size_t>("num_experts_per_tok");
    const size_t shared_intermediate_size =
        intermediate_size
        * model_config->get_or<size_t>("n_shared_experts", 1);
    const float swiglu_limit =
        model_config->get_or<float>("swiglu_limit", 10.0f);
    const bool use_packed_fp4 =
        model_config->get_or<std::string>("expert_dtype", "dense")
        == "fp4";
    const auto dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(
        gate,
        hidden_size,
        num_experts,
        experts_per_token,
        model_config->get<size_t>("vocab_size"),
        is_hash,
        model_config->get_or<float>("routed_scaling_factor", 1.5f),
        dtype,
        device);
    INFINICORE_NN_MODULE_INIT(
        experts,
        num_experts,
        hidden_size,
        intermediate_size,
        swiglu_limit,
        use_packed_fp4,
        dtype,
        device);
    INFINICORE_NN_MODULE_INIT(
        shared_experts,
        hidden_size,
        shared_intermediate_size,
        swiglu_limit,
        dtype,
        device);
}

infinicore::Tensor DeepseekV4SparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states,
    const std::optional<infinicore::Tensor> &input_ids) const {
    auto routing = gate_->forward(hidden_states, input_ids);
    const auto shape = hidden_states->shape();
    auto routed = experts_->forward(
                      hidden_states->view(
                          {shape[0] * shape[1], shape[2]}),
                      routing.selected_experts,
                      routing.routing_weights)
                      ->view(shape);
    return infinicore::op::add(
        routed, shared_experts_->forward(hidden_states));
}

} // namespace infinilm::models::deepseek_v4
