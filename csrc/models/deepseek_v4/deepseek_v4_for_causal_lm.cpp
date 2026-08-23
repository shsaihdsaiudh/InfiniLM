#include "deepseek_v4_for_causal_lm.hpp"

#include "deepseek_v4_config.hpp"
#include "deepseek_v4_rotary_embedding.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/select_last_token_hidden.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {

DeepseekV4Model::DeepseekV4Model(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      hc_mult_(model_config->get<size_t>("hc_mult")),
      rope_dim_(model_config->get<size_t>("qk_rope_head_dim")),
      rope_theta_(model_config->get_or<double>("rope_theta", 10000.0)),
      dtype_(model_config->get_dtype()),
      device_(device) {
    const auto &rank_info =
        infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.pp_size != 1 || rank_info.tp_size != 1) {
        throw std::runtime_error(
            "DeepSeek-V4 correctness path currently requires PP=TP=1");
    }
    if (hidden_size_ == 0 || hc_mult_ == 0 || rope_dim_ == 0
        || rope_dim_ % 2 != 0 || rope_theta_ <= 0.0) {
        throw std::runtime_error(
            "DeepSeek-V4 model dimensions or RoPE configuration are invalid");
    }

    INFINICORE_NN_MODULE_INIT(
        embed_tokens,
        model_config->get<size_t>("vocab_size"),
        hidden_size_,
        std::nullopt,
        dtype_,
        device_);
    const size_t num_layers =
        model_config->get<size_t>("num_hidden_layers");
    layers_.reserve(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        layers_.push_back(
            this->register_module<DeepseekV4DecoderLayer>(
                "layers." + std::to_string(layer),
                model_config,
                layer,
                device_));
    }
    INFINICORE_NN_MODULE_INIT(
        norm,
        hidden_size_,
        model_config->get<double>("rms_norm_eps"),
        dtype_,
        device_);
    INFINICORE_NN_MODULE_INIT(hc_head, model_config, device_);
}

infinicore::Tensor DeepseekV4Model::normalize_position_ids_(
    const infinicore::Tensor &position_ids,
    size_t batch_size,
    size_t sequence_length) const {
    if (!position_ids) {
        throw std::runtime_error(
            "DeepSeek-V4 requires position_ids");
    }
    if (position_ids->ndim() == 2
        && position_ids->size(0) == batch_size
        && position_ids->size(1) == sequence_length) {
        return position_ids;
    }
    if (position_ids->ndim() == 1
        && position_ids->size(0) == batch_size * sequence_length) {
        return position_ids->view({batch_size, sequence_length});
    }
    if (position_ids->ndim() == 1
        && position_ids->size(0) == sequence_length) {
        auto one_batch = position_ids->view({1, sequence_length});
        if (batch_size == 1) {
            return one_batch;
        }
        return infinicore::op::broadcast_to(
                   one_batch,
                   {static_cast<int64_t>(batch_size),
                    static_cast<int64_t>(sequence_length)})
            ->contiguous();
    }
    throw std::runtime_error(
        "DeepSeek-V4 position_ids shape does not match input_ids");
}

infinicore::Tensor DeepseekV4Model::forward(
    const infinilm::InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value() || !input.input_ids.value()
        || input.input_ids.value()->ndim() != 2
        || !input.position_ids.has_value()) {
        throw std::runtime_error(
            "DeepSeek-V4 requires 2D input_ids and position_ids");
    }
    const auto &input_ids = input.input_ids.value();
    const size_t batch_size = input_ids->size(0);
    const size_t sequence_length = input_ids->size(1);
    auto position_ids = normalize_position_ids_(
        input.position_ids.value(), batch_size, sequence_length);
    auto hidden = embed_tokens_->forward(input_ids);
    auto expanded = infinicore::op::broadcast_to(
                        hidden->unsqueeze(2),
                        {static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(sequence_length),
                         static_cast<int64_t>(hc_mult_),
                         static_cast<int64_t>(hidden_size_)})
                        ->contiguous();
    auto query_rope = deepseek_v4_rotary_embedding(
        position_ids,
        rope_dim_,
        rope_theta_,
        dtype_,
        device_);
    for (const auto &layer : layers_) {
        expanded = layer->forward(
            expanded,
            position_ids,
            query_rope.first,
            query_rope.second,
            input_ids);
    }
    return norm_->forward(hc_head_->forward(expanded));
}

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(
        lm_head,
        model_config->get<size_t>("hidden_size"),
        model_config->get<size_t>("vocab_size"),
        false,
        model_config->get_dtype(),
        device);
}

InfinilmModel::Output DeepseekV4ForCausalLM::forward(
    const Input &input) const {
    auto hidden_states = model_->forward(input);
    auto lm_head_input = hidden_states;
    if (!input.sample_all_positions && input.input_offsets.has_value()) {
        const size_t num_requests =
            input.input_offsets.value()->numel() - 1;
        if (hidden_states->size(0) == 1
            && hidden_states->size(1) > num_requests) {
            lm_head_input = infinicore::Tensor::empty(
                {1, num_requests, hidden_states->size(2)},
                hidden_states->dtype(),
                hidden_states->device());
            infinicore::op::select_last_token_hidden_(
                lm_head_input,
                hidden_states,
                input.input_offsets.value());
        }
    }
    return {lm_head_->forward(lm_head_input), hidden_states};
}

void DeepseekV4ForCausalLM::reset_cache(
    const cache::CacheConfig *cache_config) {
    cache_config_ = cache_config == nullptr
        ? nullptr
        : cache_config->unique_copy();
    infinilm::global_state::get_forward_context().kv_cache_vec.clear();
    reset_runtime_state();
}

} // namespace infinilm::models::deepseek_v4

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
} // namespace
