#include "deepseek_v4_attention.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

DeepseekV4Attention::DeepseekV4Attention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : DeepseekV4Attention(
          model_config->get<size_t>("hidden_size"),
          model_config->get<size_t>("q_lora_rank"),
          model_config->get<size_t>("num_attention_heads"),
          model_config->get<size_t>("head_dim"),
          model_config->get<size_t>("qk_rope_head_dim"),
          model_config->get<size_t>("o_groups"),
          model_config->get<size_t>("o_lora_rank"),
          model_config->get<double>("rms_norm_eps"),
          model_config->get_dtype(),
          device) {
    const auto &layer_types =
        model_config->get_config_json().at("layer_types");
    if (layer_idx >= layer_types.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 attention layer index is out of range");
    }
}

} // namespace infinilm::models::deepseek_v4
