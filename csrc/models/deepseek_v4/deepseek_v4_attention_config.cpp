#include "deepseek_v4_attention.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

std::string layer_type(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    size_t layer_idx) {
    const auto &config = model_config->get_config_json();
    const auto &layer_types = config.at("layer_types");
    if (layer_idx >= layer_types.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 attention layer index is out of range");
    }
    return layer_types.at(layer_idx).get<std::string>();
}

size_t compress_rate(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    size_t layer_idx,
    const std::string &expected_type) {
    if (layer_type(model_config, layer_idx) != expected_type) {
        return 0;
    }
    return model_config->get_config_json()
        .at("compress_rates")
        .at(expected_type)
        .get<size_t>();
}

size_t csa_value(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    size_t layer_idx,
    const char *key) {
    if (layer_type(model_config, layer_idx)
        != "compressed_sparse_attention") {
        return 0;
    }
    return model_config->get<size_t>(key);
}

} // namespace

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
          device,
          compress_rate(
              model_config,
              layer_idx,
              "heavily_compressed_attention"),
          compress_rate(
              model_config,
              layer_idx,
              "compressed_sparse_attention"),
          csa_value(model_config, layer_idx, "index_n_heads"),
          csa_value(model_config, layer_idx, "index_head_dim"),
          csa_value(model_config, layer_idx, "index_topk")) {
    const auto &layer_types =
        model_config->get_config_json().at("layer_types");
    if (layer_idx >= layer_types.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 attention layer index is out of range");
    }
}

} // namespace infinilm::models::deepseek_v4
