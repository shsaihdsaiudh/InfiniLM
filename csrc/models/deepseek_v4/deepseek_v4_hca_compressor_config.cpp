#include "deepseek_v4_hca_compressor.hpp"

namespace infinilm::models::deepseek_v4 {

DeepseekV4HCACompressor::DeepseekV4HCACompressor(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : DeepseekV4HCACompressor(
          model_config->get<size_t>("hidden_size"),
          model_config->get<size_t>("head_dim"),
          model_config->get<size_t>("qk_rope_head_dim"),
          model_config->get_config_json()
              .at("compress_rates")
              .at("heavily_compressed_attention")
              .get<size_t>(),
          model_config->get<double>("rms_norm_eps"),
          model_config->get_dtype(),
          device) {}

} // namespace infinilm::models::deepseek_v4
