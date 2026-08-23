#pragma once

#include "../../config/model_config.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

std::shared_ptr<infinilm::config::ModelConfig>
create_deepseek_v4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek_v4
