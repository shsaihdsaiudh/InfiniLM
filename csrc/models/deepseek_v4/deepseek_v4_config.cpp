#include "deepseek_v4_config.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

std::string attention_type_from_compress_ratio(size_t ratio) {
    switch (ratio) {
    case 0:
        return "sliding_attention";
    case 4:
        return "compressed_sparse_attention";
    case 128:
        return "heavily_compressed_attention";
    default:
        throw std::runtime_error(
            "DeepSeek-V4: unsupported attention compress ratio "
            + std::to_string(ratio));
    }
}

void validate_layer_types(const nlohmann::json &layer_types,
                          size_t num_hidden_layers) {
    if (!layer_types.is_array() || layer_types.size() != num_hidden_layers) {
        throw std::runtime_error(
            "DeepSeek-V4: layer_types must contain one entry per hidden layer");
    }
    for (const auto &layer_type_json : layer_types) {
        const auto layer_type = layer_type_json.get<std::string>();
        if (layer_type != "sliding_attention"
            && layer_type != "compressed_sparse_attention"
            && layer_type != "heavily_compressed_attention") {
            throw std::runtime_error(
                "DeepSeek-V4: unsupported layer type " + layer_type);
        }
    }
}

} // namespace

std::shared_ptr<infinilm::config::ModelConfig>
create_deepseek_v4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    auto &config = model_config->get_config_json();
    if (config.value("model_type", "") != "deepseek_v4") {
        throw std::runtime_error(
            "create_deepseek_v4_model_config: model_type is not deepseek_v4");
    }

    const size_t num_hidden_layers = config.at("num_hidden_layers").get<size_t>();
    const size_t head_dim = config.at("head_dim").get<size_t>();
    const size_t rotary_dim = config.at("qk_rope_head_dim").get<size_t>();
    const size_t num_experts = config.at("n_routed_experts").get<size_t>();
    const size_t experts_per_token = config.at("num_experts_per_tok").get<size_t>();
    const size_t num_hash_layers = config.value("num_hash_layers", 0UL);

    if (num_hidden_layers == 0 || head_dim == 0 || rotary_dim == 0
        || rotary_dim > head_dim || rotary_dim % 2 != 0) {
        throw std::runtime_error(
            "DeepSeek-V4: invalid layer count or partial-RoPE dimensions");
    }
    if (num_experts == 0 || experts_per_token == 0
        || experts_per_token > num_experts) {
        throw std::runtime_error("DeepSeek-V4: invalid routed expert configuration");
    }
    if (num_hash_layers > num_hidden_layers) {
        throw std::runtime_error(
            "DeepSeek-V4: num_hash_layers exceeds num_hidden_layers");
    }
    if (config.value("hc_mult", 0UL) == 0
        || config.value("hc_sinkhorn_iters", 0UL) == 0) {
        throw std::runtime_error("DeepSeek-V4: invalid mHC configuration");
    }
    if (config.value("expert_dtype", std::string{}) != "fp4") {
        throw std::runtime_error(
            "DeepSeek-V4: the initial InfiniLM path requires FP4 routed experts");
    }

    if (config.contains("layer_types")) {
        validate_layer_types(config.at("layer_types"), num_hidden_layers);
    } else {
        const auto &compress_ratios = config.at("compress_ratios");
        if (!compress_ratios.is_array()
            || compress_ratios.size() < num_hidden_layers) {
            throw std::runtime_error(
                "DeepSeek-V4: compress_ratios does not cover all hidden layers");
        }
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t layer = 0; layer < num_hidden_layers; ++layer) {
            layer_types.push_back(attention_type_from_compress_ratio(
                compress_ratios.at(layer).get<size_t>()));
        }
        config["layer_types"] = std::move(layer_types);
    }

    if (!config.contains("mlp_layer_types")) {
        std::vector<std::string> mlp_layer_types;
        mlp_layer_types.reserve(num_hidden_layers);
        for (size_t layer = 0; layer < num_hidden_layers; ++layer) {
            mlp_layer_types.push_back(layer < num_hash_layers ? "hash_moe" : "moe");
        }
        config["mlp_layer_types"] = std::move(mlp_layer_types);
    } else if (!config.at("mlp_layer_types").is_array()
               || config.at("mlp_layer_types").size() != num_hidden_layers) {
        throw std::runtime_error(
            "DeepSeek-V4: mlp_layer_types must contain one entry per hidden layer");
    }

    config["num_experts"] = num_experts;
    config["partial_rotary_factor"] =
        static_cast<double>(rotary_dim) / static_cast<double>(head_dim);
    config["mlp_bias"] = false;
    config["attention_output_bias"] = false;
    config["enable_dspark"] = false;
    if (!config.contains("dtype") && config.contains("torch_dtype")) {
        config["dtype"] = config["torch_dtype"];
    }

    // V4 applies RoPE to a contiguous tail of each query/key head.
    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);
    return model_config;
}

} // namespace infinilm::models::deepseek_v4
