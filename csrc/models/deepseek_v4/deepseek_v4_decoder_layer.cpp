#include "deepseek_v4_decoder_layer.hpp"

#include "deepseek_v4_rotary_embedding.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

// DeepseekV4DecoderLayer：一层 Transformer 的组装。
// 层的结构在构造时定型（按 layer_types 决定注意力形态），
// 前向时接收/返回 [B, S, hc_mult, hidden_size] 四流张量。
DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx),
      sliding_window_(model_config->get<size_t>("sliding_window")),
      rope_dim_(model_config->get<size_t>("qk_rope_head_dim")),
      compress_rope_theta_(
          model_config->get_or<double>("compress_rope_theta", 160000.0)),
      dtype_(model_config->get_dtype()),
      device_(device) {
    const auto &config = model_config->get_config_json();
    // 层型调度：从 config 的 layer_types 数组按层号取本层形态。
    // 取值是 "sliding_attention" / "compressed_sparse_attention" /
    // "heavily_compressed_attention" 三者之一，决定后续走哪条注意力路径。
    const auto &layer_types = config.at("layer_types");
    if (layer_idx_ >= layer_types.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 decoder layer index is out of range");
    }
    layer_type_ = layer_types.at(layer_idx_).get<std::string>();
    // 压缩分支的 RoPE 缩放配置（YaRN）。sliding 层用主 RoPE（无缩放），
    // CSA/HCA 层用这里的 compress RoPE（theta=160000、factor=16 等）。
    const nlohmann::json *rope_scaling = nullptr;
    if (config.contains("rope_parameters")
        && config.at("rope_parameters").is_object()
        && config.at("rope_parameters").contains("compress")
        && config.at("rope_parameters").at("compress").is_object()) {
        rope_scaling = &config.at("rope_parameters").at("compress");
    } else if (config.contains("rope_scaling")
               && config.at("rope_scaling").is_object()) {
        rope_scaling = &config.at("rope_scaling");
    }
    if (rope_scaling != nullptr) {
        const std::string rope_type = rope_scaling->value(
            "rope_type", rope_scaling->value("type", std::string{"default"}));
        if (rope_type == "yarn") {
            compress_yarn_ = DeepseekV4YarnScaling{
                rope_scaling->at("factor").get<double>(),
                rope_scaling->value("beta_fast", 32.0),
                rope_scaling->value("beta_slow", 1.0),
                rope_scaling->at("original_max_position_embeddings")
                    .get<size_t>(),
                rope_scaling->value("attention_factor", 1.0),
                rope_scaling->value("truncate", true)};
        } else if (rope_type != "default") {
            throw std::runtime_error(
                "DeepSeek-V4 compress RoPE supports only default or yarn");
        }
    }
    // 压缩率按层型查表：CSA=4、HCA=128（sliding 层无压缩，不查）
    if (layer_type_ != "sliding_attention") {
        compress_rate_ = config.at("compress_rates")
                             .at(layer_type_)
                             .get<size_t>();
    }
    if (sliding_window_ < 2 || rope_dim_ == 0
        || rope_dim_ % 2 != 0 || compress_rope_theta_ <= 0.0) {
        throw std::runtime_error(
            "DeepSeek-V4 decoder attention configuration is invalid");
    }

    // 一层包含的模块：
    //   self_attn（内部含 sliding/HCA/CSA 三条路径）、mlp（MoE）、
    //   两个 RMSNorm（attn 前 / ffn 前）、两个 mHC（attn 前折叠 / ffn 前折叠）
    INFINICORE_NN_MODULE_INIT(
        self_attn, model_config, layer_idx_, device_);
    INFINICORE_NN_MODULE_INIT(
        mlp, model_config, layer_idx_, device_);
    INFINICORE_NN_MODULE_INIT(
        input_layernorm,
        model_config->get<size_t>("hidden_size"),
        model_config->get<double>("rms_norm_eps"),
        dtype_,
        device_);
    INFINICORE_NN_MODULE_INIT(
        post_attention_layernorm,
        model_config->get<size_t>("hidden_size"),
        model_config->get<double>("rms_norm_eps"),
        dtype_,
        device_);
    INFINICORE_NN_MODULE_INIT(attn_hc, model_config, device_);
    INFINICORE_NN_MODULE_INIT(ffn_hc, model_config, device_);
}

// 为压缩分支（CSA/HCA）计算压缩条目的 RoPE。
// 压缩条目的位置号与"已缓冲的 token 数 + 本次长度"和压缩率有关，
// 并需要从压缩状态的起始条目号（first_entry）续算，保证增量一致性。
std::pair<infinicore::Tensor, infinicore::Tensor>
DeepseekV4DecoderLayer::compressed_rotary_(
    size_t batch_size,
    size_t sequence_length) const {
    size_t buffered = 0;
    size_t first_entry = 0;
    if (layer_type_ == "heavily_compressed_attention") {
        if (hca_state_.buffer_kv) {
            buffered = hca_state_.buffer_kv->size(1);
        }
        first_entry = hca_state_.entry_count;
    } else if (layer_type_ == "compressed_sparse_attention") {
        if (csa_state_.compressor.buffer_kv) {
            buffered = csa_state_.compressor.buffer_kv->size(1);
        }
        first_entry = csa_state_.compressor.entry_count;
    } else {
        throw std::runtime_error(
            "DeepSeek-V4 compressed RoPE requested for a sliding layer");
    }
    // 新增压缩条目数 = 当前已缓冲 token 数 + 本次序列长度，再除以压缩率
    const size_t new_entries =
        (buffered + sequence_length) / compress_rate_;
    return deepseek_v4_compressed_rotary_embedding(
        batch_size,
        new_entries,
        first_entry,
        compress_rate_,
        rope_dim_,
        compress_rope_theta_,
        dtype_,
        device_,
        compress_yarn_);
}

// 一层的前向：接收 [B,S,hc_mult,D] 四流，返回 [B,S,hc_mult,D] 四流。
// 内部顺序：mHC折叠 → RMSNorm → 注意力(按层型) → mHC混合
//           → mHC折叠 → RMSNorm → MoE → mHC混合
infinicore::Tensor DeepseekV4DecoderLayer::forward(
    const infinicore::Tensor &hidden_streams,
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &query_cos,
    const infinicore::Tensor &query_sin,
    const std::optional<infinicore::Tensor> &input_ids) const {
    // ① 第一个 mHC：四流折叠成单路，作为 attention 输入（再过 RMSNorm）
    auto attn_mix = attn_hc_->forward(hidden_streams);
    auto attention_input = input_layernorm_->forward(
        attn_mix.collapsed);

    // ② 选 RoPE：sliding 层用主 RoPE（query_cos/sin，theta=10000）；
    //    CSA/HCA 层改用压缩分支的 YaRN RoPE（compress_rope_theta=160000），
    //    与参考实现的层型调度一致。
    auto attention_cos = query_cos;
    auto attention_sin = query_sin;
    if (layer_type_ != "sliding_attention") {
        auto compressed_query_rope = deepseek_v4_rotary_embedding(
            position_ids,
            rope_dim_,
            compress_rope_theta_,
            dtype_,
            device_,
            compress_yarn_);
        attention_cos = compressed_query_rope.first;
        attention_sin = compressed_query_rope.second;
    }

    // ③ 按层型分派注意力路径；压缩层型各自带累积的压缩状态
    DeepseekV4SlidingAttentionOutput attention;
    if (layer_type_ == "sliding_attention") {
        attention = self_attn_->forward_sliding(
            attention_input,
            attention_cos,
            attention_sin,
            sliding_kv_,
            sliding_window_);
    } else {
        // 压缩层需要额外的压缩条目 RoPE（由累积状态推算位置）
        auto compressed_rope = compressed_rotary_(
            hidden_streams->size(0), hidden_streams->size(1));
        if (layer_type_ == "heavily_compressed_attention") {
            attention = self_attn_->forward_hca(
                attention_input,
                attention_cos,
                attention_sin,
                compressed_rope.first,
                compressed_rope.second,
                position_ids,
                sliding_kv_,
                &hca_state_,
                sliding_window_);
        } else if (layer_type_ == "compressed_sparse_attention") {
            attention = self_attn_->forward_csa(
                attention_input,
                attention_cos,
                attention_sin,
                compressed_rope.first,
                compressed_rope.second,
                position_ids,
                sliding_kv_,
                &csa_state_,
                sliding_window_);
        } else {
            throw std::runtime_error(
                "DeepSeek-V4 decoder has an unsupported attention type");
        }
    }
    // ④ 更新滑窗 KV，然后用第二个 mHC 把 attention 输出混合回四流
    sliding_kv_ = attention.kv_cache;
    auto mixed = attn_hc_->apply(
        hidden_streams,
        attention.output,
        attn_mix.post,
        attn_mix.comb);

    // ⑤ FFN 半层：同样的"折叠→RMSNorm→MoE→混合"流程
    auto ffn_mix = ffn_hc_->forward(mixed);
    auto ffn_input = post_attention_layernorm_->forward(
        ffn_mix.collapsed);
    auto ffn_output = mlp_->forward(ffn_input, input_ids);
    // 返回仍是 [B,S,hc_mult,D]，作为下一层的输入
    return ffn_hc_->apply(
        mixed,
        ffn_output,
        ffn_mix.post,
        ffn_mix.comb);
}

// 清空本层的运行时状态（滑窗 KV、HCA/CSA 压缩状态），用于重置推理
void DeepseekV4DecoderLayer::reset_runtime_state() const {
    sliding_kv_.reset();
    hca_state_ = {};
    csa_state_ = {};
}

} // namespace infinilm::models::deepseek_v4
