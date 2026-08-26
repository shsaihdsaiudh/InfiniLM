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

// DeepseekV4Model：V4 的 Transformer 主干（不含 lm_head）。
// 负责：embedding → 逐层 decoder → mHC 头部折叠 → 输出层 norm。
// 注意：模型从 embed 之后就把 hidden state 展开成 hc_mult 条并行流，
// 全程以四维张量 [B, S, hc_mult, hidden_size] 传递，到最后一层才折回三维。
DeepseekV4Model::DeepseekV4Model(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      hc_mult_(model_config->get<size_t>("hc_mult")),       // 并行残差流数（V4 为 4）
      rope_dim_(model_config->get<size_t>("qk_rope_head_dim")),  // 每个 head 参与 RoPE 的维度（partial RoPE）
      rope_theta_(model_config->get_or<double>("rope_theta", 10000.0)),  // 主 RoPE base（滑窗层用）
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

    // 词表 embedding：[vocab_size, hidden_size]
    INFINICORE_NN_MODULE_INIT(
        embed_tokens,
        model_config->get<size_t>("vocab_size"),
        hidden_size_,
        std::nullopt,
        dtype_,
        device_);
    // 逐层创建 decoder layer（每层内部按 layer_types 决定注意力/FFN 形态）
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
    // 最末层 RMSNorm
    INFINICORE_NN_MODULE_INIT(
        norm,
        hidden_size_,
        model_config->get<double>("rms_norm_eps"),
        dtype_,
        device_);
    // mHC 头部：把 hc_mult 条流折叠回单条 hidden，供 lm_head 使用
    INFINICORE_NN_MODULE_INIT(hc_head, model_config, device_);
}

// 把调用方传入的 position_ids 归一化成 [batch_size, sequence_length] 二维形状。
// 兼容三种传入形式：已经是二维；一维但按 [B*S] 摊平；一维只含单个序列（需要广播到 batch）。
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

// 前向主流程（V4 主干，不含 lm_head）：
//   embed [B,S,D] → 展开成 [B,S,hc_mult,D]（mHC 四流起点）
//   → 逐层 decoder（每层保持四流进出）→ hc_head 折叠回 [B,S,D] → RMSNorm
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
    // 1) 词嵌入，得到 [B, S, hidden_size]
    auto hidden = embed_tokens_->forward(input_ids);
    // 2) 关键一步：unsqueeze 出"流"维度再广播到 hc_mult 份，hidden 从此变成
    //    [B, S, hc_mult, hidden_size] 四维，四流残差结构从这里开始。
    //    （初值为四路完全相同，之后由各层 mHC 逐步拉开差异。）
    auto expanded = infinicore::op::broadcast_to(
                        hidden->unsqueeze(2),
                        {static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(sequence_length),
                         static_cast<int64_t>(hc_mult_),
                         static_cast<int64_t>(hidden_size_)})
                        ->contiguous();
    // 3) 主 RoPE（滑窗层用它；CSA/HCA 层内部会换用压缩分支的 YaRN RoPE）
    auto query_rope = deepseek_v4_rotary_embedding(
        position_ids,
        rope_dim_,
        rope_theta_,
        dtype_,
        device_);
    // 4) 逐层前向：每一层都接收/返回 [B, S, hc_mult, D]，同时维护各自的 KV 状态
    for (const auto &layer : layers_) {
        expanded = layer->forward(
            expanded,
            position_ids,
            query_rope.first,
            query_rope.second,
            input_ids);
    }
    // 5) hc_head 把 hc_mult 条流按权重折叠回单条 [B, S, D]，再过最终 RMSNorm
    return norm_->forward(hc_head_->forward(expanded));
}

// DeepseekV4ForCausalLM：模型外壳，在主干之上叠加 lm_head（词表投影）。
// 对外表现为"因果语言模型"：输入 token 序列，输出每个位置的下一个 token logits。
DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    // 主干（embed + 层堆叠 + norm + hc_head）
    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    // 输出投影 [hidden_size → vocab_size]，无 bias
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
    // 采样场景（非 sample_all_positions）：只对每个请求的最后一个 token 位置
    // 取 hidden 做 lm_head，避免对整条序列都算 logits。
    // input_offsets 形如 [0, len0, len0+len1, ...]，表示各请求的边界。
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
    // 返回 {logits, hidden_states}（hidden 供需要中间表示的上层使用）
    return {lm_head_->forward(lm_head_input), hidden_states};
}

// 清空运行时 KV / 压缩状态，并更新 cache 配置。
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
// 模型注册：把 deepseek_v4 这个 model_type 关联到本实现，
// 并指定 config 工厂函数 create_deepseek_v4_model_config。
// 之后 InfiniLM 遇到 model_type == "deepseek_v4" 就会走这个类。
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
} // namespace
