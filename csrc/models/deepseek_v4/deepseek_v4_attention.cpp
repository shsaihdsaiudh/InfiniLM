#include "deepseek_v4_attention.hpp"
#include "deepseek_v4_csa_compressor.hpp"
#include "deepseek_v4_hca_compressor.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/rms_norm.hpp>
#include <infinicore/ops/softmax.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

// ---- 一组张量工具（本文件内复用）----
// cast_to：把张量转成指定 dtype；若已相同则直接返回（避免多余拷贝）。
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

// broadcast_to_shape：广播到目标形状；若形状已匹配则原样返回。
infinicore::Tensor broadcast_to_shape(const infinicore::Tensor &input,
                                      const infinicore::Shape &shape) {
    if (input->shape() == shape) {
        return input;
    }
    return infinicore::op::broadcast_to(
        input, std::vector<int64_t>(shape.begin(), shape.end()));
}

// multiply_broadcast：两个张量各自广播到 shape 后逐元素相乘。
infinicore::Tensor multiply_broadcast(const infinicore::Tensor &lhs,
                                      const infinicore::Tensor &rhs,
                                      const infinicore::Shape &shape) {
    return infinicore::op::mul(
        broadcast_to_shape(lhs, shape), broadcast_to_shape(rhs, shape));
}

// add_broadcast：两个张量各自广播到 shape 后逐元素相加。
infinicore::Tensor add_broadcast(const infinicore::Tensor &lhs,
                                 const infinicore::Tensor &rhs,
                                 const infinicore::Shape &shape) {
    return infinicore::op::add(
        broadcast_to_shape(lhs, shape), broadcast_to_shape(rhs, shape));
}

// validate_input：校验 attention 输入形状。
// hidden_states 必须是 [B, S, hidden_size]；cos/sin 必须是 [B, S, rope_dim/2]
// 且与 hidden_states 的 batch/sequence 一致（partial RoPE 只旋转部分维度）。
void validate_input(const infinicore::Tensor &hidden_states,
                    const infinicore::Tensor &cos,
                    const infinicore::Tensor &sin,
                    size_t hidden_size,
                    size_t rope_head_dim) {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size) {
        throw std::runtime_error(
            "DeepSeek-V4 attention expects [batch, sequence, hidden_size]");
    }
    if (!cos || !sin || cos->ndim() != 3 || sin->ndim() != 3
        || cos->shape() != sin->shape()
        || cos->size(0) != hidden_states->size(0)
        || cos->size(1) != hidden_states->size(1)
        || cos->size(2) * 2 != rope_head_dim) {
        throw std::runtime_error(
            "DeepSeek-V4 attention cos/sin must be [batch, sequence, rope_dim/2]");
    }
}

} // namespace

// DeepseekV4Linear：一个简单的全连接层（weight 矩阵 + 无 bias）。
DeepseekV4Linear::DeepseekV4Linear(
    size_t in_features,
    size_t out_features,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : in_features_(in_features), out_features_(out_features) {
    if (in_features_ == 0 || out_features_ == 0) {
        throw std::runtime_error("DeepSeek-V4 linear dimensions must be non-zero");
    }
    INFINICORE_NN_PARAMETER_INIT(
        weight,
        ({out_features_, in_features_}, dtype, device));
}

// forward：输入 [..., in_features] → 线性映射 → [..., out_features]
// （先展平成 [rows, in_features]，乘完再还原末尾维度）
infinicore::Tensor DeepseekV4Linear::forward(
    const infinicore::Tensor &input) const {
    if (!input || input->ndim() == 0
        || input->size(input->ndim() - 1) != in_features_) {
        throw std::runtime_error("DeepSeek-V4 linear input shape mismatch");
    }
    const size_t rows = input->numel() / in_features_;
    auto output = infinicore::op::linear(
        input->contiguous()->view({rows, in_features_}),
        static_cast<infinicore::Tensor>(weight_),
        std::nullopt);
    auto shape = input->shape();
    shape.back() = out_features_;
    return output->view(shape);
}

// DeepseekV4RMSNorm：RMSNorm（无均值归一化，只除均方根）。
// 用全 1 的 F32 权重做归一化，再乘上可学习的 weight。
DeepseekV4RMSNorm::DeepseekV4RMSNorm(
    size_t hidden_size,
    double eps,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size), eps_(eps) {
    if (hidden_size_ == 0 || eps_ <= 0.0) {
        throw std::runtime_error("DeepSeek-V4 RMSNorm configuration is invalid");
    }
    INFINICORE_NN_PARAMETER_INIT(
        weight,
        ({hidden_size_}, dtype, device));
    norm_weight_f32_ = infinicore::Tensor::ones(
        {hidden_size_}, infinicore::DataType::F32, device);
}

infinicore::Tensor DeepseekV4RMSNorm::forward(
    const infinicore::Tensor &input) const {
    if (!input || input->ndim() == 0
        || input->size(input->ndim() - 1) != hidden_size_) {
        throw std::runtime_error("DeepSeek-V4 RMSNorm input shape mismatch");
    }
    const size_t rows = input->numel() / hidden_size_;
    // 先升到 F32 做 RMSNorm（避免低精度误差累积），再乘可学习 weight
    auto normalized_f32 = infinicore::op::rms_norm(
        cast_to(input, infinicore::DataType::F32)
            ->contiguous()
            ->view({rows, hidden_size_}),
        norm_weight_f32_,
        static_cast<float>(eps_))
                              ->view(input->shape());
    auto normalized = cast_to(normalized_f32, input->dtype());
    return multiply_broadcast(
        normalized,
        static_cast<infinicore::Tensor>(weight_),
        normalized->shape());
}

// DeepseekV4GroupedLinear：分组线性层，用于 V4 的 grouped output projection。
// 输入是 [B, S, num_groups, group_input]，对每组独立做线性映射，
// 相当于 num_groups 个并行的线性层共享同一个批。
DeepseekV4GroupedLinear::DeepseekV4GroupedLinear(
    size_t in_features_per_group,
    size_t out_features_per_group,
    size_t num_groups,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : in_features_per_group_(in_features_per_group),
      out_features_per_group_(out_features_per_group),
      num_groups_(num_groups) {
    if (in_features_per_group_ == 0 || out_features_per_group_ == 0
        || num_groups_ == 0) {
        throw std::runtime_error(
            "DeepSeek-V4 grouped-linear dimensions must be non-zero");
    }
    // weight 布局：[num_groups * out_per_group, in_per_group]
    INFINICORE_NN_PARAMETER_INIT(
        weight,
        ({num_groups_ * out_features_per_group_, in_features_per_group_},
         dtype,
         device));
}

infinicore::Tensor DeepseekV4GroupedLinear::forward(
    const infinicore::Tensor &input) const {
    if (!input || input->ndim() != 4
        || input->size(2) != num_groups_
        || input->size(3) != in_features_per_group_) {
        throw std::runtime_error(
            "DeepSeek-V4 grouped linear expects [batch, sequence, groups, group_input]");
    }
    const size_t batch_size = input->size(0);
    const size_t sequence_length = input->size(1);
    const size_t tokens = batch_size * sequence_length;
    // 重排成 [groups, tokens, in_per_group]，用 batch matmul 一次算所有组
    auto grouped_input = input->view(
        {tokens, num_groups_, in_features_per_group_})
                             ->permute({1, 0, 2})
                             ->contiguous();
    auto grouped_weight = static_cast<infinicore::Tensor>(weight_)
                              ->view({num_groups_,
                                      out_features_per_group_,
                                      in_features_per_group_})
                              ->permute({0, 2, 1})
                              ->contiguous();
    return infinicore::op::matmul(grouped_input, grouped_weight)
        ->permute({1, 0, 2})
        ->contiguous()
        ->view({batch_size,
                sequence_length,
                num_groups_,
                out_features_per_group_});
}

// DeepseekV4Attention：V4 注意力模块。
// 特点：单 KV 头 MQA（num_key_value_heads=1）、512 维 head_dim、
// partial RoPE（只旋转 rope_head_dim 维）、attention sink、
// q 用 LoRA（q_a/q_b）、输出用 grouped LoRA（o_a/o_b）。
// 构造时按压缩率决定挂 HCA 还是 CSA 压缩器（两者不会同时存在）。
DeepseekV4Attention::DeepseekV4Attention(
    size_t hidden_size,
    size_t q_lora_rank,
    size_t num_attention_heads,
    size_t head_dim,
    size_t rope_head_dim,
    size_t o_groups,
    size_t o_lora_rank,
    double rms_norm_eps,
    const infinicore::DataType &dtype,
    const infinicore::Device &device,
    size_t hca_compress_rate,
    size_t csa_compress_rate,
    size_t index_num_heads,
    size_t index_head_dim,
    size_t index_topk)
    : hidden_size_(hidden_size),
      q_lora_rank_(q_lora_rank),
      num_attention_heads_(num_attention_heads),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      o_groups_(o_groups),
      o_lora_rank_(o_lora_rank),
      rms_norm_eps_(rms_norm_eps) {
    if (hidden_size_ == 0 || q_lora_rank_ == 0
        || num_attention_heads_ == 0 || head_dim_ == 0
        || rope_head_dim_ == 0 || rope_head_dim_ > head_dim_
        || rope_head_dim_ % 2 != 0 || o_groups_ == 0
        || num_attention_heads_ % o_groups_ != 0
        || o_lora_rank_ == 0 || rms_norm_eps_ <= 0.0
        || (hca_compress_rate != 0 && csa_compress_rate != 0)) {
        throw std::runtime_error(
            "DeepSeek-V4 attention configuration is invalid");
    }

    // Q 投影（LoRA 结构）：hidden → q_lora_rank（q_a）→ q_lora_rank norm → 各头 q（q_b）
    INFINICORE_NN_MODULE_INIT(
        q_a_proj, hidden_size_, q_lora_rank_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        q_a_norm, q_lora_rank_, rms_norm_eps_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        q_b_proj,
        q_lora_rank_,
        num_attention_heads_ * head_dim_,
        dtype,
        device);
    // KV 投影：hidden → 单 KV 头（head_dim），共享给所有注意力头（MQA）
    INFINICORE_NN_MODULE_INIT(
        kv_proj, hidden_size_, head_dim_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        kv_norm, head_dim_, rms_norm_eps_, dtype, device);
    // 输出投影（grouped LoRA）：各头拼接 → o_a（分 o_groups 组降维）→ o_b 拼回 hidden
    INFINICORE_NN_MODULE_INIT(
        o_a_proj,
        num_attention_heads_ * head_dim_ / o_groups_,
        o_lora_rank_,
        o_groups_,
        dtype,
        device);
    INFINICORE_NN_MODULE_INIT(
        o_b_proj, o_groups_ * o_lora_rank_, hidden_size_, dtype, device);
    // attention sink：每个注意力头一个可学习的标量，softmax 前拼到 logits 末尾
    INFINICORE_NN_PARAMETER_INIT(
        sinks,
        ({num_attention_heads_}, dtype, device));

    q_b_norm_weight_f32_ = infinicore::Tensor::ones(
        {head_dim_}, infinicore::DataType::F32, device);
    // 压缩器二选一：HCA（128 倍重压缩，无 indexer）或 CSA（4 倍 + Lightning Indexer）
    if (hca_compress_rate != 0) {
        hca_compressor_ = this->register_module<DeepseekV4HCACompressor>(
            "compressor",
            hidden_size_,
            head_dim_,
            rope_head_dim_,
            hca_compress_rate,
            rms_norm_eps_,
            dtype,
            device);
    } else if (csa_compress_rate != 0) {
        csa_compressor_ = this->register_module<DeepseekV4CSACompressor>(
            "compressor",
            hidden_size_,
            q_lora_rank_,
            head_dim_,
            rope_head_dim_,
            csa_compress_rate,
            index_num_heads,
            index_head_dim,
            index_topk,
            rms_norm_eps_,
            dtype,
            device);
    }
}

// 无权重 RMSNorm：只用全 1 权重做归一化（用于 q / kv 的归一化）。
infinicore::Tensor DeepseekV4Attention::unweighted_rms_norm_(
    const infinicore::Tensor &input) const {
    const size_t rows = input->numel() / head_dim_;
    auto normalized = infinicore::op::rms_norm(
        cast_to(input, infinicore::DataType::F32)
            ->contiguous()
            ->view({rows, head_dim_}),
        q_b_norm_weight_f32_,
        static_cast<float>(rms_norm_eps_))
                          ->view(input->shape());
    return cast_to(normalized, input->dtype());
}

// partial RoPE：只对 head_dim 的后 rope_head_dim 维做旋转，其余维不动。
// conjugate=true 时旋转方向取反（用于输出侧抵消查询侧的旋转，实现 RoPE 的共轭）。
// 旋转实现：把 rope 段看成 [pair_count, 2]（偶/奇通道对），
// 新值 = [x0*cos - x1*sin, x0*sin + x1*cos]。
infinicore::Tensor DeepseekV4Attention::apply_partial_rope_(
    const infinicore::Tensor &input,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin,
    bool conjugate) const {
    if (input->ndim() != 4 || input->size(3) != head_dim_) {
        throw std::runtime_error(
            "DeepSeek-V4 partial RoPE expects [batch, sequence, heads, head_dim]");
    }
    const size_t batch_size = input->size(0);
    const size_t sequence_length = input->size(1);
    const size_t num_heads = input->size(2);
    const size_t pair_count = rope_head_dim_ / 2;
    const size_t nope_dim = head_dim_ - rope_head_dim_;

    // 不参与旋转的前 nope_dim 维，原样保留
    infinicore::Tensor nope;
    if (nope_dim != 0) {
        nope = input->narrow({{3, 0, nope_dim}})->contiguous();
    }
    // 参与旋转的 rope_head_dim 维，重排成 [pair_count, 2]
    auto rope = input->narrow({{3, nope_dim, rope_head_dim_}})
                    ->contiguous()
                    ->view({batch_size,
                            sequence_length,
                            num_heads,
                            pair_count,
                            2});
    // InfiniCore 逐元素 kernel 需要连续 narrow 视图；先物化偶/奇通道再旋转。
    // rotated = [-x1, x0]（第二通道取反放到第一位）
    auto first = rope->narrow({{4, 0, 1}})->contiguous();
    auto second = rope->narrow({{4, 1, 1}})->contiguous();
    auto rotated = infinicore::op::cat(
        {infinicore::op::mul_scalar(second, -1.0), first}, 4);

    const infinicore::Shape paired_shape{
        batch_size, sequence_length, num_heads, pair_count, 2};
    auto cos_f32 = cast_to(cos, infinicore::DataType::F32)
                       ->unsqueeze(2)
                       ->unsqueeze(4);
    auto sin_f32 = cast_to(sin, infinicore::DataType::F32)
                       ->unsqueeze(2)
                       ->unsqueeze(4);
    // 共轭时 sin 取反，实现反向旋转
    if (conjugate) {
        sin_f32 = infinicore::op::mul_scalar(sin_f32, -1.0);
    }
    // mixed = x * cos + rotated * sin（RoPE 的标准旋转公式）
    auto mixed = infinicore::op::add(
        multiply_broadcast(
            cast_to(rope, infinicore::DataType::F32), cos_f32, paired_shape),
        multiply_broadcast(
            cast_to(rotated, infinicore::DataType::F32), sin_f32, paired_shape));
    auto rotated_typed = cast_to(
        mixed->view({batch_size,
                     sequence_length,
                     num_heads,
                     rope_head_dim_}),
        input->dtype());
    // 把未旋转的 nope 段拼回前面，还原完整 head_dim
    if (nope_dim == 0) {
        return rotated_typed;
    }
    return infinicore::op::cat({nope, rotated_typed}, 3);
}

// causal_bias_：构造注意力 logits 的掩码偏置（加到 score 上）。
// 每个 query 只能看它自己及之前的 key（causal），且只看滑窗内（sliding_window）的 key；
// 不可见的位置置为 -inf，使 softmax 后概率为 0。
infinicore::Tensor DeepseekV4Attention::causal_bias_(
    size_t query_length,
    size_t kv_length,
    size_t past_length,
    size_t sliding_window,
    const infinicore::DataType &dtype,
    const infinicore::Device &device) const {
    if (kv_length != past_length + query_length) {
        throw std::runtime_error(
            "DeepSeek-V4 attention KV length does not match past + query");
    }
    std::vector<float> values(query_length * kv_length, 0.0f);
    for (size_t query = 0; query < query_length; ++query) {
        const size_t query_kv_index = past_length + query;  // 该 query 对应的 key 位置
        // 滑窗下界：当前 query 往前 sliding_window 个 key；sliding_window==0 表示全看
        const size_t first_visible = sliding_window == 0
            ? 0
            : (query_kv_index + 1 > sliding_window
                   ? query_kv_index + 1 - sliding_window
                   : 0);
        for (size_t key = 0; key < kv_length; ++key) {
            // 超出 causal 范围或滑窗范围 → -inf 屏蔽
            if (key > query_kv_index || key < first_visible) {
                values[query * kv_length + key] =
                    -std::numeric_limits<float>::infinity();
            }
        }
    }
    auto cpu = infinicore::Tensor::from_blob(
        values.data(),
        {1, 1, query_length, kv_length},
        infinicore::DataType::F32,
        infinicore::Device::cpu());
    return cast_to(cpu->to(device), dtype);
}

// forward：完整（无压缩）注意力入口——用于纯 dense 测试场景。
// 调 project_qkv 出 q/kv，再用 attention_from_projections_ 算注意力。
// 注意：正常推理走的是下面三个 forward_sliding / forward_hca / forward_csa。
DeepseekV4AttentionOutput DeepseekV4Attention::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin) const {
    validate_input(
        hidden_states, cos, sin, hidden_size_, rope_head_dim_);
    auto projections = project_qkv(hidden_states, cos, sin);
    return attention_from_projections_(
        projections.query, projections.kv, cos, sin, 0, 0);
}

// forward_sliding：滑窗注意力（sliding_attention 层用）。
// 只有滑窗 KV（最近的 sliding_window 个 token），没有压缩分支。
// 返回的 next_cache 会截断到滑窗大小，只保留窗口内的 KV 供下次增量推理。
DeepseekV4SlidingAttentionOutput DeepseekV4Attention::forward_sliding(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin,
    const std::optional<infinicore::Tensor> &past_kv,
    size_t sliding_window) const {
    if (sliding_window < 2) {
        throw std::runtime_error(
            "DeepSeek-V4 sliding attention requires sliding_window >= 2");
    }
    auto projections = project_qkv(hidden_states, cos, sin);
    const size_t batch_size = hidden_states->size(0);
    size_t past_length = 0;
    // 本次的 kv 拼上历史滑窗 KV
    auto combined_kv = projections.kv;
    if (past_kv.has_value()) {
        const auto &past = past_kv.value();
        if (!past || past->ndim() != 4
            || past->size(0) != batch_size || past->size(2) != 1
            || past->size(3) != head_dim_
            || past->dtype() != projections.kv->dtype()
            || past->device() != projections.kv->device()
            || past->size(1) > sliding_window - 1) {
            throw std::runtime_error(
                "DeepSeek-V4 sliding attention received an incompatible KV cache");
        }
        past_length = past->size(1);
        combined_kv = infinicore::op::cat(
            {past, projections.kv}, 1);
    }

    auto attention = attention_from_projections_(
        projections.query,
        combined_kv,
        cos,
        sin,
        past_length,
        sliding_window);
    // 滑窗只保留最近 sliding_window-1 个 KV（+本次的），超出窗口的丢弃
    const size_t combined_length = combined_kv->size(1);
    const size_t retained_length = std::min(
        combined_length, sliding_window - 1);
    auto next_cache = combined_kv
                          ->narrow({{1,
                                    combined_length - retained_length,
                                    retained_length}})
                          ->contiguous();
    return {attention.output, attention.attention_weights, next_cache};
}

// forward_hca：重压缩注意力（heavily_compressed_attention 层用）。
// KV 由两部分拼成：滑窗 KV + HCA 压缩出的长程 KV（128 倍压缩，全部条目参与）。
// hca_state 跨 token 累积压缩缓冲；block_bias 表示压缩条目的可见性/块掩码。
DeepseekV4SlidingAttentionOutput DeepseekV4Attention::forward_hca(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &query_cos,
    const infinicore::Tensor &query_sin,
    const infinicore::Tensor &compressed_cos,
    const infinicore::Tensor &compressed_sin,
    const infinicore::Tensor &position_ids,
    const std::optional<infinicore::Tensor> &past_sliding_kv,
    DeepseekV4HCAState *hca_state,
    size_t sliding_window) const {
    if (!hca_compressor_) {
        throw std::runtime_error(
            "DeepSeek-V4 HCA attention was constructed without a compressor");
    }
    if (sliding_window < 2) {
        throw std::runtime_error(
            "DeepSeek-V4 HCA attention requires sliding_window >= 2");
    }
    // Q/K/V 投影（同 sliding，用查询侧 RoPE）
    auto projections = project_qkv(
        hidden_states, query_cos, query_sin);
    const size_t batch_size = hidden_states->size(0);
    const size_t query_length = hidden_states->size(1);
    size_t past_length = 0;
    // 拼历史滑窗 KV
    auto sliding_kv = projections.kv;
    if (past_sliding_kv.has_value()) {
        const auto &past = past_sliding_kv.value();
        if (!past || past->ndim() != 4
            || past->size(0) != batch_size || past->size(2) != 1
            || past->size(3) != head_dim_
            || past->dtype() != projections.kv->dtype()
            || past->device() != projections.kv->device()
            || past->size(1) > sliding_window - 1) {
            throw std::runtime_error(
                "DeepSeek-V4 HCA attention received an incompatible sliding cache");
        }
        past_length = past->size(1);
        sliding_kv = infinicore::op::cat(
            {past, projections.kv}, 1);
    }

    // 用 HCA 压缩器把远距离历史压成长程 KV（含块掩码）
    auto compressed = hca_compressor_->forward(
        hidden_states,
        compressed_cos,
        compressed_sin,
        position_ids,
        hca_state);
    // 压缩 KV 形状从 [B, entries, 1, head] 重排成 [B, entries, head] 与滑窗 KV 一致
    auto long_range_kv = compressed.compressed_kv
                             ->permute({0, 2, 1, 3})
                             ->contiguous();
    // 拼成完整 KV：滑窗部分 + 长程压缩部分
    auto combined_kv = long_range_kv->size(1) == 0
        ? sliding_kv
        : infinicore::op::cat({sliding_kv, long_range_kv}, 1);

    // 滑窗部分的掩码（causal + 滑窗）
    auto sliding_bias = causal_bias_(
        query_length,
        sliding_kv->size(1),
        past_length,
        sliding_window,
        combined_kv->dtype(),
        combined_kv->device());
    sliding_bias = broadcast_to_shape(
        sliding_bias,
        {batch_size, 1, query_length, sliding_kv->size(1)});

    // 长程部分的掩码：有块掩码用压缩器给的，否则全可见（零）
    infinicore::Tensor combined_bias = sliding_bias;
    if (long_range_kv->size(1) != 0) {
        infinicore::Tensor long_range_bias;
        if (compressed.block_bias.has_value()) {
            long_range_bias = compressed.block_bias.value();
        } else {
            long_range_bias = infinicore::Tensor::zeros(
                {batch_size, 1, query_length, long_range_kv->size(1)},
                combined_kv->dtype(),
                combined_kv->device());
        }
        combined_bias = infinicore::op::cat(
            {sliding_bias, long_range_bias}, 3);
    }

    // 拼接后的完整 KV 上算注意力（注意这里不再传 sliding_window，因为掩码已手工拼好）
    auto attention = attention_from_projections_(
        projections.query,
        combined_kv,
        query_cos,
        query_sin,
        0,
        0,
        combined_bias);
    // 更新滑窗缓存（截断到窗口大小）
    const size_t sliding_length = sliding_kv->size(1);
    const size_t retained_length = std::min(
        sliding_length, sliding_window - 1);
    auto next_sliding_cache = sliding_kv
                                  ->narrow({{1,
                                            sliding_length - retained_length,
                                            retained_length}})
                                  ->contiguous();
    return {
        attention.output,
        attention.attention_weights,
        next_sliding_cache};
}

// forward_csa：压缩稀疏注意力（compressed_sparse_attention 层用）。
// 与 HCA 结构类似，但压缩器是 CSA（4 倍压缩 + Lightning Indexer 稀疏挑选 top-k 条目）。
// 额外传入 projections.q_residual 给 indexer 做打分输入。
DeepseekV4SlidingAttentionOutput DeepseekV4Attention::forward_csa(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &query_cos,
    const infinicore::Tensor &query_sin,
    const infinicore::Tensor &compressed_cos,
    const infinicore::Tensor &compressed_sin,
    const infinicore::Tensor &position_ids,
    const std::optional<infinicore::Tensor> &past_sliding_kv,
    DeepseekV4CSAState *csa_state,
    size_t sliding_window) const {
    if (!csa_compressor_) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA attention was constructed without a compressor");
    }
    if (sliding_window < 2) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA attention requires sliding_window >= 2");
    }
    auto projections = project_qkv(
        hidden_states, query_cos, query_sin);
    const size_t batch_size = hidden_states->size(0);
    const size_t query_length = hidden_states->size(1);
    size_t past_length = 0;
    auto sliding_kv = projections.kv;
    if (past_sliding_kv.has_value()) {
        const auto &past = past_sliding_kv.value();
        if (!past || past->ndim() != 4
            || past->size(0) != batch_size || past->size(2) != 1
            || past->size(3) != head_dim_
            || past->dtype() != projections.kv->dtype()
            || past->device() != projections.kv->device()
            || past->size(1) > sliding_window - 1) {
            throw std::runtime_error(
                "DeepSeek-V4 CSA attention received an incompatible sliding cache");
        }
        past_length = past->size(1);
        sliding_kv = infinicore::op::cat(
            {past, projections.kv}, 1);
    }

    // CSA 压缩器前向（内部含 Lightning Indexer 稀疏选择）
    auto compressed = csa_compressor_->forward(
        hidden_states,
        projections.q_residual,
        query_cos,
        query_sin,
        compressed_cos,
        compressed_sin,
        position_ids,
        csa_state);
    auto long_range_kv = compressed.compressed_kv
                             ->permute({0, 2, 1, 3})
                             ->contiguous();
    auto combined_kv = long_range_kv->size(1) == 0
        ? sliding_kv
        : infinicore::op::cat({sliding_kv, long_range_kv}, 1);

    // 滑窗掩码 + 长程掩码（CSA 的 block_bias 是 indexer 选出的条目掩码）
    auto sliding_bias = causal_bias_(
        query_length,
        sliding_kv->size(1),
        past_length,
        sliding_window,
        combined_kv->dtype(),
        combined_kv->device());
    sliding_bias = broadcast_to_shape(
        sliding_bias,
        {batch_size, 1, query_length, sliding_kv->size(1)});
    auto combined_bias = long_range_kv->size(1) == 0
        ? sliding_bias
        : infinicore::op::cat(
              {sliding_bias, compressed.block_bias}, 3);

    auto attention = attention_from_projections_(
        projections.query,
        combined_kv,
        query_cos,
        query_sin,
        0,
        0,
        combined_bias);
    const size_t sliding_length = sliding_kv->size(1);
    const size_t retained_length = std::min(
        sliding_length, sliding_window - 1);
    auto next_sliding_cache = sliding_kv
                                  ->narrow({{1,
                                            sliding_length - retained_length,
                                            retained_length}})
                                  ->contiguous();
    return {
        attention.output,
        attention.attention_weights,
        next_sliding_cache};
}

// attention_from_projections_：给定已投影好的 query 和 KV，完成注意力计算：
//   scores = Q·Kᵀ/√d → +bias（掩码）→ 拼 sink → softmax → 加权 V
//   → 输出侧反向 RoPE（conjugate）→ grouped o 投影
// 三种注意力路径（sliding/HCA/CSA）最终都汇聚到这里，只是传入的 KV 和掩码不同。
DeepseekV4AttentionOutput DeepseekV4Attention::attention_from_projections_(
    const infinicore::Tensor &query,
    const infinicore::Tensor &kv,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin,
    size_t past_length,
    size_t sliding_window,
    const std::optional<infinicore::Tensor> &attention_bias) const {
    if (!query || !kv || query->ndim() != 4 || kv->ndim() != 4
        || query->size(0) != kv->size(0)
        || query->size(2) != num_attention_heads_
        || query->size(3) != head_dim_ || kv->size(2) != 1
        || kv->size(3) != head_dim_
        || (!attention_bias.has_value()
            && kv->size(1) != past_length + query->size(1))) {
        throw std::runtime_error(
            "DeepSeek-V4 attention projection shapes are incompatible");
    }
    const size_t batch_size = query->size(0);
    const size_t query_length = query->size(1);
    const size_t kv_length = kv->size(1);

    // Q·Kᵀ：query [B,H,Q,head] → [B, H*Q, head]，KV 共享单头 → [B, head, K]
    // 缩放 1/√head_dim
    auto query_for_scores = query->permute({0, 2, 1, 3})
                                ->contiguous()
                                ->view({batch_size,
                                        num_attention_heads_ * query_length,
                                        head_dim_});
    auto kv_shared = kv->squeeze(2)->contiguous();
    auto scores = infinicore::op::matmul(
                      query_for_scores,
                      kv_shared->permute({0, 2, 1})->contiguous(),
                      1.0f / std::sqrt(static_cast<float>(head_dim_)))
                      ->view({batch_size,
                              num_attention_heads_,
                              query_length,
                              kv_length});
    // 掩码偏置：调用方给了就用作掩码（HCA/CSA 是手工拼好的滑窗+长程掩码），
    // 否则现算（纯滑窗的 causal + 滑窗掩码）
    infinicore::Tensor bias;
    if (attention_bias.has_value()) {
        bias = attention_bias.value();
        if (!bias || bias->ndim() != 4
            || bias->size(0) != batch_size || bias->size(1) != 1
            || bias->size(2) != query_length
            || bias->size(3) != kv_length
            || bias->dtype() != scores->dtype()
            || bias->device() != scores->device()) {
            throw std::runtime_error(
                "DeepSeek-V4 explicit attention bias shape is incompatible");
        }
    } else {
        bias = causal_bias_(
            query_length,
            kv_length,
            past_length,
            sliding_window,
            scores->dtype(),
            scores->device());
    }
    scores = add_broadcast(scores, bias, scores->shape());

    // attention sink：把每个头一个可学习标量拼到 logits 末尾，
    // 让 softmax 时每个 query 始终有一个"吸收"项，防止极端情况下概率退化。
    auto sink = static_cast<infinicore::Tensor>(sinks_)
                    ->view({1, num_attention_heads_, 1, 1});
    sink = broadcast_to_shape(
        sink,
        {batch_size, num_attention_heads_, query_length, 1});
    auto combined_logits = infinicore::op::cat({scores, sink}, 3);
    auto probabilities = infinicore::op::softmax(combined_logits, -1);
    // 去掉 sink 那一列，得到真正的注意力权重
    auto attention_weights = probabilities
                                 ->narrow({{3, 0, kv_length}})
                                 ->contiguous();

    // 加权求和 V：attn_weights [B, H*Q, K] × KV [B, K, head] → 输出 [B, H, Q, head]
    auto attention_output = infinicore::op::matmul(
                                attention_weights->view(
                                    {batch_size,
                                     num_attention_heads_ * query_length,
                                     kv_length}),
                                kv_shared)
                                ->view({batch_size,
                                        num_attention_heads_,
                                        query_length,
                                        head_dim_})
                                ->permute({0, 2, 1, 3})
                                ->contiguous();
    // 输出侧用共轭 RoPE 抵消查询侧旋转，保持相对位置编码的正确性
    attention_output = apply_partial_rope_(
        attention_output, cos, sin, true);

    // grouped output projection：先把所有头拼成 [B, Q, 全部head*head_dim]，
    // 分 o_groups 组经 o_a 降维到 o_lora_rank，再 o_b 拼回 hidden_size
    auto grouped = attention_output->view(
        {batch_size,
         query_length,
         o_groups_,
         num_attention_heads_ * head_dim_ / o_groups_});
    grouped = o_a_proj_->forward(grouped)
                  ->view({batch_size,
                          query_length,
                          o_groups_ * o_lora_rank_});
    auto output = o_b_proj_->forward(grouped);
    return {output, attention_weights};
}

// project_qkv：把 hidden_states 投影成 Q/KV，并完成各自的归一化与 RoPE。
// 返回 q_residual（CSA indexer 打分用）、query、kv。
DeepseekV4AttentionProjections DeepseekV4Attention::project_qkv(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin) const {
    validate_input(
        hidden_states, cos, sin, hidden_size_, rope_head_dim_);
    const size_t batch_size = hidden_states->size(0);
    const size_t sequence_length = hidden_states->size(1);

    // Q（LoRA 两段式）：hidden → q_a → q_a_norm → q_b → [B,S,H,head]
    auto q_residual = q_a_proj_->forward(hidden_states);
    q_residual = q_a_norm_->forward(q_residual);
    auto query = q_b_proj_->forward(q_residual)
                     ->view({batch_size,
                             sequence_length,
                             num_attention_heads_,
                             head_dim_});
    query = unweighted_rms_norm_(query);          // 无权重 RMSNorm
    query = apply_partial_rope_(query, cos, sin, false);  // 正向 RoPE

    // KV（单头 MQA）：hidden → kv_proj → kv_norm → [B,S,1,head] + 正向 RoPE
    auto kv = kv_proj_->forward(hidden_states);
    kv = kv_norm_->forward(kv)
             ->view({batch_size, sequence_length, 1, head_dim_});
    kv = apply_partial_rope_(kv, cos, sin, false);
    return {q_residual, query, kv};
}

} // namespace infinilm::models::deepseek_v4
