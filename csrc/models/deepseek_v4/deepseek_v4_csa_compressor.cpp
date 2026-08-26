#include "deepseek_v4_csa_compressor.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/relu.hpp>
#include <infinicore/ops/softmax.hpp>
#include <infinicore/ops/sum.hpp>
#include <infinicore/ops/topk.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

// cast_to：转 dtype；相同则直接返回。
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

infinicore::Tensor broadcast_to_shape(const infinicore::Tensor &input,
                                      const infinicore::Shape &shape) {
    if (input->shape() == shape) {
        return input;
    }
    return infinicore::op::broadcast_to(
        input, std::vector<int64_t>(shape.begin(), shape.end()));
}

// 生成全 -inf 的张量（用于掩码初始值）。
infinicore::Tensor negative_infinity(
    const infinicore::Shape &shape,
    const infinicore::DataType &dtype,
    const infinicore::Device &device) {
    std::vector<float> values(
        std::accumulate(
            shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>()),
        -std::numeric_limits<float>::infinity());
    auto cpu = infinicore::Tensor::from_blob(
        values.data(),
        shape,
        infinicore::DataType::F32,
        infinicore::Device::cpu());
    return cast_to(cpu->to(device), dtype);
}

// 把 position_ids 拷到 CPU 的 int64 向量（掩码构造需要逐位置循环）。
std::vector<int64_t> positions_to_host(
    const infinicore::Tensor &position_ids) {
    if (!position_ids || position_ids->ndim() != 2) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA position_ids must be [batch, sequence]");
    }
    auto cpu = position_ids->to(infinicore::Device::cpu())->contiguous();
    std::vector<int64_t> result(cpu->numel());
    if (cpu->dtype() == infinicore::DataType::I64) {
        std::memcpy(result.data(), cpu->data(), cpu->nbytes());
    } else if (cpu->dtype() == infinicore::DataType::I32) {
        const auto *source =
            reinterpret_cast<const int32_t *>(cpu->data());
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = source[i];
        }
    } else {
        throw std::runtime_error(
            "DeepSeek-V4 CSA position_ids must use I32 or I64");
    }
    if (std::any_of(result.begin(), result.end(), [](int64_t position) {
            return position < 0;
        })) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA position_ids must be non-negative");
    }
    return result;
}

// partial RoPE（同 HCA/attention，输入 4 维 [B,S,heads,head_dim]）。
infinicore::Tensor apply_partial_rope(
    const infinicore::Tensor &input,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin,
    size_t head_dim,
    size_t rope_head_dim) {
    if (!input || input->ndim() != 4 || input->size(3) != head_dim
        || !cos || !sin || cos->ndim() != 3 || sin->ndim() != 3
        || cos->shape() != sin->shape()
        || cos->size(0) != input->size(0)
        || cos->size(1) != input->size(1)
        || cos->size(2) * 2 != rope_head_dim) {
        throw std::runtime_error("DeepSeek-V4 CSA RoPE shape mismatch");
    }
    const size_t batch_size = input->size(0);
    const size_t sequence_length = input->size(1);
    const size_t num_heads = input->size(2);
    const size_t pair_count = rope_head_dim / 2;
    const size_t nope_dim = head_dim - rope_head_dim;

    infinicore::Tensor nope;
    if (nope_dim != 0) {
        nope = input->narrow({{3, 0, nope_dim}})->contiguous();
    }
    auto rope = input->narrow({{3, nope_dim, rope_head_dim}})
                    ->contiguous()
                    ->view({batch_size,
                            sequence_length,
                            num_heads,
                            pair_count,
                            2});
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
    auto mixed = infinicore::op::add(
        infinicore::op::mul(
            cast_to(rope, infinicore::DataType::F32),
            broadcast_to_shape(cos_f32, paired_shape)),
        infinicore::op::mul(
            cast_to(rotated, infinicore::DataType::F32),
            broadcast_to_shape(sin_f32, paired_shape)));
    auto rotated_typed = cast_to(
        mixed->view({batch_size,
                     sequence_length,
                     num_heads,
                     rope_head_dim}),
        input->dtype());
    if (nope_dim == 0) {
        return rotated_typed;
    }
    return infinicore::op::cat({nope, rotated_typed}, 3);
}

// 校验 CSA 流状态的一致性和形状。
void validate_stream_state(const DeepseekV4CSAStreamState &state,
                           size_t batch_size,
                           size_t projected_dim,
                           size_t head_dim,
                           size_t compress_rate,
                           const infinicore::Tensor &reference) {
    auto validate = [&](const infinicore::Tensor &tensor,
                        size_t width,
                        const char *name) {
        if (tensor
            && (tensor->ndim() != 3 || tensor->size(0) != batch_size
                || tensor->size(2) != width
                || tensor->dtype() != reference->dtype()
                || tensor->device() != reference->device())) {
            throw std::runtime_error(
                std::string("DeepSeek-V4 CSA incompatible state tensor: ")
                + name);
        }
    };
    validate(state.buffer_kv, projected_dim, "buffer_kv");
    validate(state.buffer_gate, projected_dim, "buffer_gate");
    validate(state.overlap_kv, head_dim, "overlap_kv");
    validate(state.overlap_gate, head_dim, "overlap_gate");
    validate(state.compressed_kv, head_dim, "compressed_kv");
    if (static_cast<bool>(state.buffer_kv)
            != static_cast<bool>(state.buffer_gate)
        || (state.buffer_kv
            && (state.buffer_kv->size(1) != state.buffer_gate->size(1)
                || state.buffer_kv->size(1) >= compress_rate))
        || static_cast<bool>(state.overlap_kv)
               != static_cast<bool>(state.overlap_gate)
        || (state.overlap_kv
            && (state.overlap_kv->size(1) != compress_rate
                || state.overlap_gate->size(1) != compress_rate))
        || (state.compressed_kv
            && state.compressed_kv->size(1) != state.entry_count)) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA stream state is inconsistent");
    }
}

// 一次压缩流的输出：全部压缩条目 + 本次新压缩的条目。
struct CompressedStreamOutput {
    infinicore::Tensor compressed_kv;
    infinicore::Tensor new_compressed_kv;
};

// compress_stream：CSA/HCA 共用的"压缩流"核心（与 HCA 的 forward 类似）。
// 关键不同：CSA 用【滑动窗口池化】——每个窗口的 KV 会"借用上一个窗口的最后
// compress_rate 个 token"作为前一跳（overlap），实现窗口间的信息传递。
// 此外 CSA 的 kv/gate 投影宽度是 2*head_dim（前 head_dim 是当前段，后 head_dim 是下一跳），
// 池化时用"当前段 + 上一跳"拼成 2*rate 长度一起 softmax。
CompressedStreamOutput compress_stream(
    const infinicore::Tensor &hidden_states,
    const std::shared_ptr<DeepseekV4Linear> &kv_proj,
    const std::shared_ptr<DeepseekV4Linear> &gate_proj,
    const std::shared_ptr<DeepseekV4RMSNorm> &kv_norm,
    const infinicore::Tensor &position_bias,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin,
    size_t head_dim,
    size_t rope_head_dim,
    size_t compress_rate,
    DeepseekV4CSAStreamState *state) {
    const size_t batch_size = hidden_states->size(0);
    // ① 投影（宽度 2*head_dim）
    auto source_kv = kv_proj->forward(hidden_states);
    auto source_gate = gate_proj->forward(hidden_states);
    const size_t projected_dim = 2 * head_dim;
    // ② 拼上历史 buffer
    if (state != nullptr) {
        validate_stream_state(
            *state,
            batch_size,
            projected_dim,
            head_dim,
            compress_rate,
            source_kv);
        if (state->buffer_kv && state->buffer_kv->size(1) != 0) {
            source_kv = infinicore::op::cat(
                {state->buffer_kv, source_kv}, 1);
            source_gate = infinicore::op::cat(
                {state->buffer_gate, source_gate}, 1);
        }
    }

    // ③ 可完整窗口数
    const size_t source_length = source_kv->size(1);
    const size_t usable =
        source_length / compress_rate * compress_rate;
    const size_t new_entries = usable / compress_rate;
    if (!cos || !sin || cos->ndim() != 3 || sin->ndim() != 3
        || cos->shape() != sin->shape() || cos->size(0) != batch_size
        || cos->size(1) != new_entries
        || cos->size(2) * 2 != rope_head_dim) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA cos/sin count does not match closed windows");
    }

    // ④ 余下存回 buffer
    if (state != nullptr) {
        const size_t remainder = source_length - usable;
        state->buffer_kv = source_kv
                               ->narrow({{1, usable, remainder}})
                               ->contiguous();
        state->buffer_gate = source_gate
                                 ->narrow({{1, usable, remainder}})
                                 ->contiguous();
    }

    // ⑤ 滑动窗口池化
    infinicore::Tensor new_compressed;
    if (new_entries == 0) {
        new_compressed = infinicore::Tensor::empty(
            {batch_size, 0, head_dim},
            source_kv->dtype(),
            source_kv->device());
    } else {
        // 重排成 [new_entries, rate, 2*head_dim]
        auto chunk_kv = source_kv
                            ->narrow({{1, 0, usable}})
                            ->contiguous()
                            ->view({batch_size,
                                    new_entries,
                                    compress_rate,
                                    projected_dim});
        auto chunk_gate = source_gate
                              ->narrow({{1, 0, usable}})
                              ->contiguous()
                              ->view({batch_size,
                                      new_entries,
                                      compress_rate,
                                      projected_dim});
        // + position_bias
        chunk_gate = infinicore::op::add(
            chunk_gate,
            broadcast_to_shape(
                position_bias->view(
                    {1, 1, compress_rate, projected_dim}),
                chunk_gate->shape()));

        // 每个窗口的后 head_dim 是"当前段"，前 head_dim 是"下一跳"（借给下一个窗口）
        auto current_kv = chunk_kv
                              ->narrow({{3, head_dim, head_dim}})
                              ->contiguous();
        auto current_gate = chunk_gate
                                ->narrow({{3, head_dim, head_dim}})
                                ->contiguous();
        // "上一跳"：来自 state->overlap（上个窗口留下的前 head_dim）
        infinicore::Tensor first_kv;
        infinicore::Tensor first_gate;
        if (state != nullptr && state->overlap_kv) {
            first_kv = state->overlap_kv->unsqueeze(1);
            first_gate = state->overlap_gate->unsqueeze(1);
        } else {
            first_kv = infinicore::Tensor::zeros(
                {batch_size, 1, compress_rate, head_dim},
                source_kv->dtype(),
                source_kv->device());
            first_gate = negative_infinity(
                {batch_size, 1, compress_rate, head_dim},
                source_gate->dtype(),
                source_gate->device());
        }
        // previous = 第一个窗口的上一跳 + 其余窗口的"自己的前 head_dim"
        auto previous_kv = first_kv;
        auto previous_gate = first_gate;
        if (new_entries > 1) {
            previous_kv = infinicore::op::cat(
                {first_kv,
                 chunk_kv
                     ->narrow({{1, 0, new_entries - 1},
                               {3, 0, head_dim}})
                     ->contiguous()},
                1);
            previous_gate = infinicore::op::cat(
                {first_gate,
                 chunk_gate
                     ->narrow({{1, 0, new_entries - 1},
                               {3, 0, head_dim}})
                     ->contiguous()},
                1);
        }
        // window = [previous(上一跳, rate个), current(当前段, rate个)] 拼成 2*rate 长
        auto window_kv = infinicore::op::cat(
            {previous_kv, current_kv}, 2);
        auto window_gate = infinicore::op::cat(
            {previous_gate, current_gate}, 2);
        // 在 2*rate 长度上 softmax，加权池化
        auto weights = cast_to(
            infinicore::op::softmax(
                cast_to(window_gate, infinicore::DataType::F32), 2),
            window_kv->dtype());
        auto pooled = infinicore::op::sum(
            infinicore::op::mul(window_kv, weights), {2}, false);
        // 归一化 + RoPE
        new_compressed = kv_norm->forward(pooled);
        new_compressed = apply_partial_rope(
                             new_compressed->unsqueeze(2),
                             cos,
                             sin,
                             head_dim,
                             rope_head_dim)
                             ->squeeze(2);

        // 更新 overlap：最后一个窗口的前 head_dim 留给下次
        if (state != nullptr) {
            state->overlap_kv = chunk_kv
                                    ->narrow({{1, new_entries - 1, 1},
                                              {3, 0, head_dim}})
                                    ->squeeze(1)
                                    ->contiguous();
            state->overlap_gate = chunk_gate
                                      ->narrow({{1, new_entries - 1, 1},
                                                {3, 0, head_dim}})
                                      ->squeeze(1)
                                      ->contiguous();
        }
    }

    // ⑥ 追加进累积条目
    auto all_compressed = new_compressed;
    if (state != nullptr) {
        if (!state->compressed_kv) {
            state->compressed_kv = new_compressed;
        } else if (new_entries != 0) {
            state->compressed_kv = infinicore::op::cat(
                {state->compressed_kv, new_compressed}, 1);
        }
        state->entry_count += new_entries;
        all_compressed = state->compressed_kv;
    }
    return {all_compressed, new_compressed};
}

// 构造 indexer 打分的 causal 掩码（同 HCA 的块掩码）：条目号 >= 阈值的不可见。
infinicore::Tensor causal_score_bias(
    const infinicore::Tensor &position_ids,
    size_t compressed_length,
    size_t compress_rate,
    const infinicore::DataType &dtype,
    const infinicore::Device &device) {
    const size_t batch_size = position_ids->size(0);
    const size_t sequence_length = position_ids->size(1);
    if (compressed_length == 0) {
        return infinicore::Tensor::empty(
            {batch_size, sequence_length, 0}, dtype, device);
    }
    const auto positions = positions_to_host(position_ids);
    std::vector<float> values(
        batch_size * sequence_length * compressed_length, 0.0f);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t query = 0; query < sequence_length; ++query) {
            const size_t threshold =
                (static_cast<size_t>(
                     positions[batch * sequence_length + query])
                 + 1)
                / compress_rate;
            for (size_t entry = threshold;
                 entry < compressed_length;
                 ++entry) {
                values[(batch * sequence_length + query)
                           * compressed_length
                       + entry] =
                    -std::numeric_limits<float>::infinity();
            }
        }
    }
    auto cpu = infinicore::Tensor::from_blob(
        values.data(),
        {batch_size, sequence_length, compressed_length},
        infinicore::DataType::F32,
        infinicore::Device::cpu());
    return cast_to(cpu->to(device), dtype);
}

// 把 top-k 选出的条目号中"不可见的"标记为 -1（causal 之后仍可能选到看不见的条目）。
infinicore::Tensor mark_invalid_topk(
    const infinicore::Tensor &indices,
    const infinicore::Tensor &position_ids,
    size_t compress_rate) {
    if (indices->size(2) == 0) {
        return indices;
    }
    auto indices_cpu = indices->to(infinicore::Device::cpu())->contiguous();
    std::vector<int32_t> values(indices_cpu->numel());
    std::memcpy(values.data(), indices_cpu->data(), indices_cpu->nbytes());
    const auto positions = positions_to_host(position_ids);
    const size_t batch_size = indices->size(0);
    const size_t sequence_length = indices->size(1);
    const size_t top_k = indices->size(2);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t query = 0; query < sequence_length; ++query) {
            const size_t threshold =
                (static_cast<size_t>(
                     positions[batch * sequence_length + query])
                 + 1)
                / compress_rate;
            for (size_t rank = 0; rank < top_k; ++rank) {
                auto &index = values[
                    (batch * sequence_length + query) * top_k + rank];
                if (index < 0 || static_cast<size_t>(index) >= threshold) {
                    index = -1;
                }
            }
        }
    }
    auto cpu = infinicore::Tensor::from_blob(
        values.data(), indices->shape(), infinicore::DataType::I32,
        infinicore::Device::cpu());
    return cpu->to(indices->device());
}

// 把选中的条目号转成块掩码：选中的条目位置为 0（可见），其余 -inf（屏蔽）。
infinicore::Tensor make_block_bias(
    const infinicore::Tensor &indices,
    size_t compressed_length,
    const infinicore::DataType &dtype,
    const infinicore::Device &device) {
    const size_t batch_size = indices->size(0);
    const size_t sequence_length = indices->size(1);
    if (compressed_length == 0) {
        return infinicore::Tensor::empty(
            {batch_size, 1, sequence_length, 0}, dtype, device);
    }
    auto indices_cpu = indices->to(infinicore::Device::cpu())->contiguous();
    std::vector<int32_t> selected(indices_cpu->numel());
    std::memcpy(selected.data(), indices_cpu->data(), indices_cpu->nbytes());
    std::vector<float> values(
        batch_size * sequence_length * compressed_length,
        -std::numeric_limits<float>::infinity());
    const size_t top_k = indices->size(2);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t query = 0; query < sequence_length; ++query) {
            for (size_t rank = 0; rank < top_k; ++rank) {
                const int32_t index = selected[
                    (batch * sequence_length + query) * top_k + rank];
                if (index >= 0
                    && static_cast<size_t>(index) < compressed_length) {
                    values[(batch * sequence_length + query)
                               * compressed_length
                           + static_cast<size_t>(index)] = 0.0f;
                }
            }
        }
    }
    auto cpu = infinicore::Tensor::from_blob(
        values.data(),
        {batch_size, 1, sequence_length, compressed_length},
        infinicore::DataType::F32,
        infinicore::Device::cpu());
    return cast_to(cpu->to(device), dtype);
}

} // namespace

// DeepseekV4IndexerScorer：Lightning Indexer 的打分器——为"每个 query 对每个压缩条目"打分，
// 决定哪些压缩条目最值得被当前 query 关注（之后 top-k 选）。
// 打分 = Σ_头 relu(query·compressed_kv / √d) × weights（多头的加权聚合）。
DeepseekV4IndexerScorer::DeepseekV4IndexerScorer(
    size_t hidden_size,
    size_t num_heads,
    size_t head_dim,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : num_heads_(num_heads), head_dim_(head_dim) {
    if (hidden_size == 0 || num_heads_ == 0 || head_dim_ == 0) {
        throw std::runtime_error(
            "DeepSeek-V4 indexer scorer configuration is invalid");
    }
    INFINICORE_NN_MODULE_INIT(
        weights_proj, hidden_size, num_heads_, dtype, device);
}

// 前向：query [B,S,H,head] 和 compressed_kv [B,entries,head] 算相关性分数，
// 再用 weights_proj(hidden) 加权各头，聚合出每个 (query, entry) 的打分。
infinicore::Tensor DeepseekV4IndexerScorer::forward(
    const infinicore::Tensor &query,
    const infinicore::Tensor &compressed_kv,
    const infinicore::Tensor &hidden_states) const {
    if (!query || query->ndim() != 4
        || query->size(2) != num_heads_
        || query->size(3) != head_dim_
        || !compressed_kv || compressed_kv->ndim() != 3
        || compressed_kv->size(0) != query->size(0)
        || compressed_kv->size(2) != head_dim_
        || !hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(0) != query->size(0)
        || hidden_states->size(1) != query->size(1)) {
        throw std::runtime_error(
            "DeepSeek-V4 indexer scorer input shape mismatch");
    }
    const size_t batch_size = query->size(0);
    const size_t sequence_length = query->size(1);
    const size_t compressed_length = compressed_kv->size(1);
    if (compressed_length == 0) {
        return infinicore::Tensor::empty(
            {batch_size, sequence_length, 0},
            infinicore::DataType::F32,
            query->device());
    }
    // query × compressed_kv：每个 (query 位置, 每个头) 对每个条目的点积
    auto scores = infinicore::op::matmul(
                      cast_to(query, infinicore::DataType::F32)
                          ->view({batch_size,
                                  sequence_length * num_heads_,
                                  head_dim_}),
                      cast_to(compressed_kv, infinicore::DataType::F32)
                          ->permute({0, 2, 1})
                          ->contiguous())
                      ->view({batch_size,
                              sequence_length,
                              num_heads_,
                              compressed_length});
    // relu + 缩放（只保留正向相关性）
    scores = infinicore::op::mul_scalar(
        infinicore::op::relu(scores),
        1.0 / std::sqrt(static_cast<double>(head_dim_)));
    // weights：hidden → weights_proj → 每头一个权重，加权聚合
    auto weights = infinicore::op::mul_scalar(
                       cast_to(
                           weights_proj_->forward(hidden_states),
                           infinicore::DataType::F32),
                       1.0 / std::sqrt(static_cast<double>(num_heads_)))
                       ->unsqueeze(3);
    weights = broadcast_to_shape(weights, scores->shape());
    // 沿头维度加权求和 → [B, S, entries]：每个 query 对每个条目的分数
    return infinicore::op::sum(
        infinicore::op::mul(scores, weights), {2}, false);
}

// DeepseekV4Indexer：Lightning Indexer——稀疏挑选压缩条目。
// 在 CSA 里，压缩条目可能很多，不是全看，而是：
//   1. 用压缩器（compress_stream）把历史压成条目
//   2. 用 scorer 为每个 query 对每个条目打分
//   3. 加 causal 掩码 → softmax（保序）→ top-k 选出最相关的条目
// 这对应 V4 配置 index_n_heads=64, index_head_dim=128, index_topk=512。
DeepseekV4Indexer::DeepseekV4Indexer(
    size_t hidden_size,
    size_t q_lora_rank,
    size_t num_heads,
    size_t head_dim,
    size_t rope_head_dim,
    size_t compress_rate,
    size_t index_topk,
    double rms_norm_eps,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      q_lora_rank_(q_lora_rank),
      num_heads_(num_heads),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      compress_rate_(compress_rate),
      index_topk_(index_topk) {
    if (hidden_size_ == 0 || q_lora_rank_ == 0 || num_heads_ == 0
        || head_dim_ == 0 || rope_head_dim_ == 0
        || rope_head_dim_ > head_dim_ || rope_head_dim_ % 2 != 0
        || compress_rate_ == 0 || index_topk_ == 0
        || rms_norm_eps <= 0.0) {
        throw std::runtime_error(
            "DeepSeek-V4 indexer configuration is invalid");
    }
    // 自己的压缩器（kv/gate 投影宽度 2*head_dim）+ query 投影 + scorer
    INFINICORE_NN_MODULE_INIT(
        kv_proj, hidden_size_, 2 * head_dim_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        gate_proj, hidden_size_, 2 * head_dim_, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(
        position_bias,
        ({compress_rate_, 2 * head_dim_}, dtype, device));
    INFINICORE_NN_MODULE_INIT(
        kv_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        q_b_proj, q_lora_rank_, num_heads_ * head_dim_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        scorer, hidden_size_, num_heads_, head_dim_, dtype, device);
}

// 前向：压缩 → 打分 → causal掩码 → softmax → top-k → 过滤不可见条目
DeepseekV4Indexer::Output DeepseekV4Indexer::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &q_residual,
    const infinicore::Tensor &query_cos,
    const infinicore::Tensor &query_sin,
    const infinicore::Tensor &compressed_cos,
    const infinicore::Tensor &compressed_sin,
    const infinicore::Tensor &position_ids,
    DeepseekV4CSAStreamState *state) const {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size_
        || !q_residual || q_residual->ndim() != 3
        || q_residual->size(0) != hidden_states->size(0)
        || q_residual->size(1) != hidden_states->size(1)
        || q_residual->size(2) != q_lora_rank_) {
        throw std::runtime_error(
            "DeepSeek-V4 indexer hidden/query residual shape mismatch");
    }
    // ① 用自己的压缩器压缩历史成条目
    auto compressed = compress_stream(
        hidden_states,
        kv_proj_,
        gate_proj_,
        kv_norm_,
        static_cast<infinicore::Tensor>(position_bias_),
        compressed_cos,
        compressed_sin,
        head_dim_,
        rope_head_dim_,
        compress_rate_,
        state);
    const size_t batch_size = hidden_states->size(0);
    const size_t sequence_length = hidden_states->size(1);
    // ② query 投影 + RoPE
    auto query = q_b_proj_->forward(q_residual)
                     ->view({batch_size,
                             sequence_length,
                             num_heads_,
                             head_dim_});
    query = apply_partial_rope(
        query,
        query_cos,
        query_sin,
        head_dim_,
        rope_head_dim_);
    // ③ 打分
    auto scores = scorer_->forward(
        query, compressed.compressed_kv, hidden_states);
    const size_t compressed_length = compressed.compressed_kv->size(1);
    const size_t top_k = std::min(index_topk_, compressed_length);
    infinicore::Tensor indices;
    if (top_k == 0) {
        indices = infinicore::Tensor::empty(
            {batch_size, sequence_length, 0},
            infinicore::DataType::I32,
            hidden_states->device());
    } else {
        // ④ causal 掩码（看不到未来的条目）
        scores = infinicore::op::add(
            scores,
            causal_score_bias(
                position_ids,
                compressed_length,
                compress_rate_,
                scores->dtype(),
                scores->device()));
        // ⑤ 保序 softmax：规避 CUDA top-k 在全负数行漏选的边界问题。
        //    softmax 保持顺序、把有限候选映射成正数、causal 的 -inf 保持 0 概率。
        scores = infinicore::op::softmax(scores, 2);
        // ⑥ top-k 选条目
        indices = infinicore::op::topk(
                      scores, top_k, 2, true, true)
                      .second;
        // ⑦ 把不可见的条目号标记为 -1（causal 屏蔽后仍可能混入）
        indices = mark_invalid_topk(
            indices, position_ids, compress_rate_);
    }
    return {indices, compressed.compressed_kv};
}

// DeepseekV4CSACompressor：CSA 压缩器（4 倍压缩 + Lightning Indexer 稀疏挑选）。
// = compress_stream（压缩历史成条目） + indexer（挑出每个 query 最相关的 top-k 条目）。
DeepseekV4CSACompressor::DeepseekV4CSACompressor(
    size_t hidden_size,
    size_t q_lora_rank,
    size_t attention_head_dim,
    size_t rope_head_dim,
    size_t compress_rate,
    size_t index_num_heads,
    size_t index_head_dim,
    size_t index_topk,
    double rms_norm_eps,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      q_lora_rank_(q_lora_rank),
      attention_head_dim_(attention_head_dim),
      rope_head_dim_(rope_head_dim),
      compress_rate_(compress_rate),
      index_topk_(index_topk) {
    if (hidden_size_ == 0 || q_lora_rank_ == 0
        || attention_head_dim_ == 0 || rope_head_dim_ == 0
        || rope_head_dim_ > attention_head_dim_
        || rope_head_dim_ > index_head_dim
        || rope_head_dim_ % 2 != 0 || compress_rate_ == 0
        || index_num_heads == 0 || index_head_dim == 0
        || index_topk_ == 0 || rms_norm_eps <= 0.0) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA compressor configuration is invalid");
    }
    // 自己的压缩器（给注意力用的，head_dim = attention_head_dim）
    INFINICORE_NN_MODULE_INIT(
        kv_proj, hidden_size_, 2 * attention_head_dim_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        gate_proj, hidden_size_, 2 * attention_head_dim_, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(
        position_bias,
        ({compress_rate_, 2 * attention_head_dim_}, dtype, device));
    INFINICORE_NN_MODULE_INIT(
        kv_norm, attention_head_dim_, rms_norm_eps, dtype, device);
    // 嵌套一个 indexer（它内部有自己的压缩器 + scorer）
    INFINICORE_NN_MODULE_INIT(
        indexer,
        hidden_size_,
        q_lora_rank_,
        index_num_heads,
        index_head_dim,
        rope_head_dim_,
        compress_rate_,
        index_topk_,
        rms_norm_eps,
        dtype,
        device);
}

// 前向：压缩出全部条目 + 用 indexer 选出每个 query 的 top-k 条目 + 块掩码
DeepseekV4CSAOutput DeepseekV4CSACompressor::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &q_residual,
    const infinicore::Tensor &query_cos,
    const infinicore::Tensor &query_sin,
    const infinicore::Tensor &compressed_cos,
    const infinicore::Tensor &compressed_sin,
    const infinicore::Tensor &position_ids,
    DeepseekV4CSAState *state) const {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size_
        || !q_residual || q_residual->ndim() != 3
        || q_residual->size(0) != hidden_states->size(0)
        || q_residual->size(1) != hidden_states->size(1)
        || q_residual->size(2) != q_lora_rank_
        || !position_ids || position_ids->ndim() != 2
        || position_ids->size(0) != hidden_states->size(0)
        || position_ids->size(1) != hidden_states->size(1)) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA compressor input shape mismatch");
    }
    // ① 用注意力侧的压缩器压缩历史成条目
    auto compressed = compress_stream(
        hidden_states,
        kv_proj_,
        gate_proj_,
        kv_norm_,
        static_cast<infinicore::Tensor>(position_bias_),
        compressed_cos,
        compressed_sin,
        attention_head_dim_,
        rope_head_dim_,
        compress_rate_,
        state == nullptr ? nullptr : &state->compressor);
    // ② 用 indexer 选出每个 query 的 top-k 条目
    auto index = indexer_->forward(
        hidden_states,
        q_residual,
        query_cos,
        query_sin,
        compressed_cos,
        compressed_sin,
        position_ids,
        state == nullptr ? nullptr : &state->indexer);
    // ③ 校验压缩条目数一致（压缩器与 indexer 各自累积，必须同步）
    if (index.compressed_kv->size(1)
        != compressed.compressed_kv->size(1)) {
        throw std::runtime_error(
            "DeepSeek-V4 CSA compressor/indexer histories diverged");
    }
    // ④ 把选中的条目转成块掩码（选中=0可见，其余=-inf）
    auto block_bias = make_block_bias(
        index.topk_indices,
        compressed.compressed_kv->size(1),
        compressed.compressed_kv->dtype(),
        compressed.compressed_kv->device());
    return {
        compressed.compressed_kv->unsqueeze(1),
        index.topk_indices,
        block_bias};
}

} // namespace infinilm::models::deepseek_v4
