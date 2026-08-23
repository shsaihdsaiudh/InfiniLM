#include "deepseek_v4_hca_compressor.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/softmax.hpp>
#include <infinicore/ops/sum.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

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

void validate_state_tensor(const infinicore::Tensor &tensor,
                           size_t batch_size,
                           size_t head_dim,
                           const infinicore::Tensor &reference,
                           const char *name) {
    if (!tensor) {
        return;
    }
    if (tensor->ndim() != 3 || tensor->size(0) != batch_size
        || tensor->size(2) != head_dim
        || tensor->dtype() != reference->dtype()
        || tensor->device() != reference->device()) {
        throw std::runtime_error(
            std::string("DeepSeek-V4 HCA incompatible state tensor: ") + name);
    }
}

} // namespace

DeepseekV4HCACompressor::DeepseekV4HCACompressor(
    size_t hidden_size,
    size_t head_dim,
    size_t rope_head_dim,
    size_t compress_rate,
    double rms_norm_eps,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      compress_rate_(compress_rate) {
    if (hidden_size_ == 0 || head_dim_ == 0 || rope_head_dim_ == 0
        || rope_head_dim_ > head_dim_ || rope_head_dim_ % 2 != 0
        || compress_rate_ == 0 || rms_norm_eps <= 0.0) {
        throw std::runtime_error("DeepSeek-V4 HCA configuration is invalid");
    }
    INFINICORE_NN_MODULE_INIT(
        kv_proj, hidden_size_, head_dim_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        gate_proj, hidden_size_, head_dim_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        kv_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(
        position_bias,
        ({compress_rate_, head_dim_}, dtype, device));
}

infinicore::Tensor DeepseekV4HCACompressor::apply_partial_rope_(
    const infinicore::Tensor &input,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin) const {
    if (!input || input->ndim() != 3 || input->size(2) != head_dim_
        || !cos || !sin || cos->ndim() != 3 || sin->ndim() != 3
        || cos->shape() != sin->shape()
        || cos->size(0) != input->size(0)
        || cos->size(1) != input->size(1)
        || cos->size(2) * 2 != rope_head_dim_) {
        throw std::runtime_error("DeepSeek-V4 HCA RoPE shape mismatch");
    }
    const size_t batch_size = input->size(0);
    const size_t sequence_length = input->size(1);
    const size_t pair_count = rope_head_dim_ / 2;
    const size_t nope_dim = head_dim_ - rope_head_dim_;

    infinicore::Tensor nope;
    if (nope_dim != 0) {
        nope = input->narrow({{2, 0, nope_dim}})->contiguous();
    }
    auto rope = input->narrow({{2, nope_dim, rope_head_dim_}})
                    ->contiguous()
                    ->view({batch_size, sequence_length, pair_count, 2});
    auto first = rope->narrow({{3, 0, 1}})->contiguous();
    auto second = rope->narrow({{3, 1, 1}})->contiguous();
    auto rotated = infinicore::op::cat(
        {infinicore::op::mul_scalar(second, -1.0), first}, 3);
    const infinicore::Shape paired_shape{
        batch_size, sequence_length, pair_count, 2};
    auto cos_f32 = cast_to(cos, infinicore::DataType::F32)->unsqueeze(3);
    auto sin_f32 = cast_to(sin, infinicore::DataType::F32)->unsqueeze(3);
    auto mixed = infinicore::op::add(
        infinicore::op::mul(
            cast_to(rope, infinicore::DataType::F32),
            broadcast_to_shape(cos_f32, paired_shape)),
        infinicore::op::mul(
            cast_to(rotated, infinicore::DataType::F32),
            broadcast_to_shape(sin_f32, paired_shape)));
    auto rotated_typed = cast_to(
        mixed->view(
            {batch_size, sequence_length, rope_head_dim_}),
        input->dtype());
    if (nope_dim == 0) {
        return rotated_typed;
    }
    return infinicore::op::cat({nope, rotated_typed}, 2);
}

std::optional<infinicore::Tensor>
DeepseekV4HCACompressor::make_block_bias_(
    const infinicore::Tensor &position_ids,
    size_t compressed_length,
    const infinicore::DataType &dtype,
    const infinicore::Device &device) const {
    if (!position_ids || position_ids->ndim() != 2) {
        throw std::runtime_error(
            "DeepSeek-V4 HCA position_ids must be [batch, sequence]");
    }
    const size_t batch_size = position_ids->size(0);
    const size_t sequence_length = position_ids->size(1);
    if (sequence_length == 1 || compressed_length == 0) {
        return std::nullopt;
    }

    // Correctness path: materialize positions on CPU while constructing the
    // irregular block bias. The integrated graph path should replace this
    // with a device-side mask kernel before production serving.
    auto positions_cpu = position_ids->to(infinicore::Device::cpu())
                             ->contiguous();
    std::vector<int64_t> positions(positions_cpu->numel());
    if (positions_cpu->dtype() == infinicore::DataType::I64) {
        std::memcpy(
            positions.data(), positions_cpu->data(), positions_cpu->nbytes());
    } else if (positions_cpu->dtype() == infinicore::DataType::I32) {
        const auto *source =
            reinterpret_cast<const int32_t *>(positions_cpu->data());
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = source[i];
        }
    } else {
        throw std::runtime_error(
            "DeepSeek-V4 HCA position_ids must use I32 or I64");
    }

    std::vector<float> values(
        batch_size * sequence_length * compressed_length, 0.0f);
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t query = 0; query < sequence_length; ++query) {
            const int64_t position =
                positions[batch * sequence_length + query];
            if (position < 0) {
                throw std::runtime_error(
                    "DeepSeek-V4 HCA position_ids must be non-negative");
            }
            const size_t threshold =
                (static_cast<size_t>(position) + 1) / compress_rate_;
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
        {batch_size, 1, sequence_length, compressed_length},
        infinicore::DataType::F32,
        infinicore::Device::cpu());
    return cast_to(cpu->to(device), dtype);
}

DeepseekV4HCAOutput DeepseekV4HCACompressor::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &cos,
    const infinicore::Tensor &sin,
    const infinicore::Tensor &position_ids,
    DeepseekV4HCAState *state) const {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size_) {
        throw std::runtime_error(
            "DeepSeek-V4 HCA expects [batch, sequence, hidden_size]");
    }
    const size_t batch_size = hidden_states->size(0);
    if (!position_ids || position_ids->ndim() != 2
        || position_ids->size(0) != batch_size
        || position_ids->size(1) != hidden_states->size(1)) {
        throw std::runtime_error(
            "DeepSeek-V4 HCA position_ids do not match hidden states");
    }
    auto source_kv = kv_proj_->forward(hidden_states);
    auto source_gate = gate_proj_->forward(hidden_states);

    if (state != nullptr) {
        validate_state_tensor(
            state->buffer_kv, batch_size, head_dim_, source_kv, "buffer_kv");
        validate_state_tensor(
            state->buffer_gate, batch_size, head_dim_, source_gate, "buffer_gate");
        validate_state_tensor(
            state->compressed_kv,
            batch_size,
            head_dim_,
            source_kv,
            "compressed_kv");
        if (static_cast<bool>(state->buffer_kv)
                != static_cast<bool>(state->buffer_gate)
            || (state->buffer_kv
                && state->buffer_kv->size(1)
                       != state->buffer_gate->size(1))) {
            throw std::runtime_error(
                "DeepSeek-V4 HCA buffered KV/gate state is inconsistent");
        }
        if (state->compressed_kv
            && state->compressed_kv->size(1) != state->entry_count) {
            throw std::runtime_error(
                "DeepSeek-V4 HCA compressed history/count is inconsistent");
        }
        if (state->buffer_kv && state->buffer_kv->size(1) != 0) {
            source_kv = infinicore::op::cat(
                {state->buffer_kv, source_kv}, 1);
            source_gate = infinicore::op::cat(
                {state->buffer_gate, source_gate}, 1);
        }
    }

    const size_t source_length = source_kv->size(1);
    const size_t usable = source_length / compress_rate_ * compress_rate_;
    const size_t new_entries = usable / compress_rate_;
    if (!cos || !sin || cos->ndim() != 3 || sin->ndim() != 3
        || cos->shape() != sin->shape() || cos->size(0) != batch_size
        || cos->size(1) != new_entries
        || cos->size(2) * 2 != rope_head_dim_) {
        throw std::runtime_error(
            "DeepSeek-V4 HCA cos/sin count does not match closed windows");
    }

    if (state != nullptr) {
        const size_t remainder = source_length - usable;
        state->buffer_kv = source_kv
                               ->narrow({{1, usable, remainder}})
                               ->contiguous();
        state->buffer_gate = source_gate
                                 ->narrow({{1, usable, remainder}})
                                 ->contiguous();
    }

    infinicore::Tensor new_compressed;
    if (new_entries == 0) {
        new_compressed = infinicore::Tensor::empty(
            {batch_size, 0, head_dim_},
            source_kv->dtype(),
            source_kv->device());
    } else {
        auto chunk_kv = source_kv
                            ->narrow({{1, 0, usable}})
                            ->contiguous()
                            ->view({batch_size,
                                    new_entries,
                                    compress_rate_,
                                    head_dim_});
        auto chunk_gate = source_gate
                              ->narrow({{1, 0, usable}})
                              ->contiguous()
                              ->view({batch_size,
                                      new_entries,
                                      compress_rate_,
                                      head_dim_});
        const infinicore::Shape window_shape{
            batch_size, new_entries, compress_rate_, head_dim_};
        auto gate_logits = infinicore::op::add(
            cast_to(chunk_gate, infinicore::DataType::F32),
            broadcast_to_shape(
                cast_to(
                    static_cast<infinicore::Tensor>(position_bias_),
                    infinicore::DataType::F32)
                    ->view({1, 1, compress_rate_, head_dim_}),
                window_shape));
        auto gate_weights = cast_to(
            infinicore::op::softmax(gate_logits, 2),
            chunk_kv->dtype());
        auto pooled = infinicore::op::sum(
            infinicore::op::mul(chunk_kv, gate_weights), {2}, false);
        new_compressed = kv_norm_->forward(pooled);
        new_compressed = apply_partial_rope_(
            new_compressed, cos, sin);
    }

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
    auto block_bias = make_block_bias_(
        position_ids,
        all_compressed->size(1),
        all_compressed->dtype(),
        all_compressed->device());
    return {all_compressed->unsqueeze(1), new_compressed, block_bias};
}

} // namespace infinilm::models::deepseek_v4
