#include "deepseek_v4_hyper_connection.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/reciprocal.hpp>
#include <infinicore/ops/rms_norm.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <infinicore/ops/softmax.hpp>
#include <infinicore/ops/sum.hpp>

#include <optional>
#include <stdexcept>

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

infinicore::Tensor add_epsilon(const infinicore::Tensor &input,
                               const infinicore::Tensor &epsilon) {
    std::vector<int64_t> shape(input->shape().begin(), input->shape().end());
    return infinicore::op::add(
        input, infinicore::op::broadcast_to(epsilon, shape));
}

infinicore::Tensor broadcast_to_shape(const infinicore::Tensor &input,
                                      const infinicore::Shape &shape) {
    if (input->shape() == shape) {
        return input;
    }
    return infinicore::op::broadcast_to(
        input, std::vector<int64_t>(shape.begin(), shape.end()));
}

infinicore::Tensor multiply_broadcast(const infinicore::Tensor &lhs,
                                      const infinicore::Tensor &rhs,
                                      const infinicore::Shape &shape) {
    return infinicore::op::mul(
        broadcast_to_shape(lhs, shape), broadcast_to_shape(rhs, shape));
}

infinicore::Tensor add_broadcast(const infinicore::Tensor &lhs,
                                 const infinicore::Tensor &rhs,
                                 const infinicore::Shape &shape) {
    return infinicore::op::add(
        broadcast_to_shape(lhs, shape), broadcast_to_shape(rhs, shape));
}

infinicore::Tensor normalize_axis(const infinicore::Tensor &input,
                                  size_t axis,
                                  const infinicore::Tensor &epsilon) {
    auto denominator = infinicore::op::sum(input, {axis}, true);
    denominator = add_epsilon(denominator, epsilon);
    return multiply_broadcast(
        input, infinicore::op::reciprocal(denominator), input->shape());
}

void validate_hidden_streams(const infinicore::Tensor &hidden_streams,
                             size_t hidden_size,
                             size_t hc_mult) {
    if (!hidden_streams || hidden_streams->ndim() != 4
        || hidden_streams->size(2) != hc_mult
        || hidden_streams->size(3) != hidden_size) {
        throw std::runtime_error(
            "DeepSeek-V4 mHC expects [batch, sequence, hc_mult, hidden_size]");
    }
}

} // namespace

DeepseekV4HyperConnection::DeepseekV4HyperConnection(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : DeepseekV4HyperConnection(
        model_config->get<size_t>("hidden_size"),
        model_config->get<size_t>("hc_mult"),
        model_config->get<size_t>("hc_sinkhorn_iters"),
        model_config->get_or<double>("hc_eps", 1e-6),
        model_config->get<double>("rms_norm_eps"),
        device) {}

DeepseekV4HyperConnection::DeepseekV4HyperConnection(
    size_t hidden_size,
    size_t hc_mult,
    size_t sinkhorn_iters,
    double hc_eps,
    double rms_norm_eps,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      hc_mult_(hc_mult),
      sinkhorn_iters_(sinkhorn_iters),
      hc_eps_(hc_eps),
      rms_norm_eps_(rms_norm_eps) {
    if (hidden_size_ == 0 || hc_mult_ == 0 || sinkhorn_iters_ == 0
        || hc_eps_ <= 0.0 || rms_norm_eps_ <= 0.0) {
        throw std::runtime_error("DeepSeek-V4: invalid mHC dimensions or epsilon");
    }

    const size_t mix_size = (2 + hc_mult_) * hc_mult_;
    INFINICORE_NN_PARAMETER_INIT(
        fn,
        ({mix_size, hc_mult_ * hidden_size_},
         infinicore::DataType::F32,
         device));
    INFINICORE_NN_PARAMETER_INIT(
        base,
        ({mix_size}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(
        scale,
        ({3}, infinicore::DataType::F32, device));

    norm_weight_ = infinicore::Tensor::ones(
        {hc_mult_ * hidden_size_}, infinicore::DataType::F32, device);
    epsilon_ = infinicore::Tensor::ones(
        {1}, infinicore::DataType::F32, device);
    infinicore::op::mul_scalar_(epsilon_, epsilon_, hc_eps_);
}

DeepseekV4HyperConnectionOutput DeepseekV4HyperConnection::forward(
    const infinicore::Tensor &hidden_streams) const {
    validate_hidden_streams(hidden_streams, hidden_size_, hc_mult_);

    const size_t batch_size = hidden_streams->size(0);
    const size_t sequence_length = hidden_streams->size(1);
    const size_t mix_size = (2 + hc_mult_) * hc_mult_;
    const size_t last_axis = 2;

    auto flat = hidden_streams->view(
        {batch_size, sequence_length, hc_mult_ * hidden_size_});
    auto flat_f32 = cast_to(flat, infinicore::DataType::F32);
    auto normalized = infinicore::op::rms_norm(
        flat_f32, norm_weight_, static_cast<float>(rms_norm_eps_));
    auto mix = infinicore::op::linear(
        normalized->view(
            {batch_size * sequence_length, hc_mult_ * hidden_size_}),
        static_cast<infinicore::Tensor>(fn_),
        std::nullopt)
                   ->view({batch_size, sequence_length, mix_size});
    if (mix->size(last_axis) != mix_size) {
        throw std::runtime_error("DeepSeek-V4: unexpected mHC projection shape");
    }

    auto pre_w = mix->narrow({{last_axis, 0, hc_mult_}});
    auto post_w = mix->narrow({{last_axis, hc_mult_, hc_mult_}});
    auto comb_w = mix->narrow(
        {{last_axis, 2 * hc_mult_, hc_mult_ * hc_mult_}});

    auto pre_b = static_cast<infinicore::Tensor>(base_)->narrow(
        {{0, 0, hc_mult_}});
    auto post_b = static_cast<infinicore::Tensor>(base_)->narrow(
        {{0, hc_mult_, hc_mult_}});
    auto comb_b = static_cast<infinicore::Tensor>(base_)->narrow(
        {{0, 2 * hc_mult_, hc_mult_ * hc_mult_}})
                      ->view({hc_mult_, hc_mult_});

    auto pre_scale = static_cast<infinicore::Tensor>(scale_)->narrow(
        {{0, 0, 1}});
    auto post_scale = static_cast<infinicore::Tensor>(scale_)->narrow(
        {{0, 1, 1}});
    auto comb_scale = static_cast<infinicore::Tensor>(scale_)->narrow(
        {{0, 2, 1}});

    auto pre = infinicore::op::sigmoid(add_broadcast(
        multiply_broadcast(pre_w, pre_scale, pre_w->shape()),
        pre_b,
        pre_w->shape()));
    pre = add_epsilon(pre, epsilon_);

    auto post = infinicore::op::sigmoid(add_broadcast(
        multiply_broadcast(post_w, post_scale, post_w->shape()),
        post_b,
        post_w->shape()));
    post = infinicore::op::mul_scalar(post, 2.0);

    auto comb_w_matrix = comb_w->view(
        {batch_size, sequence_length, hc_mult_, hc_mult_});
    auto comb_logits = add_broadcast(
        multiply_broadcast(
            comb_w_matrix, comb_scale, comb_w_matrix->shape()),
        comb_b,
        comb_w_matrix->shape());
    auto comb = infinicore::op::softmax(comb_logits, -1);
    comb = add_epsilon(comb, epsilon_);
    comb = normalize_axis(comb, 2, epsilon_);
    for (size_t iteration = 1; iteration < sinkhorn_iters_; ++iteration) {
        comb = normalize_axis(comb, 3, epsilon_);
        comb = normalize_axis(comb, 2, epsilon_);
    }

    auto hidden_f32 = cast_to(hidden_streams, infinicore::DataType::F32);
    auto collapsed = infinicore::op::sum(
        multiply_broadcast(
            pre->unsqueeze(3), hidden_f32, hidden_f32->shape()),
        {2},
        false);
    collapsed = cast_to(collapsed, hidden_streams->dtype());

    return {post, comb, collapsed};
}

infinicore::Tensor DeepseekV4HyperConnection::apply(
    const infinicore::Tensor &hidden_streams,
    const infinicore::Tensor &sublayer_output,
    const infinicore::Tensor &post,
    const infinicore::Tensor &comb) const {
    validate_hidden_streams(hidden_streams, hidden_size_, hc_mult_);
    if (!sublayer_output || sublayer_output->ndim() != 3
        || sublayer_output->size(0) != hidden_streams->size(0)
        || sublayer_output->size(1) != hidden_streams->size(1)
        || sublayer_output->size(2) != hidden_size_) {
        throw std::runtime_error(
            "DeepSeek-V4 mHC sublayer output has an incompatible shape");
    }

    const auto dtype = hidden_streams->dtype();
    auto post_typed = cast_to(post, dtype);
    auto comb_typed = cast_to(comb, dtype);
    auto placed = multiply_broadcast(
        post_typed->unsqueeze(3),
        sublayer_output->unsqueeze(2),
        hidden_streams->shape());
    const size_t batch_tokens = hidden_streams->size(0) * hidden_streams->size(1);
    auto mixed = infinicore::op::matmul(
        comb_typed->permute({0, 1, 3, 2})
            ->contiguous()
            ->view({batch_tokens, hc_mult_, hc_mult_}),
        hidden_streams->view({batch_tokens, hc_mult_, hidden_size_}))
                     ->view(hidden_streams->shape());
    return infinicore::op::add(placed, mixed);
}

DeepseekV4HyperHead::DeepseekV4HyperHead(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : DeepseekV4HyperHead(
        model_config->get<size_t>("hidden_size"),
        model_config->get<size_t>("hc_mult"),
        model_config->get_or<double>("hc_eps", 1e-6),
        model_config->get<double>("rms_norm_eps"),
        device) {}

DeepseekV4HyperHead::DeepseekV4HyperHead(
    size_t hidden_size,
    size_t hc_mult,
    double hc_eps,
    double rms_norm_eps,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      hc_mult_(hc_mult),
      hc_eps_(hc_eps),
      rms_norm_eps_(rms_norm_eps) {
    if (hidden_size_ == 0 || hc_mult_ == 0 || hc_eps_ <= 0.0
        || rms_norm_eps_ <= 0.0) {
        throw std::runtime_error(
            "DeepSeek-V4: invalid hyper-head dimensions or epsilon");
    }

    INFINICORE_NN_PARAMETER_INIT(
        hc_fn,
        ({hc_mult_, hc_mult_ * hidden_size_},
         infinicore::DataType::F32,
         device));
    INFINICORE_NN_PARAMETER_INIT(
        hc_base,
        ({hc_mult_}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(
        hc_scale,
        ({1}, infinicore::DataType::F32, device));

    norm_weight_ = infinicore::Tensor::ones(
        {hc_mult_ * hidden_size_}, infinicore::DataType::F32, device);
    epsilon_ = infinicore::Tensor::ones(
        {1}, infinicore::DataType::F32, device);
    infinicore::op::mul_scalar_(epsilon_, epsilon_, hc_eps_);
}

infinicore::Tensor DeepseekV4HyperHead::forward(
    const infinicore::Tensor &hidden_streams) const {
    validate_hidden_streams(hidden_streams, hidden_size_, hc_mult_);
    const size_t batch_size = hidden_streams->size(0);
    const size_t sequence_length = hidden_streams->size(1);

    auto flat = hidden_streams->view(
        {batch_size, sequence_length, hc_mult_ * hidden_size_});
    auto flat_f32 = cast_to(flat, infinicore::DataType::F32);
    auto normalized = infinicore::op::rms_norm(
        flat_f32, norm_weight_, static_cast<float>(rms_norm_eps_));
    auto mixes = infinicore::op::linear(
        normalized->view(
            {batch_size * sequence_length, hc_mult_ * hidden_size_}),
        static_cast<infinicore::Tensor>(hc_fn_),
        std::nullopt)
                     ->view({batch_size, sequence_length, hc_mult_});
    auto pre = infinicore::op::sigmoid(add_broadcast(
        multiply_broadcast(
            mixes,
            static_cast<infinicore::Tensor>(hc_scale_),
            mixes->shape()),
        static_cast<infinicore::Tensor>(hc_base_),
        mixes->shape()));
    pre = add_epsilon(pre, epsilon_);

    auto hidden_f32 = cast_to(hidden_streams, infinicore::DataType::F32);
    auto collapsed = infinicore::op::sum(
        multiply_broadcast(
            pre->unsqueeze(3), hidden_f32, hidden_f32->shape()),
        {2},
        false);
    return cast_to(collapsed, hidden_streams->dtype());
}

} // namespace infinilm::models::deepseek_v4
