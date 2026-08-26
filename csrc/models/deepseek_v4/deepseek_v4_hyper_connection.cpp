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

// ---- 张量工具（本文件内复用）----
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

// add_epsilon：给张量每个元素加一个 epsilon（标量广播）。
// 用于数值稳定，防止除零 / 归一化时出现 0 分母。
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

// normalize_axis：沿指定 axis 做行/列归一化——用该轴的和做分母（加 epsilon），
// 把该轴上的分量归一成总和为 1。Sinkhorn 迭代的核心操作。
infinicore::Tensor normalize_axis(const infinicore::Tensor &input,
                                  size_t axis,
                                  const infinicore::Tensor &epsilon) {
    auto denominator = infinicore::op::sum(input, {axis}, true);
    denominator = add_epsilon(denominator, epsilon);
    return multiply_broadcast(
        input, infinicore::op::reciprocal(denominator), input->shape());
}

// 校验四流张量形状必须是 [batch, sequence, hc_mult, hidden_size]。
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

// DeepseekV4HyperConnection：mHC（流超连接）——V4 的核心机制。
// 每个子层（attention/FFN）前有一个实例，做两件事：
//   forward：从四流算出 pre（折叠权重）、post（放大权重）、comb（流间混合矩阵），
//            并把四流折叠成单路（collapsed）供子层计算。
//   apply：  用 post/comb 把子层输出和原四流重新组合回四路。
// 这里的 "pre/post" 是逐流缩放标量，"comb" 是经 Sinkhorn 约束的双随机矩阵。
DeepseekV4HyperConnection::DeepseekV4HyperConnection(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : DeepseekV4HyperConnection(
        model_config->get<size_t>("hidden_size"),
        model_config->get<size_t>("hc_mult"),          // 流数（V4 为 4）
        model_config->get<size_t>("hc_sinkhorn_iters"), // Sinkhorn 迭代次数（默认 20）
        model_config->get_or<double>("hc_eps", 1e-6),   // 数值稳定 epsilon
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

    // 三个可学习参数：
    //   fn：把归一化后的四流（hc_mult*hidden_size 维）映射成 mix_size 个混合量。
    //       mix_size = (2 + hc_mult) * hc_mult，即 pre(1*hc_mult) + post(1*hc_mult)
    //       + comb(hc_mult*hc_mult) 拼在一起。
    //   base：偏置，与 fn 输出的三段一一对应。
    //   scale：pre/post/comb 各一个缩放标量（3 个）。
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

// mHC 的"折叠前"阶段：从四流 hidden_streams 算出三样东西并折叠成单路。
//   返回 {post, comb, collapsed}
//     - collapsed：四流按 pre 权重加权求和成单路 [B,S,D]，给子层当输入
//     - post：子层输出每个流上的放大权重（sigmoid×2）
//     - comb：4×4 流间混合矩阵（Sinkhorn 约束成双随机矩阵）
DeepseekV4HyperConnectionOutput DeepseekV4HyperConnection::forward(
    const infinicore::Tensor &hidden_streams) const {
    validate_hidden_streams(hidden_streams, hidden_size_, hc_mult_);

    const size_t batch_size = hidden_streams->size(0);
    const size_t sequence_length = hidden_streams->size(1);
    const size_t mix_size = (2 + hc_mult_) * hc_mult_;
    const size_t last_axis = 2;

    // 四流展平后先 RMSNorm（F32），再线性映射出 mix 量。
    // mix 的最后一个轴被切成三段：pre 段 / post 段 / comb 段
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

    // 切出三段：pre 权重（hc_mult 个）、post 权重（hc_mult 个）、comb 原始矩阵
    auto pre_w = mix->narrow({{last_axis, 0, hc_mult_}});
    auto post_w = mix->narrow({{last_axis, hc_mult_, hc_mult_}});
    auto comb_w = mix->narrow(
        {{last_axis, 2 * hc_mult_, hc_mult_ * hc_mult_}});

    // 从 base 里切出对应的偏置（comb 的偏置重排成 hc_mult×hc_mult 矩阵）
    auto pre_b = static_cast<infinicore::Tensor>(base_)->narrow(
        {{0, 0, hc_mult_}});
    auto post_b = static_cast<infinicore::Tensor>(base_)->narrow(
        {{0, hc_mult_, hc_mult_}});
    auto comb_b = static_cast<infinicore::Tensor>(base_)->narrow(
        {{0, 2 * hc_mult_, hc_mult_ * hc_mult_}})
                      ->view({hc_mult_, hc_mult_});

    // 从 scale 里切出三个缩放标量
    auto pre_scale = static_cast<infinicore::Tensor>(scale_)->narrow(
        {{0, 0, 1}});
    auto post_scale = static_cast<infinicore::Tensor>(scale_)->narrow(
        {{0, 1, 1}});
    auto comb_scale = static_cast<infinicore::Tensor>(scale_)->narrow(
        {{0, 2, 1}});

    // pre：sigmoid(scale*w + b) + epsilon，作为折叠时每路权重（在 (0,1)）
    auto pre = infinicore::op::sigmoid(add_broadcast(
        multiply_broadcast(pre_w, pre_scale, pre_w->shape()),
        pre_b,
        pre_w->shape()));
    pre = add_epsilon(pre, epsilon_);

    // post：sigmoid(scale*w + b)×2，作为子层输出每路的放大权重
    auto post = infinicore::op::sigmoid(add_broadcast(
        multiply_broadcast(post_w, post_scale, post_w->shape()),
        post_b,
        post_w->shape()));
    post = infinicore::op::mul_scalar(post, 2.0);

    // comb：先把原始 logits 做 softmax 得初始矩阵，再做 Sinkhorn 迭代：
    //   softmax → 行归一 → 列归一 → 行归一 → ...（sinkhorn_iters 次）
    // 最终得到行和、列和都为 1 的双随机矩阵（流间混合矩阵）。
    auto comb_w_matrix = comb_w->view(
        {batch_size, sequence_length, hc_mult_, hc_mult_});
    auto comb_logits = add_broadcast(
        multiply_broadcast(
            comb_w_matrix, comb_scale, comb_w_matrix->shape()),
        comb_b,
        comb_w_matrix->shape());
    auto comb = infinicore::op::softmax(comb_logits, -1);
    comb = add_epsilon(comb, epsilon_);
    comb = normalize_axis(comb, 2, epsilon_);   // 先归一"行"
    for (size_t iteration = 1; iteration < sinkhorn_iters_; ++iteration) {
        comb = normalize_axis(comb, 3, epsilon_);  // 归一"列"
        comb = normalize_axis(comb, 2, epsilon_);  // 再归一"行"
    }

    // 折叠：collapsed = Σ_流 pre * hidden_streams（每路乘 pre 再按流求和）→ 单路
    auto hidden_f32 = cast_to(hidden_streams, infinicore::DataType::F32);
    auto collapsed = infinicore::op::sum(
        multiply_broadcast(
            pre->unsqueeze(3), hidden_f32, hidden_f32->shape()),
        {2},
        false);
    collapsed = cast_to(collapsed, hidden_streams->dtype());

    return {post, comb, collapsed};
}

// mHC 的"折叠后"阶段：把子层输出和原四流重新组合回四路。
//   output_new = post * sublayer_output（每路放大） + comb^T @ hidden_streams（流间混合）
// comb 是前向算出的双随机矩阵，这里用它的转置对四流做线性混合。
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
    // 第一项：子层输出放大——post 逐流乘到子层输出上，广播成四路
    auto placed = multiply_broadcast(
        post_typed->unsqueeze(3),
        sublayer_output->unsqueeze(2),
        hidden_streams->shape());
    // 第二项：流间混合——comb^T（转置后 [4,4]）× 四流 [4,D]，得到混合后的四流
    const size_t batch_tokens = hidden_streams->size(0) * hidden_streams->size(1);
    auto mixed = infinicore::op::matmul(
        comb_typed->permute({0, 1, 3, 2})
            ->contiguous()
            ->view({batch_tokens, hc_mult_, hc_mult_}),
        hidden_streams->view({batch_tokens, hc_mult_, hidden_size_}))
                     ->view(hidden_streams->shape());
    // 两项相加 = 新的四流，返回给下一层
    return infinicore::op::add(placed, mixed);
}

// DeepseekV4HyperHead：mHC 头部，在模型最末层把四流折叠回单路 [B,S,D]。
// 与 HyperConnection 的 forward 类似，但没有 comb（不需要再混合回四流），
// 只算 per-stream 的折叠权重 pre，加权求和得到最终 hidden。
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

    // 可学习参数：hc_fn（四流 → 每流一个权重）、hc_base、hc_scale
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

// 前向：四流 → RMSNorm → 线性出每流权重 → sigmoid → 加权求和 → 单路 [B,S,D]
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
    // 四流 → 每流一个权重（hc_mult 个）
    auto mixes = infinicore::op::linear(
        normalized->view(
            {batch_size * sequence_length, hc_mult_ * hidden_size_}),
        static_cast<infinicore::Tensor>(hc_fn_),
        std::nullopt)
                     ->view({batch_size, sequence_length, hc_mult_});
    // sigmoid + epsilon 得到 (0,1) 的折叠权重
    auto pre = infinicore::op::sigmoid(add_broadcast(
        multiply_broadcast(
            mixes,
            static_cast<infinicore::Tensor>(hc_scale_),
            mixes->shape()),
        static_cast<infinicore::Tensor>(hc_base_),
        mixes->shape()));
    pre = add_epsilon(pre, epsilon_);

    // 折叠：collapsed = Σ_流 pre * hidden_streams
    auto hidden_f32 = cast_to(hidden_streams, infinicore::DataType::F32);
    auto collapsed = infinicore::op::sum(
        multiply_broadcast(
            pre->unsqueeze(3), hidden_f32, hidden_f32->shape()),
        {2},
        false);
    return cast_to(collapsed, hidden_streams->dtype());
}

} // namespace infinilm::models::deepseek_v4
