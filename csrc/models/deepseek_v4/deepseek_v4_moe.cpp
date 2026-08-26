#include "deepseek_v4_moe.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/fmin.hpp>
#include <infinicore/ops/fused_moe_mxfp4.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/silu.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

// upper_bound：把张量每个元素截断到上限 limit（相当于 min(x, limit)）。
// 用 fmin（逐元素取小）实现。
infinicore::Tensor upper_bound(const infinicore::Tensor &input,
                               float limit) {
    auto bound = infinicore::Tensor::ones(
        input->shape(), input->dtype(), input->device());
    infinicore::op::mul_scalar_(bound, bound, limit);
    return infinicore::op::fmin(input, bound);
}

// lower_bound：把张量每个元素截断到下限 limit（相当于 max(x, limit)）。
// 用"取负 → upper_bound → 再取负"实现（InfiniCore 没有直接的 max 算子）。
infinicore::Tensor lower_bound(const infinicore::Tensor &input,
                               float limit) {
    auto negated = infinicore::op::mul_scalar(input, -1.0);
    return infinicore::op::mul_scalar(
        upper_bound(negated, -limit), -1.0);
}

// deepseek_v4_swiglu：V4 的 SwiGLU 激活（带 clamp）。
//   gate 截断到上限 limit，up 截断到 [-limit, limit]，
//   然后 swiglu = silu(gate) × up。
// 这就是 V4 配置 swiglu_limit=10 的落地（gate 上限10、up 在[-10,10]）。
infinicore::Tensor deepseek_v4_swiglu(
    const infinicore::Tensor &gate,
    const infinicore::Tensor &up,
    float limit) {
    auto bounded_gate = upper_bound(gate, limit);
    auto bounded_up = lower_bound(upper_bound(up, limit), -limit);
    return infinicore::op::mul(
        infinicore::op::silu(bounded_gate), bounded_up);
}

// tensor_to_f32：把张量拷到 CPU 的 float 向量（router 的逐 token 逻辑用）。
std::vector<float> tensor_to_f32(const infinicore::Tensor &tensor) {
    auto source = cast_to(tensor, infinicore::DataType::F32)
                      ->to(infinicore::Device::cpu())
                      ->contiguous();
    std::vector<float> values(source->numel());
    std::memcpy(values.data(), source->data(), source->nbytes());
    return values;
}

// tensor_to_i64：把张量拷到 CPU 的 int64 向量（支持 I32/I64）。
std::vector<int64_t> tensor_to_i64(const infinicore::Tensor &tensor) {
    auto source = tensor->to(infinicore::Device::cpu())->contiguous();
    std::vector<int64_t> values(source->numel());
    if (source->dtype() == infinicore::DataType::I64) {
        std::memcpy(values.data(), source->data(), source->nbytes());
    } else if (source->dtype() == infinicore::DataType::I32) {
        const auto *input =
            reinterpret_cast<const int32_t *>(source->data());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = input[i];
        }
    } else {
        throw std::runtime_error(
            "DeepSeek-V4 router indices must use I32 or I64");
    }
    return values;
}

// sqrt_softplus：V4 的路由打分函数。
//   softplus(x) = log(1+e^x)（数值稳定的写法），再开根号。
//   这是 V4 的 scoring_func=sqrtsoftplus。
float sqrt_softplus(float value) {
    const float softplus =
        std::max(value, 0.0f) + std::log1p(std::exp(-std::abs(value)));
    return std::sqrt(softplus);
}

} // namespace

// DeepseekV4MLP：dense FFN（用于共享专家）。
//   gate_proj / up_proj / down_proj 三个线性层，中间过 swiglu。
DeepseekV4MLP::DeepseekV4MLP(
    size_t hidden_size,
    size_t intermediate_size,
    float swiglu_limit,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      swiglu_limit_(swiglu_limit) {
    if (hidden_size_ == 0 || intermediate_size_ == 0
        || swiglu_limit_ <= 0.0f) {
        throw std::runtime_error(
            "DeepSeek-V4 MLP configuration is invalid");
    }
    INFINICORE_NN_MODULE_INIT(
        gate_proj, hidden_size_, intermediate_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        up_proj, hidden_size_, intermediate_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        down_proj, intermediate_size_, hidden_size_, dtype, device);
}

// 前向：hidden → gate/up → swiglu(gate,up) → down
infinicore::Tensor DeepseekV4MLP::forward(
    const infinicore::Tensor &hidden_states) const {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size_) {
        throw std::runtime_error(
            "DeepSeek-V4 MLP expects [batch, sequence, hidden_size]");
    }
    auto gate = gate_proj_->forward(hidden_states);
    auto up = up_proj_->forward(hidden_states);
    return down_proj_->forward(
        deepseek_v4_swiglu(gate, up, swiglu_limit_));
}

// DeepseekV4Router：MoE 路由——为每个 token 选出用哪些专家，以及各自权重。
// 两种模式：
//   is_hash=true   ：hash router——直接用 tid2eid[input_id] 查表选专家（前3层用），
//                      gate 分数只用于加权，不用于选择。
//   is_hash=false  ：标准 router——gate 分数 + correction bias 排序，选 top-k。
// 路由权重：选中专家的 score 除以所有选中专家 score 之和（归一化）× routed_scaling_factor。
DeepseekV4Router::DeepseekV4Router(
    size_t hidden_size,
    size_t num_experts,
    size_t experts_per_token,
    size_t vocab_size,
    bool is_hash,
    float routed_scaling_factor,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : hidden_size_(hidden_size),
      num_experts_(num_experts),
      experts_per_token_(experts_per_token),
      vocab_size_(vocab_size),
      is_hash_(is_hash),
      routed_scaling_factor_(routed_scaling_factor) {
    if (hidden_size_ == 0 || num_experts_ == 0
        || experts_per_token_ == 0
        || experts_per_token_ > num_experts_
        || vocab_size_ == 0 || routed_scaling_factor_ <= 0.0f) {
        throw std::runtime_error(
            "DeepSeek-V4 router configuration is invalid");
    }
    // weight：算 gate logits（[num_experts, hidden_size]）
    INFINICORE_NN_PARAMETER_INIT(
        weight,
        ({num_experts_, hidden_size_}, dtype, device));
    if (is_hash_) {
        // hash 模式：查表权重 tid2eid[token_id][topk] = 选哪个专家
        INFINICORE_NN_PARAMETER_INIT(
            tid2eid,
            ({vocab_size_, experts_per_token_},
             infinicore::DataType::I64,
             device));
    } else {
        // 标准模式：每个专家一个 correction bias（加到 logits 上再排序）
        INFINICORE_NN_PARAMETER_INIT(
            e_score_correction_bias,
            ({num_experts_}, infinicore::DataType::F32, device));
    }
}

// 前向：hidden → gate logits → sqrt_softplus 打分 → 选专家 + 算路由权重
DeepseekV4RouterOutput DeepseekV4Router::forward(
    const infinicore::Tensor &hidden_states,
    const std::optional<infinicore::Tensor> &input_ids) const {
    if (!hidden_states || hidden_states->ndim() != 3
        || hidden_states->size(2) != hidden_size_) {
        throw std::runtime_error(
            "DeepSeek-V4 router expects [batch, sequence, hidden_size]");
    }
    const size_t batch_size = hidden_states->size(0);
    const size_t sequence_length = hidden_states->size(1);
    const size_t tokens = batch_size * sequence_length;
    // ① 算 gate logits：hidden × weight → [tokens, num_experts]
    auto logits = infinicore::op::linear(
        hidden_states->view({tokens, hidden_size_}),
        static_cast<infinicore::Tensor>(weight_),
        std::nullopt);
    // ② sqrt_softplus 打分
    const auto logits_host = tensor_to_f32(logits);
    std::vector<float> scores(logits_host.size());
    std::transform(
        logits_host.begin(), logits_host.end(), scores.begin(),
        sqrt_softplus);
    // ③ 准备选择所需的数据：
    //    hash 模式 → 读 tid2eid 查表 + token ids
    //    标准模式 → 读 correction bias
    std::vector<int64_t> hash_table;
    std::vector<int64_t> token_ids;
    std::vector<float> correction;
    if (is_hash_) {
        if (!input_ids.has_value() || !input_ids.value()
            || input_ids.value()->numel() != tokens) {
            throw std::runtime_error(
                "DeepSeek-V4 hash router requires one input id per token");
        }
        hash_table = tensor_to_i64(
            static_cast<infinicore::Tensor>(tid2eid_));
        token_ids = tensor_to_i64(input_ids.value());
    } else {
        correction = tensor_to_f32(
            static_cast<infinicore::Tensor>(e_score_correction_bias_));
    }

    std::vector<int32_t> selected(tokens * experts_per_token_);
    std::vector<float> routing(tokens * experts_per_token_);
    std::vector<size_t> expert_order(num_experts_);
    std::iota(expert_order.begin(), expert_order.end(), 0);
    // ④ 逐 token 选专家
    for (size_t token = 0; token < tokens; ++token) {
        if (is_hash_) {
            // hash 模式：直接查表 tid2eid[token_id][rank] = 专家号，不排序
            const int64_t token_id = token_ids[token];
            if (token_id < 0
                || static_cast<size_t>(token_id) >= vocab_size_) {
                throw std::runtime_error(
                    "DeepSeek-V4 hash router input id is out of range");
            }
            for (size_t rank = 0; rank < experts_per_token_; ++rank) {
                const int64_t expert = hash_table[
                    static_cast<size_t>(token_id) * experts_per_token_
                    + rank];
                if (expert < 0
                    || static_cast<size_t>(expert) >= num_experts_) {
                    throw std::runtime_error(
                        "DeepSeek-V4 tid2eid contains an invalid expert");
                }
                selected[token * experts_per_token_ + rank] =
                    static_cast<int32_t>(expert);
            }
        } else {
            // 标准模式：按 score + correction 排序，取前 experts_per_token 个
            std::partial_sort(
                expert_order.begin(),
                expert_order.begin() + experts_per_token_,
                expert_order.end(),
                [&](size_t lhs, size_t rhs) {
                    return scores[token * num_experts_ + lhs]
                               + correction[lhs]
                         > scores[token * num_experts_ + rhs]
                               + correction[rhs];
                });
            for (size_t rank = 0; rank < experts_per_token_; ++rank) {
                selected[token * experts_per_token_ + rank] =
                    static_cast<int32_t>(expert_order[rank]);
            }
        }
        // ⑤ 算路由权重：选中专家的 score / 所有选中 score 之和 × scaling_factor
        float denominator = 1e-20f;   // 防除零
        for (size_t rank = 0; rank < experts_per_token_; ++rank) {
            const auto expert = static_cast<size_t>(
                selected[token * experts_per_token_ + rank]);
            denominator += scores[token * num_experts_ + expert];
        }
        for (size_t rank = 0; rank < experts_per_token_; ++rank) {
            const auto expert = static_cast<size_t>(
                selected[token * experts_per_token_ + rank]);
            routing[token * experts_per_token_ + rank] =
                scores[token * num_experts_ + expert]
                / denominator * routed_scaling_factor_;
        }
    }

    // 返回选中的专家号和路由权重（拷回设备）
    auto selected_cpu = infinicore::Tensor::from_blob(
        selected.data(),
        {tokens, experts_per_token_},
        infinicore::DataType::I32,
        infinicore::Device::cpu());
    auto routing_cpu = infinicore::Tensor::from_blob(
        routing.data(),
        {tokens, experts_per_token_},
        infinicore::DataType::F32,
        infinicore::Device::cpu());
    return {
        routing_cpu->to(hidden_states->device()),
        selected_cpu->to(hidden_states->device()),
        logits};
}

// DeepseekV4Experts：专家层本身。
// 两种模式：
//   use_packed_fp4=true ：用 MXFP4 打包权重（U8 载体 + E8M0 scale），
//                          前向走 infinicore 的 fused_moe_mxfp4（带 SwigluLimit10）。
//   use_packed_fp4=false：普通权重，逐 token 逐专家手算。
DeepseekV4Experts::DeepseekV4Experts(
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size,
    float swiglu_limit,
    bool use_packed_fp4,
    const infinicore::DataType &dtype,
    const infinicore::Device &device)
    : num_experts_(num_experts),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      swiglu_limit_(swiglu_limit),
      use_packed_fp4_(use_packed_fp4) {
    if (num_experts_ == 0 || hidden_size_ == 0
        || intermediate_size_ == 0 || swiglu_limit_ <= 0.0f) {
        throw std::runtime_error(
            "DeepSeek-V4 experts configuration is invalid");
    }
    if (use_packed_fp4_) {
        register_packed_parameters_(device);
    } else {
        // 普通模式：每专家 gate_up_proj（w13 拼一起）+ down_proj
        INFINICORE_NN_PARAMETER_INIT(
            gate_up_proj,
            ({num_experts_, 2 * intermediate_size_, hidden_size_},
             dtype,
             device));
        INFINICORE_NN_PARAMETER_INIT(
            down_proj,
            ({num_experts_, hidden_size_, intermediate_size_},
             dtype,
             device));
    }
}

// 注册 MXFP4 打包参数。关键：打包权重是【共享一个大存储 + 每专家一个 slice 视图】。
// packed_w13_：所有专家的 w1/w3 权重打包在一个张量里（U8），
//   每个专家通过 narrow 切出自己的 slice 注册成独立参数。
// 这样做是为了 checkpoint 加载时名字能对上（experts.N.w1.weight_packed 等）。
void DeepseekV4Experts::register_packed_parameters_(
    const infinicore::Device &device) {
    if (hidden_size_ % 32 != 0 || intermediate_size_ % 32 != 0) {
        throw std::runtime_error(
            "DeepSeek-V4 MXFP4 dimensions must be divisible by 32");
    }
    // 大存储：形状是 [num_experts, rows, cols/2]（FP4 打包成 2 元素 1 字节）
    packed_w13_ = infinicore::Tensor::empty(
        {num_experts_, 2 * intermediate_size_, hidden_size_ / 2},
        infinicore::DataType::U8,
        device);
    w13_scale_ = infinicore::Tensor::empty(
        {num_experts_, 2 * intermediate_size_, hidden_size_ / 32},
        infinicore::DataType::U8,
        device);
    packed_w2_ = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, intermediate_size_ / 2},
        infinicore::DataType::U8,
        device);
    w2_scale_ = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, intermediate_size_ / 32},
        infinicore::DataType::U8,
        device);
    // 注册某个专家的某块为独立参数（narrow 出 slice 视图）
    auto register_slice = [&](const std::string &name,
                              const infinicore::Tensor &storage,
                              size_t expert,
                              size_t row_start,
                              size_t row_count) {
        auto view = storage
                        ->narrow({{0, expert, 1},
                                  {1, row_start, row_count}})
                        ->squeeze(0);
        this->register_parameter(
            name, infinicore::nn::Parameter(view));
    };
    // 每个专家的 w1/w3/w2 各自注册（从共享存储里切）
    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        register_slice(
            prefix + "w1.weight_packed",
            packed_w13_,
            expert,
            0,
            intermediate_size_);
        register_slice(
            prefix + "w1.weight_scale",
            w13_scale_,
            expert,
            0,
            intermediate_size_);
        register_slice(
            prefix + "w3.weight_packed",
            packed_w13_,
            expert,
            intermediate_size_,
            intermediate_size_);
        register_slice(
            prefix + "w3.weight_scale",
            w13_scale_,
            expert,
            intermediate_size_,
            intermediate_size_);
        register_slice(
            prefix + "w2.weight_packed",
            packed_w2_,
            expert,
            0,
            hidden_size_);
        register_slice(
            prefix + "w2.weight_scale",
            w2_scale_,
            expert,
            0,
            hidden_size_);
    }
}

// 专家前向：
//   FP4 模式 → 调 fused_moe_mxfp4（含 SwigluLimit10）——这是 InfiniCore 新增算子的消费点。
//   普通模式 → 逐 token 逐专家手算（gate_up → swiglu → down，乘路由权重累加）。
infinicore::Tensor DeepseekV4Experts::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &selected_experts,
    const infinicore::Tensor &routing_weights) const {
    if (!hidden_states || hidden_states->ndim() != 2
        || hidden_states->size(1) != hidden_size_
        || !selected_experts || selected_experts->ndim() != 2
        || !routing_weights || routing_weights->ndim() != 2
        || selected_experts->shape() != routing_weights->shape()
        || selected_experts->size(0) != hidden_states->size(0)) {
        throw std::runtime_error(
            "DeepSeek-V4 experts input shape mismatch");
    }
    // FP4 路径：整个 MoE 的 gate_up/激活/down 一步 fused 完成，
    //   激活类型是 SwigluLimit10（gate≤10、up∈[-10,10]）
    if (use_packed_fp4_) {
        return infinicore::op::fused_moe_mxfp4(
            hidden_states,
            selected_experts,
            routing_weights,
            packed_w13_,
            w13_scale_,
            packed_w2_,
            w2_scale_,
            infinicore::op::FusedMoeActivation::SwigluLimit10);
    }

    // 普通路径：逐 token 逐专家计算（数值参考实现，慢但正确）
    const auto selected = tensor_to_i64(selected_experts);
    const auto routing = tensor_to_f32(routing_weights);
    const size_t tokens = hidden_states->size(0);
    const size_t top_k = selected_experts->size(1);
    std::vector<infinicore::Tensor> token_outputs;
    token_outputs.reserve(tokens);
    for (size_t token = 0; token < tokens; ++token) {
        auto token_input = hidden_states
                               ->narrow({{0, token, 1}})
                               ->contiguous();
        auto token_output = infinicore::Tensor::zeros(
            {1, hidden_size_},
            hidden_states->dtype(),
            hidden_states->device());
        for (size_t rank = 0; rank < top_k; ++rank) {
            const int64_t expert = selected[token * top_k + rank];
            if (expert < 0
                || static_cast<size_t>(expert) >= num_experts_) {
                throw std::runtime_error(
                    "DeepSeek-V4 selected expert is out of range");
            }
            // gate_up：hidden × W13 → [2*intermediate]，切出 gate 和 up
            auto gate_up_weight =
                static_cast<infinicore::Tensor>(gate_up_proj_)
                    ->narrow({{0, static_cast<size_t>(expert), 1}})
                    ->squeeze(0)
                    ->contiguous();
            auto gate_up = infinicore::op::linear(
                token_input, gate_up_weight, std::nullopt);
            auto gate = gate_up
                            ->narrow({{1, 0, intermediate_size_}})
                            ->contiguous();
            auto up = gate_up
                          ->narrow(
                              {{1, intermediate_size_, intermediate_size_}})
                          ->contiguous();
            // swiglu（带 clamp）
            auto activated = deepseek_v4_swiglu(
                gate, up, swiglu_limit_);
            // down：activated × W2 → hidden
            auto down_weight =
                static_cast<infinicore::Tensor>(down_proj_)
                    ->narrow({{0, static_cast<size_t>(expert), 1}})
                    ->squeeze(0)
                    ->contiguous();
            auto contribution = infinicore::op::linear(
                activated, down_weight, std::nullopt);
            // 乘路由权重累加
            contribution = infinicore::op::mul_scalar(
                contribution, routing[token * top_k + rank]);
            token_output = infinicore::op::add(
                token_output, contribution);
        }
        token_outputs.push_back(token_output);
    }
    return infinicore::op::cat(token_outputs, 0);
}

// DeepseekV4SparseMoeBlock：一层的完整 MoE 模块 = 路由(gate) + 路由专家(experts) + 共享专家(shared)。
// 构造时按 mlp_layer_types[layer] 判断本层是 hash_moe 还是标准 MoE，
// 以及 expert_dtype 是否 fp4（决定 experts 用打包 FP4 还是普通权重）。
DeepseekV4SparseMoeBlock::DeepseekV4SparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    const auto &config = model_config->get_config_json();
    // 从 mlp_layer_types 判断本层是不是 hash_moe（前 3 层）
    const auto &mlp_layer_types = config.at("mlp_layer_types");
    if (layer_idx >= mlp_layer_types.size()) {
        throw std::runtime_error(
            "DeepSeek-V4 MoE layer index is out of range");
    }
    const bool is_hash =
        mlp_layer_types.at(layer_idx).get<std::string>() == "hash_moe";
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size =
        model_config->get<size_t>("moe_intermediate_size");
    const size_t num_experts = model_config->get<size_t>("num_experts");
    const size_t experts_per_token =
        model_config->get<size_t>("num_experts_per_tok");
    const size_t shared_intermediate_size =
        intermediate_size
        * model_config->get_or<size_t>("n_shared_experts", 1);
    // V4 的 swiglu_limit（gate/up 截断上限，默认 10）
    const float swiglu_limit =
        model_config->get_or<float>("swiglu_limit", 10.0f);
    // expert_dtype == "fp4" 时走打包 FP4 路径
    const bool use_packed_fp4 =
        model_config->get_or<std::string>("expert_dtype", "dense")
        == "fp4";
    const auto dtype = model_config->get_dtype();

    // 三个子模块：gate（路由）、experts（路由专家）、shared_experts（共享专家）
    INFINICORE_NN_MODULE_INIT(
        gate,
        hidden_size,
        num_experts,
        experts_per_token,
        model_config->get<size_t>("vocab_size"),
        is_hash,
        model_config->get_or<float>("routed_scaling_factor", 1.5f),
        dtype,
        device);
    INFINICORE_NN_MODULE_INIT(
        experts,
        num_experts,
        hidden_size,
        intermediate_size,
        swiglu_limit,
        use_packed_fp4,
        dtype,
        device);
    INFINICORE_NN_MODULE_INIT(
        shared_experts,
        hidden_size,
        shared_intermediate_size,
        swiglu_limit,
        dtype,
        device);
}

// 前向：路由选专家 → 路由专家算 → 加共享专家结果
infinicore::Tensor DeepseekV4SparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states,
    const std::optional<infinicore::Tensor> &input_ids) const {
    // ① 路由：得到每个 token 选中的专家 + 路由权重（hash 需要 input_ids 查表）
    auto routing = gate_->forward(hidden_states, input_ids);
    const auto shape = hidden_states->shape();
    // ② 路由专家：把 [B,S,D] 展平成 [B*S, D]，按路由结果算，再还原形状
    auto routed = experts_->forward(
                      hidden_states->view(
                          {shape[0] * shape[1], shape[2]}),
                      routing.selected_experts,
                      routing.routing_weights)
                      ->view(shape);
    // ③ 加上共享专家（每个 token 都过，不路由）
    return infinicore::op::add(
        routed, shared_experts_->forward(hidden_states));
}

} // namespace infinilm::models::deepseek_v4
