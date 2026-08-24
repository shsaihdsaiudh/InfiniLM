#include "fp8_blockwise.hpp"

#include <infinicore/ops/fp8_blockwise_dequantize.hpp>
#include <infinicore/ops/linear.hpp>

#include <optional>
#include <stdexcept>

namespace infinilm::quantization {

FP8Blockwise::FP8Blockwise(const nlohmann::json &quant_config)
    : BaseQuantization(quant_config) {
    auto block_size = get_or<std::vector<size_t>>("weight_block_size", {128, 128});
    if (block_size.size() != 2 || block_size[0] == 0 || block_size[1] == 0) {
        throw std::runtime_error("FP8Blockwise: weight_block_size must be [BM, BN] with positive entries");
    }
    block_m_ = block_size[0];
    block_n_ = block_size[1];
}

std::vector<ParamDescriptor> FP8Blockwise::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {
    if (in_features % block_n_ != 0 || out_features % block_m_ != 0) {
        throw std::runtime_error("FP8Blockwise: in_features (" + std::to_string(in_features) + ") and out_features (" + std::to_string(out_features) + ") must be divisible by weight_block_size [" + std::to_string(block_m_) + ", " + std::to_string(block_n_) + "]");
    }
    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, infinicore::DataType::F8, split_dim, tp_rank, tp_size});
    // Scale is split across TP ranks along the same dim as the weight; the
    // per-rank shard size follows from the weight shard divided by the block.
    descs.push_back({"weight_scale_inv", {out_features / block_m_, in_features / block_n_}, infinicore::DataType::F32, split_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim >= 0 ? 0 : -1, split_dim >= 0 ? tp_rank : 0, split_dim >= 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor FP8Blockwise::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();

    // Naive path (correctness first): dequantize the block-scaled FP8 weight
    // to the activation dtype, then run the standard GEMM.
    auto weight = infinicore::op::fp8_blockwise_dequantize(
        params.at("weight"), params.at("weight_scale_inv"), input->dtype());

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }
    return infinicore::op::linear(input_contiguous, weight, bias_opt, alpha);
}

std::vector<SplitParam> FP8Blockwise::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {
    std::vector<SplitParam> result;
    const auto &weight = params.at("weight");
    const auto &weight_scale_inv = params.at("weight_scale_inv");
    const auto bias_it = params.find("bias");

    // SplitInfo start/size are in units of the weight's narrow_dim extent.
    // weight_scale_inv holds one entry per block, so its narrow range along
    // the same dim is divided by the block size (BM for dim0, BN for dim1).
    const size_t block = (narrow_dim == 0) ? block_m_ : block_n_;

    for (const auto &split : splits) {
        if (split.start % block != 0 || split.size % block != 0) {
            throw std::runtime_error("FP8Blockwise: split range at start=" + std::to_string(split.start) + " size=" + std::to_string(split.size) + " is not aligned to block size " + std::to_string(block));
        }
        result.push_back({split.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight->narrow({{static_cast<size_t>(narrow_dim), split.start, split.size}}),
                              narrow_dim, tp_rank, tp_size, split.num_shards)});
        result.push_back({split.prefix + ".weight_scale_inv",
                          infinicore::nn::Parameter(
                              weight_scale_inv->narrow({{static_cast<size_t>(narrow_dim), split.start / block, split.size / block}}),
                              narrow_dim, tp_rank, tp_size, split.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({split.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, split.start, split.size}}),
                                  0, tp_rank, tp_size, split.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> FP8Blockwise::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &,
    int) const {
    const auto weight_it = params.find("weight");
    const auto scale_it = params.find("weight_scale_inv");
    if (weight_it == params.end() || scale_it == params.end()) {
        throw std::runtime_error(
            "FP8Blockwise: post-load processing requires weight and weight_scale_inv");
    }
    return nullptr;
}

} // namespace infinilm::quantization
