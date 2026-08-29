#pragma once

#include "graph_compiler.hpp"

#include <unordered_map>

namespace infinilm::engine {
class PagedCompiler : public GraphCompiler {
public:
    PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier);

    void compile() override;

    Compiled get_compiled(const InfinilmModel::Input &input) override;

    std::pair<std::shared_ptr<infinicore::graph::Graph>, infinicore::Tensor>
    get_sampling_compiled(size_t batch_size) override;

private:
    std::vector<size_t> decode_batch_sizes_;

    infinicore::Tensor block_tables_holder_;

    struct CompiledResult {
        InfinilmModel::Input input;
        Compiled compiled;
        // Small per-step inputs live as views into two contiguous device
        // buffers so each replay needs only two H2D copies. Host staging is
        // reused across steps.
        infinicore::Tensor pack_i64; // input_ids | position_ids | slot_mapping
        infinicore::Tensor pack_i32; // total_seq_lens | input_offsets | cu_seqlens
        std::vector<int64_t> stage_i64;
        std::vector<int32_t> stage_i32;
        size_t position_id_axes = 1;
        // Captured per-request argmax over the decode graph's logits blob,
        // replayed for greedy batches to avoid ~3 kernel launches per request.
        std::shared_ptr<infinicore::graph::Graph> sampling_graph;
        infinicore::Tensor sampling_out;
    };

    std::unordered_map<
        size_t, // num_requests
        CompiledResult>
        compiled_map_decode_;
};
} // namespace infinilm::engine
