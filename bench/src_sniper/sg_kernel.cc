#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "reader.h"

#include "graphbrew/partition/cagra/popt.h"
#include "sniper_sim/sniper_harness.h"

// ECG mode 6 (per-edge mask) builder — shared with cache_sim and gem5.
#include "ecg_mode6_builder.h"
// Shared per-edge next-ref epoch builder (SNIPER_ECG_EXTRACT delivery; same SSOT as
// cache_sim/gem5).
#include "ecg_epoch_builder.h"

// File-backed kernel diagnostic target. Native execution is intentionally kept
// lightweight for checking .sg parameters and sideband export. Do not use this
// as a default Sniper/SDE workload until the frontend high-memory run mode is
// fixed; roi_matrix.py keeps it guarded by default.

namespace {

using ScoreT = float;
constexpr float kDamp = 0.85f;
constexpr WeightT kDistInf = std::numeric_limits<WeightT>::max() / 2;

struct Options {
    std::string benchmark = "pr";
    std::string graph_path;
    int max_iters = 2;
    NodeID source = 0;
    WeightT delta = 1;
    std::string reorder_spec;   // -o value (e.g. "5" = DBG); empty = no reorder
    bool symmetrize = false;    // -s
};

bool has_value(int index, int argc) {
    return index + 1 < argc;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--benchmark" || arg == "-B") && has_value(i, argc)) {
            options.benchmark = argv[++i];
        } else if (arg == "-f" && has_value(i, argc)) {
            options.graph_path = argv[++i];
        } else if (arg == "-i" && has_value(i, argc)) {
            options.max_iters = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "-r" && has_value(i, argc)) {
            options.source = static_cast<NodeID>(std::atol(argv[++i]));
        } else if (arg == "-d" && has_value(i, argc)) {
            options.delta = static_cast<WeightT>(std::atol(argv[++i]));
            if (options.delta <= 0) options.delta = 1;
        } else if (arg == "-o" && has_value(i, argc)) {
            options.reorder_spec = argv[++i];   // forward to Builder reorder (was discarded!)
        } else if ((arg == "-g" || arg == "-k" || arg == "-n" || arg == "-t") && has_value(i, argc)) {
            ++i;
        } else if (arg == "-s") {
            options.symmetrize = true;
        } else if (arg == "-a" || arg == "-v" || arg == "--") {
            continue;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: sg_kernel --benchmark pr|bfs|sssp|bc|cc -f graph.sg [-i iters] [-r source] [-d delta]\n";
            std::exit(0);
        }
    }
    return options;
}

// Build a minimal GAPBS CLI argv from the parsed options so the graph is loaded
// through Builder.MakeGraph() — IDENTICAL path to cache_sim/gem5 (bench/src_sim,
// bench/src_gem5 pr.cc), which applies the -o reorder. Reading the .sg directly
// (the old behaviour) silently skipped the reorder, making all Sniper degree-
// policy runs operate on UNREORDERED graphs.
namespace {
std::vector<std::string> build_gapbs_args(const Options& opt) {
    std::vector<std::string> args = {"sg_kernel", "-f", opt.graph_path};
    if (opt.symmetrize) args.push_back("-s");
    if (!opt.reorder_spec.empty()) {
        args.push_back("-o");
        args.push_back(opt.reorder_spec);
    }
    return args;
}
}  // namespace

Graph load_graph(const Options& opt) {
    std::vector<std::string> args = build_gapbs_args(opt);
    std::vector<char*> cargv;
    cargv.reserve(args.size());
    for (auto& s : args) cargv.push_back(const_cast<char*>(s.c_str()));
    CLApp cli(static_cast<int>(cargv.size()), cargv.data(), "sg_kernel");
    cli.ParseArgs();
    Builder b(cli);
    return b.MakeGraph();
}

WGraph load_weighted_graph(const Options& opt) {
    std::vector<std::string> args = build_gapbs_args(opt);
    std::vector<char*> cargv;
    cargv.reserve(args.size());
    for (auto& s : args) cargv.push_back(const_cast<char*>(s.c_str()));
    CLDelta<WeightT> cli(static_cast<int>(cargv.size()), cargv.data(), "sg_kernel");
    cli.ParseArgs();
    WeightedBuilder b(cli);
    return b.MakeGraph();
}

template <typename GraphType, typename ValueT>
void export_popt_for_graph(const GraphType& graph) {
    constexpr int kNumEpochs = 256;
    const int num_vtx_per_line = std::max<int>(1, 64 / sizeof(ValueT));
    pvector<uint8_t> popt_matrix;
    // Only BFS + SSSP call this helper; both traverse OUT-edges reading the dest
    // property, so the reref matrix is the graph TRANSPOSE (CSC/in_neigh,
    // traverseCSR=false) — matching cache_sim. PR uses its own inline call (true).
    // Undirected graphs force true internally (out==in), so this is do-no-harm there.
    makeOffsetMatrix(graph, popt_matrix, num_vtx_per_line, kNumEpochs, /*traverseCSR=*/false);
    int num_cache_lines = (graph.num_nodes() + num_vtx_per_line - 1) / num_vtx_per_line;
    sniper_export_popt_matrix(popt_matrix.data(), num_cache_lines, kNumEpochs, graph.num_nodes());
}

int run_pr(const Graph& graph, int max_iters) {
    const ScoreT init_score = graph.num_nodes() > 0 ? 1.0f / graph.num_nodes() : 0.0f;
    const ScoreT base_score = graph.num_nodes() > 0 ? (1.0f - kDamp) / graph.num_nodes() : 0.0f;
    pvector<ScoreT> scores(graph.num_nodes(), init_score);
    pvector<ScoreT> contrib(graph.num_nodes(), 0.0f);

    for (NodeID node = 0; node < graph.num_nodes(); ++node) {
        int64_t degree = graph.out_degree(node);
        contrib[node] = degree > 0 ? scores[node] / degree : 0.0f;
    }

    SniperPropertyRegion regions[2] = {
        {"scores", reinterpret_cast<uint64_t>(scores.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(ScoreT),
            static_cast<uint32_t>(graph.num_nodes()), sizeof(ScoreT), true},
        {"contrib", reinterpret_cast<uint64_t>(contrib.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(ScoreT),
            static_cast<uint32_t>(graph.num_nodes()), sizeof(ScoreT), true},
    };
    SniperEdgeRegion edge_regions[2];
    int num_edge_regions = sniper_make_edge_regions(graph, edge_regions, 2, true);
    sniper_export_context(regions, 2, graph, nullptr, edge_regions, num_edge_regions);

    // Build POPT matrix inline (was export_popt_for_graph helper) so we
    // can reuse the matrix to derive the per-edge mode 6 mask.
    constexpr int kNumEpochs = 256;
    const int num_vtx_per_line = std::max<int>(1, 64 / static_cast<int>(sizeof(ScoreT)));
    pvector<uint8_t> popt_matrix;
    makeOffsetMatrix(graph, popt_matrix, num_vtx_per_line, kNumEpochs);
    const int popt_num_cache_lines =
        (graph.num_nodes() + num_vtx_per_line - 1) / num_vtx_per_line;
    sniper_export_popt_matrix(popt_matrix.data(), popt_num_cache_lines,
                              kNumEpochs, graph.num_nodes());

    // Per-edge next-reference epoch delivery for ECG_GRASP_POPT. PR pulls
    // IN-edges reading contrib[neighbor], so each entry in node's in-neighbour
    // list carries neighbor's next-reference epoch (the default PR direction,
    // push_out_edges=false). Build before ROI; deliver immediately before the
    // governed contrib[] demand, matching cache_sim/gem5.
    const bool ecg_extract_on = graphbrew_sniper::ecg_extract_enabled();
    const uint32_t ecg_epoch_count = static_cast<uint32_t>(
        graphbrew_sniper::env_int_clamped(
            "ECG_EDGE_MASK_EPOCHS", kNumEpochs, 2, 65535));
    std::vector<std::vector<uint16_t>> in_edge_epochs_by_src;
    if (ecg_extract_on) {
        ecg_epoch::buildInEdgeEpochs(
            graph, num_vtx_per_line, ecg_epoch_count,
            /*linemin=*/true, in_edge_epochs_by_src);
        const char* debug = std::getenv("ECG_DEBUG");
        if (debug && debug[0] && std::string(debug) != "0") {
            uint64_t total = 0;
            uint64_t nonzero = 0;
            uint16_t min_epoch = std::numeric_limits<uint16_t>::max();
            uint16_t max_epoch = 0;
            for (const auto& epochs : in_edge_epochs_by_src) {
                for (uint16_t epoch : epochs) {
                    ++total;
                    if (epoch != 0) ++nonzero;
                    min_epoch = std::min(min_epoch, epoch);
                    max_epoch = std::max(max_epoch, epoch);
                }
            }
            if (total == 0) min_epoch = 0;
            std::fprintf(stderr,
                         "[ECG-EPOCH-BUILD sim=sniper kernel=pr total=%llu "
                         "nonzero=%llu min=%u max=%u ne=%u]\n",
                         (unsigned long long)total,
                         (unsigned long long)nonzero,
                         static_cast<unsigned>(min_epoch),
                         static_cast<unsigned>(max_epoch),
                         ecg_epoch_count);
        }
    }

    // Lookahead distance for ECG_PFX hints. node+1 is too close on
    // small graphs (Sniper's cache_cntlr.cc:1146 filters
    // already-in-cache addresses, dropping ECG_PFX hints whose target
    // line is still warm from the previous vertex). Pull from env so
    // sweeps can tune per-graph; default 8 gives PREFETCH_INTERVAL
    // ~250 cycles × 8 vertex iterations ≈ 2000 cycles of head-start.
    const char* pfx_lookahead_env = std::getenv("SNIPER_ECG_PFX_LOOKAHEAD");
    const NodeID pfx_lookahead =
        (pfx_lookahead_env && pfx_lookahead_env[0])
            ? std::max(1, std::atoi(pfx_lookahead_env))
            : 8;

    // === Mode 6: Per-Edge ECG Mask (paper's ECG design) ===
    // SNIPER_ECG_PFX_MODE (preferred) or ECG_PREFETCH_MODE selects the
    // prefetch policy. Mode 6 = per-edge mask path; anything else falls
    // back to the trivial linear-lookahead path below.
    const char* mode_env = std::getenv("SNIPER_ECG_PFX_MODE");
    if (!mode_env || !mode_env[0]) mode_env = std::getenv("ECG_PREFETCH_MODE");
    const int ecg_pfx_mode = (mode_env && mode_env[0]) ? std::atoi(mode_env) : 0;
    const char* ecg_enable_env = std::getenv("SNIPER_ENABLE_ECG_PFX_HINTS");
    const bool ecg_enabled = ecg_enable_env && std::string(ecg_enable_env) != "0";

    // Build mode-6 fat-mask array BEFORE entering ROI (otherwise
    // Sniper cycle-accurately simulates the offline construction
    // pass, adding 3M-edge × allocation-heavy work to the timed
    // region — see rubber-duck #1 from sprint 6f-6 closeout).
    std::vector<std::vector<uint64_t>> in_edge_masks_by_src;
    if (ecg_enabled && ecg_pfx_mode == 6) {
        std::vector<uint8_t> avg_reref_by_line;
        ecg_mode6::computeAvgRerefByLine(popt_matrix.data(), popt_num_cache_lines,
                                         kNumEpochs, avg_reref_by_line);
        std::vector<uint8_t> tiers;
        ecg_mode6::computeDegreeTiers(graph, tiers);
        ecg_mode6::buildInEdgeMasks(graph, tiers, avg_reref_by_line,
                                    pfx_lookahead, num_vtx_per_line,
                                    in_edge_masks_by_src, "sniper-sg-PR");
        std::printf("[sniper-sg ECG mode 6] lookahead=%d (per-edge mask path active)\n",
                    pfx_lookahead);
    }

    // === Kernel-side hint dedup with O(1) bitmap (bug 5A fix + rubber-duck #2) ===
    //
    // Each SNIPER_ECG_PFX_TARGET call traps to Sniper main-thread
    // (Pin context-switch, ~5-50us each). For graphs like
    // delaunay_n19 (3M edges × 2 iter = 6.3M calls), this dominates
    // wall time. Suppress calls where the target was emitted within
    // the last KERNEL_DEDUP_WINDOW emissions.
    //
    // First implementation used a linear-scan ring buffer (O(window)
    // per edge → 3M × 256 = 1.6B comparisons that Sniper
    // cycle-accurately simulated). Replaced with an O(1) bitmap
    // indexed by cache-line / hash of cache-line: each emission sets
    // a bit; check is single load. To preserve the recency-window
    // semantics we age the bitmap every WINDOW emissions by clearing
    // and replaying.
    //
    // Tunable via SNIPER_ECG_PFX_KERNEL_DEDUP env var.
    int kernel_dedup_window;
    {
        const char* v = std::getenv("SNIPER_ECG_PFX_KERNEL_DEDUP");
        kernel_dedup_window = (v && v[0]) ? std::atoi(v) : 256;
        if (kernel_dedup_window < 0) kernel_dedup_window = 0;
        if (kernel_dedup_window > (1 << 16)) kernel_dedup_window = (1 << 16);
    }

    // Sprint 6f-7 Phase 3: per-edge AMPLIFY (matches cache_sim mode 6).
    //
    // For each edge, after emitting the encoded prefetch_target from
    // the mask, also emit prefetches for the next AMPLIFY masks'
    // decoded destinations. AMPLIFY=0 (default) preserves prior
    // single-target-per-edge behavior. AMPLIFY=N adds N sequential
    // next-dest prefetches per edge (mirrors cache_sim's
    // ECG_EDGE_MASK_AMPLIFY env var).
    //
    // Per cache_sim Phase 2.6 finding, AMPLIFY saturates at 1 because
    // the dedup window absorbs additional candidates. AMPLIFY=1 is
    // the cache_sim-validated sweet spot.
    int amplify;
    {
        const char* v = std::getenv("SNIPER_ECG_EDGE_MASK_AMPLIFY");
        amplify = (v && v[0]) ? std::atoi(v) : 0;
        if (amplify < 0) amplify = 0;
        if (amplify > 8) amplify = 8;
    }
    // Bitmap sized to property-array cache-line count. One bit per
    // property cache line.
    const uint32_t num_property_lines =
        (graph.num_nodes() + num_vtx_per_line - 1) / num_vtx_per_line;
    std::vector<uint64_t> dedup_bitmap;
    if (kernel_dedup_window > 0) {
        dedup_bitmap.assign((num_property_lines + 63) / 64, 0);
    }
    uint64_t kernel_emit_count = 0;
    uint64_t kernel_dedup_count = 0;
    uint64_t emit_since_clear = 0;

    SNIPER_ROI_BEGIN();

    for (int iter = 0; iter < max_iters; ++iter) {
        for (NodeID node = 0; node < graph.num_nodes(); ++node) {
            SNIPER_SET_VERTEX(node);
            ScoreT incoming_total = 0.0f;

            // Mode 6: per-edge ECG fat-mask path (paper's ECG ISA design).
            //
            // The fat-mask REPLACES the CSR edge entry: instead of
            // loading a 4-byte vertex ID from CSR + a separate 8-byte
            // mask, we load ONE 8-byte fat-mask that contains both
            // the dest (lower 24 bits) and the prefetch info (upper
            // bits). This matches §3.2 of the paper: ecg_extract
            // takes a single 64-bit fat-ID register and decodes
            // vertex + DBG + POPT + prefetch atomically.
            //
            // Earlier revision (sprint 6f-6 initial port) read BOTH
            // the CSR and the mask per edge, doubling memory cost
            // and producing an 8x DRAM-traffic regression on Sniper.
            // The current implementation iterates ONLY the mask
            // array; CSR is bypassed entirely in this path.
            if (ecg_enabled && ecg_pfx_mode == 6 &&
                node < static_cast<NodeID>(in_edge_masks_by_src.size())) {
                const auto& src_masks = in_edge_masks_by_src[node];
                const size_t num_masks = src_masks.size();
                for (size_t edge_idx = 0; edge_idx < num_masks; ++edge_idx) {
                    const uint64_t mask = src_masks[edge_idx];
                    NodeID neighbor = static_cast<NodeID>(ecg_mode6::extractDest(mask));
                    if (neighbor < 0 || neighbor >= graph.num_nodes()) continue;
                    if (ecg_extract_on &&
                        static_cast<size_t>(node) < in_edge_epochs_by_src.size()) {
                        const auto& eps = in_edge_epochs_by_src[node];
                        uint16_t ep = (edge_idx < eps.size()) ? eps[edge_idx]
                            : static_cast<uint16_t>(ecg_epoch_count - 1);
                        SNIPER_ECG_EXTRACT(neighbor, ep);
                    }
                    uint32_t prefetch_target = ecg_mode6::extractPrefetchTarget(mask);
                    if (prefetch_target != 0 &&
                        prefetch_target < static_cast<uint32_t>(graph.num_nodes())) {
                        // O(1) bitmap dedup (rubber-duck #2 fix).
                        // Suppress emission if cache-line was already
                        // emitted within the recency window.
                        uint32_t target_line = prefetch_target /
                            static_cast<uint32_t>(num_vtx_per_line);
                        bool seen = false;
                        if (kernel_dedup_window > 0) {
                            size_t word_idx = target_line / 64;
                            uint64_t bit = uint64_t{1} << (target_line % 64);
                            if (word_idx < dedup_bitmap.size()) {
                                if (dedup_bitmap[word_idx] & bit) {
                                    seen = true;
                                } else {
                                    dedup_bitmap[word_idx] |= bit;
                                }
                            }
                        }
                        if (!seen) {
                            SNIPER_ECG_PFX_TARGET(prefetch_target);
                            kernel_emit_count++;
                            emit_since_clear++;
                            // Age the bitmap every WINDOW emissions
                            // to keep dedup window-bounded recency
                            // semantics (without per-emit O(window)
                            // scans).
                            if (kernel_dedup_window > 0 &&
                                emit_since_clear >= static_cast<uint64_t>(kernel_dedup_window)) {
                                std::fill(dedup_bitmap.begin(), dedup_bitmap.end(), 0);
                                emit_since_clear = 0;
                            }
                        } else {
                            kernel_dedup_count++;
                        }
                    }
                    // Sprint 6f-7 Phase 3: AMPLIFY = emit next-N decoded
                    // dests as additional prefetches. Mirrors cache_sim
                    // mode 6 AMPLIFY semantics. AMPLIFY=0 (default)
                    // = no extra emissions = unchanged from before.
                    for (int step = 1; step <= amplify; ++step) {
                        const size_t fwd_idx = edge_idx + static_cast<size_t>(step);
                        if (fwd_idx >= num_masks) break;
                        const uint32_t fwd_dest = ecg_mode6::extractDest(src_masks[fwd_idx]);
                        if (fwd_dest == 0 ||
                            fwd_dest >= static_cast<uint32_t>(graph.num_nodes())) continue;
                        uint32_t fwd_line = fwd_dest /
                            static_cast<uint32_t>(num_vtx_per_line);
                        bool fwd_seen = false;
                        if (kernel_dedup_window > 0) {
                            size_t word_idx = fwd_line / 64;
                            uint64_t bit = uint64_t{1} << (fwd_line % 64);
                            if (word_idx < dedup_bitmap.size()) {
                                if (dedup_bitmap[word_idx] & bit) {
                                    fwd_seen = true;
                                } else {
                                    dedup_bitmap[word_idx] |= bit;
                                }
                            }
                        }
                        if (!fwd_seen) {
                            SNIPER_ECG_PFX_TARGET(fwd_dest);
                            kernel_emit_count++;
                            emit_since_clear++;
                            if (kernel_dedup_window > 0 &&
                                emit_since_clear >= static_cast<uint64_t>(kernel_dedup_window)) {
                                std::fill(dedup_bitmap.begin(), dedup_bitmap.end(), 0);
                                emit_since_clear = 0;
                            }
                        } else {
                            kernel_dedup_count++;
                        }
                    }
                    incoming_total += contrib[neighbor];
                }
            } else {
                // Default: trivial linear lookahead (preserves prior
                // behavior when mode != 6).
                NodeID pfx_target = node + pfx_lookahead;
                if (pfx_target < graph.num_nodes()) {
                    SNIPER_ECG_PFX_TARGET(pfx_target);
                }
                size_t edge_pos = 0;
                for (NodeID neighbor : graph.in_neigh(node)) {
                    if (ecg_extract_on &&
                        static_cast<size_t>(node) < in_edge_epochs_by_src.size()) {
                        const auto& eps = in_edge_epochs_by_src[node];
                        uint16_t ep = (edge_pos < eps.size()) ? eps[edge_pos]
                            : static_cast<uint16_t>(ecg_epoch_count - 1);
                        SNIPER_ECG_EXTRACT(neighbor, ep);
                    }
                    ++edge_pos;
                    incoming_total += contrib[neighbor];
                }
            }

            scores[node] = base_score + kDamp * incoming_total;
            int64_t degree = graph.out_degree(node);
            contrib[node] = degree > 0 ? scores[node] / degree : 0.0f;
        }
    }
    SNIPER_ROI_END();

    if (ecg_enabled && ecg_pfx_mode == 6) {
        std::printf("[sniper-sg ECG mode 6] emit=%llu kernel-dedup-skip=%llu (window=%d)\n",
                    static_cast<unsigned long long>(kernel_emit_count),
                    static_cast<unsigned long long>(kernel_dedup_count),
                    kernel_dedup_window);
    }

    ScoreT checksum = 0.0f;
    for (ScoreT score : scores) checksum += score;
    std::cout << "GraphBrew Sniper SG PR checksum: " << checksum << std::endl;
    return std::fabs(checksum) > 0.0f ? 0 : 1;
}

int run_bfs(const Graph& graph, NodeID source) {
    if (source < 0 || source >= graph.num_nodes()) source = 0;
    pvector<NodeID> parent(graph.num_nodes(), -1);
    parent[source] = source;

    SniperPropertyRegion regions[1] = {
        {"parent", reinterpret_cast<uint64_t>(parent.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(NodeID),
         static_cast<uint32_t>(graph.num_nodes()), sizeof(NodeID)},
    };
    SniperEdgeRegion edge_regions[2];
    int num_edge_regions = sniper_make_edge_regions(graph, edge_regions, 2);
    sniper_export_context(regions, 1, graph, nullptr, edge_regions, num_edge_regions);
    export_popt_for_graph<Graph, NodeID>(graph);

    // SNIPER_ECG_EXTRACT (delivery-faithful, mirrors gem5 ecg.load EVICT): deliver each
    // demand edge's next-ref epoch so ECG_GRASP_POPT ranks parent[] by a delivered epoch
    // instead of the host-side findNextRef matrix. BFS-TD pushes OUT-edges writing
    // parent[dest]; dest is next-referenced by its IN-neighbours -> push_out_edges=true
    // (the transpose; same builder as cache_sim/gem5). Gated on SNIPER_ENABLE_ECG_EXTRACT.
    constexpr uint32_t kNumVtxPerLine = 64 / sizeof(NodeID);
    const bool ecg_extract_on = graphbrew_sniper::ecg_extract_enabled();
    const uint32_t ecg_epoch_count = static_cast<uint32_t>(
        graphbrew_sniper::env_int_clamped("ECG_EDGE_MASK_EPOCHS", 256, 2, 65535));
    std::vector<std::vector<uint16_t>> out_edge_epochs;
    if (ecg_extract_on) {
        ecg_epoch::buildInEdgeEpochs(graph, kNumVtxPerLine, ecg_epoch_count,
                                     /*linemin=*/true, out_edge_epochs,
                                     /*push_out_edges=*/true);
    }

    SNIPER_ROI_BEGIN();
    std::queue<NodeID> frontier;
    frontier.push(source);
    while (!frontier.empty()) {
        NodeID node = frontier.front();
        frontier.pop();
        SNIPER_SET_VERTEX(node);
        // ECG_PFX hint: emit the head of the frontier (the node we'll expand next) so the
        // prefetcher can warm parent[next_node]. Env-gated (SNIPER_ENABLE_ECG_PFX_HINTS).
        if (!frontier.empty()) {
            SNIPER_ECG_PFX_TARGET(frontier.front());
        }
        const std::vector<uint16_t>* eps =
            (ecg_extract_on && static_cast<size_t>(node) < out_edge_epochs.size())
                ? &out_edge_epochs[node] : nullptr;
        size_t edge_pos = 0;
        for (NodeID neighbor : graph.out_neigh(node)) {
            // Deliver neighbor's epoch BEFORE reading parent[neighbor] so cache_set_ecg
            // stamps the property line on fill.
            if (eps) {
                uint16_t ep = (edge_pos < eps->size()) ? (*eps)[edge_pos]
                    : static_cast<uint16_t>(ecg_epoch_count - 1);
                SNIPER_ECG_EXTRACT(neighbor, ep);
            }
            ++edge_pos;
            if (parent[neighbor] == -1) {
                parent[neighbor] = node;
                frontier.push(neighbor);
            }
        }
    }
    SNIPER_ROI_END();

    int64_t reached = 0;
    for (NodeID value : parent) reached += value >= 0 ? 1 : 0;
    std::cout << "GraphBrew Sniper SG BFS reached: " << static_cast<long long>(reached) << std::endl;
    return reached > 0 ? 0 : 1;
}

int run_sssp(const WGraph& graph, NodeID source, WeightT delta) {
    (void)delta;
    if (source < 0 || source >= graph.num_nodes()) source = 0;
    pvector<WeightT> dist(graph.num_nodes(), kDistInf);
    pvector<uint8_t> in_queue(graph.num_nodes(), 0);
    dist[source] = 0;

    SniperPropertyRegion regions[1] = {
        {"dist", reinterpret_cast<uint64_t>(dist.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(WeightT),
         static_cast<uint32_t>(graph.num_nodes()), sizeof(WeightT)},
    };
    SniperEdgeRegion edge_regions[2];
    int num_edge_regions = sniper_make_edge_regions(graph, edge_regions, 2);
    sniper_export_context(regions, 1, graph, nullptr, edge_regions, num_edge_regions);
    export_popt_for_graph<WGraph, WeightT>(graph);

    // SNIPER_ECG_EXTRACT (delivery-faithful, mirrors gem5 ecg.load EVICT): deliver each
    // relaxed edge's next-ref epoch so ECG_GRASP_POPT ranks dist[] by a delivered epoch.
    // SSSP relaxes OUT-edges reading dist[dest]; dest is next-referenced by its
    // IN-neighbours -> push_out_edges=true (transpose). Gated on SNIPER_ENABLE_ECG_EXTRACT.
    constexpr uint32_t kNumVtxPerLine = 64 / sizeof(WeightT);
    const bool ecg_extract_on = graphbrew_sniper::ecg_extract_enabled();
    const uint32_t ecg_epoch_count = static_cast<uint32_t>(
        graphbrew_sniper::env_int_clamped("ECG_EDGE_MASK_EPOCHS", 256, 2, 65535));
    std::vector<std::vector<uint16_t>> out_edge_epochs;
    if (ecg_extract_on) {
        ecg_epoch::buildInEdgeEpochs(graph, kNumVtxPerLine, ecg_epoch_count,
                                     /*linemin=*/true, out_edge_epochs,
                                     /*push_out_edges=*/true);
    }

    SNIPER_ROI_BEGIN();
    std::queue<NodeID> frontier;
    frontier.push(source);
    in_queue[source] = 1;
    while (!frontier.empty()) {
        NodeID node = frontier.front();
        frontier.pop();
        in_queue[node] = 0;
        SNIPER_SET_VERTEX(node);
        // ECG_PFX hint: emit the head of the frontier (next node to expand) so the
        // prefetcher can warm dist[next]. Env-gated.
        if (!frontier.empty()) {
            SNIPER_ECG_PFX_TARGET(frontier.front());
        }
        const std::vector<uint16_t>* eps =
            (ecg_extract_on && static_cast<size_t>(node) < out_edge_epochs.size())
                ? &out_edge_epochs[node] : nullptr;
        size_t edge_pos = 0;
        for (WNode edge : graph.out_neigh(node)) {
            // Deliver edge.v's epoch BEFORE reading dist[edge.v] so cache_set_ecg stamps
            // the property line on fill.
            if (eps) {
                uint16_t ep = (edge_pos < eps->size()) ? (*eps)[edge_pos]
                    : static_cast<uint16_t>(ecg_epoch_count - 1);
                SNIPER_ECG_EXTRACT(edge.v, ep);
            }
            ++edge_pos;
            WeightT candidate = dist[node] + edge.w;
            if (candidate < dist[edge.v]) {
                dist[edge.v] = candidate;
                if (!in_queue[edge.v]) {
                    frontier.push(edge.v);
                    in_queue[edge.v] = 1;
                }
            }
        }
    }
    SNIPER_ROI_END();

    int64_t reached = 0;
    uint64_t checksum = 0;
    for (WeightT value : dist) {
        if (value < kDistInf) {
            reached++;
            checksum += static_cast<uint64_t>(value);
        }
    }
    std::cout << "GraphBrew Sniper SG SSSP reached/checksum: "
              << static_cast<long long>(reached) << " / " << checksum << std::endl;
    return reached > 0 ? 0 : 1;
}

// ── CC (Afforest) union-find helpers — single-threaded (equivalence workload) ──
// Same logic as bench/src_sniper/cc.cc with the atomics removed (no CAS needed on
// one thread); the eviction DECISION is thread-count-agnostic.
void cc_link(NodeID u, NodeID v, pvector<NodeID>& comp) {
    NodeID p1 = comp[u];
    NodeID p2 = comp[v];
    while (p1 != p2) {
        NodeID high = p1 > p2 ? p1 : p2;
        NodeID low = p1 + (p2 - high);
        NodeID p_high = comp[high];
        if (p_high == low) break;
        if (p_high == high) { comp[high] = low; break; }
        p1 = comp[comp[high]];
        p2 = comp[low];
    }
}

void cc_compress(const Graph& g, pvector<NodeID>& comp) {
    for (NodeID n = 0; n < g.num_nodes(); n++)
        while (comp[n] != comp[comp[n]])
            comp[n] = comp[comp[n]];
}

// Betweenness Centrality (Brandes) — single-threaded port of the audited
// Brandes_Sniper (bench/src_sniper/bc.cc): four grasp-protected vertex property
// regions (scores/depth/path_counts/deltas), transpose P-OPT reref matrix keyed on
// depth (BC pushes OUT-edges reading depth[dest] -> traverseCSR=false), and per-edge
// ECG epoch delivery. Mirrors the cache_sim/gem5 BC so the shared eviction decision
// is exercised identically across the three simulators.
int run_bc(const Graph& graph, int num_iters) {
    pvector<ScoreT> scores(graph.num_nodes(), ScoreT(0));
    pvector<int32_t> depth(graph.num_nodes(), int32_t(-1));
    pvector<int64_t> path_counts(graph.num_nodes(), int64_t(0));
    pvector<ScoreT> deltas(graph.num_nodes(), ScoreT(0));

    SniperPropertyRegion regions[4] = {
        {"scores", reinterpret_cast<uint64_t>(scores.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(ScoreT),
         static_cast<uint32_t>(graph.num_nodes()), sizeof(ScoreT), true},
        {"depth", reinterpret_cast<uint64_t>(depth.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(int32_t),
         static_cast<uint32_t>(graph.num_nodes()), sizeof(int32_t), true},
        {"path_counts", reinterpret_cast<uint64_t>(path_counts.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(int64_t),
         static_cast<uint32_t>(graph.num_nodes()), sizeof(int64_t), true},
        {"deltas", reinterpret_cast<uint64_t>(deltas.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(ScoreT),
         static_cast<uint32_t>(graph.num_nodes()), sizeof(ScoreT), true},
    };
    SniperEdgeRegion edge_regions[2];
    int num_edge_regions = sniper_make_edge_regions(graph, edge_regions, 2, true);
    sniper_export_context(regions, 4, graph, nullptr, edge_regions, num_edge_regions);

    constexpr int kNumVtxPerLine = 64 / sizeof(int32_t);
    constexpr int kNumEpochs = 256;
    pvector<uint8_t> popt_matrix;
    int popt_num_cache_lines = (graph.num_nodes() + kNumVtxPerLine - 1) / kNumVtxPerLine;
    makeOffsetMatrix(graph, popt_matrix, kNumVtxPerLine, kNumEpochs, /*traverseCSR=*/false);
    sniper_export_popt_matrix(popt_matrix.data(), popt_num_cache_lines,
                              kNumEpochs, graph.num_nodes());

    const bool ecg_extract_on = graphbrew_sniper::ecg_extract_enabled();
    const uint32_t ecg_epoch_count = static_cast<uint32_t>(
        graphbrew_sniper::env_int_clamped("ECG_EDGE_MASK_EPOCHS", kNumEpochs, 2, 65535));
    std::vector<std::vector<uint16_t>> out_edge_epochs;
    if (ecg_extract_on) {
        ecg_epoch::buildInEdgeEpochs(graph, kNumVtxPerLine, ecg_epoch_count,
                                     /*linemin=*/true, out_edge_epochs,
                                     /*push_out_edges=*/true);
    }
    auto deliver = [&](NodeID u, size_t edge_pos, NodeID v) {
        if (!ecg_extract_on || static_cast<size_t>(u) >= out_edge_epochs.size()) return;
        const auto& eps = out_edge_epochs[u];
        uint16_t ep = (edge_pos < eps.size()) ? eps[edge_pos]
                      : static_cast<uint16_t>(ecg_epoch_count - 1);
        SNIPER_ECG_EXTRACT(v, ep);
    };

    if (num_iters < 1) num_iters = 1;
    SNIPER_ROI_BEGIN();
    for (int iter = 0; iter < num_iters; iter++) {
        NodeID source = static_cast<NodeID>(iter % graph.num_nodes());
        for (NodeID n = 0; n < graph.num_nodes(); n++) {
            depth[n] = -1; path_counts[n] = 0; deltas[n] = 0;
        }
        depth[source] = 0;
        path_counts[source] = 1;

        // Forward BFS, single-threaded, recording per-level frontiers.
        std::vector<std::vector<NodeID>> levels;
        levels.push_back(std::vector<NodeID>{source});
        int cur_level = 0;
        while (!levels[cur_level].empty()) {
            std::vector<NodeID> next_level;
            for (NodeID u : levels[cur_level]) {
                SNIPER_SET_VERTEX(u);
                size_t edge_pos = 0;
                for (NodeID v : graph.out_neigh(u)) {
                    deliver(u, edge_pos, v);
                    ++edge_pos;
                    if (depth[v] == -1) {
                        depth[v] = cur_level + 1;
                        next_level.push_back(v);
                    }
                    if (depth[v] == cur_level + 1)
                        path_counts[v] += path_counts[u];
                }
            }
            if (next_level.empty()) break;
            levels.push_back(std::move(next_level));
            cur_level++;
        }

        // Backward dependency accumulation, deepest level first.
        for (int d = static_cast<int>(levels.size()) - 1; d > 0; d--) {
            for (NodeID w : levels[d]) {
                ScoreT delta_w = 0;
                for (NodeID v : graph.out_neigh(w)) {
                    if (depth[v] == depth[w] + 1)
                        delta_w += static_cast<ScoreT>(path_counts[w]) /
                                   path_counts[v] * (1.0f + deltas[v]);
                }
                deltas[w] = delta_w;
                if (w != source) scores[w] += delta_w;
            }
        }
    }
    SNIPER_ROI_END();

    double checksum = 0;
    for (ScoreT s : scores) checksum += s;
    std::cout << "GraphBrew Sniper SG BC checksum: " << checksum << std::endl;
    return graph.num_nodes() > 0 ? 0 : 1;
}

// Connected Components (Afforest) — single-threaded port of the audited
// Afforest_Sniper (bench/src_sniper/cc.cc): one grasp-protected comp[] region,
// transpose P-OPT reref matrix (CC reads comp[dest] over OUT-edges -> traverseCSR=
// false), and per-edge ECG epoch delivery. CC is the documented DO-NO-HARM cell
// (low property reuse, ECG ~= GRASP), certified for policy-compliance not a win.
int run_cc(const Graph& graph, int neighbor_rounds) {
    if (neighbor_rounds < 1) neighbor_rounds = 2;
    pvector<NodeID> comp(graph.num_nodes());
    for (NodeID n = 0; n < graph.num_nodes(); n++) comp[n] = n;

    SniperPropertyRegion regions[1] = {
        {"comp", reinterpret_cast<uint64_t>(comp.data()),
         static_cast<uint64_t>(graph.num_nodes()) * sizeof(NodeID),
         static_cast<uint32_t>(graph.num_nodes()), sizeof(NodeID), true},
    };
    SniperEdgeRegion edge_regions[2];
    int num_edge_regions = sniper_make_edge_regions(graph, edge_regions, 2, true);
    sniper_export_context(regions, 1, graph, nullptr, edge_regions, num_edge_regions);

    constexpr int kNumVtxPerLine = 64 / sizeof(NodeID);
    constexpr int kNumEpochs = 256;
    pvector<uint8_t> popt_matrix;
    int popt_num_cache_lines = (graph.num_nodes() + kNumVtxPerLine - 1) / kNumVtxPerLine;
    makeOffsetMatrix(graph, popt_matrix, kNumVtxPerLine, kNumEpochs, /*traverseCSR=*/false);
    sniper_export_popt_matrix(popt_matrix.data(), popt_num_cache_lines,
                              kNumEpochs, graph.num_nodes());

    const bool ecg_extract_on = graphbrew_sniper::ecg_extract_enabled();
    const uint32_t ecg_epoch_count = static_cast<uint32_t>(
        graphbrew_sniper::env_int_clamped("ECG_EDGE_MASK_EPOCHS", kNumEpochs, 2, 65535));
    std::vector<std::vector<uint16_t>> out_edge_epochs;
    if (ecg_extract_on) {
        ecg_epoch::buildInEdgeEpochs(graph, kNumVtxPerLine, ecg_epoch_count,
                                     /*linemin=*/true, out_edge_epochs,
                                     /*push_out_edges=*/true);
    }
    auto deliver = [&](NodeID u, size_t edge_pos, NodeID v) {
        if (!ecg_extract_on || static_cast<size_t>(u) >= out_edge_epochs.size()) return;
        const auto& eps = out_edge_epochs[u];
        uint16_t ep = (edge_pos < eps.size()) ? eps[edge_pos]
                      : static_cast<uint16_t>(ecg_epoch_count - 1);
        SNIPER_ECG_EXTRACT(v, ep);
    };

    SNIPER_ROI_BEGIN();
    // Phase 1: sample the r-th out-neighbour of each vertex, compress.
    for (int r = 0; r < neighbor_rounds; r++) {
        for (NodeID u = 0; u < graph.num_nodes(); u++) {
            SNIPER_SET_VERTEX(u);
            auto out_neigh = graph.out_neigh(u);
            auto it = out_neigh.begin();
            for (int i = 0; i < r && it != out_neigh.end(); ++i, ++it) {}
            if (it != out_neigh.end()) {
                deliver(u, static_cast<size_t>(r), *it);
                cc_link(u, *it, comp);
            }
        }
        cc_compress(graph, comp);
    }

    // Most frequent component = the giant component skipped in phase 2.
    std::unordered_map<NodeID, int64_t> count;
    for (NodeID n = 0; n < graph.num_nodes(); n++) count[comp[n]]++;
    NodeID largest = graph.num_nodes() > 0 ? comp[0] : 0;
    int64_t largest_count = -1;
    for (const auto& kv : count) {
        if (kv.second > largest_count) { largest_count = kv.second; largest = kv.first; }
    }

    // Phase 2: full traversal for vertices outside the giant component.
    for (NodeID u = 0; u < graph.num_nodes(); u++) {
        if (comp[u] == largest) continue;
        SNIPER_SET_VERTEX(u);
        size_t edge_pos = 0;
        for (NodeID v : graph.out_neigh(u)) {
            deliver(u, edge_pos, v);
            ++edge_pos;
            cc_link(u, v, comp);
        }
    }
    cc_compress(graph, comp);
    SNIPER_ROI_END();

    int64_t num_comps = 0;
    for (NodeID n = 0; n < graph.num_nodes(); n++)
        if (comp[n] == n) num_comps++;
    std::cout << "GraphBrew Sniper SG CC components: "
              << static_cast<long long>(num_comps) << std::endl;
    return graph.num_nodes() > 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Options options = parse_options(argc, argv);
    if (options.graph_path.empty()) {
        std::cerr << "sg_kernel requires -f graph.sg" << std::endl;
        return 2;
    }

    if (options.benchmark == "pr") {
        Graph graph = load_graph(options);
        return run_pr(graph, options.max_iters);
    }
    if (options.benchmark == "bfs") {
        Graph graph = load_graph(options);
        return run_bfs(graph, options.source);
    }
    if (options.benchmark == "sssp") {
        WGraph graph = load_weighted_graph(options);
        return run_sssp(graph, options.source, options.delta);
    }

    if (options.benchmark == "bc") {
        Graph graph = load_graph(options);
        return run_bc(graph, options.max_iters);
    }
    if (options.benchmark == "cc") {
        Graph graph = load_graph(options);
        return run_cc(graph, /*neighbor_rounds=*/2);
    }

    std::cerr << "unsupported sg_kernel benchmark: " << options.benchmark << std::endl;
    return 2;
}
