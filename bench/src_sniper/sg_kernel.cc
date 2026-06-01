#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "benchmark.h"
#include "graph.h"
#include "pvector.h"
#include "reader.h"

#include "graphbrew/partition/cagra/popt.h"
#include "sniper_sim/sniper_harness.h"

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
        } else if ((arg == "-g" || arg == "-k" || arg == "-o" || arg == "-n" || arg == "-t") && has_value(i, argc)) {
            ++i;
        } else if (arg == "-s" || arg == "-a" || arg == "-v" || arg == "--") {
            continue;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: sg_kernel --benchmark pr|bfs|sssp -f graph.sg [-i iters] [-r source] [-d delta]\n";
            std::exit(0);
        }
    }
    return options;
}

Graph load_graph(const std::string& path) {
    Reader<NodeID> reader(path);
    return reader.ReadSerializedGraph();
}

WGraph load_weighted_graph(const std::string& path) {
    Reader<NodeID, WNode, WeightT> reader(path);
    return reader.ReadSerializedGraph();
}

template <typename GraphType, typename ValueT>
void export_popt_for_graph(const GraphType& graph) {
    constexpr int kNumEpochs = 256;
    const int num_vtx_per_line = std::max<int>(1, 64 / sizeof(ValueT));
    pvector<uint8_t> popt_matrix;
    makeOffsetMatrix(graph, popt_matrix, num_vtx_per_line, kNumEpochs);
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
    export_popt_for_graph<Graph, ScoreT>(graph);

    SNIPER_ROI_BEGIN();
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
    for (int iter = 0; iter < max_iters; ++iter) {
        for (NodeID node = 0; node < graph.num_nodes(); ++node) {
            SNIPER_SET_VERTEX(node);
            // ECG_PFX hint: emit a future vertex so the prefetcher can
            // warm its scores/contrib lines before we touch them. The
            // emission is env-gated by SNIPER_ENABLE_ECG_PFX_HINTS
            // inside set_prefetch_target, so non-ECG_PFX runs pay
            // nothing.
            NodeID pfx_target = node + pfx_lookahead;
            if (pfx_target < graph.num_nodes()) {
                SNIPER_ECG_PFX_TARGET(pfx_target);
            }
            ScoreT incoming_total = 0.0f;
            for (NodeID neighbor : graph.in_neigh(node)) {
                incoming_total += contrib[neighbor];
            }
            scores[node] = base_score + kDamp * incoming_total;
            int64_t degree = graph.out_degree(node);
            contrib[node] = degree > 0 ? scores[node] / degree : 0.0f;
        }
    }
    SNIPER_ROI_END();

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

    SNIPER_ROI_BEGIN();
    std::queue<NodeID> frontier;
    frontier.push(source);
    while (!frontier.empty()) {
        NodeID node = frontier.front();
        frontier.pop();
        SNIPER_SET_VERTEX(node);
        // ECG_PFX hint: emit the head of the frontier (the node we'll
        // expand next after this iteration completes) so the
        // prefetcher can warm parent[next_node] before we touch it.
        // Lookahead = current frontier depth, which on real BFS is
        // typically 10s of nodes — gives Sniper's PREFETCH_INTERVAL
        // ~250 cycles × queue-depth iterations of head-start.
        // Filtering inside set_prefetch_target via
        // SNIPER_ENABLE_ECG_PFX_HINTS so non-ECG_PFX runs pay nothing.
        if (!frontier.empty()) {
            SNIPER_ECG_PFX_TARGET(frontier.front());
        }
        for (NodeID neighbor : graph.out_neigh(node)) {
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

    SNIPER_ROI_BEGIN();
    std::queue<NodeID> frontier;
    frontier.push(source);
    in_queue[source] = 1;
    while (!frontier.empty()) {
        NodeID node = frontier.front();
        frontier.pop();
        in_queue[node] = 0;
        SNIPER_SET_VERTEX(node);
        // ECG_PFX hint: emit the head of the frontier (next node to
        // expand after this iteration) so the prefetcher can warm
        // its dist[next]/edge entries before we touch them.
        // Lookahead = current frontier depth — gives PREFETCH_INTERVAL
        // queue-depth iterations of head-start. Env-gated.
        if (!frontier.empty()) {
            SNIPER_ECG_PFX_TARGET(frontier.front());
        }
        for (WNode edge : graph.out_neigh(node)) {
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

}  // namespace

int main(int argc, char** argv) {
    Options options = parse_options(argc, argv);
    if (options.graph_path.empty()) {
        std::cerr << "sg_kernel requires -f graph.sg" << std::endl;
        return 2;
    }

    if (options.benchmark == "pr") {
        Graph graph = load_graph(options.graph_path);
        return run_pr(graph, options.max_iters);
    }
    if (options.benchmark == "bfs") {
        Graph graph = load_graph(options.graph_path);
        return run_bfs(graph, options.source);
    }
    if (options.benchmark == "sssp") {
        WGraph graph = load_weighted_graph(options.graph_path);
        return run_sssp(graph, options.source, options.delta);
    }

    std::cerr << "unsupported sg_kernel benchmark: " << options.benchmark << std::endl;
    return 2;
}
