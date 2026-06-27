// ============================================================================
// Betweenness Centrality (Brandes) for gem5 SE-mode simulation
// ============================================================================
// Single-threaded Brandes BC for gem5. BFS forward + backward accumulation.
// ============================================================================

#include <cstring>
#include <iostream>
#include <queue>
#include <stack>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

#include "ecg_epoch_builder.h"
#include "ecg_mode6_builder.h"

#include "gem5_sim/gem5_harness.h"

using namespace std;

typedef float ScoreT;

pvector<ScoreT> Brandes_Gem5(const Graph &g, int num_iters) {
    constexpr size_t kPropAlign = 4096;  // page-align hot property arrays (see pr.cc)
    pvector<ScoreT> scores(g.num_nodes(), ScoreT(0), kPropAlign);
    pvector<int32_t> depth(g.num_nodes(), int32_t(0), kPropAlign);
    pvector<int64_t> path_counts(g.num_nodes(), int64_t(0), kPropAlign);
    pvector<ScoreT> deltas(g.num_nodes(), ScoreT(0), kPropAlign);

    gem5_report_region("scores", scores.data(), g.num_nodes(), sizeof(ScoreT));
    gem5_report_region("depth", depth.data(), g.num_nodes(), sizeof(int32_t));
    gem5_report_region("path_counts", path_counts.data(), g.num_nodes(), sizeof(int64_t));
        gem5_report_region("deltas", deltas.data(), g.num_nodes(), sizeof(ScoreT));

        // GRASP HPCA20 protects vertex-indexed property arrays. BC has four
        // such arrays (all indexed by vertex id), so we mark all of them as
        // grasp_region=true. classifyGRASP() applies the same hot/moderate
        // boundary per region; marking only one of four arrays (the original
        // behaviour) caused the other three to thrash under SRRIP. Mirror of
        // the cache_sim fix in bench/src_sim/bc.cc — see
        // wiki/Baseline-Literature-Faithfulness.md "BC multi-property fix".
        Gem5PropertyRegion regions[4] = {
        {"scores", reinterpret_cast<uint64_t>(scores.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(ScoreT),
            static_cast<uint32_t>(g.num_nodes()), sizeof(ScoreT), true},
        {"depth", reinterpret_cast<uint64_t>(depth.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(int32_t),
            static_cast<uint32_t>(g.num_nodes()), sizeof(int32_t), true},
        {"path_counts", reinterpret_cast<uint64_t>(path_counts.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(int64_t),
            static_cast<uint32_t>(g.num_nodes()), sizeof(int64_t), true},
           {"deltas", reinterpret_cast<uint64_t>(deltas.data()),
            static_cast<uint64_t>(g.num_nodes()) * sizeof(ScoreT),
            static_cast<uint32_t>(g.num_nodes()), sizeof(ScoreT), true},
    };
    Gem5EdgeRegion edge_regions[2];
    int num_edge_regions = gem5_make_edge_regions(g, edge_regions, 2);

    // Per-edge next-ref EPOCH budget (mirror gem5 bfs.cc) keyed on depth (int32). BC pushes
    // along OUT-edges reading depth[dest]; dest's property is next-referenced by its
    // IN-neighbours -> push_out_edges=true.
    constexpr int kNumVtxPerLine = 64 / sizeof(int32_t);
    uint8_t edge_id_bits = 1;
    while ((1ULL << edge_id_bits) < static_cast<uint64_t>(g.num_nodes())) edge_id_bits++;
    uint32_t edge_epoch_count = 2;
    if (edge_id_bits < 32) {
        uint32_t spare = 32u - edge_id_bits;
        uint32_t ne_cap = (spare >= 16) ? 65535u : (1u << spare);
        edge_epoch_count = std::min<uint32_t>(65535u, std::max<uint32_t>(2u, ne_cap));
    }
    gem5_export_context(regions, 4, g, GEM5_SIDEBAND_PATH,
                        edge_regions, num_edge_regions, edge_epoch_count);

    // A5: deliver depth[dest]'s next-ref epoch for ECG_GRASP_POPT via the fused ecg.load EVICT
    // (RISC-V); gated on GEM5_ENABLE_ECG_PLOAD. X86 falls back to a plain indexed load (no
    // delivery -> cache_sim authoritative). depth is the irregular property read in the
    // forward BFS; the other BC arrays are read sequentially or are 8-byte (path_counts).
    const bool ecg_extract_on = gem5_ecg_extract_enabled();
    std::vector<std::vector<uint16_t>> out_edge_epochs;
    if (ecg_extract_on) {
        ecg_epoch::buildInEdgeEpochs(g, static_cast<uint32_t>(kNumVtxPerLine),
                                     edge_epoch_count, /*linemin=*/true,
                                     out_edge_epochs, /*push_out_edges=*/true);
    }
    const bool ecg_load_evict_on = gem5_ecg_pload_enabled() && ecg_extract_on;
    const int  ecg_evict_wc = ecg_mode6::ecgEvictWidthClass(g.num_nodes());
    if (ecg_load_evict_on)
        fprintf(stderr, "[ECG_PLOAD] BC fused ecg.load EVICT delivery (depth) ACTIVE\n");

    GEM5_RESET_STATS();
    GEM5_WORK_BEGIN(GEM5_WORK_COMPUTE);

    // Pick sources round-robin
    for (int iter = 0; iter < num_iters; iter++) {
        NodeID source = iter % g.num_nodes();

        // Reset
        for (NodeID n = 0; n < g.num_nodes(); n++) {
            depth[n] = -1;
            path_counts[n] = 0;
            deltas[n] = 0;
        }
        depth[source] = 0;
        path_counts[source] = 1;

        // Forward BFS
        stack<NodeID> order;
        queue<NodeID> q;
        q.push(source);
        while (!q.empty()) {
            NodeID u = q.front(); q.pop();
            GEM5_SET_VERTEX(u);
            order.push(u);
            const std::vector<uint16_t>* u_epochs =
                (ecg_load_evict_on && static_cast<size_t>(u) < out_edge_epochs.size())
                    ? &out_edge_epochs[u] : nullptr;
            size_t edge_pos = 0;
            for (NodeID v : g.out_neigh(u)) {
                // Read depth[v]; on the ecg.load EVICT path the load also stamps depth[v]'s
                // line with v's epoch (push_out_edges=true transpose), so ECG_GRASP_POPT ranks
                // depth[] by next-reference.
                int32_t dv;
                if (u_epochs) {
                    uint16_t epoch = (edge_pos < u_epochs->size())
                        ? (*u_epochs)[edge_pos]
                        : static_cast<uint16_t>(edge_epoch_count - 1);
                    uint64_t fat = ecg_mode6::packEvict(static_cast<uint32_t>(v),
                                                        epoch, ecg_evict_wc);
                    uint32_t bits = gem5_ecg_load_evict(depth.data(), fat, ecg_evict_wc);
                    std::memcpy(&dv, &bits, sizeof(int32_t));
                } else {
                    dv = depth[v];
                }
                ++edge_pos;
                if (dv == -1) {
                    depth[v] = depth[u] + 1;
                    q.push(v);
                    dv = depth[u] + 1;
                }
                if (dv == depth[u] + 1)
                    path_counts[v] += path_counts[u];
            }
        }

        // Backward accumulation
        while (!order.empty()) {
            NodeID w = order.top(); order.pop();
            for (NodeID v : g.out_neigh(w)) {
                if (depth[v] == depth[w] + 1) {
                    deltas[w] += (ScoreT)path_counts[w] / path_counts[v]
                                 * (1.0f + deltas[v]);
                }
            }
            if (w != source)
                scores[w] += deltas[w];
        }
    }

    GEM5_WORK_END(GEM5_WORK_COMPUTE);
    GEM5_DUMP_STATS();
    return scores;
}

void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
    vector<pair<NodeID, ScoreT>> sp(g.num_nodes());
    for (NodeID n = 0; n < g.num_nodes(); n++) sp[n] = {n, scores[n]};
    int k = min(5, (int)g.num_nodes());
    partial_sort(sp.begin(), sp.begin() + k, sp.end(),
                [](auto &a, auto &b) { return a.second > b.second; });
    for (int i = 0; i < k; i++)
        cout << sp[i].first << ": " << sp[i].second << endl;
}

bool BCVerifier(const Graph &g, const pvector<ScoreT> &scores, int num_iters) {
    // Accept if scores are non-negative
    for (NodeID n = 0; n < g.num_nodes(); n++)
        if (scores[n] < 0) return false;
    return true;
}

int main(int argc, char *argv[]) {
    CLIterApp cli(argc, argv, "bc-gem5", 1);
    if (!cli.ParseArgs()) return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();

    auto BCBound = [&cli](const Graph &g) {
        return Brandes_Gem5(g, cli.num_iters());
    };
    auto PrintBound = [](const Graph &g, const pvector<ScoreT> &s) {
        PrintTopScores(g, s);
    };
    auto VerifyBound = [&cli](const Graph &g, const pvector<ScoreT> &s) {
        return BCVerifier(g, s, cli.num_iters());
    };
    BenchmarkKernel(cli, g, BCBound, PrintBound, VerifyBound);
    return 0;
}
