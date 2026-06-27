// ============================================================================
// SSSP delta-stepping for Sniper simulation
// ============================================================================
// Mirrors the audited gem5 SSSP wrapper while using Sniper ROI markers and
// sideband export helpers.
// ============================================================================

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"

#include "graphbrew/partition/cagra/popt.h"
#include "sniper_sim/sniper_harness.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const size_t kMaxBin = numeric_limits<size_t>::max() / 2;
const size_t kBinSizeThreshold = 1000;

inline void RelaxEdges_Sniper(const WGraph &g, NodeID u, WeightT delta,
                              pvector<WeightT> &dist,
                              vector<vector<NodeID>> &local_bins) {
    SNIPER_SET_VERTEX(u);
    int pfx_lookahead = graphbrew_sniper::env_int_clamped(
        "SNIPER_ECG_PFX_LOOKAHEAD",
        graphbrew_sniper::env_int_clamped("ECG_PREFETCH_LOOKAHEAD", 4, 0, 64),
        0, 64);
    auto out_neigh = g.out_neigh(u);
    for (auto it = out_neigh.begin(); it != out_neigh.end(); ++it) {
        WNode wn = *it;
        if (pfx_lookahead > 0) {
            auto jt = it;
            for (int step = 0; step < pfx_lookahead; step++) {
                ++jt;
                if (jt == out_neigh.end()) break;
                SNIPER_ECG_PFX_TARGET((*jt).v);
                break;
            }
        } else {
            SNIPER_ECG_PFX_TARGET(wn.v);
        }
        WeightT old_dist = dist[wn.v];
        WeightT new_dist = dist[u] + wn.w;
        while (new_dist < old_dist) {
            if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
                size_t dest_bin = new_dist / delta;
                if (dest_bin >= local_bins.size()) {
                    local_bins.resize(dest_bin + 1);
                }
                local_bins[dest_bin].push_back(wn.v);
                break;
            }
            old_dist = dist[wn.v];
        }
    }
}

pvector<WeightT> DeltaStep_Sniper(const WGraph &g, NodeID source, WeightT delta) {
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    dist[source] = 0;

    sniper_report_region("dist", dist.data(), g.num_nodes(), sizeof(WeightT));

    SniperPropertyRegion regions[1] = {
        {"dist", reinterpret_cast<uint64_t>(dist.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(WeightT),
         static_cast<uint32_t>(g.num_nodes()), sizeof(WeightT)},
    };
    SniperEdgeRegion edge_regions[2];
    int num_edge_regions = sniper_make_edge_regions(g, edge_regions, 2);
    sniper_export_context(regions, 1, g, nullptr, edge_regions, num_edge_regions);

    {
        constexpr int numVtxPerLine = 64 / sizeof(WeightT);
        constexpr int numEpochs = 256;
        static pvector<uint8_t> popt_matrix;
        // SSSP relaxes OUT-edges reading dist[dest]; the next-ref of dist[v] is over v's
        // IN-neighbours, so the reref matrix is the graph TRANSPOSE (CSC/in_neigh,
        // traverseCSR=false) — matching cache_sim's natural_csr=false. Default true=out_neigh
        // is only correct for PR's in-pull. Undirected forces true internally (do-no-harm).
        makeOffsetMatrix(g, popt_matrix, numVtxPerLine, numEpochs, /*traverseCSR=*/false);
        int numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
        sniper_export_popt_matrix(popt_matrix.data(), numCacheLines,
                                  numEpochs, g.num_nodes());
    }

    SNIPER_ROI_BEGIN();

    pvector<NodeID> frontier(g.num_edges_directed());
    size_t shared_indexes[2] = {0, kMaxBin};
    size_t frontier_tails[2] = {1, 0};
    frontier[0] = source;

    #pragma omp parallel
    {
        vector<vector<NodeID>> local_bins(0);
        size_t iter = 0;
        while (shared_indexes[iter & 1] != kMaxBin) {
            size_t &curr_bin_index = shared_indexes[iter & 1];
            size_t &next_bin_index = shared_indexes[(iter + 1) & 1];
            size_t &curr_frontier_tail = frontier_tails[iter & 1];
            size_t &next_frontier_tail = frontier_tails[(iter + 1) & 1];

            #pragma omp for nowait schedule(dynamic, 64)
            for (size_t i = 0; i < curr_frontier_tail; i++) {
                NodeID u = frontier[i];
                if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index)) {
                    RelaxEdges_Sniper(g, u, delta, dist, local_bins);
                }
            }

            while (curr_bin_index < local_bins.size() &&
                   !local_bins[curr_bin_index].empty() &&
                   local_bins[curr_bin_index].size() < kBinSizeThreshold) {
                vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
                local_bins[curr_bin_index].resize(0);
                for (NodeID u : curr_bin_copy) {
                    RelaxEdges_Sniper(g, u, delta, dist, local_bins);
                }
            }

            for (size_t i = curr_bin_index; i < local_bins.size(); i++) {
                if (!local_bins[i].empty()) {
                    #pragma omp critical
                    next_bin_index = min(next_bin_index, i);
                    break;
                }
            }

            #pragma omp barrier
            #pragma omp single nowait
            {
                curr_bin_index = kMaxBin;
                curr_frontier_tail = 0;
            }

            if (next_bin_index < local_bins.size()) {
                size_t copy_start = fetch_and_add(next_frontier_tail,
                                                  local_bins[next_bin_index].size());
                copy(local_bins[next_bin_index].begin(), local_bins[next_bin_index].end(),
                     frontier.data() + copy_start);
                local_bins[next_bin_index].resize(0);
            }

            iter++;
            #pragma omp barrier
        }
    }

    SNIPER_ROI_END();
    return dist;
}

void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
    int reachable = 0;
    for (NodeID n = 0; n < g.num_nodes(); n++) {
        if (dist[n] < kDistInf) reachable++;
    }
    cout << "Reachable: " << reachable << "/" << static_cast<int>(g.num_nodes()) << endl;
}

bool SSSPVerifier(const WGraph &g, NodeID source, const pvector<WeightT> &dist) {
    pvector<WeightT> oracle(g.num_nodes(), kDistInf);
    oracle[source] = 0;
    typedef pair<WeightT, NodeID> WN;
    priority_queue<WN, vector<WN>, greater<WN>> pq;
    pq.push({0, source});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > oracle[u]) continue;
        for (WNode wn : g.out_neigh(u)) {
            WeightT nd = oracle[u] + wn.w;
            if (nd < oracle[wn.v]) {
                oracle[wn.v] = nd;
                pq.push({nd, wn.v});
            }
        }
    }
    for (NodeID n = 0; n < g.num_nodes(); n++) {
        if (dist[n] != oracle[n]) return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    CLDelta<WeightT> cli(argc, argv, "sssp-sniper");
    if (!cli.ParseArgs()) return -1;
    omp_set_num_threads(1);
    WeightedBuilder b(cli);
    WGraph g = b.MakeGraph();
    SourcePicker<WGraph> sp(g, cli.start_vertex());
    // Separate verifier picker seeded identically to sp (matches cache_sim/canonical) so
    // the kernel and verifier draw the SAME source — without it the single picker hands
    // the verifier a different source and verification FAILs unless -r is set.
    SourcePicker<WGraph> vsp(g, cli.start_vertex());

    auto SSSPBound = [&sp, &cli](const WGraph &g) {
        return DeltaStep_Sniper(g, sp.PickNext(), cli.delta());
    };
    auto PrintBound = [](const WGraph &g, const pvector<WeightT> &d) {
        PrintSSSPStats(g, d);
    };
    auto VerifyBound = [&vsp](const WGraph &g, const pvector<WeightT> &d) {
        return SSSPVerifier(g, vsp.PickNext(), d);
    };
    BenchmarkKernel(cli, g, SSSPBound, PrintBound, VerifyBound);
    return 0;
}
