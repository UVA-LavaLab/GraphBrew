// ============================================================================
// SSSP delta-stepping for gem5 SE-mode simulation
// ============================================================================

#include <cinttypes>
#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
#include <omp.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"

#include "graphbrew/partition/cagra/popt.h"

#include "gem5_sim/gem5_harness.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const size_t kMaxBin = numeric_limits<size_t>::max() / 2;
const size_t kBinSizeThreshold = 1000;

inline void RelaxEdges_Gem5(const WGraph &g, NodeID u, WeightT delta,
                            pvector<WeightT> &dist,
                            vector<vector<NodeID>> &local_bins) {
    GEM5_SET_VERTEX(u);
    for (WNode wn : g.out_neigh(u)) {
        WeightT old_dist = dist[wn.v];
        WeightT new_dist = dist[u] + wn.w;
        while (new_dist < old_dist) {
            if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
                size_t dest_bin = new_dist / delta;
                if (dest_bin >= local_bins.size())
                    local_bins.resize(dest_bin + 1);
                local_bins[dest_bin].push_back(wn.v);
                break;
            }
            old_dist = dist[wn.v];
        }
    }
}

pvector<WeightT> DeltaStep_Gem5(const WGraph &g, NodeID source, WeightT delta) {
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    dist[source] = 0;

    gem5_report_region("dist", dist.data(), g.num_nodes(), sizeof(WeightT));

    Gem5PropertyRegion regions[1] = {
        {"dist", reinterpret_cast<uint64_t>(dist.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(WeightT),
         static_cast<uint32_t>(g.num_nodes()), sizeof(WeightT)},
    };
    Gem5EdgeRegion edge_regions[2];
    int num_edge_regions = gem5_make_edge_regions(g, edge_regions, 2);
    gem5_export_context(regions, 1, g, GEM5_SIDEBAND_PATH,
                        edge_regions, num_edge_regions);

    {
        constexpr int numVtxPerLine = 64 / sizeof(WeightT);
        constexpr int numEpochs = 256;
        static pvector<uint8_t> popt_matrix;
        makeOffsetMatrix(g, popt_matrix, numVtxPerLine, numEpochs);
        int numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
        gem5_export_popt_matrix(popt_matrix.data(), numCacheLines,
                                numEpochs, g.num_nodes());
    }

    GEM5_RESET_STATS();
    GEM5_WORK_BEGIN(GEM5_WORK_COMPUTE);

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
                if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index))
                    RelaxEdges_Gem5(g, u, delta, dist, local_bins);
            }

            while (curr_bin_index < local_bins.size() &&
                   !local_bins[curr_bin_index].empty() &&
                   local_bins[curr_bin_index].size() < kBinSizeThreshold) {
                vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
                local_bins[curr_bin_index].resize(0);
                for (NodeID u : curr_bin_copy)
                    RelaxEdges_Gem5(g, u, delta, dist, local_bins);
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
                copy(local_bins[next_bin_index].begin(),
                     local_bins[next_bin_index].end(),
                     frontier.data() + copy_start);
                local_bins[next_bin_index].resize(0);
            }

            iter++;
            #pragma omp barrier
        }
    }

    GEM5_WORK_END(GEM5_WORK_COMPUTE);
    GEM5_DUMP_STATS();
    return dist;
}

void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
    int reachable = 0;
    for (NodeID n = 0; n < g.num_nodes(); n++)
        if (dist[n] < kDistInf) reachable++;
    cout << "Reachable: " << reachable << "/" << static_cast<int>(g.num_nodes()) << endl;
}

bool SSSPVerifier(const WGraph &g, NodeID source, const pvector<WeightT> &dist) {
    pvector<WeightT> oracle(g.num_nodes(), kDistInf);
    oracle[source] = 0;
    typedef pair<WeightT, NodeID> WN;
    priority_queue<WN, vector<WN>, greater<WN>> pq;
    pq.push({0, source});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > oracle[u]) continue;
        for (WNode wn : g.out_neigh(u)) {
            WeightT nd = oracle[u] + wn.w;
            if (nd < oracle[wn.v]) { oracle[wn.v] = nd; pq.push({nd, wn.v}); }
        }
    }
    for (NodeID n = 0; n < g.num_nodes(); n++)
        if (dist[n] != oracle[n]) return false;
    return true;
}

int main(int argc, char *argv[]) {
    CLDelta<WeightT> cli(argc, argv, "sssp-gem5");
    if (!cli.ParseArgs()) return -1;
    omp_set_num_threads(1);
    {
        WeightedBuilder b(cli);
        WGraph g = b.MakeGraph();
        SourcePicker<WGraph> sp(g, cli.start_vertex());

        auto SSSPBound = [&sp, &cli](const WGraph &g) {
            return DeltaStep_Gem5(g, sp.PickNext(), cli.delta());
        };
        auto PrintBound = [](const WGraph &g, const pvector<WeightT> &d) {
            PrintSSSPStats(g, d);
        };
        auto VerifyBound = [&sp](const WGraph &g, const pvector<WeightT> &d) {
            return SSSPVerifier(g, sp.PickNext(), d);
        };
        BenchmarkKernel(cli, g, SSSPBound, PrintBound, VerifyBound);
    }
    return 0;
}
