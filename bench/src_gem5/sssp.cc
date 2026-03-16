// ============================================================================
// SSSP (Dijkstra single-threaded) for gem5 SE-mode simulation
// ============================================================================
// Single-threaded Dijkstra for gem5 SE mode (delta-stepping requires OpenMP).
// ============================================================================

#include <cinttypes>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"

#include "gem5_sim/gem5_harness.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;

pvector<WeightT> Dijkstra_Gem5(const WGraph &g, NodeID source) {
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    dist[source] = 0;

    gem5_report_region("dist", dist.data(), g.num_nodes(), sizeof(WeightT));

    Gem5PropertyRegion regions[1] = {
        {"dist", reinterpret_cast<uint64_t>(dist.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(WeightT),
         static_cast<uint32_t>(g.num_nodes()), sizeof(WeightT)},
    };
    gem5_export_context(regions, 1, g);

    GEM5_RESET_STATS();
    GEM5_WORK_BEGIN(GEM5_WORK_COMPUTE);

    typedef pair<WeightT, NodeID> WN;
    priority_queue<WN, vector<WN>, greater<WN>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (WNode wn : g.out_neigh(u)) {
            WeightT new_dist = dist[u] + wn.w;
            if (new_dist < dist[wn.v]) {
                dist[wn.v] = new_dist;
                pq.push({new_dist, wn.v});
            }
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
    {
        WeightedBuilder b(cli);
        WGraph g = b.MakeGraph();
        SourcePicker<WGraph> sp(g, cli.start_vertex());

        auto SSSPBound = [&sp](const WGraph &g) {
            return Dijkstra_Gem5(g, sp.PickNext());
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
