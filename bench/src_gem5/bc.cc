// ============================================================================
// Betweenness Centrality (Brandes) for gem5 SE-mode simulation
// ============================================================================
// Single-threaded Brandes BC for gem5. BFS forward + backward accumulation.
// ============================================================================

#include <iostream>
#include <queue>
#include <stack>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

#include "gem5_sim/gem5_harness.h"

using namespace std;

typedef float ScoreT;

pvector<ScoreT> Brandes_Gem5(const Graph &g, int num_iters) {
    pvector<ScoreT> scores(g.num_nodes(), 0);
    pvector<int32_t> depth(g.num_nodes());
    pvector<int64_t> path_counts(g.num_nodes());
    pvector<ScoreT> deltas(g.num_nodes());

    gem5_report_region("scores", scores.data(), g.num_nodes(), sizeof(ScoreT));
    gem5_report_region("depth", depth.data(), g.num_nodes(), sizeof(int32_t));
    gem5_report_region("path_counts", path_counts.data(), g.num_nodes(), sizeof(int64_t));

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
            order.push(u);
            for (NodeID v : g.out_neigh(u)) {
                if (depth[v] == -1) {
                    depth[v] = depth[u] + 1;
                    q.push(v);
                }
                if (depth[v] == depth[u] + 1)
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
