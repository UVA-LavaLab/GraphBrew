// ============================================================================
// BFS (Top-Down only) for gem5 SE-mode simulation
// ============================================================================
// Simple queue-based BFS for single-threaded gem5 SE mode.
// Bottom-up phase requires parallel/atomics which gem5 SE doesn't support well.
// ============================================================================

#include <iostream>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

#include "gem5_sim/gem5_harness.h"

using namespace std;

pvector<NodeID> BFS_Gem5(const Graph &g, NodeID source) {
    pvector<NodeID> parent(g.num_nodes(), -1);
    parent[source] = source;

    gem5_report_region("parent", parent.data(), g.num_nodes(), sizeof(NodeID));

    GEM5_RESET_STATS();
    GEM5_WORK_BEGIN(GEM5_WORK_COMPUTE);

    queue<NodeID> frontier;
    frontier.push(source);
    while (!frontier.empty()) {
        NodeID u = frontier.front();
        frontier.pop();
        for (NodeID v : g.out_neigh(u)) {
            if (parent[v] == -1) {
                parent[v] = u;
                frontier.push(v);
            }
        }
    }

    GEM5_WORK_END(GEM5_WORK_COMPUTE);
    GEM5_DUMP_STATS();
    return parent;
}

void PrintBFSStats(const Graph &g, const pvector<NodeID> &parent) {
    int64_t visited = 0;
    for (NodeID n = 0; n < g.num_nodes(); n++)
        if (parent[n] >= 0) visited++;
    cout << "Visited: " << static_cast<int>(visited) << "/" << static_cast<int>(g.num_nodes()) << endl;
}

bool BFSVerifier(const Graph &g, NodeID source, const pvector<NodeID> &parent) {
    if (parent[source] != source) return false;
    pvector<int32_t> depth(g.num_nodes(), -1);
    depth[source] = 0;
    queue<NodeID> q;
    q.push(source);
    while (!q.empty()) {
        NodeID u = q.front(); q.pop();
        for (NodeID v : g.out_neigh(u)) {
            if (depth[v] == -1) {
                depth[v] = depth[u] + 1;
                q.push(v);
            }
        }
    }
    for (NodeID n = 0; n < g.num_nodes(); n++) {
        if (depth[n] >= 0 && parent[n] < 0) return false;
        if (depth[n] < 0 && parent[n] >= 0) return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    CLApp cli(argc, argv, "bfs-gem5");
    if (!cli.ParseArgs()) return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();
    SourcePicker<Graph> sp(g, cli.start_vertex());

    auto BFSBound = [&sp](const Graph &g) {
        return BFS_Gem5(g, sp.PickNext());
    };
    auto PrintBound = [](const Graph &g, const pvector<NodeID> &p) {
        PrintBFSStats(g, p);
    };
    auto VerifyBound = [&sp](const Graph &g, const pvector<NodeID> &p) {
        return BFSVerifier(g, sp.PickNext(), p);
    };
    BenchmarkKernel(cli, g, BFSBound, PrintBound, VerifyBound);
    return 0;
}
