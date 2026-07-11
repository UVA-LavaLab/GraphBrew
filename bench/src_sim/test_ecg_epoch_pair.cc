#include "ecg_epoch_builder.h"

#include <cstdint>
#include <cstdio>
#include <vector>

struct TinyGraph {
    std::vector<std::vector<int>> out;
    std::vector<std::vector<int>> in;

    explicit TinyGraph(uint32_t n) : out(n), in(n) {}

    void addEdge(int u, int v) {
        out[u].push_back(v);
        in[v].push_back(u);
    }

    uint32_t num_nodes() const { return static_cast<uint32_t>(out.size()); }
    uint64_t num_edges_directed() const {
        uint64_t total = 0;
        for (const auto& row : out) total += row.size();
        return total;
    }
    const std::vector<int>& out_neigh(uint32_t v) const { return out[v]; }
    const std::vector<int>& in_neigh(uint32_t v) const { return in[v]; }
};

static bool checkDirection(const TinyGraph& graph, bool push)
{
    std::vector<std::vector<uint16_t>> single;
    std::vector<std::vector<ecg_epoch::EpochPair>> pairs;
    ecg_epoch::buildInEdgeEpochs(
        graph, 2, 32, true, single, push);
    ecg_epoch::buildInEdgeEpochPairs(
        graph, 2, 32, true, pairs, push);

    if (single.size() != pairs.size()) return false;
    for (size_t src = 0; src < single.size(); ++src) {
        if (single[src].size() != pairs[src].size()) return false;
        for (size_t edge = 0; edge < single[src].size(); ++edge) {
            if (!pairs[src][edge].valid) return false;
            if (pairs[src][edge].first != single[src][edge]) return false;
        }
    }
    if (push) {
        // Edge 1->2: property line 2's readers after src=1 are reader 5,
        // then reader 0 after wrap. NE=32,N=6 => epochs 26 then 0.
        if (pairs[1].empty() ||
            pairs[1][0].first != 26 ||
            pairs[1][0].second != 0)
            return false;
    }
    return true;
}

int main()
{
    TinyGraph graph(6);
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 1);
    graph.addEdge(2, 4);
    graph.addEdge(3, 1);
    graph.addEdge(3, 5);
    graph.addEdge(4, 1);
    graph.addEdge(5, 2);

    const bool pull_ok = checkDirection(graph, false);
    const bool push_ok = checkDirection(graph, true);
    std::printf("[test_ecg_epoch_pair] pull=%s push=%s\n",
                pull_ok ? "OK" : "FAIL",
                push_ok ? "OK" : "FAIL");
    return pull_ok && push_ok ? 0 : 1;
}
