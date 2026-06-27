// ============================================================================
// Connected Components (Afforest) for gem5 SE-mode simulation
// ============================================================================
// Single-threaded Afforest (subgraph sampling) for gem5 SE mode.
// ============================================================================

#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
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

void Link(NodeID u, NodeID v, pvector<NodeID>& comp) {
    NodeID p1 = comp[u], p2 = comp[v];
    while (p1 != p2) {
        NodeID high = p1 > p2 ? p1 : p2;
        NodeID low = p1 + (p2 - high);
        if (comp[high] == high) { comp[high] = low; break; }
        NodeID p_high = comp[high];
        p1 = comp[p_high];
        p2 = comp[low];
    }
}

void Compress(const Graph &g, pvector<NodeID>& comp) {
    for (NodeID n = 0; n < g.num_nodes(); n++)
        while (comp[n] != comp[comp[n]])
            comp[n] = comp[comp[n]];
}

pvector<NodeID> Afforest_Gem5(const Graph &g, int32_t neighbor_rounds = 2) {
    constexpr size_t kPropAlign = 4096;  // page-align hot property array (see pr.cc)
    pvector<NodeID> comp(g.num_nodes(), NodeID(0), kPropAlign);
    for (NodeID n = 0; n < g.num_nodes(); n++) comp[n] = n;

    gem5_report_region("comp", comp.data(), g.num_nodes(), sizeof(NodeID));

    Gem5PropertyRegion regions[1] = {
        {"comp", reinterpret_cast<uint64_t>(comp.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(NodeID),
         static_cast<uint32_t>(g.num_nodes()), sizeof(NodeID)},
    };
    Gem5EdgeRegion edge_regions[2];
    int num_edge_regions = gem5_make_edge_regions(g, edge_regions, 2);

    // Per-edge next-ref EPOCH budget (mirror gem5 bc.cc) keyed on comp (int32). cc traverses
    // OUT-edges reading comp[dest]; dest's comp is next-referenced by its IN-neighbours ->
    // push_out_edges=true.
    constexpr int kNumVtxPerLine = 64 / sizeof(NodeID);
    uint8_t edge_id_bits = 1;
    while ((1ULL << edge_id_bits) < static_cast<uint64_t>(g.num_nodes())) edge_id_bits++;
    uint32_t edge_epoch_count = 2;
    if (edge_id_bits < 32) {
        uint32_t spare = 32u - edge_id_bits;
        uint32_t ne_cap = (spare >= 16) ? 65535u : (1u << spare);
        edge_epoch_count = std::min<uint32_t>(65535u, std::max<uint32_t>(2u, ne_cap));
    }
    gem5_export_context(regions, 1, g, GEM5_SIDEBAND_PATH,
                        edge_regions, num_edge_regions, edge_epoch_count);

    // A5: deliver comp[dest]'s next-ref epoch for ECG_GRASP_POPT via the fused ecg.load EVICT
    // (RISC-V); gated on GEM5_ENABLE_ECG_PLOAD. X86 falls back to a plain indexed load. The
    // ecg.load warms+stamps comp[v] before Link re-reads it (comp[] is the irregular per-neighbour
    // property; the union-find pointer-chasing reads stay plain).
    const bool ecg_extract_on = gem5_ecg_extract_enabled();
    std::vector<std::vector<uint16_t>> out_edge_epochs;
    if (ecg_extract_on) {
        ecg_epoch::buildInEdgeEpochs(g, static_cast<uint32_t>(kNumVtxPerLine),
                                     edge_epoch_count, /*linemin=*/true,
                                     out_edge_epochs, /*push_out_edges=*/true);
    }
    const bool ecg_load_evict_on = gem5_ecg_pload_enabled() && ecg_extract_on;
    const int  ecg_evict_wc = ecg_mode6::ecgEvictWidthClass(g.num_nodes());
    auto deliver_comp = [&](NodeID u, size_t edge_pos, NodeID v) {
        if (!ecg_load_evict_on || static_cast<size_t>(u) >= out_edge_epochs.size()) return;
        const auto& eps = out_edge_epochs[u];
        uint16_t epoch = (edge_pos < eps.size()) ? eps[edge_pos]
            : static_cast<uint16_t>(edge_epoch_count - 1);
        (void)gem5_ecg_load_evict(comp.data(),
                                  ecg_mode6::packEvict(static_cast<uint32_t>(v), epoch, ecg_evict_wc),
                                  ecg_evict_wc);
    };
    if (ecg_load_evict_on)
        fprintf(stderr, "[ECG_PLOAD] CC fused ecg.load EVICT delivery (comp) ACTIVE\n");

    GEM5_RESET_STATS();
    GEM5_WORK_BEGIN(GEM5_WORK_COMPUTE);

    // Phase 1: sparse sampling
    for (int32_t r = 0; r < neighbor_rounds; r++) {
        for (NodeID u = 0; u < g.num_nodes(); u++) {
            GEM5_SET_VERTEX(u);
            auto it = g.out_neigh(u).begin();
            for (int32_t i = 0; i < r && it != g.out_neigh(u).end(); ++i, ++it) {}
            if (it != g.out_neigh(u).end()) {
                deliver_comp(u, static_cast<size_t>(r), *it);
                Link(u, *it, comp);
            }
        }
        Compress(g, comp);
    }

    // Find largest component
    unordered_map<NodeID, int64_t> count;
    for (NodeID n = 0; n < g.num_nodes(); n++) count[comp[n]]++;
    NodeID largest = max_element(count.begin(), count.end(),
        [](auto &a, auto &b){ return a.second < b.second; })->first;

    // Phase 2: full edge traversal skipping largest
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        GEM5_SET_VERTEX(u);
        if (comp[u] == largest) continue;
        size_t edge_pos = 0;
        for (NodeID v : g.out_neigh(u)) {
            deliver_comp(u, edge_pos, v);
            ++edge_pos;
            Link(u, v, comp);
        }
    }
    Compress(g, comp);

    GEM5_WORK_END(GEM5_WORK_COMPUTE);
    GEM5_DUMP_STATS();
    return comp;
}

void PrintCompStats(const Graph &g, const pvector<NodeID> &comp) {
    unordered_map<NodeID, int64_t> count;
    for (NodeID n = 0; n < g.num_nodes(); n++) count[comp[n]]++;
    cout << "Components: " << count.size() << endl;
}

bool CCVerifier(const Graph &g, const pvector<NodeID> &comp) {
    // Check: all connected vertices have same component
    for (NodeID u = 0; u < g.num_nodes(); u++)
        for (NodeID v : g.out_neigh(u))
            if (comp[u] != comp[v]) return false;
    return true;
}

int main(int argc, char *argv[]) {
    CLApp cli(argc, argv, "cc-gem5");
    if (!cli.ParseArgs()) return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();

    auto CCBound = [](const Graph &g) { return Afforest_Gem5(g); };
    auto PrintBound = [](const Graph &g, const pvector<NodeID> &c) { PrintCompStats(g, c); };
    auto VerifyBound = [](const Graph &g, const pvector<NodeID> &c) { return CCVerifier(g, c); };
    BenchmarkKernel(cli, g, CCBound, PrintBound, VerifyBound);
    return 0;
}
