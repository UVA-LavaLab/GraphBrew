// ============================================================================
// PageRank (Pull, Gauss-Seidel) for gem5 SE-mode simulation
// ============================================================================
// Single-threaded PageRank for gem5. No in-process cache simulation —
// gem5's memory subsystem tracks all accesses automatically.
// The GRASP/ECG replacement policies learn property regions online.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

// P-OPT rereference matrix builder (same as standalone cache_sim)
#include "graphbrew/partition/cagra/popt.h"

#include "gem5_sim/gem5_harness.h"

// ECG mode 6 (per-edge mask) builder — shared with cache_sim and Sniper.
#include "ecg_mode6_builder.h"

using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;

pvector<ScoreT> PageRankPullGS_Gem5(const Graph &g, int max_iters,
                                     double epsilon = 0) {
    const ScoreT init_score = 1.0f / g.num_nodes();
    const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
    pvector<ScoreT> scores(g.num_nodes(), init_score);
    pvector<ScoreT> outgoing_contrib(g.num_nodes());

    gem5_report_region("scores", scores.data(), g.num_nodes(), sizeof(ScoreT));
    gem5_report_region("contrib", outgoing_contrib.data(), g.num_nodes(), sizeof(ScoreT));

    // Export graph context sideband file for gem5 replacement policies.
    // This is the gem5 equivalent of cache_sim's registerPropertyArray() +
    // initTopology(). The SimObjects lazily load this on first eviction.
    Gem5PropertyRegion regions[2] = {
        {"scores",  reinterpret_cast<uint64_t>(scores.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(ScoreT),
            static_cast<uint32_t>(g.num_nodes()), sizeof(ScoreT), true},
        {"contrib", reinterpret_cast<uint64_t>(outgoing_contrib.data()),
         static_cast<uint64_t>(g.num_nodes()) * sizeof(ScoreT),
            static_cast<uint32_t>(g.num_nodes()), sizeof(ScoreT), true},
    };
    Gem5EdgeRegion edge_regions[2];
    int num_edge_regions = gem5_make_edge_regions(g, edge_regions, 2, true);
    gem5_export_context(regions, 2, g, GEM5_SIDEBAND_PATH,
                        edge_regions, num_edge_regions);

    // Build P-OPT rereference matrix (matching standalone src_sim/pr.cc)
    // Predicts future cache line accesses from graph structure.
    static pvector<uint8_t> popt_matrix;
    constexpr int kNumVtxPerLine = 64 / sizeof(ScoreT);  // 16 floats per line
    constexpr int kNumEpochs = 256;
    int popt_num_cache_lines = (g.num_nodes() + kNumVtxPerLine - 1) / kNumVtxPerLine;
    {
        makeOffsetMatrix(g, popt_matrix, kNumVtxPerLine, kNumEpochs);
        gem5_export_popt_matrix(popt_matrix.data(), popt_num_cache_lines,
                                kNumEpochs, g.num_nodes());
    }

    // === ECG Adaptive Prefetch ===
    // Compute bit layout dynamically from graph size (FatIDConfig logic).
    // For 32-bit edge IDs with N vertices needing B bits, we have (32-B)
    // spare bits for metadata. With 16K vertices (14 bits), we get 18 spare
    // bits → DBG=2, POPT=8, PFX=6 → 64-entry hot table.
    //
    // The hot table contains the top-K highest-degree vertices, sorted by
    // degree. Each neighbor access can prefetch one of these hub vertices
    // based on which hub is most likely to be needed next (and not already
    // in cache from a recent prefetch — dedup window prevents redundancy).
    
    // Compute spare bits (matching FatIDConfig::computeFromGraph)
    uint8_t id_bits = 1;
    while ((1ULL << id_bits) < (uint64_t)g.num_nodes()) id_bits++;
    uint8_t container_bits = (g.num_nodes() > (1LL << 30)) ? 64 : 32;
    uint8_t spare = container_bits - id_bits;
    if (spare < 2 && container_bits == 32) { container_bits = 64; spare = container_bits - id_bits; }

    // Compute prefetch bits (matching FatIDConfig allocation tiers)
    uint8_t pfx_bits = 0;
    if (spare >= 16)     pfx_bits = min((int)(spare - 10), 6);
    else if (spare >= 10) pfx_bits = min((int)(spare - 6), 4);
    else if (spare >= 6)  pfx_bits = spare - 4;
    // else: pfx_bits = 0
    
    int hot_table_size = pfx_bits > 0 ? (1 << pfx_bits) : 0;
    hot_table_size = min(hot_table_size, (int)g.num_nodes());
    constexpr int PREFETCH_WINDOW = 16;  // Dedup window

    vector<NodeID> hot_table(hot_table_size);
    if (hot_table_size > 0) {
        // Build hot table: top-K by degree
        vector<pair<int64_t, NodeID>> deg_vtx(g.num_nodes());
        for (NodeID n = 0; n < g.num_nodes(); n++)
            deg_vtx[n] = {g.out_degree(n), n};
        partial_sort(deg_vtx.begin(),
                     deg_vtx.begin() + hot_table_size,
                     deg_vtx.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < hot_table_size; i++)
            hot_table[i] = deg_vtx[i].second;
        
        printf("ECG adaptive prefetch: %d-bit container, %d-bit ID, %d spare, "
               "%d prefetch bits -> %d-entry hot table\n",
               container_bits, id_bits, spare, pfx_bits, hot_table_size);
        printf("  Top hubs: [%d(d=%ld)", hot_table[0], (long)g.out_degree(hot_table[0]));
        for (int i = 1; i < min(hot_table_size, 5); i++)
            printf(", %d(d=%ld)", hot_table[i], (long)g.out_degree(hot_table[i]));
        if (hot_table_size > 5) printf(", ... +%d more", hot_table_size - 5);
        printf("]\n");
    }
    
    // Build per-vertex prefetch index: for each vertex, what's its rank
    // in the hot table? This lets us quickly check if a neighbor is a hub.
    vector<int> hub_rank(g.num_nodes(), -1);  // -1 = not a hub
    for (int i = 0; i < hot_table_size; i++)
        hub_rank[hot_table[i]] = i;

    // Check environment variable to enable/disable ECG_PFX hint emission.
    const char* ecg_prefetch_env = getenv("GEM5_ENABLE_ECG_PFX_HINTS");
    if (!ecg_prefetch_env) ecg_prefetch_env = getenv("ECG_PREFETCH");
    bool ecg_prefetch_enabled = ecg_prefetch_env && string(ecg_prefetch_env) != "0";
    int pfx_lookahead = gem5_env_int_clamped("GEM5_ECG_PFX_LOOKAHEAD", 4, 0, 64);

    // === Mode 6: Per-Edge ECG Mask (paper's ECG design, sprint 6f-5) ===
    // Build the per-edge mask array once before iteration begins. When
    // GEM5_ECG_PFX_MODE=6 the inner loop reads pre-encoded prefetch targets
    // from this array instead of computing them at runtime.
    // (Also accepts ECG_PREFETCH_MODE for compatibility with roi_matrix.py's
    // cache_sim env naming.)
    int ecg_pfx_mode = gem5_env_int_clamped("GEM5_ECG_PFX_MODE", -1, -1, 7);
    if (ecg_pfx_mode < 0) {
        ecg_pfx_mode = gem5_env_int_clamped("ECG_PREFETCH_MODE", 0, 0, 7);
    }
    int edge_mask_lookahead = gem5_env_int_clamped("GEM5_ECG_EDGE_MASK_LOOKAHEAD",
        gem5_env_int_clamped("ECG_EDGE_MASK_LOOKAHEAD", 8, 1, 64), 1, 64);
    vector<vector<uint64_t>> in_edge_masks_by_src;
    if (ecg_prefetch_enabled && ecg_pfx_mode == 6) {
        vector<uint8_t> avg_reref_by_line;
        ecg_mode6::computeAvgRerefByLine(popt_matrix.data(), popt_num_cache_lines,
                                         kNumEpochs, avg_reref_by_line);
        vector<uint8_t> tiers;
        ecg_mode6::computeDegreeTiers(g, tiers);
        ecg_mode6::buildInEdgeMasks(g, tiers, avg_reref_by_line,
                                    edge_mask_lookahead, kNumVtxPerLine,
                                    in_edge_masks_by_src, "gem5-PR");
        printf("[gem5 ECG mode 6] lookahead=%d (per-edge mask path active)\n",
               edge_mask_lookahead);
    }

    for (NodeID n = 0; n < g.num_nodes(); n++)
        outgoing_contrib[n] = init_score / g.out_degree(n);

    GEM5_RESET_STATS();
    GEM5_WORK_BEGIN(GEM5_WORK_COMPUTE);

    // Prefetch dedup window — tracks recently prefetched hub indices
    vector<NodeID> pfx_window(PREFETCH_WINDOW, -1);
    int pfx_window_pos = 0;

    for (int iter = 0; iter < max_iters; iter++) {
        double error = 0;
        for (NodeID u = 0; u < g.num_nodes(); u++) {
            GEM5_SET_VERTEX(u);
            ScoreT incoming_total = 0;

            auto in_neigh = g.in_neigh(u);

            // === Mode 6: per-edge ECG mask path (ECG paper's design) ===
            // Pre-encoded mask at in_edge_masks_by_src[u][edge_pos] carries
            // the dest, DBG/POPT bits, and a POPT-ranked prefetch target.
            // We decode dest from the mask and issue the prefetch hint;
            // demand load on outgoing_contrib[v] happens as normal.
            if (ecg_prefetch_enabled && ecg_pfx_mode == 6
                && u < static_cast<NodeID>(in_edge_masks_by_src.size())) {
                const auto& src_masks = in_edge_masks_by_src[u];
                size_t edge_pos = 0;
                for (auto it = in_neigh.begin(); it != in_neigh.end(); ++it, ++edge_pos) {
                    uint64_t mask = (edge_pos < src_masks.size()) ? src_masks[edge_pos] : 0;
                    NodeID v = static_cast<NodeID>(ecg_mode6::extractDest(mask));
                    uint32_t prefetch_target = ecg_mode6::extractPrefetchTarget(mask);
                    if (prefetch_target != 0) {
                        bool in_window = false;
                        for (int w = 0; w < PREFETCH_WINDOW; w++) {
                            if (pfx_window[w] == static_cast<NodeID>(prefetch_target)) {
                                in_window = true;
                                break;
                            }
                        }
                        if (!in_window) {
                            // S69PRE-M1-MASK: emit FULL mode-6 mask via ecg.extract
                            // when ISA-delivered metadata channel is enabled.
                            // Else fall back to the legacy prefetch-target-only path.
                            if (gem5_ecg_extract_enabled()) {
                                GEM5_ECG_EXTRACT_MASK(mask);
                            } else {
                                GEM5_ECG_PFX_TARGET(prefetch_target);
                            }
                            pfx_window[pfx_window_pos % PREFETCH_WINDOW] = prefetch_target;
                            pfx_window_pos++;
                        }
                    }
                    incoming_total += outgoing_contrib[v];
                }
                ScoreT old_score = scores[u];
                scores[u] = base_score + kDamp * incoming_total;
                error += fabs(scores[u] - old_score);
                outgoing_contrib[u] = scores[u] / g.out_degree(u);
                continue;
            }

            for (auto it = in_neigh.begin(); it != in_neigh.end(); ++it) {
                NodeID v = *it;
                if (ecg_prefetch_enabled && pfx_lookahead > 0) {
                    NodeID pfx_target = -1;
                    int best_rank = hot_table_size + 1;
                    auto jt = it;
                    for (int step = 0; step < pfx_lookahead; step++) {
                        ++jt;
                        if (jt == in_neigh.end()) break;
                        NodeID candidate = *jt;
                        if (candidate >= 0 && candidate < static_cast<NodeID>(hub_rank.size()) &&
                            hub_rank[candidate] >= 0 && hub_rank[candidate] < best_rank) {
                            best_rank = hub_rank[candidate];
                            pfx_target = candidate;
                        }
                    }
                    if (pfx_target >= 0) {
                        bool in_window = false;
                        for (int w = 0; w < PREFETCH_WINDOW; w++) {
                            if (pfx_window[w] == pfx_target) {
                                in_window = true;
                                break;
                            }
                        }
                        if (!in_window) {
                            GEM5_ECG_PFX_TARGET(pfx_target);
                            pfx_window[pfx_window_pos % PREFETCH_WINDOW] = pfx_target;
                            pfx_window_pos++;
                        }
                    }
                }
                incoming_total += outgoing_contrib[v];
                
                // ECG per-edge prefetch: if neighbor v is a hub vertex,
                // prefetch the NEXT hub in the hot table (the one after v's
                // rank). This brings in the most likely next high-reuse
                // vertex before it's demanded.
                if (ecg_prefetch_enabled && pfx_lookahead == 0 && hub_rank[v] >= 0) {
                    int next_hub = (hub_rank[v] + 1) % hot_table_size;
                    NodeID pfx_target = hot_table[next_hub];
                    
                    // Dedup: skip if recently prefetched
                    bool in_window = false;
                    for (int w = 0; w < PREFETCH_WINDOW; w++) {
                        if (pfx_window[w] == pfx_target) {
                            in_window = true;
                            break;
                        }
                    }
                    if (!in_window) {
                        GEM5_ECG_PFX_TARGET(pfx_target);
                        pfx_window[pfx_window_pos % PREFETCH_WINDOW] = pfx_target;
                        pfx_window_pos++;
                    }
                }
            }
            ScoreT old_score = scores[u];
            scores[u] = base_score + kDamp * incoming_total;
            error += fabs(scores[u] - old_score);
            outgoing_contrib[u] = scores[u] / g.out_degree(u);
        }
        if (error < epsilon) break;
    }

    GEM5_WORK_END(GEM5_WORK_COMPUTE);
    GEM5_DUMP_STATS();
    return scores;
}

void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
    vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
    for (NodeID n = 0; n < g.num_nodes(); n++)
        score_pairs[n] = make_pair(n, scores[n]);
    int k = min(5, (int)g.num_nodes());
    partial_sort(score_pairs.begin(), score_pairs.begin() + k, score_pairs.end(),
                [](auto a, auto b) { return a.second > b.second; });
    for (int i = 0; i < k; i++)
        cout << score_pairs[i].first << ": " << score_pairs[i].second << endl;
}

bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores, double target_error) {
    const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
    pvector<ScoreT> incoming_sums(g.num_nodes(), 0);
    double error = 0;
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
        for (NodeID v : g.out_neigh(u))
            incoming_sums[v] += outgoing_contrib;
    }
    for (NodeID n = 0; n < g.num_nodes(); n++)
        error += fabs(base_score + kDamp * incoming_sums[n] - scores[n]);
    cout << "Total Error: " << error << endl;
    return error < target_error;
}

int main(int argc, char *argv[]) {
    CLPageRank cli(argc, argv, "pagerank-gem5", 1e-4, 20);
    if (!cli.ParseArgs()) return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();

    auto PRBound = [&cli](const Graph &g) {
        return PageRankPullGS_Gem5(g, cli.max_iters(), cli.tolerance());
    };
    auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
        return PRVerifier(g, scores, cli.tolerance());
    };
    BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
    return 0;
}
