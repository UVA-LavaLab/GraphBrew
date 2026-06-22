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
#include "ecg_epoch_builder.h"

using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;

pvector<ScoreT> PageRankPullGS_Gem5(const Graph &g, int max_iters,
                                     double epsilon = 0) {
    const ScoreT init_score = 1.0f / g.num_nodes();
    const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
    // Page-align the property arrays so their cache set/line mapping is pinned
    // and does not drift with unrelated heap allocations (e.g. sideband path
    // strings). Without this, a few bytes of heap shift change conflict misses
    // in the tiny ROI caches and confound the per-policy comparison.
    constexpr size_t kPropAlign = 4096;
    pvector<ScoreT> scores(g.num_nodes(), init_score, kPropAlign);
    pvector<ScoreT> outgoing_contrib(g.num_nodes(), ScoreT(0), kPropAlign);

    uint8_t edge_id_bits = 1;
    while ((1ULL << edge_id_bits) < static_cast<uint64_t>(g.num_nodes())) edge_id_bits++;
    uint32_t edge_epoch_count = 2;
    if (edge_id_bits < 32) {
        uint32_t spare = 32u - edge_id_bits;
        uint32_t ne_cap = (spare >= 16) ? 65535u : (1u << spare);
        if (ne_cap < 2) ne_cap = 2;
        edge_epoch_count = std::min<uint32_t>(65535u, ne_cap);
    }

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
                        edge_regions, num_edge_regions, edge_epoch_count);

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

    // Path A (epoch-filtered next-K lookahead): prefetch the next-K in-neighbors,
    // each carrying its epoch via GEM5_ECG_PFX_TARGET_EPOCH. Mirrors cache_sim.
    int lean_pfx_k = gem5_env_int_clamped("ECG_EDGE_MASK_PREFETCH", 0, 0, 64);
    int pfx_epoch_filter = gem5_env_int_clamped("ECG_PREFETCH_EPOCH_FILTER", 0, 0, 2);
    int pfx_epoch_thresh_pct =
        gem5_env_int_clamped("ECG_PREFETCH_EPOCH_THRESH_PCT", 50, 0, 100);

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
    vector<vector<uint16_t>> in_edge_epochs_by_src;
    // === Single-stream packed record (LEAN+PACK; matches cache_sim) ===
    // The scattered vector<vector<uint64_t>> mask above is a SEPARATE 8-byte
    // non-property stream that pollutes the LLC and displaces the property
    // (contrib) the epoch eviction is meant to protect — the root cause of gem5
    // ECG_GRASP_POPT scoring WORSE than LRU. cache_sim avoids this by packing the
    // epoch into the spare high bits of the 4-byte edge word and reading ONE
    // contiguous stream. Mirror that with a flat, CSR-ordered uint32 array
    // (dest | epoch<<id_bits): reading record r delivers BOTH the neighbor and
    // its next-ref epoch with the footprint of a single 4-byte CSR edge.
    vector<uint32_t> in_edge_packed_flat;
    vector<uint64_t> packed_off;
    uint32_t pack_id_bits = 1, pack_id_mask = 1;
    bool packed_ok = false;
    bool ecg_extract_enabled = gem5_ecg_extract_enabled();
    if ((ecg_prefetch_enabled || ecg_extract_enabled) && ecg_pfx_mode == 6) {
        vector<uint8_t> avg_reref_by_line;
        ecg_mode6::computeAvgRerefByLine(popt_matrix.data(), popt_num_cache_lines,
                                         kNumEpochs, avg_reref_by_line);
        vector<uint8_t> tiers;
        ecg_mode6::computeDegreeTiers(g, tiers);
        ecg_mode6::buildInEdgeMasks(g, tiers, avg_reref_by_line,
                                    edge_mask_lookahead, kNumVtxPerLine,
                                    in_edge_masks_by_src, "gem5-PR");
        ecg_epoch::buildInEdgeEpochs(g, kNumVtxPerLine, edge_epoch_count, true,
                                     in_edge_epochs_by_src);
        uint64_t pfx_total = 0, pfx_truncated = 0;
        for (size_t src = 0; src < in_edge_masks_by_src.size(); ++src) {
            auto& masks = in_edge_masks_by_src[src];
            const auto& epochs = in_edge_epochs_by_src[src];
            for (size_t i = 0; i < masks.size(); ++i) {
                uint64_t mask = masks[i];
                uint16_t epoch = (i < epochs.size()) ? epochs[i]
                    : static_cast<uint16_t>(edge_epoch_count - 1);
                // Transfer the POPT-best prefetch target (packed at bit 33 by
                // buildInEdgeMasks) into the packMaskEpoch target field at bit 49,
                // where the ecg.extract.wide ISA op and the prefetcher read it.
                uint32_t pfx_target = ecg_mode6::extractPrefetchTarget(mask);
                if (pfx_target != 0) {
                    pfx_total++;
                    // packMaskEpochWide carries a 24-bit prefetch target (<=16,777,215)
                    // by reclaiming the vestigial dbg(2)+popt(7) fields (doc S10.2). Only
                    // graphs with > 2^24 vertices still truncate (then use cache_sim or
                    // the 16-byte record). The ecg.extract.wide ISA op decodes [40:64].
                    if (pfx_target > 0xFFFFFFu) pfx_truncated++;
                }
                masks[i] = ecg_mode6::packMaskEpochWide(
                    ecg_mode6::extractDest(mask),
                    epoch,
                    pfx_target);
            }
        }
        if (pfx_truncated > 0) {
            std::cerr << "[gem5 ECG_PFX WARNING] " << pfx_truncated << "/" << pfx_total
                      << " prefetch targets exceed the 24-bit ISA mask field (>16,777,215) and "
                         "are TRUNCATED to wrong vertices. The gem5 ECG_PFX ISA testbed is "
                         "valid only for graphs <=16,777,215 vertices; use cache_sim for "
                         "larger-graph prefetch evaluation (no field limit). Set "
                         "ECG_PFX_STRICT_TARGET=1 to abort instead.\n";
            if (std::getenv("ECG_PFX_STRICT_TARGET")) {
                std::cerr << "[gem5 ECG_PFX] ECG_PFX_STRICT_TARGET set -> aborting.\n";
                std::abort();
            }
        }
        printf("[gem5 ECG mode 6] lookahead=%d ne=%u (per-edge epoch mask path active)\n",
               edge_mask_lookahead, edge_epoch_count);

        // Build the flat, contiguous, CSR-ordered 4-byte packed record stream
        // when dest+epoch fit in 32 bits. This REPLACES the scattered 8-byte
        // mask reads in the demand path, eliminating the LLC-polluting second
        // stream (the gem5-vs-cache_sim divergence root cause).
        {
            uint32_t nn = static_cast<uint32_t>(g.num_nodes());
            while ((1u << pack_id_bits) < nn && pack_id_bits < 31) pack_id_bits++;
            pack_id_mask = (pack_id_bits >= 32) ? 0xFFFFFFFFu
                                                : ((1u << pack_id_bits) - 1);
            uint32_t epoch_bits = 1;
            while ((1u << epoch_bits) < edge_epoch_count && epoch_bits < 16) epoch_bits++;
            if (pack_id_bits + epoch_bits <= 32) {
                packed_off.assign(static_cast<size_t>(nn) + 1, 0);
                for (uint32_t u = 0; u < nn; u++)
                    packed_off[u + 1] = packed_off[u] +
                        static_cast<uint64_t>(g.in_degree(u));
                in_edge_packed_flat.assign(packed_off[nn], 0);
                for (uint32_t u = 0; u < nn; u++) {
                    const auto& eps = in_edge_epochs_by_src[u];
                    size_t i = 0;
                    for (auto v_raw : g.in_neigh(u)) {
                        uint32_t v = static_cast<uint32_t>(v_raw);
                        uint16_t ep = (i < eps.size()) ? eps[i]
                            : static_cast<uint16_t>(edge_epoch_count - 1);
                        in_edge_packed_flat[packed_off[u] + i] =
                            (v & pack_id_mask) |
                            (static_cast<uint32_t>(ep) << pack_id_bits);
                        i++;
                    }
                }
                packed_ok = true;
                printf("[gem5 ECG mode 6] single-stream packed record ON: "
                       "id_bits=%u epoch_bits=%u (4-byte contiguous, no separate "
                       "mask array)\n", pack_id_bits, epoch_bits);
            } else {
                printf("[gem5 ECG mode 6] packed record OFF: id_bits=%u + "
                       "epoch_bits=%u > 32; using scattered mask fallback\n",
                       pack_id_bits, epoch_bits);
            }
        }
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
            if ((ecg_prefetch_enabled || ecg_extract_enabled) && ecg_pfx_mode == 6
                && u < static_cast<NodeID>(in_edge_masks_by_src.size())) {
                const auto& src_masks = in_edge_masks_by_src[u];
                size_t edge_pos = 0;
                for (auto it = in_neigh.begin(); it != in_neigh.end(); ++it, ++edge_pos) {
                    uint64_t mask;
                    NodeID v;
                    if (packed_ok && (!ecg_prefetch_enabled || lean_pfx_k > 0)) {
                        // EVICTION-ONLY demand mask (dest+epoch, NO fat-mask pfx
                        // target). Path A also takes this branch: it prefetches
                        // via the epoch-filtered lookahead below, so the demand
                        // must NOT also emit a fat-mask Path-B target. One
                        // contiguous 4-byte record read — the
                        // single edge stream that also carries the epoch (no
                        // separate scattered mask array polluting the LLC). The
                        // 4-byte record holds dest+epoch only, which is all the
                        // ECG_RP eviction path needs.
                        uint32_t rec = in_edge_packed_flat[packed_off[u] + edge_pos];
                        v = static_cast<NodeID>(rec & pack_id_mask);
                        uint16_t ep = static_cast<uint16_t>(rec >> pack_id_bits);
                        // Rebuild the 64-bit WIDE layout ecg.extract decodes
                        // (dest[0:24], epoch[24:40]); ecg.extract is a register
                        // op (no memory access).
                        mask = (static_cast<uint64_t>(v) & 0xFFFFFFULL)
                             | (static_cast<uint64_t>(ep) << 24);
                    } else {
                        // PREFETCH path (or unpacked): use the FULL 64-bit WIDE mask,
                        // which carries the 24-bit prefetch target in bits [40:64]. The
                        // 4-byte packed record drops that field, so the prefetch
                        // hint would be lost (pfx_target=0 => no hint emitted); the
                        // prefetch target needs the wider record / full mask.
                        mask = (edge_pos < src_masks.size()) ? src_masks[edge_pos] : 0;
                        v = static_cast<NodeID>(ecg_mode6::extractDest(mask));
                    }
                    if (ecg_extract_enabled) {
                        GEM5_ECG_EXTRACT_MASK(mask);
                    }
                    if (lean_pfx_k > 0 && ecg_prefetch_enabled) {
                        // Path A: prefetch the next-K in-neighbors' contrib[], each
                        // carrying its own epoch. cand is the streamed edge id (full
                        // width). Mirrors cache_sim bench/src_sim/pr.cc Path A.
                        uint32_t ne = edge_epoch_count;
                        uint32_t cur_ep_k = ecg_epoch::currentEpoch(u, g.num_nodes(), ne);
                        uint32_t thresh = static_cast<uint32_t>(
                            (static_cast<uint64_t>(pfx_epoch_thresh_pct) * ne) / 100);
                        auto jt = it;
                        size_t cpos = edge_pos;
                        for (int step = 0; step < lean_pfx_k; step++) {
                            ++jt; ++cpos;
                            if (jt == in_neigh.end()) break;
                            NodeID cand = *jt;
                            if (cand < 0) continue;
                            uint16_t cand_ep = packed_ok
                                ? static_cast<uint16_t>(
                                      in_edge_packed_flat[packed_off[u] + cpos]
                                          >> pack_id_bits)
                                : (cpos < src_masks.size()
                                       ? static_cast<uint16_t>(
                                             ecg_mode6::extractEpochWide(src_masks[cpos]))
                                       : 0);
                            if (!ecg_epoch::prefetchKeep(cand_ep, cur_ep_k, ne,
                                                         pfx_epoch_filter, thresh))
                                continue;
                            GEM5_ECG_PFX_TARGET_EPOCH(static_cast<uint32_t>(cand),
                                                      cand_ep);
                        }
                    } else {
                        // Path B: single packed prefetch target.
                        uint32_t prefetch_target =
                            ecg_mode6::extractPrefetchTargetWide(mask);
                        if (prefetch_target != 0) {
                            bool in_window = false;
                            for (int w = 0; w < PREFETCH_WINDOW; w++) {
                                if (pfx_window[w] ==
                                    static_cast<NodeID>(prefetch_target)) {
                                    in_window = true;
                                    break;
                                }
                            }
                            if (!in_window) {
                                // S69PRE-M1-MASK: emit FULL mode-6 mask via
                                // ecg.extract when the ISA-delivered metadata
                                // channel is enabled. Else fall back to the
                                // legacy prefetch-target-only path.
                                if (!ecg_extract_enabled) {
                                    GEM5_ECG_PFX_TARGET(prefetch_target);
                                }
                                pfx_window[pfx_window_pos % PREFETCH_WINDOW] =
                                    prefetch_target;
                                pfx_window_pos++;
                            }
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
