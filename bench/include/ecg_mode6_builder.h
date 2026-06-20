// ============================================================================
// ECG Mode 6: Per-Edge Mask Builder (shared across cache_sim/gem5/Sniper)
// ============================================================================
//
// The ECG paper's "correct" prefetcher design: each edge in the CSR carries
// a packed mask encoding (dest_id | DBG_tier | POPT_quant | prefetch_target).
// The prefetch_target is selected from src's iteration context as the
// "next-K POPT-best dest" in src's in_neighbors after the current one.
//
// This header is included by:
//   * bench/src_sim/{pr,bfs,sssp}.cc   (cache_sim mode 6 path)
//   * bench/src_gem5/{pr,bfs,sssp}.cc  (gem5 cycle-accurate mode 6 path)
//   * bench/src_sniper/{pr,bfs,sssp}.cc (Sniper cycle-accurate mode 6 path)
//
// Caller responsibilities:
//   1) Build POPT offset matrix via popt.h::makeOffsetMatrix and pass it in
//      as `popt_matrix`.
//   2) Compute the tier array (degree-bucketed) and pass it as `tiers`,
//      or pass empty vector to skip DBG encoding.
//   3) Provide k_lookahead (typical default: 8). 0 disables prefetch encoding.
//
// Output:
//   `out_masks[src][edge_pos]` is a 64-bit mask:
//     [0:24]  dest_id (24 bits)
//     [24:26] DBG tier (2 bits)
//     [26:33] POPT quant (7 bits, 0-127)
//     [33:64] prefetch target (31 bits, 0 = no prefetch)
//   Epoch-carrying masks use:
//     [33:49] next-ref epoch (16 bits)
//     [49:64] prefetch target (15 bits, 0 = no prefetch)
//
// For gem5/Sniper kernels that only need the prefetch target, use the
// convenience helper `extractPrefetchTarget(mask)`.

#ifndef ECG_MODE6_BUILDER_H
#define ECG_MODE6_BUILDER_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ecg_mode6 {

// === Mask field accessors (constexpr inline for free encoding/decoding) ===

constexpr int kDestBits     = 24;
constexpr int kDbgBits      = 2;
constexpr int kPoptBits     = 7;
constexpr int kPrefetchBits = 31;
constexpr int kEpochBits    = 16;
constexpr int kPrefetchEpochBits = 15;
constexpr int kDestShift     = 0;
constexpr int kDbgShift      = kDestBits;
constexpr int kPoptShift     = kDestBits + kDbgBits;
constexpr int kPrefetchShift = kDestBits + kDbgBits + kPoptBits;
constexpr int kEpochShift    = kPrefetchShift;
constexpr int kPrefetchEpochShift = kEpochShift + kEpochBits;

inline uint32_t extractDest(uint64_t mask)         { return static_cast<uint32_t>((mask >> kDestShift)     & 0xFFFFFFu); }
inline uint8_t  extractDbg(uint64_t mask)          { return static_cast<uint8_t> ((mask >> kDbgShift)      & 0x3u); }
inline uint8_t  extractPopt(uint64_t mask)         { return static_cast<uint8_t> ((mask >> kPoptShift)     & 0x7Fu); }
inline uint16_t extractEpoch(uint64_t mask)        { return static_cast<uint16_t>((mask >> kEpochShift)    & 0xFFFFu); }
inline uint32_t extractPrefetchTarget(uint64_t m)  { return static_cast<uint32_t>((m    >> kPrefetchShift) & 0x7FFFFFFFu); }
inline uint32_t extractPrefetchTargetEpoch(uint64_t m) { return static_cast<uint32_t>((m >> kPrefetchEpochShift) & 0x7FFFu); }

inline uint64_t packMask(uint32_t dest, uint8_t dbg, uint8_t popt, uint32_t pfx) {
    return (static_cast<uint64_t>(dest & 0xFFFFFFu)) |
           (static_cast<uint64_t>(dbg  & 0x3u)   << kDbgShift) |
           (static_cast<uint64_t>(popt & 0x7Fu)  << kPoptShift) |
           (static_cast<uint64_t>(pfx  & 0x7FFFFFFFu) << kPrefetchShift);
}

inline uint64_t packMaskEpoch(uint32_t dest, uint8_t dbg, uint8_t popt,
                              uint16_t epoch, uint32_t pfx) {
    return (static_cast<uint64_t>(dest & 0xFFFFFFu)) |
           (static_cast<uint64_t>(dbg  & 0x3u)   << kDbgShift) |
           (static_cast<uint64_t>(popt & 0x7Fu)  << kPoptShift) |
           (static_cast<uint64_t>(epoch & 0xFFFFu) << kEpochShift) |
           (static_cast<uint64_t>(pfx  & 0x7FFFu) << kPrefetchEpochShift);
}

// === Helper: derive avg_reref_by_line from POPT matrix ===
//
// POPT matrix layout (from popt.h::makeOffsetMatrix):
//   matrix[epoch * num_cache_lines + cline] = uint8_t
//     - high bit (0x80) set: invalid/no-data entry
//     - low 7 bits: quantized re-reference distance (0..127)
inline void computeAvgRerefByLine(
    const uint8_t* popt_matrix,
    uint32_t num_cache_lines,
    uint32_t num_epochs,
    std::vector<uint8_t>& out_avg_reref)
{
    out_avg_reref.assign(num_cache_lines, 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t cl_i = 0; cl_i < static_cast<int64_t>(num_cache_lines); ++cl_i) {
        uint32_t cline = static_cast<uint32_t>(cl_i);
        uint32_t total_dist = 0;
        uint32_t count = 0;
        for (uint32_t e = 0; e < num_epochs; e++) {
            uint8_t entry = popt_matrix[e * num_cache_lines + cline];
            if ((entry & 0x80) == 0) {
                total_dist += (entry & 0x7F);
                count++;
            }
        }
        out_avg_reref[cline] = count > 0
            ? static_cast<uint8_t>(std::min(total_dist / count, uint32_t(127)))
            : 0;
    }
}

// === Per-vertex tier computation (degree-bucketed, mirrors cache_sim) ===
template <typename GraphT>
void computeDegreeTiers(const GraphT& g, std::vector<uint8_t>& out_tiers) {
    uint32_t n = g.num_nodes();
    out_tiers.assign(n, 0);
    if (n == 0) return;

    std::vector<int64_t> degs(n);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (uint32_t v = 0; v < n; v++)
        degs[v] = g.out_degree(v);

    // Compute simple quartile cutoffs by sorting a copy.
    std::vector<int64_t> sorted_degs = degs;
    std::sort(sorted_degs.begin(), sorted_degs.end());
    int64_t q1 = sorted_degs[(n * 1) / 4];
    int64_t q2 = sorted_degs[(n * 2) / 4];
    int64_t q3 = sorted_degs[(n * 3) / 4];

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (uint32_t v = 0; v < n; v++) {
        int64_t d = degs[v];
        uint8_t tier = (d >= q3) ? 3 : (d >= q2) ? 2 : (d >= q1) ? 1 : 0;
        out_tiers[v] = tier;
    }
}

// === Main builder: in_edge masks for src-side iteration (PR pull pattern) ===
//
// PR pull iterates over each src's in_neighbors. For each (src, edge_pos)
// where edge_pos indexes into src.in_neigh, we encode a mask whose
// prefetch_target field names the next-K POPT-best vertex in src's
// in_neighbors at positions edge_pos+1..edge_pos+K.
//
// The builder is thread-safe across src (outer loop parallelizes).
//
// Arguments:
//   g                 — input graph
//   tiers             — per-vertex tier array (or empty to skip DBG encoding)
//   avg_reref_by_line — derived from POPT matrix via computeAvgRerefByLine
//                       (or empty to skip POPT-ranked target selection)
//   k_lookahead       — how many positions ahead to scan for POPT-best target
//                       (0 disables prefetch encoding)
//   num_vtx_per_line  — used to map vertex -> cache line for reref lookup
//                       (default: 16, matching cache_sim's numVtxPerLine)
//   out_masks         — output: out_masks[src][edge_pos] is the 64-bit mask
//   stats_log_label   — optional label printed to stdout summarizing build stats
//                       (pass empty string to suppress logging)
template <typename GraphT>
void buildInEdgeMasks(
    const GraphT& g,
    const std::vector<uint8_t>& tiers,
    const std::vector<uint8_t>& avg_reref_by_line,
    int k_lookahead,
    int num_vtx_per_line,
    std::vector<std::vector<uint64_t>>& out_masks,
    const char* stats_log_label = nullptr)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    uint32_t n = g.num_nodes();
    out_masks.clear();
    out_masks.resize(n);
    if (n == 0) return;
    if (num_vtx_per_line < 1) num_vtx_per_line = 16;

    uint64_t edge_count = 0;
    uint64_t encoded_count = 0;
    const bool have_tiers = !tiers.empty();
    const bool have_reref = !avg_reref_by_line.empty();

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 128) reduction(+:edge_count, encoded_count)
#endif
    for (uint32_t src = 0; src < n; src++) {
        std::vector<uint32_t> neighbors;
        neighbors.reserve(64);
        for (auto v : g.in_neigh(src))
            neighbors.push_back(static_cast<uint32_t>(v));

        auto& masks = out_masks[src];
        masks.resize(neighbors.size(), 0);

        for (size_t i = 0; i < neighbors.size(); i++) {
            uint32_t dest = neighbors[i];
            edge_count++;

            uint8_t dbg = (have_tiers && dest < tiers.size()) ? tiers[dest] : 0;

            uint8_t popt = 0;
            if (have_reref) {
                uint32_t dcl = dest / static_cast<uint32_t>(num_vtx_per_line);
                if (dcl < avg_reref_by_line.size())
                    popt = avg_reref_by_line[dcl] & 0x7F;
            }

            uint32_t prefetch_target = 0;
            if (k_lookahead > 0 && have_reref) {
                uint8_t best = 128;
                int probe = std::min<int>(k_lookahead,
                    static_cast<int>(neighbors.size()) - static_cast<int>(i) - 1);
                for (int step = 1; step <= probe; step++) {
                    uint32_t cand = neighbors[i + step];
                    uint32_t cl = cand / static_cast<uint32_t>(num_vtx_per_line);
                    if (cl < avg_reref_by_line.size()) {
                        uint8_t d = avg_reref_by_line[cl];
                        if (d < best) {
                            best = d;
                            prefetch_target = cand;
                        }
                    }
                }
            }
            if (prefetch_target != 0) encoded_count++;

            masks[i] = packMask(dest, dbg, popt, prefetch_target);
        }
    }

    if (stats_log_label && stats_log_label[0]) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - t0).count();
        double pct = edge_count > 0
            ? (100.0 * static_cast<double>(encoded_count) / static_cast<double>(edge_count))
            : 0.0;
        std::printf("[ecg_mode6 %s] vertices=%u edges=%llu encoded=%llu (%.1f%%) build_s=%.4f\n",
            stats_log_label, n,
            static_cast<unsigned long long>(edge_count),
            static_cast<unsigned long long>(encoded_count),
            pct, us / 1e6);
    }
}

} // namespace ecg_mode6

#endif // ECG_MODE6_BUILDER_H
