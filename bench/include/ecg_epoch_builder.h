// ============================================================================
// ECG next-reference epoch builder (shared cache_sim/gem5/Sniper helpers)
// ============================================================================

#ifndef ECG_EPOCH_BUILDER_H
#define ECG_EPOCH_BUILDER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ecg_epoch {

struct EpochPair {
    uint16_t first = 0;
    uint16_t second = 0;
    bool valid = false;
};

// Demand epoch for vertex u under a deterministic ID-order pull sweep (PR):
// position in [0,ne) proportional to u/num_nodes. SSOT for all 3 sims.
inline uint32_t currentEpoch(int64_t u, int64_t num_nodes, uint32_t ne) {
    return num_nodes > 0
        ? static_cast<uint32_t>(
              (static_cast<uint64_t>(u) * ne) /
              static_cast<uint64_t>(num_nodes))
        : 0u;
}

// Path A "filtered DROPLET" epoch gate — SSOT for the lookahead-prefetch decision
// across cache_sim / gem5 / Sniper. Returns true to prefetch the candidate.
inline bool prefetchKeep(uint16_t cand_ep, uint32_t cur_ep, uint32_t ne,
                         int filter, uint32_t thresh) {
    if (filter == 0 || ne <= 1) return true;
    uint32_t dist = (static_cast<uint32_t>(cand_ep) + ne - cur_ep) % ne;
    if (filter == 1 && dist < thresh) return false;
    if (filter == 2 && dist > thresh) return false;
    return true;
}

template <typename GraphT>
void buildReaderCsr(const GraphT& g, bool push_out_edges,
                    std::vector<uint64_t>& off,
                    std::vector<uint32_t>& readers) {
    const uint32_t n = static_cast<uint32_t>(g.num_nodes());
    off.assign(static_cast<size_t>(n) + 1, 0);
    readers.clear();
    readers.reserve(static_cast<size_t>(g.num_edges_directed()));
    for (uint32_t v = 0; v < n; ++v) {
        off[v] = readers.size();
        if (push_out_edges) {
            for (auto w_raw : g.in_neigh(v)) {
                const uint32_t w = static_cast<uint32_t>(w_raw);
                if (w < n) readers.push_back(w);
            }
        } else {
            for (auto w_raw : g.out_neigh(v)) {
                const uint32_t w = static_cast<uint32_t>(w_raw);
                if (w < n) readers.push_back(w);
            }
        }
        std::sort(
            readers.begin() + static_cast<std::ptrdiff_t>(off[v]),
            readers.end());
    }
    off[n] = readers.size();
}

template <typename GraphT>
void accessedVertices(const GraphT& g, uint32_t src, bool push_out_edges,
                      std::vector<uint32_t>& accessed) {
    accessed.clear();
    if (push_out_edges) {
        for (auto dest_raw : g.out_neigh(src))
            accessed.push_back(static_cast<uint32_t>(dest_raw));
    } else {
        for (auto dest_raw : g.in_neigh(src))
            accessed.push_back(static_cast<uint32_t>(dest_raw));
    }
}

// Build one per-edge next-reference epoch. PR uses pull/in edges by default;
// BFS/SSSP use push_out_edges=true.
template <typename GraphT>
void buildInEdgeEpochs(const GraphT& g,
                       uint32_t numVtxPerLine,
                       uint32_t ne,
                       bool linemin,
                       std::vector<std::vector<uint16_t>>& out,
                       bool push_out_edges = false) {
    const uint32_t n = static_cast<uint32_t>(g.num_nodes());
    out.clear();
    out.resize(n);
    if (n == 0) return;
    if (numVtxPerLine == 0) numVtxPerLine = 16;
    if (ne < 2) ne = 2;
    if (ne > 65535) ne = 65535;

    std::vector<uint64_t> off;
    std::vector<uint32_t> readers;
    buildReaderCsr(g, push_out_edges, off, readers);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 128)
#endif
    for (int64_t src_i = 0; src_i < static_cast<int64_t>(n); ++src_i) {
        const uint32_t src = static_cast<uint32_t>(src_i);
        std::vector<uint32_t> accessed;
        accessedVertices(g, src, push_out_edges, accessed);
        auto& epochs = out[src];
        epochs.resize(accessed.size(), static_cast<uint16_t>(ne - 1));

        for (size_t edge_pos = 0; edge_pos < accessed.size(); ++edge_pos) {
            const uint32_t dest = accessed[edge_pos];
            if (dest >= n) continue;
            const uint32_t v0 = linemin
                ? (dest / numVtxPerLine) * numVtxPerLine : dest;
            const uint32_t v1 = linemin
                ? std::min<uint32_t>(v0 + numVtxPerLine, n)
                : std::min<uint32_t>(dest + 1, n);

            uint32_t best_dist = std::numeric_limits<uint32_t>::max();
            uint32_t best_epoch = ne - 1;
            for (uint32_t w = v0; w < v1; ++w) {
                const uint64_t a = off[w], b = off[w + 1];
                if (a >= b) continue;
                auto begin = readers.begin() + static_cast<std::ptrdiff_t>(a);
                auto end = readers.begin() + static_cast<std::ptrdiff_t>(b);
                auto it = std::upper_bound(begin, end, src);
                uint32_t reader;
                uint32_t dist;
                if (it != end) {
                    reader = *it;
                    dist = reader - src;
                } else {
                    reader = *begin;
                    dist = reader + n - src;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_epoch = static_cast<uint32_t>(
                        (static_cast<uint64_t>(reader) * ne) / n);
                    if (best_epoch >= ne) best_epoch = ne - 1;
                }
            }
            epochs[edge_pos] = static_cast<uint16_t>(best_epoch);
        }
    }
}

// Build the next TWO per-line reference epochs for every accessed edge.
// first is equivalent to buildInEdgeEpochs; second is the next candidate after
// it in circular traversal order. If only one candidate exists, repeat first.
template <typename GraphT>
void buildInEdgeEpochPairs(const GraphT& g,
                           uint32_t numVtxPerLine,
                           uint32_t ne,
                           bool linemin,
                           std::vector<std::vector<EpochPair>>& out,
                           bool push_out_edges = false) {
    const uint32_t n = static_cast<uint32_t>(g.num_nodes());
    out.clear();
    out.resize(n);
    if (n == 0) return;
    if (numVtxPerLine == 0) numVtxPerLine = 16;
    if (ne < 2) ne = 2;
    if (ne > 65535) ne = 65535;

    std::vector<uint64_t> off;
    std::vector<uint32_t> readers;
    buildReaderCsr(g, push_out_edges, off, readers);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 128)
#endif
    for (int64_t src_i = 0; src_i < static_cast<int64_t>(n); ++src_i) {
        const uint32_t src = static_cast<uint32_t>(src_i);
        std::vector<uint32_t> accessed;
        accessedVertices(g, src, push_out_edges, accessed);
        auto& pairs = out[src];
        pairs.resize(accessed.size());

        for (size_t edge_pos = 0; edge_pos < accessed.size(); ++edge_pos) {
            const uint32_t dest = accessed[edge_pos];
            if (dest >= n) continue;
            const uint32_t v0 = linemin
                ? (dest / numVtxPerLine) * numVtxPerLine : dest;
            const uint32_t v1 = linemin
                ? std::min<uint32_t>(v0 + numVtxPerLine, n)
                : std::min<uint32_t>(dest + 1, n);

            std::vector<std::pair<uint32_t, uint16_t>> candidates;
            candidates.reserve(2 * (v1 - v0));
            for (uint32_t w = v0; w < v1; ++w) {
                const uint64_t a = off[w], b = off[w + 1];
                if (a >= b) continue;
                auto begin = readers.begin() + static_cast<std::ptrdiff_t>(a);
                auto end = readers.begin() + static_cast<std::ptrdiff_t>(b);
                auto it = std::upper_bound(begin, end, src);
                bool wrapped = false;
                for (int k = 0; k < 2; ++k) {
                    if (it == end) {
                        it = begin;
                        wrapped = true;
                    }
                    const uint32_t selected = *it;
                    uint32_t epoch = static_cast<uint32_t>(
                        (static_cast<uint64_t>(selected) * ne) / n);
                    if (epoch >= ne) epoch = ne - 1;
                    candidates.emplace_back(
                        (wrapped ? selected + n : selected) - src,
                        static_cast<uint16_t>(epoch));
                    ++it;
                }
            }
            if (candidates.empty()) continue;
            std::sort(
                candidates.begin(), candidates.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.first < rhs.first;
                });
            pairs[edge_pos].first = candidates[0].second;
            pairs[edge_pos].second = candidates.size() > 1
                ? candidates[1].second : candidates[0].second;
            pairs[edge_pos].valid = true;
        }
    }
}

} // namespace ecg_epoch

#endif // ECG_EPOCH_BUILDER_H
