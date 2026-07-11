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

// Schedule-2 wire format shared by gem5/Sniper kernels:
// dest[0:32] | first[32:48] | second[48:64].
inline uint64_t packEpochPairRecord(uint32_t dest, uint16_t first,
                                    uint16_t second) {
    return static_cast<uint64_t>(dest) |
           (static_cast<uint64_t>(first) << 32) |
           (static_cast<uint64_t>(second) << 48);
}

inline uint32_t extractEpochPairDest(uint64_t record) {
    return static_cast<uint32_t>(record);
}

inline uint16_t extractEpochPairFirst(uint64_t record) {
    return static_cast<uint16_t>((record >> 32) & 0xFFFFu);
}

inline uint16_t extractEpochPairSecond(uint64_t record) {
    return static_cast<uint16_t>((record >> 48) & 0xFFFFu);
}

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

inline EpochPair nextEpochPairForLine(
        const std::vector<uint64_t>& off,
        const std::vector<uint32_t>& readers,
        uint32_t n, uint32_t src, uint32_t dest,
        uint32_t numVtxPerLine, uint32_t ne, bool linemin) {
    EpochPair pair;
    if (dest >= n) return pair;
    const uint32_t v0 = linemin
        ? (dest / numVtxPerLine) * numVtxPerLine : dest;
    const uint32_t v1 = linemin
        ? std::min<uint32_t>(v0 + numVtxPerLine, n)
        : std::min<uint32_t>(dest + 1, n);

    uint64_t best_distance[2] = {
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(),
    };
    uint16_t best_epoch[2] = {0, 0};
    auto consider = [&](uint64_t distance, uint16_t epoch) {
        if (distance < best_distance[0]) {
            best_distance[1] = best_distance[0];
            best_epoch[1] = best_epoch[0];
            best_distance[0] = distance;
            best_epoch[0] = epoch;
        } else if (distance < best_distance[1]) {
            best_distance[1] = distance;
            best_epoch[1] = epoch;
        }
    };

    for (uint32_t w = v0; w < v1; ++w) {
        const uint64_t a = off[w], b = off[w + 1];
        if (a >= b) continue;
        auto begin = readers.begin() + static_cast<std::ptrdiff_t>(a);
        auto end = readers.begin() + static_cast<std::ptrdiff_t>(b);
        auto it = std::upper_bound(begin, end, src);
        uint32_t completed_cycles = 0;
        for (int k = 0; k < 2; ++k) {
            if (it == end) {
                it = begin;
                ++completed_cycles;
            }
            const uint32_t selected = *it;
            uint32_t epoch = static_cast<uint32_t>(
                (static_cast<uint64_t>(selected) * ne) / n);
            if (epoch >= ne) epoch = ne - 1;
            const uint64_t absolute =
                static_cast<uint64_t>(selected) +
                static_cast<uint64_t>(completed_cycles) * n;
            consider(
                absolute - src, static_cast<uint16_t>(epoch));
            ++it;
        }
    }

    if (best_distance[0] == std::numeric_limits<uint64_t>::max())
        return pair;
    pair.first = best_epoch[0];
    pair.second =
        best_distance[1] == std::numeric_limits<uint64_t>::max()
            ? best_epoch[0] : best_epoch[1];
    pair.valid = true;
    return pair;
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
            pairs[edge_pos] = nextEpochPairForLine(
                off, readers, n, src, accessed[edge_pos],
                numVtxPerLine, ne, linemin);
        }
    }
}

// Build the packed Schedule-2 stream directly, avoiding a second O(E) nested
// EpochPair representation in gem5/Sniper.
template <typename GraphT>
void buildInEdgeEpochPairRecords(
        const GraphT& g, uint32_t numVtxPerLine, uint32_t ne, bool linemin,
        std::vector<uint64_t>& record_off,
        std::vector<uint64_t>& records,
        bool push_out_edges = false) {
    const uint32_t n = static_cast<uint32_t>(g.num_nodes());
    record_off.assign(static_cast<size_t>(n) + 1, 0);
    records.clear();
    if (n == 0) return;
    if (numVtxPerLine == 0) numVtxPerLine = 16;
    if (ne < 2) ne = 2;
    if (ne > 65535) ne = 65535;

    std::vector<uint64_t> off;
    std::vector<uint32_t> readers;
    buildReaderCsr(g, push_out_edges, off, readers);
    for (uint32_t src = 0; src < n; ++src) {
        std::vector<uint32_t> accessed;
        accessedVertices(g, src, push_out_edges, accessed);
        record_off[src + 1] = record_off[src] + accessed.size();
    }
    records.assign(record_off[n], 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 128)
#endif
    for (int64_t src_i = 0; src_i < static_cast<int64_t>(n); ++src_i) {
        const uint32_t src = static_cast<uint32_t>(src_i);
        std::vector<uint32_t> accessed;
        accessedVertices(g, src, push_out_edges, accessed);
        for (size_t edge = 0; edge < accessed.size(); ++edge) {
            const EpochPair pair = nextEpochPairForLine(
                off, readers, n, src, accessed[edge],
                numVtxPerLine, ne, linemin);
            records[record_off[src] + edge] = packEpochPairRecord(
                accessed[edge], pair.first, pair.second);
        }
    }
}

} // namespace ecg_epoch

#endif // ECG_EPOCH_BUILDER_H
