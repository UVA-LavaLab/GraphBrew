// ============================================================================
// ECG next-reference epoch builder (shared cache_sim/gem5/Sniper helpers)
// ============================================================================

#ifndef ECG_EPOCH_BUILDER_H
#define ECG_EPOCH_BUILDER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ecg_epoch {

// Demand epoch for vertex u under a deterministic ID-order pull sweep (PR):
// position in [0,ne) proportional to u/num_nodes. SSOT for all 3 sims.
inline uint32_t currentEpoch(int64_t u, int64_t num_nodes, uint32_t ne) {
    return num_nodes > 0
        ? static_cast<uint32_t>((static_cast<uint64_t>(u) * ne) / static_cast<uint64_t>(num_nodes))
        : 0u;
}

// Path A "filtered DROPLET" epoch gate — SSOT for the lookahead-prefetch decision
// across cache_sim / gem5 / Sniper. Returns true to prefetch the candidate.
// filter: 0=off, 1=skip NEAR (dist<thresh), 2=skip FAR (dist>thresh); dist is the
// circular epoch distance from the demand epoch cur_ep.
inline bool prefetchKeep(uint16_t cand_ep, uint32_t cur_ep, uint32_t ne,
                         int filter, uint32_t thresh) {
    if (filter == 0 || ne <= 1) return true;
    uint32_t dist = (static_cast<uint32_t>(cand_ep) + ne - cur_ep) % ne;
    if (filter == 1 && dist < thresh) return false;
    if (filter == 2 && dist > thresh) return false;
    return true;
}

// Build per-edge next-reference epochs. By default (push_out_edges=false) this is PR's
// in-PULL pattern: src reads in_neigh(src)'s property, and a dest's property is next
// referenced by dest's OUT-neighbours (the other vertices that pull it). For PUSH/out
// kernels (BFS top-down, SSSP — push_out_edges=true) src writes out_neigh(src)'s property,
// and a dest's property is next referenced by dest's IN-neighbours. Swapping the two
// neighbour directions makes the per-edge epoch the graph TRANSPOSE for the kernel (the
// same direction cache_sim's buildOutEdgeMasks uses), so non-PR ECG_GRASP_POPT delivers a
// correct-direction epoch instead of nothing.
template <typename GraphT>
void buildInEdgeEpochs(const GraphT& g,
                       uint32_t numVtxPerLine,
                       uint32_t ne,
                       bool linemin,
                       std::vector<std::vector<uint16_t>>& out,
                       bool push_out_edges = false)
{
    const uint32_t n = static_cast<uint32_t>(g.num_nodes());
    out.clear();
    out.resize(n);
    if (n == 0) return;
    if (numVtxPerLine == 0) numVtxPerLine = 16;
    if (ne < 2) ne = 2;

    std::vector<uint64_t> off(static_cast<size_t>(n) + 1, 0);
    std::vector<uint32_t> nbr;
    nbr.reserve(static_cast<size_t>(g.num_edges_directed()));

    for (uint32_t v = 0; v < n; ++v) {
        off[v] = nbr.size();
        // dest is next-referenced by the vertices that ACCESS it: out_neigh for PR's pull,
        // in_neigh for push/out kernels (the transpose).
        if (push_out_edges) {
            for (auto w_raw : g.in_neigh(v)) {
                const uint32_t w = static_cast<uint32_t>(w_raw);
                if (w < n) nbr.push_back(w);
            }
        } else {
            for (auto w_raw : g.out_neigh(v)) {
                const uint32_t w = static_cast<uint32_t>(w_raw);
                if (w < n) nbr.push_back(w);
            }
        }
        std::sort(nbr.begin() + static_cast<std::ptrdiff_t>(off[v]), nbr.end());
    }
    off[n] = nbr.size();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 128)
#endif
    for (int64_t src_i = 0; src_i < static_cast<int64_t>(n); ++src_i) {
        const uint32_t src = static_cast<uint32_t>(src_i);
        std::vector<uint32_t> in;
        in.reserve(64);
        // The edges src ACCESSES: in_neigh for PR's pull, out_neigh for push/out kernels.
        if (push_out_edges) {
            for (auto dest_raw : g.out_neigh(src)) {
                const uint32_t dest = static_cast<uint32_t>(dest_raw);
                in.push_back(dest);
            }
        } else {
            for (auto dest_raw : g.in_neigh(src)) {
                const uint32_t dest = static_cast<uint32_t>(dest_raw);
                in.push_back(dest);
            }
        }

        auto& epochs = out[src];
        epochs.resize(in.size(), static_cast<uint16_t>(ne - 1));

        for (size_t edge_pos = 0; edge_pos < in.size(); ++edge_pos) {
            const uint32_t dest = in[edge_pos];
            if (dest >= n) continue;

            const uint32_t v0 = linemin
                ? (dest / numVtxPerLine) * numVtxPerLine
                : dest;
            const uint32_t v1 = linemin
                ? std::min<uint32_t>(v0 + numVtxPerLine, n)
                : std::min<uint32_t>(dest + 1, n);

            uint32_t best_dist = std::numeric_limits<uint32_t>::max();
            uint32_t best_epoch = ne - 1;
            for (uint32_t w = v0; w < v1; ++w) {
                const uint64_t a = off[w];
                const uint64_t b = off[w + 1];
                if (a >= b) continue;

                auto begin = nbr.begin() + static_cast<std::ptrdiff_t>(a);
                auto end = nbr.begin() + static_cast<std::ptrdiff_t>(b);
                auto it = std::upper_bound(begin, end, src);

                uint32_t next_nbr = 0;
                uint32_t dist = 0;
                if (it != end) {
                    next_nbr = *it;
                    dist = next_nbr - src;
                } else {
                    next_nbr = *begin;
                    dist = next_nbr + n - src;
                }

                if (dist < best_dist) {
                    best_dist = dist;
                    best_epoch = static_cast<uint32_t>(
                        (static_cast<uint64_t>(next_nbr) * ne) /
                        std::max<uint32_t>(1u, n));
                    if (best_epoch >= ne) best_epoch = ne - 1;
                }
            }

            epochs[edge_pos] = static_cast<uint16_t>(best_epoch);
        }
    }
}

} // namespace ecg_epoch

#endif // ECG_EPOCH_BUILDER_H
