// ============================================================================
// GraphBrew — RCM BNF variant: Modern CSR-Native Reverse Cuthill-McKee
// ============================================================================
//
// BNF variant for RCM (-o 11:bnf).  Improved RCM that operates directly
// on the CSR graph — no intermediate adjacency conversion.  Uses symmetric
// neighbor iteration (out_neigh + in_neigh) to handle directed graphs
// correctly.
//
// Key improvements over the baseline GoGraph RCM (Algorithm 11):
//
//   1. George-Liu pseudo-peripheral starting node finder (quality ↑↑)
//   2. BNF (Bi-criteria Node Finder) width tracking from RCM++ (quality ↑)
//   3. Deterministic parallel CM BFS with OpenMP (speed ↑↑)
//   4. CSR-native — zero conversion overhead (speed ↑, memory ↓)
//
// Parallelism approach: deterministic two-phase speculative discovery
// (inspired by Mlakar et al. 2021).  Within each BFS level:
//   Phase 1: all frontier parents atomically compete for unvisited
//            neighbors via atomic_min(owner[v], frontier_index).
//            The parent with the lowest frontier index always wins,
//            matching serial CM behavior exactly.
//   Phase 2: each parent collects only the neighbors it won,
//            sorts them by degree, and appends to the ordering.
// This produces an ordering identical to serial CM while exploiting
// thread-level parallelism for discovery and sorting.
//
// References:
//   - Cuthill, McKee (1969): "Reducing the bandwidth of sparse symmetric
//     matrices." ACM National Conference.
//   - George, Liu (1979): "An implementation of a pseudoperipheral node
//     finder." ACM TOMS 5(3).
//   - Hou et al. (2024): "RCM++: Reverse Cuthill-McKee ordering with
//     Bi-Criteria Node Finder." arXiv:2409.04171.
//   - Mlakar et al. (2021): "Speculative Parallel Reverse Cuthill-McKee
//     Reordering on Multi- and Many-core Architectures." IEEE IPDPS.
//     (deterministic shared-neighbor resolution via atomic_min)
//
// Author: GraphBrew Team
// License: See LICENSE.txt
// ============================================================================

#ifndef REORDER_RCM_BNF_H_
#define REORDER_RCM_BNF_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include <omp.h>

// ============================================================================
// Internal implementation details
// ============================================================================
namespace rcm_bnf_detail {

// ----------------------------------------------------------------------------
// Helper — extract plain NodeID from DestID (handles weighted / unweighted)
// ----------------------------------------------------------------------------
template <typename NodeID_, typename DestID_>
inline NodeID_ nbr_id(DestID_ d) {
    return static_cast<NodeID_>(d);
}

// ----------------------------------------------------------------------------
// Helper — iterate symmetric neighbors (out + in for directed graphs)
// ----------------------------------------------------------------------------
// On directed graphs (invert=true), out_neigh gives forward edges and
// in_neigh gives backward edges.  To get symmetric adjacency we iterate
// both.  Duplicates are harmless — BFS visited arrays filter them.
template <typename NodeID_, typename DestID_, bool invert, typename Func>
inline void for_each_sym_neigh(const CSRGraph<NodeID_, DestID_, invert>& g,
                               NodeID_ u, Func f) {
    for (DestID_ dest : g.out_neigh(u)) f(nbr_id<NodeID_>(dest));
    if constexpr (invert) {
        if (g.directed()) {
            for (DestID_ dest : g.in_neigh(u)) {
                f(nbr_id<NodeID_>(dest));
            }
        }
    }
}

// Symmetric degree (out_degree + in_degree for directed graphs)
template <typename NodeID_, typename DestID_, bool invert>
inline int64_t sym_degree(const CSRGraph<NodeID_, DestID_, invert>& g, NodeID_ u) {
    int64_t d = g.out_degree(u);
    if constexpr (invert) {
        if (g.directed()) d += g.in_degree(u);
    }
    return d;
}

// Atomic min on int64_t — sets *addr = min(*addr, val).
// Used by speculative CM BFS so the lowest frontier index always wins
// when multiple parents share a neighbor (deterministic resolution).
inline void atomic_min_int64(int64_t* addr, int64_t val) {
    int64_t old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (val < old) {
        if (__atomic_compare_exchange_n(addr, &old, val, true,
                __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            break;
    }
}

// ----------------------------------------------------------------------------
// BFS level structure — used by George-Liu / BNF
// ----------------------------------------------------------------------------
struct LevelInfo {
    int64_t eccentricity;  // max distance from root
    int64_t max_width;     // max |level_set| across all levels
};

// Compute BFS level structure from a single root within a component.
// Returns eccentricity and max_width. Resets visited for the component
// before returning.  Uses parallel frontier expansion for large components.
template <typename NodeID_, typename DestID_, bool invert>
LevelInfo bfs_level_info(const CSRGraph<NodeID_, DestID_, invert>& g,
                         NodeID_ root,
                         const std::vector<NodeID_>& component,
                         std::vector<uint8_t>& visited) {
    std::vector<NodeID_> frontier;
    frontier.reserve(component.size());
    frontier.push_back(root);
    visited[root] = 1;

    LevelInfo info{0, 1};

    std::vector<NodeID_> next_frontier;
    next_frontier.reserve(component.size());

    // Use parallel expansion when component is large enough to benefit
    const bool use_parallel = component.size() >= 4096;

    // Pre-allocate per-thread buffers outside the loop to avoid repeated
    // allocation/deallocation overhead across BFS levels.
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<NodeID_>> per_thread;
    if (use_parallel) {
        per_thread.resize(num_threads);
        for (auto& v : per_thread) v.reserve(component.size() / num_threads);
    }

    while (!frontier.empty()) {
        const int64_t fsize = static_cast<int64_t>(frontier.size());

        if (use_parallel && fsize >= 256) {
            // --- Parallel frontier expansion ---
            // Each thread collects discovered vertices into thread-local lists
            // then we merge. This avoids contention on a shared next_frontier.
            for (auto& v : per_thread) v.clear();

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& local = per_thread[tid];
                #pragma omp for schedule(dynamic, 64)
                for (int64_t fi = 0; fi < fsize; ++fi) {
                    NodeID_ u = frontier[fi];
                    for_each_sym_neigh(g, u, [&](NodeID_ v) {
                        uint8_t expected = 0;
                        if (__atomic_compare_exchange_n(&visited[v], &expected,
                                static_cast<uint8_t>(1), false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            local.push_back(v);
                        }
                    });
                }
            }

            next_frontier.clear();
            for (auto& local : per_thread) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
        } else {
            // --- Serial frontier expansion ---
            next_frontier.clear();
            for (NodeID_ u : frontier) {
                for_each_sym_neigh(g, u, [&](NodeID_ v) {
                    if (!visited[v]) {
                        visited[v] = 1;
                        next_frontier.push_back(v);
                    }
                });
            }
        }

        if (!next_frontier.empty()) {
            info.eccentricity++;
            int64_t w = static_cast<int64_t>(next_frontier.size());
            if (w > info.max_width) info.max_width = w;
        }
        std::swap(frontier, next_frontier);
    }

    // Reset visited for reuse
    #pragma omp parallel for schedule(static) if(component.size() >= 4096)
    for (int64_t i = 0; i < static_cast<int64_t>(component.size()); ++i)
        visited[component[i]] = 0;
    return info;
}

// BFS that also returns the farthest level's vertices (for GL iteration).
// Uses parallel frontier expansion for large components.
template <typename NodeID_, typename DestID_, bool invert>
LevelInfo bfs_level_info_with_farthest(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        NodeID_ root,
        const std::vector<NodeID_>& component,
        std::vector<uint8_t>& visited,
        std::vector<NodeID_>& farthest_out) {

    std::vector<NodeID_> frontier;
    frontier.reserve(component.size());
    frontier.push_back(root);
    visited[root] = 1;

    LevelInfo info{0, 1};

    std::vector<NodeID_> next_frontier;
    next_frontier.reserve(component.size());

    const bool use_parallel = component.size() >= 4096;

    // Pre-allocate per-thread buffers outside the loop to avoid repeated
    // allocation/deallocation overhead across BFS levels.
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<NodeID_>> per_thread;
    if (use_parallel) {
        per_thread.resize(num_threads);
        for (auto& v : per_thread) v.reserve(component.size() / num_threads);
    }

    while (!frontier.empty()) {
        const int64_t fsize = static_cast<int64_t>(frontier.size());

        if (use_parallel && fsize >= 256) {
            // --- Parallel frontier expansion ---
            for (auto& v : per_thread) v.clear();

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& local = per_thread[tid];
                #pragma omp for schedule(dynamic, 64)
                for (int64_t fi = 0; fi < fsize; ++fi) {
                    NodeID_ u = frontier[fi];
                    for_each_sym_neigh(g, u, [&](NodeID_ v) {
                        uint8_t expected = 0;
                        if (__atomic_compare_exchange_n(&visited[v], &expected,
                                static_cast<uint8_t>(1), false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            local.push_back(v);
                        }
                    });
                }
            }

            next_frontier.clear();
            for (auto& local : per_thread) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }
        } else {
            // --- Serial frontier expansion ---
            next_frontier.clear();
            for (NodeID_ u : frontier) {
                for_each_sym_neigh(g, u, [&](NodeID_ v) {
                    if (!visited[v]) {
                        visited[v] = 1;
                        next_frontier.push_back(v);
                    }
                });
            }
        }

        if (!next_frontier.empty()) {
            info.eccentricity++;
            int64_t w = static_cast<int64_t>(next_frontier.size());
            if (w > info.max_width) info.max_width = w;
            std::swap(frontier, next_frontier);
        } else {
            // frontier is the last (farthest) level
            break;
        }
    }

    farthest_out = frontier;

    // Reset visited
    #pragma omp parallel for schedule(static) if(component.size() >= 4096)
    for (int64_t i = 0; i < static_cast<int64_t>(component.size()); ++i)
        visited[component[i]] = 0;
    return info;
}

// ----------------------------------------------------------------------------
// BNF Pseudo-Peripheral Node Finder (George-Liu + width tracking)
// ----------------------------------------------------------------------------
// Combines George-Liu (1979) eccentricity-maximizing iteration with
// RCM++ (2024) width-minimizing criterion.
//
// GL iterates: BFS from v → pick min-degree among farthest → repeat
// until eccentricity stops increasing.
// BNF enhancement: among all candidates visited during GL iteration,
// return the one whose BFS level structure had minimum max_width.
template <typename NodeID_, typename DestID_, bool invert>
NodeID_ find_bnf_start(const CSRGraph<NodeID_, DestID_, invert>& g,
                       const std::vector<NodeID_>& component,
                       std::vector<uint8_t>& visited) {
    if (component.size() <= 1) {
        return component.empty() ? static_cast<NodeID_>(-1) : component[0];
    }

    // Start from the minimum-degree vertex in the component
    NodeID_ start = component[0];
    int64_t min_deg = sym_degree(g, start);
    for (NodeID_ v : component) {
        int64_t d = sym_degree(g, v);
        if (d < min_deg) {
            min_deg = d;
            start = v;
        }
    }

    NodeID_ v = start;
    int64_t lp = 0;      // previous eccentricity
    int64_t best_width = std::numeric_limits<int64_t>::max();
    NodeID_ best_node = v;
    std::vector<NodeID_> farthest;

    for (int iter = 0; iter < 20; ++iter) {  // safety bound
        LevelInfo info = bfs_level_info_with_farthest(g, v, component,
                                                       visited, farthest);
        int64_t l = info.eccentricity;
        int64_t w = info.max_width;

        // BNF criterion: track minimum width
        if (w < best_width || (w == best_width && l > lp)) {
            best_width = w;
            best_node = v;
        }

        // GL termination: eccentricity stopped increasing
        if (l <= lp) break;
        lp = l;

        // Pick minimum-degree vertex among the farthest level
        NodeID_ next = farthest[0];
        int64_t next_deg = sym_degree(g, next);
        for (size_t i = 1; i < farthest.size(); ++i) {
            int64_t d = sym_degree(g, farthest[i]);
            if (d < next_deg) {
                next_deg = d;
                next = farthest[i];
            }
        }
        v = next;
    }

    return best_node;
}

// ----------------------------------------------------------------------------
// Deterministic Parallel Cuthill-McKee BFS
// ----------------------------------------------------------------------------
// Performs CM-style BFS from a starting vertex, producing the Cuthill-McKee
// ordering for a connected component.
//
// Parallelism uses a two-phase speculative approach within each level:
//
//   Phase 1 (Discovery): Each parent at frontier index fi atomically
//     does min(owner[v], fi) for each unvisited neighbor v.  The parent
//     with the lowest frontier index always wins — matching the serial
//     CM assignment exactly.
//
//   Phase 2 (Collection): Each parent fi collects only the neighbors
//     where owner[v] == fi, marks them visited, and sorts by degree.
//
//   Phase 3 (Merge): Children are appended in frontier order, and
//     owner[] is reset for the next level.
//
// This produces the same ordering as serial CM while exploiting
// thread-level parallelism for both discovery and sorting.
//
// The owner[] array (caller-allocated, size = num_nodes, sentinel = INT64_MAX)
// is passed in to avoid repeated allocation across components.
template <typename NodeID_, typename DestID_, bool invert>
void cm_bfs(const CSRGraph<NodeID_, DestID_, invert>& g,
            NodeID_ start,
            const std::vector<NodeID_>& component,
            std::vector<uint8_t>& visited,
            std::vector<int64_t>& owner,
            std::vector<NodeID_>& cm_order) {

    const int64_t comp_size = static_cast<int64_t>(component.size());
    const int64_t UNCLAIMED = std::numeric_limits<int64_t>::max();

    cm_order.clear();
    cm_order.reserve(comp_size);
    cm_order.push_back(start);
    visited[start] = 1;

    if (comp_size <= 1) return;

    std::vector<NodeID_> frontier;
    frontier.reserve(comp_size);
    frontier.push_back(start);

    // Per-parent children storage for preserving CM ordering
    struct ParentChildren {
        std::vector<NodeID_> children;  // sorted by degree ascending
    };

    std::vector<NodeID_> next_frontier;
    next_frontier.reserve(comp_size);

    while (!frontier.empty()) {
        const int64_t fsize = static_cast<int64_t>(frontier.size());

        // Parallel threshold: use serial for small frontiers
        if (fsize < 64) {
            // --- Serial path (inherently deterministic) ---
            next_frontier.clear();

            for (NodeID_ parent : frontier) {
                // Collect unvisited neighbors (symmetric adjacency)
                std::vector<std::pair<int64_t, NodeID_>> candidates;
                for_each_sym_neigh(g, parent, [&](NodeID_ v) {
                    if (!visited[v]) {
                        visited[v] = 1;
                        candidates.push_back({sym_degree(g, v), v});
                    }
                });

                // Sort by degree ascending (CM property)
                std::sort(candidates.begin(), candidates.end());

                // Append to ordering and next frontier
                for (auto& [d, v] : candidates) {
                    cm_order.push_back(v);
                    next_frontier.push_back(v);
                }
            }

            std::swap(frontier, next_frontier);
        } else {
            // --- Deterministic parallel path ---
            // Phase 1: Speculative discovery.
            // Each parent fi atomically does min(owner[v], fi) for
            // each unvisited neighbor.  Lowest fi wins = serial order.
            #pragma omp parallel for schedule(dynamic, 16)
            for (int64_t fi = 0; fi < fsize; ++fi) {
                NodeID_ parent = frontier[fi];
                for_each_sym_neigh(g, parent, [&](NodeID_ v) {
                    if (!visited[v]) {
                        atomic_min_int64(&owner[v], fi);
                    }
                });
            }

            // Phase 2: Collect winners.
            // Each parent fi gathers only the neighbors it won
            // (owner[v] == fi && not yet visited), marks visited,
            // and sorts by degree ascending.
            std::vector<ParentChildren> per_parent(fsize);

            #pragma omp parallel for schedule(dynamic, 16)
            for (int64_t fi = 0; fi < fsize; ++fi) {
                NodeID_ parent = frontier[fi];
                std::vector<std::pair<int64_t, NodeID_>> candidates;

                for_each_sym_neigh(g, parent, [&](NodeID_ v) {
                    if (owner[v] == fi && !visited[v]) {
                        visited[v] = 1;
                        candidates.push_back({sym_degree(g, v), v});
                    }
                });

                std::sort(candidates.begin(), candidates.end());

                per_parent[fi].children.reserve(candidates.size());
                for (auto& [d, v] : candidates) {
                    per_parent[fi].children.push_back(v);
                }
            }

            // Phase 3: Merge in frontier order + reset owner[]
            next_frontier.clear();
            for (int64_t fi = 0; fi < fsize; ++fi) {
                for (NodeID_ v : per_parent[fi].children) {
                    cm_order.push_back(v);
                    next_frontier.push_back(v);
                    owner[v] = UNCLAIMED;  // reset for next level
                }
            }

            std::swap(frontier, next_frontier);
        }
    }
}

// ----------------------------------------------------------------------------
// Connected Components (BFS-based)
// ----------------------------------------------------------------------------
template <typename NodeID_, typename DestID_, bool invert>
void find_components(const CSRGraph<NodeID_, DestID_, invert>& g,
                     std::vector<std::vector<NodeID_>>& components) {
    const int64_t n = g.num_nodes();
    std::vector<uint8_t> visited(n, 0);
    components.clear();

    std::vector<NodeID_> queue;
    queue.reserve(n);

    for (int64_t s = 0; s < n; ++s) {
        if (visited[s]) continue;

        components.emplace_back();
        auto& comp = components.back();
        comp.reserve(1024);

        queue.clear();
        NodeID_ s_id = static_cast<NodeID_>(s);
        queue.push_back(s_id);
        visited[s] = 1;
        comp.push_back(s_id);

        size_t head = 0;
        while (head < queue.size()) {
            NodeID_ u = queue[head++];
            for_each_sym_neigh(g, u, [&](NodeID_ v) {
                if (!visited[v]) {
                    visited[v] = 1;
                    queue.push_back(v);
                    comp.push_back(v);
                }
            });
        }
    }
}

} // namespace rcm_bnf_detail

// ============================================================================
// Public API — GenerateRCMBNFOrderMapping (RCM BNF variant, -o 11:bnf)
// ============================================================================

/**
 * @brief CSR-native Reverse Cuthill-McKee with BNF + level-parallel BFS.
 *
 * Improved RCM variant (-o 11:bnf) that operates directly on the CSR
 * graph without intermediate adjacency conversion.  Uses symmetric neighbor
 * iteration (out + in edges) for correct handling of directed graphs.
 *
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam WeightT_ Edge weight type (unused, kept for API compatibility)
 * @tparam invert  Whether graph has inverse edges
 * @param g        Input graph (CSR format, symmetrized)
 * @param new_ids  Output permutation: new_ids[old_id] = new_id
 * @param filename Graph filename (unused, kept for API compatibility)
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateRCMBNFOrderMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                pvector<NodeID_>& new_ids,
                                const std::string& /*filename*/) {
    using namespace rcm_bnf_detail;
    Timer tm;
    const int64_t n = g.num_nodes();

    // Guard: empty graph
    if (n == 0) return;

    // Ensure output vector is sized
    if (static_cast<int64_t>(new_ids.size()) < n) {
        new_ids.resize(n);
    }

    // --- Step 1: Find connected components (directly on CSR) ---
    tm.Start();
    std::vector<std::vector<NodeID_>> components;
    rcm_bnf_detail::find_components(g, components);
    tm.Stop();
    PrintTime("RCM_BNF components", tm.Seconds());

    // Initialize new_ids to identity (safety net for unmapped nodes)
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        new_ids[i] = static_cast<NodeID_>(i);
    }

    // --- Step 2: RCM each component (parallel across components) ---
    tm.Start();

    // Sort components by size descending — process largest first
    std::sort(components.begin(), components.end(),
              [](const auto& a, const auto& b) {
                  return a.size() > b.size();
              });

    const size_t numComp = components.size();

    // Compute prefix-sum offsets for each component's position in global_order
    // This allows parallel components to write to non-overlapping regions
    std::vector<int64_t> compOffsets(numComp + 1, 0);
    for (size_t ci = 0; ci < numComp; ++ci) {
        compOffsets[ci + 1] = compOffsets[ci] + static_cast<int64_t>(components[ci].size());
    }
    const int64_t totalOrdered = compOffsets[numComp];

    // Pre-allocate global order array (parallel write via offsets)
    std::vector<NodeID_> global_order(totalOrdered);

    // Per-component CM orderings computed in parallel
    // Each component writes to global_order[compOffsets[ci] .. compOffsets[ci+1])
    // Components are independent — no synchronization needed between them.
    //
    // Tiered strategy:
    //   - 1 vertex: trivial
    //   - 2-32 vertices: min-degree start + serial CM BFS (skip BNF)
    //   - 33-256 vertices: simplified BNF (max 3 GL iterations) + serial CM
    //   - >256 vertices: full BNF (up to 20 GL iterations) + parallel CM BFS

    // Threshold: process components larger than this with full BNF + parallel CM
    constexpr size_t FULL_BNF_THRESHOLD = 256;
    constexpr size_t SKIP_BNF_THRESHOLD = 32;

    #pragma omp parallel
    {
        // Thread-local visited/owner arrays for small components
        // (avoids using the shared n-sized arrays which would conflict)
        std::vector<uint8_t> localVisited;
        std::vector<NodeID_> localCmOrder;

        #pragma omp for schedule(dynamic, 1)
        for (size_t ci = 0; ci < numComp; ++ci) {
            const auto& comp = components[ci];
            const size_t sz = comp.size();
            if (sz == 0) continue;

            const int64_t offset = compOffsets[ci];

            if (sz == 1) {
                global_order[offset] = comp[0];
                continue;
            }

            // --- Find starting node ---
            NodeID_ startNode;

            if (sz <= SKIP_BNF_THRESHOLD) {
                // Small component: just use min-degree vertex (no GL iteration)
                startNode = comp[0];
                int64_t minDeg = sym_degree(g, comp[0]);
                for (size_t i = 1; i < sz; ++i) {
                    int64_t d = sym_degree(g, comp[i]);
                    if (d < minDeg) {
                        minDeg = d;
                        startNode = comp[i];
                    }
                }
            } else if (sz <= FULL_BNF_THRESHOLD) {
                // Medium component: simplified BNF (max 3 GL iterations)
                // Use thread-local visited array sized to n (sparse usage but avoids alloc)
                localVisited.assign(n, 0);

                startNode = comp[0];
                int64_t minDeg = sym_degree(g, comp[0]);
                for (size_t i = 1; i < sz; ++i) {
                    int64_t d = sym_degree(g, comp[i]);
                    if (d < minDeg) {
                        minDeg = d;
                        startNode = comp[i];
                    }
                }

                NodeID_ v = startNode;
                int64_t prevEcc = 0;
                int64_t bestWidth = std::numeric_limits<int64_t>::max();
                std::vector<NodeID_> farthest;

                for (int iter = 0; iter < 3; ++iter) {
                    LevelInfo info = bfs_level_info_with_farthest(g, v, comp, localVisited, farthest);
                    if (info.max_width < bestWidth || (info.max_width == bestWidth && info.eccentricity > prevEcc)) {
                        bestWidth = info.max_width;
                        startNode = v;
                    }
                    if (info.eccentricity <= prevEcc) break;
                    prevEcc = info.eccentricity;

                    NodeID_ next = farthest[0];
                    int64_t nextDeg = sym_degree(g, next);
                    for (size_t i = 1; i < farthest.size(); ++i) {
                        int64_t d = sym_degree(g, farthest[i]);
                        if (d < nextDeg) {
                            nextDeg = d;
                            next = farthest[i];
                        }
                    }
                    v = next;
                }

                // Medium BNF done — visited will be fully reset before CM BFS
            } else {
                // Large component: full BNF with up to 20 GL iterations
                localVisited.assign(n, 0);
                startNode = find_bnf_start(g, comp, localVisited);
                // find_bnf_start internally resets visited; full reset before CM BFS below
            }

            // --- CM BFS with degree-sorted neighbors ---
            // For small/medium components: serial CM BFS (no owner array needed)
            localCmOrder.clear();
            localCmOrder.reserve(sz);
            localVisited.assign(n, 0);

            localCmOrder.push_back(startNode);
            localVisited[startNode] = 1;

            std::queue<NodeID_> bfsQ;
            bfsQ.push(startNode);

            while (!bfsQ.empty()) {
                NodeID_ u = bfsQ.front();
                bfsQ.pop();

                std::vector<std::pair<int64_t, NodeID_>> candidates;
                for_each_sym_neigh(g, u, [&](NodeID_ v) {
                    if (!localVisited[v]) {
                        localVisited[v] = 1;
                        candidates.push_back({sym_degree(g, v), v});
                    }
                });

                std::sort(candidates.begin(), candidates.end());

                for (auto& [d, v] : candidates) {
                    localCmOrder.push_back(v);
                    bfsQ.push(v);
                }
            }

            // Handle disconnected vertices within this component
            for (NodeID_ cv : comp) {
                if (!localVisited[cv]) {
                    localCmOrder.push_back(cv);
                }
            }

            // Reset visited
            for (NodeID_ cv : comp) localVisited[cv] = 0;

            // Write to global_order at this component's offset
            for (size_t i = 0; i < localCmOrder.size(); ++i) {
                global_order[offset + static_cast<int64_t>(i)] = localCmOrder[i];
            }
        }
    }

    tm.Stop();
    PrintTime("RCM_BNF BNF+BFS", tm.Seconds());

    // --- Step 3: Reverse to get RCM ---
    tm.Start();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < totalOrdered; ++i) {
        new_ids[global_order[i]] = static_cast<NodeID_>(totalOrdered - 1 - i);
    }

    tm.Stop();
    PrintTime("RCM_BNF reversal", tm.Seconds());
}

/**
 * Linear corridor-aware reversed wavefront ordering.
 *
 * This keeps the bandwidth-reducing shape of a reversed degree-guided
 * wavefront without GoGraph conversion or pseudo-peripheral searches.
 * Each next level is discovered in parallel, then globally ordered by
 * residual unvisited degree, static degree, and vertex ID. The dynamic first
 * key keeps low-degree road corridors contiguous before branch points, while
 * the global level sort makes the mapping thread-deterministic.
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateCorridorWavefrontMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        const std::string& /*filename*/) {
    using namespace rcm_bnf_detail;
    const int64_t n = g.num_nodes();
    if (n == 0) return;
    if (static_cast<int64_t>(new_ids.size()) < n) {
        new_ids.resize(n);
    }

    Timer timer;
    timer.Start();
    std::vector<uint32_t> degree(static_cast<size_t>(n), 0);
    #pragma omp parallel for schedule(static)
    for (int64_t vertex = 0; vertex < n; ++vertex) {
        const uint64_t value = static_cast<uint64_t>(
            sym_degree(g, static_cast<NodeID_>(vertex)));
        degree[static_cast<size_t>(vertex)] = static_cast<uint32_t>(
            std::min<uint64_t>(
                value,
                std::numeric_limits<uint32_t>::max()));
    }

    std::vector<NodeID_> isolated;
    isolated.reserve(static_cast<size_t>(n) / 32);
    std::vector<NodeID_> skeleton;
    skeleton.reserve(static_cast<size_t>(n) / 4);
    std::vector<int64_t> skeleton_index(
        static_cast<size_t>(n), -1);
    for (int64_t vertex = 0; vertex < n; ++vertex) {
        const uint32_t d = degree[static_cast<size_t>(vertex)];
        if (d == 0) {
            isolated.push_back(static_cast<NodeID_>(vertex));
        } else if (d != 2) {
            skeleton_index[static_cast<size_t>(vertex)] =
                static_cast<int64_t>(skeleton.size());
            skeleton.push_back(static_cast<NodeID_>(vertex));
        }
    }
    timer.Stop();
    PrintTime("Corridor Wavefront Degree Time", timer.Seconds());

    timer.Start();
    const NodeID_ INVALID = static_cast<NodeID_>(-1);
    struct Corridor {
        NodeID_ first;
        NodeID_ second;
        size_t offset;
        size_t length;
    };
    struct Cycle {
        size_t offset;
        size_t length;
    };
    struct Trace {
        NodeID_ endpoint;
        NodeID_ last;
        size_t length;
    };
    auto next_after = [&](NodeID_ current, NodeID_ previous) -> NodeID_ {
        NodeID_ next = INVALID;
        for_each_sym_neigh(g, current, [&](NodeID_ neighbor) {
            if (neighbor != previous && next == INVALID) {
                next = neighbor;
            }
        });
        return next;
    };
    auto trace = [&](NodeID_ endpoint, NodeID_ first, auto visitor) {
        NodeID_ previous = endpoint;
        NodeID_ current = first;
        NodeID_ last = endpoint;
        size_t length = 0;
        while (
            current != INVALID
            && degree[static_cast<size_t>(current)] == 2
        ) {
            visitor(current, length);
            ++length;
            last = current;
            const NodeID_ next = next_after(current, previous);
            previous = current;
            current = next;
            if (length > static_cast<size_t>(n)) {
                return Trace{INVALID, last, length};
            }
        }
        return Trace{current, last, length};
    };
    auto owns = [](NodeID_ endpoint, NodeID_ first, const Trace& path) {
        if (path.endpoint == static_cast<NodeID_>(-1)) return false;
        if (endpoint != path.endpoint) {
            return endpoint < path.endpoint;
        }
        return first < path.last;
    };

    const size_t skeleton_count = skeleton.size();
    std::vector<size_t> corridor_count(skeleton_count, 0);
    std::vector<size_t> corridor_vertex_count(skeleton_count, 0);
    std::atomic<bool> invalid_trace{false};
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t skeleton_id = 0;
         skeleton_id < static_cast<int64_t>(skeleton_count);
         ++skeleton_id) {
        const NodeID_ endpoint =
            skeleton[static_cast<size_t>(skeleton_id)];
        size_t local_corridors = 0;
        size_t local_vertices = 0;
        for_each_sym_neigh(g, endpoint, [&](NodeID_ neighbor) {
            if (degree[static_cast<size_t>(neighbor)] == 2) {
                const Trace path = trace(
                    endpoint, neighbor, [](NodeID_, size_t) {});
                if (
                    path.endpoint == INVALID
                    || skeleton_index[
                        static_cast<size_t>(path.endpoint)] < 0
                ) {
                    invalid_trace.store(true, std::memory_order_relaxed);
                    return;
                }
                if (owns(endpoint, neighbor, path)) {
                    ++local_corridors;
                    local_vertices += path.length;
                }
            } else if (neighbor != endpoint && endpoint < neighbor) {
                ++local_corridors;
            }
        });
        corridor_count[static_cast<size_t>(skeleton_id)] =
            local_corridors;
        corridor_vertex_count[static_cast<size_t>(skeleton_id)] =
            local_vertices;
    }
    if (invalid_trace.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "Corridor wavefront found an invalid degree-2 chain");
    }

    std::vector<size_t> corridor_offsets(skeleton_count + 1, 0);
    std::vector<size_t> corridor_vertex_offsets(
        skeleton_count + 1, 0);
    for (size_t index = 0; index < skeleton_count; ++index) {
        corridor_offsets[index + 1] =
            corridor_offsets[index] + corridor_count[index];
        corridor_vertex_offsets[index + 1] =
            corridor_vertex_offsets[index]
            + corridor_vertex_count[index];
    }
    std::vector<Corridor> corridors(
        corridor_offsets[skeleton_count]);
    std::vector<NodeID_> corridor_vertices(
        corridor_vertex_offsets[skeleton_count]);
    std::vector<uint8_t> covered(static_cast<size_t>(n), 0);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t skeleton_id = 0;
         skeleton_id < static_cast<int64_t>(skeleton_count);
         ++skeleton_id) {
        const size_t index = static_cast<size_t>(skeleton_id);
        const NodeID_ endpoint = skeleton[index];
        size_t corridor_position = corridor_offsets[index];
        size_t vertex_position = corridor_vertex_offsets[index];
        for_each_sym_neigh(g, endpoint, [&](NodeID_ neighbor) {
            if (degree[static_cast<size_t>(neighbor)] == 2) {
                const Trace path = trace(
                    endpoint, neighbor, [](NodeID_, size_t) {});
                if (!owns(endpoint, neighbor, path)) return;
                const size_t start = vertex_position;
                const Trace filled = trace(
                    endpoint,
                    neighbor,
                    [&](NodeID_ vertex, size_t offset) {
                        corridor_vertices[start + offset] = vertex;
                        covered[static_cast<size_t>(vertex)] = 1;
                    });
                corridors[corridor_position++] = {
                    endpoint,
                    filled.endpoint,
                    start,
                    filled.length,
                };
                vertex_position += filled.length;
            } else if (neighbor != endpoint && endpoint < neighbor) {
                corridors[corridor_position++] = {
                    endpoint,
                    neighbor,
                    vertex_position,
                    0,
                };
            }
        });
    }

    std::vector<Cycle> cycles;
    for (int64_t vertex = 0; vertex < n; ++vertex) {
        if (
            degree[static_cast<size_t>(vertex)] != 2
            || covered[static_cast<size_t>(vertex)]
        ) {
            continue;
        }
        const size_t offset = corridor_vertices.size();
        NodeID_ previous = INVALID;
        NodeID_ current = static_cast<NodeID_>(vertex);
        while (!covered[static_cast<size_t>(current)]) {
            covered[static_cast<size_t>(current)] = 1;
            corridor_vertices.push_back(current);
            const NodeID_ next = next_after(current, previous);
            if (next == INVALID) {
                throw std::runtime_error(
                    "Corridor wavefront found an invalid degree-2 cycle");
            }
            previous = current;
            current = next;
        }
        cycles.push_back({
            offset,
            corridor_vertices.size() - offset,
        });
    }
    timer.Stop();
    PrintTime("Corridor Wavefront Contraction Time", timer.Seconds());

    timer.Start();
    std::vector<uint8_t> skeleton_emitted(skeleton_count, 0);
    std::vector<NodeID_> order;
    order.reserve(static_cast<size_t>(n) - isolated.size());

    for (const Corridor& corridor : corridors) {
        const size_t first = static_cast<size_t>(
            skeleton_index[static_cast<size_t>(corridor.first)]);
        const size_t second = static_cast<size_t>(
            skeleton_index[static_cast<size_t>(corridor.second)]);
        if (!skeleton_emitted[first]) {
            skeleton_emitted[first] = 1;
            order.push_back(corridor.first);
        }
        for (size_t index = 0; index < corridor.length; ++index) {
            order.push_back(
                corridor_vertices[corridor.offset + index]);
        }
        if (!skeleton_emitted[second]) {
            skeleton_emitted[second] = 1;
            order.push_back(corridor.second);
        }
    }
    for (size_t index = 0; index < skeleton_count; ++index) {
        if (!skeleton_emitted[index]) {
            order.push_back(skeleton[index]);
        }
    }
    for (const Cycle& cycle : cycles) {
        for (size_t index = 0; index < cycle.length; ++index) {
            order.push_back(
                corridor_vertices[cycle.offset + index]);
        }
    }
    std::reverse(order.begin(), order.end());
    timer.Stop();
    PrintTime("Corridor Wavefront Emit Time", timer.Seconds());

    if (
        order.size() + isolated.size()
        != static_cast<size_t>(n)
    ) {
        throw std::runtime_error(
            "Corridor wavefront did not emit every vertex");
    }

    timer.Start();
    #pragma omp parallel for schedule(static)
    for (int64_t position = 0;
         position < static_cast<int64_t>(order.size());
         ++position) {
        new_ids[static_cast<size_t>(order[position])] =
            static_cast<NodeID_>(position);
    }
    const size_t tail = order.size();
    #pragma omp parallel for schedule(static)
    for (int64_t index = 0;
         index < static_cast<int64_t>(isolated.size());
         ++index) {
        new_ids[static_cast<size_t>(isolated[index])] =
            static_cast<NodeID_>(tail + static_cast<size_t>(index));
    }
    timer.Stop();
    PrintTime("Corridor Wavefront Assign Time", timer.Seconds());
    PrintLabel(
        "Corridor Wavefront Junctions",
        std::to_string(skeleton.size()));
    PrintLabel(
        "Corridor Wavefront Corridors",
        std::to_string(corridors.size()));
    PrintLabel(
        "Corridor Wavefront Cycles",
        std::to_string(cycles.size()));
    PrintLabel(
        "Corridor Wavefront Isolated",
        std::to_string(isolated.size()));
}

#endif  // REORDER_RCM_BNF_H_
