// ============================================================================
// GraphBrew - Basic Reordering Algorithms
// ============================================================================
// This header implements the basic/fundamental reordering algorithms:
//   - ORIGINAL (0): Keep original vertex ordering
//   - RANDOM (1):   Random permutation of vertices  
//   - SORT (2):     Sort vertices by degree
//
// These serve as baselines for comparison with more sophisticated algorithms.
//
// Author: GraphBrew Team
// License: See LICENSE.txt
// ============================================================================

#ifndef REORDER_BASIC_H_
#define REORDER_BASIC_H_

#include "reorder_types.h"

namespace graphbrew::random_detail {

inline uint64_t SplitMix64(uint64_t& state) {
    uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30))
        * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27))
        * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

inline uint64_t UniformBounded(uint64_t& state, uint64_t bound) {
    const uint64_t threshold = -bound % bound;
    while (true) {
        const uint64_t value = SplitMix64(state);
        if (value >= threshold) return value % bound;
    }
}

}  // namespace graphbrew::random_detail

// ============================================================================
// ORIGINAL ORDERING (Algorithm 0)
// ============================================================================

/**
 * @brief Keep the original vertex ordering (identity permutation)
 * 
 * This is the baseline ordering where each vertex keeps its original ID.
 * Use this to measure the performance impact of reordering algorithms.
 * 
 * Complexity: O(n) - simple parallel initialization
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type  
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph (CSR format)
 * @param new_ids Output permutation: new_ids[old_id] = new_id
 * 
 * @example
 *   pvector<NodeID> new_ids(g.num_nodes());
 *   GenerateOriginalMapping(g, new_ids);
 *   // new_ids[i] == i for all i
 */
template <typename NodeID_, typename DestID_, bool invert>
void GenerateOriginalMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                             pvector<NodeID_>& new_ids) {
    const int64_t num_nodes = g.num_nodes();
    
    Timer t;
    t.Start();
    
    // Identity mapping: each vertex keeps its original ID
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++) {
        new_ids[i] = static_cast<NodeID_>(i);
    }
    
    t.Stop();
    PrintTime("Original Map Time", t.Seconds());
}

// ============================================================================
// RANDOM ORDERING (Algorithm 1)
// ============================================================================

/**
 * @brief Generate a random permutation of vertices
 * 
 * Shuffles vertices randomly. Uses a fixed seed (0) for reproducibility.
 * This is a controlled shuffled labeling, not a worst-case claim.
 * 
 * Complexity: O(n) - parallel shuffle with granularity-based slicing
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph (CSR format)
 * @param new_ids Output permutation: new_ids[old_id] = new_id
 * 
 * @note Uses a specified SplitMix64/Fisher-Yates permutation.
 * @note Reproducible across OpenMP thread counts: seed=0.
 */
template <typename NodeID_, typename DestID_, bool invert>
void GenerateRandomMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                           pvector<NodeID_>& new_ids) {
    Timer t;
    t.Start();
    
    const int64_t num_nodes = g.num_nodes();
    pvector<NodeID_> permutation(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        permutation[i] = static_cast<NodeID_>(i);
    }

    uint64_t state = 0;
    for (uint64_t remaining = static_cast<uint64_t>(num_nodes);
         remaining > 1;
         --remaining) {
        uint64_t selected =
            graphbrew::random_detail::UniformBounded(state, remaining);
        std::swap(
            permutation[remaining - 1],
            permutation[selected]);
    }

    #pragma omp parallel for
    for (int64_t old_id = 0; old_id < num_nodes; ++old_id) {
        new_ids[old_id] = permutation[old_id];
    }

    PrintLabel("Random Seed", "0");
    t.Stop();
    PrintTime("Random Map Time", t.Seconds());
}

/**
 * @brief Alternative random mapping using compare-and-swap
 * 
 * A slower but lock-free implementation. Each vertex claims a random
 * position using CAS operations.
 * 
 * @deprecated Use GenerateRandomMapping instead (faster)
 */
template <typename NodeID_, typename DestID_, bool invert>
void GenerateRandomMapping_v2(const CSRGraph<NodeID_, DestID_, invert>& g,
                              pvector<NodeID_>& new_ids) {
    Timer t;
    t.Start();
    
    // Claimed positions tracking
    pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);
    
    // Each vertex tries to claim a random position using per-thread PRNG
    // (std::rand() is not thread-safe — concurrent calls cause data races)
    #pragma omp parallel
    {
        std::mt19937 gen(42 + omp_get_thread_num());
        std::uniform_int_distribution<NodeID_> dis(0, g.num_nodes() - 1);
        #pragma omp for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v) {
            while (true) {
                NodeID_ randID = dis(gen);
                if (claimedVtxs[randID] != 1) {
                    if (compare_and_swap(claimedVtxs[randID], NodeID_(0), NodeID_(1))) {
                        new_ids[v] = randID;
                        break;
                    }
                }
            }
        }
    }
    
    // Verify all vertices got an ID
    #pragma omp parallel for
    for (NodeID_ v = 0; v < g.num_nodes(); ++v) {
        assert(new_ids[v] != static_cast<NodeID_>(-1));
    }
    
    t.Stop();
    PrintTime("Random Map Time", t.Seconds());
}

// ============================================================================
// SORT ORDERING (Algorithm 2)
// ============================================================================

/**
 * @brief Sort vertices by degree (descending by default)
 * 
 * Orders vertices so that high-degree vertices come first. This improves
 * cache utilization because frequently accessed vertices (high degree)
 * are placed at the beginning of memory.
 * 
 * Complexity: O(n log n) - parallel sort
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph (CSR format)
 * @param new_ids Output permutation: new_ids[old_id] = new_id
 * @param useOutdeg If true, sort by out-degree; else by in-degree
 * @param lesser If true, sort ascending (low degree first); else descending
 * 
 * @example
 *   // Sort by out-degree, high degree first (default)
 *   GenerateSortMapping(g, new_ids, true, false);
 *   
 *   // Sort by in-degree, low degree first  
 *   GenerateSortMapping(g, new_ids, false, true);
 */
template <typename NodeID_, typename DestID_, bool invert>
void GenerateSortMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                         pvector<NodeID_>& new_ids, 
                         bool useOutdeg,
                         bool lesser = false) {
    
    using DegreeNodePair = std::pair<int64_t, NodeID_>;
    
    Timer t;
    t.Start();
    
    const int64_t num_nodes = g.num_nodes();
    pvector<DegreeNodePair> degree_id_pairs(num_nodes);
    
    // Collect (degree, vertex_id) pairs
    if (useOutdeg) {
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            int64_t out_degree_v = g.out_degree(v);
            degree_id_pairs[v] = std::make_pair(out_degree_v, v);
        }
    } else {
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            int64_t in_degree_v = g.in_degree(v);
            degree_id_pairs[v] = std::make_pair(in_degree_v, v);
        }
    }
    
    // Sort by degree (descending by default, ascending if lesser=true)
    auto comparator = [lesser](const DegreeNodePair& a, const DegreeNodePair& b) {
        return lesser ? (a.first < b.first) : (a.first > b.first);
    };
    
    __gnu_parallel::stable_sort(degree_id_pairs.begin(), 
                                 degree_id_pairs.end(), 
                                 comparator);
    
    // Build permutation: vertex at position n gets new ID n
    #pragma omp parallel for
    for (int64_t n = 0; n < num_nodes; ++n) {
        new_ids[degree_id_pairs[n].second] = n;
    }
    
    t.Stop();
    PrintTime("Sort Map Time", t.Seconds());
}

/**
 * @brief Sort mapping variant for RabbitOrder preprocessing
 * 
 * Sorts by (out_degree, in_degree) with special handling for
 * isolated vertices (degree 0). Isolated vertices are placed at the end.
 * 
 * This preprocessing step improves RabbitOrder's community detection
 * convergence by grouping similar-degree vertices.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph (CSR format)
 * @param new_ids Output permutation: new_ids[old_id] = new_id
 * @param useOutdeg Unused (always uses both degrees)
 * @param lesser Unused (always descending)
 */
template <typename NodeID_, typename DestID_, bool invert>
void GenerateSortMappingRabbit(const CSRGraph<NodeID_, DestID_, invert>& g,
                               pvector<NodeID_>& new_ids,
                               bool useOutdeg,
                               bool lesser = false) {
    
    using DegreeTuple = std::tuple<int64_t, int64_t, NodeID_>;
    
    Timer t;
    t.Start();
    
    const int64_t num_nodes = g.num_nodes();
    pvector<DegreeTuple> degree_id_pairs(num_nodes);
    
    // Collect (out_degree, in_degree, vertex_id) tuples
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        int64_t out_degree_v = g.out_degree(v);
        int64_t in_degree_v = g.in_degree(v);
        degree_id_pairs[v] = std::make_tuple(out_degree_v, in_degree_v, v);
    }
    
    // Custom comparator: isolated vertices go to end
    auto comparator = [](const DegreeTuple& a, const DegreeTuple& b) {
        int64_t out_a = std::get<0>(a);
        int64_t out_b = std::get<0>(b);
        int64_t in_a = std::get<1>(a);
        int64_t in_b = std::get<1>(b);
        
        // Keep isolated vertices (degree=0) at the end
        if (out_a == 0 && in_a == 0) return false;
        if (out_b == 0 && in_b == 0) return true;
        
        // Primary sort: out-degree descending
        if (out_a != out_b) return out_a > out_b;
        
        // Secondary sort: in-degree descending
        return in_a > in_b;
    };
    
    __gnu_parallel::stable_sort(degree_id_pairs.begin(), 
                                 degree_id_pairs.end(), 
                                 comparator);
    
    // Build permutation
    #pragma omp parallel for
    for (int64_t n = 0; n < num_nodes; ++n) {
        new_ids[std::get<2>(degree_id_pairs[n])] = n;
    }
    
    t.Stop();
    PrintTime("Sort Map Time", t.Seconds());
}

#endif  // REORDER_BASIC_H_
