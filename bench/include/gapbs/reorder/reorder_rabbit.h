/**
 * @file reorder_rabbit.h
 * @brief RabbitOrder Algorithm Implementation (Algorithm ID: 8)
 * 
 * RabbitOrder is a community-aware reordering algorithm that:
 *   1. Detects communities using hierarchical agglomeration
 *   2. Orders vertices within communities to maximize locality
 *   3. Places related communities adjacent in memory
 * 
 * Paper: "Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis"
 *        by J. Arai et al., IPDPS 2016
 * 
 * Best for: Social networks, web graphs, and graphs with community structure
 * 
 * Complexity: O(n log n + m) approximately
 * 
 * @note Requires RABBIT_ENABLE macro and Boost library for full functionality.
 *       Falls back to original ordering if RABBIT_ENABLE is not defined.
 */

#ifndef REORDER_RABBIT_H_
#define REORDER_RABBIT_H_

#include <vector>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <omp.h>

#include "../pvector.h"
#include "../timer.h"
#include "../graph.h"

#ifdef RABBIT_ENABLE
#include "../../rabbit/edge_list.hpp"
#include "../../rabbit/rabbit_order.hpp"
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm.hpp>
#endif

// Include the full graph header instead of forward declaration
// CSRGraph has 6 template parameters which makes forward declaration complex
#include "../graph.h"

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

#ifdef RABBIT_ENABLE
/**
 * @brief Adjacency list representation for RabbitOrder
 * 
 * Vector of vectors where each inner vector contains pairs of
 * (neighbor_id, edge_weight) for efficient community detection.
 */
using adjacency_list = std::vector<std::vector<std::pair<rabbit_order::vint, float>>>;
#endif

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

#ifdef RABBIT_ENABLE

/**
 * @brief Count unique elements in a range
 * 
 * @tparam InputIt Input iterator type
 * @param f First iterator
 * @param l Last iterator
 * @return Number of unique elements
 */
template <typename InputIt>
typename std::iterator_traits<InputIt>::difference_type
count_uniq_rabbit(const InputIt f, const InputIt l) {
    std::vector<typename std::iterator_traits<InputIt>::value_type> ys(f, l);
    return boost::size(boost::unique(boost::sort(ys)));
}

/**
 * @brief Compute modularity of a community assignment
 * 
 * Modularity measures the quality of a community partition.
 * Higher values indicate better community structure.
 * 
 * Q = Σ[e_ii - a_i²] where:
 *   - e_ii = fraction of edges within community i
 *   - a_i = fraction of edges incident to community i
 * 
 * @param adj Adjacency list representation
 * @param coms Community assignment for each vertex
 * @return Modularity value in range [-0.5, 1.0]
 */
inline double compute_modularity_rabbit(const adjacency_list& adj,
                                        const rabbit_order::vint* const coms) {
    const rabbit_order::vint n = static_cast<rabbit_order::vint>(adj.size());
    double m2 = 0.0;  // Total weight of bidirectional edges
    
    // Find max community ID for array sizing
    rabbit_order::vint max_com = 0;
    #pragma omp parallel for reduction(max : max_com)
    for (rabbit_order::vint v = 0; v < n; ++v) {
        if (coms[v] > max_com) max_com = coms[v];
    }
    const size_t com_array_size = static_cast<size_t>(max_com) + 1;
    
    // Use flat arrays for better cache performance
    std::vector<double> degs_all(com_array_size, 0.0);
    std::vector<double> degs_loop(com_array_size, 0.0);
    
    // Thread-local arrays to avoid critical sections
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> thread_degs_all(num_threads,
        std::vector<double>(com_array_size, 0.0));
    std::vector<std::vector<double>> thread_degs_loop(num_threads,
        std::vector<double>(com_array_size, 0.0));
    
    #pragma omp parallel reduction(+ : m2)
    {
        int tid = omp_get_thread_num();
        auto& local_all = thread_degs_all[tid];
        auto& local_loop = thread_degs_loop[tid];
        
        #pragma omp for schedule(dynamic, 1024)
        for (rabbit_order::vint v = 0; v < n; ++v) {
            const rabbit_order::vint c = coms[v];
            for (const auto& e : adj[v]) {
                m2 += e.second;
                local_all[c] += e.second;
                if (coms[e.first] == c)
                    local_loop[c] += e.second;
            }
        }
    }
    
    // Parallel reduction of thread-local arrays
    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < com_array_size; ++c) {
        for (int t = 0; t < num_threads; ++t) {
            degs_all[c] += thread_degs_all[t][c];
            degs_loop[c] += thread_degs_loop[t][c];
        }
    }
    
    // Compute modularity
    double q = 0.0;
    #pragma omp parallel for reduction(+ : q) schedule(static, 1024)
    for (size_t c = 0; c < com_array_size; ++c) {
        const double all = degs_all[c];
        const double loop = degs_loop[c];
        if (all > 0.0) {
            q += loop / m2 - (all / m2) * (all / m2);
        }
    }
    
    return q;
}

/**
 * @brief Detect communities and print results
 * 
 * Used for analysis and debugging purposes.
 * 
 * @param adj Adjacency list to process
 */
inline void detect_community_rabbit(adjacency_list adj) {
    auto _adj = adj;  // Copy for modularity computation
    
    const double tstart = rabbit_order::now_sec();
    auto g = rabbit_order::aggregate(std::move(_adj));
    const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
    
    #pragma omp parallel for
    for (rabbit_order::vint v = 0; v < g.n(); ++v)
        c[v] = rabbit_order::trace_com(v, &g);
    
    PrintTime("Community Time", rabbit_order::now_sec() - tstart);
    
    const double q = compute_modularity_rabbit(adj, c.get());
    std::cerr << "Modularity: " << q << std::endl;
}

/**
 * @brief Generate reordering and print permutation to stdout
 * 
 * @param adj Adjacency list to process
 */
inline void reorder_rabbit(adjacency_list adj) {
    const double tstart = rabbit_order::now_sec();
    const auto g = rabbit_order::aggregate(std::move(adj));
    const auto p = rabbit_order::compute_perm(g);
    PrintTime("Permutation generation Time", rabbit_order::now_sec() - tstart);
    
    std::copy(&p[0], &p[g.n()],
              std::ostream_iterator<rabbit_order::vint>(std::cout, "\n"));
}

/**
 * @brief Internal reordering with full statistics
 * 
 * Generates permutation and computes modularity/community stats.
 * 
 * @tparam NodeID_ Node identifier type
 * @param adj Adjacency list to process
 * @param new_ids Output permutation array
 */
template <typename NodeID_>
void reorder_internal_rabbit(adjacency_list adj, pvector<NodeID_>& new_ids) {
    auto _adj = adj;  // Copy for modularity computation
    
    const double tstart = rabbit_order::now_sec();
    auto g = rabbit_order::aggregate(std::move(_adj));
    const auto p = rabbit_order::compute_perm(g);
    const double tend = rabbit_order::now_sec();
    
    // Compute community assignments for statistics
    const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
    #pragma omp parallel for
    for (rabbit_order::vint v = 0; v < g.n(); ++v)
        c[v] = rabbit_order::trace_com(v, &g);
    
    const double q = compute_modularity_rabbit(adj, c.get());
    
    // Count unique communities
    std::unordered_set<rabbit_order::vint> unique_comms;
    for (rabbit_order::vint v = 0; v < g.n(); ++v)
        unique_comms.insert(c[v]);
    const size_t num_communities = unique_comms.size();
    
    PrintTime("RabbitOrder Communities", static_cast<double>(num_communities));
    PrintTime("RabbitOrder Modularity", q);
    PrintTime("RabbitOrder Map Time", tend - tstart);
    
    // Ensure output is large enough
    if (new_ids.size() < g.n())
        new_ids.resize(g.n());
    
    #pragma omp parallel for
    for (size_t i = 0; i < g.n(); ++i) {
        new_ids[i] = static_cast<NodeID_>(p[i]);
    }
}

/**
 * @brief Internal reordering for sub-graphs (minimal output)
 * 
 * Optimized version without modularity computation for use in
 * hierarchical/sub-graph reordering.
 * 
 * @tparam NodeID_ Node identifier type
 * @param adj Adjacency list to process
 * @param new_ids Output permutation array
 */
template <typename NodeID_>
void reorder_internal_single_rabbit(adjacency_list adj, pvector<NodeID_>& new_ids) {
    auto _adj = adj;
    
    const double tstart = rabbit_order::now_sec();
    auto g = rabbit_order::aggregate(std::move(_adj));
    const auto p = rabbit_order::compute_perm(g);
    const double tend = rabbit_order::now_sec();
    
    // Compute community assignments (used internally)
    const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
    #pragma omp parallel for
    for (rabbit_order::vint v = 0; v < g.n(); ++v)
        c[v] = rabbit_order::trace_com(v, &g);
    
    PrintTime("Sub-RabbitOrder Map Time", tend - tstart);
    
    if (new_ids.size() < g.n())
        new_ids.resize(g.n());
    
    #pragma omp parallel for
    for (size_t i = 0; i < g.n(); ++i) {
        new_ids[i] = static_cast<NodeID_>(p[i]);
    }
}

#endif  // RABBIT_ENABLE

// ============================================================================
// GRAPH CONVERSION FUNCTIONS
// ============================================================================

#ifdef RABBIT_ENABLE

/**
 * @brief Convert CSR graph to RabbitOrder adjacency list format
 * 
 * @tparam NodeID_ Node identifier type
 * @tparam DestID_ Destination identifier type
 * @tparam WeightT_ Edge weight type
 * @tparam invert Whether graph stores incoming edges
 * @param g Input CSR graph
 * @return Adjacency list for RabbitOrder processing
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
adjacency_list readRabbitOrderGraphCSR(const CSRGraph<NodeID_, DestID_, invert>& g) {
    using boost::adaptors::transformed;
    
    const int64_t num_nodes = g.num_nodes();
    adjacency_list adj(num_nodes);
    
    #pragma omp parallel for schedule(dynamic, 1024)
    for (NodeID_ i = 0; i < num_nodes; ++i) {
        adj[i].reserve(g.out_degree(i));
        for (DestID_ neighbor : g.out_neigh(i)) {
            if (g.is_weighted()) {
                NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                float weight = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                adj[i].emplace_back(dest, weight);
            } else {
                adj[i].emplace_back(static_cast<rabbit_order::vint>(neighbor), 1.0f);
            }
        }
    }
    
    return adj;
}

/**
 * @brief Convert edge list to RabbitOrder adjacency list format
 * 
 * @param edges Input edge list (src, dest, weight tuples)
 * @return Adjacency list for RabbitOrder processing
 */
inline adjacency_list readRabbitOrderAdjacencylist(
    const std::vector<edge_list::edge>& edges) {
    
    // Find number of vertices
    rabbit_order::vint max_v = 0;
    #pragma omp parallel for reduction(max : max_v)
    for (size_t i = 0; i < edges.size(); ++i) {
        rabbit_order::vint src = std::get<0>(edges[i]);
        rabbit_order::vint dst = std::get<1>(edges[i]);
        if (src > max_v) max_v = src;
        if (dst > max_v) max_v = dst;
    }
    
    const size_t n = static_cast<size_t>(max_v) + 1;
    adjacency_list adj(n);
    
    // Count degrees first for efficient allocation
    std::vector<size_t> degrees(n, 0);
    for (const auto& e : edges) {
        ++degrees[std::get<0>(e)];
    }
    
    // Reserve space
    for (size_t i = 0; i < n; ++i) {
        adj[i].reserve(degrees[i]);
    }
    
    // Fill adjacency list
    for (const auto& e : edges) {
        adj[std::get<0>(e)].emplace_back(std::get<1>(e), std::get<2>(e));
    }
    
    return adj;
}

#endif  // RABBIT_ENABLE

// ============================================================================
// MAIN API FUNCTIONS
// ============================================================================

/**
 * @brief Generate RabbitOrder vertex permutation
 * 
 * RabbitOrder combines community detection with cache-aware ordering.
 * It detects communities using hierarchical agglomeration, then orders
 * vertices to maximize spatial locality within and across communities.
 * 
 * Algorithm steps:
 *   1. Build hierarchical community structure via graph aggregation
 *   2. Compute vertex permutation respecting community boundaries
 *   3. Order vertices within communities for cache efficiency
 * 
 * @tparam NodeID_ Node identifier type
 * @tparam DestID_ Destination identifier type
 * @tparam WeightT_ Edge weight type
 * @tparam invert Whether graph stores incoming edges
 * @param g Input CSR graph
 * @param new_ids Output: new_ids[old_id] = new_id mapping
 * 
 * @note Falls back to original ordering if RABBIT_ENABLE is not defined
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateRabbitOrderMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                pvector<NodeID_>& new_ids) {
#ifdef RABBIT_ENABLE
    auto adj = readRabbitOrderGraphCSR<NodeID_, DestID_, WeightT_, invert>(g);
    reorder_internal_rabbit<NodeID_>(std::move(adj), new_ids);
#else
    // Fallback to original ordering
    Timer t;
    t.Start();
    const int64_t num_nodes = g.num_nodes();
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; i++) {
        new_ids[i] = static_cast<NodeID_>(i);
    }
    t.Stop();
    PrintTime("RabbitOrder (disabled) Map Time", t.Seconds());
#endif
}

#ifdef RABBIT_ENABLE
/**
 * @brief Generate RabbitOrder mapping from edge list
 * 
 * Alternative entry point when graph is in edge list format.
 * 
 * @tparam NodeID_ Node identifier type
 * @param edges Input edge list
 * @param new_ids Output permutation
 */
template <typename NodeID_>
void GenerateRabbitOrderMappingEdgelist(const std::vector<edge_list::edge>& edges,
                                        pvector<NodeID_>& new_ids) {
    auto adj = readRabbitOrderAdjacencylist(edges);
    reorder_internal_single_rabbit<NodeID_>(std::move(adj), new_ids);
}

/**
 * @brief Compute modularity of an edge list
 * 
 * @tparam NodeID_ Node identifier type
 * @tparam WeightT_ Edge weight type
 * @param edgesList Input edge list
 * @param is_weighted Whether edges have weights
 * @return Modularity value
 */
template <typename NodeID_, typename WeightT_>
double GenerateRabbitModularityEdgelist(
    const std::vector<std::tuple<NodeID_, NodeID_, WeightT_>>& edgesList,
    bool is_weighted) {
    
    std::vector<edge_list::edge> edges(edgesList.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < edges.size(); ++i) {
        rabbit_order::vint src = std::get<0>(edgesList[i]);
        rabbit_order::vint dest = std::get<1>(edgesList[i]);
        float weight = is_weighted ? static_cast<float>(std::get<2>(edgesList[i])) : 1.0f;
        edges[i] = std::make_tuple(src, dest, weight);
    }
    
    auto adj = readRabbitOrderAdjacencylist(edges);
    auto _adj = adj;
    
    auto g = rabbit_order::aggregate(std::move(_adj));
    const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
    
    #pragma omp parallel for
    for (rabbit_order::vint v = 0; v < g.n(); ++v)
        c[v] = rabbit_order::trace_com(v, &g);
    
    return compute_modularity_rabbit(adj, c.get());
}
#endif  // RABBIT_ENABLE

#endif  // REORDER_RABBIT_H_
