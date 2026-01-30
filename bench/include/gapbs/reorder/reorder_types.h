// ============================================================================
// GraphBrew - Reordering Types and Common Utilities
// ============================================================================
// This header defines common types, structures, and utility functions used
// across all reordering algorithm implementations.
//
// Architecture:
//   - reorder_types.h     : Types, forward declarations, utilities (this file)
//   - reorder_basic.h     : Basic algorithms (Original, Random, Sort)
//   - reorder_hub.h       : Hub-based algorithms (HubSort, HubCluster, DBG)
//   - reorder_rabbit.h    : RabbitOrder (Louvain-based community detection)
//   - reorder_classic.h   : Classic algorithms (GOrder, COrder, RCM)
//   - reorder_gve_leiden.h: GVE-Leiden core algorithm implementation
//   - reorder_leiden.h    : Leiden-based orderings (LeidenOrder, LeidenCSR, etc.)
//   - reorder_graphbrew.h : GraphBrew hybrid and Adaptive algorithms
//   - reorder.h           : Main dispatcher that includes all headers
//
// Author: GraphBrew Team
// License: See LICENSE.txt
// ============================================================================

#ifndef REORDER_TYPES_H_
#define REORDER_TYPES_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

// Include the full graph.h instead of forward declaring CSRGraph
// This ensures template parameters match
#include "../graph.h"
#include "../pvector.h"
#include "../timer.h"
#include "../util.h"

// ============================================================================
// TYPE ALIASES
// ============================================================================

// Default node weight type for weighted graphs
#ifndef TYPE
#define TYPE float
#endif

// Community ID type - uint32_t supports up to 4B communities
using CommunityID = uint32_t;

// ============================================================================
// RESULT STRUCTURES
// ============================================================================

/**
 * @brief Result structure for GVE-Leiden community detection
 * 
 * Contains the final community assignments, quality metrics, and
 * iteration statistics from the Leiden algorithm execution.
 * 
 * @tparam K Community ID type (default: uint32_t)
 */
template <typename K = uint32_t>
struct GVELeidenResult {
    // ---------- Community Assignment ----------
    /** Final community assignment: final_community[v] = community ID of vertex v */
    std::vector<K> final_community;
    
    /** Community assignments per pass for hierarchical structure */
    std::vector<std::vector<K>> community_per_pass;
    
    // ---------- Quality Metrics ----------
    /** Final modularity score (higher = better community structure) */
    double modularity = 0.0;
    
    // ---------- Iteration Statistics ----------
    /** Total local-moving iterations across all passes */
    int total_iterations = 0;
    
    /** Number of aggregation passes performed */
    int total_passes = 0;
};

/**
 * @brief Dendrogram result with explicit tree structure
 * 
 * Uses parent/child/sibling pointers for efficient tree traversal.
 * This is more memory-efficient than storing community_per_pass.
 * 
 * @tparam K Community ID type (default: uint32_t)
 */
template <typename K = uint32_t>
struct GVEDendroResult {
    std::vector<K> final_community;        ///< Final community assignment
    std::vector<int64_t> parent;           ///< Dendrogram parent (-1 = root)
    std::vector<int64_t> first_child;      ///< First child (-1 = leaf)
    std::vector<int64_t> sibling;          ///< Next sibling at same level (-1 = last)
    std::vector<int64_t> subtree_size;     ///< Size of subtree rooted here
    std::vector<double> weight;            ///< Node weight (degree)
    std::vector<int64_t> roots;            ///< Root nodes (top-level communities)
    int total_iterations = 0;
    int total_passes = 0;
    double modularity = 0.0;
};

/**
 * @brief Atomic dendrogram result for lock-free parallel building
 * 
 * Uses atomic operations for concurrent dendrogram construction.
 * Based on RabbitOrder's lock-free merge approach.
 * 
 * @tparam K Community ID type (default: uint32_t)
 */
template <typename K = uint32_t>
struct GVEAtomicDendroResult {
    std::vector<K> final_community;
    std::unique_ptr<std::atomic<int64_t>[]> parent;
    std::unique_ptr<std::atomic<int64_t>[]> first_child;
    std::unique_ptr<std::atomic<int64_t>[]> sibling;
    std::unique_ptr<std::atomic<int64_t>[]> subtree_size;
    std::vector<double> weight;
    std::vector<int64_t> roots;
    int64_t num_nodes = 0;
    int total_iterations = 0;
    int total_passes = 0;
    double modularity = 0.0;
    
    GVEAtomicDendroResult() = default;
    
    /** Initialize atomic dendrogram for n vertices */
    void init(int64_t n, const std::vector<double>& vtot) {
        num_nodes = n;
        final_community.resize(n);
        parent = std::make_unique<std::atomic<int64_t>[]>(n);
        first_child = std::make_unique<std::atomic<int64_t>[]>(n);
        sibling = std::make_unique<std::atomic<int64_t>[]>(n);
        subtree_size = std::make_unique<std::atomic<int64_t>[]>(n);
        weight.resize(n);
        
        #pragma omp parallel for
        for (int64_t v = 0; v < n; ++v) {
            final_community[v] = static_cast<K>(v);
            parent[v].store(-1, std::memory_order_relaxed);
            first_child[v].store(-1, std::memory_order_relaxed);
            sibling[v].store(-1, std::memory_order_relaxed);
            subtree_size[v].store(1, std::memory_order_relaxed);
            weight[v] = vtot[v];
        }
        
        total_iterations = 0;
        total_passes = 0;
        modularity = 0.0;
    }
    
    /** Convert atomic result to non-atomic for traversal */
    GVEDendroResult<K> toNonAtomic() const {
        GVEDendroResult<K> result;
        
        result.final_community = final_community;
        result.parent.resize(num_nodes);
        result.first_child.resize(num_nodes);
        result.sibling.resize(num_nodes);
        result.subtree_size.resize(num_nodes);
        result.weight = weight;
        result.total_iterations = total_iterations;
        result.total_passes = total_passes;
        result.modularity = modularity;
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            result.parent[v] = parent[v].load(std::memory_order_relaxed);
            result.first_child[v] = first_child[v].load(std::memory_order_relaxed);
            result.sibling[v] = sibling[v].load(std::memory_order_relaxed);
            result.subtree_size[v] = subtree_size[v].load(std::memory_order_relaxed);
        }
        
        // Collect roots
        for (int64_t v = 0; v < num_nodes; ++v) {
            if (result.parent[v] == -1) {
                result.roots.push_back(v);
            }
        }
        
        return result;
    }
};

/**
 * @brief Result structure for GVE-Rabbit hybrid ordering
 * 
 * Combines GVE-Leiden community detection with RabbitOrder-style
 * dendrogram building for efficient hierarchical ordering.
 * 
 * @tparam K Community ID type (default: uint32_t)
 */
template <typename K = uint32_t>
struct GVERabbitResult {
    std::vector<K> final_community;
    std::vector<int64_t> parent;
    std::vector<int64_t> first_child;
    std::vector<int64_t> sibling;
    std::vector<int64_t> subtree_size;
    std::vector<double> weight;
    std::vector<int64_t> roots;
    int total_iterations = 0;
    double modularity = 0.0;
};

// ============================================================================
// DENDROGRAM HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Lock-free atomic merge: Prepend vertex v to u's child list using CAS
 * 
 * This is the core RabbitOrder-style operation for building dendrograms.
 * Uses compare-and-swap for thread-safe concurrent construction.
 * 
 * @tparam K Community ID type
 * @param dendro The atomic dendrogram structure
 * @param v Vertex being merged (child)
 * @param u Community representative (parent)
 * @return true if merge was performed, false if skipped
 */
template <typename K = uint32_t>
inline bool atomicMergeToDendro(
    GVEAtomicDendroResult<K>& dendro,
    int64_t v,
    int64_t u) {
    
    if (v == u) return false;
    if (u < 0 || u >= dendro.num_nodes) return false;
    
    // Skip if v already has a parent
    if (dendro.parent[v].load(std::memory_order_acquire) != -1) {
        return false;
    }
    
    // Try to claim v by setting its parent to u
    int64_t expected_parent = -1;
    if (!dendro.parent[v].compare_exchange_strong(expected_parent, u,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    
    // Successfully claimed v - add to u's child list with CAS loop
    int64_t old_first_child = dendro.first_child[u].load(std::memory_order_acquire);
    do {
        dendro.sibling[v].store(old_first_child, std::memory_order_relaxed);
    } while (!dendro.first_child[u].compare_exchange_weak(old_first_child, v,
                std::memory_order_acq_rel, std::memory_order_acquire));
    
    // Update subtree size (atomic add)
    int64_t v_size = dendro.subtree_size[v].load(std::memory_order_relaxed);
    dendro.subtree_size[u].fetch_add(v_size, std::memory_order_relaxed);
    
    return true;
}

/**
 * @brief Non-atomic merge for sequential dendrogram building
 * 
 * Simpler version without atomic operations for single-threaded use.
 * 
 * @tparam K Community ID type
 * @param dendro The dendrogram result structure
 * @param v Vertex being merged (child)
 * @param u Community representative (parent)
 * @return true if merge was performed
 */
template <typename K = uint32_t>
inline bool mergeToDendro(
    GVEDendroResult<K>& dendro,
    int64_t v,
    int64_t u) {
    
    if (v == u) return false;
    if (dendro.parent[v] != -1) return false;
    
    dendro.parent[v] = u;
    dendro.sibling[v] = dendro.first_child[u];
    dendro.first_child[u] = v;
    dendro.subtree_size[u] += dendro.subtree_size[v];
    
    return true;
}

/**
 * @brief Initialize dendrogram with leaf nodes
 * 
 * Each vertex starts as its own community (leaf node) with
 * no parent, children, or siblings.
 * 
 * @tparam K Community ID type
 * @param dendro The dendrogram result structure to initialize
 * @param num_nodes Number of vertices
 * @param vtot Total weight (degree) of each vertex
 */
template <typename K = uint32_t>
inline void initDendrogram(
    GVEDendroResult<K>& dendro,
    const int64_t num_nodes,
    const std::vector<double>& vtot) {
    
    dendro.parent.resize(num_nodes, -1);
    dendro.first_child.resize(num_nodes, -1);
    dendro.sibling.resize(num_nodes, -1);
    dendro.subtree_size.resize(num_nodes, 1);
    dendro.weight.resize(num_nodes);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        dendro.weight[v] = vtot[v];
    }
}

/**
 * @brief Features computed for a community/subgraph
 * 
 * Used by AdaptiveOrder to select the best reordering algorithm
 * for each community based on its structural properties.
 */
struct CommunityFeatures {
    // ---------- Size Metrics ----------
    size_t num_nodes = 0;           ///< Number of vertices in community
    size_t num_edges = 0;           ///< Number of internal edges
    
    // ---------- Density Metrics ----------
    double internal_density = 0.0;  ///< edges / possible_edges
    double avg_degree = 0.0;        ///< Mean degree of vertices
    double degree_variance = 0.0;   ///< Normalized variance in degrees
    
    // ---------- Structure Metrics ----------
    double hub_concentration = 0.0;  ///< Fraction of edges from top 10% nodes
    double modularity = 0.0;         ///< Community/subgraph modularity score
    double clustering_coeff = 0.0;   ///< Local clustering coefficient (sampled)
    
    // ---------- Path Metrics ----------
    double avg_path_length = 0.0;    ///< Estimated average path length
    double diameter_estimate = 0.0;  ///< Estimated graph diameter
    
    // ---------- Hierarchy Metrics ----------
    double community_count = 0.0;    ///< Number of sub-communities detected
    
    // ---------- Performance Metrics ----------
    double reorder_time = 0.0;       ///< Estimated reordering time (if known)
};

/**
 * @brief Dendrogram node for hierarchical community structure
 * 
 * Used by LeidenDendrogram to build and traverse the community
 * hierarchy for ordering vertices.
 */
struct LeidenDendrogramNode {
    int64_t parent = -1;       ///< Parent node in dendrogram (-1 for root)
    int64_t first_child = -1;  ///< First child node (-1 for leaf)
    int64_t sibling = -1;      ///< Next sibling node (-1 if last)
    int64_t vertex_id = -1;    ///< Original vertex ID (-1 for internal nodes)
    size_t subtree_size = 1;   ///< Number of leaves in subtree
    double weight = 0.0;       ///< Degree sum for this subtree
    int level = 0;             ///< Level in hierarchy (0 = leaf)
    int64_t dfs_start = -1;    ///< Starting DFS position for parallel ordering
    
    LeidenDendrogramNode() = default;
};

// ============================================================================
// PERCEPTRON WEIGHTS FOR ADAPTIVE ORDER
// ============================================================================

/**
 * @brief Perceptron weights for ML-based algorithm selection
 * 
 * AdaptiveOrder uses these weights to compute a score for each
 * reordering algorithm based on graph features. The algorithm
 * with the highest score is selected.
 * 
 * Score = bias + sum(weight_i * feature_i)
 */
struct PerceptronWeights {
    // ---------- Base ----------
    double bias = 0.0;              ///< Base preference for this algorithm
    
    // ---------- Feature Weights ----------
    double w_modularity = 0.0;       ///< Weight for modularity feature
    double w_log_nodes = 0.0;        ///< Weight for log(num_nodes)
    double w_log_edges = 0.0;        ///< Weight for log(num_edges)
    double w_density = 0.0;          ///< Weight for density feature
    double w_avg_degree = 0.0;       ///< Weight for average degree
    double w_degree_variance = 0.0;  ///< Weight for degree variance
    double w_hub_concentration = 0.0;///< Weight for hub concentration
    double w_clustering_coeff = 0.0; ///< Weight for clustering coefficient
    double w_avg_path_length = 0.0;  ///< Weight for average path length
    double w_diameter = 0.0;         ///< Weight for diameter estimate
    double w_community_count = 0.0;  ///< Weight for community count
    double w_reorder_time = 0.0;     ///< Weight for reorder time estimate
    
    // ---------- Cache Impact Weights ----------
    double cache_l1_impact = 0.0;    ///< L1 cache impact weight
    double cache_l2_impact = 0.0;    ///< L2 cache impact weight
    double cache_l3_impact = 0.0;    ///< L3 cache impact weight
    double cache_dram_penalty = 0.0; ///< DRAM access penalty weight
    
    // ---------- Per-Benchmark Weights ----------
    std::map<std::string, double> benchmark_weights;
};

// ============================================================================
// UTILITY MACROS
// ============================================================================

/**
 * @brief Print a labeled time value
 * @param label Description of the timing
 * @param seconds Time in seconds
 */
#ifndef REORDER_PRINT_TIME
#define REORDER_PRINT_TIME(label, seconds) \
    printf("%-21s%3.5lf\n", (std::string(label) + ":").c_str(), seconds)
#endif

// ============================================================================
// COMMON UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Compute total edge weight for a vertex
 * 
 * For unweighted graphs, returns the degree. For weighted graphs,
 * returns the sum of edge weights.
 * 
 * @tparam W Weight type
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @param v Vertex ID
 * @param g CSR graph
 * @param symmetric Whether graph is symmetric (affects weight counting)
 * @return Total edge weight for vertex v
 */
template <typename W, typename NodeID_, typename DestID_>
inline W computeVertexTotalWeight(NodeID_ v, 
                                   const CSRGraph<NodeID_, DestID_, true>& g,
                                   bool symmetric) {
    W total = W(0);
    for (auto neighbor : g.out_neigh(v)) {
        if constexpr (std::is_same_v<NodeID_, DestID_>) {
            total += W(1);  // Unweighted
        } else {
            total += static_cast<W>(neighbor.w);  // Weighted
        }
    }
    // For symmetric graphs stored with both directions, don't double count
    return total;
}

/**
 * @brief Verify that a permutation is valid
 * 
 * Checks that new_ids is a valid bijection from [0, n) to [0, n).
 * 
 * @tparam NodeID_ Node ID type
 * @param new_ids The permutation to verify
 * @param n Number of vertices
 * @return true if valid permutation, false otherwise
 */
template <typename NodeID_>
bool verifyPermutation(const pvector<NodeID_>& new_ids, NodeID_ n) {
    std::vector<bool> seen(n, false);
    for (NodeID_ i = 0; i < n; ++i) {
        if (new_ids[i] < 0 || new_ids[i] >= n) {
            return false;  // Out of range
        }
        if (seen[new_ids[i]]) {
            return false;  // Duplicate
        }
        seen[new_ids[i]] = true;
    }
    return true;
}

/**
 * @brief Count unique communities in an assignment
 * 
 * @tparam K Community ID type
 * @param communities Vector of community assignments
 * @return Number of unique communities
 */
template <typename K>
size_t countUniqueCommunities(const std::vector<K>& communities) {
    std::unordered_set<K> unique(communities.begin(), communities.end());
    return unique.size();
}

/**
 * @brief Auto-compute Leiden resolution based on graph density
 * 
 * Higher resolution = more communities
 * Lower resolution = fewer, larger communities
 * 
 * Formula: γ = clip(0.5 + 0.25 × log₁₀(avg_degree + 1), 0.5, 1.2)
 * If CV(degree) > 2: γ = max(γ, 1.0)  // Guardrail for power-law graphs
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @param g CSR graph
 * @return Recommended resolution parameter
 */
template <typename NodeID_, typename DestID_>
double computeAutoResolution(const CSRGraph<NodeID_, DestID_, true>& g) {
    const int64_t n = g.num_nodes();
    const int64_t m = g.num_edges();
    
    if (n == 0) return 1.0;
    
    double avg_degree = static_cast<double>(m) / n;
    
    // Compute degree variance for CV guardrail
    double sum_sq = 0.0;
    #pragma omp parallel for reduction(+:sum_sq)
    for (int64_t v = 0; v < n; ++v) {
        double d = static_cast<double>(g.out_degree(v));
        double diff = d - avg_degree;
        sum_sq += diff * diff;
    }
    double variance = sum_sq / n;
    double cv = (avg_degree > 0) ? std::sqrt(variance) / avg_degree : 0.0;
    
    // Base resolution from average degree
    double resolution = 0.5 + 0.25 * std::log10(avg_degree + 1);
    resolution = std::max(0.5, std::min(1.2, resolution));
    
    // CV guardrail for power-law/hubby graphs
    if (cv > 2.0) {
        resolution = std::max(resolution, 1.0);
    }
    
    return resolution;
}

// ============================================================================
// SIGNAL HANDLER FOR DEBUGGING
// ============================================================================

/**
 * @brief Install SIGSEGV handler for debugging memory issues
 * 
 * Useful during development to get stack traces on segfaults.
 */
inline void install_sigsegv_handler() {
#ifdef _DEBUG
    struct sigaction sa;
    sa.sa_handler = [](int) {
        std::cerr << "SIGSEGV received - memory access error" << std::endl;
        std::abort();
    };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, nullptr);
#endif
}

// ============================================================================
// MODULARITY COMPUTATION
// ============================================================================

/**
 * @brief Compute modularity for a community assignment on CSR graph
 * 
 * Modularity Q = (1/2m) * Σ[A_ij - R * k_i * k_j / (2m)] * δ(c_i, c_j)
 * where m = total edge weight, R = resolution parameter
 * 
 * Uses parallel reduction for efficiency on large graphs.
 * 
 * @tparam K Community ID type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type (NodeID_T for unweighted, NodeWeight for weighted)
 * @param g CSR graph
 * @param community Community assignment vector
 * @param resolution Resolution parameter (default 1.0)
 * @return Modularity value in range [-0.5, 1.0]
 */
template <typename K, typename NodeID_T, typename DestID_T>
double computeModularityCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<K>& community,
    double resolution = 1.0) {
    
    const int64_t num_nodes = g.num_nodes();
    const bool graph_is_symmetric = !g.directed();
    const double M = static_cast<double>(g.num_edges());
    
    if (M == 0) return 0.0;
    
    // Compute vertex degrees (weighted)
    std::vector<double> vtot(num_nodes, 0.0);
    #pragma omp parallel for
    for (int64_t u = 0; u < num_nodes; ++u) {
        double total = 0.0;
        for (auto neighbor : g.out_neigh(u)) {
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                total += 1.0;
            } else {
                total += static_cast<double>(neighbor.w);
            }
        }
        if (!graph_is_symmetric) {
            for (auto neighbor : g.in_neigh(u)) {
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    total += 1.0;
                } else {
                    total += static_cast<double>(neighbor.w);
                }
            }
        }
        vtot[u] = total;
    }
    
    // Compute modularity using parallel reduction
    double Q = 0.0;
    #pragma omp parallel for reduction(+:Q)
    for (int64_t u = 0; u < num_nodes; ++u) {
        K cu = community[u];
        
        for (auto neighbor : g.out_neigh(u)) {
            NodeID_T v;
            double w;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = neighbor;
                w = 1.0;
            } else {
                v = neighbor.v;
                w = static_cast<double>(neighbor.w);
            }
            
            K cv = community[v];
            if (cu == cv) {
                Q += w - resolution * vtot[u] * vtot[v] / (2.0 * M);
            }
        }
        
        if (!graph_is_symmetric) {
            for (auto neighbor : g.in_neigh(u)) {
                NodeID_T v;
                double w;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    v = neighbor;
                    w = 1.0;
                } else {
                    v = neighbor.v;
                    w = static_cast<double>(neighbor.w);
                }
                
                K cv = community[v];
                if (cu == cv) {
                    Q += w - resolution * vtot[u] * vtot[v] / (2.0 * M);
                }
            }
        }
    }
    
    return Q / (2.0 * M);
}

// ============================================================================
// VERTEX EDGE SCANNING
// ============================================================================

/**
 * @brief Scan all edges connected to vertex u and accumulate by community
 * 
 * Scans both out-edges and in-edges (for non-symmetric graphs).
 * Accumulates edge weights into the hash map by community.
 * 
 * @tparam K Community ID type
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param u Vertex to scan
 * @param vcom Community assignment array
 * @param hash Hash map to accumulate weights by community
 * @param d Target community to track separately
 * @param g CSR graph
 * @param graph_is_symmetric Whether graph is symmetric (avoids double-counting)
 * @return Total weight to community d
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
inline W scanVertexEdges(
    NodeID_T u,
    const K* vcom,
    std::unordered_map<K, W>& hash,
    K d,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric) {
    
    W ku_to_d = W(0);
    
    // Scan out-neighbors
    for (auto neighbor : g.out_neigh(u)) {
        NodeID_T v;
        W w;
        if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
            v = neighbor;
            w = W(1);
        } else {
            v = neighbor.v;
            w = static_cast<W>(neighbor.w);
        }
        
        K c = vcom[v];
        hash[c] += w;
        if (c == d) ku_to_d += w;
    }
    
    // For non-symmetric graphs, also scan in-neighbors
    if (!graph_is_symmetric) {
        for (auto neighbor : g.in_neigh(u)) {
            NodeID_T v;
            W w;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = neighbor;
                w = W(1);
            } else {
                v = neighbor.v;
                w = static_cast<W>(neighbor.w);
            }
            
            K c = vcom[v];
            hash[c] += w;
            if (c == d) ku_to_d += w;
        }
    }
    
    return ku_to_d;
}

/**
 * @brief Compute total edge weight for a vertex
 * 
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param u Vertex ID
 * @param g CSR graph
 * @param graph_is_symmetric Whether graph is symmetric
 * @return Total edge weight for vertex u
 */
template <typename W, typename NodeID_T, typename DestID_T>
inline W computeVertexTotalWeightCSR(
    NodeID_T u,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric) {
    
    W total = W(0);
    
    for (auto neighbor : g.out_neigh(u)) {
        if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
            total += W(1);
        } else {
            total += static_cast<W>(neighbor.w);
        }
    }
    
    if (!graph_is_symmetric) {
        for (auto neighbor : g.in_neigh(u)) {
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                total += W(1);
            } else {
                total += static_cast<W>(neighbor.w);
            }
        }
    }
    
    return total;
}

/**
 * @brief Mark all neighbors of a vertex as affected
 * 
 * Used in local-moving phase to mark vertices that need re-evaluation
 * after a neighbor has moved communities.
 * 
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param u Vertex whose neighbors should be marked
 * @param vaff Affected flag array
 * @param g CSR graph
 * @param graph_is_symmetric Whether graph is symmetric
 */
template <typename NodeID_T, typename DestID_T>
inline void markNeighborsAffected(
    NodeID_T u,
    std::vector<char>& vaff,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric) {
    
    for (auto neighbor : g.out_neigh(u)) {
        NodeID_T v;
        if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
            v = neighbor;
        } else {
            v = neighbor.v;
        }
        vaff[v] = 1;
    }
    
    if (!graph_is_symmetric) {
        for (auto neighbor : g.in_neigh(u)) {
            NodeID_T v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = neighbor;
            } else {
                v = neighbor.v;
            }
            vaff[v] = 1;
        }
    }
}

/**
 * @brief Build dendrogram from community assignments
 * 
 * Creates a hierarchical structure from flat community assignments.
 * For each community, the vertex with highest weight becomes the representative (root),
 * and all other vertices in that community become children of the representative.
 *
 * @tparam K Community ID type
 * @tparam W Weight type
 * @param dendro Dendrogram result structure to populate
 * @param vcom Community assignment for each vertex
 * @param vtot Total weight (degree) of each vertex
 * @param num_nodes Number of nodes in the graph
 */
template <typename K, typename W>
void buildDendrogramFromCommunities(
    GVEDendroResult<K>& dendro,
    const std::vector<K>& vcom,
    const std::vector<W>& vtot,
    int64_t num_nodes) {
    
    // Use vector instead of unordered_map for efficiency (community ID = vertex ID)
    // Track representative per community: rep[c] = vertex with max weight in community c
    std::vector<int64_t> comm_rep(num_nodes, -1);
    
    // Find representative (highest weight vertex) for each community
    for (int64_t v = 0; v < num_nodes; ++v) {
        K c = vcom[v];
        if (comm_rep[c] == -1 || vtot[v] > vtot[comm_rep[c]]) {
            comm_rep[c] = v;
        }
    }
    
    // Clear existing dendrogram structure (rebuild fresh each pass)
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        dendro.first_child[v] = -1;
        dendro.sibling[v] = -1;
        dendro.parent[v] = -1;
        dendro.subtree_size[v] = 1;
        dendro.weight[v] = vtot[v];
    }
    
    // Link non-representatives to their representative (sequential to avoid races)
    for (int64_t v = 0; v < num_nodes; ++v) {
        K c = vcom[v];
        int64_t rep = comm_rep[c];
        if (v != rep && rep >= 0) {
            // Prepend v to rep's child list
            dendro.sibling[v] = dendro.first_child[rep];
            dendro.first_child[rep] = v;
            dendro.parent[v] = rep;
            dendro.subtree_size[rep] += dendro.subtree_size[v];
            dendro.weight[rep] += dendro.weight[v];
        }
    }
}

/**
 * Traverse dendrogram in DFS order to produce vertex ordering.
 * 
 * @tparam K Community ID type
 * @tparam NodeID_T Node ID type for new_ids array
 * @param dendro The dendrogram structure to traverse
 * @param new_ids Output array mapping original vertex -> new position
 * @param hub_first If true, visit higher-weight children first
 */
template <typename K, typename NodeID_T>
void traverseDendrogramDFS(
    const GVEDendroResult<K>& dendro,
    std::vector<NodeID_T>& new_ids,
    bool hub_first = true) {
    
    const int64_t num_nodes = static_cast<int64_t>(dendro.parent.size());
    
    // Find root nodes (nodes with no parent)
    std::vector<int64_t> roots;
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (dendro.parent[v] == -1) {
            roots.push_back(v);
        }
    }
    
    // Sort roots by weight (hub-first)
    if (hub_first) {
        std::sort(roots.begin(), roots.end(),
            [&dendro](int64_t a, int64_t b) {
                return dendro.weight[a] > dendro.weight[b];
            });
    }
    
    // DFS traversal with stack
    std::vector<int64_t> stack;
    int64_t current_id = 0;
    
    for (int64_t root : roots) {
        stack.push_back(root);
        
        while (!stack.empty()) {
            int64_t v = stack.back();
            stack.pop_back();
            
            // Assign new ID
            new_ids[v] = current_id++;
            
            // Collect children and sort by weight if hub-first
            std::vector<int64_t> children;
            int64_t child = dendro.first_child[v];
            while (child != -1) {
                children.push_back(child);
                child = dendro.sibling[child];
            }
            
            if (hub_first && !children.empty()) {
                std::sort(children.begin(), children.end(),
                    [&dendro](int64_t a, int64_t b) {
                        return dendro.weight[a] < dendro.weight[b]; // Reverse for stack
                    });
            }
            
            // Push children to stack (in reverse order for correct DFS)
            for (int64_t c : children) {
                stack.push_back(c);
            }
        }
    }
}

#endif  // REORDER_TYPES_H_
