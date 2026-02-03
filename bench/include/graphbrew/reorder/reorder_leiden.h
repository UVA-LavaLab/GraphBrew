/**
 * @file reorder_leiden.h
 * @brief Leiden-based community detection and graph reordering - API Reference
 *
 * This header provides documentation and utility types for Leiden-based
 * reordering algorithms. The core implementations remain in builder.h due to
 * tight integration with the graph building template system.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * LEIDENORDER (ID 15) - Classic Leiden via igraph library
 *   Format: -o 15:resolution
 *   Example: ./bench/bin/pr -f graph.el -o 15:0.75
 *   
 *   Uses igraph's Leiden implementation. Good for quality-focused detection.
 *   Requires igraph library to be installed.
 *
 * LEIDENDENDROGRAM (ID 16) - Hierarchical Leiden reordering
 *   Format: -o 16:resolution:variant
 *   Variants: dfs, dfshub, dfssize, bfs, hybrid (default: hybrid)
 *   Example: ./bench/bin/pr -f graph.el -o 16:0.75:hybrid
 *   
 *   Builds community hierarchy and uses tree traversal for final ordering.
 *   Best for hierarchical graph structures.
 *
 * LEIDENCSR (ID 17) - Fast GVE-Leiden on native CSR
 *   Format: -o 17:variant:resolution:iterations:passes
 *   Variants: gve, gveopt, gverabbit, dfs, bfs, hubsort, fast, modularity
 *   Example: ./bench/bin/pr -f graph.el -o 17:gve:0.75:10:10
 *   
 *   Native CSR implementation - no igraph dependency. Multiple variants:
 *   - gve: Standard GVE-Leiden (default, good balance)
 *   - gveopt: Cache-optimized (faster on large graphs)
 *   - gverabbit: GVE + RabbitOrder within communities
 *   - fast: Speed-optimized (fewer iterations)
 *   - modularity: Quality-optimized (more iterations)
 *
 * ============================================================================
 * IMPLEMENTATION DETAILS
 * ============================================================================
 *
 * Core Functions in builder.h:
 * ----------------------------
 * GVELeidenCSR<K>()       - Standard GVE-Leiden algorithm
 * GVELeidenOpt<K>()       - Cache-optimized variant
 * GVELeidenDendo<K>()     - With incremental dendrogram
 * GVELeidenOptDendo<K>()  - Optimized with dendrogram
 *
 * Wrapper Functions in builder.h:
 * --------------------------------
 * GenerateLeidenMapping()              - Entry for LeidenOrder (15)
 * GenerateLeidenDendrogramMappingUnified() - Entry for LeidenDendrogram (16)
 * GenerateLeidenCSRMappingUnified()    - Entry for LeidenCSR (17)
 *
 * Result Structures in builder.h:
 * -------------------------------
 * GVELeidenResult<K>       - Community assignments + modularity
 * GVEDendroResult<K>       - Result with explicit tree structure
 * GVEAtomicDendroResult<K> - Lock-free parallel dendrogram
 *
 * ============================================================================
 * ALGORITHM PARAMETERS
 * ============================================================================
 *
 * resolution (default: 0.75)
 *   Controls community granularity. Range: 0.0 - 2.0
 *   Lower = fewer, larger communities
 *   Higher = more, smaller communities
 *
 * tolerance (default: 1e-2)
 *   Convergence threshold for modularity improvement.
 *
 * aggregation_tolerance (default: 0.8)
 *   Threshold for super-graph aggregation.
 *
 * max_iterations (default: 10)
 *   Maximum local moving iterations per pass.
 *
 * max_passes (default: 10)
 *   Maximum number of aggregation passes.
 *
 * ============================================================================
 * DENDROGRAM TRAVERSAL VARIANTS
 * ============================================================================
 *
 * For LeidenDendrogram (-o 16), the traversal variant affects final ordering:
 *
 * DFS (dfs):
 *   Depth-first traversal. Groups related vertices closely.
 *
 * DFS-Hub (dfshub):
 *   DFS with hub vertices (high degree) processed first within each level.
 *   Better cache locality for power-law graphs.
 *
 * DFS-Size (dfssize):
 *   DFS with larger subtrees processed first.
 *   Groups big communities together.
 *
 * BFS (bfs):
 *   Breadth-first traversal. Level-order processing.
 *
 * Hybrid (hybrid) [default]:
 *   Combines DFS-Hub for initial levels, BFS for deep levels.
 *   Best general-purpose choice.
 *
 * ============================================================================
 * PERFORMANCE RECOMMENDATIONS
 * ============================================================================
 *
 * Small graphs (<100K vertices):
 *   Use -o 15 (igraph) or -o 17:gve for quality
 *
 * Medium graphs (100K - 10M vertices):
 *   Use -o 17:gve or -o 17:gveopt
 *
 * Large graphs (>10M vertices):
 *   Use -o 17:gveopt for speed
 *   Use -o 17:fast if quality is less critical
 *
 * Hierarchical structures:
 *   Use -o 16 with appropriate traversal variant
 *
 * ============================================================================
 * EXAMPLES
 * ============================================================================
 *
 * Basic Leiden (igraph):
 *   ./bench/bin/pr -f graph.el -o 15 -n 5
 *
 * High-resolution (more communities):
 *   ./bench/bin/pr -f graph.el -o 15:1.5 -n 5
 *
 * Fast GVE-Leiden:
 *   ./bench/bin/pr -f graph.el -o 17:gve:0.75 -n 5
 *
 * Cache-optimized for large graphs:
 *   ./bench/bin/pr -f graph.el -o 17:gveopt:0.75 -n 5
 *
 * Quality-focused (more iterations):
 *   ./bench/bin/pr -f graph.el -o 17:modularity:1.0:20:20 -n 5
 *
 * Dendrogram with hybrid traversal:
 *   ./bench/bin/pr -f graph.el -o 16:0.75:hybrid -n 5
 */

#ifndef REORDER_LEIDEN_H_
#define REORDER_LEIDEN_H_

#include <string>
#include <cstdint>
#include "reorder_types.h"

namespace graphbrew {
namespace leiden {

// ============================================================================
// VARIANT ENUMERATIONS
// ============================================================================

/**
 * @brief Variant selection for LeidenCSR (-o 17)
 */
enum class LeidenCSRVariant {
    GVE,        ///< Standard GVE-Leiden (default)
    GVEOpt,     ///< Cache-optimized GVE-Leiden
    GVERabbit,  ///< GVE + RabbitOrder within communities
    DFS,        ///< DFS ordering of community tree
    BFS,        ///< BFS ordering of community tree
    HubSort,    ///< Hub-first ordering within communities
    Fast,       ///< Speed-optimized (fewer iterations)
    Modularity  ///< Quality-optimized (more iterations)
};

/**
 * @brief Variant selection for LeidenDendrogram (-o 16)
 */
enum class DendrogramTraversal {
    DFS,        ///< Depth-first traversal
    DFSHub,     ///< DFS with hub vertices first
    DFSSize,    ///< DFS with larger subtrees first
    BFS,        ///< Breadth-first traversal
    Hybrid      ///< Combined DFS-Hub and BFS (default)
};

// ============================================================================
// STRING PARSING UTILITIES
// ============================================================================

inline LeidenCSRVariant ParseLeidenCSRVariant(const std::string& s) {
    if (s == "gve" || s.empty()) return LeidenCSRVariant::GVE;
    if (s == "gveopt") return LeidenCSRVariant::GVEOpt;
    if (s == "gverabbit") return LeidenCSRVariant::GVERabbit;
    if (s == "dfs") return LeidenCSRVariant::DFS;
    if (s == "bfs") return LeidenCSRVariant::BFS;
    if (s == "hubsort") return LeidenCSRVariant::HubSort;
    if (s == "fast") return LeidenCSRVariant::Fast;
    if (s == "modularity") return LeidenCSRVariant::Modularity;
    return LeidenCSRVariant::GVE;
}

inline DendrogramTraversal ParseDendrogramTraversal(const std::string& s) {
    if (s == "dfs") return DendrogramTraversal::DFS;
    if (s == "dfshub") return DendrogramTraversal::DFSHub;
    if (s == "dfssize") return DendrogramTraversal::DFSSize;
    if (s == "bfs") return DendrogramTraversal::BFS;
    if (s == "hybrid" || s.empty()) return DendrogramTraversal::Hybrid;
    return DendrogramTraversal::Hybrid;
}

inline std::string LeidenCSRVariantToString(LeidenCSRVariant v) {
    switch (v) {
        case LeidenCSRVariant::GVE: return "gve";
        case LeidenCSRVariant::GVEOpt: return "gveopt";
        case LeidenCSRVariant::GVERabbit: return "gverabbit";
        case LeidenCSRVariant::DFS: return "dfs";
        case LeidenCSRVariant::BFS: return "bfs";
        case LeidenCSRVariant::HubSort: return "hubsort";
        case LeidenCSRVariant::Fast: return "fast";
        case LeidenCSRVariant::Modularity: return "modularity";
        default: return "unknown";
    }
}

inline std::string DendrogramTraversalToString(DendrogramTraversal t) {
    switch (t) {
        case DendrogramTraversal::DFS: return "dfs";
        case DendrogramTraversal::DFSHub: return "dfshub";
        case DendrogramTraversal::DFSSize: return "dfssize";
        case DendrogramTraversal::BFS: return "bfs";
        case DendrogramTraversal::Hybrid: return "hybrid";
        default: return "unknown";
    }
}

// ============================================================================
// DEFAULT PARAMETERS - Leiden Algorithm Constants
// These are in graphbrew::leiden namespace (not nested further)
// ============================================================================

constexpr double DEFAULT_RESOLUTION = 0.75;
constexpr double DEFAULT_TOLERANCE = 1e-2;
constexpr double DEFAULT_AGGREGATION_TOLERANCE = 0.8;
constexpr double DEFAULT_QUALITY_FACTOR = 10.0;
constexpr int DEFAULT_MAX_ITERATIONS = 10;
constexpr int DEFAULT_MAX_PASSES = 10;

// Fast variant parameters
constexpr int FAST_MAX_ITERATIONS = 5;
constexpr int FAST_MAX_PASSES = 5;

// Modularity variant parameters (quality-focused)
constexpr int MODULARITY_MAX_ITERATIONS = 20;
constexpr int MODULARITY_MAX_PASSES = 20;

// Sort variant uses higher tolerance to allow more passes
constexpr double SORT_AGGREGATION_TOLERANCE = 0.95;

// ============================================================================
// MODULARITY CALCULATION UTILITIES
// ============================================================================

/**
 * @brief Calculate delta modularity for moving vertex between communities
 * 
 * Uses the Leiden/Louvain modularity gain formula:
 *   ΔQ = (K_i→c - K_i→d)/m - R·K_i·(Σ_c - Σ_d + K_i)/(2m²)
 * 
 * @tparam W Weight type (typically double)
 * @param ki_to_c Weight of edges from vertex i to community c
 * @param ki_to_d Weight of edges from vertex i to community d (current)
 * @param ki Total weight of vertex i (degree)
 * @param sigma_c Total weight of community c
 * @param sigma_d Total weight of community d
 * @param M Total edge weight in graph (must be > 0)
 * @param R Resolution parameter
 * @return Delta modularity (positive = beneficial move), or 0 if M is zero
 */
template <typename W>
inline W gveDeltaModularity(W ki_to_c, W ki_to_d, W ki, W sigma_c, W sigma_d, W M, W R) {
    // Guard against division by zero (can happen with degenerate graphs)
    if (M <= W(0)) return W(0);
    return (ki_to_c - ki_to_d) / M - R * ki * (sigma_c - sigma_d + ki) / (W(2.0) * M * M);
}

} // namespace leiden
} // namespace graphbrew

// ============================================================================
// LEIDEN ALGORITHM IMPLEMENTATIONS
// These are standalone template functions that work with CSRGraph directly
// ============================================================================

// Bring leiden constants into global scope for implementations below
using graphbrew::leiden::DEFAULT_TOLERANCE;
using graphbrew::leiden::DEFAULT_AGGREGATION_TOLERANCE;
using graphbrew::leiden::DEFAULT_QUALITY_FACTOR;
using graphbrew::leiden::DEFAULT_MAX_ITERATIONS;
using graphbrew::leiden::DEFAULT_MAX_PASSES;
using graphbrew::leiden::FAST_MAX_ITERATIONS;
using graphbrew::leiden::FAST_MAX_PASSES;
using graphbrew::leiden::MODULARITY_MAX_ITERATIONS;
using graphbrew::leiden::MODULARITY_MAX_PASSES;
using graphbrew::leiden::SORT_AGGREGATION_TOLERANCE;
// Note: DEFAULT_RESOLUTION not brought to global scope to avoid conflict with adaptive::DEFAULT_RESOLUTION

#include <graph.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>

/**
 * @brief Parallel local moving phase for Leiden algorithm
 * 
 * Two-phase approach:
 * 1. Find best community for each vertex (parallel, read-only)
 * 2. Apply moves and update weights (parallel with atomics)
 * 
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param g CSR graph
 * @param community Current community assignment (modified)
 * @param comm_weight Community weights (modified)
 * @param vertex_weight Vertex weights (degree)
 * @param total_weight Total edge weight in graph
 * @param resolution Modularity resolution parameter
 * @param max_iterations Maximum iterations
 * @return Total number of moves made
 */
template<typename NodeID_T, typename DestID_T>
int64_t LeidenLocalMoveParallel(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    std::vector<int64_t>& community,
    std::vector<double>& comm_weight,
    const std::vector<double>& vertex_weight,
    double total_weight,
    double resolution,
    int max_iterations)
{
    const int64_t num_vertices = g.num_nodes();
    int64_t total_moves = 0;
    
    // Storage for proposed moves
    std::vector<int64_t> best_comm(num_vertices);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        int64_t moves_this_iter = 0;
        
        // Phase 1: Find best community for each vertex (read-only)
        #pragma omp parallel
        {
            // Thread-local dense array for counting (sparse usage)
            std::vector<double> count_arr(num_vertices, 0.0);
            std::vector<int64_t> touched_comms;
            touched_comms.reserve(512);
            
            #pragma omp for schedule(static)
            for (int64_t v = 0; v < num_vertices; ++v) {
                int64_t deg = g.out_degree(v);
                best_comm[v] = community[v];
                
                if (deg == 0) continue;
                
                int64_t current_comm = community[v];
                double v_weight = vertex_weight[v];
                
                // Count edges to each neighbor community
                touched_comms.clear();
                for (auto u : g.out_neigh(v)) {
                    int64_t c = community[u];
                    if (count_arr[c] == 0.0) {
                        touched_comms.push_back(c);
                    }
                    count_arr[c] += 1.0;
                }
                
                // Get edges to current community
                double edges_to_current = count_arr[current_comm];
                
                // Find best community
                double best_delta = 0.0;
                double sigma_current = comm_weight[current_comm] - v_weight;
                double leave_delta = edges_to_current - resolution * v_weight * sigma_current / total_weight;
                
                for (int64_t c : touched_comms) {
                    if (c == current_comm) continue;
                    
                    double edges_to_c = count_arr[c];
                    double sigma_c = comm_weight[c];
                    double join_delta = edges_to_c - resolution * v_weight * sigma_c / total_weight;
                    double delta = join_delta - leave_delta;
                    
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_comm[v] = c;
                    }
                }
                
                // Reset count array (only touched entries)
                for (int64_t c : touched_comms) {
                    count_arr[c] = 0.0;
                }
            }
        }
        
        // Phase 2: Apply moves and count
        #pragma omp parallel for schedule(static) reduction(+:moves_this_iter)
        for (int64_t v = 0; v < num_vertices; ++v) {
            if (best_comm[v] != community[v]) {
                int64_t old_comm = community[v];
                int64_t new_comm = best_comm[v];
                double v_weight = vertex_weight[v];
                
                #pragma omp atomic
                comm_weight[old_comm] -= v_weight;
                #pragma omp atomic
                comm_weight[new_comm] += v_weight;
                
                community[v] = new_comm;
                moves_this_iter++;
            }
        }
        
        total_moves += moves_this_iter;
        if (moves_this_iter == 0) break;
    }
    
    return total_moves;
}

/**
 * @brief Main Leiden community detection algorithm
 * 
 * Quality-focused community detection using iterative local moving.
 * 
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param g CSR graph
 * @param final_community Output: final community assignment
 * @param resolution Modularity resolution (default 1.0)
 * @param max_passes Maximum aggregation passes (default 3)
 * @param max_iterations Maximum iterations per pass (default 20)
 */
template<typename NodeID_T, typename DestID_T>
void LeidenCommunityDetection(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    std::vector<int64_t>& final_community,
    double resolution = 1.0,
    int max_passes = 3,
    int max_iterations = 20)
{
    const int64_t num_vertices = g.num_nodes();
    // Guard against zero edges to prevent FPE
    const double total_weight = std::max(1.0, static_cast<double>(g.num_edges_directed()));
    
    std::vector<int64_t> community(num_vertices);
    std::vector<double> vertex_weight(num_vertices);
    std::vector<double> comm_weight(num_vertices);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        community[v] = v;
        vertex_weight[v] = static_cast<double>(g.out_degree(v));
        comm_weight[v] = vertex_weight[v];
    }
    
    int64_t total_moves = 0;
    int pass = 0;
    
    // Use bitmap instead of unordered_set for O(1) community counting
    std::vector<char> comm_exists(num_vertices, 0);
    
    for (pass = 0; pass < max_passes; ++pass) {
        int64_t moves = LeidenLocalMoveParallel<NodeID_T, DestID_T>(
            g, community, comm_weight, vertex_weight,
            total_weight, resolution, max_iterations);
        
        // Count communities using bitmap (O(N) instead of hash table)
        std::fill(comm_exists.begin(), comm_exists.end(), 0);
        for (int64_t v = 0; v < num_vertices; ++v) {
            comm_exists[community[v]] = 1;
        }
        int64_t num_comms = 0;
        for (int64_t c = 0; c < num_vertices; ++c) {
            if (comm_exists[c]) num_comms++;
        }
        
        printf("Leiden pass %d: %ld moves, %ld communities\n", pass + 1, moves, num_comms);
        
        total_moves += moves;
        if (moves == 0) break;
    }
    
    // Compress communities using array-based remapping (O(N) instead of hash table)
    std::vector<int64_t> comm_remap(num_vertices, -1);
    int64_t num_comms = 0;
    for (int64_t v = 0; v < num_vertices; ++v) {
        int64_t c = community[v];
        if (comm_remap[c] == -1) {
            comm_remap[c] = num_comms++;
        }
    }
    
    final_community.resize(num_vertices);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        final_community[v] = comm_remap[community[v]];
    }
    
    printf("Leiden: %ld total moves, %d passes, %ld final communities\n",
           total_moves, pass, num_comms);
}

/**
 * @brief Fast modularity-based community detection using parallel Union-Find
 * 
 * Combines:
 * - Lock-free Union-Find with atomic CAS for merging
 * - Best-fit modularity merging (scans all neighbors)
 * - Label propagation refinement with hash-based counting
 * 
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param g CSR graph
 * @param vertex_strength Output: vertex strengths (degrees)
 * @param final_community Output: final community assignment
 * @param resolution Modularity resolution (default 1.0)
 * @param max_passes Maximum label propagation passes (default 3)
 */
template<typename NodeID_T, typename DestID_T>
void FastModularityCommunityDetection(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    std::vector<double>& vertex_strength,
    std::vector<int64_t>& final_community,
    double resolution = 1.0,
    int max_passes = 3)
{
    const int64_t num_vertices = g.num_nodes();
    // Guard against zero edges to prevent FPE
    const double total_weight = std::max(1.0, static_cast<double>(g.num_edges_directed()));
    
    // Initialize vertex strengths
    vertex_strength.resize(num_vertices);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        vertex_strength[v] = static_cast<double>(g.out_degree(v));
    }
    
    // Atomic parent array for lock-free Union-Find
    std::vector<std::atomic<int64_t>> parent(num_vertices);
    std::vector<std::atomic<double>> comm_strength(num_vertices);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        parent[v].store(v, std::memory_order_relaxed);
        comm_strength[v].store(vertex_strength[v], std::memory_order_relaxed);
    }
    
    // Lock-free find with path compression
    auto find = [&](int64_t x) -> int64_t {
        int64_t root = x;
        while (true) {
            int64_t p = parent[root].load(std::memory_order_relaxed);
            if (p == root) break;
            root = p;
        }
        int64_t curr = x;
        while (curr != root) {
            int64_t p = parent[curr].load(std::memory_order_relaxed);
            parent[curr].store(root, std::memory_order_relaxed);
            curr = p;
        }
        return root;
    };
    
    // Process in degree order (smaller first)
    std::vector<std::pair<int64_t, int64_t>> deg_order(num_vertices);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        deg_order[v] = {g.out_degree(v), v};
    }
    __gnu_parallel::sort(deg_order.begin(), deg_order.end());
    
    std::atomic<int64_t> total_merges{0};
    
    // Phase 1: Parallel Union-Find with best-fit modularity merging
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1024)
        for (int64_t i = 0; i < num_vertices; ++i) {
            int64_t v = deg_order[i].second;
            int64_t deg = deg_order[i].first;
            if (deg == 0) continue;
            
            int64_t v_root = find(v);
            double v_str = comm_strength[v_root].load(std::memory_order_relaxed);
            
            int64_t best_root = v_root;
            double best_delta = 0.0;
            
            for (auto u : g.out_neigh(v)) {
                int64_t u_root = find(u);
                if (u_root == v_root) continue;
                
                double u_str = comm_strength[u_root].load(std::memory_order_relaxed);
                double delta = 1.0 - resolution * v_str * u_str / total_weight;
                
                if (delta > best_delta) {
                    best_delta = delta;
                    best_root = u_root;
                }
            }
            
            if (best_root != v_root && best_delta > 0) {
                int64_t from = (v_root < best_root) ? v_root : best_root;
                int64_t to = (v_root < best_root) ? best_root : v_root;
                
                int64_t expected = from;
                if (parent[from].compare_exchange_weak(expected, to, 
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    double from_str = comm_strength[from].load(std::memory_order_relaxed);
                    double old_to_str = comm_strength[to].load(std::memory_order_relaxed);
                    while (!comm_strength[to].compare_exchange_weak(old_to_str, 
                            old_to_str + from_str, std::memory_order_relaxed));
                    total_merges.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    
    printf("LeidenFast: %ld merges in parallel Union-Find phase\n", 
           total_merges.load());
    
    // Compress all paths
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        find(v);
    }
    
    // Phase 2: Label propagation refinement
    std::vector<int64_t> labels(num_vertices);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        labels[v] = parent[v].load(std::memory_order_relaxed);
    }
    
    for (int pass = 1; pass < max_passes; ++pass) {
        std::atomic<int64_t> moves{0};
        
        #pragma omp parallel
        {
            std::unordered_map<int64_t, int64_t> label_counts;
            label_counts.reserve(256);
            
            #pragma omp for schedule(dynamic, 2048)
            for (int64_t i = 0; i < num_vertices; ++i) {
                int64_t v = deg_order[i].second;
                int64_t deg = g.out_degree(v);
                if (deg == 0) continue;
                
                int64_t current_label = labels[v];
                
                label_counts.clear();
                for (auto u : g.out_neigh(v)) {
                    label_counts[labels[u]]++;
                }
                
                int64_t best_label = current_label;
                int64_t best_count = 0;
                int64_t current_count = 0;
                
                for (auto& [lbl, cnt] : label_counts) {
                    if (lbl == current_label) {
                        current_count = cnt;
                    }
                    if (cnt > best_count) {
                        best_count = cnt;
                        best_label = lbl;
                    }
                }
                
                if (best_label != current_label && best_count > current_count) {
                    labels[v] = best_label;
                    moves.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        
        int64_t move_count = moves.load();
        printf("LeidenFast: %ld moves in LP pass %d\n", move_count, pass);
        
        if (move_count == 0 || move_count < num_vertices / 1000) break;
    }
    
    // Compress labels to contiguous IDs
    std::unordered_map<int64_t, int64_t> label_remap;
    int64_t num_comms = 0;
    for (int64_t v = 0; v < num_vertices; ++v) {
        int64_t l = labels[v];
        auto it = label_remap.find(l);
        if (it == label_remap.end()) {
            label_remap[l] = num_comms++;
        }
    }
    
    final_community.resize(num_vertices);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        final_community[v] = label_remap[labels[v]];
    }
    
    printf("LeidenFast: %ld final communities\n", num_comms);
}

/**
 * @brief Build final vertex ordering from communities
 * 
 * Orders vertices so that:
 * 1. Larger communities come first
 * 2. Within each community, higher degree vertices come first
 * 
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param g CSR graph
 * @param vertex_strength Vertex strengths (degrees)
 * @param community Community assignment
 * @param ordered_vertices Output: vertices in final order
 */
template<typename NodeID_T, typename DestID_T>
void BuildCommunityOrdering(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<double>& vertex_strength,
    const std::vector<int64_t>& community,
    std::vector<int64_t>& ordered_vertices)
{
    const int64_t num_vertices = g.num_nodes();
    
    // Count communities and compute their total strengths
    int64_t num_comms = 0;
    for (int64_t v = 0; v < num_vertices; ++v) {
        num_comms = std::max(num_comms, community[v] + 1);
    }
    
    std::vector<double> comm_strength(num_comms, 0.0);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        int64_t c = community[v];
        #pragma omp atomic
        comm_strength[c] += vertex_strength[v];
    }
    
    // Build ordering: (community strength DESC, vertex degree DESC, vertex ID)
    std::vector<std::tuple<int64_t, int64_t, int64_t>> order(num_vertices);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_vertices; ++v) {
        int64_t c = community[v];
        order[v] = std::make_tuple(
            -static_cast<int64_t>(comm_strength[c] * 1000),
            -static_cast<int64_t>(vertex_strength[v]),
            v
        );
    }
    __gnu_parallel::sort(order.begin(), order.end());
    
    ordered_vertices.resize(num_vertices);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_vertices; ++i) {
        ordered_vertices[i] = std::get<2>(order[i]);
    }
}

/**
 * Local-moving on the community graph (virtual aggregation).
 * 
 * Performs local-moving phase on the aggregated community graph where each
 * node represents a community from the previous level. This allows hierarchical
 * refinement without materializing the full aggregated graph.
 * 
 * @tparam K Community ID type
 * @tparam W Weight type
 * @param comm_graph Community adjacency: comm -> (neighbor_comm -> edge_weight)
 * @param comm_weight Total weight of each community
 * @param M Total edge weight in graph
 * @param R Resolution parameter
 * @param max_iterations Maximum iterations for local moving
 * @param tolerance Convergence tolerance (unused currently)
 * @return Mapping from old community ID to new super-community ID
 */
template <typename K, typename W>
std::unordered_map<K, K> communityLocalMove(
    const std::unordered_map<K, std::unordered_map<K, W>>& comm_graph,
    const std::unordered_map<K, W>& comm_weight,
    double M, double R, int max_iterations, double tolerance) {
    
    // Get list of communities
    std::vector<K> comms;
    for (auto& [c, _] : comm_weight) {
        comms.push_back(c);
    }
    
    // Initialize each community as its own super-community
    std::unordered_map<K, K> super_comm;
    std::unordered_map<K, W> super_weight;
    for (K c : comms) {
        super_comm[c] = c;
        auto it = comm_weight.find(c);
        super_weight[c] = (it != comm_weight.end()) ? it->second : W(0);
    }
    
    // Iterate
    for (int iter = 0; iter < max_iterations; ++iter) {
        int moves = 0;
        
        for (K c : comms) {
            K d = super_comm[c];  // Current super-community
            
            auto kc_it = comm_weight.find(c);
            W kc = (kc_it != comm_weight.end()) ? kc_it->second : W(0);
            
            // Scan neighbor super-communities
            std::unordered_map<K, W> neighbor_weight;
            W kc_to_d = W(0);
            
            auto it = comm_graph.find(c);
            if (it != comm_graph.end()) {
                for (auto& [nc, w] : it->second) {
                    auto sc_it = super_comm.find(nc);
                    if (sc_it != super_comm.end()) {
                        K snc = sc_it->second;
                        neighbor_weight[snc] += w;
                        if (snc == d) kc_to_d += w;
                    }
                }
            }
            
            // Find best super-community
            K best_sc = d;
            W best_delta = W(0);
            
            for (auto& [sc, kc_to_sc] : neighbor_weight) {
                if (sc == d) continue;
                
                auto sw_sc_it = super_weight.find(sc);
                auto sw_d_it = super_weight.find(d);
                W sigma_sc = (sw_sc_it != super_weight.end()) ? sw_sc_it->second : W(0);
                W sigma_d = (sw_d_it != super_weight.end()) ? sw_d_it->second : W(0);
                
                W delta = graphbrew::leiden::gveDeltaModularity<W>(
                    kc_to_sc, kc_to_d, kc, sigma_sc, sigma_d, static_cast<W>(M), static_cast<W>(R));
                
                if (delta > best_delta) {
                    best_delta = delta;
                    best_sc = sc;
                }
            }
            
            // Move if positive gain
            if (best_sc != d) {
                super_weight[d] -= kc;
                super_weight[best_sc] += kc;
                super_comm[c] = best_sc;
                moves++;
            }
        }
        
        if (moves == 0) break;
    }
    
    return super_comm;
}

// ============================================================================
// GVE-LEIDEN LOCAL MOVING PHASE
// ============================================================================

/**
 * @brief GVE-Leiden Local-Moving Phase (Algorithm 2)
 * 
 * Iteratively moves vertices to communities that maximize modularity.
 * Uses flag-based vertex pruning for efficiency.
 * 
 * @tparam K Community ID type
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param vcom Community membership (updated)
 * @param ctot Community total weight (updated)
 * @param vaff Vertex affected flag (updated)
 * @param vtot Vertex total weight
 * @param num_nodes Number of vertices
 * @param g CSR graph
 * @param graph_is_symmetric Whether graph is symmetric
 * @param M Total edge weight
 * @param R Resolution parameter
 * @param L Maximum iterations
 * @param tolerance Convergence tolerance
 * @return Number of iterations performed
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
int gveLeidenLocalMoveCSR(
    std::vector<K>& vcom,
    std::vector<W>& ctot,
    std::vector<char>& vaff,
    const std::vector<W>& vtot,
    const int64_t num_nodes,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric,
    double M, double R, int L, double tolerance) {
    
    int iterations = 0;
    W total_delta = W(0);
    
    // Thread-local hashtables for scanning communities
    const int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<K, W>> thread_hash(num_threads);
    
    for (int iter = 0; iter < L; ++iter) {
        total_delta = W(0);
        int moves_this_iter = 0;
        
        #pragma omp parallel reduction(+:total_delta, moves_this_iter)
        {
            int tid = omp_get_thread_num();
            auto& hash = thread_hash[tid];
            
            #pragma omp for schedule(dynamic, 2048)
            for (int64_t u = 0; u < num_nodes; ++u) {
                if (!vaff[u]) continue;
                
                K d = vcom[u];  // Current community
                W ku = vtot[u]; // Vertex total weight
                
                // Clear and scan ALL edges connected to u
                hash.clear();
                W ku_to_d = ::scanVertexEdges<K, W, NodeID_T, DestID_T>(
                    static_cast<NodeID_T>(u), vcom.data(), hash, d, g, graph_is_symmetric);
                
                // Find best community to move to
                K best_c = d;
                W best_delta = W(0);
                
                for (auto& [c, ku_to_c] : hash) {
                    if (c == d) continue;
                    
                    W sigma_c = ctot[c];
                    W sigma_d = ctot[d];
                    
                    W delta = graphbrew::leiden::gveDeltaModularity<W>(
                        ku_to_c, ku_to_d, ku, sigma_c, sigma_d, static_cast<W>(M), static_cast<W>(R));
                    
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_c = c;
                    }
                }
                
                // Move vertex if positive gain
                if (best_c != d) {
                    // Update community weights atomically
                    #pragma omp atomic
                    ctot[d] -= ku;
                    #pragma omp atomic
                    ctot[best_c] += ku;
                    
                    vcom[u] = best_c;
                    
                    // Mark neighbors as affected
                    ::markNeighborsAffected<NodeID_T, DestID_T>(
                        static_cast<NodeID_T>(u), vaff, g, graph_is_symmetric);
                    
                    total_delta += best_delta;
                    moves_this_iter++;
                }
                
                vaff[u] = 0;
            }
        }
        
        iterations++;
        
        // Check convergence
        if (moves_this_iter == 0 || total_delta <= tolerance) break;
    }
    
    return iterations;
}

/**
 * @brief GVE-Leiden Refinement Phase (Algorithm 3)
 * 
 * Only isolated vertices (sole occupant of their community) can move.
 * Moves are constrained within community bounds from local-moving phase.
 * 
 * @tparam K Community ID type
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param vcom Community membership (updated)
 * @param ctot Community total weight (updated)
 * @param vaff Vertex affected flag (updated)
 * @param vcob Community bounds (from local-moving)
 * @param vtot Vertex total weight
 * @param num_nodes Number of vertices
 * @param g CSR graph
 * @param graph_is_symmetric Whether graph is symmetric
 * @param M Total edge weight
 * @param R Resolution parameter
 * @return Number of moves made
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
int gveLeidenRefineCSR(
    std::vector<K>& vcom,
    std::vector<W>& ctot,
    std::vector<char>& vaff,
    const std::vector<K>& vcob,
    const std::vector<W>& vtot,
    const int64_t num_nodes,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric,
    double M, double R) {
    
    int moves = 0;
    
    // Thread-local hashtables
    const int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<K, W>> thread_hash(num_threads);
    
    // Single pass refinement (per paper)
    #pragma omp parallel reduction(+:moves)
    {
        int tid = omp_get_thread_num();
        auto& hash = thread_hash[tid];
        
        #pragma omp for schedule(dynamic, 2048)
        for (int64_t u = 0; u < num_nodes; ++u) {
            K d = vcom[u];
            K b = vcob[u];
            W ku = vtot[u];
            W sigma_d = ctot[d];
            
            // Only isolated vertices can move
            W expected_d = ku;
            W actual_d;
            
            #pragma omp atomic read
            actual_d = ctot[d];
            
            if (actual_d > expected_d * 1.001) {
                continue;
            }
            
            // Scan communities within the same bound
            hash.clear();
            W ku_to_d = W(0);
            
            // Scan out-neighbors within bounds
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
                
                if (vcob[v] != b) continue;
                
                K c = vcom[v];
                hash[c] += w;
                if (c == d) ku_to_d += w;
            }
            
            // Scan in-neighbors within bounds (for non-symmetric)
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
                    
                    if (vcob[v] != b) continue;
                    
                    K c = vcom[v];
                    hash[c] += w;
                    if (c == d) ku_to_d += w;
                }
            }
            
            // Find best community within bounds
            K best_c = d;
            W best_delta = W(0);
            
            for (auto& [c, ku_to_c] : hash) {
                if (c == d) continue;
                
                W sigma_c;
                #pragma omp atomic read
                sigma_c = ctot[c];
                
                W delta = graphbrew::leiden::gveDeltaModularity<W>(
                    ku_to_c, ku_to_d, ku, sigma_c, sigma_d, static_cast<W>(M), static_cast<W>(R));
                
                if (delta > best_delta) {
                    best_delta = delta;
                    best_c = c;
                }
            }
            
            // Move if positive gain and still isolated
            if (best_c != d) {
                W old_sigma_d;
                #pragma omp atomic capture
                {
                    old_sigma_d = ctot[d];
                    ctot[d] -= ku;
                }
                
                if (old_sigma_d > ku * 1.001) {
                    #pragma omp atomic
                    ctot[d] += ku;
                    continue;
                }
                
                #pragma omp atomic
                ctot[best_c] += ku;
                
                vcom[u] = best_c;
                moves++;
                
                ::markNeighborsAffected<NodeID_T, DestID_T>(
                    static_cast<NodeID_T>(u), vaff, g, graph_is_symmetric);
            }
        }
    }
    
    return moves;
}

/**
 * @brief Compute community-to-community edge weights for virtual aggregation
 * 
 * @tparam K Community ID type
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type
 * @param comm_graph Output: community adjacency map
 * @param comm_weight Output: total weight per community
 * @param vcom Community assignments
 * @param vtot Vertex total weights
 * @param num_nodes Number of vertices
 * @param g CSR graph
 * @param graph_is_symmetric Whether graph is symmetric
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
void computeCommunityGraphCSR(
    std::unordered_map<K, std::unordered_map<K, W>>& comm_graph,
    std::unordered_map<K, W>& comm_weight,
    const std::vector<K>& vcom,
    const std::vector<W>& vtot,
    const int64_t num_nodes,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric) {
    
    comm_graph.clear();
    comm_weight.clear();
    
    // Compute community total weights
    for (int64_t v = 0; v < num_nodes; ++v) {
        comm_weight[vcom[v]] += vtot[v];
    }
    
    // Compute community-to-community edges
    for (int64_t u = 0; u < num_nodes; ++u) {
        K cu = vcom[u];
        
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
            
            K cv = vcom[v];
            if (cu != cv) {
                comm_graph[cu][cv] += w;
            }
        }
        
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
                
                K cv = vcom[v];
                if (cu != cv) {
                    comm_graph[cu][cv] += w;
                }
            }
        }
    }
}

// ============================================================================
// OPTIMIZED GVE-LEIDEN FUNCTIONS (Cache-Optimized Variants)
// ============================================================================

/**
 * @brief Optimized local move scan using flat array instead of hash map.
 * Uses prefetching for better cache performance.
 *
 * @tparam K Community ID type
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type (may include weight)
 * @tparam WeightT_ Weight type for weighted graphs
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T, typename WeightT_ = int32_t>
inline W gveOptScanVertex(
    NodeID_T u,
    const K* __restrict__ vcom,
    W* __restrict__ comm_weights,  // Pre-allocated flat array [num_communities]
    K* __restrict__ touched_comms, // List of touched communities
    int& num_touched,
    K d,  // Current community
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric) {
    
    W ku_to_d = W(0);
    num_touched = 0;
    
    // Scan out-neighbors with prefetching
    auto out_begin = g.out_neigh(u).begin();
    auto out_end = g.out_neigh(u).end();
    const int PREFETCH_DIST = 8;
    
    for (auto it = out_begin; it != out_end; ++it) {
        // Prefetch ahead
        if (it + PREFETCH_DIST < out_end) {
            NodeID_T prefetch_v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                prefetch_v = *(it + PREFETCH_DIST);
            } else {
                prefetch_v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*(it + PREFETCH_DIST)).v;
            }
            __builtin_prefetch(&vcom[prefetch_v], 0, 3);
        }
        
        NodeID_T v;
        W w;
        if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
            v = *it;
            w = W(1);
        } else {
            v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).v;
            w = static_cast<W>(static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).w);
        }
        
        K c = vcom[v];
        if (comm_weights[c] == W(0)) {
            touched_comms[num_touched++] = c;
        }
        comm_weights[c] += w;
        if (c == d) ku_to_d += w;
    }
    
    // For non-symmetric graphs, also scan in-neighbors
    if (!graph_is_symmetric) {
        auto in_begin = g.in_neigh(u).begin();
        auto in_end = g.in_neigh(u).end();
        
        for (auto it = in_begin; it != in_end; ++it) {
            if (it + PREFETCH_DIST < in_end) {
                NodeID_T prefetch_v;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    prefetch_v = *(it + PREFETCH_DIST);
                } else {
                    prefetch_v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*(it + PREFETCH_DIST)).v;
                }
                __builtin_prefetch(&vcom[prefetch_v], 0, 3);
            }
            
            NodeID_T v;
            W w;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = *it;
                w = W(1);
            } else {
                v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).v;
                w = static_cast<W>(static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).w);
            }
            
            K c = vcom[v];
            if (comm_weights[c] == W(0)) {
                touched_comms[num_touched++] = c;
            }
            comm_weights[c] += w;
            if (c == d) ku_to_d += w;
        }
    }
    
    return ku_to_d;
}

/**
 * @brief Optimized GVE-Leiden Local-Moving Phase
 * Uses flat arrays instead of hash maps for better cache performance.
 *
 * @tparam K Community ID type
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @tparam WeightT_ Weight type for weighted graphs
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T, typename WeightT_ = int32_t>
int gveOptLocalMove(
    std::vector<K>& vcom,           // Community membership (updated)
    std::vector<W>& ctot,           // Community total weight (updated)
    std::vector<char>& vaff,        // Vertex affected flag (updated)
    const std::vector<W>& vtot,     // Vertex total weight
    const int64_t num_nodes,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric,
    double M, double R, int L, double tolerance) {
    
    using namespace graphbrew::leiden;
    
    int iterations = 0;
    W total_delta = W(0);
    
    const int num_threads = omp_get_max_threads();
    
    // Thread-local flat arrays for scanning (much faster than hash maps)
    std::vector<std::vector<W>> thread_comm_weights(num_threads);
    std::vector<std::vector<K>> thread_touched_comms(num_threads);
    
    // Pre-allocate for each thread
    for (int t = 0; t < num_threads; ++t) {
        thread_comm_weights[t].resize(num_nodes, W(0));
        thread_touched_comms[t].resize(g.num_edges() / num_threads + 1000);
    }
    
    for (int iter = 0; iter < L; ++iter) {
        total_delta = W(0);
        int moves_this_iter = 0;
        
        #pragma omp parallel reduction(+:total_delta, moves_this_iter)
        {
            int tid = omp_get_thread_num();
            W* comm_weights = thread_comm_weights[tid].data();
            K* touched_comms = thread_touched_comms[tid].data();
            
            #pragma omp for schedule(guided, 1024)
            for (int64_t u = 0; u < num_nodes; ++u) {
                if (!vaff[u]) continue;
                
                K d = vcom[u];  // Current community
                W ku = vtot[u]; // Vertex total weight
                
                // Scan neighbors using flat array
                int num_touched = 0;
                W ku_to_d = gveOptScanVertex<K, W, NodeID_T, DestID_T, WeightT_>(
                    u, vcom.data(), comm_weights, touched_comms, num_touched,
                    d, g, graph_is_symmetric);
                
                // Find best community
                K best_c = d;
                W best_delta = W(0);
                
                for (int i = 0; i < num_touched; ++i) {
                    K c = touched_comms[i];
                    if (c == d) continue;
                    
                    W ku_to_c = comm_weights[c];
                    W sigma_c = ctot[c];
                    W sigma_d = ctot[d];
                    
                    W delta = gveDeltaModularity<W>(ku_to_c, ku_to_d, ku, sigma_c, sigma_d, static_cast<W>(M), static_cast<W>(R));
                    
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_c = c;
                    }
                }
                
                // Clear touched communities (reset for next vertex)
                for (int i = 0; i < num_touched; ++i) {
                    comm_weights[touched_comms[i]] = W(0);
                }
                
                // Move vertex if positive gain
                if (best_c != d) {
                    #pragma omp atomic
                    ctot[d] -= ku;
                    #pragma omp atomic
                    ctot[best_c] += ku;
                    
                    vcom[u] = best_c;
                    
                    // Mark neighbors as affected
                    markNeighborsAffected<NodeID_T, DestID_T>(u, vaff, g, graph_is_symmetric);
                    
                    total_delta += best_delta;
                    moves_this_iter++;
                }
                
                vaff[u] = 0;
            }
        }
        
        iterations++;
        if (moves_this_iter == 0 || total_delta <= tolerance) break;
    }
    
    return iterations;
}

/**
 * @brief Optimized GVE-Leiden Refinement Phase
 * Uses flat arrays and prefetching for better cache performance.
 *
 * @tparam K Community ID type
 * @tparam W Weight type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @tparam WeightT_ Weight type for weighted graphs
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T, typename WeightT_ = int32_t>
int gveOptRefine(
    std::vector<K>& vcom,           // Community membership (updated)
    std::vector<W>& ctot,           // Community total weight (updated)
    std::vector<char>& vaff,        // Vertex affected flag (updated)
    const std::vector<K>& vcob,     // Community bounds (from local-moving)
    const std::vector<W>& vtot,     // Vertex total weight
    const int64_t num_nodes,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    bool graph_is_symmetric,
    double M, double R) {
    
    using namespace graphbrew::leiden;
    
    int moves = 0;
    
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<W>> thread_comm_weights(num_threads);
    std::vector<std::vector<K>> thread_touched_comms(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_comm_weights[t].resize(num_nodes, W(0));
        thread_touched_comms[t].resize(g.num_edges() / num_threads + 1000);
    }
    
    #pragma omp parallel reduction(+:moves)
    {
        int tid = omp_get_thread_num();
        W* comm_weights = thread_comm_weights[tid].data();
        K* touched_comms = thread_touched_comms[tid].data();
        
        #pragma omp for schedule(guided, 1024)
        for (int64_t u = 0; u < num_nodes; ++u) {
            K d = vcom[u];
            K b = vcob[u];
            W ku = vtot[u];
            
            // Check if isolated
            W actual_d;
            #pragma omp atomic read
            actual_d = ctot[d];
            
            if (actual_d > ku * 1.001) continue;
            
            // Scan communities within bounds
            int num_touched = 0;
            W ku_to_d = W(0);
            
            // Scan with prefetching
            auto out_begin = g.out_neigh(u).begin();
            auto out_end = g.out_neigh(u).end();
            const int PREFETCH_DIST = 8;
            
            for (auto it = out_begin; it != out_end; ++it) {
                if (it + PREFETCH_DIST < out_end) {
                    NodeID_T prefetch_v;
                    if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                        prefetch_v = *(it + PREFETCH_DIST);
                    } else {
                        prefetch_v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*(it + PREFETCH_DIST)).v;
                    }
                    __builtin_prefetch(&vcom[prefetch_v], 0, 3);
                    __builtin_prefetch(&vcob[prefetch_v], 0, 3);
                }
                
                NodeID_T v;
                W w;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    v = *it;
                    w = W(1);
                } else {
                    v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).v;
                    w = static_cast<W>(static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).w);
                }
                
                if (vcob[v] != b) continue;
                
                K c = vcom[v];
                if (comm_weights[c] == W(0)) {
                    touched_comms[num_touched++] = c;
                }
                comm_weights[c] += w;
                if (c == d) ku_to_d += w;
            }
            
            if (!graph_is_symmetric) {
                auto in_begin = g.in_neigh(u).begin();
                auto in_end = g.in_neigh(u).end();
                
                for (auto it = in_begin; it != in_end; ++it) {
                    if (it + PREFETCH_DIST < in_end) {
                        NodeID_T prefetch_v;
                        if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                            prefetch_v = *(it + PREFETCH_DIST);
                        } else {
                            prefetch_v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*(it + PREFETCH_DIST)).v;
                        }
                        __builtin_prefetch(&vcom[prefetch_v], 0, 3);
                        __builtin_prefetch(&vcob[prefetch_v], 0, 3);
                    }
                    
                    NodeID_T v;
                    W w;
                    if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                        v = *it;
                        w = W(1);
                    } else {
                        v = static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).v;
                        w = static_cast<W>(static_cast<NodeWeight<NodeID_T, WeightT_>>(*it).w);
                    }
                    
                    if (vcob[v] != b) continue;
                    
                    K c = vcom[v];
                    if (comm_weights[c] == W(0)) {
                        touched_comms[num_touched++] = c;
                    }
                    comm_weights[c] += w;
                    if (c == d) ku_to_d += w;
                }
            }
            
            // Find best community within bounds
            K best_c = d;
            W best_delta = W(0);
            W sigma_d = actual_d;
            
            for (int i = 0; i < num_touched; ++i) {
                K c = touched_comms[i];
                if (c == d) continue;
                
                W ku_to_c = comm_weights[c];
                W sigma_c;
                #pragma omp atomic read
                sigma_c = ctot[c];
                
                W delta = gveDeltaModularity<W>(ku_to_c, ku_to_d, ku, sigma_c, sigma_d, static_cast<W>(M), static_cast<W>(R));
                
                if (delta > best_delta) {
                    best_delta = delta;
                    best_c = c;
                }
            }
            
            // Clear touched communities
            for (int i = 0; i < num_touched; ++i) {
                comm_weights[touched_comms[i]] = W(0);
            }
            
            // Move if positive gain and still isolated
            if (best_c != d) {
                W old_sigma_d;
                #pragma omp atomic capture
                {
                    old_sigma_d = ctot[d];
                    ctot[d] -= ku;
                }
                
                if (old_sigma_d > ku * 1.001) {
                    #pragma omp atomic
                    ctot[d] += ku;
                    continue;
                }
                
                #pragma omp atomic
                ctot[best_c] += ku;
                
                vcom[u] = best_c;
                moves++;
                
                markNeighborsAffected<NodeID_T, DestID_T>(u, vaff, g, graph_is_symmetric);
            }
        }
    }
    
    return moves;
}

/**
 * GVE-Leiden algorithm for community detection on CSR graphs.
 * 
 * This implements the complete Leiden algorithm with:
 * 1. LOCAL-MOVING phase - vertices move to maximize modularity
 * 2. REFINEMENT phase - ensures well-connected communities
 * 3. AGGREGATION phase - builds super-graph for next pass
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam W Weight type (should be double)
 * @tparam NodeID_T Node ID type from graph
 * @tparam DestID_T Destination ID type (may include weights)
 * 
 * @param g The input CSR graph (must be weighted-enabled)
 * @param resolution Resolution parameter for modularity (default 1.0)
 * @param tolerance Convergence tolerance (default 1e-2)
 * @param aggregation_tolerance Stop if progress < this (default 0.8)
 * @param tolerance_drop Factor to decrease tolerance each pass (default 10.0)
 * @param max_iterations Max iterations per local-moving phase (default 20)
 * @param max_passes Maximum number of Leiden passes (default 10)
 * 
 * @return GVELeidenResult containing community assignments and modularity
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVELeidenResult<K> GVELeidenCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = 10) {
    
    const int64_t num_nodes = g.num_nodes();
    
    // Detect if graph is symmetric
    bool graph_is_symmetric = !g.directed();
    
    // Compute M (total edge weight)
    const int64_t num_edges_stored = g.num_edges();
    // Guard against zero edges (degenerate graph)
    const double M = std::max(1.0, static_cast<double>(num_edges_stored));
    
    GVELeidenResult<K> result;
    result.total_iterations = 0;
    result.total_passes = 0;
    
    // Initialize data structures for original graph level
    std::vector<K> vcom(num_nodes);      // Current community
    std::vector<K> vcob(num_nodes);      // Community bounds (from local-moving for refinement)
    std::vector<W> vtot(num_nodes);      // Vertex total weight
    std::vector<W> ctot(num_nodes);      // Community total weight
    std::vector<char> vaff(num_nodes, 1); // Vertex affected flag
    
    // Initialize in parallel
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vcom[v] = static_cast<K>(v);
        W vtot_v = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
        vtot[v] = vtot_v;
        ctot[v] = vtot_v;
    }
    
    double current_tolerance = tolerance;
    size_t prev_communities = num_nodes;
    
    // Main Leiden loop (Algorithm 1 from paper)
    for (int pass = 0; pass < max_passes; ++pass) {
        
        // ================================================================
        // PHASE 1: LOCAL-MOVING (Algorithm 2)
        // Vertices move to communities that maximize modularity
        // ================================================================
        int local_iters = gveLeidenLocalMoveCSR<K, W, NodeID_T, DestID_T>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric,
            M, resolution, max_iterations, current_tolerance);
        
        result.total_iterations += local_iters;
        
        // Store community bounds (result of local-moving phase)
        // These bounds constrain refinement phase
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcob[v] = vcom[v];
        }
        
        // Count communities after local-moving
        std::unordered_set<K> unique_comms_local;
        for (int64_t v = 0; v < num_nodes; ++v) {
            unique_comms_local.insert(vcom[v]);
        }
        size_t num_communities_after_local = unique_comms_local.size();
        
        // ================================================================
        // PHASE 2: REFINEMENT (Optional - only on first pass for Leiden guarantee)
        // For subsequent passes, skip refinement for better aggregation
        // ================================================================
        
        std::vector<K> vcom_refined(num_nodes);
        size_t num_refined_communities;
        int refine_moves = 0;
        
        if (pass == 0) {
            // First pass: do full refinement for well-connected communities
            std::vector<W> ctot_refined(num_nodes);
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = static_cast<K>(v);  // Each vertex is its own refined community
                ctot_refined[v] = vtot[v];
            }
            
            // Reset affected flags for refinement
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vaff[v] = 1;
            }
            
            // Refinement iterations - vertices can only move within their community bound
            refine_moves = gveLeidenRefineCSR<K, W, NodeID_T, DestID_T>(
                vcom_refined, ctot_refined, vaff, vcob, vtot,
                num_nodes, g, graph_is_symmetric, M, resolution);
            
            // Renumber refined communities contiguously
            // First find max community ID
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            // Build renumber table as vector for thread-safe parallel access
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        } else {
            // Subsequent passes: vcom contains community IDs from previous pass
            // These IDs are sparse (not contiguous), so we need to renumber
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = vcom[v];
            }
            
            // Find max community ID in vcom
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            // Renumber to contiguous IDs
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        }
        
        // Map refined communities to their bound communities
        // This creates the hierarchical structure
        std::vector<K> refined_to_bound(num_refined_communities, K(-1));
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            K bc = vcob[v];
            if (refined_to_bound[rc] == K(-1)) {
                refined_to_bound[rc] = bc;
            }
        }
        
        // Build super-graph using compact representation
        // Each community stores its total weight
        std::vector<W> super_weight(num_refined_communities, W(0));
        
        // Compute community weights in parallel
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom_refined[v];
            #pragma omp atomic
            super_weight[c] += vtot[v];
        }
        
        // Build edge list for super-graph (more memory efficient for sparse graphs)
        // Format: vector of (source_comm, dest_comm, weight)
        struct SuperEdge {
            K src, dst;
            W weight;
        };
        
        // Thread-local edge lists
        const int num_threads = omp_get_max_threads();
        std::vector<std::vector<SuperEdge>> thread_edges(num_threads);
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_edges = thread_edges[tid];
            local_edges.reserve(g.num_edges() / num_threads / 10);  // Estimate
            
            // Thread-local hash for deduplication within chunk
            std::unordered_map<uint64_t, W> edge_hash;
            
            #pragma omp for schedule(dynamic, 4096)
            for (int64_t u = 0; u < num_nodes; ++u) {
                K cu = vcom_refined[u];  // Already renumbered
                
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
                    
                    K cv = vcom_refined[v];  // Already renumbered
                    if (cu != cv && cu < num_refined_communities && cv < num_refined_communities) {
                        uint64_t key = (static_cast<uint64_t>(cu) << 32) | cv;
                        edge_hash[key] += w;
                    }
                }
            }
            
            // Convert hash to vector
            for (auto& [key, w] : edge_hash) {
                K src = static_cast<K>(key >> 32);
                K dst = static_cast<K>(key & 0xFFFFFFFF);
                local_edges.push_back({src, dst, w});
            }
        }
        
        // Merge into global edge hash
        std::unordered_map<uint64_t, W> global_edge_hash;
        for (int t = 0; t < num_threads; ++t) {
            for (auto& e : thread_edges[t]) {
                uint64_t key = (static_cast<uint64_t>(e.src) << 32) | e.dst;
                global_edge_hash[key] += e.weight;
            }
        }
        
        // Convert to adjacency list for super-graph
        std::vector<std::vector<std::pair<K, W>>> super_adj(num_refined_communities);
        for (auto& [key, w] : global_edge_hash) {
            K src = static_cast<K>(key >> 32);
            K dst = static_cast<K>(key & 0xFFFFFFFF);
            if (src < num_refined_communities) {
                super_adj[src].emplace_back(dst, w);
            }
        }
        
        // ================================================================
        // LOCAL-MOVING ON SUPER-GRAPH
        // Merge refined communities
        // ================================================================
        
        std::vector<K> super_comm(num_refined_communities);
        std::vector<W> super_ctot(num_refined_communities);
        
        #pragma omp parallel for
        for (size_t c = 0; c < num_refined_communities; ++c) {
            super_comm[c] = c;
            super_ctot[c] = super_weight[c];
        }
        
        // Iterate local-moving on super-graph (NO bounds constraint - allow full merging)
        for (int iter = 0; iter < max_iterations; ++iter) {
            int moves = 0;
            
            for (size_t c = 0; c < num_refined_communities; ++c) {
                K d = super_comm[c];  // Current super-community
                W kc = super_weight[c];
                W sigma_d = super_ctot[d];
                
                // Scan ALL neighbor super-communities (no bounds restriction)
                std::unordered_map<K, W> neighbor_sc_weight;
                W kc_to_d = W(0);
                
                for (auto& [nc, w] : super_adj[c]) {
                    K snc = super_comm[nc];
                    neighbor_sc_weight[snc] += w;
                    if (snc == d) kc_to_d += w;
                }
                
                // Find best super-community (no bounds restriction)
                K best_sc = d;
                W best_delta = W(0);
                
                for (auto& [sc, kc_to_sc] : neighbor_sc_weight) {
                    if (sc == d) continue;
                    
                    W sigma_sc = super_ctot[sc];
                    W delta = graphbrew::leiden::gveDeltaModularity<W>(kc_to_sc, kc_to_d, kc, sigma_sc, sigma_d, M, resolution);
                    
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_sc = sc;
                    }
                }
                
                // Move if positive gain
                if (best_sc != d) {
                    super_ctot[d] -= kc;
                    super_ctot[best_sc] += kc;
                    super_comm[c] = best_sc;
                    moves++;
                }
            }
            
            if (moves == 0) break;
        }
        
        // Map back to original vertices
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            vcom[v] = super_comm[rc];
        }
        
        // Renumber final communities contiguously
        // First find max community ID for vector sizing
        K max_final_comm = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_final_comm = std::max(max_final_comm, vcom[v]);
        }
        
        // Build renumber table as vector for thread-safe parallel access
        std::vector<K> final_renumber(max_final_comm + 1, K(-1));
        K next_final_id = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom[v];
            if (final_renumber[c] == K(-1)) {
                final_renumber[c] = next_final_id++;
            }
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcom[v] = final_renumber[vcom[v]];
        }
        
        size_t num_final_communities = static_cast<size_t>(next_final_id);
        
        // Store pass result
        std::vector<K> pass_community(num_nodes);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            pass_community[v] = vcom[v];
        }
        result.community_per_pass.push_back(pass_community);
        result.total_passes++;
        
        printf("  Pass %d: local=%d iters, refine=%d moves, communities: %zu -> %zu -> %zu\n",
               pass + 1, local_iters, refine_moves,
               prev_communities, num_refined_communities, num_final_communities);
        
        // Check convergence
        if (num_final_communities >= prev_communities || num_final_communities == 1) {
            break;
        }
        
        // Check aggregation tolerance - stop if not enough progress
        double progress = static_cast<double>(num_final_communities) / prev_communities;
        if (progress >= aggregation_tolerance) {
            break;
        }
        
        prev_communities = num_final_communities;
        
        // Recompute ctot for next pass
        std::fill(ctot.begin(), ctot.end(), W(0));
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            #pragma omp atomic
            ctot[vcom[v]] += vtot[v];
        }
        
        // Reset affected flags for next pass
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vaff[v] = 1;
        }
        
        current_tolerance /= tolerance_drop;
    }
    
    // Use final community assignment
    if (!result.community_per_pass.empty()) {
        result.final_community = result.community_per_pass.back();
    } else {
        result.final_community.resize(num_nodes);
        std::iota(result.final_community.begin(), result.final_community.end(), 0);
        result.community_per_pass.push_back(result.final_community);
    }
    
    // Compute final modularity using the standard formula
    result.modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, result.final_community, resolution);
    
    printf("GVELeiden: %d passes, %d total iterations, modularity=%.6f\n",
           result.total_passes, result.total_iterations, result.modularity);
    
    return result;
}

/**
 * GVELeidenOpt - Optimized GVE-Leiden with cache optimizations
 * 
 * Key optimizations over standard GVELeidenCSR:
 * - Flat arrays instead of hash maps for community scanning
 * - Prefetching for community lookups
 * - Guided scheduling for better load balancing
 * - Optimized super-graph construction with sorted edge merging
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam W Weight type (should be double)
 * @tparam NodeID_T Node ID type from graph
 * @tparam DestID_T Destination ID type (may include weights)
 * 
 * @param g The input CSR graph (must be weighted-enabled)
 * @param resolution Resolution parameter for modularity (default 1.0)
 * @param tolerance Convergence tolerance (default 1e-2)
 * @param aggregation_tolerance Stop if progress < this (default 0.8)
 * @param tolerance_drop Factor to decrease tolerance each pass (default 10.0)
 * @param max_iterations Max iterations per local-moving phase (default 20)
 * @param max_passes Maximum number of Leiden passes (default 10)
 * @param skip_refine Skip refinement phase for speed (default false)
 * 
 * @return GVELeidenResult containing community assignments and modularity
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVELeidenResult<K> GVELeidenOptCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = 10,
    bool skip_refine = false) {
    
    const int64_t num_nodes = g.num_nodes();
    bool graph_is_symmetric = !g.directed();
    const int64_t num_edges_stored = g.num_edges();
    const double M = std::max(1.0, static_cast<double>(num_edges_stored));  // Guard against zero edges
    
    GVELeidenResult<K> result;
    result.total_iterations = 0;
    result.total_passes = 0;
    
    // Initialize data structures
    std::vector<K> vcom(num_nodes);
    std::vector<K> vcob(num_nodes);
    std::vector<W> vtot(num_nodes);
    std::vector<W> ctot(num_nodes);
    std::vector<char> vaff(num_nodes, 1);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vcom[v] = static_cast<K>(v);
        W vtot_v = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
        vtot[v] = vtot_v;
        ctot[v] = vtot_v;
    }
    
    double current_tolerance = tolerance;
    size_t prev_communities = num_nodes;
    
    for (int pass = 0; pass < max_passes; ++pass) {
        
        // PHASE 1: LOCAL-MOVING (optimized)
        int local_iters = gveOptLocalMove<K, W, NodeID_T, DestID_T>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric,
            M, resolution, max_iterations, current_tolerance);
        
        result.total_iterations += local_iters;
        
        // Store community bounds
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcob[v] = vcom[v];
        }
        
        // Count communities
        std::vector<char> comm_exists(num_nodes, 0);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            comm_exists[vcom[v]] = 1;
        }
        size_t num_communities_after_local = 0;
        for (int64_t c = 0; c < num_nodes; ++c) {
            if (comm_exists[c]) num_communities_after_local++;
        }
        
        // PHASE 2: REFINEMENT (optimized) - can be skipped for speed
        std::vector<K> vcom_refined(num_nodes);
        size_t num_refined_communities;
        
        if (!skip_refine && pass == 0) {
            std::vector<W> ctot_refined(num_nodes);
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = static_cast<K>(v);
                ctot_refined[v] = vtot[v];
                vaff[v] = 1;
            }
            
            gveOptRefine<K, W, NodeID_T, DestID_T>(
                vcom_refined, ctot_refined, vaff, vcob, vtot,
                num_nodes, g, graph_is_symmetric, M, resolution);
            
            // Renumber refined communities
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        } else {
            // Skip refinement: just renumber communities
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = vcom[v];
            }
            
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        }
        
        // Map refined to bound
        std::vector<K> refined_to_bound(num_refined_communities, K(-1));
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            if (refined_to_bound[rc] == K(-1)) {
                refined_to_bound[rc] = vcob[v];
            }
        }
        
        // PHASE 3: AGGREGATION (optimized with sorted edge merging)
        std::vector<W> super_weight(num_refined_communities, W(0));
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom_refined[v];
            #pragma omp atomic
            super_weight[c] += vtot[v];
        }
        
        // Build super-graph edges using sorted merge (more cache-friendly)
        const int num_threads = omp_get_max_threads();
        
        // Each thread collects edges
        struct SuperEdge {
            K src, dst;
            W weight;
            bool operator<(const SuperEdge& o) const {
                return src < o.src || (src == o.src && dst < o.dst);
            }
        };
        
        std::vector<std::vector<SuperEdge>> thread_edges(num_threads);
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_edges = thread_edges[tid];
            local_edges.reserve(g.num_edges() / num_threads / 8);
            
            #pragma omp for schedule(guided, 2048)
            for (int64_t u = 0; u < num_nodes; ++u) {
                K cu = vcom_refined[u];
                
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
                    
                    K cv = vcom_refined[v];
                    if (cu != cv) {
                        local_edges.push_back({cu, cv, w});
                    }
                }
            }
            
            // Sort local edges for efficient merging
            std::sort(local_edges.begin(), local_edges.end());
            
            // Merge duplicates within thread
            if (!local_edges.empty()) {
                std::vector<SuperEdge> merged;
                merged.reserve(local_edges.size() / 2);
                merged.push_back(local_edges[0]);
                
                for (size_t i = 1; i < local_edges.size(); ++i) {
                    if (local_edges[i].src == merged.back().src && 
                        local_edges[i].dst == merged.back().dst) {
                        merged.back().weight += local_edges[i].weight;
                    } else {
                        merged.push_back(local_edges[i]);
                    }
                }
                local_edges = std::move(merged);
            }
        }
        
        // Merge all thread edges using k-way merge
        std::vector<SuperEdge> all_edges;
        size_t total_edges = 0;
        for (int t = 0; t < num_threads; ++t) {
            total_edges += thread_edges[t].size();
        }
        all_edges.reserve(total_edges);
        
        for (int t = 0; t < num_threads; ++t) {
            all_edges.insert(all_edges.end(), 
                thread_edges[t].begin(), thread_edges[t].end());
        }
        
        // Sort and merge globally
        __gnu_parallel::sort(all_edges.begin(), all_edges.end());
        
        // Build adjacency list
        std::vector<std::vector<std::pair<K, W>>> super_adj(num_refined_communities);
        
        if (!all_edges.empty()) {
            K prev_src = all_edges[0].src;
            K prev_dst = all_edges[0].dst;
            W acc_weight = all_edges[0].weight;
            
            for (size_t i = 1; i < all_edges.size(); ++i) {
                if (all_edges[i].src == prev_src && all_edges[i].dst == prev_dst) {
                    acc_weight += all_edges[i].weight;
                } else {
                    if (prev_src < num_refined_communities) {
                        super_adj[prev_src].emplace_back(prev_dst, acc_weight);
                    }
                    prev_src = all_edges[i].src;
                    prev_dst = all_edges[i].dst;
                    acc_weight = all_edges[i].weight;
                }
            }
            if (prev_src < num_refined_communities) {
                super_adj[prev_src].emplace_back(prev_dst, acc_weight);
            }
        }
        
        // Local-moving on super-graph
        std::vector<K> super_comm(num_refined_communities);
        std::vector<W> super_ctot(num_refined_communities);
        
        #pragma omp parallel for
        for (size_t c = 0; c < num_refined_communities; ++c) {
            super_comm[c] = c;
            super_ctot[c] = super_weight[c];
        }
        
        // Use flat array for super-graph local move
        std::vector<W> sg_comm_weights(num_refined_communities, W(0));
        std::vector<K> sg_touched(num_refined_communities);
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            int moves = 0;
            
            for (size_t c = 0; c < num_refined_communities; ++c) {
                K d = super_comm[c];
                W kc = super_weight[c];
                W sigma_d = super_ctot[d];
                
                // Scan neighbors using flat array
                int num_touched = 0;
                W kc_to_d = W(0);
                
                for (auto& [nc, w] : super_adj[c]) {
                    K snc = super_comm[nc];
                    if (sg_comm_weights[snc] == W(0)) {
                        sg_touched[num_touched++] = snc;
                    }
                    sg_comm_weights[snc] += w;
                    if (snc == d) kc_to_d += w;
                }
                
                // Find best
                K best_sc = d;
                W best_delta = W(0);
                
                for (int i = 0; i < num_touched; ++i) {
                    K sc = sg_touched[i];
                    if (sc == d) continue;
                    
                    W kc_to_sc = sg_comm_weights[sc];
                    W sigma_sc = super_ctot[sc];
                    W delta = graphbrew::leiden::gveDeltaModularity<W>(kc_to_sc, kc_to_d, kc, sigma_sc, sigma_d, M, resolution);
                    
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_sc = sc;
                    }
                }
                
                // Clear
                for (int i = 0; i < num_touched; ++i) {
                    sg_comm_weights[sg_touched[i]] = W(0);
                }
                
                if (best_sc != d) {
                    super_ctot[d] -= kc;
                    super_ctot[best_sc] += kc;
                    super_comm[c] = best_sc;
                    moves++;
                }
            }
            
            if (moves == 0) break;
        }
        
        // Map back to original vertices
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            vcom[v] = super_comm[rc];
        }
        
        // Renumber final communities
        K max_final_comm = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_final_comm = std::max(max_final_comm, vcom[v]);
        }
        
        std::vector<K> final_renumber(max_final_comm + 1, K(-1));
        K next_final_id = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom[v];
            if (final_renumber[c] == K(-1)) {
                final_renumber[c] = next_final_id++;
            }
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcom[v] = final_renumber[vcom[v]];
        }
        
        size_t current_communities = static_cast<size_t>(next_final_id);
        
        // Store dendrogram
        result.community_per_pass.push_back(vcom);
        result.total_passes++;
        
        // Check convergence
        double reduction = static_cast<double>(prev_communities - current_communities) 
                         / static_cast<double>(prev_communities);
        
        if (current_communities >= prev_communities || reduction < (1.0 - aggregation_tolerance)) {
            break;
        }
        
        // Reinitialize for next pass
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            ctot[vcom[v]] = W(0);
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            #pragma omp atomic
            ctot[vcom[v]] += vtot[v];
            vaff[v] = 1;
        }
        
        prev_communities = current_communities;
        current_tolerance = tolerance / tolerance_drop;
    }
    
    // Final community assignment
    result.final_community = vcom;
    
    // Compute modularity
    result.modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, result.final_community, resolution);
    
    printf("GVELeidenOpt: %d passes, %d total iterations, modularity=%.6f\n",
           result.total_passes, result.total_iterations, result.modularity);
    
    return result;
}

/**
 * Build super-graph adjacency list using CSR-style approach
 * Directly builds std::vector<std::vector<std::pair<K, W>>> format
 * Used as drop-in replacement for sort-based aggregation
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
inline void buildSuperGraphAdjList(
    std::vector<std::vector<std::pair<K, W>>>& super_adj,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<K>& vcom,
    size_t num_nodes,
    size_t num_communities) {
    
    const int num_threads = omp_get_max_threads();
    
    // Build community-to-vertices mapping
    std::vector<size_t> coff(num_communities + 1, 0);
    std::vector<K> cedg(num_nodes);
    
    // Count vertices per community
    for (size_t v = 0; v < num_nodes; ++v) {
        coff[vcom[v] + 1]++;
    }
    
    // Exclusive scan to get offsets
    for (size_t c = 1; c <= num_communities; ++c) {
        coff[c] += coff[c - 1];
    }
    
    // Fill vertex lists
    std::vector<size_t> count(num_communities, 0);
    for (size_t v = 0; v < num_nodes; ++v) {
        K c = vcom[v];
        cedg[coff[c] + count[c]++] = static_cast<K>(v);
    }
    
    // Thread-local hashtables for edge aggregation
    std::vector<std::vector<W>> thread_vcout(num_threads);
    std::vector<std::vector<K>> thread_vcs(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_vcout[t].resize(num_communities, W(0));
        thread_vcs[t].reserve(1000);
    }
    
    // Initialize output
    super_adj.clear();
    super_adj.resize(num_communities);
    
    // Aggregate edges community by community (parallel)
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t c = 0; c < num_communities; ++c) {
        int tid = omp_get_thread_num();
        auto& vcs = thread_vcs[tid];
        W* vcout = thread_vcout[tid].data();
        
        vcs.clear();
        
        // Scan all vertices in this community
        for (size_t i = coff[c]; i < coff[c + 1]; ++i) {
            K u = cedg[i];
            
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
                
                K cv = vcom[v];
                if (cv != static_cast<K>(c)) {  // Only inter-community edges
                    if (vcout[cv] == W(0)) {
                        vcs.push_back(cv);
                    }
                    vcout[cv] += w;
                }
            }
        }
        
        // Write to adjacency list
        super_adj[c].reserve(vcs.size());
        for (K d : vcs) {
            super_adj[c].emplace_back(d, vcout[d]);
            vcout[d] = W(0);  // Reset for next community
        }
    }
}

//==========================================================================
// Adaptive Resolution Helper Functions
//==========================================================================

/**
 * Runtime metrics collected during each Leiden pass
 * Used to inform adaptive resolution adjustments
 */
template <typename W>
struct LeidenPassMetrics {
    size_t num_communities;      // Communities after this pass
    size_t prev_communities;     // Communities before this pass
    int local_move_iterations;   // How many local-move iterations needed
    W max_community_weight;      // Largest community weight
    W total_weight;              // Sum of all weights
    size_t total_super_edges;    // Edges in super-graph
    double reduction_ratio;      // num_communities / prev_communities
    double size_imbalance;       // max_weight / avg_weight
    double super_avg_degree;     // Average degree in super-graph
};

/**
 * Compute adaptive resolution adjustment based on runtime metrics
 * 
 * @param current_resolution Current resolution value
 * @param metrics Runtime metrics from the current pass
 * @param original_avg_degree Average degree of original graph
 * @param max_iterations Max local-move iterations (to detect convergence issues)
 * @return Adjusted resolution for the next pass
 */
template <typename W>
inline double computeAdaptiveResolution(
    double current_resolution,
    const LeidenPassMetrics<W>& metrics,
    double original_avg_degree,
    int max_iterations) {
    
    double adjusted = current_resolution;
    
    // Signal 1: Community reduction ratio
    // If reducing too slowly (ratio > 0.5), lower resolution to encourage merging
    // If reducing too fast (ratio < 0.05), raise resolution to slow down
    if (metrics.reduction_ratio > 0.5) {
        adjusted *= 0.85;  // Not reducing enough, encourage merging
    } else if (metrics.reduction_ratio < 0.05) {
        adjusted *= 1.25;  // Reducing too aggressively, slow down
    }
    
    // Signal 2: Community size imbalance
    // If giant communities exist (imbalance > 50), raise resolution to break them
    if (metrics.size_imbalance > 50.0) {
        adjusted *= 1.15;  // Break giant communities
    } else if (metrics.size_imbalance > 100.0) {
        adjusted *= 1.25;  // Very unbalanced, more aggressive
    }
    
    // Signal 3: Local-move convergence
    // If converged in 1 iteration, communities are very stable - try finer resolution
    // If hit max iterations, might be struggling - try coarser
    if (metrics.local_move_iterations == 1) {
        adjusted *= 1.1;   // Too stable, try finer granularity
    } else if (metrics.local_move_iterations >= max_iterations) {
        adjusted *= 0.95;  // Struggling to converge, try coarser
    }
    
    // Signal 4: Super-graph density evolution
    // As super-graph gets denser, may need higher resolution
    if (metrics.super_avg_degree > 0 && original_avg_degree > 0) {
        double density_ratio = metrics.super_avg_degree / original_avg_degree;
        if (density_ratio > 2.0) {
            // Super-graph is significantly denser, scale up resolution
            adjusted *= (1.0 + 0.1 * std::log2(density_ratio));
        }
    }
    
    // Clamp to reasonable range [0.1, 5.0]
    adjusted = std::max(0.1, std::min(5.0, adjusted));
    
    return adjusted;
}

/**
 * Collect runtime metrics from current pass state
 */
template <typename K, typename W>
inline LeidenPassMetrics<W> collectPassMetrics(
    const std::vector<W>& super_weight,
    const std::vector<std::vector<std::pair<K, W>>>& super_adj,
    size_t num_communities,
    size_t prev_communities,
    int local_move_iterations) {
    
    LeidenPassMetrics<W> metrics;
    metrics.num_communities = num_communities;
    metrics.prev_communities = prev_communities;
    metrics.local_move_iterations = local_move_iterations;
    
    // Compute community weight statistics
    W max_weight = W(0);
    W total_weight = W(0);
    for (size_t c = 0; c < num_communities; ++c) {
        max_weight = std::max(max_weight, super_weight[c]);
        total_weight += super_weight[c];
    }
    metrics.max_community_weight = max_weight;
    metrics.total_weight = total_weight;
    
    // Compute size imbalance
    W avg_weight = (num_communities > 0) ? total_weight / num_communities : W(1);
    metrics.size_imbalance = (avg_weight > 0) ? static_cast<double>(max_weight) / avg_weight : 1.0;
    
    // Compute reduction ratio
    metrics.reduction_ratio = (prev_communities > 0) 
        ? static_cast<double>(num_communities) / prev_communities 
        : 1.0;
    
    // Compute super-graph edge count and average degree
    size_t total_edges = 0;
    for (size_t c = 0; c < num_communities; ++c) {
        total_edges += super_adj[c].size();
    }
    metrics.total_super_edges = total_edges;
    metrics.super_avg_degree = (num_communities > 0) 
        ? static_cast<double>(total_edges) / num_communities 
        : 0.0;
    
    return metrics;
}

//==========================================================================
// GVELeidenAdaptiveCSR - Leiden with Dynamic Resolution Adjustment
//==========================================================================

/**
 * GVELeidenAdaptiveCSR - Leiden algorithm with adaptive resolution
 * 
 * Key innovation: Resolution is adjusted dynamically at each pass based on:
 * 1. Community reduction rate (how fast communities are merging)
 * 2. Community size imbalance (detecting giant communities)
 * 3. Local-move convergence speed (iteration count)
 * 4. Super-graph density evolution (denser graphs need higher resolution)
 * 
 * Benefits:
 * - More balanced community hierarchy
 * - Better adaptation to graph structure at each level
 * - Avoids over-merging in later passes
 * - Potentially better cache locality for graph algorithms
 * 
 * @param g Input graph (must be symmetric)
 * @param initial_resolution Starting resolution (will be adapted)
 * @param tolerance Convergence tolerance for local-moving
 * @param aggregation_tolerance Minimum reduction to continue
 * @param tolerance_drop Factor to reduce tolerance each pass
 * @param max_iterations Max local-move iterations per pass
 * @param max_passes Maximum number of passes
 * @return GVELeidenResult with community assignments
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVELeidenResult<K> GVELeidenAdaptiveCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double initial_resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = 10) {
    
    const int64_t num_nodes = g.num_nodes();
    
    // GUARD: Empty graph - return empty result
    if (num_nodes == 0) {
        GVELeidenResult<K> empty_result;
        empty_result.total_iterations = 0;
        empty_result.total_passes = 0;
        return empty_result;
    }
    
    bool graph_is_symmetric = !g.directed();
    const int64_t num_edges_stored = g.num_edges();
    const double M = std::max(1.0, static_cast<double>(num_edges_stored));  // Guard against zero edges
    
    // Compute original graph's average degree for density comparison
    const double original_avg_degree = static_cast<double>(num_edges_stored) / num_nodes;
    
    GVELeidenResult<K> result;
    result.total_iterations = 0;
    result.total_passes = 0;
    
    // Initialize data structures
    std::vector<K> vcom(num_nodes);
    std::vector<K> vcob(num_nodes);
    std::vector<W> vtot(num_nodes);
    std::vector<W> ctot(num_nodes);
    std::vector<char> vaff(num_nodes, 1);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vcom[v] = static_cast<K>(v);
        W vtot_v = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
        vtot[v] = vtot_v;
        ctot[v] = vtot_v;
    }
    
    double current_tolerance = tolerance;
    size_t prev_communities = num_nodes;
    double resolution = initial_resolution;  // Will be adapted each pass
    
    printf("GVELeidenAdaptive: initial_resolution=%.4f, max_passes=%d\n", 
           initial_resolution, max_passes);
    
    for (int pass = 0; pass < max_passes; ++pass) {
        
        // PHASE 1: LOCAL-MOVING with current (possibly adapted) resolution
        int local_iters = gveOptLocalMove<K, W, NodeID_T, DestID_T>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric,
            M, resolution, max_iterations, current_tolerance);
        
        result.total_iterations += local_iters;
        
        // Store community bounds for refinement
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcob[v] = vcom[v];
        }
        
        // Count communities after local-moving
        std::vector<char> comm_exists(num_nodes, 0);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            comm_exists[vcom[v]] = 1;
        }
        size_t num_communities_after_local = 0;
        for (int64_t c = 0; c < num_nodes; ++c) {
            if (comm_exists[c]) num_communities_after_local++;
        }
        
        // PHASE 2: REFINEMENT (only on first pass)
        std::vector<K> vcom_refined(num_nodes);
        size_t num_refined_communities;
        
        if (pass == 0) {
            std::vector<W> ctot_refined(num_nodes);
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = static_cast<K>(v);
                ctot_refined[v] = vtot[v];
                vaff[v] = 1;
            }
            
            gveOptRefine<K, W, NodeID_T, DestID_T>(
                vcom_refined, ctot_refined, vaff, vcob, vtot,
                num_nodes, g, graph_is_symmetric, M, resolution);
            
            // Renumber refined communities
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        } else {
            // Skip refinement on later passes: just renumber
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = vcom[v];
            }
            
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> renumber(max_comm + 1, K(-1));
            K next_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K c = vcom_refined[v];
                if (renumber[c] == K(-1)) {
                    renumber[c] = next_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_id);
        }
        
        // PHASE 3: AGGREGATION using CSR-based approach
        std::vector<W> super_weight(num_refined_communities, W(0));
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom_refined[v];
            #pragma omp atomic
            super_weight[c] += vtot[v];
        }
        
        // Build super-graph
        std::vector<std::vector<std::pair<K, W>>> super_adj;
        buildSuperGraphAdjList<K, W, NodeID_T, DestID_T>(
            super_adj, g, vcom_refined, num_nodes, num_refined_communities);
        
        // ======== ADAPTIVE RESOLUTION: Collect metrics and adjust ========
        auto metrics = collectPassMetrics<K, W>(
            super_weight, super_adj, num_refined_communities, 
            prev_communities, local_iters);
        
        double next_resolution = computeAdaptiveResolution<W>(
            resolution, metrics, original_avg_degree, max_iterations);
        
        printf("  Pass %d: res=%.3f, comms=%zu→%zu (%.1f%%), iters=%d, imbalance=%.1f → next_res=%.3f\n",
               pass, resolution, prev_communities, num_refined_communities,
               metrics.reduction_ratio * 100, local_iters, metrics.size_imbalance,
               next_resolution);
        // ================================================================
        
        // Local-moving on super-graph
        std::vector<K> super_comm(num_refined_communities);
        std::vector<W> super_ctot(num_refined_communities);
        
        #pragma omp parallel for
        for (size_t c = 0; c < num_refined_communities; ++c) {
            super_comm[c] = c;
            super_ctot[c] = super_weight[c];
        }
        
        // Super-graph local move with flat array
        std::vector<W> sg_comm_weights(num_refined_communities, W(0));
        std::vector<K> sg_touched(num_refined_communities);
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            int moves = 0;
            
            for (size_t c = 0; c < num_refined_communities; ++c) {
                K d = super_comm[c];
                W kc = super_weight[c];
                W sigma_d = super_ctot[d];
                
                int num_touched = 0;
                W kc_to_d = W(0);
                
                for (auto& [nc, w] : super_adj[c]) {
                    K snc = super_comm[nc];
                    if (sg_comm_weights[snc] == W(0)) {
                        sg_touched[num_touched++] = snc;
                    }
                    sg_comm_weights[snc] += w;
                    if (snc == d) kc_to_d += w;
                }
                
                K best_comm = d;
                W best_delta = W(0);
                
                for (int i = 0; i < num_touched; ++i) {
                    K e = sg_touched[i];
                    W kc_to_e = sg_comm_weights[e];
                    W sigma_e = super_ctot[e];
                    
                    W delta_removal = resolution * kc * (sigma_d - kc) / M;
                    W delta_addition;
                    if (e == d) {
                        delta_addition = kc_to_d - delta_removal;
                    } else {
                        delta_addition = kc_to_e - resolution * kc * sigma_e / M;
                    }
                    
                    W delta = delta_addition - delta_removal + kc_to_d;
                    
                    if (delta > best_delta || (delta == best_delta && e < best_comm)) {
                        best_delta = delta;
                        best_comm = e;
                    }
                }
                
                if (best_comm != d && best_delta > 0) {
                    super_ctot[d] -= kc;
                    super_ctot[best_comm] += kc;
                    super_comm[c] = best_comm;
                    moves++;
                }
                
                for (int i = 0; i < num_touched; ++i) {
                    sg_comm_weights[sg_touched[i]] = W(0);
                }
            }
            
            if (moves == 0) break;
        }
        
        // Map back to original vertices
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            vcom[v] = super_comm[rc];
        }
        
        // Renumber final communities
        K max_final_comm = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_final_comm = std::max(max_final_comm, vcom[v]);
        }
        
        std::vector<K> final_renumber(max_final_comm + 1, K(-1));
        K next_final_id = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom[v];
            if (final_renumber[c] == K(-1)) {
                final_renumber[c] = next_final_id++;
            }
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcom[v] = final_renumber[vcom[v]];
        }
        
        size_t current_communities = static_cast<size_t>(next_final_id);
        
        // Store dendrogram
        result.community_per_pass.push_back(vcom);
        result.total_passes++;
        
        // Check convergence
        double reduction = static_cast<double>(prev_communities - current_communities) 
                         / static_cast<double>(prev_communities);
        
        if (current_communities >= prev_communities || reduction < (1.0 - aggregation_tolerance)) {
            break;
        }
        
        // Update for next pass
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            ctot[vcom[v]] = W(0);
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            #pragma omp atomic
            ctot[vcom[v]] += vtot[v];
            vaff[v] = 1;
        }
        
        prev_communities = current_communities;
        current_tolerance = tolerance / tolerance_drop;
        resolution = next_resolution;  // Apply adaptive resolution for next pass
    }
    
    result.final_community = vcom;
    result.modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, result.final_community, initial_resolution);
    
    printf("GVELeidenAdaptive: %d passes, %d iters, final_modularity=%.6f\n",
           result.total_passes, result.total_iterations, result.modularity);
    
    return result;
}

//==========================================================================
// GVELeidenOpt2CSR - GVELeidenOpt with CSR-based Aggregation
//==========================================================================

/**
 * GVELeidenOpt2CSR - Same as GVELeidenOptCSR but uses CSR-based aggregation
 * 
 * Key change: Replaces sort-based super-graph building with community-first
 * scanning approach (like leiden.hxx). This eliminates the O(E log E) sort
 * and uses O(E) linear scanning instead.
 * 
 * Expected speedup: ~30-40% faster detection time
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVELeidenResult<K> GVELeidenOpt2CSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = 10,
    bool skip_refine = false) {
    
    const int64_t num_nodes = g.num_nodes();
    bool graph_is_symmetric = !g.directed();
    const int64_t num_edges_stored = g.num_edges();
    const double M = std::max(1.0, static_cast<double>(num_edges_stored));  // Guard against zero edges
    
    GVELeidenResult<K> result;
    result.total_iterations = 0;
    result.total_passes = 0;
    
    // Initialize data structures
    std::vector<K> vcom(num_nodes);
    std::vector<K> vcob(num_nodes);  // Community bounds for refinement
    std::vector<W> vtot(num_nodes);
    std::vector<W> ctot(num_nodes);
    std::vector<char> vaff(num_nodes, 1);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vcom[v] = static_cast<K>(v);
        W vtot_v = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
        vtot[v] = vtot_v;
        ctot[v] = vtot_v;
    }
    
    double current_tolerance = tolerance;
    size_t prev_communities = num_nodes;
    
    for (int pass = 0; pass < max_passes; ++pass) {
        
        // PHASE 1: LOCAL-MOVING (using optimized flat-array approach)
        int local_iters = gveOptLocalMove<K, W, NodeID_T, DestID_T>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric,
            M, resolution, max_iterations, current_tolerance);
        
        result.total_iterations += local_iters;
        
        // Store community bounds
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcob[v] = vcom[v];
        }
        
        // Count communities
        std::vector<char> comm_exists(num_nodes, 0);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            comm_exists[vcom[v]] = 1;
        }
        size_t num_communities_after_local = 0;
        for (int64_t c = 0; c < num_nodes; ++c) {
            if (comm_exists[c]) num_communities_after_local++;
        }
        
        // PHASE 2: REFINEMENT (optimized) - can be skipped for speed
        std::vector<K> vcom_refined(num_nodes);
        size_t num_refined_communities;
        
        if (!skip_refine && pass == 0) {
            std::vector<W> ctot_refined(num_nodes);
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = static_cast<K>(v);
                ctot_refined[v] = vtot[v];
                vaff[v] = 1;
            }
            
            gveOptRefine<K, W, NodeID_T, DestID_T>(
                vcom_refined, ctot_refined, vaff, vcob, vtot,
                num_nodes, g, graph_is_symmetric, M, resolution);
            
            // Renumber refined communities
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        } else {
            // Skip refinement: just renumber communities
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = vcom[v];
            }
            
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> renumber(max_comm + 1, K(-1));
            K next_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K c = vcom_refined[v];
                if (renumber[c] == K(-1)) {
                    renumber[c] = next_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_id);
        }
        
        // Map from refined community to original local-move community (for later lookup)
        std::vector<K> refined_to_bound(num_refined_communities);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            if (rc < num_refined_communities) {
                refined_to_bound[rc] = vcob[v];
            }
        }
        
        // PHASE 3: AGGREGATION - Using optimized approach
        // Use thread-local reduction instead of atomics
        const int num_threads = omp_get_max_threads();
        std::vector<std::vector<W>> thread_super_weight(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            thread_super_weight[t].resize(num_refined_communities, W(0));
        }
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_weight = thread_super_weight[tid];
            
            #pragma omp for
            for (int64_t v = 0; v < num_nodes; ++v) {
                K c = vcom_refined[v];
                local_weight[c] += vtot[v];
            }
        }
        
        std::vector<W> super_weight(num_refined_communities, W(0));
        for (int t = 0; t < num_threads; ++t) {
            for (size_t c = 0; c < num_refined_communities; ++c) {
                super_weight[c] += thread_super_weight[t][c];
            }
        }
        
        // Build super-graph edges using CSR-based community scanning
        std::vector<std::vector<std::pair<K, W>>> super_adj;
        buildSuperGraphAdjList<K, W, NodeID_T, DestID_T>(
            super_adj, g, vcom_refined, num_nodes, num_refined_communities);
        
        // Local-moving on super-graph
        std::vector<K> super_comm(num_refined_communities);
        std::vector<W> super_ctot(num_refined_communities);
        
        #pragma omp parallel for
        for (size_t c = 0; c < num_refined_communities; ++c) {
            super_comm[c] = c;
            super_ctot[c] = super_weight[c];
        }
        
        // Limit super-graph iterations for speed (main optimization!)
        // Large graphs don't need many iterations on the coarsened graph
        int super_max_iter = std::min(3, max_iterations);
        
        // Use flat array for super-graph local move
        std::vector<W> sg_comm_weights(num_refined_communities, W(0));
        std::vector<K> sg_touched(num_refined_communities);
        
        for (int iter = 0; iter < super_max_iter; ++iter) {
            int moves = 0;
            
            for (size_t c = 0; c < num_refined_communities; ++c) {
                K d = super_comm[c];
                W kc = super_weight[c];
                W sigma_d = super_ctot[d];
                
                // Scan neighbors using flat array
                int num_touched = 0;
                W kc_to_d = W(0);
                
                for (auto& [nc, w] : super_adj[c]) {
                    K snc = super_comm[nc];
                    if (sg_comm_weights[snc] == W(0)) {
                        sg_touched[num_touched++] = snc;
                    }
                    sg_comm_weights[snc] += w;
                    if (snc == d) kc_to_d += w;
                }
                
                // Find best community
                K best_comm = d;
                W best_delta = W(0);
                
                for (int i = 0; i < num_touched; ++i) {
                    K e = sg_touched[i];
                    W kc_to_e = sg_comm_weights[e];
                    W sigma_e = super_ctot[e];
                    
                    W delta_removal = resolution * kc * (sigma_d - kc) / M;
                    W delta_addition;
                    if (e == d) {
                        delta_addition = kc_to_d - delta_removal;
                    } else {
                        delta_addition = kc_to_e - resolution * kc * sigma_e / M;
                    }
                    
                    W delta = delta_addition - delta_removal + kc_to_d;
                    
                    if (delta > best_delta || (delta == best_delta && e < best_comm)) {
                        best_delta = delta;
                        best_comm = e;
                    }
                }
                
                // Move if better
                if (best_comm != d && best_delta > 0) {
                    super_ctot[d] -= kc;
                    super_ctot[best_comm] += kc;
                    super_comm[c] = best_comm;
                    moves++;
                }
                
                // Clear touched communities
                for (int i = 0; i < num_touched; ++i) {
                    sg_comm_weights[sg_touched[i]] = W(0);
                }
            }
            
            if (moves == 0) break;
        }
        
        // Map super-graph communities back to original vertices
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            vcom[v] = super_comm[rc];
        }
        
        // Renumber final communities
        K max_final_comm = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_final_comm = std::max(max_final_comm, vcom[v]);
        }
        
        std::vector<K> final_renumber(max_final_comm + 1, K(-1));
        K next_final_id = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom[v];
            if (final_renumber[c] == K(-1)) {
                final_renumber[c] = next_final_id++;
            }
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcom[v] = final_renumber[vcom[v]];
        }
        
        size_t current_communities = static_cast<size_t>(next_final_id);
        
        // Store dendrogram
        result.community_per_pass.push_back(vcom);
        result.total_passes++;
        
        // Check convergence
        double reduction = static_cast<double>(prev_communities - current_communities) 
                         / static_cast<double>(prev_communities);
        
        if (current_communities >= prev_communities || reduction < (1.0 - aggregation_tolerance)) {
            break;
        }
        
        // Reinitialize for next pass (use sequential clear + parallel fill for simplicity)
        K max_comm_reinit = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_comm_reinit = std::max(max_comm_reinit, vcom[v]);
        }
        
        // Clear only used communities
        for (K c = 0; c <= max_comm_reinit; ++c) {
            ctot[c] = W(0);
        }
        
        // Parallel fill using shared vector (unavoidable atomics here, but fewer)
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            #pragma omp atomic
            ctot[vcom[v]] += vtot[v];
            vaff[v] = 1;
        }
        
        prev_communities = current_communities;
        current_tolerance = tolerance / tolerance_drop;
    }
    
    // Final community assignment
    result.final_community = vcom;
    
    // Compute modularity
    result.modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, result.final_community, resolution);
    
    printf("GVELeidenOpt2: %d passes, %d total iterations, modularity=%.6f\n",
           result.total_passes, result.total_iterations, result.modularity);
    
    return result;
}

//==========================================================================
// GVELeidenFastCSR - CSR Buffer Reuse Optimization (leiden.hxx style)
//==========================================================================

/**
 * Simple CSR super-graph structure for aggregation
 * Pre-allocated buffers that can be reused between passes
 */
template <typename K, typename W>
struct SuperGraphCSR {
    std::vector<size_t> offsets;    // Edge offsets per community
    std::vector<K> degrees;         // Current degree of each community
    std::vector<K> edge_dst;        // Destination community IDs
    std::vector<W> edge_weight;     // Edge weights
    size_t num_communities = 0;
    size_t max_edges = 0;
    
    void resize(size_t num_comms, size_t est_edges) {
        num_communities = num_comms;
        max_edges = est_edges;
        offsets.resize(num_comms + 1);
        degrees.resize(num_comms);
        edge_dst.resize(est_edges);
        edge_weight.resize(est_edges);
    }
    
    void clear() {
        std::fill(degrees.begin(), degrees.end(), K(0));
    }
};

/**
 * Build community-to-vertices mapping (which vertices belong to each community)
 */
template <typename K>
inline void buildCommunityVertices(
    std::vector<size_t>& coff,      // Community vertex offsets
    std::vector<K>& cedg,           // Vertices in each community
    const std::vector<K>& vcom,     // Community assignment
    size_t num_nodes,
    size_t num_communities) {
    
    // Count vertices per community
    std::vector<K> count(num_communities, 0);
    for (size_t v = 0; v < num_nodes; ++v) {
        count[vcom[v]]++;
    }
    
    // Exclusive scan to get offsets
    coff.resize(num_communities + 1);
    coff[0] = 0;
    for (size_t c = 0; c < num_communities; ++c) {
        coff[c + 1] = coff[c] + count[c];
    }
    
    // Fill vertex lists
    cedg.resize(num_nodes);
    std::fill(count.begin(), count.end(), 0);
    for (size_t v = 0; v < num_nodes; ++v) {
        K c = vcom[v];
        cedg[coff[c] + count[c]++] = static_cast<K>(v);
    }
}

/**
 * Aggregate edges using community-first approach (leiden.hxx style)
 * Much faster than vertex-by-vertex + global sort
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
inline void aggregateEdgesCSR(
    SuperGraphCSR<K, W>& super,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<K>& vcom,
    const std::vector<size_t>& coff,
    const std::vector<K>& cedg,
    size_t num_communities) {
    
    const int num_threads = omp_get_max_threads();
    
    // Thread-local hashtables for edge aggregation
    std::vector<std::vector<K>> thread_vcs(num_threads);
    std::vector<std::vector<W>> thread_vcout(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_vcout[t].resize(num_communities, W(0));
        thread_vcs[t].reserve(1000);
    }
    
    // Phase 1: Compute degree of each super-node (parallel)
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t c = 0; c < num_communities; ++c) {
        int tid = omp_get_thread_num();
        auto& vcs = thread_vcs[tid];
        W* vcout = thread_vcout[tid].data();
        
        vcs.clear();
        
        // Scan all vertices in this community
        for (size_t i = coff[c]; i < coff[c + 1]; ++i) {
            K u = cedg[i];
            
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
                
                K cv = vcom[v];
                if (cv != static_cast<K>(c)) {  // Only inter-community edges
                    if (vcout[cv] == W(0)) {
                        vcs.push_back(cv);
                    }
                    vcout[cv] += w;
                }
            }
        }
        
        // Store degree and clear hashtable
        super.degrees[c] = static_cast<K>(vcs.size());
        for (K d : vcs) {
            vcout[d] = W(0);
        }
    }
    
    // Phase 2: Compute offsets via exclusive scan
    super.offsets[0] = 0;
    for (size_t c = 0; c < num_communities; ++c) {
        super.offsets[c + 1] = super.offsets[c] + super.degrees[c];
    }
    
    // Ensure edge arrays are large enough
    size_t total_edges = super.offsets[num_communities];
    if (total_edges > super.edge_dst.size()) {
        super.edge_dst.resize(total_edges);
        super.edge_weight.resize(total_edges);
    }
    
    // Reset degrees for use as insertion counters
    std::fill(super.degrees.begin(), super.degrees.end(), K(0));
    
    // Phase 3: Fill edge arrays (parallel)
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t c = 0; c < num_communities; ++c) {
        int tid = omp_get_thread_num();
        auto& vcs = thread_vcs[tid];
        W* vcout = thread_vcout[tid].data();
        
        vcs.clear();
        
        // Scan all vertices in this community
        for (size_t i = coff[c]; i < coff[c + 1]; ++i) {
            K u = cedg[i];
            
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
                
                K cv = vcom[v];
                if (cv != static_cast<K>(c)) {
                    if (vcout[cv] == W(0)) {
                        vcs.push_back(cv);
                    }
                    vcout[cv] += w;
                }
            }
        }
        
        // Write edges to CSR (sequential within community, parallel across)
        size_t offset = super.offsets[c];
        for (K d : vcs) {
            super.edge_dst[offset] = d;
            super.edge_weight[offset] = vcout[d];
            vcout[d] = W(0);
            offset++;
        }
        super.degrees[c] = static_cast<K>(vcs.size());
    }
}

/**
 * GVELeidenFastCSR - Optimized GVE-Leiden with CSR buffer reuse
 * 
 * Key optimization: Uses pre-allocated CSR buffers for super-graph construction
 * instead of sort-based edge merging. Adopts leiden.hxx's aggregation strategy.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam W Weight type (should be double)
 * @tparam NodeID_T Node ID type from graph
 * @tparam DestID_T Destination ID type (may include weights)
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVELeidenResult<K> GVELeidenFastCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = 10) {
    
    const int64_t num_nodes = g.num_nodes();
    bool graph_is_symmetric = !g.directed();
    const int64_t num_edges_stored = g.num_edges();
    const double M = std::max(1.0, static_cast<double>(num_edges_stored));  // Guard against zero edges
    
    GVELeidenResult<K> result;
    result.total_iterations = 0;
    result.total_passes = 0;
    
    // Initialize data structures
    std::vector<K> vcom(num_nodes);
    std::vector<K> vcob(num_nodes);
    std::vector<W> vtot(num_nodes);
    std::vector<W> ctot(num_nodes);
    std::vector<char> vaff(num_nodes, 1);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vcom[v] = static_cast<K>(v);
        W vtot_v = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
        vtot[v] = vtot_v;
        ctot[v] = vtot_v;
    }
    
    // Pre-allocate CSR buffers for super-graphs (reused between passes)
    SuperGraphCSR<K, W> super_y, super_z;
    size_t est_communities = num_nodes / 10;  // Initial estimate
    size_t est_edges = g.num_edges() / 5;
    super_y.resize(est_communities, est_edges);
    super_z.resize(est_communities, est_edges);
    
    // Community-to-vertices mapping
    std::vector<size_t> coff;
    std::vector<K> cedg;
    
    double current_tolerance = tolerance;
    size_t prev_communities = num_nodes;
    
    for (int pass = 0; pass < max_passes; ++pass) {
        
        // PHASE 1: LOCAL-MOVING (using optimized flat-array approach)
        int local_iters = gveOptLocalMove<K, W, NodeID_T, DestID_T>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric,
            M, resolution, max_iterations, current_tolerance);
        
        result.total_iterations += local_iters;
        
        // Store community bounds
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcob[v] = vcom[v];
        }
        
        // Count communities
        std::vector<char> comm_exists(num_nodes, 0);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            comm_exists[vcom[v]] = 1;
        }
        size_t num_communities_after_local = 0;
        for (int64_t c = 0; c < num_nodes; ++c) {
            if (comm_exists[c]) num_communities_after_local++;
        }
        
        // PHASE 2: REFINEMENT
        std::vector<K> vcom_refined(num_nodes);
        size_t num_refined_communities;
        
        if (pass == 0) {
            std::vector<W> ctot_refined(num_nodes);
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = static_cast<K>(v);
                ctot_refined[v] = vtot[v];
                vaff[v] = 1;
            }
            
            gveOptRefine<K, W, NodeID_T, DestID_T>(
                vcom_refined, ctot_refined, vaff, vcob, vtot,
                num_nodes, g, graph_is_symmetric, M, resolution);
            
            // Renumber refined communities
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        } else {
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = vcom[v];
            }
            
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        }
        
        // Store community assignment for this pass
        result.community_per_pass.push_back(vcom_refined);
        result.total_passes = pass + 1;
        
        // Check aggregation tolerance
        double progress = static_cast<double>(num_refined_communities) / prev_communities;
        if (progress >= aggregation_tolerance) {
            vcom = vcom_refined;
            break;
        }
        
        prev_communities = num_refined_communities;
        
        // PHASE 3: CSR-BASED AGGREGATION (leiden.hxx style)
        // Build community-to-vertices mapping
        buildCommunityVertices(coff, cedg, vcom_refined, num_nodes, num_refined_communities);
        
        // Resize super-graph buffers if needed
        if (num_refined_communities > super_y.num_communities) {
            super_y.resize(num_refined_communities, g.num_edges());
            super_z.resize(num_refined_communities, g.num_edges());
        }
        super_y.clear();
        
        // Aggregate edges using CSR approach
        aggregateEdgesCSR<K, W, NodeID_T, DestID_T>(
            super_y, g, vcom_refined, coff, cedg, num_refined_communities);
        
        // Update for next pass
        vcom = vcom_refined;
        
        // Compute super-node weights
        std::vector<W> super_weight(num_refined_communities, W(0));
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            #pragma omp atomic
            super_weight[vcom[v]] += vtot[v];
        }
        
        // Reinitialize for next pass on super-graph
        ctot.assign(num_refined_communities, W(0));
        for (size_t c = 0; c < num_refined_communities; ++c) {
            ctot[c] = super_weight[c];
        }
        
        vtot = super_weight;
        vaff.assign(num_refined_communities, 1);
        vcom.resize(num_refined_communities);
        std::iota(vcom.begin(), vcom.end(), K(0));
        
        current_tolerance = tolerance / tolerance_drop;
    }
    
    // Final community assignment
    result.final_community = vcom;
    
    // Compute modularity
    result.modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, result.final_community, resolution);
    
    printf("GVELeidenFast: %d passes, %d iters, modularity=%.6f\n",
           result.total_passes, result.total_iterations, result.modularity);
    
    return result;
}

//==========================================================================
// GVELeidenTurboCSR - Maximum Speed Variant
//==========================================================================

/**
 * GVELeidenTurboCSR - Speed-optimized GVE-Leiden for fast reordering
 * 
 * Optimizations over GVELeidenOptCSR:
 * 1. SKIP REFINEMENT: Refinement adds ~20-30% overhead for marginal quality gain
 * 2. EARLY TERMINATION: Stop when modularity gain < threshold (not just moves=0)
 * 3. BATCHED ATOMICS: Accumulate local changes, write once per iteration
 * 4. REDUCED ITERATIONS: Default max 5 iterations (sufficient for convergence)
 * 5. AGGRESSIVE TOLERANCE: Higher tolerance for faster convergence
 * 
 * Trade-off: ~10-15% lower modularity for ~40-50% faster detection
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam W Weight type (should be double)
 * @tparam NodeID_T Node ID type from graph
 * @tparam DestID_T Destination ID type (may include weights)
 * @param g Input graph (must be symmetric for Leiden)
 * @param resolution Resolution parameter (default 1.0)
 * @param max_iterations Max iterations per pass (default 5)
 * @param max_passes Maximum Leiden passes (default 10)
 * @param early_stop_threshold Stop if modularity gain < this (default 0.001)
 * 
 * @return GVELeidenResult with community assignments and modularity
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVELeidenResult<K> GVELeidenTurboCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    int max_iterations = 5,
    int max_passes = 10,
    double early_stop_threshold = 0.001) {
    
    const int64_t num_nodes = g.num_nodes();
    bool graph_is_symmetric = !g.directed();
    const int64_t num_edges_stored = g.num_edges();
    const double M = std::max(1.0, static_cast<double>(num_edges_stored));  // Guard against zero edges
    
    GVELeidenResult<K> result;
    result.total_iterations = 0;
    result.total_passes = 0;
    
    // Initialize data structures
    std::vector<K> vcom(num_nodes);
    std::vector<W> vtot(num_nodes);
    std::vector<W> ctot(num_nodes);
    std::vector<char> vaff(num_nodes, 1);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vcom[v] = static_cast<K>(v);
        W vtot_v = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
        vtot[v] = vtot_v;
        ctot[v] = vtot_v;
    }
    
    const int num_threads = omp_get_max_threads();
    double prev_modularity = 0.0;
    size_t prev_communities = num_nodes;
    
    // Thread-local structures for batched updates
    std::vector<std::vector<W>> thread_comm_weights(num_threads);
    std::vector<std::vector<K>> thread_touched_comms(num_threads);
    std::vector<std::vector<std::pair<K, W>>> thread_ctot_deltas(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_comm_weights[t].resize(num_nodes, W(0));
        thread_touched_comms[t].reserve(1000);
        thread_ctot_deltas[t].reserve(10000);
    }
    
    for (int pass = 0; pass < max_passes; ++pass) {
        
        // PHASE 1: LOCAL-MOVING (turbo version - batched atomics)
        int local_iters = 0;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            int moves_this_iter = 0;
            
            // Clear thread-local delta accumulators
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                thread_ctot_deltas[tid].clear();
            }
            
            #pragma omp parallel reduction(+:moves_this_iter)
            {
                int tid = omp_get_thread_num();
                W* comm_weights = thread_comm_weights[tid].data();
                auto& touched_comms = thread_touched_comms[tid];
                auto& ctot_deltas = thread_ctot_deltas[tid];
                
                #pragma omp for schedule(static, 2048)
                for (int64_t u = 0; u < num_nodes; ++u) {
                    if (!vaff[u]) continue;
                    
                    K d = vcom[u];
                    W ku = vtot[u];
                    
                    // Scan neighbors
                    touched_comms.clear();
                    W ku_to_d = W(0);
                    
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
                        
                        K cv = vcom[v];
                        if (comm_weights[cv] == W(0)) {
                            touched_comms.push_back(cv);
                        }
                        comm_weights[cv] += w;
                        if (cv == d) ku_to_d += w;
                    }
                    
                    // For symmetric graphs, in_neigh == out_neigh, skip
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
                            
                            K cv = vcom[v];
                            if (comm_weights[cv] == W(0)) {
                                touched_comms.push_back(cv);
                            }
                            comm_weights[cv] += w;
                            if (cv == d) ku_to_d += w;
                        }
                    }
                    
                    // Find best community
                    K best_c = d;
                    W best_delta = W(0);
                    W sigma_d = ctot[d];
                    
                    for (K c : touched_comms) {
                        if (c == d) continue;
                        W ku_to_c = comm_weights[c];
                        W sigma_c = ctot[c];
                        
                        // Delta modularity formula
                        W delta = (ku_to_c - ku_to_d) / M - 
                                  resolution * ku * (sigma_c - sigma_d + ku) / (2.0 * M * M);
                        
                        if (delta > best_delta) {
                            best_delta = delta;
                            best_c = c;
                        }
                    }
                    
                    // Clear touched
                    for (K c : touched_comms) {
                        comm_weights[c] = W(0);
                    }
                    
                    // Move if beneficial - batch the update
                    if (best_c != d) {
                        vcom[u] = best_c;
                        ctot_deltas.push_back({d, -ku});
                        ctot_deltas.push_back({best_c, ku});
                        
                        // Mark neighbors affected (still need this)
                        for (auto neighbor : g.out_neigh(u)) {
                            NodeID_T v;
                            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                                v = neighbor;
                            } else {
                                v = neighbor.v;
                            }
                            vaff[v] = 1;
                        }
                        
                        moves_this_iter++;
                    }
                    vaff[u] = 0;
                }
            }
            
            // Apply batched ctot updates
            #pragma omp parallel for
            for (int t = 0; t < num_threads; ++t) {
                for (auto& [c, delta] : thread_ctot_deltas[t]) {
                    #pragma omp atomic
                    ctot[c] += delta;
                }
            }
            
            local_iters++;
            if (moves_this_iter == 0) break;
        }
        
        result.total_iterations += local_iters;
        
        // Store pass communities (NO REFINEMENT - skip phase 2)
        result.community_per_pass.push_back(vcom);
        result.total_passes = pass + 1;
        
        // Count communities and check early termination
        std::vector<char> comm_exists(num_nodes, 0);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            comm_exists[vcom[v]] = 1;
        }
        
        size_t num_communities = 0;
        for (int64_t c = 0; c < num_nodes; ++c) {
            if (comm_exists[c]) num_communities++;
        }
        
        // Early termination: minimal community change (but allow at least 2 passes)
        if (pass > 0 && num_communities >= prev_communities * 0.98) {
            break;
        }
        
        // Early termination: check modularity gain (only after pass 1)
        double current_modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, vcom, resolution);
        if (pass > 1 && (current_modularity - prev_modularity) < early_stop_threshold) {
            break;
        }
        prev_modularity = current_modularity;
        prev_communities = num_communities;
        
        // PHASE 3: SIMPLIFIED AGGREGATION (no sorting, direct hash)
        // Skip if converged
        if (num_communities == prev_communities) break;
        
        // Renumber communities contiguously
        std::vector<K> comm_map(num_nodes, K(-1));
        K next_id = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom[v];
            if (comm_map[c] == K(-1)) {
                comm_map[c] = next_id++;
            }
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcom[v] = comm_map[vcom[v]];
        }
        
        // Build super-graph using simple concurrent hash
        std::vector<W> super_weight(num_communities, W(0));
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            #pragma omp atomic
            super_weight[vcom[v]] += vtot[v];
        }
        
        // Build super-graph edges
        std::vector<std::unordered_map<K, W>> super_adj(num_communities);
        
        // Sequential edge collection (simpler, still fast for small super-graphs)
        for (int64_t u = 0; u < num_nodes; ++u) {
            K cu = vcom[u];
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
                K cv = vcom[v];
                if (cu != cv) {
                    super_adj[cu][cv] += w;
                }
            }
        }
        
        // Continue with super-graph (simplified: just update ctot and reset)
        ctot.assign(num_communities, W(0));
        for (size_t c = 0; c < num_communities; ++c) {
            ctot[c] = super_weight[c];
        }
        
        // Reset for next pass
        vaff.assign(num_communities, 1);
        vcom.resize(num_communities);
        vtot = super_weight;
        std::iota(vcom.begin(), vcom.end(), K(0));
    }
    
    // Final community assignment
    result.final_community = vcom;
    
    // Compute final modularity
    result.modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, result.final_community, resolution);
    
    printf("GVELeidenTurbo: %d passes, %d iters, modularity=%.6f\n",
           result.total_passes, result.total_iterations, result.modularity);
    
    return result;
}

/**
 * GVELeidenDendo - GVE-Leiden with INCREMENTAL atomic dendrogram building
 * 
 * This is a clone of GVELeidenCSR that builds the dendrogram incrementally
 * using RabbitOrder-style atomic CAS instead of post-processing.
 * 
 * Key optimization: Instead of storing community_per_pass and rebuilding
 * the tree in post-processing, we build parent-child relationships as
 * vertices merge during detection using lock-free atomic operations.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam W Weight type (should be double)
 * @tparam NodeID_T Node ID type from graph
 * @tparam DestID_T Destination ID type (may include weights)
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVEDendroResult<K> GVELeidenDendoCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = 10) {
    
    const int64_t num_nodes = g.num_nodes();
    bool graph_is_symmetric = !g.directed();
    const int64_t num_edges_stored = g.num_edges();
    const double M = std::max(1.0, static_cast<double>(num_edges_stored));  // Guard against zero edges
    
    // Initialize atomic dendrogram (replaces community_per_pass storage)
    GVEAtomicDendroResult<K> atomic_dendro;
    std::vector<W> vtot(num_nodes);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vtot[v] = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
    }
    atomic_dendro.init(num_nodes, vtot);
    
    // Initialize data structures (same as GVELeidenCSR)
    std::vector<K> vcom(num_nodes);
    std::vector<K> vcob(num_nodes);
    std::vector<W> ctot(num_nodes);
    std::vector<char> vaff(num_nodes, 1);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vcom[v] = static_cast<K>(v);
        ctot[v] = vtot[v];
    }
    
    double current_tolerance = tolerance;
    size_t prev_communities = num_nodes;
    std::atomic<int64_t> total_merges(0);
    
    // Main Leiden loop (same structure as GVELeidenCSR)
    for (int pass = 0; pass < max_passes; ++pass) {
        
        // PHASE 1: LOCAL-MOVING
        int local_iters = gveLeidenLocalMoveCSR<K, W, NodeID_T, DestID_T>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric,
            M, resolution, max_iterations, current_tolerance);
        
        atomic_dendro.total_iterations += local_iters;
        
        // Store community bounds
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcob[v] = vcom[v];
        }
        
        // PHASE 2: REFINEMENT (same as GVELeidenCSR)
        std::vector<K> vcom_refined(num_nodes);
        size_t num_refined_communities;
        int refine_moves = 0;
        
        if (pass == 0) {
            std::vector<W> ctot_refined(num_nodes);
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = static_cast<K>(v);
                ctot_refined[v] = vtot[v];
                vaff[v] = 1;
            }
            
            refine_moves = gveLeidenRefineCSR<K, W, NodeID_T, DestID_T>(
                vcom_refined, ctot_refined, vaff, vcob, vtot,
                num_nodes, g, graph_is_symmetric, M, resolution);
            
            // === ATOMIC DENDROGRAM UPDATE during refinement ===
            // Vertices that moved in refinement get merged into their new representative
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (rc != static_cast<K>(v)) {
                    if (atomicMergeToDendro<K>(atomic_dendro, v, rc)) {
                        total_merges.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            
            // Renumber refined communities
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        } else {
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = vcom[v];
            }
            
            K max_comm = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                max_comm = std::max(max_comm, vcom_refined[v]);
            }
            
            std::vector<K> refine_renumber(max_comm + 1, K(-1));
            K next_refined_id = 0;
            for (int64_t v = 0; v < num_nodes; ++v) {
                K rc = vcom_refined[v];
                if (refine_renumber[rc] == K(-1)) {
                    refine_renumber[rc] = next_refined_id++;
                }
            }
            
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                vcom_refined[v] = refine_renumber[vcom_refined[v]];
            }
            
            num_refined_communities = static_cast<size_t>(next_refined_id);
        }
        
        // Build super-graph (same as GVELeidenCSR)
        std::vector<W> super_weight(num_refined_communities, W(0));
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom_refined[v];
            #pragma omp atomic
            super_weight[c] += vtot[v];
        }
        
        const int num_threads = omp_get_max_threads();
        std::vector<std::unordered_map<uint64_t, W>> thread_edge_hash(num_threads);
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& edge_hash = thread_edge_hash[tid];
            
            #pragma omp for schedule(dynamic, 4096)
            for (int64_t u = 0; u < num_nodes; ++u) {
                K cu = vcom_refined[u];
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
                    K cv = vcom_refined[v];
                    if (cu != cv && cu < num_refined_communities && cv < num_refined_communities) {
                        uint64_t key = (static_cast<uint64_t>(cu) << 32) | cv;
                        edge_hash[key] += w;
                    }
                }
            }
        }
        
        std::unordered_map<uint64_t, W> global_edge_hash;
        for (int t = 0; t < num_threads; ++t) {
            for (auto& [key, w] : thread_edge_hash[t]) {
                global_edge_hash[key] += w;
            }
        }
        
        std::vector<std::vector<std::pair<K, W>>> super_adj(num_refined_communities);
        for (auto& [key, w] : global_edge_hash) {
            K src = static_cast<K>(key >> 32);
            K dst = static_cast<K>(key & 0xFFFFFFFF);
            if (src < num_refined_communities) {
                super_adj[src].emplace_back(dst, w);
            }
        }
        
        // LOCAL-MOVING ON SUPER-GRAPH
        std::vector<K> super_comm(num_refined_communities);
        std::vector<W> super_ctot(num_refined_communities);
        
        #pragma omp parallel for
        for (size_t c = 0; c < num_refined_communities; ++c) {
            super_comm[c] = c;
            super_ctot[c] = super_weight[c];
        }
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            int moves = 0;
            for (size_t c = 0; c < num_refined_communities; ++c) {
                K d = super_comm[c];
                W kc = super_weight[c];
                W sigma_d = super_ctot[d];
                
                std::unordered_map<K, W> neighbor_sc_weight;
                W kc_to_d = W(0);
                
                for (auto& [nc, w] : super_adj[c]) {
                    K snc = super_comm[nc];
                    neighbor_sc_weight[snc] += w;
                    if (snc == d) kc_to_d += w;
                }
                
                K best_sc = d;
                W best_delta = W(0);
                
                for (auto& [sc, kc_to_sc] : neighbor_sc_weight) {
                    if (sc == d) continue;
                    W sigma_sc = super_ctot[sc];
                    W delta = graphbrew::leiden::gveDeltaModularity<W>(kc_to_sc, kc_to_d, kc, sigma_sc, sigma_d, M, resolution);
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_sc = sc;
                    }
                }
                
                if (best_sc != d) {
                    super_ctot[d] -= kc;
                    super_ctot[best_sc] += kc;
                    super_comm[c] = best_sc;
                    moves++;
                }
            }
            if (moves == 0) break;
        }
        
        // Map back to original vertices
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            K rc = vcom_refined[v];
            vcom[v] = super_comm[rc];
        }
        
        // Renumber final communities
        K max_final_comm = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_final_comm = std::max(max_final_comm, vcom[v]);
        }
        
        std::vector<K> final_renumber(max_final_comm + 1, K(-1));
        K next_final_id = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            K c = vcom[v];
            if (final_renumber[c] == K(-1)) {
                final_renumber[c] = next_final_id++;
            }
        }
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            vcom[v] = final_renumber[vcom[v]];
        }
        
        size_t num_final_communities = static_cast<size_t>(next_final_id);
        
        atomic_dendro.total_passes++;
        
        printf("  Pass %d: local=%d iters, refine=%d, merges=%lld, communities: %zu -> %zu -> %zu\n",
               pass + 1, local_iters, refine_moves, total_merges.load(),
               prev_communities, num_refined_communities, num_final_communities);
        
        // Check convergence
        if (num_final_communities >= prev_communities || num_final_communities == 1) break;
        
        double progress = static_cast<double>(num_final_communities) / prev_communities;
        if (progress >= aggregation_tolerance) break;
        
        prev_communities = num_final_communities;
        
        // Recompute for next pass
        std::fill(ctot.begin(), ctot.end(), W(0));
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            #pragma omp atomic
            ctot[vcom[v]] += vtot[v];
            vaff[v] = 1;
        }
        
        current_tolerance /= tolerance_drop;
    }
    
    // Convert to non-atomic result
    auto result = atomic_dendro.toNonAtomic();
    result.final_community = vcom;
    result.modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, vcom, resolution);
    
    // Build final dendrogram from community assignment
    // (This fills in any vertices not yet merged)
    buildDendrogramFromCommunities<K, W>(result, vcom, vtot, num_nodes);
    
    // Update roots
    result.roots.clear();
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (result.parent[v] == -1) {
            result.roots.push_back(v);
        }
    }
    
    printf("GVELeidenDendo: %d passes, %d iters, modularity=%.6f, roots=%zu, merges=%lld\n",
           result.total_passes, result.total_iterations, result.modularity,
           result.roots.size(), total_merges.load());
    
    return result;
}

/**
 * GVELeidenOptDendo - Optimized GVE-Leiden with incremental dendrogram
 * 
 * Clone of GVELeidenOpt with dendrogram building.
 * Uses optimized flat-array scanning and builds tree from final communities.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam W Weight type (should be double)
 * @tparam NodeID_T Node ID type from graph
 * @tparam DestID_T Destination ID type (may include weights)
 */
template <typename K, typename W, typename NodeID_T, typename DestID_T>
GVEDendroResult<K> GVELeidenOptDendoCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    double tolerance = 1e-2,
    double aggregation_tolerance = DEFAULT_AGGREGATION_TOLERANCE,
    double tolerance_drop = DEFAULT_QUALITY_FACTOR,
    int max_iterations = 20,
    int max_passes = 10) {
    
    // For OptDendo, we use the same approach as GVELeidenOpt
    // but build dendrogram from the final community assignment
    
    const int64_t num_nodes = g.num_nodes();
    bool graph_is_symmetric = !g.directed();
    
    // Run the optimized GVE-Leiden algorithm
    auto gve_result = GVELeidenOptCSR<K, W, NodeID_T, DestID_T>(
        g, resolution, tolerance, aggregation_tolerance,
        tolerance_drop, max_iterations, max_passes);
    
    // Compute vertex weights for dendrogram
    std::vector<W> vtot(num_nodes);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        vtot[v] = computeVertexTotalWeightCSR<W, NodeID_T, DestID_T>(v, g, graph_is_symmetric);
    }
    
    // Build dendrogram from final communities
    GVEDendroResult<K> result;
    result.total_iterations = gve_result.total_iterations;
    result.total_passes = gve_result.total_passes;
    result.modularity = gve_result.modularity;
    result.final_community = gve_result.final_community;
    
    result.parent.resize(num_nodes, -1);
    result.first_child.resize(num_nodes, -1);
    result.sibling.resize(num_nodes, -1);
    result.subtree_size.resize(num_nodes, 1);
    result.weight.resize(num_nodes);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        result.weight[v] = vtot[v];
    }
    
    buildDendrogramFromCommunities<K, W>(result, result.final_community, vtot, num_nodes);
    
    // Collect roots
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (result.parent[v] == -1) {
            result.roots.push_back(v);
        }
    }
    
    printf("GVELeidenOptDendo: %d passes, %d iters, modularity=%.6f, roots=%zu\n",
           result.total_passes, result.total_iterations, result.modularity, result.roots.size());
    
    return result;
}

/**
 * FastLeidenCSR - Union-Find based community detection on CSR graphs
 * 
 * Uses modularity-guided community merging with Union-Find for efficiency:
 * 1. Initialize each vertex in its own community
 * 2. Process vertices in degree order (small first)
 * 3. For each vertex, find best neighbor community to merge with
 * 4. Merge if modularity delta > 0
 * 5. Multiple passes allow communities to coalesce further
 *
 * @tparam K Community ID type (default uint32_t)
 * @tparam NodeID_T Node ID type from graph
 * @tparam DestID_T Destination ID type (may include weights)
 * @param g Input CSR graph
 * @param resolution Resolution parameter for modularity
 * @param max_iterations Max iterations (unused, for API compatibility)
 * @param max_passes Maximum number of merge passes
 * @return Vector of community assignments per pass (finest to coarsest)
 */
template <typename K, typename NodeID_T, typename DestID_T>
std::vector<std::vector<K>> FastLeidenCSR(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    int max_iterations = 10,
    int max_passes = 10)
{
    const int64_t num_nodes = g.num_nodes();
    const int64_t num_edges = g.num_edges_directed();
    // Guard against zero edges to prevent FPE
    const double total_weight = std::max(1.0, static_cast<double>(num_edges));
    
    std::vector<std::vector<K>> community_per_pass;
    
    // Union-Find arrays
    std::vector<K> parent(num_nodes);
    std::vector<double> strength(num_nodes);  // sum of degrees in community
    
    // Initialize: each vertex in its own community
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        parent[i] = i;
        strength[i] = static_cast<double>(g.out_degree(i));
    }
    
    // Find root with path compression
    auto find = [&](K x) -> K {
        K root = x;
        while (parent[root] != root) root = parent[root];
        // Path compression
        while (parent[x] != root) {
            K next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };
    
    // Degree-ordered processing (smaller degrees first)
    std::vector<std::pair<int64_t, K>> deg_order(num_nodes);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        deg_order[v] = {g.out_degree(v), v};
    }
    std::sort(deg_order.begin(), deg_order.end());
    
    // Multi-pass merging
    for (int pass = 0; pass < max_passes; ++pass) {
        int64_t merge_count = 0;
        
        for (auto& [deg, v] : deg_order) {
            K v_root = find(v);
            double v_str = strength[v_root];
            
            // Find best neighbor to merge with
            double best_delta = 0.0;
            K best_target = v_root;
            
            for (auto u : g.out_neigh(v)) {
                K u_root = find(u);
                if (u_root == v_root) continue;
                
                double edge_weight = 1.0;
                double u_str = strength[u_root];
                
                // Modularity delta = edge_weight - resolution * v_str * u_str / total_weight
                double delta = edge_weight - resolution * v_str * u_str / total_weight;
                if (delta > best_delta) {
                    best_delta = delta;
                    best_target = u_root;
                }
            }
            
            // Merge if positive gain
            if (best_target != v_root) {
                // Union: smaller root becomes child of larger
                if (strength[v_root] < strength[best_target]) {
                    parent[v_root] = best_target;
                    strength[best_target] += v_str;
                } else {
                    parent[best_target] = v_root;
                    strength[v_root] += strength[best_target];
                }
                merge_count++;
            }
        }
        
        // Extract communities for this pass
        std::vector<K> community(num_nodes);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            community[i] = find(i);
        }
        
        // Renumber to contiguous IDs
        std::unordered_map<K, K> comm_renumber;
        K next_comm = 0;
        for (int64_t i = 0; i < num_nodes; ++i) {
            K c = community[i];
            if (comm_renumber.find(c) == comm_renumber.end()) {
                comm_renumber[c] = next_comm++;
            }
            community[i] = comm_renumber[c];
        }
        
        community_per_pass.push_back(community);
        
        printf("LeidenCSR pass %d: %ld merges, %u communities\n", pass + 1, merge_count, next_comm);
        
        // Stop if no merges (fully converged)
        if (merge_count == 0) break;
    }
    
    return community_per_pass;
}

//==========================================================================
// GenerateLeidenCSRMapping - Standalone function for CSR-based Leiden ordering
//==========================================================================

/**
 * GenerateLeidenCSRMapping - Fast Leiden-style ordering directly on CSR
 * 
 * Uses fast label propagation directly on CSR for community detection,
 * then orders vertices by community + degree.
 * 
 * @param g Input graph (CSR format, symmetric)
 * @param new_ids Output permutation array
 * @param reordering_options [resolution, max_iterations, max_passes]
 * @param flavor Ordering flavor: 0=DFS, 1=BFS, 2=HubSort (default)
 * 
 * Template parameters:
 * - K: Community ID type (typically uint32_t)
 * - NodeID_T: Node ID type
 * - DestID_T: Destination ID type (may include weight)
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateLeidenCSRMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options,
    int flavor = 2)
{
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 10;
    int max_passes = 1;  // Single pass is fastest with comparable quality
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("LeidenCSR: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run fast label propagation directly on CSR
    auto community_per_pass = FastLeidenCSR<K, NodeID_T, DestID_T>(g, resolution, max_iterations, max_passes);
    
    tm.Stop();
    PrintTime("LeidenCSR Community Detection", tm.Seconds());
    PrintTime("LeidenCSR Passes", community_per_pass.size());
    
    if (community_per_pass.empty()) {
        // Fallback: original ordering
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            new_ids[i] = i;
        }
        return;
    }
    
    // Get degrees for secondary sort
    tm.Start();
    std::vector<K> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        degrees[i] = g.out_degree(i);
    }
    
    // Create sort indices
    std::vector<size_t> sort_indices(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        sort_indices[i] = i;
    }
    
    const size_t num_passes = community_per_pass.size();
    
    // Apply ordering based on flavor
    switch (flavor) {
        case 0: { // DFS (standard)
            // DFS-like ordering: sort by all passes (coarsest to finest) then degree
            std::cout << "LeidenCSR Ordering: DFS (standard)" << std::endl;
            
            __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
                [&community_per_pass, &degrees, num_passes](size_t a, size_t b) {
                    // Compare all passes from coarsest (last) to finest (first)
                    for (size_t p = num_passes; p > 0; --p) {
                        K comm_a = community_per_pass[p - 1][a];
                        K comm_b = community_per_pass[p - 1][b];
                        if (comm_a != comm_b) {
                            return comm_a < comm_b;
                        }
                    }
                    // Within same community: degree ascending
                    return degrees[a] < degrees[b];
                });
            break;
        }
        
        case 1: { // BFS
            // BFS-like ordering: sort by level (pass index where community changes)
            // then by community at each level, then by degree
            std::cout << "LeidenCSR Ordering: BFS (level-first)" << std::endl;
            
            // Compute level for each node (first pass where it differs from neighbors)
            std::vector<int> node_level(num_nodes, num_passes);
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v) {
                for (size_t p = 0; p < num_passes; ++p) {
                    if (community_per_pass[p][v] != community_per_pass[num_passes-1][v]) {
                        node_level[v] = p;
                        break;
                    }
                }
            }
            
            __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
                [&community_per_pass, &degrees, &node_level, num_passes](size_t a, size_t b) {
                    // Primary: sort by last pass community (coarsest)
                    K comm_a = community_per_pass[num_passes - 1][a];
                    K comm_b = community_per_pass[num_passes - 1][b];
                    if (comm_a != comm_b) return comm_a < comm_b;
                    // Secondary: sort by level (BFS order)
                    if (node_level[a] != node_level[b]) return node_level[a] < node_level[b];
                    // Tertiary: degree descending
                    return degrees[a] > degrees[b];
                });
            break;
        }
        
        case 2: // HubSort (default)
        default: {
            // Hub sort within communities: sort by (last community, degree DESC)
            // Simpler than full hierarchical sort but good for hub locality
            std::cout << "LeidenCSR Ordering: HubSort (community + degree)" << std::endl;
            
            __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
                [&community_per_pass, &degrees, num_passes](size_t a, size_t b) {
                    // Primary: sort by last pass community
                    K comm_a = community_per_pass[num_passes - 1][a];
                    K comm_b = community_per_pass[num_passes - 1][b];
                    if (comm_a != comm_b) return comm_a < comm_b;
                    // Secondary: degree descending (hubs first)
                    return degrees[a] > degrees[b];
                });
            break;
        }
    }
    
    // Assign new IDs based on sorted order
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        new_ids[sort_indices[i]] = i;
    }
    
    tm.Stop();
    double map_time = tm.Seconds();
    
    // Print community count and modularity from last pass
    if (!community_per_pass.empty()) {
        // Count unique communities (don't use max+1 as communities may not be contiguous)
        std::unordered_set<K> unique_comms(community_per_pass.back().begin(), 
                                            community_per_pass.back().end());
        PrintTime("LeidenCSR Communities", static_cast<double>(unique_comms.size()));
        
        // Compute and print modularity
        double modularity = computeModularityCSR<K, NodeID_T, DestID_T>(g, community_per_pass.back(), resolution);
        PrintTime("LeidenCSR Modularity", modularity);
    }
    PrintTime("LeidenCSR Map Time", map_time);
}

//==========================================================================
// GenerateLeidenFastMapping - Union-Find + Label Propagation based ordering
//==========================================================================

/**
 * GenerateLeidenFastMapping - Main entry point for LeidenFast algorithm
 * 
 * Improved version with:
 * - Parallel Union-Find with atomic CAS
 * - Best-fit modularity merging (not first-fit)
 * - Hash-based label propagation (faster than sorted array)
 * - Proper convergence detection
 * 
 * @param g Input graph (CSR format, symmetric)
 * @param new_ids Output permutation array
 * @param reordering_options [resolution, max_passes]
 */
template <typename NodeID_T, typename DestID_T>
void GenerateLeidenFastMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options)
{
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_passes = 3;  // Default to 3 passes for good quality
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::FIXED;
        res_cfg.value = 1.0;
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_passes = std::stoi(reordering_options[1]);
    }
    
    printf("LeidenFast: resolution=%.4f (%s), max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_passes);
    
    // Run community detection
    std::vector<double> vertex_strength;
    std::vector<int64_t> community;
    
    FastModularityCommunityDetection<NodeID_T, DestID_T>(
        g, vertex_strength, community, resolution, max_passes);
    
    tm.Stop();
    PrintTime("LeidenFast Community Detection", tm.Seconds());
    
    // Build ordering
    tm.Start();
    std::vector<int64_t> ordered_vertices;
    BuildCommunityOrdering<NodeID_T, DestID_T>(
        g, vertex_strength, community, ordered_vertices);
    
    // Assign new IDs
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        new_ids[ordered_vertices[i]] = i;
    }
    
    tm.Stop();
    PrintTime("LeidenFast Ordering", tm.Seconds());
}

//==========================================================================
// GenerateLeidenMapping2 - Quality-focused Leiden reordering
//==========================================================================

/**
 * GenerateLeidenMapping2 - Quality-focused Leiden reordering
 * 
 * Uses LeidenCommunityDetection for high-quality communities, then
 * orders by community strength (degree sum) and vertex degree.
 * 
 * @param g Input graph (CSR format, symmetric)
 * @param new_ids Output permutation array
 * @param reordering_options [resolution, max_passes, max_iterations]
 */
template <typename NodeID_T, typename DestID_T>
void GenerateLeidenMapping2(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options)
{
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_passes = 1;      // Single pass is usually enough
    int max_iterations = 4;  // 4 iterations for good community quality
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_passes = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_iterations = std::stoi(reordering_options[2]);
    }
    
    printf("Leiden: resolution=%.4f (%s), max_passes=%d, max_iterations=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_passes, max_iterations);
    
    std::vector<int64_t> community;
    LeidenCommunityDetection<NodeID_T, DestID_T>(g, community, resolution, max_passes, max_iterations);
    
    tm.Stop();
    PrintTime("Leiden Community Detection", tm.Seconds());
    
    tm.Start();
    
    int64_t num_comms = *std::max_element(community.begin(), community.end()) + 1;
    std::vector<double> comm_strength(num_comms, 0.0);
    
    // Use thread-local reduction instead of atomics for better performance
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> thread_strength(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_strength[t].resize(num_comms, 0.0);
    }
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_strength = thread_strength[tid];
        
        #pragma omp for
        for (int64_t v = 0; v < num_nodes; ++v) {
            int64_t c = community[v];
            local_strength[c] += static_cast<double>(g.out_degree(v));
        }
    }
    
    // Merge thread-local results
    for (int t = 0; t < num_threads; ++t) {
        for (int64_t c = 0; c < num_comms; ++c) {
            comm_strength[c] += thread_strength[t][c];
        }
    }
    
    std::vector<std::tuple<int64_t, int64_t, int64_t>> order(num_nodes);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        int64_t c = community[v];
        order[v] = std::make_tuple(
            -static_cast<int64_t>(comm_strength[c] * 1000),
            -static_cast<int64_t>(g.out_degree(v)),
            v
        );
    }
    __gnu_parallel::sort(order.begin(), order.end());
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        new_ids[std::get<2>(order[i])] = i;
    }
    
    tm.Stop();
    PrintTime("Leiden Ordering", tm.Seconds());
}

//==========================================================================
// GenerateGVELeidenCSRMapping - True Leiden ordering using GVE-Leiden
//==========================================================================

/**
 * GenerateGVELeidenCSRMapping - True Leiden ordering using GVE-Leiden algorithm
 * 
 * Uses the GVE-Leiden implementation which follows the ACM paper
 * "Fast Leiden Algorithm for Community Detection in Shared Memory Setting"
 * 
 * Key features:
 * - Proper refinement phase (only isolated vertices move)
 * - Community bounds constraint
 * - Well-connected communities guaranteed
 * - Dendrogram-based ordering
 * - Isolated vertex separation (degree-0 vertices grouped at end)
 * 
 * @param g Input graph (CSR format, symmetric)
 * @param new_ids Output permutation array
 * @param reordering_options [resolution, max_iterations, max_passes]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenCSRMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    // ================================================================
    // ISOLATED VERTEX SEPARATION
    // ================================================================
    std::vector<int64_t> isolated_vertices;
    std::vector<int64_t> active_vertices;
    isolated_vertices.reserve(num_nodes / 10);
    active_vertices.reserve(num_nodes);
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (g.out_degree(v) == 0) {
            isolated_vertices.push_back(v);
        } else {
            active_vertices.push_back(v);
        }
    }
    
    const int64_t num_isolated = isolated_vertices.size();
    const int64_t num_active = active_vertices.size();
    
    printf("GVELeidenCSR: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    if (num_isolated > 0) {
        printf("GVELeidenCSR: %ld active vertices, %ld isolated vertices (%.1f%%)\n",
               num_active, num_isolated, 100.0 * num_isolated / num_nodes);
    }
    
    // Run GVE-Leiden algorithm
    auto result = GVELeidenCSR<K, double, NodeID_T, DestID_T>(g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    PrintTime("GVELeiden Community Detection", tm.Seconds());
    
    // Use community hierarchy for ordering
    tm.Start();
    
    // Get degrees for secondary sort
    std::vector<K> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        degrees[i] = g.out_degree(i);
    }
    
    // Build dendrogram from community passes
    size_t num_communities = 0;
    int64_t current_id = 0;
    
    if (!result.community_per_pass.empty()) {
        std::vector<LeidenDendrogramNode> nodes;
        std::vector<int64_t> roots;
        buildLeidenDendrogram(nodes, roots, result.community_per_pass, degrees, num_nodes);
        
        // Count real communities
        size_t real_communities = 0;
        for (int64_t r : roots) {
            if (degrees[r] > 0) {
                real_communities++;
            }
        }
        num_communities = real_communities;
        
        // Use DFS with hub-first ordering
        orderDendrogramDFSParallel(nodes, roots, new_ids, true, false);
        
        // Post-process: Move isolated vertices to the end
        std::vector<int64_t> active_order;
        active_order.reserve(num_active);
        for (int64_t v = 0; v < num_nodes; ++v) {
            if (degrees[v] > 0) {
                active_order.push_back(v);
            }
        }
        
        std::sort(active_order.begin(), active_order.end(),
            [&new_ids](int64_t a, int64_t b) {
                return new_ids[a] < new_ids[b];
            });
        
        current_id = 0;
        for (int64_t v : active_order) {
            new_ids[v] = current_id++;
        }
        for (int64_t v : isolated_vertices) {
            new_ids[v] = current_id++;
        }
    } else {
        // Fallback: Sort by community then degree
        std::vector<size_t> sort_indices;
        sort_indices.reserve(num_active);
        for (int64_t v : active_vertices) {
            sort_indices.push_back(v);
        }
        
        __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
            [&result, &degrees](size_t a, size_t b) {
                K comm_a = result.final_community[a];
                K comm_b = result.final_community[b];
                if (comm_a != comm_b) return comm_a < comm_b;
                return degrees[a] > degrees[b];
            });
        
        current_id = 0;
        for (size_t v : sort_indices) {
            new_ids[v] = current_id++;
        }
        for (int64_t v : isolated_vertices) {
            new_ids[v] = current_id++;
        }
        
        std::unordered_set<K> unique_comms;
        for (int64_t v : active_vertices) {
            unique_comms.insert(result.final_community[v]);
        }
        num_communities = unique_comms.size();
    }
    
    tm.Stop();
    double ordering_time = tm.Seconds();
    
    PrintTime("GVELeiden Communities", static_cast<double>(num_communities));
    if (num_isolated > 0) {
        PrintTime("GVELeiden Isolated", static_cast<double>(num_isolated));
    }
    PrintTime("GVELeiden Modularity", result.modularity);
    PrintTime("GVELeiden Map Time", ordering_time);
}

//==========================================================================
// GenerateGVELeidenOptMapping - Optimized GVE-Leiden ordering
//==========================================================================

/**
 * GenerateGVELeidenOptMapping - Optimized GVE-Leiden ordering
 * 
 * Uses the optimized GVE-Leiden implementation with:
 * - Flat arrays instead of hash maps
 * - Prefetching for community lookups
 * - Guided scheduling for better load balancing
 * - Sorted edge merging for super-graph construction
 * 
 * @param g Input graph (CSR format, symmetric)
 * @param new_ids Output permutation array
 * @param reordering_options [resolution, max_iterations, max_passes]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenOptMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    // ================================================================
    // ISOLATED VERTEX SEPARATION
    // ================================================================
    std::vector<int64_t> isolated_vertices;
    std::vector<int64_t> active_vertices;
    isolated_vertices.reserve(num_nodes / 10);
    active_vertices.reserve(num_nodes);
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (g.out_degree(v) == 0) {
            isolated_vertices.push_back(v);
        } else {
            active_vertices.push_back(v);
        }
    }
    
    const int64_t num_isolated = isolated_vertices.size();
    const int64_t num_active = active_vertices.size();
    
    printf("GVELeidenOpt: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    printf("GVELeidenOpt: %ld active vertices, %ld isolated vertices (%.1f%%)\n",
           num_active, num_isolated, 100.0 * num_isolated / num_nodes);
    
    // Run optimized GVE-Leiden algorithm
    auto result = GVELeidenOptCSR<K, double, NodeID_T, DestID_T>(g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    PrintTime("GVELeidenOpt Community Detection", tm.Seconds());
    
    // Use community hierarchy for ordering
    tm.Start();
    
    // Get degrees for secondary sort
    std::vector<K> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        degrees[i] = g.out_degree(i);
    }
    
    // Build dendrogram from community passes
    size_t num_communities = 0;
    int64_t current_id = 0;
    
    if (!result.community_per_pass.empty()) {
        std::vector<LeidenDendrogramNode> nodes;
        std::vector<int64_t> roots;
        buildLeidenDendrogram(nodes, roots, result.community_per_pass, degrees, num_nodes);
        
        // Count real communities
        size_t real_communities = 0;
        for (int64_t r : roots) {
            if (degrees[r] > 0) {
                real_communities++;
            }
        }
        num_communities = real_communities;
        
        // Use DFS with hub-first ordering
        orderDendrogramDFSParallel(nodes, roots, new_ids, true, false);
        
        // Post-process: Move isolated vertices to the end
        std::vector<int64_t> active_order;
        active_order.reserve(num_active);
        for (int64_t v = 0; v < num_nodes; ++v) {
            if (degrees[v] > 0) {
                active_order.push_back(v);
            }
        }
        
        std::sort(active_order.begin(), active_order.end(),
            [&new_ids](int64_t a, int64_t b) {
                return new_ids[a] < new_ids[b];
            });
        
        current_id = 0;
        for (int64_t v : active_order) {
            new_ids[v] = current_id++;
        }
        for (int64_t v : isolated_vertices) {
            new_ids[v] = current_id++;
        }
        
    } else {
        // Fallback: Sort by community then degree
        std::vector<size_t> sort_indices;
        sort_indices.reserve(num_active);
        for (int64_t v : active_vertices) {
            sort_indices.push_back(v);
        }
        
        __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
            [&result, &degrees](size_t a, size_t b) {
                K comm_a = result.final_community[a];
                K comm_b = result.final_community[b];
                if (comm_a != comm_b) return comm_a < comm_b;
                return degrees[a] > degrees[b];
            });
        
        current_id = 0;
        for (size_t v : sort_indices) {
            new_ids[v] = current_id++;
        }
        for (int64_t v : isolated_vertices) {
            new_ids[v] = current_id++;
        }
        
        std::unordered_set<K> unique_comms;
        for (int64_t v : active_vertices) {
            unique_comms.insert(result.final_community[v]);
        }
        num_communities = unique_comms.size();
    }
    
    tm.Stop();
    double ordering_time = tm.Seconds();
    
    PrintTime("GVELeidenOpt Communities", static_cast<double>(num_communities));
    PrintTime("GVELeidenOpt Isolated", static_cast<double>(num_isolated));
    PrintTime("GVELeidenOpt Modularity", result.modularity);
    PrintTime("GVELeidenOpt Map Time", ordering_time);
}

/**
 * GenerateGVELeidenOpt2Mapping - GVE-Leiden with CSR-based aggregation
 * 
 * Same as GenerateGVELeidenOptMapping but uses buildSuperGraphAdjList
 * for faster aggregation (no global sort).
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenOpt2Mapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("GVELeidenOpt2: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run GVE-Leiden Opt2 with CSR aggregation
    auto result = GVELeidenOpt2CSR<K, double, NodeID_T, DestID_T>(
        g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    double detection_time = tm.Seconds();
    tm.Start();  // Restart for mapping phase
    PrintTime("GVELeidenOpt2 Community Detection", detection_time);
    PrintTime("GVELeidenOpt2 Passes", static_cast<double>(result.total_passes));
    PrintTime("GVELeidenOpt2 Modularity", result.modularity);
    
    const int64_t num_nodes = g.num_nodes();
    new_ids.resize(num_nodes);
    
    // Generate multi-level sort ordering (like LeidenOrder)
    struct VertexKey {
        K pass0_comm;
        K pass1_comm;
        K pass2_comm;
        int64_t neg_degree;
        int64_t vertex;
        
        bool operator<(const VertexKey& o) const {
            if (pass0_comm != o.pass0_comm) return pass0_comm < o.pass0_comm;
            if (pass1_comm != o.pass1_comm) return pass1_comm < o.pass1_comm;
            if (pass2_comm != o.pass2_comm) return pass2_comm < o.pass2_comm;
            if (neg_degree != o.neg_degree) return neg_degree < o.neg_degree;
            return vertex < o.vertex;
        }
    };
    
    std::vector<VertexKey> keys(num_nodes);
    int num_passes = result.community_per_pass.size();
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        keys[v].pass0_comm = (num_passes > 0) ? result.community_per_pass[num_passes-1][v] : 0;
        keys[v].pass1_comm = (num_passes > 1) ? result.community_per_pass[num_passes-2][v] : 0;
        keys[v].pass2_comm = (num_passes > 2) ? result.community_per_pass[num_passes-3][v] : 0;
        keys[v].neg_degree = -static_cast<int64_t>(g.out_degree(v));
        keys[v].vertex = v;
    }
    
    __gnu_parallel::sort(keys.begin(), keys.end());
    
    // Assign new IDs based on sort order
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        new_ids[keys[i].vertex] = static_cast<NodeID_T>(i);
    }
    
    // Count communities and isolated
    std::vector<char> comm_seen(num_nodes, 0);
    int64_t num_communities = 0;
    int64_t num_isolated = 0;
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        K c = result.final_community[v];
        if (!comm_seen[c]) {
            comm_seen[c] = 1;
            num_communities++;
        }
        if (g.out_degree(v) == 0) num_isolated++;
    }
    
    tm.Stop();
    double ordering_time = tm.Seconds();
    
    PrintTime("GVELeidenOpt2 Communities", static_cast<double>(num_communities));
    PrintTime("GVELeidenOpt2 Isolated", static_cast<double>(num_isolated));
    PrintTime("GVELeidenOpt2 Map Time", ordering_time);
}

/**
 * GenerateGVELeidenAdaptiveMapping - GVE-Leiden with dynamic resolution
 * 
 * Uses GVELeidenAdaptiveCSR which adjusts resolution at each pass based on:
 * - Community reduction rate
 * - Community size imbalance
 * - Local-move convergence speed
 * - Super-graph density evolution
 * 
 * This is the only algorithm that truly supports ResolutionMode::DYNAMIC.
 * Pass "dynamic" or "dynamic:2.0" as resolution option.
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenAdaptiveMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::DYNAMIC;  // Default to dynamic for adaptive
        res_cfg.initial_value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double initial_resolution = res_cfg.isDynamic() ? res_cfg.initial_value : res_cfg.value;
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("GVELeidenAdaptive: initial_resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           initial_resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run adaptive GVE-Leiden
    auto result = GVELeidenAdaptiveCSR<K, double, NodeID_T, DestID_T>(
        g, initial_resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    double detection_time = tm.Seconds();
    tm.Start();
    
    PrintTime("GVELeidenAdaptive Community Detection", detection_time);
    PrintTime("GVELeidenAdaptive Passes", static_cast<double>(result.total_passes));
    PrintTime("GVELeidenAdaptive Modularity", result.modularity);
    
    const int64_t num_nodes = g.num_nodes();
    new_ids.resize(num_nodes);
    
    // Generate multi-level sort ordering
    struct VertexKey {
        K pass0_comm;
        K pass1_comm;
        K pass2_comm;
        int64_t neg_degree;
        int64_t vertex;
        
        bool operator<(const VertexKey& o) const {
            if (pass0_comm != o.pass0_comm) return pass0_comm < o.pass0_comm;
            if (pass1_comm != o.pass1_comm) return pass1_comm < o.pass1_comm;
            if (pass2_comm != o.pass2_comm) return pass2_comm < o.pass2_comm;
            if (neg_degree != o.neg_degree) return neg_degree < o.neg_degree;
            return vertex < o.vertex;
        }
    };
    
    std::vector<VertexKey> keys(num_nodes);
    int num_passes = result.community_per_pass.size();
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        keys[v].pass0_comm = (num_passes > 0) ? result.community_per_pass[num_passes-1][v] : 0;
        keys[v].pass1_comm = (num_passes > 1) ? result.community_per_pass[num_passes-2][v] : 0;
        keys[v].pass2_comm = (num_passes > 2) ? result.community_per_pass[num_passes-3][v] : 0;
        keys[v].neg_degree = -static_cast<int64_t>(g.out_degree(v));
        keys[v].vertex = v;
    }
    
    __gnu_parallel::sort(keys.begin(), keys.end());
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        new_ids[keys[i].vertex] = static_cast<NodeID_T>(i);
    }
    
    // Count communities
    std::vector<char> comm_seen(num_nodes, 0);
    int64_t num_communities = 0;
    int64_t num_isolated = 0;
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        K c = result.final_community[v];
        if (!comm_seen[c]) {
            comm_seen[c] = 1;
            num_communities++;
        }
        if (g.out_degree(v) == 0) num_isolated++;
    }
    
    tm.Stop();
    double ordering_time = tm.Seconds();
    
    PrintTime("GVELeidenAdaptive Communities", static_cast<double>(num_communities));
    PrintTime("GVELeidenAdaptive Isolated", static_cast<double>(num_isolated));
    PrintTime("GVELeidenAdaptive Map Time", ordering_time);
}

/**
 * GenerateGVELeidenDendoMapping - GVE-Leiden with incremental dendrogram
 * 
 * Uses GVELeidenDendoCSR for community detection with atomic dendrogram building,
 * then traverses dendrogram to generate vertex ordering.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @param g Input graph (must be symmetric for Leiden)
 * @param new_ids Output mapping vector
 * @param reordering_options Options: [resolution, max_iterations, max_passes]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenDendoMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("GVELeidenDendo: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run algorithm
    auto result = GVELeidenDendoCSR<K, double, NodeID_T, DestID_T>(
        g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    double detection_time = tm.Seconds();
    PrintTime("GVELeidenDendo Community Detection", detection_time);
    
    Timer tm2;
    tm2.Start();
    
    // Traverse dendrogram to generate ordering
    // Convert pvector to std::vector for the standalone function
    std::vector<NodeID_T> temp_ids(new_ids.size());
    traverseDendrogramDFS<K, NodeID_T>(result, temp_ids, true);  // hub_first = true
    
    // Copy back to pvector
    #pragma omp parallel for
    for (size_t i = 0; i < temp_ids.size(); ++i) {
        new_ids[i] = temp_ids[i];
    }
    
    tm2.Stop();
    double ordering_time = tm2.Seconds();
    
    PrintTime("GVELeidenDendo Communities", static_cast<double>(result.roots.size()));
    PrintTime("GVELeidenDendo Modularity", result.modularity);
    PrintTime("GVELeidenDendo Map Time", ordering_time);
}

/**
 * GenerateGVELeidenOptDendoMapping - Optimized GVE-Leiden with incremental dendrogram
 * 
 * Uses GVELeidenOptDendoCSR for community detection with atomic dendrogram building
 * and flat-array optimizations, then traverses dendrogram to generate vertex ordering.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @param g Input graph (must be symmetric for Leiden)
 * @param new_ids Output mapping vector
 * @param reordering_options Options: [resolution, max_iterations, max_passes]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenOptDendoMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("GVELeidenOptDendo: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run algorithm
    auto result = GVELeidenOptDendoCSR<K, double, NodeID_T, DestID_T>(
        g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    double detection_time = tm.Seconds();
    PrintTime("GVELeidenOptDendo Community Detection", detection_time);
    
    Timer tm2;
    tm2.Start();
    
    // Traverse dendrogram to generate ordering
    // Convert pvector to std::vector for the standalone function
    std::vector<NodeID_T> temp_ids(new_ids.size());
    traverseDendrogramDFS<K, NodeID_T>(result, temp_ids, true);  // hub_first = true
    
    // Copy back to pvector
    #pragma omp parallel for
    for (size_t i = 0; i < temp_ids.size(); ++i) {
        new_ids[i] = temp_ids[i];
    }
    
    tm2.Stop();
    double ordering_time = tm2.Seconds();
    
    PrintTime("GVELeidenOptDendo Communities", static_cast<double>(result.roots.size()));
    PrintTime("GVELeidenOptDendo Modularity", result.modularity);
    PrintTime("GVELeidenOptDendo Map Time", ordering_time);
}

//==========================================================================
// GenerateGVELeidenFastMapping - CSR Buffer Reuse Optimization
//==========================================================================

/**
 * GenerateGVELeidenFastMapping - GVE-Leiden with CSR buffer reuse
 * 
 * Uses GVELeidenFastCSR for community detection with pre-allocated CSR buffers
 * for super-graph construction (leiden.hxx style), then applies multi-level sort.
 * 
 * Expected ~30-40% faster detection than GVELeidenOptCSR.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @param g Input graph (must be symmetric for Leiden)
 * @param new_ids Output mapping vector
 * @param reordering_options Options: [resolution, max_iterations, max_passes]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenFastMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("GVELeidenFast: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run GVE-Leiden with CSR buffer reuse (faster aggregation)
    auto result = GVELeidenFastCSR<K, double, NodeID_T, DestID_T>(
        g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    double detection_time = tm.Seconds();
    PrintTime("GVELeidenFast Community Detection", detection_time);
    
    Timer tm2;
    tm2.Start();
    
    // Get degrees for secondary sort
    std::vector<int64_t> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        degrees[i] = g.out_degree(i);
    }
    
    // Separate isolated and active vertices
    std::vector<int64_t> active_vertices;
    std::vector<int64_t> isolated_vertices;
    active_vertices.reserve(num_nodes);
    isolated_vertices.reserve(num_nodes / 10);
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (degrees[v] > 0) {
            active_vertices.push_back(v);
        } else {
            isolated_vertices.push_back(v);
        }
    }
    
    const int num_passes = static_cast<int>(result.community_per_pass.size());
    
    // Count unique communities
    std::unordered_set<K> unique_comms;
    for (int64_t v : active_vertices) {
        unique_comms.insert(result.final_community[v]);
    }
    size_t num_communities = unique_comms.size();
    
    // LeidenOrder-style multi-level sort
    __gnu_parallel::sort(active_vertices.begin(), active_vertices.end(),
        [&result, &degrees, num_passes](int64_t a, int64_t b) {
            for (int p = num_passes - 1; p >= 0; --p) {
                K ca = result.community_per_pass[p][a];
                K cb = result.community_per_pass[p][b];
                if (ca != cb) return ca < cb;
            }
            return degrees[a] > degrees[b];
        });
    
    // Assign new IDs
    int64_t current_id = 0;
    for (int64_t v : active_vertices) {
        new_ids[v] = current_id++;
    }
    for (int64_t v : isolated_vertices) {
        new_ids[v] = current_id++;
    }
    
    tm2.Stop();
    double ordering_time = tm2.Seconds();
    
    PrintTime("GVELeidenFast Passes", static_cast<double>(num_passes));
    PrintTime("GVELeidenFast Communities", static_cast<double>(num_communities));
    PrintTime("GVELeidenFast Modularity", result.modularity);
    PrintTime("GVELeidenFast Map Time", ordering_time);
}

//==========================================================================
// GenerateGVELeidenOptSortMapping - Sort-based ordering (Strategy 1)
//==========================================================================

/**
 * GenerateGVELeidenOptSortMapping - Optimized GVE-Leiden with LeidenOrder-style sort
 * 
 * Uses GVELeidenOptCSR for community detection, then applies a multi-level
 * lexicographic sort by [community_per_pass..., -degree] instead of dendrogram DFS.
 * This mimics LeidenOrder's ordering strategy while using GVE's fast detection.
 * 
 * Key insight: Sorting by all hierarchy levels + degree achieves the same cache
 * locality as dendrogram DFS but is simpler and often faster.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @param g Input graph (must be symmetric for Leiden)
 * @param new_ids Output mapping vector
 * @param reordering_options Options: [resolution, max_iterations, max_passes]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenOptSortMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("GVELeidenOptSort: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run optimized GVE-Leiden with lower aggregation_tolerance to allow more passes
    // (SORT_AGGREGATION_TOLERANCE instead of DEFAULT - continue even if only 5% reduction per pass)
    auto result = GVELeidenOptCSR<K, double, NodeID_T, DestID_T>(
        g, resolution, DEFAULT_TOLERANCE, SORT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    tm.Stop();
    double detection_time = tm.Seconds();
    PrintTime("GVELeidenOptSort Community Detection", detection_time);
    
    Timer tm2;
    tm2.Start();
    
    // Get degrees for secondary sort (hub-first within communities)
    std::vector<int64_t> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        degrees[i] = g.out_degree(i);
    }
    
    // Separate isolated (degree-0) and active vertices
    std::vector<int64_t> active_vertices;
    std::vector<int64_t> isolated_vertices;
    active_vertices.reserve(num_nodes);
    isolated_vertices.reserve(num_nodes / 10);
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (degrees[v] > 0) {
            active_vertices.push_back(v);
        } else {
            isolated_vertices.push_back(v);
        }
    }
    
    const int num_passes = static_cast<int>(result.community_per_pass.size());
    
    // Count unique communities in final pass
    std::unordered_set<K> unique_comms;
    for (int64_t v : active_vertices) {
        unique_comms.insert(result.final_community[v]);
    }
    size_t num_communities = unique_comms.size();
    
    // LeidenOrder-style multi-level sort: [pass_N, pass_N-1, ..., pass_0, -degree]
    // This groups vertices hierarchically by their community at each level
    __gnu_parallel::sort(active_vertices.begin(), active_vertices.end(),
        [&result, &degrees, num_passes](int64_t a, int64_t b) {
            // Compare from coarsest (last pass) to finest (first pass)
            for (int p = num_passes - 1; p >= 0; --p) {
                K ca = result.community_per_pass[p][a];
                K cb = result.community_per_pass[p][b];
                if (ca != cb) return ca < cb;
            }
            // Within same community at all levels, sort by degree descending (hubs first)
            return degrees[a] > degrees[b];
        });
    
    // Assign new IDs: active vertices first (in sorted order), then isolated
    int64_t current_id = 0;
    for (int64_t v : active_vertices) {
        new_ids[v] = current_id++;
    }
    for (int64_t v : isolated_vertices) {
        new_ids[v] = current_id++;
    }
    
    tm2.Stop();
    double ordering_time = tm2.Seconds();
    
    PrintTime("GVELeidenOptSort Passes", static_cast<double>(num_passes));
    PrintTime("GVELeidenOptSort Communities", static_cast<double>(num_communities));
    PrintTime("GVELeidenOptSort Modularity", result.modularity);
    PrintTime("GVELeidenOptSort Map Time", ordering_time);
}

//==========================================================================
// GenerateGVELeidenTurboMapping - Maximum Speed Variant
//==========================================================================

/**
 * GenerateGVELeidenTurboMapping - Speed-optimized GVE-Leiden with sort-based ordering
 * 
 * Combines GVELeidenTurboCSR (no refinement, early termination, batched atomics)
 * with LeidenOrder-style multi-level sort for maximum speed.
 * 
 * Expected ~40-50% faster detection than GVELeidenOptCSR with ~10% lower modularity.
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @param g Input graph (must be symmetric for Leiden)
 * @param new_ids Output mapping vector
 * @param reordering_options Options: [resolution, max_iterations, max_passes]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVELeidenTurboMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 5;  // Lower default for turbo
    int max_passes = 10;
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::AUTO;
        res_cfg.value = computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
        max_passes = std::stoi(reordering_options[2]);
    }
    
    printf("GVELeidenTurbo: resolution=%.4f (%s), max_iterations=%d, max_passes=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations, max_passes);
    
    // Run optimized GVE-Leiden with refinement SKIPPED for speed
    auto result = GVELeidenOptCSR<K, double, NodeID_T, DestID_T>(
        g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE, DEFAULT_QUALITY_FACTOR, max_iterations, max_passes, true);  // skip_refine=true
    
    tm.Stop();
    double detection_time = tm.Seconds();
    PrintTime("GVELeidenTurbo Community Detection", detection_time);
    
    Timer tm2;
    tm2.Start();
    
    // Get degrees for secondary sort
    std::vector<int64_t> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        degrees[i] = g.out_degree(i);
    }
    
    // Separate isolated and active vertices
    std::vector<int64_t> active_vertices;
    std::vector<int64_t> isolated_vertices;
    active_vertices.reserve(num_nodes);
    isolated_vertices.reserve(num_nodes / 10);
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (degrees[v] > 0) {
            active_vertices.push_back(v);
        } else {
            isolated_vertices.push_back(v);
        }
    }
    
    const int num_passes = static_cast<int>(result.community_per_pass.size());
    
    // Count unique communities
    std::unordered_set<K> unique_comms;
    for (int64_t v : active_vertices) {
        unique_comms.insert(result.final_community[v]);
    }
    size_t num_communities = unique_comms.size();
    
    // LeidenOrder-style multi-level sort
    __gnu_parallel::sort(active_vertices.begin(), active_vertices.end(),
        [&result, &degrees, num_passes](int64_t a, int64_t b) {
            for (int p = num_passes - 1; p >= 0; --p) {
                K ca = result.community_per_pass[p][a];
                K cb = result.community_per_pass[p][b];
                if (ca != cb) return ca < cb;
            }
            return degrees[a] > degrees[b];
        });
    
    // Assign new IDs
    int64_t current_id = 0;
    for (int64_t v : active_vertices) {
        new_ids[v] = current_id++;
    }
    for (int64_t v : isolated_vertices) {
        new_ids[v] = current_id++;
    }
    
    tm2.Stop();
    double ordering_time = tm2.Seconds();
    
    PrintTime("GVELeidenTurbo Passes", static_cast<double>(num_passes));
    PrintTime("GVELeidenTurbo Communities", static_cast<double>(num_communities));
    PrintTime("GVELeidenTurbo Modularity", result.modularity);
    PrintTime("GVELeidenTurbo Map Time", ordering_time);
}

/**
 * GVERabbitCoreResult - Local result structure for GVERabbitCore
 * 
 * This is a simplified result structure that contains the information
 * needed by GenerateGVERabbitMapping. It's separate from GVERabbitResult
 * in reorder_types.h which has a more complex structure.
 */
template <typename K = uint32_t>
struct GVERabbitCoreResult {
    std::vector<K> community;       // Final community assignment
    double modularity;
    double aggregation_time;
    double refinement_time;
};

/**
 * GVERabbitCore - GVE-Leiden with limited iterations for speed
 * 
 * Fast community detection hybrid using GVE-Leiden's local moving
 * with fewer iterations and single pass for RabbitOrder-style speed.
 * 
 * @tparam K Community ID type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @param g Input graph (must be symmetric)
 * @param resolution Resolution parameter (default 1.0)
 * @param max_iterations Maximum iterations (default 5)
 * @return GVERabbitCoreResult with community assignments and metadata
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
GVERabbitCoreResult<K> GVERabbitCore(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    double resolution = 1.0,
    int max_iterations = 5) {
    
    Timer tm;
    tm.Start();
    
    // Use GVE-Leiden with limited iterations for speed
    // Parameters: resolution, tolerance, agg_tolerance, tolerance_drop, max_iterations, max_passes
    // Fewer iterations (3) and single pass (1) for speed
    auto leiden_result = GVELeidenCSR<K, double, NodeID_T, DestID_T>(g, resolution, 
        0.01,    // tolerance
        0.8,     // aggregation_tolerance  
        10.0,    // tolerance_drop
        std::min(max_iterations, 5),  // max_iterations (cap at 5 for speed)
        1);      // max_passes (single pass)
    
    tm.Stop();
    
    // Convert to GVERabbitCoreResult
    GVERabbitCoreResult<K> result;
    result.community = std::move(leiden_result.final_community);
    result.modularity = leiden_result.modularity;
    result.aggregation_time = tm.Seconds();
    result.refinement_time = 0;
    
    return result;
}

/**
 * GenerateGVERabbitMapping - GVE-Rabbit hybrid ordering
 * 
 * Fast variant of GVE-Leiden:
 * - Uses GVE-Leiden's optimized local moving
 * - Limited iterations for speed
 * - Single aggregation pass
 * 
 * @tparam K Community ID type (default uint32_t)
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination type
 * @param g Input graph (must be symmetric)
 * @param new_ids Output mapping vector
 * @param reordering_options Options: [resolution, max_iterations]
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void GenerateGVERabbitMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& new_ids,
    const std::vector<std::string>& reordering_options) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options using unified resolution parser
    ResolutionConfig res_cfg;
    int max_iterations = 5;  // Fewer iterations for speed
    
    if (!reordering_options.empty() && !reordering_options[0].empty()) {
        res_cfg = parseResolution<NodeID_T, DestID_T>(reordering_options[0], g);
    } else {
        res_cfg.mode = ResolutionMode::FIXED;
        res_cfg.value = 1.0;  // Default to 1.0 for GVERabbit
    }
    double resolution = res_cfg.getResolution();
    
    if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
        max_iterations = std::stoi(reordering_options[1]);
    }
    
    // Isolated vertex separation
    std::vector<int64_t> isolated_vertices;
    std::vector<int64_t> active_vertices;
    isolated_vertices.reserve(num_nodes / 10);
    active_vertices.reserve(num_nodes);
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (g.out_degree(v) == 0) {
            isolated_vertices.push_back(v);
        } else {
            active_vertices.push_back(v);
        }
    }
    
    const int64_t num_isolated = isolated_vertices.size();
    const int64_t num_active = active_vertices.size();
    
    printf("GVERabbit: resolution=%.4f (%s), max_iterations=%d\n",
           resolution, resolutionModeString(res_cfg.mode).c_str(), max_iterations);
    if (num_isolated > 0) {
        printf("GVERabbit: %ld active vertices, %ld isolated vertices (%.1f%%)\n",
               num_active, num_isolated, 100.0 * num_isolated / num_nodes);
    }
    
    // Run GVE-Rabbit algorithm (uses GVELeidenCSR with limited iterations)
    auto result = GVERabbitCore<K, NodeID_T, DestID_T>(g, resolution, max_iterations);
    
    PrintTime("GVERabbit Aggregation", result.aggregation_time);
    PrintTime("GVERabbit Refinement", result.refinement_time);
    
    tm.Stop();
    double total_time = tm.Seconds();
    
    // Build ordering from communities
    tm.Start();
    
    // Get degrees for hub-first ordering - use int64_t for sort key optimization
    std::vector<int64_t> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        degrees[v] = g.out_degree(v);
    }
    
    // Optimized sort: embed negative degree in sort key to avoid lambda cache misses
    // Key format: (community << 32) | (MAX_DEGREE - degree) for hub-first
    struct SortKey {
        K community;
        int64_t neg_degree;  // Store negative for descending degree
        int64_t vertex;
        
        bool operator<(const SortKey& o) const {
            if (community != o.community) return community < o.community;
            if (neg_degree != o.neg_degree) return neg_degree < o.neg_degree;
            return vertex < o.vertex;
        }
    };
    
    std::vector<SortKey> order_keys(num_nodes);
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        order_keys[v] = {result.community[v], -degrees[v], v};
    }
    
    // Sort: group by community, hub-first within community (using operator<)
    __gnu_parallel::sort(order_keys.begin(), order_keys.end());
    
    // Assign new IDs: active vertices first, isolated at end (parallelized)
    new_ids.resize(num_nodes);
    
    // First pass: count active vertices
    int64_t num_active_from_sort = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
        if (degrees[order_keys[i].vertex] > 0) {
            num_active_from_sort++;
        }
    }
    
    // Assign IDs in sorted order for active vertices
    int64_t current_id = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
        int64_t v = order_keys[i].vertex;
        if (degrees[v] > 0) {
            new_ids[v] = current_id++;
        }
    }
    // Assign remaining IDs to isolated vertices
    for (int64_t v : isolated_vertices) {
        new_ids[v] = current_id++;
    }
    
    tm.Stop();
    double ordering_time = tm.Seconds();
    
    // Count real communities using bitmap instead of unordered_set
    K max_comm = 0;
    for (int64_t v : active_vertices) {
        max_comm = std::max(max_comm, result.community[v]);
    }
    std::vector<char> comm_seen(max_comm + 1, 0);
    size_t num_communities = 0;
    for (int64_t v : active_vertices) {
        K c = result.community[v];
        if (!comm_seen[c]) {
            comm_seen[c] = 1;
            num_communities++;
        }
    }
    
    PrintTime("GVERabbit Communities", static_cast<double>(num_communities));
    if (num_isolated > 0) {
        PrintTime("GVERabbit Isolated", static_cast<double>(num_isolated));
    }
    PrintTime("GVERabbit Modularity", result.modularity);
    PrintTime("GVERabbit Ordering", ordering_time);
    PrintTime("GVERabbit Total", total_time + ordering_time);
}

#endif // REORDER_LEIDEN_H_
