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
// DEFAULT PARAMETERS
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
 * @param M Total edge weight in graph
 * @param R Resolution parameter
 * @return Delta modularity (positive = beneficial move)
 */
template <typename W>
inline W gveDeltaModularity(W ki_to_c, W ki_to_d, W ki, W sigma_c, W sigma_d, W M, W R) {
    return (ki_to_c - ki_to_d) / M - R * ki * (sigma_c - sigma_d + ki) / (W(2.0) * M * M);
}

} // namespace leiden
} // namespace graphbrew

// ============================================================================
// LEIDEN ALGORITHM IMPLEMENTATIONS
// These are standalone template functions that work with CSRGraph directly
// ============================================================================

#include "../graph.h"
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
    const double total_weight = static_cast<double>(g.num_edges_directed());
    
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
    
    for (pass = 0; pass < max_passes; ++pass) {
        int64_t moves = LeidenLocalMoveParallel<NodeID_T, DestID_T>(
            g, community, comm_weight, vertex_weight,
            total_weight, resolution, max_iterations);
        
        // Count communities
        std::unordered_set<int64_t> unique_comms(community.begin(), community.end());
        int64_t num_comms = unique_comms.size();
        
        printf("Leiden pass %d: %ld moves, %ld communities\n", pass + 1, moves, num_comms);
        
        total_moves += moves;
        if (moves == 0) break;
    }
    
    // Compress communities
    std::unordered_map<int64_t, int64_t> comm_remap;
    int64_t num_comms = 0;
    for (int64_t v = 0; v < num_vertices; ++v) {
        int64_t c = community[v];
        if (comm_remap.find(c) == comm_remap.end()) {
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
    const double total_weight = static_cast<double>(g.num_edges_directed());
    
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

#endif // REORDER_LEIDEN_H_
