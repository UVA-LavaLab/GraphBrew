/**
 * @file reorder_graphbrew.h
 * @brief GraphBrew and Adaptive multi-level reordering - API Reference
 *
 * This header provides documentation and utility types for GraphBrew's advanced
 * reordering algorithms. The core implementations remain in builder.h due to
 * tight integration with the Leiden community detection system.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * GRAPHBREWORDER (ID 12) - Multi-level community-aware reordering
 *   Format: -o 12:cluster_variant:final_algo:resolution:levels
 *   Example: ./bench/bin/pr -f graph.el -o 12:gve:8:0.75
 *   
 *   Combines community detection with per-community reordering:
 *   1. Detect communities using specified clustering algorithm
 *   2. Apply final reordering algorithm within each community
 *   3. Arrange communities to maximize locality
 *
 * ADAPTIVEORDER (ID 14) - ML-based algorithm selection
 *   Format: -o 14:mode:recursive_depth
 *   Example: ./bench/bin/pr -f graph.el -o 14:1
 *   
 *   Automatically selects the best reordering algorithm based on graph features:
 *   1. Extract graph features (modularity, degree distribution, etc.)
 *   2. Use trained perceptron to predict best algorithm
 *   3. Apply selected algorithm
 *
 * ============================================================================
 * GRAPHBREWORDER DETAILS
 * ============================================================================
 *
 * Cluster Variants:
 * -----------------
 * leiden (default): Standard igraph Leiden
 *   - High quality community detection
 *   - Requires igraph library
 *   
 * gve: GVE-Leiden (native CSR)
 *   - Fast parallel implementation
 *   - Good balance of speed and quality
 *   
 * gveopt: Cache-optimized GVE-Leiden
 *   - Best for large graphs (>10M edges)
 *   - Uses prefetching and cache-aware access
 *   
 * rabbit: RabbitOrder's Louvain
 *   - Uses RabbitOrder's internal clustering
 *   - Good for power-law graphs
 *   
 * hubcluster: HubCluster partitioning
 *   - Fast hub-based clustering
 *   - Best for graphs with clear hub structure
 *
 * Final Algorithms (per-community):
 * ---------------------------------
 * Any algorithm ID 0-11 can be used:
 *   0: ORIGINAL - Keep original order within community
 *   1: RANDOM - Random shuffle within community
 *   2: SORT - Sort by vertex ID
 *   3: HUBSORT - Sort by degree (hubs first)
 *   4: HUBCLUSTER - Cluster hubs with neighbors
 *   5: DBG - Degree-based grouping
 *   6: HUBSORTDBG - HubSort within DBG buckets
 *   7: HUBCLUSTERDBG - HubCluster within DBG (recommended)
 *   8: RABBITORDER - Full RabbitOrder (default, best quality)
 *   9: GORDER - G-Order optimization
 *   10: CORDER - Cache-aware ordering
 *   11: RCM - Reverse Cuthill-McKee
 *
 * ============================================================================
 * ADAPTIVEORDER DETAILS
 * ============================================================================
 *
 * Selection Modes:
 * ----------------
 * Mode 0 (full_graph):
 *   Analyze entire graph, select single best algorithm.
 *   Fast but doesn't adapt to local structure.
 *
 * Mode 1 (per_community) [default]:
 *   Run community detection, then select algorithm per community.
 *   Best balance of quality and overhead.
 *
 * Mode 2 (recursive):
 *   Recursively partition and select algorithms.
 *   Best quality but higher overhead.
 *
 * Graph Features Used:
 * --------------------
 * - modularity: Community structure strength (0.0 - 1.0)
 * - avg_degree: Average vertex degree
 * - degree_variance: Variance in degree distribution
 * - max_degree_ratio: max_degree / avg_degree
 * - density: edge_count / (n * (n-1))
 * - num_communities: Detected community count
 * - avg_community_size: n / num_communities
 * - clustering_coefficient: Local clustering metric
 *
 * Graph Type Classification:
 * --------------------------
 * Based on features, graphs are classified:
 * - social: High clustering, power-law degrees
 * - web: Very high max_degree_ratio, sparse
 * - road: Low degree variance, mesh-like
 * - random: Low modularity, uniform degrees
 * - kron: Synthetic Kronecker patterns
 *
 * Algorithm Selection:
 * --------------------
 * Pre-trained perceptron weights stored in:
 *   scripts/weights/active/type_*.json
 *
 * Selection formula:
 *   score = bias + Σ(weight_i * normalized_feature_i)
 * Highest scoring algorithm is selected.
 *
 * ============================================================================
 * IMPLEMENTATION DETAILS
 * ============================================================================
 *
 * Core Functions in builder.h:
 * ----------------------------
 * GenerateGraphBrewMappingUnified() - Main GraphBrew entry point
 * GenerateGraphBrewGVEMapping()     - GVE variant
 * GenerateGraphBrewRabbitMapping()  - Rabbit variant
 * GenerateGraphBrewHubClusterMapping() - HubCluster variant
 * GenerateGraphBrewMapping()        - Generic implementation
 *
 * GenerateAdaptiveMapping()         - Main Adaptive entry point
 * GenerateAdaptiveMappingFullGraph() - Full-graph mode
 * GenerateAdaptiveMappingRecursive() - Recursive mode
 *
 * Helper Functions:
 * -----------------
 * SelectFastestReorderFromWeights() - Perceptron-based selection
 * LoadPerceptronWeights()           - Load weights from JSON
 * ComputeGraphFeatures()            - Extract graph features
 * DetectGraphType()                 - Classify graph type
 *
 * ============================================================================
 * PERFORMANCE RECOMMENDATIONS
 * ============================================================================
 *
 * GraphBrewOrder:
 * ---------------
 * Small graphs (<1M edges): Use -o 12:leiden:8 (quality)
 * Medium graphs (1M-100M):  Use -o 12:gve:8 (balanced)
 * Large graphs (>100M):     Use -o 12:gveopt:7 (speed)
 * 
 * If community structure is weak (modularity < 0.3):
 *   Consider using -o 7 (HubClusterDBG) directly instead
 *
 * AdaptiveOrder:
 * --------------
 * Unknown graph type: Use -o 14:1 (per-community adaptive)
 * Speed critical:     Use -o 14:0 (full-graph, single analysis)
 * Best quality:       Use -o 14:2:3 (recursive, depth 3)
 *
 * ============================================================================
 * EXAMPLES
 * ============================================================================
 *
 * Default GraphBrew (Leiden + RabbitOrder):
 *   ./bench/bin/pr -f graph.el -o 12 -n 5
 *
 * GraphBrew with GVE and HubClusterDBG:
 *   ./bench/bin/pr -f graph.el -o 12:gve:7 -n 5
 *
 * GraphBrew with high resolution (more communities):
 *   ./bench/bin/pr -f graph.el -o 12:gve:8:1.5 -n 5
 *
 * Per-community adaptive:
 *   ./bench/bin/pr -f graph.el -o 14 -n 5
 *
 * Full-graph adaptive:
 *   ./bench/bin/pr -f graph.el -o 14:0 -n 5
 *
 * Recursive adaptive (depth 3):
 *   ./bench/bin/pr -f graph.el -o 14:2:3 -n 5
 */

#ifndef REORDER_GRAPHBREW_H_
#define REORDER_GRAPHBREW_H_

#include <string>
#include <cstdint>
#include <cmath>
#include <limits>
#include "reorder_types.h"

namespace graphbrew {

// ============================================================================
// GRAPHBREW CLUSTER VARIANTS
// ============================================================================

/**
 * @brief Community detection algorithm for GraphBrewOrder
 */
enum class GraphBrewCluster {
    Leiden,      ///< Standard Leiden via igraph (default)
    GVE,         ///< GVE-Leiden (native CSR, fast)
    GVEOpt,      ///< Cache-optimized GVE-Leiden
    Rabbit,      ///< RabbitOrder's Louvain variant
    HubCluster   ///< HubCluster-based partitioning
};

inline GraphBrewCluster ParseGraphBrewCluster(const std::string& s) {
    if (s == "leiden" || s.empty()) return GraphBrewCluster::Leiden;
    if (s == "gve") return GraphBrewCluster::GVE;
    if (s == "gveopt") return GraphBrewCluster::GVEOpt;
    if (s == "rabbit") return GraphBrewCluster::Rabbit;
    if (s == "hubcluster") return GraphBrewCluster::HubCluster;
    return GraphBrewCluster::Leiden;
}

inline std::string GraphBrewClusterToString(GraphBrewCluster c) {
    switch (c) {
        case GraphBrewCluster::Leiden: return "leiden";
        case GraphBrewCluster::GVE: return "gve";
        case GraphBrewCluster::GVEOpt: return "gveopt";
        case GraphBrewCluster::Rabbit: return "rabbit";
        case GraphBrewCluster::HubCluster: return "hubcluster";
        default: return "unknown";
    }
}

// ============================================================================
// ADAPTIVE ORDER MODES
// ============================================================================

/**
 * @brief Selection mode for AdaptiveOrder
 */
enum class AdaptiveMode {
    FullGraph = 0,    ///< Analyze entire graph, select one algorithm
    PerCommunity = 1, ///< Select algorithm per community (default)
    Recursive = 2     ///< Recursively partition and select
};

inline AdaptiveMode ParseAdaptiveMode(int mode) {
    switch (mode) {
        case 0: return AdaptiveMode::FullGraph;
        case 1: return AdaptiveMode::PerCommunity;
        case 2: return AdaptiveMode::Recursive;
        default: return AdaptiveMode::PerCommunity;
    }
}

inline std::string AdaptiveModeToString(AdaptiveMode m) {
    switch (m) {
        case AdaptiveMode::FullGraph: return "full_graph";
        case AdaptiveMode::PerCommunity: return "per_community";
        case AdaptiveMode::Recursive: return "recursive";
        default: return "unknown";
    }
}

// Note: GraphType, DetectGraphType, GetGraphType, GraphTypeToString, BenchmarkType, 
// GetBenchmarkType, SelectionMode, SelectionModeToString, GetSelectionMode are now 
// defined in reorder_types.h at global scope.
// Import them into graphbrew namespace for backward compatibility.
using ::GraphType;
using ::GRAPH_GENERIC;
using ::GRAPH_SOCIAL;
using ::GRAPH_ROAD;
using ::GRAPH_WEB;
using ::GRAPH_POWERLAW;
using ::GRAPH_UNIFORM;
using ::GraphTypeToString;
using ::GetGraphType;
using ::DetectGraphType;
using ::BenchmarkType;
using ::BENCH_GENERIC;
using ::BENCH_PR;
using ::BENCH_BFS;
using ::BENCH_CC;
using ::BENCH_SSSP;
using ::BENCH_BC;
using ::BENCH_TC;
using ::GetBenchmarkType;
using ::SelectionMode;
using ::MODE_FASTEST_REORDER;
using ::MODE_FASTEST_EXECUTION;
using ::MODE_BEST_ENDTOEND;
using ::MODE_BEST_AMORTIZATION;
using ::SelectionModeToString;
using ::GetSelectionMode;

// ============================================================================
// DEFAULT PARAMETERS
// ============================================================================

// GraphBrew defaults
constexpr double DEFAULT_GRAPHBREW_RESOLUTION = 0.75;
constexpr int DEFAULT_GRAPHBREW_LEVELS = 1;
constexpr int DEFAULT_FINAL_ALGO = 8;  // RabbitOrder

// Adaptive defaults
constexpr int DEFAULT_ADAPTIVE_MODE = 1;  // PerCommunity
constexpr int DEFAULT_RECURSIVE_DEPTH = 3;

// Community size thresholds
constexpr size_t SMALL_COMMUNITY_THRESHOLD = 100;
constexpr size_t LARGE_COMMUNITY_THRESHOLD = 10000;

// Feature thresholds for graph type detection
namespace thresholds {
    constexpr double SOCIAL_MIN_MODULARITY = 0.4;
    constexpr double SOCIAL_MIN_DEGREE_RATIO = 10.0;
    constexpr double WEB_MIN_DEGREE_RATIO = 100.0;
    constexpr double WEB_MAX_DENSITY = 0.001;
    constexpr double ROAD_MAX_DEGREE_VARIANCE = 2.0;
    constexpr double ROAD_MAX_AVG_DEGREE = 5.0;
    constexpr double RANDOM_MAX_MODULARITY = 0.2;
}

// PerceptronWeights and CommunityFeatures are defined in reorder_types.h at global scope
using ::PerceptronWeights;
using ::CommunityFeatures;

// ============================================================================
// DYNAMIC THRESHOLD COMPUTATION
// ============================================================================

/**
 * Compute dynamic minimum community size threshold.
 * 
 * Key design decisions:
 * 1. Relative to average community size (not absolute)
 * 2. Also relative to sqrt(N) as classic heuristic 
 * 3. Has minimum floor (ABSOLUTE_MIN = 50)
 * 4. Always relative to community structure discovered
 * 
 * @param num_nodes Total nodes in graph
 * @param num_communities Number of communities detected
 * @param avg_community_size Average community size (num_nodes / num_communities)
 * @return Dynamic threshold for minimum community size
 */
inline size_t ComputeDynamicMinCommunitySize(size_t num_nodes, 
                                              size_t num_communities,
                                              size_t avg_community_size = 0)
{
    const size_t ABSOLUTE_MIN = 50;      // Never go below this
    const size_t FACTOR = 4;             // Communities < avg/4 are "small"
    const size_t MAX_THRESHOLD = 2000;   // Cap for very large graphs
    
    // Compute average if not provided
    if (avg_community_size == 0 && num_communities > 0) {
        avg_community_size = num_nodes / num_communities;
    }
    
    // Threshold based on average community size
    size_t avg_based = (avg_community_size > 0) ? avg_community_size / FACTOR : ABSOLUTE_MIN;
    
    // Threshold based on sqrt(N) - classic heuristic
    size_t sqrt_based = static_cast<size_t>(std::sqrt(static_cast<double>(num_nodes)));
    
    // Take minimum of avg-based and sqrt-based (don't want threshold too high)
    // But ensure at least ABSOLUTE_MIN
    size_t threshold = std::max(ABSOLUTE_MIN, std::min(avg_based, sqrt_based));
    
    // Cap at MAX_THRESHOLD for very large graphs
    threshold = std::min(threshold, MAX_THRESHOLD);
    
    return threshold;
}

/**
 * Compute dynamic threshold for when to apply local reordering.
 * Communities smaller than this get simple degree-sorting instead of
 * expensive subgraph construction + algorithm application.
 * 
 * This is slightly higher than min_community_size because subgraph
 * construction has overhead.
 * 
 * @param num_nodes Total nodes in graph
 * @param num_communities Number of communities
 * @param avg_community_size Average community size
 * @return Threshold for local reordering (higher = more batching)
 */
inline size_t ComputeDynamicLocalReorderThreshold(size_t num_nodes,
                                                   size_t num_communities,
                                                   size_t avg_community_size = 0)
{
    // Local reorder threshold is 2x the min community size
    // because subgraph construction has significant overhead
    size_t min_size = ComputeDynamicMinCommunitySize(num_nodes, num_communities, avg_community_size);
    return std::min(min_size * 2, static_cast<size_t>(5000));  // Cap at 5000
}

} // namespace graphbrew

#endif // REORDER_GRAPHBREW_H_
