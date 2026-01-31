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
// ADAPTIVE ORDER MODES - imported from reorder_types.h
// ============================================================================
using ::AdaptiveMode;
using ::ParseAdaptiveMode;
using ::AdaptiveModeToString;
using ::AdaptiveModeToInt;

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

// Import selection functions from reorder_types.h
using ::SelectBestReorderingForCommunity;
using ::SelectReorderingWithMode;
using ::SelectReorderingPerceptron;
using ::SelectReorderingPerceptronWithFeatures;

// ============================================================================
// DYNAMIC THRESHOLD COMPUTATION
// ============================================================================

// ComputeDynamicMinCommunitySize is now in reorder_types.h at global scope
using ::ComputeDynamicMinCommunitySize;

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

// ============================================================================
// GRAPHBREW CONFIGURATION
// ============================================================================

/**
 * @brief Configuration for GraphBrew reordering algorithm
 * 
 * Parses command-line options and provides a unified configuration
 * for all GraphBrew variants.
 */
struct GraphBrewConfig {
    GraphBrewCluster cluster = GraphBrewCluster::Leiden;
    int final_algo_id = DEFAULT_FINAL_ALGO;
    double resolution = DEFAULT_GRAPHBREW_RESOLUTION;
    int num_levels = DEFAULT_GRAPHBREW_LEVELS;
    size_t frequency_threshold = 10;
    bool use_fast_final = false;
    
    /**
     * Parse configuration from reordering options
     * 
     * Supports two formats:
     * 1. New format: cluster_variant:final_algo:resolution:levels
     * 2. Old format: frequency_threshold:final_algo:resolution:iterations:passes
     * 
     * @param options Vector of option strings
     * @param auto_resolution Default resolution if not specified
     * @return Parsed configuration
     */
    static GraphBrewConfig FromOptions(const std::vector<std::string>& options,
                                        double auto_resolution = DEFAULT_GRAPHBREW_RESOLUTION) {
        GraphBrewConfig cfg;
        cfg.resolution = auto_resolution;
        
        if (options.empty() || options[0].empty()) {
            return cfg;
        }
        
        const std::string& first_opt = options[0];
        bool is_numeric = !first_opt.empty() && 
            std::all_of(first_opt.begin(), first_opt.end(), ::isdigit);
        
        if (is_numeric) {
            // Old format: frequency_threshold:final_algo:resolution:iterations:passes
            cfg.frequency_threshold = std::stoi(first_opt);
            if (options.size() > 1 && !options[1].empty()) {
                cfg.final_algo_id = std::stoi(options[1]);
            }
            if (options.size() > 2 && !options[2].empty()) {
                cfg.resolution = std::stod(options[2]);
                if (cfg.resolution > 3) cfg.resolution = 1.0;
            }
            cfg.cluster = GraphBrewCluster::Leiden;
        } else {
            // New format: cluster_variant:final_algo:resolution:levels
            std::string variant = first_opt;
            
            // Handle "fast" suffix variants
            if (variant.size() >= 4 && variant.substr(variant.size() - 4) == "fast") {
                cfg.use_fast_final = true;
                variant = variant.substr(0, variant.size() - 4);
                if (variant.empty()) variant = "gve";
            }
            
            cfg.cluster = ParseGraphBrewCluster(variant);
            
            // Set default final algorithm based on fast flag
            if (cfg.use_fast_final) {
                cfg.final_algo_id = 6;  // HubSortDBG
            }
            
            if (options.size() > 1 && !options[1].empty()) {
                cfg.final_algo_id = std::stoi(options[1]);
            }
            if (options.size() > 2 && !options[2].empty()) {
                cfg.resolution = std::stod(options[2]);
                if (cfg.resolution > 3) cfg.resolution = 1.0;
            }
            if (options.size() > 3 && !options[3].empty()) {
                cfg.num_levels = std::stoi(options[3]);
            }
        }
        
        return cfg;
    }
    
    /**
     * Print configuration for verbose output
     */
    void print() const {
        printf("GraphBrewOrder: cluster=%s, final_algo=%d, resolution=%.2f, levels=%d\n",
               GraphBrewClusterToString(cluster).c_str(), final_algo_id, resolution, num_levels);
    }
    
    /**
     * Build internal options vector for backward compatibility
     */
    std::vector<std::string> toInternalOptions() const {
        std::vector<std::string> opts;
        opts.push_back(std::to_string(frequency_threshold));
        opts.push_back(std::to_string(final_algo_id));
        opts.push_back(std::to_string(resolution));
        return opts;
    }
};

} // namespace graphbrew

#endif // REORDER_GRAPHBREW_H_
