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

// ============================================================================
// GRAPH TYPE CLASSIFICATION
// ============================================================================

/**
 * @brief Graph type for algorithm selection
 */
enum class GraphType {
    Social,   ///< Social networks (high clustering)
    Web,      ///< Web graphs (high max degree ratio)
    Road,     ///< Road networks (low degree variance)
    Random,   ///< Random graphs (low modularity)
    Kron,     ///< Kronecker/synthetic graphs
    Unknown   ///< Unknown/mixed type
};

inline std::string GraphTypeToString(GraphType t) {
    switch (t) {
        case GraphType::Social: return "social";
        case GraphType::Web: return "web";
        case GraphType::Road: return "road";
        case GraphType::Random: return "random";
        case GraphType::Kron: return "kron";
        case GraphType::Unknown: return "unknown";
        default: return "unknown";
    }
}

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

// ============================================================================
// BENCHMARK AND SELECTION TYPES
// ============================================================================

/**
 * @brief Benchmark types for algorithm selection
 * 
 * Different graph algorithms have different access patterns:
 * - PageRank: Iterative, benefits from cache locality
 * - BFS: Traversal-heavy, benefits from frontier locality
 * - CC: Union-find based, benefits from parent locality
 * - SSSP: Priority queue based, benefits from distance locality
 * - BC: All-pairs traversal, benefits from both
 * - TC: Neighborhood intersection, benefits from neighbor locality
 */
enum BenchmarkType {
    BENCH_GENERIC = 0,  ///< Generic/default - no benchmark-specific adjustment
    BENCH_PR,           ///< PageRank - iterative, benefits from cache locality
    BENCH_BFS,          ///< Breadth-First Search - traversal-heavy
    BENCH_CC,           ///< Connected Components - union-find based
    BENCH_SSSP,         ///< Single-Source Shortest Path - priority queue based
    BENCH_BC,           ///< Betweenness Centrality - all-pairs traversal
    BENCH_TC            ///< Triangle Counting - neighborhood intersection
};

/**
 * @brief Convert benchmark name string to enum
 * @param name Benchmark name (e.g., "pr", "bfs", "cc")
 * @return BenchmarkType enum value (BENCH_GENERIC if unrecognized)
 */
inline BenchmarkType GetBenchmarkType(const std::string& name) {
    if (name.empty() || name == "generic" || name == "GENERIC" || name == "all") return BENCH_GENERIC;
    if (name == "pr" || name == "PR" || name == "pagerank" || name == "PageRank") return BENCH_PR;
    if (name == "bfs" || name == "BFS") return BENCH_BFS;
    if (name == "cc" || name == "CC") return BENCH_CC;
    if (name == "sssp" || name == "SSSP") return BENCH_SSSP;
    if (name == "bc" || name == "BC") return BENCH_BC;
    if (name == "tc" || name == "TC") return BENCH_TC;
    return BENCH_GENERIC;
}

/**
 * @brief Selection mode for AdaptiveOrder algorithm selection
 * 
 * Controls how AdaptiveOrder selects the best reordering algorithm:
 * 
 * FASTEST_REORDER (0): Select algorithm with lowest reordering time
 *   - Use when reordering cost dominates (single execution)
 *   - Falls back to this mode for UNKNOWN/UNTRAINED graphs
 * 
 * FASTEST_EXECUTION (1): Select algorithm with best cache performance
 *   - Use for repeated algorithm executions
 *   - Uses perceptron weights to predict execution speedup
 *   - Ignores reordering overhead
 * 
 * BEST_ENDTOEND (2): Minimize (reorder_time + execution_time)
 *   - Balanced approach for typical workloads
 *   - Combines perceptron score with reorder time penalty
 * 
 * BEST_AMORTIZATION (3): Minimize iterations to amortize reorder cost
 *   - For scenarios where you need minimum runs to break even
 *   - Considers: reorder_time / (predicted_speedup - 1.0)
 */
enum SelectionMode {
    MODE_FASTEST_REORDER = 0,      ///< Minimize reordering time
    MODE_FASTEST_EXECUTION = 1,    ///< Minimize execution time (perceptron)
    MODE_BEST_ENDTOEND = 2,        ///< Minimize total time
    MODE_BEST_AMORTIZATION = 3     ///< Minimize iterations to amortize
};

/**
 * @brief Convert selection mode to string
 */
inline std::string SelectionModeToString(SelectionMode mode) {
    switch (mode) {
        case MODE_FASTEST_REORDER:    return "fastest-reorder";
        case MODE_FASTEST_EXECUTION:  return "fastest-execution";
        case MODE_BEST_ENDTOEND:      return "best-endtoend";
        case MODE_BEST_AMORTIZATION:  return "best-amortization";
        default:                      return "unknown";
    }
}

/**
 * @brief Convert string to selection mode
 */
inline SelectionMode GetSelectionMode(const std::string& name) {
    if (name == "0" || name == "fastest-reorder" || name == "reorder") return MODE_FASTEST_REORDER;
    if (name == "1" || name == "fastest-execution" || name == "execution" || name == "cache") return MODE_FASTEST_EXECUTION;
    if (name == "2" || name == "best-endtoend" || name == "endtoend" || name == "e2e") return MODE_BEST_ENDTOEND;
    if (name == "3" || name == "best-amortization" || name == "amortization" || name == "amortize") return MODE_BEST_AMORTIZATION;
    return MODE_FASTEST_EXECUTION;  // Default
}

/**
 * @brief Perceptron weights for algorithm selection
 * 
 * These weights are used by AdaptiveOrder to score each candidate algorithm.
 * Higher scores indicate better predicted performance for the given features.
 */
struct PerceptronWeights {
    // Core weights
    double bias;                ///< baseline score
    double w_modularity;        ///< correlation with modularity
    double w_log_nodes;         ///< scale effect
    double w_log_edges;         ///< size effect  
    double w_density;           ///< sparsity effect
    double w_avg_degree;        ///< connectivity effect
    double w_degree_variance;   ///< power-law / uniformity effect
    double w_hub_concentration; ///< hub dominance effect
    
    // Extended graph structure weights
    double w_clustering_coeff = 0.0;   ///< local clustering coefficient effect
    double w_avg_path_length = 0.0;    ///< average path length sensitivity
    double w_diameter = 0.0;           ///< diameter effect
    double w_community_count = 0.0;    ///< sub-community count effect
    
    // Cache impact weights
    double cache_l1_impact = 0.0;      ///< bonus for high L1 hit rate
    double cache_l2_impact = 0.0;      ///< bonus for high L2 hit rate
    double cache_l3_impact = 0.0;      ///< bonus for high L3 hit rate
    double cache_dram_penalty = 0.0;   ///< penalty for DRAM accesses
    
    // Reorder time weight
    double w_reorder_time = 0.0;       ///< penalty for slow reordering
    
    // Metadata from training
    double avg_speedup = 1.0;          ///< average speedup observed
    double avg_reorder_time = 0.0;     ///< average reorder time in seconds
    
    // Benchmark-specific multipliers
    double bench_pr = 1.0;
    double bench_bfs = 1.0;
    double bench_cc = 1.0;
    double bench_sssp = 1.0;
    double bench_bc = 1.0;
    double bench_tc = 1.0;
    
    /**
     * Calculate iterations needed to amortize reorder cost.
     * Lower = better (pays off faster)
     * Returns INFINITY if speedup <= 1.0 (never pays off)
     */
    double iterationsToAmortize() const {
        if (avg_speedup <= 1.0) {
            return std::numeric_limits<double>::infinity();
        }
        double time_saved_per_iter = (avg_speedup - 1.0) / avg_speedup;
        if (time_saved_per_iter <= 0) {
            return std::numeric_limits<double>::infinity();
        }
        return avg_reorder_time / time_saved_per_iter;
    }
    
    /**
     * Get benchmark-specific multiplier
     */
    double getBenchmarkMultiplier(BenchmarkType bench) const {
        switch (bench) {
            case BENCH_PR:   return bench_pr;
            case BENCH_BFS:  return bench_bfs;
            case BENCH_CC:   return bench_cc;
            case BENCH_SSSP: return bench_sssp;
            case BENCH_BC:   return bench_bc;
            case BENCH_TC:   return bench_tc;
            case BENCH_GENERIC:
            default:         return 1.0;
        }
    }
    
    /**
     * Compute base score (without benchmark adjustment)
     */
    double scoreBase(const CommunityFeatures& feat) const {
        double log_nodes = std::log10(static_cast<double>(feat.num_nodes) + 1.0);
        double log_edges = std::log10(static_cast<double>(feat.num_edges) + 1.0);
        
        double s = bias 
             + w_modularity * feat.modularity
             + w_log_nodes * log_nodes
             + w_log_edges * log_edges
             + w_density * feat.internal_density
             + w_avg_degree * feat.avg_degree / 100.0
             + w_degree_variance * feat.degree_variance
             + w_hub_concentration * feat.hub_concentration;
        
        s += w_clustering_coeff * feat.clustering_coeff;
        s += w_avg_path_length * feat.avg_path_length / 10.0;
        s += w_diameter * feat.diameter_estimate / 50.0;
        s += w_community_count * std::log10(feat.community_count + 1.0);
        
        s += cache_l1_impact * 0.5;
        s += cache_l2_impact * 0.3;
        s += cache_l3_impact * 0.2;
        s += cache_dram_penalty;
        
        s += w_reorder_time * feat.reorder_time;
        
        return s;
    }
    
    /**
     * Compute score with optional benchmark-specific adjustment
     */
    double score(const CommunityFeatures& feat, BenchmarkType bench = BENCH_GENERIC) const {
        return scoreBase(feat) * getBenchmarkMultiplier(bench);
    }
    
    /**
     * Compute score (backward compatible)
     */
    double score(const CommunityFeatures& feat) const {
        return scoreBase(feat);
    }
};

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
