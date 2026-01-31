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
// These functions are defined in reorder_types.h and imported here for convenience

using ::ComputeDynamicMinCommunitySize;
using ::ComputeDynamicLocalReorderThreshold;

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

// ============================================================================
// GRAPHBREW STANDALONE IMPLEMENTATIONS
// ============================================================================
// These implementations use global types from reorder_types.h and can be
// called without a BuilderBase instance.

/**
 * @brief GraphBrew with GVE-Leiden clustering (standalone)
 * 
 * Uses GVE-Leiden or GVE-LeidenOpt for community detection, then applies
 * the final reordering algorithm to each community.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination type  
 * @tparam WeightT_ Edge weight type
 * @tparam invert Whether graph has inverse edges
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateGraphBrewGVEMappingStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    bool useOutdeg,
    std::vector<std::string> reordering_options,
    int numLevels = 2,
    bool recursion = false,
    bool use_optimized = false) {
    
    Timer tm, tm_2;
    using EL = EdgeList<NodeID_, DestID_>;
    using EP = EdgePair<NodeID_, DestID_>;
    
    const int64_t num_nodes = g.num_nodes();
    
    // Parse options
    size_t frequency_threshold = 10;
    int final_algo_id = 8;  // RabbitOrder
    double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
    int max_iterations = 20;
    int max_passes = 10;
    
    if (!reordering_options.empty()) {
        frequency_threshold = std::stoi(reordering_options[0]);
    }
    if (reordering_options.size() > 1) {
        final_algo_id = std::stoi(reordering_options[1]);
    }
    if (reordering_options.size() > 2) {
        resolution = std::stod(reordering_options[2]);
        if (resolution > 3) resolution = 1.0;
    }
    
    ReorderingAlgo final_algo = static_cast<ReorderingAlgo>(final_algo_id);
    
    if (recursion && numLevels > 0) {
        numLevels -= 1;
    }
    
    printf("GraphBrewGVE: resolution=%.4f, final_algo=%s, levels=%d, optimized=%d\n",
           resolution, ReorderingAlgoStr(final_algo).c_str(), numLevels, use_optimized);
    
    tm.Start();
    
    // Run GVE-Leiden community detection
    std::vector<K> comm_ids(num_nodes);
    size_t num_communities;
    
    if (use_optimized) {
        auto result = GVELeidenOptCSR<K, WeightT_, NodeID_, DestID_>(g, resolution, 1e-2, 0.8, 10.0, max_iterations, max_passes);
        comm_ids = result.final_community;
        K max_comm = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_comm = std::max(max_comm, comm_ids[v]);
        }
        num_communities = static_cast<size_t>(max_comm + 1);
        printf("GVELeidenOpt: %d passes, modularity=%.6f, communities=%zu\n",
               result.total_passes, result.modularity, num_communities);
    } else {
        auto result = GVELeidenCSR<K, WeightT_, NodeID_, DestID_>(g, resolution, 1e-2, 0.8, 10.0, max_iterations, max_passes);
        comm_ids = result.final_community;
        K max_comm = 0;
        for (int64_t v = 0; v < num_nodes; ++v) {
            max_comm = std::max(max_comm, comm_ids[v]);
        }
        num_communities = static_cast<size_t>(max_comm + 1);
        printf("GVELeidenCSR: %d passes, modularity=%.6f, communities=%zu\n",
               result.total_passes, result.modularity, num_communities);
    }
    
    tm.Stop();
    PrintTime("GraphBrewGVE Community Detection", tm.Seconds());
    
    // Count community sizes
    std::vector<size_t> comm_freq(num_communities, 0);
    for (int64_t v = 0; v < num_nodes; ++v) {
        comm_freq[comm_ids[v]]++;
    }
    
    // Compute average community size for dynamic threshold
    size_t non_empty_communities = 0;
    for (size_t c = 0; c < num_communities; ++c) {
        if (comm_freq[c] > 0) non_empty_communities++;
    }
    size_t avg_community_size = (non_empty_communities > 0) ? 
        static_cast<size_t>(num_nodes) / non_empty_communities : static_cast<size_t>(num_nodes);
    
    // Dynamic threshold for minimum community size
    const size_t MIN_COMMUNITY_SIZE = ComputeDynamicMinCommunitySize(
        static_cast<size_t>(num_nodes), non_empty_communities, avg_community_size);
    
    // Determine actual threshold
    size_t actual_freq_threshold;
    if (frequency_threshold > 0 && frequency_threshold < non_empty_communities) {
        std::vector<size_t> sorted_freq = comm_freq;
        std::nth_element(sorted_freq.begin(), 
                         sorted_freq.begin() + frequency_threshold - 1,
                         sorted_freq.end(),
                         std::greater<size_t>());
        actual_freq_threshold = sorted_freq[frequency_threshold - 1];
    } else {
        actual_freq_threshold = MIN_COMMUNITY_SIZE;
    }
    
    printf("GraphBrewGVE: Dynamic threshold=%zu (avg_comm=%zu, num_comm=%zu)\n",
           actual_freq_threshold, avg_community_size, non_empty_communities);
    
    // Classify communities
    std::vector<NodeID_> small_community_nodes;
    std::vector<bool> is_small_community(num_communities, false);
    
    for (size_t c = 0; c < num_communities; ++c) {
        if (comm_freq[c] > 0 && comm_freq[c] < actual_freq_threshold) {
            is_small_community[c] = true;
        }
    }
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (is_small_community[comm_ids[v]]) {
            small_community_nodes.push_back(static_cast<NodeID_>(v));
        }
    }
    
    // Get large communities
    std::vector<std::pair<size_t, size_t>> freq_comm_pairs;
    for (size_t c = 0; c < num_communities; ++c) {
        if (comm_freq[c] > 0 && !is_small_community[c]) {
            freq_comm_pairs.emplace_back(comm_freq[c], c);
        }
    }
    std::sort(freq_comm_pairs.begin(), freq_comm_pairs.end(), std::greater<>());
    
    std::vector<size_t> top_communities;
    std::vector<bool> is_top_community(num_communities, false);
    for (auto& [freq, comm] : freq_comm_pairs) {
        top_communities.push_back(comm);
        is_top_community[comm] = true;
    }
    
    printf("GraphBrewGVE: %zu large communities for reordering\n", top_communities.size());
    
    // Build edge lists per community
    tm_2.Start();
    
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<EL>> thread_edge_lists(num_threads);
    
    #pragma omp parallel for
    for (int t = 0; t < num_threads; ++t) {
        thread_edge_lists[t].resize(num_communities);
    }
    
    const NodeID_ BLOCK_SIZE = 1024;
    const bool graph_is_weighted = g.is_weighted();
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_edge_lists = thread_edge_lists[tid];
        
        #pragma omp for schedule(dynamic, 1) nowait
        for (NodeID_ block_start = 0; block_start < num_nodes; block_start += BLOCK_SIZE) {
            NodeID_ block_end = std::min(block_start + BLOCK_SIZE, static_cast<NodeID_>(num_nodes));
            
            for (NodeID_ i = block_start; i < block_end; ++i) {
                size_t src_comm = comm_ids[i];
                if (is_top_community[src_comm]) {
                    auto& target_list = local_edge_lists[src_comm];
                    if (graph_is_weighted) {
                        for (DestID_ neighbor : g.out_neigh(i)) {
                            NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                            WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                            target_list.push_back(EP(i, NodeWeight<NodeID_, WeightT_>(dest, weight)));
                        }
                    } else {
                        for (DestID_ neighbor : g.out_neigh(i)) {
                            target_list.push_back(EP(i, neighbor));
                        }
                    }
                }
            }
        }
    }
    
    // Merge thread edge lists
    std::vector<size_t> comm_sizes(num_communities, 0);
    #pragma omp parallel for
    for (size_t c = 0; c < num_communities; ++c) {
        for (int t = 0; t < num_threads; ++t) {
            comm_sizes[c] += thread_edge_lists[t][c].size();
        }
    }
    
    std::vector<EL> community_edge_lists(num_communities);
    #pragma omp parallel for
    for (size_t c = 0; c < num_communities; ++c) {
        if (comm_sizes[c] > 0) {
            community_edge_lists[c].reserve(comm_sizes[c]);
        }
    }
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < num_communities; ++c) {
        if (comm_sizes[c] > 0) {
            auto& comm_edge_list = community_edge_lists[c];
            for (const auto& local_lists : thread_edge_lists) {
                const auto& local_list = local_lists[c];
                comm_edge_list.insert(comm_edge_list.end(), local_list.begin(), local_list.end());
            }
        }
    }
    
    // Apply final reordering to each community
    const size_t LARGE_COMMUNITY_THRESHOLD = 500000;
    const int saved_omp_threads = omp_get_max_threads();
    
    std::vector<std::vector<std::pair<size_t, NodeID_>>> community_id_mappings(num_communities);
    pvector<NodeID_> new_ids_sub(num_nodes);
    
    for (size_t idx = 0; idx < top_communities.size(); ++idx) {
        size_t comm_id = top_communities[idx];
        auto& edge_list = community_edge_lists[comm_id];
        
        if (edge_list.empty()) continue;
        
        const bool is_large = edge_list.size() >= LARGE_COMMUNITY_THRESHOLD;
        
        if (is_large) {
            omp_set_num_threads(saved_omp_threads);
        } else {
            omp_set_num_threads(1);
        }
        
        std::fill(new_ids_sub.begin(), new_ids_sub.end(), (NodeID_)-1);
        
        // Use standalone edge-list reordering
        GenerateMappingLocalEdgelistStandalone<NodeID_, DestID_, WeightT_, invert>(
            edge_list, new_ids_sub, final_algo, useOutdeg, reordering_options);
        
        // Collect results
        for (size_t i = 0; i < new_ids_sub.size(); ++i) {
            if (new_ids_sub[i] != (NodeID_)-1 && comm_ids[i] == comm_id) {
                community_id_mappings[comm_id].emplace_back(i, new_ids_sub[i]);
            }
        }
    }
    
    omp_set_num_threads(saved_omp_threads);
    tm_2.Stop();
    
    // Compute final mapping
    NodeID_ current_id = 0;
    
    // Process small communities first
    if (!small_community_nodes.empty()) {
        std::unordered_set<NodeID_> small_node_set(
            small_community_nodes.begin(), small_community_nodes.end());
        
        auto merged_feat = ComputeMergedCommunityFeatures(g, small_community_nodes, small_node_set);
        ReorderingAlgo small_algo = SelectAlgorithmForSmallGroup(merged_feat);
        
        printf("GraphBrewGVE: Grouped %zu nodes from small communities -> %s\n",
               small_community_nodes.size(), ReorderingAlgoStr(small_algo).c_str());
        
        if (small_algo == ORIGINAL || small_community_nodes.size() < 100) {
            std::vector<std::pair<int64_t, NodeID_>> degree_node_pairs;
            degree_node_pairs.reserve(small_community_nodes.size());
            for (NodeID_ node : small_community_nodes) {
                int64_t deg = useOutdeg ? g.out_degree(node) : g.in_degree(node);
                degree_node_pairs.emplace_back(-deg, node);
            }
            std::sort(degree_node_pairs.begin(), degree_node_pairs.end());
            for (auto& [neg_deg, node] : degree_node_pairs) {
                new_ids[node] = current_id++;
            }
        } else {
            ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                g, small_community_nodes, small_node_set, small_algo, useOutdeg, 
                new_ids, current_id);
        }
    }
    
    // Process large communities
    std::vector<size_t> comm_start_indices(top_communities.size() + 1);
    comm_start_indices[0] = current_id;
    for (size_t c = 0; c < top_communities.size(); ++c) {
        comm_start_indices[c + 1] = comm_start_indices[c] + 
            community_id_mappings[top_communities[c]].size();
    }
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < top_communities.size(); ++c) {
        size_t comm_id = top_communities[c];
        auto& id_pairs = community_id_mappings[comm_id];
        const size_t start_idx = comm_start_indices[c];
        
        for (size_t i = 0; i < id_pairs.size(); ++i) {
            id_pairs[i].second = start_idx + i;
            new_ids[id_pairs[i].first] = (NodeID_)(start_idx + i);
        }
    }
    
    if (!recursion) {
        PrintTime("GraphBrewGVE Map Time", tm_2.Seconds());
    }
    PrintTime("GraphBrewGVE Total Time", tm.Seconds() + tm_2.Seconds());
}

/**
 * @brief GraphBrew with HubCluster-based grouping (standalone)
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateGraphBrewHubClusterMappingStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    bool useOutdeg,
    std::vector<std::string> reordering_options,
    int numLevels = 2) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    const int64_t num_edges = g.num_edges();
    const int64_t avg_degree = num_edges / num_nodes;
    
    int final_algo_id = 8;
    if (reordering_options.size() > 1) {
        final_algo_id = std::stoi(reordering_options[1]);
    }
    ReorderingAlgo final_algo = static_cast<ReorderingAlgo>(final_algo_id);
    
    printf("GraphBrewHubCluster: final_algo=%s, avg_degree=%lld\n",
           ReorderingAlgoStr(final_algo).c_str(), (long long)avg_degree);
    
    // Hub-based clustering
    std::vector<K> comm_ids(num_nodes);
    std::vector<int64_t> hub_vertices;
    
    for (int64_t v = 0; v < num_nodes; ++v) {
        if (g.out_degree(v) > 2 * avg_degree) {
            hub_vertices.push_back(v);
        }
    }
    
    std::fill(comm_ids.begin(), comm_ids.end(), 0);
    
    K next_comm = 1;
    for (int64_t hub : hub_vertices) {
        comm_ids[hub] = next_comm;
        for (DestID_ neighbor : g.out_neigh(hub)) {
            NodeID_ n;
            if constexpr (std::is_same_v<DestID_, NodeID_>) {
                n = neighbor;
            } else {
                n = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
            }
            if (comm_ids[n] == 0) {
                comm_ids[n] = next_comm;
            }
        }
        next_comm++;
    }
    
    printf("GraphBrewHubCluster: %zu hubs, %u communities\n", 
           hub_vertices.size(), next_comm);
    
    tm.Stop();
    PrintTime("GraphBrewHubCluster Community Detection", tm.Seconds());
    
    // Degree-sorted ordering within communities
    typedef std::pair<K, std::pair<int64_t, NodeID_>> comm_deg_node_t;
    std::vector<comm_deg_node_t> sort_keys(num_nodes);
    
    #pragma omp parallel for
    for (int64_t v = 0; v < num_nodes; ++v) {
        sort_keys[v] = {comm_ids[v], {-g.out_degree(v), static_cast<NodeID_>(v)}};
    }
    
    __gnu_parallel::sort(sort_keys.begin(), sort_keys.end());
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        new_ids[sort_keys[i].second.second] = static_cast<NodeID_>(i);
    }
    
    PrintTime("GraphBrewHubCluster Map Time", tm.Seconds());
}

/**
 * @brief Unified GraphBrew entry point (standalone)
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateGraphBrewMappingUnifiedStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    bool useOutdeg,
    std::vector<std::string> reordering_options) {
    
    double auto_resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
    auto cfg = graphbrew::GraphBrewConfig::FromOptions(reordering_options, auto_resolution);
    cfg.print();
    
    auto internal_options = cfg.toInternalOptions();
    
    switch (cfg.cluster) {
        case graphbrew::GraphBrewCluster::GVE:
            GenerateGraphBrewGVEMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
                g, new_ids, useOutdeg, internal_options, cfg.num_levels, false, false);
            break;
        case graphbrew::GraphBrewCluster::GVEOpt:
            GenerateGraphBrewGVEMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
                g, new_ids, useOutdeg, internal_options, cfg.num_levels, false, true);
            break;
        case graphbrew::GraphBrewCluster::HubCluster:
            GenerateGraphBrewHubClusterMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
                g, new_ids, useOutdeg, internal_options, cfg.num_levels);
            break;
        case graphbrew::GraphBrewCluster::Rabbit:
        case graphbrew::GraphBrewCluster::Leiden:
        default:
            GenerateGraphBrewGVEMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
                g, new_ids, useOutdeg, internal_options, cfg.num_levels, false, true);
            break;
    }
}

#endif // REORDER_GRAPHBREW_H_
