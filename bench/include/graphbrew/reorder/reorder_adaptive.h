/**
 * @file reorder_adaptive.h
 * @brief AdaptiveOrder - ML-based Algorithm Selection
 *
 * This header provides the AdaptiveOrder algorithm configuration and utilities.
 * AdaptiveOrder automatically selects the best reordering algorithm based on
 * graph features using a trained perceptron model.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * AdaptiveOrder (ID 14) analyzes graph structure and selects the best algorithm:
 * 
 * 1. FEATURE EXTRACTION
 *    - modularity: Community structure strength (0.0 - 1.0)
 *    - degree_variance: Normalized variance in degree distribution
 *    - hub_concentration: Fraction of edges from top 10% degree nodes
 *    - avg_degree: Average vertex degree
 *    - density: edge_count / (n * (n-1))
 *
 * 2. TYPE MATCHING
 *    - Compare features against trained type centroids (type_0, type_1, etc.)
 *    - Find closest matching graph type using Euclidean distance
 *    - Load corresponding perceptron weights
 *
 * 3. ALGORITHM SELECTION
 *    - Compute perceptron score for each candidate algorithm
 *    - Select algorithm with highest score
 *    - Safety fallback for unavailable algorithms
 *
 * ============================================================================
 * COMMAND LINE FORMAT
 * ============================================================================
 *
 * -o 14[:mode[:depth[:resolution[:min_size[:selection_mode[:graph_name]]]]]]
 *
 * Parameters:
 *   mode: 0=full_graph, 1=per_community (default), 2=recursive
 *   depth: Maximum recursion depth (default: 0)
 *   resolution: Leiden resolution (default: auto)
 *   min_size: Minimum community size for recursion (default: 50000)
 *   selection_mode: 0=fastest-reorder, 1=fastest-execution (default),
 *                   2=best-endtoend, 3=best-amortization
 *   graph_name: Graph name for loading reorder times
 *
 * Examples:
 *   -o 14                           # Default: per-community mode
 *   -o 14:0                         # Full graph mode
 *   -o 14:1:2                       # Per-community with depth=2
 *   -o 14:1:0:1.0:50000:0          # fastest-reorder for unknown graphs
 *
 * ============================================================================
 * SELECTION MODES
 * ============================================================================
 *
 * MODE_FASTEST_REORDER (0):
 *   Select algorithm with lowest reordering time.
 *   Best for: Unknown graphs, one-shot reordering.
 *
 * MODE_FASTEST_EXECUTION (1) [default]:
 *   Use perceptron to predict best cache performance.
 *   Best for: Repeated traversals on known graph types.
 *
 * MODE_BEST_ENDTOEND (2):
 *   Balance perceptron score with reorder time penalty.
 *   Best for: Single execution where total time matters.
 *
 * MODE_BEST_AMORTIZATION (3):
 *   Minimize iterations to amortize reorder cost.
 *   Best for: When you know the iteration count.
 *
 * ============================================================================
 * RECURSIVE MODES
 * ============================================================================
 *
 * Full Graph (mode=0):
 *   Analyze entire graph, select single best algorithm.
 *   Fast but doesn't adapt to local structure.
 *
 * Per Community (mode=1) [default]:
 *   1. Run Leiden community detection
 *   2. Compute features per community
 *   3. Select best algorithm per community
 *   Best balance of quality and overhead.
 *
 * Recursive (mode=2):
 *   1. Run community detection
 *   2. For large communities with good structure:
 *      - Recursively apply AdaptiveOrder
 *   3. For other communities:
 *      - Select algorithm based on features
 *   Best quality but higher overhead.
 *
 * Author: GraphBrew Team
 * License: See LICENSE.txt
 */

#ifndef REORDER_ADAPTIVE_H_
#define REORDER_ADAPTIVE_H_

#include <string>
#include <vector>
#include <cstddef>
#include "reorder_types.h"

namespace adaptive {

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Default parameters for AdaptiveOrder - use unified defaults
constexpr int DEFAULT_MODE = 1;           // Per-community
constexpr int DEFAULT_RECURSION_DEPTH = 0;
constexpr double DEFAULT_RESOLUTION = reorder::DEFAULT_RESOLUTION;
constexpr size_t DEFAULT_MIN_RECURSE_SIZE = 50000;

// Thresholds for recursion decisions
constexpr double RECURSION_MODULARITY_THRESHOLD = 0.3;
constexpr size_t MIN_SIZE_FOR_RECURSION = 10000;

// Feature thresholds for heuristic fallback
namespace thresholds {
    constexpr double HIGH_DEGREE_VARIANCE = 0.8;
    constexpr double HIGH_HUB_CONCENTRATION = 0.3;
    constexpr double LOW_MODULARITY = 0.2;
    constexpr double HIGH_DENSITY = 0.1;
}

// ============================================================================
// ADAPTIVE MODE - imported from reorder_types.h
// ============================================================================
// AdaptiveMode, ParseAdaptiveMode, AdaptiveModeToString, AdaptiveModeToInt
// are defined in reorder_types.h at global scope
using ::AdaptiveMode;
using ::ParseAdaptiveMode;
using ::AdaptiveModeToString;
using ::AdaptiveModeToInt;

// ============================================================================
// CONFIGURATION STRUCTURE
// ============================================================================

/**
 * @brief Configuration for AdaptiveOrder algorithm
 */
struct AdaptiveConfig {
    AdaptiveMode mode = AdaptiveMode::PerCommunity;
    int max_depth = DEFAULT_RECURSION_DEPTH;
    double resolution = DEFAULT_RESOLUTION;
    size_t min_recurse_size = DEFAULT_MIN_RECURSE_SIZE;
    SelectionMode selection_mode = MODE_FASTEST_EXECUTION;
    std::string graph_name = "";
    BenchmarkType benchmark = BENCH_GENERIC;
    bool verbose = false;
    
    /**
     * Parse configuration from reordering options
     * Format: mode:depth:resolution:min_size:selection_mode:graph_name
     */
    static AdaptiveConfig FromOptions(const std::vector<std::string>& options) {
        AdaptiveConfig cfg;
        
        // Parse mode (param 0)
        if (options.size() > 0 && !options[0].empty()) {
            try {
                int mode_int = std::stoi(options[0]);
                cfg.mode = ParseAdaptiveMode(mode_int);
            } catch (...) {}
        }
        
        // Parse max_depth (param 1)
        if (options.size() > 1 && !options[1].empty()) {
            try {
                cfg.max_depth = std::stoi(options[1]);
            } catch (...) {}
        }
        
        // Parse resolution (param 2)
        if (options.size() > 2 && !options[2].empty()) {
            try {
                cfg.resolution = std::stod(options[2]);
            } catch (...) {}
        }
        
        // Parse min_recurse_size (param 3)
        if (options.size() > 3 && !options[3].empty()) {
            try {
                cfg.min_recurse_size = std::stoull(options[3]);
            } catch (...) {}
        }
        
        // Parse selection_mode (param 4)
        if (options.size() > 4 && !options[4].empty()) {
            try {
                int sm = std::stoi(options[4]);
                switch (sm) {
                    case 0: cfg.selection_mode = MODE_FASTEST_REORDER; break;
                    case 1: cfg.selection_mode = MODE_FASTEST_EXECUTION; break;
                    case 2: cfg.selection_mode = MODE_BEST_ENDTOEND; break;
                    case 3: cfg.selection_mode = MODE_BEST_AMORTIZATION; break;
                    default: cfg.selection_mode = MODE_FASTEST_EXECUTION; break;
                }
            } catch (...) {}
        }
        
        // Parse graph_name (param 5)
        if (options.size() > 5 && !options[5].empty()) {
            cfg.graph_name = options[5];
        }
        
        return cfg;
    }
    
    /**
     * Print configuration for verbose output
     */
    void print() const {
        std::cout << "AdaptiveOrder Configuration:\n";
        std::cout << "  Mode: " << AdaptiveModeToString(mode) << "\n";
        std::cout << "  Max Depth: " << max_depth << "\n";
        std::cout << "  Resolution: " << resolution << "\n";
        std::cout << "  Min Recurse Size: " << min_recurse_size << "\n";
        std::cout << "  Selection Mode: " << SelectionModeToString(selection_mode) << "\n";
        if (!graph_name.empty()) {
            std::cout << "  Graph Name: " << graph_name << "\n";
        }
    }
};

// ============================================================================
// HEURISTIC FALLBACK
// ============================================================================

/**
 * Heuristic algorithm selection for edge cases.
 * 
 * Used when:
 * - Perceptron selects unavailable algorithm (e.g., RabbitOrder when disabled)
 * - Very small communities where perceptron overhead isn't worth it
 * 
 * @param feat Community features
 * @return Selected algorithm based on simple heuristics
 */
inline ReorderingAlgo SelectHeuristicFallback(const CommunityFeatures& feat) {
    // High degree variance → hub-based approaches work well
    if (feat.degree_variance > thresholds::HIGH_DEGREE_VARIANCE) {
        return HubClusterDBG;
    }
    
    // High hub concentration → hub sorting helps
    if (feat.hub_concentration > thresholds::HIGH_HUB_CONCENTRATION) {
        return HubSort;
    }
    
    // Low modularity, high density → simple grouping is sufficient
    if (feat.modularity < thresholds::LOW_MODULARITY && 
        feat.internal_density > thresholds::HIGH_DENSITY) {
        return Sort;
    }
    
    // Default: DBG is a safe middle ground
    return DBG;
}

/**
 * Check if a community is suitable for recursive processing.
 * 
 * Criteria:
 * 1. Large enough to benefit from recursion
 * 2. Has sufficient community structure (modularity)
 * 3. Not too dense (dense graphs don't need complex reordering)
 * 
 * @param feat Community features
 * @param min_size Minimum size for recursion
 * @return true if community should be recursively processed
 */
inline bool ShouldRecurse(const CommunityFeatures& feat, size_t min_size) {
    // Too small
    if (feat.num_nodes < min_size) return false;
    
    // Has community structure
    if (feat.modularity < RECURSION_MODULARITY_THRESHOLD) return false;
    
    // Not too dense
    if (feat.internal_density > 0.1) return false;
    
    return true;
}

} // namespace adaptive

// Import key types and functions into global scope for convenience
using adaptive::AdaptiveMode;
using adaptive::AdaptiveConfig;
using adaptive::SelectHeuristicFallback;
using adaptive::ShouldRecurse;

// ============================================================================
// STANDALONE ADAPTIVE IMPLEMENTATIONS
// ============================================================================

/**
 * @brief Full-Graph Adaptive Mode (Standalone)
 * 
 * Analyzes entire graph features and selects a single best algorithm.
 * Uses ApplyBasicReorderingStandalone for algorithm dispatch.
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateAdaptiveMappingFullGraphStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    bool useOutdeg,
    const std::vector<std::string>& reordering_options = {}) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    const int64_t num_edges = g.num_edges_directed();
    
    std::cout << "=== Full-Graph Adaptive Mode (Standalone) ===\n";
    std::cout << "Nodes: " << num_nodes << ", Edges: " << num_edges << "\n";
    
    // GUARD: Empty graph - create identity mapping
    if (num_nodes == 0) {
        tm.Stop();
        PrintTime("AdaptiveOrder Total Time", tm.Seconds());
        return;
    }
    
    // Compute global features
    auto features = ::ComputeSampledDegreeFeatures(g, 10000, true);
    
    double global_modularity = features.estimated_modularity;
    double global_degree_variance = features.degree_variance;
    double global_hub_concentration = features.hub_concentration;
    double global_avg_degree = (num_nodes > 0) ? static_cast<double>(num_edges) / num_nodes : 0.0;
    double clustering_coeff = features.clustering_coeff;
    
    // Detect graph type
    GraphType detected_graph_type = DetectGraphType(
        global_modularity, global_degree_variance, global_hub_concentration,
        global_avg_degree, static_cast<size_t>(num_nodes));
    
    std::cout << "Graph Type: " << GraphTypeToString(detected_graph_type) << "\n";
    PrintTime("Degree Variance", global_degree_variance);
    PrintTime("Hub Concentration", global_hub_concentration);
    
    // Create community features for the whole graph
    CommunityFeatures global_feat;
    global_feat.num_nodes = num_nodes;
    global_feat.num_edges = num_edges;
    global_feat.internal_density = global_avg_degree / (num_nodes - 1);
    global_feat.degree_variance = global_degree_variance;
    global_feat.hub_concentration = global_hub_concentration;
    global_feat.clustering_coeff = clustering_coeff;
    
    // Select best algorithm
    ReorderingAlgo best_algo = SelectBestReorderingForCommunity(
        global_feat, global_modularity, global_degree_variance, global_hub_concentration,
        global_avg_degree, static_cast<size_t>(num_nodes), num_edges,
        BENCH_GENERIC, detected_graph_type);
    
    std::cout << "\n=== Selected Algorithm: " << ReorderingAlgoStr(best_algo) << " ===\n";
    
    // Use standalone dispatcher
    ApplyBasicReorderingStandalone<NodeID_, DestID_, WeightT_, invert>(
        g, new_ids, best_algo, useOutdeg, "");
    
    tm.Stop();
    PrintTime("Full-Graph Adaptive Time", tm.Seconds());
}

/**
 * @brief Recursive Adaptive Mapping (Standalone)
 * 
 * Uses GVE-Leiden for community detection (native, no external library).
 * For each community, selects best algorithm based on features.
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateAdaptiveMappingRecursiveStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    bool useOutdeg,
    const std::vector<std::string>& reordering_options,
    int depth = 0,
    bool verbose = true,
    SelectionMode selection_mode = MODE_FASTEST_EXECUTION,
    const std::string& graph_name = "") {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    const int64_t num_edges = g.num_edges_directed();
    
    // Parse options
    int MAX_DEPTH = 0;
    size_t MIN_COMMUNITY_FOR_RECURSION = 50000;
    double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
    int max_iterations = 30;
    int max_passes = 30;
    
    if (reordering_options.size() > 0) {
        double first_val = std::stod(reordering_options[0]);
        if (first_val >= 0 && first_val <= 10 && std::floor(first_val) == first_val) {
            MAX_DEPTH = static_cast<int>(first_val);
        } else {
            resolution = first_val;
        }
    }
    if (reordering_options.size() > 1) {
        resolution = std::stod(reordering_options[1]);
    }
    if (reordering_options.size() > 2) {
        MIN_COMMUNITY_FOR_RECURSION = std::stoul(reordering_options[2]);
    }
    
    if (depth == 0 && verbose) {
        PrintTime("Max Depth", static_cast<double>(MAX_DEPTH));
        PrintTime("Resolution", resolution);
        PrintTime("Min Recurse Size", static_cast<double>(MIN_COMMUNITY_FOR_RECURSION));
    }
    
    // Use GVE-Leiden for community detection (native CSR)
    // Note: W must be double for proper modularity computation (not WeightT_)
    auto leiden_result = GVELeidenOptCSR<K, double, NodeID_, DestID_>(
        g, resolution, DEFAULT_TOLERANCE, DEFAULT_AGGREGATION_TOLERANCE,
        DEFAULT_QUALITY_FACTOR, max_iterations, max_passes);
    
    std::vector<K> comm_ids_k = leiden_result.final_community;
    double global_modularity = leiden_result.modularity;
    
    // Convert to size_t
    std::vector<size_t> comm_ids(num_nodes);
    K max_comm = 0;
    for (int64_t v = 0; v < num_nodes; ++v) {
        comm_ids[v] = static_cast<size_t>(comm_ids_k[v]);
        max_comm = std::max(max_comm, comm_ids_k[v]);
    }
    size_t num_communities = static_cast<size_t>(max_comm + 1);
    
    if (depth == 0 && verbose) {
        PrintTime("Modularity", global_modularity);
        PrintTime("Num Communities", static_cast<double>(num_communities));
    }
    
    // Compute global features
    auto deg_features = ::ComputeSampledDegreeFeatures(g, 10000, true);
    double global_degree_variance = deg_features.degree_variance;
    double global_hub_concentration = deg_features.hub_concentration;
    double global_avg_degree = (num_nodes > 0) ? static_cast<double>(num_edges) / num_nodes : 0.0;
    
    // Detect graph type
    GraphType detected_graph_type = DetectGraphType(
        global_modularity, global_degree_variance, global_hub_concentration,
        global_avg_degree, static_cast<size_t>(num_nodes));
    
    if (depth == 0 && verbose) {
        std::cout << "Graph Type: " << GraphTypeToString(detected_graph_type) << "\n";
        PrintTime("Degree Variance", global_degree_variance);
        PrintTime("Hub Concentration", global_hub_concentration);
    }
    
    // Count community sizes
    std::vector<size_t> comm_freq(num_communities, 0);
    for (int64_t v = 0; v < num_nodes; ++v) {
        comm_freq[comm_ids[v]]++;
    }
    
    // Compute dynamic thresholds
    size_t non_empty_communities = 0;
    for (size_t c = 0; c < num_communities; ++c) {
        if (comm_freq[c] > 0) non_empty_communities++;
    }
    size_t avg_community_size = (non_empty_communities > 0) ?
        static_cast<size_t>(num_nodes) / non_empty_communities : static_cast<size_t>(num_nodes);
    
    const size_t MIN_FEATURES_SAMPLE = ComputeDynamicMinCommunitySize(
        static_cast<size_t>(num_nodes), non_empty_communities, avg_community_size);
    const size_t MIN_LOCAL_REORDER = ComputeDynamicLocalReorderThreshold(
        static_cast<size_t>(num_nodes), non_empty_communities, avg_community_size);
    
    if (depth == 0 && verbose) {
        printf("\n=== Adaptive Reordering Selection (Depth %d, Modularity: %.4f) ===\n",
               depth, global_modularity);
        printf("Dynamic thresholds: MIN_FEATURES=%zu, MIN_LOCAL_REORDER=%zu (avg_comm=%zu, num_comm=%zu)\n",
               MIN_FEATURES_SAMPLE, MIN_LOCAL_REORDER, avg_community_size, non_empty_communities);
    }
    
    // Collect small community nodes
    std::vector<NodeID_> small_community_nodes;
    std::vector<bool> is_small_community(num_communities, false);
    
    for (size_t c = 0; c < num_communities; ++c) {
        if (comm_freq[c] > 0 && comm_freq[c] < MIN_LOCAL_REORDER) {
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
        if (comm_freq[c] >= MIN_LOCAL_REORDER) {
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
    
    // Process communities and assign new IDs
    NodeID_ current_id = 0;
    
    // First: handle small communities
    if (!small_community_nodes.empty()) {
        std::unordered_set<NodeID_> small_node_set(
            small_community_nodes.begin(), small_community_nodes.end());
        
        auto merged_feat = ::ComputeMergedCommunityFeatures(g, small_community_nodes, small_node_set);
        ReorderingAlgo small_algo = ::SelectAlgorithmForSmallGroup(merged_feat);
        
        if (verbose) {
            printf("AdaptiveOrder: Grouped %zu small communities (%zu nodes, %zu edges) -> %s\n",
                   non_empty_communities - top_communities.size(),
                   small_community_nodes.size(), merged_feat.num_edges,
                   ReorderingAlgoStr(small_algo).c_str());
        }
        
        if (small_algo == ORIGINAL || small_community_nodes.size() < 100) {
            // Simple degree sort
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
                g, small_community_nodes, small_node_set, small_algo, useOutdeg, new_ids, current_id);
        }
    }
    
    // Then: process large communities
    for (size_t comm_id : top_communities) {
        // Collect nodes in this community
        std::vector<NodeID_> comm_nodes;
        comm_nodes.reserve(comm_freq[comm_id]);
        for (int64_t v = 0; v < num_nodes; ++v) {
            if (comm_ids[v] == comm_id) {
                comm_nodes.push_back(static_cast<NodeID_>(v));
            }
        }
        
        std::unordered_set<NodeID_> comm_node_set(comm_nodes.begin(), comm_nodes.end());
        
        // Compute features for this community
        auto feat = ComputeCommunityFeaturesStandalone<NodeID_, DestID_, invert>(
            comm_nodes, g, comm_node_set);
        
        // Select algorithm for this community
        ReorderingAlgo selected_algo = SelectBestReorderingForCommunity(
            feat, global_modularity, global_degree_variance, global_hub_concentration,
            global_avg_degree, static_cast<size_t>(num_nodes), num_edges,
            BENCH_GENERIC, detected_graph_type);
        
        if (verbose && comm_nodes.size() >= MIN_FEATURES_SAMPLE) {
            printf("  Community %zu: %zu nodes, %zu edges -> %s\n",
                   comm_id, comm_nodes.size(), feat.num_edges, 
                   ReorderingAlgoStr(selected_algo).c_str());
        }
        
        // Apply algorithm
        ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, comm_nodes, comm_node_set, selected_algo, useOutdeg, new_ids, current_id);
    }
    
    tm.Stop();
    if (!verbose || depth == 0) {
        PrintTime("Adaptive Map Time", tm.Seconds());
    }
}

/**
 * @brief Main Adaptive entry point (Standalone)
 * 
 * Parses options and dispatches to appropriate mode.
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateAdaptiveMappingStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    bool useOutdeg,
    const std::vector<std::string>& reordering_options) {
    
    SelectionMode selection_mode = MODE_FASTEST_EXECUTION;
    std::string graph_name = "";
    
    if (reordering_options.size() > 3) {
        try {
            int mode_val = std::stoi(reordering_options[3]);
            if (mode_val >= 0 && mode_val <= 3) {
                selection_mode = static_cast<SelectionMode>(mode_val);
            } else if (mode_val == 100) {
                // Legacy full-graph mode
                GenerateAdaptiveMappingFullGraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                    g, new_ids, useOutdeg, reordering_options);
                return;
            }
        } catch (...) {
            selection_mode = GetSelectionMode(reordering_options[3]);
        }
    }
    
    if (reordering_options.size() > 4) {
        graph_name = reordering_options[4];
    }
    
    printf("AdaptiveOrder: Selection Mode: %s", SelectionModeToString(selection_mode).c_str());
    if (!graph_name.empty()) {
        printf(" (graph: %s)", graph_name.c_str());
    }
    printf("\n");
    fflush(stdout);
    
    GenerateAdaptiveMappingRecursiveStandalone<NodeID_, DestID_, WeightT_, invert>(
        g, new_ids, useOutdeg, reordering_options, 0, true, selection_mode, graph_name);
}

#endif // REORDER_ADAPTIVE_H_
