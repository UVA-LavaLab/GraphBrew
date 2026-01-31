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

// Default parameters for AdaptiveOrder
constexpr int DEFAULT_MODE = 1;           // Per-community
constexpr int DEFAULT_RECURSION_DEPTH = 0;
constexpr double DEFAULT_RESOLUTION = 1.0;
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
// ADAPTIVE MODE ENUM
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

inline int AdaptiveModeToInt(AdaptiveMode m) {
    return static_cast<int>(m);
}

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

#endif // REORDER_ADAPTIVE_H_
