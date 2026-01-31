// ============================================================================
// GraphBrew - Reordering Algorithm Dispatcher
// ============================================================================
// This is the main header that includes all reordering algorithm implementations
// and provides the unified GenerateMapping() dispatch function.
//
// Architecture Overview:
// ┌─────────────────────────────────────────────────────────────────────────┐
// │                           reorder.h (this file)                         │
// │                     Main dispatcher and includes                        │
// └─────────────────────────────────────────────────────────────────────────┘
//                                    │
//          ┌──────────────┬─────────┼─────────┬──────────────┐
//          │              │         │         │              │
//          ▼              ▼         ▼         ▼              ▼
// ┌────────────┐  ┌────────────┐  ┌───┐  ┌────────────┐  ┌────────────┐
// │reorder_    │  │reorder_    │  │...│  │reorder_    │  │reorder_    │
// │basic.h     │  │hub.h       │  │   │  │leiden.h    │  │graphbrew.h │
// │(0,1,2)     │  │(3,4,5,6,7) │  │   │  │(15,16,17)  │  │(12,14)     │
// └────────────┘  └────────────┘  └───┘  └────────────┘  └────────────┘
//          │              │         │         │              │
//          └──────────────┴─────────┴─────────┴──────────────┘
//                                    │
//                                    ▼
//                         ┌─────────────────────┐
//                         │  reorder_types.h    │
//                         │  Common types and   │
//                         │  utilities          │
//                         └─────────────────────┘
//
// Algorithm IDs:
//   0  - ORIGINAL:       Keep original ordering
//   1  - RANDOM:         Random permutation
//   2  - SORT:           Sort by degree
//   3  - HUBSORT:        Sort hubs first
//   4  - HUBCLUSTER:     Cluster hubs
//   5  - DBG:            Degree-based grouping
//   6  - HUBSORTDBG:     HubSort within DBG
//   7  - HUBCLUSTERDBG:  HubCluster within DBG (recommended for power-law)
//   8  - RABBITORDER:    Community detection + aggregation
//   9  - GORDER:         Dynamic programming ordering
//   10 - CORDER:         Cache-aware ordering
//   11 - RCMORDER:       Reverse Cuthill-McKee
//   12 - GRAPHBREWORDER: Leiden + per-community reordering
//   13 - MAP:            Load ordering from file
//   14 - ADAPTIVEORDER:  ML-based algorithm selection
//   15 - LEIDENORDER:    Leiden community ordering
//   16 - LEIDENDENDROGRAM: Leiden with dendrogram traversal
//   17 - LEIDENCSR:      GVE-Leiden CSR-native ordering
//
// Usage:
//   #include "reorder/reorder.h"
//   
//   // In your BuilderBase class or standalone:
//   pvector<NodeID> new_ids(g.num_nodes(), UINT_E_MAX);
//   GenerateMapping(g, new_ids, HubClusterDBG, true, {});
//
// Author: GraphBrew Team
// License: See LICENSE.txt
// ============================================================================

#ifndef REORDER_H_
#define REORDER_H_

// ============================================================================
// INCLUDE ALL ALGORITHM HEADERS
// ============================================================================

#include "reorder_types.h"   // Common types and utilities
#include "reorder_basic.h"   // ORIGINAL, RANDOM, SORT (0-2)
#include "reorder_hub.h"     // HUBSORT, HUBCLUSTER, DBG variants (3-7)
#include "reorder_rabbit.h"  // RABBITORDER (8)
#include "reorder_classic.h" // GORDER, CORDER, RCM (9-11)

// The following are documentation headers for complex algorithms still in builder.h:
#include "reorder_leiden.h"    // LEIDENORDER, LEIDENDENDROGRAM, LEIDENCSR (15-17) docs
#include "reorder_graphbrew.h" // GRAPHBREWORDER, ADAPTIVEORDER (12, 14) docs

// ============================================================================
// ALGORITHM STRING CONVERSION
// ============================================================================

/**
 * @brief Convert ReorderingAlgo enum to human-readable string
 * 
 * Used for logging, debugging, and output formatting.
 * 
 * @param algo The reordering algorithm enum value
 * @return String name of the algorithm
 */
inline const std::string ReorderingAlgoStr(ReorderingAlgo algo) {
    switch (algo) {
        case ORIGINAL:        return "Original";
        case Random:          return "Random";
        case Sort:            return "Sort";
        case HubSort:         return "HubSort";
        case HubCluster:      return "HubCluster";
        case DBG:             return "DBG";
        case HubSortDBG:      return "HubSortDBG";
        case HubClusterDBG:   return "HubClusterDBG";
        case RabbitOrder:     return "RabbitOrder";
        case GOrder:          return "GOrder";
        case COrder:          return "COrder";
        case RCMOrder:        return "RCMOrder";
        case GraphBrewOrder:  return "GraphBrewOrder";
        case MAP:             return "MAP";
        case AdaptiveOrder:   return "AdaptiveOrder";
        case LeidenOrder:     return "LeidenOrder";
        case LeidenDendrogram: return "LeidenDendrogram";
        case LeidenCSR:       return "LeidenCSR";
        default:
            std::cerr << "Unknown Reordering Algorithm: " << static_cast<int>(algo) << std::endl;
            std::abort();
    }
}

/**
 * @brief Convert integer ID to ReorderingAlgo enum
 * 
 * Used when parsing command-line arguments.
 * 
 * @param value Integer algorithm ID (0-17)
 * @return Corresponding ReorderingAlgo enum value
 */
inline ReorderingAlgo getReorderingAlgo(int value) {
    switch (value) {
        case 0:  return ORIGINAL;
        case 1:  return Random;
        case 2:  return Sort;
        case 3:  return HubSort;
        case 4:  return HubCluster;
        case 5:  return DBG;
        case 6:  return HubSortDBG;
        case 7:  return HubClusterDBG;
        case 8:  return RabbitOrder;
        case 9:  return GOrder;
        case 10: return COrder;
        case 11: return RCMOrder;
        case 12: return GraphBrewOrder;
        case 13: return MAP;
        case 14: return AdaptiveOrder;
        case 15: return LeidenOrder;
        case 16: return LeidenDendrogram;
        case 17: return LeidenCSR;
        default:
            std::cerr << "Unknown algorithm ID: " << value << std::endl;
            return ORIGINAL;
    }
}

/**
 * @brief Convert string to ReorderingAlgo enum
 * 
 * Parses command-line argument string (may be just a number).
 * 
 * @param arg String representation (e.g., "7" or "7:option")
 * @return Corresponding ReorderingAlgo enum value
 */
inline ReorderingAlgo getReorderingAlgo(const char* arg) {
    return getReorderingAlgo(std::atoi(arg));
}

// ============================================================================
// ALGORITHM NAME TO ENUM MAPPING (for perceptron weights)
// ============================================================================

/**
 * @brief Map algorithm name strings to enum values
 * 
 * Used when loading perceptron weights from JSON files.
 * Supports both camelCase and UPPERCASE naming conventions.
 */
inline const std::map<std::string, ReorderingAlgo>& getAlgorithmNameMap() {
    static const std::map<std::string, ReorderingAlgo> name_to_algo = {
        // Standard names
        {"ORIGINAL", ORIGINAL},
        {"Original", ORIGINAL},
        {"RANDOM", Random},
        {"Random", Random},
        {"SORT", Sort},
        {"Sort", Sort},
        {"HubSort", HubSort},
        {"HUBSORT", HubSort},
        {"HubCluster", HubCluster},
        {"HUBCLUSTER", HubCluster},
        {"DBG", DBG},
        {"HubSortDBG", HubSortDBG},
        {"HUBSORTDBG", HubSortDBG},
        {"HubClusterDBG", HubClusterDBG},
        {"HUBCLUSTERDBG", HubClusterDBG},
        {"RabbitOrder", RabbitOrder},
        {"RABBITORDER", RabbitOrder},
        {"GOrder", GOrder},
        {"GORDER", GOrder},
        {"COrder", COrder},
        {"CORDER", COrder},
        {"RCMOrder", RCMOrder},
        {"RCMORDER", RCMOrder},
        {"RCM", RCMOrder},
        {"GraphBrewOrder", GraphBrewOrder},
        {"GRAPHBREWORDER", GraphBrewOrder},
        {"MAP", MAP},
        {"AdaptiveOrder", AdaptiveOrder},
        {"ADAPTIVEORDER", AdaptiveOrder},
        {"LeidenOrder", LeidenOrder},
        {"LEIDENORDER", LeidenOrder},
        {"LeidenDendrogram", LeidenDendrogram},
        {"LEIDENDENDROGRAM", LeidenDendrogram},
        {"LeidenCSR", LeidenCSR},
        {"LEIDENCSR", LeidenCSR},
    };
    return name_to_algo;
}

// ============================================================================
// ALGORITHM CATEGORY HELPERS
// ============================================================================

/**
 * @brief Check if algorithm is a basic/simple algorithm
 * 
 * Basic algorithms have O(n) or O(n log n) complexity and
 * don't require community detection.
 */
inline bool isBasicAlgorithm(ReorderingAlgo algo) {
    return algo == ORIGINAL || algo == Random || algo == Sort;
}

/**
 * @brief Check if algorithm is hub-based
 * 
 * Hub-based algorithms focus on high-degree vertices.
 */
inline bool isHubBasedAlgorithm(ReorderingAlgo algo) {
    return algo == HubSort || algo == HubCluster || 
           algo == DBG || algo == HubSortDBG || algo == HubClusterDBG;
}

/**
 * @brief Check if algorithm uses community detection
 * 
 * Community-based algorithms detect graph structure before reordering.
 */
inline bool isCommunityBasedAlgorithm(ReorderingAlgo algo) {
    return algo == RabbitOrder || algo == LeidenOrder || 
           algo == LeidenDendrogram || algo == LeidenCSR ||
           algo == GraphBrewOrder || algo == AdaptiveOrder;
}

/**
 * @brief Check if algorithm is a Leiden variant
 */
inline bool isLeidenAlgorithm(ReorderingAlgo algo) {
    return algo == LeidenOrder || algo == LeidenDendrogram || algo == LeidenCSR;
}

/**
 * @brief Check if algorithm supports options/variants
 * 
 * Some algorithms accept format strings like "8:csr" or "17:gve:1.0"
 */
inline bool hasVariants(ReorderingAlgo algo) {
    return algo == RabbitOrder || algo == LeidenDendrogram || 
           algo == LeidenCSR || algo == GraphBrewOrder || 
           algo == AdaptiveOrder || algo == MAP;
}

// ============================================================================
// DEFAULT PERCEPTRON WEIGHTS
// ============================================================================

/**
 * @brief Get default perceptron weights for AdaptiveOrder
 * 
 * These weights are used when no trained weights are available.
 * They encode domain knowledge about when each algorithm performs well.
 * 
 * Positive weight = algorithm prefers higher values of that feature
 * Negative weight = algorithm prefers lower values
 * 
 * NOTE: The comprehensive weights with all algorithms are in BuilderBase::GetPerceptronWeights().
 * This function provides a minimal subset for quick reference.
 * 
 * @return Map of algorithm enum to default PerceptronWeights
 */
inline const std::map<ReorderingAlgo, PerceptronWeights>& getDefaultPerceptronWeights() {
    static const std::map<ReorderingAlgo, PerceptronWeights> default_weights = {
        // ORIGINAL: Baseline, no strong preferences
        {ORIGINAL, {
            .bias = 0.0,
            .w_modularity = 0.0,
            .w_log_nodes = 0.0,
            .w_log_edges = 0.0,
            .w_density = 0.0,
            .w_avg_degree = 0.0,
            .w_degree_variance = 0.0,
            .w_hub_concentration = 0.0,
            .w_clustering_coeff = 0.0,
            .w_avg_path_length = 0.0,
            .w_diameter = 0.0,
            .w_community_count = 0.0,
            .cache_l1_impact = 0.0,
            .cache_l2_impact = 0.0,
            .cache_l3_impact = 0.0,
            .cache_dram_penalty = 0.0,
            .w_reorder_time = 0.0
        }},
        
        // HUBCLUSTERDBG: Good for power-law graphs
        {HubClusterDBG, {
            .bias = 0.7,
            .w_modularity = -0.1,
            .w_log_nodes = 0.05,
            .w_log_edges = 0.05,
            .w_density = -0.2,
            .w_avg_degree = 0.0,
            .w_degree_variance = 0.3,
            .w_hub_concentration = 0.4,
            .w_clustering_coeff = 0.0,
            .w_avg_path_length = 0.0,
            .w_diameter = 0.0,
            .w_community_count = 0.0,
            .cache_l1_impact = 0.0,
            .cache_l2_impact = 0.0,
            .cache_l3_impact = 0.0,
            .cache_dram_penalty = 0.0,
            .w_reorder_time = 0.1
        }},
        
        // RABBITORDER: Good for community-structured graphs
        {RabbitOrder, {
            .bias = 0.6,
            .w_modularity = 0.25,
            .w_log_nodes = 0.08,
            .w_log_edges = 0.08,
            .w_density = -0.15,
            .w_avg_degree = 0.05,
            .w_degree_variance = 0.1,
            .w_hub_concentration = 0.1,
            .w_clustering_coeff = 0.0,
            .w_avg_path_length = 0.0,
            .w_diameter = 0.0,
            .w_community_count = 0.0,
            .cache_l1_impact = 0.0,
            .cache_l2_impact = 0.0,
            .cache_l3_impact = 0.0,
            .cache_dram_penalty = 0.0,
            .w_reorder_time = -0.1
        }},
        
        // LEIDENCSR: Best quality for community graphs
        {LeidenCSR, {
            .bias = 0.65,
            .w_modularity = 0.3,
            .w_log_nodes = 0.1,
            .w_log_edges = 0.1,
            .w_density = -0.1,
            .w_avg_degree = 0.05,
            .w_degree_variance = 0.15,
            .w_hub_concentration = 0.1,
            .w_clustering_coeff = 0.0,
            .w_avg_path_length = 0.0,
            .w_diameter = 0.0,
            .w_community_count = 0.0,
            .cache_l1_impact = 0.0,
            .cache_l2_impact = 0.0,
            .cache_l3_impact = 0.0,
            .cache_dram_penalty = 0.0,
            .w_reorder_time = -0.2
        }},
        
        // GRAPHBREWORDER: Leiden + per-community ordering
        {GraphBrewOrder, {
            .bias = 0.6,
            .w_modularity = 0.25,
            .w_log_nodes = 0.1,
            .w_log_edges = 0.08,
            .w_density = -0.3,
            .w_avg_degree = 0.0,
            .w_degree_variance = 0.2,
            .w_hub_concentration = 0.1,
            .w_clustering_coeff = 0.0,
            .w_avg_path_length = 0.0,
            .w_diameter = 0.0,
            .w_community_count = 0.0,
            .cache_l1_impact = 0.0,
            .cache_l2_impact = 0.0,
            .cache_l3_impact = 0.0,
            .cache_dram_penalty = 0.0,
            .w_reorder_time = -0.5
        }},
    };
    return default_weights;
}

#endif  // REORDER_H_
