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
// │basic.h     │  │hub.h       │  │   │  │leiden.h    │  │adaptive.h  │
// │(0,1,2)     │  │(3,4,5,6,7) │  │   │  │(15,16,17)  │  │(14)        │
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
// Note: LeidenCSR (16) has been deprecated — GraphBrew (12) subsumes it.
// LeidenOrder (15) uses external/leiden/leiden.hxx directly.

// Note: reorder_adaptive.h is included at the END of this file
// after all dispatcher functions are defined.
// GraphBrewOrder (12) implementation is now fully in builder.h via GraphBrew pipeline.

// ============================================================================
// ALGORITHM STRING CONVERSION
// ============================================================================
// NOTE: ReorderingAlgoStr and getReorderingAlgo are now defined in reorder_types.h
// to allow use by reorder_adaptive.h before this file.

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
           algo == GraphBrewOrder || algo == AdaptiveOrder;
}

/**
 * @brief Check if algorithm is a Leiden variant
 */
inline bool isLeidenAlgorithm(ReorderingAlgo algo) {
    return algo == LeidenOrder;
}

/**
 * @brief Check if algorithm supports options/variants
 * 
 * Some algorithms accept format strings like "8:csr" or "17:gve:1.0"
 */
inline bool hasVariants(ReorderingAlgo algo) {
    return algo == RabbitOrder || 
           algo == GraphBrewOrder || 
           algo == AdaptiveOrder || algo == MAP;
}

// ============================================================================
// DEFAULT PERCEPTRON WEIGHTS
// ============================================================================

/**
 * @brief Get default perceptron weights for AdaptiveOrder
 * 
 * Delegates to the comprehensive GetPerceptronWeights() function in reorder_types.h
 * which contains weights for all algorithms.
 * 
 * Positive weight = algorithm prefers higher values of that feature
 * Negative weight = algorithm prefers lower values
 * 
 * @return Map of algorithm enum to default PerceptronWeights
 */
inline const std::map<ReorderingAlgo, PerceptronWeights>& getDefaultPerceptronWeights() {
    return GetPerceptronWeights();
}


// ============================================================================
// BASIC REORDERING DISPATCHER
// ============================================================================

/**
 * @brief Apply a basic reordering algorithm to a graph (standalone version)
 * 
 * This dispatcher calls the appropriate reordering function based on the 
 * algorithm ID. Used by GraphBrew, Adaptive, and other multi-level algorithms
 * that need to apply per-community reordering.
 * 
 * Supports algorithms: ORIGINAL, Random, Sort, HubSort, HubCluster, DBG,
 * HubSortDBG, HubClusterDBG, GOrder, COrder, RCMOrder
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam WeightT_ Edge weight type (for GOrder/RCMOrder)
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph
 * @param new_ids Output permutation
 * @param algo Algorithm to apply
 * @param useOutdeg Use out-degree (true) or in-degree (false)
 * @param filename Optional filename for GOrder/RCMOrder
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
inline void ApplyBasicReorderingStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    ReorderingAlgo algo,
    bool useOutdeg,
    const std::string& filename = "") {
    
    switch (algo) {
        case HubSort:
            ::GenerateHubSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case HubCluster:
            ::GenerateHubClusterMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case DBG:
            ::GenerateDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case HubSortDBG:
            ::GenerateHubSortDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case HubClusterDBG:
            ::GenerateHubClusterDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case Sort:
            ::GenerateSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg, false);
            break;
        case Random:
            ::GenerateRandomMapping<NodeID_, DestID_, invert>(g, new_ids);
            break;
        case GOrder:
            ::GenerateGOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, filename);
            break;
        case COrder:
            ::GenerateCOrderMapping<NodeID_, DestID_, invert>(g, new_ids);
            break;
        case RCMOrder:
            ::GenerateRCMOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, filename);
            break;
        case ORIGINAL:
        default:
            ::GenerateOriginalMapping<NodeID_, DestID_, invert>(g, new_ids);
            break;
    }
}

// ============================================================================
// COMMUNITY SUBGRAPH REORDERING (Standalone)
// ============================================================================

/**
 * @brief Reorder a community subgraph and assign global IDs (standalone version)
 * 
 * This function:
 * 1. Extracts induced subgraph for the community nodes
 * 2. Applies the specified reordering algorithm
 * 3. Maps local IDs back to global IDs
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type  
 * @tparam WeightT_ Edge weight type
 * @tparam invert Whether graph has inverse edges
 * @param g The full graph
 * @param nodes Community node IDs
 * @param node_set Set for O(1) membership test
 * @param algo Algorithm to apply
 * @param useOutdeg Use out-degree (true) or in-degree (false)
 * @param new_ids Output global mapping (modified in place)
 * @param current_id Starting global ID (updated after assignment)
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void ReorderCommunitySubgraphStandalone(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    const std::vector<NodeID_>& nodes,
    const std::unordered_set<NodeID_>& node_set,
    ReorderingAlgo algo,
    bool useOutdeg,
    pvector<NodeID_>& new_ids,
    NodeID_& current_id)
{
    const size_t comm_size = nodes.size();
    if (comm_size == 0) return;
    
    // Build global-to-local / local-to-global mappings
    std::unordered_map<NodeID_, NodeID_> global_to_local;
    std::vector<NodeID_> local_to_global(comm_size);
    for (size_t i = 0; i < comm_size; ++i) {
        global_to_local[nodes[i]] = static_cast<NodeID_>(i);
        local_to_global[i] = nodes[i];
    }
    
    // Create edge list for induced subgraph
    std::vector<std::pair<NodeID_, DestID_>> sub_edges;
    for (NodeID_ node : nodes) {
        NodeID_ local_src = global_to_local[node];
        for (DestID_ neighbor : g.out_neigh(node)) {
            NodeID_ dest = static_cast<NodeID_>(neighbor);
            if (node_set.count(dest)) {
                NodeID_ local_dst = global_to_local[dest];
                sub_edges.push_back({local_src, static_cast<DestID_>(local_dst)});
            }
        }
    }
    
    // Build CSR graph from edge list and apply reordering
    // GUARD: If subgraph has no internal edges (all edges go to other communities),
    // skip reordering and just assign nodes in original order. This can happen on
    // graphs with extreme structure like Kronecker graphs where small communities
    // may have nodes only connected to other communities.
    if (sub_edges.empty()) {
        // No internal edges - assign nodes in original order
        for (NodeID_ node : nodes) {
            new_ids[node] = current_id++;
        }
        return;
    }
    
    CSRGraph<NodeID_, DestID_, invert> sub_g = 
        MakeLocalGraphFromELStandalone<NodeID_, DestID_, invert>(sub_edges, false);
    pvector<NodeID_> sub_new_ids(comm_size, -1);
    ApplyBasicReorderingStandalone<NodeID_, DestID_, WeightT_, invert>(
        sub_g, sub_new_ids, algo, useOutdeg);
    
    // Map local reordered IDs back to global IDs
    std::vector<NodeID_> reordered_nodes(comm_size);
    for (size_t i = 0; i < comm_size; ++i) {
        if (sub_new_ids[i] >= 0 && sub_new_ids[i] < static_cast<NodeID_>(comm_size)) {
            reordered_nodes[sub_new_ids[i]] = local_to_global[i];
        } else {
            reordered_nodes[i] = local_to_global[i];
        }
    }
    
    // Assign global IDs
    for (NodeID_ node : reordered_nodes) {
        new_ids[node] = current_id++;
    }
}

// ============================================================================
// EDGELIST-BASED REORDERING DISPATCHER (Standalone)
// ============================================================================

/**
 * @brief Apply reordering algorithm to an edge list (standalone version)
 * 
 * This dispatcher creates a local graph from the edge list and applies
 * the specified reordering algorithm. Used by GraphBrew and Adaptive
 * for per-community reordering.
 * 
 * Handles both basic algorithms (0-11) and RabbitOrder (8) specially.
 * For RabbitOrder, supports "csr" (default) and "boost" variants.
 * 
 * CRITICAL: For small subgraphs (<100K edges), disables nested parallelism
 * to avoid thread explosion overhead.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam WeightT_ Edge weight type
 * @tparam invert Whether graph has inverse edges
 * @param el Edge list to reorder
 * @param new_ids Output permutation
 * @param algo Algorithm to apply
 * @param useOutdeg Use out-degree (true) or in-degree (false)
 * @param reordering_options Algorithm-specific options
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateMappingLocalEdgelistStandalone(
    EdgeList<NodeID_, DestID_>& el,
    pvector<NodeID_>& new_ids,
    ReorderingAlgo algo,
    bool useOutdeg,
    const std::vector<std::string>& reordering_options = {})
{
    // CRITICAL: Disable nested parallelism for small subgraphs
    const size_t MIN_EDGES_FOR_PARALLEL = 100000;
    const bool is_small_subgraph = el.size() < MIN_EDGES_FOR_PARALLEL;
    
    // Save current settings
    int prev_nested = omp_get_nested();
    int prev_max_levels = omp_get_max_active_levels();
    int prev_num_threads = omp_get_max_threads();
    
    if (is_small_subgraph) {
        omp_set_nested(0);
        omp_set_max_active_levels(1);
        omp_set_num_threads(1);
    } else {
        omp_set_nested(1);
        omp_set_max_active_levels(2);
    }
    
    // Convert EdgeList (pvector<Edge>) to vector<pair> for standalone functions
    std::vector<std::pair<NodeID_, DestID_>> edges_vec(el.size());
    #pragma omp parallel for
    for (size_t i = 0; i < el.size(); ++i) {
        edges_vec[i] = {el[i].u, el[i].v};
    }
    
    // Build CSR graph from edge list
    CSRGraph<NodeID_, DestID_, invert> g = 
        MakeLocalGraphFromELStandalone<NodeID_, DestID_, invert>(edges_vec);
    
    // Dispatch to appropriate algorithm
    switch (algo) {
        case HubSort:
            ::GenerateHubSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case HubCluster:
            ::GenerateHubClusterMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case DBG:
            ::GenerateDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case HubSortDBG:
            ::GenerateHubSortDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case HubClusterDBG:
            ::GenerateHubClusterDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
            break;
        case Sort:
            ::GenerateSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg, false);
            break;
        case Random:
            ::GenerateRandomMapping<NodeID_, DestID_, invert>(g, new_ids);
            break;
        case RabbitOrder:
        {
            // RabbitOrder with variants: csr (default), boost
            std::string variant = "csr";
            if (!reordering_options.empty() && !reordering_options[0].empty()) {
                variant = reordering_options[0];
            }
            
            if (variant == "boost") {
                // Boost-based RabbitOrder needs preprocessing
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                ::GenerateSortMappingRabbit<NodeID_, DestID_, invert>(g, new_ids_local, true, true);
                auto g_trans = RelabelByMappingStandalone<NodeID_, DestID_, invert>(g, new_ids_local);
                ::GenerateRabbitOrderMapping<NodeID_, DestID_, WeightT_, invert>(g_trans, new_ids_local_2);
                
                // Combine mappings
                if (is_small_subgraph) {
                    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
                        if (new_ids_local[n] != static_cast<NodeID_>(-1) && 
                            new_ids_local[n] < g.num_nodes()) {
                            new_ids[n] = new_ids_local_2[new_ids_local[n]];
                        }
                    }
                } else {
                    #pragma omp parallel for
                    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
                        if (new_ids_local[n] != static_cast<NodeID_>(-1) && 
                            new_ids_local[n] < g.num_nodes()) {
                            new_ids[n] = new_ids_local_2[new_ids_local[n]];
                        }
                    }
                }
            } else {
                // Native CSR implementation with degree preprocessing
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                ::GenerateSortMappingRabbit<NodeID_, DestID_, invert>(g, new_ids_local, true, true);
                auto g_trans = RelabelByMappingStandalone<NodeID_, DestID_, invert>(g, new_ids_local);
                ::GenerateRabbitOrderCSRMapping<NodeID_, DestID_, WeightT_, invert>(g_trans, new_ids_local_2);
                
                // Combine mappings
                if (is_small_subgraph) {
                    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
                        new_ids[n] = new_ids_local_2[new_ids_local[n]];
                    }
                } else {
                    #pragma omp parallel for
                    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
                        new_ids[n] = new_ids_local_2[new_ids_local[n]];
                    }
                }
            }
        }
        break;
        case GOrder:
            ::GenerateGOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, "");
            break;
        case COrder:
            ::GenerateCOrderMapping<NodeID_, DestID_, invert>(g, new_ids);
            break;
        case RCMOrder:
            ::GenerateRCMOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, "");
            break;
        case ORIGINAL:
        default:
            ::GenerateOriginalMapping<NodeID_, DestID_, invert>(g, new_ids);
            break;
    }
    
    // Restore OpenMP settings
    omp_set_nested(prev_nested);
    omp_set_max_active_levels(prev_max_levels);
    omp_set_num_threads(prev_num_threads);
}

// ============================================================================
// INCLUDE ADAPTIVE HEADER
// ============================================================================
// Included last because they depend on functions defined above
// (GenerateMappingLocalEdgelistStandalone, ReorderCommunitySubgraphStandalone)

#include "reorder_graphbrew.h" // GRAPHBREWORDER (12) - GraphBrew unified reordering framework
#include "reorder_adaptive.h"  // ADAPTIVEORDER (14) config (implementations in builder.h)

#endif  // REORDER_H_
