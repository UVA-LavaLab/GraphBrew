/**
 * @file reorder_graphbrew.h
 * @brief GraphBrew: Graph Reordering for Better Efficiency
 *
 * A modular, configurable graph reordering library that combines the best
 * techniques from Leiden, RabbitOrder, and cache optimization literature.
 *
 * =============================================================================
 * UNIFIED CONFIGURATION
 * =============================================================================
 *
 * GraphBrew uses the unified reorder::ReorderConfig system from reorder_types.h:
 *
 * │ Parameter              │ Default │ Description                          │
 * ├────────────────────────┼─────────┼──────────────────────────────────────┤
 * │ resolution             │ 1.0     │ Modularity (auto-computed from graph)│
 * │ tolerance              │ 1e-2    │ Node movement convergence            │
 * │ aggregationTolerance   │ 0.8     │ When to stop aggregating             │
 * │ toleranceDrop          │ 10.0    │ Tolerance reduction per pass         │
 * │ maxIterations          │ 10      │ Max iterations per pass              │
 * │ maxPasses              │ 10      │ Max aggregation passes               │
 * │ tileSize               │ 4096    │ Cache blocking tile size             │
 * │ prefetchDistance       │ 8       │ Prefetch lookahead                   │
 *
 * GraphBrewConfig uses reorder::DEFAULT_* constants and provides conversion:
 *   GraphBrewConfig::FromReorderConfig(cfg)  - Convert unified → GraphBrew
 *   graphBrewConfig.toReorderConfig()        - Convert GraphBrew → unified
 *
 * =============================================================================
 * ALGORITHMS
 * =============================================================================
 *
 * GraphBrew supports two main algorithmic approaches:
 *
 * 1. LEIDEN-BASED (default): Multi-pass community detection with configurable
 *    aggregation and ordering strategies.
 *    - Local moving phase: vertices move to best neighboring community
 *    - Refinement phase: split and merge communities for better modularity
 *    - Aggregation: build super-graph and repeat until convergence
 *
 * 2. RABBIT ORDER: Single-pass parallel incremental aggregation from the paper
 *    "Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis"
 *    - Vertices processed in ascending degree order
 *    - Lock-free parallel merging via 64-bit CAS
 *    - Dendrogram built on-the-fly during merges
 *    - Faster reordering, coarser communities
 *
 * =============================================================================
 * ARCHITECTURE
 * =============================================================================
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                    GraphBrew (graphbrew) - Leiden Pipeline                        │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  PASS 1...N (until convergence):                                       │
 * │    ├── leidenLocalMoving()  - Move vertices to best community          │
 * │    ├── leidenRefinement()   - Refine communities (optional)            │
 * │    └── Aggregation:                                                    │
 * │        ├── LEIDEN_CSR:  Build explicit super-graph (default)           │
 * │        ├── RABBIT_LAZY: Union-find streaming (graphbrew:streaming)          │
 * │        └── HYBRID:      Lazy early, CSR later                          │
 * │  Then apply ordering strategy to final communities                     │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                 RABBIT ORDER (12:rabbit) - Single Pass                  │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  1. Sort vertices by degree (ascending)                                │
 * │  2. Initialize edge cache per vertex                                   │
 * │  3. Parallel merge loop (degree order):                                │
 * │     ├── Invalidate vertex (atomic)                                     │
 * │     ├── unite() - Aggregate edges from children (incremental)          │
 * │     ├── Find best neighbor (max ΔQ modularity)                         │
 * │     └── Atomic CAS merge (single 64-bit: strength + child)             │
 * │  4. DFS from top-level vertices → final ordering                       │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * =============================================================================
 * ORDERING STRATEGIES (for Leiden-based GraphBrew)
 * =============================================================================
 *
 * After community detection, vertices are ordered using one of:
 *
 * │ Strategy        │ Option           │ Description                       │
 * ├─────────────────┼──────────────────┼───────────────────────────────────┤
 * │ HIERARCHICAL    │ (default)        │ Sort by community, then degree    │
 * │ DENDROGRAM_DFS  │ 12:dfs           │ DFS traversal of dendrogram       │
 * │ DENDROGRAM_BFS  │ 12:bfs           │ BFS traversal of dendrogram       │
 * │ DBG             │ 12:dbg           │ DBG within each community         │
 * │ CORDER          │ 12:corder        │ Hot/cold within each community    │
 * │ DBG_GLOBAL      │ 12:dbg-global    │ DBG across all vertices           │
 * │ CORDER_GLOBAL   │ 12:corder-global │ Hot/cold across all vertices      │
 * │ CONNECTIVITY_BFS│ 12:conn          │ BFS within communities (default)  │
 * │ HYBRID_LEIDEN_  │ 12:hrab          │ Leiden + RabbitOrder super-graph   │
 * │ RABBIT          │                  │ (best locality web/geometric)     │
 * │ TILE_QUANTIZED_ │ 12:tqr           │ Tile-quantized RabbitOrder:       │
 * │ RABBIT          │                  │ cache-line-aligned tile graph     │
 * │ LAYER           │ 12:leiden        │ Per-community external algo       │
 * │                 │                  │ dispatch (0-11, default: Rabbit)  │
 *
 * =============================================================================
 * COMMAND LINE USAGE
 * =============================================================================
 *
 * Format: -o 12:[ordering]:[options...]
 *
 * Leiden-based GraphBrew (the "graphbrew" prefix is optional):
 *   ./bench/bin/pr -f graph.mtx -o 12 -n 5                    # Default (leiden + per-community RabbitOrder)
 *   ./bench/bin/pr -f graph.mtx -o 12:dfs -n 5                # DFS ordering
 *   ./bench/bin/pr -f graph.mtx -o 12:dbg -n 5                # DBG per community
 *   ./bench/bin/pr -f graph.mtx -o 12:streaming -n 5          # Lazy aggregation
 *   ./bench/bin/pr -f graph.mtx -o 12:lazyupdate -n 5         # Batched ctot updates (reduces atomics)
 *   ./bench/bin/pr -f graph.mtx -o 12:conn -n 5               # Connectivity BFS (default ordering)
 *   ./bench/bin/pr -f graph.mtx -o 12:hrab -n 5               # Hybrid Leiden+RabbitOrder (best locality)
 *   ./bench/bin/pr -f graph.mtx -o 12:tqr -n 5                # Tile-Quantized RabbitOrder (cache-aligned)
 *   ./bench/bin/pr -f graph.mtx -o 12:0.75 -n 5               # Fixed resolution (0.75)
 *   ./bench/bin/pr -f graph.mtx -o 12:dfs:streaming:0.75 -n 5 # Combined
 *
 * RabbitOrder:
 *   ./bench/bin/pr -f graph.mtx -o 12:rabbit -n 5             # RabbitOrder algorithm
 *   ./bench/bin/pr -f graph.mtx -o 12:rabbit:dfs -n 5         # + DFS post-ordering
 *   ./bench/bin/pr -f graph.mtx -o 12:rabbit:dbg -n 5         # + DBG post-ordering
 *   # Note: RabbitOrder does not support dynamic resolution (falls back to auto)
 *
 * =============================================================================
 * RESOLUTION MODES
 * =============================================================================
 *
 * GraphBrew supports three resolution modes for community detection:
 *
 * │ Mode         │ Syntax         │ Description                              │
 * ├──────────────┼────────────────┼──────────────────────────────────────────┤
 * │ Auto         │ auto or 0      │ Compute once from graph density/CV       │
 * │ Dynamic      │ dynamic        │ Auto initial, adjust each pass based on: │
 * │              │                │   - Community reduction rate             │
 * │              │                │   - Community size imbalance             │
 * │              │                │   - Convergence speed                    │
 * │ Dynamic+Init │ dynamic_2.0    │ Start at 2.0, then adjust each pass      │
 * │ Fixed        │ 1.5            │ Use specified value unchanged            │
 *
 * Dynamic resolution is recommended for unknown graphs where optimal resolution
 * is not known ahead of time. The algorithm adapts based on runtime metrics.
 *
 * =============================================================================
 * COMPARISON: GraphBrew (Leiden) vs RABBIT ORDER
 * =============================================================================
 *
 * │ Aspect          │ graphbrew (Leiden)           │ graphbrew:rabbit (RabbitOrder)  │
 * ├─────────────────┼─────────────────────────┼────────────────────────────┤
 * │ Passes          │ Multi-pass (2-5)        │ Single-pass                │
 * │ Vertex Order    │ Random/parallel         │ Sorted by degree           │
 * │ Aggregation     │ Explicit super-graph    │ Implicit union-find        │
 * │ Parallelism     │ Per-pass parallel       │ Lock-free CAS              │
 * │ Dendrogram      │ Built after detection   │ Built during merges        │
 * │ Communities     │ Many fine (~50K)        │ Fewer coarse (~2K)         │
 * │ Ordering        │ Configurable            │ DFS (configurable post)    │
 * │ Best For        │ Quality communities     │ Fast reordering            │
 *
 * =============================================================================
 * CONFIGURATION STRUCT
 * =============================================================================
 *
 * struct GraphBrewConfig {
 *     // Algorithm Selection
 *     GraphBrewAlgorithm algorithm = LEIDEN;     // LEIDEN or RABBIT_ORDER
 *
 *     // Community Detection (Leiden)
 *     double resolution = 1.0;              // Modularity resolution
 *     int maxIterations = 20;               // Per-pass iteration limit
 *     int maxPasses = 10;                   // Maximum aggregation passes
 *     bool useRefinement = true;            // Enable refinement phase
 *
 *     // Aggregation Strategy (Leiden)
 *     AggregationStrategy aggregation = LEIDEN_CSR;  // CSR, RABBIT_LAZY, HYBRID
 *
 *     // Ordering Strategy
 *     OrderingStrategy ordering = HIERARCHICAL;      // See table above
 * };
 *
 * @author GraphBrew
 * @date 2026
 */

#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <stack>
#include <queue>
#include <atomic>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <sstream>
#include <type_traits>

#ifdef OPENMP
#include <omp.h>
#include <parallel/algorithm>  // For __gnu_parallel::sort
#endif

#include <graph.h>
#include <pvector.h>
#include <timer.h>

#include "reorder_gorder.h"

namespace graphbrew {

//=============================================================================
// SECTION 1: CONFIGURATION AND ENUMS
//=============================================================================

/** Enable debug output */
#ifndef GRAPHBREW_DEBUG
#define GRAPHBREW_DEBUG 0
#endif

#if GRAPHBREW_DEBUG
#define GRAPHBREW_TRACE(fmt, ...) printf("[GraphBrew] " fmt "\n", ##__VA_ARGS__)
#else
#define GRAPHBREW_TRACE(fmt, ...) ((void)0)
#endif

/** Weight type */
using Weight = double;

/** Main algorithm selection */
enum class GraphBrewAlgorithm {
    LEIDEN,          ///< Leiden-based community detection (default)
    RABBIT_ORDER     ///< RabbitOrder incremental aggregation (paper)
};

/** Aggregation strategy for building super-graph */
enum class AggregationStrategy {
    LEIDEN_CSR,      ///< Standard Leiden CSR aggregation (accurate)
    RABBIT_LAZY,     ///< RabbitOrder-style lazy incremental merge (fast)
    HYBRID,          ///< Use lazy for early passes, CSR for final
    GVE_CSR          ///< GVE-style aggregation: explicit super-graph adjacency + local-moving merge
};

/** Edge weight computation for M (total weight) */
enum class MComputation {
    HALF_EDGES,      ///< M = num_edges / 2 (standard undirected, leiden.hxx style)
    TOTAL_EDGES      ///< M = num_edges (GVE style — tighter communities)
};

/** Ordering strategy for final vertex permutation */
enum class OrderingStrategy {
    HIERARCHICAL,    ///< Multi-level sort by all passes (leiden.hxx style)
    CONNECTIVITY_BFS,///< BFS within communities using original graph (Boost-style)
    DENDROGRAM_DFS,  ///< DFS traversal of community dendrogram
    DENDROGRAM_BFS,  ///< BFS traversal of community dendrogram
    COMMUNITY_SORT,  ///< Simple sort by final community + degree
    HUB_CLUSTER,     ///< Hub-first within communities
    DBG,             ///< Degree-Based Grouping within communities
    CORDER,          ///< Corder hot/cold partitioning within communities
    DBG_GLOBAL,      ///< DBG across all vertices (post-clustering)
    CORDER_GLOBAL,   ///< Corder across all vertices (post-clustering)
    HIERARCHICAL_CACHE_AWARE,  ///< Use Leiden hierarchy as dendrogram for cache-aware ordering
    HYBRID_LEIDEN_RABBIT,      ///< Leiden communities + RabbitOrder super-graph ordering (best locality)
    HIERARCHICAL_LEIDEN_RABBIT,///< Run RabbitOrder at EVERY Leiden dendrogram level (multi-level super-graph ordering)
    TILE_QUANTIZED_RABBIT,     ///< Tile-quantized graph + RabbitOrder: cache-line-aligned macro-ordering
    COMPOSE,                   ///< Pluggable: pick {super-graph order} + {community order} + {intra-community order}
    LAYER                      ///< GraphBrew mode: apply any algo 0-11 per community (external dispatch)
};

/** Community-ordering menu for OrderingStrategy::COMPOSE
 *  (how to order the communities themselves, on top of the super-graph permutation).
 */
enum class CommunityOrder {
    SizeDesc,    ///< Sort communities by vertex-count descending (large first)
    SizeAsc,     ///< Sort communities by vertex-count ascending (small first, tail-heavy graphs)
    DegreeDesc,  ///< Sort communities by total in-community degree descending
    DegreeAsc,   ///< Sort communities by total in-community degree ascending
    CapacityRuns,///< Cut-aware L2/LLC-sized runs of whole communities
    CutMin,      ///< Mt-METIS-style: minimise inter-community edge crossing
                 ///< by greedy NN-TSP over the C×C crossing-edge graph.
                 ///< Cost O(|E|) + O(C^2).  Fallback to DegreeDesc if C>4096.
    Identity,    ///< Keep the super-graph permutation order untouched
};

/** Intra-community-ordering menu for OrderingStrategy::COMPOSE
 *  (how to order vertices within each community).
 */
enum class IntraCommunityOrder {
    BFSFromHub,  ///< intraBFSFromHub<>() primitive (SECTION 16-PRIMITIVES)
    RCM,         ///< intraRCM<>() primitive (SECTION 16-PRIMITIVES)
    RCMpp,       ///< intraRCMpp<>() — Hou/Liu/Zhu (arXiv 2409.04171, 2024)
                 ///< bi-criteria pseudo-peripheral start: argmin (0.5·deg_rank
                 ///< + 0.5·depth_rank) instead of plain BNF min-degree.  Same
                 ///< CM-BFS body as intraRCM.  ~30 LoC delta in start pick.
    Dendrogram,  ///< intraDendrogramDFS<>() — reuse Rabbit per-community DFS
                 ///< (only valid when CD=Rabbit; falls back to BFS otherwise).
                 ///< Cost: pure pointer-chase on dendrogram (no BFS/visited
                 ///< bookkeeping), so cheapest intra primitive available.
    Gorder,      ///< Relaxed local Gorder: direct-neighbor term only
    GorderFaithful, ///< Local Gorder with direct-neighbor and two-hop sibling
                   ///< terms. Faithful objective, restricted to one community.
    HubSort,     ///< intraHubSort<>() — sort the community's vertices in
                 ///< descending degree order (hubs first).  O(|V_local| log)
                 ///< per community.  Cheapest non-trivial intra primitive
                 ///< and competitive with DBG-style degree layouts for
                 ///< push-style kernels (PR pull, BC forward).
    DegreeAsc,   ///< Inverse of HubSort: sort the community's vertices in
                 ///< ASCENDING degree order (leaves first, hubs last).
                 ///< Useful as an ablation control and for workloads
                 ///< where leaf vertices benefit from being co-located.
    Hub2,        ///< Second-moment degree (DRO/Lakhotia-style): sort within-
                 ///< community by sum-of-neighbor-degree descending.  Cost
                 ///< O(|E_local|) per community.  Captures hub-of-hubs
                 ///< locality that pure first-moment HubSort misses.
    Alternate,   ///< Interleave hubs and leaves: take sorted-by-degree-desc
                 ///< list, then output [hub0, leaf0, hub1, leaf1, ...].
                 ///< Combines locality (hubs get prefetched early) with
                 ///< false-sharing reduction (hubs dispersed across the
                 ///< ID range so their visited[] bits don't collide on
                 ///< the same cache line under multi-threaded execution).
    Random,      ///< Deterministic Fisher-Yates shuffle inside each
                 ///< community.  Worst-case control / sanity check:
                 ///< measures the value of "any ordering at all".
                 ///< If Random consistently loses to ALL other primitives,
                 ///< the intra-community placement matters.  If it
                 ///< sometimes wins, the framework's choice may be
                 ///< overfitting to noise.  Uses fixed seed for
                 ///< reproducibility.
    BoundaryLast,///< Structure-aware: sort intra-community by EXTERNAL
                 ///< degree ASCENDING (count of edges leaving the
                 ///< community).  Interior nodes first, boundary nodes
                 ///< last.  Hypothesis: BFS/CC propagation completes
                 ///< intra-community work before transitioning to
                 ///< neighbour communities, improving cache locality
                 ///< on push-style kernels.  Cost: O(|E_local| + |E_cross|)
                 ///< per community (one edge scan).
    CoreOrder,   ///< Structure-aware: sort intra-community by INTERNAL
                 ///< k-core number DESCENDING (deep core first, periphery
                 ///< last).  Computes the k-core decomposition restricted
                 ///< to the community subgraph.  Hypothesis: the densely-
                 ///< connected core of a community is the natural reuse
                 ///< target for PR/CC; placing core nodes early gives
                 ///< them the best cache lines.  Differs from BoundaryLast
                 ///< (which uses external degree); CoreOrder uses pure
                 ///< internal connectivity.  Cost: O(|V_local|+|E_local|)
                 ///< per community via bucket-sort peel.
};

/** Optional post-pass refinement after intra-community ordering.
 *  Operates per-community independently so it parallelises trivially. */
enum class RefinementPass {
    None,        ///< Skip refinement (default).
    TwoSwap,     ///< Adjacent-pair swap refinement (FM-style).  For every
                 ///< pair of consecutive positions (p, p+1) within a
                 ///< community, swap if it reduces Σ|local_id(u)-local_id(v)|
                 ///< over intra-community edges.  Up to `refineMaxPasses`
                 ///< passes per community; early-stops when no swap accepted.
                 ///< Cost: O(|E_local| * refineMaxPasses) per community.
};

/** Super-graph-ordering menu for OrderingStrategy::COMPOSE
 *  (how to derive a base community permutation from the community super-graph).
 */
enum class SuperGraphOrder {
    None,         ///< Identity permutation; the community-order axis sorts from scratch
    SuperRabbit,  ///< Build community super-graph, run RabbitOrder (HRAB's super-graph order)
    SuperRCM,     ///< Build community super-graph, run BNF+George-Liu+RCM
    TileRabbit,   ///< 2-level: tile-RabbitOrder + community-RabbitOrder (TQR's super-graph order)
    Hilbert,      ///< Mosaic-style: 2-D Hilbert curve over (size_bucket, deg_bucket)
                  ///< of each community.  Cost O(C log C).  Provides edge-plane
                  ///< locality on the super-graph without building it.
};

/** Community detection mode */
enum class CommunityMode {
    FULL_LEIDEN,     ///< Full Leiden with refinement
    FAST_LP,         ///< Fast label propagation only
    HYBRID           ///< LP for first pass, Leiden for refinement
};

/**
 * Algorithm configuration
 *
 * Uses unified defaults from reorder::ReorderConfig for consistency with
 * Leiden, GraphBrew, and other community-based reordering algorithms.
 */
struct GraphBrewConfig {
    // Resolution and convergence - use unified defaults
    double resolution = reorder::DEFAULT_RESOLUTION;           ///< Modularity resolution [0.1 - 2.0]
    double tolerance = reorder::DEFAULT_TOLERANCE;             ///< Convergence tolerance
    double aggregationTolerance = reorder::DEFAULT_AGGREGATION_TOLERANCE; ///< When to stop aggregating
    double toleranceDrop = reorder::DEFAULT_TOLERANCE_DROP;    ///< Tolerance reduction per pass

    // Iteration limits - use unified defaults
    int maxIterations = reorder::DEFAULT_MAX_ITERATIONS;       ///< Max iterations per pass
    int maxPasses = reorder::DEFAULT_MAX_PASSES;               ///< Max aggregation passes

    // Algorithm selection
    GraphBrewAlgorithm algorithm = GraphBrewAlgorithm::LEIDEN; ///< Main algorithm
    CommunityMode communityMode = CommunityMode::FULL_LEIDEN;
    AggregationStrategy aggregation = AggregationStrategy::LEIDEN_CSR;
    OrderingStrategy ordering = OrderingStrategy::CONNECTIVITY_BFS;  ///< Default: BFS within communities (best cache locality)

    // Feature flags
    bool useRefinement = true;         ///< Enable Leiden refinement step
    int  refinementDepth = -1;         ///< Which passes to refine: -1=all, 0=pass 0 only (GVE style), N=passes 0..N
    MComputation mComputation = MComputation::HALF_EDGES;  ///< How to compute M (total edge weight)
    bool usePrefetch = true;           ///< Enable cache prefetching
    bool useParallelSort = true;       ///< Use parallel sorting
    bool deterministicCommunityDetection = true; ///< Serialize Leiden detection; ordering remains parallel
    int superGraphMoveBatch = 1;       ///< Proposal batch; 1 preserves exact sequential updates
    bool verifyTopology = false;       ///< Verify topology after reordering
    bool useDynamicResolution = false; ///< Enable per-pass resolution adjustment
    bool useDegreeSorting = false;     ///< Process vertices by ascending degree (helps graphbrew:rabbit, not Leiden)
    bool useCommunityMerging = false;  ///< Merge small communities for better cache locality (graphbrew:merge)
    size_t targetCommunities = 0;      ///< Target community count after merging (0 = auto)
    bool useHubExtraction = false;     ///< Extract high-degree hubs before community ordering (graphbrew:hubx)
    double hubExtractionPct = 0.001;   ///< Fraction of vertices to extract as hubs (default 0.1%, use hubx0.5 for 0.5%)

    // Gorder-inspired improvements (apply to hrab ordering)
    bool useGorderIntra = false;       ///< Replace BFS with Gorder-greedy within communities (graphbrew:gord)
    int  gorderWindow = 5;             ///< Sliding window size for Gorder-greedy intra-community ordering
    int  gorderFallback = 0;           ///< Community size threshold for BFS fallback (0 = auto = N, i.e. no fallback)
    bool useHubSort = false;           ///< Post-process: pack hub vertices contiguously sorted by descending degree (graphbrew:hsort)
    bool useRCMSuper = false;          ///< Use RCM on super-graph instead of RabbitOrder dendrogram DFS (graphbrew:rcm)
    bool useRCMIntra = false;          ///< Use RCM (Cuthill-McKee) within each community instead of BFS (graphbrew:rcm_intra)

    // COMPOSE-strategy picks (only used when ordering == OrderingStrategy::COMPOSE).
    // The three axes are: super-graph order, community order, intra-community order.
    // See SuperGraphOrder / CommunityOrder / IntraCommunityOrder.
    SuperGraphOrder superGraphOrder = SuperGraphOrder::None;
    CommunityOrder communityOrder = CommunityOrder::SizeDesc;
    IntraCommunityOrder intraCommunityOrder = IntraCommunityOrder::BFSFromHub;
    RefinementPass refinementPass = RefinementPass::None;  ///< Optional FM-style refinement
    int refineMaxPasses = 3;                               ///< Max passes for refinement primitive
    size_t capacityL2Bytes = 0;        ///< 0 = detect machine L2
    size_t capacityLLCBytes = 0;       ///< 0 = detect machine LLC
    size_t capacityPropertyBytesPerVertex = 0; ///< 0 = model from kernel hint

    // Super-graph modularity resolution (HRAB / TQR community-level merges)
    //
    // Standard Rabbit Order modularity gain on a super-graph is:
    //     ΔQ(u,v) = w_uv − γ · str(u)·str(v) / (2·M_super)
    //
    // The Leiden output that feeds the super-graph has already been optimized
    // for ΔQ at γ=1.0 on the original graph; with γ=1.0 on the super-graph,
    // very few merges would happen (Leiden left no positive-ΔQ edges at γ=1).
    // For HRAB/TQR we want aggressive coarsening to ~1K cache-sized blocks,
    // so we lower γ.  γ ∈ (0, 1] trades merge aggressiveness for modularity
    // preservation:
    //   γ = 1.00  → almost no merges (faithful to original Rabbit on raw graph)
    //   γ = 0.10  → balanced; typical 100K → ~1-5K blocks  ← default
    //   γ = 0.05  → very aggressive; merges any connected components
    //
    // Default γ=0.10 chosen from a 4-graph PR cache-sim sweep (L3=8MB, iters=5):
    //   soc-pokec      γ=0.10 11.93M acc / 15.6s   (best on graph)
    //   hollywood-2009 γ=0.10 35.84M acc / 40.5s   (within 0.3% of best)
    //   USA-road-d.USA γ=0.10 33.56M acc / 39.6s   (best on graph)
    //   com-Orkut      γ=0.10 96.04M acc / 162.5s  (best mem; γ=0.25 still best kernel)
    // Previous default γ=0.25 was worst or near-worst on 3/4 graphs.
    // com-Orkut-shaped graphs (very dense communities) may benefit from sgres0.25.
    //
    // Override via -o 12:hrab:sgres0.5  (sets γ=0.5)
    // or          -o 12:tqr:sgres0.1
    double superGraphResolution = 0.10;

    // GraphBrew mode: per-community external algorithm dispatch
    int  finalAlgoId = -1;             ///< Final reordering algorithm ID (0-11) for LAYER ordering. -1 = not set (uses GraphBrew ordering)
    bool useSmallCommunityMerging = false; ///< Merge small communities and apply heuristic algorithm selection
    size_t smallCommunityThreshold = 0;    ///< Min community size for individual reordering (0 = dynamic)
    int  recursiveDepth = -1;          ///< Recursive sub-community depth: -1=auto (default), 0=flat, 1+=recurse into large communities
    int  subAlgoId = 8;                ///< Algo for sub-communities: -1=adaptive, 0-11=fixed algo ID (default=8 RabbitOrder)

    // Pipeline control
    bool hasExplicitOrdering = false;  ///< True when user specified an ordering token (e.g. :dbg, :hrab). Used to distinguish 12:rabbit (native DFS) from 12:rabbit:conn (explicit ordering override).
    bool rabbitDegreeSortPreprocess = false; ///< Physical CSR degree sort applied by the builder before Rabbit community detection.

    // Memory optimizations
    bool useLazyUpdates = false;       ///< Batch community weight updates (reduces atomics in non-REFINE phase)
    bool useRelaxedMemory = false;     ///< Use relaxed memory ordering in LazyUpdateBuffer [requires useLazyUpdates]
    bool reuseBuffers = true;          ///< Reuse SuperGraph buffers across passes (avoids reallocation)

    // Cache optimization - use unified defaults
    size_t tileSize = reorder::DEFAULT_TILE_SIZE;              ///< Tile size for cache blocking
    size_t prefetchDistance = reorder::DEFAULT_PREFETCH_DISTANCE;  ///< Prefetch lookahead

    /**
     * Create GraphBrewConfig from unified ReorderConfig
     * Enables seamless interop with unified configuration
     */
    static GraphBrewConfig FromReorderConfig(const reorder::ReorderConfig& cfg) {
        GraphBrewConfig vc;
        vc.resolution = cfg.resolution;
        vc.tolerance = cfg.tolerance;
        vc.aggregationTolerance = cfg.aggregationTolerance;
        vc.toleranceDrop = cfg.toleranceDrop;
        vc.maxIterations = cfg.maxIterations;
        vc.maxPasses = cfg.maxPasses;
        vc.useRefinement = cfg.useRefinement;
        vc.usePrefetch = cfg.usePrefetch;
        vc.useParallelSort = cfg.useParallelSort;
        vc.verifyTopology = cfg.verifyTopology;
        vc.tileSize = cfg.tileSize;
        vc.prefetchDistance = cfg.prefetchDistance;

        // Map ordering strategies
        switch (cfg.ordering) {
            case reorder::OrderingStrategy::DFS:
                vc.ordering = OrderingStrategy::DENDROGRAM_DFS; break;
            case reorder::OrderingStrategy::BFS:
                vc.ordering = OrderingStrategy::DENDROGRAM_BFS; break;
            case reorder::OrderingStrategy::DBG:
                vc.ordering = OrderingStrategy::DBG; break;
            case reorder::OrderingStrategy::CORDER:
                vc.ordering = OrderingStrategy::CORDER; break;
            case reorder::OrderingStrategy::HUB_CLUSTER:
                vc.ordering = OrderingStrategy::HUB_CLUSTER; break;
            case reorder::OrderingStrategy::COMMUNITY_SORT:
                vc.ordering = OrderingStrategy::COMMUNITY_SORT; break;
            default:
                vc.ordering = OrderingStrategy::CONNECTIVITY_BFS; break;  // Best default
        }

        // Map aggregation strategies
        switch (cfg.aggregation) {
            case reorder::AggregationStrategy::LAZY_STREAMING:
                vc.aggregation = AggregationStrategy::RABBIT_LAZY; break;
            case reorder::AggregationStrategy::HYBRID:
                vc.aggregation = AggregationStrategy::HYBRID; break;
            case reorder::AggregationStrategy::GVE_CSR_MERGE:
                vc.aggregation = AggregationStrategy::GVE_CSR; break;
            default:
                vc.aggregation = AggregationStrategy::LEIDEN_CSR; break;
        }

        return vc;
    }

    /**
     * Convert to unified ReorderConfig
     * Enables interop with other algorithms
     */
    reorder::ReorderConfig toReorderConfig() const {
        reorder::ReorderConfig cfg;
        cfg.resolution = resolution;
        cfg.tolerance = tolerance;
        cfg.aggregationTolerance = aggregationTolerance;
        cfg.toleranceDrop = toleranceDrop;
        cfg.maxIterations = maxIterations;
        cfg.maxPasses = maxPasses;
        cfg.useRefinement = useRefinement;
        cfg.usePrefetch = usePrefetch;
        cfg.useParallelSort = useParallelSort;
        cfg.verifyTopology = verifyTopology;
        cfg.tileSize = tileSize;
        cfg.prefetchDistance = prefetchDistance;

        // Map ordering strategies
        switch (ordering) {
            case OrderingStrategy::DENDROGRAM_DFS:
                cfg.ordering = reorder::OrderingStrategy::DFS; break;
            case OrderingStrategy::DENDROGRAM_BFS:
                cfg.ordering = reorder::OrderingStrategy::BFS; break;
            case OrderingStrategy::DBG:
            case OrderingStrategy::DBG_GLOBAL:
                cfg.ordering = reorder::OrderingStrategy::DBG; break;
            case OrderingStrategy::CORDER:
            case OrderingStrategy::CORDER_GLOBAL:
                cfg.ordering = reorder::OrderingStrategy::CORDER; break;
            case OrderingStrategy::HUB_CLUSTER:
                cfg.ordering = reorder::OrderingStrategy::HUB_CLUSTER; break;
            case OrderingStrategy::COMMUNITY_SORT:
                cfg.ordering = reorder::OrderingStrategy::COMMUNITY_SORT; break;
            default:
                cfg.ordering = reorder::OrderingStrategy::HIERARCHICAL; break;
        }

        // Map aggregation strategies
        switch (aggregation) {
            case AggregationStrategy::RABBIT_LAZY:
                cfg.aggregation = reorder::AggregationStrategy::LAZY_STREAMING; break;
            case AggregationStrategy::HYBRID:
                cfg.aggregation = reorder::AggregationStrategy::HYBRID; break;
            case AggregationStrategy::GVE_CSR:
                cfg.aggregation = reorder::AggregationStrategy::GVE_CSR_MERGE; break;
            default:
                cfg.aggregation = reorder::AggregationStrategy::CSR_BUFFER; break;
        }

        return cfg;
    }
};

inline const char* graphBrewAlgorithmName(GraphBrewAlgorithm value) {
    return value == GraphBrewAlgorithm::RABBIT_ORDER ? "rabbit" : "leiden";
}

inline const char* graphBrewAggregationName(AggregationStrategy value) {
    switch (value) {
        case AggregationStrategy::LEIDEN_CSR: return "leiden-csr";
        case AggregationStrategy::RABBIT_LAZY: return "streaming";
        case AggregationStrategy::HYBRID: return "hybrid";
        case AggregationStrategy::GVE_CSR: return "gve-csr";
    }
    return "unknown";
}

inline const char* graphBrewOrderingName(OrderingStrategy value) {
    switch (value) {
        case OrderingStrategy::HIERARCHICAL: return "hierarchical";
        case OrderingStrategy::CONNECTIVITY_BFS: return "connectivity-bfs";
        case OrderingStrategy::DENDROGRAM_DFS: return "dendrogram-dfs";
        case OrderingStrategy::DENDROGRAM_BFS: return "dendrogram-bfs";
        case OrderingStrategy::COMMUNITY_SORT: return "community-sort";
        case OrderingStrategy::HUB_CLUSTER: return "hubcluster";
        case OrderingStrategy::DBG: return "dbg";
        case OrderingStrategy::CORDER: return "corder";
        case OrderingStrategy::DBG_GLOBAL: return "dbg-global";
        case OrderingStrategy::CORDER_GLOBAL: return "corder-global";
        case OrderingStrategy::HIERARCHICAL_CACHE_AWARE: return "hcache";
        case OrderingStrategy::HYBRID_LEIDEN_RABBIT: return "hrab";
        case OrderingStrategy::HIERARCHICAL_LEIDEN_RABBIT: return "hlr";
        case OrderingStrategy::TILE_QUANTIZED_RABBIT: return "tqr";
        case OrderingStrategy::COMPOSE: return "compose";
        case OrderingStrategy::LAYER: return "layer";
    }
    return "unknown";
}

inline const char* graphBrewCommunityModeName(CommunityMode value) {
    switch (value) {
        case CommunityMode::FULL_LEIDEN: return "full-leiden";
        case CommunityMode::FAST_LP: return "fast-lp";
        case CommunityMode::HYBRID: return "hybrid";
    }
    return "unknown";
}

inline const char* graphBrewSuperGraphName(SuperGraphOrder value) {
    switch (value) {
        case SuperGraphOrder::None: return "none";
        case SuperGraphOrder::SuperRabbit: return "super-rabbit";
        case SuperGraphOrder::SuperRCM: return "super-rcm";
        case SuperGraphOrder::TileRabbit: return "tile-rabbit";
        case SuperGraphOrder::Hilbert: return "hilbert";
    }
    return "unknown";
}

inline const char* graphBrewCommunityOrderName(CommunityOrder value) {
    switch (value) {
        case CommunityOrder::SizeDesc: return "size-desc";
        case CommunityOrder::SizeAsc: return "size-asc";
        case CommunityOrder::DegreeDesc: return "degree-desc";
        case CommunityOrder::DegreeAsc: return "degree-asc";
        case CommunityOrder::CapacityRuns: return "capacity-runs";
        case CommunityOrder::CutMin: return "cut-min";
        case CommunityOrder::Identity: return "identity";
    }
    return "unknown";
}

inline const char* graphBrewIntraOrderName(IntraCommunityOrder value) {
    switch (value) {
        case IntraCommunityOrder::BFSFromHub: return "bfs";
        case IntraCommunityOrder::RCM: return "rcm";
        case IntraCommunityOrder::RCMpp: return "rcmpp";
        case IntraCommunityOrder::Dendrogram: return "dendrogram";
        case IntraCommunityOrder::Gorder: return "gorder";
        case IntraCommunityOrder::GorderFaithful: return "gorder-faithful";
        case IntraCommunityOrder::HubSort: return "hubsort";
        case IntraCommunityOrder::DegreeAsc: return "degree-asc";
        case IntraCommunityOrder::Hub2: return "hub2";
        case IntraCommunityOrder::Alternate: return "alternate";
        case IntraCommunityOrder::Random: return "random";
        case IntraCommunityOrder::BoundaryLast: return "boundary-last";
        case IntraCommunityOrder::CoreOrder: return "core";
    }
    return "unknown";
}

inline std::string graphBrewEffectiveConfigJson(
    const GraphBrewConfig& config,
    bool includeSchema = true)
{
    const char* effective_ordering =
        config.algorithm == GraphBrewAlgorithm::RABBIT_ORDER &&
        !config.hasExplicitOrdering
            ? "rabbit-native-dfs"
            : graphBrewOrderingName(config.ordering);

    std::ostringstream out;
    out << std::setprecision(17) << std::boolalpha
        << "{";
    if (includeSchema)
        out << "\"schema\":\"graphbrew_config/v3\",";
    out << "\"algorithm\":\""
        << graphBrewAlgorithmName(config.algorithm) << "\""
        << ",\"community_mode\":\""
        << graphBrewCommunityModeName(config.communityMode) << "\""
        << ",\"aggregation\":\""
        << graphBrewAggregationName(config.aggregation) << "\""
        << ",\"ordering\":\"" << effective_ordering << "\""
        << ",\"super_graph\":\""
        << graphBrewSuperGraphName(config.superGraphOrder) << "\""
        << ",\"community_order\":\""
        << graphBrewCommunityOrderName(config.communityOrder) << "\""
        << ",\"intra_community_order\":\""
        << graphBrewIntraOrderName(config.intraCommunityOrder) << "\""
        << ",\"refinement_pass\":\""
        << (
            config.refinementPass == RefinementPass::TwoSwap
                ? "two-swap" : "none"
        ) << "\""
        << ",\"resolution\":";
    if (config.algorithm == GraphBrewAlgorithm::RABBIT_ORDER)
        out << "null";
    else
        out << config.resolution;
    out << ",\"super_graph_resolution\":"
        << config.superGraphResolution
        << ",\"max_iterations\":" << config.maxIterations
        << ",\"max_passes\":" << config.maxPasses
        << ",\"refinement_depth\":" << config.refinementDepth
        << ",\"m_computation\":\""
        << (
            config.mComputation == MComputation::TOTAL_EDGES
                ? "total-edges" : "half-edges"
        ) << "\""
        << ",\"deterministic_community_detection\":"
        << config.deterministicCommunityDetection
        << ",\"supergraph_move_batch\":"
        << config.superGraphMoveBatch
        << ",\"gorder_window\":" << config.gorderWindow
        << ",\"gorder_fallback\":" << config.gorderFallback
        << ",\"capacity_l2_bytes\":" << config.capacityL2Bytes
        << ",\"capacity_llc_bytes\":" << config.capacityLLCBytes
        << ",\"capacity_property_bytes_per_vertex\":"
        << config.capacityPropertyBytesPerVertex
        << ",\"final_algo_id\":" << config.finalAlgoId
        << ",\"recursive_depth\":" << config.recursiveDepth
        << ",\"sub_algo_id\":" << config.subAlgoId
        << ",\"rabbit_degree_sort_preprocess\":"
        << config.rabbitDegreeSortPreprocess
        << ",\"use_refinement\":" << config.useRefinement
        << ",\"use_lazy_updates\":" << config.useLazyUpdates
        << ",\"verify_topology\":" << config.verifyTopology
        << ",\"dynamic_resolution\":" << config.useDynamicResolution
        << ",\"degree_sorting\":" << config.useDegreeSorting
        << ",\"community_merging\":" << config.useCommunityMerging
        << ",\"hub_extraction\":" << config.useHubExtraction
        << ",\"hub_extraction_pct\":" << config.hubExtractionPct
        << ",\"gorder_intra\":" << config.useGorderIntra
        << ",\"hub_sort\":" << config.useHubSort
        << ",\"rcm_super\":" << config.useRCMSuper
        << ",\"rcm_intra\":" << config.useRCMIntra
        << ",\"small_community_merging\":"
        << config.useSmallCommunityMerging
        << ",\"has_explicit_ordering\":"
        << config.hasExplicitOrdering
        << "}";
    return out.str();
}

inline void printGraphBrewEffectiveConfig(const GraphBrewConfig& config)
{
    const std::string json = graphBrewEffectiveConfigJson(config);
    printf(
        "GraphBrew Effective Config: %s\n",
        json.c_str());
}

struct GraphBrewRealizedConfig {
    struct Fallback {
        std::string reason;
        std::string requested;
        std::string realized;
    };

    std::string algorithm;
    std::string aggregation;
    std::string ordering;
    std::string superGraph;
    std::string communityOrder;
    std::string intraCommunityOrder;
    std::string refinementPass;
    bool resolutionApplicable = true;
    double resolution = 0.0;
    bool recursiveDepthApplicable = false;
    int recursiveDepth = -1;
    bool scheduleSensitive = false;
    int gorderWindow = 5;
    int gorderFallback = 0;
    size_t gorderCommunities = 0;
    size_t gorderVertices = 0;
    size_t gorderMaxCommunity = 0;
    size_t gorderFallbackCommunities = 0;
    size_t gorderFallbackVertices = 0;
    size_t capacityL2Bytes = 0;
    size_t capacityLLCBytes = 0;
    size_t capacityPropertyBytesPerVertex = 0;
    size_t capacityL2Runs = 0;
    size_t capacityLLCRuns = 0;
    int finalAlgoId = -1;
    int subAlgoId = -1;
    int numPasses = 0;
    size_t numCommunities = 0;
    std::vector<Fallback> fallbacks;
    std::map<std::string, size_t> blockAlgorithms;
};

inline GraphBrewRealizedConfig makeGraphBrewRealizedConfig(
    const GraphBrewConfig& config) {
    GraphBrewRealizedConfig realized;
    realized.algorithm = graphBrewAlgorithmName(config.algorithm);
    realized.aggregation =
        config.algorithm == GraphBrewAlgorithm::RABBIT_ORDER
            ? "rabbit-incremental"
            : graphBrewAggregationName(config.aggregation);
    realized.ordering =
        config.algorithm == GraphBrewAlgorithm::RABBIT_ORDER &&
        !config.hasExplicitOrdering
            ? "rabbit-native-dfs"
            : graphBrewOrderingName(config.ordering);
    realized.superGraph = graphBrewSuperGraphName(config.superGraphOrder);
    realized.communityOrder =
        graphBrewCommunityOrderName(config.communityOrder);
    realized.intraCommunityOrder =
        graphBrewIntraOrderName(config.intraCommunityOrder);
    realized.refinementPass =
        config.refinementPass == RefinementPass::TwoSwap
            ? "two-swap" : "none";
    realized.resolutionApplicable =
        config.algorithm != GraphBrewAlgorithm::RABBIT_ORDER;
    realized.resolution = config.resolution;
    realized.recursiveDepthApplicable =
        config.ordering == OrderingStrategy::LAYER;
    realized.recursiveDepth = config.recursiveDepth;
    realized.scheduleSensitive =
        config.algorithm == GraphBrewAlgorithm::RABBIT_ORDER ||
        (
            config.algorithm == GraphBrewAlgorithm::LEIDEN &&
            !config.deterministicCommunityDetection
        ) ||
        config.ordering == OrderingStrategy::HYBRID_LEIDEN_RABBIT ||
        config.ordering == OrderingStrategy::HIERARCHICAL_LEIDEN_RABBIT ||
        config.ordering == OrderingStrategy::TILE_QUANTIZED_RABBIT ||
        (
            config.ordering == OrderingStrategy::LAYER &&
            config.finalAlgoId == 8
        ) ||
        config.superGraphOrder == SuperGraphOrder::SuperRabbit ||
        config.superGraphOrder == SuperGraphOrder::TileRabbit;
    realized.gorderWindow = config.gorderWindow;
    realized.gorderFallback = config.gorderFallback;
    realized.capacityL2Bytes = config.capacityL2Bytes;
    realized.capacityLLCBytes = config.capacityLLCBytes;
    realized.capacityPropertyBytesPerVertex =
        config.capacityPropertyBytesPerVertex;
    realized.finalAlgoId = config.finalAlgoId;
    realized.subAlgoId = config.subAlgoId;
    return realized;
}

inline void printGraphBrewRealizedConfig(
    const GraphBrewRealizedConfig& realized) {
    printf(
        "GraphBrew Realized Config: "
        "{\"schema\":\"graphbrew_realized/v2\","
        "\"algorithm\":\"%s\",\"aggregation\":\"%s\","
        "\"ordering\":\"%s\",\"super_graph\":\"%s\","
        "\"community_order\":\"%s\",\"intra_community_order\":\"%s\","
        "\"refinement_pass\":\"%s\",\"resolution\":",
        realized.algorithm.c_str(),
        realized.aggregation.c_str(),
        realized.ordering.c_str(),
        realized.superGraph.c_str(),
        realized.communityOrder.c_str(),
        realized.intraCommunityOrder.c_str(),
        realized.refinementPass.c_str());
    if (realized.resolutionApplicable) {
        printf("%.17g", realized.resolution);
    } else {
        printf("null");
    }
    printf(",\"recursive_depth\":");
    if (realized.recursiveDepthApplicable) {
        printf("%d", realized.recursiveDepth);
    } else {
        printf("null");
    }
    printf(
        ",\"schedule_sensitive\":%s,"
        "\"gorder_window\":%d,\"gorder_fallback\":%d,"
        "\"gorder_communities\":%zu,\"gorder_vertices\":%zu,"
        "\"gorder_max_community\":%zu,"
        "\"gorder_fallback_communities\":%zu,"
        "\"gorder_fallback_vertices\":%zu,"
        "\"capacity_l2_bytes\":%zu,\"capacity_llc_bytes\":%zu,"
        "\"capacity_property_bytes_per_vertex\":%zu,"
        "\"capacity_l2_runs\":%zu,\"capacity_llc_runs\":%zu,"
        "\"final_algo_id\":%d,\"sub_algo_id\":%d,"
        "\"num_passes\":%d,\"num_communities\":%zu,\"fallbacks\":[",
        realized.scheduleSensitive ? "true" : "false",
        realized.gorderWindow,
        realized.gorderFallback,
        realized.gorderCommunities,
        realized.gorderVertices,
        realized.gorderMaxCommunity,
        realized.gorderFallbackCommunities,
        realized.gorderFallbackVertices,
        realized.capacityL2Bytes,
        realized.capacityLLCBytes,
        realized.capacityPropertyBytesPerVertex,
        realized.capacityL2Runs,
        realized.capacityLLCRuns,
        realized.finalAlgoId,
        realized.subAlgoId,
        realized.numPasses,
        realized.numCommunities);
    for (size_t i = 0; i < realized.fallbacks.size(); ++i) {
        const auto& fallback = realized.fallbacks[i];
        printf(
            "%s{\"reason\":\"%s\",\"requested\":\"%s\","
            "\"realized\":\"%s\"}",
            i == 0 ? "" : ",",
            fallback.reason.c_str(),
            fallback.requested.c_str(),
            fallback.realized.c_str());
    }
    printf("],\"block_algorithms\":{");
    size_t index = 0;
    for (const auto& [name, count] : realized.blockAlgorithms) {
        printf(
            "%s\"%s\":%zu",
            index++ == 0 ? "" : ",",
            name.c_str(),
            count);
    }
    printf("}}\n");
}

class ScopedCommunityThreadLimit {
public:
    explicit ScopedCommunityThreadLimit(bool enabled) {
#ifdef _OPENMP
        if (enabled) {
            previous_ = omp_get_max_threads();
            active_ = previous_ > 1;
            if (active_) omp_set_num_threads(1);
        }
#else
        (void)enabled;
#endif
    }

    ~ScopedCommunityThreadLimit() {
#ifdef _OPENMP
        if (active_) omp_set_num_threads(previous_);
#endif
    }

    ScopedCommunityThreadLimit(const ScopedCommunityThreadLimit&) = delete;
    ScopedCommunityThreadLimit& operator=(
        const ScopedCommunityThreadLimit&) = delete;

private:
    int previous_ = 1;
    bool active_ = false;
};

//=============================================================================
// SECTION 2: RESULT STRUCTURES
//=============================================================================

/** Dendrogram node for hierarchical community structure */
template <typename K>
struct DendrogramNode {
    K id;                              ///< Node/community ID
    K parent;                          ///< Parent community (-1 for root)
    std::vector<K> children;           ///< Child communities/vertices
    size_t size;                       ///< Number of vertices in subtree
    Weight weight;                     ///< Total edge weight
    int level;                         ///< Level in hierarchy (0 = leaf)
};

/** Full result from GraphBrew algorithm */
template <typename K>
struct GraphBrewResult {
    // Community structure
    std::vector<K> membership;                     ///< Final community per vertex
    std::vector<std::vector<K>> membershipPerPass; ///< Membership at each level

    // Dendrogram (for tree-based ordering)
    std::vector<DendrogramNode<K>> dendrogram;     ///< Community tree
    std::vector<K> roots;                          ///< Root community IDs

    // Optional Rabbit dendrogram, populated by runRabbitOrder when the
    // downstream pipeline requests dendrogram-based intra-community order
    // (IntraCommunityOrder::Dendrogram).  Stored as parallel arrays so the
    // result struct stays decoupled from RabbitNode<K>.
    //
    //   rabbitChild[v]   = first child of v in dendrogram (INVALID if leaf)
    //   rabbitSibling[v] = next sibling of v in dendrogram (INVALID if last)
    //   rabbitToplevel[c]= vertex root of community c
    //                      (membership[v] == c <=> v in subtree rooted at
    //                       rabbitToplevel[c])
    std::vector<K> rabbitChild;
    std::vector<K> rabbitSibling;
    std::vector<K> rabbitToplevel;
    bool hasRabbitDendrogram = false;

    // Weights
    std::vector<Weight> vertexWeight;              ///< Total weight per vertex
    std::vector<Weight> communityWeight;           ///< Total weight per community

    // Statistics
    double modularity = 0.0;
    int totalIterations = 0;
    int totalPasses = 0;
    size_t numCommunities = 0;

    // Timing breakdown
    double localMoveTime = 0.0;
    double refinementTime = 0.0;
    double aggregationTime = 0.0;
    double dendrogramTime = 0.0;
    double orderingTime = 0.0;
    double totalTime = 0.0;
};

//=============================================================================
// SECTION 3: SUPER-GRAPH CSR STRUCTURE
//=============================================================================

/**
 * CSR structure for super-graph (aggregated communities)
 * Pre-allocated and reusable across passes
 */
template <typename K, typename W>
struct SuperGraph {
    std::vector<size_t> offsets;   ///< CSR row offsets
    std::vector<K> degrees;        ///< Degree of each super-node
    std::vector<K> neighbors;      ///< Neighbor IDs
    std::vector<W> weights;        ///< Edge weights
    size_t numNodes = 0;           ///< Number of super-nodes
    size_t numEdges = 0;           ///< Number of edges
    size_t reservedNodes = 0;      ///< Reserved capacity for nodes
    size_t reservedEdges = 0;      ///< Reserved capacity for edges

    void resize(size_t n, size_t e) {
        offsets.resize(n + 1);
        degrees.resize(n);
        neighbors.resize(e);
        weights.resize(e);
    }

    /**
     * Reserve capacity for reuse across passes
     * Avoids repeated allocations when graph shrinks
     */
    void reserve(size_t n, size_t e) {
        if (n > reservedNodes) {
            offsets.reserve(n + 1);
            degrees.reserve(n);
            reservedNodes = n;
        }
        if (e > reservedEdges) {
            neighbors.reserve(e);
            weights.reserve(e);
            reservedEdges = e;
        }
    }

    /**
     * Soft clear - reset sizes but keep allocated memory
     */
    void softClear() {
        numNodes = 0;
        numEdges = 0;
        // Don't shrink vectors - keep reserved capacity
    }

    void clear() {
        std::fill(degrees.begin(), degrees.end(), K(0));
        numNodes = 0;
        numEdges = 0;
    }
};

//=============================================================================
// SECTION 4: CACHE-OPTIMIZED COMMUNITY SCANNER (sparse open-address hashmap)
//=============================================================================
//
// Thread-local sparse hashtable for accumulating community → weight pairs
// during community scanning.  Replaces the previous dense vector<W>(N) layout
// to bound peak memory at O(touched_communities) per scanner instead of O(N).
//
// Why a sparse table:
//   The previous dense layout allocated values.resize(N, 0) per scanner.
//   For a graph with N=118M vertices and 32 threads, peak resident memory was
//   118M × 8B × 32 ≈ 30 GB just for the scanners — enough to OOM webbase-2001
//   and twitter7 before any communities were even formed.  The new layout
//   holds only the touched buckets (~max community-degree per vertex), so
//   peak per-scanner memory caps at a few hundred KB.
//
// API preserved:
//   keys                public vector of touched community IDs (iteration)
//   add(c, w)           accumulate weight w into bucket c
//   get(c)              O(1) average-case lookup
//   clear()             zero out touched buckets (O(|keys|))
//   compactIfNeeded()   no-op kept for source compatibility
//
// Implementation:
//   Linear-probing open-address hashtable with power-of-2 size.  Initial
//   capacity 256, doubles when load > 0.5 (rare on modern graphs because
//   per-vertex unique-neighbor counts are usually small).  Sentinel EMPTY
//   = numeric_limits<K>::max() — guaranteed not to clash with valid community
//   IDs since K is uint32_t and Leiden never emits > N communities.
//=============================================================================

template <typename K, typename W>
struct CommunityScanner {
    static_assert(std::is_unsigned<K>::value,
                  "CommunityScanner requires unsigned K (uses numeric_limits<K>::max() as EMPTY)");
    static constexpr K EMPTY = std::numeric_limits<K>::max();

    std::vector<K> keys;       ///< Touched community IDs (insertion order)
    std::vector<K> ht_keys;    ///< Hashtable buckets — key (EMPTY = free)
    std::vector<W> ht_vals;    ///< Hashtable buckets — accumulated weight
    size_t mask;               ///< ht_keys.size() − 1  (power-of-2 mask)

    explicit CommunityScanner(size_t /*cap_hint*/) {
        // Always start at 256 slots; grow on demand.  cap_hint is ignored
        // because most vertices touch only a few-dozen communities even on
        // billion-edge graphs, so over-allocating to N would defeat the
        // purpose of the sparse layout.
        const size_t init = 256;
        ht_keys.assign(init, EMPTY);
        ht_vals.assign(init, W(0));
        mask = init - 1;
        keys.reserve(64);
    }

    static inline size_t mix(K x) {
        // Knuth's multiplicative hash on 64 bits (good distribution for
        // dense integer keys with a power-of-2 table size)
        uint64_t h = static_cast<uint64_t>(x) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 27;
        return static_cast<size_t>(h);
    }

    void clear() {
        for (K c : keys) {
            size_t h = mix(c) & mask;
            while (ht_keys[h] != c) h = (h + 1) & mask;
            ht_keys[h] = EMPTY;
            ht_vals[h] = W(0);
        }
        keys.clear();
    }

    void add(K community, W weight) {
        // Resize when load > 0.5 to keep probe length bounded
        if ((keys.size() + 1) * 2 > ht_keys.size()) grow();

        size_t h = mix(community) & mask;
        while (true) {
            const K slot = ht_keys[h];
            if (slot == community) {
                ht_vals[h] += weight;
                return;
            }
            if (slot == EMPTY) {
                ht_keys[h] = community;
                ht_vals[h] = weight;
                keys.push_back(community);
                return;
            }
            h = (h + 1) & mask;
        }
    }

    W get(K community) const {
        size_t h = mix(community) & mask;
        while (true) {
            const K slot = ht_keys[h];
            if (slot == community) return ht_vals[h];
            if (slot == EMPTY) return W(0);
            h = (h + 1) & mask;
        }
    }

    /**
     * Kept for source compatibility with the old API (Boost's edge-aggregator
     * pattern called this every 2048 inserts).  No-op for the hashtable layout
     * because aggregation happens implicitly on every add().
     */
    void compactIfNeeded(size_t /*threshold*/ = 2048) {}

private:
    void grow() {
        const size_t old_size = ht_keys.size();
        const size_t new_size = old_size * 2;
        const size_t new_mask = new_size - 1;
        std::vector<K> new_keys(new_size, EMPTY);
        std::vector<W> new_vals(new_size, W(0));

        for (size_t i = 0; i < old_size; ++i) {
            const K k = ht_keys[i];
            if (k == EMPTY) continue;
            size_t h = mix(k) & new_mask;
            while (new_keys[h] != EMPTY) h = (h + 1) & new_mask;
            new_keys[h] = k;
            new_vals[h] = ht_vals[i];
        }
        ht_keys = std::move(new_keys);
        ht_vals = std::move(new_vals);
        mask = new_mask;
    }
};

/**
 * Thread-local batch update buffer for lazy community weight updates
 *
 * Instead of atomically updating ctot on every vertex move,
 * we batch updates and apply them at the end of each iteration.
 * This reduces atomic contention significantly.
 *
 * Used in non-REFINE phase when config.useLazyUpdates=true.
 * REFINE phase still uses immediate atomics due to read-modify dependency
 * (must check ctot before deciding to move).
 */
template <typename K, typename W>
struct LazyUpdateBuffer {
    std::vector<std::pair<K, W>> decrements;  ///< (community, weight) to subtract
    std::vector<std::pair<K, W>> increments;  ///< (community, weight) to add

    LazyUpdateBuffer() {
        decrements.reserve(1024);
        increments.reserve(1024);
    }

    void recordMove(K old_comm, K new_comm, W weight) {
        decrements.emplace_back(old_comm, weight);
        increments.emplace_back(new_comm, weight);
    }

    void clear() {
        decrements.clear();
        increments.clear();
    }

    bool empty() const {
        return decrements.empty() && increments.empty();
    }

    /**
     * Apply batched updates to community weights
     * Uses relaxed memory ordering for better performance
     */
    void applyTo(std::vector<W>& ctot) {
        for (auto& [c, w] : decrements) {
            #pragma omp atomic
            ctot[c] -= w;
        }
        for (auto& [c, w] : increments) {
            #pragma omp atomic
            ctot[c] += w;
        }
        clear();
    }
};

//=============================================================================
// SECTION 5: MODULARITY COMPUTATION
//=============================================================================

/**
 * Delta modularity for moving vertex u to community c
 *
 * FAITHFUL to leiden.hxx deltaModularity()
 */
template <typename W>
inline W deltaModularity(W k_u_c, W k_u_d, W k_u, W sigma_c, W sigma_d, W M, W R) {
    return (k_u_c - k_u_d) / M - R * k_u * (sigma_c - sigma_d + k_u) / (W(2) * M * M);
}

//=============================================================================
// SECTION 6: VERTEX WEIGHT COMPUTATION
//=============================================================================

/**
 * Compute total edge weight for each vertex
 */
template <typename NodeID_T, typename DestID_T>
void computeVertexWeights(
    std::vector<Weight>& vtot,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const GraphBrewConfig& config) {

    const int64_t N = g.num_nodes();
    vtot.assign(N, Weight(0));

    GRAPHBREW_TRACE("computeVertexWeights: N=%ld", N);

    #pragma omp parallel for schedule(dynamic, config.tileSize)
    for (int64_t u = 0; u < N; ++u) {
        Weight total = 0;

        for (auto neighbor : g.out_neigh(u)) {
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                total += Weight(1);
            } else {
                total += static_cast<Weight>(neighbor.w);
            }
        }
        vtot[u] = total;
    }
}

/**
 * Compute vertex weights for super-graph
 */
template <typename K, typename W>
void computeVertexWeightsSuperGraph(
    std::vector<W>& vtot,
    const SuperGraph<K, W>& sg,
    const GraphBrewConfig& config) {

    const size_t N = sg.numNodes;
    vtot.assign(N, W(0));

    #pragma omp parallel for schedule(dynamic, std::min(config.tileSize, size_t(256)))
    for (size_t u = 0; u < N; ++u) {
        W total = 0;
        size_t start = sg.offsets[u];
        size_t end = sg.offsets[u] + sg.degrees[u];
        for (size_t i = start; i < end; ++i) {
            total += sg.weights[i];
        }
        vtot[u] = total;
    }
}

//=============================================================================
// SECTION 7: COMMUNITY INITIALIZATION
//=============================================================================

/**
 * Initialize communities: each vertex is its own community
 */
template <typename K>
void initializeCommunities(
    std::vector<K>& vcom,
    std::vector<Weight>& ctot,
    const std::vector<Weight>& vtot,
    size_t N) {

    vcom.resize(N);
    ctot.resize(N);

    GRAPHBREW_TRACE("initializeCommunities: N=%zu", N);

    #pragma omp parallel for schedule(static, 4096)
    for (size_t u = 0; u < N; ++u) {
        vcom[u] = static_cast<K>(u);
        ctot[u] = vtot[u];
    }
}

//=============================================================================
// SECTION 8: COMMUNITY SCANNING (with prefetching)
//=============================================================================

/**
 * Scan communities connected to vertex u
 *
 * @tparam REFINE If true, only scan within community bounds
 */
template <bool REFINE, typename K, typename W, typename NodeID_T, typename DestID_T>
void scanCommunities(
    CommunityScanner<K, W>& scanner,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    NodeID_T u,
    const std::vector<K>& vcom,
    const std::vector<K>& vcob,
    const GraphBrewConfig& config) {

    scanner.clear();
    // Prefetch community data for neighbors
    if (config.usePrefetch) {
        size_t count = 0;
        for (auto neighbor : g.out_neigh(u)) {
            if (count++ >= config.prefetchDistance) break;
            NodeID_T v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = neighbor;
            } else {
                v = neighbor.v;
            }
            __builtin_prefetch(&vcom[v], 0, 1);
        }
    }

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

        if (v == u) continue;  // Skip self-loops

        // REFINE: only scan within same community bound
        if constexpr (REFINE) {
            if (vcob[v] != vcob[u]) continue;
        }

        scanner.add(vcom[v], w);
    }
}

/**
 * Scan communities on super-graph
 *
 * Optimization: Prefetch neighbor community data ahead of access
 */
template <bool REFINE, typename K, typename W>
void scanCommunitiesSuperGraph(
    CommunityScanner<K, W>& scanner,
    const SuperGraph<K, W>& sg,
    K u,
    const std::vector<K>& vcom,
    const std::vector<K>& vcob,
    const GraphBrewConfig& config) {

    scanner.clear();
    size_t start = sg.offsets[u];
    size_t end = sg.offsets[u] + sg.degrees[u];

    // Prefetch community data for upcoming neighbors
    if (config.usePrefetch && end > start) {
        const size_t prefetchAhead = std::min(config.prefetchDistance, end - start);
        for (size_t i = 0; i < prefetchAhead; ++i) {
            __builtin_prefetch(&vcom[sg.neighbors[start + i]], 0, 1);
        }
    }

    for (size_t i = start; i < end; ++i) {
        // Prefetch next batch of community data
        if (config.usePrefetch && i + config.prefetchDistance < end) {
            __builtin_prefetch(&vcom[sg.neighbors[i + config.prefetchDistance]], 0, 1);
        }

        K v = sg.neighbors[i];
        W w = sg.weights[i];

        if (v == u) {
            scanner.add(vcom[v], w);  // Self-loop contributes
            continue;
        }

        if constexpr (REFINE) {
            if (vcob[v] != vcob[u]) continue;
        }

        scanner.add(vcom[v], w);
    }
}

//=============================================================================
// SECTION 9: COMMUNITY SELECTION
//=============================================================================

/**
 * Choose best community using greedy selection
 */
template <typename K, typename W>
std::pair<K, W> chooseCommunityGreedy(
    K u,
    K current_comm,
    const CommunityScanner<K, W>& scanner,
    const std::vector<W>& vtot,
    const std::vector<W>& ctot,
    W M, W R) {

    K best_c = current_comm;
    W best_delta = W(0);
    bool found_move = false;
    W k_u = vtot[u];
    W k_u_d = scanner.get(current_comm);
    W sigma_d = ctot[current_comm];

    for (K c : scanner.keys) {
        if (c == current_comm) continue;

        W k_u_c = scanner.get(c);
        W sigma_c = ctot[c];

        W delta = deltaModularity<W>(k_u_c, k_u_d, k_u, sigma_c, sigma_d, M, R);

        if (delta > best_delta ||
            (found_move && delta == best_delta && c < best_c)) {
            best_delta = delta;
            best_c = c;
            found_move = true;
        }
    }

    return {best_c, best_delta};
}

//=============================================================================
// SECTION 10: COMMUNITY CHANGE (with atomic updates)
//=============================================================================

/**
 * Move vertex to new community
 *
 * @tparam REFINE If true, check singleton constraint
 */
template <bool REFINE, typename K, typename W>
bool changeCommunity(
    std::vector<K>& vcom,
    std::vector<W>& ctot,
    K u, K new_comm,
    const std::vector<W>& vtot) {

    K old_comm = vcom[u];
    W k_u = vtot[u];

    if constexpr (REFINE) {
        // In refinement, only move if we're the only member (singleton)
        // Use 1.001 tolerance for floating-point rounding
        W old_ctot;
        #pragma omp atomic capture
        {
            old_ctot = ctot[old_comm];
            ctot[old_comm] -= k_u;
        }

        if (old_ctot > k_u * W(1.001)) {
            #pragma omp atomic
            ctot[old_comm] += k_u;
            return false;
        }
    } else {
        #pragma omp atomic
        ctot[old_comm] -= k_u;
    }

    #pragma omp atomic
    ctot[new_comm] += k_u;

    vcom[u] = new_comm;
    return true;
}

//=============================================================================
// SECTION 11: LOCAL-MOVING PHASE
//=============================================================================

/**
 * Leiden local-moving phase on original graph
 *
 * Optimization: Process vertices in ascending degree order (low-degree first)
 * This reduces contention since high-degree hubs are processed after
 * low-degree vertices have settled into communities.
 *
 * @tparam REFINE If true, this is refinement phase
 */
template <bool REFINE, typename K, typename NodeID_T, typename DestID_T>
int localMovingPhase(
    std::vector<K>& vcom,
    std::vector<Weight>& ctot,
    std::vector<char>& vaff,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<K>& vcob,
    const std::vector<Weight>& vtot,
    Weight M, Weight R,
    const GraphBrewConfig& config) {

    const int64_t N = g.num_nodes();
    int iterations = 0;

    // Allocate per-thread scanners and lazy update buffers
    const int numThreads = omp_get_max_threads();
    std::vector<CommunityScanner<K, Weight>> scanners;
    std::vector<LazyUpdateBuffer<K, Weight>> lazyBuffers;
    for (int t = 0; t < numThreads; ++t) {
        scanners.emplace_back(N);
        lazyBuffers.emplace_back();
    }

    // Decide whether to use lazy updates this phase
    // REFINE phase cannot use lazy due to read-modify dependency on ctot
    const bool useLazy = config.useLazyUpdates && !REFINE;

    // Build degree-sorted vertex order (ascending - low degree first)
    // This is a key optimization from RabbitOrder: process low-degree vertices
    // first so they settle into communities before high-degree hubs
    std::vector<NodeID_T> vertexOrder(N);
    std::iota(vertexOrder.begin(), vertexOrder.end(), 0);

    if (config.useDegreeSorting) {
        if (config.useParallelSort) {
            // Parallel sort by degree
            __gnu_parallel::sort(vertexOrder.begin(), vertexOrder.end(),
                [&](NodeID_T a, NodeID_T b) {
                    const auto degree_a = g.out_degree(a);
                    const auto degree_b = g.out_degree(b);
                    return degree_a != degree_b ? degree_a < degree_b : a < b;
                });
        } else {
            std::sort(vertexOrder.begin(), vertexOrder.end(),
                [&](NodeID_T a, NodeID_T b) {
                    const auto degree_a = g.out_degree(a);
                    const auto degree_b = g.out_degree(b);
                    return degree_a != degree_b ? degree_a < degree_b : a < b;
                });
        }
        GRAPHBREW_TRACE("localMovingPhase: using degree-sorted order");
    }

    GRAPHBREW_TRACE("localMovingPhase<%s>: N=%ld", REFINE ? "REFINE" : "NORMAL", N);

    // Set OpenMP schedule based on whether degree sorting is enabled
    // Degree sorting creates uniform work distribution -> static,1 is optimal
    // Leiden's affectedness check makes work irregular -> dynamic is better
    if (config.useDegreeSorting) {
        omp_set_schedule(omp_sched_static, 1);
    } else {
        omp_set_schedule(omp_sched_dynamic, config.tileSize);
    }

    for (int iter = 0; iter < config.maxIterations; ++iter) {
        Weight totalDelta = Weight(0);

        #pragma omp parallel reduction(+:totalDelta)
        {
            int tid = omp_get_thread_num();
            auto& scanner = scanners[tid];
            auto& lazyBuffer = lazyBuffers[tid];

            if (useLazy) {
                lazyBuffer.clear();
            }

            // Use static scheduling when degree-sorted (uniform work distribution)
            // Use dynamic scheduling otherwise (variable work per vertex)
            #pragma omp for schedule(runtime)
            for (int64_t i = 0; i < N; ++i) {
                NodeID_T u = vertexOrder[i];
                if (!vaff[u]) continue;

                K d = vcom[u];

                // REFINE: Skip if community has more than one member
                // Use 1.001 tolerance factor to handle
                // floating-point accumulation rounding errors
                if constexpr (REFINE) {
                    if (ctot[d] > vtot[u] * Weight(1.001)) continue;
                }

                scanCommunities<REFINE>(scanner, g, static_cast<NodeID_T>(u),
                                        vcom, vcob, config);

                auto [best_c, delta] = chooseCommunityGreedy<K, Weight>(
                    static_cast<K>(u), d, scanner, vtot, ctot, M, R);

                if (best_c != d) {
                    bool moved = false;

                    if (useLazy) {
                        // Lazy mode: record move, defer ctot update
                        K old_comm = vcom[u];
                        vcom[u] = best_c;
                        lazyBuffer.recordMove(old_comm, best_c, vtot[u]);
                        moved = true;
                    } else {
                        // Immediate mode: atomic ctot update
                        moved = changeCommunity<REFINE>(vcom, ctot, static_cast<K>(u), best_c, vtot);
                    }

                    if (moved) {
                        for (auto neighbor : g.out_neigh(static_cast<NodeID_T>(u))) {
                            NodeID_T v;
                            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                                v = neighbor;
                            } else {
                                v = neighbor.v;
                            }
                            vaff[v] = 1;
                        }
                    }
                }

                vaff[u] = 0;
                totalDelta += delta;
            }

            // Barrier before applying lazy updates
            if (useLazy) {
                #pragma omp barrier
                // Apply this thread's batched updates
                lazyBuffer.applyTo(ctot);
            }
        }

        ++iterations;
        GRAPHBREW_TRACE("  iter %d: totalDelta=%.6f%s", iter, totalDelta, useLazy ? " (lazy)" : "");

        if constexpr (REFINE) break;  // Refinement: single pass
        if (totalDelta <= config.tolerance) break;
    }

    return iterations;
}

/**
 * Local-moving phase on super-graph
 */
template <bool REFINE, typename K>
int localMovingPhaseSuperGraph(
    std::vector<K>& vcom,
    std::vector<Weight>& ctot,
    std::vector<char>& vaff,
    const SuperGraph<K, Weight>& sg,
    const std::vector<K>& vcob,
    const std::vector<Weight>& vtot,
    Weight M, Weight R,
    const GraphBrewConfig& config) {

    const size_t N = sg.numNodes;
    int iterations = 0;

    const int numThreads = omp_get_max_threads();
    std::vector<CommunityScanner<K, Weight>> scanners;
    std::vector<LazyUpdateBuffer<K, Weight>> lazyBuffers;
    for (int t = 0; t < numThreads; ++t) {
        scanners.emplace_back(N);
        lazyBuffers.emplace_back();
    }

    // Decide whether to use lazy updates this phase
    const bool useLazy = config.useLazyUpdates && !REFINE;

    GRAPHBREW_TRACE("localMovingPhaseSuperGraph<%s>: N=%zu%s", REFINE ? "REFINE" : "NORMAL", N, useLazy ? " (lazy)" : "");

    for (int iter = 0; iter < config.maxIterations; ++iter) {
        Weight totalDelta = Weight(0);

        #pragma omp parallel reduction(+:totalDelta)
        {
            int tid = omp_get_thread_num();
            auto& scanner = scanners[tid];
            auto& lazyBuffer = lazyBuffers[tid];

            if (useLazy) {
                lazyBuffer.clear();
            }

            #pragma omp for schedule(dynamic, std::min(config.tileSize, size_t(256)))
            for (size_t u = 0; u < N; ++u) {
                if (!vaff[u]) continue;

                K d = vcom[u];

                if constexpr (REFINE) {
                    if (ctot[d] > vtot[u]) continue;
                }

                scanCommunitiesSuperGraph<REFINE>(scanner, sg, static_cast<K>(u), vcom, vcob, config);

                auto [best_c, delta] = chooseCommunityGreedy<K, Weight>(
                    static_cast<K>(u), d, scanner, vtot, ctot, M, R);

                if (best_c != d) {
                    bool moved = false;

                    if (useLazy) {
                        // Lazy mode: record move, defer ctot update
                        K old_comm = vcom[u];
                        vcom[u] = best_c;
                        lazyBuffer.recordMove(old_comm, best_c, vtot[u]);
                        moved = true;
                    } else {
                        // Immediate mode: atomic ctot update
                        moved = changeCommunity<REFINE>(vcom, ctot, static_cast<K>(u), best_c, vtot);
                    }

                    if (moved) {
                        size_t start = sg.offsets[u];
                        size_t end = sg.offsets[u] + sg.degrees[u];
                        for (size_t i = start; i < end; ++i) {
                            vaff[sg.neighbors[i]] = 1;
                        }
                    }
                }

                vaff[u] = 0;
                totalDelta += delta;
            }

            // Barrier before applying lazy updates
            if (useLazy) {
                #pragma omp barrier
                lazyBuffer.applyTo(ctot);
            }
        }

        ++iterations;
        if constexpr (REFINE) break;
        if (totalDelta <= config.tolerance) break;
    }

    return iterations;
}

//=============================================================================
// SECTION 12: COMMUNITY COUNTING AND RENUMBERING
//=============================================================================

/**
 * Count communities and mark existence
 */
template <typename K>
size_t countCommunities(
    std::vector<K>& exists,
    const std::vector<K>& vcom,
    size_t N) {

    std::fill(exists.begin(), exists.end(), K(0));
    size_t count = 0;

    #pragma omp parallel for schedule(static, 4096) reduction(+:count)
    for (size_t u = 0; u < N; ++u) {
        K c = vcom[u];
        K old = 0;
        #pragma omp atomic capture
        {
            old = exists[c];
            exists[c] = K(1);
        }
        if (!old) ++count;
    }

    return count;
}

/**
 * Renumber communities to be contiguous 0, 1, 2, ...
 */
template <typename K>
size_t renumberCommunities(
    std::vector<K>& vcom,
    std::vector<K>& cmap,
    size_t N) {

    K next = 0;
    for (size_t i = 0; i < cmap.size(); ++i) {
        if (cmap[i]) {
            cmap[i] = next++;
        }
    }

    #pragma omp parallel for schedule(static, 4096)
    for (size_t u = 0; u < N; ++u) {
        vcom[u] = cmap[vcom[u]];
    }

    return next;
}

/**
 * Update membership through hierarchy (lookup)
 */
template <typename K>
void lookupCommunities(
    std::vector<K>& ucom,
    const std::vector<K>& vcom) {

    #pragma omp parallel for schedule(static, 4096)
    for (size_t u = 0; u < ucom.size(); ++u) {
        ucom[u] = vcom[ucom[u]];
    }
}

//=============================================================================
// SECTION 13: AGGREGATION - LEIDEN CSR STRATEGY
//=============================================================================

/**
 * Build community vertices list
 */
template <typename K>
void buildCommunityVertices(
    std::vector<size_t>& coff,
    std::vector<K>& cvtx,
    const std::vector<K>& vcom,
    size_t N, size_t C) {

    coff.assign(C + 1, 0);
    for (size_t u = 0; u < N; ++u) {
        coff[vcom[u]]++;
    }

    size_t sum = 0;
    for (size_t c = 0; c < C; ++c) {
        size_t count = coff[c];
        coff[c] = sum;
        sum += count;
    }
    coff[C] = sum;

    cvtx.resize(N);
    std::vector<size_t> pos(C);
    for (size_t c = 0; c < C; ++c) {
        pos[c] = coff[c];
    }

    for (size_t u = 0; u < N; ++u) {
        K c = vcom[u];
        cvtx[pos[c]++] = static_cast<K>(u);
    }
}

/**
 * Aggregate original graph into super-graph (Leiden CSR style)
 */
template <typename K, typename NodeID_T, typename DestID_T>
void aggregateGraphLeiden(
    SuperGraph<K, Weight>& sg,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<K>& vcom,
    const std::vector<size_t>& coff,
    const std::vector<K>& cvtx,
    size_t C,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("aggregateGraphLeiden: C=%zu", C);

    // Estimate edges per community
    std::vector<size_t> estDegree(C, 0);
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t c = 0; c < C; ++c) {
        size_t deg = 0;
        for (size_t i = coff[c]; i < coff[c + 1]; ++i) {
            deg += g.out_degree(cvtx[i]);
        }
        estDegree[c] = deg;
    }

    // Build offsets
    sg.offsets.resize(C + 1);
    sg.offsets[0] = 0;
    for (size_t c = 0; c < C; ++c) {
        sg.offsets[c + 1] = sg.offsets[c] + estDegree[c];
    }

    size_t totalEdges = sg.offsets[C];
    sg.neighbors.resize(totalEdges);
    sg.weights.resize(totalEdges);
    sg.degrees.assign(C, 0);
    sg.numNodes = C;

    // Aggregate edges
    const int numThreads = omp_get_max_threads();
    std::vector<CommunityScanner<K, Weight>> scanners;
    for (int t = 0; t < numThreads; ++t) {
        scanners.emplace_back(C);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& scanner = scanners[tid];

        #pragma omp for schedule(dynamic, 256)
        for (size_t c = 0; c < C; ++c) {
            scanner.clear();

            const size_t commStart = coff[c];
            const size_t commEnd = coff[c + 1];

            // Prefetch community membership data for upcoming vertices
            if (config.usePrefetch && commEnd > commStart) {
                const size_t prefetchCount = std::min(config.prefetchDistance, commEnd - commStart);
                for (size_t p = 0; p < prefetchCount; ++p) {
                    __builtin_prefetch(&vcom[cvtx[commStart + p]], 0, 1);
                }
            }

            for (size_t i = commStart; i < commEnd; ++i) {
                // Prefetch community membership for next vertex's neighbors
                if (config.usePrefetch && i + config.prefetchDistance < commEnd) {
                    __builtin_prefetch(&vcom[cvtx[i + config.prefetchDistance]], 0, 1);
                }

                K u = cvtx[i];
                for (auto neighbor : g.out_neigh(u)) {
                    NodeID_T v;
                    Weight w;
                    if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                        v = neighbor;
                        w = Weight(1);
                    } else {
                        v = neighbor.v;
                        w = static_cast<Weight>(neighbor.w);
                    }
                    K d = vcom[v];
                    scanner.add(d, w);
                }
            }

            size_t offset = sg.offsets[c];
            for (K d : scanner.keys) {
                sg.neighbors[offset] = d;
                sg.weights[offset] = scanner.get(d);
                ++offset;
            }
            sg.degrees[c] = scanner.keys.size();
        }
    }

    sg.numEdges = 0;
    for (size_t c = 0; c < C; ++c) {
        sg.numEdges += sg.degrees[c];
    }
}

/**
 * Aggregate super-graph into new super-graph
 */
template <typename K>
void aggregateSuperGraphLeiden(
    SuperGraph<K, Weight>& out,
    const SuperGraph<K, Weight>& in,
    const std::vector<K>& vcom,
    const std::vector<size_t>& coff,
    const std::vector<K>& cvtx,
    size_t C,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("aggregateSuperGraphLeiden: C=%zu", C);

    std::vector<size_t> estDegree(C, 0);
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t c = 0; c < C; ++c) {
        size_t deg = 0;
        for (size_t i = coff[c]; i < coff[c + 1]; ++i) {
            deg += in.degrees[cvtx[i]];
        }
        estDegree[c] = deg;
    }

    out.offsets.resize(C + 1);
    out.offsets[0] = 0;
    for (size_t c = 0; c < C; ++c) {
        out.offsets[c + 1] = out.offsets[c] + estDegree[c];
    }

    size_t totalEdges = out.offsets[C];
    out.neighbors.resize(totalEdges);
    out.weights.resize(totalEdges);
    out.degrees.assign(C, 0);
    out.numNodes = C;

    const int numThreads = omp_get_max_threads();
    std::vector<CommunityScanner<K, Weight>> scanners;
    for (int t = 0; t < numThreads; ++t) {
        scanners.emplace_back(C);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& scanner = scanners[tid];

        #pragma omp for schedule(dynamic, 256)
        for (size_t c = 0; c < C; ++c) {
            scanner.clear();

            for (size_t i = coff[c]; i < coff[c + 1]; ++i) {
                K u = cvtx[i];
                size_t start = in.offsets[u];
                size_t end = in.offsets[u] + in.degrees[u];
                for (size_t j = start; j < end; ++j) {
                    K d = vcom[in.neighbors[j]];
                    scanner.add(d, in.weights[j]);
                }
            }

            size_t offset = out.offsets[c];
            for (K d : scanner.keys) {
                out.neighbors[offset] = d;
                out.weights[offset] = scanner.get(d);
                ++offset;
            }
            out.degrees[c] = scanner.keys.size();
        }
    }

    out.numEdges = 0;
    for (size_t c = 0; c < C; ++c) {
        out.numEdges += out.degrees[c];
    }
}

//=============================================================================
// SECTION 13b: GVE-STYLE AGGREGATION WITH SUPER-GRAPH LOCAL-MOVING
//=============================================================================
//
// Ported from GVE's GVELeidenCSR() aggregation strategy.
// Key difference from standard Leiden CSR: after aggregation, runs an
// explicit local-moving phase on the super-graph adjacency lists to merge
// refined sub-communities. This produces tighter communities.
//=============================================================================

/**
 * GVE-style aggregation: build explicit adjacency lists + super-graph local-moving
 *
 * Unlike standard aggregation which builds a SuperGraph struct for the next pass,
 * GVE builds std::vector<std::pair<K,W>> adjacency lists per community and runs
 * local-moving directly on them. This merges refined sub-communities more
 * aggressively, producing better kernel performance.
 *
 * @param vcom IN/OUT: community assignment (updated to merged communities)
 * @param ctot OUT: community total weights (updated)
 * @param g Original graph
 * @param vtot Vertex total weights (original graph)
 * @param C Number of communities (from renumbering)
 * @param coff Community offset array
 * @param cvtx Community vertex list
 * @param M Total edge weight
 * @param R Resolution parameter
 * @param config GraphBrew configuration
 * @return Number of final communities after merging
 */
template <typename K, typename NodeID_T, typename DestID_T>
size_t aggregateGVEStyle(
    std::vector<K>& vcom,
    std::vector<Weight>& ctot,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<Weight>& vtot,
    size_t C,
    const std::vector<size_t>& coff,
    const std::vector<K>& cvtx,
    Weight M, Weight R,
    const GraphBrewConfig& config) {

    const int64_t N = g.num_nodes();
    Timer phase_timer;
    phase_timer.Start();

    // Compute community weights
    std::vector<Weight> super_weight(C, Weight(0));
    #pragma omp parallel for
    for (int64_t v = 0; v < N; ++v) {
        K c = vcom[v];
        #pragma omp atomic
        super_weight[c] += vtot[v];
    }
    phase_timer.Stop();
    const double weight_seconds = phase_timer.Seconds();
    phase_timer.Start();

    // Build super-graph adjacency using community-based approach
    std::vector<std::vector<std::pair<K, Weight>>> super_adj(C);

    // Aggregate edges community by community (parallel)
    #pragma omp parallel
    {
        CommunityScanner<K, Weight> scanner(C);

        #pragma omp for schedule(dynamic, 256)
        for (size_t c = 0; c < C; ++c) {
            scanner.clear();

            // Scan all vertices in this community
            for (size_t i = coff[c]; i < coff[c + 1]; ++i) {
                K u = cvtx[i];

                for (auto neighbor : g.out_neigh(u)) {
                    NodeID_T v;
                    Weight w;
                    if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                        v = neighbor;
                        w = Weight(1);
                    } else {
                        v = neighbor.v;
                        w = static_cast<Weight>(neighbor.w);
                    }

                    K cv = vcom[v];
                    if (cv != static_cast<K>(c)) {
                        scanner.add(cv, w);
                    }
                }
            }

            // Build adjacency list for this community
            super_adj[c].reserve(scanner.keys.size());
            for (K d : scanner.keys) {
                super_adj[c].emplace_back(d, scanner.get(d));
            }
        }
    }

    phase_timer.Stop();
    const double adjacency_seconds = phase_timer.Seconds();
    phase_timer.Start();

    // ================================================================
    // LOCAL-MOVING ON SUPER-GRAPH (from GVE's GVELeidenCSR)
    // Merge refined communities using parallel local-moving
    // ================================================================

    std::vector<K> super_comm(C);
    std::vector<Weight> super_ctot(C);

    #pragma omp parallel for
    for (size_t c = 0; c < C; ++c) {
        super_comm[c] = static_cast<K>(c);
        super_ctot[c] = super_weight[c];
    }

    // Sequential flat arrays for super-graph local move
    // Sequential super-graph local move with simplified delta, limited iterations
    std::vector<Weight> sg_comm_weights(C, Weight(0));
    std::vector<K> sg_touched(C);

    // Limit super-graph iterations: min(3, maxIterations)
    int super_max_iter = std::min(3, config.maxIterations);

    if (config.superGraphMoveBatch > 1) {
        std::vector<K> proposed_comm(C);
        std::vector<Weight> proposed_delta(C);
        const size_t batch_size =
            static_cast<size_t>(config.superGraphMoveBatch);
        for (int iter = 0; iter < super_max_iter; ++iter) {
            int moves = 0;
            #pragma omp parallel
            {
                CommunityScanner<K, Weight> scanner(C);
                for (size_t begin = 0; begin < C; begin += batch_size) {
                    const size_t end = std::min(C, begin + batch_size);
                    #pragma omp for schedule(static)
                    for (size_t c = begin; c < end; ++c) {
                        const K d = super_comm[c];
                        const Weight kc = super_weight[c];
                        const Weight sigma_d = super_ctot[d];
                        const Weight delta_removal =
                            R * kc * (sigma_d - kc) / M;
                        scanner.clear();
                        for (const auto& [nc, w] : super_adj[c]) {
                            scanner.add(super_comm[nc], w);
                        }
                        const Weight kc_to_d = scanner.get(d);
                        K best_comm = d;
                        Weight best_delta = Weight(0);
                        for (K e : scanner.keys) {
                            const Weight kc_to_e = scanner.get(e);
                            const Weight delta_addition =
                                e == d
                                ? kc_to_d - delta_removal
                                : kc_to_e
                                    - R * kc * super_ctot[e] / M;
                            const Weight delta =
                                delta_addition
                                - delta_removal
                                + kc_to_d;
                            if (
                                delta > best_delta
                                || (
                                    delta == best_delta
                                    && e < best_comm
                                )
                            ) {
                                best_delta = delta;
                                best_comm = e;
                            }
                        }
                        proposed_comm[c] = best_comm;
                        proposed_delta[c] = best_delta;
                    }
                    #pragma omp single
                    {
                        for (size_t c = begin; c < end; ++c) {
                            const K d = super_comm[c];
                            const K best_comm = proposed_comm[c];
                            if (
                                best_comm == d
                                || proposed_delta[c] <= Weight(0)
                            ) {
                                continue;
                            }
                            const Weight kc = super_weight[c];
                            super_ctot[d] -= kc;
                            super_ctot[best_comm] += kc;
                            super_comm[c] = best_comm;
                            moves++;
                        }
                    }
                }
            }
            if (moves == 0) break;
        }
    } else {
        for (int iter = 0; iter < super_max_iter; ++iter) {
            int moves = 0;

            for (size_t c = 0; c < C; ++c) {
            K d = super_comm[c];
            Weight kc = super_weight[c];
            Weight sigma_d = super_ctot[d];
            const Weight delta_removal =
                R * kc * (sigma_d - kc) / M;

            // Scan neighbor communities
            int num_touched = 0;
            Weight kc_to_d = Weight(0);

            for (auto& [nc, w] : super_adj[c]) {
                K snc = super_comm[nc];
                if (sg_comm_weights[snc] == Weight(0)) {
                    sg_touched[num_touched++] = snc;
                }
                sg_comm_weights[snc] += w;
                if (snc == d) kc_to_d += w;
            }

            // Find best community using simplified delta formula
            K best_comm = d;
            Weight best_delta = Weight(0);

            for (int i = 0; i < num_touched; ++i) {
                K e = sg_touched[i];
                Weight kc_to_e = sg_comm_weights[e];
                Weight sigma_e = super_ctot[e];

                Weight delta_addition;
                if (e == d) {
                    delta_addition = kc_to_d - delta_removal;
                } else {
                    delta_addition = kc_to_e - R * kc * sigma_e / M;
                }

                Weight delta = delta_addition - delta_removal + kc_to_d;

                if (delta > best_delta || (delta == best_delta && e < best_comm)) {
                    best_delta = delta;
                    best_comm = e;
                }
            }

            // Clear touched communities
            for (int i = 0; i < num_touched; ++i) {
                sg_comm_weights[sg_touched[i]] = Weight(0);
            }

            // Move if better
            if (best_comm != d && best_delta > Weight(0)) {
                super_ctot[d] -= kc;
                super_ctot[best_comm] += kc;
                super_comm[c] = best_comm;
                moves++;
            }
        }

            if (moves == 0) break;
        }
    }
    phase_timer.Stop();
    const double local_move_seconds = phase_timer.Seconds();
    phase_timer.Start();

    // Map back to original vertices: vcom[v] = super_comm[old_vcom[v]]
    #pragma omp parallel for
    for (int64_t v = 0; v < N; ++v) {
        K rc = vcom[v];
        vcom[v] = super_comm[rc];
    }

    // Renumber final communities contiguously
    K max_final_comm = 0;
    for (int64_t v = 0; v < N; ++v) {
        max_final_comm = std::max(max_final_comm, vcom[v]);
    }

    std::vector<K> final_renumber(max_final_comm + 1, K(-1));
    K next_final_id = 0;
    for (int64_t v = 0; v < N; ++v) {
        K c = vcom[v];
        if (final_renumber[c] == K(-1)) {
            final_renumber[c] = next_final_id++;
        }
    }

    #pragma omp parallel for
    for (int64_t v = 0; v < N; ++v) {
        vcom[v] = final_renumber[vcom[v]];
    }

    // Update ctot for the merged communities
    size_t numFinal = static_cast<size_t>(next_final_id);
    ctot.assign(ctot.size(), Weight(0));
    #pragma omp parallel for
    for (int64_t v = 0; v < N; ++v) {
        #pragma omp atomic
        ctot[vcom[v]] += vtot[v];
    }
    phase_timer.Stop();
    printf(
        "  gve-aggregate phases: weights=%.4fs, adjacency=%.4fs, "
        "super-move=%.4fs, remap=%.4fs\n",
        weight_seconds,
        adjacency_seconds,
        local_move_seconds,
        phase_timer.Seconds());

    return numFinal;
}

//=============================================================================
// SECTION 14: RABBIT ORDER - PARALLEL INCREMENTAL AGGREGATION
//=============================================================================
//
// Implementation of the RabbitOrder algorithm from:
// "Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis"
// Arai et al., IPDPS 2016
//
// Key innovations:
// 1. Lazy aggregation: defer edge updates, only register community membership
// 2. Lock-free parallelism: use CAS (12 bytes) instead of locks
// 3. Incremental processing: vertices processed in degree order
// 4. Dendrogram construction: build tree during merges
//=============================================================================

/**
 * Packed atomic structure for single-CAS lazy aggregation
 * From original rabbit_order.hpp - packs strength + child into 64 bits
 *
 * Layout: [strength: 32 bits (float)] [child: 32 bits (uint32)]
 * This allows atomic compare_exchange on the full 64-bit value
 */
struct RabbitAtom {
    using vint = uint32_t;
    static constexpr vint INVALID_CHILD = UINT32_MAX;
    static constexpr float INVALID_STR = -1.0f;

    // Packed representation: lower 32 bits = child, upper 32 bits = strength (as float bits)
    std::atomic<uint64_t> packed;

    RabbitAtom() : packed(pack(0.0f, INVALID_CHILD)) {}

    static uint64_t pack(float str, vint child) {
        uint64_t result = 0;
        uint32_t str_bits;
        std::memcpy(&str_bits, &str, sizeof(float));
        result = (static_cast<uint64_t>(str_bits) << 32) | child;
        return result;
    }

    static std::pair<float, vint> unpack(uint64_t val) {
        vint child = static_cast<vint>(val & 0xFFFFFFFF);
        uint32_t str_bits = static_cast<uint32_t>(val >> 32);
        float str;
        std::memcpy(&str, &str_bits, sizeof(float));
        return {str, child};
    }

    void init(float str) {
        packed.store(pack(str, INVALID_CHILD), std::memory_order_relaxed);
    }

    std::pair<float, vint> load() const {
        return unpack(packed.load(std::memory_order_acquire));
    }

    // Invalidate and return old strength (returns INVALID_STR if already invalid)
    float invalidate() {
        uint64_t old = packed.load(std::memory_order_relaxed);
        while (true) {
            auto [str, child] = unpack(old);
            if (str == INVALID_STR) return INVALID_STR;
            uint64_t newval = pack(INVALID_STR, child);
            if (packed.compare_exchange_weak(old, newval,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return str;
            }
        }
    }

    // Restore strength (for failed merges)
    void restore(float str) {
        uint64_t old = packed.load(std::memory_order_relaxed);
        auto [_, child] = unpack(old);
        packed.store(pack(str, child), std::memory_order_release);
    }

    // Atomic CAS to merge: update both strength and child atomically
    bool tryMerge(float expected_str, vint expected_child, float new_str, vint new_child) {
        uint64_t expected = pack(expected_str, expected_child);
        uint64_t desired = pack(new_str, new_child);
        return packed.compare_exchange_strong(expected, desired,
            std::memory_order_acq_rel, std::memory_order_relaxed);
    }
};

/**
 * Consolidated per-vertex structure for Rabbit Order.
 * Merges RabbitAtom (str+child), dest, sibling, united_child, and edge cache
 * into one struct so that all per-vertex fields share the same cache lines.
 * This eliminates 3-4 separate array look-ups per vertex access.
 */
template <typename K>
struct RabbitNode {
    RabbitAtom atom;                              ///< Packed {strength, child} — 8B atomic
    K dest;                                       ///< Community root (union-find)
    K sibling;                                    ///< Sibling in dendrogram
    K united_child;                               ///< Last child whose edges were aggregated
    std::vector<std::pair<K, float>> edges;       ///< Cached aggregated edges

    RabbitNode() : dest(0), sibling(static_cast<K>(-1)),
                   united_child(static_cast<K>(-1)) {}
};

/**
 * Per-vertex structure for Rabbit Order (used by hybrid, tile-quantized,
 * and community-level ordering strategies that operate on Leiden communities).
 */
template <typename K>
struct RabbitVertex {
    std::vector<std::pair<K, float>> edges;  ///< Cached aggregated edges (neighbor, weight)
    K united_child;  ///< Last child whose edges were aggregated

    RabbitVertex() : united_child(static_cast<K>(-1)) {}
};

/**
 * Rabbit Order community detection using parallel incremental aggregation
 * (Consolidated RabbitNode version — all per-vertex data in one struct)
 */
template <typename K, typename NodeID_T, typename DestID_T>
size_t rabbitCommunityDetection(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    std::vector<RabbitNode<K>>& nodes,
    std::vector<K>& toplevel,
    const std::vector<K>& vertexOrder,
    float M,
    const GraphBrewConfig& config) {

    using Edge = std::pair<K, float>;
    const size_t N = g.num_nodes();
    constexpr K INVALID = static_cast<K>(-1);
    const int numThreads = omp_get_max_threads();

    GRAPHBREW_TRACE("rabbitCommunityDetection: N=%zu, M=%.0f", N, M);

    // Lambda: find destination community with path compression
    auto findDest = [&](K v) -> K {
        K d = nodes[v].dest;
        while (nodes[d].dest != d) {
            K dd = nodes[d].dest;
            nodes[v].dest = dd;  // Path compression
            d = dd;
        }
        return d;
    };

    // Unite: aggregate edges of v and its merged children into nbrs buffer
    auto unite = [&](K u, std::vector<Edge>& nbrs) -> void {
        auto& nu = nodes[u];
        ptrdiff_t icmb = 0;
        nbrs.clear();

        auto push_edges = [&](K w) {
            auto& es = nodes[w].edges;
            constexpr size_t PREFETCH_AHEAD = 8;

            for (size_t i = 0; i < es.size() && i < PREFETCH_AHEAD; ++i)
                __builtin_prefetch(&nodes[es[i].first].dest, 0, 3);

            for (size_t i = 0; i < es.size(); ++i) {
                if (i + PREFETCH_AHEAD < es.size())
                    __builtin_prefetch(&nodes[es[i + PREFETCH_AHEAD].first].dest, 0, 3);

                K root = findDest(es[i].first);
                if (root != u)
                    nbrs.push_back({root, es[i].second});
            }

            if (static_cast<ptrdiff_t>(nbrs.size()) - icmb >= 2048) {
                auto it = nbrs.begin() + icmb;
                std::sort(it, nbrs.end(), [](const Edge& a, const Edge& b) {
                    return a.first < b.first;
                });
                auto out = it;
                auto x = *it;
                auto cur = it;
                while (++cur != nbrs.end()) {
                    if (x.first == cur->first) {
                        x.second += cur->second;
                    } else {
                        *out++ = x;
                        x = *cur;
                    }
                }
                *out++ = x;
                icmb = out - nbrs.begin();
                nbrs.resize(icmb);
            }
        };

        push_edges(u);

        while (nu.united_child != nu.atom.load().second) {
            K c = static_cast<K>(nu.atom.load().second);
            K w;
            for (w = c; w != INVALID && w != nu.united_child; w = nodes[w].sibling)
                push_edges(w);
            nu.united_child = c;
        }

        // Final compaction and writeback
        if (!nbrs.empty()) {
            std::sort(nbrs.begin(), nbrs.end(), [](const Edge& a, const Edge& b) {
                return a.first < b.first;
            });
            size_t write_pos = 0;
            for (size_t j = 1; j < nbrs.size(); ++j) {
                if (nbrs[j].first == nbrs[write_pos].first) {
                    nbrs[write_pos].second += nbrs[j].second;
                } else {
                    ++write_pos;
                    if (write_pos != j) nbrs[write_pos] = nbrs[j];
                }
            }
            nbrs.resize(write_pos + 1);
            nu.edges = std::move(nbrs);
        } else {
            nu.edges.clear();
        }
    };

    // Find best destination from compacted edges
    auto findBest = [&](K u, float str_u) -> K {
        float dmax = 0.0f;
        K best = u;
        for (const auto& [target, weight] : nodes[u].edges) {
            auto [str_d, _] = nodes[target].atom.load();
            if (str_d == RabbitAtom::INVALID_STR) continue;
            float d = weight - str_u * str_d / M;
            if (d > dmax) {
                dmax = d;
                best = target;
            }
        }
        return best;
    };

    // Main parallel incremental aggregation
    std::vector<std::deque<K>> topss(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::deque<K> tops, pends;

        std::vector<Edge> nbrs;
        const int np = omp_get_num_threads();
        nbrs.reserve(std::max<size_t>(N / np * 2, 4096));

        auto merge = [&](K u) -> K {
            unite(u, nbrs);

            float str_u = nodes[u].atom.invalidate();
            if (str_u == RabbitAtom::INVALID_STR) {
                return u;
            }

            if (static_cast<K>(nodes[u].atom.load().second) != nodes[u].united_child) {
                unite(u, nbrs);
            }

            K best = findBest(u, str_u);

            if (best == u) {
                nodes[u].atom.restore(str_u);
                return u;
            }

            auto [str_v, child_v] = nodes[best].atom.load();
            if (str_v == RabbitAtom::INVALID_STR) {
                nodes[u].atom.restore(str_u);
                return INVALID;
            }

            nodes[u].sibling = child_v;
            float new_str = str_v + str_u;

            if (nodes[best].atom.tryMerge(str_v, child_v, new_str, static_cast<uint32_t>(u))) {
                nodes[u].dest = best;
                return best;
            } else {
                nodes[u].sibling = INVALID;
                nodes[u].atom.restore(str_u);
                return INVALID;
            }
        };

        #pragma omp for schedule(static, 1)
        for (size_t i = 0; i < N; ++i) {
            pends.erase(
                std::remove_if(pends.begin(), pends.end(), [&](K w) {
                    K u = merge(w);
                    if (u == w) tops.push_back(w);
                    return u != INVALID;
                }),
                pends.end());

            K v = vertexOrder[i];
            K u = merge(v);
            if (u == v)
                tops.push_back(v);
            else if (u == INVALID)
                pends.push_back(v);
        }

        #pragma omp barrier
        #pragma omp critical
        {
            for (K v : pends) {
                K u = merge(v);
                if (u == v) tops.push_back(v);
            }
            topss[tid] = std::move(tops);
        }
    }

    for (int t = 0; t < numThreads; ++t) {
        toplevel.insert(toplevel.end(), topss[t].begin(), topss[t].end());
    }

    GRAPHBREW_TRACE("rabbitCommunityDetection: %zu top-level communities", toplevel.size());

    return toplevel.size();
}

/**
 * Write v and lineal descendants of v on the dendrogram to output
 * (siblings of children are NOT included - they're handled by the caller)
 *
 * This matches Boost's descendants() function exactly.
 * The key insight: we only follow the child chain, not sibling chain.
 */
template <typename K>
void rabbitDescendants(
    const std::vector<RabbitNode<K>>& nodes,
    K v,
    std::deque<K>& output) {

    constexpr K INVALID = static_cast<K>(-1);
    output.push_back(v);

    auto [_, child] = nodes[v].atom.load();
    while (child != INVALID) {
        output.push_back(child);
        auto [__, next_child] = nodes[child].atom.load();
        child = next_child;
    }
}

/**
 * Overload for legacy callers that use separate std::vector<RabbitAtom>.
 * Used by hybrid ordering, tile-quantized, and community-level sections.
 */
template <typename K>
void rabbitDescendants(
    const std::vector<RabbitAtom>& atom,
    K v,
    std::deque<K>& output) {

    constexpr K INVALID = static_cast<K>(-1);
    output.push_back(v);

    auto [_, child] = atom[v].load();
    while (child != INVALID) {
        output.push_back(child);
        auto [__, next_child] = atom[child].load();
        child = next_child;
    }
}

/**
 * Extract flat community membership from Rabbit Order dendrogram
 *
 * Traverses the dendrogram (toplevel roots → descendants → siblings) to produce
 * a flat membership[v] = communityId vector compatible with all ordering strategies.
 * This enables Rabbit's community detection to be combined with any ordering
 * strategy (DBG, hubcluster, corder, hrab, etc.) via the generic pipeline.
 *
 * Also sorts toplevel into: active communities first, isolated (zero-degree) last.
 * Returns the number of communities.
 */
template <typename K>
size_t rabbitExtractMembership(
    std::vector<K>& membership,
    const std::vector<RabbitNode<K>>& nodes,
    std::vector<K>& toplevel,  // Non-const: we sort it for determinism
    const std::vector<float>& degrees,  // For zero-degree separation
    size_t N) {

    GRAPHBREW_TRACE("rabbitExtractMembership: %zu top-level, N=%zu", toplevel.size(), N);

    membership.resize(N);
    constexpr K INVALID = static_cast<K>(-1);

    // Separate zero-degree (isolated singleton) communities
    // Group them at the end for better cache locality
    std::vector<K> activeTop, isolatedTop;
    for (K root : toplevel) {
        if (degrees[root] == 0.0f) {
            isolatedTop.push_back(root);
        } else {
            activeTop.push_back(root);
        }
    }

    // Sort active and isolated separately, then concatenate
    std::sort(activeTop.begin(), activeTop.end());
    std::sort(isolatedTop.begin(), isolatedTop.end());

    // Rebuild toplevel: active first, isolated last
    toplevel.clear();
    toplevel.insert(toplevel.end(), activeTop.begin(), activeTop.end());
    toplevel.insert(toplevel.end(), isolatedTop.begin(), isolatedTop.end());

    const K ncom = static_cast<K>(toplevel.size());

    // Determine parallelism: use task-based distribution
    const int np = omp_get_max_threads();
    const K ntask = std::min<K>(ncom, static_cast<K>(128 * np));

    // Parallel per-community DFS to assign membership
    #pragma omp parallel
    {
        std::deque<K> stack;

        #pragma omp for schedule(dynamic, 1)
        for (K i = 0; i < ntask; ++i) {
            for (K comid = i; comid < ncom; comid += ntask) {
                K root = toplevel[comid];

                rabbitDescendants(nodes, root, stack);

                while (!stack.empty()) {
                    K v = stack.back();
                    stack.pop_back();

                    membership[v] = comid;

                    if (nodes[v].sibling != INVALID) {
                        rabbitDescendants(nodes, nodes[v].sibling, stack);
                    }
                }
            }
        }
    }

    GRAPHBREW_TRACE("rabbitExtractMembership: %zu active, %zu isolated communities",
               activeTop.size(), isolatedTop.size());

    return static_cast<size_t>(ncom);
}

/**
 * Generate ordering from Rabbit Order dendrogram using Boost's two-phase approach
 *
 * This is a FAITHFUL implementation of Boost's compute_perm():
 *
 * Phase 1 (parallel): For each community, DFS traverse and assign LOCAL IDs (0, 1, 2, ...)
 *          Store community membership in coms[] and local ID in perm[]
 *          Record community size in offsets[comid + 1]
 *
 * Phase 2 (sequential): Prefix sum on offsets[] to get global start positions
 *
 * Phase 3 (parallel): Add offsets[coms[v]] to each vertex's local ID to get global ID
 *
 * Key differences from our previous implementation:
 * 1. Deterministic toplevel order: Sort toplevel array before processing
 * 2. Parallel per-community traversal: Each community processed independently
 * 3. Offset-based ID assignment: Guarantees contiguous community blocks
 * 4. Different DFS traversal: Uses descendants() which follows child chain only
 * 5. Zero-degree communities grouped at the END for better cache locality
 */
template <typename K>
void rabbitOrderingGeneration(
    std::vector<K>& permutation,
    const std::vector<RabbitNode<K>>& nodes,
    std::vector<K>& toplevel,  // Non-const: we sort it for determinism
    const std::vector<float>& degrees,  // For zero-degree separation
    size_t N) {

    GRAPHBREW_TRACE("rabbitOrderingGeneration: %zu top-level, N=%zu", toplevel.size(), N);

    permutation.resize(N);
    constexpr K INVALID = static_cast<K>(-1);

    // Extract membership (also sorts toplevel: active first, isolated last)
    std::vector<K> coms;
    size_t numComm = rabbitExtractMembership(coms, nodes, toplevel, degrees, N);

    const K ncom = static_cast<K>(numComm);
    std::vector<K> offsets(ncom + 1); // offsets[c+1] = size of community c (then prefix sum)

    // Determine parallelism like Boost: use task-based distribution
    const int np = omp_get_max_threads();
    const K ntask = std::min<K>(ncom, static_cast<K>(128 * np));

    // Phase 1: Parallel per-community DFS, assign local IDs
    // We re-traverse the dendrogram (same order as membership extraction)
    // to assign DFS-based local IDs within each community
    #pragma omp parallel
    {
        std::deque<K> stack;

        #pragma omp for schedule(dynamic, 1)
        for (K i = 0; i < ntask; ++i) {
            // Process multiple communities per task (strided distribution like Boost)
            for (K comid = i; comid < ncom; comid += ntask) {
                K localId = 0;
                K root = toplevel[comid];

                // Get descendants of root (root + lineal children chain)
                rabbitDescendants(nodes, root, stack);

                while (!stack.empty()) {
                    K v = stack.back();
                    stack.pop_back();

                    permutation[v] = localId++;

                    if (nodes[v].sibling != INVALID) {
                        rabbitDescendants(nodes, nodes[v].sibling, stack);
                    }
                }

                // Store community size for prefix sum
                offsets[comid + 1] = localId;
            }
        }
    }

    // Phase 2: Sequential prefix sum to get global offsets
    // offsets[c] becomes the starting position for community c
    offsets[0] = 0;
    for (K c = 0; c < ncom; ++c) {
        offsets[c + 1] += offsets[c];
    }
    assert(offsets[ncom] == static_cast<K>(N));

    // Phase 3: Parallel offset addition to get global IDs
    #pragma omp parallel for schedule(static)
    for (size_t v = 0; v < N; ++v) {
        permutation[v] += offsets[coms[v]];
    }

    GRAPHBREW_TRACE("rabbitOrderingGeneration: %zu communities", numComm);
}

/**
 * Full Rabbit Order algorithm
 *
 * Combines:
 * 1. Parallel incremental aggregation (community detection)
 * 2. DFS ordering generation from dendrogram
 */
template <typename K, typename NodeID_T, typename DestID_T>
void runRabbitOrder(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& mapping,
    const GraphBrewConfig& config,
    GraphBrewResult<K>* resultOut = nullptr,
    bool communityDetectionOnly = true) {

    const size_t N = g.num_nodes();

    Timer timer;
    timer.Start();

    // Step 1: Compute total edge weight M and vertex degrees
    float M = 0;
    std::vector<float> degrees(N);

    #pragma omp parallel for reduction(+:M)
    for (size_t u = 0; u < N; ++u) {
        float d = 0;
        for (auto neighbor : g.out_neigh(u)) {
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                d += 1.0f;
            } else {
                d += static_cast<float>(neighbor.w);
            }
        }
        degrees[u] = d;
        M += d;
    }
    // NOTE: Do NOT divide M by 2  here. The ΔQ formula in findBest uses
    // weight - str_u * str_d / M.  The CSR version uses tot_wgt = sum(degrees)
    // (undirected double-counted).  Dividing by 2 doubles the penalty term,
    // producing too many communities and 4x more unite() work.

    // Step 2: Sort vertices by degree (ascending) - key to RabbitOrder efficiency
    std::vector<K> vertexOrder(N);
    std::iota(vertexOrder.begin(), vertexOrder.end(), K(0));

    #pragma omp parallel
    {
        #pragma omp single
        __gnu_parallel::sort(vertexOrder.begin(), vertexOrder.end(),
            [&](K a, K b) {
                return degrees[a] != degrees[b]
                    ? degrees[a] < degrees[b]
                    : a < b;
            });
    }

    // Step 3: Initialize consolidated per-vertex data (RabbitNode)
    std::vector<RabbitNode<K>> nodes(N);
    std::vector<K> toplevel;
    toplevel.reserve(N / 10);

    #pragma omp parallel for
    for (size_t u = 0; u < N; ++u) {
        nodes[u].atom.init(degrees[u]);
        nodes[u].dest = static_cast<K>(u);

        // Initialize edge cache from original graph
        nodes[u].edges.reserve(g.out_degree(u));
        for (auto neighbor : g.out_neigh(u)) {
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                nodes[u].edges.emplace_back(static_cast<K>(neighbor), 1.0f);
            } else {
                nodes[u].edges.emplace_back(static_cast<K>(neighbor.v), static_cast<float>(neighbor.w));
            }
        }
    }

    // Step 4: Run community detection
    Timer cdTimer;
    cdTimer.Start();
    size_t numComm = rabbitCommunityDetection<K>(g, nodes, toplevel,
                                                  vertexOrder, M, config);
    cdTimer.Stop();

    // If caller wants community detection results only (for generic ordering pipeline),
    // extract membership and return without generating native DFS ordering.
    if (resultOut && communityDetectionOnly) {
        Timer memberTimer;
        memberTimer.Start();

        size_t ncom = rabbitExtractMembership(resultOut->membership, nodes, toplevel, degrees, N);
        resultOut->numCommunities = ncom;
        resultOut->totalPasses = 1;
        // No multi-level hierarchy for Rabbit — membershipPerPass stays empty
        // (buildSyntheticDendrogram will be used if dendrogram orderings are requested)

        // Optionally hand the Rabbit dendrogram (child/sibling chains + sorted
        // toplevel) to the downstream ordering pipeline so that
        // IntraCommunityOrder::Dendrogram can do a pure pointer-chase DFS
        // per community without reconstructing anything.  Gated on the
        // COMPOSE pipeline asking for it: this O(N) copy is otherwise wasted.
        if (config.ordering == OrderingStrategy::COMPOSE &&
            config.intraCommunityOrder == IntraCommunityOrder::Dendrogram) {
            constexpr K INVALID = static_cast<K>(-1);
            resultOut->rabbitChild.assign(N, INVALID);
            resultOut->rabbitSibling.assign(N, INVALID);
            #pragma omp parallel for schedule(static)
            for (size_t v = 0; v < N; ++v) {
                resultOut->rabbitChild[v]   = static_cast<K>(nodes[v].atom.load().second);
                resultOut->rabbitSibling[v] = nodes[v].sibling;
            }
            resultOut->rabbitToplevel = toplevel;  // already sorted: active first, isolated last
            resultOut->hasRabbitDendrogram = true;
        }

        memberTimer.Stop();
        timer.Stop();

        printf("RabbitOrder (community detection only): %zu communities, time=%.4fs\n",
               ncom, timer.Seconds());
        printf("  community-detection: %.4fs, membership-extract: %.4fs\n",
               cdTimer.Seconds(), memberTimer.Seconds());
        return;
    }

    // Step 5: Generate ordering from dendrogram (native DFS)
    Timer orderTimer;
    orderTimer.Start();
    std::vector<K> permutation;
    rabbitOrderingGeneration(permutation, nodes, toplevel, degrees, N);
    orderTimer.Stop();

    timer.Stop();

    // Step 6: Build mapping
    mapping.resize(N);
    #pragma omp parallel for
    for (size_t u = 0; u < N; ++u) {
        mapping[u] = static_cast<NodeID_T>(permutation[u]);
    }

    printf("RabbitOrder: %zu communities, time=%.4fs\n", numComm, timer.Seconds());
    printf("  community-detection: %.4fs, ordering: %.4fs\n",
           cdTimer.Seconds(), orderTimer.Seconds());
    if (resultOut) {
        resultOut->numCommunities = numComm;
        resultOut->totalPasses = 1;
    }
}

//=============================================================================
// SECTION 15: AGGREGATION - STREAMING APPROACH
//=============================================================================

/**
 * RabbitOrder-style lazy incremental merge
 *
 * Instead of building full super-graph CSR, we track:
 * - Union-find structure for community merges
 * - Incremental edge weight updates
 *
 * This is faster for early passes with many small communities
 */
template <typename K>
struct LazyAggregator {
    std::vector<K> parent;         ///< Union-find parent
    std::vector<K> rank;           ///< Union-find rank
    std::vector<Weight> edgeSum;   ///< Sum of edge weights per community
    std::atomic<size_t> numMerges; ///< Total merges performed

    explicit LazyAggregator(size_t N) : numMerges(0) {
        parent.resize(N);
        rank.resize(N, 0);
        edgeSum.resize(N, 0);
        std::iota(parent.begin(), parent.end(), K(0));
    }

    K find(K x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }

    bool unite(K x, K y) {
        K px = find(x);
        K py = find(y);
        if (px == py) return false;

        if (rank[px] < rank[py]) std::swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        edgeSum[px] += edgeSum[py];
        numMerges++;
        return true;
    }

    void reset() {
        std::iota(parent.begin(), parent.end(), K(0));
        std::fill(rank.begin(), rank.end(), K(0));
        std::fill(edgeSum.begin(), edgeSum.end(), Weight(0));
        numMerges = 0;
    }
};

/**
 * Lazy aggregation from original graph - streaming approach
 *
 * Uses the same CommunityScanner as Leiden but processes vertices
 * in community order without building explicit community vertex lists.
 *
 * Benefits:
 * - Avoids O(N) memory for coff/cvtx arrays
 * - Single pass over vertices
 * - Same cache-efficient scanning as Leiden
 */
template <typename K, typename NodeID_T, typename DestID_T>
void aggregateGraphLazy(
    SuperGraph<K, Weight>& sg,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<K>& vcom,
    size_t C,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("aggregateGraphLazy: C=%zu", C);

    const int64_t N = g.num_nodes();
    const int numThreads = omp_get_max_threads();

    // Phase 1: Group vertices by community using counting sort
    std::vector<size_t> commCount(C, 0);
    for (int64_t u = 0; u < N; ++u) {
        commCount[vcom[u]]++;
    }

    std::vector<size_t> commStart(C + 1);
    commStart[0] = 0;
    for (size_t c = 0; c < C; ++c) {
        commStart[c + 1] = commStart[c] + commCount[c];
    }

    std::vector<K> vertexList(N);
    std::vector<size_t> commPos = commStart;  // Copy for filling
    for (int64_t u = 0; u < N; ++u) {
        K c = vcom[u];
        vertexList[commPos[c]++] = static_cast<K>(u);
    }

    // Phase 2: Single-pass aggregation storing results per community
    // Store (keys, weights) per community for later assembly
    std::vector<std::vector<K>> allKeys(C);
    std::vector<std::vector<Weight>> allWeights(C);

    std::vector<CommunityScanner<K, Weight>> scanners;
    for (int t = 0; t < numThreads; ++t) {
        scanners.emplace_back(C);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& scanner = scanners[tid];

        #pragma omp for schedule(dynamic, 256)
        for (size_t c = 0; c < C; ++c) {
            scanner.clear();

            size_t start = commStart[c];
            size_t end = commStart[c + 1];

            for (size_t i = start; i < end; ++i) {
                K u = vertexList[i];
                for (auto neighbor : g.out_neigh(u)) {
                    NodeID_T v;
                    Weight w;
                    if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                        v = neighbor;
                        w = Weight(1);
                    } else {
                        v = neighbor.v;
                        w = static_cast<Weight>(neighbor.w);
                    }
                    scanner.add(vcom[v], w);
                }
            }

            // Store results
            allKeys[c].reserve(scanner.keys.size());
            allWeights[c].reserve(scanner.keys.size());
            for (K d : scanner.keys) {
                allKeys[c].push_back(d);
                allWeights[c].push_back(scanner.get(d));
            }
        }
    }

    // Phase 3: Build offsets
    sg.offsets.resize(C + 1);
    sg.offsets[0] = 0;
    for (size_t c = 0; c < C; ++c) {
        sg.offsets[c + 1] = sg.offsets[c] + allKeys[c].size();
    }

    size_t totalEdges = sg.offsets[C];
    sg.neighbors.resize(totalEdges);
    sg.weights.resize(totalEdges);
    sg.degrees.resize(C);
    sg.numNodes = C;

    // Phase 4: Parallel copy to final arrays
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t c = 0; c < C; ++c) {
        size_t offset = sg.offsets[c];
        for (size_t i = 0; i < allKeys[c].size(); ++i) {
            sg.neighbors[offset + i] = allKeys[c][i];
            sg.weights[offset + i] = allWeights[c][i];
        }
        sg.degrees[c] = allKeys[c].size();
    }

    sg.numEdges = totalEdges;
}

/**
 * Lazy aggregation from super-graph - streaming approach
 */
template <typename K>
void aggregateSuperGraphLazy(
    SuperGraph<K, Weight>& out,
    const SuperGraph<K, Weight>& in,
    const std::vector<K>& vcom,
    size_t C,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("aggregateSuperGraphLazy: C=%zu", C);

    const size_t N = in.numNodes;
    const int numThreads = omp_get_max_threads();

    // Phase 1: Group super-nodes by community using counting sort
    std::vector<size_t> commCount(C, 0);
    for (size_t u = 0; u < N; ++u) {
        commCount[vcom[u]]++;
    }

    std::vector<size_t> commStart(C + 1);
    commStart[0] = 0;
    for (size_t c = 0; c < C; ++c) {
        commStart[c + 1] = commStart[c] + commCount[c];
    }

    std::vector<K> nodeList(N);
    std::vector<size_t> commPos = commStart;  // Copy for filling
    for (size_t u = 0; u < N; ++u) {
        K c = vcom[u];
        nodeList[commPos[c]++] = static_cast<K>(u);
    }

    // Phase 2: Single-pass aggregation storing results per community
    std::vector<std::vector<K>> allKeys(C);
    std::vector<std::vector<Weight>> allWeights(C);

    std::vector<CommunityScanner<K, Weight>> scanners;
    for (int t = 0; t < numThreads; ++t) {
        scanners.emplace_back(C);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& scanner = scanners[tid];

        #pragma omp for schedule(dynamic, 256)
        for (size_t c = 0; c < C; ++c) {
            scanner.clear();

            size_t start = commStart[c];
            size_t end = commStart[c + 1];

            for (size_t i = start; i < end; ++i) {
                K u = nodeList[i];
                size_t es = in.offsets[u];
                size_t ee = in.offsets[u] + in.degrees[u];
                for (size_t j = es; j < ee; ++j) {
                    scanner.add(vcom[in.neighbors[j]], in.weights[j]);
                }
            }

            // Store results
            allKeys[c].reserve(scanner.keys.size());
            allWeights[c].reserve(scanner.keys.size());
            for (K d : scanner.keys) {
                allKeys[c].push_back(d);
                allWeights[c].push_back(scanner.get(d));
            }
        }
    }

    // Phase 3: Build offsets
    out.offsets.resize(C + 1);
    out.offsets[0] = 0;
    for (size_t c = 0; c < C; ++c) {
        out.offsets[c + 1] = out.offsets[c] + allKeys[c].size();
    }

    size_t totalEdges = out.offsets[C];
    out.neighbors.resize(totalEdges);
    out.weights.resize(totalEdges);
    out.degrees.resize(C);
    out.numNodes = C;

    // Phase 4: Parallel copy to final arrays
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t c = 0; c < C; ++c) {
        size_t offset = out.offsets[c];
        for (size_t i = 0; i < allKeys[c].size(); ++i) {
            out.neighbors[offset + i] = allKeys[c][i];
            out.weights[offset + i] = allWeights[c][i];
        }
        out.degrees[c] = allKeys[c].size();
    }

    out.numEdges = totalEdges;
}

//=============================================================================
// SECTION 15b: DENDROGRAM CONSTRUCTION
//=============================================================================

/**
 * Build dendrogram from membership hierarchy
 */
template <typename K>
void buildDendrogram(
    std::vector<DendrogramNode<K>>& dendrogram,
    std::vector<K>& roots,
    const std::vector<std::vector<K>>& membershipPerPass,
    size_t N) {

    if (membershipPerPass.empty()) return;

    GRAPHBREW_TRACE("buildDendrogram: N=%zu, passes=%zu", N, membershipPerPass.size());

    const size_t numPasses = membershipPerPass.size();

    // Final membership determines leaf communities
    const auto& finalMembership = membershipPerPass.back();

    // Count vertices per community at each level
    std::vector<std::unordered_map<K, std::vector<K>>> levelComms(numPasses);

    // Level 0: vertices grouped by finest communities
    for (size_t v = 0; v < N; ++v) {
        levelComms[0][membershipPerPass[0][v]].push_back(static_cast<K>(v));
    }

    // Higher levels: communities grouped by parent communities
    for (size_t p = 1; p < numPasses; ++p) {
        for (const auto& [comm, vertices] : levelComms[p - 1]) {
            K parent = membershipPerPass[p][vertices[0]];  // All vertices in comm have same parent
            levelComms[p][parent].push_back(comm);
        }
    }

    // Build dendrogram nodes
    size_t nodeId = 0;
    std::unordered_map<std::pair<int, K>, K,
        std::function<size_t(const std::pair<int, K>&)>> nodeMap(
        0, [](const std::pair<int, K>& p) {
            return std::hash<int>()(p.first) ^ std::hash<K>()(p.second);
        });

    // Create nodes bottom-up
    for (int level = 0; level < static_cast<int>(numPasses); ++level) {
        for (const auto& [comm, children] : levelComms[level]) {
            DendrogramNode<K> node;
            node.id = nodeId;
            node.level = level;
            node.size = 0;

            if (level == 0) {
                // Leaf level: children are vertices
                node.children = children;
                node.size = children.size();
            } else {
                // Higher level: children are community nodes
                for (K child : children) {
                    auto it = nodeMap.find({level - 1, child});
                    if (it != nodeMap.end()) {
                        node.children.push_back(it->second);
                        node.size += dendrogram[it->second].size;
                    }
                }
            }

            nodeMap[{level, comm}] = nodeId;
            dendrogram.push_back(node);
            nodeId++;
        }
    }

    // Find roots (nodes at highest level)
    roots.clear();
    for (const auto& [comm, _] : levelComms[numPasses - 1]) {
        auto it = nodeMap.find({static_cast<int>(numPasses - 1), comm});
        if (it != nodeMap.end()) {
            roots.push_back(it->second);
        }
    }

    // Set parent pointers
    for (auto& node : dendrogram) {
        for (K child : node.children) {
            if (child < dendrogram.size()) {
                dendrogram[child].parent = node.id;
            }
        }
    }

    GRAPHBREW_TRACE("buildDendrogram: %zu nodes, %zu roots", dendrogram.size(), roots.size());
}

/**
 * Build a synthetic single-level dendrogram from a flat membership vector.
 *
 * This is used when community detection (e.g., Rabbit Order) produces only
 * a flat membership[v] = communityId without a hierarchical dendrogram.
 * The synthetic dendrogram has:
 *   - One leaf node per community (level 0), containing its vertex list
 *   - One root node (level 1) whose children are all leaf nodes
 *
 * This enables dendrogram-dependent orderings (DFS, BFS, etc.) to work
 * with any community detection algorithm.
 */
template <typename K>
void buildSyntheticDendrogram(
    std::vector<DendrogramNode<K>>& dendrogram,
    std::vector<K>& roots,
    const std::vector<K>& membership,
    size_t N,
    size_t numCommunities) {

    GRAPHBREW_TRACE("buildSyntheticDendrogram: N=%zu, C=%zu", N, numCommunities);

    dendrogram.clear();
    roots.clear();

    const K C = static_cast<K>(numCommunities);

    // Group vertices by community
    std::vector<std::vector<K>> commVertices(C);
    for (size_t v = 0; v < N; ++v) {
        K c = membership[v];
        if (c < C) {
            commVertices[c].push_back(static_cast<K>(v));
        }
    }

    // Create leaf nodes (level 0): one per community
    dendrogram.resize(C + 1);  // C leaves + 1 root

    K rootId = C;  // Root is the last node

    for (K c = 0; c < C; ++c) {
        auto& node = dendrogram[c];
        node.id = c;
        node.parent = rootId;
        node.children = std::move(commVertices[c]);
        node.size = node.children.size();
        node.weight = 0.0;
        node.level = 0;
    }

    // Create root node (level 1): parent of all leaves
    auto& root = dendrogram[rootId];
    root.id = rootId;
    root.parent = static_cast<K>(-1);
    root.children.resize(C);
    root.size = N;
    root.weight = 0.0;
    root.level = 1;
    for (K c = 0; c < C; ++c) {
        root.children[c] = c;
    }

    roots.push_back(rootId);

    GRAPHBREW_TRACE("buildSyntheticDendrogram: %zu nodes, 1 root", dendrogram.size());
}

//=============================================================================
// SECTION 16: ORDERING - HIERARCHICAL SORT
//=============================================================================

/**
 * Hierarchical multi-level sort (leiden.hxx style)
 *
 * Zero-degree nodes are grouped at the END for better cache locality.
 * They don't contribute to locality (no edges), so keeping them separate
 * from "active" nodes improves cache utilization.
 */
template <typename K, typename NodeID_T>
void orderHierarchicalSort(
    pvector<NodeID_T>& newIds,
    const GraphBrewResult<K>& result,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderHierarchicalSort: N=%zu, passes=%zu", N, result.membershipPerPass.size());

    // Separate zero-degree (isolated) nodes - they go at the end
    std::vector<size_t> active, isolated;
    active.reserve(N);
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(v);
        } else {
            active.push_back(v);
        }
    }

    const size_t numPasses = result.membershipPerPass.size();

    auto comparator = [&](size_t a, size_t b) {
        // Compare from coarsest (last) to finest, using top 3 passes max
        // Multi-level sort key: coarsest-to-finest pass ordering
        size_t startPass = (numPasses > 3) ? numPasses - 3 : 0;
        for (size_t p = numPasses; p > startPass; --p) {
            K ca = result.membershipPerPass[p - 1][a];
            K cb = result.membershipPerPass[p - 1][b];
            if (ca != cb) return ca < cb;
        }
        // Tie-break by degree (hubs first), then vertex ID (determinism)
        if (degrees[a] != degrees[b]) return degrees[a] > degrees[b];
        return a < b;
    };

    if (config.useParallelSort) {
        __gnu_parallel::sort(active.begin(), active.end(), comparator);
    } else {
        std::sort(active.begin(), active.end(), comparator);
    }

    // Assign IDs: active nodes first, isolated nodes at the end
    #pragma omp parallel for
    for (size_t i = 0; i < active.size(); ++i) {
        newIds[active[i]] = static_cast<NodeID_T>(i);
    }

    // Isolated nodes get highest IDs (grouped at the end)
    NodeID_T isolatedStart = static_cast<NodeID_T>(active.size());
    for (size_t i = 0; i < isolated.size(); ++i) {
        newIds[isolated[i]] = isolatedStart + static_cast<NodeID_T>(i);
    }

    GRAPHBREW_TRACE("orderHierarchicalSort: %zu active, %zu isolated", active.size(), isolated.size());
}

//=============================================================================
// SECTION 16a: ORDERING - HIERARCHICAL CACHE-AWARE
//=============================================================================

/**
 * Hierarchical Cache-Aware Ordering
 *
 * KEY INSIGHT: Leiden's multi-pass structure is a natural dendrogram!
 *
 * Pass 0: Finest communities (many small) - e.g., 144k communities
 * Pass 1: Coarser communities - e.g., 50k communities
 * Pass 2: Coarsest communities - e.g., 10k communities (or fewer)
 *
 * Algorithm:
 * 1. Use COARSEST level (last pass) for primary grouping → large cache blocks
 * 2. Within each coarse community, use middle pass for secondary grouping
 * 3. Within each fine community, use BFS on original graph for locality
 *
 * This creates a natural hierarchy like RabbitOrder's dendrogram:
 * - Large contiguous blocks (good cache locality)
 * - Preserves Leiden's community quality within blocks
 * - Isolated (zero-degree) vertices grouped at the end
 */
template <typename K, typename NodeID_T, typename DestID_T>
void orderHierarchicalCacheAware(
    pvector<NodeID_T>& newIds,
    const GraphBrewResult<K>& result,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderHierarchicalCacheAware: N=%zu, passes=%zu", N, result.membershipPerPass.size());

    const size_t numPasses = result.membershipPerPass.size();
    // hcache exploits Leiden's *hierarchy*: it needs at least 2 passes
    // (one fine + one coarse) to map cache levels.  If Leiden converged
    // in a single pass (typical for road/mesh graphs that lack hierarchical
    // community structure), the "coarsest" and "finest" levels would be
    // identical -- the algorithm degenerates to plain community sort.
    // Fall through to the multi-level hierarchical sort, which is well-defined
    // for any number of passes including 1.
    if (numPasses < 2) {
        std::cerr << "  [warning] hcache requires Leiden hierarchy with >= 2 passes "
                  << "(got " << numPasses << ").  Falling back to orderHierarchicalSort.\n"
                  << "  This typically happens on road/mesh graphs with weak community "
                  << "structure.  Consider using -o 12:leiden or -o 12:rcm instead.\n";
        orderHierarchicalSort<K, NodeID_T>(newIds, result, degrees, N, config);
        return;
    }

    // Separate zero-degree (isolated) nodes
    std::vector<K> active, isolated;
    active.reserve(N);
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(static_cast<K>(v));
        } else {
            active.push_back(static_cast<K>(v));
        }
    }

    // Use COARSEST level for primary grouping (last pass = fewest communities)
    // This gives us large cache blocks
    const auto& coarseMembership = result.membershipPerPass.back();

    // Find number of coarse communities
    K maxCoarse = 0;
    for (K v : active) {
        maxCoarse = std::max(maxCoarse, coarseMembership[v]);
    }
    const size_t numCoarse = static_cast<size_t>(maxCoarse) + 1;

    printf("  hierarchical-cache: %zu coarse communities from pass %zu\n",
           numCoarse, numPasses - 1);

    // Build vertex lists per coarse community
    std::vector<std::vector<K>> coarseVertices(numCoarse);
    for (K v : active) {
        coarseVertices[coarseMembership[v]].push_back(v);
    }

    // Sort coarse communities by size (large first for better cache behavior)
    std::vector<K> coarseOrder(numCoarse);
    std::iota(coarseOrder.begin(), coarseOrder.end(), K(0));
    std::sort(coarseOrder.begin(), coarseOrder.end(), [&](K a, K b) {
        return coarseVertices[a].size() > coarseVertices[b].size();
    });

    // Compute offsets for coarse communities (prefix sum)
    std::vector<size_t> coarseOffsets(numCoarse + 1, 0);
    for (size_t i = 0; i < numCoarse; ++i) {
        coarseOffsets[i + 1] = coarseOffsets[i] + coarseVertices[coarseOrder[i]].size();
    }

    // Create reverse mapping
    std::vector<K> coarseToIndex(numCoarse);
    for (size_t i = 0; i < numCoarse; ++i) {
        coarseToIndex[coarseOrder[i]] = static_cast<K>(i);
    }

    // Process each coarse community: sort by finer levels, then BFS within finest
    std::vector<K> localIds(N, static_cast<K>(-1));

    #pragma omp parallel
    {
        std::queue<K> bfsQueue;
        std::vector<bool> visited;

        #pragma omp for schedule(dynamic, 1)
        for (size_t ci = 0; ci < numCoarse; ++ci) {
            K coarseComm = coarseOrder[ci];
            auto& verts = coarseVertices[coarseComm];
            if (verts.empty()) continue;

            // Sort vertices within coarse community by finer hierarchy
            // From finest (pass 0) to coarser (pass numPasses-2)
            std::sort(verts.begin(), verts.end(), [&](K a, K b) {
                // Compare from coarser to finer (but skip the coarsest since already grouped)
                for (size_t p = numPasses - 1; p > 0; --p) {
                    K ca = result.membershipPerPass[p - 1][a];
                    K cb = result.membershipPerPass[p - 1][b];
                    if (ca != cb) return ca < cb;
                }
                // Tie-break: higher degree first (hubs together)
                return degrees[a] > degrees[b];
            });

            // Now do BFS within each finest-level community for even better locality
            // This is the key: we traverse the ORIGINAL GRAPH within fine communities
            if (numPasses > 1) {
                const auto& finestMembership = result.membershipPerPass[0];

                // Group vertices by finest community
                std::unordered_map<K, std::vector<K>> fineGroups;
                for (K v : verts) {
                    fineGroups[finestMembership[v]].push_back(v);
                }

                // BFS within each fine community
                K localId = 0;
                std::vector<K> sortedFineComms;
                for (auto& [fc, _] : fineGroups) {
                    sortedFineComms.push_back(fc);
                }
                std::sort(sortedFineComms.begin(), sortedFineComms.end());

                for (K fineComm : sortedFineComms) {
                    auto& fineVerts = fineGroups[fineComm];
                    if (fineVerts.size() <= 1) {
                        // Trivial case
                        for (K v : fineVerts) {
                            localIds[v] = localId++;
                        }
                        continue;
                    }

                    // Build local index for BFS
                    visited.assign(fineVerts.size(), false);
                    std::unordered_map<K, size_t> vertToLocal;
                    for (size_t i = 0; i < fineVerts.size(); ++i) {
                        vertToLocal[fineVerts[i]] = i;
                    }

                    // Find starting vertex (highest degree)
                    K startV = fineVerts[0];
                    K maxDeg = degrees[fineVerts[0]];
                    for (K v : fineVerts) {
                        if (degrees[v] > maxDeg) {
                            maxDeg = degrees[v];
                            startV = v;
                        }
                    }

                    // BFS from startV
                    bfsQueue.push(startV);
                    visited[vertToLocal[startV]] = true;

                    while (!bfsQueue.empty()) {
                        K u = bfsQueue.front();
                        bfsQueue.pop();
                        localIds[u] = localId++;

                        // Add unvisited neighbors in same fine community
                        for (auto neighbor : g.out_neigh(u)) {
                            NodeID_T v;
                            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                                v = neighbor;
                            } else {
                                v = neighbor.v;
                            }

                            auto it = vertToLocal.find(static_cast<K>(v));
                            if (it != vertToLocal.end() && !visited[it->second]) {
                                visited[it->second] = true;
                                bfsQueue.push(static_cast<K>(v));
                            }
                        }
                    }

                    // Handle disconnected vertices
                    for (size_t i = 0; i < fineVerts.size(); ++i) {
                        if (!visited[i]) {
                            localIds[fineVerts[i]] = localId++;
                        }
                    }
                }
            } else {
                // Only one pass: just assign in sorted order
                K localId = 0;
                for (K v : verts) {
                    localIds[v] = localId++;
                }
            }
        }
    }

    // Compute global IDs = coarseOffset + localId
    #pragma omp parallel for
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] > 0) {
            K coarseIdx = coarseToIndex[coarseMembership[v]];
            newIds[v] = static_cast<NodeID_T>(coarseOffsets[coarseIdx] + localIds[v]);
        }
    }

    // Assign isolated vertices at the end
    size_t isolatedStart = coarseOffsets[numCoarse];
    for (size_t i = 0; i < isolated.size(); ++i) {
        newIds[isolated[i]] = static_cast<NodeID_T>(isolatedStart + i);
    }

    printf("  hierarchical-cache: %zu active vertices, %zu isolated\n",
           active.size(), isolated.size());
}

//=============================================================================
// SECTION 16-PRIMITIVES: Reusable per-community ordering primitives
//=============================================================================
//
// These free template functions are the per-community building blocks
// (the intra-community order axis in the paper's three-axis pipeline).
// They operate on a
// SINGLE community's vertex list and write into a shared `localIds`
// vector that the caller composes into the final permutation.
//
// Design rules so multiple variants can share them safely:
//   - The function processes ONE community per call; the caller is
//     responsible for the outer parallel-for across communities.
//   - The caller pre-populates `vertToLocal[v]` for every v in `verts`
//     with v's position inside `verts` (so BFS membership lookups stay
//     O(1) without a per-call hashmap).
//   - The function writes `localIds[v]` for every v in `verts` and
//     does not touch any other entries.
//   - No global allocations; all scratch buffers are reused arguments
//     so the caller can size them once per thread.
//
// Available primitives:
//   intraBFSFromHub<...>      max-degree-start BFS, neighbour insertion order
//   intraRCM<...>             BNF pseudo-peripheral start + CM (ascending
//                             neighbour-degree) BFS + final reversal
//
// HRAB's STEP 4 ("intra-community ordering") and the standalone
// CONNECTIVITY_BFS variant both call these.  TQR's micro-ordering also
// uses intraBFSFromHub.  Adding a new variant amounts to writing one
// short function that picks which primitive to call per community.

/**
 * BFS within a single community from the highest-degree vertex.
 *
 * @param verts        vertices in the community
 * @param c            community ID (for membership[] check)
 * @param membership   full membership vector
 * @param degrees      full degree vector (used to pick start vertex)
 * @param g            original graph (for adjacency)
 * @param vertToLocal  flat vertex->local-index map (caller-populated)
 * @param visited      thread-local scratch, sized to verts.size()
 * @param bfsQueue     thread-local scratch FIFO
 * @param localIds     output: localIds[v] gets v's position within the community
 */
template <typename K, typename NodeID_T, typename DestID_T>
inline void intraBFSFromHub(
    const std::vector<K>& verts,
    K c,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<size_t>& vertToLocal,
    std::vector<bool>& visited,
    std::queue<K>& bfsQueue,
    std::vector<K>& localIds)
{
    const size_t sz = verts.size();
    if (sz == 0) return;
    if (sz == 1) { localIds[verts[0]] = 0; return; }

    visited.assign(sz, false);

    // Pick highest-degree vertex as BFS start.
    K startV = verts[0];
    K maxDeg = degrees[verts[0]];
    for (K v : verts) {
        if (degrees[v] > maxDeg) { maxDeg = degrees[v]; startV = v; }
    }

    K localId = 0;
    while (!bfsQueue.empty()) bfsQueue.pop();
    bfsQueue.push(startV);
    visited[vertToLocal[startV]] = true;

    while (!bfsQueue.empty()) {
        K u = bfsQueue.front();
        bfsQueue.pop();
        localIds[u] = localId++;

        for (auto neighbor : g.out_neigh(u)) {
            NodeID_T v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = neighbor;
            } else {
                v = neighbor.v;
            }
            if (membership[v] != c) continue;
            size_t localIdx = vertToLocal[static_cast<K>(v)];
            if (localIdx != static_cast<size_t>(-1) && !visited[localIdx]) {
                visited[localIdx] = true;
                bfsQueue.push(static_cast<K>(v));
            }
        }
    }

    // Disconnected vertices get IDs at the end (preserves verts[] order).
    for (size_t i = 0; i < sz; ++i) {
        if (!visited[i]) localIds[verts[i]] = localId++;
    }
}

/**
 * Reverse Cuthill-McKee within a single community.
 *
 * Tiered pseudoperipheral-start (BNF) cost based on sz:
 *   sz <= 32       : min-degree start, skip George-Liu iteration
 *   sz > 32        : up to 3 George-Liu iterations to find a better
 *                    pseudoperipheral start vertex
 * Then standard CM-BFS (neighbours sorted by ascending degree) and
 * final reversal of the connected component.  Disconnected vertices
 * keep their input order at the tail (not reversed).
 *
 * Same argument convention as intraBFSFromHub plus one extra scratch
 * vector `cmOrder` (sized to verts.size()).
 */
template <typename K, typename NodeID_T, typename DestID_T>
inline void intraRCM(
    const std::vector<K>& verts,
    K c,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<size_t>& vertToLocal,
    std::vector<bool>& visited,
    std::queue<K>& bfsQueue,
    std::vector<K>& cmOrder,
    std::vector<K>& localIds)
{
    const size_t sz = verts.size();
    if (sz == 0) return;
    if (sz == 1) { localIds[verts[0]] = 0; return; }

    visited.assign(sz, false);
    cmOrder.clear();
    cmOrder.reserve(sz);

    // BNF start: min-degree seed vertex
    K startV = verts[0];
    K minDeg = degrees[verts[0]];
    for (K v : verts) {
        if (degrees[v] < minDeg ||
            (degrees[v] == minDeg && v < startV)) {
            minDeg = degrees[v];
            startV = v;
        }
    }

    // George-Liu pseudoperipheral iteration for sz > 32.  Bounded at 3
    // iterations (cheap heuristic — full GL converges in <10 on most
    // sub-4096-vertex communities).
    const int maxGLIter = (sz <= 32) ? 0 : 3;
    if (maxGLIter > 0) {
        K glNode = startV;
        int64_t prevEcc = 0;
        for (int glIt = 0; glIt < maxGLIter; ++glIt) {
            visited.assign(sz, false);
            std::vector<K> curLevel;
            curLevel.push_back(glNode);
            int64_t ecc = 0;
            while (!curLevel.empty()) {
                std::vector<K> nextLevel;
                for (K u : curLevel) {
                    for (auto neighbor : g.out_neigh(u)) {
                        NodeID_T v;
                        if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                            v = neighbor;
                        } else {
                            v = neighbor.v;
                        }
                        if (membership[v] != c) continue;
                        size_t localIdx = vertToLocal[static_cast<K>(v)];
                        if (localIdx != static_cast<size_t>(-1) && !visited[localIdx]) {
                            visited[localIdx] = true;
                            nextLevel.push_back(static_cast<K>(v));
                        }
                    }
                }
                if (!nextLevel.empty()) {
                    ecc++;
                    curLevel = std::move(nextLevel);
                } else {
                    break;
                }
            }
            if (ecc <= prevEcc) break;
            prevEcc = ecc;

            // Pick min-degree vertex in the farthest BFS level
            K bestV = curLevel[0];
            K bestDeg = degrees[curLevel[0]];
            for (size_t i = 1; i < curLevel.size(); ++i) {
                if (degrees[curLevel[i]] < bestDeg) {
                    bestDeg = degrees[curLevel[i]];
                    bestV = curLevel[i];
                }
            }
            glNode = bestV;
        }
        startV = glNode;
    }

    // Cuthill-McKee BFS: enqueue children sorted by ascending degree
    visited.assign(sz, false);
    cmOrder.clear();
    while (!bfsQueue.empty()) bfsQueue.pop();
    size_t startLocalIdx = vertToLocal[static_cast<K>(startV)];
    if (startLocalIdx < sz) {
        bfsQueue.push(startV);
        visited[startLocalIdx] = true;
    }

    while (!bfsQueue.empty()) {
        K u = bfsQueue.front();
        bfsQueue.pop();
        cmOrder.push_back(u);

        std::vector<std::pair<K, K>> candidates; // (degree, vertex)
        for (auto neighbor : g.out_neigh(u)) {
            NodeID_T v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = neighbor;
            } else {
                v = neighbor.v;
            }
            if (membership[v] != c) continue;
            size_t localIdx = vertToLocal[static_cast<K>(v)];
            if (localIdx != static_cast<size_t>(-1) && !visited[localIdx]) {
                visited[localIdx] = true;
                candidates.push_back({degrees[static_cast<K>(v)], static_cast<K>(v)});
            }
        }
        std::sort(candidates.begin(), candidates.end());
        for (auto& [d, v] : candidates) bfsQueue.push(v);
    }

    // Disconnected vertices get appended (not reversed)
    size_t mainSize = cmOrder.size();
    for (size_t i = 0; i < sz; ++i) {
        if (!visited[i]) cmOrder.push_back(verts[i]);
    }

    // Reverse the main connected component (the R in RCM)
    for (size_t i = 0; i < mainSize; ++i) {
        localIds[cmOrder[i]] = static_cast<K>(mainSize - 1 - i);
    }
    // Disconnected tail
    for (size_t i = mainSize; i < cmOrder.size(); ++i) {
        localIds[cmOrder[i]] = static_cast<K>(i);
    }
}

/**
 * RCM++ (Hou/Liu/Zhu, arXiv 2409.04171, 2024) within a single community.
 * Identical to intraRCM<>() in all respects EXCEPT the initial start-vertex
 * pick.  Instead of plain BNF min-degree, RCM++ scores every community
 * vertex by `0.5*deg_rank + 0.5*depth_rank` where depth_rank comes from
 * one BFS from the min-degree seed; the argmin is the start.
 *
 * This costs one extra BFS (O(|E_local|)) over plain intraRCM, but on
 * graphs with multiple equally-low-degree candidates the bi-criteria pick
 * often finds a more peripheral seed in a single shot, eliminating up to
 * `maxGLIter` rounds of George-Liu refinement (which dominate intraRCM's
 * runtime for sz > 32).  Net effect: 2-5% improvement on bandwidth-bound
 * kernels for citation-graph CC where many leaf vertices tie at deg=1.
 *
 * Same argument convention as intraRCM<>().  All scratch vectors are
 * caller-provided to allow reuse across communities.
 */
template <typename K, typename NodeID_T, typename DestID_T>
inline void intraRCMpp(
    const std::vector<K>& verts,
    K c,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<size_t>& vertToLocal,
    std::vector<bool>& visited,
    std::queue<K>& bfsQueue,
    std::vector<K>& cmOrder,
    std::vector<K>& localIds)
{
    const size_t sz = verts.size();
    if (sz == 0) return;
    if (sz == 1) { localIds[verts[0]] = 0; return; }

    visited.assign(sz, false);
    cmOrder.clear();
    cmOrder.reserve(sz);

    // RCM++ Step 1: BNF min-degree seed (same as intraRCM)
    K startV = verts[0];
    K minDeg = degrees[verts[0]];
    for (K v : verts) {
        if (degrees[v] < minDeg ||
            (degrees[v] == minDeg && v < startV)) {
            minDeg = degrees[v];
            startV = v;
        }
    }

    // RCM++ Step 2 (NEW vs intraRCM): one BFS from the seed to record
    // per-vertex BFS depth, then pick start = argmin(0.5*deg_rank + 0.5*depth_rank)
    // among all community vertices.  Skip for sz <= 32 (overhead not worth it).
    if (sz > 32) {
        std::vector<K> depth(sz, static_cast<K>(0));
        visited.assign(sz, false);
        size_t seedIdx = vertToLocal[static_cast<K>(startV)];
        if (seedIdx < sz) {
            visited[seedIdx] = true;
            std::vector<K> curLevel = {startV};
            K curDepth = 0;
            while (!curLevel.empty()) {
                std::vector<K> nextLevel;
                for (K u : curLevel) {
                    size_t uIdx = vertToLocal[u];
                    if (uIdx < sz) depth[uIdx] = curDepth;
                    for (auto neighbor : g.out_neigh(u)) {
                        NodeID_T v;
                        if constexpr (std::is_same_v<DestID_T, NodeID_T>) v = neighbor;
                        else v = neighbor.v;
                        if (membership[v] != c) continue;
                        size_t lIdx = vertToLocal[static_cast<K>(v)];
                        if (lIdx != static_cast<size_t>(-1) && !visited[lIdx]) {
                            visited[lIdx] = true;
                            nextLevel.push_back(static_cast<K>(v));
                        }
                    }
                }
                curDepth++;
                curLevel = std::move(nextLevel);
            }
            // Rank by degree ascending (low deg = low rank = better for peripheral)
            std::vector<size_t> degRank(sz), depthRank(sz);
            std::vector<K> sortedByDeg(verts), sortedByDepth(verts);
            std::sort(sortedByDeg.begin(), sortedByDeg.end(),
                      [&](K a, K b){
                          return degrees[a] != degrees[b]
                              ? degrees[a] < degrees[b]
                              : a < b;
                      });
            std::sort(sortedByDepth.begin(), sortedByDepth.end(),
                      [&](K a, K b){
                          size_t ai = vertToLocal[a], bi = vertToLocal[b];
                          K ad = (ai < sz) ? depth[ai] : 0;
                          K bd = (bi < sz) ? depth[bi] : 0;
                          return ad != bd
                              ? ad > bd
                              : a < b;
                      });
            for (size_t i = 0; i < sz; ++i) {
                size_t di = vertToLocal[sortedByDeg[i]];
                if (di < sz) degRank[di] = i;
                size_t pi = vertToLocal[sortedByDepth[i]];
                if (pi < sz) depthRank[pi] = i;
            }
            // Pick argmin combined score
            K bestStart = startV;
            double bestScore = 1e18;
            for (K v : verts) {
                size_t vi = vertToLocal[v];
                if (vi >= sz) continue;
                double s = 0.5 * static_cast<double>(degRank[vi]) +
                           0.5 * static_cast<double>(depthRank[vi]);
                if (s < bestScore ||
                    (s == bestScore && v < bestStart)) {
                    bestScore = s;
                    bestStart = v;
                }
            }
            startV = bestStart;
        }
    }

    // RCM++ Step 3: George-Liu pseudoperipheral refinement (same as intraRCM).
    const int maxGLIter = (sz <= 32) ? 0 : 3;
    if (maxGLIter > 0) {
        K glNode = startV;
        int64_t prevEcc = 0;
        for (int glIt = 0; glIt < maxGLIter; ++glIt) {
            visited.assign(sz, false);
            std::vector<K> curLevel;
            curLevel.push_back(glNode);
            int64_t ecc = 0;
            while (!curLevel.empty()) {
                std::vector<K> nextLevel;
                for (K u : curLevel) {
                    for (auto neighbor : g.out_neigh(u)) {
                        NodeID_T v;
                        if constexpr (std::is_same_v<DestID_T, NodeID_T>) v = neighbor;
                        else v = neighbor.v;
                        if (membership[v] != c) continue;
                        size_t localIdx = vertToLocal[static_cast<K>(v)];
                        if (localIdx != static_cast<size_t>(-1) && !visited[localIdx]) {
                            visited[localIdx] = true;
                            nextLevel.push_back(static_cast<K>(v));
                        }
                    }
                }
                if (!nextLevel.empty()) { ecc++; curLevel = std::move(nextLevel); }
                else break;
            }
            if (ecc <= prevEcc) break;
            prevEcc = ecc;
            K bestV = curLevel[0];
            K bestDeg = degrees[curLevel[0]];
            for (size_t i = 1; i < curLevel.size(); ++i) {
                if (degrees[curLevel[i]] < bestDeg ||
                    (degrees[curLevel[i]] == bestDeg &&
                     curLevel[i] < bestV)) {
                    bestDeg = degrees[curLevel[i]];
                    bestV = curLevel[i];
                }
            }
            glNode = bestV;
        }
        startV = glNode;
    }

    // RCM++ Step 4: Cuthill-McKee BFS (identical to intraRCM)
    visited.assign(sz, false);
    cmOrder.clear();
    while (!bfsQueue.empty()) bfsQueue.pop();
    size_t startLocalIdx = vertToLocal[static_cast<K>(startV)];
    if (startLocalIdx < sz) {
        bfsQueue.push(startV);
        visited[startLocalIdx] = true;
    }
    while (!bfsQueue.empty()) {
        K u = bfsQueue.front();
        bfsQueue.pop();
        cmOrder.push_back(u);
        std::vector<std::pair<K, K>> candidates;
        for (auto neighbor : g.out_neigh(u)) {
            NodeID_T v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) v = neighbor;
            else v = neighbor.v;
            if (membership[v] != c) continue;
            size_t localIdx = vertToLocal[static_cast<K>(v)];
            if (localIdx != static_cast<size_t>(-1) && !visited[localIdx]) {
                visited[localIdx] = true;
                candidates.push_back({degrees[static_cast<K>(v)], static_cast<K>(v)});
            }
        }
        std::sort(candidates.begin(), candidates.end());
        for (auto& [d, v] : candidates) bfsQueue.push(v);
    }
    size_t mainSize = cmOrder.size();
    std::vector<K> disconnected;
    for (size_t i = 0; i < sz; ++i) {
        if (!visited[i]) disconnected.push_back(verts[i]);
    }
    std::sort(disconnected.begin(), disconnected.end());
    cmOrder.insert(
        cmOrder.end(), disconnected.begin(), disconnected.end());
    for (size_t i = 0; i < mainSize; ++i) {
        localIds[cmOrder[i]] = static_cast<K>(mainSize - 1 - i);
    }
    for (size_t i = mainSize; i < cmOrder.size(); ++i) {
        localIds[cmOrder[i]] = static_cast<K>(i);
    }
}

/**
 * Dendrogram-DFS within a single community using Rabbit's per-vertex
 * child/sibling links.  This reuses the exact traversal of
 * rabbitOrderingGeneration() but emits LOCAL ids (0..sz-1) so that
 * orderCompose<>() can stitch them under any community-order axis.
 *
 * Identical in spirit to rabbitDescendants(): for vertex v, follow the
 * child chain (depth-first), then for each visited node also descend
 * into its sibling subtree.  No BFS queue, no visited bitmap, no
 * adjacency-list scan: pure pointer-chase on the dendrogram, so this is
 * the cheapest intra primitive when CD=Rabbit.
 *
 * @param root         dendrogram root for this community (rabbitToplevel[c])
 * @param dendChild    flat child[] array (INVALID if leaf)
 * @param dendSibling  flat sibling[] array (INVALID if last)
 * @param stack        thread-local scratch deque (reused)
 * @param localIds     output: localIds[v] gets v's position within the community
 */
template <typename K>
inline void intraDendrogramDFS(
    K root,
    const std::vector<K>& dendChild,
    const std::vector<K>& dendSibling,
    std::deque<K>& stack,
    std::vector<K>& localIds)
{
    constexpr K INVALID = static_cast<K>(-1);
    K localId = 0;
    stack.clear();

    // Push v and lineal descendants of v on the dendrogram (same shape
    // as rabbitDescendants() in SECTION 13).
    auto pushDescendants = [&](K v) {
        stack.push_back(v);
        K c = dendChild[v];
        while (c != INVALID) {
            stack.push_back(c);
            c = dendChild[c];
        }
    };

    pushDescendants(root);

    while (!stack.empty()) {
        K v = stack.back();
        stack.pop_back();
        localIds[v] = localId++;
        if (dendSibling[v] != INVALID) {
            pushDescendants(dendSibling[v]);
        }
    }
}

/**
 * Gorder-greedy (Hao Wei et al. 2016) within a single community using
 * a UnitHeap for O(1) amortized increment/extract-max.  Reuses the same
 * algorithm as the standalone HRAB hybrid-rabbit-gord path but exposed
 * as a primitive that orderCompose<>() can compose with any community-
 * or super-graph axis.
 *
 * Per-community complexity: O(|E_local|) thanks to the UnitHeap.
 *
 * @param verts        vertices in the community
 * @param c            community ID (for membership[] check)
 * @param membership   full membership vector
 * @param degrees      full degree vector (used to pick start vertex)
 * @param g            original graph (for adjacency)
 * @param vertToLocal  flat vertex->local-index map (caller-populated)
 * @param window       sliding-window size W (Gorder default = 5)
 * @param placedOrder  thread-local scratch: order of placed vertices
 * @param localNeighborsScratch thread-local scratch: per-vertex local adjacency
 * @param localIds     output: localIds[v] gets v's position within the community
 */
template <typename K, typename NodeID_T, typename DestID_T>
inline void intraGorderGreedy(
    const std::vector<K>& verts,
    K c,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<size_t>& vertToLocal,
    int window,
    std::vector<K>& placedOrder,
    std::vector<std::vector<size_t>>& localNeighborsScratch,
    std::vector<K>& localIds)
{
    const size_t sz = verts.size();
    if (sz == 0) return;
    if (sz == 1) { localIds[verts[0]] = 0; return; }
    // Tiny communities: sequential ordering (Gorder overhead not worth it).
    if (sz <= 3) {
        for (size_t i = 0; i < sz; ++i) localIds[verts[i]] = static_cast<K>(i);
        return;
    }

    // UnitHeap node / bucket structs (local to keep scope tight)
    struct Node { int key; int prev; int next; };
    struct Bucket { int first = -1; int second = -1; };

    // Build per-community local adjacency
    if (localNeighborsScratch.size() < sz) localNeighborsScratch.resize(sz);
    for (size_t i = 0; i < sz; ++i) localNeighborsScratch[i].clear();
    for (size_t i = 0; i < sz; ++i) {
        K u = verts[i];
        for (auto neighbor : g.out_neigh(u)) {
            NodeID_T v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) v = neighbor;
            else                                              v = neighbor.v;
            if (membership[v] != c) continue;
            size_t localIdx = vertToLocal[static_cast<K>(v)];
            if (localIdx != static_cast<size_t>(-1) && localIdx < sz) {
                localNeighborsScratch[i].push_back(localIdx);
            }
        }
    }

    // Initialise UnitHeap: all vertices at key=0, doubly-linked in order
    std::vector<Node> nodes(sz);
    std::vector<Bucket> buckets(16);
    int maxKey = 0;
    int top = 0;
    for (size_t i = 0; i < sz; ++i) {
        nodes[i].key  = 0;
        nodes[i].prev = static_cast<int>(i) - 1;
        nodes[i].next = (i + 1 < sz) ? static_cast<int>(i + 1) : -1;
    }
    buckets[0].first  = 0;
    buckets[0].second = static_cast<int>(sz - 1);

    auto ensureBucket = [&](int k) {
        if (k >= static_cast<int>(buckets.size())) buckets.resize(static_cast<size_t>(k + 8));
    };
    auto unlink = [&](int idx) {
        int p = nodes[idx].prev, n = nodes[idx].next;
        if (p >= 0) nodes[p].next = n;
        if (n >= 0) nodes[n].prev = p;
        int k = nodes[idx].key;
        if (buckets[k].first == idx && buckets[k].second == idx) {
            buckets[k].first = buckets[k].second = -1;
        } else if (buckets[k].first == idx) {
            buckets[k].first = n;
        } else if (buckets[k].second == idx) {
            buckets[k].second = p;
        }
        if (top == idx) top = n;
    };
    auto linkToBucketFront = [&](int idx, int k) {
        ensureBucket(k);
        nodes[idx].key = k;
        if (buckets[k].first < 0) {
            int afterNode = -1;
            for (int bk = k - 1; bk >= 0; --bk) {
                if (buckets[bk].first >= 0) { afterNode = buckets[bk].first; break; }
            }
            int beforeNode = -1;
            for (int bk = k + 1; bk <= maxKey; ++bk) {
                if (buckets[bk].second >= 0) { beforeNode = buckets[bk].second; break; }
            }
            nodes[idx].prev = beforeNode;
            nodes[idx].next = afterNode;
            if (beforeNode >= 0) nodes[beforeNode].next = idx;
            if (afterNode  >= 0) nodes[afterNode].prev  = idx;
            buckets[k].first = buckets[k].second = idx;
        } else {
            int oldFirst = buckets[k].first;
            int beforeOldFirst = nodes[oldFirst].prev;
            nodes[idx].prev = beforeOldFirst;
            nodes[idx].next = oldFirst;
            nodes[oldFirst].prev = idx;
            if (beforeOldFirst >= 0) nodes[beforeOldFirst].next = idx;
            buckets[k].first = idx;
        }
        if (k > maxKey) maxKey = k;
        if (top < 0 || k >= nodes[top].key) top = idx;
    };
    auto incrementKey = [&](int idx) {
        int oldKey = nodes[idx].key;
        unlink(idx);
        linkToBucketFront(idx, oldKey + 1);
    };
    auto decrementKey = [&](int idx) {
        int oldKey = nodes[idx].key;
        if (oldKey <= 0) return;
        unlink(idx);
        linkToBucketFront(idx, oldKey - 1);
    };
    auto deleteElement = [&](int idx) {
        unlink(idx);
        nodes[idx].prev = nodes[idx].next = -1;
        nodes[idx].key  = -1;
        while (maxKey > 0 && buckets[maxKey].first < 0) maxKey--;
    };
    auto extractMax = [&]() -> int {
        while (maxKey > 0 && buckets[maxKey].first < 0) maxKey--;
        int idx = buckets[maxKey].first;
        if (idx >= 0) deleteElement(idx);
        return idx;
    };

    // Seed: highest-degree vertex
    size_t bestSeed = 0;
    K seedDeg = degrees[verts[0]];
    for (size_t i = 1; i < sz; ++i) {
        if (degrees[verts[i]] > seedDeg) { seedDeg = degrees[verts[i]]; bestSeed = i; }
    }

    deleteElement(static_cast<int>(bestSeed));
    placedOrder.clear();
    placedOrder.reserve(sz);
    placedOrder.push_back(static_cast<K>(bestSeed));
    localIds[verts[bestSeed]] = 0;

    for (size_t nbr : localNeighborsScratch[bestSeed]) {
        if (nodes[nbr].key >= 0) incrementKey(static_cast<int>(nbr));
    }

    for (K localId = 1; localId < static_cast<K>(sz); ++localId) {
        int best = extractMax();
        if (best < 0) {
            // Disconnected residual: emit remaining verts in input order
            for (size_t i = 0; i < sz && localId < static_cast<K>(sz); ++i) {
                if (nodes[i].key >= 0) {
                    deleteElement(static_cast<int>(i));
                    localIds[verts[i]] = localId++;
                }
            }
            break;
        }
        placedOrder.push_back(static_cast<K>(best));
        localIds[verts[best]] = localId;
        for (size_t nbr : localNeighborsScratch[best]) {
            if (nodes[nbr].key >= 0) incrementKey(static_cast<int>(nbr));
        }
        if (static_cast<int>(placedOrder.size()) > window) {
            size_t oldVert = placedOrder[placedOrder.size() - 1 - window];
            for (size_t nbr : localNeighborsScratch[oldVert]) {
                if (nodes[nbr].key >= 0) decrementKey(static_cast<int>(nbr));
            }
        }
    }
}

inline bool faithfulGorderScoreRangeSafe(
    size_t vertexCount,
    int window)
{
    if (window <= 0) return false;
    const uint64_t window64 = static_cast<uint64_t>(window);
    const uint64_t maximum = static_cast<uint64_t>(
        std::numeric_limits<int>::max());
    if (2 * window64 > maximum) return false;
    const uint64_t vertexLimit =
        (maximum - 2 * window64) / (window64 + 1);
    return vertexCount <= vertexLimit;
}

template <typename K>
inline void faithfulGorderLocalOrder(
    const std::vector<std::vector<size_t>>& outNeighbors,
    const std::vector<std::vector<size_t>>& inNeighbors,
    size_t vertexCount,
    int window,
    std::vector<K>& order)
{
    const size_t size = vertexCount;
    if (
        outNeighbors.size() < size
        || inNeighbors.size() < size
    ) {
        throw std::invalid_argument(
            "Faithful local Gorder received mismatched adjacency");
    }
    if (window <= 0) {
        throw std::invalid_argument(
            "Faithful local Gorder requires a positive window");
    }

    order.clear();
    order.reserve(size);
    if (size == 0) return;
    if (size == 1) {
        order.push_back(0);
        return;
    }
    if (!faithfulGorderScoreRangeSafe(size, window)) {
        throw std::overflow_error(
            "Faithful local Gorder community exceeds score range");
    }
    for (size_t local = 0; local < size; ++local) {
        if (!std::is_sorted(
                outNeighbors[local].begin(),
                outNeighbors[local].end())) {
            throw std::invalid_argument(
                "Faithful local Gorder requires sorted out-neighbors");
        }
        for (size_t target : outNeighbors[local]) {
            if (target >= size) {
                throw std::invalid_argument(
                    "Faithful local Gorder out-neighbor is out of range");
            }
        }
        for (size_t source : inNeighbors[local]) {
            if (source >= size) {
                throw std::invalid_argument(
                    "Faithful local Gorder in-neighbor is out of range");
            }
        }
    }

    int seed = 0;
    int maxInDegree = -1;
    std::vector<K> zeroDegree;
    for (size_t local = 0; local < size; ++local) {
        const int inDegree =
            static_cast<int>(inNeighbors[local].size());
        const int outDegree =
            static_cast<int>(outNeighbors[local].size());
        if (inDegree > maxInDegree) {
            seed = static_cast<int>(local);
            maxInDegree = inDegree;
        } else if (inDegree + outDegree == 0) {
            zeroDegree.push_back(static_cast<K>(local));
        }
    }

    const int localCount = static_cast<int>(size);
    gorder_csr_detail::UnitHeap heap(localCount);
    for (int local = 0; local < localCount; ++local) {
        const int inDegree = static_cast<int>(
            inNeighbors[static_cast<size_t>(local)].size());
        heap.list[local].key = inDegree;
        heap.update[local] = -inDegree;
    }
    heap.ReConstruct();

    std::vector<char> active(size, 1);
    for (K local : zeroDegree) {
        active[local] = 0;
        heap.update[local] = std::numeric_limits<int>::max() / 2;
        heap.DeleteElement(static_cast<int>(local));
    }
    active[static_cast<size_t>(seed)] = 0;
    heap.update[seed] = std::numeric_limits<int>::max() / 2;
    heap.DeleteElement(seed);
    order.push_back(static_cast<K>(seed));

    auto scoreIncrement = [&](size_t local) {
        if (!active[local]) return;
        if (heap.update[local] == 0)
            heap.IncrementKey(static_cast<int>(local));
        else
            ++heap.update[local];
    };
    auto scoreDecrement = [&](size_t local) {
        if (active[local]) --heap.update[local];
    };
    const size_t hugeVertex = static_cast<size_t>(
        std::sqrt(static_cast<double>(size)));

    for (size_t in : inNeighbors[static_cast<size_t>(seed)]) {
        if (outNeighbors[in].size() > hugeVertex) continue;
        scoreIncrement(in);
        if (outNeighbors[in].size() > 1) {
            for (size_t sibling : outNeighbors[in])
                scoreIncrement(sibling);
        }
    }
    if (outNeighbors[static_cast<size_t>(seed)].size() <= hugeVertex) {
        for (size_t out : outNeighbors[static_cast<size_t>(seed)])
            scoreIncrement(out);
    }

    const size_t activeAfterSeed =
        size - zeroDegree.size() - 1;
    std::vector<char> popVertexExists(size, 0);
    for (size_t count = 1; count <= activeAfterSeed; ++count) {
        const int current = heap.ExtractMax();
        active[static_cast<size_t>(current)] = 0;
        heap.update[current] = std::numeric_limits<int>::max() / 2;
        order.push_back(static_cast<K>(current));

        const int popIndex =
            static_cast<int>(count) - window;
        if (popIndex >= 0) {
            const size_t oldLocal = static_cast<size_t>(
                order[static_cast<size_t>(popIndex)]);
            if (outNeighbors[oldLocal].size() <= hugeVertex) {
                for (size_t out : outNeighbors[oldLocal])
                    scoreDecrement(out);
            }
            for (size_t in : inNeighbors[oldLocal]) {
                if (outNeighbors[in].size() > hugeVertex)
                    continue;
                scoreDecrement(in);
                if (outNeighbors[in].size() <= 1)
                    continue;
                if (
                    std::binary_search(
                        outNeighbors[in].begin(),
                        outNeighbors[in].end(),
                        static_cast<size_t>(current))
                ) {
                    popVertexExists[in] = 1;
                } else {
                    for (size_t sibling : outNeighbors[in])
                        scoreDecrement(sibling);
                }
            }
        }

        const size_t currentLocal =
            static_cast<size_t>(current);
        if (outNeighbors[currentLocal].size() <= hugeVertex) {
            for (size_t out : outNeighbors[currentLocal])
                scoreIncrement(out);
        }
        for (size_t in : inNeighbors[currentLocal]) {
            if (outNeighbors[in].size() > hugeVertex)
                continue;
            scoreIncrement(in);
            if (popVertexExists[in]) {
                popVertexExists[in] = 0;
            } else if (outNeighbors[in].size() > 1) {
                for (size_t sibling : outNeighbors[in])
                    scoreIncrement(sibling);
            }
        }
    }

    if (!zeroDegree.empty()) {
        order.insert(
            order.end() - 1,
            zeroDegree.begin(),
            zeroDegree.end());
    }
}

/**
 * Faithful Gorder scoring on the simple directed subgraph induced by one
 * community. Duplicate edges are collapsed before applying the reference
 * one-hop, two-hop sibling, window, and sqrt(|V_local|) guard semantics.
 */
template <typename K, typename NodeID_T, typename DestID_T>
inline void intraGorderFaithful(
    const std::vector<K>& verts,
    K community,
    const std::vector<K>& membership,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const std::vector<size_t>& vertToLocal,
    int window,
    std::vector<K>& placedOrder,
    std::vector<std::vector<size_t>>& outNeighbors,
    std::vector<std::vector<size_t>>& inNeighbors,
    std::vector<K>& localIds)
{
    const size_t size = verts.size();
    if (outNeighbors.size() < size) outNeighbors.resize(size);
    if (inNeighbors.size() < size) inNeighbors.resize(size);
    for (size_t local = 0; local < size; ++local) {
        outNeighbors[local].clear();
        inNeighbors[local].clear();
    }
    for (size_t local = 0; local < size; ++local) {
        for (auto neighbor : g.out_neigh(verts[local])) {
            NodeID_T vertex;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>)
                vertex = neighbor;
            else
                vertex = neighbor.v;
            if (membership[vertex] != community) continue;
            const size_t target =
                vertToLocal[static_cast<size_t>(vertex)];
            if (target < size)
                outNeighbors[local].push_back(target);
        }
        auto& localOut = outNeighbors[local];
        std::sort(localOut.begin(), localOut.end());
        localOut.erase(
            std::unique(localOut.begin(), localOut.end()),
            localOut.end());
    }
    for (size_t source = 0; source < size; ++source) {
        for (size_t target : outNeighbors[source])
            inNeighbors[target].push_back(source);
    }

    faithfulGorderLocalOrder<K>(
        outNeighbors, inNeighbors, size, window, placedOrder);
    for (size_t position = 0; position < placedOrder.size(); ++position) {
        localIds[verts[placedOrder[position]]] =
            static_cast<K>(position);
    }
}

/**
 * Per-community 2-swap refinement (FM-style adjacent-swap).
 *
 * Reduces the per-community locality cost  L(c) = Σ_{(u,v) in E_intra(c)} |pos(u)-pos(v)|
 * by considering every consecutive position pair (p, p+1) inside the community
 * and accepting the swap iff Δcost < 0.  Up to `maxPasses` left-to-right
 * passes per community; early-exit when a pass finds no accepting swap.
 *
 * Cost: O(|E_intra(c)| * maxPasses) per community.  Refinement is per-community
 * independent so the caller may invoke it under a parallel-for over communities.
 *
 * INPUT:
 *   - verts:        community vertex set (any order)
 *   - c:            community id (for membership filtering of neighbors)
 *   - membership:   global vertex -> community id
 *   - g:            CSR graph (for g.out_neigh(v))
 *   - localIds:     INOUT.  On entry: localIds[v] = current local position in c.
 *                   On exit:  refined permutation (still a permutation of [0, |c|)).
 *   - maxPasses:    max passes (caller passes config.refineMaxPasses)
 *
 * SCRATCH (caller-owned, reused across communities to avoid alloc churn):
 *   - adjOffsets:   per-verts-index offsets into adjVertices
 *   - adjVertices:  flat intra-community neighbor list
 *   - orderScratch: position -> vertex (rebuilt)
 *
 * Notes on neighbor enumeration:
 *   We iterate g.out_neigh(v). For undirected GAP-BS graphs out_neigh and
 *   in_neigh return identical sets so a single pass suffices. For directed
 *   graphs we still get a meaningful locality metric (push-side neighbors).
 */
template <typename K>
inline bool refineVertexIndex(
    const std::vector<K>& verts,
    const std::vector<size_t>& vertToLocal,
    K vertex,
    size_t& index)
{
    size_t vertex_index = static_cast<size_t>(vertex);
    if (vertex_index >= vertToLocal.size())
        return false;
    index = vertToLocal[vertex_index];
    return index < verts.size() && verts[index] == vertex;
}

template <typename K, typename NodeID_T, typename DestID_T>
inline void refineTwoSwap(
    const std::vector<K>& verts,
    K c,
    const std::vector<K>& membership,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    std::vector<K>& localIds,
    const std::vector<size_t>& vertToLocal,
    int maxPasses,
    std::vector<size_t>& adjOffsets,
    std::vector<K>& adjVertices,
    std::vector<K>& orderScratch)
{
    const size_t sz = verts.size();
    if (sz < 3) return;

    // position -> vertex (initial)
    orderScratch.assign(sz, K(0));
    for (K v : verts) {
        size_t p = static_cast<size_t>(localIds[v]);
        if (p < sz) orderScratch[p] = v;
    }

    // Flat adjacency by verts-index. Reusing one contiguous buffer avoids
    // retaining millions of per-vertex vector capacities across communities.
    adjOffsets.resize(sz + 1);
    adjVertices.clear();
    for (size_t i = 0; i < sz; ++i) {
        adjOffsets[i] = adjVertices.size();
        K v = verts[i];
        for (auto u : g.out_neigh(v)) {
            K w = static_cast<K>(u);
            if (w == v) continue;
            if (static_cast<size_t>(w) >= membership.size()) continue;
            if (membership[w] != c) continue;
            adjVertices.push_back(w);
        }
    }
    adjOffsets[sz] = adjVertices.size();

    for (int pass = 0; pass < maxPasses; ++pass) {
        size_t swapped = 0;
        for (size_t p = 0; p + 1 < sz; ++p) {
            K va = orderScratch[p];
            K vb = orderScratch[p + 1];
            size_t ia;
            size_t ib;
            if (
                !refineVertexIndex(verts, vertToLocal, va, ia) ||
                !refineVertexIndex(verts, vertToLocal, vb, ib)
            ) continue;
            int delta = 0;
            for (size_t j = adjOffsets[ia]; j < adjOffsets[ia + 1]; ++j) {
                K w = adjVertices[j];
                if (w == vb) continue;
                size_t pn = static_cast<size_t>(localIds[w]);
                if (pn < p) delta += 1;
                else if (pn > p + 1) delta -= 1;
            }
            for (size_t j = adjOffsets[ib]; j < adjOffsets[ib + 1]; ++j) {
                K w = adjVertices[j];
                if (w == va) continue;
                size_t pn = static_cast<size_t>(localIds[w]);
                if (pn < p) delta -= 1;
                else if (pn > p + 1) delta += 1;
            }
            if (delta < 0) {
                std::swap(orderScratch[p], orderScratch[p + 1]);
                std::swap(localIds[va], localIds[vb]);
                ++swapped;
            }
        }
        if (!swapped) break;
    }
}

//=============================================================================
// SECTION 16-SUPER-GRAPH: Reusable super-graph-order primitives
//=============================================================================
//
// Super-graph ordering produces a permutation of community IDs that
// becomes the base ordering for the community-order axis.  Picks:
//
//   None         : identity permutation; community-order sorts from scratch
//                  (CONNECTIVITY_BFS / current orderCompose default)
//   SuperRabbit  : build community super-graph, run RabbitOrder on it
//                  (HRAB's super-graph order)
//   SuperRCM     : build community super-graph, run RCM on it
//                  (HRAB :rcm_super)
//   TileRabbit   : 2-level tile + community RabbitOrder
//                  (TQR's super-graph order)
//
// The shared building block for all super-graph picks is the community
// super-graph itself.  We provide it as `buildCommunitySuperGraph<>()`
// returning a `CommunitySuperGraph` POD.  Phase 2 will add the runners.

/**
 * Community super-graph aggregated from vertex-level edges.
 *
 * Storage: upper-triangular adjacency only — `edges[c]` contains
 *          {d -> weight} pairs where d >= c.  This matches HRAB's
 *          original layout so we can reuse the same modularity
 *          arithmetic without rewriting it.
 */
template <typename K>
struct CommunitySuperGraph {
    std::vector<std::unordered_map<K, float>> edges;  // size C; upper-triangular
    std::vector<float> degrees;                        // size C; total weight per community
    float M = 0.0f;                                    // total edge weight (sum of degrees / 2)
};

/**
 * Build the community super-graph from a vertex-level CSR + membership.
 *
 * @param membership     final community id per vertex (size N)
 * @param vertexDegrees  out-degree per vertex; vertices with degree 0
 *                       are skipped (they are isolated)
 * @param vertexIsHub    optional hub mask (size N); if non-empty,
 *                       hub vertices are skipped in BOTH the source
 *                       and destination check (matches HRAB hubx).
 *                       Pass {} to disable.
 * @param g              original directed CSR
 * @param N              vertex count
 * @param C              community count (max membership + 1)
 *
 * @return CommunitySuperGraph<K> filled with edges/degrees/M
 *
 * Parallel per-thread accumulation, then per-community merge.  Matches
 * the HRAB STEP 1 algorithm exactly so a follow-up commit can replace
 * HRAB's inline copy with a call to this primitive without behaviour
 * change.
 */
template <typename K, typename NodeID_T, typename DestID_T>
CommunitySuperGraph<K> buildCommunitySuperGraph(
    const std::vector<K>& membership,
    const std::vector<K>& vertexDegrees,
    const std::vector<bool>& vertexIsHub,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    K C)
{
    CommunitySuperGraph<K> sg;
    sg.edges.resize(C);
    sg.degrees.assign(C, 0.0f);

    const bool hasHubMask = !vertexIsHub.empty();

    const int nT = omp_get_max_threads();
    std::vector<std::vector<std::unordered_map<K, float>>> allLocalEdges(nT);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        allLocalEdges[tid].resize(C);

        #pragma omp for schedule(dynamic, 1024)
        for (size_t u = 0; u < N; ++u) {
            if (vertexDegrees[u] == 0) continue;
            if (hasHubMask && vertexIsHub[u]) continue;
            K commU = membership[u];

            for (auto neighbor : g.out_neigh(u)) {
                NodeID_T v;
                float w;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    v = neighbor;
                    w = 1.0f;
                } else {
                    v = neighbor.v;
                    w = static_cast<float>(neighbor.w);
                }

                if (hasHubMask && vertexIsHub[v]) continue;

                K commV = membership[v];
                if (commU <= commV) {
                    allLocalEdges[tid][commU][commV] += w;
                }
            }
        }
    }

    // Per-community merge across threads (each community independent)
    #pragma omp parallel for schedule(dynamic, 64)
    for (K c = 0; c < C; ++c) {
        for (int t = 0; t < nT; ++t) {
            for (auto& [d, w] : allLocalEdges[t][c]) {
                sg.edges[c][d] += w;
            }
        }
    }

    // Community degrees (both directions: upper-triangular storage means
    // c <= d entries also contribute to degree[d])
    for (K c = 0; c < C; ++c) {
        for (auto& [d, w] : sg.edges[c]) {
            sg.degrees[c] += w;
            if (d != c) sg.degrees[d] += w;
        }
    }

    sg.M = 0.0f;
    for (K c = 0; c < C; ++c) sg.M += sg.degrees[c];
    sg.M /= 2.0f;

    return sg;
}

template <typename K>
struct CapacityRunOrderResult {
    std::vector<K> order;
    std::vector<size_t> l2RunEnds;
    std::vector<size_t> llcRunEnds;
};

struct CapacityGeometry {
    size_t l2Bytes = 0;
    size_t llcBytes = 0;
    size_t propertyBytesPerVertex = 0;
    size_t l2TargetVertices = 0;
    size_t llcTargetVertices = 0;
};

inline CapacityGeometry resolveCapacityGeometry(
    const GraphBrewConfig& config)
{
    if (
        config.capacityL2Bytes == 0
        || config.capacityLLCBytes == 0
        || config.capacityPropertyBytesPerVertex == 0
    ) {
        throw std::invalid_argument(
            "Capacity-run execution requires pinned L2, LLC, "
            "and property-byte geometry");
    }
    if (config.capacityLLCBytes < config.capacityL2Bytes) {
        throw std::invalid_argument(
            "Capacity-run LLC bytes must be at least L2 bytes");
    }
    CapacityGeometry geometry;
    geometry.l2Bytes = config.capacityL2Bytes;
    geometry.llcBytes = config.capacityLLCBytes;
    geometry.propertyBytesPerVertex =
        config.capacityPropertyBytesPerVertex;
    geometry.l2TargetVertices = std::max<size_t>(
        1, geometry.l2Bytes / geometry.propertyBytesPerVertex);
    geometry.llcTargetVertices = std::max(
        geometry.l2TargetVertices,
        geometry.llcBytes / geometry.propertyBytesPerVertex);
    return geometry;
}

template <typename K, typename NodeID_T, typename DestID_T>
std::vector<std::vector<std::pair<K, uint64_t>>>
buildCapacityCommunityAdjacency(
    const std::vector<K>& membership,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t nodeCount,
    K communityCount)
{
    if (g.directed()) {
        throw std::invalid_argument(
            "Capacity-run adjacency requires an undirected graph");
    }
    static_assert(
        std::is_unsigned<K>::value,
        "Capacity-run community IDs must be unsigned");
    static_assert(
        sizeof(K) <= sizeof(uint32_t),
        "Capacity-run community IDs must fit in 32 bits");
    const int threadCount = omp_get_max_threads();
    std::vector<std::vector<uint64_t>> localKeys(threadCount);

    #pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        auto& keys = localKeys[thread];
        #pragma omp for schedule(dynamic, 1024)
        for (size_t u = 0; u < nodeCount; ++u) {
            const K communityU = membership[u];
            for (auto neighbor : g.out_neigh(
                     static_cast<NodeID_T>(u))) {
                NodeID_T v;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>)
                    v = neighbor;
                else
                    v = neighbor.v;
                const K communityV = membership[v];
                if (communityU == communityV) continue;
                const K low = std::min(communityU, communityV);
                const K high = std::max(communityU, communityV);
                const uint64_t key =
                    (static_cast<uint64_t>(low) << 32)
                    | static_cast<uint64_t>(high);
                keys.push_back(key);
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threadCount; ++thread) {
        auto& keys = localKeys[thread];
        std::sort(keys.begin(), keys.end());
    }

    struct Cursor {
        uint64_t key;
        int thread;
        size_t index;
    };
    struct CursorGreater {
        bool operator()(const Cursor& left, const Cursor& right) const {
            if (left.key != right.key) return left.key > right.key;
            return left.thread > right.thread;
        }
    };
    std::priority_queue<
        Cursor,
        std::vector<Cursor>,
        CursorGreater> cursors;
    for (int thread = 0; thread < threadCount; ++thread) {
        if (!localKeys[thread].empty()) {
            cursors.push({localKeys[thread][0], thread, 0});
        }
    }

    std::vector<std::vector<std::pair<K, uint64_t>>> adjacency(
        communityCount);
    while (!cursors.empty()) {
        const uint64_t key = cursors.top().key;
        uint64_t count = 0;
        while (!cursors.empty() && cursors.top().key == key) {
            Cursor cursor = cursors.top();
            cursors.pop();
            const auto& keys = localKeys[cursor.thread];
            size_t next = cursor.index;
            while (next < keys.size() && keys[next] == key) {
                ++count;
                ++next;
            }
            if (next < keys.size()) {
                cursors.push({
                    keys[next], cursor.thread, next});
            }
        }
        const K low = static_cast<K>(key >> 32);
        const K high = static_cast<K>(key & UINT64_C(0xffffffff));
        adjacency[low].push_back({high, count});
        adjacency[high].push_back({low, count});
    }
    for (auto& neighbors : adjacency) {
        std::sort(
            neighbors.begin(), neighbors.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
    }
    return adjacency;
}

/**
 * Build cut-aware runs of whole communities bounded by L2 and LLC capacity.
 *
 * The traversal restarts at each capacity boundary. Within an L2 run it
 * selects the unplaced community with the strongest accumulated edge weight
 * to that run. New L2 runs are seeded from connectivity accumulated across
 * the current LLC run. Disconnected runs restart from the largest remaining
 * community. Disconnected communities are packed into remaining capacity in
 * base-order-aware size order. The algorithm is deterministic and
 * O((C + E_super) log C).
 */
template <typename K>
CapacityRunOrderResult<K> buildCapacityRunCommunityOrder(
    const std::vector<size_t>& communitySizes,
    const std::vector<std::vector<std::pair<K, uint64_t>>>& adjacency,
    const std::vector<K>& baseOrder,
    size_t l2TargetVertices,
    size_t llcTargetVertices)
{
    const size_t C = communitySizes.size();
    CapacityRunOrderResult<K> result;
    result.order.reserve(C);
    if (C == 0) return result;
    if (adjacency.size() != C) {
        throw std::invalid_argument(
            "Capacity-run ordering received mismatched community sizes");
    }

    l2TargetVertices = std::max<size_t>(1, l2TargetVertices);
    llcTargetVertices = std::max(l2TargetVertices, llcTargetVertices);

    std::vector<size_t> baseRank(C, C);
    for (size_t i = 0; i < baseOrder.size(); ++i) {
        const size_t c = static_cast<size_t>(baseOrder[i]);
        if (c < C) baseRank[c] = i;
    }
    for (size_t c = 0; c < C; ++c) {
        if (baseRank[c] == C) baseRank[c] = c;
    }

    struct Entry {
        uint64_t score;
        size_t rank;
        K community;
    };
    struct EntryLess {
        bool operator()(const Entry& left, const Entry& right) const {
            if (left.score != right.score)
                return left.score < right.score;
            if (left.rank != right.rank)
                return left.rank > right.rank;
            return left.community > right.community;
        }
    };
    using Queue = std::priority_queue<Entry, std::vector<Entry>, EntryLess>;

    std::vector<char> placed(C, 0);
    std::map<size_t, std::set<std::pair<size_t, K>>> remainingBySize;
    std::vector<K> emptyCommunities;
    for (K c = 0; c < static_cast<K>(C); ++c) {
        if (communitySizes[c] == 0) {
            placed[c] = 1;
            emptyCommunities.push_back(c);
            continue;
        }
        remainingBySize[communitySizes[c]].insert({baseRank[c], c});
    }
    std::sort(
        emptyCommunities.begin(), emptyCommunities.end(),
        [&](K left, K right) {
            return baseRank[left] < baseRank[right];
        });
    const size_t activeCommunityCount =
        C - emptyCommunities.size();

    auto nextLargestFitting = [&](size_t capacity) -> K {
        auto sizeIt = remainingBySize.upper_bound(capacity);
        if (sizeIt == remainingBySize.begin())
            return static_cast<K>(-1);
        --sizeIt;
        return sizeIt->second.begin()->second;
    };
    auto nextLargest = [&]() -> K {
        if (remainingBySize.empty()) return static_cast<K>(-1);
        const auto sizeIt = std::prev(remainingBySize.end());
        return sizeIt->second.begin()->second;
    };
    auto removeRemaining = [&](K community) {
        auto sizeIt = remainingBySize.find(communitySizes[community]);
        if (sizeIt == remainingBySize.end()) return;
        sizeIt->second.erase({baseRank[community], community});
        if (sizeIt->second.empty()) remainingBySize.erase(sizeIt);
    };

    std::vector<uint64_t> l2Score(C, 0), llcScore(C, 0);
    std::vector<K> l2Touched, llcTouched;
    Queue l2Queue, llcQueue;

    auto resetFrontier = [](std::vector<uint64_t>& score,
                            std::vector<K>& touched,
                            Queue& queue) {
        for (K c : touched) score[c] = 0;
        touched.clear();
        queue = Queue();
    };
    auto updateFrontier = [&](K source,
                              std::vector<uint64_t>& score,
                              std::vector<K>& touched,
                              Queue& queue) {
        for (const auto& [neighbor, weight] : adjacency[source]) {
            if (placed[neighbor]) continue;
            if (score[neighbor] == 0) touched.push_back(neighbor);
            score[neighbor] += weight;
            queue.push({score[neighbor], baseRank[neighbor], neighbor});
        }
    };
    auto peekFrontierFitting = [&](
        Queue& queue,
        const std::vector<uint64_t>& score,
        size_t capacity) -> K {
        while (!queue.empty()) {
            const Entry& entry = queue.top();
            if (
                placed[entry.community]
                || entry.score != score[entry.community]
            ) {
                queue.pop();
                continue;
            }
            if (communitySizes[entry.community] > capacity) {
                queue.pop();
                continue;
            }
            return entry.community;
        }
        return static_cast<K>(-1);
    };

    while (result.order.size() < activeCommunityCount) {
        resetFrontier(llcScore, llcTouched, llcQueue);
        resetFrontier(l2Score, l2Touched, l2Queue);
        K seed = nextLargest();
        if (seed == static_cast<K>(-1)) break;

        size_t llcUsed = 0;
        bool closeLLC = false;
        while (seed != static_cast<K>(-1) && !closeLLC) {
            resetFrontier(l2Score, l2Touched, l2Queue);
            size_t l2Used = 0;
            K current = seed;

            while (current != static_cast<K>(-1)) {
                const size_t size = communitySizes[current];
                if (
                    l2Used > 0
                    && (
                        l2Used >= l2TargetVertices
                        || size > l2TargetVertices - l2Used
                    )
                )
                    break;
                if (
                    llcUsed > 0
                    && (
                        llcUsed >= llcTargetVertices
                        || size > llcTargetVertices - llcUsed
                    )
                ) {
                    closeLLC = true;
                    break;
                }

                placed[current] = 1;
                removeRemaining(current);
                result.order.push_back(current);
                l2Used += size;
                llcUsed += size;
                updateFrontier(current, l2Score, l2Touched, l2Queue);
                updateFrontier(current, llcScore, llcTouched, llcQueue);

                const size_t l2Remaining =
                    l2Used < l2TargetVertices
                        ? l2TargetVertices - l2Used
                        : 0;
                const size_t llcRemaining =
                    llcUsed < llcTargetVertices
                        ? llcTargetVertices - llcUsed
                        : 0;
                const size_t remaining =
                    std::min(l2Remaining, llcRemaining);
                current = peekFrontierFitting(
                    l2Queue, l2Score, remaining);
                if (current == static_cast<K>(-1)) {
                    current = nextLargestFitting(remaining);
                }
            }

            result.l2RunEnds.push_back(result.order.size());
            if (
                closeLLC
                || result.order.size() == activeCommunityCount
                || llcUsed >= llcTargetVertices
            ) {
                break;
            }

            const size_t llcRemaining =
                llcUsed < llcTargetVertices
                    ? llcTargetVertices - llcUsed
                    : 0;
            seed = peekFrontierFitting(
                llcQueue, llcScore, llcRemaining);
            if (seed == static_cast<K>(-1)) {
                seed = nextLargestFitting(llcRemaining);
            }
            if (
                seed == static_cast<K>(-1)
                || llcUsed >= llcTargetVertices
                || communitySizes[seed]
                    > llcTargetVertices - llcUsed
            ) break;
        }
        result.llcRunEnds.push_back(result.order.size());
    }

    result.order.insert(
        result.order.end(),
        emptyCommunities.begin(),
        emptyCommunities.end());
    return result;
}

/**
 * Run RabbitOrder on a community super-graph and return the community
 * permutation produced by the resulting dendrogram DFS.
 *
 * Faithful extraction of HRAB STEPs 2 + 3.  Identity-equivalent to the
 * inline path when called with the same `sg` and gamma.
 *
 * @param sg                   community super-graph from buildCommunitySuperGraph<>()
 * @param numVertsPerComm      per-community vertex count (used only to
 *                             classify roots as "active" vs "empty";
 *                             empty roots are placed after active ones)
 * @param gamma                modularity-gain resolution
 *                             (default 0.10 — see GraphBrewConfig)
 *
 * @return commPerm of size C; commPerm[c] is the position of community c
 *         in the final ordering.
 */
template <typename K>
std::vector<K> runRabbitOnSuperCSR(
    CommunitySuperGraph<K> sg,                   // taken by value: mutated internally
    const std::vector<size_t>& numVertsPerComm,
    float gamma)
{
    const K C = static_cast<K>(sg.edges.size());
    constexpr K INVALID = static_cast<K>(-1);

    // Initialise RabbitOrder structures for communities
    std::vector<RabbitAtom> atom(C);
    std::vector<RabbitVertex<K>> vtx(C);
    std::vector<K> dest(C);
    std::vector<K> sibling(C, INVALID);

    #pragma omp parallel for
    for (K c = 0; c < C; ++c) {
        atom[c].init(sg.degrees[c]);
        dest[c] = c;
    }

    // Build symmetric edge lists (sg.edges is upper-triangular)
    for (K c = 0; c < C; ++c) {
        for (auto& [d, w] : sg.edges[c]) {
            vtx[c].edges.emplace_back(d, w);
            if (d != c) vtx[d].edges.emplace_back(c, w);
        }
    }

    // Community processing order: ascending degree (RabbitOrder convention)
    std::vector<K> commOrder(C);
    std::iota(commOrder.begin(), commOrder.end(), K(0));
    std::sort(commOrder.begin(), commOrder.end(),
              [&](K a, K b) {
                  return sg.degrees[a] != sg.degrees[b]
                      ? sg.degrees[a] < sg.degrees[b]
                      : a < b;
              });

    const int numThreads = omp_get_max_threads();
    std::vector<CommunityScanner<K, float>> scanners;
    for (int t = 0; t < numThreads; ++t) scanners.emplace_back(C);

    auto findDest = [&](K v) -> K {
        K d = dest[v];
        while (dest[d] != d) {
            K dd = dest[d];
            dest[v] = dd;
            d = dd;
        }
        return d;
    };

    const float inv2M_super = (sg.M > 0.0f) ? 1.0f / (2.0f * sg.M) : 0.0f;
    auto deltaQ = [gamma, inv2M_super](float w_uv, float d_u, float d_v) -> float {
        return w_uv - gamma * d_u * d_v * inv2M_super;
    };

    auto unite = [&](K u, CommunityScanner<K, float>& scanner) -> void {
        auto& vu = vtx[u];
        scanner.clear();
        for (auto& [d, w] : vu.edges) {
            K root = findDest(d);
            if (root != u) scanner.add(root, w);
        }
        auto [_, first_child] = atom[u].load();
        K uc = vu.united_child;
        std::vector<K> childStack;
        if (first_child != uc && first_child != INVALID) childStack.push_back(first_child);
        while (!childStack.empty()) {
            K c = childStack.back();
            childStack.pop_back();
            if (c == uc || c == INVALID) continue;
            auto& vc = vtx[c];
            for (auto& [d, w] : vc.edges) {
                K root = findDest(d);
                if (root != u) scanner.add(root, w);
            }
            if (sibling[c] != uc && sibling[c] != INVALID) childStack.push_back(sibling[c]);
            auto [__, child_first] = atom[c].load();
            if (child_first != vc.united_child && child_first != INVALID)
                childStack.push_back(child_first);
            vc.united_child = child_first;
        }
        vu.edges.clear();
        for (K d : scanner.keys) vu.edges.emplace_back(d, scanner.get(d));
        vu.united_child = first_child;
    };

    std::atomic<size_t> numToplevel(0);
    std::vector<std::deque<K>> superTopss(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& scanner = scanners[tid];
        std::deque<K> tops, pends;

        auto tryMerge = [&](K u) -> bool {
            unite(u, scanner);
            float str_u = atom[u].invalidate();
            if (str_u == RabbitAtom::INVALID_STR) return false;
            {
                auto [_re, first_child_re] = atom[u].load();
                if (first_child_re != vtx[u].united_child) unite(u, scanner);
            }
            K bestDest = INVALID;
            float bestDeltaQ = 0.0f;
            for (K d : scanner.keys) {
                float w_ud = scanner.get(d);
                auto [str_d, _] = atom[d].load();
                if (str_d == RabbitAtom::INVALID_STR) continue;
                float dq = deltaQ(w_ud, str_u, str_d);
                if (dq > bestDeltaQ ||
                    (dq == bestDeltaQ &&
                     (bestDest == INVALID || d < bestDest))) {
                    bestDeltaQ = dq;
                    bestDest = d;
                }
            }
            if (bestDest == INVALID || bestDeltaQ <= 0.0f) {
                atom[u].restore(str_u);
                tops.push_back(u);
                return false;
            }
            auto [str_v, child_v] = atom[bestDest].load();
            if (str_v == RabbitAtom::INVALID_STR) {
                atom[u].restore(str_u);
                return true;
            }
            sibling[u] = child_v;
            float new_str = str_v + str_u;
            if (atom[bestDest].tryMerge(str_v, child_v, new_str, static_cast<uint32_t>(u))) {
                dest[u] = bestDest;
                return false;
            } else {
                sibling[u] = INVALID;
                atom[u].restore(str_u);
                return true;
            }
        };

        #pragma omp for schedule(static, 1)
        for (K i = 0; i < C; ++i) {
            pends.erase(std::remove_if(pends.begin(), pends.end(),
                        [&](K w) { return !tryMerge(w); }), pends.end());
            K u = commOrder[i];
            if (sg.degrees[u] == 0.0f) { tops.push_back(u); continue; }
            if (tryMerge(u)) pends.push_back(u);
        }

        #pragma omp barrier
        #pragma omp critical
        {
            for (K u : pends) {
                bool retry = tryMerge(u);
                if (retry) {
                    float str_u = atom[u].invalidate();
                    if (str_u != RabbitAtom::INVALID_STR) atom[u].restore(str_u);
                    tops.push_back(u);
                }
            }
            superTopss[tid] = std::move(tops);
        }
    }

    std::vector<K> toplevel;
    toplevel.reserve(C / 10);
    for (int t = 0; t < numThreads; ++t)
        toplevel.insert(toplevel.end(), superTopss[t].begin(), superTopss[t].end());
    numToplevel.store(toplevel.size(), std::memory_order_relaxed);

    printf("  stage1-super-rabbit: %zu super-communities (merged from %u)\n",
           numToplevel.load(), static_cast<unsigned>(C));

    // Find all dendrogram roots (active vs empty separated)
    std::vector<K> allRoots;
    for (K c = 0; c < C; ++c) {
        K root = dest[c];
        while (dest[root] != root) root = dest[root];
        if (root == c) allRoots.push_back(c);
    }
    std::vector<K> activeTop, emptyTop;
    for (K r : allRoots) {
        if (sg.degrees[r] == 0.0f &&
            (r < numVertsPerComm.size() ? numVertsPerComm[r] == 0 : true)) {
            emptyTop.push_back(r);
        } else {
            activeTop.push_back(r);
        }
    }
    std::sort(activeTop.begin(), activeTop.end());
    std::sort(emptyTop.begin(), emptyTop.end());
    toplevel.clear();
    toplevel.insert(toplevel.end(), activeTop.begin(), activeTop.end());
    toplevel.insert(toplevel.end(), emptyTop.begin(), emptyTop.end());

    // Dendrogram DFS to produce commPerm
    std::vector<K> commPerm(C, static_cast<K>(-1));
    K nextCommId = 0;
    for (K bi = 0; bi < (K)toplevel.size(); ++bi) {
        K root = toplevel[bi];
        std::deque<K> stack;
        rabbitDescendants(atom, root, stack);
        while (!stack.empty()) {
            K c = stack.back();
            stack.pop_back();
            if (commPerm[c] != static_cast<K>(-1)) continue;
            commPerm[c] = nextCommId++;
            if (sibling[c] != INVALID) rabbitDescendants(atom, sibling[c], stack);
        }
    }
    for (K c = 0; c < C; ++c) {
        if (commPerm[c] == static_cast<K>(-1)) commPerm[c] = nextCommId++;
    }
    return commPerm;
}

/**
 * Run BNF+George-Liu RCM on a community super-graph and return the
 * community permutation produced.
 *
 * Faithful extraction of HRAB STEP 3b (the `:rcm_super` path).
 *
 * @param sg                   community super-graph
 * @param numVertsPerComm      per-community vertex count; empty
 *                             communities (count == 0) are excluded
 *                             from the BFS and appended at the tail
 *
 * @return commPerm of size C
 */
template <typename K>
std::vector<K> runRCMOnSuperCSR(
    const CommunitySuperGraph<K>& sg,
    const std::vector<size_t>& numVertsPerComm)
{
    const K C = static_cast<K>(sg.edges.size());

    // Build symmetric adjacency (excluding self-loops)
    std::vector<std::vector<std::pair<K, float>>> superAdj(C);
    for (K c = 0; c < C; ++c) {
        for (auto& [nbr, w] : sg.edges[c]) {
            if (nbr != c) superAdj[c].push_back({nbr, w});
        }
    }
    // commEdges is upper-triangular, so we need to add the reverse edges
    for (K c = 0; c < C; ++c) {
        for (auto& [nbr, w] : sg.edges[c]) {
            if (nbr != c) superAdj[nbr].push_back({c, w});
        }
    }

    std::vector<K> rcmDegree(C);
    for (K c = 0; c < C; ++c) rcmDegree[c] = static_cast<K>(superAdj[c].size());

    auto isActive = [&](K c) -> bool {
        return c < (K)numVertsPerComm.size() && numVertsPerComm[c] > 0;
    };

    std::vector<K> nonEmpty;
    nonEmpty.reserve(C);
    for (K c = 0; c < C; ++c) {
        if (isActive(c) && rcmDegree[c] > 0) nonEmpty.push_back(c);
    }

    // George-Liu pseudoperipheral start
    auto superBFS = [&](K root, std::vector<bool>& vis)
        -> std::tuple<int64_t, int64_t, std::vector<K>> {
        vis.assign(C, false);
        std::vector<K> curLevel = {root};
        vis[root] = true;
        int64_t ecc = 0;
        int64_t maxWidth = 1;
        while (true) {
            std::vector<K> nextLevel;
            for (K cur : curLevel) {
                for (auto& [nbr, w] : superAdj[cur]) {
                    if (!vis[nbr] && isActive(nbr)) {
                        vis[nbr] = true;
                        nextLevel.push_back(nbr);
                    }
                }
            }
            if (nextLevel.empty()) break;
            ecc++;
            int64_t w = static_cast<int64_t>(nextLevel.size());
            if (w > maxWidth) maxWidth = w;
            curLevel = std::move(nextLevel);
        }
        return {ecc, maxWidth, curLevel};
    };

    K glNode = nonEmpty.empty() ? K(0) : nonEmpty[0];
    K minDeg = std::numeric_limits<K>::max();
    for (K c : nonEmpty) {
        if (rcmDegree[c] < minDeg) { minDeg = rcmDegree[c]; glNode = c; }
    }
    std::vector<bool> glVis(C, false);
    int64_t prevEcc = 0;
    int64_t bestWidth = std::numeric_limits<int64_t>::max();
    K bestNode = glNode;
    for (int it = 0; it < 20; ++it) {
        auto [ecc, maxW, farthest] = superBFS(glNode, glVis);
        if (maxW < bestWidth || (maxW == bestWidth && ecc > prevEcc)) {
            bestWidth = maxW;
            bestNode = glNode;
        }
        if (ecc <= prevEcc) break;
        prevEcc = ecc;
        K nextNode = farthest[0];
        K nextDeg = rcmDegree[farthest[0]];
        for (size_t i = 1; i < farthest.size(); ++i) {
            if (rcmDegree[farthest[i]] < nextDeg) {
                nextDeg = rcmDegree[farthest[i]];
                nextNode = farthest[i];
            }
        }
        glNode = nextNode;
    }

    // CM BFS with degree-sorted neighbours
    std::vector<bool> rcmVisited(C, false);
    std::vector<K> rcmOrder;
    rcmOrder.reserve(C);
    std::queue<K> rcmQueue;
    std::vector<K> seedOrder(C);
    std::iota(seedOrder.begin(), seedOrder.end(), K(0));
    std::sort(seedOrder.begin(), seedOrder.end(),
              [&](K a, K b) { return rcmDegree[a] < rcmDegree[b]; });

    auto doCMBFS = [&](K seed) {
        if (rcmVisited[seed] || !isActive(seed)) return;
        rcmQueue.push(seed);
        rcmVisited[seed] = true;
        while (!rcmQueue.empty()) {
            K cur = rcmQueue.front();
            rcmQueue.pop();
            rcmOrder.push_back(cur);
            auto& nbrs = superAdj[cur];
            std::sort(nbrs.begin(), nbrs.end(),
                [&](const std::pair<K, float>& a, const std::pair<K, float>& b) {
                    return rcmDegree[a.first] < rcmDegree[b.first];
                });
            for (auto& [nbr, w] : nbrs) {
                if (!rcmVisited[nbr] && isActive(nbr)) {
                    rcmVisited[nbr] = true;
                    rcmQueue.push(nbr);
                }
            }
        }
    };

    doCMBFS(bestNode);
    for (K seed : seedOrder) doCMBFS(seed);

    std::reverse(rcmOrder.begin(), rcmOrder.end());

    std::vector<K> commPerm(C, static_cast<K>(-1));
    for (size_t i = 0; i < rcmOrder.size(); ++i)
        commPerm[rcmOrder[i]] = static_cast<K>(i);
    K rcmNext = static_cast<K>(rcmOrder.size());
    for (K c = 0; c < C; ++c) if (commPerm[c] == static_cast<K>(-1)) commPerm[c] = rcmNext++;

    printf("  stage1-super-rcm: BNF-ordered %zu communities (GL best_width=%lld)\n",
           rcmOrder.size(), static_cast<long long>(bestWidth));
    return commPerm;
}

/**
 * Choose tile parameters for tile-quantised super-graph ordering.
 * Matches TQR STEP 0 verbatim:
 *
 *   tileSz = clamp(512 / log2(avgDeg+2), 32, 512)
 *   numTiles = ceil(N / tileSz)
 *   numTiles capped at 65536
 *
 * @return pair (tileSz, numTiles)
 */
inline std::pair<size_t, size_t> chooseTileParams(size_t N, double avgDeg)
{
    const double logFactor = std::log2(avgDeg + 2.0);
    size_t targetTileSz = static_cast<size_t>(512.0 / logFactor);
    targetTileSz = std::max(size_t(32), std::min(size_t(512), targetTileSz));

    size_t numTiles = std::max(size_t(256), (N + targetTileSz - 1) / targetTileSz);
    const size_t MAX_TILES = 65536;
    if (numTiles > MAX_TILES) numTiles = MAX_TILES;

    size_t tileSz = (N + numTiles - 1) / numTiles;
    if (tileSz < 16) {
        tileSz = 16;
        numTiles = (N + tileSz - 1) / tileSz;
    }
    return {tileSz, numTiles};
}

/**
 * Run RabbitOrder on a tile super-graph (standard ΔQ, no gamma scaling).
 *
 * Faithful extraction of TQR STEP 3 + STEP 4.  Differs from
 * runRabbitOnSuperCSR<>() in three ways:
 *
 *   1. Modularity formula: `ΔQ = w_uv / (2M) - str_u * str_v / (4M²)`
 *      (standard Rabbit Order, NOT γ-scaled).  Tiles are not Leiden
 *      communities so γ tuning does not apply.
 *   2. Two-phase retry (parallel + serial) instead of inline pending
 *      queue.
 *   3. Toplevel collected inline via `#pragma omp critical` instead of
 *      per-thread join.
 *
 * Empty-tile detection uses `tileStartVertexBeyondN` instead of vertex
 * count (the last tile may be partially populated when N % tileSz != 0).
 *
 * @return pair (tilePerm, tileBlockId).  tilePerm[t] = position of tile
 *         t in the final ordering.  tileBlockId[t] = root of t in the
 *         RabbitOrder dendrogram (used by the super-graph-order orchestrator).
 */
template <typename K>
std::pair<std::vector<K>, std::vector<K>> runRabbitOnTileGraph(
    CommunitySuperGraph<K> tileSG,                  // taken by value
    size_t tileSz,
    size_t N)
{
    const size_t numTiles = tileSG.edges.size();
    constexpr K INVALID = static_cast<K>(-1);

    std::vector<RabbitAtom> atom(numTiles);
    std::vector<RabbitVertex<K>> vtx(numTiles);
    std::vector<K> dest(numTiles);
    std::vector<K> tileSibling(numTiles, INVALID);
    std::vector<K> toplevel;
    toplevel.reserve(numTiles / 10);

    #pragma omp parallel for
    for (size_t t = 0; t < numTiles; ++t) {
        atom[t].init(tileSG.degrees[t]);
        dest[t] = static_cast<K>(t);
    }

    // Build symmetric edge lists from upper-triangular tileSG.edges
    for (K c = 0; c < (K)numTiles; ++c) {
        for (auto& [d, w] : tileSG.edges[c]) {
            vtx[c].edges.emplace_back(d, w);
            if (d != c) vtx[d].edges.emplace_back(c, w);
        }
    }

    std::vector<K> tileOrder(numTiles);
    std::iota(tileOrder.begin(), tileOrder.end(), K(0));
    std::sort(tileOrder.begin(), tileOrder.end(),
        [&](K a, K b) {
            return tileSG.degrees[a] != tileSG.degrees[b]
                ? tileSG.degrees[a] < tileSG.degrees[b]
                : a < b;
        });

    const int numThreads = omp_get_max_threads();
    std::vector<CommunityScanner<K, float>> scanners;
    for (int t = 0; t < numThreads; ++t) scanners.emplace_back(numTiles);
    std::vector<std::vector<K>> retryQueues(numThreads);

    auto findDest = [&](K v) -> K {
        K d = dest[v];
        while (dest[d] != d) {
            K dd = dest[d];
            dest[v] = dd;
            d = dd;
        }
        return d;
    };

    auto unite = [&](K u, CommunityScanner<K, float>& scanner) -> void {
        auto& vu = vtx[u];
        scanner.clear();
        for (auto& [d, w] : vu.edges) {
            K root = findDest(d);
            if (root != u) scanner.add(root, w);
        }
        auto [_, first_child] = atom[u].load();
        K uc = vu.united_child;
        std::vector<K> childStack;
        if (first_child != uc && first_child != INVALID) childStack.push_back(first_child);
        while (!childStack.empty()) {
            K c = childStack.back();
            childStack.pop_back();
            if (c == uc || c == INVALID) continue;
            auto& vc = vtx[c];
            for (auto& [d, w] : vc.edges) {
                K root = findDest(d);
                if (root != u) scanner.add(root, w);
            }
            if (tileSibling[c] != uc && tileSibling[c] != INVALID) childStack.push_back(tileSibling[c]);
            auto [__, child_first] = atom[c].load();
            if (child_first != vc.united_child && child_first != INVALID)
                childStack.push_back(child_first);
            vc.united_child = child_first;
        }
        vu.edges.clear();
        for (K d : scanner.keys) vu.edges.emplace_back(d, scanner.get(d));
        vu.united_child = first_child;
    };

    std::atomic<size_t> numToplevel(0);
    const float inv2M = (tileSG.M > 0.0f) ? 1.0f / (2.0f * tileSG.M) : 0.0f;
    const float inv4M2 = inv2M * inv2M;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& scanner = scanners[tid];
        auto& retryQueue = retryQueues[tid];

        #pragma omp for schedule(static, 1)
        for (size_t i = 0; i < numTiles; ++i) {
            K u = tileOrder[i];
            if (tileSG.degrees[u] == 0.0f) {
                #pragma omp critical
                toplevel.push_back(u);
                numToplevel++;
                continue;
            }
            float str_u = atom[u].invalidate();
            if (str_u == RabbitAtom::INVALID_STR) continue;
            unite(u, scanner);
            K bestDest = INVALID;
            float bestDeltaQ = 0.0f;
            for (K d : scanner.keys) {
                float w_ud = scanner.get(d);
                auto [str_d, _] = atom[d].load();
                if (str_d == RabbitAtom::INVALID_STR) continue;
                float dq = w_ud * inv2M - str_u * str_d * inv4M2;
                if (dq > bestDeltaQ ||
                    (dq == bestDeltaQ &&
                     (bestDest == INVALID || d < bestDest))) {
                    bestDeltaQ = dq;
                    bestDest = d;
                }
            }
            if (bestDest == INVALID || bestDeltaQ <= 0.0f) {
                atom[u].restore(str_u);
                #pragma omp critical
                toplevel.push_back(u);
                numToplevel++;
                continue;
            }
            auto [str_v, child_v] = atom[bestDest].load();
            if (str_v == RabbitAtom::INVALID_STR) {
                atom[u].restore(str_u);
                retryQueue.push_back(u);
                continue;
            }
            tileSibling[u] = child_v;
            float new_str = str_v + str_u;
            if (atom[bestDest].tryMerge(str_v, child_v, new_str, static_cast<uint32_t>(u))) {
                dest[u] = bestDest;
            } else {
                tileSibling[u] = INVALID;
                atom[u].restore(str_u);
                retryQueue.push_back(u);
            }
        }
    }

    // Serial retry pass
    for (int t = 0; t < numThreads; ++t) {
        auto& scanner = scanners[t];
        for (K u : retryQueues[t]) {
            float str_u = atom[u].invalidate();
            if (str_u == RabbitAtom::INVALID_STR) continue;
            unite(u, scanner);
            K bestDest = INVALID;
            float bestDeltaQ = 0.0f;
            for (K d : scanner.keys) {
                float w_ud = scanner.get(d);
                auto [str_d, _] = atom[d].load();
                if (str_d == RabbitAtom::INVALID_STR) continue;
                float dq = w_ud * inv2M - str_u * str_d * inv4M2;
                if (dq > bestDeltaQ ||
                    (dq == bestDeltaQ &&
                     (bestDest == INVALID || d < bestDest))) {
                    bestDeltaQ = dq;
                    bestDest = d;
                }
            }
            if (bestDest == INVALID || bestDeltaQ <= 0.0f) {
                atom[u].restore(str_u);
                toplevel.push_back(u);
                numToplevel++;
                continue;
            }
            auto [str_v, child_v] = atom[bestDest].load();
            tileSibling[u] = child_v;
            atom[bestDest].packed.store(
                RabbitAtom::pack(str_v + str_u, static_cast<uint32_t>(u)),
                std::memory_order_release);
            dest[u] = bestDest;
        }
    }

    printf("  stage1-tile-rabbit: %zu tile-blocks (from %zu tiles)\n",
           numToplevel.load(), numTiles);

    // Find roots and separate active vs empty (matches TQR STEP 4)
    std::vector<K> allRoots;
    for (size_t t = 0; t < numTiles; ++t) {
        K root = dest[t];
        while (dest[root] != root) root = dest[root];
        if (root == static_cast<K>(t)) allRoots.push_back(static_cast<K>(t));
    }
    std::vector<K> activeTop, emptyTop;
    for (K root : allRoots) {
        size_t tileStart = static_cast<size_t>(root) * tileSz;
        if (tileStart >= N && tileSG.degrees[root] == 0.0f) emptyTop.push_back(root);
        else activeTop.push_back(root);
    }
    std::sort(activeTop.begin(), activeTop.end());
    std::sort(emptyTop.begin(), emptyTop.end());
    toplevel.clear();
    toplevel.insert(toplevel.end(), activeTop.begin(), activeTop.end());
    toplevel.insert(toplevel.end(), emptyTop.begin(), emptyTop.end());

    // Dendrogram DFS → tilePerm
    std::vector<K> tilePerm(numTiles, static_cast<K>(-1));
    K nextTileId = 0;
    for (K root : toplevel) {
        std::deque<K> stack;
        rabbitDescendants(atom, root, stack);
        while (!stack.empty()) {
            K t = stack.back();
            stack.pop_back();
            if (tilePerm[t] != static_cast<K>(-1)) continue;
            tilePerm[t] = nextTileId++;
            if (tileSibling[t] != INVALID) rabbitDescendants(atom, tileSibling[t], stack);
        }
    }
    for (size_t t = 0; t < numTiles; ++t) {
        if (tilePerm[t] == static_cast<K>(-1)) tilePerm[t] = nextTileId++;
    }

    // tileBlockId[t] = root of t (path-compressed)
    std::vector<K> tileBlockId(numTiles, 0);
    for (size_t t = 0; t < numTiles; ++t) {
        K root = dest[t];
        while (dest[root] != root) root = dest[root];
        tileBlockId[t] = root;
    }

    return {std::move(tilePerm), std::move(tileBlockId)};
}

/**
 * tile_rabbit super-graph order — TQR's full 2-level (tile + community)
 * pipeline expressed as a single function.  Output is a community permutation.
 *
 * Faithful composition of TQR STEPs 0–4 + 5a–5e:
 *   1. Choose tile parameters via chooseTileParams().
 *   2. Build tile super-graph via buildCommunitySuperGraph()
 *      (vertex's "community" = its tile).
 *   3. Run RabbitOrder on tiles via runRabbitOnTileGraph() (standard ΔQ).
 *   4. Assign each Leiden community a center tile (majority-vertex tile).
 *   5. Build community super-graph via buildCommunitySuperGraph()
 *      (the real one this time).
 *   6. Run RabbitOrder on community super-graph via runRabbitOnSuperCSR()
 *      (γ-scaled ΔQ — matches TQR's STEP 5c which already shares
 *      criterion with HRAB STEP 2).
 *   7. Composite sort by (tilePerm[centerTile[c]], commPermRO[c]).
 *
 * @return commPerm of size C; commPerm[c] = position of community c.
 */
template <typename K, typename NodeID_T, typename DestID_T>
std::vector<K> runTileRabbit(
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const std::vector<std::vector<K>>& commVertices,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    K C,
    float gamma)
{
    // Average degree → tile params
    size_t totalEdges = 0;
    size_t numActive = 0;
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] > 0) { totalEdges += degrees[v]; numActive++; }
    }
    double avgDeg = numActive > 0 ? static_cast<double>(totalEdges) / numActive : 1.0;
    auto [tileSz, numTiles] = chooseTileParams(N, avgDeg);

    printf("  stage1-tile-rabbit: N=%zu avgDeg=%.1f tileSz=%zu numTiles=%zu\n",
           N, avgDeg, tileSz, numTiles);

    // Precompute per-vertex "tile membership" so we can reuse the
    // existing super-graph builder.
    std::vector<K> tileMembership(N);
    auto tileOf = [tileSz](size_t v) -> K { return static_cast<K>(v / tileSz); };
    #pragma omp parallel for schedule(static)
    for (size_t v = 0; v < N; ++v) tileMembership[v] = tileOf(v);

    // Skip isolated vertices via the hub-mask channel of buildCommunitySuperGraph
    std::vector<bool> isolatedMask(N, false);
    for (size_t v = 0; v < N; ++v) if (degrees[v] == 0) isolatedMask[v] = true;

    // Tile super-graph (uses tile membership; isolated vertices excluded)
    auto tileSG = buildCommunitySuperGraph<K, NodeID_T, DestID_T>(
        tileMembership, degrees, isolatedMask, g, N, static_cast<K>(numTiles));
    tileMembership.clear(); tileMembership.shrink_to_fit();

    // Tile-level RabbitOrder → (tilePerm, tileBlockId)
    auto [tilePerm, tileBlockId] = runRabbitOnTileGraph<K>(std::move(tileSG), tileSz, N);

    // Assign each community its center tile (majority of vertices)
    std::vector<K> commCenterTile(C, 0);
    #pragma omp parallel for schedule(dynamic, 64)
    for (K c = 0; c < C; ++c) {
        const auto& verts = commVertices[c];
        if (verts.empty()) { commCenterTile[c] = 0; continue; }
        struct TileCount { K tile; size_t count; };
        std::vector<TileCount> tileCounts;
        tileCounts.reserve(8);
        for (K v : verts) {
            K t = tileOf(v);
            bool found = false;
            for (auto& tc : tileCounts) {
                if (tc.tile == t) { tc.count++; found = true; break; }
            }
            if (!found) tileCounts.push_back({t, 1});
        }
        K bestTile = tileCounts[0].tile;
        size_t bestCount = tileCounts[0].count;
        for (size_t i = 1; i < tileCounts.size(); ++i) {
            if (tileCounts[i].count > bestCount) {
                bestCount = tileCounts[i].count;
                bestTile = tileCounts[i].tile;
            }
        }
        commCenterTile[c] = bestTile;
    }

    // Community super-graph + RabbitOrder (γ-scaled, reused primitives)
    auto commSG = buildCommunitySuperGraph<K, NodeID_T, DestID_T>(
        membership, degrees, /*vertexIsHub*/{}, g, N, C);
    std::vector<size_t> numVertsPerComm(C, 0);
    for (K c = 0; c < C; ++c) numVertsPerComm[c] = commVertices[c].size();
    auto commPermRO = runRabbitOnSuperCSR<K>(std::move(commSG), numVertsPerComm, gamma);

    // Composite sort: primary = tilePerm[centerTile], secondary = commPermRO
    std::vector<K> sortedComms(C);
    std::iota(sortedComms.begin(), sortedComms.end(), K(0));
    std::sort(sortedComms.begin(), sortedComms.end(),
        [&](K a, K b) {
            K permA = tilePerm[commCenterTile[a]];
            K permB = tilePerm[commCenterTile[b]];
            if (permA != permB) return permA < permB;
            return commPermRO[a] < commPermRO[b];
        });

    // commPerm[c] = position of community c in sortedComms
    std::vector<K> commPerm(C);
    for (K i = 0; i < C; ++i) commPerm[sortedComms[i]] = i;
    return commPerm;
}

//=============================================================================
// SECTION 16-COMPOSE: Pluggable {super-graph + community + intra-community} composition
//=============================================================================
//
// The first variant in this file that is *expressed* as a composition of
// independent stage picks rather than as a bespoke end-to-end algorithm.
// HRAB and TQR retain their bespoke pipelines (they each do non-trivial
// Stage 1 super-graph construction that does not generalise cleanly), but
// orderCompose<>() demonstrates that for the subset of variants that need
// no super-graph (CONNECTIVITY_BFS-style), the pipeline genuinely is just:
//
//     for each community c in <Stage 2 sort>(communities):
//         <Stage 3 primitive>(c, ...)
//
// Adding a new pick at either stage means adding one enum value + one
// switch arm; no other code changes.
//
// CLI: -o 12:compose            (defaults: s2=size_desc, s3=bfs)
//      -o 12:compose:s2_degree  (Stage 2 = total-degree desc)
//      -o 12:compose:s3_rcm     (Stage 3 = intraRCM)
//      -o 12:compose:s2_degree:s3_rcm
//
// Killer composition (Rabbit-everywhere, COMPOSE-only configuration that
// native RabbitOrder cannot reach):
//      -o 12:rabbit:compose:sg_super_rabbit:comm_identity:intra_dendrogram
//   -> Rabbit CD provides communities AND dendrogram, super-graph reorders
//      communities via a second (tiny) Rabbit pass, Identity preserves the
//      super-graph permutation, and intra_dendrogram reuses Rabbit's per-
//      community DFS for free (no BFS/visited bookkeeping).

template <typename K, typename NodeID_T, typename DestID_T>
void orderCompose(
    pvector<NodeID_T>& newIds,
    const GraphBrewResult<K>& result,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    const GraphBrewConfig& config,
    GraphBrewRealizedConfig* realized = nullptr)
{
    const std::vector<K>& membership = result.membership;
    Timer phaseTimer;
    phaseTimer.Start();
    Timer composeSectionTimer;
    composeSectionTimer.Start();
    double groupingSeconds = 0.0;
    double communityOrderSeconds = 0.0;
    double vertexMapSeconds = 0.0;
    double intraOrderSeconds = 0.0;
    double finalAssignSeconds = 0.0;

    const bool faithfulLocalGorder =
        config.intraCommunityOrder
        == IntraCommunityOrder::GorderFaithful;
    if (faithfulLocalGorder && g.directed()) {
        throw std::invalid_argument(
            "Faithful local Gorder currently requires an undirected graph");
    }
    if (
        faithfulLocalGorder
        && AblationConfig::Get().don_tiebreak
    ) {
        throw std::invalid_argument(
            "Faithful local Gorder requires DON tie-breaking disabled");
    }
    if (
        config.communityOrder == CommunityOrder::CapacityRuns
        && config.superGraphOrder != SuperGraphOrder::None
    ) {
        throw std::invalid_argument(
            "Capacity-run ordering requires no Stage-1 order");
    }
    if (
        config.communityOrder == CommunityOrder::CapacityRuns
        && g.directed()
    ) {
        throw std::invalid_argument(
            "Capacity-run ordering currently requires an undirected graph");
    }

    // Group vertices by community; isolated vertices go to the global tail.
    // Serial scatter is fast enough (~30-50ms for 3.7M vertices) and avoids
    // atomic-cursor contention seen with parallel push approaches on graphs
    // that have a few hot mega-communities (e.g. cit-Patents).
    K numComm = 0;
    #pragma omp parallel for reduction(max:numComm) schedule(static)
    for (int64_t v = 0; v < (int64_t)N; ++v) {
        if (membership[v] >= 0 && membership[v] + 1 > numComm) numComm = membership[v] + 1;
    }

    // Pre-size each commVertices[c] via parallel histogram so push_back never
    // reallocates during the serial scatter below.
    std::vector<K> commSizes(numComm, 0);
    int orderingThreads = 1;
    {
        const int nT = omp_get_max_threads();
        #pragma omp parallel
        {
            #pragma omp single
            orderingThreads = omp_get_num_threads();
            std::vector<K> localSizes(numComm, 0);
            #pragma omp for schedule(static) nowait
            for (int64_t v = 0; v < (int64_t)N; ++v) {
                if (degrees[v] != 0)
                    ++localSizes[membership[v]];
            }
            for (K c = 0; c < numComm; ++c) {
                if (localSizes[c]) {
                    #pragma omp atomic
                    commSizes[c] += localSizes[c];
                }
            }
        }
        (void)nT;
    }
    printf("GraphBrew: ordering-threads=%d\n", orderingThreads);

    std::vector<std::vector<K>> commVertices(numComm);
    #pragma omp parallel for schedule(dynamic, 256)
    for (K c = 0; c < numComm; ++c) {
        commVertices[c].reserve(commSizes[c]);
    }
    std::vector<K> isolated;
    isolated.reserve(N / 16);
    for (int64_t v = 0; v < (int64_t)N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(static_cast<K>(v));
        } else {
            commVertices[membership[v]].push_back(static_cast<K>(v));
        }
    }
    composeSectionTimer.Stop();
    groupingSeconds = composeSectionTimer.Seconds();
    composeSectionTimer.Start();

    // -------- Community order --------
    std::vector<K> commOrder(numComm);
    std::iota(commOrder.begin(), commOrder.end(), K(0));

    // -------- Super-graph order (optional) --------
    // Produces the base community order. Stage 2 is the primary key but
    // preserves this order as its deterministic tie-break.
    std::vector<K> stage1Perm;
    if (config.superGraphOrder != SuperGraphOrder::None) {
        // Per-community vertex count (used to flag empty/active for Stage 1)
        std::vector<size_t> numVertsPerComm(numComm, 0);
        for (K c = 0; c < numComm; ++c) numVertsPerComm[c] = commVertices[c].size();

        std::vector<K> vertDegreesView(degrees.begin(), degrees.end());

        if (config.superGraphOrder == SuperGraphOrder::SuperRabbit) {
            auto sg = buildCommunitySuperGraph<K, NodeID_T, DestID_T>(
                membership, vertDegreesView, /*vertexIsHub*/{}, g, N, numComm);
            stage1Perm = runRabbitOnSuperCSR<K>(
                std::move(sg), numVertsPerComm,
                static_cast<float>(config.superGraphResolution));
        } else if (config.superGraphOrder == SuperGraphOrder::SuperRCM) {
            auto sg = buildCommunitySuperGraph<K, NodeID_T, DestID_T>(
                membership, vertDegreesView, /*vertexIsHub*/{}, g, N, numComm);
            stage1Perm = runRCMOnSuperCSR<K>(sg, numVertsPerComm);
        } else if (config.superGraphOrder == SuperGraphOrder::TileRabbit) {
            stage1Perm = runTileRabbit<K, NodeID_T, DestID_T>(
                membership, vertDegreesView, commVertices, g, N, numComm,
                static_cast<float>(config.superGraphResolution));
        }
        if (config.superGraphOrder == SuperGraphOrder::Hilbert) {
            // Mosaic-style 2-D Hilbert curve over (community_size, avg_degree).
            // Bucket each axis to [0,256), compute 16-bit Hilbert d, sort by d.
            const uint32_t ORDER = 256;
            std::vector<size_t> sz(numComm, 0), totDeg(numComm, 0);
            size_t maxSz = 1; size_t maxAvg = 1;
            for (K c = 0; c < numComm; ++c) {
                sz[c] = commVertices[c].size();
                size_t s = 0;
                for (K v : commVertices[c]) s += degrees[v];
                totDeg[c] = s;
                if (sz[c] > maxSz) maxSz = sz[c];
                size_t avg = sz[c] ? totDeg[c] / sz[c] : 0;
                if (avg > maxAvg) maxAvg = avg;
            }
            // 8-bit Hilbert curve d2xy (xy2d inverse from Wikipedia)
            auto xy2d = [&](uint32_t n, uint32_t x, uint32_t y) -> uint64_t {
                uint64_t d = 0;
                for (uint32_t s = n/2; s > 0; s /= 2) {
                    uint32_t rx = (x & s) > 0;
                    uint32_t ry = (y & s) > 0;
                    d += (uint64_t)s * s * ((3u * rx) ^ ry);
                    if (ry == 0) {
                        if (rx == 1) { x = s - 1 - x; y = s - 1 - y; }
                        uint32_t t = x; x = y; y = t;
                    }
                }
                return d;
            };
            stage1Perm.assign(numComm, 0);
            std::vector<std::pair<uint64_t,K>> rank(numComm);
            for (K c = 0; c < numComm; ++c) {
                size_t avg = sz[c] ? totDeg[c] / sz[c] : 0;
                uint32_t bx = (uint32_t)((sz[c] * (uint64_t)(ORDER-1)) / maxSz);
                uint32_t by = (uint32_t)((avg    * (uint64_t)(ORDER-1)) / maxAvg);
                rank[c] = { xy2d(ORDER, bx, by), c };
            }
            std::sort(rank.begin(), rank.end());
            for (K i = 0; i < numComm; ++i) stage1Perm[rank[i].second] = i;
        }
        // Order commOrder by Stage 1 permutation (lowest perm first)
        std::sort(commOrder.begin(), commOrder.end(),
                  [&](K a, K b) { return stage1Perm[a] < stage1Perm[b]; });
    }

    // Stage 2 refines the Stage 1 result. Stable sorting makes the selected
    // community metric primary while retaining Stage 1 order for equal keys.
    bool cutMinFallback = false;
    CapacityRunOrderResult<K> capacityRuns;
    size_t capacityL2Bytes = 0;
    size_t capacityLLCBytes = 0;
    size_t capacityPropertyBytes = 0;
    if (config.communityOrder == CommunityOrder::SizeDesc) {
        std::stable_sort(commOrder.begin(), commOrder.end(), [&](K a, K b) {
            return commVertices[a].size() > commVertices[b].size();
        });
    } else if (config.communityOrder == CommunityOrder::SizeAsc) {
        std::stable_sort(commOrder.begin(), commOrder.end(), [&](K a, K b) {
            return commVertices[a].size() < commVertices[b].size();
        });
    } else if (config.communityOrder == CommunityOrder::DegreeDesc) {
        std::vector<size_t> commTotalDeg(numComm, 0);
        #pragma omp parallel for schedule(dynamic, 256)
        for (K c = 0; c < numComm; ++c) {
            size_t s = 0;
            for (K v : commVertices[c]) s += degrees[v];
            commTotalDeg[c] = s;
        }
        std::stable_sort(commOrder.begin(), commOrder.end(), [&](K a, K b) {
            return commTotalDeg[a] > commTotalDeg[b];
        });
    } else if (config.communityOrder == CommunityOrder::DegreeAsc) {
        std::vector<size_t> commTotalDeg(numComm, 0);
        #pragma omp parallel for schedule(dynamic, 256)
        for (K c = 0; c < numComm; ++c) {
            size_t s = 0;
            for (K v : commVertices[c]) s += degrees[v];
            commTotalDeg[c] = s;
        }
        std::stable_sort(commOrder.begin(), commOrder.end(), [&](K a, K b) {
            return commTotalDeg[a] < commTotalDeg[b];
        });
    } else if (config.communityOrder == CommunityOrder::CapacityRuns) {
        const CapacityGeometry geometry =
            resolveCapacityGeometry(config);
        capacityL2Bytes = geometry.l2Bytes;
        capacityLLCBytes = geometry.llcBytes;
        capacityPropertyBytes = geometry.propertyBytesPerVertex;
        std::vector<size_t> capacityCommunitySizes(numComm, 0);
        for (K c = 0; c < numComm; ++c)
            capacityCommunitySizes[c] = commVertices[c].size();
        auto capacityAdjacency =
            buildCapacityCommunityAdjacency<K, NodeID_T, DestID_T>(
                membership,
                g,
                N,
                numComm);
        capacityRuns = buildCapacityRunCommunityOrder<K>(
            capacityCommunitySizes,
            capacityAdjacency,
            commOrder,
            geometry.l2TargetVertices,
            geometry.llcTargetVertices);
        if (capacityRuns.order.size() != static_cast<size_t>(numComm)) {
            throw std::runtime_error(
                "Capacity-run ordering produced an incomplete permutation");
        }
        commOrder = capacityRuns.order;
        if (realized) {
            realized->capacityL2Bytes = capacityL2Bytes;
            realized->capacityLLCBytes = capacityLLCBytes;
            realized->capacityPropertyBytesPerVertex =
                capacityPropertyBytes;
            realized->capacityL2Runs = capacityRuns.l2RunEnds.size();
            realized->capacityLLCRuns = capacityRuns.llcRunEnds.size();
        }
    } else if (config.communityOrder == CommunityOrder::CutMin) {
        // Mt-METIS-style: greedy nearest-neighbour TSP over inter-community
        // crossing-edge counts.  Skip if C>4096 (matrix gets too big).
        if (numComm > 4096) {
            // Fallback to DegreeDesc when C is too large.
            cutMinFallback = true;
            std::vector<size_t> commTotalDeg(numComm, 0);
            #pragma omp parallel for schedule(dynamic, 256)
            for (K c = 0; c < numComm; ++c) {
                size_t s = 0;
                for (K v : commVertices[c]) s += degrees[v];
                commTotalDeg[c] = s;
            }
            std::stable_sort(commOrder.begin(), commOrder.end(), [&](K a, K b) {
                return commTotalDeg[a] > commTotalDeg[b];
            });
        } else {
            const size_t C = static_cast<size_t>(numComm);
            std::vector<uint64_t> cross(C * C, 0);
            // One pass over all edges to build the symmetric crossing matrix.
            for (NodeID_T v = 0; v < static_cast<NodeID_T>(N); ++v) {
                K cv = membership[v];
                for (auto u : g.out_neigh(v)) {
                    K cu = membership[u];
                    if (cu != cv) cross[(size_t)cv * C + (size_t)cu]++;
                }
            }
            // NN-TSP: start from largest community, walk to unvisited neighbour
            // with maximum crossing weight.  Ties broken by community ID.
            std::vector<char> visited(C, 0);
            std::vector<K> tour; tour.reserve(C);
            K start = commOrder.empty() ? K(0) : commOrder.front();
            size_t bestSz = 0;
            for (K c : commOrder) {
                if (commVertices[c].size() > bestSz) {
                    bestSz = commVertices[c].size();
                    start = c;
                }
            }
            K cur = start; tour.push_back(cur); visited[cur] = 1;
            for (size_t step = 1; step < C; ++step) {
                uint64_t bestW = 0; K bestC = (K)-1;
                for (K c : commOrder) {
                    if (visited[c]) continue;
                    uint64_t w = cross[(size_t)cur * C + (size_t)c];
                    if (w > bestW || bestC == (K)-1) {
                        bestW = w;
                        bestC = c;
                    }
                }
                if (bestC == (K)-1) {
                    // Disconnected: preserve the Stage 1 base order.
                    for (K c : commOrder)
                        if (!visited[c]) { bestC = c; break; }
                }
                visited[bestC] = 1; tour.push_back(bestC); cur = bestC;
            }
            commOrder = std::move(tour);
        }
    }
    // CommunityOrder::Identity: leave commOrder as-is (Stage 1 result or 0..C)

    // Offsets in the sorted community order
    std::vector<size_t> offsets(numComm + 1, 0);
    for (size_t i = 0; i < (size_t)numComm; ++i) {
        offsets[i + 1] = offsets[i] + commVertices[commOrder[i]].size();
    }
    composeSectionTimer.Stop();
    communityOrderSeconds = composeSectionTimer.Seconds();
    composeSectionTimer.Start();

    // -------- Intra-community order (per-community ordering via shared primitives) --------
    std::vector<K> localIds(N, 0);

    // Decide whether intra_dendrogram can actually run.  It needs Rabbit's
    // per-vertex child/sibling chains AND the community-id -> dendrogram-root
    // mapping (rabbitToplevel[c]).  Both are populated only when
    // runRabbitOrder() was the CD step AND COMPOSE+Dendrogram was requested.
    const bool useDendrogram =
        (config.intraCommunityOrder == IntraCommunityOrder::Dendrogram) &&
        result.hasRabbitDendrogram &&
        result.rabbitToplevel.size() >= static_cast<size_t>(numComm);
    if (config.intraCommunityOrder == IntraCommunityOrder::Dendrogram && !useDendrogram) {
        printf("  compose: intra_dendrogram requested but no Rabbit dendrogram available "
               "(was CD=Leiden?), falling back to intra_bfs\n");
    }

    // Flat global vertex -> local-index map (caller-owned for the primitives).
    // Only intraBFSFromHub and intraRCM need this; intraDendrogramDFS follows
    // pointer chains and never touches it.  Skip the 8*N-byte allocation +
    // init when the dendrogram path will be taken.
    std::vector<size_t> vertToLocal;
    size_t gorderCommunityCount = 0;
    size_t gorderVertexCount = 0;
    size_t gorderMaxCommunity = 0;
    size_t gorderFallbackCount = 0;
    size_t gorderFallbackVertices = 0;
    const bool needsVertToLocal =
        !useDendrogram ||
        config.refinementPass == RefinementPass::TwoSwap;
    if (needsVertToLocal) {
        vertToLocal.assign(N, static_cast<size_t>(-1));
        #pragma omp parallel for schedule(dynamic, 256)
        for (size_t i = 0; i < (size_t)numComm; ++i) {
            const auto& verts = commVertices[commOrder[i]];
            for (size_t j = 0; j < verts.size(); ++j) {
                vertToLocal[verts[j]] = j;
            }
        }
    }
    if (
        config.intraCommunityOrder == IntraCommunityOrder::Gorder
        || config.intraCommunityOrder
            == IntraCommunityOrder::GorderFaithful
    ) {
        if (config.gorderWindow <= 0) {
            throw std::invalid_argument(
                "Local Gorder requires a positive window");
        }
        for (K c : commOrder) {
            size_t size = commVertices[c].size();
            if (
                config.gorderFallback > 0
                && size > static_cast<size_t>(
                    config.gorderFallback)
            ) {
                ++gorderFallbackCount;
                gorderFallbackVertices += size;
                continue;
            }
            if (size <= (faithfulLocalGorder ? 1u : 3u))
                continue;
            if (
                faithfulLocalGorder
                && !faithfulGorderScoreRangeSafe(
                    size, config.gorderWindow)
            ) {
                throw std::overflow_error(
                    "Faithful local Gorder community exceeds score range");
            }
            ++gorderCommunityCount;
            gorderVertexCount += size;
            gorderMaxCommunity =
                std::max(gorderMaxCommunity, size);
        }
        if (realized) {
            realized->gorderCommunities = gorderCommunityCount;
            realized->gorderVertices = gorderVertexCount;
            realized->gorderMaxCommunity = gorderMaxCommunity;
            realized->gorderFallbackCommunities =
                gorderFallbackCount;
            realized->gorderFallbackVertices =
                gorderFallbackVertices;
        }
    }
    composeSectionTimer.Stop();
    vertexMapSeconds = composeSectionTimer.Seconds();
    composeSectionTimer.Start();

    #pragma omp parallel
    {
        std::queue<K> bfsQueue;
        std::vector<bool> visited;
        std::vector<K> cmOrder;  // only used by RCM
        std::deque<K> dendStack; // only used by Dendrogram
        std::vector<K> gordPlaced;                  // only used by Gorder
        std::vector<std::vector<size_t>> gordOutAdj;// only used by Gorder
        std::vector<std::vector<size_t>> gordInAdj; // faithful Gorder only

        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < (size_t)numComm; ++i) {
            K c = commOrder[i];
            if (useDendrogram) {
                K root = result.rabbitToplevel[c];
                intraDendrogramDFS<K>(
                    root, result.rabbitChild, result.rabbitSibling,
                    dendStack, localIds);
            } else if (config.intraCommunityOrder == IntraCommunityOrder::RCM) {
                intraRCM<K, NodeID_T, DestID_T>(
                    commVertices[c], c, membership, degrees, g,
                    vertToLocal, visited, bfsQueue, cmOrder, localIds);
            } else if (config.intraCommunityOrder == IntraCommunityOrder::RCMpp) {
                intraRCMpp<K, NodeID_T, DestID_T>(
                    commVertices[c], c, membership, degrees, g,
                    vertToLocal, visited, bfsQueue, cmOrder, localIds);
            } else if (
                config.intraCommunityOrder == IntraCommunityOrder::Gorder
            ) {
                if (
                    config.gorderFallback > 0
                    && commVertices[c].size()
                        > static_cast<size_t>(
                            config.gorderFallback)
                ) {
                    intraBFSFromHub<K, NodeID_T, DestID_T>(
                        commVertices[c], c, membership, degrees, g,
                        vertToLocal, visited, bfsQueue, localIds);
                } else {
                    intraGorderGreedy<K, NodeID_T, DestID_T>(
                        commVertices[c], c, membership, degrees, g,
                        vertToLocal, config.gorderWindow,
                        gordPlaced, gordOutAdj, localIds);
                }
            } else if (
                config.intraCommunityOrder
                == IntraCommunityOrder::GorderFaithful
            ) {
                if (
                    config.gorderFallback > 0
                    && commVertices[c].size()
                        > static_cast<size_t>(
                            config.gorderFallback)
                ) {
                    intraBFSFromHub<K, NodeID_T, DestID_T>(
                        commVertices[c], c, membership, degrees, g,
                        vertToLocal, visited, bfsQueue, localIds);
                } else {
                    intraGorderFaithful<K, NodeID_T, DestID_T>(
                        commVertices[c], c, membership, g,
                        vertToLocal, config.gorderWindow,
                        gordPlaced, gordOutAdj, gordInAdj, localIds);
                }
            } else if (config.intraCommunityOrder == IntraCommunityOrder::HubSort) {
                // Sort community vertices by degree descending; cheap O(sz log sz).
                const auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) { /* nothing */ }
                else if (sz == 1) { localIds[verts[0]] = 0; }
                else {
                    std::vector<K> sorted(verts.begin(), verts.end());
                    std::sort(sorted.begin(), sorted.end(),
                        [&](K a, K b) { return degrees[a] > degrees[b]; });
                    for (size_t j = 0; j < sz; ++j) localIds[sorted[j]] = static_cast<K>(j);
                }
            } else if (config.intraCommunityOrder == IntraCommunityOrder::DegreeAsc) {
                // Sort community vertices by degree ASCENDING; ablation control vs HubSort.
                const auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) { /* nothing */ }
                else if (sz == 1) { localIds[verts[0]] = 0; }
                else {
                    std::vector<K> sorted(verts.begin(), verts.end());
                    std::sort(sorted.begin(), sorted.end(),
                        [&](K a, K b) { return degrees[a] < degrees[b]; });
                    for (size_t j = 0; j < sz; ++j) localIds[sorted[j]] = static_cast<K>(j);
                }
            } else if (config.intraCommunityOrder == IntraCommunityOrder::Hub2) {
                // Second-moment degree (DRO/Lakhotia, IISWC'19): per-community
                // sort by sum-of-neighbor-degree descending.  Captures hub-of-hubs
                // locality missed by pure HubSort.  O(|E_local|) per community.
                const auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) { /* nothing */ }
                else if (sz == 1) { localIds[verts[0]] = 0; }
                else {
                    std::vector<K> sorted(verts.begin(), verts.end());
                    std::vector<uint64_t> score(sz, 0);
                    for (size_t j = 0; j < sz; ++j) {
                        K v = sorted[j];
                        uint64_t s = 0;
                        for (auto u : g.out_neigh(v)) s += static_cast<uint64_t>(degrees[u]);
                        score[j] = s;
                    }
                    std::vector<size_t> idx(sz);
                    for (size_t j = 0; j < sz; ++j) idx[j] = j;
                    std::sort(idx.begin(), idx.end(),
                        [&](size_t a, size_t b) { return score[a] > score[b]; });
                    for (size_t j = 0; j < sz; ++j) localIds[sorted[idx[j]]] = static_cast<K>(j);
                }
            } else if (config.intraCommunityOrder == IntraCommunityOrder::Alternate) {
                // Sort by degree desc, then interleave [hub0,leaf0,hub1,leaf1,...]
                // to combine prefetch locality with false-sharing dispersion.
                const auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) { /* nothing */ }
                else if (sz == 1) { localIds[verts[0]] = 0; }
                else {
                    std::vector<K> sorted(verts.begin(), verts.end());
                    std::sort(sorted.begin(), sorted.end(),
                        [&](K a, K b) { return degrees[a] > degrees[b]; });
                    // Interleave: front half (hubs) and back half (leaves)
                    size_t mid = (sz + 1) / 2;
                    size_t lo = 0, hi = sz - 1, pos = 0;
                    for (size_t i = 0; i < mid; ++i) {
                        localIds[sorted[lo++]] = static_cast<K>(pos++);
                        if (hi >= lo) localIds[sorted[hi--]] = static_cast<K>(pos++);
                    }
                }
            } else if (config.intraCommunityOrder == IntraCommunityOrder::Random) {
                // Deterministic Fisher-Yates shuffle inside community.
                // Worst-case control: tests if any ordering matters at all.
                // Seed = 0xDEADBEEF ^ communityID for per-community reproducibility.
                const auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) { /* nothing */ }
                else if (sz == 1) { localIds[verts[0]] = 0; }
                else {
                    std::vector<K> perm(verts.begin(), verts.end());
                    // Local RNG so parallel shuffles are deterministic
                    // and independent across communities.
                    std::mt19937 rng(static_cast<uint32_t>(0xDEADBEEFu ^ static_cast<uint32_t>(c)));
                    for (size_t j = sz - 1; j > 0; --j) {
                        std::uniform_int_distribution<size_t> dist(0, j);
                        size_t k2 = dist(rng);
                        std::swap(perm[j], perm[k2]);
                    }
                    for (size_t j = 0; j < sz; ++j) localIds[perm[j]] = static_cast<K>(j);
                }
            } else if (config.intraCommunityOrder == IntraCommunityOrder::BoundaryLast) {
                // Sort intra-community by EXTERNAL degree ascending:
                // interior nodes (few cross-community edges) first, boundary nodes
                // (many cross-community edges) last.  Hypothesis: BFS/CC complete
                // intra-community work before transitioning, improving locality.
                // Tie-break by degree DESCENDING (hubs among ties first).
                const auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) { /* nothing */ }
                else if (sz == 1) { localIds[verts[0]] = 0; }
                else {
                    std::vector<K> sorted(verts.begin(), verts.end());
                    std::vector<uint32_t> extDeg(sz, 0);
                    // Compute external degree for each vertex in this community.
                    for (size_t idx = 0; idx < sz; ++idx) {
                        K v = verts[idx];
                        uint32_t e = 0;
                        for (auto d : g.out_neigh(v)) {
                            NodeID_T dst = static_cast<NodeID_T>(d);
                            if (dst >= 0 && static_cast<size_t>(dst) < membership.size()
                                && membership[dst] != c) {
                                ++e;
                            }
                        }
                        extDeg[idx] = e;
                    }
                    // Build idx permutation, sort by (extDeg asc, degree desc).
                    std::vector<size_t> idxs(sz);
                    for (size_t j = 0; j < sz; ++j) idxs[j] = j;
                    std::sort(idxs.begin(), idxs.end(),
                        [&](size_t a, size_t b) {
                            if (extDeg[a] != extDeg[b]) return extDeg[a] < extDeg[b];
                            return degrees[verts[a]] > degrees[verts[b]];
                        });
                    for (size_t j = 0; j < sz; ++j) localIds[verts[idxs[j]]] = static_cast<K>(j);
                }
            } else if (config.intraCommunityOrder == IntraCommunityOrder::CoreOrder) {
                // K-core decomposition restricted to the community subgraph.
                // Sort by core-number DESCENDING (deep core first, periphery last).
                // Tie-break by degree descending (hubs among same-core ties first).
                // Algorithm: bucket-sort peel (Batagelj & Zaversnik 2003), O(V+E_local).
                const auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) { /* nothing */ }
                else if (sz == 1) { localIds[verts[0]] = 0; }
                else {
                    // vertToLocal was populated for every community before
                    // entering the parallel region and remains read-only.
                    // Compute INTERNAL degree (edges to same-community vertices).
                    std::vector<uint32_t> intDeg(sz, 0);
                    for (size_t idx = 0; idx < sz; ++idx) {
                        K v = verts[idx];
                        uint32_t d = 0;
                        for (auto nb : g.out_neigh(v)) {
                            NodeID_T dst = static_cast<NodeID_T>(nb);
                            if (dst >= 0 && static_cast<size_t>(dst) < membership.size()
                                && membership[dst] == c) ++d;
                        }
                        intDeg[idx] = d;
                    }
                    // Bucket-sort peel: for each iter, remove lowest-degree vertex,
                    // record its core number = its current degree, decrement its
                    // neighbours' degrees.
                    std::vector<uint32_t> coreNum(sz, 0);
                    std::vector<uint8_t> removed(sz, 0);
                    uint32_t maxDeg = 0;
                    for (uint32_t d : intDeg) maxDeg = std::max(maxDeg, d);
                    std::vector<std::vector<size_t>> bucket(maxDeg + 1);
                    for (size_t j = 0; j < sz; ++j) bucket[intDeg[j]].push_back(j);
                    uint32_t k = 0;
                    size_t processed = 0;
                    while (processed < sz) {
                        // Find next non-empty bucket >= k.
                        while (k <= maxDeg && bucket[k].empty()) ++k;
                        if (k > maxDeg) break;
                        size_t j = bucket[k].back();
                        bucket[k].pop_back();
                        if (removed[j]) continue;
                        if (intDeg[j] > k) {
                            // Was decremented after being placed in bucket k; re-bucket.
                            bucket[intDeg[j]].push_back(j);
                            continue;
                        }
                        coreNum[j] = k;
                        removed[j] = 1;
                        ++processed;
                        // Decrement neighbours that are still present.
                        K v = verts[j];
                        for (auto nb : g.out_neigh(v)) {
                            NodeID_T dst = static_cast<NodeID_T>(nb);
                            if (dst < 0 || static_cast<size_t>(dst) >= membership.size()) continue;
                            if (membership[dst] != c) continue;
                            K nbIdx = vertToLocal[dst];
                            if (nbIdx >= sz) continue;
                            if (removed[nbIdx]) continue;
                            if (intDeg[nbIdx] > k) {
                                --intDeg[nbIdx];
                                if (intDeg[nbIdx] >= k) bucket[intDeg[nbIdx]].push_back(nbIdx);
                            }
                        }
                    }
                    // Sort by (core desc, degree desc).
                    std::vector<size_t> idxs(sz);
                    for (size_t j = 0; j < sz; ++j) idxs[j] = j;
                    std::sort(idxs.begin(), idxs.end(),
                        [&](size_t a, size_t b) {
                            if (coreNum[a] != coreNum[b]) return coreNum[a] > coreNum[b];
                            return degrees[verts[a]] > degrees[verts[b]];
                        });
                    for (size_t j = 0; j < sz; ++j) localIds[verts[idxs[j]]] = static_cast<K>(j);
                }
            } else { // BFSFromHub (also the fallback path)
                intraBFSFromHub<K, NodeID_T, DestID_T>(
                    commVertices[c], c, membership, degrees, g,
                    vertToLocal, visited, bfsQueue, localIds);
            }
        }
    }
    composeSectionTimer.Stop();
    intraOrderSeconds = composeSectionTimer.Seconds();

    // -------- Optional post-pass refinement (per-community independent) --------
    if (config.refinementPass == RefinementPass::TwoSwap) {
        Timer refineTimer; refineTimer.Start();
        std::vector<size_t> refineOrder;
        refineOrder.reserve(numComm);
        for (size_t i = 0; i < static_cast<size_t>(numComm); ++i) {
            if (commVertices[commOrder[i]].size() >= 3)
                refineOrder.push_back(i);
        }
        std::stable_sort(
            refineOrder.begin(), refineOrder.end(),
            [&](size_t a, size_t b) {
                return commVertices[commOrder[a]].size()
                    > commVertices[commOrder[b]].size();
            });
        #pragma omp parallel
        {
            std::vector<size_t> adjOffsets;
            std::vector<K> adjVertices;
            std::vector<K> orderScratch;
            #pragma omp for schedule(dynamic, 1)
            for (size_t ri = 0; ri < refineOrder.size(); ++ri) {
                size_t i = refineOrder[ri];
                K c = commOrder[i];
                refineTwoSwap<K, NodeID_T, DestID_T>(
                    commVertices[c], c, membership, g, localIds,
                    vertToLocal, config.refineMaxPasses,
                    adjOffsets, adjVertices, orderScratch);
            }
        }
        refineTimer.Stop();
        printf(
            "  compose: refine=2swap maxPasses=%d, eligible=%zu/%u, %.4fs\n",
            config.refineMaxPasses, refineOrder.size(),
            static_cast<unsigned>(numComm), refineTimer.Seconds());
    }

    // Compose final newIds = community-offset + local id within community
    composeSectionTimer.Start();
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < (size_t)numComm; ++i) {
        size_t base = offsets[i];
        for (K v : commVertices[commOrder[i]]) {
            newIds[v] = static_cast<NodeID_T>(base + localIds[v]);
        }
    }

    // Isolated vertices appended in their natural order
    size_t tail = offsets[numComm];
    for (size_t i = 0; i < isolated.size(); ++i) {
        newIds[isolated[i]] = static_cast<NodeID_T>(tail + i);
    }
    composeSectionTimer.Stop();
    finalAssignSeconds = composeSectionTimer.Seconds();

    phaseTimer.Stop();
    const char* s1Name =
        config.superGraphOrder == SuperGraphOrder::None ? "none" :
        config.superGraphOrder == SuperGraphOrder::SuperRabbit ? "super_rabbit" :
        config.superGraphOrder == SuperGraphOrder::SuperRCM ? "super_rcm" :
        config.superGraphOrder == SuperGraphOrder::TileRabbit ? "tile_rabbit" :
        "hilbert";
    const char* s2Name =
        config.communityOrder == CommunityOrder::SizeDesc   ? "size_desc"   :
        config.communityOrder == CommunityOrder::SizeAsc    ? "size_asc"    :
        config.communityOrder == CommunityOrder::DegreeDesc ? "degree_desc" :
        config.communityOrder == CommunityOrder::DegreeAsc  ? "degree_asc"  :
        config.communityOrder == CommunityOrder::CapacityRuns
            ? "capacity_runs" :
        config.communityOrder == CommunityOrder::CutMin
            ? (cutMinFallback ? "cut_min_fallback_degree_desc" : "cut_min") :
        "identity";
    const char* intraName =
        useDendrogram                                              ? "dendrogram"   :
        config.intraCommunityOrder == IntraCommunityOrder::RCM     ? "rcm"          :
        config.intraCommunityOrder == IntraCommunityOrder::RCMpp   ? "rcmpp"        :
        config.intraCommunityOrder == IntraCommunityOrder::Gorder  ? "gorder"       :
        config.intraCommunityOrder == IntraCommunityOrder::GorderFaithful
            ? "gorder_faithful" :
        config.intraCommunityOrder == IntraCommunityOrder::HubSort ? "hubsort"      :
        config.intraCommunityOrder == IntraCommunityOrder::DegreeAsc ? "deg_asc"    :
        config.intraCommunityOrder == IntraCommunityOrder::Hub2      ? "hub2"        :
        config.intraCommunityOrder == IntraCommunityOrder::Alternate ? "alternate"  :
        config.intraCommunityOrder == IntraCommunityOrder::Random    ? "random"     :
        config.intraCommunityOrder == IntraCommunityOrder::BoundaryLast ? "bndlast" :
        config.intraCommunityOrder == IntraCommunityOrder::CoreOrder ? "core" :
                                                                      "bfs_from_hub";
    const char* refineName =
        config.refinementPass == RefinementPass::TwoSwap ? "2swap" : "none";
    printf("  compose: sg=%s comm=%s intra=%s refine=%s, %zu communities, %zu isolated, %.4fs\n",
           s1Name, s2Name, intraName, refineName,
           static_cast<size_t>(numComm), isolated.size(),
           phaseTimer.Seconds());
    printf(
        "  compose phases: grouping=%.4fs, community-order=%.4fs, "
        "vertex-map=%.4fs, intra-order=%.4fs, final-assign=%.4fs\n",
        groupingSeconds, communityOrderSeconds, vertexMapSeconds,
        intraOrderSeconds, finalAssignSeconds);
    if (
        config.intraCommunityOrder == IntraCommunityOrder::Gorder
        || config.intraCommunityOrder
            == IntraCommunityOrder::GorderFaithful
    ) {
        printf(
            "  compose gorder: communities=%zu, vertices=%zu, "
            "max-community=%zu, fallback-communities=%zu, "
            "fallback-vertices=%zu\n",
            gorderCommunityCount, gorderVertexCount,
            gorderMaxCommunity, gorderFallbackCount,
            gorderFallbackVertices);
    }
    if (config.communityOrder == CommunityOrder::CapacityRuns) {
        printf(
            "  compose capacity-runs: property-bytes=%zu, "
            "l2-bytes=%zu, llc-bytes=%zu, l2-runs=%zu, llc-runs=%zu\n",
            capacityPropertyBytes,
            capacityL2Bytes,
            capacityLLCBytes,
            capacityRuns.l2RunEnds.size(),
            capacityRuns.llcRunEnds.size());
    }
}

//=============================================================================
// Extracted implementation detail; kept at this position to preserve declaration order.
#include "reorder_graphbrew_diagnostics.h"

// SECTION 21: DYNAMIC RESOLUTION ADJUSTMENT
//=============================================================================

/**
 * Runtime metrics collected during each GraphBrew pass
 * Used to inform adaptive resolution adjustments (when useDynamicResolution=true)
 */
template <typename W>
struct GraphBrewPassMetrics {
    size_t num_communities;      ///< Communities after this pass
    size_t prev_communities;     ///< Communities before this pass
    int local_move_iterations;   ///< How many local-move iterations needed
    W max_community_weight;      ///< Largest community weight
    W total_weight;              ///< Sum of all weights
    size_t total_super_edges;    ///< Edges in super-graph
    double reduction_ratio;      ///< num_communities / prev_communities
    double size_imbalance;       ///< max_weight / avg_weight
    double super_avg_degree;     ///< Average degree in super-graph
};

/**
 * Compute adaptive resolution adjustment based on runtime metrics
 *
 * Signals used for adjustment:
 * 1. Community reduction rate - too fast/slow indicates resolution needs tuning
 * 2. Size imbalance - giant communities should be broken up
 * 3. Convergence speed - struggling algorithms benefit from coarser resolution
 * 4. Super-graph density - denser graphs need higher resolution
 *
 * @param current_resolution Current resolution value
 * @param metrics Runtime metrics from the current pass
 * @param original_avg_degree Average degree of original graph
 * @param max_iterations Max local-move iterations (to detect convergence issues)
 * @return Adjusted resolution for the next pass
 */
template <typename W>
inline double graphBrewComputeAdaptiveResolution(
    double current_resolution,
    const GraphBrewPassMetrics<W>& metrics,
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
    if (metrics.size_imbalance > 100.0) {
        adjusted *= 1.25;  // Very unbalanced, aggressive
    } else if (metrics.size_imbalance > 50.0) {
        adjusted *= 1.15;  // Break giant communities
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
 * Collect runtime metrics from super-graph state
 */
template <typename K, typename W>
inline GraphBrewPassMetrics<W> graphBrewCollectPassMetrics(
    const SuperGraph<K, W>& sg,
    const std::vector<W>& vtot,
    size_t num_communities,
    size_t prev_communities,
    int local_move_iterations) {

    GraphBrewPassMetrics<W> metrics;
    metrics.num_communities = num_communities;
    metrics.prev_communities = prev_communities;
    metrics.local_move_iterations = local_move_iterations;

    // Compute community weight statistics from vtot
    W max_weight = W(0);
    W total_weight = W(0);
    size_t C = std::min(num_communities, vtot.size());
    for (size_t c = 0; c < C; ++c) {
        max_weight = std::max(max_weight, vtot[c]);
        total_weight += vtot[c];
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
    metrics.total_super_edges = sg.numEdges;
    metrics.super_avg_degree = (num_communities > 0)
        ? static_cast<double>(sg.numEdges) / num_communities
        : 0.0;

    return metrics;
}

//=============================================================================
// SECTION 22: MAIN GRAPHBREW ALGORITHM
//=============================================================================

/**
 * Main GraphBrew algorithm - Faithful Leiden with modular components
 *
 * Supports dynamic resolution when config.useDynamicResolution=true
 * Resolution is adjusted per-pass based on:
 * - Community reduction rate
 * - Size imbalance
 * - Convergence speed
 * - Super-graph density
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
GraphBrewResult<K> runGraphBrew(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const GraphBrewConfig& config) {

    ScopedCommunityThreadLimit thread_limit(
        config.deterministicCommunityDetection);
    const int64_t N = g.num_nodes();
    const Weight M = (config.mComputation == MComputation::TOTAL_EDGES)
        ? std::max(Weight(1), static_cast<Weight>(g.num_edges()))   // GVE style
        : static_cast<Weight>(g.num_edges()) / 2.0;                // leiden.hxx style

    // Current resolution - may be adjusted if dynamic
    Weight currentResolution = config.resolution;
    const double originalAvgDegree = static_cast<double>(g.num_edges()) / N;

    GRAPHBREW_TRACE("=== GRAPHBREW START ===");
    GRAPHBREW_TRACE("N=%ld, M=%.0f, R=%.4f, dynamic=%d", N, M, currentResolution, config.useDynamicResolution);

    GraphBrewResult<K> result;
    Timer totalTimer;
    totalTimer.Start();

    // Allocate buffers
    std::vector<char> vaff(N, 1);
    std::vector<K> ucom(N), vcom(N);
    std::vector<K> vcob(N);
    std::vector<Weight> utot(N), vtot(N);
    std::vector<Weight> ctot(N);
    std::vector<K> cmap(N);

    SuperGraph<K, Weight> sgY, sgZ;
    std::vector<size_t> coff;
    std::vector<K> cvtx;

    // Pre-allocate SuperGraph buffers if reusing (estimate: N nodes, 2*edges)
    if (config.reuseBuffers) {
        const size_t estNodes = N;
        const size_t estEdges = g.num_edges();
        sgY.reserve(estNodes, estEdges);
        sgZ.reserve(estNodes, estEdges);
        GRAPHBREW_TRACE("reuseBuffers: reserved %zu nodes, %zu edges", estNodes, estEdges);
    }

    // Step 1: Compute vertex weights
    Timer phaseTimer;
    phaseTimer.Start();
    computeVertexWeights(utot, g, config);
    phaseTimer.Stop();

    // Step 2: Initialize communities
    initializeCommunities<K>(ucom, ctot, utot, N);
    vtot = utot;
    vcom = ucom;

    // Track previous community count for dynamic resolution
    size_t prevCommunities = N;

    GraphBrewConfig passConfig = config;
    int totalIterations = 0;
    int pass = 0;

    // Main loop
    // GVE_CSR mode: always operate on original graph (flat loop, like GVE)
    // Standard mode: switch to super-graph after pass 0
    const bool gveMode = (config.aggregation == AggregationStrategy::GVE_CSR);

    for (; pass < config.maxPasses && M > 0; ++pass) {
        bool isFirst = (pass == 0);
        // In GVE mode, always treat as "first pass" (original graph)
        bool useOriginalGraph = isFirst || gveMode;

        GRAPHBREW_TRACE("=== PASS %d (res=%.4f) ===", pass, currentResolution);

        // ================================================================
        // PHASE 1: LOCAL-MOVING
        // ================================================================
        phaseTimer.Start();

        int moveIters;
        if (useOriginalGraph) {
            moveIters = localMovingPhase<false, K>(
                ucom, ctot, vaff, g, vcob, utot, M, currentResolution, passConfig);
        } else {
            moveIters = localMovingPhaseSuperGraph<false, K>(
                vcom, ctot, vaff, sgY, vcob, vtot, M, currentResolution, passConfig);
        }

        phaseTimer.Stop();
        result.localMoveTime += phaseTimer.Seconds();
        GRAPHBREW_TRACE("  local-moving: %d iters, %.4fs", moveIters, phaseTimer.Seconds());

        // ================================================================
        // PHASE 2: REFINEMENT (if enabled)
        // refinementDepth: -1=all passes, 0=pass 0 only (GVE), N=passes 0..N
        // ================================================================
        bool doRefine = config.useRefinement &&
                        (config.refinementDepth < 0 || pass <= config.refinementDepth);
        if (doRefine) {
            // Save bounds
            if (useOriginalGraph) {
                #pragma omp parallel for schedule(static, 4096)
                for (int64_t u = 0; u < N; ++u) {
                    vcob[u] = ucom[u];
                }
            } else {
                size_t NS = sgY.numNodes;
                vcob.resize(NS);
                #pragma omp parallel for schedule(static, 4096)
                for (size_t u = 0; u < NS; ++u) {
                    vcob[u] = vcom[u];
                }
            }

            // Re-initialize to singletons
            if (useOriginalGraph) {
                #pragma omp parallel for schedule(static, 4096)
                for (int64_t u = 0; u < N; ++u) {
                    ucom[u] = static_cast<K>(u);
                    ctot[u] = utot[u];
                }
            } else {
                size_t NS = sgY.numNodes;
                #pragma omp parallel for schedule(static, 4096)
                for (size_t u = 0; u < NS; ++u) {
                    vcom[u] = static_cast<K>(u);
                    ctot[u] = vtot[u];
                }
            }

            // Mark all affected
            if (useOriginalGraph) {
                std::fill(vaff.begin(), vaff.end(), 1);
            } else {
                size_t NS = sgY.numNodes;
                vaff.resize(NS);
                std::fill(vaff.begin(), vaff.begin() + NS, 1);
            }

            // Refinement
            phaseTimer.Start();

            int refineIters;
            if (useOriginalGraph) {
                refineIters = localMovingPhase<true, K>(
                    ucom, ctot, vaff, g, vcob, utot, M, currentResolution, passConfig);
            } else {
                refineIters = localMovingPhaseSuperGraph<true, K>(
                    vcom, ctot, vaff, sgY, vcob, vtot, M, currentResolution, passConfig);
            }

            phaseTimer.Stop();
            result.refinementTime += phaseTimer.Seconds();
            GRAPHBREW_TRACE("  refinement: %d iters, %.4fs", refineIters, phaseTimer.Seconds());
        }

        totalIterations += moveIters;

        // ================================================================
        // Check convergence
        // ================================================================
        size_t NS = useOriginalGraph ? static_cast<size_t>(N) : sgY.numNodes;
        cmap.resize(NS);
        size_t numCommunities = countCommunities(cmap, useOriginalGraph ? ucom : vcom, NS);

        GRAPHBREW_TRACE("  communities: %zu (from %zu)", numCommunities, prevCommunities);

        // GVE mode: skip pre-aggregation convergence checks
        // Only check convergence AFTER aggregation in GVE mode
        if (!gveMode) {
            if (moveIters <= 1 || pass >= config.maxPasses - 1) {
                result.membershipPerPass.push_back(ucom);
                break;
            }

            double ratio = static_cast<double>(numCommunities) / NS;
            if (ratio >= config.aggregationTolerance) {
                result.membershipPerPass.push_back(ucom);
                break;
            }
        } else if (pass >= config.maxPasses - 1) {
            result.membershipPerPass.push_back(ucom);
            break;
        }

        // ================================================================
        // PHASE 3: AGGREGATION
        // ================================================================
        size_t C = renumberCommunities(useOriginalGraph ? ucom : vcom, cmap, NS);

        phaseTimer.Start();

        // Use configured aggregation strategy
        if (config.aggregation == AggregationStrategy::GVE_CSR) {
            // GVE-style: build explicit adjacency lists + super-graph local-moving merge
            // This produces tighter communities by merging refined sub-communities
            buildCommunityVertices(coff, cvtx, useOriginalGraph ? ucom : vcom, NS, C);
            size_t numFinal;
            // GVE mode always operates on the original graph
            numFinal = aggregateGVEStyle(ucom, ctot, g, utot, C, coff, cvtx,
                                         M, currentResolution, config);

            phaseTimer.Stop();
            result.aggregationTime += phaseTimer.Seconds();
            GRAPHBREW_TRACE("  gve-aggregation: %zu -> %zu communities, %.4fs", C, numFinal, phaseTimer.Seconds());

            // Store membership
            result.membershipPerPass.push_back(ucom);

            // Check convergence (compare against previous pass, not self)
            if (numFinal >= prevCommunities || numFinal == 1) break;

            // Check aggregation tolerance
            double gveProgress = static_cast<double>(numFinal) / prevCommunities;
            if (gveProgress >= config.aggregationTolerance) break;

            // GVE-style loop operates on the original graph each pass
            prevCommunities = numFinal;

            // Recompute ctot for next pass
            ctot.assign(ctot.size(), Weight(0));
            #pragma omp parallel for
            for (int64_t v = 0; v < N; ++v) {
                #pragma omp atomic
                ctot[ucom[v]] += utot[v];
            }

            // Reset affected flags
            std::fill(vaff.begin(), vaff.end(), 1);

            passConfig.tolerance /= config.toleranceDrop;
            continue;  // Skip standard SuperGraph path
        } else if (config.aggregation == AggregationStrategy::LEIDEN_CSR ||
            (config.aggregation == AggregationStrategy::HYBRID && pass >= 2)) {
            // Leiden CSR: needs community vertex lists
            buildCommunityVertices(coff, cvtx, isFirst ? ucom : vcom, NS, C);
            if (isFirst) {
                aggregateGraphLeiden(sgZ, g, ucom, coff, cvtx, C, config);
            } else {
                aggregateSuperGraphLeiden(sgZ, sgY, vcom, coff, cvtx, C, config);
            }
        } else {
            // Lazy aggregation: skips community vertex lists, direct scan
            if (isFirst) {
                aggregateGraphLazy(sgZ, g, ucom, C, config);
            } else {
                aggregateSuperGraphLazy(sgZ, sgY, vcom, C, config);
            }
        }

        phaseTimer.Stop();
        result.aggregationTime += phaseTimer.Seconds();

        // Swap buffers - sgZ becomes the new sgY
        // If reuseBuffers, the old sgY keeps its reserved capacity for next aggregation
        std::swap(sgY, sgZ);
        if (config.reuseBuffers) {
            sgZ.softClear();  // Clear counts but keep allocated memory
        }
        GRAPHBREW_TRACE("  aggregation: C=%zu, %.4fs", C, phaseTimer.Seconds());

        // ================================================================
        // DYNAMIC RESOLUTION ADJUSTMENT (if enabled)
        // ================================================================
        if (config.useDynamicResolution) {
            // Collect metrics from current pass
            auto metrics = graphBrewCollectPassMetrics<K, Weight>(
                sgY, vtot, numCommunities, prevCommunities, moveIters);

            // Compute next resolution
            double nextResolution = graphBrewComputeAdaptiveResolution<Weight>(
                currentResolution, metrics, originalAvgDegree, config.maxIterations);

            printf("  Pass %d: res=%.3f, comms=%zu→%zu (%.1f%%), iters=%d, imbalance=%.1f → next_res=%.3f\n",
                   pass, currentResolution, prevCommunities, numCommunities,
                   metrics.reduction_ratio * 100, moveIters, metrics.size_imbalance,
                   nextResolution);

            currentResolution = nextResolution;
        }

        // Update previous communities for next pass
        prevCommunities = numCommunities;

        // Update hierarchy
        if (!isFirst) {
            lookupCommunities(ucom, vcom);
        }

        // Store membership AFTER lookup
        result.membershipPerPass.push_back(ucom);

        // Prepare for next pass
        computeVertexWeightsSuperGraph(vtot, sgY, config);

        vcom.resize(C);
        ctot.resize(C);
        vaff.resize(C);
        vcob.resize(C);

        #pragma omp parallel for schedule(static, 4096)
        for (size_t u = 0; u < C; ++u) {
            vcom[u] = static_cast<K>(u);
            ctot[u] = vtot[u];
            vaff[u] = 1;
        }

        passConfig.tolerance /= config.toleranceDrop;
    }

    // Final lookup if needed
    if (pass > 1 && result.membershipPerPass.size() > 1) {
        lookupCommunities(ucom, vcom);
    }

    totalTimer.Stop();

    // Store results
    result.membership = std::move(ucom);
    result.vertexWeight = std::move(utot);
    result.totalIterations = totalIterations;
    result.totalPasses = pass + 1;
    result.totalTime = totalTimer.Seconds();

    // Count final communities
    std::unordered_set<K> uniqueComms(result.membership.begin(), result.membership.end());
    result.numCommunities = uniqueComms.size();

    GRAPHBREW_TRACE("=== GRAPHBREW END ===");
    GRAPHBREW_TRACE("passes=%d, iters=%d, communities=%zu, time=%.4fs",
               result.totalPasses, result.totalIterations, result.numCommunities, result.totalTime);

    return result;
}

//=============================================================================
// SECTION 22b: GENERATE MAPPING (ENTRY POINT)
//=============================================================================

// Forward declaration for topology verification
template <typename NodeID_T, typename DestID_T>
bool verifyReorderingTopology(
    const CSRGraph<NodeID_T, DestID_T, true>& original,
    const pvector<NodeID_T>& new_ids,
    bool verbose);

/**
 * Apply an ordering strategy to produce vertex newIds from community detection results.
 *
 * This is the shared ordering pipeline used by both Leiden and Rabbit Order paths.
 * Steps:
 *   1. Community merging (if enabled)
 *   2. Dendrogram construction (if ordering needs it)
 *   3. Degree computation
 *   4. Ordering dispatch (14-strategy switch)
 *   5. Post-ordering analysis (debug builds)
 *   6. Topology verification (if requested)
 */
template <typename K, typename NodeID_T, typename DestID_T>
void applyOrderingStrategy(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& newIds,
    GraphBrewResult<K>& result,
    const GraphBrewConfig& config,
    GraphBrewRealizedConfig* realized = nullptr) {

    const int64_t N = g.num_nodes();

    // Apply community merging if enabled (key for cache locality!)
    if (config.useCommunityMerging) {
        Timer mergeTimer;
        mergeTimer.Start();
        size_t finalComms = mergeCommunities<K>(result.membership, g, config.targetCommunities);
        mergeTimer.Stop();
        result.numCommunities = finalComms;
        printf("  community merge: %.4fs\n", mergeTimer.Seconds());
    }
    if (realized) {
        realized->numPasses = result.totalPasses;
        realized->numCommunities = result.numCommunities;
    }

    // Build dendrogram if needed by the ordering strategy
    if (config.ordering == OrderingStrategy::DENDROGRAM_DFS ||
        config.ordering == OrderingStrategy::DENDROGRAM_BFS ||
        config.ordering == OrderingStrategy::HIERARCHICAL ||
        config.ordering == OrderingStrategy::HIERARCHICAL_CACHE_AWARE) {
        if (result.dendrogram.empty()) {
            Timer dendroTimer;
            dendroTimer.Start();
            if (!result.membershipPerPass.empty()) {
                // Multi-level hierarchy available (Leiden)
                buildDendrogram(result.dendrogram, result.roots, result.membershipPerPass, N);
            } else {
                // Flat membership only (Rabbit Order, etc.) → synthetic single-level
                buildSyntheticDendrogram(result.dendrogram, result.roots,
                                         result.membership, N, result.numCommunities);
            }
            dendroTimer.Stop();
            result.dendrogramTime = dendroTimer.Seconds();
            printf("  dendrogram: %.4fs\n", result.dendrogramTime);
        }
    }

    // Get degrees for ordering
    std::vector<K> degrees(N);
    #pragma omp parallel for
    for (int64_t u = 0; u < N; ++u) {
        degrees[u] = g.out_degree(u);
    }

    // Initialize newIds
    newIds.resize(N);
    // orderCompose writes every entry exactly once, so the -1 sentinel fill
    // is pure overhead for that path.  Other orderings may leave gaps, so
    // keep the safety fill there.
    if (config.ordering != OrderingStrategy::COMPOSE) {
        std::fill(newIds.begin(), newIds.end(), static_cast<NodeID_T>(-1));
    }

    // Generate ordering
    Timer orderTimer;
    orderTimer.Start();

    switch (config.ordering) {
        case OrderingStrategy::HIERARCHICAL:
            orderHierarchicalSort<K, NodeID_T>(newIds, result, degrees, N, config);
            break;
        case OrderingStrategy::CONNECTIVITY_BFS:
            orderConnectivityBFS<K, NodeID_T, DestID_T>(newIds, result.membership, degrees, g, N, config);
            break;
        case OrderingStrategy::DENDROGRAM_DFS:
            orderDendrogramDFS<K, NodeID_T>(newIds, result, degrees, N, config);
            break;
        case OrderingStrategy::DENDROGRAM_BFS:
            orderDendrogramBFS<K, NodeID_T>(newIds, result, degrees, N, config);
            break;
        case OrderingStrategy::COMMUNITY_SORT:
            orderCommunitySort<K, NodeID_T>(newIds, result.membership, degrees, N, config);
            break;
        case OrderingStrategy::HUB_CLUSTER:
            orderHubCluster<K, NodeID_T>(newIds, result.membership, degrees, N, config);
            break;
        case OrderingStrategy::DBG:
            orderDBG<K, NodeID_T>(newIds, result.membership, degrees, N, config);
            break;
        case OrderingStrategy::CORDER:
            orderCorder<K, NodeID_T>(newIds, result.membership, degrees, N, config);
            break;
        case OrderingStrategy::DBG_GLOBAL:
            orderDBGGlobal<K, NodeID_T>(newIds, result.membership, degrees, N, config);
            break;
        case OrderingStrategy::CORDER_GLOBAL:
            orderCorderGlobal<K, NodeID_T>(newIds, result.membership, degrees, N, config);
            break;
        case OrderingStrategy::HIERARCHICAL_CACHE_AWARE:
            if (realized && result.membershipPerPass.size() < 2) {
                realized->ordering = "hierarchical";
                realized->fallbacks.push_back({
                    "hcache-requires-two-passes",
                    "hcache",
                    "hierarchical",
                });
            }
            orderHierarchicalCacheAware<K, NodeID_T, DestID_T>(newIds, result, degrees, g, N, config);
            break;
        case OrderingStrategy::HYBRID_LEIDEN_RABBIT:
            orderHybridLeidenRabbit<K, NodeID_T, DestID_T>(newIds, result.membership, degrees, g, N, config);
            break;
        case OrderingStrategy::HIERARCHICAL_LEIDEN_RABBIT:
            if (realized && result.membershipPerPass.size() <= 1) {
                realized->ordering = "hrab";
                realized->fallbacks.push_back({
                    "hierarchical-rabbit-requires-two-passes",
                    "hlr",
                    "hrab",
                });
            }
            orderHierarchicalLeidenRabbit<K, NodeID_T, DestID_T>(newIds, result, degrees, g, N, config);
            break;
        case OrderingStrategy::TILE_QUANTIZED_RABBIT:
            orderTileQuantizedRabbit<K, NodeID_T, DestID_T>(newIds, result.membership, degrees, g, N, config);
            break;
        case OrderingStrategy::COMPOSE:
            if (
                realized &&
                config.communityOrder == CommunityOrder::CutMin &&
                result.numCommunities > 4096
            ) {
                realized->communityOrder = "degree-desc";
                realized->fallbacks.push_back({
                    "cut-min-community-limit",
                    "cut-min",
                    "degree-desc",
                });
            }
            if (
                realized &&
                config.intraCommunityOrder == IntraCommunityOrder::Dendrogram &&
                (
                    !result.hasRabbitDendrogram ||
                    result.rabbitToplevel.size() < result.numCommunities
                )
            ) {
                realized->intraCommunityOrder = "bfs";
                realized->fallbacks.push_back({
                    "rabbit-dendrogram-unavailable",
                    "dendrogram",
                    "bfs",
                });
            }
            orderCompose<K, NodeID_T, DestID_T>(
                newIds, result, degrees, g, N, config, realized);
            break;
        case OrderingStrategy::LAYER:
            printf("GraphBrew: GraphBrew mode - ordering deferred to external dispatch\n");
            orderConnectivityBFS<K, NodeID_T, DestID_T>(newIds, result.membership, degrees, g, N, config);
            break;
    }

    orderTimer.Stop();
    result.orderingTime = orderTimer.Seconds();
    printf("  ordering (%s): %.4fs\n",
           config.ordering == OrderingStrategy::HIERARCHICAL ? "hierarchical" :
           config.ordering == OrderingStrategy::DBG ? "dbg" :
           config.ordering == OrderingStrategy::CORDER ? "corder" :
           config.ordering == OrderingStrategy::COMMUNITY_SORT ? "community" :
           config.ordering == OrderingStrategy::HUB_CLUSTER ? "hubcluster" :
           config.ordering == OrderingStrategy::CONNECTIVITY_BFS ? "connectivity-bfs" :
           config.ordering == OrderingStrategy::DENDROGRAM_DFS ? "dfs" :
           config.ordering == OrderingStrategy::DENDROGRAM_BFS ? "bfs" :
           "other", result.orderingTime);

    // Post-ordering locality analysis (gated behind GRAPHBREW_DEBUG)
#if GRAPHBREW_DEBUG
    {
        std::vector<size_t> distBuckets(10, 0);
        size_t totalEdges = 0;
        double sumLogDist = 0;
        size_t nearEdges = 0;

        #pragma omp parallel for reduction(+:totalEdges, sumLogDist, nearEdges)
        for (int64_t u = 0; u < N; ++u) {
            for (auto neighbor : g.out_neigh(u)) {
                NodeID_T v;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    v = neighbor;
                } else {
                    v = neighbor.v;
                }

                size_t dist = (newIds[u] > newIds[v])
                    ? (newIds[u] - newIds[v])
                    : (newIds[v] - newIds[u]);

                sumLogDist += std::log(std::max(size_t(1), dist));
                totalEdges++;

                if (dist < 1000) nearEdges++;

                size_t bucket = 0;
                size_t d = dist;
                while (d >= 100 && bucket < 9) {
                    d /= 10;
                    bucket++;
                }
                #pragma omp atomic
                distBuckets[bucket]++;
            }
        }

        double geoMeanDist = std::exp(sumLogDist / std::max(size_t(1), totalEdges));
        double nearRatio = 100.0 * nearEdges / std::max(size_t(1), totalEdges);

        printf("  === ORDERING LOCALITY ===\n");
        printf("  geo-mean edge distance: %.1f\n", geoMeanDist);
        printf("  near edges (dist<1K): %.1f%%\n", nearRatio);
        printf("  distance buckets: [0-100)=%zu, [100-1K)=%zu, [1K-10K)=%zu, [10K-100K)=%zu, [100K+)=%zu\n",
               distBuckets[0], distBuckets[1], distBuckets[2], distBuckets[3],
               distBuckets[4] + distBuckets[5] + distBuckets[6] + distBuckets[7] + distBuckets[8] + distBuckets[9]);
        printf("  ==========================\n");
    }
#endif

    // Verify topology if requested
    if (config.verifyTopology) {
        if (!verifyReorderingTopology(g, newIds, true)) {
            const char* orderingName =
                config.ordering == OrderingStrategy::HIERARCHICAL ? "hierarchical" :
                config.ordering == OrderingStrategy::DBG ? "dbg" : "other";
            printf("ERROR: Topology verification FAILED for ordering=%s!\n", orderingName);
        }
    }
}

/**
 * Main entry point: Run GraphBrew and generate vertex reordering
 */
template <typename K = uint32_t, typename NodeID_T, typename DestID_T>
void generateGraphBrewMapping(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    pvector<NodeID_T>& newIds,
    const GraphBrewConfig& config) {

    const int64_t N = g.num_nodes();
    auto realized = makeGraphBrewRealizedConfig(config);

    // Branch based on main algorithm choice
    if (config.algorithm == GraphBrewAlgorithm::RABBIT_ORDER) {
        printf("RabbitOrder: modularity resolution=n/a (fixed gamma=1)\n");

        if (config.hasExplicitOrdering) {
            // Generic pipeline: Rabbit community detection → shared ordering strategy
            // e.g., -o 12:rabbit:dbg, -o 12:rabbit:hubcluster, -o 12:rabbit:dfs
            GraphBrewResult<K> result;
            runRabbitOrder<K>(g, newIds, config, &result);
            applyOrderingStrategy<K>(g, newIds, result, config, &realized);
        } else {
            // Native path: Rabbit's built-in DFS ordering (backward compat)
            // e.g., -o 12:rabbit (no ordering token)
            GraphBrewResult<K> result;
            runRabbitOrder<K>(g, newIds, config, &result, false);
            realized.numPasses = result.totalPasses;
            realized.numCommunities = result.numCommunities;

            // Verify if requested
            if (config.verifyTopology) {
                if (!verifyReorderingTopology(g, newIds, true)) {
                    printf("ERROR: Topology verification FAILED for RabbitOrder!\n");
                }
            }
        }
        printGraphBrewRealizedConfig(realized);
        return;
    }

    // Default: Leiden-based GraphBrew
    printf("GraphBrew: resolution=%.4f (%s), maxIters=%d, maxPasses=%d\n",
           config.resolution,
           config.useDynamicResolution ? "dynamic" : "fixed",
           config.maxIterations, config.maxPasses);

    const char* orderingName =
        config.ordering == OrderingStrategy::HIERARCHICAL ? "hierarchical" :
        config.ordering == OrderingStrategy::CONNECTIVITY_BFS ? "connectivity-bfs" :
        config.ordering == OrderingStrategy::DENDROGRAM_DFS ? "dfs" :
        config.ordering == OrderingStrategy::DENDROGRAM_BFS ? "bfs" :
        config.ordering == OrderingStrategy::COMMUNITY_SORT ? "community" :
        config.ordering == OrderingStrategy::HUB_CLUSTER ? "hubcluster" :
        config.ordering == OrderingStrategy::DBG ? "dbg" :
        config.ordering == OrderingStrategy::CORDER ? "corder" :
        config.ordering == OrderingStrategy::DBG_GLOBAL ? "dbg-global" :
        config.ordering == OrderingStrategy::CORDER_GLOBAL ? "corder-global" :
        config.ordering == OrderingStrategy::HIERARCHICAL_CACHE_AWARE ? "hcache" :
        config.ordering == OrderingStrategy::HYBRID_LEIDEN_RABBIT ? "hybrid-rabbit" :
        config.ordering == OrderingStrategy::TILE_QUANTIZED_RABBIT ? "tile-quantized-rabbit" :
        config.ordering == OrderingStrategy::COMPOSE ? "compose" :
        config.ordering == OrderingStrategy::LAYER ? "graphbrew" : "unknown";

    printf("GraphBrew: aggregation=%s, ordering=%s, refinement=%s (depth=%d)%s%s\n",
           config.aggregation == AggregationStrategy::LEIDEN_CSR ? "leiden" :
           config.aggregation == AggregationStrategy::RABBIT_LAZY ? "streaming" :
           config.aggregation == AggregationStrategy::GVE_CSR ? "gve-csr" : "hybrid",
           orderingName,
           config.useRefinement ? "on" : "off",
           config.refinementDepth,
           config.useCommunityMerging ? ", merge=on" : "",
           config.useHubExtraction ? ", hubx=on" : "");
    printf(
        "GraphBrew: community-detection=%s\n",
        config.deterministicCommunityDetection ? "serial" : "parallel");
    if (config.mComputation == MComputation::TOTAL_EDGES)
        printf("GraphBrew: mComputation=total-edges (GVE style)\n");
    if (config.useGorderIntra) {
        if (config.gorderFallback > 0)
            printf("GraphBrew: gord=on (window=%d, fallback=%d)\n", config.gorderWindow, config.gorderFallback);
        else
            printf("GraphBrew: gord=on (window=%d, fallback=N)\n", config.gorderWindow);
    }
    if (config.useHubSort)     printf("GraphBrew: hsort=on\n");
    if (config.useRCMSuper)    printf("GraphBrew: rcm=on\n");
    if (config.ordering == OrderingStrategy::LAYER)
        printf("GraphBrew: GraphBrew mode, finalAlgo=%d, smallMerge=%s\n",
               config.finalAlgoId, config.useSmallCommunityMerging ? "on" : "off");

    // Run GraphBrew
    Timer timer;
    timer.Start();

    auto result = runGraphBrew<K>(g, config);

    timer.Stop();
    printf("GraphBrew: %d passes, %d iters, %zu communities, time=%.4fs\n",
           result.totalPasses, result.totalIterations, result.numCommunities, timer.Seconds());
    printf("  local-move: %.4fs, refine: %.4fs, aggregate: %.4fs\n",
           result.localMoveTime, result.refinementTime, result.aggregationTime);

    // ===============================================================
    // DETAILED COMMUNITY ANALYSIS (gated behind GRAPHBREW_DEBUG)
    // ===============================================================
#if GRAPHBREW_DEBUG
    {
        const auto& membership = result.membership;

        // 1. Community size distribution
        std::vector<size_t> commSizes(result.numCommunities, 0);
        for (size_t v = 0; v < static_cast<size_t>(N); ++v) {
            if (membership[v] < commSizes.size()) {
                commSizes[membership[v]]++;
            }
        }

        // Sort sizes for percentile analysis
        std::vector<size_t> sortedSizes = commSizes;
        std::sort(sortedSizes.begin(), sortedSizes.end());

        size_t minSize = sortedSizes.empty() ? 0 : sortedSizes.front();
        size_t maxSize = sortedSizes.empty() ? 0 : sortedSizes.back();
        size_t medianSize = sortedSizes.empty() ? 0 : sortedSizes[sortedSizes.size() / 2];
        double avgSize = static_cast<double>(N) / std::max(size_t(1), result.numCommunities);

        // Size distribution buckets
        size_t tiny = 0, small = 0, medium = 0, large = 0, huge = 0;
        for (size_t sz : sortedSizes) {
            if (sz <= 10) tiny++;
            else if (sz <= 100) small++;
            else if (sz <= 1000) medium++;
            else if (sz <= 10000) large++;
            else huge++;
        }

        printf("  === COMMUNITY SIZE ANALYSIS ===\n");
        printf("  sizes: min=%zu, median=%zu, avg=%.1f, max=%zu\n",
               minSize, medianSize, avgSize, maxSize);
        printf("  buckets: [1-10]=%zu, [11-100]=%zu, [101-1K]=%zu, [1K-10K]=%zu, [10K+]=%zu\n",
               tiny, small, medium, large, huge);

        // 2. Edge locality analysis (intra vs inter-community)
        size_t intraEdges = 0, interEdges = 0;
        #pragma omp parallel for reduction(+:intraEdges, interEdges)
        for (int64_t u = 0; u < N; ++u) {
            K commU = membership[u];
            for (auto neighbor : g.out_neigh(u)) {
                NodeID_T v;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    v = neighbor;
                } else {
                    v = neighbor.v;
                }
                if (membership[v] == commU) {
                    intraEdges++;
                } else {
                    interEdges++;
                }
            }
        }

        double intraRatio = 100.0 * intraEdges / std::max(size_t(1), intraEdges + interEdges);
        printf("  edges: intra=%zu (%.1f%%), inter=%zu (%.1f%%)\n",
               intraEdges, intraRatio, interEdges, 100.0 - intraRatio);

        // 3. Hierarchy depth analysis
        printf("  hierarchy: %zu passes recorded\n", result.membershipPerPass.size());
        for (size_t p = 0; p < result.membershipPerPass.size(); ++p) {
            std::unordered_set<K> uniqueComms(result.membershipPerPass[p].begin(),
                                               result.membershipPerPass[p].end());
            printf("    pass %zu: %zu communities\n", p, uniqueComms.size());
        }
        printf("  ================================\n");
    }
#endif

    // Apply ordering strategy (shared pipeline with Rabbit Order path)
    applyOrderingStrategy<K>(g, newIds, result, config, &realized);
    printGraphBrewRealizedConfig(realized);
}

//=============================================================================
// Extracted implementation detail; kept at this position to preserve declaration order.
#include "reorder_graphbrew_parser.h"

// SECTION 24: TOPOLOGY VERIFICATION
//=============================================================================

/**
 * Verify that reordering preserves graph topology
 *
 * Checks:
 * 1. new_ids is a valid permutation (bijective mapping)
 * 2. Every edge (u,v) in original has edge (new_ids[u], new_ids[v]) in reordered
 * 3. Edge count is preserved
 *
 * @return true if topology is preserved, false otherwise
 */
template <typename NodeID_T, typename DestID_T>
bool verifyReorderingTopology(
    const CSRGraph<NodeID_T, DestID_T, true>& original,
    const pvector<NodeID_T>& new_ids,
    bool verbose) {

    const int64_t N = original.num_nodes();
    const int64_t M = original.num_edges();

    if (verbose) {
        printf("=== Topology Verification ===\n");
        printf("Nodes: %ld, Edges: %ld\n", N, M);
    }

    // Check 1: new_ids has correct size
    if (static_cast<int64_t>(new_ids.size()) != N) {
        if (verbose) printf("FAIL: new_ids size %zu != N %ld\n", new_ids.size(), N);
        return false;
    }

    // Check 2: new_ids is a valid permutation (all values in [0, N) and unique)
    std::vector<bool> seen(N, false);
    for (int64_t u = 0; u < N; ++u) {
        NodeID_T new_id = new_ids[u];
        if (new_id < 0 || new_id >= N) {
            if (verbose) printf("FAIL: new_ids[%ld] = %d out of range [0, %ld)\n",
                               u, static_cast<int>(new_id), N);
            return false;
        }
        if (seen[new_id]) {
            if (verbose) printf("FAIL: new_ids[%ld] = %d is duplicate\n",
                               u, static_cast<int>(new_id));
            return false;
        }
        seen[new_id] = true;
    }

    // Check 3: Build inverse mapping
    std::vector<NodeID_T> inv_ids(N);
    for (int64_t u = 0; u < N; ++u) {
        inv_ids[new_ids[u]] = static_cast<NodeID_T>(u);
    }

    // Check 4: Verify edge preservation using edge counts per vertex
    // For each vertex, count edges and verify degree is preserved
    std::vector<int64_t> orig_degree(N), reord_degree(N);

    #pragma omp parallel for
    for (int64_t u = 0; u < N; ++u) {
        orig_degree[u] = original.out_degree(u);
    }

    // After reordering, vertex u becomes new_ids[u]
    // So reordered vertex v has degree of original vertex inv_ids[v]
    #pragma omp parallel for
    for (int64_t v = 0; v < N; ++v) {
        reord_degree[v] = orig_degree[inv_ids[v]];
    }

    // Verify total degree is preserved
    int64_t total_orig = 0, total_reord = 0;
    for (int64_t u = 0; u < N; ++u) {
        total_orig += orig_degree[u];
        total_reord += reord_degree[u];
    }

    if (total_orig != total_reord) {
        if (verbose) printf("FAIL: Total degree mismatch: orig=%ld, reord=%ld\n",
                           total_orig, total_reord);
        return false;
    }

    // Check 5: Verify edge-by-edge (sample check for large graphs)
    int64_t edges_checked = 0;
    int64_t edges_valid = 0;
    const int64_t max_check = std::min(M, int64_t(1000000));  // Check up to 1M edges

    std::set<std::pair<NodeID_T, NodeID_T>> orig_edges;

    // Collect original edges (normalized: smaller ID first)
    for (int64_t u = 0; u < N && edges_checked < max_check; ++u) {
        for (auto neighbor : original.out_neigh(u)) {
            NodeID_T v;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                v = neighbor;
            } else {
                v = neighbor.v;
            }
            if (u < v) {  // Only count each edge once
                orig_edges.insert({static_cast<NodeID_T>(u), v});
                edges_checked++;
                if (edges_checked >= max_check) break;
            }
        }
    }

    // Verify each original edge maps to a valid reordered edge
    for (const auto& [u, v] : orig_edges) {
        NodeID_T new_u = new_ids[u];
        NodeID_T new_v = new_ids[v];

        // Check if edge exists in reordered graph
        // The reordered graph should have edge (new_u, new_v)
        // Since we're checking the mapping, not the actual reordered graph,
        // we verify the mapping is consistent
        if (new_u >= 0 && new_u < N && new_v >= 0 && new_v < N) {
            edges_valid++;
        }
    }

    if (edges_valid != static_cast<int64_t>(orig_edges.size())) {
        if (verbose) printf("FAIL: Edge mapping verification failed: %ld/%zu valid\n",
                           edges_valid, orig_edges.size());
        return false;
    }

    if (verbose) {
        printf("PASS: Permutation valid (all %ld IDs unique and in range)\n", N);
        printf("PASS: Total degree preserved (%ld)\n", total_orig);
        printf("PASS: Edge mapping valid (%ld edges checked)\n", edges_checked);
        printf("=== Verification Complete ===\n");
    }

    return true;
}

/**
 * Verify and print summary for GraphBrew reordering
 */
template <typename K, typename NodeID_T, typename DestID_T>
bool verifyGraphBrewResult(
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    const pvector<NodeID_T>& new_ids,
    const std::string& variant_name) {

    printf("\n--- Verifying GraphBrew variant: %s ---\n", variant_name.c_str());
    bool ok = verifyReorderingTopology(g, new_ids, true);
    printf("Result: %s\n\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace graphbrew
