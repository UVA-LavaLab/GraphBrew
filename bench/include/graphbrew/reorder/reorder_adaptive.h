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
 *    Uses the shared Tier-0 schema:
 *    - log graph dimensions, average degree, degree CV, hub concentration
 *    - normalized sampled edge span and cache-line neighbor overlap
 *    - kernel-specific property/LLC ratio, kernel class, and reuse bucket
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
 * -o 14[:_[:_[:model[:criterion]]]]
 *
 * Parameters (positions relative to algorithm ID):
 *   0-2: Reserved (currently unused by standalone entry point)
 *   3: model: perceptron (deployable model)
 *   4: criterion: fastest-reorder, fastest-execution (default),
 *                 best-endtoend, or best-amortization
 *
 * Examples:
 *   -o 14                                  # perceptron + fastest-execution
 *   -o 14::::perceptron:best-endtoend      # independent model/criterion
 * Decision-tree and hybrid artifacts remain offline-only until they are
 * retrained on the Tier-0 schema.
 *
 * Graph names, runtime benchmark-database kNN, and exact-name oracle lookup
 * are prohibited in deployable AdaptiveOrder. Exact-name comparisons belong
 * to the offline OracleUpperBound analysis.
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
 * RUNTIME BEHAVIOR
 * ============================================================================
 *
 * The standalone entry point (GenerateAdaptiveMappingStandalone) always
 * delegates to GenerateAdaptiveMappingFullGraphStandalone. Full-graph
 * mode was found to outperform per-community mode because:
 *   1. Training data is whole-graph, so features match better
 *   2. No Leiden partitioning overhead
 *   3. Cross-community edge patterns are preserved
 *
 * GenerateAdaptiveMappingRecursiveStandalone exists but is not called
 * from the CLI entry point.
 *
 * Author: GraphBrew Team
 * License: See LICENSE.txt
 */

#ifndef REORDER_ADAPTIVE_H_
#define REORDER_ADAPTIVE_H_

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
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
    BenchmarkType benchmark = BENCH_GENERIC;
    bool verbose = false;
    
    /**
     * Parse configuration from reordering options
     * Format: mode:depth:resolution:min_size:selection_mode
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
    }
};

struct DeployableSelectionPolicy {
    SelectionModel model = SELECTION_MODEL_PERCEPTRON;
    SelectionCriterion criterion = CRITERION_FASTEST_EXECUTION;
    double reuse_count = 1.0;
    bool reuse_count_explicit = false;
};

inline bool IsDeployableSelectionModel(SelectionModel model) {
    return (
        model == SELECTION_MODEL_PERCEPTRON
        || model == SELECTION_MODEL_BUDGETED_RULE
    );
}

inline DeployableSelectionPolicy ParseDeployableSelectionPolicy(
    const std::vector<std::string>& options) {
    DeployableSelectionPolicy policy;
    for (size_t i = 0; i < std::min<size_t>(3, options.size()); ++i) {
        if (!options[i].empty() && options[i] != "_") {
            throw std::invalid_argument(
                "AdaptiveOrder reserved fields must be empty");
        }
    }
    const std::string model_spec = (
        options.size() > 3 ? options[3] : "");
    const std::string criterion_spec = (
        options.size() > 4 ? options[4] : "");
    const std::string reuse_spec = (
        options.size() > 5 ? options[5] : "");

    if (!model_spec.empty() && model_spec != "_") {
        if (!criterion_spec.empty() && criterion_spec != "_") {
            if (model_spec.find(':') != std::string::npos) {
                throw std::invalid_argument(
                    "AdaptiveOrder model and criterion were specified twice");
            }
            policy.model = GetSelectionModel(model_spec);
            policy.criterion = GetSelectionCriterion(criterion_spec);
        } else {
            auto parsed = ParseModelCriterion(model_spec);
            policy.model = parsed.first;
            policy.criterion = parsed.second;
        }
    } else if (!criterion_spec.empty() && criterion_spec != "_") {
        throw std::invalid_argument(
            "AdaptiveOrder criterion requires an explicit model");
    }

    if (!reuse_spec.empty() && reuse_spec != "_") {
        size_t parsed = 0;
        policy.reuse_count = std::stod(reuse_spec, &parsed);
        if (
            parsed != reuse_spec.size()
            || !std::isfinite(policy.reuse_count)
            || policy.reuse_count <= 0.0
        ) {
            throw std::invalid_argument(
                "AdaptiveOrder reuse count must be positive");
        }
        policy.reuse_count_explicit = true;
    }

    for (size_t i = 6; i < options.size(); ++i) {
        if (!options[i].empty() && options[i] != "_") {
            throw std::invalid_argument(
                "AdaptiveOrder no longer accepts graph identity or "
                "additional runtime selection fields");
        }
    }
    if (!IsDeployableSelectionModel(policy.model)) {
        throw std::invalid_argument(
            SelectionModelToString(policy.model)
            + " is offline-only and cannot drive deployable AdaptiveOrder");
    }
    if (
        policy.model == SELECTION_MODEL_BUDGETED_RULE
        && !policy.reuse_count_explicit
    ) {
        throw std::invalid_argument(
            "budgeted-rule requires an explicit reuse count");
    }
    return policy;
}

inline bool ShouldUseBudgetedLeidenGorder(
    const CommunityFeatures& features,
    BenchmarkType benchmark,
    double reuse_count
) {
    const bool pr_kernel = (
        benchmark == BENCH_PR
        || benchmark == BENCH_PR_SPMV
    );
    return (
        pr_kernel
        && reuse_count <= 1.0
        && features.num_nodes >= 1000
        && features.avg_degree >= 45.0
        && features.degree_variance < 3.0
        && features.hub_concentration >= 0.3
    );
}

inline PerceptronSelection SelectBudgetedOneUseRule(
    const CommunityFeatures& features,
    BenchmarkType benchmark,
    double reuse_count
) {
    if (ShouldUseBudgetedLeidenGorder(
        features, benchmark, reuse_count
    )) {
        PerceptronSelection selected;
        selected.algo = GraphBrewOrder;
        selected.variant_name = (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_gorder:gw32:gordf500:cd_parallel:1:1"
        );
        selected.canonical_spec = selected.variant_name;
        selected.predicted_spec = selected.variant_name;
        selected.options = {
            "leiden",
            "compose",
            "sg_none",
            "comm_identity",
            "intra_gorder",
            "gw32",
            "gordf500",
            "cd_parallel",
            "1",
            "1",
        };
        selected.override_reason = "budgeted-one-use-match";
        selected.confidence = 1.0;
        return selected;
    }
#ifdef RABBIT_ENABLE
    auto fallback = ResolveDeployableAdaptiveArm("8:csr");
#else
    auto fallback = ResolveDeployableAdaptiveArm("5");
#endif
    fallback.predicted_spec = fallback.canonical_spec;
    fallback.override_reason = "budgeted-one-use-fallback";
    fallback.confidence = 1.0;
    return fallback;
}

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

inline PerceptronSelection EnforceDeployableAdaptivePortfolio(
    const PerceptronSelection& predicted) {
    try {
        auto applied = ResolveDeployableAdaptiveArm(
            predicted.variant_name, predicted.score);
        applied.margin = predicted.margin;
        applied.confidence = predicted.confidence;
        applied.explored = predicted.explored;
        applied.predicted_spec = predicted.predicted_spec;
        applied.override_reason = predicted.override_reason;
        return applied;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error(
            "Deployable adaptive model emitted non-portfolio label: "
            + predicted.variant_name);
    }
}

template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void ApplyDeployableAdaptiveArm(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids,
    const PerceptronSelection& selected,
    bool useOutdeg) {
    if (selected.algo == GraphBrewOrder) {
        if constexpr (!invert) {
            throw std::invalid_argument(
                "Deployable GraphBrew adaptive arms require inverse CSR");
        } else {
            auto config = graphbrew::parseGraphBrewConfig(
                selected.options, true);
            graphbrew::generateGraphBrewMapping<uint32_t>(
                g, new_ids, config);
            return;
        }
    }
    ApplyBasicReorderingStandalone<NodeID_, DestID_, WeightT_, invert>(
        g, new_ids, selected, useOutdeg, "");
}

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
    const auto selection_policy =
        adaptive::ParseDeployableSelectionPolicy(reordering_options);
    
    std::cout << "=== Full-Graph Adaptive Mode (Standalone) ===\n";
    std::cout << "Nodes: " << num_nodes << ", Edges: " << num_edges << "\n";
    
    // GUARD: Empty graph - create identity mapping
    if (num_nodes == 0) {
        tm.Stop();
        PrintTime("AdaptiveOrder Total Time", tm.Seconds());
        return;
    }
    
    Timer feature_timer;
    feature_timer.Start();
    CommunityFeatures global_feat =
        ::ComputeTier0SampledGraphFeatures(g);
    const BenchmarkType benchmark = GetBenchmarkTypeHint();
    const uint64_t property_bytes = ModeledPropertyBytes(
        benchmark,
        static_cast<uint64_t>(num_nodes),
        static_cast<uint64_t>(num_edges));
    const size_t llc_bytes = GetLLCSizeBytes();
    Tier0FeatureContext tier0_context;
    tier0_context.property_wsr_llc = (
        llc_bytes > 0
        ? static_cast<double>(property_bytes) / llc_bytes
        : 0.0);
    tier0_context.kernel_class = benchmark;
    tier0_context.reuse_count = selection_policy.reuse_count;
    global_feat.working_set_ratio =
        tier0_context.property_wsr_llc;
    global_feat.kernel_class =
        static_cast<double>(benchmark);
    global_feat.reuse_count = tier0_context.reuse_count;
    const auto tier0_values = ExtractTier0Features(
        global_feat, tier0_context);
    feature_timer.Stop();

    const double global_modularity = 0.0;
    const double global_degree_variance = global_feat.degree_variance;
    const double global_hub_concentration =
        global_feat.hub_concentration;
    const double global_avg_degree = global_feat.avg_degree;
    
    // Detect graph type
    GraphType detected_graph_type = DetectGraphType(
        global_modularity, global_degree_variance, global_hub_concentration,
        global_avg_degree, static_cast<size_t>(num_nodes));
    
    std::cout << "Graph Type: " << GraphTypeToString(detected_graph_type) << "\n";
    PrintTime("Degree Variance", global_degree_variance);
    PrintTime("Hub Concentration", global_hub_concentration);
    PrintTime("Normalized Edge Span", global_feat.avg_reuse_distance);
    PrintTime("Window Neighbor Overlap",
              global_feat.window_neighbor_overlap);
    PrintTime("Packing Factor", global_feat.packing_factor);
    PrintTime(
        "Property Working Set Bytes",
        static_cast<double>(property_bytes));
    PrintTime("LLC Capacity Bytes", static_cast<double>(llc_bytes));
    PrintTime(
        "Property WSR LLC",
        tier0_context.property_wsr_llc);
    const auto old_flags = std::cout.flags();
    const auto old_precision = std::cout.precision();
    std::cout << std::defaultfloat << std::setprecision(17);
    std::cout << "Adaptive Tier0 Features: {";
    for (size_t i = 0; i < TIER0_FEATURE_COUNT; ++i) {
        if (i > 0) std::cout << ",";
        std::cout << "\"" << TIER0_FEATURE_NAMES[i]
                  << "\":" << tier0_values[i];
    }
    std::cout << "}\n";
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
    PrintTime("Adaptive Feature Time", feature_timer.Seconds());

    Timer model_timer;
    model_timer.Start();
    std::cout << "Selection: model="
              << SelectionModelToString(selection_policy.model)
              << " criterion="
              << SelectionCriterionToString(selection_policy.criterion)
              << "\n";
    
    // Select best algorithm
    PerceptronSelection best;
    if (
        selection_policy.model
        == SELECTION_MODEL_BUDGETED_RULE
    ) {
        if (
            selection_policy.criterion
            != CRITERION_BEST_ENDTOEND
        ) {
            throw std::invalid_argument(
                "budgeted-rule requires best-endtoend criterion");
        }
        best = adaptive::SelectBudgetedOneUseRule(
            global_feat,
            benchmark,
            selection_policy.reuse_count);
    } else {
        best = SelectBestReorderingForCommunityWithModelCriterion(
            global_feat, global_modularity,
            global_degree_variance, global_hub_concentration,
            global_avg_degree, static_cast<size_t>(num_nodes),
            num_edges, selection_policy.model,
            selection_policy.criterion,
            benchmark, detected_graph_type);
    }
    
    const int top_k = AblationConfig::Get().top_k;
    if (top_k > 1) {
        throw std::invalid_argument(
            "ADAPTIVE_TOP_K is offline-only; deployable AdaptiveOrder "
            "cannot trial multiple reorderers");
    }

    const std::string predicted_label = (
        best.predicted_spec.empty()
        ? best.variant_name
        : best.predicted_spec);
    if (
        selection_policy.model
        != SELECTION_MODEL_BUDGETED_RULE
    ) {
        best = EnforceDeployableAdaptivePortfolio(best);
    }
    model_timer.Stop();
    PrintTime("Adaptive Model Time", model_timer.Seconds());
    PrintTime(
        "Adaptive Selection Time",
        feature_timer.Seconds() + model_timer.Seconds());
    PrintLabel("Adaptive Predicted", predicted_label);
    PrintLabel("Adaptive Applied", best.canonical_spec);
    PrintLabel(
        "Adaptive Override Reason",
        best.override_reason.empty() ? "none" : best.override_reason);
    PrintTime("Adaptive Confidence", best.confidence);
    {
        auto& staged =
            graphbrew::database::GetStagedReorderMeta();
        std::ostringstream spec;
        spec << "14:model="
             << SelectionModelToString(selection_policy.model)
             << ":criterion="
             << SelectionCriterionToString(
                    selection_policy.criterion)
             << ":applied=" << best.canonical_spec;
        staged.algorithm_spec = spec.str();
        staged.schedule_sensitive =
            best.canonical_spec.find("8:") == 0
            || best.canonical_spec.find("rabbit") !=
                std::string::npos;
    }
    std::cout << "\n=== Selected Algorithm: " << best.canonical_spec
              << " ===\n";
    Timer arm_timer;
    arm_timer.Start();
    ApplyDeployableAdaptiveArm<NodeID_, DestID_, WeightT_, invert>(
        g, new_ids, best, useOutdeg);
    arm_timer.Stop();
    PrintTime("Adaptive Arm Map Time", arm_timer.Seconds());
    
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
    SelectionModel selection_model = SELECTION_MODEL_PERCEPTRON,
    SelectionCriterion selection_criterion =
        CRITERION_FASTEST_EXECUTION) {
    throw std::invalid_argument(
        "Recursive AdaptiveOrder is retired; deployable selection is "
        "full-graph over the frozen exact portfolio");
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    const int64_t num_edges = g.num_edges_directed();
    
    // GUARD: Empty graph - nothing to partition
    if (num_nodes == 0) {
        tm.Stop();
        PrintTime("Adaptive Map Time", tm.Seconds());
        return;
    }
    
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
        // Print active ablation toggles
        AblationConfig::Get().print();
    }
    
    // Ablation: ADAPTIVE_NO_LEIDEN=1 — skip Leiden, treat whole graph as one community.
    // All nodes go to community 0, bypassing partitioning entirely.
    
    // Use GraphBrew's Leiden engine for community detection (native CSR)
    // NOTE: Parallel Leiden (OMP_NUM_THREADS > 1) is non-deterministic due to
    // concurrent community updates in localMovingPhase. For reproducible results,
    // set OMP_NUM_THREADS=1 or use precomputed label maps (--precompute).
    Timer t_leiden;
    t_leiden.Start();
    
    std::vector<K> comm_ids_k;
    double global_modularity = 0.0;
    
    if (AblationConfig::Get().no_leiden) {
        // Ablation: skip Leiden, treat entire graph as one community
        comm_ids_k.assign(num_nodes, K(0));
        global_modularity = 0.0;
        if (verbose) printf("ABLATION: Leiden skipped, single community\n");
    } else {
        graphbrew::GraphBrewConfig gb_config;
        gb_config.resolution = resolution;
        gb_config.maxIterations = max_iterations;
        gb_config.maxPasses = max_passes;
        gb_config.ordering = graphbrew::OrderingStrategy::COMMUNITY_SORT;
        auto gb_result = graphbrew::runGraphBrew<K>(g, gb_config);
        comm_ids_k = gb_result.membership;
        global_modularity = gb_result.modularity;
    }
    
    t_leiden.Stop();
    
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
    Timer t_features;
    t_features.Start();
    auto deg_features = ::ComputeSampledDegreeFeatures(g, 0, true);
    double global_degree_variance = deg_features.degree_variance;
    double global_hub_concentration = deg_features.hub_concentration;
    double global_avg_degree = (num_nodes > 0) ? static_cast<double>(num_edges) / num_nodes : 0.0;
    t_features.Stop();
    
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
    
    // Stage timers for per-community work
    double t_comm_features_total = 0.0;
    double t_comm_scoring_total = 0.0;
    double t_comm_reorder_total = 0.0;
    
    // First: handle small communities
    double t_small_total = 0.0;
    if (!small_community_nodes.empty()) {
        Timer t_small;
        t_small.Start();
        std::unordered_set<NodeID_> small_node_set(
            small_community_nodes.begin(), small_community_nodes.end());
        
        auto merged_feat = ::ComputeMergedCommunityFeatures(g, small_community_nodes, small_node_set);
        
        // Use whole-graph features for the merged small-community group.
        // The perceptron was trained on whole-graph features, and the merged
        // group is 96-98% of nodes, so global features are a much better match
        // than recomputed subgraph features. This avoids the distribution
        // mismatch between training data and runtime features.
        CommunityFeatures comm_feat;
        comm_feat.num_nodes = num_nodes;
        comm_feat.num_edges = num_edges;
        comm_feat.internal_density = global_avg_degree / std::max(1.0, static_cast<double>(num_nodes - 1));
        comm_feat.degree_variance = deg_features.degree_variance;
        comm_feat.hub_concentration = deg_features.hub_concentration;
        comm_feat.clustering_coeff = deg_features.clustering_coeff;
        comm_feat.packing_factor = deg_features.packing_factor;
        comm_feat.forward_edge_fraction = deg_features.forward_edge_fraction;
        comm_feat.working_set_ratio = deg_features.working_set_ratio;
        comm_feat.vertex_significance_skewness = deg_features.vertex_significance_skewness;
        comm_feat.window_neighbor_overlap = deg_features.window_neighbor_overlap;
        comm_feat.sampled_locality_score = deg_features.sampled_locality_score;
        comm_feat.avg_reuse_distance = deg_features.avg_reuse_distance;
        comm_feat.packing_factor_cl = deg_features.packing_factor_cl;
        comm_feat.locality_score_pairwise = deg_features.locality_score_pairwise;
        comm_feat.reuse_distance_lru = deg_features.reuse_distance_lru;
        const BenchmarkType bench_hint = GetBenchmarkTypeHint();
        if (verbose) {
            const char* bnames[] = {"GENERIC","PR","BFS","CC","SSSP","BC","TC","PR_SPMV","CC_SV"};
            printf("AdaptiveOrder: Benchmark hint = %s (%d)\n", 
                   bench_hint < 9 ? bnames[bench_hint] : "?", bench_hint);
        }
        PerceptronSelection small_sel =
            SelectBestReorderingForCommunityWithModelCriterion(
            comm_feat, global_modularity, global_degree_variance, global_hub_concentration,
            global_avg_degree, static_cast<size_t>(num_nodes), num_edges,
            selection_model, selection_criterion,
            bench_hint, detected_graph_type);
        
        // Complexity guard: GOrder is O(n*m*w) and CORDER is O(n*m) — prohibitively
        // slow for large merged groups. Fall back to fast O(n+m) alternatives when
        // the merged group exceeds a node threshold.
        constexpr size_t EXPENSIVE_ALGO_MAX_NODES = 20000;
        if (small_community_nodes.size() > EXPENSIVE_ALGO_MAX_NODES) {
            if (small_sel.algo == GOrder || small_sel.algo == COrder) {
                // Re-select excluding expensive algorithms: use HubSort family or DBG
                // which are O(n log n) and produce reasonable locality
                if (deg_features.hub_concentration > 0.5 && deg_features.degree_variance > 1.5) {
                    small_sel = ResolveVariantSelection("HUBCLUSTERDBG", small_sel.score);
                } else if (deg_features.hub_concentration > 0.3) {
                    small_sel = ResolveVariantSelection("HUBSORT", small_sel.score);
                } else {
                    small_sel = ResolveVariantSelection("DBG", small_sel.score);
                }
                if (verbose) {
                    printf("  -> Complexity guard: large group (%zu > %zu nodes), using %s instead\n",
                           small_community_nodes.size(), EXPENSIVE_ALGO_MAX_NODES,
                           small_sel.variant_name.c_str());
                }
            }
        }
        
        if (verbose) {
            printf("AdaptiveOrder: Grouped %zu small communities (%zu nodes, %zu edges) -> %s\n",
                   non_empty_communities - top_communities.size(),
                   small_community_nodes.size(), merged_feat.num_edges,
                   small_sel.variant_name.c_str());
        }
        
        if (small_sel.algo == ORIGINAL || small_community_nodes.size() < 100) {
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
        } else if (small_sel.variant_name == "DON_LITE") {
            // DON-Lite neural ordering override.
            // Build local subgraph and apply MLP-based reordering.
            std::unordered_map<NodeID_, NodeID_> g2l;
            std::vector<NodeID_> l2g(small_community_nodes.size());
            for (size_t i = 0; i < small_community_nodes.size(); ++i) {
                g2l[small_community_nodes[i]] = static_cast<NodeID_>(i);
                l2g[i] = small_community_nodes[i];
            }
            std::vector<std::pair<NodeID_, DestID_>> sub_edges;
            for (NodeID_ node : small_community_nodes) {
                NodeID_ ls = g2l[node];
                for (DestID_ nb : g.out_neigh(node)) {
                    if (small_node_set.count(static_cast<NodeID_>(nb)))
                        sub_edges.push_back({ls, static_cast<DestID_>(g2l[static_cast<NodeID_>(nb)])});
                }
            }
            if (!sub_edges.empty()) {
                auto sub_g = MakeLocalGraphFromELStandalone<NodeID_, DestID_, invert>(sub_edges, false);
                pvector<NodeID_> sub_ids(small_community_nodes.size(), -1);
                GenerateDonLiteMapping<NodeID_, DestID_, invert>(sub_g, sub_ids, useOutdeg);
                // Fix: Validate sub_ids before using as index — unmapped entries
                // are still -1 (UINT32_MAX for unsigned), which would OOB.
                std::vector<NodeID_> reordered(small_community_nodes.size());
                std::vector<bool> placed(small_community_nodes.size(), false);
                for (size_t i = 0; i < small_community_nodes.size(); ++i) {
                    NodeID_ sid = sub_ids[i];
                    if (sid < small_community_nodes.size()) {
                        reordered[sid] = l2g[i];
                        placed[sid] = true;
                    }
                }
                // Append any unplaced nodes (unmapped by DON-Lite)
                size_t fill = 0;
                for (size_t i = 0; i < small_community_nodes.size(); ++i) {
                    if (!placed[i]) {
                        while (fill < small_community_nodes.size() && placed[fill]) ++fill;
                        if (fill < small_community_nodes.size()) {
                            reordered[fill] = l2g[i];
                            placed[fill] = true;
                        }
                    }
                }
                for (NodeID_ node : reordered)
                    new_ids[node] = current_id++;
            } else {
                for (NodeID_ node : small_community_nodes)
                    new_ids[node] = current_id++;
            }
        } else {
            ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                g, small_community_nodes, small_node_set, small_sel, useOutdeg, new_ids, current_id);
        }
        t_small.Stop();
        t_small_total = t_small.Seconds();
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
        Timer t_cf;
        t_cf.Start();
        auto feat = ComputeCommunityFeaturesStandalone<NodeID_, DestID_, invert>(
            comm_nodes, g, comm_node_set);
        t_cf.Stop();
        t_comm_features_total += t_cf.Seconds();
        
        // Select algorithm for this community
        Timer t_cs;
        t_cs.Start();
        PerceptronSelection selected =
            SelectBestReorderingForCommunityWithModelCriterion(
            feat, global_modularity, global_degree_variance, global_hub_concentration,
            global_avg_degree, static_cast<size_t>(num_nodes), num_edges,
            selection_model, selection_criterion,
            GetBenchmarkTypeHint(), detected_graph_type);
        t_cs.Stop();
        t_comm_scoring_total += t_cs.Seconds();
        
        // Per-community complexity guard: GOrder O(n*m*w) is expensive even for
        // mid-size communities when there are hundreds of them. Also, GOrder can
        // produce invalid permutations on some subgraph topologies.
        // Only block for communities above EXPENSIVE_ALGO_MAX_NODES — small
        // communities where GOrder is genuinely optimal should not be overridden.
        constexpr size_t EXPENSIVE_ALGO_MAX_NODES = 20000;
        if ((selected.algo == GOrder || selected.algo == COrder) &&
            comm_nodes.size() > EXPENSIVE_ALGO_MAX_NODES) {
            if (feat.hub_concentration > 0.5 && feat.degree_variance > 1.5) {
                selected = ResolveVariantSelection("HUBCLUSTERDBG", selected.score);
            } else if (feat.hub_concentration > 0.3) {
                selected = ResolveVariantSelection("HUBSORT", selected.score);
            } else {
                selected = ResolveVariantSelection("DBG", selected.score);
            }
        }
        
        if (verbose && comm_nodes.size() >= MIN_FEATURES_SAMPLE) {
            printf("  Community %zu: %zu nodes, %zu edges -> %s\n",
                   comm_id, comm_nodes.size(), feat.num_edges, 
                   selected.variant_name.c_str());
        }
        
        // Apply algorithm
        Timer t_cr;
        t_cr.Start();
        if (selected.variant_name == "DON_LITE") {
            // DON-Lite neural ordering override for a large community.
            std::unordered_map<NodeID_, NodeID_> g2l;
            std::vector<NodeID_> l2g(comm_nodes.size());
            for (size_t i = 0; i < comm_nodes.size(); ++i) {
                g2l[comm_nodes[i]] = static_cast<NodeID_>(i);
                l2g[i] = comm_nodes[i];
            }
            std::vector<std::pair<NodeID_, DestID_>> sub_edges;
            for (NodeID_ node : comm_nodes) {
                NodeID_ ls = g2l[node];
                for (DestID_ nb : g.out_neigh(node)) {
                    if (comm_node_set.count(static_cast<NodeID_>(nb)))
                        sub_edges.push_back({ls, static_cast<DestID_>(g2l[static_cast<NodeID_>(nb)])});
                }
            }
            if (!sub_edges.empty()) {
                auto sub_g = MakeLocalGraphFromELStandalone<NodeID_, DestID_, invert>(sub_edges, false);
                pvector<NodeID_> sub_ids(comm_nodes.size(), -1);
                GenerateDonLiteMapping<NodeID_, DestID_, invert>(sub_g, sub_ids, useOutdeg);
                // Fix: Validate sub_ids before using as index — unmapped entries
                // are still -1 (UINT32_MAX for unsigned), which would OOB.
                std::vector<NodeID_> reordered(comm_nodes.size());
                std::vector<bool> placed(comm_nodes.size(), false);
                for (size_t i = 0; i < comm_nodes.size(); ++i) {
                    NodeID_ sid = sub_ids[i];
                    if (sid < comm_nodes.size()) {
                        reordered[sid] = l2g[i];
                        placed[sid] = true;
                    }
                }
                // Append any unplaced nodes (unmapped by DON-Lite)
                size_t fill = 0;
                for (size_t i = 0; i < comm_nodes.size(); ++i) {
                    if (!placed[i]) {
                        while (fill < comm_nodes.size() && placed[fill]) ++fill;
                        if (fill < comm_nodes.size()) {
                            reordered[fill] = l2g[i];
                            placed[fill] = true;
                        }
                    }
                }
                for (NodeID_ node : reordered)
                    new_ids[node] = current_id++;
            } else {
                for (NodeID_ node : comm_nodes)
                    new_ids[node] = current_id++;
            }
        } else {
            ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
                g, comm_nodes, comm_node_set, selected, useOutdeg, new_ids, current_id);
        }
        t_cr.Stop();
        t_comm_reorder_total += t_cr.Seconds();
    }
    
    tm.Stop();
    if (!verbose || depth == 0) {
        PrintTime("Adaptive Map Time", tm.Seconds());
    }
    
    // Stage breakdown (always print at depth 0 when verbose)
    if (depth == 0 && verbose) {
        printf("\n=== AdaptiveOrder Stage Breakdown ===\n");
        PrintTime("  Leiden Partitioning", t_leiden.Seconds());
        PrintTime("  Global Features", t_features.Seconds());
        PrintTime("  Small Communities", t_small_total);
        PrintTime("  Comm Features (sum)", t_comm_features_total);
        PrintTime("  Comm Scoring (sum)", t_comm_scoring_total);
        PrintTime("  Comm Reorder (sum)", t_comm_reorder_total);
        double accounted = t_leiden.Seconds() + t_features.Seconds() + t_small_total
                         + t_comm_features_total + t_comm_scoring_total + t_comm_reorder_total;
        PrintTime("  Overhead (unaccounted)", tm.Seconds() - accounted);
        PrintTime("  Total", tm.Seconds());
        printf("Large communities: %zu, Small-group nodes: %zu\n",
               top_communities.size(), small_community_nodes.size());
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
    
    // Default: full-graph mode. The perceptron selects the best algorithm for
    // the whole graph based on graph features and benchmark type. Per-community
    // reordering (recursive mode) was found to degrade performance because:
    //   1. Leiden decomposition disrupts original memory layout
    //   2. Community-level features differ from training data (whole-graph)
    //   3. Cross-community edge patterns are not captured
    GenerateAdaptiveMappingFullGraphStandalone<NodeID_, DestID_, WeightT_, invert>(
        g, new_ids, useOutdeg, reordering_options);
}

#endif // REORDER_ADAPTIVE_H_
