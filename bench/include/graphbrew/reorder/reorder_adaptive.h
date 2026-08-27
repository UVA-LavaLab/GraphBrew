/**
 * @file reorder_adaptive.h
 * @brief AdaptiveOrder runtime selection and retained offline-model support.
 *
 * Deployable deterministic rules extract lightweight graph context and choose
 * one ordering without runtime training, graph identity, benchmark-row
 * lookups, or candidate trials.
 *
 * Validated CLI:
 *
 *   -o 14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>
 *   -o 14:_:_:_:native-midreuse-rule:best-endtoend:40
 *
 * Reuse must be explicit. The low-reuse rule accepts 1 or 2, while the native
 * mid-reuse rule accepts exactly 40. Decisions are deterministic; a selected
 * GraphBrew composition can still be schedule-sensitive.
 *
 * Perceptron, decision-tree, hybrid, kNN, and oracle code remains for offline
 * experiments and compatibility. A legacy recursive per-community helper is
 * also retained, but it is not part of the validated full-graph rule.
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
constexpr size_t NATIVE_MIDREUSE_MIN_NODES = 1ULL << 17;
constexpr double NATIVE_MIDREUSE_COUNT = 40.0;

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
        || model == SELECTION_MODEL_ALLKERNEL_LOWREUSE_RULE
        || model == SELECTION_MODEL_NATIVE_MIDREUSE_RULE
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
        (
            policy.model == SELECTION_MODEL_BUDGETED_RULE
            || policy.model
                == SELECTION_MODEL_ALLKERNEL_LOWREUSE_RULE
            || policy.model
                == SELECTION_MODEL_NATIVE_MIDREUSE_RULE
        )
        && !policy.reuse_count_explicit
    ) {
        throw std::invalid_argument(
            SelectionModelToString(policy.model)
            + " requires an explicit reuse count");
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

inline bool ShouldUseAllKernelLowReuseRule(
    const CommunityFeatures& features,
    BenchmarkType benchmark,
    double reuse_count
) {
    const bool supported_kernel = (
        benchmark == BENCH_PR
        || benchmark == BENCH_PR_SPMV
        || benchmark == BENCH_BFS
        || benchmark == BENCH_CC
        || benchmark == BENCH_CC_SV
        || benchmark == BENCH_BC
        || benchmark == BENCH_SSSP
    );
    const double wsr = features.working_set_ratio;
    return (
        supported_kernel
        && reuse_count <= 2.0
        && features.num_nodes >= 1000
        && wsr <= 3.2
        && (
            (
                features.degree_variance <= 2.68
                && (
                    features.avg_degree <= 60.0
                    || wsr <= 0.82
                )
            )
            || features.degree_variance > 8.0
        )
    );
}

inline PerceptronSelection SelectAllKernelLowReuseRule(
    const CommunityFeatures& features,
    BenchmarkType benchmark,
    double reuse_count
) {
    if (ShouldUseAllKernelLowReuseRule(
        features, benchmark, reuse_count
    )) {
        PerceptronSelection selected;
        selected.algo = GraphBrewOrder;
        selected.variant_name = (
            "12:leiden:compose:sg_none:comm_size_desc:"
            "intra_gorder:gw8:cd_parallel:sgmb4096:"
            "gordf5000:norefine:2:2"
        );
        selected.canonical_spec = selected.variant_name;
        selected.predicted_spec = selected.variant_name;
        selected.options = {
            "leiden",
            "compose",
            "sg_none",
            "comm_size_desc",
            "intra_gorder",
            "gw8",
            "cd_parallel",
            "sgmb4096",
            "gordf5000",
            "norefine",
            "2",
            "2",
        };
        selected.override_reason =
            "allkernel-lowreuse-match";
        selected.confidence = 1.0;
        return selected;
    }
#ifdef RABBIT_ENABLE
    PerceptronSelection fallback;
    fallback.algo = RabbitOrder;
    fallback.variant_name = "8:boost";
    fallback.canonical_spec = fallback.variant_name;
    fallback.predicted_spec = fallback.variant_name;
    fallback.options = {"boost"};
#else
    auto fallback = ResolveDeployableAdaptiveArm("5");
#endif
    fallback.override_reason =
        "allkernel-lowreuse-fallback";
    fallback.confidence = 1.0;
    return fallback;
}

inline bool IsNativeMidReuseSupportedKernel(BenchmarkType benchmark) {
    return (
        benchmark == BENCH_GENERIC
        || benchmark == BENCH_PR
        || benchmark == BENCH_PR_SPMV
        || benchmark == BENCH_BFS
        || benchmark == BENCH_BC
    );
}

inline PerceptronSelection SelectNativeMidReuseRule(
    const CommunityFeatures& features,
    BenchmarkType benchmark,
    double reuse_count
) {
    if (std::abs(reuse_count - NATIVE_MIDREUSE_COUNT) > 1e-9) {
        throw std::invalid_argument(
            "native-midreuse-rule requires reuse count 40");
    }

    if (
        features.num_nodes < NATIVE_MIDREUSE_MIN_NODES
        || !IsNativeMidReuseSupportedKernel(benchmark)
    ) {
        auto fallback = ResolveDeployableAdaptiveArm("0");
        fallback.override_reason = (
            features.num_nodes < NATIVE_MIDREUSE_MIN_NODES
            ? "native-midreuse-small-graph"
            : "native-midreuse-unsupported-kernel");
        fallback.confidence = 1.0;
        return fallback;
    }

    PerceptronSelection selected;
    selected.algo = HubClusterDBG;
    selected.variant_name = "7";
    selected.canonical_spec = selected.variant_name;
    selected.predicted_spec = selected.variant_name;
    selected.override_reason = "native-midreuse-match";
    selected.confidence = 1.0;
    return selected;
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
            auto config = graphbrew::parseGraphBrewCliConfig(
                selected.options,
                LeidenAutoResolution<NodeID_, DestID_>(g));
            const auto realized =
                graphbrew::makeGraphBrewRealizedConfig(config);
            auto& staged =
                graphbrew::database::GetStagedReorderMeta();
            staged.schedule_sensitive =
                staged.schedule_sensitive
                || realized.scheduleSensitive;
            staged.thread_policy_sensitive =
                staged.thread_policy_sensitive
                || config.algorithm ==
                    graphbrew::GraphBrewAlgorithm::LEIDEN;
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
    const BenchmarkType benchmark = GetBenchmarkTypeHint();
    CommunityFeatures global_feat;
    if (
        selection_policy.model
        == SELECTION_MODEL_NATIVE_MIDREUSE_RULE
    ) {
        global_feat.num_nodes =
            static_cast<size_t>(std::max<int64_t>(0, num_nodes));
        global_feat.num_edges =
            static_cast<size_t>(std::max<int64_t>(0, num_edges));
        global_feat.avg_degree = (
            num_nodes > 0
            ? static_cast<double>(num_edges) / num_nodes
            : 0.0);
    } else {
        global_feat = ::ComputeTier0SampledGraphFeatures(g);
    }
    const uint64_t property_bytes = (
        selection_policy.model
            == SELECTION_MODEL_NATIVE_MIDREUSE_RULE
        ? 0
        : ModeledPropertyBytes(
            benchmark,
            static_cast<uint64_t>(num_nodes),
            static_cast<uint64_t>(num_edges)));
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
    } else if (
        selection_policy.model
        == SELECTION_MODEL_ALLKERNEL_LOWREUSE_RULE
    ) {
        if (
            selection_policy.criterion
            != CRITERION_BEST_ENDTOEND
        ) {
            throw std::invalid_argument(
                "allkernel-lowreuse-rule requires "
                "best-endtoend criterion");
        }
        best = adaptive::SelectAllKernelLowReuseRule(
            global_feat,
            benchmark,
            selection_policy.reuse_count);
    } else if (
        selection_policy.model
        == SELECTION_MODEL_NATIVE_MIDREUSE_RULE
    ) {
        if (
            selection_policy.criterion
            != CRITERION_BEST_ENDTOEND
        ) {
            throw std::invalid_argument(
                "native-midreuse-rule requires "
                "best-endtoend criterion");
        }
        best = adaptive::SelectNativeMidReuseRule(
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
        && selection_policy.model
            != SELECTION_MODEL_ALLKERNEL_LOWREUSE_RULE
        && selection_policy.model
            != SELECTION_MODEL_NATIVE_MIDREUSE_RULE
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

/** Retired recursive entry point retained to fail closed for old callers. */
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
    (void)g;
    (void)new_ids;
    (void)useOutdeg;
    (void)reordering_options;
    (void)depth;
    (void)verbose;
    (void)selection_model;
    (void)selection_criterion;
    throw std::invalid_argument(
        "Recursive AdaptiveOrder is retired; deployable selection is "
        "full-graph over the frozen exact portfolio");
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
    
    // Deployable AdaptiveOrder always selects one full-graph arm.
    GenerateAdaptiveMappingFullGraphStandalone<NodeID_, DestID_, WeightT_, invert>(
        g, new_ids, useOutdeg, reordering_options);
}

#endif // REORDER_ADAPTIVE_H_
