#include <cstdint>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "graphbrew/analysis/adaptive_source_policy.h"
#include "graphbrew/reorder/reorder_graphbrew.h"

namespace
{

struct MockGraph
{
    std::vector<std::vector<int>> adjacency;
    std::vector<int> original_ids;

    int64_t num_nodes() const
    {
        return static_cast<int64_t>(adjacency.size());
    }

    int64_t num_edges_directed() const
    {
        int64_t edges = 0;
        for (const auto& neighbors : adjacency)
            edges += static_cast<int64_t>(neighbors.size());
        return edges;
    }

    bool directed() const
    {
        return false;
    }

    int64_t out_degree(int64_t node) const
    {
        return static_cast<int64_t>(adjacency[node].size());
    }

    const std::vector<int>& out_neigh(int64_t node) const
    {
        return adjacency[node];
    }

    const int* get_org_ids() const
    {
        return original_ids.empty() ? nullptr : original_ids.data();
    }

    int get_org_id(int64_t internal) const
    {
        return original_ids[internal];
    }

    int get_internal_id(int64_t original) const
    {
        for (size_t internal = 0; internal < original_ids.size(); ++internal)
            if (original_ids[internal] == original)
                return static_cast<int>(internal);
        throw std::out_of_range("mock original ID is not mapped");
    }
};

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

Builder MakeBuilder()
{
    static char program[] = "test_graphbrew_config";
    static char *argv[] = {program, nullptr};
    static CLBase cli(1, argv, program);
    return Builder(cli);
}

void TestPresetTailParsing()
{
    Builder builder = MakeBuilder();
    auto config = builder.ParseGraphBrewConfig(
        {
            "leiden",
            "compose",
            "sg_super_rabbit",
            "comm_identity",
            "intra_gorder",
            "gw8",
        },
        0.75);

    Require(
        config.algorithm == graphbrew::GraphBrewAlgorithm::LEIDEN,
        "leiden preset changed the community-detection algorithm");
    Require(
        config.aggregation == graphbrew::AggregationStrategy::GVE_CSR,
        "leiden preset lost GVE aggregation");
    Require(
        config.mComputation == graphbrew::MComputation::TOTAL_EDGES,
        "leiden preset lost total-edge M semantics");
    Require(
        config.ordering == graphbrew::OrderingStrategy::COMPOSE,
        "named preset tail did not select COMPOSE");
    Require(
        config.superGraphOrder == graphbrew::SuperGraphOrder::SuperRabbit,
        "named preset tail lost the super-graph axis");
    Require(
        config.communityOrder == graphbrew::CommunityOrder::Identity,
        "named preset tail lost the community axis");
    Require(
        config.intraCommunityOrder == graphbrew::IntraCommunityOrder::Gorder,
        "named preset tail lost the intra-community axis");
    Require(
        config.gorderWindow == 8,
        "Leiden-Gorder8 did not retain gw8");
}

void TestBudgetedAdaptiveRule()
{
    Require(
        GetSelectionModel("budgeted-rule")
            == SELECTION_MODEL_BUDGETED_RULE,
        "budgeted-rule selection model parsing changed");
    bool missing_reuse_rejected = false;
    try
    {
        adaptive::ParseDeployableSelectionPolicy(
            {"", "", "", "budgeted-rule", "best-endtoend"});
    }
    catch (const std::invalid_argument&)
    {
        missing_reuse_rejected = true;
    }
    Require(
        missing_reuse_rejected,
        "budgeted-rule accepted an implicit reuse count");

    CommunityFeatures features;
    features.num_nodes = 10000;
    features.avg_degree = 60.0;
    features.degree_variance = 1.5;
    features.hub_concentration = 0.4;
    Require(
        adaptive::ShouldUseBudgetedLeidenGorder(
            features, BENCH_PR, 1.0),
        "budgeted rule rejected an eligible PR graph");
    Require(
        !adaptive::ShouldUseBudgetedLeidenGorder(
            features, BENCH_BFS, 1.0),
        "budgeted rule accepted a non-PR kernel");
    Require(
        !adaptive::ShouldUseBudgetedLeidenGorder(
            features, BENCH_PR, 20.0),
        "budgeted rule accepted a high-reuse context");

    auto config = graphbrew::parseGraphBrewCliConfig({
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
    }, 0.75);
    Require(
        config.aggregation == graphbrew::AggregationStrategy::GVE_CSR
            && config.mComputation
                == graphbrew::MComputation::TOTAL_EDGES
            && config.refinementDepth == 0
            && config.resolution == 0.75,
        "budgeted rule diverged from the public leiden preset");
    Require(
        config.ordering == graphbrew::OrderingStrategy::COMPOSE,
        "budgeted rule lost COMPOSE ordering");
    Require(
        config.maxIterations == 1 && config.maxPasses == 1,
        "budgeted rule iteration budget changed");
    Require(
        config.gorderWindow == 32
            && config.gorderFallback == 500,
        "budgeted rule Gorder budget changed");
    Require(
        !config.deterministicCommunityDetection,
        "budgeted rule community policy changed");
    Require(
        graphbrew::makeGraphBrewRealizedConfig(config).scheduleSensitive,
        "parallel budgeted rule was not marked schedule-sensitive");
}

void TestRabbitComposeParsing()
{
    Builder builder = MakeBuilder();
    auto config = builder.ParseGraphBrewConfig(
        {
            "rabbit",
            "compose",
            "sg_none",
            "comm_identity",
            "intra_hubsort",
        },
        0.75);

    Require(
        config.algorithm == graphbrew::GraphBrewAlgorithm::RABBIT_ORDER,
        "rabbit preset did not select Rabbit community detection");
    Require(
        config.ordering == graphbrew::OrderingStrategy::COMPOSE,
        "rabbit preset tail did not select COMPOSE");
    Require(
        config.superGraphOrder == graphbrew::SuperGraphOrder::None,
        "rabbit control did not retain sg_none");
    Require(
        config.communityOrder == graphbrew::CommunityOrder::Identity,
        "rabbit control did not retain comm_identity");
    Require(
        config.intraCommunityOrder == graphbrew::IntraCommunityOrder::HubSort,
        "rabbit control did not retain intra_hubsort");

    bool rejected = false;
    try
    {
        (void)builder.ParseGraphBrewConfig(
            {
                "rabbit",
                "compose",
                "sg_super_rabbit",
                "intra_hubsort",
            },
            0.75);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(
        rejected,
        "COMPOSE accepted a super-graph axis without an explicit block sort");
}

void TestNamedDepthAndStrictTokens()
{
    Builder builder = MakeBuilder();
    auto config = builder.ParseGraphBrewConfig(
        {"leiden", "depth1"}, 0.75);
    Require(config.recursiveDepth == 1, "depth1 did not set recursion depth");

    bool rejected = false;
    try
    {
        (void)builder.ParseGraphBrewConfig(
            {"leiden", "definitely_unknown"}, 0.75);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "unknown GraphBrew token was silently accepted");

    for (const std::string &token : {
             "depth1junk",
             "final5junk",
             "gw8junk",
             "hubx0.5junk",
             "dynamicjunk",
             "nan",
         })
    {
        rejected = false;
        try
        {
            (void)graphbrew::parseGraphBrewConfig({token}, true);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        Require(rejected, "malformed GraphBrew token was accepted");
    }

    auto dynamic = builder.ParseGraphBrewConfig(
        {"leiden", "", "dynamic_2.0"}, 0.75);
    Require(dynamic.useDynamicResolution, "dynamic_2.0 did not enable dynamic mode");
    Require(dynamic.resolution == 2.0, "dynamic_2.0 lost its initial resolution");

    rejected = false;
    try
    {
        (void)builder.ParseGraphBrewConfig({"leiden", "8.0"}, 0.75);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "malformed positional final algorithm was accepted");
}

void TestHubSortAliasesAreUnambiguous()
{
    auto legacy = graphbrew::parseGraphBrewConfig({"hubsort"}, true);
    Require(legacy.useHubSort, "bare hubsort no longer selects legacy hub packing");
    Require(
        legacy.intraCommunityOrder ==
            graphbrew::IntraCommunityOrder::BFSFromHub,
        "bare hubsort unexpectedly changed the COMPOSE intra axis");

    auto compose = graphbrew::parseGraphBrewConfig(
        {"compose", "intra_hubsort"}, true);
    Require(
        compose.intraCommunityOrder ==
            graphbrew::IntraCommunityOrder::HubSort,
        "intra_hubsort did not select the COMPOSE primitive");
}

void TestCommunityZeroCanBeSelected()
{
    graphbrew::CommunityScanner<uint32_t, double> scanner(3);
    scanner.add(2, 5.0);
    scanner.add(0, 5.0);
    std::vector<double> vertex_weight{1.0, 1.0, 1.0};
    std::vector<double> community_weight{1.0, 1.0, 1.0};

    auto [best, delta] = graphbrew::chooseCommunityGreedy<uint32_t, double>(
        1, 1, scanner, vertex_weight, community_weight, 10.0, 1.0);
    Require(delta > 0.0, "test setup did not produce a positive move");
    Require(best == 0, "community zero was treated as a no-move sentinel");
}

void TestAdaptiveDeployablePolicy()
{
    auto defaults = adaptive::ParseDeployableSelectionPolicy({});
    Require(
        defaults.model == SELECTION_MODEL_PERCEPTRON,
        "AdaptiveOrder default model changed");
    Require(
        defaults.criterion == CRITERION_FASTEST_EXECUTION,
        "AdaptiveOrder default criterion changed");

    auto explicit_policy = adaptive::ParseDeployableSelectionPolicy(
        {"", "", "", "perceptron", "best-endtoend"});
    Require(
        explicit_policy.model == SELECTION_MODEL_PERCEPTRON,
        "AdaptiveOrder did not parse an independent model");
    Require(
        explicit_policy.criterion == CRITERION_BEST_ENDTOEND,
        "AdaptiveOrder did not parse an independent criterion");

    auto combined = adaptive::ParseDeployableSelectionPolicy(
        {"", "", "", "perceptron:best-amortization"});
    Require(
        combined.model == SELECTION_MODEL_PERCEPTRON
            && combined.criterion == CRITERION_BEST_AMORTIZATION,
        "AdaptiveOrder legacy combined model/criterion parsing changed");

    for (const std::vector<std::string>& options : {
             std::vector<std::string>{"", "", "", "knn", "execution"},
             std::vector<std::string>{"", "", "", "decision-tree", "execution"},
             std::vector<std::string>{"", "", "", "perceptron", "twitter7"},
             std::vector<std::string>{"", "", "", "unknown-model"},
             std::vector<std::string>{"garbage"},
             std::vector<std::string>{"", "", "", "perceptron", "execution",
                                      "twitter7"},
         })
    {
        bool rejected = false;
        try
        {
            (void)adaptive::ParseDeployableSelectionPolicy(options);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        Require(
            rejected,
            "AdaptiveOrder accepted an identity/oracle/unknown CLI path");
    }

    CommunityFeatures features;
    bool rejected = false;
    try
    {
        (void)SelectReorderingWithModelCriterion(
            features, 0.0, 0.0, 0.0, 1000, 5000,
            SELECTION_MODEL_KNN_DATABASE,
            CRITERION_FASTEST_EXECUTION);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    Require(rejected, "Deployable selection accepted the kNN database");

    rejected = false;
    try
    {
        (void)SelectReorderingWithModelCriterion(
            features, 0.0, 0.0, 0.0, 1000, 5000,
            SELECTION_MODEL_PERCEPTRON,
            CRITERION_FASTEST_EXECUTION,
            "known-graph");
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    Require(rejected, "Deployable selection consumed a graph identity");

    Require(
        DEPLOYABLE_ADAPTIVE_ARM_COUNT == 5,
        "Deployable adaptive portfolio size changed");
    Require(
        NormalizeDeployableAdaptiveArm("RABBITORDER_csr") == "8:csr",
        "Adaptive portfolio alias normalization changed");
    auto compose = ResolveDeployableAdaptiveArm(
        "12:rabbit:compose:sg_super_rabbit:"
        "comm_identity:intra_hubsort");
    Require(
        compose.canonical_spec
            == "12:rabbit:compose:sg_super_rabbit:"
               "comm_identity:intra_hubsort",
        "Adaptive compose arm lost its exact canonical spec");
    auto compose_config = graphbrew::parseGraphBrewConfig(
        compose.options, true);
    Require(
        compose_config.algorithm
                == graphbrew::GraphBrewAlgorithm::RABBIT_ORDER
            && compose_config.ordering
                == graphbrew::OrderingStrategy::COMPOSE
            && compose_config.superGraphOrder
                == graphbrew::SuperGraphOrder::SuperRabbit
            && compose_config.communityOrder
                == graphbrew::CommunityOrder::Identity
            && compose_config.intraCommunityOrder
                == graphbrew::IntraCommunityOrder::HubSort,
        "Adaptive compose arm does not dispatch its exact recipe");

    rejected = false;
    try
    {
        (void)EnforceDeployableAdaptivePortfolio(
            ResolveVariantSelection("RABBIT"));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Require(
        rejected,
        "Adaptive family label was not rejected before dispatch");
}

void TestTier0FeatureSchema()
{
    Require(TIER0_FEATURE_COUNT == 10, "Tier-0 feature count changed");
    Require(
        std::string(TIER0_FEATURE_NAMES[0]) == "log10_nodes"
            && std::string(TIER0_FEATURE_NAMES[9]) == "reuse_bucket",
        "Tier-0 feature name/order contract changed");

    CommunityFeatures features;
    features.num_nodes = 99;
    features.num_edges = 999;
    features.avg_degree = 4.0;
    features.degree_variance = 1.5;
    features.hub_concentration = 0.25;
    features.avg_reuse_distance = 0.1;
    features.window_neighbor_overlap = 0.2;

    Tier0FeatureContext context;
    context.property_wsr_llc = 3.0;
    context.kernel_class = BENCH_BFS;
    context.reuse_count = 20;
    auto values = ExtractTier0Features(features, context);
    const double expected[] = {
        2.0, 3.0, 4.0, 1.5, 0.25, 0.1, 0.2, 3.0, 2.0, 3.0,
    };
    for (size_t i = 0; i < TIER0_FEATURE_COUNT; ++i)
    {
        Require(
            std::abs(values[i] - expected[i]) < 1e-12,
            "Tier-0 C++ feature transform changed");
    }
    PerceptronWeights weights;
    weights.bias = 1.0;
    for (size_t i = 0; i < TIER0_FEATURE_COUNT; ++i)
        weights.tier0_weights[i] = static_cast<double>(i + 1);
    features.working_set_ratio = context.property_wsr_llc;
    features.kernel_class = static_cast<double>(context.kernel_class);
    features.reuse_count = context.reuse_count;
    double expected_score = 1.0;
    for (size_t i = 0; i < TIER0_FEATURE_COUNT; ++i)
        expected_score += (i + 1) * expected[i];
    Require(
        std::abs(weights.scoreTier0(features) - expected_score) < 1e-12,
        "Tier-0 C++ perceptron scoring changed");

    bool rejected = false;
    try
    {
        context.property_wsr_llc = -1.0;
        (void)ExtractTier0Features(features, context);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    Require(rejected, "Tier-0 accepted a missing kernel working set");

    Require(
        ModeledPropertyBytes(BENCH_PR, 100, 500) == 800,
        "PR property working-set formula changed");
    Require(
        ModeledPropertyBytes(BENCH_BFS, 100, 500) == 826,
        "BFS property working-set formula changed");
    Require(
        ModeledPropertyBytes(BENCH_CC, 100, 500) == 413,
        "CC property working-set formula changed");
    Require(
        ModeledPropertyBytes(BENCH_SSSP, 100, 500) == 2400,
        "SSSP property working-set formula changed");
    Require(
        ModeledPropertyBytes(BENCH_BC, 100, 500) == 2063,
        "BC property working-set formula changed");
}

void TestAdaptiveArtifactAndCriterionContract()
{
    nlohmann::json entry = {{"bias", 0.0}};
    for (const char* feature : TIER0_FEATURE_NAMES)
        entry[std::string("w_t0_") + feature] = 0.0;
    entry["w_reorder_time"] = 0.0;
    entry["_metadata"] = {
        {"avg_speedup", 1.0},
        {"avg_reorder_time", 0.0},
    };

    nlohmann::json artifact = {
        {"_schema", "adaptive-tier0/v1"},
        {"weights", nlohmann::json::object()},
    };
    for (const char* arm : DEPLOYABLE_ADAPTIVE_ARM_SPECS)
        artifact["weights"][arm] = entry;

    std::map<std::string, PerceptronWeights> parsed;
    Require(
        ParseWeightsFromJSON(artifact.dump(), parsed)
            && parsed.size() == DEPLOYABLE_ADAPTIVE_ARM_COUNT,
        "strict adaptive artifact did not parse");

    artifact["_schema"] = "adaptive-legacy/v0";
    artifact["_note"] = "adaptive-tier0/v1";
    bool rejected = false;
    try
    {
        (void)ParseWeightsFromJSON(artifact.dump(), parsed);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Require(rejected, "adaptive schema accepted a decoy note");

    artifact["_schema"] = "adaptive-tier0/v1";
    artifact["weights"]["ORIGINAL"] = entry;
    artifact["weights"]["ORIGINAL"]["bias"] = 1.0;
    rejected = false;
    try
    {
        (void)ParseWeightsFromJSON(artifact.dump(), parsed);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Require(rejected, "adaptive artifact accepted conflicting aliases");

    std::map<std::string, PerceptronWeights> weights;
    for (const char* arm : DEPLOYABLE_ADAPTIVE_ARM_SPECS)
    {
        PerceptronWeights weight;
        weight.has_reorder_weight = true;
        weight.has_cost_metadata = true;
        weight.avg_speedup = 1.0;
        weight.avg_reorder_time = 0.0;
        weights.emplace(arm, weight);
    }
    CommunityFeatures features;
    Require(
        SelectBestEndToEndFromWeights(features, weights).canonical_spec
            == "0",
        "best-endtoend tie did not prefer portfolio order");
    Require(
        SelectBestAmortizationFromWeights(weights).canonical_spec
            == "0",
        "all-infeasible amortization did not select ORIGINAL");

    weights["5"].avg_speedup = 1.1;
    weights["5"].avg_reorder_time = 0.95;
    weights["8:csr"].avg_speedup = 10.0;
    weights["8:csr"].avg_reorder_time = 9.5;
    Require(
        std::abs(weights["5"].iterationsToAmortize() - 10.45) < 1e-12,
        "amortization formula changed");
    Require(
        SelectBestAmortizationFromWeights(weights).canonical_spec
            == "5",
        "amortization selected the wrong counterexample arm");

    weights["8:csr"].avg_speedup = 1.1;
    weights["8:csr"].avg_reorder_time = 0.95;
    Require(
        SelectBestAmortizationFromWeights(weights).canonical_spec
            == "5",
        "amortization tie did not follow portfolio order");
}

void TestAdaptiveModelIndicesFailClosed()
{
    CommunityFeatures features;
    ModelTree tree;
    tree.nodes.resize(2);
    tree.nodes[0].feature_idx = MODEL_TREE_N_FEATURES;
    tree.nodes[0].left = 1;
    tree.nodes[0].right = 1;
    tree.nodes[1].feature_idx = -1;
    tree.nodes[1].leaf_class = "ORIGINAL";

    bool rejected = false;
    try
    {
        (void)tree.predict(features);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Require(rejected, "Adaptive model accepted an invalid feature index");

    tree.nodes[0].feature_idx = 0;
    tree.nodes[0].right = 99;
    rejected = false;
    try
    {
        (void)tree.predict(features);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    Require(rejected, "Adaptive model accepted an invalid child index");
}

void TestTier0PackingUsesSampledTopDegreeVertices()
{
    MockGraph graph;
    graph.adjacency.resize(40);
    for (int node : {2, 6, 10, 14, 18, 26, 30, 34, 38})
        graph.adjacency[node] = {39};
    graph.adjacency[22] = {16, 17, 18, 19, 20, 21, 23};

    auto features = ComputeTier0SampledGraphFeatures(graph, 10, 0);
    Require(
        std::abs(features.packing_factor - 1.0) < 1e-12,
        "Tier-0 packing did not use the actual sampled top-degree hub");
    Require(
        features.avg_reuse_distance > 0.0
            && features.avg_reuse_distance <= 1.0,
        "Tier-0 normalized edge span is out of range");
}

void TestExplicitSourceListContract()
{
    optind = 1;
    char program[] = "test_sources";
    char file_flag[] = "-f";
    char file_name[] = "tiny.el";
    char source_flag[] = "-r";
    char source_list[] = "2,7,11";
    char* argv[] = {
        program, file_flag, file_name, source_flag, source_list, nullptr,
    };
    CLApp cli(5, argv, program);
    Require(cli.ParseArgs(), "source-list CLI parsing failed");
    Require(
        cli.start_vertices()
            == std::vector<int64_t>({2, 7, 11}),
        "source-list CLI changed source order");
    Require(
        cli.num_trials() == 3,
        "source-list CLI did not derive one trial per source");

    optind = 1;
    char repeated_program[] = "test_repeated_sources";
    char repeat_flag[] = "-R";
    char repeat_count[] = "2";
    char* repeated_argv[] = {
        repeated_program,
        file_flag,
        file_name,
        source_flag,
        source_list,
        repeat_flag,
        repeat_count,
        nullptr,
    };
    CLApp repeated_cli(7, repeated_argv, repeated_program);
    Require(repeated_cli.ParseArgs(), "source-repeat CLI parsing failed");
    Require(
        repeated_cli.num_trials() == 6
            && repeated_cli.source_repeats() == 2,
        "source repeats did not expand every source");

    optind = 1;
    char single_program[] = "test_single_source";
    char trial_flag[] = "-n";
    char trial_count[] = "5";
    char single_source[] = "2";
    char* single_argv[] = {
        single_program,
        file_flag,
        file_name,
        source_flag,
        single_source,
        trial_flag,
        trial_count,
        nullptr,
    };
    CLApp single_cli(7, single_argv, single_program);
    Require(single_cli.ParseArgs(), "single-source CLI parsing failed");
    Require(
        single_cli.num_trials() == 5,
        "single-source repeated-trial semantics changed");

    MockGraph graph;
    graph.adjacency = {
        {1},
        {0, 2},
        {1},
        {0, 1, 2},
    };
    graph.original_ids = {2, 0, 3, 1};
    SourcePicker<MockGraph> picker(
        graph, std::vector<int64_t>{0, 1}, 2);
    Require(
        picker.PickNext() == 1
            && picker.last_original_source() == 0
            && picker.last_source_out_degree() == 2,
        "first explicit source did not resolve before timing");
    Require(
        picker.PickNext() == 3
            && picker.last_original_source() == 1
            && picker.last_source_out_degree() == 3,
        "second explicit source did not preserve original-ID order");
    bool rejected = false;
    try
    {
        (void)picker.PickNext();
    }
    catch (const std::out_of_range&)
    {
        rejected = true;
    }
    Require(rejected, "exhausted explicit source list fell back to random");

    SourcePicker<MockGraph> repeated_picker(
        graph, std::vector<int64_t>{0, 1}, 4, 2);
    std::vector<int> repeated_sources;
    for (int trial = 0; trial < 4; ++trial)
        repeated_sources.push_back(repeated_picker.PickNext());
    Require(
        repeated_sources == std::vector<int>({1, 1, 3, 3}),
        "source repeats are not consecutive within each source");
}

void TestAdaptiveSourcePolicySelection()
{
    MockGraph graph;
    graph.adjacency.resize(16);
    graph.original_ids.resize(16);
    for (int node = 0; node < 16; ++node)
    {
        graph.original_ids[node] = 15 - node;
        const int next = (node + 1) % 16;
        const int previous = (node + 15) % 16;
        graph.adjacency[node] = {previous, next};
        if (node % 2 == 0)
            graph.adjacency[node].push_back((node + 2) % 16);
    }
    std::vector<int> components(16, 0);
    const auto first = graphbrew::analysis::SelectAdaptiveSources(
        graph, components);
    const auto second = graphbrew::analysis::SelectAdaptiveSources(
        graph, components);
    Require(
        first.sources.size() == 8
            && second.sources.size() == first.sources.size(),
        "adaptive source policy did not select eight sources");
    std::vector<int64_t> originals;
    for (size_t index = 0; index < first.sources.size(); ++index)
    {
        const auto& source = first.sources[index];
        const auto& repeated = second.sources[index];
        originals.push_back(source.original);
        Require(
            source.original == repeated.original
                && source.requested_octile == index
                && source.realized_octile == index,
            "adaptive source policy is not deterministic by octile");
        Require(
            source.reachable_vertices == 16
                && source.reachable_fraction == 1.0
                && source.out_degree > 0,
            "adaptive source policy lost reachability metadata");
    }
    std::sort(originals.begin(), originals.end());
    Require(
        std::adjacent_find(originals.begin(), originals.end())
            == originals.end(),
        "adaptive source policy selected duplicate original IDs");

    MockGraph permuted;
    permuted.adjacency.resize(16);
    permuted.original_ids.resize(16);
    auto remap = [](int old_internal) {
        return (old_internal * 5) % 16;
    };
    for (int old_internal = 0; old_internal < 16; ++old_internal)
    {
        const int new_internal = remap(old_internal);
        permuted.original_ids[new_internal] =
            graph.original_ids[old_internal];
        for (int old_neighbor : graph.adjacency[old_internal])
            permuted.adjacency[new_internal].push_back(
                remap(old_neighbor));
    }
    const auto relabeled = graphbrew::analysis::SelectAdaptiveSources(
        permuted, components);
    std::vector<int64_t> relabeled_originals;
    for (const auto& source : relabeled.sources)
        relabeled_originals.push_back(source.original);
    Require(
        relabeled_originals
            == std::vector<int64_t>({
                first.sources[0].original,
                first.sources[1].original,
                first.sources[2].original,
                first.sources[3].original,
                first.sources[4].original,
                first.sources[5].original,
                first.sources[6].original,
                first.sources[7].original,
            }),
        "adaptive source policy changed under internal relabeling");

    MockGraph tied;
    tied.adjacency.resize(16);
    tied.original_ids.resize(16);
    std::vector<int> tied_components(16);
    for (int node = 0; node < 16; ++node)
    {
        tied.original_ids[node] = node;
        const int base = node < 8 ? 0 : 8;
        tied.adjacency[node] = {
            base + (node - base + 7) % 8,
            base + (node - base + 1) % 8,
        };
        tied_components[node] = base;
    }
    const auto tied_selection =
        graphbrew::analysis::SelectAdaptiveSources(
            tied, tied_components);
    Require(
        tied_selection.largest_component_min_original == 0
            && tied_selection.second_largest_component_size == 8,
        "adaptive source policy component tie-break changed");

    MockGraph tied_permuted;
    tied_permuted.adjacency.resize(16);
    tied_permuted.original_ids.resize(16);
    std::vector<int> tied_permuted_components(16);
    for (int old_internal = 0; old_internal < 16; ++old_internal)
    {
        const int new_internal = remap(old_internal);
        tied_permuted.original_ids[new_internal] =
            tied.original_ids[old_internal];
        tied_permuted_components[new_internal] = (
            old_internal < 8 ? remap(0) : remap(8));
        for (int old_neighbor : tied.adjacency[old_internal])
            tied_permuted.adjacency[new_internal].push_back(
                remap(old_neighbor));
    }
    const auto tied_relabeled =
        graphbrew::analysis::SelectAdaptiveSources(
            tied_permuted, tied_permuted_components);
    for (
        size_t index = 0;
        index < graphbrew::analysis::ADAPTIVE_SOURCE_COUNT;
        ++index
    )
    {
        Require(
            tied_selection.sources[index].original
                == tied_relabeled.sources[index].original,
            "adaptive source component tie-break is labeling-dependent");
    }
}

void TestGraphNameNormalization()
{
    Require(
        ExtractGraphNameFromPath(
            "results/graphs/cit-Patents/cit-Patents.sg")
            == "cit-Patents",
        "nested graph-name normalization changed");
    Require(
        ExtractGraphNameFromPath(
            "C:\\graphs\\com-Orkut\\com-Orkut.wsg")
            == "com-Orkut",
        "Windows-style graph-name normalization changed");
    Require(
        ExtractGraphNameFromPath("twitter7.el.gz") == "twitter7",
        "compressed graph-name normalization changed");
    Require(
        ExtractGraphNameFromPath("USA-road-d.USA")
            == "USA-road-d.USA",
        "extension-free graph name changed");
}

} // namespace

int main()
{
    try
    {
        TestPresetTailParsing();
        TestBudgetedAdaptiveRule();
        TestRabbitComposeParsing();
        TestNamedDepthAndStrictTokens();
        TestHubSortAliasesAreUnambiguous();
        TestCommunityZeroCanBeSelected();
        TestAdaptiveDeployablePolicy();
        TestTier0FeatureSchema();
        TestAdaptiveArtifactAndCriterionContract();
        TestAdaptiveModelIndicesFailClosed();
        TestTier0PackingUsesSampledTopDegreeVertices();
        TestExplicitSourceListContract();
        TestAdaptiveSourcePolicySelection();
        TestGraphNameNormalization();
    }
    catch (const std::exception &error)
    {
        std::cerr << "GraphBrew config test failed: "
                  << error.what() << std::endl;
        return 1;
    }
    std::cout << "GraphBrew config tests passed" << std::endl;
    return 0;
}
