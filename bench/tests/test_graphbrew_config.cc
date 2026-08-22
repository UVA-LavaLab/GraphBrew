#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
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

Graph BuildDirectedGraph(
    const std::vector<std::vector<NodeID>>& outgoing)
{
    const size_t nodeCount = outgoing.size();
    std::vector<std::vector<NodeID>> incoming(nodeCount);
    size_t edgeCount = 0;
    for (size_t source = 0; source < nodeCount; ++source) {
        edgeCount += outgoing[source].size();
        for (NodeID target : outgoing[source])
            incoming[target].push_back(static_cast<NodeID>(source));
    }

    auto buildIndex = [&](const std::vector<std::vector<NodeID>>& rows,
                          NodeID*& neighbors) {
        neighbors = new NodeID[edgeCount];
        auto** index = new NodeID*[nodeCount + 1];
        size_t position = 0;
        for (size_t node = 0; node < nodeCount; ++node) {
            index[node] = neighbors + position;
            for (NodeID neighbor : rows[node])
                neighbors[position++] = neighbor;
        }
        index[nodeCount] = neighbors + position;
        return index;
    };

    NodeID* outNeighbors = nullptr;
    NodeID* inNeighbors = nullptr;
    NodeID** outIndex = buildIndex(outgoing, outNeighbors);
    NodeID** inIndex = buildIndex(incoming, inNeighbors);
    return Graph(
        static_cast<int64_t>(nodeCount),
        outIndex,
        outNeighbors,
        inIndex,
        inNeighbors);
}

Graph BuildSymmetricGraph(
    const std::vector<std::vector<NodeID>>& adjacency)
{
    size_t edgeCount = 0;
    for (const auto& neighbors : adjacency)
        edgeCount += neighbors.size();
    auto* neighbors = new NodeID[edgeCount];
    auto** index = new NodeID*[adjacency.size() + 1];
    size_t position = 0;
    for (size_t node = 0; node < adjacency.size(); ++node) {
        index[node] = neighbors + position;
        for (NodeID neighbor : adjacency[node])
            neighbors[position++] = neighbor;
    }
    index[adjacency.size()] = neighbors + position;
    return Graph(
        static_cast<int64_t>(adjacency.size()),
        index,
        neighbors);
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

void TestAllKernelLowReuseRule()
{
    Require(
        GetSelectionModel("allkernel-lowreuse-rule")
            == SELECTION_MODEL_ALLKERNEL_LOWREUSE_RULE,
        "all-kernel low-reuse model parsing changed");

    bool missing_reuse_rejected = false;
    try
    {
        adaptive::ParseDeployableSelectionPolicy({
            "", "", "", "allkernel-lowreuse-rule",
            "best-endtoend",
        });
    }
    catch (const std::invalid_argument&)
    {
        missing_reuse_rejected = true;
    }
    Require(
        missing_reuse_rejected,
        "all-kernel low-reuse rule accepted implicit reuse");

    CommunityFeatures features;
    features.num_nodes = 10000;
    features.avg_degree = 25.0;
    features.degree_variance = 1.5;
    features.hub_concentration = 0.4;
    features.working_set_ratio = 0.5;
    Require(
        adaptive::ShouldUseAllKernelLowReuseRule(
            features, BENCH_CC, 2.0),
        "all-kernel rule rejected an eligible CC context");
    Require(
        !adaptive::ShouldUseAllKernelLowReuseRule(
            features, BENCH_TC, 2.0),
        "all-kernel rule accepted an unmeasured kernel");
    Require(
        !adaptive::ShouldUseAllKernelLowReuseRule(
            features, BENCH_CC, 3.0),
        "all-kernel rule accepted high reuse");

    auto selected = adaptive::SelectAllKernelLowReuseRule(
        features, BENCH_BFS, 1.0);
    Require(
        selected.canonical_spec.find(
            "sgmb4096:gordf5000:norefine:2:2"
        ) != std::string::npos,
        "all-kernel rule selected the wrong composition");
    auto config = graphbrew::parseGraphBrewCliConfig(
        selected.options, 0.75);
    Require(
        config.superGraphMoveBatch == 4096
            && config.gorderFallback == 5000
            && !config.useRefinement
            && config.maxIterations == 2
            && config.maxPasses == 2,
        "all-kernel selected config changed");

    features.working_set_ratio = 4.0;
    auto fallback = adaptive::SelectAllKernelLowReuseRule(
        features, BENCH_BFS, 1.0);
#ifdef RABBIT_ENABLE
    Require(
        fallback.canonical_spec == "8:boost",
        "all-kernel rule lost Boost fallback");
#else
    Require(
        fallback.canonical_spec == "5",
        "all-kernel rule lost no-Boost fallback");
#endif
}

void TestSuperGraphMoveBatchParsing()
{
    auto config = graphbrew::parseGraphBrewConfig(
        {"sgmb256"}, true);
    Require(
        config.superGraphMoveBatch == 256,
        "super-graph move batch token was ignored");
}

void TestCapacityRunCommunityOrder()
{
    auto config = graphbrew::parseGraphBrewConfig({
        "compose",
        "comm_capacity_runs",
        "capl2k4",
        "capllck16",
        "capv8",
    }, true);
    Require(
        config.communityOrder
                == graphbrew::CommunityOrder::CapacityRuns
            && config.capacityL2Bytes == 4 * 1024
            && config.capacityLLCBytes == 16 * 1024
            && config.capacityPropertyBytesPerVertex == 8,
        "capacity-run parser lost explicit cache geometry");
    Require(
        std::string(graphbrew::graphBrewCommunityOrderName(
            config.communityOrder)) == "capacity-runs",
        "capacity-run community-order name changed");
    const auto geometry = graphbrew::resolveCapacityGeometry(config);
    Require(
        geometry.l2Bytes == 4 * 1024
            && geometry.llcBytes == 16 * 1024
            && geometry.propertyBytesPerVertex == 8,
        "capacity-run geometry resolution changed explicit values");
    bool geometryRejected = false;
    try {
        (void)graphbrew::resolveCapacityGeometry(
            graphbrew::GraphBrewConfig{});
    } catch (const std::invalid_argument&) {
        geometryRejected = true;
    }
    Require(
        geometryRejected,
        "capacity-run execution accepted unpinned geometry");

    auto faithful = graphbrew::parseGraphBrewConfig({
        "compose",
        "intra_gorder_faithful",
        "gw8",
    }, true);
    Require(
        faithful.intraCommunityOrder
                == graphbrew::IntraCommunityOrder::GorderFaithful
            && faithful.gorderWindow == 8
            && std::string(graphbrew::graphBrewIntraOrderName(
                faithful.intraCommunityOrder)) == "gorder-faithful",
        "faithful local Gorder token changed");
    Require(
        graphbrew::parseGraphBrewConfig(
            {"compose", "intra_gorder2"}, true)
                .intraCommunityOrder
            == graphbrew::IntraCommunityOrder::GorderFaithful,
        "faithful local Gorder alias changed");

    std::vector<std::vector<std::pair<uint32_t, uint64_t>>>
        adjacency(6);
    auto connect = [&](uint32_t left, uint32_t right, uint64_t weight) {
        adjacency[left].push_back({right, weight});
        adjacency[right].push_back({left, weight});
    };
    connect(0, 1, 10);
    connect(1, 2, 9);
    connect(2, 3, 8);
    connect(4, 5, 10);
    const auto result =
        graphbrew::buildCapacityRunCommunityOrder<uint32_t>(
            {3, 3, 3, 3, 8, 1},
            adjacency,
            {0, 1, 2, 3, 4, 5},
            6,
            12);
    Require(
        result.order == std::vector<uint32_t>({4, 5, 0, 1, 2, 3}),
        "capacity-run ordering changed deterministic cut-aware traversal");
    Require(
        result.l2RunEnds == std::vector<size_t>({1, 3, 5, 6})
            && result.llcRunEnds == std::vector<size_t>({3, 6}),
        "capacity-run boundaries changed");
    const auto tighter =
        graphbrew::buildCapacityRunCommunityOrder<uint32_t>(
            {3, 3, 3, 3, 8, 1},
            adjacency,
            {0, 1, 2, 3, 4, 5},
            3,
            6);
    Require(
        tighter.order != result.order,
        "capacity-run geometry did not affect community ordering");

    const auto sparse = graphbrew::buildCapacityRunCommunityOrder<uint32_t>(
        {0, 2, 0},
        std::vector<
            std::vector<std::pair<uint32_t, uint64_t>>>(3),
        {2, 0, 1},
        4,
        8);
    Require(
        sparse.order == std::vector<uint32_t>({1, 2, 0})
            && sparse.l2RunEnds == std::vector<size_t>({1})
            && sparse.llcRunEnds == std::vector<size_t>({1}),
        "capacity-run ordering counted empty communities as cache runs");

    const auto disconnected =
        graphbrew::buildCapacityRunCommunityOrder<uint32_t>(
            {2, 2, 2},
            std::vector<
                std::vector<std::pair<uint32_t, uint64_t>>>(3),
            {2, 0, 1},
            4,
            8);
    Require(
        disconnected.order == std::vector<uint32_t>({2, 0, 1})
            && disconnected.l2RunEnds
                == std::vector<size_t>({2, 3})
            && disconnected.llcRunEnds
                == std::vector<size_t>({3}),
        "capacity-run disconnected packing lost base order or capacity");

    std::vector<std::vector<std::pair<uint32_t, uint64_t>>>
        blockedFrontier(4);
    auto connectBlocked = [&](
        uint32_t left,
        uint32_t right,
        uint64_t weight)
    {
        blockedFrontier[left].push_back({right, weight});
        blockedFrontier[right].push_back({left, weight});
    };
    connectBlocked(0, 1, 100);
    connectBlocked(0, 2, 1);
    connectBlocked(3, 0, 50);
    const auto fittedFrontier =
        graphbrew::buildCapacityRunCommunityOrder<uint32_t>(
            {1, 5, 1, 6},
            blockedFrontier,
            {0, 1, 2, 3},
            3,
            100);
    Require(
        fittedFrontier.order
                == std::vector<uint32_t>({3, 0, 2, 1})
            && fittedFrontier.l2RunEnds
                == std::vector<size_t>({1, 3, 4})
            && fittedFrontier.llcRunEnds
                == std::vector<size_t>({4}),
        "capacity-run ordering stopped at an oversized frontier head");

    Graph capacityGraph = BuildSymmetricGraph({
        {1, 2},
        {0, 4},
        {0, 3, 4},
        {2},
        {1, 2, 5},
        {4, 6},
        {5},
        {},
    });
    graphbrew::GraphBrewResult<uint32_t> capacityResult;
    capacityResult.membership = {0, 0, 1, 1, 2, 2, 2, 3};
    const std::vector<uint32_t> capacityDegrees = {
        2, 2, 3, 1, 3, 2, 1, 0};
    graphbrew::GraphBrewConfig capacityConfig;
    capacityConfig.ordering =
        graphbrew::OrderingStrategy::COMPOSE;
    capacityConfig.communityOrder =
        graphbrew::CommunityOrder::CapacityRuns;
    capacityConfig.capacityL2Bytes = 3;
    capacityConfig.capacityLLCBytes = 5;
    capacityConfig.capacityPropertyBytesPerVertex = 1;
    pvector<NodeID> oneThreadMapping(8);
    pvector<NodeID> fourThreadMapping(8);
    graphbrew::GraphBrewRealizedConfig oneThreadRealized;
    graphbrew::GraphBrewRealizedConfig fourThreadRealized;
    #ifdef OPENMP
    const int previousThreadCount = omp_get_max_threads();
    omp_set_num_threads(1);
    #endif
    graphbrew::orderCompose<uint32_t, NodeID, NodeID>(
        oneThreadMapping,
        capacityResult,
        capacityDegrees,
        capacityGraph,
        8,
        capacityConfig,
        &oneThreadRealized);
    #ifdef OPENMP
    omp_set_num_threads(4);
    #endif
    graphbrew::orderCompose<uint32_t, NodeID, NodeID>(
        fourThreadMapping,
        capacityResult,
        capacityDegrees,
        capacityGraph,
        8,
        capacityConfig,
        &fourThreadRealized);
    #ifdef OPENMP
    omp_set_num_threads(previousThreadCount);
    #endif
    auto capacityPermutation = std::vector<NodeID>(
        oneThreadMapping.begin(), oneThreadMapping.end());
    std::sort(capacityPermutation.begin(), capacityPermutation.end());
    Require(
        std::equal(
            oneThreadMapping.begin(),
            oneThreadMapping.end(),
            fourThreadMapping.begin())
            && capacityPermutation
                == std::vector<NodeID>({0, 1, 2, 3, 4, 5, 6, 7})
            && oneThreadRealized.capacityL2Runs == 3
            && oneThreadRealized.capacityLLCRuns == 2
            && fourThreadRealized.capacityL2Runs == 3
            && fourThreadRealized.capacityLLCRuns == 2,
        "capacity-run graph path changed with thread partitioning");

    Graph directedTailGraph = BuildDirectedGraph({
        {1, 2, 3},
        {0},
        {0},
        {},
    });
    bool directedRejected = false;
    try {
        (void)graphbrew::buildCapacityCommunityAdjacency<
            uint32_t, NodeID, NodeID>(
                std::vector<uint32_t>{0, 0, 1, 2},
                directedTailGraph,
                4,
                3);
    } catch (const std::invalid_argument&) {
        directedRejected = true;
    }
    graphbrew::GraphBrewResult<uint32_t> directedCapacityResult;
    directedCapacityResult.membership = {0, 0, 1, 2};
    pvector<NodeID> directedCapacityMapping(4);
    bool composeDirectedRejected = false;
    try {
        graphbrew::orderCompose<uint32_t, NodeID, NodeID>(
            directedCapacityMapping,
            directedCapacityResult,
            std::vector<uint32_t>{3, 1, 1, 0},
            directedTailGraph,
            4,
            capacityConfig);
    } catch (const std::invalid_argument&) {
        composeDirectedRejected = true;
    }
    Require(
        directedRejected && composeDirectedRejected,
        "capacity-run ordering accepted a directed graph");

    std::mt19937 generator(0xC0FFEEu);
    for (size_t trial = 0; trial < 64; ++trial)
    {
        const size_t communityCount = 1 + generator() % 32;
        std::vector<size_t> sizes(communityCount);
        std::vector<uint32_t> baseOrder(communityCount);
        std::iota(baseOrder.begin(), baseOrder.end(), 0);
        std::shuffle(baseOrder.begin(), baseOrder.end(), generator);
        for (size_t& size : sizes) size = generator() % 17;

        std::vector<
            std::vector<std::pair<uint32_t, uint64_t>>>
            randomAdjacency(communityCount);
        for (uint32_t left = 0; left < communityCount; ++left)
        {
            for (
                uint32_t right = left + 1;
                right < communityCount;
                ++right
            )
            {
                if (generator() % 5 != 0) continue;
                const uint64_t weight = 1 + generator() % 31;
                randomAdjacency[left].push_back({right, weight});
                randomAdjacency[right].push_back({left, weight});
            }
        }

        const size_t l2Target = 1 + generator() % 24;
        const size_t llcTarget = l2Target + generator() % 48;
        const auto randomResult =
            graphbrew::buildCapacityRunCommunityOrder<uint32_t>(
                sizes,
                randomAdjacency,
                baseOrder,
                l2Target,
                llcTarget);
        const auto repeated =
            graphbrew::buildCapacityRunCommunityOrder<uint32_t>(
                sizes,
                randomAdjacency,
                baseOrder,
                l2Target,
                llcTarget);
        Require(
            randomResult.order == repeated.order
                && randomResult.l2RunEnds == repeated.l2RunEnds
                && randomResult.llcRunEnds == repeated.llcRunEnds,
            "capacity-run ordering became nondeterministic");

        std::vector<uint32_t> sortedOrder(baseOrder.size());
        std::iota(sortedOrder.begin(), sortedOrder.end(), 0);
        auto emitted = randomResult.order;
        std::sort(emitted.begin(), emitted.end());
        Require(
            emitted == sortedOrder,
            "capacity-run ordering emitted an invalid permutation");

        const size_t activeCount = static_cast<size_t>(std::count_if(
            sizes.begin(), sizes.end(),
            [](size_t size) { return size > 0; }));
        auto checkRuns = [&](const std::vector<size_t>& ends,
                             size_t target) {
            size_t begin = 0;
            for (size_t end : ends)
            {
                Require(
                    begin < end && end <= activeCount,
                    "capacity-run boundary was not strictly increasing");
                size_t vertices = 0;
                for (size_t index = begin; index < end; ++index)
                    vertices += sizes[randomResult.order[index]];
                Require(
                    vertices <= target || end == begin + 1,
                    "capacity-run boundary exceeded its target");
                begin = end;
            }
            Require(
                begin == activeCount,
                "capacity-run boundaries did not cover active communities");
        };
        if (activeCount == 0)
        {
            Require(
                randomResult.l2RunEnds.empty()
                    && randomResult.llcRunEnds.empty(),
                "capacity-run ordering created runs for empty communities");
        }
        else
        {
            checkRuns(randomResult.l2RunEnds, l2Target);
            checkRuns(randomResult.llcRunEnds, llcTarget);
        }
        for (size_t index = activeCount; index < communityCount; ++index)
        {
            Require(
                sizes[randomResult.order[index]] == 0,
                "capacity-run ordering interleaved an empty community");
        }
    }
}

bool SlowFaithfulUniqueOrder(
    const std::vector<std::vector<size_t>>& outNeighbors,
    int window,
    std::vector<uint32_t>& order)
{
    const size_t size = outNeighbors.size();
    std::vector<std::vector<size_t>> inNeighbors(size);
    for (size_t source = 0; source < size; ++source) {
        for (size_t target : outNeighbors[source])
            inNeighbors[target].push_back(source);
    }

    size_t seed = 0;
    size_t maximumInDegree = 0;
    bool seedTie = false;
    for (size_t vertex = 0; vertex < size; ++vertex) {
        if (inNeighbors[vertex].size() > maximumInDegree) {
            maximumInDegree = inNeighbors[vertex].size();
            seed = vertex;
            seedTie = false;
        } else if (
            inNeighbors[vertex].size() == maximumInDegree
        ) {
            seedTie = true;
        }
    }
    if (maximumInDegree == 0 || seedTie) return false;

    const size_t hugeVertex = static_cast<size_t>(
        std::sqrt(static_cast<double>(size)));
    std::vector<char> placed(size, 0);
    placed[seed] = 1;
    order = {static_cast<uint32_t>(seed)};
    while (order.size() < size) {
        uint64_t bestScore = 0;
        size_t best = size;
        bool tie = false;
        for (size_t candidate = 0; candidate < size; ++candidate) {
            if (placed[candidate]) continue;
            uint64_t score = 0;
            const size_t begin =
                order.size() > static_cast<size_t>(window)
                    ? order.size() - static_cast<size_t>(window)
                    : 0;
            for (size_t index = begin; index < order.size(); ++index) {
                const size_t active = order[index];
                if (
                    outNeighbors[active].size() <= hugeVertex
                    && std::binary_search(
                        outNeighbors[active].begin(),
                        outNeighbors[active].end(),
                        candidate)
                ) {
                    ++score;
                }
                for (size_t predecessor : inNeighbors[active]) {
                    if (
                        outNeighbors[predecessor].size()
                        > hugeVertex
                    ) {
                        continue;
                    }
                    if (candidate == predecessor) ++score;
                    if (
                        outNeighbors[predecessor].size() > 1
                        && std::binary_search(
                            outNeighbors[predecessor].begin(),
                            outNeighbors[predecessor].end(),
                            candidate)
                    ) {
                        ++score;
                    }
                }
            }
            if (best == size || score > bestScore) {
                bestScore = score;
                best = candidate;
                tie = false;
            } else if (score == bestScore) {
                tie = true;
            }
        }
        if (best == size || tie) return false;
        placed[best] = 1;
        order.push_back(static_cast<uint32_t>(best));
    }
    return true;
}

void TestFaithfulLocalGorder()
{
    Require(
        !AblationConfig::Get().don_tiebreak,
        "faithful Gorder differential test requires DON disabled");

    const std::vector<std::vector<size_t>> uniqueAdjacency = {
        {1, 6, 8},
        {0, 2, 3, 4, 5},
        {1, 3, 8},
        {1, 2, 7, 8},
        {1, 8},
        {1, 6, 7},
        {0, 5},
        {3, 5},
        {0, 2, 3, 4},
    };
    std::vector<uint32_t> slowOrder;
    std::vector<uint32_t> faithfulOrder;
    std::vector<std::vector<size_t>> uniqueIn(
        uniqueAdjacency.size());
    for (size_t source = 0; source < uniqueAdjacency.size(); ++source) {
        for (size_t target : uniqueAdjacency[source])
            uniqueIn[target].push_back(source);
    }
    Require(
        SlowFaithfulUniqueOrder(uniqueAdjacency, 3, slowOrder),
        "independent faithful Gorder oracle has a score tie");
    graphbrew::faithfulGorderLocalOrder<uint32_t>(
        uniqueAdjacency,
        uniqueIn,
        uniqueAdjacency.size(),
        3,
        faithfulOrder);
    Require(
        faithfulOrder == slowOrder
            && faithfulOrder
                == std::vector<uint32_t>(
                    {1, 8, 6, 0, 5, 7, 3, 2, 4}),
        "faithful local Gorder diverged from independent scores");

    auto compareWithReference = [](
        const std::vector<std::vector<NodeID>>& outgoing,
        int window)
    {
        std::vector<std::vector<size_t>> localOut(outgoing.size());
        std::vector<std::vector<size_t>> localIn(outgoing.size());
        for (size_t source = 0; source < outgoing.size(); ++source) {
            for (NodeID target : outgoing[source]) {
                localOut[source].push_back(
                    static_cast<size_t>(target));
                localIn[target].push_back(source);
            }
        }
        for (auto& neighbors : localOut)
            std::sort(neighbors.begin(), neighbors.end());

        std::vector<uint32_t> localOrder;
        graphbrew::faithfulGorderLocalOrder<uint32_t>(
            localOut,
            localIn,
            outgoing.size(),
            window,
            localOrder);

        Graph graph = BuildDirectedGraph(outgoing);
        std::vector<int> referenceMapping;
        gorder_csr_detail::gorder_greedy_csr(
            graph,
            static_cast<int>(outgoing.size()),
            window,
            referenceMapping);
        std::vector<uint32_t> referenceOrder(outgoing.size());
        for (size_t vertex = 0; vertex < outgoing.size(); ++vertex) {
            referenceOrder[referenceMapping[vertex]] =
                static_cast<uint32_t>(vertex);
        }
        Require(
            localOrder == referenceOrder,
            "faithful local Gorder diverged from exact Gorder");
    };

    const std::vector<std::vector<NodeID>> path = {
        {1},
        {0, 2},
        {1, 3},
        {2},
    };
    compareWithReference(path, 1);
    compareWithReference(path, 2);
    compareWithReference(path, 8);

    const std::vector<std::vector<NodeID>> tinyStar = {
        {1, 2},
        {0},
        {0},
    };
    compareWithReference(tinyStar, 2);
    std::vector<std::vector<size_t>> starOut = {
        {1, 2},
        {0},
        {0},
    };
    const auto starIn = starOut;
    std::vector<uint32_t> starOrder;
    graphbrew::faithfulGorderLocalOrder<uint32_t>(
        starOut, starIn, starOut.size(), 2, starOrder);
    Require(
        !starOrder.empty() && starOrder.front() == 0,
        "faithful local Gorder skipped its three-vertex hub");

    const std::vector<std::vector<NodeID>> directedSibling = {
        {1, 2},
        {3},
        {3},
        {0},
    };
    compareWithReference(directedSibling, 1);
    compareWithReference(directedSibling, 2);
    compareWithReference(directedSibling, 5);

    const std::vector<std::vector<NodeID>> guardBoundary = {
        {1, 2, 3},
        {0, 4, 5, 6},
        {0},
        {0},
        {1},
        {1},
        {1},
        {8},
        {7},
    };
    compareWithReference(guardBoundary, 3);
    compareWithReference({
        {},
        {2},
        {1},
    }, 2);
    compareWithReference({
        {},
        {},
        {},
        {},
    }, 2);

    std::mt19937 differentialGenerator(0x6F726465u);
    for (size_t trial = 0; trial < 256; ++trial) {
        const size_t nodeCount =
            2 + differentialGenerator() % 11;
        std::vector<std::vector<NodeID>> outgoing(nodeCount);
        for (size_t source = 0; source < nodeCount; ++source) {
            for (size_t target = 0; target < nodeCount; ++target) {
                if (differentialGenerator() % 5 == 0) {
                    outgoing[source].push_back(
                        static_cast<NodeID>(target));
                }
            }
        }
        compareWithReference(
            outgoing,
            1 + static_cast<int>(trial % 8));
    }

    std::vector<std::vector<size_t>> zeroOut = {
        {1},
        {0},
        {},
        {},
    };
    const auto zeroIn = zeroOut;
    std::vector<uint32_t> zeroOrder;
    graphbrew::faithfulGorderLocalOrder<uint32_t>(
        zeroOut, zeroIn, zeroOut.size(), 2, zeroOrder);
    Require(
        zeroOrder == std::vector<uint32_t>({0, 2, 3, 1}),
        "faithful local Gorder changed zero-degree placement");

    bool rejected = false;
    try {
        graphbrew::faithfulGorderLocalOrder<uint32_t>(
            zeroOut, zeroIn, zeroOut.size(), 0, zeroOrder);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "faithful local Gorder accepted a nonpositive window");

    Graph clique = BuildDirectedGraph({
        {1, 2, 3},
        {0, 2, 3},
        {0, 1, 3},
        {0, 1, 2},
    });
    std::vector<NodeID> vertices = {0, 1, 2, 3};
    std::vector<NodeID> membership(4, 0);
    std::vector<NodeID> degrees(4, 3);
    std::vector<size_t> vertexToLocal = {0, 1, 2, 3};
    std::vector<NodeID> placedOrder;
    std::vector<std::vector<size_t>> neighborScratch;
    std::vector<NodeID> localIds(4, -1);
    graphbrew::intraGorderGreedy<NodeID, NodeID, NodeID>(
        vertices,
        0,
        membership,
        degrees,
        clique,
        vertexToLocal,
        5,
        placedOrder,
        neighborScratch,
        localIds);
    Require(
        localIds == std::vector<NodeID>({0, 3, 2, 1}),
        "legacy local Gorder mapping changed");

    Graph duplicateGraph = BuildDirectedGraph({
        {1, 1},
        {0},
    });
    graphbrew::GraphBrewResult<uint32_t> directedResult;
    directedResult.membership = {0, 0};
    graphbrew::GraphBrewConfig directedConfig;
    directedConfig.ordering =
        graphbrew::OrderingStrategy::COMPOSE;
    directedConfig.intraCommunityOrder =
        graphbrew::IntraCommunityOrder::GorderFaithful;
    pvector<NodeID> directedMapping;
    rejected = false;
    try {
        graphbrew::orderCompose<uint32_t, NodeID, NodeID>(
            directedMapping,
            directedResult,
            std::vector<uint32_t>{2, 1},
            duplicateGraph,
            2,
            directedConfig);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "faithful local Gorder accepted a directed COMPOSE graph");

    vertices = {0, 1};
    membership.assign(2, 0);
    vertexToLocal = {0, 1};
    localIds.assign(2, -1);
    std::vector<std::vector<size_t>> outScratch;
    std::vector<std::vector<size_t>> inScratch;
    graphbrew::intraGorderFaithful<NodeID, NodeID, NodeID>(
        vertices,
        0,
        membership,
        duplicateGraph,
        vertexToLocal,
        2,
        placedOrder,
        outScratch,
        inScratch,
        localIds);
    Require(
        outScratch[0] == std::vector<size_t>({1})
            && localIds[0] >= 0
            && localIds[1] >= 0
            && localIds[0] != localIds[1],
        "faithful local Gorder did not collapse duplicate edges");

    Graph reuseGraph = BuildDirectedGraph({
        {1},
        {0, 2},
        {1, 3},
        {2, 4},
        {3},
        {6},
        {5},
    });
    membership = {0, 0, 0, 0, 0, 1, 1};
    vertexToLocal = {0, 1, 2, 3, 4, 0, 1};
    localIds.assign(7, -1);
    outScratch.clear();
    inScratch.clear();
    graphbrew::intraGorderFaithful<NodeID, NodeID, NodeID>(
        std::vector<NodeID>{0, 1, 2, 3, 4},
        0,
        membership,
        reuseGraph,
        vertexToLocal,
        3,
        placedOrder,
        outScratch,
        inScratch,
        localIds);
    const bool firstOrderSized =
        placedOrder.size() == 5;
    const std::vector<NodeID> firstCommunity(
        localIds.begin(), localIds.begin() + 5);
    graphbrew::intraGorderFaithful<NodeID, NodeID, NodeID>(
        std::vector<NodeID>{5, 6},
        1,
        membership,
        reuseGraph,
        vertexToLocal,
        3,
        placedOrder,
        outScratch,
        inScratch,
        localIds);
    const bool secondOrderSized =
        placedOrder.size() == 2;
    graphbrew::intraGorderFaithful<NodeID, NodeID, NodeID>(
        std::vector<NodeID>{},
        2,
        membership,
        reuseGraph,
        vertexToLocal,
        3,
        placedOrder,
        outScratch,
        inScratch,
        localIds);
    const bool emptyOrderSized = placedOrder.empty();
    auto firstSorted = firstCommunity;
    auto secondSorted = std::vector<NodeID>{
        localIds[5], localIds[6]};
    std::sort(firstSorted.begin(), firstSorted.end());
    std::sort(secondSorted.begin(), secondSorted.end());
    Require(
        std::equal(
            firstCommunity.begin(),
            firstCommunity.end(),
            localIds.begin())
            && firstOrderSized
            && secondOrderSized
            && emptyOrderSized
            && firstSorted
                == std::vector<NodeID>({0, 1, 2, 3, 4})
            && secondSorted == std::vector<NodeID>({0, 1}),
        "faithful local Gorder corrupted reused shrinking scratch");

    Graph endToEndGraph = BuildSymmetricGraph({
        {1},
        {0, 2},
        {1, 3},
        {2, 4},
        {3},
        {6},
        {5},
        {},
    });
    graphbrew::GraphBrewResult<uint32_t> endToEndResult;
    endToEndResult.membership = {0, 0, 0, 0, 0, 1, 1, 2};
    std::vector<uint32_t> endToEndDegrees = {
        1, 2, 2, 2, 1, 1, 1, 0};
    graphbrew::GraphBrewConfig endToEndConfig;
    endToEndConfig.ordering =
        graphbrew::OrderingStrategy::COMPOSE;
    endToEndConfig.intraCommunityOrder =
        graphbrew::IntraCommunityOrder::GorderFaithful;
    graphbrew::GraphBrewRealizedConfig realized;
    pvector<NodeID> endToEndMapping;
    endToEndMapping.resize(8);
    #ifdef OPENMP
    const int previousThreadCount = omp_get_max_threads();
    omp_set_num_threads(1);
    #endif
    graphbrew::orderCompose<uint32_t, NodeID, NodeID>(
        endToEndMapping,
        endToEndResult,
        endToEndDegrees,
        endToEndGraph,
        8,
        endToEndConfig,
        &realized);
    auto sortedMapping = std::vector<NodeID>(
        endToEndMapping.begin(), endToEndMapping.end());
    std::sort(sortedMapping.begin(), sortedMapping.end());
    Require(
        sortedMapping
                == std::vector<NodeID>({0, 1, 2, 3, 4, 5, 6, 7})
            && realized.gorderCommunities == 2
            && realized.gorderVertices == 7
            && realized.gorderMaxCommunity == 5
            && realized.gorderFallbackCommunities == 0
            && realized.gorderFallbackVertices == 0,
        "faithful local Gorder end-to-end mapping or metadata changed");

    const auto oneThreadMapping = std::vector<NodeID>(
        endToEndMapping.begin(), endToEndMapping.end());
    const auto oneThreadRealized = realized;
    pvector<NodeID> fourThreadMapping(8);
    graphbrew::GraphBrewRealizedConfig fourThreadRealized;
    #ifdef OPENMP
    omp_set_num_threads(4);
    #endif
    graphbrew::orderCompose<uint32_t, NodeID, NodeID>(
        fourThreadMapping,
        endToEndResult,
        endToEndDegrees,
        endToEndGraph,
        8,
        endToEndConfig,
        &fourThreadRealized);
    Require(
        std::equal(
            oneThreadMapping.begin(),
            oneThreadMapping.end(),
            fourThreadMapping.begin())
            && fourThreadRealized.gorderCommunities
                == oneThreadRealized.gorderCommunities
            && fourThreadRealized.gorderVertices
                == oneThreadRealized.gorderVertices
            && fourThreadRealized.gorderMaxCommunity
                == oneThreadRealized.gorderMaxCommunity
            && fourThreadRealized.gorderFallbackCommunities
                == oneThreadRealized.gorderFallbackCommunities
            && fourThreadRealized.gorderFallbackVertices
                == oneThreadRealized.gorderFallbackVertices,
        "faithful local Gorder changed across thread counts");

    endToEndConfig.gorderFallback = 1;
    realized = graphbrew::GraphBrewRealizedConfig{};
    #ifdef OPENMP
    omp_set_num_threads(1);
    #endif
    graphbrew::orderCompose<uint32_t, NodeID, NodeID>(
        endToEndMapping,
        endToEndResult,
        endToEndDegrees,
        endToEndGraph,
        8,
        endToEndConfig,
        &realized);
    #ifdef OPENMP
    omp_set_num_threads(previousThreadCount);
    #endif
    Require(
        realized.gorderCommunities == 0
            && realized.gorderVertices == 0
            && realized.gorderMaxCommunity == 0
            && realized.gorderFallbackCommunities == 2
            && realized.gorderFallbackVertices == 7,
        "faithful local Gorder fallback metadata diverged from dispatch");
}

void TestNoveltyParserParityCorpus()
{
    std::ifstream input(
        "bench/tests/data/graphbrew_novelty_parser_cases.json");
    Require(input.good(), "novelty parser corpus is missing");
    nlohmann::json corpus;
    input >> corpus;
    for (const auto& entry : corpus.at("cases")) {
        const auto tokens =
            entry.at("tokens").get<std::vector<std::string>>();
        bool accepted = true;
        graphbrew::GraphBrewConfig config;
        try {
            config = graphbrew::parseGraphBrewCliConfig(tokens, 0.5);
        } catch (const std::invalid_argument&) {
            accepted = false;
        }
        Require(
            accepted == entry.at("valid").get<bool>(),
            entry.at("name").get_ref<const std::string&>().c_str());
        if (!accepted) continue;
        Require(
            std::string(graphbrew::graphBrewOrderingName(
                config.ordering))
                    == entry.at("ordering").get<std::string>()
                && std::string(graphbrew::graphBrewCommunityOrderName(
                    config.communityOrder))
                    == entry.at("community_order").get<std::string>()
                && std::string(graphbrew::graphBrewIntraOrderName(
                    config.intraCommunityOrder))
                    == entry.at("intra_community_order")
                        .get<std::string>(),
            "novelty parser corpus final state changed");
    }
}

void TestEffectiveConfigIdentity()
{
    Require(
        std::string(
            graphbrew::database::REORDER_SEMANTICS_VERSION)
            == "graphbrew-reorder/v3",
        "native reorder semantics version changed");
    const auto base = graphbrew::parseGraphBrewConfig(
        {"compose", "comm_size", "intra_gorder"}, true);
    auto serial = base;
    auto parallel = base;
    serial.deterministicCommunityDetection = true;
    parallel.deterministicCommunityDetection = false;
    auto refined = base;
    refined.refinementPass = graphbrew::RefinementPass::TwoSwap;
    auto totalEdges = base;
    totalEdges.mComputation = graphbrew::MComputation::TOTAL_EDGES;
    auto lazy = base;
    lazy.useLazyUpdates = true;
    auto verified = base;
    verified.verifyTopology = true;
    auto faithful = base;
    faithful.intraCommunityOrder =
        graphbrew::IntraCommunityOrder::GorderFaithful;
    const auto capacity4 = graphbrew::parseGraphBrewConfig({
        "compose",
        "comm_capacity_runs",
        "capl2k4",
        "capllck16",
        "capv8",
    }, true);
    const auto capacity8 = graphbrew::parseGraphBrewConfig({
        "compose",
        "comm_capacity_runs",
        "capl2k8",
        "capllck16",
        "capv8",
    }, true);

    auto identity = [](const graphbrew::GraphBrewConfig& config) {
        return graphbrew::graphBrewEffectiveConfigJson(config, false);
    };
    Require(
        identity(serial) != identity(parallel)
            && identity(base) != identity(refined)
            && identity(base) != identity(totalEdges)
            && identity(base) != identity(lazy)
            && identity(base) != identity(verified)
            && identity(base) != identity(faithful)
            && identity(capacity4) != identity(capacity8),
        "effective GraphBrew identities collided");
    Require(
        identity(base).find("\"schema\"") == std::string::npos
            && graphbrew::graphBrewEffectiveConfigJson(base).find(
                "\"schema\":\"graphbrew_config/v3\"")
                != std::string::npos,
        "effective config schema inclusion changed");

    bool rejected = false;
    try {
        (void)graphbrew::parseGraphBrewConfig({
            "compose",
            "comm_capacity_runs",
            "capl2k16",
            "capllck4",
            "capv8",
        });
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "non-strict capacity parsing clamped invalid geometry");
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
        TestAllKernelLowReuseRule();
        TestSuperGraphMoveBatchParsing();
        TestCapacityRunCommunityOrder();
        TestNoveltyParserParityCorpus();
        TestEffectiveConfigIdentity();
        TestFaithfulLocalGorder();
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
