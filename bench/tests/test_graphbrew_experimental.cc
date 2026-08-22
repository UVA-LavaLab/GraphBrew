#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "graphbrew/reorder/experimental/compression_layout.h"
#include "graphbrew/reorder/experimental/dual_index.h"
#include "graphbrew/reorder/experimental/locality_probe.h"
#include "graphbrew/reorder/experimental/spectral_blocks.h"

namespace
{

using Node = uint32_t;
using WeightedAdjacency =
    std::vector<std::vector<std::pair<Node, double>>>;

void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::vector<Node>> MakePath(size_t size)
{
    std::vector<std::vector<Node>> adjacency(size);
    for (size_t vertex = 0; vertex < size; ++vertex) {
        if (vertex > 0)
            adjacency[vertex].push_back(
                static_cast<Node>(vertex - 1));
        if (vertex + 1 < size)
            adjacency[vertex].push_back(
                static_cast<Node>(vertex + 1));
    }
    return adjacency;
}

WeightedAdjacency Weighted(
    const std::vector<std::vector<Node>>& adjacency)
{
    WeightedAdjacency weighted(adjacency.size());
    for (size_t source = 0; source < adjacency.size(); ++source) {
        for (Node target : adjacency[source])
            weighted[source].push_back({target, 1.0});
    }
    return weighted;
}

bool IsPermutation(const std::vector<Node>& order)
{
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    std::vector<Node> expected(order.size());
    std::iota(expected.begin(), expected.end(), Node(0));
    return sorted == expected;
}

std::vector<std::pair<Node, Node>> DecodeDualEdges(
    const graphbrew::experimental::DualIndexLayout<Node>& layout)
{
    std::vector<std::pair<Node, Node>> edges;
    for (size_t sourceId = 0;
         sourceId < layout.nodeCount();
         ++sourceId) {
        const Node source = layout.sourceToOriginal[sourceId];
        for (size_t edge = layout.outOffsets[sourceId];
             edge < layout.outOffsets[sourceId + 1];
             ++edge) {
            edges.push_back({
                source,
                layout.destinationToOriginal[
                    layout.outDestinationIds[edge]],
            });
        }
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

void TestDualIndexLayout()
{
    const std::vector<std::vector<Node>> outgoing = {
        {0, 1, 1, 2},
        {2},
        {0, 3},
        {1},
    };
    const std::vector<Node> sourceOrder = {2, 0, 3, 1};
    const std::vector<Node> destinationOrder = {1, 3, 0, 2};
    const auto layout =
        graphbrew::experimental::buildDualIndexLayout(
            outgoing, sourceOrder, destinationOrder);

    Require(
        graphbrew::experimental::validateDualIndexLayout(
            layout, outgoing),
        "dual-index layout changed the directed edge multiset");
    Require(
        layout.directedEdgeCount() == 8
            && layout.originalToSource[2] == 0
            && layout.originalToDestination[1] == 0
            && layout.sourceToOriginal[
                layout.originalToSource[3]] == 3
            && layout.destinationToOriginal[
                layout.originalToDestination[2]] == 2,
        "dual-index translation maps are inconsistent");
    Require(
        layout.outOffsets
                == std::vector<size_t>({0, 2, 6, 7, 8})
            && layout.outDestinationIds
                == std::vector<Node>({1, 2, 0, 0, 2, 3, 0, 3})
            && layout.inOffsets
                == std::vector<size_t>({0, 3, 4, 6, 8})
            && layout.inSourceIds
                == std::vector<Node>({1, 1, 2, 0, 0, 1, 1, 3}),
        "dual-index CSR/CSC arrays changed");

    const uint64_t expectedBytes =
        4 * outgoing.size() * sizeof(Node)
        + 2 * (outgoing.size() + 1) * sizeof(size_t)
        + 2 * layout.directedEdgeCount() * sizeof(Node);
    Require(
        layout.modeledBytes() == expectedBytes,
        "dual-index memory accounting changed");

    bool rejected = false;
    try {
        (void)graphbrew::experimental::buildDualIndexLayout(
            outgoing,
            std::vector<Node>{0, 0, 2, 3},
            destinationOrder);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "dual-index layout accepted a non-permutation");

    rejected = false;
    try {
        (void)graphbrew::experimental::buildDualIndexLayout(
            outgoing,
            sourceOrder,
            std::vector<Node>{0, 1, 1, 3});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "dual-index layout accepted a bad destination permutation");

    auto corrupted = layout;
    corrupted.outOffsets.pop_back();
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator accepted truncated offsets");
    corrupted = layout;
    corrupted.outDestinationIds[0] = 99;
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator accepted an invalid neighbor");
    corrupted = layout;
    corrupted.outOffsets[1] =
        corrupted.outDestinationIds.size() + 1;
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator accepted an oversized middle offset");
    corrupted = layout;
    corrupted.originalToSource[0] =
        corrupted.originalToSource[1];
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator accepted bad inverse maps");
    corrupted = layout;
    corrupted.outDestinationIds[3] = 1;
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator skipped outgoing multiset validation");
    corrupted = layout;
    corrupted.inSourceIds[1] = 2;
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator skipped incoming multiset validation");
    corrupted = layout;
    corrupted.inOffsets.pop_back();
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator accepted truncated incoming offsets");
    corrupted = layout;
    corrupted.inSourceIds[0] = 99;
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator accepted an invalid incoming neighbor");
    corrupted = layout;
    corrupted.originalToDestination[0] =
        corrupted.originalToDestination[1];
    Require(
        !graphbrew::experimental::validateDualIndexLayout(
            corrupted, outgoing),
        "dual-index validator accepted a bad destination inverse");

    const auto empty =
        graphbrew::experimental::buildDualIndexLayout<Node>(
            {}, {}, {});
    Require(
        empty.outOffsets == std::vector<size_t>({0})
            && empty.inOffsets == std::vector<size_t>({0})
            && graphbrew::experimental::validateDualIndexLayout(
                empty, {}),
        "dual-index empty-graph handling changed");

    std::vector<uint8_t> byteOrder(256);
    std::iota(byteOrder.begin(), byteOrder.end(), uint8_t(0));
    Require(
        graphbrew::experimental::invertOrder(byteOrder)
            == byteOrder,
        "experimental permutation capacity is off by one");

    bool signedRejected = false;
    try {
        (void)graphbrew::experimental::buildDualIndexLayout<int>(
            {{-1}}, {0}, {0});
    } catch (const std::invalid_argument&) {
        signedRejected = true;
    }
    Require(
        signedRejected,
        "dual-index layout accepted a negative target");

    const auto degreeZero =
        graphbrew::experimental::buildDualIndexLayout<Node>(
            {{1}, {}, {1}, {}},
            {1, 0, 2, 3},
            {0, 1, 2, 3});
    Require(
        degreeZero.outOffsets[1] == degreeZero.outOffsets[0]
            && degreeZero.outOffsets[4]
                == degreeZero.outOffsets[3]
            && degreeZero.inOffsets[1] == degreeZero.inOffsets[0]
            && degreeZero.inOffsets[4] == degreeZero.inOffsets[3]
            && graphbrew::experimental::validateDualIndexLayout(
                degreeZero, {{1}, {}, {1}, {}}),
        "dual-index layout mishandled empty boundary rows");

    std::mt19937 generator(0xD0411D3u);
    for (size_t trial = 0; trial < 200; ++trial) {
        const size_t size = generator() % 33;
        std::vector<std::vector<Node>> randomOutgoing(size);
        for (size_t source = 0; source < size; ++source) {
            const size_t arcs = generator() % 5;
            for (size_t edge = 0; edge < arcs; ++edge) {
                randomOutgoing[source].push_back(
                    static_cast<Node>(generator() % size));
            }
        }
        std::vector<Node> randomSource(size);
        std::vector<Node> randomDestination(size);
        std::iota(
            randomSource.begin(), randomSource.end(), Node(0));
        std::iota(
            randomDestination.begin(),
            randomDestination.end(),
            Node(0));
        std::shuffle(
            randomSource.begin(), randomSource.end(), generator);
        std::shuffle(
            randomDestination.begin(),
            randomDestination.end(),
            generator);
        const auto randomLayout =
            graphbrew::experimental::buildDualIndexLayout(
                randomOutgoing,
                randomSource,
                randomDestination);
        std::vector<std::pair<Node, Node>> expected;
        for (size_t source = 0; source < size; ++source) {
            for (Node target : randomOutgoing[source]) {
                expected.push_back({
                    static_cast<Node>(source), target});
            }
        }
        std::sort(expected.begin(), expected.end());
        Require(
            graphbrew::experimental::validateDualIndexLayout(
                randomLayout, randomOutgoing)
                && DecodeDualEdges(randomLayout) == expected,
            "dual-index randomized edge preservation failed");
    }
}

void TestExactSpectralOrder()
{
    const auto path = MakePath(8);
    const auto pathResult =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            Weighted(path));
    Require(
        pathResult.status
                == graphbrew::experimental::SpectralOrderStatus::Success
            && pathResult.order
                == std::vector<Node>({0, 1, 2, 3, 4, 5, 6, 7})
            && pathResult.lambda2 > 0.0
            && pathResult.lambda3 > pathResult.lambda2
            && std::abs(
                pathResult.lambda2 - 0.0761204675) < 1e-8
            && std::abs(
                pathResult.lambda3 - 0.2928932188) < 1e-8
            && pathResult.minimumEigengap > 0.0
            && pathResult.residual < 1e-8,
        "exact spectral ordering failed on a path");

    auto tinyScale = Weighted(path);
    auto hugeScale = Weighted(path);
    for (auto& row : tinyScale)
        for (auto& edge : row) edge.second *= 1e-12;
    for (auto& row : hugeScale)
        for (auto& edge : row) edge.second *= 1e12;
    Require(
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            tinyScale).order == pathResult.order
            && graphbrew::experimental::exactFiedlerBlockOrder<Node>(
                hugeScale).order == pathResult.order,
        "spectral ordering changed under uniform weight scaling");

    auto star = std::vector<std::vector<Node>>(6);
    for (Node leaf = 1; leaf < 6; ++leaf) {
        star[0].push_back(leaf);
        star[leaf].push_back(0);
    }
    const auto starResult =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            Weighted(star));
    Require(
        starResult.status
                == graphbrew::experimental::SpectralOrderStatus::Degenerate
            && starResult.order.empty(),
        "spectral ordering hid a repeated Fiedler eigenvalue");

    WeightedAdjacency weakBridge = {
        {{1, 1.0}},
        {{0, 1.0}, {2, 1e-12}},
        {{1, 1e-12}},
    };
    Require(
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            weakBridge).status
            == graphbrew::experimental::SpectralOrderStatus::Degenerate
            && graphbrew::experimental::exactFiedlerBlockOrder<Node>(
                weakBridge).order.empty(),
        "spectral ordering accepted a numerically unresolved Fiedler mode");

    auto disconnected = Weighted(MakePath(8));
    disconnected[3].erase(
        std::remove_if(
            disconnected[3].begin(),
            disconnected[3].end(),
            [](const auto& edge) { return edge.first == 4; }),
        disconnected[3].end());
    disconnected[4].erase(
        std::remove_if(
            disconnected[4].begin(),
            disconnected[4].end(),
            [](const auto& edge) { return edge.first == 3; }),
        disconnected[4].end());
    disconnected[6].erase(
        std::remove_if(
            disconnected[6].begin(),
            disconnected[6].end(),
            [](const auto& edge) { return edge.first == 7; }),
        disconnected[6].end());
    disconnected[7].clear();
    const auto disconnectedResult =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            disconnected,
            {},
            std::vector<size_t>{1, 1, 1, 1, 1, 1, 1, 0});
    Require(
        disconnectedResult.status
                == graphbrew::experimental::SpectralOrderStatus::Success
            && disconnectedResult.componentCount == 2
            && disconnectedResult.order.back() == 7
            && IsPermutation(disconnectedResult.order),
        "spectral component or empty-vertex handling changed");

    const auto isolated =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            WeightedAdjacency(4));
    const auto allEmpty =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            WeightedAdjacency(4),
            {},
            std::vector<size_t>(4, 0));
    Require(
        isolated.status
                == graphbrew::experimental::SpectralOrderStatus::Trivial
            && allEmpty.status
                == graphbrew::experimental::SpectralOrderStatus::Trivial
            && IsPermutation(isolated.order)
            && IsPermutation(allEmpty.order),
        "spectral trivial-component status changed");

    const auto tooLarge =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            Weighted(MakePath(65)));
    Require(
        tooLarge.status
                == graphbrew::experimental::SpectralOrderStatus::TooLarge
            && tooLarge.order.empty(),
        "spectral validity size limit was ignored");

    const auto totalTooLarge =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            WeightedAdjacency(
                graphbrew::experimental::kSpectralMaxTotalVertices + 1));
    Require(
        totalTooLarge.status
                == graphbrew::experimental::SpectralOrderStatus::TooLarge
            && totalTooLarge.order.empty(),
        "spectral total-size limit was ignored");

    bool rejected = false;
    try {
        (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            Weighted(path),
            std::vector<Node>{0, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "spectral ordering accepted a short base order");
    rejected = false;
    try {
        (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            WeightedAdjacency{
                {{1, 5e-14}},
                {},
            });
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "spectral ordering accepted asymmetric support");

    for (double scale : {1e-12, 1.0}) {
        rejected = false;
        try {
            (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
                WeightedAdjacency{
                    {{1, scale}},
                    {{0, scale * 1.05}},
                });
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Require(
            rejected,
            "spectral symmetry check changed under uniform scaling");
    }
    const auto hugeFinite =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            WeightedAdjacency{
                {{1, 1e308}},
                {{0, 1e308}},
            });
    Require(
        hugeFinite.status
            == graphbrew::experimental::SpectralOrderStatus::Trivial,
        "spectral canonical averaging overflowed finite weights");

    rejected = false;
    try {
        (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            {}, std::vector<Node>{0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "spectral empty graph accepted a nonempty base order");
    rejected = false;
    try {
        (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            {}, {}, std::vector<size_t>{1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "spectral empty graph accepted nonempty masses");

    WeightedAdjacency twoComponents(6);
    for (Node start : {Node(0), Node(3)}) {
        twoComponents[start].push_back({start + 1, 1.0});
        twoComponents[start + 1].push_back({start, 1.0});
        twoComponents[start + 1].push_back({start + 2, 1.0});
        twoComponents[start + 2].push_back({start + 1, 1.0});
    }
    const auto baseFirst =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            twoComponents,
            std::vector<Node>{3, 4, 5, 0, 1, 2},
            std::vector<size_t>{100, 100, 100, 1, 1, 1});
    Require(
        baseFirst.status
                == graphbrew::experimental::SpectralOrderStatus::Success
            && baseFirst.order.front() == 3,
        "spectral components stopped following base-order rank");

    WeightedAdjacency barbell(6);
    auto addUndirected = [&](Node left, Node right) {
        barbell[left].push_back({right, 1.0});
        barbell[right].push_back({left, 1.0});
    };
    addUndirected(0, 1);
    addUndirected(0, 2);
    addUndirected(1, 2);
    addUndirected(2, 3);
    addUndirected(3, 4);
    addUndirected(3, 5);
    addUndirected(4, 5);
    const auto tied =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            barbell,
            std::vector<Node>{1, 0, 2, 3, 5, 4});
    const auto tiedPosition =
        graphbrew::experimental::invertOrder(tied.order);
    Require(
        tied.status
                == graphbrew::experimental::SpectralOrderStatus::Success
            && tiedPosition[1] < tiedPosition[0]
            && tiedPosition[5] < tiedPosition[4],
        "spectral tied coordinates lost base-order tie-breaking");

    const auto capPath =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            Weighted(MakePath(
                graphbrew::experimental::
                    kSpectralMaxComponentVertices)));
    Require(
        capPath.status
            == graphbrew::experimental::SpectralOrderStatus::Success,
        "spectral solver failed at its component cap");

    for (const auto& invalidGraph : {
             WeightedAdjacency{{{1, -1.0}}, {{0, -1.0}}},
             WeightedAdjacency{{
                 {0, std::numeric_limits<double>::quiet_NaN()}}},
             WeightedAdjacency{{{2, 1.0}}, {}},
         }) {
        rejected = false;
        try {
            (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
                invalidGraph);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Require(
            rejected,
            "spectral ordering accepted invalid adjacency data");
    }
    rejected = false;
    try {
        (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            Weighted(MakePath(2)),
            {},
            std::vector<size_t>{0, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "spectral ordering accepted edges incident to empty mass");
    rejected = false;
    try {
        (void)graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            Weighted(MakePath(2)),
            std::vector<Node>{0, 0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "spectral ordering accepted a non-permutation base order");

    const std::vector<Node> oldToNew = {2, 0, 3, 1};
    std::vector<Node> newToOld(4);
    for (Node old = 0; old < 4; ++old)
        newToOld[oldToNew[old]] = old;
    auto original = Weighted(MakePath(4));
    WeightedAdjacency relabeled(4);
    for (Node oldSource = 0; oldSource < 4; ++oldSource) {
        for (const auto& edge : original[oldSource]) {
            relabeled[oldToNew[oldSource]].push_back({
                oldToNew[edge.first], edge.second});
        }
    }
    const auto relabeledResult =
        graphbrew::experimental::exactFiedlerBlockOrder<Node>(
            relabeled,
            std::vector<Node>{2, 0, 3, 1});
    std::vector<Node> decoded;
    for (Node vertex : relabeledResult.order)
        decoded.push_back(newToOld[vertex]);
    Require(
        relabeledResult.status
                == graphbrew::experimental::SpectralOrderStatus::Success
            && decoded
                == graphbrew::experimental::exactFiedlerBlockOrder<Node>(
                    original).order,
        "spectral ordering is not invariant to relabeling");
}

void TestStructuralLocalityProbe()
{
    const auto path = MakePath(6);
    const std::vector<Node> cheap = {0, 5, 1, 4, 2, 3};
    const std::vector<Node> expensive = {0, 1, 2, 3, 4, 5};
    graphbrew::experimental::StructuralProbeConfig config;
    config.window = 1;
    const auto result =
        graphbrew::experimental::compareLocalOrders(
            path, cheap, expensive, config);
    Require(
        result.chooseExpensive
            && result.cheapWindowScore == 3
            && result.expensiveWindowScore == 10
            && result.cheapDirectedEdgeSpan == 18
            && result.expensiveDirectedEdgeSpan == 10
            && result.expensiveWindowScore
                > result.cheapWindowScore
            && result.expensiveDirectedEdgeSpan
                < result.cheapDirectedEdgeSpan
            && result.inputArcVisits == 10
            && result.normalizedDirectedEdges == 10
            && result.edgeSpanVisits == 20
            && result.windowPairEvaluations == 10
            && result.directLinkSearches == 20
            && result.predecessorComparisons == 23
            && std::abs(
                result.relativeScoreGain - 7.0 / 3.0) < 1e-12,
        "structural locality probe missed a path-locality improvement");

    std::vector<std::vector<Node>> clique(4);
    for (Node source = 0; source < 4; ++source) {
        for (Node target = 0; target < 4; ++target) {
            if (source != target) clique[source].push_back(target);
        }
    }
    const auto tie =
        graphbrew::experimental::compareLocalOrders(
            clique,
            std::vector<Node>{0, 1, 2, 3},
            std::vector<Node>{3, 2, 1, 0},
            config);
    Require(
        !tie.chooseExpensive,
        "structural locality probe selected an equivalent clique order");

    graphbrew::experimental::StructuralProbeConfig exactConfig;
    exactConfig.window = 1;
    const auto oneWay =
        graphbrew::experimental::compareLocalOrders(
            std::vector<std::vector<Node>>{{1}, {}},
            std::vector<Node>{0, 1},
            std::vector<Node>{0, 1},
            exactConfig);
    const auto reciprocal =
        graphbrew::experimental::compareLocalOrders(
            std::vector<std::vector<Node>>{{1}, {0}},
            std::vector<Node>{0, 1},
            std::vector<Node>{0, 1},
            exactConfig);
    const auto duplicate =
        graphbrew::experimental::compareLocalOrders(
            std::vector<std::vector<Node>>{{1, 1}, {}},
            std::vector<Node>{0, 1},
            std::vector<Node>{0, 1},
            exactConfig);
    const auto selfLoop =
        graphbrew::experimental::compareLocalOrders(
            std::vector<std::vector<Node>>{{0, 1}, {}},
            std::vector<Node>{0, 1},
            std::vector<Node>{0, 1},
            exactConfig);
    Require(
        oneWay.cheapWindowScore == 1
            && reciprocal.cheapWindowScore == 2
            && duplicate.cheapWindowScore == 1
            && duplicate.inputArcVisits == 2
            && duplicate.normalizedDirectedEdges == 1
            && duplicate.edgeSpanVisits == 2
            && selfLoop.cheapWindowScore == 2
            && selfLoop.normalizedDirectedEdges == 2
            && selfLoop.cheapDirectedEdgeSpan == 1,
        "structural probe direct/common-predecessor semantics changed");

    const auto commonPredecessor =
        graphbrew::experimental::compareLocalOrders(
            std::vector<std::vector<Node>>{{1, 2}, {}, {}, {}},
            std::vector<Node>{0, 3, 1, 2},
            std::vector<Node>{0, 3, 1, 2},
            exactConfig);
    Require(
        commonPredecessor.cheapWindowScore == 1,
        "structural probe lost its common-predecessor term");
    Require(
        !graphbrew::experimental::shouldChooseExpensive(
            UINT64_C(9007199254740992),
            UINT64_C(9007199254740993),
            UINT64_C(9007199254740992),
            UINT64_C(9007199254740993),
            0,
            10000)
            && graphbrew::experimental::shouldChooseExpensive(
                UINT64_C(9007199254740992),
                UINT64_C(9007199254740993),
                100,
                100,
                0,
                10000)
            && graphbrew::experimental::shouldChooseExpensive(
                100, 103, 100, 100, 300, 10000)
            && !graphbrew::experimental::shouldChooseExpensive(
                100, 200, 100, 101, 0, 10000)
            && graphbrew::experimental::shouldChooseExpensive(
                100, 200, 100, 101, 0, 11000),
        "structural probe predicate lost exact ratio semantics");

    const auto empty =
        graphbrew::experimental::compareLocalOrders<Node>(
            {}, {}, {}, exactConfig);
    Require(
        !empty.chooseExpensive
            && empty.cheapWindowScore == 0
            && empty.expensiveWindowScore == 0,
        "structural probe empty-graph handling changed");

    bool rejected = false;
    try {
        (void)graphbrew::experimental::compareLocalOrders<int>(
            {{-1}}, {0}, {0}, exactConfig);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "structural probe accepted a negative edge target");

    rejected = false;
    try {
        std::vector<std::vector<Node>> tooManyVertices(
            graphbrew::experimental::kStructuralProbeMaxVertices + 1);
        std::vector<Node> order(tooManyVertices.size());
        std::iota(order.begin(), order.end(), Node(0));
        (void)graphbrew::experimental::compareLocalOrders(
            tooManyVertices, order, order, exactConfig);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "structural probe ignored its vertex limit");
    rejected = false;
    try {
        auto badWindow = exactConfig;
        badWindow.window =
            graphbrew::experimental::kStructuralProbeMaxWindow + 1;
        (void)graphbrew::experimental::compareLocalOrders(
            path, cheap, expensive, badWindow);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "structural probe ignored its window limit");
    rejected = false;
    try {
        std::vector<std::vector<Node>> tooManyEdges(1);
        tooManyEdges[0].assign(
            graphbrew::experimental::
                kStructuralProbeMaxDirectedEdges + 1,
            0);
        (void)graphbrew::experimental::compareLocalOrders(
            tooManyEdges,
            std::vector<Node>{0},
            std::vector<Node>{0},
            exactConfig);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "structural probe ignored its edge limit");
}

void TestCompressionLayout()
{
    using graphbrew::experimental::compression_detail::modeledBits;
    using graphbrew::experimental::compression_detail::varintBytes;
    using graphbrew::experimental::compression_detail::zigZag;
    Require(
        zigZag(0) == 0
            && zigZag(-1) == 1
            && zigZag(1) == 2
            && zigZag(-2) == 3
            && zigZag(std::numeric_limits<int64_t>::min())
                == std::numeric_limits<uint64_t>::max()
            && modeledBits(0) == 1
            && modeledBits(1) == 1
            && modeledBits(2) == 2
            && modeledBits(127) == 7
            && modeledBits(128) == 8
            && varintBytes(0) == 1
            && varintBytes(127) == 1
            && varintBytes(128) == 2
            && varintBytes(16383) == 2
            && varintBytes(16384) == 3,
        "compression integer-code contract changed");

    const auto exactMetrics =
        graphbrew::experimental::measureGapEncoding(
            std::vector<std::vector<Node>>{{0, 1, 1}, {0}},
            std::vector<Node>{0, 1},
            4);
    Require(
        exactMetrics.modeledGapBits == 4
            && exactMetrics.modeledVarintBytes == 4
            && exactMetrics.fixedNeighborBytes == 4 * sizeof(Node)
            && exactMetrics.fixedOffsetBytes == 3 * sizeof(size_t)
            && exactMetrics.weightBytes == 16
            && exactMetrics.directedEdges == 4,
        "compression exact row encoding changed");

    const auto nonUnitGap =
        graphbrew::experimental::measureGapEncoding(
            std::vector<std::vector<Node>>{
                {1, 4}, {}, {}, {}, {}},
            std::vector<Node>{0, 1, 2, 3, 4});
    Require(
        nonUnitGap.modeledGapBits == 4
            && nonUnitGap.modeledVarintBytes == 2,
        "compression encoding stopped using successive neighbor gaps");

    const auto emptyMetrics =
        graphbrew::experimental::measureGapEncoding<Node>(
            {}, {});
    Require(
        emptyMetrics.modeledGapBits == 0
            && emptyMetrics.modeledVarintBytes == 0
            && emptyMetrics.fixedOffsetBytes == sizeof(size_t),
        "compression empty-graph accounting changed");

    const auto path = MakePath(16);
    std::vector<Node> natural(16);
    std::iota(natural.begin(), natural.end(), Node(0));
    const std::vector<Node> separated = {
        0, 2, 4, 6, 8, 10, 12, 14,
        1, 3, 5, 7, 9, 11, 13, 15,
    };
    const auto naturalMetrics =
        graphbrew::experimental::measureGapEncoding(
            path, natural, 4);
    const auto separatedMetrics =
        graphbrew::experimental::measureGapEncoding(
            path, separated, 4);
    Require(
        naturalMetrics.modeledGapBits
                < separatedMetrics.modeledGapBits
            && naturalMetrics.fixedNeighborBytes
                == separatedMetrics.fixedNeighborBytes
            && naturalMetrics.weightBytes
                == naturalMetrics.directedEdges * 4,
        "compression metrics confused gap and fixed-width bytes");

    const std::vector<Node> refinementSeed = {
        2, 10, 0, 14, 6, 5, 3, 8,
        7, 11, 15, 1, 12, 13, 9, 4,
    };
    const auto refined =
        graphbrew::experimental::refineCompressionOrder(
            path, refinementSeed, 32);
    Require(
        IsPermutation(refined.order)
            && refined.before.modeledGapBits == 101
            && refined.after.modeledGapBits == 95
            && refined.acceptedSwaps == 4
            && graphbrew::experimental::measureGapEncoding(
                path, refined.order).modeledGapBits
                == refined.after.modeledGapBits,
        "compression refinement worsened or misreported its objective");

    const auto zeroPasses =
        graphbrew::experimental::refineCompressionOrder(
            path, natural, 0);
    const auto repeated =
        graphbrew::experimental::refineCompressionOrder(
            path, refinementSeed, 32);
    Require(
        zeroPasses.order == natural
            && zeroPasses.acceptedSwaps == 0
            && zeroPasses.before.modeledGapBits
                == zeroPasses.after.modeledGapBits
            && repeated.order == refined.order
            && repeated.after.modeledGapBits
                == refined.after.modeledGapBits,
        "compression refinement lost zero-pass or repeat determinism");

    const std::vector<std::vector<Node>> edgeless(4);
    const std::vector<Node> edgelessOrder = {2, 0, 3, 1};
    const auto equalCost =
        graphbrew::experimental::refineCompressionOrder(
            edgeless, edgelessOrder, 4);
    Require(
        equalCost.order == edgelessOrder
            && equalCost.acceptedSwaps == 0
            && equalCost.before.modeledGapBits
                == equalCost.after.modeledGapBits,
        "compression refinement accepted an equal-cost swap");

    bool rejected = false;
    try {
        (void)graphbrew::experimental::refineCompressionOrder(
            MakePath(257),
            [] {
                std::vector<Node> order(257);
                std::iota(order.begin(), order.end(), Node(0));
                return order;
            }());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "compression refinement ignored its validity size limit");

    rejected = false;
    try {
        (void)graphbrew::experimental::refineCompressionOrder(
            path,
            natural,
            graphbrew::experimental::kCompressionMaxPasses + 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "compression refinement ignored its pass limit");

    rejected = false;
    try {
        std::vector<std::vector<Node>> tooManyEdges(1);
        tooManyEdges[0].assign(
            graphbrew::experimental::kCompressionMaxDirectedEdges + 1,
            0);
        (void)graphbrew::experimental::refineCompressionOrder(
            tooManyEdges, std::vector<Node>{0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "compression refinement ignored its edge limit");

    rejected = false;
    try {
        std::vector<std::vector<Node>> excessiveWork(256);
        for (auto& row : excessiveWork) {
            row.resize(256);
            std::iota(row.begin(), row.end(), Node(0));
        }
        std::vector<Node> order(256);
        std::iota(order.begin(), order.end(), Node(0));
        (void)graphbrew::experimental::refineCompressionOrder(
            excessiveWork,
            order,
            graphbrew::experimental::kCompressionMaxPasses);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "compression refinement ignored its work limit");

    rejected = false;
    bool rejectedEarly = false;
    try {
        (void)graphbrew::experimental::refineCompressionOrder(
            MakePath(257), std::vector<Node>{0, 1});
    } catch (const std::invalid_argument& error) {
        rejected = true;
        rejectedEarly =
            std::string(error.what()).find("cover every vertex")
            != std::string::npos;
    }
    Require(
        rejected && rejectedEarly,
        "compression refinement delayed order-size validation");

    rejected = false;
    try {
        (void)graphbrew::experimental::measureGapEncoding<int>(
            {{-1}}, {0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(
        rejected,
        "compression metrics accepted a negative edge target");
}

}  // namespace

int main()
{
    try {
        TestDualIndexLayout();
        TestExactSpectralOrder();
        TestStructuralLocalityProbe();
        TestCompressionLayout();
    } catch (const std::exception& error) {
        std::cerr << "GraphBrew experimental test failure: "
                  << error.what() << std::endl;
        return 1;
    }
    std::cout << "GraphBrew experimental validity tests passed"
              << std::endl;
    return 0;
}
