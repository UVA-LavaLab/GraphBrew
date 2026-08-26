#include <algorithm>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "graphbrew/reorder/reorder_graphbrew.h"

namespace
{

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

Graph BuildDisconnectedCommunityGraph()
{
    pvector<SGOffset> offsets(6);
    offsets[0] = 0;
    offsets[1] = 1;
    offsets[2] = 3;
    offsets[3] = 4;
    offsets[4] = 5;
    offsets[5] = 6;
    auto *neighbors = new NodeID[6]{
        1,
        0, 2,
        1,
        4,
        3,
    };
    auto **index = Graph::GenIndex(offsets, neighbors);
    return Graph(5, index, neighbors);
}

Graph BuildEdgelessGraph()
{
    pvector<SGOffset> offsets(4);
    offsets[0] = 0;
    offsets[1] = 0;
    offsets[2] = 0;
    offsets[3] = 0;
    auto *neighbors = new NodeID[1]{0};
    auto **index = Graph::GenIndex(offsets, neighbors);
    return Graph(3, index, neighbors);
}

void TestDisconnectedTailAndNonzeroBase()
{
    Graph graph = BuildDisconnectedCommunityGraph();
    std::vector<NodeID> vertices{0, 1, 2, 3, 4};
    std::vector<NodeID> membership(5, 0);
    std::vector<NodeID> degrees{1, 2, 1, 1, 1};
    std::vector<size_t> vertexToLocal{0, 1, 2, 3, 4};
    std::vector<bool> visited;
    std::queue<NodeID> queue;
    std::vector<NodeID> localIds(5, -1);
    pvector<NodeID> directIds(5);

    graphbrew::intraBFSFromHub<
        NodeID, NodeID, NodeID>(
        vertices,
        0,
        membership,
        degrees,
        graph,
        vertexToLocal,
        visited,
        queue,
        localIds);
    graphbrew::intraBFSFromHubDirect<
        NodeID, NodeID, NodeID>(
        vertices,
        0,
        membership,
        degrees,
        graph,
        vertexToLocal,
        visited,
        queue,
        7,
        directIds);

    for (NodeID vertex : vertices)
    {
        Require(
            directIds[vertex] == 7 + localIds[vertex],
            "direct emission changed BFS or disconnected-tail order");
    }
    std::vector<NodeID> sorted(
        directIds.begin(), directIds.end());
    std::sort(sorted.begin(), sorted.end());
    Require(
        sorted == std::vector<NodeID>({7, 8, 9, 10, 11}),
        "direct emission did not produce a contiguous nonzero-base range");
}

void TestEdgelessCompaction()
{
    Graph graph = BuildEdgelessGraph();
    pvector<NodeID> newIds(3);
    graphbrew::GraphBrewConfig config;
    config.algorithm = graphbrew::GraphBrewAlgorithm::LEIDEN;
    config.ordering = graphbrew::OrderingStrategy::COMPOSE;
    config.communityOrder = graphbrew::CommunityOrder::Identity;
    config.intraCommunityOrder =
        graphbrew::IntraCommunityOrder::BFSCompact;
    config.maxIterations = 1;
    config.maxPasses = 1;
    config.useRefinement = false;
    config.refinementPass = graphbrew::RefinementPass::None;

    graphbrew::generateGraphBrewMapping<uint32_t>(
        graph, newIds, config);
    std::vector<NodeID> sorted(newIds.begin(), newIds.end());
    std::sort(sorted.begin(), sorted.end());
    Require(
        sorted == std::vector<NodeID>({0, 1, 2}),
        "edgeless compact mapping is not a permutation");
}

} // namespace

int main()
{
    try
    {
        TestDisconnectedTailAndNonzeroBase();
        TestEdgelessCompaction();
        std::cout << "GraphBrew direct-emission tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
}
