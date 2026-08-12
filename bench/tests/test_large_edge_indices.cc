#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "graphbrew/reorder/reorder_classic.h"

namespace
{

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

Graph BuildGraph()
{
    pvector<SGOffset> offsets(7);
    offsets[0] = 0;
    offsets[1] = 2;
    offsets[2] = 5;
    offsets[3] = 8;
    offsets[4] = 10;
    offsets[5] = 13;
    offsets[6] = 14;
    auto *neighbors = new NodeID[14]{
        1, 2,
        0, 2, 3,
        0, 1, 4,
        1, 4,
        2, 3, 5,
        4,
    };
    auto **index = Graph::GenIndex(offsets, neighbors);
    return Graph(6, index, neighbors);
}

Graph BuildDirectedGraph()
{
    pvector<SGOffset> out_offsets(5);
    out_offsets[0] = 0;
    out_offsets[1] = 2;
    out_offsets[2] = 3;
    out_offsets[3] = 4;
    out_offsets[4] = 4;
    auto *out_neighbors = new NodeID[4]{1, 2, 2, 0};
    auto **out_index = Graph::GenIndex(out_offsets, out_neighbors);

    pvector<SGOffset> in_offsets(5);
    in_offsets[0] = 0;
    in_offsets[1] = 1;
    in_offsets[2] = 2;
    in_offsets[3] = 4;
    in_offsets[4] = 4;
    auto *in_neighbors = new NodeID[4]{2, 0, 0, 1};
    auto **in_index = Graph::GenIndex(in_offsets, in_neighbors);
    return Graph(
        4, out_index, out_neighbors, in_index, in_neighbors);
}

using DirectedGraphNoInverse = CSRGraph<NodeID, NodeID, false>;

DirectedGraphNoInverse BuildDirectedGraphNoInverse()
{
    pvector<SGOffset> offsets(5);
    offsets[0] = 0;
    offsets[1] = 2;
    offsets[2] = 3;
    offsets[3] = 4;
    offsets[4] = 4;
    auto *neighbors = new NodeID[4]{1, 2, 2, 0};
    auto **index = DirectedGraphNoInverse::GenIndex(offsets, neighbors);
    return DirectedGraphNoInverse(
        4, index, neighbors, nullptr, nullptr);
}

template <typename GraphT>
std::vector<std::pair<int, int>> CollectEdges(const GraphT &graph)
{
    std::vector<std::pair<int, int>> edges;
    for (NodeID source = 0; source < graph.num_nodes(); ++source)
    {
        for (NodeID destination : graph.out_neigh(source))
            edges.emplace_back(source, destination);
    }
    return edges;
}

void RequireSameGraph(
    const Gorder::GoGraph &first,
    const Gorder::GoGraph &second)
{
    Require(first.vsize == second.vsize, "GoGraph vertex counts differ");
    Require(first.edgenum == second.edgenum, "GoGraph edge counts differ");
    Require(first.outedge == second.outedge, "GoGraph out-edges differ");
    Require(first.inedge == second.inedge, "GoGraph in-edges differ");
    Require(first.order_l1 == second.order_l1, "GoGraph first orders differ");
    Require(first.graph.size() == second.graph.size(), "GoGraph sizes differ");
    for (size_t i = 0; i < first.graph.size(); ++i)
    {
        Require(
            first.graph[i].outstart == second.graph[i].outstart,
            "GoGraph out-offsets differ");
        Require(
            first.graph[i].outdegree == second.graph[i].outdegree,
            "GoGraph out-degrees differ");
        Require(
            first.graph[i].instart == second.graph[i].instart,
            "GoGraph in-offsets differ");
        Require(
            first.graph[i].indegree == second.graph[i].indegree,
            "GoGraph in-degrees differ");
    }
}

} // namespace

int main()
{
    try
    {
#ifndef Release
        throw std::runtime_error(
            "test must use production GOrder Release semantics");
#endif
        constexpr std::int64_t twitter_directed_edges = 2405026092LL;
        static_assert(
            twitter_directed_edges > std::numeric_limits<std::int32_t>::max());
        static_assert(sizeof(Gorder::EdgeIndex) >= sizeof(std::int64_t));

        Gorder::Vertex vertex;
        vertex.outstart = twitter_directed_edges;
        vertex.instart = twitter_directed_edges;
        Require(
            vertex.outstart == twitter_directed_edges,
            "Gorder out-edge offset was truncated");
        Require(
            vertex.instart == twitter_directed_edges,
            "Gorder in-edge offset was truncated");
        Require(
            graphbrew::classic_detail::EdgePosition(
                twitter_directed_edges, 7) ==
                static_cast<std::size_t>(twitter_directed_edges + 7),
            "Classic reorder edge position was truncated");
        Require(
            graphbrew::classic_detail::PreferGOrderCSR(
                twitter_directed_edges),
            "Large Gorder graph did not select the CSR implementation");
        Require(
            !graphbrew::classic_detail::PreferGOrderCSR(
                std::numeric_limits<std::int32_t>::max()),
            "Small Gorder graph unexpectedly selected the CSR implementation");

        Graph graph = BuildGraph();
        std::vector<std::pair<int, int>> edges = CollectEdges(graph);

        Gorder::GoGraph edge_list_graph;
        edge_list_graph.readGraphEdgelist(edges, graph.num_nodes());
        Gorder::GoGraph csr_graph;
        graphbrew::classic_detail::InitializeGoGraphFromCSR<
            NodeID, NodeID, WeightT, true>(graph, csr_graph);
        RequireSameGraph(edge_list_graph, csr_graph);

        edge_list_graph.Transform();
        csr_graph.Transform();
        RequireSameGraph(edge_list_graph, csr_graph);

        std::vector<int> edge_list_order;
        std::vector<int> csr_order;
        edge_list_graph.RCMOrder(edge_list_order);
        csr_graph.RCMOrder(csr_order);
        Require(
            edge_list_order == csr_order,
            "Direct CSR initialization changed the RCM ordering");

        pvector<NodeID> legacy_ids(graph.num_nodes());
        pvector<NodeID> csr_ids(graph.num_nodes());
        GenerateGOrderMapping<NodeID, NodeID, WeightT, true>(
            graph, legacy_ids, "test");
        GenerateGOrderCSRMapping<NodeID, NodeID, WeightT, true>(
            graph, csr_ids, "test");
        Require(
            std::equal(
                legacy_ids.begin(), legacy_ids.end(), csr_ids.begin()),
            "CSR Gorder implementation changed the ordering");

        Graph directed_graph = BuildDirectedGraph();
        std::vector<std::pair<int, int>> directed_edges =
            CollectEdges(directed_graph);
        Gorder::GoGraph directed_legacy;
        directed_legacy.readGraphEdgelist(
            directed_edges, directed_graph.num_nodes());
        Gorder::GoGraph directed_csr;
        graphbrew::classic_detail::InitializeGoGraphFromCSR<
            NodeID, NodeID, WeightT, true>(
            directed_graph, directed_csr);
        RequireSameGraph(directed_legacy, directed_csr);

        DirectedGraphNoInverse no_inverse_graph =
            BuildDirectedGraphNoInverse();
        std::vector<std::pair<int, int>> no_inverse_edges =
            CollectEdges(no_inverse_graph);
        Gorder::GoGraph no_inverse_legacy;
        no_inverse_legacy.readGraphEdgelist(
            no_inverse_edges, no_inverse_graph.num_nodes());
        Gorder::GoGraph no_inverse_csr;
        graphbrew::classic_detail::InitializeGoGraphFromCSR<
            NodeID, NodeID, WeightT, false>(
            no_inverse_graph, no_inverse_csr);
        RequireSameGraph(no_inverse_legacy, no_inverse_csr);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Large edge-index test failed: "
                  << error.what() << std::endl;
        return 1;
    }

    std::cout << "Large edge-index tests passed" << std::endl;
    return 0;
}
