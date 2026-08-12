#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
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

Graph BuildPathGraph(NodeID nodes)
{
    pvector<SGOffset> offsets(nodes + 1);
    size_t edges = 0;
    for (NodeID vertex = 0; vertex < nodes; ++vertex)
    {
        offsets[vertex] = edges;
        if (vertex > 0)
            ++edges;
        if (vertex + 1 < nodes)
            ++edges;
    }
    offsets[nodes] = edges;
    auto *neighbors = new NodeID[edges];
    size_t position = 0;
    for (NodeID vertex = 0; vertex < nodes; ++vertex)
    {
        if (vertex > 0)
            neighbors[position++] = vertex - 1;
        if (vertex + 1 < nodes)
            neighbors[position++] = vertex + 1;
    }
    auto **index = Graph::GenIndex(offsets, neighbors);
    return Graph(nodes, index, neighbors);
}

void ReferenceRefine(
    const std::vector<NodeID> &vertices,
    NodeID community,
    const std::vector<NodeID> &membership,
    const Graph &graph,
    std::vector<NodeID> &local_ids,
    int max_passes)
{
    const size_t size = vertices.size();
    std::vector<NodeID> order(size, 0);
    for (NodeID vertex : vertices)
        order[local_ids[vertex]] = vertex;

    std::vector<std::vector<NodeID>> adjacency(size);
    for (size_t i = 0; i < size; ++i)
    {
        NodeID vertex = vertices[i];
        for (NodeID neighbor : graph.out_neigh(vertex))
        {
            if (neighbor == vertex)
                continue;
            if (membership[neighbor] == community)
                adjacency[i].push_back(neighbor);
        }
    }

    for (int pass = 0; pass < max_passes; ++pass)
    {
        size_t swapped = 0;
        for (size_t position = 0; position + 1 < size; ++position)
        {
            NodeID first = order[position];
            NodeID second = order[position + 1];
            size_t first_index = 0;
            size_t second_index = 0;
            for (size_t i = 0; i < size; ++i)
            {
                if (vertices[i] == first)
                    first_index = i;
                if (vertices[i] == second)
                    second_index = i;
            }

            int delta = 0;
            for (NodeID neighbor : adjacency[first_index])
            {
                if (neighbor == second)
                    continue;
                size_t neighbor_position = local_ids[neighbor];
                if (neighbor_position < position)
                    ++delta;
                else if (neighbor_position > position + 1)
                    --delta;
            }
            for (NodeID neighbor : adjacency[second_index])
            {
                if (neighbor == first)
                    continue;
                size_t neighbor_position = local_ids[neighbor];
                if (neighbor_position < position)
                    --delta;
                else if (neighbor_position > position + 1)
                    ++delta;
            }
            if (delta < 0)
            {
                std::swap(order[position], order[position + 1]);
                std::swap(local_ids[first], local_ids[second]);
                ++swapped;
            }
        }
        if (swapped == 0)
            break;
    }
}

} // namespace

int main()
{
    try
    {
        Graph graph = BuildGraph();
        std::vector<NodeID> vertices{3, 0, 5, 1, 4, 2};
        std::vector<NodeID> membership(6, 0);
        std::vector<NodeID> expected{4, 1, 5, 0, 3, 2};
        std::vector<NodeID> actual = expected;
        std::vector<size_t> vertex_to_local(6);
        for (size_t i = 0; i < vertices.size(); ++i)
            vertex_to_local[vertices[i]] = i;

        ReferenceRefine(
            vertices, 0, membership, graph, expected, 3);

        std::vector<size_t> adjacency_offsets;
        std::vector<NodeID> adjacency_vertices;
        std::vector<NodeID> order;
        graphbrew::refineTwoSwap<NodeID, NodeID, NodeID>(
            vertices, 0, membership, graph, actual, vertex_to_local, 3,
            adjacency_offsets, adjacency_vertices, order);

        Require(
            actual == expected,
            "Flat two-swap refinement changed ordering semantics");
        std::vector<NodeID> sorted = actual;
        std::sort(sorted.begin(), sorted.end());
        for (NodeID i = 0; i < static_cast<NodeID>(sorted.size()); ++i)
            Require(sorted[i] == i, "Two-swap output is not a permutation");

        std::vector<NodeID> foreign_vertices{3, 4, 5};
        std::vector<size_t> foreign_lookup{0, 1, 2, 0, 1, 2};
        size_t foreign_index = 0;
        Require(
            !graphbrew::refineVertexIndex(
                foreign_vertices, foreign_lookup, NodeID(0), foreign_index),
            "Two-swap accepted a vertex from another community");

        std::vector<NodeID> split_membership{0, 0, 0, 1, 1, 1};
        std::vector<NodeID> split_local_ids{99, 0, 0, 0, 2, 0};
        std::vector<NodeID> split_before = split_local_ids;
        std::vector<size_t> split_lookup{2, 0, 0, 0, 1, 2};
        graphbrew::refineTwoSwap<NodeID, NodeID, NodeID>(
            foreign_vertices, 1, split_membership, graph,
            split_local_ids, split_lookup, 3,
            adjacency_offsets, adjacency_vertices, order);
        Require(
            split_local_ids == split_before,
            "Two-swap modified a foreign-community vertex through a stale slot");

        constexpr NodeID large_size = 70;
        Graph large_graph = BuildPathGraph(large_size);
        std::vector<NodeID> large_vertices(large_size);
        std::vector<NodeID> large_membership(large_size, 0);
        std::vector<NodeID> large_expected(large_size);
        std::vector<size_t> large_lookup(large_size);
        for (NodeID vertex = 0; vertex < large_size; ++vertex)
        {
            large_vertices[vertex] = vertex;
            large_expected[vertex] = (vertex * 37) % large_size;
            large_lookup[vertex] = vertex;
        }
        std::vector<NodeID> large_actual = large_expected;
        ReferenceRefine(
            large_vertices, 0, large_membership, large_graph,
            large_expected, 3);
        graphbrew::refineTwoSwap<NodeID, NodeID, NodeID>(
            large_vertices, 0, large_membership, large_graph,
            large_actual, large_lookup, 3,
            adjacency_offsets, adjacency_vertices, order);
        Require(
            large_actual == large_expected,
            "Large flat two-swap changed ordering semantics");
    }
    catch (const std::exception &error)
    {
        std::cerr << "Two-swap refinement test failed: "
                  << error.what() << std::endl;
        return 1;
    }

    std::cout << "Two-swap refinement tests passed" << std::endl;
    return 0;
}
