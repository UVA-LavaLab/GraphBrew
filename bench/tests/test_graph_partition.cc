#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "partition/compact_csr.h"

namespace
{

using Node = std::int32_t;

class TestGraph
{
public:
    explicit TestGraph(std::size_t num_nodes, bool directed = true)
        : out_(num_nodes), in_(num_nodes), directed_(directed)
    {
    }

    void AddEdge(Node source, Node destination)
    {
        out_.at(static_cast<std::size_t>(source)).push_back(destination);
        in_.at(static_cast<std::size_t>(destination)).push_back(source);
        if (!directed_)
        {
            out_.at(static_cast<std::size_t>(destination)).push_back(source);
            in_.at(static_cast<std::size_t>(source)).push_back(destination);
        }
    }

    std::int64_t num_nodes() const
    {
        return static_cast<std::int64_t>(out_.size());
    }

    std::int64_t num_edges_directed() const
    {
        std::int64_t total = 0;
        for (const auto &neighbors : out_)
            total += static_cast<std::int64_t>(neighbors.size());
        return total;
    }

    bool directed() const
    {
        return directed_;
    }

    std::int64_t out_degree(Node vertex) const
    {
        return static_cast<std::int64_t>(
            out_.at(static_cast<std::size_t>(vertex)).size());
    }

    std::int64_t in_degree(Node vertex) const
    {
        return static_cast<std::int64_t>(
            in_.at(static_cast<std::size_t>(vertex)).size());
    }

    const std::vector<Node> &out_neigh(Node vertex) const
    {
        return out_.at(static_cast<std::size_t>(vertex));
    }

    const std::vector<Node> &in_neigh(Node vertex) const
    {
        return directed_
            ? in_.at(static_cast<std::size_t>(vertex))
            : out_.at(static_cast<std::size_t>(vertex));
    }

private:
    std::vector<std::vector<Node>> out_;
    std::vector<std::vector<Node>> in_;
    bool directed_;
};

void Require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

TestGraph MakeSkewedGraph()
{
    TestGraph graph(12, true);
    for (Node destination = 1; destination < 12; ++destination)
        graph.AddEdge(0, destination);
    for (Node source = 1; source + 1 < 12; ++source)
        graph.AddEdge(source, source + 1);
    graph.AddEdge(11, 0);
    return graph;
}

void TestExactDirectedShards()
{
    const TestGraph graph = MakeSkewedGraph();
    auto partitioned = PartitionedGraph<Node>::Build(
        graph, 3, GraphPartitionBalance::kTotalEdges);
    partitioned.VerifyExact(graph);

    Require(partitioned.num_partitions() == 3, "expected three shards");
    Require(
        partitioned.num_edges_directed() == graph.num_edges_directed(),
        "edge count changed during partitioning");
    Require(
        partitioned.partition(0).vertex_begin == 0,
        "first shard must begin at vertex zero");
    Require(
        partitioned.partition(2).vertex_end == graph.num_nodes(),
        "last shard must end at the graph vertex count");
    Require(
        partitioned.total_ghosts() > 0,
        "cross-shard graph must produce ghost metadata");

    for (const auto &shard : partitioned.partitions())
    {
        for (const Node local_id : shard.out_neighbors)
        {
            Require(
                local_id >= 0 &&
                    static_cast<std::size_t>(local_id) <
                        shard.vertex_count() + shard.ghost_count(),
                "outgoing neighbor is not encoded as a local slot");
        }
        for (const Node local_id : shard.in_neighbors)
        {
            Require(
                local_id >= 0 &&
                    static_cast<std::size_t>(local_id) <
                        shard.vertex_count() + shard.ghost_count(),
                "incoming neighbor is not encoded as a local slot");
        }
        for (std::size_t index = 0;
             index < shard.ghost_count(); ++index)
        {
            Require(
                partitioned.owner(shard.ghost_global(index)) ==
                    shard.ghost_owner(index),
                "ghost owner lookup mismatch");
            Require(
                shard.ghost_owner(index) != shard.id,
                "locally owned vertex recorded as a ghost");
            Require(
                shard.global_vertex_from_slot(
                    shard.ghost_local_id(index)) ==
                    shard.ghost_global(index),
                "ghost local slot does not map back to its global vertex");
        }
    }
}

void TestDeterministicBuild()
{
    const TestGraph graph = MakeSkewedGraph();
#ifdef _OPENMP
    const int original_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    auto first = PartitionedGraph<Node>::Build(
        graph, 4, GraphPartitionBalance::kTotalEdges);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    auto second = PartitionedGraph<Node>::Build(
        graph, 4, GraphPartitionBalance::kTotalEdges);
#ifdef _OPENMP
    omp_set_num_threads(original_threads);
#endif

    Require(
        first.num_partitions() == second.num_partitions(),
        "deterministic build changed shard count");
    for (std::size_t index = 0;
         index < first.num_partitions(); ++index)
    {
        const auto &lhs = first.partition(index);
        const auto &rhs = second.partition(index);
        Require(
            lhs.vertex_begin == rhs.vertex_begin &&
                lhs.vertex_end == rhs.vertex_end,
            "deterministic build changed ownership cuts");
        Require(
            lhs.out_offsets == rhs.out_offsets &&
                lhs.out_neighbors == rhs.out_neighbors &&
                lhs.in_offsets == rhs.in_offsets &&
                lhs.in_neighbors == rhs.in_neighbors &&
                lhs.ghost_globals == rhs.ghost_globals &&
                lhs.ghost_owners == rhs.ghost_owners,
            "deterministic build changed shard contents");
    }
}

void TestEdgeBalanceImprovesSkew()
{
    const TestGraph graph = MakeSkewedGraph();
    auto vertex_balanced = PartitionedGraph<Node>::Build(
        graph, 3, GraphPartitionBalance::kVertices);
    auto edge_balanced = PartitionedGraph<Node>::Build(
        graph, 3, GraphPartitionBalance::kTotalEdges);

    const auto actual_edge_imbalance =
        [](const PartitionedGraph<Node> &partitioned)
        {
            std::uint64_t total = 0;
            std::uint64_t maximum = 0;
            for (const auto &shard : partitioned.partitions())
            {
                const std::uint64_t load =
                    shard.out_neighbors.size() +
                    shard.in_neighbors.size();
                total += load;
                maximum = std::max(maximum, load);
            }
            return static_cast<double>(maximum) /
                   (static_cast<double>(total) /
                    partitioned.num_partitions());
        };
    Require(
        actual_edge_imbalance(edge_balanced) <
            actual_edge_imbalance(vertex_balanced),
        "edge-balanced cuts did not improve the skewed graph");
}

void TestUndirectedStorageIsNotDuplicated()
{
    TestGraph graph(6, false);
    graph.AddEdge(0, 1);
    graph.AddEdge(1, 2);
    graph.AddEdge(2, 3);
    graph.AddEdge(3, 4);
    graph.AddEdge(4, 5);

    auto partitioned = PartitionedGraph<Node>::Build(
        graph, 2, GraphPartitionBalance::kTotalEdges);
    partitioned.VerifyExact(graph);
    for (const auto &shard : partitioned.partitions())
    {
        Require(shard.symmetric, "undirected shard lost symmetry metadata");
        Require(
            shard.in_offsets.empty() && shard.in_neighbors.empty(),
            "undirected shard duplicated its incoming CSR");
    }
}

void TestDegenerateInputs()
{
    TestGraph isolated(5, true);
    auto partitioned = PartitionedGraph<Node>::Build(
        isolated, 20, GraphPartitionBalance::kTotalEdges);
    partitioned.VerifyExact(isolated);
    Require(
        partitioned.num_partitions() == 5,
        "partition count must clamp to the vertex count");
    for (const auto &shard : partitioned.partitions())
        Require(shard.vertex_count() == 1, "isolated shard must own one vertex");

    TestGraph empty(0, true);
    auto empty_partitioned = PartitionedGraph<Node>::Build(
        empty, 4, GraphPartitionBalance::kTotalEdges);
    empty_partitioned.VerifyExact(empty);
    Require(
        empty_partitioned.num_partitions() == 0,
        "empty graph must not create empty shards");

    bool rejected_zero = false;
    try
    {
        auto invalid = PartitionedGraph<Node>::Build(
            isolated, 0, GraphPartitionBalance::kTotalEdges);
        (void)invalid;
    }
    catch (const std::invalid_argument &)
    {
        rejected_zero = true;
    }
    Require(rejected_zero, "zero partition count was not rejected");
}

} // namespace

int main()
{
    try
    {
        TestExactDirectedShards();
        TestDeterministicBuild();
        TestEdgeBalanceImprovesSkew();
        TestUndirectedStorageIsNotDuplicated();
        TestDegenerateInputs();
    }
    catch (const std::exception &error)
    {
        std::cerr << "graph partition test failed: "
                  << error.what() << std::endl;
        return 1;
    }
    std::cout << "graph partition tests passed" << std::endl;
    return 0;
}
