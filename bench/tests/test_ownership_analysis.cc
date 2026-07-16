#include <cstdint>
#include <stdexcept>
#include <vector>

#include "graphbrew/partition/ownership_analysis.h"

namespace
{

using Node = std::int32_t;

class TestGraph
{
public:
    explicit TestGraph(std::size_t nodes)
        : out_(nodes), in_(nodes)
    {
    }

    void AddEdge(Node source, Node target)
    {
        out_.at(source).push_back(target);
        in_.at(target).push_back(source);
    }

    std::int64_t num_nodes() const { return out_.size(); }
    std::int64_t num_edges_directed() const
    {
        std::int64_t total = 0;
        for (const auto &neighbors : out_)
            total += neighbors.size();
        return total;
    }
    bool directed() const { return true; }
    std::int64_t out_degree(Node vertex) const
    {
        return out_.at(vertex).size();
    }
    std::int64_t in_degree(Node vertex) const
    {
        return in_.at(vertex).size();
    }
    const std::vector<Node> &out_neigh(Node vertex) const
    {
        return out_.at(vertex);
    }
    const std::vector<Node> &in_neigh(Node vertex) const
    {
        return in_.at(vertex);
    }

private:
    std::vector<std::vector<Node>> out_;
    std::vector<std::vector<Node>> in_;
};

} // namespace

int main()
{
    auto Require = [](bool condition, const char *message) {
        if (!condition)
            throw std::runtime_error(message);
    };
    TestGraph graph(6);
    graph.AddEdge(0, 1);
    graph.AddEdge(0, 2);
    graph.AddEdge(1, 3);
    graph.AddEdge(2, 3);
    graph.AddEdge(3, 4);
    graph.AddEdge(4, 5);
    graph.AddEdge(5, 0);

    const std::vector<Node> identity{0, 1, 2, 3, 4, 5};
    const auto contiguous_owners =
        graphbrew::partition::BuildContiguousOwners<Node>(
            graph,
            identity,
            2,
            GraphPartitionBalance::kTotalEdges);
    const auto contiguous =
        graphbrew::partition::EvaluateOwnership<Node>(
            graph,
            contiguous_owners,
            GraphPartitionBalance::kTotalEdges);
    const auto materialized = PartitionedGraph<Node>::Build(
        graph, 2, GraphPartitionBalance::kTotalEdges);
    Require(
        contiguous.total_ghost_slots == materialized.total_ghosts(),
        "contiguous ghost count mismatch");
    Require(
        contiguous.total_storage_bytes ==
            materialized.total_storage_bytes(),
        "contiguous storage mismatch");
    Require(
        contiguous.max_storage_bytes ==
            materialized.max_shard_storage_bytes(),
        "contiguous max storage mismatch");
    Require(
        contiguous.remote_out_fraction ==
            materialized.remote_out_edge_fraction(),
        "contiguous remote-out mismatch");
    Require(
        contiguous.remote_in_fraction ==
            materialized.remote_in_edge_fraction(),
        "contiguous remote-in mismatch");

    const std::vector<std::uint32_t> membership{0, 0, 0, 1, 1, 1};
    const auto first =
        graphbrew::partition::BuildCommunityOwners<Node>(
            graph,
            membership,
            2,
            GraphPartitionBalance::kTotalEdges);
    const auto second =
        graphbrew::partition::BuildCommunityOwners<Node>(
            graph,
            membership,
            2,
            GraphPartitionBalance::kTotalEdges);
    Require(first == second, "community assignment is not deterministic");
    const auto community =
        graphbrew::partition::EvaluateOwnership<Node>(
            graph,
            first.first,
            GraphPartitionBalance::kTotalEdges,
            2,
            true);
    Require(community.partition_count == 2, "partition count mismatch");
    Require(
        community.total_remote_out == community.total_remote_in,
        "community remote totals mismatch");
    Require(
        community.total_ownership_metadata_bytes ==
            graph.num_nodes() * sizeof(Node),
        "noncontiguous ownership metadata mismatch");
    Require(
        !community.owner_fingerprint.empty(),
        "owner fingerprint is empty");

    const auto clamped =
        graphbrew::partition::BuildContiguousOwners<Node>(
            graph,
            identity,
            16,
            GraphPartitionBalance::kTotalEdges);
    const auto clamped_metrics =
        graphbrew::partition::EvaluateOwnership<Node>(
            graph,
            clamped,
            GraphPartitionBalance::kTotalEdges,
            6);
    Require(
        clamped_metrics.partition_count == 6,
        "partition count was not clamped to vertex count");
    return 0;
}
