#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "partition/compact_csr.h"
#include "partition/diagnostics.h"
#include "partition/shard_manifest.h"

namespace
{

using Node = std::int32_t;
using Offset = std::uint64_t;

class TestGraph
{
public:
    explicit TestGraph(std::size_t nodes)
        : out_(nodes), in_(nodes), org_ids_(nodes)
    {
        for (std::size_t node = 0; node < nodes; ++node)
            org_ids_[node] = static_cast<Node>(node);
    }

    void AddEdge(Node source, Node destination)
    {
        out_.at(static_cast<std::size_t>(source)).push_back(destination);
        in_.at(static_cast<std::size_t>(destination)).push_back(source);
    }

    std::int64_t num_nodes() const
    {
        return static_cast<std::int64_t>(out_.size());
    }

    std::int64_t num_edges_directed() const
    {
        std::int64_t edges = 0;
        for (const auto &neighbors : out_)
            edges += static_cast<std::int64_t>(neighbors.size());
        return edges;
    }

    bool directed() const
    {
        return true;
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
        return in_.at(static_cast<std::size_t>(vertex));
    }

    Node *get_org_ids() const
    {
        return const_cast<Node *>(org_ids_.data());
    }

private:
    std::vector<std::vector<Node>> out_;
    std::vector<std::vector<Node>> in_;
    std::vector<Node> org_ids_;
};

std::filesystem::path TemporaryRoot()
{
    const auto nonce =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("graphbrew-shard-manifest-" + std::to_string(nonce));
}

TestGraph MakeGraph()
{
    TestGraph graph(6);
    graph.AddEdge(0, 1);
    graph.AddEdge(0, 2);
    graph.AddEdge(1, 3);
    graph.AddEdge(2, 3);
    graph.AddEdge(3, 4);
    graph.AddEdge(4, 5);
    graph.AddEdge(5, 0);
    return graph;
}

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void CorruptFirstByte(const std::filesystem::path &path)
{
    std::fstream file(
        path,
        std::ios::binary | std::ios::in | std::ios::out);
    Require(static_cast<bool>(file), "cannot open artifact for corruption");
    char value = 0;
    file.read(&value, 1);
    Require(file.gcount() == 1, "artifact is empty");
    value ^= 0x5a;
    file.seekp(0);
    file.write(&value, 1);
    Require(static_cast<bool>(file), "artifact corruption write failed");
}

} // namespace

int main()
{
    const std::filesystem::path root = TemporaryRoot();
    const std::filesystem::path package = root / "package";
    try
    {
        const TestGraph graph = MakeGraph();
        const auto mapping =
            graphbrew::partition::BuildOriginalIdMapping(graph);
        auto partitioned = PartitionedGraph<Node>::Build(
            graph, 2, GraphPartitionBalance::kTotalEdges);
        graphbrew::partition::ShardPackageMetadata metadata;
        metadata.graph_id = "unit-graph";
        metadata.policy_name = "Original";
        metadata.policy_id = 0;

        const auto manifest_path =
            graphbrew::partition::WriteShardPackage(
                package,
                graph,
                mapping,
                partitioned,
                metadata);
        const auto manifest =
            graphbrew::partition::ValidateShardPackage<Node, Offset>(
                manifest_path);
        Require(
            manifest.at("schema") == "graph.shard.v1",
            "manifest schema mismatch");
        Require(
            manifest.at("shards").size() == 2,
            "manifest shard count mismatch");
        Require(
            manifest.at("mapping")
                .at("fingerprint")
                .get<std::string>() == mapping.fingerprint,
            "manifest mapping fingerprint mismatch");

        auto repartitioned = PartitionedGraph<Node>::Build(
            graph, 3, GraphPartitionBalance::kTotalEdges);
        const auto replaced_manifest =
            graphbrew::partition::WriteShardPackage(
                package,
                graph,
                mapping,
                repartitioned,
                metadata);
        const auto replaced =
            graphbrew::partition::ValidateShardPackage<Node, Offset>(
                replaced_manifest);
        Require(
            replaced.at("shards").size() == 3,
            "replacement package shard count mismatch");
        Require(
            std::filesystem::exists(
                package / "shards/0002/out_offsets.bin"),
            "replacement package lacks the third shard");

        auto unsafe_manifest = replaced;
        unsafe_manifest["mapping"]["internal_to_source"]["path"] =
            "../outside.bin";
        const std::filesystem::path unsafe_path =
            package / "unsafe-manifest.json";
        {
            std::ofstream unsafe_file(unsafe_path);
            unsafe_file << unsafe_manifest.dump(2) << '\n';
        }
        bool rejected_traversal = false;
        try
        {
            const auto ignored =
                graphbrew::partition::ValidateShardPackage<Node, Offset>(
                    unsafe_path);
            (void)ignored;
        }
        catch (const std::invalid_argument &)
        {
            rejected_traversal = true;
        }
        Require(
            rejected_traversal,
            "parent-traversing shard artifact was not rejected");

        const auto &out_artifact =
            replaced.at("shards")
                .at(0)
                .at("arrays")
                .at("out_offsets");
        const std::filesystem::path out_path =
            package /
            out_artifact.at("path").get<std::string>();
        CorruptFirstByte(out_path);
        bool rejected_corruption = false;
        try
        {
            const auto ignored =
                graphbrew::partition::ValidateShardPackage<Node, Offset>(
                    replaced_manifest);
            (void)ignored;
        }
        catch (const std::invalid_argument &)
        {
            rejected_corruption = true;
        }
        Require(
            rejected_corruption,
            "corrupted shard artifact was not rejected");
    }
    catch (const std::exception &error)
    {
        std::cerr << "shard manifest test failed: "
                  << error.what() << std::endl;
        std::filesystem::remove_all(root);
        return 1;
    }
    std::filesystem::remove_all(root);
    std::cout << "shard manifest tests passed" << std::endl;
    return 0;
}
