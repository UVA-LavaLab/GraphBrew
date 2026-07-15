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

template <typename Fn>
bool RejectsInvalid(Fn &&fn)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (const std::out_of_range &)
    {
        return true;
    }
    return false;
}

// Write a copy of `manifest` (mutated by `mutate`) as a sibling manifest inside
// the package directory so relative artifact paths still resolve, then return
// its path.
template <typename Mutate>
std::filesystem::path WriteTamperedManifest(
    const std::filesystem::path &package,
    const nlohmann::json &manifest,
    const std::string &name,
    Mutate &&mutate)
{
    nlohmann::json copy = manifest;
    mutate(copy);
    const std::filesystem::path path = package / name;
    std::ofstream file(path);
    file << copy.dump(2) << '\n';
    return path;
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

        // Lightweight single-shard load API used by Blox workers.
        const auto header =
            graphbrew::partition::LoadShardManifestHeader<Node, Offset>(
                replaced_manifest, true);
        Require(header.nodes == 6, "header node count mismatch");
        Require(header.partition_count == 3, "header partition count mismatch");
        Require(header.graph_id == "unit-graph", "header graph id mismatch");
        Require(header.policy_name == "Original", "header policy name mismatch");
        Require(header.policy_id == 0, "header policy id mismatch");
        Require(
            header.policy_options.empty(),
            "header policy options should be empty");
        Require(header.balance == "total", "header balance mismatch");
        Require(header.directed, "header directed flag mismatch");
        Require(
            header.directed_edges == graph.num_edges_directed(),
            "header directed_edges mismatch");
        Require(!header.identity.empty(), "header identity is empty");
        Require(
            header.ownership.size() == 3 &&
                header.ownership.front().first == 0 &&
                header.ownership.back().second == 6,
            "header ownership does not cover every vertex");
        Require(
            header.mapping_loaded &&
                header.internal_to_source.size() == 6,
            "header mapping was not loaded");

        // Each shard loads on its own and matches the manifest's cached scalars.
        std::uint64_t summed_out = 0;
        for (std::size_t id = 0; id < header.partition_count; ++id)
        {
            const auto shard =
                graphbrew::partition::LoadShardPackageShard<Node, Offset>(
                    replaced_manifest, id);
            const auto &json_shard = replaced.at("shards").at(id);
            Require(
                shard.owned_begin ==
                    json_shard.at("owned_begin").get<std::size_t>() &&
                shard.owned_end ==
                    json_shard.at("owned_end").get<std::size_t>(),
                "single-shard ownership mismatch");
            Require(
                shard.storage_bytes ==
                    json_shard.at("storage_bytes").get<std::uint64_t>() &&
                shard.balance_weight ==
                    json_shard.at("balance_weight").get<std::uint64_t>() &&
                shard.remote_out_edges ==
                    json_shard.at("remote_out_edges").get<std::uint64_t>(),
                "single-shard scalar metadata mismatch");
            Require(
                shard.out_offsets.size() == shard.owned_count() + 1,
                "single-shard out CSR size mismatch");
            summed_out += shard.out_neighbors.size();
        }
        Require(
            summed_out ==
                static_cast<std::uint64_t>(graph.num_edges_directed()),
            "single-shard edge totals mismatch");
        Require(
            RejectsInvalid([&]() {
                const auto ignored =
                    graphbrew::partition::LoadShardPackageShard<Node, Offset>(
                        replaced_manifest, 3);
                (void)ignored;
            }),
            "out-of-range shard id was not rejected");

        // Tightened header validation rejects tampered consumer-facing fields.
        const auto expect_reject =
            [&](const std::string &name, auto mutate) {
                const std::filesystem::path path =
                    WriteTamperedManifest(package, replaced, name, mutate);
                Require(
                    RejectsInvalid([&]() {
                        const auto ignored =
                            graphbrew::partition::ValidateShardPackage<
                                Node, Offset>(path);
                        (void)ignored;
                    }),
                    ("tampered manifest accepted: " + name).c_str());
            };
        expect_reject("bad-graph-id.json", [](nlohmann::json &m) {
            m["graph"]["id"] = "";
        });
        expect_reject("bad-identity.json", [](nlohmann::json &m) {
            m["graph"]["identity"] = "0000000000000000";
        });
        expect_reject("bad-directed.json", [](nlohmann::json &m) {
            m["graph"]["directed"] = false;
        });
        expect_reject("bad-directed-edges.json", [](nlohmann::json &m) {
            m["graph"]["directed_edges"] =
                m["graph"]["directed_edges"].get<std::int64_t>() + 1;
        });
        expect_reject("bad-policy-name.json", [](nlohmann::json &m) {
            m["policy"]["name"] = "";
        });
        expect_reject("bad-policy-id.json", [](nlohmann::json &m) {
            m["policy"]["id"] = "not-an-int";
        });
        expect_reject("bad-policy-options.json", [](nlohmann::json &m) {
            m["policy"]["options"] = "not-an-array";
        });

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
