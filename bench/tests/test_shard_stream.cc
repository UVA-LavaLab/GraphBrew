#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph.h"
#include "reader.h"
#include "writer.h"

#include "partition/compact_csr.h"
#include "partition/diagnostics.h"
#include "partition/sg_mmap_view.h"
#include "partition/shard_manifest.h"

namespace
{

using Node = std::int32_t;
using Offset = std::uint64_t;
using View = graphbrew::partition::SerializedGraphView<Node, std::int64_t>;

using GapGraph = CSRGraph<Node, Node>;
using GapWriter = WriterBase<Node, Node>;
using GapReader = Reader<Node, Node, Node>;

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::filesystem::path TemporaryRoot()
{
    const auto nonce =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("graphbrew-shard-stream-" + std::to_string(nonce));
}

// Build a directed CSRGraph from explicit outgoing/incoming adjacency. The CSR
// index/neighbor arrays are heap allocated exactly as gapbs expects so the
// CSRGraph destructor can release them.
GapGraph BuildDirected(
    const std::vector<std::vector<Node>> &out_adj,
    const std::vector<std::vector<Node>> &in_adj)
{
    const std::int64_t n = static_cast<std::int64_t>(out_adj.size());
    std::int64_t m = 0;
    for (const auto &row : out_adj)
        m += static_cast<std::int64_t>(row.size());

    Node *out_neighs = new Node[static_cast<std::size_t>(m)];
    Node **out_index = new Node *[static_cast<std::size_t>(n + 1)];
    std::int64_t pos = 0;
    for (std::int64_t v = 0; v < n; ++v)
    {
        out_index[v] = out_neighs + pos;
        for (const Node dst : out_adj[static_cast<std::size_t>(v)])
            out_neighs[pos++] = dst;
    }
    out_index[n] = out_neighs + pos;

    Node *in_neighs = new Node[static_cast<std::size_t>(m)];
    Node **in_index = new Node *[static_cast<std::size_t>(n + 1)];
    pos = 0;
    for (std::int64_t v = 0; v < n; ++v)
    {
        in_index[v] = in_neighs + pos;
        for (const Node src : in_adj[static_cast<std::size_t>(v)])
            in_neighs[pos++] = src;
    }
    in_index[n] = in_neighs + pos;

    return GapGraph(n, out_index, out_neighs, in_index, in_neighs);
}

// Build a symmetric CSRGraph from a single adjacency (each edge listed in both
// directions). Uses the shared-index constructor, matching an undirected `.sg`.
GapGraph BuildSymmetric(const std::vector<std::vector<Node>> &adj)
{
    const std::int64_t n = static_cast<std::int64_t>(adj.size());
    std::int64_t m = 0;
    for (const auto &row : adj)
        m += static_cast<std::int64_t>(row.size());

    Node *neighs = new Node[static_cast<std::size_t>(m)];
    Node **index = new Node *[static_cast<std::size_t>(n + 1)];
    std::int64_t pos = 0;
    for (std::int64_t v = 0; v < n; ++v)
    {
        index[v] = neighs + pos;
        for (const Node dst : adj[static_cast<std::size_t>(v)])
            neighs[pos++] = dst;
    }
    index[n] = neighs + pos;

    return GapGraph(n, index, neighs);
}

void WriteSg(GapGraph &graph, const std::filesystem::path &path)
{
    GapWriter writer(graph);
    writer.WriteGraph(path.string(), true);
}

// Assert that the mmap view exposes exactly the same topology as the graph the
// gapbs Reader materialised from the same `.sg` bytes.
void CheckViewMatchesReader(const View &view, const GapGraph &graph)
{
    Require(view.num_nodes() == graph.num_nodes(),
            "view/reader num_nodes mismatch");
    Require(view.num_edges_directed() == graph.num_edges_directed(),
            "view/reader num_edges_directed mismatch");
    Require(view.num_edges() == graph.num_edges(),
            "view/reader num_edges mismatch");
    Require(view.directed() == graph.directed(),
            "view/reader directed mismatch");

    for (Node v = 0; v < static_cast<Node>(view.num_nodes()); ++v)
    {
        Require(view.out_degree(v) == graph.out_degree(v),
                "view/reader out_degree mismatch");
        Require(view.in_degree(v) == graph.in_degree(v),
                "view/reader in_degree mismatch");

        auto view_out = view.out_neigh(v);
        auto it = view_out.begin();
        for (const Node neighbor : graph.out_neigh(v))
        {
            Require(it != view_out.end(),
                    "view out_neigh shorter than reader");
            Require(*it == neighbor, "view/reader out_neigh mismatch");
            ++it;
        }
        Require(it == view_out.end(),
                "view out_neigh longer than reader");

        auto view_in = view.in_neigh(v);
        auto in_it = view_in.begin();
        for (const Node neighbor : graph.in_neigh(v))
        {
            Require(in_it != view_in.end(),
                    "view in_neigh shorter than reader");
            Require(*in_it == neighbor, "view/reader in_neigh mismatch");
            ++in_it;
        }
        Require(in_it == view_in.end(),
                "view in_neigh longer than reader");
    }

    const Node *view_ids = view.get_org_ids();
    const Node *reader_ids = graph.get_org_ids();
    for (std::int64_t v = 0; v < view.num_nodes(); ++v)
        Require(view_ids[v] == reader_ids[v], "view/reader org_ids mismatch");
}

bool FilesEqual(
    const std::filesystem::path &a,
    const std::filesystem::path &b)
{
    std::ifstream fa(a, std::ios::binary);
    std::ifstream fb(b, std::ios::binary);
    if (!fa || !fb)
        return false;
    std::istreambuf_iterator<char> begin_a(fa), end;
    std::istreambuf_iterator<char> begin_b(fb);
    return std::equal(begin_a, end, begin_b) &&
           std::istreambuf_iterator<char>(fb) == end;
}

// Recursively assert two package directories are byte-for-byte identical.
void RequireIdenticalTrees(
    const std::filesystem::path &a,
    const std::filesystem::path &b)
{
    std::vector<std::filesystem::path> a_files;
    std::vector<std::filesystem::path> b_files;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(a))
        if (entry.is_regular_file())
            a_files.push_back(
                std::filesystem::relative(entry.path(), a));
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(b))
        if (entry.is_regular_file())
            b_files.push_back(
                std::filesystem::relative(entry.path(), b));

    std::sort(a_files.begin(), a_files.end());
    std::sort(b_files.begin(), b_files.end());
    Require(a_files == b_files,
            "streamed and legacy packages have different file sets");
    for (const auto &relative : a_files)
        Require(FilesEqual(a / relative, b / relative),
                "streamed and legacy package files differ byte-wise");
}

graphbrew::partition::ShardPackageMetadata MakeMetadata()
{
    graphbrew::partition::ShardPackageMetadata metadata;
    metadata.graph_id = "stream-graph";
    metadata.policy_name = "Original";
    metadata.policy_id = 0;
    return metadata;
}

// Legacy path (in-memory PartitionedGraph::Build + WriteShardPackage) and
// streaming path must produce byte-identical packages, and streaming must keep
// only one shard resident.
void CheckStreamingByteIdentical(
    const View &view,
    std::size_t partitions,
    GraphPartitionBalance balance,
    const std::filesystem::path &legacy_dir,
    const std::filesystem::path &stream_dir)
{
    const auto mapping =
        graphbrew::partition::BuildOriginalIdMapping(view);
    const auto metadata = MakeMetadata();

    auto partitioned = PartitionedGraph<Node>::Build(
        view, partitions, balance);
    const auto legacy_manifest =
        graphbrew::partition::WriteShardPackage(
            legacy_dir, view, mapping, partitioned, metadata);

    graphbrew::partition::ShardStreamStats stats;
    const auto stream_manifest =
        graphbrew::partition::StreamShardPackage(
            stream_dir, view, mapping, partitions, balance,
            metadata, &stats);

    const auto legacy_json =
        graphbrew::partition::ValidateShardPackage<Node, Offset>(
            legacy_manifest);
    const auto stream_json =
        graphbrew::partition::ValidateShardPackage<Node, Offset>(
            stream_manifest);
    Require(legacy_json == stream_json,
            "streamed manifest JSON differs from legacy");

    RequireIdenticalTrees(legacy_dir, stream_dir);

    const std::size_t expected_shards =
        std::min(partitions, static_cast<std::size_t>(view.num_nodes()));
    Require(stats.max_live_shards <= 1,
            "streaming exporter kept more than one shard live");
    if (expected_shards > 0)
        Require(stats.max_live_shards == 1,
                "streaming exporter never reported a live shard");
    if (expected_shards > 1)
        Require(stats.max_live_shard_bytes < stats.total_shard_bytes,
                "streaming peak shard was not smaller than the whole package");
}

void RunGraph(
    const std::filesystem::path &root,
    const std::filesystem::path &sg_path,
    GapGraph &graph,
    const char *label)
{
    WriteSg(graph, sg_path);
    // WriteGraph appends the .sg suffix itself.
    const std::filesystem::path actual_sg =
        sg_path.string() + ".sg";

    View view(actual_sg.string());
    {
        GapReader reader(actual_sg.string());
        GapGraph loaded = reader.ReadSerializedGraph();
        CheckViewMatchesReader(view, loaded);
    }

    const std::size_t nodes =
        static_cast<std::size_t>(view.num_nodes());
    const std::vector<std::size_t> partition_counts = {
        1, 2, 3, nodes};
    const std::vector<GraphPartitionBalance> balances = {
        GraphPartitionBalance::kVertices,
        GraphPartitionBalance::kOutgoingEdges,
        GraphPartitionBalance::kTotalEdges};

    for (const std::size_t partitions : partition_counts)
    {
        if (partitions == 0)
            continue;
        for (const GraphPartitionBalance balance : balances)
        {
            const std::string tag =
                std::string(label) + "-P" +
                std::to_string(partitions) + "-B" +
                GraphPartitionBalanceName(balance);
            const std::filesystem::path legacy_dir =
                root / (tag + "-legacy");
            const std::filesystem::path stream_dir =
                root / (tag + "-stream");
            CheckStreamingByteIdentical(
                view, partitions, balance, legacy_dir, stream_dir);
        }
    }
}

void CorruptFirstByte(const std::filesystem::path &path)
{
    std::fstream file(
        path, std::ios::binary | std::ios::in | std::ios::out);
    Require(static_cast<bool>(file), "cannot open artifact for corruption");
    char value = 0;
    file.read(&value, 1);
    Require(file.gcount() == 1, "artifact is empty");
    value ^= 0x5a;
    file.seekp(0);
    file.write(&value, 1);
    Require(static_cast<bool>(file), "artifact corruption write failed");
}

// Streamed packages must support in-place replacement and reject corruption,
// mirroring the guarantees the in-memory writer already provides.
void CheckReplacementAndCorruption(
    const std::filesystem::path &root,
    const View &view)
{
    const auto mapping =
        graphbrew::partition::BuildOriginalIdMapping(view);
    const auto metadata = MakeMetadata();
    const std::filesystem::path package = root / "streamed-package";

    const auto first_manifest =
        graphbrew::partition::StreamShardPackage(
            package, view, mapping, 2,
            GraphPartitionBalance::kTotalEdges, metadata);
    const auto first_json =
        graphbrew::partition::ValidateShardPackage<Node, Offset>(
            first_manifest);
    Require(first_json.at("shards").size() == 2,
            "streamed package shard count mismatch");

    const auto replaced_manifest =
        graphbrew::partition::StreamShardPackage(
            package, view, mapping, 3,
            GraphPartitionBalance::kTotalEdges, metadata);
    const auto replaced_json =
        graphbrew::partition::ValidateShardPackage<Node, Offset>(
            replaced_manifest);
    Require(replaced_json.at("shards").size() == 3,
            "streamed replacement shard count mismatch");
    Require(
        std::filesystem::exists(package / "shards/0002/out_offsets.bin"),
        "streamed replacement lacks the third shard");

    const auto &artifact =
        replaced_json.at("shards").at(0).at("arrays").at("out_offsets");
    const std::filesystem::path corrupt_path =
        package / artifact.at("path").get<std::string>();
    CorruptFirstByte(corrupt_path);
    bool rejected = false;
    try
    {
        const auto ignored =
            graphbrew::partition::ValidateShardPackage<Node, Offset>(
                replaced_manifest);
        (void)ignored;
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "corrupted streamed shard artifact was not rejected");
}

} // namespace

int main()
{
    const std::filesystem::path root = TemporaryRoot();
    std::filesystem::create_directories(root);
    try
    {
        // Directed graph.
        std::vector<std::vector<Node>> out_adj = {
            {1, 2}, {3}, {3, 4}, {4}, {5}, {0}};
        std::vector<std::vector<Node>> in_adj(out_adj.size());
        for (Node u = 0; u < static_cast<Node>(out_adj.size()); ++u)
            for (const Node v : out_adj[static_cast<std::size_t>(u)])
                in_adj[static_cast<std::size_t>(v)].push_back(u);
        GapGraph directed = BuildDirected(out_adj, in_adj);
        RunGraph(root, root / "directed", directed, "directed");

        // Symmetric graph (each edge in both directions).
        std::vector<std::vector<Node>> sym_adj(7);
        const std::vector<std::pair<Node, Node>> edges = {
            {0, 1}, {0, 2}, {1, 3}, {2, 3},
            {3, 4}, {4, 5}, {5, 6}, {6, 0}};
        for (const auto &edge : edges)
        {
            sym_adj[static_cast<std::size_t>(edge.first)]
                .push_back(edge.second);
            sym_adj[static_cast<std::size_t>(edge.second)]
                .push_back(edge.first);
        }
        GapGraph symmetric = BuildSymmetric(sym_adj);
        RunGraph(root, root / "symmetric", symmetric, "symmetric");

        // Replacement and corruption behaviour on a streamed package.
        {
            const std::filesystem::path sg_path =
                (root / "symmetric").string() + ".sg";
            View view(sg_path.string());
            CheckReplacementAndCorruption(root, view);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "shard stream test failed: "
                  << error.what() << std::endl;
        std::filesystem::remove_all(root);
        return 1;
    }
    std::filesystem::remove_all(root);
    std::cout << "shard stream tests passed" << std::endl;
    return 0;
}
