#ifndef GRAPHBREW_PARTITION_SHARD_MANIFEST_H_
#define GRAPHBREW_PARTITION_SHARD_MANIFEST_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../external/nlohmann_json.hpp"
#include "partition/compact_csr.h"
#include "partition/diagnostics.h"

namespace graphbrew
{
namespace partition
{

inline constexpr const char *kShardManifestSchema = "graph.shard.v1";

struct ShardPackageMetadata
{
    std::string graph_id;
    std::string policy_name;
    int policy_id = 0;
    std::vector<std::string> policy_options;
};

inline void ValidateRelativeArtifactPath(
    const std::filesystem::path &path)
{
    if (path.empty() || path.is_absolute())
        throw std::invalid_argument(
            "Shard artifact path must be non-empty and relative");
    for (const auto &component : path)
    {
        if (component == "..")
            throw std::invalid_argument(
                "Shard artifact path must not traverse parents");
    }
}

inline bool HostIsLittleEndian()
{
    const std::uint32_t value = 1;
    return *reinterpret_cast<const std::uint8_t *>(&value) == 1;
}

template <typename T>
std::string IntegralTypeName()
{
    static_assert(std::is_integral<T>::value, "array type must be integral");
    std::ostringstream name;
    name << (std::is_signed<T>::value ? 'i' : 'u')
         << (sizeof(T) * 8);
    return name.str();
}

template <typename T>
std::string IntegralRangeFingerprint(const std::vector<T> &values)
{
    OrderedFingerprint fingerprint;
    fingerprint.AddRange(values);
    return fingerprint.Hex();
}

template <typename T>
void WriteIntegralArray(
    const std::filesystem::path &path,
    const std::vector<T> &values)
{
    static_assert(std::is_integral<T>::value, "array type must be integral");
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error(
            "Cannot write shard array: " + path.string());
    if (HostIsLittleEndian())
    {
        if (!values.empty())
        {
            output.write(
                reinterpret_cast<const char *>(values.data()),
                static_cast<std::streamsize>(
                    values.size() * sizeof(T)));
        }
    }
    else
    {
        using Unsigned = typename std::make_unsigned<T>::type;
        for (const T value : values)
        {
            const Unsigned encoded =
                static_cast<Unsigned>(value);
            for (std::size_t byte = 0; byte < sizeof(T); ++byte)
            {
                output.put(static_cast<char>(
                    (encoded >> (byte * 8)) & 0xffu));
            }
        }
    }
    if (!output)
        throw std::runtime_error(
            "Failed writing shard array: " + path.string());
}

template <typename T>
std::vector<T> ReadIntegralArray(
    const std::filesystem::path &path,
    std::size_t count)
{
    static_assert(std::is_integral<T>::value, "array type must be integral");
    std::vector<T> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error(
            "Cannot read shard array: " + path.string());
    if (HostIsLittleEndian())
    {
        if (!values.empty())
        {
            input.read(
                reinterpret_cast<char *>(values.data()),
                static_cast<std::streamsize>(
                    values.size() * sizeof(T)));
        }
    }
    else
    {
        using Unsigned = typename std::make_unsigned<T>::type;
        for (std::size_t index = 0; index < count; ++index)
        {
            Unsigned decoded = 0;
            for (std::size_t byte = 0; byte < sizeof(T); ++byte)
            {
                const int value = input.get();
                if (value == std::char_traits<char>::eof())
                    throw std::runtime_error(
                        "Truncated shard array: " + path.string());
                decoded |= static_cast<Unsigned>(
                    static_cast<std::uint8_t>(value))
                    << (byte * 8);
            }
            values[index] = static_cast<T>(decoded);
        }
    }
    if (!input)
        throw std::runtime_error(
            "Truncated shard array: " + path.string());
    char extra = 0;
    if (input.get(extra))
        throw std::runtime_error(
            "Shard array has trailing bytes: " + path.string());
    return values;
}

template <typename T>
nlohmann::json WriteArrayArtifact(
    const std::filesystem::path &package_root,
    const std::filesystem::path &relative_path,
    const std::vector<T> &values)
{
    ValidateRelativeArtifactPath(relative_path);
    WriteIntegralArray(package_root / relative_path, values);
    return {
        {"path", relative_path.generic_string()},
        {"type", IntegralTypeName<T>()},
        {"count", values.size()},
        {"element_bytes", sizeof(T)},
        {"fingerprint", IntegralRangeFingerprint(values)},
    };
}

inline std::string ShardDirectoryName(std::uint32_t id)
{
    std::ostringstream name;
    name << "shards/" << std::setfill('0') << std::setw(4) << id;
    return name.str();
}

inline void ReplacePackageDirectory(
    const std::filesystem::path &temporary,
    const std::filesystem::path &output)
{
    const auto filename = output.filename().string();
    if (filename.empty() || filename == "." || filename == ".." ||
        output == output.root_path())
    {
        throw std::invalid_argument(
            "Unsafe graph shard output directory");
    }
    const std::filesystem::path parent =
        output.parent_path().empty()
            ? std::filesystem::path(".")
            : output.parent_path();
    const std::filesystem::path backup =
        parent / (filename + ".graph-shard-backup");
    if (std::filesystem::exists(backup))
    {
        if (!std::filesystem::exists(output))
            std::filesystem::rename(backup, output);
        else
            std::filesystem::remove_all(backup);
    }
    const bool had_output = std::filesystem::exists(output);
    if (had_output)
    {
        if (std::filesystem::is_symlink(output))
            throw std::invalid_argument(
                "Graph shard output directory must not be a symlink");
        std::filesystem::rename(output, backup);
    }
    try
    {
        std::filesystem::rename(temporary, output);
    }
    catch (...)
    {
        if (had_output && std::filesystem::exists(backup))
            std::filesystem::rename(backup, output);
        throw;
    }
    std::filesystem::remove_all(backup);
}

// Create the atomic-replacement staging directory for a shard package. Returns
// {normalized output path, temporary staging path} and guarantees the staging
// directory exists and is empty. Shared by the in-memory and streaming writers
// so both stage identically before ReplacePackageDirectory.
inline std::pair<std::filesystem::path, std::filesystem::path>
PrepareShardPackageDirectories(const std::filesystem::path &output)
{
    const std::filesystem::path normalized =
        output.lexically_normal();
    if (normalized.empty() || normalized == normalized.root_path())
        throw std::invalid_argument(
            "Unsafe graph shard output directory");
    const std::filesystem::path parent =
        normalized.parent_path().empty()
            ? std::filesystem::path(".")
            : normalized.parent_path();
    std::filesystem::create_directories(parent);
    const std::filesystem::path temporary =
        parent /
        (normalized.filename().string() + ".graph-shard-tmp");
    std::filesystem::remove_all(temporary);
    std::filesystem::create_directories(temporary);
    return {normalized, temporary};
}

// Build the schema/graph/encoding/policy/mapping portion of a manifest and
// write the mapping sidecar artifacts into `temporary`. Both writers share this
// so their manifests are byte-identical up to the partitioning/shards sections.
template <typename Offset_, typename GraphT>
nlohmann::json BuildShardManifestHeader(
    const std::filesystem::path &temporary,
    const GraphT &graph,
    const OriginalIdMapping<GraphNode<GraphT>> &mapping,
    const ShardPackageMetadata &metadata)
{
    using Node = GraphNode<GraphT>;
    if (metadata.graph_id.empty() || metadata.policy_name.empty())
        throw std::invalid_argument(
            "Shard package metadata requires graph and policy names");
    if (
        mapping.internal_to_source.size() !=
            static_cast<std::size_t>(graph.num_nodes()) ||
        mapping.source_to_internal.size() !=
            static_cast<std::size_t>(graph.num_nodes()))
    {
        throw std::invalid_argument(
            "Shard package source mapping size mismatch");
    }

    nlohmann::json manifest;
    manifest["schema"] = kShardManifestSchema;
    const std::string source_topology_fingerprint =
        SourceTopologyFingerprint(graph, mapping);
    manifest["graph"] = {
        {"id", metadata.graph_id},
        {"nodes", graph.num_nodes()},
        {"directed_edges", graph.num_edges_directed()},
        {"directed", graph.directed()},
        {
            "source_topology_fingerprint",
            source_topology_fingerprint,
        },
        {
            "identity",
            source_topology_fingerprint,
        },
    };
    manifest["encoding"] = {
        {"byte_order", "little"},
        {"node_id_type", IntegralTypeName<Node>()},
        {"offset_type", IntegralTypeName<Offset_>()},
        {"edge_value_type", "none"},
        {
            "local_slot_layout",
            "owned_then_ghost",
        },
        {
            "owned_slot_rule",
            "global_id=owned_begin+slot",
        },
        {
            "ghost_slot_rule",
            "ghost_index=slot-owned_count",
        },
        {
            "symmetric_incoming",
            "alias_outgoing",
        },
        {
            "optional_array_kinds",
            {"out_weights", "in_weights"},
        },
    };
    manifest["policy"] = {
        {"name", metadata.policy_name},
        {"id", metadata.policy_id},
        {"options", metadata.policy_options},
    };
    manifest["mapping"] = {
        {"fingerprint", mapping.fingerprint},
        {
            "internal_to_source",
            WriteArrayArtifact(
                temporary,
                "mapping/internal_to_source.bin",
                mapping.internal_to_source),
        },
        {
            "source_to_internal",
            WriteArrayArtifact(
                temporary,
                "mapping/source_to_internal.bin",
                mapping.source_to_internal),
        },
    };
    return manifest;
}

// Write one shard's six CSR/ghost arrays into `temporary` and return the shard's
// manifest object. Shared by both writers so per-shard manifest bytes and
// on-disk arrays are identical regardless of how the shard was produced.
template <typename PartitionT>
nlohmann::json WriteShardArtifacts(
    const std::filesystem::path &temporary,
    const PartitionT &part)
{
    const std::filesystem::path directory =
        ShardDirectoryName(part.id);
    nlohmann::json arrays;
    arrays["out_offsets"] = WriteArrayArtifact(
        temporary, directory / "out_offsets.bin", part.out_offsets);
    arrays["out_neighbors"] = WriteArrayArtifact(
        temporary, directory / "out_neighbors.bin", part.out_neighbors);
    arrays["in_offsets"] = WriteArrayArtifact(
        temporary, directory / "in_offsets.bin", part.in_offsets);
    arrays["in_neighbors"] = WriteArrayArtifact(
        temporary, directory / "in_neighbors.bin", part.in_neighbors);
    arrays["ghost_globals"] = WriteArrayArtifact(
        temporary, directory / "ghost_globals.bin", part.ghost_globals);
    arrays["ghost_owners"] = WriteArrayArtifact(
        temporary, directory / "ghost_owners.bin", part.ghost_owners);
    return {
        {"id", part.id},
        {"owned_begin", part.vertex_begin},
        {"owned_end", part.vertex_end},
        {"symmetric", part.symmetric},
        {"balance_weight", part.balance_weight},
        {"remote_out_edges", part.remote_out_edges},
        {"remote_in_edges", part.remote_in_edges},
        {"storage_bytes", part.storage_bytes()},
        {"arrays", std::move(arrays)},
    };
}

// Serialize `manifest` into the staging directory and atomically swap it into
// place at `normalized`.
inline std::filesystem::path FinalizeShardPackage(
    const std::filesystem::path &temporary,
    const std::filesystem::path &normalized,
    const nlohmann::json &manifest)
{
    const std::filesystem::path manifest_path =
        temporary / "manifest.json";
    {
        std::ofstream file(manifest_path, std::ios::trunc);
        if (!file)
            throw std::runtime_error(
                "Cannot write graph shard manifest");
        file << manifest.dump(2) << '\n';
        if (!file)
            throw std::runtime_error(
                "Failed writing graph shard manifest");
    }
    ReplacePackageDirectory(temporary, normalized);
    return normalized / "manifest.json";
}

template <typename GraphT, typename PartitionedGraphT>
std::filesystem::path WriteShardPackage(
    const std::filesystem::path &output,
    const GraphT &graph,
    const OriginalIdMapping<GraphNode<GraphT>> &mapping,
    const PartitionedGraphT &partitioned,
    const ShardPackageMetadata &metadata)
{
    using Offset = typename PartitionedGraphT::Offset;
    if (partitioned.num_nodes() != graph.num_nodes())
        throw std::invalid_argument(
            "Shard package graph and partition vertex counts differ");

    std::filesystem::path normalized;
    std::filesystem::path temporary;
    std::tie(normalized, temporary) =
        PrepareShardPackageDirectories(output);

    nlohmann::json manifest =
        BuildShardManifestHeader<Offset>(
            temporary, graph, mapping, metadata);
    manifest["partitioning"] = {
        {"count", partitioned.num_partitions()},
        {"balance", GraphPartitionBalanceName(partitioned.balance())},
        {"shard_fingerprint", CompactShardFingerprint(partitioned)},
        {"ghost_fingerprint", GhostMetadataFingerprint(partitioned)},
        {"remote_out_fraction", partitioned.remote_out_edge_fraction()},
        {"remote_in_fraction", partitioned.remote_in_edge_fraction()},
        {"ghost_count", partitioned.total_ghosts()},
        {"ghost_bytes", partitioned.total_ghost_metadata_bytes()},
        {"ghost_byte_fraction", partitioned.ghost_metadata_fraction()},
        {"total_shard_bytes", partitioned.total_storage_bytes()},
        {"max_shard_bytes", partitioned.max_shard_storage_bytes()},
        {"balance_imbalance", partitioned.max_balance_imbalance()},
        {"storage_imbalance", partitioned.max_shard_storage_imbalance()},
    };
    manifest["shards"] = nlohmann::json::array();

    for (const auto &part : partitioned.partitions())
    {
        manifest["shards"].push_back(
            WriteShardArtifacts(temporary, part));
    }

    return FinalizeShardPackage(temporary, normalized, manifest);
}

// Instrumentation for the streaming exporter's memory profile. A structural
// witness that at most one shard's arrays are resident at any moment.
struct ShardStreamStats
{
    std::size_t max_live_shards = 0;
    std::uint64_t max_live_shard_bytes = 0;
    std::uint64_t total_shard_bytes = 0;
};

// One-shard-at-a-time exporter. Derives the same balanced ownership as
// PartitionedGraph::Build, then materializes, writes and discards each shard in
// turn so peak extra memory is O(N) scratch plus the single largest shard
// instead of every shard at once. The emitted package is byte-identical to
// WriteShardPackage(PartitionedGraph::Build(...)).
template <typename GraphT>
std::filesystem::path StreamShardPackage(
    const std::filesystem::path &output,
    const GraphT &graph,
    const OriginalIdMapping<GraphNode<GraphT>> &mapping,
    std::size_t requested_partitions,
    GraphPartitionBalance balance,
    const ShardPackageMetadata &metadata,
    ShardStreamStats *stats = nullptr)
{
    using Node = GraphNode<GraphT>;
    using Partition = CompactGraphPartition<Node>;
    using Offset = typename Partition::Offset;

    GraphPartitionPlan<Node> plan =
        BuildGraphPartitionPlan<Node>(
            graph, requested_partitions, balance);

    std::filesystem::path normalized;
    std::filesystem::path temporary;
    std::tie(normalized, temporary) =
        PrepareShardPackageDirectories(output);

    nlohmann::json manifest =
        BuildShardManifestHeader<Offset>(
            temporary, graph, mapping, metadata);

    const std::size_t partition_count = plan.partition_count;
    const std::size_t num_nodes =
        static_cast<std::size_t>(plan.num_nodes);
    const std::uint64_t num_edges_directed = plan.num_edges_directed;

    // Running fingerprints reproduced incrementally, one shard at a time, using
    // the exact folds CompactShardFingerprint/GhostMetadataFingerprint use.
    OrderedFingerprint shard_fp;
    shard_fp.Add(1);
    shard_fp.Add(graph.num_nodes());
    shard_fp.Add(graph.num_edges_directed());
    shard_fp.Add(plan.directed ? 1 : 0);
    shard_fp.AddIntegral(balance);
    shard_fp.Add(partition_count);
    OrderedFingerprint ghost_fp;
    ghost_fp.Add(1);
    ghost_fp.Add(partition_count);

    std::vector<std::uint32_t> ghost_stamp(num_nodes, 0);
    std::vector<Node> ghost_slot_by_vertex(num_nodes, 0);

    std::size_t total_ghosts = 0;
    std::uint64_t total_out_edges = 0;
    std::uint64_t total_in_edges = 0;
    std::uint64_t remote_out_edges = 0;
    std::uint64_t remote_in_edges = 0;
    std::uint64_t total_ghost_bytes = 0;
    std::uint64_t total_storage = 0;
    std::uint64_t max_storage = 0;
    std::uint64_t max_weight = 0;
    std::uint64_t total_weight = 0;

    std::size_t live_shards = 0;
    std::size_t peak_live_shards = 0;

    nlohmann::json shards = nlohmann::json::array();
    for (std::size_t id = 0; id < partition_count; ++id)
    {
        Partition part = BuildCompactShard<Node>(
            graph, plan, id, ghost_stamp, ghost_slot_by_vertex);
        ++live_shards;
        peak_live_shards = std::max(peak_live_shards, live_shards);

        AccumulateCompactShardFingerprint(shard_fp, part);
        AccumulateGhostFingerprint(ghost_fp, part);

        const std::uint64_t storage_bytes = part.storage_bytes();
        total_ghosts += part.ghost_count();
        total_out_edges = graphbrew_compact_detail::CheckedAdd(
            total_out_edges, part.out_neighbors.size());
        total_in_edges = graphbrew_compact_detail::CheckedAdd(
            total_in_edges,
            plan.directed
                ? part.in_neighbors.size()
                : part.out_neighbors.size());
        remote_out_edges = graphbrew_compact_detail::CheckedAdd(
            remote_out_edges, part.remote_out_edges);
        remote_in_edges = graphbrew_compact_detail::CheckedAdd(
            remote_in_edges, part.remote_in_edges);
        total_ghost_bytes = graphbrew_compact_detail::CheckedAdd(
            total_ghost_bytes, part.ghost_metadata_bytes());
        total_storage = graphbrew_compact_detail::CheckedAdd(
            total_storage, storage_bytes);
        max_storage = std::max(max_storage, storage_bytes);
        max_weight = std::max(max_weight, part.balance_weight);
        total_weight = graphbrew_compact_detail::CheckedAdd(
            total_weight, part.balance_weight);

        shards.push_back(WriteShardArtifacts(temporary, part));

        // Release this shard's arrays before building the next one.
        part = Partition();
        --live_shards;
    }

    if (total_out_edges != num_edges_directed)
        throw std::logic_error(
            "Streamed outgoing edge count does not match the graph");
    if (total_in_edges != num_edges_directed)
        throw std::logic_error(
            "Streamed incoming edge count does not match the graph");

    const auto edge_fraction =
        [num_edges_directed](std::uint64_t remote) -> double
        {
            if (num_edges_directed == 0)
                return 0.0;
            return static_cast<double>(remote) /
                   static_cast<double>(num_edges_directed);
        };
    const double ghost_byte_fraction =
        total_storage == 0
            ? 0.0
            : static_cast<double>(total_ghost_bytes) /
                  static_cast<double>(total_storage);
    const double balance_imbalance =
        total_weight == 0
            ? 1.0
            : static_cast<double>(max_weight) /
                  (static_cast<double>(total_weight) /
                   static_cast<double>(partition_count));
    const double storage_imbalance =
        (partition_count == 0 || total_storage == 0)
            ? 1.0
            : static_cast<double>(max_storage) /
                  (static_cast<double>(total_storage) /
                   static_cast<double>(partition_count));

    manifest["partitioning"] = {
        {"count", partition_count},
        {"balance", GraphPartitionBalanceName(balance)},
        {"shard_fingerprint", shard_fp.Hex()},
        {"ghost_fingerprint", ghost_fp.Hex()},
        {"remote_out_fraction", edge_fraction(remote_out_edges)},
        {"remote_in_fraction", edge_fraction(remote_in_edges)},
        {"ghost_count", total_ghosts},
        {"ghost_bytes", total_ghost_bytes},
        {"ghost_byte_fraction", ghost_byte_fraction},
        {"total_shard_bytes", total_storage},
        {"max_shard_bytes", max_storage},
        {"balance_imbalance", balance_imbalance},
        {"storage_imbalance", storage_imbalance},
    };
    manifest["shards"] = std::move(shards);

    if (stats != nullptr)
    {
        stats->max_live_shards = peak_live_shards;
        stats->max_live_shard_bytes = max_storage;
        stats->total_shard_bytes = total_storage;
    }

    return FinalizeShardPackage(temporary, normalized, manifest);
}

inline const nlohmann::json &RequireObjectField(
    const nlohmann::json &object,
    const char *name)
{
    if (!object.contains(name) || !object.at(name).is_object())
        throw std::invalid_argument(
            std::string("Missing shard manifest object: ") + name);
    return object.at(name);
}

inline std::filesystem::path ResolveArtifact(
    const std::filesystem::path &root,
    const nlohmann::json &artifact)
{
    const std::filesystem::path relative =
        artifact.at("path").get<std::string>();
    ValidateRelativeArtifactPath(relative);
    const std::filesystem::path candidate =
        (root / relative).lexically_normal();
    if (std::filesystem::is_symlink(candidate))
        throw std::invalid_argument(
            "Shard artifact must not be a symlink");
    const std::filesystem::path canonical_root =
        std::filesystem::weakly_canonical(root);
    const std::filesystem::path resolved =
        std::filesystem::weakly_canonical(candidate);
    const auto mismatch = std::mismatch(
        canonical_root.begin(),
        canonical_root.end(),
        resolved.begin(),
        resolved.end());
    if (mismatch.first != canonical_root.end())
        throw std::invalid_argument(
            "Shard artifact escapes the package directory");
    return resolved;
}

template <typename T>
std::vector<T> ValidateAndReadArrayArtifact(
    const std::filesystem::path &root,
    const nlohmann::json &artifact)
{
    if (
        artifact.at("type").get<std::string>() != IntegralTypeName<T>() ||
        artifact.at("element_bytes").get<std::size_t>() != sizeof(T))
    {
        throw std::invalid_argument(
            "Shard array type does not match the manifest contract");
    }
    const std::size_t count =
        artifact.at("count").get<std::size_t>();
    const std::filesystem::path path =
        ResolveArtifact(root, artifact);
    if (!std::filesystem::is_regular_file(path))
        throw std::invalid_argument(
            "Shard artifact is missing: " + path.string());
    if (
        count >
        std::numeric_limits<std::uintmax_t>::max() / sizeof(T))
    {
        throw std::overflow_error(
            "Shard artifact byte count overflow");
    }
    const std::uintmax_t expected =
        static_cast<std::uintmax_t>(count) * sizeof(T);
    if (std::filesystem::file_size(path) != expected)
        throw std::invalid_argument(
            "Shard artifact byte count mismatch: " + path.string());
    std::vector<T> values = ReadIntegralArray<T>(path, count);
    if (
        IntegralRangeFingerprint(values) !=
        artifact.at("fingerprint").get<std::string>())
    {
        throw std::invalid_argument(
            "Shard artifact fingerprint mismatch: " + path.string());
    }
    return values;
}

inline std::string RequireStringField(
    const nlohmann::json &object,
    const char *name)
{
    if (!object.contains(name) || !object.at(name).is_string())
        throw std::invalid_argument(
            std::string("Shard manifest field must be a string: ") + name);
    return object.at(name).get<std::string>();
}

// Open, parse, and schema-check a manifest file. Shared by the full validator
// and the lightweight single-shard loaders so all entry points reject a
// foreign or malformed schema identically.
inline nlohmann::json ParseShardManifestFile(
    const std::filesystem::path &manifest_path)
{
    std::ifstream file(manifest_path);
    if (!file)
        throw std::invalid_argument(
            "Cannot open graph shard manifest");
    nlohmann::json manifest;
    file >> manifest;
    if (
        !manifest.is_object() ||
        manifest.value("schema", "") != kShardManifestSchema)
    {
        throw std::invalid_argument(
            "Unsupported graph shard manifest schema");
    }
    return manifest;
}

// Validated, scalar-only view of a package that Blox workers can obtain without
// reading or materializing any shard's CSR/ghost arrays. `ownership[i]` is the
// half-open owned-vertex range of shard i, derived from the manifest scalars
// and proven contiguous and covering. Mapping arrays are populated only when
// requested.
template <typename NodeID_>
struct ShardManifestHeader
{
    std::size_t nodes = 0;
    bool directed = false;
    std::int64_t directed_edges = 0;
    std::string graph_id;
    std::string identity;
    std::string policy_name;
    int policy_id = 0;
    std::vector<std::string> policy_options;
    std::string balance;
    std::size_t partition_count = 0;
    std::vector<std::pair<std::size_t, std::size_t>> ownership;
    bool mapping_loaded = false;
    std::vector<NodeID_> internal_to_source;
    std::vector<NodeID_> source_to_internal;
};

// A single validated shard's arrays plus its derived scalars. This is the unit
// a Blox worker loads for the shard it owns.
template <typename NodeID_, typename Offset_>
struct LoadedShard
{
    std::uint32_t id = 0;
    std::size_t owned_begin = 0;
    std::size_t owned_end = 0;
    bool symmetric = false;
    std::vector<Offset_> out_offsets;
    std::vector<NodeID_> out_neighbors;
    std::vector<Offset_> in_offsets;
    std::vector<NodeID_> in_neighbors;
    std::vector<NodeID_> ghost_globals;
    std::vector<std::uint32_t> ghost_owners;
    std::uint64_t remote_out_edges = 0;
    std::uint64_t remote_in_edges = 0;
    std::uint64_t storage_bytes = 0;
    std::uint64_t balance_weight = 0;

    std::size_t owned_count() const
    {
        return owned_end - owned_begin;
    }
};

template <typename OffsetT>
bool ShardOffsetsAreValid(
    const std::vector<OffsetT> &offsets,
    std::size_t neighbors)
{
    if (offsets.empty() || offsets.front() != 0 ||
        offsets.back() != neighbors)
        return false;
    return std::is_sorted(offsets.begin(), offsets.end());
}

// Validate the schema/graph/encoding/policy/partitioning/ownership metadata and
// (optionally) the source mapping without touching any shard arrays. Tightened
// to check graph.id/identity/directed/directed_edges and policy.name/id/options
// because the Blox reader consumes those fields verbatim.
template <typename NodeID_, typename Offset_>
ShardManifestHeader<NodeID_> ValidateShardManifestHeaderJson(
    const nlohmann::json &manifest,
    const std::filesystem::path &root,
    bool load_mapping)
{
    const auto &graph = RequireObjectField(manifest, "graph");
    const auto &encoding = RequireObjectField(manifest, "encoding");
    const auto &policy = RequireObjectField(manifest, "policy");
    const auto &mapping = RequireObjectField(manifest, "mapping");
    const auto &partitioning =
        RequireObjectField(manifest, "partitioning");
    if (
        encoding.at("byte_order").get<std::string>() != "little" ||
        encoding.at("node_id_type").get<std::string>() !=
            IntegralTypeName<NodeID_>() ||
        encoding.at("offset_type").get<std::string>() !=
            IntegralTypeName<Offset_>() ||
        encoding.at("local_slot_layout").get<std::string>() !=
            "owned_then_ghost" ||
        encoding.at("owned_slot_rule").get<std::string>() !=
            "global_id=owned_begin+slot" ||
        encoding.at("ghost_slot_rule").get<std::string>() !=
            "ghost_index=slot-owned_count" ||
        encoding.at("symmetric_incoming").get<std::string>() !=
            "alias_outgoing" ||
        encoding.at("edge_value_type").get<std::string>() !=
            "none")
    {
        throw std::invalid_argument(
            "Graph shard encoding is incompatible");
    }

    ShardManifestHeader<NodeID_> header;

    header.graph_id = RequireStringField(graph, "id");
    if (header.graph_id.empty())
        throw std::invalid_argument(
            "Graph shard manifest graph id is empty");

    if (!graph.contains("directed") || !graph.at("directed").is_boolean())
        throw std::invalid_argument(
            "Graph shard manifest directed flag is missing or not a boolean");
    header.directed = graph.at("directed").get<bool>();

    if (
        !graph.contains("directed_edges") ||
        !graph.at("directed_edges").is_number_integer())
    {
        throw std::invalid_argument(
            "Graph shard manifest directed_edges is missing or not integral");
    }
    const std::int64_t directed_edges =
        graph.at("directed_edges").get<std::int64_t>();
    if (directed_edges < 0)
        throw std::invalid_argument(
            "Graph shard manifest directed_edges is negative");
    header.directed_edges = directed_edges;

    if (!graph.contains("nodes") || !graph.at("nodes").is_number_integer())
        throw std::invalid_argument(
            "Graph shard manifest node count is missing or not integral");
    const std::int64_t raw_nodes = graph.at("nodes").get<std::int64_t>();
    if (raw_nodes < 0)
        throw std::invalid_argument(
            "Graph shard manifest node count is negative");
    header.nodes = static_cast<std::size_t>(raw_nodes);

    header.identity = RequireStringField(graph, "identity");
    const std::string topology =
        RequireStringField(graph, "source_topology_fingerprint");
    if (header.identity.empty() || topology.empty())
        throw std::invalid_argument(
            "Graph shard manifest topology fingerprint is empty");
    if (header.identity != topology)
        throw std::invalid_argument(
            "Graph shard manifest identity and topology fingerprint disagree");

    header.policy_name = RequireStringField(policy, "name");
    if (header.policy_name.empty())
        throw std::invalid_argument(
            "Graph shard manifest policy name is empty");
    if (!policy.contains("id") || !policy.at("id").is_number_integer())
        throw std::invalid_argument(
            "Graph shard manifest policy id is missing or not integral");
    header.policy_id = policy.at("id").get<int>();
    if (!policy.contains("options") || !policy.at("options").is_array())
        throw std::invalid_argument(
            "Graph shard manifest policy options must be an array");
    for (const auto &option : policy.at("options"))
    {
        if (!option.is_string())
            throw std::invalid_argument(
                "Graph shard manifest policy option must be a string");
        header.policy_options.push_back(option.get<std::string>());
    }

    header.balance = RequireStringField(partitioning, "balance");
    if (
        header.balance != "vertices" &&
        header.balance != "out" &&
        header.balance != "total")
    {
        throw std::invalid_argument(
            "Graph shard manifest balance policy is unknown");
    }
    header.partition_count =
        partitioning.at("count").get<std::size_t>();

    if (!manifest.contains("shards") || !manifest.at("shards").is_array())
        throw std::invalid_argument(
            "Graph shard manifest lacks shards");
    const auto &shards = manifest.at("shards");
    if (shards.size() != header.partition_count)
        throw std::invalid_argument(
            "Graph shard count mismatch");

    std::size_t expected_begin = 0;
    header.ownership.reserve(shards.size());
    for (std::size_t index = 0; index < shards.size(); ++index)
    {
        const auto &shard = shards.at(index);
        if (
            shard.at("id").get<std::size_t>() != index ||
            shard.at("owned_begin").get<std::size_t>() !=
                expected_begin)
        {
            throw std::invalid_argument(
                "Graph shard ownership is not contiguous");
        }
        const std::size_t owned_end =
            shard.at("owned_end").get<std::size_t>();
        if (owned_end <= expected_begin || owned_end > header.nodes)
            throw std::invalid_argument(
                "Graph shard ownership range is invalid");
        if (shard.at("symmetric").get<bool>() == header.directed)
            throw std::invalid_argument(
                "Graph shard symmetry contradicts graph.directed");
        header.ownership.emplace_back(expected_begin, owned_end);
        expected_begin = owned_end;
    }
    if (expected_begin != header.nodes)
        throw std::invalid_argument(
            "Graph shard ownership does not cover every vertex");

    if (load_mapping)
    {
        header.internal_to_source =
            ValidateAndReadArrayArtifact<NodeID_>(
                root, mapping.at("internal_to_source"));
        header.source_to_internal =
            ValidateAndReadArrayArtifact<NodeID_>(
                root, mapping.at("source_to_internal"));
        if (
            header.internal_to_source.size() != header.nodes ||
            header.source_to_internal.size() != header.nodes)
        {
            throw std::invalid_argument(
                "Graph shard mapping count mismatch");
        }
        for (std::size_t internal = 0; internal < header.nodes; ++internal)
        {
            const auto source = static_cast<std::size_t>(
                header.internal_to_source[internal]);
            if (
                source >= header.nodes ||
                static_cast<std::size_t>(
                    header.source_to_internal[source]) != internal)
            {
                throw std::invalid_argument(
                    "Graph shard mappings are not inverses");
            }
        }
        if (
            OriginalIdMappingFingerprint(header.internal_to_source) !=
            mapping.at("fingerprint").get<std::string>())
        {
            throw std::invalid_argument(
                "Graph shard mapping fingerprint mismatch");
        }
        header.mapping_loaded = true;
    }
    else
    {
        RequireStringField(mapping, "fingerprint");
    }

    return header;
}

// Validate and load exactly one shard's arrays, checking its CSR structure,
// local slot bounds, ghost metadata against every shard's ownership, and its
// cached scalar metadata. Shared by the full validator and the single-shard
// loader so both apply identical shard-level checks.
template <typename NodeID_, typename Offset_>
LoadedShard<NodeID_, Offset_> ValidateShardEntry(
    const std::filesystem::path &root,
    const nlohmann::json &shard,
    const ShardManifestHeader<NodeID_> &header,
    std::size_t index)
{
    LoadedShard<NodeID_, Offset_> loaded;
    loaded.id = static_cast<std::uint32_t>(index);
    loaded.owned_begin = header.ownership[index].first;
    loaded.owned_end = header.ownership[index].second;
    const std::size_t owned = loaded.owned_count();

    const auto &arrays = RequireObjectField(shard, "arrays");
    loaded.out_offsets =
        ValidateAndReadArrayArtifact<Offset_>(
            root, arrays.at("out_offsets"));
    loaded.out_neighbors =
        ValidateAndReadArrayArtifact<NodeID_>(
            root, arrays.at("out_neighbors"));
    loaded.in_offsets =
        ValidateAndReadArrayArtifact<Offset_>(
            root, arrays.at("in_offsets"));
    loaded.in_neighbors =
        ValidateAndReadArrayArtifact<NodeID_>(
            root, arrays.at("in_neighbors"));
    loaded.ghost_globals =
        ValidateAndReadArrayArtifact<NodeID_>(
            root, arrays.at("ghost_globals"));
    loaded.ghost_owners =
        ValidateAndReadArrayArtifact<std::uint32_t>(
            root, arrays.at("ghost_owners"));

    if (
        loaded.out_offsets.size() != owned + 1 ||
        !ShardOffsetsAreValid(
            loaded.out_offsets, loaded.out_neighbors.size()))
    {
        throw std::invalid_argument(
            "Graph shard outgoing CSR is invalid");
    }
    loaded.symmetric = shard.at("symmetric").get<bool>();
    if (
        loaded.symmetric
            ? (!loaded.in_offsets.empty() || !loaded.in_neighbors.empty())
            : (loaded.in_offsets.size() != owned + 1 ||
               !ShardOffsetsAreValid(
                   loaded.in_offsets, loaded.in_neighbors.size())))
    {
        throw std::invalid_argument(
            "Graph shard incoming CSR is invalid");
    }
    if (loaded.ghost_globals.size() != loaded.ghost_owners.size())
        throw std::invalid_argument(
            "Graph shard ghost arrays differ in size");

    const std::size_t slots = owned + loaded.ghost_globals.size();
    std::uint64_t remote_out_edges = 0;
    for (const NodeID_ slot : loaded.out_neighbors)
    {
        if (static_cast<std::size_t>(slot) >= slots)
            throw std::invalid_argument(
                "Graph shard outgoing local slot is invalid");
        if (static_cast<std::size_t>(slot) >= owned)
            ++remote_out_edges;
    }
    std::uint64_t remote_in_edges = 0;
    for (const NodeID_ slot : loaded.in_neighbors)
    {
        if (static_cast<std::size_t>(slot) >= slots)
            throw std::invalid_argument(
                "Graph shard incoming local slot is invalid");
        if (static_cast<std::size_t>(slot) >= owned)
            ++remote_in_edges;
    }
    if (loaded.symmetric)
        remote_in_edges = remote_out_edges;
    for (
        std::size_t ghost = 0;
        ghost < loaded.ghost_globals.size();
        ++ghost)
    {
        const std::size_t owner = loaded.ghost_owners[ghost];
        const std::size_t global =
            static_cast<std::size_t>(loaded.ghost_globals[ghost]);
        if (
            global >= header.nodes ||
            owner >= header.ownership.size() ||
            owner == index ||
            global < header.ownership[owner].first ||
            global >= header.ownership[owner].second)
        {
            throw std::invalid_argument(
                "Graph shard ghost metadata is invalid");
        }
    }

    loaded.remote_out_edges = remote_out_edges;
    loaded.remote_in_edges = remote_in_edges;
    loaded.storage_bytes =
        static_cast<std::uint64_t>(loaded.out_offsets.size()) *
            sizeof(Offset_) +
        static_cast<std::uint64_t>(loaded.in_offsets.size()) *
            sizeof(Offset_) +
        static_cast<std::uint64_t>(loaded.out_neighbors.size()) *
            sizeof(NodeID_) +
        static_cast<std::uint64_t>(loaded.in_neighbors.size()) *
            sizeof(NodeID_) +
        static_cast<std::uint64_t>(loaded.ghost_globals.size()) *
            sizeof(NodeID_) +
        static_cast<std::uint64_t>(loaded.ghost_owners.size()) *
            sizeof(std::uint32_t);
    loaded.balance_weight =
        header.balance == "vertices"
            ? owned
            : header.balance == "out"
                ? loaded.out_neighbors.size()
                : loaded.out_neighbors.size() +
                  (loaded.symmetric ? 0 : loaded.in_neighbors.size());
    if (
        shard.at("remote_out_edges").get<std::uint64_t>() !=
            loaded.remote_out_edges ||
        shard.at("remote_in_edges").get<std::uint64_t>() !=
            loaded.remote_in_edges ||
        shard.at("storage_bytes").get<std::uint64_t>() !=
            loaded.storage_bytes ||
        shard.at("balance_weight").get<std::uint64_t>() !=
            loaded.balance_weight)
    {
        throw std::invalid_argument(
            "Graph shard scalar metadata is inconsistent");
    }
    return loaded;
}

// Lightweight header load for Blox workers: validates the manifest scalars and
// ownership map (and optionally the source mapping) without reading a single
// shard's CSR/ghost arrays.
template <typename NodeID_, typename Offset_>
ShardManifestHeader<NodeID_> LoadShardManifestHeader(
    const std::filesystem::path &manifest_path,
    bool load_mapping = false)
{
    const nlohmann::json manifest =
        ParseShardManifestFile(manifest_path);
    return ValidateShardManifestHeaderJson<NodeID_, Offset_>(
        manifest, manifest_path.parent_path(), load_mapping);
}

// Load and validate exactly one shard, materializing only that shard's arrays
// (plus O(P) ownership scalars and, optionally, the O(N) source mapping). Blox
// workers use this instead of ValidateShardPackage so they never materialize
// every shard just to read their own.
template <typename NodeID_, typename Offset_>
LoadedShard<NodeID_, Offset_> LoadShardPackageShard(
    const std::filesystem::path &manifest_path,
    std::size_t shard_id,
    ShardManifestHeader<NodeID_> *header_out = nullptr,
    bool load_mapping = false)
{
    const nlohmann::json manifest =
        ParseShardManifestFile(manifest_path);
    const std::filesystem::path root = manifest_path.parent_path();
    const ShardManifestHeader<NodeID_> header =
        ValidateShardManifestHeaderJson<NodeID_, Offset_>(
            manifest, root, load_mapping);
    if (shard_id >= header.partition_count)
        throw std::out_of_range(
            "Requested graph shard id is outside the package");
    LoadedShard<NodeID_, Offset_> loaded =
        ValidateShardEntry<NodeID_, Offset_>(
            root, manifest.at("shards").at(shard_id), header, shard_id);
    if (header_out != nullptr)
        *header_out = header;
    return loaded;
}

// Exhaustive validation: header, mapping, every shard, and the cross-shard edge
// totals against graph.directed_edges. Callers that only need one shard should
// prefer LoadShardPackageShard.
template <typename NodeID_, typename Offset_>
nlohmann::json ValidateShardPackage(
    const std::filesystem::path &manifest_path)
{
    const nlohmann::json manifest =
        ParseShardManifestFile(manifest_path);
    const std::filesystem::path root = manifest_path.parent_path();
    const ShardManifestHeader<NodeID_> header =
        ValidateShardManifestHeaderJson<NodeID_, Offset_>(
            manifest, root, /*load_mapping=*/true);

    const auto &shards = manifest.at("shards");
    std::uint64_t total_out_edges = 0;
    std::uint64_t total_in_edges = 0;
    for (std::size_t index = 0; index < shards.size(); ++index)
    {
        const LoadedShard<NodeID_, Offset_> loaded =
            ValidateShardEntry<NodeID_, Offset_>(
                root, shards.at(index), header, index);
        total_out_edges += loaded.out_neighbors.size();
        total_in_edges += loaded.symmetric
            ? loaded.out_neighbors.size()
            : loaded.in_neighbors.size();
    }
    const std::uint64_t directed_edges =
        static_cast<std::uint64_t>(header.directed_edges);
    if (total_out_edges != directed_edges || total_in_edges != directed_edges)
        throw std::invalid_argument(
            "Graph shard edge totals disagree with graph.directed_edges");
    return manifest;
}

} // namespace partition
} // namespace graphbrew

#endif // GRAPHBREW_PARTITION_SHARD_MANIFEST_H_
