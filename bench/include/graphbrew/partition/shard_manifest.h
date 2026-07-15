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

template <typename GraphT, typename PartitionedGraphT>
std::filesystem::path WriteShardPackage(
    const std::filesystem::path &output,
    const GraphT &graph,
    const OriginalIdMapping<GraphNode<GraphT>> &mapping,
    const PartitionedGraphT &partitioned,
    const ShardPackageMetadata &metadata)
{
    using Node = GraphNode<GraphT>;
    using Offset = typename PartitionedGraphT::Offset;
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
    if (partitioned.num_nodes() != graph.num_nodes())
        throw std::invalid_argument(
            "Shard package graph and partition vertex counts differ");

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
        {"offset_type", IntegralTypeName<Offset>()},
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
        const std::filesystem::path directory =
            ShardDirectoryName(part.id);
        nlohmann::json arrays;
        arrays["out_offsets"] = WriteArrayArtifact(
            temporary,
            directory / "out_offsets.bin",
            part.out_offsets);
        arrays["out_neighbors"] = WriteArrayArtifact(
            temporary,
            directory / "out_neighbors.bin",
            part.out_neighbors);
        arrays["in_offsets"] = WriteArrayArtifact(
            temporary,
            directory / "in_offsets.bin",
            part.in_offsets);
        arrays["in_neighbors"] = WriteArrayArtifact(
            temporary,
            directory / "in_neighbors.bin",
            part.in_neighbors);
        arrays["ghost_globals"] = WriteArrayArtifact(
            temporary,
            directory / "ghost_globals.bin",
            part.ghost_globals);
        arrays["ghost_owners"] = WriteArrayArtifact(
            temporary,
            directory / "ghost_owners.bin",
            part.ghost_owners);
        manifest["shards"].push_back({
            {"id", part.id},
            {"owned_begin", part.vertex_begin},
            {"owned_end", part.vertex_end},
            {"symmetric", part.symmetric},
            {"balance_weight", part.balance_weight},
            {"remote_out_edges", part.remote_out_edges},
            {"remote_in_edges", part.remote_in_edges},
            {"storage_bytes", part.storage_bytes()},
            {"arrays", std::move(arrays)},
        });
    }

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

template <typename NodeID_, typename Offset_>
nlohmann::json ValidateShardPackage(
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

    const auto &graph = RequireObjectField(manifest, "graph");
    const auto &encoding = RequireObjectField(manifest, "encoding");
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

    const std::size_t nodes =
        graph.at("nodes").get<std::size_t>();
    const auto internal_to_source =
        ValidateAndReadArrayArtifact<NodeID_>(
            manifest_path.parent_path(),
            mapping.at("internal_to_source"));
    const auto source_to_internal =
        ValidateAndReadArrayArtifact<NodeID_>(
            manifest_path.parent_path(),
            mapping.at("source_to_internal"));
    if (
        internal_to_source.size() != nodes ||
        source_to_internal.size() != nodes)
    {
        throw std::invalid_argument(
            "Graph shard mapping count mismatch");
    }
    for (std::size_t internal = 0; internal < nodes; ++internal)
    {
        const auto source = static_cast<std::size_t>(
            internal_to_source[internal]);
        if (
            source >= nodes ||
            static_cast<std::size_t>(
                source_to_internal[source]) != internal)
        {
            throw std::invalid_argument(
                "Graph shard mappings are not inverses");
        }
    }
    if (
        OriginalIdMappingFingerprint(internal_to_source) !=
        mapping.at("fingerprint").get<std::string>())
    {
        throw std::invalid_argument(
            "Graph shard mapping fingerprint mismatch");
    }

    if (!manifest.contains("shards") || !manifest.at("shards").is_array())
        throw std::invalid_argument(
            "Graph shard manifest lacks shards");
    const auto &shards = manifest.at("shards");
    if (
        shards.size() !=
        partitioning.at("count").get<std::size_t>())
    {
        throw std::invalid_argument(
            "Graph shard count mismatch");
    }

    std::size_t expected_begin = 0;
    std::vector<std::pair<std::size_t, std::size_t>> ownership;
    ownership.reserve(shards.size());
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
        if (owned_end <= expected_begin || owned_end > nodes)
            throw std::invalid_argument(
                "Graph shard ownership range is invalid");
        ownership.emplace_back(expected_begin, owned_end);
        expected_begin = owned_end;
    }
    if (expected_begin != nodes)
        throw std::invalid_argument(
            "Graph shard ownership does not cover every vertex");

    const auto offsets_are_valid =
        [](const auto &offsets, std::size_t neighbors)
        {
            if (
                offsets.empty() ||
                offsets.front() != 0 ||
                offsets.back() != neighbors)
            {
                return false;
            }
            return std::is_sorted(offsets.begin(), offsets.end());
        };
    for (std::size_t index = 0; index < shards.size(); ++index)
    {
        const auto &shard = shards.at(index);
        const std::size_t owned =
            ownership[index].second - ownership[index].first;
        const auto &arrays = RequireObjectField(shard, "arrays");
        const auto out_offsets =
            ValidateAndReadArrayArtifact<Offset_>(
                manifest_path.parent_path(), arrays.at("out_offsets"));
        const auto out_neighbors =
            ValidateAndReadArrayArtifact<NodeID_>(
                manifest_path.parent_path(), arrays.at("out_neighbors"));
        const auto in_offsets =
            ValidateAndReadArrayArtifact<Offset_>(
                manifest_path.parent_path(), arrays.at("in_offsets"));
        const auto in_neighbors =
            ValidateAndReadArrayArtifact<NodeID_>(
                manifest_path.parent_path(), arrays.at("in_neighbors"));
        const auto ghost_globals =
            ValidateAndReadArrayArtifact<NodeID_>(
                manifest_path.parent_path(), arrays.at("ghost_globals"));
        const auto ghost_owners =
            ValidateAndReadArrayArtifact<std::uint32_t>(
                manifest_path.parent_path(), arrays.at("ghost_owners"));
        if (
            out_offsets.size() != owned + 1 ||
            !offsets_are_valid(
                out_offsets, out_neighbors.size()))
        {
            throw std::invalid_argument(
                "Graph shard outgoing CSR is invalid");
        }
        const bool symmetric =
            shard.at("symmetric").get<bool>();
        if (
            symmetric
                ? (!in_offsets.empty() || !in_neighbors.empty())
                : (in_offsets.size() != owned + 1 ||
                   !offsets_are_valid(
                       in_offsets, in_neighbors.size())))
        {
            throw std::invalid_argument(
                "Graph shard incoming CSR is invalid");
        }
        if (ghost_globals.size() != ghost_owners.size())
            throw std::invalid_argument(
                "Graph shard ghost arrays differ in size");
        const std::size_t slots = owned + ghost_globals.size();
        std::uint64_t remote_out_edges = 0;
        for (const NodeID_ slot : out_neighbors)
        {
            if (static_cast<std::size_t>(slot) >= slots)
                throw std::invalid_argument(
                    "Graph shard outgoing local slot is invalid");
            if (static_cast<std::size_t>(slot) >= owned)
                ++remote_out_edges;
        }
        std::uint64_t remote_in_edges = 0;
        for (const NodeID_ slot : in_neighbors)
        {
            if (static_cast<std::size_t>(slot) >= slots)
                throw std::invalid_argument(
                    "Graph shard incoming local slot is invalid");
            if (static_cast<std::size_t>(slot) >= owned)
                ++remote_in_edges;
        }
        if (symmetric)
            remote_in_edges = remote_out_edges;
        for (std::size_t ghost = 0; ghost < ghost_globals.size(); ++ghost)
        {
            const std::size_t owner = ghost_owners[ghost];
            const std::size_t global =
                static_cast<std::size_t>(ghost_globals[ghost]);
            if (
                global >= nodes ||
                owner >= shards.size() ||
                owner == index ||
                global < ownership[owner].first ||
                global >= ownership[owner].second)
            {
                throw std::invalid_argument(
                    "Graph shard ghost metadata is invalid");
            }
        }
        const std::uint64_t storage_bytes =
            static_cast<std::uint64_t>(out_offsets.size()) *
                sizeof(Offset_) +
            static_cast<std::uint64_t>(in_offsets.size()) *
                sizeof(Offset_) +
            static_cast<std::uint64_t>(out_neighbors.size()) *
                sizeof(NodeID_) +
            static_cast<std::uint64_t>(in_neighbors.size()) *
                sizeof(NodeID_) +
            static_cast<std::uint64_t>(ghost_globals.size()) *
                sizeof(NodeID_) +
            static_cast<std::uint64_t>(ghost_owners.size()) *
                sizeof(std::uint32_t);
        const std::string balance =
            partitioning.at("balance").get<std::string>();
        const std::uint64_t expected_weight =
            balance == "vertices"
                ? owned
                : balance == "out"
                    ? out_neighbors.size()
                    : out_neighbors.size() +
                      (symmetric ? 0 : in_neighbors.size());
        if (
            shard.at("remote_out_edges").get<std::uint64_t>() !=
                remote_out_edges ||
            shard.at("remote_in_edges").get<std::uint64_t>() !=
                remote_in_edges ||
            shard.at("storage_bytes").get<std::uint64_t>() !=
                storage_bytes ||
            shard.at("balance_weight").get<std::uint64_t>() !=
                expected_weight)
        {
            throw std::invalid_argument(
                "Graph shard scalar metadata is inconsistent");
        }
    }
    return manifest;
}

} // namespace partition
} // namespace graphbrew

#endif // GRAPHBREW_PARTITION_SHARD_MANIFEST_H_
