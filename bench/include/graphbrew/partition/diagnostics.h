#ifndef GRAPHBREW_PARTITION_DIAGNOSTICS_H_
#define GRAPHBREW_PARTITION_DIAGNOSTICS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace graphbrew
{
namespace partition
{

inline std::uint64_t RotateLeft(std::uint64_t value, unsigned shift)
{
    shift &= 63u;
    if (shift == 0)
        return value;
    return (value << shift) | (value >> (64u - shift));
}

inline std::uint64_t Mix64(std::uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

inline std::string Hex64(std::uint64_t value)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

class OrderedFingerprint
{
public:
    void Add(std::uint64_t value)
    {
        const std::uint64_t mixed =
            Mix64(value + 0x9e3779b97f4a7c15ULL + count_);
        first_ ^= mixed;
        first_ *= 0x100000001b3ULL;
        second_ += mixed ^ 0xd6e8feb86659fd93ULL;
        second_ = RotateLeft(second_, 27);
        second_ *= 0x94d049bb133111ebULL;
        ++count_;
    }

    template <typename T>
    void AddIntegral(T value)
    {
        static_assert(
            std::is_integral<T>::value || std::is_enum<T>::value,
            "fingerprints accept only integral or enum values");
        if constexpr (std::is_enum<T>::value)
        {
            using Raw = typename std::underlying_type<T>::type;
            using Unsigned = typename std::make_unsigned<Raw>::type;
            Add(static_cast<std::uint64_t>(
                static_cast<Unsigned>(static_cast<Raw>(value))));
        }
        else
        {
            using Unsigned = typename std::make_unsigned<T>::type;
            Add(static_cast<std::uint64_t>(
                static_cast<Unsigned>(value)));
        }
    }

    template <typename Range>
    void AddRange(const Range &range)
    {
        Add(static_cast<std::uint64_t>(range.size()));
        for (const auto value : range)
            AddIntegral(value);
    }

    std::string Hex() const
    {
        const std::uint64_t final_first =
            Mix64(first_ ^ Mix64(count_));
        const std::uint64_t final_second =
            Mix64(second_ + Mix64(count_ ^ 0xa0761d6478bd642fULL));
        return Hex64(final_first) + Hex64(final_second);
    }

private:
    std::uint64_t first_ = 0xcbf29ce484222325ULL;
    std::uint64_t second_ = 0x6eed0e9da4d94a4fULL;
    std::uint64_t count_ = 0;
};

template <typename GraphT>
using GraphNode = typename std::remove_cv<
    typename std::remove_pointer<
        decltype(std::declval<const GraphT &>().get_org_ids())>::type>::type;

template <typename NodeID_>
struct OriginalIdMapping
{
    std::vector<NodeID_> internal_to_source;
    std::vector<NodeID_> source_to_internal;
    std::string fingerprint;
};

template <typename Range>
std::string OriginalIdMappingFingerprint(const Range &internal_to_source)
{
    OrderedFingerprint fingerprint;
    fingerprint.Add(1);
    fingerprint.Add(internal_to_source.size());
    for (const auto source : internal_to_source)
        fingerprint.AddIntegral(source);
    return fingerprint.Hex();
}

template <typename GraphT>
OriginalIdMapping<GraphNode<GraphT>> BuildOriginalIdMapping(
    const GraphT &graph)
{
    using Node = GraphNode<GraphT>;
    const std::int64_t raw_nodes = graph.num_nodes();
    if (raw_nodes < 0)
        throw std::invalid_argument(
            "Cannot validate a negative graph vertex count");
    const std::size_t nodes = static_cast<std::size_t>(raw_nodes);
    const Node *org_ids = graph.get_org_ids();
    if (nodes != 0 && org_ids == nullptr)
        throw std::logic_error(
            "Graph has no internal-to-source vertex mapping");

    OriginalIdMapping<Node> mapping;
    mapping.internal_to_source.resize(nodes);
    mapping.source_to_internal.resize(nodes);
    std::vector<std::uint8_t> seen(nodes, 0);
    for (std::size_t internal = 0; internal < nodes; ++internal)
    {
        const Node source = org_ids[internal];
        if constexpr (std::is_signed<Node>::value)
        {
            if (source < 0)
                throw std::logic_error(
                    "Graph source mapping contains a negative vertex");
        }
        const std::size_t source_index =
            static_cast<std::size_t>(source);
        if (source_index >= nodes)
            throw std::logic_error(
                "Graph source mapping contains an out-of-range vertex");
        if (seen[source_index] != 0)
            throw std::logic_error(
                "Graph source mapping is not a permutation");
        seen[source_index] = 1;
        mapping.internal_to_source[internal] = source;
        mapping.source_to_internal[source_index] =
            static_cast<Node>(internal);
    }

    mapping.fingerprint =
        OriginalIdMappingFingerprint(mapping.internal_to_source);
    return mapping;
}

class UnorderedEdgeFingerprint
{
public:
    void Add(std::uint64_t source, std::uint64_t destination)
    {
        const std::uint64_t source_hash =
            Mix64(source ^ 0x243f6a8885a308d3ULL);
        const std::uint64_t destination_hash =
            Mix64(destination ^ 0x13198a2e03707344ULL);
        const std::uint64_t edge_hash =
            Mix64(source_hash ^ RotateLeft(destination_hash, 1));
        sum_first_ += edge_hash;
        sum_second_ +=
            Mix64(edge_hash ^ 0xa4093822299f31d0ULL);
        xor_first_ ^= RotateLeft(
            edge_hash, static_cast<unsigned>(edge_hash));
        xor_second_ ^= Mix64(
            edge_hash + 0x082efa98ec4e6c89ULL);
        ++count_;
    }

    std::string Hex(
        std::uint64_t nodes,
        bool directed) const
    {
        OrderedFingerprint metadata;
        metadata.Add(1);
        metadata.Add(nodes);
        metadata.Add(count_);
        metadata.Add(directed ? 1 : 0);
        return metadata.Hex() +
               Hex64(Mix64(sum_first_)) +
               Hex64(Mix64(sum_second_)) +
               Hex64(Mix64(xor_first_)) +
               Hex64(Mix64(xor_second_));
    }

private:
    std::uint64_t sum_first_ = 0;
    std::uint64_t sum_second_ = 0;
    std::uint64_t xor_first_ = 0;
    std::uint64_t xor_second_ = 0;
    std::uint64_t count_ = 0;
};

template <typename GraphT>
std::string SourceTopologyFingerprint(
    const GraphT &graph,
    const OriginalIdMapping<GraphNode<GraphT>> &mapping)
{
    using Node = GraphNode<GraphT>;
    const std::size_t nodes =
        static_cast<std::size_t>(graph.num_nodes());
    if (mapping.internal_to_source.size() != nodes)
        throw std::invalid_argument(
            "Topology fingerprint mapping size mismatch");

    UnorderedEdgeFingerprint fingerprint;
    for (std::size_t internal = 0; internal < nodes; ++internal)
    {
        const Node source = mapping.internal_to_source[internal];
        for (const auto raw_neighbor : graph.out_neigh(
                 static_cast<Node>(internal)))
        {
            const Node neighbor = static_cast<Node>(raw_neighbor);
            if constexpr (std::is_signed<Node>::value)
            {
                if (neighbor < 0)
                    throw std::logic_error(
                        "Graph topology contains a negative neighbor");
            }
            const std::size_t neighbor_index =
                static_cast<std::size_t>(neighbor);
            if (neighbor_index >= nodes)
                throw std::logic_error(
                    "Graph topology contains an out-of-range neighbor");
            fingerprint.Add(
                static_cast<std::uint64_t>(source),
                static_cast<std::uint64_t>(
                    mapping.internal_to_source[neighbor_index]));
        }
    }
    return fingerprint.Hex(nodes, graph.directed());
}

template <typename PartitionedGraphT>
std::string CompactShardFingerprint(
    const PartitionedGraphT &graph)
{
    OrderedFingerprint fingerprint;
    fingerprint.Add(1);
    fingerprint.Add(graph.num_nodes());
    fingerprint.Add(graph.num_edges_directed());
    fingerprint.Add(graph.directed() ? 1 : 0);
    fingerprint.AddIntegral(graph.balance());
    fingerprint.Add(graph.num_partitions());
    for (const auto &part : graph.partitions())
    {
        fingerprint.Add(part.id);
        fingerprint.AddIntegral(part.vertex_begin);
        fingerprint.AddIntegral(part.vertex_end);
        fingerprint.Add(part.balance_weight);
        fingerprint.Add(part.remote_out_edges);
        fingerprint.Add(part.remote_in_edges);
        fingerprint.Add(part.symmetric ? 1 : 0);
        fingerprint.AddRange(part.out_offsets);
        fingerprint.AddRange(part.out_neighbors);
        fingerprint.AddRange(part.in_offsets);
        fingerprint.AddRange(part.in_neighbors);
        fingerprint.AddRange(part.ghost_globals);
        fingerprint.AddRange(part.ghost_owners);
    }
    return fingerprint.Hex();
}

template <typename PartitionedGraphT>
std::string GhostMetadataFingerprint(
    const PartitionedGraphT &graph)
{
    OrderedFingerprint fingerprint;
    fingerprint.Add(1);
    fingerprint.Add(graph.num_partitions());
    for (const auto &part : graph.partitions())
    {
        fingerprint.Add(part.id);
        fingerprint.AddIntegral(part.vertex_begin);
        fingerprint.AddIntegral(part.vertex_end);
        fingerprint.AddRange(part.ghost_globals);
        fingerprint.AddRange(part.ghost_owners);
    }
    return fingerprint.Hex();
}

template <typename NodeID_>
struct BfsDepthSummary
{
    NodeID_ source_id = 0;
    std::size_t reachable_vertices = 0;
    std::uint64_t max_depth = 0;
    std::string fingerprint;
};

// The source-space depth fingerprint is a cross-policy correctness oracle.
// It intentionally ignores parent choice: any valid BFS tree for the same
// source graph must produce the same reachability and depth vector.
template <typename GraphT, typename ParentRange>
BfsDepthSummary<GraphNode<GraphT>> SourceDepthFingerprint(
    const GraphT &graph,
    const OriginalIdMapping<GraphNode<GraphT>> &mapping,
    GraphNode<GraphT> source_internal,
    const ParentRange &parent)
{
    using Node = GraphNode<GraphT>;
    const std::size_t nodes =
        static_cast<std::size_t>(graph.num_nodes());
    if (
        parent.size() != nodes ||
        mapping.internal_to_source.size() != nodes)
    {
        throw std::invalid_argument(
            "BFS depth fingerprint size mismatch");
    }
    if constexpr (std::is_signed<Node>::value)
    {
        if (source_internal < 0)
            throw std::invalid_argument(
                "BFS source is negative");
    }
    if (static_cast<std::size_t>(source_internal) >= nodes)
        throw std::invalid_argument(
            "BFS source is out of range");

    constexpr std::int64_t kUnknown = -2;
    std::vector<std::int64_t> depth(nodes, kUnknown);
    std::vector<std::uint8_t> state(nodes, 0);
    depth[static_cast<std::size_t>(source_internal)] = 0;
    state[static_cast<std::size_t>(source_internal)] = 2;

    for (std::size_t start = 0; start < nodes; ++start)
    {
        if (state[start] == 2)
            continue;
        if (parent[start] < 0)
        {
            depth[start] = -1;
            state[start] = 2;
            continue;
        }

        std::vector<std::size_t> path;
        std::size_t current = start;
        std::int64_t base_depth = kUnknown;
        while (state[current] != 2)
        {
            if (state[current] == 1)
                throw std::logic_error(
                    "BFS parent result contains a cycle");
            state[current] = 1;
            path.push_back(current);

            const Node raw_parent = parent[current];
            if (raw_parent < 0)
            {
                base_depth = -1;
                break;
            }
            const std::size_t parent_index =
                static_cast<std::size_t>(raw_parent);
            if (parent_index >= nodes)
                throw std::logic_error(
                    "BFS parent result is out of range");
            if (parent_index == current)
            {
                if (current !=
                    static_cast<std::size_t>(source_internal))
                {
                    throw std::logic_error(
                        "BFS parent result contains a non-source root");
                }
                depth[current] = 0;
                state[current] = 2;
                path.pop_back();
                base_depth = 0;
                break;
            }
            current = parent_index;
        }
        if (base_depth == kUnknown)
            base_depth = depth[current];

        for (auto it = path.rbegin(); it != path.rend(); ++it)
        {
            if (base_depth < 0)
            {
                depth[*it] = -1;
            }
            else
            {
                ++base_depth;
                depth[*it] = base_depth;
            }
            state[*it] = 2;
        }
    }

    std::vector<std::int64_t> source_depth(nodes, -1);
    BfsDepthSummary<Node> summary;
    summary.source_id = mapping.internal_to_source[
        static_cast<std::size_t>(source_internal)];
    for (std::size_t internal = 0; internal < nodes; ++internal)
    {
        const std::size_t source = static_cast<std::size_t>(
            mapping.internal_to_source[internal]);
        source_depth[source] = depth[internal];
        if (depth[internal] >= 0)
        {
            ++summary.reachable_vertices;
            summary.max_depth = std::max(
                summary.max_depth,
                static_cast<std::uint64_t>(depth[internal]));
        }
    }

    OrderedFingerprint fingerprint;
    fingerprint.Add(1);
    fingerprint.AddIntegral(summary.source_id);
    fingerprint.Add(source_depth.size());
    for (const std::int64_t value : source_depth)
        fingerprint.AddIntegral(value);
    summary.fingerprint = fingerprint.Hex();
    return summary;
}

} // namespace partition
} // namespace graphbrew

#endif // GRAPHBREW_PARTITION_DIAGNOSTICS_H_
