#ifndef GRAPHBREW_PARTITION_OWNERSHIP_ANALYSIS_H_
#define GRAPHBREW_PARTITION_OWNERSHIP_ANALYSIS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "graphbrew/partition/compact_csr.h"
#include "graphbrew/partition/diagnostics.h"

namespace graphbrew {
namespace partition {

template <typename NodeID_, typename DestID_>
NodeID_ OwnershipNeighborVertex(const DestID_ &neighbor)
{
    if constexpr (std::is_same<NodeID_, DestID_>::value)
        return neighbor;
    else
        return neighbor.v;
}

template <typename NodeID_>
struct OwnershipMetrics
{
    std::size_t partition_count = 0;
    std::vector<std::uint32_t> owner_by_vertex;
    std::vector<std::uint64_t> owned_vertices;
    std::vector<std::uint64_t> out_edges;
    std::vector<std::uint64_t> in_edges;
    std::vector<std::uint64_t> balance_weights;
    std::vector<std::uint64_t> remote_out;
    std::vector<std::uint64_t> remote_in;
    std::vector<std::uint64_t> ghost_slots;
    std::vector<std::uint64_t> ownership_metadata_bytes;
    std::vector<std::uint64_t> storage_bytes;
    std::uint64_t total_remote_out = 0;
    std::uint64_t total_remote_in = 0;
    std::uint64_t total_ghost_slots = 0;
    std::uint64_t total_ownership_metadata_bytes = 0;
    std::uint64_t total_storage_bytes = 0;
    std::uint64_t max_storage_bytes = 0;
    double remote_out_fraction = 0;
    double remote_in_fraction = 0;
    double max_remote_out_fraction = 0;
    double max_remote_in_fraction = 0;
    double vertex_imbalance = 1;
    double out_edge_imbalance = 1;
    double in_edge_imbalance = 1;
    double balance_imbalance = 1;
    double storage_imbalance = 1;
    std::string owner_fingerprint;
};

inline double OwnershipImbalance(
    const std::vector<std::uint64_t> &values)
{
    if (values.empty())
        return 1.0;
    const long double total = std::accumulate(
        values.begin(), values.end(), static_cast<long double>(0));
    if (total == 0)
        return 1.0;
    const long double mean = total / values.size();
    return static_cast<double>(
        *std::max_element(values.begin(), values.end()) / mean);
}

inline std::uint64_t OwnershipCheckedMultiply(
    std::uint64_t lhs,
    std::uint64_t rhs)
{
    if (
        lhs != 0 &&
        rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
    {
        throw std::overflow_error(
            "Ownership byte count overflow");
    }
    return lhs * rhs;
}

template <typename NodeID_, typename GraphT>
OwnershipMetrics<NodeID_> EvaluateOwnership(
    const GraphT &graph,
    const std::vector<std::uint32_t> &owners,
    GraphPartitionBalance balance,
    std::size_t requested_partitions = 0,
    bool requires_owned_vertex_map = false)
{
    namespace detail = graphbrew_compact_detail;
    const std::size_t nodes =
        static_cast<std::size_t>(graph.num_nodes());
    if (owners.size() != nodes)
        throw std::invalid_argument(
            "Ownership map does not cover every vertex");
    if (owners.empty())
        return {};
    const std::uint32_t max_owner =
        *std::max_element(owners.begin(), owners.end());
    const std::size_t minimum_partitions =
        static_cast<std::size_t>(max_owner) + 1;
    const std::size_t partitions = requested_partitions == 0
        ? minimum_partitions
        : requested_partitions;
    if (partitions < minimum_partitions)
        throw std::invalid_argument(
            "Requested partition count excludes an ownership ID");
    OwnershipMetrics<NodeID_> result;
    result.partition_count = partitions;
    result.owner_by_vertex = owners;
    result.owned_vertices.assign(partitions, 0);
    result.out_edges.assign(partitions, 0);
    result.in_edges.assign(partitions, 0);
    result.balance_weights.assign(partitions, 0);
    result.remote_out.assign(partitions, 0);
    result.remote_in.assign(partitions, 0);
    result.ghost_slots.assign(partitions, 0);
    result.ownership_metadata_bytes.assign(partitions, 0);
    result.storage_bytes.assign(partitions, 0);
    std::vector<std::unordered_set<NodeID_>> ghosts(partitions);

    for (std::size_t raw_vertex = 0;
         raw_vertex < nodes; ++raw_vertex)
    {
        const NodeID_ vertex = static_cast<NodeID_>(raw_vertex);
        const std::size_t owner = owners[raw_vertex];
        if (owner >= partitions)
            throw std::invalid_argument("Ownership map has invalid owner");
        const std::uint64_t out_degree =
            detail::CheckedDegree(graph.out_degree(vertex));
        const std::uint64_t in_degree = graph.directed()
            ? detail::CheckedDegree(graph.in_degree(vertex))
            : out_degree;
        ++result.owned_vertices[owner];
        result.out_edges[owner] = detail::CheckedAdd(
            result.out_edges[owner], out_degree);
        result.in_edges[owner] = detail::CheckedAdd(
            result.in_edges[owner], in_degree);
        const std::uint64_t weight =
            balance == GraphPartitionBalance::kVertices
                ? 1
                : (
                    balance == GraphPartitionBalance::kOutgoingEdges
                        ? out_degree
                        : (
                            graph.directed()
                                ? detail::CheckedAdd(
                                    out_degree, in_degree)
                                : out_degree));
        result.balance_weights[owner] = detail::CheckedAdd(
            result.balance_weights[owner], weight);

        for (const auto &neighbor : graph.out_neigh(vertex))
        {
            const NodeID_ remote =
                OwnershipNeighborVertex<NodeID_>(neighbor);
            const std::size_t remote_owner =
                owners.at(static_cast<std::size_t>(remote));
            if (remote_owner != owner)
            {
                result.remote_out[owner] = detail::CheckedAdd(
                    result.remote_out[owner], 1);
                ghosts[owner].insert(remote);
            }
        }
        if (graph.directed())
        {
            for (const auto &neighbor : graph.in_neigh(vertex))
            {
                const NodeID_ remote =
                    OwnershipNeighborVertex<NodeID_>(neighbor);
                const std::size_t remote_owner =
                    owners.at(static_cast<std::size_t>(remote));
                if (remote_owner != owner)
                {
                    result.remote_in[owner] = detail::CheckedAdd(
                        result.remote_in[owner], 1);
                    ghosts[owner].insert(remote);
                }
            }
        }
    }
    if (!graph.directed())
        result.remote_in = result.remote_out;

    const std::uint64_t edge_total =
        static_cast<std::uint64_t>(graph.num_edges_directed());
    for (std::size_t shard = 0; shard < partitions; ++shard)
    {
        result.ghost_slots[shard] = ghosts[shard].size();
        const std::uint64_t offsets =
            detail::CheckedAdd(result.owned_vertices[shard], 1);
        std::uint64_t bytes = OwnershipCheckedMultiply(
            offsets, sizeof(std::uint64_t));
        bytes = detail::CheckedAdd(
            bytes,
            OwnershipCheckedMultiply(
                result.out_edges[shard], sizeof(NodeID_)));
        if (graph.directed())
        {
            bytes = detail::CheckedAdd(
                bytes,
                OwnershipCheckedMultiply(
                    offsets, sizeof(std::uint64_t)));
            bytes = detail::CheckedAdd(
                bytes,
                OwnershipCheckedMultiply(
                    result.in_edges[shard], sizeof(NodeID_)));
        }
        bytes = detail::CheckedAdd(
            bytes,
            OwnershipCheckedMultiply(
                result.ghost_slots[shard],
                sizeof(NodeID_) + sizeof(std::uint32_t)));
        if (requires_owned_vertex_map)
        {
            result.ownership_metadata_bytes[shard] =
                OwnershipCheckedMultiply(
                    result.owned_vertices[shard],
                    sizeof(NodeID_));
            bytes = detail::CheckedAdd(
                bytes,
                result.ownership_metadata_bytes[shard]);
        }
        result.storage_bytes[shard] = bytes;
        result.total_remote_out = detail::CheckedAdd(
            result.total_remote_out, result.remote_out[shard]);
        result.total_remote_in = detail::CheckedAdd(
            result.total_remote_in, result.remote_in[shard]);
        result.total_ghost_slots = detail::CheckedAdd(
            result.total_ghost_slots, result.ghost_slots[shard]);
        result.total_ownership_metadata_bytes = detail::CheckedAdd(
            result.total_ownership_metadata_bytes,
            result.ownership_metadata_bytes[shard]);
        result.total_storage_bytes = detail::CheckedAdd(
            result.total_storage_bytes, bytes);
        result.max_storage_bytes =
            std::max(result.max_storage_bytes, bytes);
        if (result.out_edges[shard] != 0)
        {
            result.max_remote_out_fraction = std::max(
                result.max_remote_out_fraction,
                static_cast<double>(result.remote_out[shard]) /
                    result.out_edges[shard]);
        }
        if (result.in_edges[shard] != 0)
        {
            result.max_remote_in_fraction = std::max(
                result.max_remote_in_fraction,
                static_cast<double>(result.remote_in[shard]) /
                    result.in_edges[shard]);
        }
    }
    if (result.total_remote_out != result.total_remote_in)
        throw std::logic_error(
            "Ownership remote incoming/outgoing totals disagree");
    if (edge_total != 0)
    {
        result.remote_out_fraction =
            static_cast<double>(result.total_remote_out) / edge_total;
        result.remote_in_fraction =
            static_cast<double>(result.total_remote_in) / edge_total;
    }
    result.vertex_imbalance = OwnershipImbalance(
        result.owned_vertices);
    result.out_edge_imbalance = OwnershipImbalance(
        result.out_edges);
    result.in_edge_imbalance = OwnershipImbalance(
        result.in_edges);
    result.balance_imbalance = OwnershipImbalance(
        result.balance_weights);
    result.storage_imbalance = OwnershipImbalance(
        result.storage_bytes);
    OrderedFingerprint fingerprint;
    fingerprint.AddRange(owners);
    result.owner_fingerprint = fingerprint.Hex();
    return result;
}

template <typename NodeID_, typename GraphT, typename MappingT>
std::vector<std::uint32_t> BuildContiguousOwners(
    const GraphT &graph,
    const MappingT &new_ids,
    std::size_t partition_count,
    GraphPartitionBalance balance)
{
    const auto base = BuildGraphPartitionPlan<NodeID_>(
        graph, partition_count, balance);
    const std::size_t nodes =
        static_cast<std::size_t>(graph.num_nodes());
    if (new_ids.size() != nodes)
        throw std::invalid_argument(
            "Reordered mapping does not cover every vertex");
    std::vector<std::uint64_t> reordered_weights(nodes);
    std::vector<bool> seen(nodes, false);
    for (std::size_t vertex = 0; vertex < nodes; ++vertex)
    {
        const std::size_t reordered =
            static_cast<std::size_t>(new_ids[vertex]);
        if (reordered >= nodes || seen[reordered])
            throw std::invalid_argument(
                "Reordered mapping is not a permutation");
        seen[reordered] = true;
        reordered_weights[reordered] = base.weights[vertex];
    }
    const auto ranges = graphbrew_compact_detail::BuildBalancedRanges(
        reordered_weights, base.partition_count);
    std::vector<std::uint32_t> owner_new(nodes);
    for (std::size_t shard = 0; shard < ranges.size(); ++shard)
    {
        std::fill(
            owner_new.begin() +
                static_cast<std::ptrdiff_t>(ranges[shard].first),
            owner_new.begin() +
                static_cast<std::ptrdiff_t>(ranges[shard].second),
            static_cast<std::uint32_t>(shard));
    }
    std::vector<std::uint32_t> owners(nodes);
    for (std::size_t vertex = 0; vertex < nodes; ++vertex)
        owners[vertex] = owner_new[
            static_cast<std::size_t>(new_ids[vertex])];
    return owners;
}

template <typename NodeID_, typename GraphT, typename MembershipT>
std::pair<std::vector<std::uint32_t>, bool>
BuildCommunityOwners(
    const GraphT &graph,
    const MembershipT &membership,
    std::size_t requested_partitions,
    GraphPartitionBalance balance)
{
    const std::size_t nodes =
        static_cast<std::size_t>(graph.num_nodes());
    if (membership.size() != nodes || requested_partitions == 0)
        throw std::invalid_argument(
            "Community ownership input is invalid");
    const auto weights = BuildGraphPartitionPlan<NodeID_>(
        graph, requested_partitions, balance).weights;
    struct Community
    {
        std::uint64_t weight = 0;
        std::size_t vertices = 0;
        std::size_t canonical = 0;
        std::vector<std::size_t> members;
    };
    std::unordered_map<
        typename MembershipT::value_type,
        std::size_t> community_index;
    std::vector<Community> communities;
    for (std::size_t vertex = 0; vertex < nodes; ++vertex)
    {
        const auto label = membership[vertex];
        auto inserted = community_index.emplace(
            label, communities.size());
        if (inserted.second)
        {
            Community community;
            community.canonical = vertex;
            communities.push_back(std::move(community));
        }
        Community &community = communities[inserted.first->second];
        community.weight = graphbrew_compact_detail::CheckedAdd(
            community.weight, weights[vertex]);
        ++community.vertices;
        community.canonical =
            std::min(community.canonical, vertex);
        community.members.push_back(vertex);
    }
    std::vector<std::size_t> order(communities.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(), order.end(),
        [&](std::size_t lhs, std::size_t rhs) {
            const Community &a = communities[lhs];
            const Community &b = communities[rhs];
            if (a.weight != b.weight)
                return a.weight > b.weight;
            if (a.vertices != b.vertices)
                return a.vertices > b.vertices;
            return a.canonical < b.canonical;
        });
    const std::size_t partitions =
        std::min(requested_partitions, nodes);
    std::vector<std::uint64_t> shard_weights(partitions, 0);
    std::vector<std::uint64_t> shard_vertices(partitions, 0);
    std::vector<std::uint32_t> owners(nodes);
    for (const std::size_t community_id : order)
    {
        std::size_t shard = 0;
        for (std::size_t candidate = 1;
             candidate < partitions; ++candidate)
        {
            if (
                shard_weights[candidate] < shard_weights[shard] ||
                (
                    shard_weights[candidate] == shard_weights[shard] &&
                    (
                        shard_vertices[candidate] <
                            shard_vertices[shard] ||
                        (
                            shard_vertices[candidate] ==
                                shard_vertices[shard] &&
                            candidate < shard))))
            {
                shard = candidate;
            }
        }
        const Community &community = communities[community_id];
        shard_weights[shard] = graphbrew_compact_detail::CheckedAdd(
            shard_weights[shard], community.weight);
        shard_vertices[shard] = graphbrew_compact_detail::CheckedAdd(
            shard_vertices[shard], community.vertices);
        for (const std::size_t vertex : community.members)
            owners[vertex] = static_cast<std::uint32_t>(shard);
    }
    const bool all_nonempty = std::all_of(
        shard_vertices.begin(), shard_vertices.end(),
        [](std::uint64_t count) { return count != 0; });
    return {
        std::move(owners),
        all_nonempty &&
            OwnershipImbalance(shard_weights) <= 1.05,
    };
}

} // namespace partition
} // namespace graphbrew

#endif // GRAPHBREW_PARTITION_OWNERSHIP_ANALYSIS_H_
