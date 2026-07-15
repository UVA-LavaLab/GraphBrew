#ifndef GRAPHBREW_PARTITION_COMPACT_CSR_H_
#define GRAPHBREW_PARTITION_COMPACT_CSR_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

enum class GraphPartitionBalance
{
    kVertices,
    kOutgoingEdges,
    kTotalEdges,
};

inline const char *GraphPartitionBalanceName(GraphPartitionBalance balance)
{
    switch (balance)
    {
    case GraphPartitionBalance::kVertices:
        return "vertices";
    case GraphPartitionBalance::kOutgoingEdges:
        return "out";
    case GraphPartitionBalance::kTotalEdges:
        return "total";
    }
    throw std::logic_error("Unknown graph partition balance policy");
}

inline GraphPartitionBalance ParseGraphPartitionBalance(
    const std::string &value)
{
    if (value == "vertices" || value == "vertex")
        return GraphPartitionBalance::kVertices;
    if (value == "out" || value == "outgoing")
        return GraphPartitionBalance::kOutgoingEdges;
    if (value == "total" || value == "edges")
        return GraphPartitionBalance::kTotalEdges;
    throw std::invalid_argument(
        "Unknown partition balance policy '" + value +
        "' (expected vertices, out, or total)");
}

template <typename NodeID_>
struct CompactGraphPartition
{
    using Offset = std::uint64_t;

    std::uint32_t id = 0;
    NodeID_ vertex_begin = 0;
    NodeID_ vertex_end = 0;
    std::vector<Offset> out_offsets;
    std::vector<NodeID_> out_neighbors;
    std::vector<Offset> in_offsets;
    std::vector<NodeID_> in_neighbors;
    std::vector<NodeID_> ghost_globals;
    std::vector<std::uint32_t> ghost_owners;
    std::uint64_t balance_weight = 0;
    std::uint64_t remote_out_edges = 0;
    std::uint64_t remote_in_edges = 0;
    bool symmetric = false;

    std::size_t vertex_count() const
    {
        return static_cast<std::size_t>(vertex_end - vertex_begin);
    }

    NodeID_ global_vertex(std::size_t local_vertex) const
    {
        if (local_vertex >= vertex_count())
            throw std::out_of_range("Partition-local vertex is out of range");
        return static_cast<NodeID_>(vertex_begin + local_vertex);
    }

    std::size_t local_vertex(NodeID_ global_vertex_id) const
    {
        if (
            global_vertex_id < vertex_begin ||
            global_vertex_id >= vertex_end)
        {
            throw std::out_of_range(
                "Global vertex is not owned by this partition");
        }
        return static_cast<std::size_t>(
            global_vertex_id - vertex_begin);
    }

    NodeID_ global_vertex_from_slot(NodeID_ local_id) const
    {
        if constexpr (std::is_signed<NodeID_>::value)
        {
            if (local_id < 0)
                throw std::out_of_range(
                    "Partition-local vertex slot is negative");
        }
        const std::size_t slot =
            static_cast<std::size_t>(local_id);
        if (slot < vertex_count())
            return global_vertex(slot);
        const std::size_t ghost_index = slot - vertex_count();
        if (ghost_index >= ghost_globals.size())
            throw std::out_of_range(
                "Partition-local vertex slot is out of range");
        return ghost_globals[ghost_index];
    }

    std::size_t ghost_count() const
    {
        return ghost_owners.size();
    }

    NodeID_ ghost_global(std::size_t ghost_index) const
    {
        if (ghost_index >= ghost_count())
            throw std::out_of_range(
                "Partition ghost index is out of range");
        return ghost_globals[ghost_index];
    }

    NodeID_ ghost_local_id(std::size_t ghost_index) const
    {
        if (ghost_index >= ghost_count())
            throw std::out_of_range(
                "Partition ghost index is out of range");
        return static_cast<NodeID_>(
            vertex_count() + ghost_index);
    }

    std::uint32_t ghost_owner(std::size_t ghost_index) const
    {
        return ghost_owners.at(ghost_index);
    }

    std::uint64_t storage_bytes() const
    {
        return
            static_cast<std::uint64_t>(out_offsets.size()) *
                sizeof(Offset) +
            static_cast<std::uint64_t>(out_neighbors.size()) *
                sizeof(NodeID_) +
            static_cast<std::uint64_t>(in_offsets.size()) *
                sizeof(Offset) +
            static_cast<std::uint64_t>(in_neighbors.size()) *
                sizeof(NodeID_) +
            static_cast<std::uint64_t>(ghost_globals.size()) *
                sizeof(NodeID_) +
            static_cast<std::uint64_t>(ghost_owners.size()) *
                sizeof(std::uint32_t);
    }

    std::uint64_t ghost_metadata_bytes() const
    {
        return
            static_cast<std::uint64_t>(ghost_globals.size()) *
                sizeof(NodeID_) +
            static_cast<std::uint64_t>(ghost_owners.size()) *
                sizeof(std::uint32_t);
    }

    std::size_t incoming_edge_count() const
    {
        return symmetric ? out_neighbors.size() : in_neighbors.size();
    }

    const std::vector<Offset> &incoming_offsets() const
    {
        return symmetric ? out_offsets : in_offsets;
    }

    const std::vector<NodeID_> &incoming_neighbors() const
    {
        return symmetric ? out_neighbors : in_neighbors;
    }
};

template <typename NodeID_>
class PartitionedGraph
{
    static_assert(
        std::is_integral<NodeID_>::value,
        "PartitionedGraph requires an integral vertex identifier");

public:
    using Partition = CompactGraphPartition<NodeID_>;
    using Offset = typename Partition::Offset;

    class Neighborhood
    {
    public:
        class const_iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = NodeID_;
            using difference_type = std::ptrdiff_t;
            using pointer = void;
            using reference = NodeID_;

            const_iterator(
                const Partition &partition,
                const std::vector<NodeID_> &neighbors,
                Offset index)
                : partition_(&partition),
                  neighbors_(&neighbors),
                  index_(index)
            {
            }

            NodeID_ operator*() const
            {
                return partition_->global_vertex_from_slot(
                    (*neighbors_)[static_cast<std::size_t>(index_)]);
            }

            const_iterator &operator++()
            {
                ++index_;
                return *this;
            }

            bool operator==(const const_iterator &other) const
            {
                return partition_ == other.partition_ &&
                       neighbors_ == other.neighbors_ &&
                       index_ == other.index_;
            }

            bool operator!=(const const_iterator &other) const
            {
                return !(*this == other);
            }

        private:
            const Partition *partition_;
            const std::vector<NodeID_> *neighbors_;
            Offset index_;
        };

        Neighborhood(
            const Partition &partition,
            const std::vector<NodeID_> &neighbors,
            Offset begin,
            Offset end)
            : partition_(&partition),
              neighbors_(&neighbors),
              begin_(begin),
              end_(end)
        {
        }

        const_iterator begin() const
        {
            return const_iterator(
                *partition_, *neighbors_, begin_);
        }

        const_iterator end() const
        {
            return const_iterator(
                *partition_, *neighbors_, end_);
        }

        std::size_t size() const
        {
            return static_cast<std::size_t>(end_ - begin_);
        }

    private:
        const Partition *partition_;
        const std::vector<NodeID_> *neighbors_;
        Offset begin_;
        Offset end_;
    };

    PartitionedGraph() = default;
    PartitionedGraph(const PartitionedGraph &) = delete;
    PartitionedGraph &operator=(const PartitionedGraph &) = delete;
    PartitionedGraph(PartitionedGraph &&) = default;
    PartitionedGraph &operator=(PartitionedGraph &&) = default;

    template <typename GraphT>
    static PartitionedGraph Build(
        const GraphT &graph,
        std::size_t requested_partitions,
        GraphPartitionBalance balance =
            GraphPartitionBalance::kTotalEdges)
    {
        if (requested_partitions == 0)
            throw std::invalid_argument(
                "Graph partition count must be positive");
        if (graph.num_nodes() < 0)
            throw std::invalid_argument(
                "Cannot partition a graph with a negative vertex count");
        if (graph.num_edges_directed() < 0)
            throw std::invalid_argument(
                "Cannot partition a graph with a negative edge count");
        if (
            static_cast<std::uint64_t>(graph.num_nodes()) >
            static_cast<std::uint64_t>(
                std::numeric_limits<NodeID_>::max()))
        {
            throw std::overflow_error(
                "Graph vertex count does not fit the partition vertex type");
        }

        PartitionedGraph result;
        result.num_nodes_ = static_cast<NodeID_>(graph.num_nodes());
        result.num_edges_directed_ =
            static_cast<std::uint64_t>(graph.num_edges_directed());
        result.directed_ = graph.directed();
        result.balance_ = balance;
        const std::size_t num_nodes =
            static_cast<std::size_t>(graph.num_nodes());
        if (num_nodes == 0)
            return result;

        const std::size_t partition_count =
            std::min(requested_partitions, num_nodes);
        if (
            partition_count >
            std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(
                "Graph partition count exceeds the manifest identifier width");
        }

        std::vector<std::uint64_t> weights(num_nodes, 0);
        for (std::size_t index = 0; index < num_nodes; ++index)
        {
            const NodeID_ vertex = static_cast<NodeID_>(index);
            const std::uint64_t out_degree =
                CheckedDegree(graph.out_degree(vertex));
            const std::uint64_t in_degree =
                result.directed_
                    ? CheckedDegree(graph.in_degree(vertex))
                    : out_degree;
            switch (balance)
            {
            case GraphPartitionBalance::kVertices:
                weights[index] = 1;
                break;
            case GraphPartitionBalance::kOutgoingEdges:
                weights[index] = out_degree;
                break;
            case GraphPartitionBalance::kTotalEdges:
                weights[index] = result.directed_
                    ? CheckedAdd(out_degree, in_degree)
                    : out_degree;
                break;
            }
        }

        const auto ranges = BuildBalancedRanges(weights, partition_count);
        result.partitions_.resize(partition_count);
        result.cut_ends_.reserve(partition_count);
        std::vector<std::uint32_t> owner_by_vertex(num_nodes, 0);

        for (std::size_t id = 0; id < partition_count; ++id)
        {
            Partition &partition = result.partitions_[id];
            partition.id = static_cast<std::uint32_t>(id);
            partition.vertex_begin =
                static_cast<NodeID_>(ranges[id].first);
            partition.vertex_end =
                static_cast<NodeID_>(ranges[id].second);
            partition.symmetric = !result.directed_;
            partition.balance_weight = SumRange(
                weights, ranges[id].first, ranges[id].second);
            result.cut_ends_.push_back(partition.vertex_end);
            std::fill(
                owner_by_vertex.begin() +
                    static_cast<std::ptrdiff_t>(ranges[id].first),
                owner_by_vertex.begin() +
                    static_cast<std::ptrdiff_t>(ranges[id].second),
                partition.id);

            const std::size_t owned = partition.vertex_count();
            partition.out_offsets.resize(owned + 1, 0);
            if (result.directed_)
                partition.in_offsets.resize(owned + 1, 0);

            for (std::size_t local = 0; local < owned; ++local)
            {
                const NodeID_ vertex = partition.global_vertex(local);
                partition.out_offsets[local + 1] = CheckedAdd(
                    partition.out_offsets[local],
                    CheckedDegree(graph.out_degree(vertex)));
                if (result.directed_)
                {
                    partition.in_offsets[local + 1] = CheckedAdd(
                        partition.in_offsets[local],
                        CheckedDegree(graph.in_degree(vertex)));
                }
            }

            CheckVectorSize(partition.out_offsets.back());
            partition.out_neighbors.resize(
                static_cast<std::size_t>(
                    partition.out_offsets.back()));
            if (result.directed_)
            {
                CheckVectorSize(partition.in_offsets.back());
                partition.in_neighbors.resize(
                    static_cast<std::size_t>(
                        partition.in_offsets.back()));
            }
        }

        std::atomic<bool> invalid_neighbor(false);
        #pragma omp parallel for schedule(dynamic, 256)
        for (std::int64_t raw_vertex = 0;
             raw_vertex < static_cast<std::int64_t>(num_nodes);
             ++raw_vertex)
        {
            const NodeID_ vertex = static_cast<NodeID_>(raw_vertex);
            const std::size_t owner =
                owner_by_vertex[static_cast<std::size_t>(vertex)];
            Partition &partition = result.partitions_[owner];
            const std::size_t local =
                static_cast<std::size_t>(
                    vertex - partition.vertex_begin);

            Offset out_offset = partition.out_offsets[local];
            for (const auto raw_neighbor : graph.out_neigh(vertex))
            {
                const NodeID_ neighbor =
                    static_cast<NodeID_>(raw_neighbor);
                if (!ValidVertex(neighbor, result.num_nodes_))
                {
                    invalid_neighbor.store(true);
                    continue;
                }
                partition.out_neighbors[
                    static_cast<std::size_t>(out_offset++)] = neighbor;
            }
            if (out_offset != partition.out_offsets[local + 1])
                invalid_neighbor.store(true);

            if (result.directed_)
            {
                Offset in_offset = partition.in_offsets[local];
                for (const auto raw_neighbor : graph.in_neigh(vertex))
                {
                    const NodeID_ neighbor =
                        static_cast<NodeID_>(raw_neighbor);
                    if (!ValidVertex(neighbor, result.num_nodes_))
                    {
                        invalid_neighbor.store(true);
                        continue;
                    }
                    partition.in_neighbors[
                        static_cast<std::size_t>(in_offset++)] = neighbor;
                }
                if (in_offset != partition.in_offsets[local + 1])
                    invalid_neighbor.store(true);
            }
        }
        if (invalid_neighbor.load())
            throw std::invalid_argument(
                "Graph adjacency contains an invalid vertex or degree");

        std::vector<std::uint32_t> ghost_stamp(num_nodes, 0);
        std::vector<NodeID_> ghost_slot_by_vertex(num_nodes, 0);
        std::size_t total_ghosts = 0;
        std::uint64_t total_out_edges = 0;
        std::uint64_t total_in_edges = 0;
        std::uint64_t remote_out_edges = 0;
        std::uint64_t remote_in_edges = 0;
        std::uint64_t max_weight = 0;
        std::uint64_t total_weight = 0;
        for (Partition &partition : result.partitions_)
        {
            const std::uint32_t stamp = partition.id + 1;
            const auto record_ghost =
                [&](NodeID_ neighbor)
                {
                    const std::size_t owner =
                        owner_by_vertex[
                            static_cast<std::size_t>(neighbor)];
                    if (owner == partition.id)
                        return;
                    const std::size_t index =
                        static_cast<std::size_t>(neighbor);
                    if (ghost_stamp[index] == stamp)
                        return;
                    ghost_stamp[index] = stamp;
                    partition.ghost_globals.push_back(neighbor);
                };

            for (const NodeID_ neighbor : partition.out_neighbors)
            {
                if (
                    owner_by_vertex[
                        static_cast<std::size_t>(neighbor)] !=
                    partition.id)
                {
                    ++partition.remote_out_edges;
                }
                record_ghost(neighbor);
            }
            if (result.directed_)
            {
                for (const NodeID_ neighbor : partition.in_neighbors)
                {
                    if (
                        owner_by_vertex[
                            static_cast<std::size_t>(neighbor)] !=
                        partition.id)
                    {
                        ++partition.remote_in_edges;
                    }
                    record_ghost(neighbor);
                }
            }
            else
            {
                partition.remote_in_edges = partition.remote_out_edges;
            }

            const std::size_t owned = partition.vertex_count();
            const std::size_t ghost_count = partition.ghost_globals.size();
            const std::uint64_t local_slot_count = CheckedAdd(
                owned, ghost_count);
            if (
                local_slot_count >
                static_cast<std::uint64_t>(
                    std::numeric_limits<NodeID_>::max()))
            {
                throw std::overflow_error(
                    "Partition local and ghost vertices exceed the local ID width");
            }

            partition.ghost_owners.resize(ghost_count);
            for (std::size_t index = 0;
                 index < ghost_count; ++index)
            {
                const NodeID_ global_id =
                    partition.ghost_globals[index];
                const NodeID_ local_id =
                    static_cast<NodeID_>(owned + index);
                ghost_slot_by_vertex[
                    static_cast<std::size_t>(global_id)] = local_id;
                partition.ghost_owners[index] =
                    owner_by_vertex[
                        static_cast<std::size_t>(global_id)];
            }

            const auto localize_neighbor =
                [&](NodeID_ global_id)
                {
                    if (
                        owner_by_vertex[
                            static_cast<std::size_t>(global_id)] ==
                        partition.id)
                    {
                        return static_cast<NodeID_>(
                            global_id - partition.vertex_begin);
                    }
                    return ghost_slot_by_vertex[
                        static_cast<std::size_t>(global_id)];
                };
            for (NodeID_ &neighbor : partition.out_neighbors)
                neighbor = localize_neighbor(neighbor);
            if (result.directed_)
            {
                for (NodeID_ &neighbor : partition.in_neighbors)
                    neighbor = localize_neighbor(neighbor);
            }

            total_ghosts += partition.ghost_count();
            total_out_edges = CheckedAdd(
                total_out_edges, partition.out_neighbors.size());
            total_in_edges = CheckedAdd(
                total_in_edges,
                result.directed_
                    ? partition.in_neighbors.size()
                    : partition.out_neighbors.size());
            remote_out_edges = CheckedAdd(
                remote_out_edges, partition.remote_out_edges);
            remote_in_edges = CheckedAdd(
                remote_in_edges, partition.remote_in_edges);
            max_weight = std::max(
                max_weight, partition.balance_weight);
            total_weight = CheckedAdd(
                total_weight, partition.balance_weight);
        }

        if (total_out_edges != result.num_edges_directed_)
            throw std::logic_error(
                "Partitioned outgoing edge count does not match the graph");
        if (total_in_edges != result.num_edges_directed_)
            throw std::logic_error(
                "Partitioned incoming edge count does not match the graph");

        result.total_ghosts_ = total_ghosts;
        result.remote_out_edges_ = remote_out_edges;
        result.remote_in_edges_ = remote_in_edges;
        if (total_weight == 0)
        {
            result.max_balance_imbalance_ = 1.0;
        }
        else
        {
            const double average =
                static_cast<double>(total_weight) /
                static_cast<double>(partition_count);
            result.max_balance_imbalance_ =
                static_cast<double>(max_weight) / average;
        }
        result.owner_by_vertex_ = std::move(owner_by_vertex);
        return result;
    }

    std::size_t num_partitions() const
    {
        return partitions_.size();
    }

    std::int64_t num_nodes() const
    {
        return static_cast<std::int64_t>(num_nodes_);
    }

    std::int64_t num_edges_directed() const
    {
        return static_cast<std::int64_t>(num_edges_directed_);
    }

    bool directed() const
    {
        return directed_;
    }

    GraphPartitionBalance balance() const
    {
        return balance_;
    }

    const std::vector<Partition> &partitions() const
    {
        return partitions_;
    }

    const Partition &partition(std::size_t id) const
    {
        return partitions_.at(id);
    }

    std::size_t owner(NodeID_ vertex) const
    {
        CheckVertex(vertex);
        return owner_by_vertex_.empty()
            ? OwnerFromCuts(cut_ends_, vertex)
            : owner_by_vertex_[static_cast<std::size_t>(vertex)];
    }

    const Partition &partition_for(NodeID_ vertex) const
    {
        return partitions_[owner(vertex)];
    }

    std::int64_t out_degree(NodeID_ vertex) const
    {
        const Partition &owned = partition_for(vertex);
        const std::size_t local = owned.local_vertex(vertex);
        return static_cast<std::int64_t>(
            owned.out_offsets[local + 1] -
            owned.out_offsets[local]);
    }

    std::int64_t in_degree(NodeID_ vertex) const
    {
        const Partition &owned = partition_for(vertex);
        const std::size_t local = owned.local_vertex(vertex);
        const auto &offsets = owned.incoming_offsets();
        return static_cast<std::int64_t>(
            offsets[local + 1] - offsets[local]);
    }

    Neighborhood out_neigh(NodeID_ vertex) const
    {
        const Partition &owned = partition_for(vertex);
        const std::size_t local = owned.local_vertex(vertex);
        return Neighborhood(
            owned,
            owned.out_neighbors,
            owned.out_offsets[local],
            owned.out_offsets[local + 1]);
    }

    Neighborhood in_neigh(NodeID_ vertex) const
    {
        const Partition &owned = partition_for(vertex);
        const std::size_t local = owned.local_vertex(vertex);
        const auto &offsets = owned.incoming_offsets();
        const auto &neighbors = owned.incoming_neighbors();
        return Neighborhood(
            owned, neighbors, offsets[local], offsets[local + 1]);
    }

    std::size_t total_ghosts() const
    {
        return total_ghosts_;
    }

    std::uint64_t remote_out_edges() const
    {
        return remote_out_edges_;
    }

    std::uint64_t remote_in_edges() const
    {
        return remote_in_edges_;
    }

    double remote_out_edge_fraction() const
    {
        if (num_edges_directed_ == 0)
            return 0.0;
        return static_cast<double>(remote_out_edges_) /
               static_cast<double>(num_edges_directed_);
    }

    double remote_in_edge_fraction() const
    {
        if (num_edges_directed_ == 0)
            return 0.0;
        return static_cast<double>(remote_in_edges_) /
               static_cast<double>(num_edges_directed_);
    }

    std::uint64_t total_storage_bytes() const
    {
        std::uint64_t total = 0;
        for (const Partition &part : partitions_)
            total = CheckedAdd(total, part.storage_bytes());
        return total;
    }

    std::uint64_t max_shard_storage_bytes() const
    {
        std::uint64_t maximum = 0;
        for (const Partition &part : partitions_)
            maximum = std::max(maximum, part.storage_bytes());
        return maximum;
    }

    std::uint64_t total_ghost_metadata_bytes() const
    {
        std::uint64_t total = 0;
        for (const Partition &part : partitions_)
            total = CheckedAdd(total, part.ghost_metadata_bytes());
        return total;
    }

    double ghost_metadata_fraction() const
    {
        const std::uint64_t total = total_storage_bytes();
        if (total == 0)
            return 0.0;
        return static_cast<double>(total_ghost_metadata_bytes()) /
               static_cast<double>(total);
    }

    double max_vertex_imbalance() const
    {
        return MaxImbalance(
            static_cast<std::uint64_t>(num_nodes_),
            [](const Partition &part)
            {
                return static_cast<std::uint64_t>(part.vertex_count());
            });
    }

    double max_out_edge_imbalance() const
    {
        return MaxImbalance(
            num_edges_directed_,
            [](const Partition &part)
            {
                return static_cast<std::uint64_t>(
                    part.out_neighbors.size());
            });
    }

    double max_in_edge_imbalance() const
    {
        return MaxImbalance(
            num_edges_directed_,
            [](const Partition &part)
            {
                return static_cast<std::uint64_t>(
                    part.incoming_edge_count());
            });
    }

    double max_shard_storage_imbalance() const
    {
        return MaxImbalance(
            total_storage_bytes(),
            [](const Partition &part)
            {
                return part.storage_bytes();
            });
    }

    double max_remote_out_fraction() const
    {
        double maximum = 0.0;
        for (const Partition &part : partitions_)
        {
            if (part.out_neighbors.empty())
                continue;
            maximum = std::max(
                maximum,
                static_cast<double>(part.remote_out_edges) /
                    static_cast<double>(part.out_neighbors.size()));
        }
        return maximum;
    }

    double max_remote_in_fraction() const
    {
        double maximum = 0.0;
        for (const Partition &part : partitions_)
        {
            const std::size_t incoming = part.incoming_edge_count();
            if (incoming == 0)
                continue;
            maximum = std::max(
                maximum,
                static_cast<double>(part.remote_in_edges) /
                    static_cast<double>(incoming));
        }
        return maximum;
    }

    double max_balance_imbalance() const
    {
        return max_balance_imbalance_;
    }

    void PrintStats(std::ostream &out = std::cout) const
    {
        const std::uint64_t total_storage = total_storage_bytes();
        const std::uint64_t max_shard_storage =
            max_shard_storage_bytes();
        out << "Partitioned graph has " << num_partitions()
            << " compact shards, balance="
            << GraphPartitionBalanceName(balance_)
            << ", max imbalance=" << max_balance_imbalance_
            << ", remote out=" << remote_out_edge_fraction()
            << ", remote in=" << remote_in_edge_fraction()
            << ", max remote out=" << max_remote_out_fraction()
            << ", max remote in=" << max_remote_in_fraction()
            << ", ghosts=" << total_ghosts_
            << ", ghost bytes=" << total_ghost_metadata_bytes()
            << ", ghost fraction=" << ghost_metadata_fraction()
            << ", total bytes=" << total_storage
            << ", max shard bytes=" << max_shard_storage
            << ", vertex imbalance=" << max_vertex_imbalance()
            << ", out-edge imbalance=" << max_out_edge_imbalance()
            << ", in-edge imbalance=" << max_in_edge_imbalance()
            << ", storage imbalance=" << max_shard_storage_imbalance()
            << std::endl;
        for (const Partition &part : partitions_)
        {
            out << "  shard " << part.id << ": vertices ["
                << part.vertex_begin << ", " << part.vertex_end
                << "), out_edges=" << part.out_neighbors.size()
                << ", in_edges=" << part.incoming_edge_count()
                << ", remote_out=" << part.remote_out_edges
                << ", remote_in=" << part.remote_in_edges
                << ", ghosts=" << part.ghost_count()
                << ", bytes=" << part.storage_bytes()
                << ", weight=" << part.balance_weight << std::endl;
        }
    }

    template <typename GraphT>
    void VerifyExact(const GraphT &graph) const
    {
        if (graph.num_nodes() != num_nodes())
            throw std::logic_error(
                "Partition verification vertex count mismatch");
        if (graph.num_edges_directed() != num_edges_directed())
            throw std::logic_error(
                "Partition verification edge count mismatch");
        if (partitions_.empty() != (num_nodes_ == 0))
            throw std::logic_error(
                "Partition verification empty-graph mismatch");

        NodeID_ expected_begin = 0;
        for (const Partition &part : partitions_)
        {
            if (part.vertex_begin != expected_begin)
                throw std::logic_error(
                    "Partition ownership ranges are not contiguous");
            if (part.vertex_begin >= part.vertex_end)
                throw std::logic_error(
                    "Partition ownership range is empty");
            expected_begin = part.vertex_end;

            VerifyOffsets(
                part.out_offsets, part.out_neighbors.size(),
                part.vertex_count());
            if (directed_)
            {
                VerifyOffsets(
                    part.in_offsets, part.in_neighbors.size(),
                    part.vertex_count());
            }

            for (std::size_t local = 0;
                 local < part.vertex_count(); ++local)
            {
                const NodeID_ vertex = part.global_vertex(local);
                VerifyNeighborhood(
                    graph.out_neigh(vertex),
                    out_neigh(vertex));
                if (directed_)
                {
                    VerifyNeighborhood(
                        graph.in_neigh(vertex),
                        in_neigh(vertex));
                }
            }

            if (part.ghost_globals.size() != part.ghost_count())
                throw std::logic_error(
                    "Partition ghost metadata size is inconsistent");
            for (std::size_t index = 0;
                 index < part.ghost_count(); ++index)
            {
                const NodeID_ global_id = part.ghost_global(index);
                const NodeID_ local_id = part.ghost_local_id(index);
                const std::uint32_t ghost_owner =
                    part.ghost_owner(index);
                if (owner(global_id) != ghost_owner)
                    throw std::logic_error(
                        "Partition ghost owner metadata is inconsistent");
                if (ghost_owner == part.id)
                    throw std::logic_error(
                        "Partition ghost is locally owned");
                if (
                    part.global_vertex_from_slot(local_id) !=
                    global_id)
                {
                    throw std::logic_error(
                        "Partition ghost local-slot metadata is inconsistent");
                }
            }
        }
        if (expected_begin != num_nodes_)
            throw std::logic_error(
                "Partition ownership ranges do not cover every vertex");
    }

private:
    template <typename Measure>
    double MaxImbalance(
        std::uint64_t total,
        Measure measure) const
    {
        if (partitions_.empty() || total == 0)
            return 1.0;
        std::uint64_t maximum = 0;
        for (const Partition &part : partitions_)
            maximum = std::max(maximum, measure(part));
        const double average =
            static_cast<double>(total) /
            static_cast<double>(partitions_.size());
        return static_cast<double>(maximum) / average;
    }

    static std::uint64_t CheckedDegree(std::int64_t degree)
    {
        if (degree < 0)
            throw std::invalid_argument(
                "Graph contains a negative vertex degree");
        return static_cast<std::uint64_t>(degree);
    }

    static std::uint64_t CheckedAdd(
        std::uint64_t lhs,
        std::uint64_t rhs)
    {
        if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
            throw std::overflow_error(
                "Graph partition size arithmetic overflow");
        return lhs + rhs;
    }

    static void CheckVectorSize(std::uint64_t size)
    {
        if (
            size > std::numeric_limits<std::size_t>::max() ||
            size > static_cast<std::uint64_t>(
                       std::numeric_limits<std::ptrdiff_t>::max()))
        {
            throw std::overflow_error(
                "Graph partition adjacency exceeds host address space");
        }
    }

    static bool ValidVertex(NodeID_ vertex, NodeID_ num_nodes)
    {
        if constexpr (std::is_signed<NodeID_>::value)
        {
            if (vertex < 0)
                return false;
        }
        return vertex < num_nodes;
    }

    void CheckVertex(NodeID_ vertex) const
    {
        if (!ValidVertex(vertex, num_nodes_))
            throw std::out_of_range(
                "Graph partition vertex is out of range");
    }

    static std::size_t OwnerFromCuts(
        const std::vector<NodeID_> &cut_ends,
        NodeID_ vertex)
    {
        const auto it = std::upper_bound(
            cut_ends.begin(), cut_ends.end(), vertex);
        if (it == cut_ends.end())
            throw std::out_of_range(
                "Graph partition owner lookup is out of range");
        return static_cast<std::size_t>(
            std::distance(cut_ends.begin(), it));
    }

    static std::uint64_t SumRange(
        const std::vector<std::uint64_t> &weights,
        std::size_t begin,
        std::size_t end)
    {
        std::uint64_t total = 0;
        for (std::size_t index = begin; index < end; ++index)
            total = CheckedAdd(total, weights[index]);
        return total;
    }

    static std::vector<std::pair<std::size_t, std::size_t>>
    BuildBalancedRanges(
        const std::vector<std::uint64_t> &weights,
        std::size_t partition_count)
    {
        const std::size_t num_nodes = weights.size();
        std::vector<std::uint64_t> prefix(num_nodes + 1, 0);
        for (std::size_t index = 0; index < num_nodes; ++index)
        {
            prefix[index + 1] =
                CheckedAdd(prefix[index], weights[index]);
        }

        if (prefix.back() == 0)
        {
            for (std::size_t index = 0; index < num_nodes; ++index)
                prefix[index + 1] = index + 1;
        }

        std::vector<std::size_t> cuts(partition_count + 1, 0);
        cuts.back() = num_nodes;
        const std::uint64_t total = prefix.back();
        const std::uint64_t quotient =
            total / partition_count;
        const std::uint64_t remainder =
            total % partition_count;

        for (std::size_t part = 1;
             part < partition_count; ++part)
        {
            const std::size_t min_cut = cuts[part - 1] + 1;
            const std::size_t max_cut =
                num_nodes - (partition_count - part);
            const std::uint64_t target =
                quotient * part +
                (remainder * part) / partition_count;

            auto lower = prefix.begin() +
                         static_cast<std::ptrdiff_t>(min_cut);
            auto upper = prefix.begin() +
                         static_cast<std::ptrdiff_t>(max_cut + 1);
            auto it = std::lower_bound(lower, upper, target);
            std::size_t candidate =
                it == upper
                    ? max_cut
                    : static_cast<std::size_t>(
                          std::distance(prefix.begin(), it));
            candidate = std::max(min_cut, std::min(candidate, max_cut));

            if (candidate > min_cut)
            {
                const std::size_t previous = candidate - 1;
                const std::uint64_t candidate_distance =
                    prefix[candidate] >= target
                        ? prefix[candidate] - target
                        : target - prefix[candidate];
                const std::uint64_t previous_distance =
                    prefix[previous] >= target
                        ? prefix[previous] - target
                        : target - prefix[previous];
                if (previous_distance <= candidate_distance)
                    candidate = previous;
            }
            cuts[part] = candidate;
        }

        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        ranges.reserve(partition_count);
        for (std::size_t part = 0;
             part < partition_count; ++part)
        {
            if (cuts[part] >= cuts[part + 1])
                throw std::logic_error(
                    "Balanced graph partitioning produced an empty shard");
            ranges.emplace_back(cuts[part], cuts[part + 1]);
        }
        return ranges;
    }

    static void VerifyOffsets(
        const std::vector<Offset> &offsets,
        std::size_t neighbor_count,
        std::size_t vertex_count)
    {
        if (offsets.size() != vertex_count + 1)
            throw std::logic_error(
                "Partition CSR offset count is invalid");
        if (offsets.empty() || offsets.front() != 0)
            throw std::logic_error(
                "Partition CSR must begin at offset zero");
        for (std::size_t index = 1; index < offsets.size(); ++index)
        {
            if (offsets[index] < offsets[index - 1])
                throw std::logic_error(
                    "Partition CSR offsets are not monotonic");
        }
        if (offsets.back() != neighbor_count)
            throw std::logic_error(
                "Partition CSR edge count is inconsistent");
    }

    template <typename ExpectedRange, typename ActualRange>
    static void VerifyNeighborhood(
        ExpectedRange expected,
        ActualRange actual)
    {
        auto actual_it = actual.begin();
        const auto actual_end = actual.end();
        for (const auto raw_neighbor : expected)
        {
            if (
                actual_it == actual_end ||
                *actual_it != static_cast<NodeID_>(raw_neighbor))
            {
                throw std::logic_error(
                    "Partition CSR neighborhood differs from the source graph");
            }
            ++actual_it;
        }
        if (actual_it != actual_end)
            throw std::logic_error(
                "Partition CSR neighborhood length differs from the source graph");
    }

    NodeID_ num_nodes_ = 0;
    std::uint64_t num_edges_directed_ = 0;
    bool directed_ = false;
    GraphPartitionBalance balance_ =
        GraphPartitionBalance::kTotalEdges;
    std::vector<Partition> partitions_;
    std::vector<NodeID_> cut_ends_;
    std::vector<std::uint32_t> owner_by_vertex_;
    std::size_t total_ghosts_ = 0;
    std::uint64_t remote_out_edges_ = 0;
    std::uint64_t remote_in_edges_ = 0;
    double max_balance_imbalance_ = 1.0;
};

#endif // GRAPHBREW_PARTITION_COMPACT_CSR_H_
