#ifndef GRAPHBREW_PARTITION_RUNTIME_TRAFFIC_H_
#define GRAPHBREW_PARTITION_RUNTIME_TRAFFIC_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graphbrew {
namespace partition {

inline std::uint64_t CheckedTrafficAdd(
    std::uint64_t lhs,
    std::uint64_t rhs)
{
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
        throw std::overflow_error(
            "partition runtime traffic addition overflow");
    return lhs + rhs;
}

inline std::uint64_t CheckedTrafficMultiply(
    std::uint64_t count,
    std::uint64_t bytes)
{
    if (
        count != 0 &&
        bytes > std::numeric_limits<std::uint64_t>::max() / count)
    {
        throw std::overflow_error(
            "partition runtime traffic multiplication overflow");
    }
    return count * bytes;
}

struct ShardHaloProjection
{
    std::size_t shard_id = 0;
    std::uint64_t ghost_slots = 0;
    std::uint64_t bfs_bytes_per_superstep = 0;
    std::uint64_t pr_bytes_per_iteration = 0;
    std::uint64_t cc_bytes_per_iteration = 0;
    std::uint64_t spmv_initial_bytes = 0;
};

struct HaloProjection
{
    std::vector<ShardHaloProjection> shards;
    std::uint64_t ghost_slots = 0;
    std::uint64_t bfs_bytes_per_superstep = 0;
    std::uint64_t pr_bytes_per_iteration = 0;
    std::uint64_t cc_bytes_per_iteration = 0;
    std::uint64_t spmv_initial_bytes = 0;
};

template <typename PartitionedGraphT>
HaloProjection BuildHaloProjection(
    const PartitionedGraphT &graph)
{
    HaloProjection projection;
    projection.shards.reserve(graph.num_partitions());
    for (std::size_t shard_id = 0;
         shard_id < graph.num_partitions(); ++shard_id)
    {
        const auto &shard = graph.partition(shard_id);
        const std::uint64_t ghosts =
            static_cast<std::uint64_t>(shard.ghost_count());
        ShardHaloProjection item;
        item.shard_id = shard_id;
        item.ghost_slots = ghosts;
        item.bfs_bytes_per_superstep =
            CheckedTrafficMultiply(
                CheckedTrafficMultiply(ghosts, 2),
                sizeof(std::uint32_t));
        item.pr_bytes_per_iteration =
            CheckedTrafficMultiply(
                ghosts, sizeof(std::uint32_t));
        item.cc_bytes_per_iteration =
            CheckedTrafficMultiply(
                ghosts, sizeof(std::uint32_t));
        item.spmv_initial_bytes =
            CheckedTrafficMultiply(
                ghosts, sizeof(std::uint32_t));
        projection.shards.push_back(item);
        projection.ghost_slots = CheckedTrafficAdd(
            projection.ghost_slots, item.ghost_slots);
        projection.bfs_bytes_per_superstep = CheckedTrafficAdd(
            projection.bfs_bytes_per_superstep,
            item.bfs_bytes_per_superstep);
        projection.pr_bytes_per_iteration = CheckedTrafficAdd(
            projection.pr_bytes_per_iteration,
            item.pr_bytes_per_iteration);
        projection.cc_bytes_per_iteration = CheckedTrafficAdd(
            projection.cc_bytes_per_iteration,
            item.cc_bytes_per_iteration);
        projection.spmv_initial_bytes = CheckedTrafficAdd(
            projection.spmv_initial_bytes,
            item.spmv_initial_bytes);
    }
    return projection;
}

struct BfsShardTraffic
{
    std::size_t shard_id = 0;
    std::uint64_t cpu_ghost_sync_values = 0;
    std::uint64_t cpu_ghost_sync_bytes = 0;
    std::uint64_t remote_parent_messages = 0;
    std::uint64_t remote_parent_bytes = 0;
    std::uint64_t halo_values = 0;
    std::uint64_t halo_bytes = 0;
};

struct BfsSuperstepTraffic
{
    std::size_t step = 0;
    std::string phase;
    std::vector<BfsShardTraffic> shards;
    std::uint64_t cpu_ghost_sync_values = 0;
    std::uint64_t cpu_ghost_sync_bytes = 0;
    std::uint64_t remote_parent_messages = 0;
    std::uint64_t remote_parent_bytes = 0;
    std::uint64_t halo_values = 0;
    std::uint64_t halo_bytes = 0;
};

struct BfsRuntimeTraffic
{
    HaloProjection projection;
    std::vector<BfsSuperstepTraffic> supersteps;
    std::uint64_t cpu_ghost_sync_values = 0;
    std::uint64_t cpu_ghost_sync_bytes = 0;
    std::uint64_t remote_parent_messages = 0;
    std::uint64_t remote_parent_bytes = 0;
    std::uint64_t halo_values = 0;
    std::uint64_t halo_bytes = 0;

    template <typename PartitionedGraphT>
    void initialize(const PartitionedGraphT &graph)
    {
        projection = BuildHaloProjection(graph);
        supersteps.clear();
        cpu_ghost_sync_values = 0;
        cpu_ghost_sync_bytes = 0;
        remote_parent_messages = 0;
        remote_parent_bytes = 0;
        halo_values = 0;
        halo_bytes = 0;
    }

    void record_superstep(
        std::size_t step,
        std::string phase,
        bool cpu_syncs_ghost_frontier,
        const std::vector<std::uint64_t> &remote_messages_by_shard)
    {
        if (remote_messages_by_shard.size() != projection.shards.size())
            throw std::invalid_argument(
                "BFS traffic remote-message vector has wrong shard count");
        BfsSuperstepTraffic event;
        event.step = step;
        event.phase = std::move(phase);
        event.shards.reserve(projection.shards.size());
        for (std::size_t shard_id = 0;
             shard_id < projection.shards.size(); ++shard_id)
        {
            const ShardHaloProjection &halo =
                projection.shards[shard_id];
            BfsShardTraffic shard;
            shard.shard_id = shard_id;
            if (cpu_syncs_ghost_frontier)
            {
                shard.cpu_ghost_sync_values = halo.ghost_slots;
                shard.cpu_ghost_sync_bytes =
                    CheckedTrafficMultiply(
                        halo.ghost_slots,
                        sizeof(std::uint8_t));
            }
            shard.remote_parent_messages =
                remote_messages_by_shard[shard_id];
            shard.remote_parent_bytes =
                CheckedTrafficMultiply(
                    shard.remote_parent_messages,
                    sizeof(std::int32_t));
            shard.halo_values =
                CheckedTrafficMultiply(halo.ghost_slots, 2);
            shard.halo_bytes =
                halo.bfs_bytes_per_superstep;
            event.cpu_ghost_sync_values = CheckedTrafficAdd(
                event.cpu_ghost_sync_values,
                shard.cpu_ghost_sync_values);
            event.cpu_ghost_sync_bytes = CheckedTrafficAdd(
                event.cpu_ghost_sync_bytes,
                shard.cpu_ghost_sync_bytes);
            event.remote_parent_messages = CheckedTrafficAdd(
                event.remote_parent_messages,
                shard.remote_parent_messages);
            event.remote_parent_bytes = CheckedTrafficAdd(
                event.remote_parent_bytes,
                shard.remote_parent_bytes);
            event.halo_values = CheckedTrafficAdd(
                event.halo_values,
                shard.halo_values);
            event.halo_bytes = CheckedTrafficAdd(
                event.halo_bytes,
                shard.halo_bytes);
            event.shards.push_back(shard);
        }
        cpu_ghost_sync_values = CheckedTrafficAdd(
            cpu_ghost_sync_values,
            event.cpu_ghost_sync_values);
        cpu_ghost_sync_bytes = CheckedTrafficAdd(
            cpu_ghost_sync_bytes,
            event.cpu_ghost_sync_bytes);
        remote_parent_messages = CheckedTrafficAdd(
            remote_parent_messages,
            event.remote_parent_messages);
        remote_parent_bytes = CheckedTrafficAdd(
            remote_parent_bytes,
            event.remote_parent_bytes);
        halo_values = CheckedTrafficAdd(
            halo_values,
            event.halo_values);
        halo_bytes = CheckedTrafficAdd(
            halo_bytes,
            event.halo_bytes);
        supersteps.push_back(std::move(event));
    }
};

} // namespace partition
} // namespace graphbrew

#endif // GRAPHBREW_PARTITION_RUNTIME_TRAFFIC_H_
