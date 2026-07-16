#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "graphbrew/partition/runtime_traffic.h"

namespace
{

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Shard
{
    std::size_t ghosts = 0;

    std::size_t ghost_count() const { return ghosts; }
};

struct Graph
{
    std::vector<Shard> shards;

    std::size_t num_partitions() const
    {
        return shards.size();
    }

    const Shard &partition(std::size_t index) const
    {
        return shards.at(index);
    }
};

} // namespace

int main()
{
    const Graph graph{{{2}, {1}}};
    const auto projection =
        graphbrew::partition::BuildGraphBloxHaloProjection(
            graph);
    Require(projection.ghost_slots == 3, "ghost total mismatch");
    Require(
        projection.bfs_bytes_per_superstep == 24,
        "BFS halo bytes mismatch");
    Require(
        projection.pr_bytes_per_iteration == 12 &&
            projection.cc_bytes_per_iteration == 12 &&
            projection.spmv_initial_bytes == 12,
        "word halo bytes mismatch");

    graphbrew::partition::BfsRuntimeTraffic traffic;
    traffic.initialize(graph);
    traffic.record_superstep(
        0, "p-bsp-td", false, {2, 1});
    traffic.record_superstep(
        1, "p-bsp-bu", true, {0, 0});
    Require(
        traffic.supersteps.size() == 2,
        "BFS traffic step count mismatch");
    Require(
        traffic.remote_parent_messages == 3 &&
            traffic.remote_parent_bytes == 12,
        "remote parent traffic mismatch");
    Require(
        traffic.cpu_ghost_sync_values == 3 &&
            traffic.cpu_ghost_sync_bytes == 3,
        "CPU ghost sync mismatch");
    Require(
        traffic.graphblox_halo_values == 12 &&
            traffic.graphblox_halo_bytes == 48,
        "GraphBlox BFS halo traffic mismatch");
    Require(
        traffic.supersteps[0].shards[0]
                .remote_parent_messages == 2 &&
            traffic.supersteps[1].shards[0]
                .cpu_ghost_sync_bytes == 2,
        "per-shard BFS traffic mismatch");
    return 0;
}
