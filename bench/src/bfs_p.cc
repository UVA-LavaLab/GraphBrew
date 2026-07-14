// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "benchmark.h"
#include "bfs_common.h"
#include "builder.h"
#include "command_line.h"
#include "graph_partition.h"
#include "pvector.h"
#include "timer.h"

using PartitionedBFSGraph = PartitionedGraph<NodeID>;

namespace
{

constexpr NodeID kNoParent = -1;

NodeID PendingParentSentinel()
{
    return std::numeric_limits<NodeID>::max();
}

struct PartitionBFSState
{
    explicit PartitionBFSState(
        const PartitionedBFSGraph::Partition &partition)
        : parent(partition.vertex_count(), kNoParent),
          frontier_slots(
              partition.vertex_count() + partition.ghost_count(), 0),
          pending_parent(
              partition.vertex_count(), PendingParentSentinel())
    {
    }

    std::vector<NodeID> parent;
    std::vector<std::uint8_t> frontier_slots;
    std::vector<NodeID> pending_parent;
};

struct OwnedVertex
{
    std::size_t partition = 0;
    std::size_t local = 0;
};

OwnedVertex LocateOwnedVertex(
    const PartitionedBFSGraph &graph,
    NodeID vertex)
{
    const std::size_t owner = graph.owner(vertex);
    return {
        owner,
        graph.partition(owner).local_vertex(vertex),
    };
}

bool AtomicWriteMin(NodeID &target, NodeID candidate)
{
    NodeID current = __atomic_load_n(&target, __ATOMIC_SEQ_CST);
    while (candidate < current)
    {
        const NodeID previous = current;
        if (__atomic_compare_exchange_n(
                &target,
                &current,
                candidate,
                false,
                __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST))
        {
            return previous == PendingParentSentinel();
        }
    }
    return false;
}

void SyncGhostFrontier(
    const PartitionedBFSGraph &graph,
    std::vector<PartitionBFSState> &states)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t raw_partition = 0;
         raw_partition <
             static_cast<std::int64_t>(graph.num_partitions());
         ++raw_partition)
    {
        const std::size_t partition_id =
            static_cast<std::size_t>(raw_partition);
        const auto &partition = graph.partition(partition_id);
        auto &state = states[partition_id];
        const std::size_t owned = partition.vertex_count();
        for (std::size_t ghost_index = 0;
             ghost_index < partition.ghost_count(); ++ghost_index)
        {
            const std::uint32_t owner =
                partition.ghost_owner(ghost_index);
            const auto &owner_partition = graph.partition(owner);
            const std::size_t owner_local =
                owner_partition.local_vertex(
                    partition.ghost_global(ghost_index));
            state.frontier_slots[owned + ghost_index] =
                states[owner].frontier_slots[owner_local];
        }
    }
}

std::vector<NodeID> TopDownRound(
    const PartitionedBFSGraph &graph,
    std::vector<PartitionBFSState> &states,
    const std::vector<NodeID> &frontier)
{
    std::vector<NodeID> touched;
    #pragma omp parallel
    {
        std::vector<NodeID> local_touched;
        #pragma omp for schedule(dynamic, 64) nowait
        for (std::int64_t frontier_index = 0;
             frontier_index <
                 static_cast<std::int64_t>(frontier.size());
             ++frontier_index)
        {
            const NodeID source =
                frontier[static_cast<std::size_t>(frontier_index)];
            const OwnedVertex source_location =
                LocateOwnedVertex(graph, source);
            const auto &source_partition =
                graph.partition(source_location.partition);
            const auto begin =
                source_partition.out_offsets[source_location.local];
            const auto end =
                source_partition.out_offsets[source_location.local + 1];
            for (auto edge = begin; edge < end; ++edge)
            {
                const NodeID slot =
                    source_partition.out_neighbors[
                        static_cast<std::size_t>(edge)];
                const NodeID destination =
                    source_partition.global_vertex_from_slot(slot);
                const std::size_t target_partition =
                    static_cast<std::size_t>(slot) <
                            source_partition.vertex_count()
                        ? source_location.partition
                        : source_partition.ghost_owner(
                              static_cast<std::size_t>(slot) -
                              source_partition.vertex_count());
                const std::size_t target_local =
                    graph.partition(target_partition)
                        .local_vertex(destination);
                auto &target_state = states[target_partition];
                if (target_state.parent[target_local] != kNoParent)
                    continue;
                if (AtomicWriteMin(
                        target_state.pending_parent[target_local],
                        source))
                {
                    local_touched.push_back(destination);
                }
            }
        }
        #pragma omp critical
        touched.insert(
            touched.end(),
            local_touched.begin(),
            local_touched.end());
    }
    return touched;
}

std::vector<NodeID> ApplyTopDownDiscoveries(
    const PartitionedBFSGraph &graph,
    std::vector<PartitionBFSState> &states,
    const std::vector<NodeID> &touched,
    std::int64_t &scout_count)
{
    std::vector<NodeID> discovered(touched.size());
    scout_count = 0;
    #pragma omp parallel for reduction(+ : scout_count)
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(touched.size());
         ++index)
    {
        const NodeID vertex =
            touched[static_cast<std::size_t>(index)];
        const OwnedVertex location =
            LocateOwnedVertex(graph, vertex);
        auto &state = states[location.partition];
        state.parent[location.local] =
            state.pending_parent[location.local];
        state.pending_parent[location.local] =
            PendingParentSentinel();
        discovered[static_cast<std::size_t>(index)] = vertex;
        scout_count += graph.out_degree(vertex);
    }
    return discovered;
}

std::vector<NodeID> BottomUpRound(
    const PartitionedBFSGraph &graph,
    std::vector<PartitionBFSState> &states,
    std::int64_t &scout_count)
{
    std::vector<std::vector<NodeID>> discovered_by_partition(
        graph.num_partitions());
    scout_count = 0;
    #pragma omp parallel for reduction(+ : scout_count) schedule(dynamic, 1)
    for (std::int64_t raw_partition = 0;
         raw_partition <
             static_cast<std::int64_t>(graph.num_partitions());
         ++raw_partition)
    {
        const std::size_t partition_id =
            static_cast<std::size_t>(raw_partition);
        const auto &partition = graph.partition(partition_id);
        auto &state = states[partition_id];
        auto &local_discovered =
            discovered_by_partition[partition_id];
        for (std::size_t local = 0;
             local < partition.vertex_count(); ++local)
        {
            if (state.parent[local] != kNoParent)
                continue;

            const auto &offsets = partition.incoming_offsets();
            const auto &neighbors = partition.incoming_neighbors();
            for (auto edge = offsets[local];
                 edge < offsets[local + 1]; ++edge)
            {
                const NodeID predecessor_slot =
                    neighbors[static_cast<std::size_t>(edge)];
                if (
                    state.frontier_slots[
                        static_cast<std::size_t>(
                            predecessor_slot)] == 0)
                {
                    continue;
                }
                state.parent[local] =
                    partition.global_vertex_from_slot(
                        predecessor_slot);
                const NodeID vertex =
                    partition.global_vertex(local);
                local_discovered.push_back(vertex);
                scout_count += static_cast<std::int64_t>(
                    partition.out_offsets[local + 1] -
                    partition.out_offsets[local]);
                break;
            }
        }
    }

    std::size_t discovered_count = 0;
    for (const auto &local : discovered_by_partition)
        discovered_count += local.size();
    std::vector<NodeID> discovered;
    discovered.reserve(discovered_count);
    for (const auto &local : discovered_by_partition)
        discovered.insert(discovered.end(), local.begin(), local.end());
    return discovered;
}

void AdvanceFrontier(
    const PartitionedBFSGraph &graph,
    std::vector<PartitionBFSState> &states,
    const std::vector<NodeID> &current,
    const std::vector<NodeID> &next)
{
    #pragma omp parallel for
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(current.size());
         ++index)
    {
        const OwnedVertex location = LocateOwnedVertex(
            graph, current[static_cast<std::size_t>(index)]);
        states[location.partition]
            .frontier_slots[location.local] = 0;
    }
    #pragma omp parallel for
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(next.size());
         ++index)
    {
        const OwnedVertex location = LocateOwnedVertex(
            graph, next[static_cast<std::size_t>(index)]);
        states[location.partition]
            .frontier_slots[location.local] = 1;
    }
}

pvector<NodeID> GatherParents(
    const PartitionedBFSGraph &graph,
    const std::vector<PartitionBFSState> &states)
{
    pvector<NodeID> parent(graph.num_nodes());
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t raw_partition = 0;
         raw_partition <
             static_cast<std::int64_t>(graph.num_partitions());
         ++raw_partition)
    {
        const std::size_t partition_id =
            static_cast<std::size_t>(raw_partition);
        const auto &partition = graph.partition(partition_id);
        const auto &state = states[partition_id];
        for (std::size_t local = 0;
             local < partition.vertex_count(); ++local)
        {
            parent[partition.global_vertex(local)] =
                state.parent[local];
        }
    }
    return parent;
}

} // namespace

pvector<NodeID> DOBFSPartitioned(
    const PartitionedBFSGraph &graph,
    NodeID source,
    bool logging_enabled = false,
    int alpha = 15,
    int beta = 18)
{
    if (logging_enabled)
        PrintStep("Source", static_cast<std::int64_t>(source));

    std::vector<PartitionBFSState> states;
    states.reserve(graph.num_partitions());
    for (const auto &partition : graph.partitions())
        states.emplace_back(partition);

    const OwnedVertex source_location =
        LocateOwnedVertex(graph, source);
    states[source_location.partition].parent[source_location.local] =
        source;
    states[source_location.partition]
        .frontier_slots[source_location.local] = 1;

    std::vector<NodeID> frontier{source};
    std::int64_t edges_to_check = graph.num_edges_directed();
    std::int64_t scout_count = graph.out_degree(source);
    bool bottom_up = false;
    Timer timer;

    while (!frontier.empty())
    {
        if (!bottom_up && scout_count > edges_to_check / alpha)
            bottom_up = true;

        const std::size_t old_awake_count = frontier.size();
        std::vector<NodeID> next;
        timer.Start();
        if (bottom_up)
        {
            SyncGhostFrontier(graph, states);
            next = BottomUpRound(graph, states, scout_count);
        }
        else
        {
            edges_to_check =
                std::max<std::int64_t>(
                    0, edges_to_check - scout_count);
            const std::vector<NodeID> touched =
                TopDownRound(graph, states, frontier);
            next = ApplyTopDownDiscoveries(
                graph, states, touched, scout_count);
        }
        AdvanceFrontier(graph, states, frontier, next);
        timer.Stop();

        if (logging_enabled)
        {
            PrintStep(
                bottom_up ? "p-bsp-bu" : "p-bsp-td",
                timer.Seconds(),
                static_cast<std::int64_t>(next.size()));
        }

        if (
            bottom_up &&
            !(next.size() >= old_awake_count ||
              next.size() >
                  static_cast<std::size_t>(
                      graph.num_nodes() / beta)))
        {
            bottom_up = false;
            scout_count = 1;
        }
        frontier.swap(next);
    }
    return GatherParents(graph, states);
}

int main(int argc, char *argv[])
{
    CLPartitionApp cli(
        argc, argv, "partitioned breadth-first search");
    if (!cli.ParseArgs())
        return -1;

    Builder builder(cli);
    Graph graph = builder.MakeGraph();

    Timer partition_timer;
    partition_timer.Start();
    auto partitioned = PartitionedBFSGraph::Build(
        graph,
        static_cast<std::size_t>(cli.num_partitions()),
        ParseGraphPartitionBalance(cli.partition_balance()));
    partition_timer.Stop();
    PrintTime("Partition Build Time", partition_timer.Seconds());
    partitioned.PrintStats();
    if (cli.do_verify())
        partitioned.VerifyExact(graph);

    SourcePicker<Graph> source_picker(
        graph, cli.start_vertex());
    auto bfs = [&source_picker, &cli, &partitioned](const Graph &)
    {
        return DOBFSPartitioned(
            partitioned, source_picker.PickNext(),
            cli.logging_en());
    };
    SourcePicker<Graph> verifier_source_picker(
        graph, cli.start_vertex());
    auto verifier =
        [&verifier_source_picker](
            const Graph &input,
            const pvector<NodeID> &parent)
        {
            return BFSVerifier(
                input,
                verifier_source_picker.PickNext(),
                parent);
        };
    BenchmarkKernel(
        cli, graph, bfs, PrintBFSStats, verifier);
    return 0;
}
