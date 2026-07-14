// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cstdint>
#include <limits>

#include "benchmark.h"
#include "bfs_common.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph_partition.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"

using PartitionedBFSGraph = PartitionedGraph<NodeID>;

std::int64_t PartitionedBUStep(
    const PartitionedBFSGraph &graph,
    pvector<NodeID> &parent,
    Bitmap &front,
    Bitmap &next)
{
    std::int64_t awake_count = 0;
    next.reset();
    #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
    for (NodeID vertex = 0;
         vertex < graph.num_nodes(); ++vertex)
    {
        if (parent[vertex] >= 0)
            continue;
        for (NodeID predecessor : graph.in_neigh(vertex))
        {
            if (!front.get_bit(predecessor))
                continue;
            parent[vertex] = predecessor;
            ++awake_count;
            next.set_bit_atomic(vertex);
            break;
        }
    }
    return awake_count;
}

std::int64_t PartitionedTDStep(
    const PartitionedBFSGraph &graph,
    pvector<NodeID> &parent,
    SlidingQueue<NodeID> &queue)
{
    std::int64_t scout_count = 0;
    #pragma omp parallel
    {
        QueueBuffer<NodeID> local_queue(queue);
        #pragma omp for reduction(+ : scout_count) nowait schedule(dynamic, 64)
        for (auto queue_it = queue.begin();
             queue_it < queue.end(); ++queue_it)
        {
            const NodeID vertex = *queue_it;
            for (NodeID neighbor : graph.out_neigh(vertex))
            {
                const NodeID current = parent[neighbor];
                if (
                    current < 0 &&
                    compare_and_swap(
                        parent[neighbor], current, vertex))
                {
                    local_queue.push_back(neighbor);
                    scout_count += -current;
                }
            }
        }
        local_queue.flush();
    }
    return scout_count;
}

void PartitionedQueueToBitmap(
    const SlidingQueue<NodeID> &queue,
    Bitmap &bitmap)
{
    bitmap.reset();
    #pragma omp parallel for
    for (auto queue_it = queue.begin();
         queue_it < queue.end(); ++queue_it)
    {
        bitmap.set_bit_atomic(*queue_it);
    }
}

void PartitionedBitmapToQueue(
    const PartitionedBFSGraph &graph,
    const Bitmap &bitmap,
    SlidingQueue<NodeID> &queue)
{
    #pragma omp parallel
    {
        QueueBuffer<NodeID> local_queue(queue);
        #pragma omp for nowait
        for (NodeID vertex = 0;
             vertex < graph.num_nodes(); ++vertex)
        {
            if (bitmap.get_bit(vertex))
                local_queue.push_back(vertex);
        }
        local_queue.flush();
    }
    queue.slide_window();
}

pvector<NodeID> InitPartitionedParent(
    const PartitionedBFSGraph &graph)
{
    pvector<NodeID> parent(graph.num_nodes());
    #pragma omp parallel for
    for (NodeID vertex = 0;
         vertex < graph.num_nodes(); ++vertex)
    {
        const std::int64_t degree = graph.out_degree(vertex);
        parent[vertex] =
            degree == 0 ? -1 : -static_cast<NodeID>(degree);
    }
    return parent;
}

pvector<NodeID> DOBFSPartitioned(
    const PartitionedBFSGraph &graph,
    NodeID source,
    bool logging_enabled = false,
    int alpha = 15,
    int beta = 18)
{
    if (logging_enabled)
        PrintStep("Source", static_cast<std::int64_t>(source));

    Timer timer;
    timer.Start();
    pvector<NodeID> parent = InitPartitionedParent(graph);
    timer.Stop();
    if (logging_enabled)
        PrintStep("p-i", timer.Seconds());

    parent[source] = source;
    SlidingQueue<NodeID> queue(graph.num_nodes());
    queue.push_back(source);
    queue.slide_window();
    Bitmap current(graph.num_nodes());
    current.reset();
    Bitmap front(graph.num_nodes());
    front.reset();
    std::int64_t edges_to_check = graph.num_edges_directed();
    std::int64_t scout_count = graph.out_degree(source);

    while (!queue.empty())
    {
        if (scout_count > edges_to_check / alpha)
        {
            std::int64_t awake_count;
            std::int64_t old_awake_count;
            TIME_OP(timer, PartitionedQueueToBitmap(queue, front));
            if (logging_enabled)
                PrintStep("p-e", timer.Seconds());
            awake_count = queue.size();
            queue.slide_window();
            do
            {
                timer.Start();
                old_awake_count = awake_count;
                awake_count = PartitionedBUStep(
                    graph, parent, front, current);
                front.swap(current);
                timer.Stop();
                if (logging_enabled)
                    PrintStep("p-bu", timer.Seconds(), awake_count);
            }
            while (
                awake_count >= old_awake_count ||
                awake_count > graph.num_nodes() / beta);
            TIME_OP(
                timer,
                PartitionedBitmapToQueue(graph, front, queue));
            if (logging_enabled)
                PrintStep("p-c", timer.Seconds());
            scout_count = 1;
        }
        else
        {
            timer.Start();
            edges_to_check -= scout_count;
            scout_count = PartitionedTDStep(graph, parent, queue);
            queue.slide_window();
            timer.Stop();
            if (logging_enabled)
                PrintStep(
                    "p-td", timer.Seconds(), queue.size());
        }
    }

    #pragma omp parallel for
    for (NodeID vertex = 0;
         vertex < graph.num_nodes(); ++vertex)
    {
        if (parent[vertex] < -1)
            parent[vertex] = -1;
    }
    return parent;
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
