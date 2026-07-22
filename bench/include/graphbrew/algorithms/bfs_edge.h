#ifndef GRAPHBREW_ALGORITHMS_BFS_EDGE_H_
#define GRAPHBREW_ALGORITHMS_BFS_EDGE_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "benchmark.h"
#include "bitmap.h"
#include "graphbrew/bfs_common.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/atomics.h"
#include "graphbrew/edge/edge_map.h"
#include "graphbrew/edge/edge_stream.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"

namespace graphbrew::algorithms {

template <typename FlatGraphT, typename AccessPolicy>
int64_t BFSSparsePush(
    const FlatGraphT &outgoing,
    pvector<NodeID> &parent,
    SlidingQueue<NodeID> &queue,
    AccessPolicy &access_policy) {
  int64_t scout_count = 0;
#pragma omp parallel
  {
    QueueBuffer<NodeID> local_queue(queue);
#pragma omp for reduction(+ : scout_count) nowait
    for (auto cursor = queue.begin(); cursor < queue.end(); ++cursor) {
      const NodeID source = *cursor;
      const std::size_t begin = static_cast<std::size_t>(
          outgoing.offsets_.data[source]);
      const std::size_t end = begin + static_cast<std::size_t>(
          outgoing.degrees_.data[source]);
      for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
        const NodeID destination =
            outgoing.neighbors_.data[ordinal];
        access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
            source, destination, 1, ordinal});
        const NodeID observed = edge::AtomicLoad(parent[destination]);
        if (observed < 0 &&
            edge::AtomicAssignIfEqual(
                parent[destination], observed, source)) {
          local_queue.push_back(destination);
          scout_count += -static_cast<int64_t>(observed);
        }
      }
    }
    local_queue.flush();
  }
  return scout_count;
}

template <typename FlatGraphT, typename AccessPolicy>
int64_t BFSDensePull(
    const FlatGraphT &incoming,
    pvector<NodeID> &parent,
    const Bitmap &frontier,
    Bitmap &next,
    AccessPolicy &access_policy) {
  int64_t awake_count = 0;
  next.reset();
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : awake_count)
  for (NodeID node = 0; node < incoming.num_nodes(); ++node) {
    if (parent[node] >= 0)
      continue;
    const std::size_t begin = static_cast<std::size_t>(
        incoming.offsets_.data[node]);
    const std::size_t end = begin + static_cast<std::size_t>(
        incoming.degrees_.data[node]);
    for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
      const NodeID source = incoming.neighbors_.data[ordinal];
      access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
          source, node, 1, ordinal});
      if (frontier.get_bit(source)) {
        parent[node] = source;
        next.set_bit(node);
        ++awake_count;
        break;
      }
    }
  }
  return awake_count;
}

inline void BFSQueueToBitmap(
    const SlidingQueue<NodeID> &queue,
    Bitmap &bitmap) {
#pragma omp parallel for
  for (auto cursor = queue.begin(); cursor < queue.end(); ++cursor)
    bitmap.set_bit_atomic(*cursor);
}

inline void BFSBitmapToQueue(
    const Bitmap &bitmap,
    SlidingQueue<NodeID> &queue,
    const NodeID num_nodes) {
#pragma omp parallel
  {
    QueueBuffer<NodeID> local_queue(queue);
#pragma omp for nowait
    for (NodeID node = 0; node < num_nodes; ++node) {
      if (bitmap.get_bit(node))
        local_queue.push_back(node);
    }
    local_queue.flush();
  }
  queue.slide_window();
}

template <typename OutFlatGraphT, typename InFlatGraphT,
          typename AccessPolicy>
pvector<NodeID> DirectionOptimizingBFSEdge(
    const Graph &graph,
    const OutFlatGraphT &outgoing,
    const InFlatGraphT &incoming,
    const NodeID source,
    const bool logging_enabled,
    AccessPolicy &access_policy,
    const int alpha = 15,
    const int beta = 18) {
  if (alpha <= 0 || beta <= 0)
    throw std::invalid_argument("BFS alpha and beta must be positive");

  pvector<NodeID> parent(graph.num_nodes());
#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    parent[node] = graph.out_degree(node) != 0
        ? -graph.out_degree(node)
        : -1;
  }
  if (graph.num_nodes() == 0)
    return parent;
  if (source < 0 || source >= graph.num_nodes())
    throw std::out_of_range("BFS source is outside the graph");

  if (logging_enabled)
    PrintStep("Source", static_cast<int64_t>(source));
  parent[source] = source;
  SlidingQueue<NodeID> queue(graph.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap current(graph.num_nodes());
  current.reset();
  Bitmap frontier(graph.num_nodes());
  frontier.reset();
  int64_t edges_to_check = graph.num_edges_directed();
  int64_t scout_count = graph.out_degree(source);
  int step = 0;
  Timer timer;

  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      timer.Start();
      BFSQueueToBitmap(queue, frontier);
      timer.Stop();
      graphbrew::database::AppendBenchmarkIterationEntry(
          {{"step", step++},
           {"phase", "frontier_to_bitmap"},
           {"time_s", timer.Seconds()}});
      int64_t awake_count =
          static_cast<int64_t>(queue.size());
      int64_t old_awake_count = 0;
      queue.slide_window();
      do {
        old_awake_count = awake_count;
        timer.Start();
        awake_count = BFSDensePull(
            incoming,
            parent,
            frontier,
            current,
            access_policy);
        timer.Stop();
        frontier.swap(current);
        if (logging_enabled)
          PrintStep("bu", timer.Seconds(), awake_count);
        graphbrew::database::AppendBenchmarkIterationEntry(
            {{"step", step++},
             {"phase", "bu"},
             {"time_s", timer.Seconds()},
             {"awake_count", awake_count}});
      } while (
          awake_count >= old_awake_count ||
          awake_count > graph.num_nodes() / beta);
      timer.Start();
      BFSBitmapToQueue(frontier, queue, graph.num_nodes());
      timer.Stop();
      graphbrew::database::AppendBenchmarkIterationEntry(
          {{"step", step++},
           {"phase", "bitmap_to_frontier"},
           {"time_s", timer.Seconds()}});
      scout_count = 1;
    } else {
      edges_to_check -= scout_count;
      timer.Start();
      scout_count = BFSSparsePush(
          outgoing,
          parent,
          queue,
          access_policy);
      queue.slide_window();
      timer.Stop();
      if (logging_enabled)
        PrintStep(
          "td", timer.Seconds(),
          static_cast<int64_t>(queue.size()));
      graphbrew::database::AppendBenchmarkIterationEntry(
          {{"step", step++},
           {"phase", "td"},
           {"time_s", timer.Seconds()},
           {"scout_count", scout_count},
           {"queue_size",
            static_cast<int64_t>(queue.size())}});
    }
  }

#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    if (parent[node] < -1)
      parent[node] = -1;
  }
  return parent;
}

inline nlohmann::json BFSSummary(
    const Graph &graph,
    const pvector<NodeID> &parent) {
  int64_t tree_nodes = 0;
  int64_t tree_edges = 0;
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    if (parent[node] >= 0) {
      tree_edges += graph.out_degree(node);
      ++tree_nodes;
    }
  }
  return {
      {"tree_nodes", tree_nodes},
      {"tree_edges", tree_edges},
  };
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_BFS_EDGE_H_
