#ifndef GRAPHBREW_ALGORITHMS_BFS_EDGE_H_
#define GRAPHBREW_ALGORITHMS_BFS_EDGE_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "benchmark.h"
#include "graphbrew/bfs_common.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/atomics.h"
#include "graphbrew/edge/edge_map.h"
#include "graphbrew/edge/edge_stream.h"
#include "graphbrew/edge/frontier.h"
#include "pvector.h"
#include "timer.h"

namespace graphbrew::algorithms {

template <typename FlatGraphT, typename AccessPolicy>
std::pair<int64_t, edge::Frontier<NodeID>> BFSSparsePush(
    const FlatGraphT &outgoing,
    const edge::Frontier<NodeID> &frontier,
    pvector<NodeID> &parent,
    edge::FrontierBuilder<NodeID> &builder,
    AccessPolicy &access_policy) {
  builder.PrepareForParallel();
  int64_t scout_count = 0;
  const auto &active = frontier.sparse();
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : scout_count)
  for (std::size_t index = 0; index < active.size(); ++index) {
    const NodeID source = active[index];
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
        builder.Push(destination);
        scout_count += -static_cast<int64_t>(observed);
      }
    }
  }
  return {scout_count, builder.Finish()};
}

template <typename FlatGraphT, typename AccessPolicy>
std::pair<int64_t, edge::Frontier<NodeID>> BFSDensePull(
    const FlatGraphT &incoming,
    const edge::Frontier<NodeID> &frontier,
    pvector<NodeID> &parent,
    edge::FrontierBuilder<NodeID> &builder,
    AccessPolicy &access_policy) {
  builder.PrepareForParallel();
  const auto partitions = edge::PartitionSegments(
      incoming, edge::EdgeWorkerCount());
  int64_t awake_count = 0;
#pragma omp parallel for schedule(static) reduction(+ : awake_count)
  for (std::size_t partition = 0;
       partition < partitions.size(); ++partition) {
    for (std::size_t destination =
             partitions[partition].begin_vertex;
         destination < partitions[partition].end_vertex;
         ++destination) {
      const NodeID node = static_cast<NodeID>(destination);
      if (edge::AtomicLoad(parent[node]) >= 0)
        continue;
      const std::size_t begin = static_cast<std::size_t>(
          incoming.offsets_.data[destination]);
      const std::size_t end = begin + static_cast<std::size_t>(
          incoming.degrees_.data[destination]);
      for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
        const NodeID source = incoming.neighbors_.data[ordinal];
        access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
            source, node, 1, ordinal});
        if (frontier.Contains(source)) {
          edge::AtomicStore(parent[node], source);
          builder.Push(node);
          ++awake_count;
          break;
        }
      }
    }
  }
  return {awake_count, builder.Finish()};
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
  edge::Frontier<NodeID> frontier(graph.num_nodes());
  frontier.AssignSingleton(source);
  edge::FrontierBuilder<NodeID> builder(graph.num_nodes());
  int64_t edges_to_check = graph.num_edges_directed();
  int64_t scout_count = graph.out_degree(source);
  int step = 0;
  Timer timer;

  while (!frontier.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count =
          static_cast<int64_t>(frontier.size());
      int64_t old_awake_count = 0;
      do {
        old_awake_count = awake_count;
        timer.Start();
        auto next = BFSDensePull(
            incoming,
            frontier,
            parent,
            builder,
            access_policy);
        timer.Stop();
        awake_count = next.first;
        frontier = std::move(next.second);
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
      scout_count = 1;
    } else {
      edges_to_check -= scout_count;
      timer.Start();
      auto next = BFSSparsePush(
          outgoing,
          frontier,
          parent,
          builder,
          access_policy);
      timer.Stop();
      scout_count = next.first;
      frontier = std::move(next.second);
      if (logging_enabled)
        PrintStep(
          "td", timer.Seconds(),
          static_cast<int64_t>(frontier.size()));
      graphbrew::database::AppendBenchmarkIterationEntry(
          {{"step", step++},
           {"phase", "td"},
           {"time_s", timer.Seconds()},
           {"scout_count", scout_count},
           {"queue_size",
            static_cast<int64_t>(frontier.size())}});
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
