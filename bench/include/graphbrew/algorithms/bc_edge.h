#ifndef GRAPHBREW_ALGORITHMS_BC_EDGE_H_
#define GRAPHBREW_ALGORITHMS_BC_EDGE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "benchmark.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/atomics.h"
#include "graphbrew/edge/edge_stream.h"
#include "graphbrew/edge/frontier.h"
#include "pvector.h"
#include "timer.h"

namespace graphbrew::algorithms {

using BCScore = float;
using BCPathCount = double;

template <typename FlatGraphT, typename AccessPolicy>
void BCForwardEdge(
    const Graph &graph,
    const FlatGraphT &outgoing,
    const NodeID source,
    pvector<NodeID> &depth,
    pvector<BCPathCount> &path_count,
    std::vector<std::vector<NodeID>> &levels,
    edge::FrontierBuilder<NodeID> &builder,
    AccessPolicy &access_policy) {
  depth.fill(-1);
  path_count.fill(0);
  levels.clear();
  depth[source] = 0;
  path_count[source] = 1;

  edge::Frontier<NodeID> frontier(graph.num_nodes());
  frontier.AssignSingleton(source);
  levels.push_back(frontier.sparse());
  NodeID next_depth = 1;
  while (!frontier.empty()) {
    const auto &active = frontier.sparse();
#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t index = 0; index < active.size(); ++index) {
      const NodeID source_node = active[index];
      const BCPathCount source_paths = path_count[source_node];
      const std::size_t begin = static_cast<std::size_t>(
          outgoing.offsets_.data[source_node]);
      const std::size_t end = begin + static_cast<std::size_t>(
          outgoing.degrees_.data[source_node]);
      for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
        const NodeID destination =
            outgoing.neighbors_.data[ordinal];
        access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
            source_node, destination, 1, ordinal});
        NodeID observed = edge::AtomicLoad(depth[destination]);
        if (observed == -1 &&
            edge::AtomicAssignIfEqual(
                depth[destination], observed, next_depth)) {
          builder.Push(destination);
          observed = next_depth;
        } else if (observed == -1) {
          observed = edge::AtomicLoad(depth[destination]);
        }
        if (observed == next_depth) {
#pragma omp atomic update
          path_count[destination] += source_paths;
        }
      }
    }
    auto next = builder.Finish();
    if (next.empty())
      break;
    levels.push_back(next.sparse());
    frontier = std::move(next);
    ++next_depth;
  }
}

template <typename FlatGraphT, typename AccessPolicy>
pvector<BCScore> BrandesEdge(
    const Graph &graph,
    const FlatGraphT &outgoing,
    SourcePicker<Graph> &source_picker,
    const NodeID num_sources,
    const bool logging_enabled,
    AccessPolicy &access_policy) {
  pvector<BCScore> scores(graph.num_nodes(), 0);
  if (graph.num_nodes() == 0)
    return scores;

  pvector<NodeID> depth(graph.num_nodes(), -1);
  pvector<BCPathCount> path_count(graph.num_nodes(), 0);
  edge::FrontierBuilder<NodeID> builder(graph.num_nodes());
  std::vector<std::vector<NodeID>> levels;
  Timer timer;

  for (NodeID iteration = 0;
       iteration < num_sources; ++iteration) {
    const NodeID source = source_picker.PickNext();
    if (logging_enabled)
      PrintStep("Source", static_cast<int64_t>(source));

    timer.Start();
    BCForwardEdge(
        graph,
        outgoing,
        source,
        depth,
        path_count,
        levels,
        builder,
        access_policy);
    timer.Stop();
    const double bfs_seconds = timer.Seconds();
    if (logging_enabled)
      PrintStep("b", bfs_seconds);

    pvector<BCScore> delta(graph.num_nodes(), 0);
    timer.Start();
    for (int level = static_cast<int>(levels.size()) - 1;
         level >= 0; --level) {
      const auto &vertices = levels[level];
#pragma omp parallel for schedule(dynamic, 64)
      for (std::size_t index = 0;
           index < vertices.size(); ++index) {
        const NodeID node = vertices[index];
        BCScore node_delta = 0;
        const std::size_t begin = static_cast<std::size_t>(
            outgoing.offsets_.data[node]);
        const std::size_t end = begin + static_cast<std::size_t>(
            outgoing.degrees_.data[node]);
        for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
          const NodeID successor =
              outgoing.neighbors_.data[ordinal];
          access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
              node, successor, 1, ordinal});
          if (depth[successor] == depth[node] + 1) {
            node_delta +=
                (path_count[node] / path_count[successor]) *
                (1 + delta[successor]);
          }
        }
        delta[node] = node_delta;
        scores[node] += node_delta;
      }
    }
    timer.Stop();
    const double backprop_seconds = timer.Seconds();
    if (logging_enabled)
      PrintStep("p", backprop_seconds);
    graphbrew::database::AppendBenchmarkIterationEntry(
        {{"source_iter", static_cast<int64_t>(iteration)},
         {"source_id", static_cast<int64_t>(source)},
         {"bfs_time_s", bfs_seconds},
         {"backprop_time_s", backprop_seconds},
         {"depth", static_cast<int64_t>(levels.size())}});
  }

  BCScore maximum = 0;
#pragma omp parallel for schedule(static) reduction(max : maximum)
  for (NodeID node = 0; node < graph.num_nodes(); ++node)
    maximum = std::max(maximum, scores[node]);
  if (maximum > 0) {
#pragma omp parallel for schedule(static)
    for (NodeID node = 0; node < graph.num_nodes(); ++node)
      scores[node] /= maximum;
  }
  return scores;
}

inline void PrintBCEdgeScores(
    const Graph &graph,
    const pvector<BCScore> &scores) {
  std::vector<std::pair<NodeID, BCScore>> score_pairs(
      graph.num_nodes());
  for (NodeID node : graph.vertices())
    score_pairs[node] = {node, scores[node]};
  const auto top = TopK(score_pairs, 5);
  for (const auto &entry : top)
    std::cout << entry.second << ":" << entry.first << std::endl;
}

inline bool VerifyBC(
    const Graph &graph,
    SourcePicker<Graph> &source_picker,
    const NodeID num_sources,
    const pvector<BCScore> &scores_to_test) {
  pvector<BCScore> scores(graph.num_nodes(), 0);
  for (NodeID iteration = 0;
       iteration < num_sources; ++iteration) {
    const NodeID source = source_picker.PickNext();
    pvector<int> depth(graph.num_nodes(), -1);
    depth[source] = 0;
    std::vector<BCPathCount> path_count(graph.num_nodes(), 0);
    path_count[source] = 1;
    std::vector<NodeID> queue;
    queue.reserve(graph.num_nodes());
    queue.push_back(source);
    for (auto cursor = queue.begin(); cursor != queue.end(); ++cursor) {
      const NodeID node = *cursor;
      for (NodeID neighbor : graph.out_neigh(node)) {
        if (depth[neighbor] == -1) {
          depth[neighbor] = depth[node] + 1;
          queue.push_back(neighbor);
        }
        if (depth[neighbor] == depth[node] + 1)
          path_count[neighbor] += path_count[node];
      }
    }

    std::vector<std::vector<NodeID>> levels;
    for (NodeID node : graph.vertices()) {
      if (depth[node] == -1)
        continue;
      if (depth[node] >= static_cast<int>(levels.size()))
        levels.resize(depth[node] + 1);
      levels[depth[node]].push_back(node);
    }
    pvector<BCScore> delta(graph.num_nodes(), 0);
    for (int level = static_cast<int>(levels.size()) - 1;
         level >= 0; --level) {
      for (NodeID node : levels[level]) {
        for (NodeID successor : graph.out_neigh(node)) {
          if (depth[successor] == depth[node] + 1) {
            delta[node] +=
                (path_count[node] / path_count[successor]) *
                (1 + delta[successor]);
          }
        }
        scores[node] += delta[node];
      }
    }
  }

  if (scores.size() != 0) {
    const BCScore maximum =
        *std::max_element(scores.begin(), scores.end());
    if (maximum > 0) {
      for (NodeID node : graph.vertices())
        scores[node] /= maximum;
    }
  }
  bool matches = true;
  for (NodeID node : graph.vertices()) {
    const BCScore difference =
        std::abs(scores_to_test[node] - scores[node]);
    if (difference > std::numeric_limits<BCScore>::epsilon()) {
      std::cout << node << ": " << scores[node]
                << " != " << scores_to_test[node]
                << "(" << difference << ")" << std::endl;
      matches = false;
    }
  }
  return matches;
}

inline nlohmann::json BCSummary(
    const Graph &,
    const pvector<BCScore> &scores) {
  const BCScore maximum = scores.size() == 0
      ? 0
      : *std::max_element(scores.begin(), scores.end());
  return {{"max_centrality", static_cast<double>(maximum)}};
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_BC_EDGE_H_
