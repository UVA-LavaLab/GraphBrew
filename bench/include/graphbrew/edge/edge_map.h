#ifndef GRAPHBREW_EDGE_EDGE_MAP_H_
#define GRAPHBREW_EDGE_EDGE_MAP_H_

#include <cstddef>
#include <type_traits>
#include <utility>

#include "graphbrew/edge/algorithm_access_policy.h"

namespace graphbrew::edge {

template <typename StreamT, typename Function, typename AccessPolicy>
void ParallelForEachDirected(
    const StreamT &stream,
    Function &&function,
    AccessPolicy &access_policy) {
  stream.ParallelForEachDirected([&](const auto &edge) {
    access_policy.OnEdge(edge);
    function(edge);
  });
}

template <typename StreamT, typename Function>
void ParallelForEachDirected(
    const StreamT &stream,
    Function &&function) {
  NoOpAccessPolicy access_policy;
  ParallelForEachDirected(
      stream, std::forward<Function>(function), access_policy);
}

template <typename StreamT, typename Function, typename AccessPolicy>
void ParallelForEachOrientedUndirected(
    const StreamT &stream,
    Function &&function,
    AccessPolicy &access_policy) {
  stream.ParallelForEachDirected([&](const auto &edge) {
    if (edge.source < edge.destination) {
      access_policy.OnEdge(edge);
      function(edge);
    }
  });
}

template <typename StreamT, typename Function>
void ParallelForEachOrientedUndirected(
    const StreamT &stream,
    Function &&function) {
  NoOpAccessPolicy access_policy;
  ParallelForEachOrientedUndirected(
      stream, std::forward<Function>(function), access_policy);
}

template <typename FlatGraphT, typename Function>
void ParallelForEachSegment(
    const FlatGraphT &graph,
    Function &&function) {
  using Node = std::remove_cv_t<std::remove_pointer_t<
      decltype(graph.offsets_.data)>>;
#pragma omp parallel for schedule(static)
  for (Node vertex = 0; vertex < graph.num_nodes(); ++vertex) {
    const std::size_t begin = static_cast<std::size_t>(
        graph.offsets_.data[vertex]);
    const std::size_t end = begin + static_cast<std::size_t>(
        graph.degrees_.data[vertex]);
    function(vertex, begin, end);
  }
}

}  // namespace graphbrew::edge

#endif  // GRAPHBREW_EDGE_EDGE_MAP_H_
