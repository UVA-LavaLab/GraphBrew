#ifndef GRAPHBREW_EDGE_EDGE_MAP_H_
#define GRAPHBREW_EDGE_EDGE_MAP_H_

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "graphbrew/edge/algorithm_access_policy.h"

namespace graphbrew::edge {

inline std::size_t EdgeWorkerCount() {
#ifdef _OPENMP
  return static_cast<std::size_t>(omp_get_max_threads());
#else
  return 1;
#endif
}

struct SegmentPartition {
  std::size_t begin_vertex = 0;
  std::size_t end_vertex = 0;
  std::size_t begin_edge = 0;
  std::size_t end_edge = 0;
};

template <typename FlatGraphT>
std::vector<SegmentPartition> PartitionSegments(
    const FlatGraphT &graph,
    std::size_t workers) {
  using Node = std::remove_cv_t<std::remove_pointer_t<
      decltype(graph.offsets_.data)>>;
  const std::size_t vertices =
      static_cast<std::size_t>(graph.num_nodes());
  const std::size_t edges =
      static_cast<std::size_t>(graph.num_edges());
  workers = std::max<std::size_t>(
      1, std::min(workers, std::max<std::size_t>(vertices, 1)));

  std::vector<std::size_t> boundaries(workers + 1, 0);
  boundaries.back() = vertices;
  if (edges == 0) {
    for (std::size_t worker = 1; worker < workers; ++worker)
      boundaries[worker] = (vertices * worker) / workers;
  } else {
    const auto *offsets = graph.offsets_.data;
    for (std::size_t worker = 1; worker < workers; ++worker) {
      const std::size_t target = (edges * worker) / workers;
      const auto *boundary = std::lower_bound(
          offsets, offsets + vertices + 1,
          static_cast<Node>(target));
      boundaries[worker] = std::min<std::size_t>(
          static_cast<std::size_t>(boundary - offsets), vertices);
    }
  }

  std::vector<SegmentPartition> partitions;
  partitions.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin_vertex = boundaries[worker];
    const std::size_t end_vertex = boundaries[worker + 1];
    partitions.push_back({
        begin_vertex,
        end_vertex,
        static_cast<std::size_t>(
            graph.offsets_.data[begin_vertex]),
        static_cast<std::size_t>(
            graph.offsets_.data[end_vertex])});
  }
  return partitions;
}

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
  const auto partitions =
      PartitionSegments(graph, EdgeWorkerCount());
#pragma omp parallel for schedule(static)
  for (std::size_t partition = 0; partition < partitions.size();
       ++partition) {
    for (std::size_t vertex = partitions[partition].begin_vertex;
         vertex < partitions[partition].end_vertex; ++vertex) {
      const std::size_t begin = static_cast<std::size_t>(
          graph.offsets_.data[vertex]);
      const std::size_t end = begin + static_cast<std::size_t>(
          graph.degrees_.data[vertex]);
      function(static_cast<Node>(vertex), begin, end);
    }
  }
}

}  // namespace graphbrew::edge

#endif  // GRAPHBREW_EDGE_EDGE_MAP_H_
