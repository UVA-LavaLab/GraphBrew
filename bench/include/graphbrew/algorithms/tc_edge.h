#ifndef GRAPHBREW_ALGORITHMS_TC_EDGE_H_
#define GRAPHBREW_ALGORITHMS_TC_EDGE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "benchmark.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_stream.h"
#include "pvector.h"

namespace graphbrew::algorithms {

inline bool WorthRelabellingForTC(const Graph &graph) {
  if (graph.num_nodes() == 0)
    return false;
  const int64_t average_degree =
      graph.num_edges() / graph.num_nodes();
  if (average_degree < 10)
    return false;

  SourcePicker<Graph> source_picker(graph);
  const int64_t sample_count =
      std::min<int64_t>(1000, graph.num_nodes());
  int64_t sample_total = 0;
  pvector<int64_t> samples(sample_count);
  for (int64_t sample = 0; sample < sample_count; ++sample) {
    samples[sample] =
        graph.out_degree(source_picker.PickNext());
    sample_total += samples[sample];
  }
  std::sort(samples.begin(), samples.end());
  const double sample_average =
      static_cast<double>(sample_total) / sample_count;
  const double sample_median = samples[sample_count / 2];
  return sample_average / 1.3 > sample_median;
}

template <typename FlatGraphT>
std::size_t CountTriangleIntersection(
    const FlatGraphT &outgoing,
    const NodeID high,
    const NodeID middle) {
  std::size_t high_cursor = static_cast<std::size_t>(
      outgoing.offsets_.data[high]);
  const std::size_t high_end =
      high_cursor + static_cast<std::size_t>(
          outgoing.degrees_.data[high]);
  std::size_t middle_cursor = static_cast<std::size_t>(
      outgoing.offsets_.data[middle]);
  const std::size_t middle_end =
      middle_cursor + static_cast<std::size_t>(
          outgoing.degrees_.data[middle]);
  std::size_t triangles = 0;

  while (high_cursor < high_end &&
         middle_cursor < middle_end) {
    const NodeID high_neighbor =
        outgoing.neighbors_.data[high_cursor];
    const NodeID middle_neighbor =
        outgoing.neighbors_.data[middle_cursor];
    if (high_neighbor >= middle ||
        middle_neighbor >= middle) {
      break;
    }
    if (high_neighbor < middle_neighbor) {
      ++high_cursor;
    } else if (middle_neighbor < high_neighbor) {
      ++middle_cursor;
    } else {
      ++triangles;
      ++high_cursor;
      ++middle_cursor;
    }
  }
  return triangles;
}

template <typename FlatGraphT, typename AccessPolicy>
std::size_t TriangleCountEdge(
    const FlatGraphT &outgoing,
    AccessPolicy &access_policy) {
  std::size_t total = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : total)
  for (NodeID source = 0; source < outgoing.num_nodes(); ++source) {
    const std::size_t begin = static_cast<std::size_t>(
        outgoing.offsets_.data[source]);
    const std::size_t end = begin + static_cast<std::size_t>(
        outgoing.degrees_.data[source]);
    for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
      const NodeID destination =
          outgoing.neighbors_.data[ordinal];
      if (source >= destination)
        continue;
      const edge::EdgeRecord<NodeID, NodeID> record{
          source, destination, 1, ordinal};
      access_policy.OnEdge(record);
      total += CountTriangleIntersection(
          outgoing, destination, source);
    }
  }
  return total;
}

inline void PrintTriangleEdgeStats(
    const Graph &,
    const std::size_t triangles) {
  std::cout << triangles << " triangles" << std::endl;
}

inline bool VerifyTriangleCount(
    const Graph &graph,
    const std::size_t test_total) {
  std::size_t total = 0;
  std::vector<NodeID> intersection;
  intersection.reserve(graph.num_nodes());
  for (NodeID source : graph.vertices()) {
    for (NodeID destination : graph.out_neigh(source)) {
      intersection.clear();
      std::set_intersection(
          graph.out_neigh(source).begin(),
          graph.out_neigh(source).end(),
          graph.out_neigh(destination).begin(),
          graph.out_neigh(destination).end(),
          std::back_inserter(intersection));
      total += intersection.size();
    }
  }
  total /= 6;
  if (total != test_total)
    std::cout << total << " != " << test_total << std::endl;
  return total == test_total;
}

inline nlohmann::json TriangleCountSummary(
    const Graph &,
    const std::size_t triangles) {
  return {
      {"total_triangles", static_cast<int64_t>(triangles)},
  };
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_TC_EDGE_H_
