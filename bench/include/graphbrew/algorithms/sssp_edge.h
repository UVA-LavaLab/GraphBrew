#ifndef GRAPHBREW_ALGORITHMS_SSSP_EDGE_H_
#define GRAPHBREW_ALGORITHMS_SSSP_EDGE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "benchmark.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/atomics.h"
#include "graphbrew/edge/edge_map.h"
#include "graphbrew/edge/edge_stream.h"
#include "pvector.h"
#include "timer.h"

namespace graphbrew::algorithms {

inline constexpr WeightT kSSSPDistInf =
    std::numeric_limits<WeightT>::max() / 2;
inline constexpr std::size_t kSSSPMaxBin =
    std::numeric_limits<std::size_t>::max() / 2;
inline constexpr std::size_t kSSSPBinFusionThreshold = 1000;

template <typename FlatGraphT, typename AccessPolicy>
void RelaxWeightedEdges(
    const FlatGraphT &outgoing,
    const NodeID source,
    const WeightT delta,
    pvector<WeightT> &distance,
    std::vector<std::vector<NodeID>> &local_bins,
    AccessPolicy &access_policy) {
  const std::size_t begin = static_cast<std::size_t>(
      outgoing.offsets_.data[source]);
  const std::size_t end = begin + static_cast<std::size_t>(
      outgoing.degrees_.data[source]);
  for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
    const NodeID destination = outgoing.neighbors_.data[ordinal];
    const WeightT weight = outgoing.weights_.data[ordinal];
    access_policy.OnEdge(edge::EdgeRecord<NodeID, WeightT>{
        source, destination, weight, ordinal});
    const WeightT source_distance =
        edge::AtomicLoad(distance[source]);
    if (source_distance == kSSSPDistInf)
      continue;
    const WeightT candidate = source_distance + weight;
    if (edge::AtomicMin(distance[destination], candidate)) {
      const std::size_t destination_bin =
          static_cast<std::size_t>(candidate / delta);
      if (destination_bin >= local_bins.size())
        local_bins.resize(destination_bin + 1);
      local_bins[destination_bin].push_back(destination);
    }
  }
}

template <typename FlatGraphT, typename AccessPolicy>
pvector<WeightT> DeltaStepEdge(
    const WGraph &graph,
    const FlatGraphT &outgoing,
    const NodeID source,
    const WeightT delta,
    const bool logging_enabled,
    AccessPolicy &access_policy) {
  if (delta <= 0)
    throw std::invalid_argument("SSSP delta must be positive");
  if (graph.num_nodes() == 0)
    return pvector<WeightT>(0);
  if (source < 0 || source >= graph.num_nodes())
    throw std::out_of_range("SSSP source is outside the graph");

  pvector<WeightT> distance(graph.num_nodes(), kSSSPDistInf);
  distance[source] = 0;
  std::vector<NodeID> frontier{source};
  std::vector<std::vector<std::vector<NodeID>>> thread_bins(
      edge::EdgeWorkerCount());
  std::size_t current_bin = 0;
  int64_t iteration = 0;
  Timer timer;
  timer.Start();

#pragma omp parallel
  {
    std::size_t thread = 0;
#ifdef _OPENMP
    thread = static_cast<std::size_t>(omp_get_thread_num());
#endif
    auto &local_bins = thread_bins[thread];
    while (current_bin != kSSSPMaxBin) {
      const std::size_t frontier_size = frontier.size();
#pragma omp for schedule(dynamic, 64) nowait
      for (std::size_t index = 0; index < frontier_size; ++index) {
        const NodeID node = frontier[index];
        if (edge::AtomicLoad(distance[node]) >=
            delta * static_cast<WeightT>(current_bin)) {
          RelaxWeightedEdges(
              outgoing,
              node,
              delta,
              distance,
              local_bins,
              access_policy);
        }
      }

      while (
          current_bin < local_bins.size() &&
          !local_bins[current_bin].empty() &&
          local_bins[current_bin].size() <
              kSSSPBinFusionThreshold) {
        std::vector<NodeID> fused =
            std::move(local_bins[current_bin]);
        local_bins[current_bin].clear();
        for (const NodeID node : fused) {
          RelaxWeightedEdges(
              outgoing,
              node,
              delta,
              distance,
              local_bins,
              access_policy);
        }
      }

#pragma omp barrier
#pragma omp single
      {
        timer.Stop();
        if (logging_enabled) {
          PrintStep(
              current_bin,
              timer.Millisecs(),
              static_cast<int64_t>(frontier_size));
        }
        graphbrew::database::AppendBenchmarkIterationEntry(
            {{"iter", iteration++},
             {"bin_index", static_cast<int64_t>(current_bin)},
             {"time_ms", timer.Millisecs()},
             {"frontier_size",
              static_cast<int64_t>(frontier_size)}});

        std::size_t next_bin = kSSSPMaxBin;
        for (const auto &bins : thread_bins) {
          for (std::size_t bin = current_bin;
               bin < bins.size(); ++bin) {
            if (!bins[bin].empty()) {
              next_bin = std::min(next_bin, bin);
              break;
            }
          }
        }
        if (next_bin == kSSSPMaxBin) {
          current_bin = kSSSPMaxBin;
          frontier.clear();
        } else {
          std::size_t next_size = 0;
          for (const auto &bins : thread_bins) {
            if (next_bin < bins.size())
              next_size += bins[next_bin].size();
          }
          frontier.clear();
          frontier.reserve(next_size);
          for (auto &bins : thread_bins) {
            if (next_bin >= bins.size())
              continue;
            frontier.insert(
                frontier.end(),
                bins[next_bin].begin(),
                bins[next_bin].end());
            bins[next_bin].clear();
          }
          current_bin = next_bin;
          timer.Start();
        }
      }
    }
  }

  if (logging_enabled)
    std::cout << "took " << static_cast<long long>(iteration)
              << " iterations" << std::endl;
  return distance;
}

inline void PrintSSSPEdgeStats(
    const WGraph &,
    const pvector<WeightT> &distance) {
  const int64_t reached = std::count_if(
      distance.begin(),
      distance.end(),
      [](const WeightT value) {
        return value != kSSSPDistInf;
      });
  std::cout << "SSSP Tree reaches "
            << static_cast<long long>(reached)
            << " nodes" << std::endl;
}

inline bool VerifySSSP(
    const WGraph &graph,
    const NodeID source,
    const pvector<WeightT> &distance) {
  pvector<WeightT> oracle(graph.num_nodes(), kSSSPDistInf);
  oracle[source] = 0;
  using WeightedNode = std::pair<WeightT, NodeID>;
  std::priority_queue<
      WeightedNode,
      std::vector<WeightedNode>,
      std::greater<WeightedNode>> queue;
  queue.emplace(0, source);
  while (!queue.empty()) {
    const WeightT current_distance = queue.top().first;
    const NodeID node = queue.top().second;
    queue.pop();
    if (current_distance != oracle[node])
      continue;
    for (const WNode neighbor : graph.out_neigh(node)) {
      const WeightT candidate =
          current_distance + neighbor.w;
      if (candidate < oracle[neighbor.v]) {
        oracle[neighbor.v] = candidate;
        queue.emplace(candidate, neighbor.v);
      }
    }
  }

  bool matches = true;
  for (NodeID node : graph.vertices()) {
    if (distance[node] != oracle[node]) {
      std::cout << node << ": " << distance[node]
                << " != " << oracle[node] << std::endl;
      matches = false;
    }
  }
  return matches;
}

inline nlohmann::json SSSPSummary(
    const WGraph &graph,
    const pvector<WeightT> &distance) {
  int64_t reachable = 0;
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    if (distance[node] != kSSSPDistInf)
      ++reachable;
  }
  return {{"reachable_nodes", reachable}};
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_SSSP_EDGE_H_
