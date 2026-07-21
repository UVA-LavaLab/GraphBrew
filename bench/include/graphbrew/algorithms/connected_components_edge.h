#ifndef GRAPHBREW_ALGORITHMS_CONNECTED_COMPONENTS_EDGE_H_
#define GRAPHBREW_ALGORITHMS_CONNECTED_COMPONENTS_EDGE_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/atomics.h"
#include "graphbrew/edge/edge_map.h"
#include "graphbrew/edge/edge_stream.h"
#include "pvector.h"

namespace graphbrew::algorithms {

struct ConnectedComponentsResult {
  pvector<NodeID> labels;
  int iterations = 0;
};

inline void LinkComponents(
    const NodeID first,
    const NodeID second,
    pvector<NodeID> &labels) {
  NodeID first_parent = edge::AtomicLoad(labels[first]);
  NodeID second_parent = edge::AtomicLoad(labels[second]);
  while (first_parent != second_parent) {
    const NodeID high =
        std::max(first_parent, second_parent);
    const NodeID low =
        std::min(first_parent, second_parent);
    const NodeID high_parent = edge::AtomicLoad(labels[high]);
    if (high_parent == low)
      return;
    if (high_parent == high &&
        edge::AtomicAssignIfEqual(labels[high], high, low)) {
      return;
    }
    const NodeID next_high = edge::AtomicLoad(labels[high]);
    first_parent = edge::AtomicLoad(labels[next_high]);
    second_parent = edge::AtomicLoad(labels[low]);
  }
}

inline void CompressComponents(pvector<NodeID> &labels) {
#pragma omp parallel for schedule(dynamic, 16384)
  for (NodeID node = 0;
       node < static_cast<NodeID>(labels.size()); ++node) {
    while (true) {
      const NodeID parent = edge::AtomicLoad(labels[node]);
      const NodeID grandparent = edge::AtomicLoad(labels[parent]);
      if (parent == grandparent)
        break;
      edge::AtomicStore(labels[node], grandparent);
    }
  }
}

inline NodeID SampleFrequentComponent(
    const pvector<NodeID> &labels,
    const bool logging_enabled,
    const int64_t num_samples = 1024) {
  if (labels.size() == 0)
    return -1;

  std::unordered_map<NodeID, int> sample_counts(32);
  std::mt19937 generator;
  std::uniform_int_distribution<NodeID> distribution(
      0, static_cast<NodeID>(labels.size()) - 1);
  for (int64_t sample = 0; sample < num_samples; ++sample) {
    const NodeID node = distribution(generator);
    ++sample_counts[edge::AtomicLoad(labels[node])];
  }
  const auto most_frequent = std::max_element(
      sample_counts.begin(),
      sample_counts.end(),
      [](const auto &left, const auto &right) {
        return left.second < right.second;
      });
  if (logging_enabled) {
    const float fraction =
        static_cast<float>(most_frequent->second) /
        static_cast<float>(num_samples);
    std::cout
        << "Skipping largest intermediate component (ID: "
        << most_frequent->first << ", approx. "
        << static_cast<int>(fraction * 100)
        << "% of the graph)" << std::endl;
  }
  return most_frequent->first;
}

template <typename FlatGraphT, typename AccessPolicy>
ConnectedComponentsResult AfforestEdge(
    const Graph &graph,
    const FlatGraphT &outgoing,
    const bool logging_enabled,
    const int32_t neighbor_rounds,
    AccessPolicy &access_policy) {
  pvector<NodeID> labels(graph.num_nodes());
#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < graph.num_nodes(); ++node)
    labels[node] = node;
  if (graph.num_nodes() == 0)
    return {std::move(labels), 0};

  for (int32_t round = 0; round < neighbor_rounds; ++round) {
#pragma omp parallel for schedule(dynamic, 16384)
    for (NodeID source = 0; source < graph.num_nodes(); ++source) {
      const std::size_t degree = static_cast<std::size_t>(
          outgoing.degrees_.data[source]);
      if (static_cast<std::size_t>(round) >= degree)
        continue;
      const std::size_t ordinal =
          static_cast<std::size_t>(outgoing.offsets_.data[source]) +
          static_cast<std::size_t>(round);
      const NodeID destination =
          outgoing.neighbors_.data[ordinal];
      access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
          source, destination, 1, ordinal});
      LinkComponents(source, destination, labels);
    }
    CompressComponents(labels);
    graphbrew::database::AppendBenchmarkIterationEntry(
        {{"phase", "neighbor_round"}, {"round", round}});
  }

  const NodeID frequent =
      SampleFrequentComponent(labels, logging_enabled);
  edge::EdgeStream<FlatGraphT> stream(
      outgoing, edge::EdgeStorageOrder::kSourceMajor);
  auto link_if_needed = [&](const auto &record) {
    const std::size_t sampled_end =
        static_cast<std::size_t>(
            outgoing.offsets_.data[record.source]) +
        std::min<std::size_t>(
            static_cast<std::size_t>(
                std::max<int32_t>(neighbor_rounds, 0)),
            static_cast<std::size_t>(
                outgoing.degrees_.data[record.source]));
    if (record.ordinal < sampled_end)
      return;
    const NodeID source_label =
        edge::AtomicLoad(labels[record.source]);
    const NodeID destination_label =
        edge::AtomicLoad(labels[record.destination]);
    if (source_label == frequent &&
        destination_label == frequent) {
      return;
    }
    LinkComponents(
        record.source, record.destination, labels);
  };

  if (graph.directed()) {
    edge::ParallelForEachDirected(
        stream, link_if_needed, access_policy);
  } else {
    edge::ParallelForEachOrientedUndirected(
        stream, link_if_needed, access_policy);
  }

  CompressComponents(labels);
  graphbrew::database::AppendBenchmarkIterationEntry(
      {{"phase", "final_compress"},
       {"neighbor_rounds", neighbor_rounds}});
  return {std::move(labels), 0};
}

template <typename FlatGraphT, typename AccessPolicy>
ConnectedComponentsResult ShiloachVishkinEdge(
    const Graph &graph,
    const FlatGraphT &outgoing,
    AccessPolicy &access_policy) {
  pvector<NodeID> labels(graph.num_nodes());
#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < graph.num_nodes(); ++node)
    labels[node] = node;

  edge::EdgeStream<FlatGraphT> stream(
      outgoing, edge::EdgeStorageOrder::kSourceMajor);
  bool changed = true;
  int iterations = 0;
  while (changed) {
    std::atomic<bool> hook_changed{false};
    ++iterations;
    auto hook = [&](const auto &record) {
      const NodeID source_label =
          edge::AtomicLoad(labels[record.source]);
      const NodeID destination_label =
          edge::AtomicLoad(labels[record.destination]);
      if (source_label == destination_label)
        return;
      const NodeID high =
          std::max(source_label, destination_label);
      const NodeID low =
          std::min(source_label, destination_label);
      if (edge::AtomicAssignIfEqual(
              labels[high], high, low)) {
        hook_changed.store(true, std::memory_order_relaxed);
      }
    };

    if (graph.directed()) {
      edge::ParallelForEachDirected(
          stream, hook, access_policy);
    } else {
      edge::ParallelForEachOrientedUndirected(
          stream, hook, access_policy);
    }
    CompressComponents(labels);
    changed = hook_changed.load(std::memory_order_relaxed);
  }

  std::cout << "Shiloach-Vishkin took "
            << iterations << " iterations" << std::endl;
  graphbrew::database::AppendBenchmarkIterationEntry(
      {{"num_iterations", iterations}});
  return {std::move(labels), iterations};
}

inline void PrintConnectedComponentStats(
    const Graph &,
    const ConnectedComponentsResult &result) {
  std::cout << std::endl;
  std::unordered_map<NodeID, NodeID> counts;
  for (const NodeID label : result.labels)
    ++counts[label];
  std::vector<std::pair<NodeID, NodeID>> count_vector;
  count_vector.reserve(counts.size());
  for (const auto &entry : counts)
    count_vector.push_back(entry);
  const auto top = TopK(count_vector, 5);
  std::cout << std::min<std::size_t>(5, top.size())
            << " biggest clusters" << std::endl;
  for (const auto &entry : top)
    std::cout << entry.second << ":" << entry.first << std::endl;
  std::cout << "There are " << counts.size()
            << " components" << std::endl;
}

inline bool VerifyConnectedComponents(
    const Graph &graph,
    const ConnectedComponentsResult &result) {
  const auto &labels = result.labels;
  if (labels.size() != static_cast<std::size_t>(graph.num_nodes()))
    return false;

  std::unordered_map<NodeID, NodeID> label_to_source;
  for (NodeID node : graph.vertices())
    label_to_source[labels[node]] = node;
  Bitmap visited(graph.num_nodes());
  visited.reset();
  std::vector<NodeID> frontier;
  frontier.reserve(graph.num_nodes());
  for (const auto &entry : label_to_source) {
    const NodeID label = entry.first;
    frontier.clear();
    frontier.push_back(entry.second);
    visited.set_bit(entry.second);
    for (auto cursor = frontier.begin();
         cursor != frontier.end(); ++cursor) {
      const NodeID source = *cursor;
      for (NodeID destination : graph.out_neigh(source)) {
        if (labels[destination] != label)
          return false;
        if (!visited.get_bit(destination)) {
          visited.set_bit(destination);
          frontier.push_back(destination);
        }
      }
      if (graph.directed()) {
        for (NodeID destination : graph.in_neigh(source)) {
          if (labels[destination] != label)
            return false;
          if (!visited.get_bit(destination)) {
            visited.set_bit(destination);
            frontier.push_back(destination);
          }
        }
      }
    }
  }
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    if (!visited.get_bit(node))
      return false;
  }
  return true;
}

inline nlohmann::json ConnectedComponentsSummary(
    const Graph &graph,
    const ConnectedComponentsResult &result) {
  std::unordered_map<NodeID, NodeID> counts;
  for (NodeID node = 0; node < graph.num_nodes(); ++node)
    ++counts[result.labels[node]];
  NodeID largest = 0;
  for (const auto &entry : counts)
    largest = std::max(largest, entry.second);
  return {
      {"num_components", static_cast<int64_t>(counts.size())},
      {"largest_component", largest},
  };
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_CONNECTED_COMPONENTS_EDGE_H_
