#ifndef GRAPHBREW_ALGORITHMS_PAGERANK_EDGE_H_
#define GRAPHBREW_ALGORITHMS_PAGERANK_EDGE_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "benchmark.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_map.h"
#include "graphbrew/edge/edge_stream.h"
#include "pvector.h"

namespace graphbrew::algorithms {

using PageRankScore = float;
constexpr PageRankScore kPageRankDamp = 0.85f;

struct PageRankResult {
  pvector<PageRankScore> scores;
  int iterations = 0;
};

inline std::size_t PageRankWorkers() {
#ifdef _OPENMP
  return static_cast<std::size_t>(omp_get_max_threads());
#else
  return 1;
#endif
}

template <typename FlatGraphT, typename AccessPolicy>
PageRankResult PageRankJacobiEdge(
    const Graph &graph,
    const FlatGraphT &incoming,
    const int max_iters,
    const double epsilon,
    const bool logging_enabled,
    AccessPolicy &access_policy) {
  const NodeID num_nodes = graph.num_nodes();
  if (num_nodes == 0)
    return {pvector<PageRankScore>(0), 0};

  const PageRankScore initial_score =
      1.0f / static_cast<PageRankScore>(num_nodes);
  const PageRankScore base_score =
      (1.0f - kPageRankDamp) /
      static_cast<PageRankScore>(num_nodes);
  pvector<PageRankScore> scores(num_nodes, initial_score);
  pvector<PageRankScore> outgoing_contrib(num_nodes);
  const auto partitions =
      edge::PartitionSegments(incoming, PageRankWorkers());
  int completed_iters = max_iters;

  for (int iter = 0; iter < max_iters; ++iter) {
#pragma omp parallel for schedule(static)
    for (NodeID node = 0; node < num_nodes; ++node) {
      access_policy.OnVertex(
          node, edge::AccessKind::kVertexRead);
      outgoing_contrib[node] =
          scores[node] / graph.out_degree(node);
      access_policy.OnVertex(
          node, edge::AccessKind::kVertexWrite);
    }

    double error = 0;
#pragma omp parallel for schedule(static) reduction(+ : error)
    for (std::size_t partition = 0;
         partition < partitions.size(); ++partition) {
      for (std::size_t destination =
               partitions[partition].begin_vertex;
           destination < partitions[partition].end_vertex;
           ++destination) {
        PageRankScore incoming_total = 0;
        const std::size_t begin = static_cast<std::size_t>(
            incoming.offsets_.data[destination]);
        const std::size_t end = begin + static_cast<std::size_t>(
            incoming.degrees_.data[destination]);
        for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
          const NodeID source = incoming.neighbors_.data[ordinal];
          access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
              source,
              static_cast<NodeID>(destination),
              1,
              ordinal});
          incoming_total += outgoing_contrib[source];
        }
        const NodeID node = static_cast<NodeID>(destination);
        const PageRankScore old_score = scores[node];
        const PageRankScore new_score =
            base_score + kPageRankDamp * incoming_total;
        scores[node] = new_score;
        error += std::fabs(new_score - old_score);
        access_policy.OnVertex(
            node, edge::AccessKind::kVertexWrite);
      }
    }

    if (logging_enabled)
      PrintStep(iter, error);
    graphbrew::database::AppendBenchmarkIterationEntry(
        {{"iter", iter}, {"error", error}});
    if (error < epsilon) {
      completed_iters = iter + 1;
      break;
    }
  }
  return {std::move(scores), completed_iters};
}

template <typename FlatGraphT, typename AccessPolicy>
PageRankResult PageRankAsyncEdge(
    const Graph &graph,
    const FlatGraphT &incoming,
    const int max_iters,
    const double epsilon,
    const bool logging_enabled,
    AccessPolicy &access_policy) {
  const NodeID num_nodes = graph.num_nodes();
  if (num_nodes == 0)
    return {pvector<PageRankScore>(0), 0};

  const PageRankScore initial_score =
      1.0f / static_cast<PageRankScore>(num_nodes);
  const PageRankScore base_score =
      (1.0f - kPageRankDamp) /
      static_cast<PageRankScore>(num_nodes);
  pvector<PageRankScore> scores(num_nodes, initial_score);
  auto outgoing_contrib =
      std::make_unique<std::atomic<PageRankScore>[]>(num_nodes);
#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < num_nodes; ++node) {
    outgoing_contrib[node].store(
        initial_score / graph.out_degree(node),
        std::memory_order_relaxed);
  }

  const auto partitions =
      edge::PartitionSegments(incoming, PageRankWorkers());
  int completed_iters = max_iters;
  for (int iter = 0; iter < max_iters; ++iter) {
    double error = 0;
#pragma omp parallel for schedule(static) reduction(+ : error)
    for (std::size_t partition = 0;
         partition < partitions.size(); ++partition) {
      for (std::size_t destination =
               partitions[partition].begin_vertex;
           destination < partitions[partition].end_vertex;
           ++destination) {
        PageRankScore incoming_total = 0;
        const std::size_t begin = static_cast<std::size_t>(
            incoming.offsets_.data[destination]);
        const std::size_t end = begin + static_cast<std::size_t>(
            incoming.degrees_.data[destination]);
        for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
          const NodeID source = incoming.neighbors_.data[ordinal];
          access_policy.OnEdge(edge::EdgeRecord<NodeID, NodeID>{
              source,
              static_cast<NodeID>(destination),
              1,
              ordinal});
          incoming_total += outgoing_contrib[source].load(
              std::memory_order_relaxed);
        }
        const NodeID node = static_cast<NodeID>(destination);
        const PageRankScore old_score = scores[node];
        const PageRankScore new_score =
            base_score + kPageRankDamp * incoming_total;
        scores[node] = new_score;
        error += std::fabs(new_score - old_score);
        outgoing_contrib[node].store(
            new_score / graph.out_degree(node),
            std::memory_order_relaxed);
        access_policy.OnVertex(
            node, edge::AccessKind::kVertexWrite);
      }
    }

    if (logging_enabled)
      PrintStep(iter, error);
    graphbrew::database::AppendBenchmarkIterationEntry(
        {{"iter", iter}, {"error", error}});
    if (error < epsilon) {
      completed_iters = iter + 1;
      break;
    }
  }
  PrintTime("Iterations", completed_iters);
  return {std::move(scores), completed_iters};
}

inline bool VerifyPageRank(
    const Graph &graph,
    const pvector<PageRankScore> &scores,
    const double target_error) {
  if (graph.num_nodes() == 0)
      return scores.size() == 0;

  const PageRankScore base_score =
      (1.0f - kPageRankDamp) / graph.num_nodes();
  pvector<PageRankScore> incoming_sums(graph.num_nodes(), 0);
  double error = 0;
  for (NodeID source : graph.vertices()) {
    const PageRankScore contribution =
        scores[source] / graph.out_degree(source);
    for (NodeID destination : graph.out_neigh(source))
      incoming_sums[destination] += contribution;
  }
  for (NodeID node : graph.vertices()) {
    error += std::fabs(
        base_score +
        kPageRankDamp * incoming_sums[node] -
        scores[node]);
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

inline void PrintPageRankTopScores(
    const Graph &graph,
    const PageRankResult &result,
    const int count,
    const bool use_original_ids) {
  std::vector<std::pair<NodeID, PageRankScore>> score_pairs(
      graph.num_nodes());
  const NodeID *original_ids = graph.get_org_ids();
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    const NodeID output_id =
        use_original_ids && original_ids != nullptr
            ? original_ids[node]
            : node;
    score_pairs[node] = {output_id, result.scores[node]};
  }
  const auto top = TopK(score_pairs, count);
  for (const auto &entry : top)
    std::cout << entry.second << ":" << entry.first << std::endl;
}

inline nlohmann::json PageRankSummary(
    const Graph &graph,
    const PageRankResult &result,
    const bool include_iterations) {
  nlohmann::json summary;
  double total = 0;
  PageRankScore maximum = 0;
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    total += result.scores[node];
    maximum = std::max(maximum, result.scores[node]);
  }
  summary["total_score"] = total;
  summary["max_score"] = static_cast<double>(maximum);
  if (include_iterations)
    summary["iterations_to_convergence"] = result.iterations;
  return summary;
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_PAGERANK_EDGE_H_
