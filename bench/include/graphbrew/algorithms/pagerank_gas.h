#ifndef GRAPHBREW_ALGORITHMS_PAGERANK_GAS_H_
#define GRAPHBREW_ALGORITHMS_PAGERANK_GAS_H_

#include <cmath>
#include <utility>

#include "benchmark.h"
#include "graphbrew/algorithms/pagerank_edge.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/frontier.h"
#include "graphbrew/gas/executor.h"
#include "pvector.h"

namespace graphbrew::algorithms {

struct PageRankGASState {
  PageRankScore score = 0;
  PageRankScore outgoing_contribution = 0;
};

class PageRankGASProgram {
 public:
  using State = PageRankGASState;
  using Gather = PageRankScore;
  using Convergence = double;

  explicit PageRankGASProgram(const Graph &graph)
      : graph_(graph),
        base_score_(
            (1.0f - kPageRankDamp) / graph.num_nodes()) {}

  Gather GatherIdentity(NodeID) const { return 0; }

  template <typename EdgeT>
  Gather GatherEdge(
      const EdgeT &,
      const State &source,
      const State &) const {
    return source.outgoing_contribution;
  }

  Gather GatherCombine(
      const Gather left,
      const Gather right) const {
    return left + right;
  }

  gas::ApplyResult<State, Convergence> Apply(
      const NodeID node,
      const State &old_state,
      const Gather incoming_total) const {
    const PageRankScore score =
        base_score_ + kPageRankDamp * incoming_total;
    const PageRankScore difference =
        std::fabs(score - old_state.score);
    return {
        {score, score / graph_.out_degree(node)},
        difference > 0,
        difference,
    };
  }

  template <typename EdgeT>
  bool Scatter(
      const EdgeT &,
      const State &,
      const State &,
      const State &) const {
    return true;
  }

  Convergence ConvergenceIdentity() const { return 0; }

  Convergence ConvergenceCombine(
      const Convergence left,
      const Convergence right) const {
    return left + right;
  }

 private:
  const Graph &graph_;
  PageRankScore base_score_;
};

template <
    typename IncomingFlatGraph,
    typename OutgoingFlatGraph,
    typename AccessPolicy>
PageRankResult PageRankGAS(
    const Graph &graph,
    const IncomingFlatGraph &incoming,
    const OutgoingFlatGraph &outgoing,
    const int max_iters,
    const double epsilon,
    const bool logging_enabled,
    AccessPolicy &access_policy) {
  if (graph.num_nodes() == 0)
    return {pvector<PageRankScore>(0), 0};

  const PageRankScore initial_score =
      1.0f / graph.num_nodes();
  pvector<PageRankGASState> state(graph.num_nodes());
#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    state[node] = {
        initial_score,
        initial_score / graph.out_degree(node),
    };
  }

  PageRankGASProgram program(graph);
  gas::Executor<
      IncomingFlatGraph,
      OutgoingFlatGraph,
      PageRankGASProgram,
      AccessPolicy> executor(
          incoming, outgoing, program, access_policy);
  edge::Frontier<NodeID> unused(graph.num_nodes());
  int completed_iters = max_iters;
  for (int iteration = 0;
       iteration < max_iters; ++iteration) {
    auto result = executor.Run(
        state, unused, gas::GatherSchedule::kDense);
    if (logging_enabled)
      PrintStep(iteration, result.convergence);
    graphbrew::database::AppendBenchmarkIterationEntry(
        {{"iter", iteration},
         {"error", result.convergence},
         {"changed_vertices",
          static_cast<int64_t>(result.changed_vertices)}});
    if (result.convergence < epsilon) {
      completed_iters = iteration + 1;
      break;
    }
  }

  pvector<PageRankScore> scores(graph.num_nodes());
#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < graph.num_nodes(); ++node)
    scores[node] = state[node].score;
  return {std::move(scores), completed_iters};
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_PAGERANK_GAS_H_
