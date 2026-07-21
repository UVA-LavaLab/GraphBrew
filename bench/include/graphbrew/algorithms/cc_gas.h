#ifndef GRAPHBREW_ALGORITHMS_CC_GAS_H_
#define GRAPHBREW_ALGORITHMS_CC_GAS_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "benchmark.h"
#include "graphbrew/algorithms/connected_components_edge.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/frontier.h"
#include "graphbrew/gas/executor.h"
#include "pvector.h"
#include "timer.h"

namespace graphbrew::algorithms {

class ConnectedComponentsGASProgram {
 public:
  using State = NodeID;
  using Gather = NodeID;
  using Convergence = int64_t;

  Gather GatherIdentity(NodeID) const {
    return std::numeric_limits<NodeID>::max();
  }

  template <typename EdgeT>
  Gather GatherEdge(
      const EdgeT &, const State source, State) const {
    return source;
  }

  Gather GatherCombine(
      const Gather left, const Gather right) const {
    return std::min(left, right);
  }

  gas::ApplyResult<State, Convergence> Apply(
      NodeID, const State old_state, const Gather gathered) const {
    const State state = std::min(old_state, gathered);
    const bool changed = state < old_state;
    return {state, changed, changed ? 1 : 0};
  }

  template <typename EdgeT>
  bool Scatter(
      const EdgeT &, State, State, State) const {
    return true;
  }

  Convergence ConvergenceIdentity() const { return 0; }

  Convergence ConvergenceCombine(
      const Convergence left,
      const Convergence right) const {
    return left + right;
  }
};

template <typename WeakFlatGraph, typename AccessPolicy>
ConnectedComponentsResult ConnectedComponentsGAS(
    const Graph &graph,
    const WeakFlatGraph &weak,
    const bool logging_enabled,
    AccessPolicy &access_policy) {
  pvector<NodeID> state(graph.num_nodes());
  std::vector<NodeID> initial_vertices(graph.num_nodes());
#pragma omp parallel for schedule(static)
  for (NodeID node = 0; node < graph.num_nodes(); ++node) {
    state[node] = node;
    initial_vertices[node] = node;
  }

  edge::Frontier<NodeID> active(graph.num_nodes());
  active.Assign(std::move(initial_vertices));
  ConnectedComponentsGASProgram program;
  gas::Executor<
      WeakFlatGraph,
      WeakFlatGraph,
      ConnectedComponentsGASProgram,
      AccessPolicy> executor(
          weak, weak, program, access_policy);
  int iterations = 0;
  Timer timer;
  while (!active.empty()) {
    const std::size_t active_count = active.size();
    timer.Start();
    auto result = executor.Run(
        state, active, gas::GatherSchedule::kActive);
    timer.Stop();
    ++iterations;
    if (logging_enabled) {
      PrintStep(
          static_cast<std::size_t>(iterations - 1),
          timer.Millisecs(),
          static_cast<int64_t>(active_count));
    }
    graphbrew::database::AppendBenchmarkIterationEntry(
        {{"iter", iterations - 1},
         {"active_vertices", static_cast<int64_t>(active_count)},
         {"time_ms", timer.Millisecs()},
         {"changed_vertices",
          static_cast<int64_t>(result.changed_vertices)}});
    active = std::move(result.active);
  }
  return {std::move(state), iterations};
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_CC_GAS_H_
