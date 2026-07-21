#ifndef GRAPHBREW_ALGORITHMS_SSSP_GAS_H_
#define GRAPHBREW_ALGORITHMS_SSSP_GAS_H_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "benchmark.h"
#include "graphbrew/algorithms/sssp_edge.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/frontier.h"
#include "graphbrew/gas/executor.h"
#include "pvector.h"
#include "timer.h"

namespace graphbrew::algorithms {

class SSSPGASProgram {
 public:
  using State = WeightT;
  using Gather = WeightT;
  using Convergence = int64_t;

  Gather GatherIdentity(NodeID) const { return kSSSPDistInf; }

  template <typename EdgeT>
  Gather GatherEdge(
      const EdgeT &edge,
      const State source,
      State) const {
    if (source == kSSSPDistInf)
      return kSSSPDistInf;
    return source + edge.weight;
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

template <
    typename IncomingFlatGraph,
    typename OutgoingFlatGraph,
    typename AccessPolicy>
pvector<WeightT> SSSPGAS(
    const WGraph &graph,
    const IncomingFlatGraph &incoming,
    const OutgoingFlatGraph &outgoing,
    const NodeID source,
    const bool logging_enabled,
    AccessPolicy &access_policy) {
  if (graph.num_nodes() == 0)
    return pvector<WeightT>(0);
  if (source < 0 || source >= graph.num_nodes())
    throw std::out_of_range("SSSP GAS source is outside the graph");

  pvector<WeightT> state(graph.num_nodes(), kSSSPDistInf);
  state[source] = 0;
  std::vector<NodeID> initial_vertices;
  const std::size_t begin = static_cast<std::size_t>(
      outgoing.offsets_.data[source]);
  const std::size_t end = begin + static_cast<std::size_t>(
      outgoing.degrees_.data[source]);
  initial_vertices.reserve(end - begin);
  for (std::size_t ordinal = begin; ordinal < end; ++ordinal)
    initial_vertices.push_back(outgoing.neighbors_.data[ordinal]);

  edge::Frontier<NodeID> active(graph.num_nodes());
  active.Assign(std::move(initial_vertices));
  SSSPGASProgram program;
  gas::Executor<
      IncomingFlatGraph,
      OutgoingFlatGraph,
      SSSPGASProgram,
      AccessPolicy> executor(
          incoming, outgoing, program, access_policy);
  int64_t iteration = 0;
  Timer timer;
  while (!active.empty()) {
    const std::size_t active_count = active.size();
    timer.Start();
    auto result = executor.Run(
        state, active, gas::GatherSchedule::kActive);
    timer.Stop();
    if (logging_enabled) {
      PrintStep(
          static_cast<std::size_t>(iteration),
          timer.Millisecs(),
          static_cast<int64_t>(active_count));
    }
    graphbrew::database::AppendBenchmarkIterationEntry(
        {{"iter", iteration++},
         {"active_vertices", static_cast<int64_t>(active_count)},
         {"time_ms", timer.Millisecs()},
         {"changed_vertices",
          static_cast<int64_t>(result.changed_vertices)}});
    active = std::move(result.active);
  }
  return state;
}

}  // namespace graphbrew::algorithms

#endif  // GRAPHBREW_ALGORITHMS_SSSP_GAS_H_
