#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "benchmark.h"
#include "graphbrew/edge/edge_stream.h"
#include "graphbrew/edge/frontier.h"
#include "graphbrew/gas/executor.h"

namespace {

void Require(const bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

Graph MakeDirectedGraph() {
  constexpr NodeID kNodes = 5;
  auto **out_index = new NodeID *[kNodes + 1];
  auto *out_neighbors = new NodeID[6]{1, 2, 2, 3, 3, 4};
  const int out_offsets[] = {0, 2, 4, 5, 6, 6};
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex)
    out_index[vertex] = out_neighbors + out_offsets[vertex];

  auto **in_index = new NodeID *[kNodes + 1];
  auto *in_neighbors = new NodeID[6]{0, 0, 1, 1, 2, 3};
  const int in_offsets[] = {0, 0, 1, 3, 5, 6};
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex)
    in_index[vertex] = in_neighbors + in_offsets[vertex];
  return Graph(
      kNodes, out_index, out_neighbors, in_index, in_neighbors);
}

Graph MakeRingGraph(const NodeID vertices) {
  auto **out_index = new NodeID *[vertices + 1];
  auto *out_neighbors = new NodeID[vertices];
  auto **in_index = new NodeID *[vertices + 1];
  auto *in_neighbors = new NodeID[vertices];
  for (NodeID vertex = 0; vertex < vertices; ++vertex) {
    out_neighbors[vertex] = (vertex + 1) % vertices;
    in_neighbors[vertex] =
        (vertex + vertices - 1) % vertices;
    out_index[vertex] = out_neighbors + vertex;
    in_index[vertex] = in_neighbors + vertex;
  }
  out_index[vertices] = out_neighbors + vertices;
  in_index[vertices] = in_neighbors + vertices;
  return Graph(
      vertices,
      out_index,
      out_neighbors,
      in_index,
      in_neighbors);
}

struct CountingAccessPolicy {
  std::atomic<std::size_t> edges{0};
  std::atomic<std::size_t> vertices{0};
  std::atomic<std::size_t> barriers{0};

  template <typename EdgeT>
  void OnEdge(const EdgeT &) {
    edges.fetch_add(1, std::memory_order_relaxed);
  }

  template <typename Node>
  void OnVertex(Node, graphbrew::edge::AccessKind) {
    vertices.fetch_add(1, std::memory_order_relaxed);
  }

  void OnBarrier(std::size_t) {
    barriers.fetch_add(1, std::memory_order_relaxed);
  }
};

struct SumProgram {
  using State = int;
  using Gather = int;
  using Convergence = int;

  Gather GatherIdentity(NodeID) const { return 0; }

  template <typename EdgeT>
  Gather GatherEdge(
      const EdgeT &, const State source, const State) const {
    return source;
  }

  Gather GatherCombine(
      const Gather left, const Gather right) const {
    return left + right;
  }

  graphbrew::gas::ApplyResult<State, Convergence> Apply(
      NodeID, const State old_state, const Gather gathered) const {
    return {
        gathered,
        gathered != old_state,
        std::abs(gathered - old_state),
    };
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

struct OrderedGatherProgram {
  using State = std::uint64_t;
  using Gather = std::vector<NodeID>;
  using Convergence = std::uint64_t;

  Gather GatherIdentity(NodeID) const { return {}; }

  template <typename EdgeT>
  Gather GatherEdge(
      const EdgeT &edge, const State, const State) const {
    return {edge.source};
  }

  Gather GatherCombine(Gather left, Gather right) const {
    left.insert(left.end(), right.begin(), right.end());
    return left;
  }

  graphbrew::gas::ApplyResult<State, Convergence> Apply(
      NodeID, const State old_state, const Gather &gathered) const {
    State encoded = 0;
    for (const NodeID source : gathered)
      encoded = encoded * 10 + static_cast<State>(source + 1);
    return {encoded, encoded != old_state, encoded};
  }

  template <typename EdgeT>
  bool Scatter(
      const EdgeT &, State, State, State) const {
    return false;
  }

  Convergence ConvergenceIdentity() const { return 0; }

  Convergence ConvergenceCombine(
      const Convergence left,
      const Convergence right) const {
    return left + right;
  }
};

struct SelectiveProgram {
  using State = int;
  using Gather = int;
  using Convergence = int;

  Gather GatherIdentity(NodeID) const { return 0; }

  template <typename EdgeT>
  Gather GatherEdge(
      const EdgeT &, State, State) const {
    return 0;
  }

  Gather GatherCombine(Gather, Gather) const { return 0; }

  graphbrew::gas::ApplyResult<State, Convergence> Apply(
      const NodeID node,
      const State old_state,
      Gather) const {
    const bool changed = (node % 2) == 0;
    return {
        changed ? old_state + 1 : old_state,
        changed,
        changed ? 1 : 0,
    };
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

void TestDenseAndActiveSchedules() {
  Graph graph = MakeDirectedGraph();
  FlatGraph outgoing = graphbrew::edge::FlattenOutgoing(graph);
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  SumProgram program;
  CountingAccessPolicy access;
  graphbrew::gas::Executor<
      FlatGraph,
      FlatGraph,
      SumProgram,
      CountingAccessPolicy> executor(
          incoming, outgoing, program, access);

  pvector<int> state(5);
  const int initial[] = {1, 2, 4, 8, 16};
  for (std::size_t index = 0; index < state.size(); ++index)
    state[index] = initial[index];
  graphbrew::edge::Frontier<NodeID> unused(5);
  auto dense = executor.Run(
      state,
      unused,
      graphbrew::gas::GatherSchedule::kDense);
  const int expected_dense[] = {0, 1, 3, 6, 8};
  for (std::size_t index = 0; index < state.size(); ++index) {
    Require(
        state[index] == expected_dense[index],
        "dense GAS state differs");
  }
  Require(dense.applied_vertices == 5, "dense applied count differs");
  Require(dense.changed_vertices == 5, "dense changed count differs");
  Require(dense.convergence == 13, "dense convergence differs");
  Require(
      dense.active.sparse() ==
          std::vector<NodeID>{1, 2, 3, 4},
      "dense scatter activation differs");
  Require(
      access.edges.load(std::memory_order_relaxed) == 12,
      "dense GAS hook count differs");
  Require(
      access.vertices.load(std::memory_order_relaxed) == 27,
      "dense GAS vertex hook count differs");
  Require(
      access.barriers.load(std::memory_order_relaxed) == 2,
      "dense GAS barrier hook count differs");

  for (std::size_t index = 0; index < state.size(); ++index)
    state[index] = initial[index];
  graphbrew::edge::Frontier<NodeID> active(5);
  active.AssignSingleton(2);
  auto sparse = executor.Run(
      state,
      active,
      graphbrew::gas::GatherSchedule::kActive);
  const int expected_sparse[] = {1, 2, 3, 8, 16};
  for (std::size_t index = 0; index < state.size(); ++index) {
    Require(
        state[index] == expected_sparse[index],
        "active GAS state differs");
  }
  Require(sparse.applied_vertices == 1, "active applied count differs");
  Require(sparse.changed_vertices == 1, "active changed count differs");
  Require(sparse.convergence == 1, "active convergence differs");
  Require(
      sparse.active.sparse() == std::vector<NodeID>{3},
      "active scatter activation differs");
  Require(
      access.edges.load(std::memory_order_relaxed) == 15,
      "active GAS hook count differs");
  Require(
      access.vertices.load(std::memory_order_relaxed) == 33,
      "active GAS vertex hook count differs");
  Require(
      access.barriers.load(std::memory_order_relaxed) == 4,
      "active GAS barrier hook count differs");
}

void TestStoredGatherOrder() {
  Graph graph = MakeDirectedGraph();
  FlatGraph outgoing = graphbrew::edge::FlattenOutgoing(graph);
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  OrderedGatherProgram program;
  graphbrew::edge::NoOpAccessPolicy access;
  graphbrew::gas::Executor<
      FlatGraph,
      FlatGraph,
      OrderedGatherProgram> executor(
          incoming, outgoing, program, access);

  pvector<std::uint64_t> state(5, 0);
  graphbrew::edge::Frontier<NodeID> active(5);
  active.AssignSingleton(2);
  auto result = executor.Run(
      state,
      active,
      graphbrew::gas::GatherSchedule::kActive);
  Require(state[2] == 12, "stored gather order differs");
  Require(result.convergence == 12, "ordered convergence differs");
  Require(result.active.empty(), "ordered scatter should be empty");
}

void TestMixedScheduleReuse() {
  Graph graph = MakeDirectedGraph();
  FlatGraph outgoing = graphbrew::edge::FlattenOutgoing(graph);
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  SumProgram program;
  graphbrew::edge::NoOpAccessPolicy access;
  graphbrew::gas::Executor<
      FlatGraph,
      FlatGraph,
      SumProgram> executor(
          incoming, outgoing, program, access);
  pvector<int> state(5);
  const int initial[] = {1, 2, 4, 8, 16};
  for (std::size_t index = 0; index < state.size(); ++index)
    state[index] = initial[index];
  graphbrew::edge::Frontier<NodeID> unused(5);

  auto dense = executor.Run(
      state, unused, graphbrew::gas::GatherSchedule::kDense);
  auto sparse = executor.Run(
      state, dense.active, graphbrew::gas::GatherSchedule::kActive);
  const int after_sparse[] = {0, 0, 1, 4, 6};
  for (std::size_t index = 0; index < state.size(); ++index) {
    Require(
        state[index] == after_sparse[index],
        "mixed active state differs");
  }
  Require(
      sparse.active.sparse() == std::vector<NodeID>{2, 3, 4},
      "mixed active frontier differs");

  auto final_dense = executor.Run(
      state, unused, graphbrew::gas::GatherSchedule::kDense);
  const int final_state[] = {0, 0, 0, 1, 4};
  for (std::size_t index = 0; index < state.size(); ++index) {
    Require(
        state[index] == final_state[index],
        "mixed final dense state differs");
  }
  Require(
      final_dense.applied_vertices == 5,
      "mixed final applied count differs");
}

void TestThreadGrowthAndDenseActivation() {
#ifdef _OPENMP
  const int original_threads = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
  constexpr NodeID kVertices = 128;
  Graph graph = MakeRingGraph(kVertices);
  FlatGraph outgoing = graphbrew::edge::FlattenOutgoing(graph);
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  SelectiveProgram program;
  graphbrew::edge::NoOpAccessPolicy access;
  graphbrew::gas::Executor<
      FlatGraph,
      FlatGraph,
      SelectiveProgram> executor(
          incoming, outgoing, program, access);

#ifdef _OPENMP
  const int grown_threads =
      std::max(1, std::min(4, omp_get_thread_limit()));
  omp_set_num_threads(grown_threads);
#endif
  pvector<int> state(kVertices, 0);
  std::vector<NodeID> vertices(kVertices);
  for (NodeID vertex = 0; vertex < kVertices; ++vertex)
    vertices[vertex] = vertex;
  graphbrew::edge::Frontier<NodeID> active(kVertices);
  active.Assign(std::move(vertices));
  auto result = executor.Run(
      state,
      active,
      graphbrew::gas::GatherSchedule::kActive);
  Require(
      result.applied_vertices == kVertices,
      "grown-team applied count differs");
  Require(
      result.changed_vertices == kVertices / 2,
      "grown-team changed count differs");
  Require(
      result.convergence == kVertices / 2,
      "grown-team convergence differs");
  Require(
      result.active.size() == kVertices / 2,
      "dense activation count differs");
  for (NodeID vertex = 0; vertex < kVertices; ++vertex) {
    Require(
        state[vertex] == ((vertex % 2) == 0 ? 1 : 0),
        "selective state differs");
    if ((vertex % 2) == 1)
      Require(result.active.Contains(vertex), "activation vertex differs");
  }
#ifdef _OPENMP
  omp_set_num_threads(original_threads);
#endif
}

void TestValidationFailures() {
  Graph graph = MakeDirectedGraph();
  FlatGraph outgoing = graphbrew::edge::FlattenOutgoing(graph);
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  SumProgram program;
  graphbrew::edge::NoOpAccessPolicy access;
  graphbrew::gas::Executor<
      FlatGraph,
      FlatGraph,
      SumProgram> executor(
          incoming, outgoing, program, access);

  pvector<int> wrong_state(4, 0);
  graphbrew::edge::Frontier<NodeID> active(5);
  bool threw = false;
  try {
    executor.Run(
        wrong_state,
        active,
        graphbrew::gas::GatherSchedule::kActive);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  Require(threw, "wrong GAS state size was accepted");

  pvector<int> state(5, 0);
  graphbrew::edge::Frontier<NodeID> wrong_frontier(4);
  threw = false;
  try {
    executor.Run(
        state,
        wrong_frontier,
        graphbrew::gas::GatherSchedule::kActive);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  Require(threw, "wrong GAS frontier size was accepted");

  graphbrew::edge::Frontier<NodeID> unsorted(5);
  unsorted.Push(3);
  unsorted.Push(1);
  threw = false;
  try {
    executor.Run(
        state,
        unsorted,
        graphbrew::gas::GatherSchedule::kActive);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  Require(threw, "unsorted GAS frontier was accepted");

  CSRGraphFlat<NodeID, NodeID, NodeID> wrong_outgoing(4, 0);
  threw = false;
  try {
    graphbrew::gas::Executor<
        FlatGraph,
        decltype(wrong_outgoing),
        SumProgram> invalid(
            incoming, wrong_outgoing, program, access);
    (void)invalid;
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  Require(threw, "mismatched GAS graph views were accepted");
}

}  // namespace

int main() {
  TestDenseAndActiveSchedules();
  TestStoredGatherOrder();
  TestMixedScheduleReuse();
  TestThreadGrowthAndDenseActivation();
  TestValidationFailures();
  std::cout << "test_gas_executor: PASS" << std::endl;
  return 0;
}
