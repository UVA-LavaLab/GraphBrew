#ifndef GRAPHBREW_GAS_EXECUTOR_H_
#define GRAPHBREW_GAS_EXECUTOR_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_map.h"
#include "graphbrew/edge/edge_stream.h"
#include "graphbrew/edge/frontier.h"
#include "pvector.h"

namespace graphbrew::gas {

enum class GatherSchedule {
  kDense,
  kActive,
};

template <typename State, typename Convergence>
struct ApplyResult {
  State state{};
  bool changed = false;
  Convergence convergence{};
};

template <typename Convergence>
struct SuperstepResult {
  edge::Frontier<NodeID> active;
  Convergence convergence{};
  std::size_t applied_vertices = 0;
  std::size_t changed_vertices = 0;
};

template <
    typename IncomingFlatGraph,
    typename OutgoingFlatGraph,
    typename Program,
    typename AccessPolicy = edge::NoOpAccessPolicy>
class Executor {
 public:
  // Program methods are invoked concurrently and must be const/reentrant.
  // GatherCombine must be associative; incoming row order is preserved.
  using State = typename Program::State;
  using Gather = typename Program::Gather;
  using Convergence = typename Program::Convergence;
  using Apply = ApplyResult<State, Convergence>;
  using IncomingEdge =
      typename edge::EdgeStream<IncomingFlatGraph>::Record;
  using OutgoingEdge =
      typename edge::EdgeStream<OutgoingFlatGraph>::Record;

  Executor(
      const IncomingFlatGraph &incoming,
      const OutgoingFlatGraph &outgoing,
      const Program &program,
      AccessPolicy &access_policy)
      : incoming_(incoming),
        outgoing_(outgoing),
        program_(program),
        access_policy_(access_policy),
        vertices_(static_cast<std::size_t>(incoming.num_nodes())),
        next_state_(vertices_),
        convergence_(vertices_),
        changed_(vertices_, 0),
        applied_(vertices_, 0),
        active_builder_(vertices_) {
    if (incoming.num_nodes() != outgoing.num_nodes()) {
      throw std::invalid_argument(
          "GAS incoming/outgoing vertex counts differ");
    }
  }

  SuperstepResult<Convergence> Run(
      pvector<State> &state,
      const edge::Frontier<NodeID> &active,
      const GatherSchedule schedule) {
    if (state.size() != vertices_ ||
        active.capacity() != vertices_) {
      throw std::invalid_argument(
          "GAS state/frontier size differs from graph");
    }
    if (!std::is_sorted(
            active.sparse().begin(), active.sparse().end())) {
      throw std::invalid_argument(
          "GAS active frontier must be sorted");
    }
    active_builder_.PrepareForParallel();

    const Convergence identity =
        program_.ConvergenceIdentity();

    if (schedule == GatherSchedule::kDense) {
#pragma omp parallel for schedule(static)
      for (std::size_t vertex = 0; vertex < vertices_; ++vertex) {
        next_state_[vertex] = state[vertex];
        convergence_[vertex] = identity;
        changed_[vertex] = 0;
        applied_[vertex] = 0;
      }
      const auto partitions = edge::PartitionSegments(
          incoming_, edge::EdgeWorkerCount());
#pragma omp parallel for schedule(static)
      for (std::size_t partition = 0;
           partition < partitions.size(); ++partition) {
        for (std::size_t destination =
                 partitions[partition].begin_vertex;
             destination < partitions[partition].end_vertex;
             ++destination) {
          ApplyVertex(
              static_cast<NodeID>(destination), state);
        }
      }
    } else {
      const auto &vertices = active.sparse();
#pragma omp parallel for schedule(static)
      for (std::size_t index = 0;
           index < vertices.size(); ++index) {
        const NodeID vertex = vertices[index];
        convergence_[vertex] = identity;
        changed_[vertex] = 0;
        applied_[vertex] = 0;
      }
#pragma omp parallel for schedule(dynamic, 64)
      for (std::size_t index = 0;
           index < vertices.size(); ++index) {
        ApplyVertex(vertices[index], state);
      }
    }
    access_policy_.OnBarrier(0);

    std::size_t applied_vertices = 0;
    std::size_t changed_vertices = 0;
    Convergence total = identity;
    if (schedule == GatherSchedule::kDense) {
      for (std::size_t vertex = 0; vertex < vertices_; ++vertex) {
        if (applied_[vertex] != 0)
          ++applied_vertices;
        if (changed_[vertex] != 0)
          ++changed_vertices;
        total = program_.ConvergenceCombine(
            total, convergence_[vertex]);
      }
    } else {
      for (const NodeID vertex : active.sparse()) {
        if (applied_[vertex] != 0)
          ++applied_vertices;
        if (changed_[vertex] != 0)
          ++changed_vertices;
        total = program_.ConvergenceCombine(
            total, convergence_[vertex]);
      }
    }

    if (schedule == GatherSchedule::kDense) {
#pragma omp parallel for schedule(dynamic, 64)
      for (std::size_t source = 0; source < vertices_; ++source) {
        ScatterVertex(static_cast<NodeID>(source), state);
      }
    } else {
      const auto &vertices = active.sparse();
#pragma omp parallel for schedule(dynamic, 64)
      for (std::size_t index = 0;
           index < vertices.size(); ++index) {
        ScatterVertex(vertices[index], state);
      }
    }

    auto next_active = active_builder_.Finish();
    access_policy_.OnBarrier(1);
    if (schedule == GatherSchedule::kDense) {
      state.swap(next_state_);
    } else {
      const auto &vertices = active.sparse();
#pragma omp parallel for schedule(static)
      for (std::size_t index = 0;
           index < vertices.size(); ++index) {
        const NodeID vertex = vertices[index];
        state[vertex] = next_state_[vertex];
      }
    }
    return {
        std::move(next_active),
        std::move(total),
        applied_vertices,
        changed_vertices,
    };
  }

 private:
  void ApplyVertex(
      const NodeID destination,
      const pvector<State> &state) {
    access_policy_.OnVertex(
        destination, edge::AccessKind::kVertexRead);
    Gather aggregate = program_.GatherIdentity(destination);
    const std::size_t begin = static_cast<std::size_t>(
        incoming_.offsets_.data[destination]);
    const std::size_t end = begin + static_cast<std::size_t>(
        incoming_.degrees_.data[destination]);
    for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
      const NodeID source = incoming_.neighbors_.data[ordinal];
      const IncomingEdge record{
          source,
          destination,
          incoming_.weights_.data[ordinal],
          ordinal};
      access_policy_.OnEdge(record);
      access_policy_.OnVertex(
          source, edge::AccessKind::kVertexRead);
      aggregate = program_.GatherCombine(
          std::move(aggregate),
          program_.GatherEdge(
              record, state[source], state[destination]));
    }

    const Apply result =
        program_.Apply(destination, state[destination], aggregate);
    next_state_[destination] = result.state;
    access_policy_.OnVertex(
        destination, edge::AccessKind::kVertexWrite);
    convergence_[destination] = result.convergence;
    changed_[destination] = result.changed ? 1 : 0;
    applied_[destination] = 1;
  }

  void ScatterVertex(
      const NodeID source,
      const pvector<State> &old_state) {
    if (changed_[source] == 0)
      return;
    access_policy_.OnVertex(
        source, edge::AccessKind::kVertexRead);
    const std::size_t begin = static_cast<std::size_t>(
        outgoing_.offsets_.data[source]);
    const std::size_t end = begin + static_cast<std::size_t>(
        outgoing_.degrees_.data[source]);
    for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
      const NodeID destination =
          outgoing_.neighbors_.data[ordinal];
      const OutgoingEdge record{
          source,
          destination,
          outgoing_.weights_.data[ordinal],
          ordinal};
      access_policy_.OnEdge(record);
      access_policy_.OnVertex(
          destination, edge::AccessKind::kVertexRead);
      if (program_.Scatter(
              record,
              old_state[source],
              next_state_[source],
              old_state[destination])) {
        active_builder_.Push(destination);
      }
    }
  }

  const IncomingFlatGraph &incoming_;
  const OutgoingFlatGraph &outgoing_;
  const Program &program_;
  AccessPolicy &access_policy_;
  std::size_t vertices_;
  pvector<State> next_state_;
  pvector<Convergence> convergence_;
  pvector<std::uint8_t> changed_;
  pvector<std::uint8_t> applied_;
  edge::FrontierBuilder<NodeID> active_builder_;
};

}  // namespace graphbrew::gas

#endif  // GRAPHBREW_GAS_EXECUTOR_H_
