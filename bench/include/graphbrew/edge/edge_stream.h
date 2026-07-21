#ifndef GRAPHBREW_EDGE_EDGE_STREAM_H_
#define GRAPHBREW_EDGE_EDGE_STREAM_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "graph.h"

namespace graphbrew::edge {

template <typename Neighbor, typename Node,
          bool IsWeighted = !std::is_same<Neighbor, Node>::value>
struct NeighborTraits;

template <typename Neighbor, typename Node>
struct NeighborTraits<Neighbor, Node, false> {
  using Weight = Node;

  static Node Vertex(const Neighbor neighbor) { return neighbor; }
  static Weight EdgeWeight(const Neighbor) { return Weight{1}; }
};

template <typename Neighbor, typename Node>
struct NeighborTraits<Neighbor, Node, true> {
  using Weight = decltype(std::declval<Neighbor>().w);

  static Node Vertex(const Neighbor &neighbor) { return neighbor.v; }
  static Weight EdgeWeight(const Neighbor &neighbor) { return neighbor.w; }
};

template <typename GraphT>
auto FlattenOutgoing(const GraphT &graph) {
  using Node = std::remove_cv_t<std::remove_pointer_t<
      decltype(graph.get_org_ids())>>;
  using Neighbor = std::decay_t<
      decltype(*graph.out_neigh(Node{}).begin())>;
  using Traits = NeighborTraits<Neighbor, Node>;
  using Weight = typename Traits::Weight;

  CSRGraphFlat<Node, Weight, Node> flat(
      graph.num_nodes(), graph.num_edges_directed());
  for (Node vertex = 0; vertex < graph.num_nodes(); ++vertex) {
    flat.degrees_.data[vertex] =
        static_cast<Node>(graph.out_degree(vertex));
    flat.offsets_.data[vertex + 1] =
        flat.offsets_.data[vertex] + flat.degrees_.data[vertex];
  }

#pragma omp parallel for schedule(static)
  for (Node source = 0; source < graph.num_nodes(); ++source) {
    std::size_t edge = static_cast<std::size_t>(
        flat.offsets_.data[source]);
    for (const Neighbor &neighbor : graph.out_neigh(source)) {
      flat.neighbors_.data[edge] = Traits::Vertex(neighbor);
      flat.weights_.data[edge] = Traits::EdgeWeight(neighbor);
      ++edge;
    }
  }
  return flat;
}

template <typename GraphT>
auto FlattenIncoming(const GraphT &graph) {
  using Node = std::remove_cv_t<std::remove_pointer_t<
      decltype(graph.get_org_ids())>>;
  using Neighbor = std::decay_t<
      decltype(*graph.in_neigh(Node{}).begin())>;
  using Traits = NeighborTraits<Neighbor, Node>;
  using Weight = typename Traits::Weight;

  CSRGraphFlat<Node, Weight, Node> flat(
      graph.num_nodes(), graph.num_edges_directed());
  for (Node vertex = 0; vertex < graph.num_nodes(); ++vertex) {
    flat.degrees_.data[vertex] =
        static_cast<Node>(graph.in_degree(vertex));
    flat.offsets_.data[vertex + 1] =
        flat.offsets_.data[vertex] + flat.degrees_.data[vertex];
  }

#pragma omp parallel for schedule(static)
  for (Node destination = 0; destination < graph.num_nodes();
       ++destination) {
    std::size_t edge = static_cast<std::size_t>(
        flat.offsets_.data[destination]);
    for (const Neighbor &neighbor : graph.in_neigh(destination)) {
      flat.neighbors_.data[edge] = Traits::Vertex(neighbor);
      flat.weights_.data[edge] = Traits::EdgeWeight(neighbor);
      ++edge;
    }
  }
  return flat;
}

struct EdgePartition {
  std::size_t begin = 0;
  std::size_t end = 0;
};

enum class EdgeStorageOrder {
  kSourceMajor,
  kDestinationMajor,
};

inline EdgePartition PartitionEdges(
    const std::size_t edges,
    const std::size_t worker,
    const std::size_t workers) {
  if (workers == 0 || worker >= workers)
    return {};
  return {
      (edges * worker) / workers,
      (edges * (worker + 1)) / workers,
  };
}

template <typename Node, typename Weight>
struct EdgeRecord {
  Node source{};
  Node destination{};
  Weight weight{};
  std::size_t ordinal = 0;
};

template <typename FlatGraphT>
class EdgeStream {
 public:
  using Node = std::remove_cv_t<std::remove_pointer_t<
      decltype(std::declval<FlatGraphT>().offsets_.data)>>;
  using Weight = std::remove_cv_t<std::remove_pointer_t<
      decltype(std::declval<FlatGraphT>().weights_.data)>>;
  using Record = EdgeRecord<Node, Weight>;

  explicit EdgeStream(
      const FlatGraphT &graph,
      const EdgeStorageOrder order)
      : graph_(graph), order_(order) {}

  EdgeStream(FlatGraphT &&, EdgeStorageOrder) = delete;
  EdgeStream(const FlatGraphT &&, EdgeStorageOrder) = delete;

  std::size_t num_nodes() const {
    return static_cast<std::size_t>(graph_.num_nodes());
  }

  std::size_t num_edges() const {
    return static_cast<std::size_t>(graph_.num_edges());
  }

  Node RowOf(const std::size_t edge) const {
    const auto *offsets = graph_.offsets_.data;
    const auto *upper = std::upper_bound(
        offsets, offsets + num_nodes() + 1,
        static_cast<Node>(edge));
    return static_cast<Node>((upper - offsets) - 1);
  }

  template <typename Function>
  void ForEachPartition(
      const std::size_t worker,
      const std::size_t workers,
      Function &&function) const {
    const EdgePartition partition =
        PartitionEdges(num_edges(), worker, workers);
    if (partition.begin >= partition.end)
      return;
    Node row = RowOf(partition.begin);
    for (std::size_t edge = partition.begin; edge < partition.end; ++edge) {
      while (
          static_cast<std::size_t>(row + 1) < num_nodes() &&
          edge >= static_cast<std::size_t>(
              graph_.offsets_.data[row + 1])) {
        ++row;
      }
      const Node neighbor = graph_.neighbors_.data[edge];
      const bool source_major =
          order_ == EdgeStorageOrder::kSourceMajor;
      function(Record{
          source_major ? row : neighbor,
          source_major ? neighbor : row,
          graph_.weights_.data[edge],
          edge});
    }
  }

  template <typename Function>
  void ForEachDirected(Function &&function) const {
    ForEachPartition(0, 1, std::forward<Function>(function));
  }

  template <typename Function>
  void ForEachOrientedUndirected(Function &&function) const {
    ForEachDirected([&](const Record &edge) {
      if (edge.source < edge.destination)
        function(edge);
    });
  }

  template <typename Function>
  void ParallelForEachDirected(Function &&function) const {
#ifdef _OPENMP
#pragma omp parallel
    {
      const std::size_t worker =
          static_cast<std::size_t>(omp_get_thread_num());
      const std::size_t workers =
          static_cast<std::size_t>(omp_get_num_threads());
      ForEachPartition(worker, workers, function);
    }
#else
    ForEachDirected(std::forward<Function>(function));
#endif
  }

  const FlatGraphT &graph() const { return graph_; }
  EdgeStorageOrder order() const { return order_; }

 private:
  const FlatGraphT &graph_;
  EdgeStorageOrder order_;
};

}  // namespace graphbrew::edge

#endif  // GRAPHBREW_EDGE_EDGE_STREAM_H_
