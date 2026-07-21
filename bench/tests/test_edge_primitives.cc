#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "benchmark.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/atomics.h"
#include "graphbrew/edge/edge_map.h"
#include "graphbrew/edge/edge_stream.h"
#include "graphbrew/edge/frontier.h"

namespace {

using graphbrew::edge::EdgeStream;

static_assert(
    !std::is_constructible<
        EdgeStream<FlatGraph>, const FlatGraph &>::value,
    "EdgeStream must require an explicit storage order");
static_assert(
    !std::is_constructible<
        EdgeStream<FlatGraph>,
        FlatGraph &&,
        graphbrew::edge::EdgeStorageOrder>::value,
    "EdgeStream must reject temporary flat graphs");

struct CountingAccessPolicy {
  std::atomic<std::size_t> edges{0};

  template <typename EdgeT>
  void OnEdge(const EdgeT &) {
    edges.fetch_add(1, std::memory_order_relaxed);
  }
};

void Require(bool condition, const char *message) {
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

Graph MakeUndirectedGraph() {
  constexpr NodeID kNodes = 4;
  auto **index = new NodeID *[kNodes + 1];
  auto *neighbors = new NodeID[6]{1, 0, 2, 1, 3, 2};
  const int offsets[] = {0, 1, 3, 5, 6};
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex)
    index[vertex] = neighbors + offsets[vertex];
  return Graph(kNodes, index, neighbors);
}

WGraph MakeWeightedGraph() {
  constexpr NodeID kNodes = 4;
  auto **out_index = new WNode *[kNodes + 1];
  auto *out_neighbors = new WNode[4]{
      {1, 2}, {2, 5}, {2, 3}, {3, 7}};
  const int out_offsets[] = {0, 2, 3, 4, 4};
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex)
    out_index[vertex] = out_neighbors + out_offsets[vertex];

  auto **in_index = new WNode *[kNodes + 1];
  auto *in_neighbors = new WNode[4]{
      {0, 2}, {0, 5}, {1, 3}, {2, 7}};
  const int in_offsets[] = {0, 0, 1, 3, 4};
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex)
    in_index[vertex] = in_neighbors + in_offsets[vertex];
  return WGraph(
      kNodes, out_index, out_neighbors, in_index, in_neighbors);
}

Graph MakeEmptyGraph() {
  constexpr NodeID kNodes = 4;
  auto **out_index = new NodeID *[kNodes + 1];
  auto *out_neighbors = new NodeID[0];
  auto **in_index = new NodeID *[kNodes + 1];
  auto *in_neighbors = new NodeID[0];
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex) {
    out_index[vertex] = out_neighbors;
    in_index[vertex] = in_neighbors;
  }
  return Graph(
      kNodes, out_index, out_neighbors, in_index, in_neighbors);
}

Graph MakeGraphWithLeadingIsolates() {
  constexpr NodeID kNodes = 6;
  auto **out_index = new NodeID *[kNodes + 1];
  auto *out_neighbors = new NodeID[2]{3, 4};
  const int out_offsets[] = {0, 0, 0, 1, 2, 2, 2};
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex)
    out_index[vertex] = out_neighbors + out_offsets[vertex];

  auto **in_index = new NodeID *[kNodes + 1];
  auto *in_neighbors = new NodeID[2]{2, 3};
  const int in_offsets[] = {0, 0, 0, 0, 1, 2, 2};
  for (NodeID vertex = 0; vertex <= kNodes; ++vertex)
    in_index[vertex] = in_neighbors + in_offsets[vertex];
  return Graph(
      kNodes, out_index, out_neighbors, in_index, in_neighbors);
}

void TestDirectedAndIncomingStreams() {
  Graph graph = MakeDirectedGraph();
  auto weak = graphbrew::edge::FlattenWeaklyConnected(graph);
  FlatGraph outgoing = graph.flattenGraphOut();
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  EdgeStream<FlatGraph> out_stream(
      outgoing, graphbrew::edge::EdgeStorageOrder::kSourceMajor);
  EdgeStream<FlatGraph> in_stream(
      incoming,
      graphbrew::edge::EdgeStorageOrder::kDestinationMajor);

  std::vector<std::tuple<NodeID, NodeID, std::size_t>> out_edges;
  out_stream.ForEachDirected([&](const auto &edge) {
    out_edges.emplace_back(
        edge.source, edge.destination, edge.ordinal);
  });
  Require(
      out_edges == std::vector<std::tuple<NodeID, NodeID, std::size_t>>{
          {0, 1, 0}, {0, 2, 1}, {1, 2, 2},
          {1, 3, 3}, {2, 3, 4}, {3, 4, 5}},
      "outgoing edge stream differs");

  std::vector<std::pair<NodeID, NodeID>> in_edges;
  in_stream.ForEachDirected([&](const auto &edge) {
    in_edges.emplace_back(edge.source, edge.destination);
  });
  Require(
      in_edges == std::vector<std::pair<NodeID, NodeID>>{
          {0, 1}, {0, 2}, {1, 2},
          {1, 3}, {2, 3}, {3, 4}},
      "incoming segmented stream differs");

  for (std::size_t workers : {1u, 2u, 3u, 8u}) {
    std::vector<int> seen(out_stream.num_edges(), 0);
    for (std::size_t worker = 0; worker < workers; ++worker) {
      out_stream.ForEachPartition(worker, workers, [&](const auto &edge) {
        ++seen[edge.ordinal];
      });
    }
    Require(
        std::all_of(
            seen.begin(), seen.end(),
            [](int count) { return count == 1; }),
        "edge partitions miss or duplicate edges");
  }

  std::vector<std::size_t> segment_sizes(graph.num_nodes(), 0);
  graphbrew::edge::ParallelForEachSegment(
      incoming,
      [&](NodeID vertex, std::size_t begin, std::size_t end) {
        segment_sizes[vertex] = end - begin;
      });
  Require(
      segment_sizes == std::vector<std::size_t>{0, 1, 2, 2, 1},
      "incoming segment sizes differ");

  EdgeStream<decltype(weak)> weak_stream(
      weak, graphbrew::edge::EdgeStorageOrder::kSourceMajor);
  std::vector<std::pair<NodeID, NodeID>> weak_edges;
  weak_stream.ForEachDirected([&](const auto &edge) {
    weak_edges.emplace_back(edge.source, edge.destination);
  });
  std::sort(weak_edges.begin(), weak_edges.end());
  Require(
      weak_edges == std::vector<std::pair<NodeID, NodeID>>{
          {0, 1}, {0, 2},
          {1, 0}, {1, 2}, {1, 3},
          {2, 0}, {2, 1}, {2, 3},
          {3, 1}, {3, 2}, {3, 4},
          {4, 3}},
      "weak-connectivity edge view differs");

  for (std::size_t workers : {1u, 2u, 4u, 8u}) {
    const auto partitions =
        graphbrew::edge::PartitionSegments(incoming, workers);
    std::vector<int> vertex_visits(graph.num_nodes(), 0);
    std::size_t covered_edges = 0;
    for (const auto &partition : partitions) {
      Require(
          partition.begin_edge ==
              static_cast<std::size_t>(
                  incoming.offsets_.data[partition.begin_vertex]),
          "segment partition begin edge differs");
      Require(
          partition.end_edge ==
              static_cast<std::size_t>(
                  incoming.offsets_.data[partition.end_vertex]),
          "segment partition end edge differs");
      covered_edges += partition.end_edge - partition.begin_edge;
      for (std::size_t vertex = partition.begin_vertex;
           vertex < partition.end_vertex; ++vertex) {
        ++vertex_visits[vertex];
      }
    }
    Require(
        covered_edges == in_stream.num_edges(),
        "segment partitions miss edges");
    Require(
        std::all_of(
            vertex_visits.begin(), vertex_visits.end(),
            [](int count) { return count == 1; }),
        "segment partitions miss or duplicate vertices");
  }
}

void TestOrientedAndParallelStreams() {
  Graph graph = MakeUndirectedGraph();
  FlatGraph flat = graph.flattenGraphOut();
  EdgeStream<FlatGraph> stream(
      flat, graphbrew::edge::EdgeStorageOrder::kSourceMajor);
  std::vector<std::pair<NodeID, NodeID>> oriented;
  stream.ForEachOrientedUndirected([&](const auto &edge) {
    oriented.emplace_back(edge.source, edge.destination);
  });
  Require(
      oriented == std::vector<std::pair<NodeID, NodeID>>{
          {0, 1}, {1, 2}, {2, 3}},
      "oriented edge stream differs");

  std::atomic<std::size_t> parallel_count{0};
  stream.ParallelForEachDirected([&](const auto &) {
    parallel_count.fetch_add(1, std::memory_order_relaxed);
  });
  Require(
      parallel_count.load() == stream.num_edges(),
      "parallel edge stream count differs");
}

void TestEmptyAndIsolatedRows() {
  Graph empty = MakeEmptyGraph();
  FlatGraph empty_flat = empty.flattenGraphOut();
  EdgeStream<FlatGraph> empty_stream(
      empty_flat, graphbrew::edge::EdgeStorageOrder::kSourceMajor);
  Require(empty_stream.num_nodes() == 4, "empty graph node count differs");
  Require(empty_stream.num_edges() == 0, "empty graph has edges");
  std::atomic<std::size_t> empty_visits{0};
  empty_stream.ParallelForEachDirected([&](const auto &) {
    empty_visits.fetch_add(1, std::memory_order_relaxed);
  });
  Require(empty_visits.load() == 0, "empty graph visited an edge");

  Graph graph = MakeGraphWithLeadingIsolates();
  FlatGraph outgoing = graph.flattenGraphOut();
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  EdgeStream<FlatGraph> out_stream(
      outgoing, graphbrew::edge::EdgeStorageOrder::kSourceMajor);
  EdgeStream<FlatGraph> in_stream(
      incoming,
      graphbrew::edge::EdgeStorageOrder::kDestinationMajor);

  Require(out_stream.RowOf(0) == 2, "leading empty rows lost source");
  Require(out_stream.RowOf(1) == 3, "outgoing row mapping differs");
  Require(in_stream.RowOf(0) == 3, "incoming row mapping differs");
  Require(in_stream.RowOf(1) == 4, "trailing empty row mapping differs");

  std::vector<std::pair<NodeID, NodeID>> edges;
  in_stream.ForEachDirected([&](const auto &edge) {
    edges.emplace_back(edge.source, edge.destination);
  });
  Require(
      edges == std::vector<std::pair<NodeID, NodeID>>{
          {2, 3}, {3, 4}},
      "isolated-row incoming identity differs");
}

void TestWeightedIncoming() {
  WGraph graph = MakeWeightedGraph();
  auto outgoing = graphbrew::edge::FlattenOutgoing(graph);
  auto incoming = graphbrew::edge::FlattenIncoming(graph);
  EdgeStream<decltype(outgoing)> out_stream(
      outgoing,
      graphbrew::edge::EdgeStorageOrder::kSourceMajor);
  EdgeStream<decltype(incoming)> stream(
      incoming,
      graphbrew::edge::EdgeStorageOrder::kDestinationMajor);
  std::vector<std::tuple<NodeID, NodeID, WeightT>> out_edges;
  out_stream.ForEachDirected([&](const auto &edge) {
    out_edges.emplace_back(
        edge.source, edge.destination, edge.weight);
  });
  Require(
      out_edges == std::vector<std::tuple<NodeID, NodeID, WeightT>>{
          {0, 1, 2}, {0, 2, 5}, {1, 2, 3}, {2, 3, 7}},
      "weighted outgoing edge stream differs");
  std::vector<std::tuple<NodeID, NodeID, WeightT>> edges;
  stream.ForEachDirected([&](const auto &edge) {
    edges.emplace_back(edge.source, edge.destination, edge.weight);
  });
  Require(
      edges == std::vector<std::tuple<NodeID, NodeID, WeightT>>{
          {0, 1, 2}, {0, 2, 5}, {1, 2, 3}, {2, 3, 7}},
      "weighted incoming edge stream differs");
}

void TestFrontierAndAtomics() {
  graphbrew::edge::Frontier<NodeID> frontier(8);
  frontier.Assign({3, 1, 3, 5});
  Require(frontier.size() == 3, "frontier deduplication differs");
  Require(
      frontier.Contains(1) && frontier.Contains(3) && frontier.Contains(5),
      "frontier dense membership differs");

  graphbrew::edge::Frontier<NodeID> boundary(130);
  boundary.Push(63);
  boundary.Push(64);
  boundary.Push(129);
  Require(boundary.Contains(63), "frontier lost bit 63");
  Require(boundary.Contains(64), "frontier lost bit 64");
  Require(boundary.Contains(129), "frontier lost final bit");

  graphbrew::edge::FrontierBuilder<NodeID> builder(8);
#pragma omp parallel for
  for (int index = 0; index < 8; ++index)
    builder.Push(static_cast<NodeID>(index % 4));
  auto built = builder.Finish();
  Require(built.size() == 4, "frontier builder deduplication differs");
  Require(
      built.sparse() == std::vector<NodeID>{0, 1, 2, 3},
      "frontier builder order differs");
  builder.Push(7);
  auto rebuilt = builder.Finish();
  Require(
      rebuilt.sparse() == std::vector<NodeID>{7},
      "frontier builder reuse differs");

  int value = 100;
#pragma omp parallel for
  for (int candidate = 99; candidate >= 0; --candidate)
    graphbrew::edge::AtomicMin(value, candidate);
  Require(value == 0, "AtomicMin differs");
  Require(
      graphbrew::edge::AtomicMax(value, 7) && value == 7,
      "AtomicMax differs");
  Require(
      graphbrew::edge::AtomicAssignIfEqual(value, 7, 9) && value == 9,
      "AtomicAssignIfEqual differs");
  graphbrew::edge::AtomicStore(value, 11);
  Require(
      graphbrew::edge::AtomicLoad(value) == 11,
      "AtomicStore differs");

  int maximum = -1;
  int claimed = -1;
  int claims = 0;
#pragma omp parallel for reduction(+ : claims)
  for (int candidate = 0; candidate < 1024; ++candidate) {
    graphbrew::edge::AtomicMax(maximum, candidate);
    if (graphbrew::edge::AtomicAssignIfEqual(
            claimed, -1, candidate)) {
      ++claims;
    }
  }
  Require(maximum == 1023, "parallel AtomicMax differs");
  Require(claims == 1, "parallel CAS admitted multiple winners");

  Graph graph = MakeDirectedGraph();
  FlatGraph flat = graph.flattenGraphOut();
  EdgeStream<FlatGraph> stream(
      flat, graphbrew::edge::EdgeStorageOrder::kSourceMajor);
  CountingAccessPolicy counting;
  std::atomic<std::size_t> visits{0};
  graphbrew::edge::ParallelForEachDirected(
      stream,
      [&](const auto &) {
        visits.fetch_add(1, std::memory_order_relaxed);
      },
      counting);
  Require(visits.load() == 6, "hooked edge map lost an edge");
  Require(
      counting.edges.load(std::memory_order_relaxed) == 6,
      "access policy did not observe every edge");

  graphbrew::edge::NoOpAccessPolicy access;
  access.OnVertex(0, graphbrew::edge::AccessKind::kVertexRead);
  access.OnEdge(std::make_pair(0, 1));
  access.OnBarrier(0);
}

}  // namespace

int main() {
  TestDirectedAndIncomingStreams();
  TestOrientedAndParallelStreams();
  TestEmptyAndIsolatedRows();
  TestWeightedIncoming();
  TestFrontierAndAtomics();
  std::cout << "test_edge_primitives: PASS\n";
  return 0;
}
