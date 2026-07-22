#include <iostream>
#include <memory>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/algorithms/tc_edge.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_stream.h"

bool TCVerifier(
    const Graph &graph,
    const std::size_t triangles) {
  return graphbrew::algorithms::VerifyTriangleCount(
      graph, triangles);
}

int main(int argc, char **argv) {
  CLApp cli(argc, argv, "triangle-count-edge");
  if (!cli.ParseArgs())
    return -1;

  SetBenchmarkTypeHint(BENCH_TC);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder builder(cli);
  Graph graph = builder.MakeGraph();
  if (graph.directed()) {
    std::cout << "Input graph is directed but tc requires undirected"
              << std::endl;
    return -2;
  }

  std::unique_ptr<Graph> relabelled;
  const Graph *count_graph = &graph;
  if (graphbrew::algorithms::WorthRelabellingForTC(graph)) {
    relabelled = std::make_unique<Graph>(
        Builder::RelabelByDegree(graph));
    count_graph = relabelled.get();
  }
  FlatGraph outgoing =
      graphbrew::edge::FlattenOutgoing(*count_graph);
  graphbrew::edge::NoOpAccessPolicy access_policy;
  auto kernel = [
      &outgoing,
      &access_policy](const Graph &) {
    return graphbrew::algorithms::TriangleCountEdge(
        outgoing, access_policy);
  };

  BenchmarkKernel(
      cli,
      graph,
      kernel,
      graphbrew::algorithms::PrintTriangleEdgeStats,
      TCVerifier,
      "tc_edge",
      graphbrew::algorithms::TriangleCountSummary);
  return 0;
}
