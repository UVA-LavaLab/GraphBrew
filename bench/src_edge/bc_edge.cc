#include <iostream>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/algorithms/bc_edge.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_stream.h"

bool BCVerifier(
    const Graph &graph,
    SourcePicker<Graph> &source_picker,
    const NodeID num_sources,
    const pvector<graphbrew::algorithms::BCScore> &scores) {
  return graphbrew::algorithms::VerifyBC(
      graph, source_picker, num_sources, scores);
}

int main(int argc, char **argv) {
  CLIterApp cli(argc, argv, "betweenness-centrality-edge", 1);
  if (!cli.ParseArgs())
    return -1;

  SetBenchmarkTypeHint(BENCH_BC);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  if (cli.num_iters() > 1 && cli.start_vertex() != -1)
    std::cout << "Warning: iterating from same source (-r & -i)"
              << std::endl;
  Builder builder(cli);
  Graph graph = builder.MakeGraph();
  FlatGraph outgoing = graphbrew::edge::FlattenOutgoing(graph);
  graphbrew::edge::NoOpAccessPolicy access_policy;

  SourcePicker<Graph> source_picker(
      graph, cli.start_vertex(), cli.num_trials());
  auto kernel = [
      &source_picker,
      &cli,
      &outgoing,
      &access_policy](const Graph &kernel_graph) {
    return graphbrew::algorithms::BrandesEdge(
        kernel_graph,
        outgoing,
        source_picker,
        cli.num_iters(),
        cli.logging_en(),
        access_policy);
  };
  SourcePicker<Graph> verifier_source_picker(
      graph, cli.start_vertex(), cli.num_trials());
  auto verifier = [
      &verifier_source_picker,
      &cli](
      const Graph &kernel_graph,
      const pvector<graphbrew::algorithms::BCScore> &scores) {
    return BCVerifier(
        kernel_graph,
        verifier_source_picker,
        cli.num_iters(),
        scores);
  };

  BenchmarkKernel(
      cli,
      graph,
      kernel,
      graphbrew::algorithms::PrintBCEdgeScores,
      verifier,
      "bc_edge",
      graphbrew::algorithms::BCSummary);
  return 0;
}
