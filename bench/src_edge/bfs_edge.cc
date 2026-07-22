#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/algorithms/bfs_edge.h"
#include "graphbrew/bfs_common.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_stream.h"

int main(int argc, char **argv) {
  CLApp cli(argc, argv, "breadth-first-search-edge");
  if (!cli.ParseArgs())
    return -1;

  SetBenchmarkTypeHint(BENCH_BFS);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder builder(cli);
  Graph graph = builder.MakeGraph();
  FlatGraph outgoing = graph.flattenGraphOut();
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  graphbrew::edge::NoOpAccessPolicy access_policy;

  SourcePicker<Graph> source_picker(
      graph, cli.start_vertex(), cli.num_trials());
  auto kernel = [
      &source_picker,
      &cli,
      &outgoing,
      &incoming,
      &access_policy](const Graph &kernel_graph) {
    return graphbrew::algorithms::DirectionOptimizingBFSEdge(
        kernel_graph,
        outgoing,
        incoming,
        source_picker.PickNext(),
        cli.logging_en(),
        access_policy);
  };
  SourcePicker<Graph> verifier_source_picker(
      graph, cli.start_vertex(), cli.num_trials());
  auto verifier = [&verifier_source_picker](
                      const Graph &kernel_graph,
                      const pvector<NodeID> &parent) {
    return BFSVerifier(
        kernel_graph,
        verifier_source_picker.PickNext(),
        parent);
  };

  BenchmarkKernel(
      cli,
      graph,
      kernel,
      PrintBFSStats,
      verifier,
      "bfs_edge",
      graphbrew::algorithms::BFSSummary);
  return 0;
}
