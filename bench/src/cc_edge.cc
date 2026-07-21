#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/algorithms/connected_components_edge.h"
#include "graphbrew/edge/algorithm_access_policy.h"

using graphbrew::algorithms::ConnectedComponentsResult;

bool CCVerifier(
    const Graph &graph,
    const ConnectedComponentsResult &result) {
  return graphbrew::algorithms::VerifyConnectedComponents(
      graph, result);
}

int main(int argc, char **argv) {
  CLApp cli(argc, argv, "connected-components-afforest-edge");
  if (!cli.ParseArgs())
    return -1;

  SetBenchmarkTypeHint(BENCH_CC);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder builder(cli);
  Graph graph = builder.MakeGraph();
  FlatGraph outgoing = graph.flattenGraphOut();
  graphbrew::edge::NoOpAccessPolicy access_policy;

  auto kernel = [&cli, &outgoing, &access_policy](
                    const Graph &kernel_graph) {
    return graphbrew::algorithms::AfforestEdge(
        kernel_graph,
        outgoing,
        cli.logging_en(),
        2,
        access_policy);
  };
  BenchmarkKernel(
      cli,
      graph,
      kernel,
      graphbrew::algorithms::PrintConnectedComponentStats,
      CCVerifier,
      "cc_edge",
      graphbrew::algorithms::ConnectedComponentsSummary);
  return 0;
}
