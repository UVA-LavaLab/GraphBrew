#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/algorithms/cc_gas.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_stream.h"

using graphbrew::algorithms::ConnectedComponentsResult;

bool CCVerifier(
    const Graph &graph,
    const ConnectedComponentsResult &result) {
  return graphbrew::algorithms::VerifyConnectedComponents(
      graph, result);
}

int main(int argc, char **argv) {
  CLApp cli(argc, argv, "connected-components-gas");
  if (!cli.ParseArgs())
    return -1;

  SetBenchmarkTypeHint(BENCH_CC);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder builder(cli);
  Graph graph = builder.MakeGraph();
  auto weak = graphbrew::edge::FlattenWeaklyConnected(graph);
  graphbrew::edge::NoOpAccessPolicy access_policy;
  auto kernel = [
      &cli,
      &weak,
      &access_policy](const Graph &kernel_graph) {
    return graphbrew::algorithms::ConnectedComponentsGAS(
        kernel_graph, weak, cli.logging_en(), access_policy);
  };

  BenchmarkKernel(
      cli,
      graph,
      kernel,
      graphbrew::algorithms::PrintConnectedComponentStats,
      CCVerifier,
      "cc_gas",
      graphbrew::algorithms::ConnectedComponentsSummary);
  return 0;
}
