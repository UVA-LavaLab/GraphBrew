#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/algorithms/sssp_gas.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_stream.h"

bool SSSPVerifier(
    const WGraph &graph,
    const NodeID source,
    const pvector<WeightT> &distance) {
  return graphbrew::algorithms::VerifySSSP(
      graph, source, distance);
}

int main(int argc, char **argv) {
  CLDelta<WeightT> cli(
      argc, argv, "single-source-shortest-path-gas");
  if (!cli.ParseArgs())
    return -1;

  SetBenchmarkTypeHint(BENCH_SSSP);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  WeightedBuilder builder(cli);
  WGraph graph = builder.MakeGraph();
  auto outgoing = graphbrew::edge::FlattenOutgoing(graph);
  auto incoming = graphbrew::edge::FlattenIncoming(graph);
  graphbrew::edge::NoOpAccessPolicy access_policy;

  SourcePicker<WGraph> source_picker(
      graph, cli.start_vertex(), cli.num_trials());
  auto kernel = [
      &source_picker,
      &cli,
      &outgoing,
      &incoming,
      &access_policy](const WGraph &kernel_graph) {
    return graphbrew::algorithms::SSSPGAS(
        kernel_graph,
        incoming,
        outgoing,
        source_picker.PickNext(),
        cli.logging_en(),
        access_policy);
  };
  SourcePicker<WGraph> verifier_source_picker(
      graph, cli.start_vertex(), cli.num_trials());
  auto verifier = [&verifier_source_picker](
                      const WGraph &kernel_graph,
                      const pvector<WeightT> &distance) {
    return SSSPVerifier(
        kernel_graph,
        verifier_source_picker.PickNext(),
        distance);
  };

  BenchmarkKernel(
      cli,
      graph,
      kernel,
      graphbrew::algorithms::PrintSSSPEdgeStats,
      verifier,
      "sssp_gas",
      graphbrew::algorithms::SSSPSummary);
  return 0;
}
