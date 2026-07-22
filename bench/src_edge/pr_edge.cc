#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/algorithms/pagerank_edge.h"
#include "graphbrew/edge/algorithm_access_policy.h"
#include "graphbrew/edge/edge_stream.h"

using graphbrew::algorithms::PageRankResult;

bool PRVerifier(
    const Graph &graph,
    const PageRankResult &result,
    const double target_error) {
  return graphbrew::algorithms::VerifyPageRank(
      graph, result.scores, target_error);
}

int main(int argc, char **argv) {
  CLPageRank cli(argc, argv, "pagerank-edge", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;

  SetBenchmarkTypeHint(BENCH_PR);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder builder(cli);
  Graph graph = builder.MakeGraph();
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  graphbrew::edge::NoOpAccessPolicy access_policy;

  auto kernel = [&cli, &incoming, &access_policy](
                    const Graph &kernel_graph) {
    return graphbrew::algorithms::PageRankAsyncEdge(
        kernel_graph,
        incoming,
        cli.max_iters(),
        cli.tolerance(),
        cli.logging_en(),
        access_policy);
  };
  auto verifier = [&cli](
                      const Graph &kernel_graph,
                      const PageRankResult &result) {
    return PRVerifier(
        kernel_graph, result, cli.tolerance());
  };
  auto statistics = [](
                        const Graph &kernel_graph,
                        const PageRankResult &result) {
    graphbrew::algorithms::PrintPageRankTopScores(
        kernel_graph, result, 100, true);
  };

  BenchmarkKernel(
      cli,
      graph,
      kernel,
      statistics,
      verifier,
      "pr_edge",
      [](const Graph &kernel_graph, const PageRankResult &result) {
        return graphbrew::algorithms::PageRankSummary(
            kernel_graph, result, true);
      });
  return 0;
}
