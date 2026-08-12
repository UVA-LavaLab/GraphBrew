// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

/*
   GAP Benchmark Suite
   Kernel: PageRank (PR)
   Author: Scott Beamer

   Will return pagerank scores for all vertices once total change < epsilon

   This PR implementation uses the traditional iterative approach. It performs
   updates in the pull direction to remove the need for atomics, and it allows
   new values to be immediately visible (like Gauss-Seidel method). The prior PR
   implementation is still available in src/pr_spmv.cc.
 */

using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;

// Global variable to expose iteration count to the benchmark harness.
// Set by PageRankPullGS after convergence; read by main() for JSON output.
static int g_pr_iterations_to_convergence = -1;
static int g_pr_iterations_executed = 0;
static double g_pr_final_error = 0.0;
static bool g_pr_fixed_work = false;

pvector<ScoreT> PageRankPullGS(const Graph &g, int max_iters,
                               double epsilon = 0,
                               bool logging_enabled = false,
                               bool fixed_work = false) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contrib(g.num_nodes());
  int converged_iter = -1;
  int executed_iters = 0;
  double final_error = 0.0;
#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    const auto degree = g.out_degree(n);
    outgoing_contrib[n] =
        degree ? init_score / degree : ScoreT(0);
  }
  for (int iter = 0; iter < max_iters; iter++) {
    double error = 0;
#pragma omp parallel for reduction(+ : error) schedule(dynamic, 16384)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      ScoreT incoming_total = 0;
      for (NodeID v : g.in_neigh(u)) {
        ScoreT contribution;
        __atomic_load(
            &outgoing_contrib[v], &contribution, __ATOMIC_RELAXED);
        incoming_total += contribution;
      }
      ScoreT old_score = scores[u];
      scores[u] = base_score + kDamp * incoming_total;
      error += fabs(scores[u] - old_score);
      const auto degree = g.out_degree(u);
      const ScoreT contribution =
          degree ? scores[u] / degree : ScoreT(0);
      __atomic_store(
          &outgoing_contrib[u], &contribution, __ATOMIC_RELAXED);
    }
    executed_iters = iter + 1;
    final_error = error;
    if (logging_enabled)
      PrintStep(iter, error);
    if (graphbrew::database::SelfRecordingEnabled())
      graphbrew::database::AppendBenchmarkIterationEntry(
          {{"iter", iter}, {"error", error}});
    if (converged_iter < 0 && error < epsilon)
      converged_iter = iter + 1;
    if (!fixed_work && error < epsilon)
      break;
  }
  g_pr_iterations_to_convergence = converged_iter;
  g_pr_iterations_executed = executed_iters;
  g_pr_final_error = final_error;
  g_pr_fixed_work = fixed_work;
  return scores;
}

void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  NodeID *temp_org_ids = g.get_org_ids();

  NodeID *org_ids_inv_ = new NodeID[g.num_nodes()];
#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    org_ids_inv_[temp_org_ids[n]] = n;
  }

  for (NodeID n = 0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(org_ids_inv_[n], scores[n]);
  }
  int k = 100;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incoming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    const auto degree = g.out_degree(u);
    ScoreT outgoing_contrib = degree ? scores[u] / degree : ScoreT(0);
    for (NodeID v : g.out_neigh(u))
      incoming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incoming_sums[n] - scores[n]);
    incoming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

int main(int argc, char *argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  SetBenchmarkTypeHint(BENCH_PR);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder b(cli);
  Graph g = b.MakeGraph();
  PrintLabel("PR Mode", cli.fixed_work() ? "fixed-work" : "convergence");

  auto PRBound = [&cli](const Graph &g) {
    return PageRankPullGS(g, cli.max_iters(), cli.tolerance(),
                          cli.logging_en(), cli.fixed_work());
  };
  auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound,
    "pr",
    [](const Graph &g, const pvector<ScoreT> &scores) -> nlohmann::json {
      nlohmann::json ans;
      // Sum scores for total_score check
      double total = 0;
      ScoreT max_score = 0;
      for (NodeID n = 0; n < g.num_nodes(); n++) {
        total += scores[n];
        if (scores[n] > max_score) max_score = scores[n];
      }
      ans["total_score"] = total;
      ans["max_score"] = static_cast<double>(max_score);
      ans["iterations_to_convergence"] = g_pr_iterations_to_convergence;
      ans["converged"] = g_pr_iterations_to_convergence > 0;
      ans["iterations_executed"] = g_pr_iterations_executed;
      ans["final_error"] = g_pr_final_error;
      ans["mode"] = g_pr_fixed_work ? "fixed-work" : "convergence";
      ans["directed_edges_processed"] =
          static_cast<int64_t>(g_pr_iterations_executed) *
          g.num_edges_directed();
      PrintTime("Iterations", g_pr_iterations_executed);
      printf("%-21s%.17g\n", "Final Error:", g_pr_final_error);
      return ans;
    });
  return 0;
}
