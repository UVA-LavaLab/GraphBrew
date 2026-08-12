// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"
#include "util.h"


/*
GAP Benchmark Suite
Kernel: Betweenness Centrality (BC)
Author: Scott Beamer

Will return array of approx betweenness centrality scores for each vertex

This BC implementation makes use of the Brandes [1] algorithm with
implementation optimizations from Madduri et al. [2]. It is only approximate
because it does not compute the paths from every start vertex, but only a small
subset of them. Additionally, the scores are normalized to the range [0,1].

As an optimization to save memory, this implementation uses a Bitmap to hold
succ (list of successors) found during the BFS phase that are used in the back-
propagation phase.

[1] Ulrik Brandes. "A faster algorithm for betweenness centrality." Journal of
    Mathematical Sociology, 25(2):163–177, 2001.

[2] Kamesh Madduri, David Ediger, Karl Jiang, David A Bader, and Daniel
    Chavarria-Miranda. "A faster parallel algorithm and efficient multithreaded
    implementations for evaluating betweenness centrality on massive datasets."
    International Symposium on Parallel & Distributed Processing (IPDPS), 2009.
*/


using namespace std;
typedef float ScoreT;
typedef double CountT;
constexpr ScoreT kBCAbsoluteTolerance = 1e-5f;
constexpr ScoreT kBCRelativeTolerance = 1e-5f;

constexpr bool BCScoreWithinTolerance(ScoreT expected, ScoreT actual) {
  ScoreT delta = expected > actual ? expected - actual : actual - expected;
  ScoreT scale = expected > actual ? expected : actual;
  return delta <= kBCAbsoluteTolerance + kBCRelativeTolerance * scale;
}

static_assert(BCScoreWithinTolerance(1.0f, 1.0f - 2.4e-7f));
static_assert(!BCScoreWithinTolerance(0.5f, 0.501f));

#ifdef GRAPHBREW_COUNT_WORK
static int64_t g_bc_bfs_edges = 0;
static int64_t g_bc_backprop_edges = 0;
static int64_t g_bc_max_depth = 0;
#endif


void PBFS(const Graph &g, NodeID source, pvector<CountT> &path_counts,
    Bitmap &succ, vector<SlidingQueue<NodeID>::iterator> &depth_index,
    SlidingQueue<NodeID> &queue
#ifdef GRAPHBREW_COUNT_WORK
    , int64_t &edges_examined
#endif
    ) {
  pvector<NodeID> depths(g.num_nodes(), -1);
  depths[source] = 0;
  path_counts[source] = 1;
  queue.push_back(source);
  depth_index.push_back(queue.begin());
  queue.slide_window();
  const NodeID* g_out_start = g.out_neigh(0).begin();
#ifdef GRAPHBREW_COUNT_WORK
  int64_t total_edges = 0;
#endif
  #pragma omp parallel
  {
#ifdef GRAPHBREW_COUNT_WORK
    int64_t local_edges = 0;
#endif
    NodeID depth = 0;
    QueueBuffer<NodeID> lqueue(queue);
    while (!queue.empty()) {
      depth++;
      #pragma omp for schedule(dynamic, 64) nowait
      for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        NodeID u = *q_iter;
        for (NodeID &v : g.out_neigh(u)) {
#ifdef GRAPHBREW_COUNT_WORK
          local_edges++;
#endif
          if ((depths[v] == -1) &&
              (compare_and_swap(depths[v], static_cast<NodeID>(-1), depth))) {
            lqueue.push_back(v);
          }
          if (depths[v] == depth) {
            succ.set_bit_atomic(&v - g_out_start);
            #pragma omp atomic
            path_counts[v] += path_counts[u];
          }
        }
      }
      lqueue.flush();
      #pragma omp barrier
      #pragma omp single
      {
        depth_index.push_back(queue.begin());
        queue.slide_window();
      }
    }
#ifdef GRAPHBREW_COUNT_WORK
    #pragma omp atomic
    total_edges += local_edges;
#endif
  }
  depth_index.push_back(queue.begin());
#ifdef GRAPHBREW_COUNT_WORK
  edges_examined = total_edges;
#endif
}


pvector<ScoreT> Brandes(const Graph &g, SourcePicker<Graph> &sp,
                        NodeID num_iters, bool logging_enabled = false) {
  using graphbrew::database::AppendBenchmarkIterationEntry;
  const bool record_iterations =
      graphbrew::database::SelfRecordingEnabled();
  const bool measure_steps = logging_enabled || record_iterations;
  Timer t;
  if (measure_steps) t.Start();
  pvector<ScoreT> scores(g.num_nodes(), 0);
  pvector<CountT> path_counts(g.num_nodes());
  Bitmap succ(g.num_edges_directed());
  vector<SlidingQueue<NodeID>::iterator> depth_index;
  SlidingQueue<NodeID> queue(g.num_nodes());
  if (measure_steps) t.Stop();
  if (logging_enabled)
    PrintStep("a", t.Seconds());
  const NodeID* g_out_start = g.out_neigh(0).begin();
#ifdef GRAPHBREW_COUNT_WORK
  int64_t bfs_edges = 0;
  int64_t backprop_edges = 0;
  int64_t max_depth = 0;
#endif
  for (NodeID iter=0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    if (logging_enabled)
      PrintStep("Source", static_cast<int64_t>(source));
    if (measure_steps) t.Start();
    path_counts.fill(0);
    depth_index.resize(0);
    queue.reset();
    succ.reset();
#ifdef GRAPHBREW_COUNT_WORK
    int64_t source_bfs_edges = 0;
    PBFS(
        g, source, path_counts, succ, depth_index, queue,
        source_bfs_edges);
    bfs_edges += source_bfs_edges;
#else
    PBFS(g, source, path_counts, succ, depth_index, queue);
#endif
    if (measure_steps) t.Stop();
    double bfs_time = t.Seconds();
    if (logging_enabled)
      PrintStep("b", bfs_time);
    pvector<ScoreT> deltas(g.num_nodes(), 0);
    if (measure_steps) t.Start();
#ifdef GRAPHBREW_COUNT_WORK
    max_depth = max<int64_t>(
        max_depth,
        depth_index.size() > 1 ? depth_index.size() - 1 : 0);
#endif
    for (int d=depth_index.size()-2; d >= 0; d--) {
#ifdef GRAPHBREW_COUNT_WORK
      int64_t level_edges = 0;
      #pragma omp parallel for reduction(+ : level_edges) schedule(dynamic, 64)
#else
      #pragma omp parallel for schedule(dynamic, 64)
#endif
      for (auto it = depth_index[d]; it < depth_index[d+1]; it++) {
        NodeID u = *it;
        ScoreT delta_u = 0;
        for (NodeID &v : g.out_neigh(u)) {
#ifdef GRAPHBREW_COUNT_WORK
          level_edges++;
#endif
          if (succ.get_bit(&v - g_out_start)) {
            delta_u += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
          }
        }
        deltas[u] = delta_u;
        scores[u] += delta_u;
      }
#ifdef GRAPHBREW_COUNT_WORK
      backprop_edges += level_edges;
#endif
    }
    if (measure_steps) t.Stop();
    double backprop_time = t.Seconds();
    if (logging_enabled)
      PrintStep("p", backprop_time);
    if (record_iterations)
      AppendBenchmarkIterationEntry({
          {"source_iter", static_cast<int64_t>(iter)},
          {"source_id", static_cast<int64_t>(source)},
          {"bfs_time_s", bfs_time},
          {"backprop_time_s", backprop_time},
          {"depth", static_cast<int64_t>(depth_index.size() > 1 ? depth_index.size() - 1 : 0)}});
  }
  // normalize scores
  ScoreT biggest_score = 0;
  #pragma omp parallel for reduction(max : biggest_score)
  for (NodeID n=0; n < g.num_nodes(); n++)
    biggest_score = max(biggest_score, scores[n]);
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    scores[n] = scores[n] / biggest_score;
#ifdef GRAPHBREW_COUNT_WORK
  g_bc_bfs_edges = bfs_edges;
  g_bc_backprop_edges = backprop_edges;
  g_bc_max_depth = max_depth;
#endif
  return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n : g.vertices())
    score_pairs[n] = make_pair(n, scores[n]);
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
bool BCVerifier(const Graph &g, SourcePicker<Graph> &sp, NodeID num_iters,
                const pvector<ScoreT> &scores_to_test) {
  pvector<ScoreT> scores(g.num_nodes(), 0);
  for (int iter=0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    // BFS phase, only records depth & path_counts
    pvector<int> depths(g.num_nodes(), -1);
    depths[source] = 0;
    vector<CountT> path_counts(g.num_nodes(), 0);
    path_counts[source] = 1;
    vector<NodeID> to_visit;
    to_visit.reserve(g.num_nodes());
    to_visit.push_back(source);
    for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (depths[v] == -1) {
          depths[v] = depths[u] + 1;
          to_visit.push_back(v);
        }
        if (depths[v] == depths[u] + 1)
          path_counts[v] += path_counts[u];
      }
    }
    // Get lists of vertices at each depth
    vector<vector<NodeID>> verts_at_depth;
    for (NodeID n : g.vertices()) {
      if (depths[n] != -1) {
        if (depths[n] >= static_cast<int>(verts_at_depth.size()))
          verts_at_depth.resize(depths[n] + 1);
        verts_at_depth[depths[n]].push_back(n);
      }
    }
    // Going from farthest to closest, compute "dependencies" (deltas)
    pvector<ScoreT> deltas(g.num_nodes(), 0);
    for (int depth=verts_at_depth.size()-1; depth >= 0; depth--) {
      for (NodeID u : verts_at_depth[depth]) {
        for (NodeID v : g.out_neigh(u)) {
          if (depths[v] == depths[u] + 1) {
            deltas[u] += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
          }
        }
        scores[u] += deltas[u];
      }
    }
  }
  // Normalize scores
  ScoreT biggest_score = *max_element(scores.begin(), scores.end());
  for (NodeID n : g.vertices())
    scores[n] = scores[n] / biggest_score;
  // Compare scores
  bool all_ok = true;
  ScoreT max_abs_delta = 0;
  uint64_t watch_count = 0;
  for (NodeID n : g.vertices()) {
    ScoreT delta = abs(scores_to_test[n] - scores[n]);
    max_abs_delta = max(max_abs_delta, delta);
    if (delta > 1e-6f)
      watch_count++;
    if (!BCScoreWithinTolerance(scores[n], scores_to_test[n])) {
      cout << n << ": " << scores[n] << " != " << scores_to_test[n];
      cout << "(" << delta << ")" << endl;
      all_ok = false;
    }
  }
  ostringstream max_delta_text;
  max_delta_text << scientific << setprecision(9) << max_abs_delta;
  PrintLabel("BC Verify Max Abs Delta", max_delta_text.str());
  PrintLabel("BC Verify Delta Above 1e-6", to_string(watch_count));
  return all_ok;
}


int main(int argc, char* argv[]) {
  CLIterApp cli(argc, argv, "betweenness-centrality", 1);
  if (!cli.ParseArgs())
    return -1;
  SetBenchmarkTypeHint(BENCH_BC);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  if (!cli.start_vertices().empty() && cli.num_iters() != 1)
    throw std::invalid_argument(
        "Explicit BC source lists require one Brandes source per trial");
  Builder b(cli);
  Graph g = b.MakeGraph();
  // Create SourcePicker with pre-generated consistent sources based on num_trials
  // This ensures all orderings use the same ORIGINAL vertex IDs as sources
  SourcePicker<Graph> sp(
      g, cli.start_vertices(),
      cli.start_vertices().empty()
          ? cli.num_trials() * cli.num_iters()
          : cli.num_trials(),
      cli.source_repeats());
  auto BCBound = [&sp, &cli] (const Graph &g) {
    return Brandes(g, sp, cli.num_iters(), cli.logging_en());
  };
  std::unique_ptr<SourcePicker<Graph>> vsp;
  if (cli.do_verify()) {
    vsp = std::make_unique<SourcePicker<Graph>>(
        g, cli.start_vertices(),
        cli.start_vertices().empty()
            ? cli.num_trials() * cli.num_iters()
            : cli.num_trials(),
        cli.source_repeats());
  }
  auto VerifierBound = [&vsp, &cli] (const Graph &g,
                                     const pvector<ScoreT> &scores) {
    return BCVerifier(g, *vsp, cli.num_iters(), scores);
  };
  BenchmarkKernel(cli, g, BCBound, PrintTopScores, VerifierBound,
    "bc",
    [&sp](const Graph &g, const pvector<ScoreT> &scores) -> nlohmann::json {
      nlohmann::json ans;
      ScoreT max_centrality = *std::max_element(scores.begin(), scores.end());
      ans["max_centrality"] = static_cast<double>(max_centrality);
      auto sources = sp.TakePickedSources();
      nlohmann::json source_originals = nlohmann::json::array();
      nlohmann::json source_internals = nlohmann::json::array();
      nlohmann::json source_out_degrees = nlohmann::json::array();
      for (const auto& source : sources) {
        source_originals.push_back(source.original);
        source_internals.push_back(source.internal);
        source_out_degrees.push_back(source.out_degree);
        PrintLabel(
            "Source Original", std::to_string(source.original));
        PrintLabel(
            "Source Internal", std::to_string(source.internal));
        PrintLabel(
            "Source Out Degree", std::to_string(source.out_degree));
      }
      ans["source_originals"] = std::move(source_originals);
      ans["source_internals"] = std::move(source_internals);
      ans["source_out_degrees"] = std::move(source_out_degrees);
#ifdef GRAPHBREW_COUNT_WORK
      ans["bfs_edges_examined"] = g_bc_bfs_edges;
      ans["backprop_edges_examined"] = g_bc_backprop_edges;
      ans["max_depth"] = g_bc_max_depth;
      PrintLabel("BC BFS Edges", std::to_string(g_bc_bfs_edges));
      PrintLabel(
          "BC Backprop Edges",
          std::to_string(g_bc_backprop_edges));
      PrintLabel("BC Max Depth", std::to_string(g_bc_max_depth));
#endif
      return ans;
    });
  return 0;
}
