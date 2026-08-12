// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>
#include <cstdint> // For int64_t

#include "benchmark.h"
#include "bfs_common.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in the parent array as negative numbers. Thus, the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

#ifdef GRAPHBREW_COUNT_WORK
static int64_t g_bfs_td_edges = 0;
static int64_t g_bfs_bu_edges = 0;
static int64_t g_bfs_steps = 0;
#endif

int64_t BUStep(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next
#ifdef GRAPHBREW_COUNT_WORK
               , int64_t &edges_examined
#endif
               ) {
  int64_t awake_count = 0;
#ifdef GRAPHBREW_COUNT_WORK
  int64_t step_edges = 0;
#endif
  next.reset();
#ifdef GRAPHBREW_COUNT_WORK
  #pragma omp parallel for reduction(+ : awake_count, step_edges) schedule(dynamic, 1024)
#else
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
#endif
  for (NodeID u=0; u < g.num_nodes(); u++) {
    if (parent[u] < 0) {
      for (NodeID v : g.in_neigh(u)) {
#ifdef GRAPHBREW_COUNT_WORK
        step_edges++;
#endif
        if (front.get_bit(v)) {
          parent[u] = v;
          awake_count++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
#ifdef GRAPHBREW_COUNT_WORK
  edges_examined = step_edges;
#endif
  return awake_count;
}


int64_t TDStep(const Graph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue
#ifdef GRAPHBREW_COUNT_WORK
               , int64_t &edges_examined
#endif
               ) {
  int64_t scout_count = 0;
#ifdef GRAPHBREW_COUNT_WORK
  int64_t step_edges = 0;
#endif
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
#ifdef GRAPHBREW_COUNT_WORK
    #pragma omp for reduction(+ : scout_count, step_edges) nowait
#else
    #pragma omp for reduction(+ : scout_count) nowait
#endif
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter;
      for (NodeID v : g.out_neigh(u)) {
#ifdef GRAPHBREW_COUNT_WORK
        step_edges++;
#endif
        NodeID curr_val = parent[v];
        if (curr_val < 0) {
          if (compare_and_swap(parent[v], curr_val, u)) {
            lqueue.push_back(v);
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
#ifdef GRAPHBREW_COUNT_WORK
  edges_examined = step_edges;
#endif
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    NodeID u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for nowait
    for (NodeID n=0; n < g.num_nodes(); n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
  return parent;
}

pvector<NodeID> DOBFS(const Graph &g, NodeID source, bool logging_enabled = false,
                      int alpha = 15, int beta = 18) {
  using graphbrew::database::AppendBenchmarkIterationEntry;
  int bfs_step = 0;
  const bool record_iterations =
      graphbrew::database::SelfRecordingEnabled();
  const bool measure_steps = logging_enabled || record_iterations;
#ifdef GRAPHBREW_COUNT_WORK
  constexpr bool count_work = true;
#else
  constexpr bool count_work = false;
#endif
#ifdef GRAPHBREW_COUNT_WORK
  int64_t td_edges = 0;
  int64_t bu_edges = 0;
#endif
  if (logging_enabled)
    PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  if (measure_steps) t.Start();
  pvector<NodeID> parent = InitParent(g);
  if (measure_steps) t.Stop();
  if (logging_enabled)
    PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      if (measure_steps) t.Start();
      QueueToBitmap(queue, front);
      if (measure_steps) t.Stop();
      if (logging_enabled)
        PrintStep("e", t.Seconds());
      if (record_iterations)
        AppendBenchmarkIterationEntry({{"step", bfs_step}, {"phase", "e"}, {"time_s", t.Seconds()}});
      if (record_iterations || count_work) bfs_step++;
      awake_count = queue.size();
      queue.slide_window();
      do {
        if (measure_steps) t.Start();
        old_awake_count = awake_count;
#ifdef GRAPHBREW_COUNT_WORK
        int64_t step_edges = 0;
        awake_count = BUStep(
            g, parent, front, curr, step_edges);
        bu_edges += step_edges;
#else
        awake_count = BUStep(g, parent, front, curr);
#endif
        front.swap(curr);
        if (measure_steps) t.Stop();
        if (logging_enabled)
          PrintStep("bu", t.Seconds(), awake_count);
        if (record_iterations)
          AppendBenchmarkIterationEntry({{"step", bfs_step}, {"phase", "bu"}, {"time_s", t.Seconds()}, {"awake_count", awake_count}});
        if (record_iterations || count_work) bfs_step++;
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      if (measure_steps) t.Start();
      BitmapToQueue(g, front, queue);
      if (measure_steps) t.Stop();
      if (logging_enabled)
        PrintStep("c", t.Seconds());
      if (record_iterations)
        AppendBenchmarkIterationEntry({{"step", bfs_step}, {"phase", "c"}, {"time_s", t.Seconds()}});
      if (record_iterations || count_work) bfs_step++;
      scout_count = 1;
    } else {
      if (measure_steps) t.Start();
      edges_to_check -= scout_count;
#ifdef GRAPHBREW_COUNT_WORK
      int64_t step_edges = 0;
      scout_count = TDStep(g, parent, queue, step_edges);
      td_edges += step_edges;
#else
      scout_count = TDStep(g, parent, queue);
#endif
      queue.slide_window();
      if (measure_steps) t.Stop();
      if (logging_enabled)
        PrintStep("td", t.Seconds(), queue.size());
      if (record_iterations)
        AppendBenchmarkIterationEntry({{"step", bfs_step}, {"phase", "td"}, {"time_s", t.Seconds()}, {"scout_count", scout_count}, {"queue_size", static_cast<int64_t>(queue.size())}});
      if (record_iterations || count_work) bfs_step++;
    }
  }
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    if (parent[n] < -1)
      parent[n] = -1;
#ifdef GRAPHBREW_COUNT_WORK
  g_bfs_td_edges = td_edges;
  g_bfs_bu_edges = bu_edges;
  g_bfs_steps = bfs_step;
#endif
  return parent;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
  SetBenchmarkTypeHint(BENCH_BFS);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder b(cli);
  Graph g = b.MakeGraph();
  // Create SourcePicker with pre-generated consistent sources based on num_trials
  // This ensures all orderings use the same ORIGINAL vertex IDs as sources
  SourcePicker<Graph> sp(
      g, cli.start_vertices(), cli.num_trials(),
      cli.source_repeats());
  auto BFSBound = [&sp,&cli] (const Graph &g) {
    return DOBFS(g, sp.PickNext(), cli.logging_en());
  };
  std::unique_ptr<SourcePicker<Graph>> vsp;
  if (cli.do_verify()) {
    vsp = std::make_unique<SourcePicker<Graph>>(
        g, cli.start_vertices(), cli.num_trials(),
        cli.source_repeats());
  }
  auto VerifierBound = [&vsp] (const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp->PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound,
    "bfs",
    [&sp](const Graph &g, const pvector<NodeID> &parent) -> nlohmann::json {
      nlohmann::json ans;
      int64_t tree_size = 0, n_edges = 0;
      for (NodeID n = 0; n < g.num_nodes(); n++) {
        if (parent[n] >= 0) { n_edges += g.out_degree(n); tree_size++; }
      }
      ans["tree_nodes"] = tree_size;
      ans["tree_edges"] = n_edges;
      ans["source_original"] = sp.last_original_source();
      ans["source_internal"] = sp.last_internal_source();
      PrintLabel(
          "Source Original", std::to_string(sp.last_original_source()));
      PrintLabel(
          "Source Internal", std::to_string(sp.last_internal_source()));
      PrintLabel(
          "Source Out Degree",
          std::to_string(sp.last_source_out_degree()));
      ans["source_out_degree"] = sp.last_source_out_degree();
#ifdef GRAPHBREW_COUNT_WORK
      ans["top_down_edges_examined"] = g_bfs_td_edges;
      ans["bottom_up_edges_examined"] = g_bfs_bu_edges;
      ans["edges_examined"] = g_bfs_td_edges + g_bfs_bu_edges;
      ans["bfs_steps"] = g_bfs_steps;
      PrintLabel("BFS TD Edges", std::to_string(g_bfs_td_edges));
      PrintLabel("BFS BU Edges", std::to_string(g_bfs_bu_edges));
      PrintLabel(
          "BFS Edges Examined",
          std::to_string(g_bfs_td_edges + g_bfs_bu_edges));
      PrintLabel("BFS Steps", std::to_string(g_bfs_steps));
#endif
      return ans;
    });
  return 0;
}
