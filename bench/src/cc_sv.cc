// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"


/*
GAP Benchmark Suite
Kernel: Connected Components (CC)
Author: Scott Beamer

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Shiloach-Vishkin [2] algorithm with
implementation optimizations from Bader et al. [1]. Michael Sutton contributed
a fix for directed graphs using the min-max swap from [3], and it also produces
more consistent performance for undirected graphs.

[1] David A Bader, Guojing Cong, and John Feo. "On the architectural
    requirements for efficient execution of graph algorithms." International
    Conference on Parallel Processing, Jul 2005.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.

[3] Kishore Kothapalli, Jyothish Soman, and P. J. Narayanan. "Fast GPU
    algorithms for graph connectivity." Workshop on Large Scale Parallel
    Processing, 2010.
*/


using namespace std;

#ifdef GRAPHBREW_COUNT_WORK
static int64_t g_cc_sv_iterations = 0;
static int64_t g_cc_sv_edges_examined = 0;
static int64_t g_cc_sv_compress_steps = 0;
#endif

inline NodeID LoadComp(const pvector<NodeID>& comp, NodeID node) {
  return __atomic_load_n(&comp[node], __ATOMIC_RELAXED);
}

inline void StoreComp(
    pvector<NodeID>& comp, NodeID node, NodeID value) {
  __atomic_store_n(&comp[node], value, __ATOMIC_RELAXED);
}

// The hooking condition (comp_u < comp_v) may not coincide with the edge's
// direction, so we use a min-max swap such that lower component IDs propagate
// independent of the edge's direction.
pvector<NodeID> ShiloachVishkin(const Graph &g) {
  pvector<NodeID> comp(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    comp[n] = n;
  std::atomic<bool> change(true);
#ifdef GRAPHBREW_COUNT_WORK
  int num_iter = 0;
#endif
#ifdef GRAPHBREW_COUNT_WORK
  int64_t edges_examined = 0;
  int64_t compress_steps = 0;
#endif
  while (change.load(std::memory_order_relaxed)) {
    change.store(false, std::memory_order_relaxed);
#ifdef GRAPHBREW_COUNT_WORK
    num_iter++;
#endif
#ifdef GRAPHBREW_COUNT_WORK
    int64_t iteration_edges = 0;
    #pragma omp parallel for reduction(+ : iteration_edges)
#else
    #pragma omp parallel for
#endif
    for (NodeID u=0; u < g.num_nodes(); u++) {
      for (NodeID v : g.out_neigh(u)) {
#ifdef GRAPHBREW_COUNT_WORK
        iteration_edges++;
#endif
        NodeID comp_u = LoadComp(comp, u);
        NodeID comp_v = LoadComp(comp, v);
        if (comp_u == comp_v) continue;
        // Hooking condition so lower component ID wins independent of direction
        NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
        NodeID low_comp = comp_u + (comp_v - high_comp);
        if (
            high_comp == LoadComp(comp, high_comp) &&
            compare_and_swap(
                comp[high_comp], high_comp, low_comp)) {
          change.store(true, std::memory_order_relaxed);
        }
      }
    }
#ifdef GRAPHBREW_COUNT_WORK
    edges_examined += iteration_edges;
    int64_t iteration_compress_steps = 0;
    #pragma omp parallel for reduction(+ : iteration_compress_steps)
#else
    #pragma omp parallel for
#endif
    for (NodeID n=0; n < g.num_nodes(); n++) {
      NodeID parent = LoadComp(comp, n);
      NodeID grandparent = LoadComp(comp, parent);
      while (parent != grandparent) {
        StoreComp(comp, n, grandparent);
#ifdef GRAPHBREW_COUNT_WORK
        iteration_compress_steps++;
#endif
        parent = grandparent;
        grandparent = LoadComp(comp, parent);
      }
    }
#ifdef GRAPHBREW_COUNT_WORK
    compress_steps += iteration_compress_steps;
#endif
  }
#ifdef GRAPHBREW_COUNT_WORK
  g_cc_sv_iterations = num_iter;
  g_cc_sv_edges_examined = edges_examined;
  g_cc_sv_compress_steps = compress_steps;
#endif
  return comp;
}


void PrintCompStats(const Graph &g, const pvector<NodeID> &comp) {
  cout << endl;
  unordered_map<NodeID, NodeID> count;
  for (NodeID comp_i : comp)
    count[comp_i] += 1;
  int k = 5;
  vector<pair<NodeID, NodeID>> count_vector;
  count_vector.reserve(count.size());
  for (auto kvp : count)
    count_vector.push_back(kvp);
  vector<pair<NodeID, NodeID>> top_k = TopK(count_vector, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout << k << " biggest clusters" << endl;
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
  cout << "There are " << count.size() << " components" << endl;
}


// Verifies CC result by performing a BFS from a vertex in each component
// - Asserts search does not reach a vertex with a different component label
// - If the graph is directed, it performs the search as if it was undirected
// - Asserts every vertex is visited (degree-0 vertex should have own label)
bool CCVerifier(const Graph &g, const pvector<NodeID> &comp) {
  unordered_map<NodeID, NodeID> label_to_source;
  for (NodeID n : g.vertices())
    label_to_source[comp[n]] = n;
  Bitmap visited(g.num_nodes());
  visited.reset();
  vector<NodeID> frontier;
  frontier.reserve(g.num_nodes());
  for (auto label_source_pair : label_to_source) {
    NodeID curr_label = label_source_pair.first;
    NodeID source = label_source_pair.second;
    frontier.clear();
    frontier.push_back(source);
    visited.set_bit(source);
    for (auto it = frontier.begin(); it != frontier.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (comp[v] != curr_label)
          return false;
        if (!visited.get_bit(v)) {
          visited.set_bit(v);
          frontier.push_back(v);
        }
      }
      if (g.directed()) {
        for (NodeID v : g.in_neigh(u)) {
          if (comp[v] != curr_label)
            return false;
          if (!visited.get_bit(v)) {
            visited.set_bit(v);
            frontier.push_back(v);
          }
        }
      }
    }
  }
  for (NodeID n=0; n < g.num_nodes(); n++)
    if (!visited.get_bit(n))
      return false;
  return true;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "connected-components");
  if (!cli.ParseArgs())
    return -1;
  SetBenchmarkTypeHint(BENCH_CC_SV);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder b(cli);
  Graph g = b.MakeGraph();
  BenchmarkKernel(cli, g, ShiloachVishkin, PrintCompStats, CCVerifier,
    "cc_sv",
    [](const Graph &g, const pvector<NodeID> &comp) -> nlohmann::json {
      nlohmann::json ans;
      std::unordered_map<NodeID, NodeID> count;
      for (NodeID n = 0; n < g.num_nodes(); n++) count[comp[n]]++;
      ans["num_components"] = static_cast<int64_t>(count.size());
      NodeID largest = 0;
      for (auto& kv : count) if (kv.second > largest) largest = kv.second;
      ans["largest_component"] = largest;
#ifdef GRAPHBREW_COUNT_WORK
      ans["iterations"] = g_cc_sv_iterations;
      ans["edges_examined"] = g_cc_sv_edges_examined;
      ans["compress_steps"] = g_cc_sv_compress_steps;
      PrintLabel(
          "CC-SV Iterations",
          std::to_string(g_cc_sv_iterations));
      PrintLabel(
          "CC-SV Edges Examined",
          std::to_string(g_cc_sv_edges_examined));
      PrintLabel(
          "CC-SV Compress Steps",
          std::to_string(g_cc_sv_compress_steps));
#endif
      return ans;
    });
  return 0;
}
