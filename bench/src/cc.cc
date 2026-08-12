// Copyright (c) 2018, The Hebrew University of Jerusalem (HUJI, A. Barak)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "graphbrew/analysis/adaptive_source_policy.h"
#include "pvector.h"


/*
GAP Benchmark Suite
Kernel: Connected Components (CC)
Authors: Michael Sutton, Scott Beamer

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Afforest subgraph sampling algorithm [1],
which restructures and extends the Shiloach-Vishkin algorithm [2].

[1] Michael Sutton, Tal Ben-Nun, and Amnon Barak. "Optimizing Parallel 
    Graph Connectivity Computation via Subgraph Sampling" Symposium on 
    Parallel and Distributed Processing, IPDPS 2018.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.
*/


using namespace std;

class CLConnectedComponents : public CLApp {
  string source_manifest_;

 public:
  CLConnectedComponents(int argc, char** argv, string name)
      : CLApp(argc, argv, std::move(name)) {
    get_args_ += "Y:";
    AddHelpLine(
        'Y', "file",
        "write deterministic adaptive source manifest and exit");
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    if (opt == 'Y') {
      source_manifest_ = opt_arg;
      return;
    }
    CLApp::HandleArg(opt, opt_arg);
  }

  const string& source_manifest() const {
    return source_manifest_;
  }
};

#ifdef GRAPHBREW_COUNT_WORK
static int64_t g_cc_sampled_edges = 0;
static int64_t g_cc_final_edges = 0;
static int64_t g_cc_compress_steps = 0;
static int64_t g_cc_skipped_vertices = 0;
#endif

inline NodeID LoadComp(const pvector<NodeID>& comp, NodeID node) {
  return __atomic_load_n(&comp[node], __ATOMIC_RELAXED);
}

inline void StoreComp(
    pvector<NodeID>& comp, NodeID node, NodeID value) {
  __atomic_store_n(&comp[node], value, __ATOMIC_RELAXED);
}

// Place nodes u and v in same component of lower component ID
void Link(NodeID u, NodeID v, pvector<NodeID>& comp) {
  NodeID p1 = LoadComp(comp, u);
  NodeID p2 = LoadComp(comp, v);
  while (p1 != p2) {
    NodeID high = p1 > p2 ? p1 : p2;
    NodeID low = p1 + (p2 - high);
    NodeID p_high = LoadComp(comp, high);
    // Was already 'low' or succeeded in writing 'low'
    if ((p_high == low) ||
        (p_high == high && compare_and_swap(comp[high], high, low)))
      break;
    p1 = LoadComp(comp, LoadComp(comp, high));
    p2 = LoadComp(comp, low);
  }
}


// Reduce depth of tree for each component to 1 by crawling up parents
int64_t Compress(const Graph &g, pvector<NodeID>& comp) {
#ifdef GRAPHBREW_COUNT_WORK
  int64_t steps = 0;
  #pragma omp parallel for reduction(+ : steps) schedule(dynamic, 16384)
#else
  #pragma omp parallel for schedule(dynamic, 16384)
#endif
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    NodeID parent = LoadComp(comp, n);
    NodeID grandparent = LoadComp(comp, parent);
    while (parent != grandparent) {
      StoreComp(comp, n, grandparent);
#ifdef GRAPHBREW_COUNT_WORK
      steps++;
#endif
      parent = grandparent;
      grandparent = LoadComp(comp, parent);
    }
  }
#ifdef GRAPHBREW_COUNT_WORK
  return steps;
#else
  return 0;
#endif
}


NodeID SampleFrequentElement(const pvector<NodeID>& comp,
                             bool logging_enabled = false,
                             int64_t num_samples = 1024) {
  std::unordered_map<NodeID, int> sample_counts(32);
  using kvp_type = std::unordered_map<NodeID, int>::value_type;
  // Sample elements from 'comp'
  std::mt19937 gen;
  std::uniform_int_distribution<NodeID> distribution(0, comp.size() - 1);
  for (NodeID i = 0; i < num_samples; i++) {
    NodeID n = distribution(gen);
    sample_counts[comp[n]]++;
  }
  // Find most frequent element in samples (estimate of most frequent overall)
  auto most_frequent = std::max_element(
    sample_counts.begin(), sample_counts.end(),
    [](const kvp_type& a, const kvp_type& b) { return a.second < b.second; });
  float frac_of_graph = static_cast<float>(most_frequent->second) / num_samples;
  if (logging_enabled)
    std::cout
      << "Skipping largest intermediate component (ID: " << most_frequent->first
      << ", approx. " << static_cast<int>(frac_of_graph * 100)
      << "% of the graph)" << std::endl;
  return most_frequent->first;
}


pvector<NodeID> Afforest(const Graph &g, bool logging_enabled = false,
                         int32_t neighbor_rounds = 2) {
  using graphbrew::database::AppendBenchmarkIterationEntry;
  pvector<NodeID> comp(g.num_nodes());
#ifdef GRAPHBREW_COUNT_WORK
  int64_t sampled_edges = 0;
  int64_t final_edges = 0;
  int64_t compress_steps = 0;
  int64_t skipped_vertices = 0;
#endif

  // Initialize each node to a single-node self-pointing tree
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    comp[n] = n;

  // Process a sparse sampled subgraph first for approximating components.
  // Sample by processing a fixed number of neighbors for each node (see paper)
  for (int r = 0; r < neighbor_rounds; ++r) {
#ifdef GRAPHBREW_COUNT_WORK
  #pragma omp parallel for reduction(+ : sampled_edges) schedule(dynamic,16384)
#else
  #pragma omp parallel for schedule(dynamic,16384)
#endif
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      for (NodeID v : g.out_neigh(u, r)) {
#ifdef GRAPHBREW_COUNT_WORK
        sampled_edges++;
#endif
        // Link at most one time if neighbor available at offset r
        Link(u, v, comp);
        break;
      }
    }
#ifdef GRAPHBREW_COUNT_WORK
    compress_steps += Compress(g, comp);
#else
    Compress(g, comp);
#endif
    if (graphbrew::database::SelfRecordingEnabled())
      AppendBenchmarkIterationEntry({{"phase", "neighbor_round"}, {"round", r}});
  }

  // Sample 'comp' to find the most frequent element -- due to prior
  // compression, this value represents the largest intermediate component
  NodeID c = SampleFrequentElement(comp, logging_enabled);

  // Final 'link' phase over remaining edges (excluding the largest component)
  if (!g.directed()) {
#ifdef GRAPHBREW_COUNT_WORK
    #pragma omp parallel for reduction(+ : final_edges, skipped_vertices) schedule(dynamic, 16384)
#else
    #pragma omp parallel for schedule(dynamic, 16384)
#endif
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      // Skip processing nodes in the largest component
      if (LoadComp(comp, u) == c) {
#ifdef GRAPHBREW_COUNT_WORK
        skipped_vertices++;
#endif
        continue;
      }
      // Skip over part of neighborhood (determined by neighbor_rounds)
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
#ifdef GRAPHBREW_COUNT_WORK
        final_edges++;
#endif
        Link(u, v, comp);
      }
    }
  } else {
#ifdef GRAPHBREW_COUNT_WORK
    #pragma omp parallel for reduction(+ : final_edges, skipped_vertices) schedule(dynamic, 16384)
#else
    #pragma omp parallel for schedule(dynamic, 16384)
#endif
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      if (LoadComp(comp, u) == c) {
#ifdef GRAPHBREW_COUNT_WORK
        skipped_vertices++;
#endif
        continue;
      }
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
#ifdef GRAPHBREW_COUNT_WORK
        final_edges++;
#endif
        Link(u, v, comp);
      }
      // To support directed graphs, process reverse graph completely
      for (NodeID v : g.in_neigh(u)) {
#ifdef GRAPHBREW_COUNT_WORK
        final_edges++;
#endif
        Link(u, v, comp);
      }
    }
  }
  // Finally, 'compress' for final convergence
#ifdef GRAPHBREW_COUNT_WORK
  compress_steps += Compress(g, comp);
  g_cc_sampled_edges = sampled_edges;
  g_cc_final_edges = final_edges;
  g_cc_compress_steps = compress_steps;
  g_cc_skipped_vertices = skipped_vertices;
#else
  Compress(g, comp);
#endif
  if (graphbrew::database::SelfRecordingEnabled())
    AppendBenchmarkIterationEntry({{"phase", "final_compress"}, {"neighbor_rounds", neighbor_rounds}});
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
  CLConnectedComponents cli(
      argc, argv, "connected-components-afforest");
  if (!cli.ParseArgs())
    return -1;
  SetBenchmarkTypeHint(BENCH_CC);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  Builder b(cli);
  Graph g = b.MakeGraph();
  if (!cli.source_manifest().empty()) {
    auto comp = Afforest(g, false);
    if (!CCVerifier(g, comp))
      throw runtime_error(
          "Adaptive source policy component verification failed");
    const auto labeling_features =
        ComputeTier0SampledGraphFeatures(g);
    const size_t labeling_sample_size = std::min<size_t>(
        8192,
        std::max<size_t>(
            1024,
            static_cast<size_t>(
                std::sqrt(static_cast<double>(g.num_nodes())))));
    const nlohmann::json labeling_feature_json = {
        {"normalized_edge_span",
         labeling_features.avg_reuse_distance},
        {"window_neighbor_overlap",
         labeling_features.window_neighbor_overlap},
        {"sample_size", labeling_sample_size},
        {"sample_policy", "sqrt-clamped-1024-8192/v1"},
    };
    const string graph_path = cli.filename();
    string graph_name = (
        graph_path.empty()
        ? "synthetic-scale-" + to_string(cli.scale())
        : ExtractGraphNameFromPath(graph_path));
    const string natural_suffix = ".natural";
    if (
        graph_name.size() > natural_suffix.size()
        && graph_name.compare(
            graph_name.size() - natural_suffix.size(),
            natural_suffix.size(),
            natural_suffix) == 0
    ) {
      graph_name.resize(graph_name.size() - natural_suffix.size());
    }
    graphbrew::analysis::WriteAdaptiveSourceManifest(
        g,
        comp,
        graph_name,
        graph_path,
        cli.source_manifest(),
        true,
        "CCVerifier/v1",
        labeling_feature_json);
    cout << "Adaptive Source Manifest: "
         << cli.source_manifest() << "\n";
    return 0;
  }
  auto CCBound = [&cli](const Graph& gr){ return Afforest(gr, cli.logging_en()); };
  BenchmarkKernel(cli, g, CCBound, PrintCompStats, CCVerifier,
    "cc",
    [](const Graph &g, const pvector<NodeID> &comp) -> nlohmann::json {
      nlohmann::json ans;
      std::unordered_map<NodeID, NodeID> count;
      for (NodeID n = 0; n < g.num_nodes(); n++) count[comp[n]]++;
      ans["num_components"] = static_cast<int64_t>(count.size());
      // Find largest component
      NodeID largest = 0;
      for (auto& kv : count) if (kv.second > largest) largest = kv.second;
      ans["largest_component"] = largest;
#ifdef GRAPHBREW_COUNT_WORK
      ans["sampled_edges_examined"] = g_cc_sampled_edges;
      ans["final_edges_examined"] = g_cc_final_edges;
      ans["compress_steps"] = g_cc_compress_steps;
      ans["skipped_vertices"] = g_cc_skipped_vertices;
      PrintLabel(
          "CC Sampled Edges", std::to_string(g_cc_sampled_edges));
      PrintLabel(
          "CC Final Edges", std::to_string(g_cc_final_edges));
      PrintLabel(
          "CC Compress Steps", std::to_string(g_cc_compress_steps));
      PrintLabel(
          "CC Skipped Vertices",
          std::to_string(g_cc_skipped_vertices));
#endif
      return ans;
    });
  return 0;
}
