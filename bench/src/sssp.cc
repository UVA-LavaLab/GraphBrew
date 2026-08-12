// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cinttypes>
#include <cstdlib>
#include <cstdint> // For int64_t
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"

/*
GAP Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer, Yunming Zhang

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph. This implementation
incorporates a new bucket fusion optimization [2] that significantly reduces
the number of iterations (& barriers) needed.

The bins of width delta are actually all thread-local and of type std::vector,
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies its selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and, it now appears in a lower bin. We find ignoring vertices if
their distance is less than the min distance for the current bin removes
enough redundant work to be faster than removing the vertex from older bins.

The bucket fusion optimization [2] executes the next thread-local bin in
the same iteration if the vertices in the next thread-local bin have the
same priority as those in the current shared bin. This optimization greatly
reduces the number of iterations needed without violating the priority-based
execution order, leading to significant speedup on large diameter road networks.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.

[2] Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala,
    Shoaib Kamil, Saman Amarasinghe, and Julian Shun. "Optimizing ordered graph
    algorithms with GraphIt." The 18th International Symposium on Code
Generation and Optimization (CGO), pages 158-170, 2020.
*/

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const size_t kMaxBin = numeric_limits<size_t>::max() / 2;
const size_t kBinSizeThreshold = 1000;
#ifdef GRAPHBREW_COUNT_WORK
static uint64_t g_sssp_edges_examined = 0;
static uint64_t g_sssp_relax_successes = 0;
static uint64_t g_sssp_frontier_entries = 0;
static uint64_t g_sssp_bucket_iterations = 0;
#endif

uint64_t Mix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

string FormatFingerprint(uint64_t xor_hash, uint64_t sum_hash) {
  ostringstream out;
  out << hex << setfill('0')
      << setw(16) << xor_hash
      << setw(16) << sum_hash;
  return out.str();
}

string WeightChecksum(const WGraph &g) {
  unsigned long long xor_hash = 0;
  unsigned long long sum_hash = 0;
#pragma omp parallel for reduction(^ : xor_hash) reduction(+ : sum_hash)
  for (NodeID source = 0; source < g.num_nodes(); ++source) {
    uint64_t original_source =
        static_cast<uint32_t>(g.get_org_id(source));
    for (WNode edge : g.out_neigh(source)) {
      uint64_t original_destination =
          static_cast<uint32_t>(g.get_org_id(edge.v));
      uint64_t edge_hash = Mix64(
          Mix64(original_source) ^
          (Mix64(original_destination) << 1) ^
          (Mix64(static_cast<uint32_t>(edge.w)) << 7));
      xor_hash ^= edge_hash;
      sum_hash += edge_hash;
    }
  }
  return FormatFingerprint(xor_hash, sum_hash);
}

string DistanceFingerprint(
    const WGraph &g, const pvector<WeightT> &dist) {
  unsigned long long xor_hash = 0;
  unsigned long long sum_hash = 0;
#pragma omp parallel for reduction(^ : xor_hash) reduction(+ : sum_hash)
  for (NodeID internal = 0; internal < g.num_nodes(); ++internal) {
    uint64_t original = static_cast<uint32_t>(g.get_org_id(internal));
    uint64_t distance = static_cast<uint32_t>(dist[internal]);
    uint64_t entry_hash =
        Mix64(Mix64(original) ^ (Mix64(distance) << 1));
    xor_hash ^= entry_hash;
    sum_hash += entry_hash;
  }
  return FormatFingerprint(xor_hash, sum_hash);
}

inline void RelaxEdges(const WGraph &g, NodeID u, WeightT delta,
                       pvector<WeightT> &dist,
                       vector<vector<NodeID>> &local_bins
#ifdef GRAPHBREW_COUNT_WORK
                       ,
                       uint64_t &edges_examined,
                       uint64_t &relax_successes
#endif
                       ) {
  for (WNode wn : g.out_neigh(u)) {
#ifdef GRAPHBREW_COUNT_WORK
    edges_examined++;
#endif
    WeightT old_dist = dist[wn.v];
    WeightT new_dist = dist[u] + wn.w;
    while (new_dist < old_dist) {
      if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
        size_t dest_bin = new_dist / delta;
        if (dest_bin >= local_bins.size())
          local_bins.resize(dest_bin + 1);
        local_bins[dest_bin].push_back(wn.v);
#ifdef GRAPHBREW_COUNT_WORK
        relax_successes++;
#endif
        break;
      }
      old_dist = dist[wn.v]; // swap failed, recheck dist update & retry
    }
  }
}

pvector<WeightT> DeltaStep(const WGraph &g, NodeID source, WeightT delta,
                           bool logging_enabled = false) {
  const bool record_iterations =
      graphbrew::database::SelfRecordingEnabled();
  const bool measure_steps = logging_enabled || record_iterations;
  Timer t;
  pvector<WeightT> dist(g.num_nodes(), kDistInf);
  dist[source] = 0;
  pvector<NodeID> frontier(g.num_edges_directed());
#ifdef GRAPHBREW_COUNT_WORK
  const int max_threads = omp_get_max_threads();
  vector<uint64_t> thread_edges(max_threads, 0);
  vector<uint64_t> thread_relaxes(max_threads, 0);
  vector<uint64_t> thread_frontier_entries(max_threads, 0);
  uint64_t bucket_iterations = 0;
#endif
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kMaxBin};
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;
  if (measure_steps) t.Start();
#pragma omp parallel
  {
#ifdef GRAPHBREW_COUNT_WORK
    const int tid = omp_get_thread_num();
    if (tid < 0 || tid >= max_threads)
      std::abort();
    uint64_t edges_examined = 0;
    uint64_t relax_successes = 0;
    uint64_t frontier_entries = 0;
#endif
    vector<vector<NodeID>> local_bins(0);
    size_t iter = 0;
    while (shared_indexes[iter & 1] != kMaxBin) {
      size_t &curr_bin_index = shared_indexes[iter & 1];
      size_t &next_bin_index = shared_indexes[(iter + 1) & 1];
      size_t &curr_frontier_tail = frontier_tails[iter & 1];
      size_t &next_frontier_tail = frontier_tails[(iter + 1) & 1];
#pragma omp for nowait schedule(dynamic, 64)
      for (size_t i = 0; i < curr_frontier_tail; i++) {
        NodeID u = frontier[i];
        if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index))
          RelaxEdges(
              g, u, delta, dist, local_bins
#ifdef GRAPHBREW_COUNT_WORK
              , edges_examined, relax_successes
#endif
              );
      }
      while (curr_bin_index < local_bins.size() &&
             !local_bins[curr_bin_index].empty() &&
             local_bins[curr_bin_index].size() < kBinSizeThreshold) {
        vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
        local_bins[curr_bin_index].resize(0);
        for (NodeID u : curr_bin_copy)
          RelaxEdges(
              g, u, delta, dist, local_bins
#ifdef GRAPHBREW_COUNT_WORK
              , edges_examined, relax_successes
#endif
              );
      }
      for (size_t i = curr_bin_index; i < local_bins.size(); i++) {
        if (!local_bins[i].empty()) {
#pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
#pragma omp barrier
#pragma omp single nowait
      {
        if (measure_steps) t.Stop();
        if (logging_enabled)
          PrintStep(curr_bin_index, t.Millisecs(), curr_frontier_tail);
        if (record_iterations)
          graphbrew::database::AppendBenchmarkIterationEntry({
              {"iter", static_cast<int64_t>(iter)},
              {"bin_index", static_cast<int64_t>(curr_bin_index)},
              {"time_ms", t.Millisecs()},
              {"frontier_size", static_cast<int64_t>(curr_frontier_tail)}});
        if (measure_steps) t.Start();
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
#ifdef GRAPHBREW_COUNT_WORK
        frontier_entries += local_bins[next_bin_index].size();
#endif
        local_bins[next_bin_index].resize(0);
      }
      iter++;
#pragma omp barrier
    }
#pragma omp single
    {
#ifdef GRAPHBREW_COUNT_WORK
      bucket_iterations = iter;
#endif
      if (logging_enabled)
        cout << "took " << iter << " iterations" << endl;
    }
#ifdef GRAPHBREW_COUNT_WORK
    thread_edges[tid] = edges_examined;
    thread_relaxes[tid] = relax_successes;
    thread_frontier_entries[tid] = frontier_entries;
#endif
  }
#ifdef GRAPHBREW_COUNT_WORK
  g_sssp_edges_examined =
      accumulate(thread_edges.begin(), thread_edges.end(), uint64_t(0));
  g_sssp_relax_successes =
      accumulate(thread_relaxes.begin(), thread_relaxes.end(), uint64_t(0));
  g_sssp_frontier_entries =
      1 + accumulate(
          thread_frontier_entries.begin(),
          thread_frontier_entries.end(), uint64_t(0));
  g_sssp_bucket_iterations = bucket_iterations;
#endif
  return dist;
}

void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << static_cast<long long>(num_reached)
       << " nodes" << endl;
}

// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}

int main(int argc, char *argv[]) {
  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;
  SetBenchmarkTypeHint(BENCH_SSSP);
  graphbrew::database::InitSelfRecording(cli.db_dir());
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  const string weight_checksum = WeightChecksum(g);
  PrintLabel("Weight Scheme", cli.weight_scheme());
  PrintLabel("Weight Checksum", weight_checksum);
  PrintLabel("Delta", std::to_string(cli.delta()));
  // Create SourcePicker with pre-generated consistent sources based on num_trials
  // This ensures all orderings use the same ORIGINAL vertex IDs as sources
  SourcePicker<WGraph> sp(
      g, cli.start_vertices(), cli.num_trials(), cli.source_repeats());
  auto SSSPBound = [&sp, &cli](const WGraph &g) {
    return DeltaStep(g, sp.PickNext(), cli.delta(), cli.logging_en());
  };
  std::unique_ptr<SourcePicker<WGraph>> vsp;
  if (cli.do_verify()) {
    vsp = std::make_unique<SourcePicker<WGraph>>(
        g, cli.start_vertices(), cli.num_trials(),
        cli.source_repeats());
  }
  auto VerifierBound = [&vsp](const WGraph &g, const pvector<WeightT> &dist) {
    return SSSPVerifier(g, vsp->PickNext(), dist);
  };
  BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound,
    "sssp",
    [&cli, &sp, &weight_checksum](
        const WGraph &g, const pvector<WeightT> &dist) -> nlohmann::json {
      nlohmann::json ans;
      int64_t reachable = 0;
      for (NodeID n = 0; n < g.num_nodes(); n++) {
        if (dist[n] != kDistInf) reachable++;
      }
      const string distance_fingerprint = DistanceFingerprint(g, dist);
      PrintLabel(
          "Source Original", std::to_string(sp.last_original_source()));
      PrintLabel(
          "Source Internal", std::to_string(sp.last_internal_source()));
      PrintLabel(
          "Source Out Degree",
          std::to_string(sp.last_source_out_degree()));
      PrintLabel("Distance Fingerprint", distance_fingerprint);
#ifdef GRAPHBREW_COUNT_WORK
      PrintLabel(
          "SSSP Edges Examined",
          std::to_string(g_sssp_edges_examined));
      PrintLabel(
          "SSSP Relax Successes",
          std::to_string(g_sssp_relax_successes));
      PrintLabel(
          "SSSP Frontier Entries",
          std::to_string(g_sssp_frontier_entries));
      PrintLabel(
          "SSSP Bucket Iterations",
          std::to_string(g_sssp_bucket_iterations));
#endif
      ans["reachable_nodes"] = reachable;
      ans["weight_scheme"] = cli.weight_scheme();
      ans["weight_checksum"] = weight_checksum;
      ans["delta"] = cli.delta();
      ans["source_original"] = sp.last_original_source();
      ans["source_internal"] = sp.last_internal_source();
      ans["source_out_degree"] = sp.last_source_out_degree();
      ans["distance_fingerprint"] = distance_fingerprint;
#ifdef GRAPHBREW_COUNT_WORK
      ans["edges_examined"] = g_sssp_edges_examined;
      ans["relax_successes"] = g_sssp_relax_successes;
      ans["frontier_entries"] = g_sssp_frontier_entries;
      ans["bucket_iterations"] = g_sssp_bucket_iterations;
#endif
      return ans;
    });
  return 0;
}
