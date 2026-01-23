// Copyright (c) 2024, UVA LavaLab
// SSSP with Cache Simulation

#include <iostream>
#include <fstream>
#include <limits>
#include <queue>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"

#include "cache_sim.h"
#include "graph_sim.h"

using namespace std;
using namespace cache_sim;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const size_t kMaxBin = numeric_limits<size_t>::max() / 2;

pvector<WeightT> DeltaStep_Sim(const WGraph &g, NodeID source, 
                                WeightT delta, CacheHierarchy &cache) {
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    dist[source] = 0;
    
    pvector<NodeID> frontier(g.num_edges_directed());
    size_t shared_indexes[2] = {0, kMaxBin};
    size_t frontier_tails[2] = {1, 0};
    frontier[0] = source;
    
    #pragma omp parallel
    {
        vector<vector<NodeID>> local_bins(0);
        size_t iter = 0;
        while (shared_indexes[iter&1] != kMaxBin) {
            size_t &curr_bin_index = shared_indexes[iter&1];
            size_t &next_bin_index = shared_indexes[(iter+1)&1];
            size_t &curr_frontier_tail = frontier_tails[iter&1];
            size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
            
            #pragma omp for nowait schedule(dynamic, 64)
            for (size_t i = 0; i < curr_frontier_tail; i++) {
                NodeID u = frontier[i];
                // Track: read dist[u]
                CACHE_READ(cache, dist.data(), u);
                if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index)) {
                    for (WNode wn : g.out_neigh(u)) {
                        WeightT old_dist = dist[wn.v];
                        WeightT new_dist = dist[u] + wn.w;
                        // Track: read/write dist[wn.v]
                        CACHE_READ(cache, dist.data(), wn.v);
                        if (new_dist < old_dist) {
                            CACHE_WRITE(cache, dist.data(), wn.v);
                            bool changed_dist = true;
                            while (!compare_and_swap(dist[wn.v], old_dist, new_dist)) {
                                old_dist = dist[wn.v];
                                if (old_dist <= new_dist) {
                                    changed_dist = false;
                                    break;
                                }
                            }
                            if (changed_dist) {
                                size_t dest_bin = new_dist / delta;
                                if (dest_bin >= local_bins.size())
                                    local_bins.resize(dest_bin + 1);
                                local_bins[dest_bin].push_back(wn.v);
                            }
                        }
                    }
                }
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
                curr_bin_index = kMaxBin;
                curr_frontier_tail = 0;
            }
            
            if (next_bin_index < local_bins.size()) {
                size_t copy_start = fetch_and_add(next_frontier_tail, 
                                                  local_bins[next_bin_index].size());
                copy(local_bins[next_bin_index].begin(),
                     local_bins[next_bin_index].end(), 
                     frontier.data() + copy_start);
                local_bins[next_bin_index].resize(0);
            }
            
            iter++;
            #pragma omp barrier
        }
    }
    
    return dist;
}

void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
    auto NotInf = [](WeightT d) { return d != kDistInf; };
    int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
    cout << "SSSP Tree reaches " << static_cast<long long>(num_reached) << " nodes" << endl;
}

bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    dist[source] = 0;
    
    typedef pair<WeightT, NodeID> WN;
    priority_queue<WN, vector<WN>, greater<WN>> mq;
    mq.push(make_pair(static_cast<WeightT>(0), source));
    
    while (!mq.empty()) {
        WeightT td = mq.top().first;
        NodeID u = mq.top().second;
        mq.pop();
        if (td == dist[u]) {
            for (WNode wn : g.out_neigh(u)) {
                if (td + wn.w < dist[wn.v]) {
                    dist[wn.v] = td + wn.w;
                    mq.push(make_pair(dist[wn.v], wn.v));
                }
            }
        }
    }
    
    for (NodeID n : g.vertices())
        if (dist[n] != dist_to_test[n])
            return false;
    return true;
}

int main(int argc, char *argv[]) {
    CLDelta<WeightT> cli(argc, argv, "sssp-sim");
    if (!cli.ParseArgs())
        return -1;
    
    WeightedBuilder b(cli);
    WGraph g = b.MakeGraph();
    
    CacheHierarchy cache = CacheHierarchy::fromEnvironment();
    
    SourcePicker<WGraph> sp(g, cli.start_vertex(), cli.num_trials());
    auto SSSPBound = [&sp, &cli, &cache](const WGraph &g) {
        return DeltaStep_Sim(g, sp.PickNext(), cli.delta(), cache);
    };
    SourcePicker<WGraph> vsp(g, cli.start_vertex(), cli.num_trials());
    auto VerifierBound = [&vsp](const WGraph &g, const pvector<WeightT> &dist) {
        return SSSPVerifier(g, vsp.PickNext(), dist);
    };
    
    BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
    
    cout << endl;
    cache.printStats();
    
    const char* json_file = getenv("CACHE_OUTPUT_JSON");
    if (json_file) {
        ofstream ofs(json_file);
        if (ofs.is_open()) {
            ofs << cache.toJSON() << endl;
            ofs.close();
        }
    }
    
    return 0;
}
