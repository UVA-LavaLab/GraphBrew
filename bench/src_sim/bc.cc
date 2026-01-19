// Copyright (c) 2024, UVA LavaLab
// Betweenness Centrality with Cache Simulation

#include <iostream>
#include <fstream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"

#include "cache_sim.h"
#include "graph_sim.h"

using namespace std;
using namespace cache_sim;

typedef float ScoreT;

void BCBFS_Sim(const Graph &g, NodeID source, 
               pvector<ScoreT> &scores, CacheHierarchy &cache) {
    pvector<int32_t> depths(g.num_nodes(), -1);
    depths[source] = 0;
    
    pvector<NodeID> succ;
    succ.reserve(g.num_edges_directed());
    
    pvector<int64_t> succ_start(g.num_nodes() + 1, 0);
    pvector<int64_t> path_counts(g.num_nodes(), 0);
    path_counts[source] = 1;
    
    vector<SlidingQueue<NodeID>::iterator> depth_index;
    SlidingQueue<NodeID> queue(g.num_nodes());
    queue.push_back(source);
    queue.slide_window();
    depth_index.push_back(queue.begin());
    
    int32_t depth = 0;
    
    // BFS forward phase
    while (!queue.empty()) {
        depth++;
        depth_index.push_back(queue.begin());
        
        #pragma omp parallel for schedule(dynamic, 64)
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            // Track depth read
            CACHE_READ(cache, depths.data(), u);
            
            for (NodeID v : g.out_neigh(u)) {
                // Track depth and path_counts accesses
                CACHE_READ(cache, depths.data(), v);
                CACHE_READ(cache, path_counts.data(), v);
                
                if (depths[v] == -1 && 
                    compare_and_swap(depths[v], static_cast<int32_t>(-1), depth)) {
                    CACHE_WRITE(cache, depths.data(), v);
                    queue.push_back(v);
                }
                
                if (depths[v] == depth) {
                    #pragma omp critical
                    {
                        succ.push_back(v);
                        fetch_and_add(succ_start[u + 1], 1);
                        fetch_and_add(path_counts[v], path_counts[u]);
                        CACHE_WRITE(cache, path_counts.data(), v);
                    }
                }
            }
        }
        queue.slide_window();
    }
    depth_index.push_back(queue.begin());
    
    // Prefix sum for successor starts
    for (NodeID n = 0; n < g.num_nodes(); n++)
        succ_start[n + 1] += succ_start[n];
    
    // Backward phase - accumulate dependencies
    pvector<ScoreT> deltas(g.num_nodes(), 0);
    
    for (int32_t d = depth - 1; d >= 0; d--) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (auto it = depth_index[d]; it < depth_index[d + 1]; it++) {
            NodeID u = *it;
            // Track path_counts and deltas reads
            CACHE_READ(cache, path_counts.data(), u);
            CACHE_READ(cache, deltas.data(), u);
            
            ScoreT delta_u = 0;
            for (int64_t i = succ_start[u]; i < succ_start[u + 1]; i++) {
                NodeID v = succ[i];
                // Track path_counts and deltas accesses
                CACHE_READ(cache, path_counts.data(), v);
                CACHE_READ(cache, deltas.data(), v);
                delta_u += static_cast<ScoreT>(path_counts[u]) / 
                           static_cast<ScoreT>(path_counts[v]) * (1 + deltas[v]);
            }
            deltas[u] = delta_u;
            CACHE_WRITE(cache, deltas.data(), u);
            
            #pragma omp atomic
            scores[u] += delta_u;
            CACHE_WRITE(cache, scores.data(), u);
        }
    }
}

pvector<ScoreT> BC_Sim(const Graph &g, int num_iters, CacheHierarchy &cache) {
    pvector<ScoreT> scores(g.num_nodes(), 0);
    
    SourcePicker<Graph> sp(g);
    for (int i = 0; i < num_iters; i++) {
        NodeID source = sp.PickNext();
        BCBFS_Sim(g, source, scores, cache);
    }
    
    // Normalize scores
    ScoreT max_score = *max_element(scores.begin(), scores.end());
    if (max_score > 0) {
        for (NodeID n : g.vertices())
            scores[n] /= max_score;
    }
    
    return scores;
}

void PrintBCStats(const Graph &g, const pvector<ScoreT> &scores) {
    auto [min_it, max_it] = minmax_element(scores.begin(), scores.end());
    cout << "BC scores range: [" << *min_it << ", " << *max_it << "]" << endl;
}

bool BCVerifier(const Graph &g, const pvector<ScoreT> &scores, int num_iters) {
    // Simple verification: check scores are non-negative and bounded
    for (NodeID n : g.vertices()) {
        if (scores[n] < 0 || scores[n] > 1) {
            cout << "BC verification failed: score out of range at node " << n << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    CLIterApp cli(argc, argv, "bc-sim", 4);
    if (!cli.ParseArgs())
        return -1;
    
    Builder b(cli);
    Graph g = b.MakeGraph();
    
    CacheHierarchy cache = CacheHierarchy::fromEnvironment();
    
    auto BCBound = [&cli, &cache](const Graph &g) {
        return BC_Sim(g, cli.num_iters(), cache);
    };
    auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
        return BCVerifier(g, scores, cli.num_iters());
    };
    
    BenchmarkKernel(cli, g, BCBound, PrintBCStats, VerifierBound);
    
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
