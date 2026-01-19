// Copyright (c) 2024, UVA LavaLab
// PageRank with Cache Simulation
// Tracks all memory accesses to graph data structures

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

// Cache simulation headers
#include "cache_sim.h"
#include "graph_sim.h"

using namespace std;
using namespace cache_sim;

typedef float ScoreT;
const float kDamp = 0.85;

// PageRank with cache simulation - tracks all memory accesses
pvector<ScoreT> PageRankPullGS_Sim(const Graph &g, CacheHierarchy &cache,
                                    int max_iters, double epsilon = 0,
                                    bool logging_enabled = false) {
    const ScoreT init_score = 1.0f / g.num_nodes();
    const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
    pvector<ScoreT> scores(g.num_nodes(), init_score);
    pvector<ScoreT> outgoing_contrib(g.num_nodes());
    
    // Get raw pointers for cache tracking
    ScoreT* scores_ptr = scores.data();
    ScoreT* contrib_ptr = outgoing_contrib.data();
    
    // Initialize outgoing contributions
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++) {
        // Track: read degree (via neighbor iteration bounds), write contrib[n]
        CACHE_READ(cache, scores_ptr, n);
        CACHE_WRITE(cache, contrib_ptr, n);
        outgoing_contrib[n] = init_score / g.out_degree(n);
    }
    
    for (int iter = 0; iter < max_iters; iter++) {
        double error = 0;
        
        #pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
        for (NodeID u = 0; u < g.num_nodes(); u++) {
            ScoreT incoming_total = 0;
            
            // Iterate over incoming neighbors
            for (NodeID v : g.in_neigh(u)) {
                // Track: read neighbor ID, read contrib[v]
                CACHE_READ(cache, contrib_ptr, v);
                incoming_total += outgoing_contrib[v];
            }
            
            // Track: read old score, write new score
            CACHE_READ(cache, scores_ptr, u);
            ScoreT old_score = scores[u];
            ScoreT new_score = base_score + kDamp * incoming_total;
            CACHE_WRITE(cache, scores_ptr, u);
            scores[u] = new_score;
            error += fabs(new_score - old_score);
            
            // Update contribution for next iteration
            CACHE_WRITE(cache, contrib_ptr, u);
            outgoing_contrib[u] = new_score / g.out_degree(u);
        }
        
        if (logging_enabled)
            cout << "Iteration " << iter << ": error = " << error << endl;
        
        if (error < epsilon)
            break;
    }
    
    return scores;
}

void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
    vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
    for (NodeID n = 0; n < g.num_nodes(); n++) {
        score_pairs[n] = make_pair(n, scores[n]);
    }
    int k = 5;
    partial_sort(score_pairs.begin(), score_pairs.begin() + k, score_pairs.end(),
                [](auto a, auto b) { return a.second > b.second; });
    for (int i = 0; i < k; i++) {
        cout << score_pairs[i].first << ": " << score_pairs[i].second << endl;
    }
}

bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores, double target_error) {
    const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
    pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
    double error = 0;
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
        for (NodeID v : g.out_neigh(u))
            incomming_sums[v] += outgoing_contrib;
    }
    for (NodeID n = 0; n < g.num_nodes(); n++) {
        error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
        incomming_sums[n] = 0;
    }
    cout << "Total Error: " << error << endl;
    return error < target_error;
}

int main(int argc, char *argv[]) {
    CLPageRank cli(argc, argv, "pagerank-sim", 1e-4, 20);
    if (!cli.ParseArgs())
        return -1;
    
    Builder b(cli);
    Graph g = b.MakeGraph();
    
    // Initialize cache simulator from environment
    CacheHierarchy cache = CacheHierarchy::fromEnvironment();
    
    // Run PageRank with cache simulation
    auto PRBound = [&cli, &cache](const Graph &g) {
        return PageRankPullGS_Sim(g, cache, cli.max_iters(), cli.tolerance());
    };
    auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
        return PRVerifier(g, scores, cli.tolerance());
    };
    
    BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
    
    // Print cache statistics
    cout << endl;
    cache.printStats();
    
    // Export to JSON if requested
    const char* json_file = getenv("CACHE_OUTPUT_JSON");
    if (json_file) {
        ofstream ofs(json_file);
        if (ofs.is_open()) {
            ofs << cache.toJSON() << endl;
            ofs.close();
            cout << "Cache stats exported to: " << json_file << endl;
        }
    }
    
    return 0;
}
