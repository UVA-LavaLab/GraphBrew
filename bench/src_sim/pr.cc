// Copyright (c) 2024, UVA LavaLab
// PageRank with Cache Simulation
// Tracks all memory accesses to graph data structures
// Supports both single-core and multi-core cache simulation

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
#include "cache_sim/cache_sim.h"
#include "cache_sim/graph_sim.h"

// P-OPT rereference matrix builder
#include "graphbrew/partition/cagra/popt.h"

using namespace std;
using namespace cache_sim;

typedef float ScoreT;
const float kDamp = 0.85;

// PageRank with cache simulation - template version works with both cache types
template<typename CacheType>
pvector<ScoreT> PageRankPullGS_Sim(const Graph &g, CacheType &cache,
                                    int max_iters, double epsilon = 0,
                                    bool logging_enabled = false) {
    const ScoreT init_score = 1.0f / g.num_nodes();
    const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
    pvector<ScoreT> scores(g.num_nodes(), init_score);
    pvector<ScoreT> outgoing_contrib(g.num_nodes());
    
    // Get raw pointers for cache tracking
    ScoreT* scores_ptr = scores.data();
    ScoreT* contrib_ptr = outgoing_contrib.data();

    // --- Graph-aware cache context (for GRASP/P-OPT/ECG policies) ---
    // Registers property arrays so cache policies know hot/warm/cold regions.
    // Auto-computes hot fraction from degree distribution (self-tuning).
    GraphCacheContext graph_ctx;

    // Build degree array for topology init
    pvector<uint32_t> degrees(g.num_nodes());
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
        degrees[n] = static_cast<uint32_t>(g.out_degree(n));
    graph_ctx.initTopology(degrees.data(), g.num_nodes(),
                           g.num_edges_directed(), g.directed());

    // Upstream GRASP protects the source contribution region (propertyA), not
    // the next-score destination array. Both remain property data for P-OPT/ECG.
    size_t llc_size = 8 * 1024 * 1024;  // Default 8MB, overridden by env
    llc_size = GetEnvSizeBytes("CACHE_L3_SIZE", llc_size);
    graph_ctx.registerPropertyArray(scores_ptr, g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true);
    graph_ctx.registerPropertyArray(contrib_ptr, g.num_nodes(), sizeof(ScoreT), llc_size, -1.0, true);
    cache.initGraphContext(&graph_ctx);

    // Build P-OPT rereference matrix (for POPT and ECG policies)
    // Uses graph structure to predict future cache line accesses.
    // numVtxPerLine = cache_line_size / sizeof(ScoreT) = 64/4 = 16
    static pvector<uint8_t> popt_matrix;  // Must outlive graph_ctx
    {
        const char* policy_env = getenv("CACHE_POLICY");
        std::string policy_str = policy_env ? policy_env : "";
        const char* pfx_env = getenv("ECG_PREFETCH_MODE");
        bool popt_prefetch = pfx_env && (atoi(pfx_env) == 2 || atoi(pfx_env) == 4);
        if (policy_str == "POPT" || policy_str == "ECG" || popt_prefetch) {
            constexpr int numVtxPerLine = 64 / sizeof(ScoreT);
            constexpr int numEpochs = 256;
            makeOffsetMatrix(g, popt_matrix, numVtxPerLine, numEpochs);
            int numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
            graph_ctx.initRereference(popt_matrix.data(), numCacheLines,
                                      numEpochs, g.num_nodes(), 64);
        }
    }

    // Compute per-vertex ECG mask array (supports 8/16/32-bit widths)
    graph_ctx.initMaskConfig();
    auto vertex_masks = graph_ctx.computeVertexMasks(g);  // uint32_t per vertex
    graph_ctx.initMaskArray32(vertex_masks.data(), vertex_masks.size());
    graph_ctx.printSummary();
    int pfx_lookahead = GraphSimEnvIntClamped("ECG_PREFETCH_LOOKAHEAD", 0, 0, 64);
    int pfx_top_k = GraphSimEnvIntClamped("ECG_PREFETCH_TOP_K", 1, 1, 64);
    if (pfx_lookahead > 0 && graph_ctx.mask_config.prefetch_mode > 0) {
        cout << "PR PFX lookahead: window=" << pfx_lookahead
             << " mode=" << int(graph_ctx.mask_config.prefetch_mode)
             << " top_k=" << pfx_top_k << endl;
    }
    
    // Initialize outgoing contributions
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++) {
        // Track: read degree (via neighbor iteration bounds), write contrib[n]
        SIM_CACHE_READ(cache, scores_ptr, n);
        SIM_CACHE_WRITE(cache, contrib_ptr, n);
        outgoing_contrib[n] = init_score / g.out_degree(n);
    }
    
    for (int iter = 0; iter < max_iters; iter++) {
        double error = 0;
        
        #pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
        for (NodeID u = 0; u < g.num_nodes(); u++) {
            ScoreT incoming_total = 0;
            
            // P-OPT: update current destination vertex for rereference lookup
            SIM_SET_VERTEX(cache, u);
            
            // Iterate over incoming neighbors with CSR edge tracking
            auto in_neigh = g.in_neigh(u);
            for (auto it = in_neigh.begin(); it != in_neigh.end(); ++it) {
                // Track CSR edge list read (reading neighbor ID from edge array)
                SIM_CACHE_READ_EDGE(cache, it);
                NodeID v = *it;
                // ECG: read contrib[v] with mask. With lookahead enabled, issue
                // the prefetch from upcoming incoming-neighbor IDs before their
                // demand read; otherwise use the per-vertex PFX target directly.
                //
                // Prefetch modes:
                //   1 = degree-ranked (ECG_PFX): pick most-popular among next K
                //   2 = POPT-ranked   (ECG_PFX): pick lowest-POPT-rank among next K
                //   3 = sequential    (DROPLET): prefetch ALL next K (no selection)
                //
                // Mode 3 = DROPLET-in-cache_sim. DROPLET's Sniper impl monitors
                // edge stream and stride-prefetches destination properties.
                // Cache_sim has explicit edge access markers so we can deliver
                // the same semantic (prefetch next-K in-neighbors' contrib[])
                // without the runtime stride detection. Faithful comparator
                // for the ECG_PFX claim.
                if (pfx_lookahead > 0 && graph_ctx.mask_config.prefetch_mode > 0) {
                    if (graph_ctx.mask_config.prefetch_mode == 3) {
                        // DROPLET-style: prefetch every next-K in-neighbor
                        // sequentially. No target selection — just sweep the
                        // upcoming edge stream's destinations.
                        auto jt = it;
                        for (int step = 0; step < pfx_lookahead; step++) {
                            ++jt;
                            if (jt == in_neigh.end()) break;
                            NodeID candidate = *jt;
                            if (candidate < 0) continue;
                            SIM_CACHE_PREFETCH_VERTEX(cache, contrib_ptr,
                                static_cast<uint32_t>(candidate), graph_ctx);
                        }
                        SIM_CACHE_READ_MASKED(cache, contrib_ptr, v, graph_ctx, vertex_masks[v]);
                    } else {
                        // Mode 1 = degree-ranked, Mode 2 = POPT-ranked.
                        // Top-K extension (sprint 6f-3): instead of issuing
                        // just the single best target, collect all candidates
                        // from the lookahead window and issue prefetches for
                        // the top-K ranked. K=1 reproduces the original
                        // single-best behavior; K>1 trades selection quality
                        // for higher prefetch volume (closer to DROPLET in
                        // bandwidth, but still POPT-quality filtered).
                        struct Cand { uint32_t v; uint16_t key; };
                        Cand cands[64];  // max lookahead is 64
                        int n_cand = 0;
                        auto jt = it;
                        for (int step = 0; step < pfx_lookahead; step++) {
                            ++jt;
                            if (jt == in_neigh.end()) break;
                            NodeID candidate = *jt;
                            if (candidate < 0) continue;
                            uint16_t key;
                            if (graph_ctx.mask_config.prefetch_mode == 1) {
                                // Larger out_degree = "more popular" — invert
                                // for sorting (smaller key = higher priority).
                                uint64_t od = g.out_degree(candidate);
                                key = od > 65535 ? 0 : static_cast<uint16_t>(65535 - od);
                            } else {
                                // Lower POPT rank = sooner-rereferenced = higher priority.
                                key = graph_ctx.mask_config.decodePOPT(vertex_masks[candidate]);
                            }
                            cands[n_cand++] = {static_cast<uint32_t>(candidate), key};
                        }
                        if (n_cand == 0) {
                            graph_ctx.recordPrefetchNoTarget();
                        } else if (pfx_top_k <= 1) {
                            // Fast path: single best target — match historical mode-2 behavior bit-for-bit.
                            int best = 0;
                            for (int i = 1; i < n_cand; i++)
                                if (cands[i].key < cands[best].key) best = i;
                            SIM_CACHE_PREFETCH_VERTEX(cache, contrib_ptr, cands[best].v, graph_ctx);
                        } else {
                            // Top-K path: partial sort by key (ascending), issue first K.
                            int k_eff = pfx_top_k < n_cand ? pfx_top_k : n_cand;
                            for (int i = 0; i < k_eff; i++) {
                                int best = i;
                                for (int j = i + 1; j < n_cand; j++)
                                    if (cands[j].key < cands[best].key) best = j;
                                if (best != i) std::swap(cands[i], cands[best]);
                                SIM_CACHE_PREFETCH_VERTEX(cache, contrib_ptr, cands[i].v, graph_ctx);
                            }
                        }
                        SIM_CACHE_READ_MASKED(cache, contrib_ptr, v, graph_ctx, vertex_masks[v]);
                    }
                } else {
                    SIM_CACHE_READ_MASKED_PREFETCH(cache, contrib_ptr, v, graph_ctx, vertex_masks[v]);
                }
                incoming_total += outgoing_contrib[v];
            }
            
            // Track: read old score, write new score
            SIM_CACHE_READ(cache, scores_ptr, u);
            ScoreT old_score = scores[u];
            ScoreT new_score = base_score + kDamp * incoming_total;
            SIM_CACHE_WRITE(cache, scores_ptr, u);
            scores[u] = new_score;
            error += fabs(new_score - old_score);
            
            // Update contribution for next iteration
            SIM_CACHE_WRITE(cache, contrib_ptr, u);
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
    
    // Check modes: multi-core vs single-core, ultrafast vs fast vs accurate
    bool multicore = IsMultiCoreMode();
    bool sampled = IsSampledMode();
    bool ultrafast = IsUltraFastMode();
    bool fast = IsFastMode();
    
    if (multicore) {
        // Multi-core cache simulation (private L1/L2, shared L3)
        MultiCoreCacheHierarchy cache = MultiCoreCacheHierarchy::fromEnvironment();
        
        auto PRBound = [&cli, &cache](const Graph &g) {
            return PageRankPullGS_Sim(g, cache, cli.max_iters(), cli.tolerance());
        };
        auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
            return PRVerifier(g, scores, cli.tolerance());
        };
        
        BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
        
        cout << endl;
        cache.printStats();
        
        const char* json_file = getenv("CACHE_OUTPUT_JSON");
        if (json_file) {
            ofstream ofs(json_file);
            if (ofs.is_open()) {
                ofs << cache.toJSON() << endl;
                ofs.close();
                cout << "Cache stats exported to: " << json_file << endl;
            }
        }
    } else if (sampled) {
        // SAMPLED cache simulation (~5-20x faster with statistical sampling)
        SampledCacheHierarchy cache = SampledCacheHierarchy::fromEnvironment();
        
        auto PRBound = [&cli, &cache](const Graph &g) {
            return PageRankPullGS_Sim(g, cache, cli.max_iters(), cli.tolerance());
        };
        auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
            return PRVerifier(g, scores, cli.tolerance());
        };
        
        BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
        
        cout << endl;
        cache.printStats();
        
        const char* json_file = getenv("CACHE_OUTPUT_JSON");
        if (json_file) {
            ofstream ofs(json_file);
            if (ofs.is_open()) {
                ofs << cache.toJSON() << endl;
                ofs.close();
                cout << "Cache stats exported to: " << json_file << endl;
            }
        }
    } else if (ultrafast) {
        // ULTRA-FAST cache simulation (packed structures, best performance)
        UltraFastCacheHierarchy cache = UltraFastCacheHierarchy::fromEnvironment();
        
        auto PRBound = [&cli, &cache](const Graph &g) {
            return PageRankPullGS_Sim(g, cache, cli.max_iters(), cli.tolerance());
        };
        auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
            return PRVerifier(g, scores, cli.tolerance());
        };
        
        BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
        
        cout << endl;
        cache.printStats();
        
        const char* json_file = getenv("CACHE_OUTPUT_JSON");
        if (json_file) {
            ofstream ofs(json_file);
            if (ofs.is_open()) {
                ofs << cache.toJSON() << endl;
                ofs.close();
                cout << "Cache stats exported to: " << json_file << endl;
            }
        }
    } else if (fast) {
        // FAST single-core cache simulation (no locks)
        FastCacheHierarchy cache = FastCacheHierarchy::fromEnvironment();
        
        auto PRBound = [&cli, &cache](const Graph &g) {
            return PageRankPullGS_Sim(g, cache, cli.max_iters(), cli.tolerance());
        };
        auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
            return PRVerifier(g, scores, cli.tolerance());
        };
        
        BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
        
        cout << endl;
        cache.printStats();
        
        const char* json_file = getenv("CACHE_OUTPUT_JSON");
        if (json_file) {
            ofstream ofs(json_file);
            if (ofs.is_open()) {
                ofs << cache.toJSON() << endl;
                ofs.close();
                cout << "Cache stats exported to: " << json_file << endl;
            }
        }
    } else {
        // Original single-core cache simulation (with locks, slower but full LRU)
        CacheHierarchy cache = CacheHierarchy::fromEnvironment();
        
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
    }
    
    return 0;
}
