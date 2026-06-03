// Copyright (c) 2024, UVA LavaLab
// SSSP with Cache Simulation

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"

#include "cache_sim/cache_sim.h"
#include "cache_sim/graph_sim.h"

#include "graphbrew/partition/cagra/popt.h"

using namespace std;
using namespace cache_sim;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const size_t kMaxBin = numeric_limits<size_t>::max() / 2;
const size_t kBinSizeThreshold = 1000;

template<typename CacheType>
inline void RelaxEdges_Sim(const WGraph &g, NodeID u, WeightT delta,
                           pvector<WeightT> &dist,
                           vector<vector<NodeID>> &local_bins,
                           CacheType &cache, GraphCacheContext &graph_ctx,
                           const vector<uint32_t> &vertex_masks,
                           int pfx_lookahead) {
    auto out_neigh = g.out_neigh(u);
    for (auto it = out_neigh.begin(); it != out_neigh.end(); ++it) {
        WNode wn = *it;
        if (pfx_lookahead > 0 && graph_ctx.mask_config.prefetch_mode > 0) {
            if (graph_ctx.mask_config.prefetch_mode == 3) {
                // DROPLET-style: prefetch every next-K out-neighbor.
                auto jt = it;
                for (int step = 0; step < pfx_lookahead; step++) {
                    ++jt;
                    if (jt == out_neigh.end()) break;
                    WNode candidate_node = *jt;
                    NodeID candidate = candidate_node.v;
                    if (candidate < 0) continue;
                    SIM_CACHE_PREFETCH_VERTEX(cache, dist.data(),
                        static_cast<uint32_t>(candidate), graph_ctx);
                }
            } else {
                uint32_t lookahead_target = UINT32_MAX;
                auto jt = it;
                for (int step = 0; step < pfx_lookahead; step++) {
                    ++jt;
                    if (jt == out_neigh.end()) break;
                    WNode candidate_node = *jt;
                    NodeID candidate = candidate_node.v;
                    if (candidate < 0) continue;
                    if (graph_ctx.mask_config.prefetch_mode == 1) {
                        if (lookahead_target == UINT32_MAX ||
                            g.out_degree(candidate) > g.out_degree(lookahead_target)) {
                            lookahead_target = static_cast<uint32_t>(candidate);
                        }
                    } else {
                        uint8_t candidate_popt = graph_ctx.mask_config.decodePOPT(vertex_masks[candidate]);
                        if (lookahead_target == UINT32_MAX ||
                            candidate_popt < graph_ctx.mask_config.decodePOPT(vertex_masks[lookahead_target])) {
                            lookahead_target = static_cast<uint32_t>(candidate);
                        }
                    }
                }
                if (lookahead_target != UINT32_MAX) {
                    SIM_CACHE_PREFETCH_VERTEX(cache, dist.data(), lookahead_target, graph_ctx);
                } else {
                    graph_ctx.recordPrefetchNoTarget();
                }
            }
        }
        WeightT old_dist = dist[wn.v];
        WeightT new_dist = dist[u] + wn.w;
        SIM_CACHE_READ_MASKED(cache, dist.data(), wn.v, graph_ctx, vertex_masks[wn.v]);
        while (new_dist < old_dist) {
            SIM_CACHE_WRITE(cache, dist.data(), wn.v);
            if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
                size_t dest_bin = new_dist / delta;
                if (dest_bin >= local_bins.size())
                    local_bins.resize(dest_bin + 1);
                local_bins[dest_bin].push_back(wn.v);
                break;
            }
            old_dist = dist[wn.v];
            SIM_CACHE_READ_MASKED(cache, dist.data(), wn.v, graph_ctx, vertex_masks[wn.v]);
        }
    }
}

template<typename CacheType>
pvector<WeightT> DeltaStep_Sim(const WGraph &g, NodeID source, 
                                WeightT delta, CacheType &cache) {
    pvector<WeightT> dist(g.num_nodes(), kDistInf);
    dist[source] = 0;

    // --- Graph-aware cache context ---
    GraphCacheContext graph_ctx;
    pvector<uint32_t> deg_arr(g.num_nodes());
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
        deg_arr[n] = static_cast<uint32_t>(g.out_degree(n));
    graph_ctx.initTopology(deg_arr.data(), g.num_nodes(),
                           g.num_edges_directed(), g.directed());
    size_t llc_size = 8 * 1024 * 1024;
    llc_size = GetEnvSizeBytes("CACHE_L3_SIZE", llc_size);
    graph_ctx.registerPropertyArray(dist.data(), g.num_nodes(), sizeof(WeightT), llc_size);
    cache.initGraphContext(&graph_ctx);

    // Build P-OPT rereference matrix before masks so POPT-ranked PFX can use it.
    static pvector<uint8_t> popt_matrix;
    {
        const char* policy_env = getenv("CACHE_POLICY");
        std::string policy_str = policy_env ? policy_env : "";
        const char* pfx_env = getenv("ECG_PREFETCH_MODE");
        bool popt_prefetch = pfx_env && atoi(pfx_env) == 2;
        if (policy_str == "POPT" || policy_str == "ECG" || popt_prefetch) {
            constexpr int numVtxPerLine = 64 / sizeof(WeightT);
            constexpr int numEpochs = 256;
            makeOffsetMatrix(g, popt_matrix, numVtxPerLine, numEpochs);
            int numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
            graph_ctx.initRereference(popt_matrix.data(), numCacheLines,
                                      numEpochs, g.num_nodes(), 64);
        }
    }

    // Compute per-vertex ECG mask array
    graph_ctx.initMaskConfig();
    auto vertex_masks = graph_ctx.computeVertexMasks(g);
    graph_ctx.initMaskArray32(vertex_masks.data(), vertex_masks.size());
    int pfx_lookahead = GraphSimEnvIntClamped("ECG_PREFETCH_LOOKAHEAD", 0, 0, 64);
    if (pfx_lookahead > 0 && graph_ctx.mask_config.prefetch_mode > 0) {
        cout << "SSSP relax PFX lookahead: window=" << pfx_lookahead
             << " mode=" << int(graph_ctx.mask_config.prefetch_mode) << endl;
    }

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
                SIM_SET_VERTEX(cache, u);
                SIM_CACHE_READ(cache, dist.data(), u);
                if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index))
                    RelaxEdges_Sim(g, u, delta, dist, local_bins, cache,
                                   graph_ctx, vertex_masks, pfx_lookahead);
            }

            while (curr_bin_index < local_bins.size() &&
                   !local_bins[curr_bin_index].empty() &&
                   local_bins[curr_bin_index].size() < kBinSizeThreshold) {
                vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
                local_bins[curr_bin_index].resize(0);
                for (NodeID u : curr_bin_copy) {
                    SIM_SET_VERTEX(cache, u);
                    SIM_CACHE_READ(cache, dist.data(), u);
                    RelaxEdges_Sim(g, u, delta, dist, local_bins, cache,
                                   graph_ctx, vertex_masks, pfx_lookahead);
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
    
    bool multicore = IsMultiCoreMode();
    bool fast = IsFastMode();
    
    if (multicore) {
        MultiCoreCacheHierarchy cache = MultiCoreCacheHierarchy::fromEnvironment();
        
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
    } else if (fast) {
        // FAST single-core cache simulation (no locks, ~10x faster)
        FastCacheHierarchy cache = FastCacheHierarchy::fromEnvironment();
        
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
    } else {
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
    }
    
    return 0;
}
