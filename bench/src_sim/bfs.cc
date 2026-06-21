// Copyright (c) 2024, UVA LavaLab
// BFS with Cache Simulation

#include <iostream>
#include <vector>
#include <fstream>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"

#include "cache_sim/cache_sim.h"
#include "cache_sim/graph_sim.h"

// P-OPT rereference matrix builder
#include "graphbrew/partition/cagra/popt.h"

using namespace std;
using namespace cache_sim;

template<typename CacheType>
int64_t BUStep_Sim(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
                   Bitmap &next, CacheType &cache,
                   GraphCacheContext &graph_ctx, const std::vector<uint32_t> &vertex_masks) {
    int64_t awake_count = 0;
    next.reset();
    #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        // P-OPT: update current destination vertex
        SIM_SET_VERTEX(cache, u);
        // Track: read parent[u]
        SIM_CACHE_READ(cache, parent.data(), u);
        if (parent[u] < 0) {
            auto in_neigh = g.in_neigh(u);
            // ECG_BFS_EDGE_MASKS: model BU's frontier-bitmap membership probe as a
            // cache access (a real load of front's word holding v's bit) and carry
            // the IN-edge per-edge mask, so BU is masked symmetric to TD (the
            // dual-direction capability). The transpose-correct epoch
            // (in_edge_epoch_by_src[u][edge_pos] = next out_neigh(v) > u) is when
            // v's frontier bit is next read. Gated -> the default BFS access stream
            // is unchanged. HONEST CAVEAT: a 64B bitmap line holds 512 vertices'
            // bits and the bitmap is compact/uniformly-hot by design (BU exists to
            // avoid TD's property traffic), so this is a do-no-harm completeness
            // mask whose measurable effect is ~nil on the symmetric corpus.
            const bool use_in_edge_masks =
                !graph_ctx.in_edge_masks_by_src.empty() &&
                u < (NodeID)graph_ctx.in_edge_masks_by_src.size() &&
                graph_ctx.in_edge_masks_by_src[u].size() == (size_t)g.in_degree(u);
            size_t edge_pos = 0;
            for (auto it = in_neigh.begin(); it != in_neigh.end(); ++it, ++edge_pos) {
                SIM_CACHE_READ_EDGE(cache, it);
                NodeID v = *it;
                if (use_in_edge_masks) {
                    uint64_t emask = graph_ctx.in_edge_masks_by_src[u][edge_pos];
                    graph_ctx.hints_for_thread().edge_epoch =
                        graph_ctx.in_edge_epoch_by_src[u][edge_pos];
                    SIM_CACHE_READ_MASKED(cache, front.data(), (size_t)v / 64, graph_ctx,
                                          GraphCacheContext::edgeMaskPOPT(emask));
                }
                // Track: check if v is in frontier
                if (front.get_bit(v)) {
                    // Track: write parent[u]
                    SIM_CACHE_WRITE(cache, parent.data(), u);
                    parent[u] = v;
                    awake_count++;
                    next.set_bit(u);
                    break;
                }
            }
        }
    }
    return awake_count;
}

template<typename CacheType>
int64_t TDStep_Sim(const Graph &g, pvector<NodeID> &parent,
                   SlidingQueue<NodeID> &queue, CacheType &cache,
                   GraphCacheContext &graph_ctx, const std::vector<uint32_t> &vertex_masks,
                   int pfx_lookahead, int pfx_top_k = 1) {
    int64_t scout_count = 0;
    #pragma omp parallel
    {
        QueueBuffer<NodeID> lqueue(queue);
        #pragma omp for reduction(+ : scout_count)
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            SIM_SET_VERTEX(cache, u);  // current frontier vertex = the traversal clock
            auto out_neigh = g.out_neigh(u);
            // ECG_BFS_EDGE_MASKS: consume the OUT-edge per-edge masks (the transpose-
            // correct dual-direction masks built for the out edge list) instead of the
            // per-vertex masks. The per-edge epoch is src-iteration-aware (next
            // in-neighbour of dest > u), which the single per-vertex mask cannot encode.
            const bool use_out_edge_masks =
                !graph_ctx.out_edge_masks_by_src.empty() &&
                u < (NodeID)graph_ctx.out_edge_masks_by_src.size() &&
                graph_ctx.out_edge_masks_by_src[u].size() == (size_t)g.out_degree(u);
            size_t edge_pos = 0;
            for (auto it = out_neigh.begin(); it != out_neigh.end(); ++it, ++edge_pos) {
                SIM_CACHE_READ_EDGE(cache, it);
                NodeID v = *it;
                if (pfx_lookahead > 0 && graph_ctx.mask_config.prefetch_mode > 0) {
                    if (graph_ctx.mask_config.prefetch_mode == 3) {
                        // DROPLET-style: prefetch every next-K out-neighbor
                        // sequentially (no target selection).
                        auto jt = it;
                        for (int step = 0; step < pfx_lookahead; step++) {
                            ++jt;
                            if (jt == out_neigh.end()) break;
                            NodeID candidate = *jt;
                            if (candidate < 0) continue;
                            SIM_CACHE_PREFETCH_VERTEX(cache, parent.data(),
                                static_cast<uint32_t>(candidate), graph_ctx);
                        }
                    } else {
                        // Top-K POPT/degree-ranked selection (sprint 6f-3).
                        struct Cand { uint32_t v; uint16_t key; };
                        Cand cands[64];
                        int n_cand = 0;
                        auto jt = it;
                        for (int step = 0; step < pfx_lookahead; step++) {
                            ++jt;
                            if (jt == out_neigh.end()) break;
                            NodeID candidate = *jt;
                            if (candidate < 0) continue;
                            uint16_t key;
                            if (graph_ctx.mask_config.prefetch_mode == 1) {
                                uint64_t od = g.out_degree(candidate);
                                key = od > 65535 ? 0 : static_cast<uint16_t>(65535 - od);
                            } else {
                                key = graph_ctx.mask_config.decodePOPT(vertex_masks[candidate]);
                            }
                            cands[n_cand++] = {static_cast<uint32_t>(candidate), key};
                        }
                        if (n_cand == 0) {
                            graph_ctx.recordPrefetchNoTarget();
                        } else if (pfx_top_k <= 1) {
                            int best = 0;
                            for (int i = 1; i < n_cand; i++)
                                if (cands[i].key < cands[best].key) best = i;
                            SIM_CACHE_PREFETCH_VERTEX(cache, parent.data(), cands[best].v, graph_ctx);
                        } else {
                            int k_eff = pfx_top_k < n_cand ? pfx_top_k : n_cand;
                            for (int i = 0; i < k_eff; i++) {
                                int best = i;
                                for (int j = i + 1; j < n_cand; j++)
                                    if (cands[j].key < cands[best].key) best = j;
                                if (best != i) std::swap(cands[i], cands[best]);
                                SIM_CACHE_PREFETCH_VERTEX(cache, parent.data(), cands[i].v, graph_ctx);
                            }
                        }
                    }
                }
                // Track: read parent[v]. With OUT-edge masks, carry this edge's
                // src-aware epoch + POPT (transpose-correct for the push read);
                // otherwise fall back to the per-vertex mask.
                if (use_out_edge_masks) {
                    uint64_t emask = graph_ctx.out_edge_masks_by_src[u][edge_pos];
                    graph_ctx.hints_for_thread().edge_epoch =
                        graph_ctx.out_edge_epoch_by_src[u][edge_pos];
                    SIM_CACHE_READ_MASKED(cache, parent.data(), v, graph_ctx,
                                          GraphCacheContext::edgeMaskPOPT(emask));
                } else {
                    SIM_CACHE_READ_MASKED(cache, parent.data(), v, graph_ctx, vertex_masks[v]);
                }
                NodeID curr_val = parent[v];
                if (curr_val < 0) {
                    // Track: write parent[v]
                    SIM_CACHE_WRITE(cache, parent.data(), v);
                    if (compare_and_swap(parent[v], curr_val, u)) {
                        lqueue.push_back(v);
                        scout_count += -curr_val;
                    }
                }
            }
        }
        lqueue.flush();
    }
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
        #pragma omp for
        for (NodeID n = 0; n < g.num_nodes(); n++)
            if (bm.get_bit(n))
                lqueue.push_back(n);
        lqueue.flush();
    }
    queue.slide_window();
}

pvector<NodeID> InitParent(const Graph &g) {
    pvector<NodeID> parent(g.num_nodes());
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
        parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
    return parent;
}

template<typename CacheType>
pvector<NodeID> DOBFS_Sim(const Graph &g, NodeID source, CacheType &cache,
                          int alpha = 15, int beta = 18) {
    pvector<NodeID> parent = InitParent(g);
    parent[source] = source;

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
    graph_ctx.registerPropertyArray(parent.data(), g.num_nodes(), sizeof(NodeID), llc_size);
    cache.initGraphContext(&graph_ctx);

    // Build P-OPT rereference matrix before masks so POPT-ranked PFX can use it.
    static pvector<uint8_t> popt_matrix;
    {
        const char* policy_env = getenv("CACHE_POLICY");
        std::string policy_str = policy_env ? policy_env : "";
        const char* pfx_env = getenv("ECG_PREFETCH_MODE");
        bool popt_prefetch = pfx_env && atoi(pfx_env) == 2;
        if (policy_str == "POPT" || policy_str == "ECG" || popt_prefetch) {
            constexpr int numVtxPerLine = 64 / sizeof(NodeID);
            constexpr int numEpochs = 256;
            // BFS is direction-optimizing, but its ONLY masked property read is the
            // TD (push) parent[v] over out_neigh(u); BU (pull) uses a frontier bitmap
            // (no masked read). The next reader of parent[v] is in_neigh(v), so the
            // transpose-correct rereference direction is IN/CSC (traverseCSR=false).
            // ECG_BFS_FORCE_OUT reverts to CSR for direction-transfer experiments;
            // ECG_EXACT_BFS instead uses its own visit-order skeleton clock (below).
            // (On the symmetric eval corpus in==out so this is inert; it is the
            // correct default for directed graphs. See ecg_mask_direction_and_metadata.md.)
            bool bfs_natural_csr = std::getenv("ECG_BFS_FORCE_OUT") != nullptr;
            makeOffsetMatrix(g, popt_matrix, numVtxPerLine, numEpochs,
                             ecgRerefTraverseCSR(bfs_natural_csr, g, "BFS(TD/out->in-transpose)"));
            int numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
            graph_ctx.initRereference(popt_matrix.data(), numCacheLines,
                                      numEpochs, g.num_nodes(), 64);
            graph_ctx.exact_vtx_per_line = numVtxPerLine;
            if (std::getenv("ECG_EXACT_REREF")) {
                const char* eb = std::getenv("ECG_EXACT_BITS");
                if (eb) graph_ctx.exact_bits = (uint32_t)atoi(eb);
                if (std::getenv("ECG_EXACT_BFS")) {
                    // source-specific: BFS skeleton from the kernel's own source —
                    // UNLESS ECG_BFS_MASK_SRC overrides it (source-TRANSFER test: build
                    // the mask from a DIFFERENT source than the kernel runs).
                    uint32_t mask_src = (uint32_t)source;
                    if (const char* ms = std::getenv("ECG_BFS_MASK_SRC"))
                        mask_src = (uint32_t)atoi(ms);
                    if (std::getenv("ECG_BFS_HUBSRC")) {
                        // canonical deterministic source-independent choice: the highest
                        // out-degree hub (most central -> most representative BFS layering).
                        uint32_t best = 0; uint64_t bd = 0;
                        for (uint32_t v = 0; v < (uint32_t)g.num_nodes(); ++v)
                            if (g.out_degree(v) > bd) { bd = g.out_degree(v); best = v; }
                        mask_src = best;
                    }
                    if (std::getenv("ECG_BFS_KSOURCE")) {
                        // K-source EXPECTED-REUSE consensus clock (source-independent).
                        uint32_t K = 8, seed = 12345;
                        if (const char* s = std::getenv("ECG_BFS_K")) K = (uint32_t)atoi(s);
                        if (const char* s = std::getenv("ECG_BFS_KSEED")) seed = (uint32_t)atoi(s);
                        graph_ctx.buildBFSVisitOrderKSource(g, K, seed);
                    } else if (std::getenv("ECG_BFS_BOUNDED")) {
                        // SOURCE-INDEPENDENT depth-bounded degree-seeded clustering clock.
                        uint32_t d = 8;
                        if (const char* s = std::getenv("ECG_BFS_BOUND_DEPTH")) d = (uint32_t)atoi(s);
                        if (std::getenv("ECG_BFS_COMMUNITY"))
                            graph_ctx.buildBoundedBFSOrderCommunity(g, d);
                        else
                            graph_ctx.buildBoundedBFSOrder(g, d);
                    } else if (std::getenv("ECG_BFS_DEPTHORDER")) {
                        graph_ctx.buildBFSVisitOrderByDepth(g, mask_src);
                    } else {
                        graph_ctx.buildBFSVisitOrder(g, mask_src);
                    }
                    graph_ctx.registerInAdjacencyExactBFS(g);
                } else if (std::getenv("ECG_EXACT_IN")) {
                    // SOURCE-INDEPENDENT (the RCM variation): in-adjacency mask with
                    // ID-order clock. Built ONCE, no per-source BFS. On an RCM-reordered
                    // graph ID-order ~ BFS-frontier-order for any source, so this
                    // approximates the per-source BFS mask without knowing the source.
                    graph_ctx.visit_pos.resize(g.num_nodes());
                    for (uint32_t v = 0; v < (uint32_t)g.num_nodes(); ++v)
                        graph_ctx.visit_pos[v] = v;
                    graph_ctx.registerInAdjacencyExactBFS(g);
                } else {
                    graph_ctx.registerOutAdjacencyExact(g);  // ECG_EXACT mode (sweep flavor)
                }
            }
        }
    }

    // Compute per-vertex ECG mask array
    graph_ctx.initMaskConfig();
    auto vertex_masks = graph_ctx.computeVertexMasks(g);
    graph_ctx.initMaskArray32(vertex_masks.data(), vertex_masks.size());
    // ECG_BFS_EDGE_MASKS: build the dual-direction per-edge masks so BOTH BFS
    // phases carry a src-aware, transpose-correct epoch per edge instead of the
    // single per-vertex value. TD (push) traverses out_neigh(u) reading parent[v]
    // -> OUT-edge masks (epoch from in_neigh(v)); BU (pull) traverses in_neigh(u)
    // probing the frontier bit of v -> IN-edge masks (epoch from out_neigh(v)).
    // Inert on symmetric graphs (in==out); the correct dual mask for directed graphs.
    if (std::getenv("ECG_BFS_EDGE_MASKS")) {
        graph_ctx.buildOutEdgeMasks(g);     // TD push: parent[v] over out_neigh(u)
        graph_ctx.buildInEdgeMasksBFS(g);   // BU pull: frontier bit of v over in_neigh(u)
        cout << "BFS: dual-direction per-edge masks enabled (TD=OUT-edge, BU=IN-edge)" << endl;
    }
    int pfx_lookahead = GraphSimEnvIntClamped("ECG_PREFETCH_LOOKAHEAD", 0, 0, 64);
    int pfx_top_k = GraphSimEnvIntClamped("ECG_PREFETCH_TOP_K", 1, 1, 64);
    if (pfx_lookahead > 0 && graph_ctx.mask_config.prefetch_mode > 0) {
        cout << "BFS TD PFX lookahead: window=" << pfx_lookahead
             << " mode=" << int(graph_ctx.mask_config.prefetch_mode)
             << " top_k=" << pfx_top_k << endl;
    }

    SlidingQueue<NodeID> queue(g.num_nodes());
    queue.push_back(source);
    queue.slide_window();
    Bitmap curr(g.num_nodes());
    curr.reset();
    Bitmap front(g.num_nodes());
    front.reset();
    int64_t edges_to_check = g.num_edges_directed();
    int64_t scout_count = g.out_degree(source);
    // ECG_BFS_FORCE_TD: stay top-down (skip the bottom-up phase) so the BFS-order
    // EXACT generator (which models the top-down access pattern) can be validated
    // against the actual access order. Direction-optimizing BU needs its own model.
    static const bool force_td = std::getenv("ECG_BFS_FORCE_TD") != nullptr;

    while (!queue.empty()) {
        if (!force_td && scout_count > edges_to_check / alpha) {
            int64_t awake_count, old_awake_count;
            QueueToBitmap(queue, front);
            awake_count = queue.size();
            queue.slide_window();
            do {
                old_awake_count = awake_count;
                awake_count = BUStep_Sim(g, parent, front, curr, cache, graph_ctx, vertex_masks);
                front.swap(curr);
            } while ((awake_count >= old_awake_count) ||
                     (awake_count > g.num_nodes() / beta));
            BitmapToQueue(g, front, queue);
            scout_count = 1;
        } else {
            edges_to_check -= scout_count;
            scout_count = TDStep_Sim(g, parent, queue, cache, graph_ctx, vertex_masks,
                                     pfx_lookahead, pfx_top_k);
            queue.slide_window();
        }
    }
    return parent;
}

void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
    int64_t tree_size = 0;
    int64_t n_edges = 0;
    for (NodeID n : g.vertices()) {
        if (bfs_tree[n] >= 0) {
            n_edges += g.out_degree(n);
            tree_size++;
        }
    }
    cout << "BFS Tree has " << static_cast<long long>(tree_size) << " nodes and ";
    cout << static_cast<long long>(n_edges) << " edges" << endl;
}

bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
    pvector<int> depth(g.num_nodes(), -1);
    depth[source] = 0;
    vector<NodeID> to_visit;
    to_visit.reserve(g.num_nodes());
    to_visit.push_back(source);
    for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
        NodeID u = *it;
        for (NodeID v : g.out_neigh(u)) {
            if (depth[v] == -1) {
                depth[v] = depth[u] + 1;
                to_visit.push_back(v);
            }
        }
    }
    for (NodeID u : g.vertices()) {
        if ((depth[u] != -1) && (parent[u] != -1)) {
            if (u == source) {
                if (parent[u] != u) return false;
            } else {
                bool found = false;
                for (NodeID v : g.in_neigh(u)) {
                    if (parent[u] == v) {
                        if (depth[v] != depth[u] - 1) return false;
                        found = true;
                        break;
                    }
                }
                if (!found) return false;
            }
        } else if (depth[u] != parent[u]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    CLApp cli(argc, argv, "bfs-sim");
    if (!cli.ParseArgs())
        return -1;
    
    Builder b(cli);
    Graph g = b.MakeGraph();
    
    bool multicore = IsMultiCoreMode();
    bool fast = IsFastMode();
    
    if (multicore) {
        MultiCoreCacheHierarchy cache = MultiCoreCacheHierarchy::fromEnvironment();
        
        SourcePicker<Graph> sp(g, cli.start_vertex(), cli.num_trials());
        auto BFSBound = [&sp, &cache](const Graph &g) {
            return DOBFS_Sim(g, sp.PickNext(), cache);
        };
        SourcePicker<Graph> vsp(g, cli.start_vertex(), cli.num_trials());
        auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
            return BFSVerifier(g, vsp.PickNext(), parent);
        };
        
        BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
        
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
        // FAST single-core cache simulation (no locks, ~10x faster)
        FastCacheHierarchy cache = FastCacheHierarchy::fromEnvironment();
        
        SourcePicker<Graph> sp(g, cli.start_vertex(), cli.num_trials());
        auto BFSBound = [&sp, &cache](const Graph &g) {
            return DOBFS_Sim(g, sp.PickNext(), cache);
        };
        SourcePicker<Graph> vsp(g, cli.start_vertex(), cli.num_trials());
        auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
            return BFSVerifier(g, vsp.PickNext(), parent);
        };
        
        BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
        
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
        CacheHierarchy cache = CacheHierarchy::fromEnvironment();
        
        SourcePicker<Graph> sp(g, cli.start_vertex(), cli.num_trials());
        auto BFSBound = [&sp, &cache](const Graph &g) {
            return DOBFS_Sim(g, sp.PickNext(), cache);
        };
        SourcePicker<Graph> vsp(g, cli.start_vertex(), cli.num_trials());
        auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
            return BFSVerifier(g, vsp.PickNext(), parent);
        };
        
        BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
        
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
    }
    
    return 0;
}
