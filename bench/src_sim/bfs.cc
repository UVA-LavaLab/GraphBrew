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

#include "cache_sim.h"
#include "graph_sim.h"

using namespace std;
using namespace cache_sim;

int64_t BUStep_Sim(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
                   Bitmap &next, CacheHierarchy &cache) {
    int64_t awake_count = 0;
    next.reset();
    #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        // Track: read parent[u]
        CACHE_READ(cache, parent.data(), u);
        if (parent[u] < 0) {
            for (NodeID v : g.in_neigh(u)) {
                // Track: check if v is in frontier
                if (front.get_bit(v)) {
                    // Track: write parent[u]
                    CACHE_WRITE(cache, parent.data(), u);
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

int64_t TDStep_Sim(const Graph &g, pvector<NodeID> &parent,
                   SlidingQueue<NodeID> &queue, CacheHierarchy &cache) {
    int64_t scout_count = 0;
    #pragma omp parallel
    {
        QueueBuffer<NodeID> lqueue(queue);
        #pragma omp for reduction(+ : scout_count)
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            for (NodeID v : g.out_neigh(u)) {
                // Track: read parent[v]
                CACHE_READ(cache, parent.data(), v);
                NodeID curr_val = parent[v];
                if (curr_val < 0) {
                    // Track: write parent[v]
                    CACHE_WRITE(cache, parent.data(), v);
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

pvector<NodeID> DOBFS_Sim(const Graph &g, NodeID source, CacheHierarchy &cache,
                          int alpha = 15, int beta = 18) {
    pvector<NodeID> parent = InitParent(g);
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
            QueueToBitmap(queue, front);
            awake_count = queue.size();
            queue.slide_window();
            do {
                old_awake_count = awake_count;
                awake_count = BUStep_Sim(g, parent, front, curr, cache);
                front.swap(curr);
            } while ((awake_count >= old_awake_count) ||
                     (awake_count > g.num_nodes() / beta));
            BitmapToQueue(g, front, queue);
            scout_count = 1;
        } else {
            edges_to_check -= scout_count;
            scout_count = TDStep_Sim(g, parent, queue, cache);
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
    
    return 0;
}
