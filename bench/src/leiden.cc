// Copyright (c) 2024, GraphBrew
// Leiden-based Graph Reordering with Dendrogram-style Cache Optimization
//
// This implements Leiden community detection with RabbitOrder-inspired
// dendrogram traversal for optimal cache locality.
//
// Flavors:
//   0: DFS_STANDARD    - Standard DFS traversal (like RabbitOrder)
//   1: DFS_HUB_FIRST   - DFS with high-degree nodes first within communities
//   2: DFS_SIZE_FIRST  - DFS with largest subtrees first
//   3: BFS_LEVEL       - BFS by level
//   4: HYBRID_HUB_DFS  - Hubs first globally, then DFS within communities
//   5: LAST_PASS_DEG   - Sort by last-pass community, then degree (simplest)

#include <algorithm>
#include <iostream>
#include <vector>
#include <deque>
#include <numeric>
#include <cstring>
#include <random>
#include <parallel/algorithm>
#include <unordered_map>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"

// Leiden includes
#include "main.hxx"

using namespace std;

//==============================================================================
// ORDERING FLAVORS
//==============================================================================

enum class OrderingFlavor {
    DFS_STANDARD = 0,      // Standard DFS (like RabbitOrder)
    DFS_HUB_FIRST = 1,     // DFS with high-degree nodes first
    DFS_SIZE_FIRST = 2,    // DFS with largest subtrees first
    BFS_LEVEL = 3,         // BFS by level (process all at same depth)
    HYBRID_HUB_DFS = 4,    // Hubs first globally, then DFS within communities
    LAST_PASS_DEG = 5,     // Sort by last-pass community + degree (baseline)
};

//==============================================================================
// DENDROGRAM NODE STRUCTURE
//==============================================================================

struct DendrogramNode {
    int64_t parent;      
    int64_t first_child; 
    int64_t sibling;     
    int64_t vertex_id;   // Original vertex ID (-1 for internal nodes)
    size_t subtree_size; 
    double weight;       // Degree sum
    int level;           
    
    DendrogramNode() : parent(-1), first_child(-1), sibling(-1), 
                       vertex_id(-1), subtree_size(1), weight(0.0), level(0) {}
};

//==============================================================================
// LEIDEN DENDROGRAM BUILDER
//==============================================================================

class LeidenDendrogram {
public:
    vector<DendrogramNode> nodes;
    vector<int64_t> roots;
    size_t num_vertices;
    
    LeidenDendrogram(size_t n) : num_vertices(n) {
        nodes.reserve(n * 2);
    }
    
    /**
     * Build dendrogram from Leiden's per-pass community mappings
     */
    template<typename K>
    void buildFromLeiden(const vector<vector<K>>& communityMappingPerPass,
                         const vector<K>& degrees) {
        const size_t num_passes = communityMappingPerPass.size();
        
        // Create leaf nodes for all vertices
        nodes.resize(num_vertices);
        #pragma omp parallel for
        for (size_t v = 0; v < num_vertices; ++v) {
            nodes[v].vertex_id = v;
            nodes[v].subtree_size = 1;
            nodes[v].weight = degrees[v];
            nodes[v].level = 0;
        }
        
        if (num_passes == 0) {
            // No community structure - each vertex is its own root
            for (size_t v = 0; v < num_vertices; ++v) {
                roots.push_back(v);
            }
            return;
        }
        
        // Build hierarchy from finest to coarsest
        // At each pass, group nodes by their community
        vector<int64_t> current_nodes(num_vertices);
        iota(current_nodes.begin(), current_nodes.end(), 0);
        
        for (size_t pass = 0; pass < num_passes; ++pass) {
            const auto& comm_map = communityMappingPerPass[pass];
            
            // Group current nodes by community at this pass
            unordered_map<K, vector<int64_t>> community_members;
            for (int64_t node_id : current_nodes) {
                // Find representative vertex for this node
                int64_t v = node_id;
                while (v >= 0 && nodes[v].vertex_id < 0) {
                    v = nodes[v].first_child;
                }
                if (v >= 0 && nodes[v].vertex_id >= 0) {
                    size_t vertex = nodes[v].vertex_id;
                    if (vertex < comm_map.size()) {
                        community_members[comm_map[vertex]].push_back(node_id);
                    }
                }
            }
            
            // Create internal nodes for multi-member communities
            vector<int64_t> next_nodes;
            for (auto& [comm_id, members] : community_members) {
                if (members.size() == 1) {
                    next_nodes.push_back(members[0]);
                } else {
                    // Create internal node
                    DendrogramNode internal;
                    internal.vertex_id = -1;
                    internal.level = pass + 1;
                    internal.subtree_size = 0;
                    internal.weight = 0.0;
                    
                    int64_t internal_id = nodes.size();
                    nodes.push_back(internal);
                    
                    // Sort members by weight (degree) descending for hub-first
                    sort(members.begin(), members.end(), [this](int64_t a, int64_t b) {
                        return nodes[a].weight > nodes[b].weight;
                    });
                    
                    // Link children
                    int64_t prev_sibling = -1;
                    for (int64_t child_id : members) {
                        nodes[child_id].parent = internal_id;
                        if (nodes[internal_id].first_child == -1) {
                            nodes[internal_id].first_child = child_id;
                        }
                        if (prev_sibling >= 0) {
                            nodes[prev_sibling].sibling = child_id;
                        }
                        prev_sibling = child_id;
                        
                        nodes[internal_id].subtree_size += nodes[child_id].subtree_size;
                        nodes[internal_id].weight += nodes[child_id].weight;
                    }
                    
                    next_nodes.push_back(internal_id);
                }
            }
            current_nodes = move(next_nodes);
        }
        
        // Remaining nodes are roots
        for (int64_t node_id : current_nodes) {
            roots.push_back(node_id);
        }
        
        // Sort roots by subtree size
        sort(roots.begin(), roots.end(), [this](int64_t a, int64_t b) {
            return nodes[a].subtree_size > nodes[b].subtree_size;
        });
    }
};

//==============================================================================
// ORDERING FUNCTIONS
//==============================================================================

void orderDFS(const LeidenDendrogram& dendro, pvector<NodeID>& new_ids,
              bool hub_first = false, bool size_first = false) {
    NodeID current_id = 0;
    deque<int64_t> stack;
    
    for (int64_t root : dendro.roots) {
        stack.push_back(root);
        
        while (!stack.empty()) {
            int64_t node_id = stack.back();
            stack.pop_back();
            
            const auto& node = dendro.nodes[node_id];
            
            if (node.vertex_id >= 0) {
                new_ids[node.vertex_id] = current_id++;
            } else {
                vector<int64_t> children;
                int64_t child = node.first_child;
                while (child >= 0) {
                    children.push_back(child);
                    child = dendro.nodes[child].sibling;
                }
                
                if (hub_first) {
                    sort(children.begin(), children.end(), 
                         [&dendro](int64_t a, int64_t b) {
                             return dendro.nodes[a].weight > dendro.nodes[b].weight;
                         });
                } else if (size_first) {
                    sort(children.begin(), children.end(),
                         [&dendro](int64_t a, int64_t b) {
                             return dendro.nodes[a].subtree_size > dendro.nodes[b].subtree_size;
                         });
                }
                
                for (auto it = children.rbegin(); it != children.rend(); ++it) {
                    stack.push_back(*it);
                }
            }
        }
    }
}

void orderBFS(const LeidenDendrogram& dendro, pvector<NodeID>& new_ids) {
    NodeID current_id = 0;
    deque<int64_t> queue;
    
    for (int64_t root : dendro.roots) {
        queue.push_back(root);
    }
    
    while (!queue.empty()) {
        int64_t node_id = queue.front();
        queue.pop_front();
        
        const auto& node = dendro.nodes[node_id];
        
        if (node.vertex_id >= 0) {
            new_ids[node.vertex_id] = current_id++;
        } else {
            int64_t child = node.first_child;
            while (child >= 0) {
                queue.push_back(child);
                child = dendro.nodes[child].sibling;
            }
        }
    }
}

template<typename K>
void orderHybridHubDFS(const vector<vector<K>>& communityMappingPerPass,
                        const vector<K>& degrees,
                        pvector<NodeID>& new_ids) {
    const size_t n = degrees.size();
    
    // Get last-pass communities
    const auto& last_pass = communityMappingPerPass.back();
    
    // Create (vertex, community, degree) tuples
    vector<tuple<size_t, K, K>> vertices(n);
    #pragma omp parallel for
    for (size_t v = 0; v < n; ++v) {
        vertices[v] = make_tuple(v, last_pass[v], degrees[v]);
    }
    
    // Sort by (community, degree descending)
    __gnu_parallel::sort(vertices.begin(), vertices.end(),
        [](const auto& a, const auto& b) {
            if (get<1>(a) != get<1>(b)) return get<1>(a) < get<1>(b);
            return get<2>(a) > get<2>(b);
        });
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        new_ids[get<0>(vertices[i])] = i;
    }
}

template<typename K>
void orderLastPassDeg(const vector<vector<K>>& communityMappingPerPass,
                      const vector<K>& degrees,
                      pvector<NodeID>& new_ids) {
    const size_t n = degrees.size();
    
    if (communityMappingPerPass.empty()) {
        // No community - sort by degree
        vector<pair<K, size_t>> deg_vertex(n);
        #pragma omp parallel for
        for (size_t v = 0; v < n; ++v) {
            deg_vertex[v] = {degrees[v], v};
        }
        __gnu_parallel::sort(deg_vertex.begin(), deg_vertex.end(), greater<>());
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            new_ids[deg_vertex[i].second] = i;
        }
        return;
    }
    
    const auto& last_pass = communityMappingPerPass.back();
    
    vector<tuple<K, K, size_t>> comm_deg_vertex(n);  // (comm, -deg, vertex)
    #pragma omp parallel for
    for (size_t v = 0; v < n; ++v) {
        comm_deg_vertex[v] = make_tuple(last_pass[v], (K)(UINT32_MAX - degrees[v]), v);
    }
    
    __gnu_parallel::sort(comm_deg_vertex.begin(), comm_deg_vertex.end());
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        new_ids[get<2>(comm_deg_vertex[i])] = i;
    }
}

//==============================================================================
// MAIN LEIDEN REORDERING FUNCTION
//==============================================================================

void GenerateLeidenDendrogramMapping(
    const Graph& g,
    pvector<NodeID>& new_ids,
    double resolution,
    int max_iterations,
    int max_passes,
    OrderingFlavor flavor) {
    
    Timer tm;
    tm.Start();
    
    const int64_t num_nodes = g.num_nodes();
    
    // Build Leiden-compatible graph
    using K = uint32_t;
    using V = None;
    using W = float;
    
    DiGraph<K, V, W> x;
    x.respan(num_nodes);
    
    // Add edges
    for (NodeID u = 0; u < num_nodes; ++u) {
        for (NodeID v : g.out_neigh(u)) {
            x.addEdge(u, v, 1.0f);
        }
    }
    x.update();
    
    tm.Stop();
    PrintTime("Graph Build Time", tm.Seconds());
    
    // Run Leiden algorithm
    tm.Start();
    
    // Need random number generator for Leiden
    std::random_device dev;
    std::default_random_engine rnd(dev());
    int repeat = 1;
    
    // Call Leiden with correct API: leidenStaticOmp<RANDOM, USEPARENT>(rnd, graph, {options})
    auto result = leidenStaticOmp<false, false>(
        rnd, x,
        {repeat, resolution, 1e-12, 0.8, 1.0, max_iterations, max_passes});
    
    tm.Stop();
    PrintTime("Leiden Time", tm.Seconds());
    PrintTime("Leiden Passes", result.passes);
    PrintTime("Leiden Iterations", result.iterations);
    
    // Get community mappings per pass (direct access to member variable)
    vector<vector<K>> communityMappingPerPass = x.communityMappingPerPass;
    
    PrintTime("Community Passes Stored", communityMappingPerPass.size());
    
    // Get degrees
    vector<K> degrees(num_nodes);
    #pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
        degrees[i] = g.out_degree(i);
    }
    
    // Initialize new_ids
    new_ids.resize(num_nodes);
    
    // Generate ordering based on flavor
    tm.Start();
    
    if (flavor == OrderingFlavor::LAST_PASS_DEG) {
        cout << "Ordering Flavor: LAST_PASS_DEG (baseline)" << endl;
        orderLastPassDeg(communityMappingPerPass, degrees, new_ids);
    } else if (flavor == OrderingFlavor::HYBRID_HUB_DFS) {
        cout << "Ordering Flavor: HYBRID_HUB_DFS" << endl;
        orderHybridHubDFS(communityMappingPerPass, degrees, new_ids);
    } else {
        // Build dendrogram for DFS/BFS flavors
        LeidenDendrogram dendro(num_nodes);
        dendro.buildFromLeiden(communityMappingPerPass, degrees);
        
        PrintTime("Dendrogram Nodes", dendro.nodes.size());
        PrintTime("Dendrogram Roots", dendro.roots.size());
        
        switch (flavor) {
            case OrderingFlavor::DFS_STANDARD:
                cout << "Ordering Flavor: DFS_STANDARD" << endl;
                orderDFS(dendro, new_ids, false, false);
                break;
                
            case OrderingFlavor::DFS_HUB_FIRST:
                cout << "Ordering Flavor: DFS_HUB_FIRST" << endl;
                orderDFS(dendro, new_ids, true, false);
                break;
                
            case OrderingFlavor::DFS_SIZE_FIRST:
                cout << "Ordering Flavor: DFS_SIZE_FIRST" << endl;
                orderDFS(dendro, new_ids, false, true);
                break;
                
            case OrderingFlavor::BFS_LEVEL:
                cout << "Ordering Flavor: BFS_LEVEL" << endl;
                orderBFS(dendro, new_ids);
                break;
                
            default:
                cout << "Unknown flavor, using DFS_HUB_FIRST" << endl;
                orderDFS(dendro, new_ids, true, false);
        }
    }
    
    tm.Stop();
    PrintTime("Ordering Time", tm.Seconds());
}

//==============================================================================
// VERIFICATION
//==============================================================================

bool VerifyMapping(const Graph& g, const pvector<NodeID>& mapping) {
    vector<bool> seen(g.num_nodes(), false);
    for (NodeID i = 0; i < g.num_nodes(); ++i) {
        if (mapping[i] < 0 || mapping[i] >= g.num_nodes()) {
            cout << "Invalid mapping: vertex " << i << " maps to " << mapping[i] << endl;
            return false;
        }
        if (seen[mapping[i]]) {
            cout << "Duplicate mapping: " << mapping[i] << endl;
            return false;
        }
        seen[mapping[i]] = true;
    }
    return true;
}

//==============================================================================
// PAGERANK KERNEL FOR BENCHMARKING
//==============================================================================

typedef float ScoreT;
const float kDamp = 0.85f;

pvector<ScoreT> PageRankPullGS(const Graph &g, int max_iters) {
    const ScoreT init_score = 1.0f / g.num_nodes();
    const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
    pvector<ScoreT> scores(g.num_nodes(), init_score);
    pvector<ScoreT> outgoing_contrib(g.num_nodes());
    
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
        outgoing_contrib[n] = init_score / g.out_degree(n);
    
    for (int iter = 0; iter < max_iters; iter++) {
        #pragma omp parallel for schedule(dynamic, 16384)
        for (NodeID u = 0; u < g.num_nodes(); u++) {
            ScoreT incoming_total = 0;
            for (NodeID v : g.in_neigh(u))
                incoming_total += outgoing_contrib[v];
            scores[u] = base_score + kDamp * incoming_total;
            outgoing_contrib[u] = scores[u] / g.out_degree(u);
        }
    }
    return scores;
}

//==============================================================================
// COMMAND LINE INTERFACE
//==============================================================================

class CLLeiden : public CLApp {
    double resolution_ = 1.0;
    int max_iterations_ = 20;
    int max_passes_ = 10;
    int flavor_ = 1;  // Default: DFS_HUB_FIRST
    
public:
    CLLeiden(int argc, char* argv[], string name) : CLApp(argc, argv, name) {
        get_args_ += "R:I:P:F:";
        AddHelpLine('R', "res", "Resolution parameter for Leiden", "1.0");
        AddHelpLine('I', "iter", "Max iterations per pass", "20");
        AddHelpLine('P', "pass", "Max passes", "10");
        AddHelpLine('F', "flavor", "Ordering: 0=DFS,1=DFS_HUB,2=DFS_SIZE,3=BFS,4=HYBRID,5=BASELINE", "1");
    }
    
    void HandleArg(signed char opt, char* opt_arg) override {
        switch (opt) {
            case 'R':
                resolution_ = atof(opt_arg);
                break;
            case 'I':
                max_iterations_ = atoi(opt_arg);
                break;
            case 'P':
                max_passes_ = atoi(opt_arg);
                break;
            case 'F':
                flavor_ = atoi(opt_arg);
                break;
            default:
                CLApp::HandleArg(opt, opt_arg);
        }
    }
    
    double resolution() const { return resolution_; }
    int max_iterations() const { return max_iterations_; }
    int max_passes() const { return max_passes_; }
    OrderingFlavor flavor() const { return static_cast<OrderingFlavor>(flavor_); }
};

//==============================================================================
// MAIN
//==============================================================================

int main(int argc, char* argv[]) {
    CLLeiden cli(argc, argv, "leiden-dendro");
    if (!cli.ParseArgs())
        return -1;
    
    cout << "========================================" << endl;
    cout << "Leiden Dendrogram-based Reordering" << endl;
    cout << "========================================" << endl;
    cout << "Resolution: " << cli.resolution() << endl;
    cout << "Max Iterations: " << cli.max_iterations() << endl;
    cout << "Max Passes: " << cli.max_passes() << endl;
    cout << "Flavor: " << static_cast<int>(cli.flavor()) << endl;
    cout << "  0=DFS, 1=DFS_HUB, 2=DFS_SIZE, 3=BFS, 4=HYBRID, 5=LAST_PASS_DEG" << endl;
    cout << "========================================" << endl;
    
    // Build graph
    Builder b(cli);
    Graph g = b.MakeGraph();
    
    cout << "Graph: " << (unsigned long)g.num_nodes() << " nodes, " << (unsigned long)g.num_edges() << " edges" << endl;
    
    // Generate mapping
    pvector<NodeID> new_ids;
    
    Timer total_tm;
    total_tm.Start();
    
    GenerateLeidenDendrogramMapping(g, new_ids, cli.resolution(),
                                    cli.max_iterations(), cli.max_passes(),
                                    cli.flavor());
    
    total_tm.Stop();
    PrintTime("Total Reordering Time", total_tm.Seconds());
    
    // Verify
    if (VerifyMapping(g, new_ids)) {
        cout << "Mapping verified: OK" << endl;
    } else {
        cout << "Mapping verification FAILED!" << endl;
        return -1;
    }
    
    // Benchmark: Compare original vs reordered graph
    if (cli.num_trials() > 0) {
        cout << "\n========================================" << endl;
        cout << "PageRank Benchmark (20 iterations)" << endl;
        cout << "========================================" << endl;
        
        const int pr_iters = 20;
        
        // Benchmark on original graph
        cout << "\nOriginal graph ordering:" << endl;
        double orig_total = 0.0;
        for (int trial = 0; trial < cli.num_trials(); ++trial) {
            Timer pr_tm;
            pr_tm.Start();
            auto scores = PageRankPullGS(g, pr_iters);
            pr_tm.Stop();
            PrintTime("Trial Time", pr_tm.Seconds());
            orig_total += pr_tm.Seconds();
        }
        PrintTime("Original Average", orig_total / cli.num_trials());
        
        // Note: To test reordered graph, use the pr binary with -r leiden
        cout << "\nTo benchmark reordered graph, use:" << endl;
        cout << "  ./bench/bin/pr -f <graph> -r leiden -n " << cli.num_trials() << endl;
    }
    
    // Output the mapping (optional - can be used with MAP reordering)
    cout << "\nFirst 10 vertex mappings:" << endl;
    NodeID limit = (g.num_nodes() < 10) ? g.num_nodes() : 10;
    for (NodeID i = 0; i < limit; ++i) {
        cout << "  " << i << " -> " << new_ids[i] << endl;
    }
    
    return 0;
}
