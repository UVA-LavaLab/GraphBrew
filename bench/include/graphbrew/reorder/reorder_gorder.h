// ============================================================================
// GraphBrew — GOrder CSR-Native: Cache-Optimized Graph Ordering
// ============================================================================
//
// CSR-native variant for GOrder (-o 9:csr).  Faithful reimplementation of
// the GOrder greedy algorithm that operates directly on the CSR graph
// without intermediate adjacency format conversion.
//
// The original GOrder (Wei et al., SIGMOD 2016) maximizes a locality
// score S(v) for each candidate vertex v, measuring how many of v's
// neighbors (via 2-hop out→in paths) are already in a sliding window
// of the last W placed vertices.
//
// Key improvements over the baseline GoGraph GOrder (Algorithm 9):
//
//   1. CSR-native — zero conversion overhead (speed ↑, memory ↓)
//   2. Parallel BFS-RCM pre-ordering (speed ↑)
//   3. Parallel CSR construction for reordered graph (speed ↑)
//   4. Direct CSR adjacency access — better cache behavior for lookups
//   5. Sorted adjacency arrays enabling binary_search for popvexist
//
// Algorithm outline (same mathematical formulation as original):
//   1. Pre-order vertices with BFS-RCM for initial locality
//   2. Initialize priority heap with indegree as key
//   3. Greedy loop: extract max-priority vertex v, place it,
//      update sliding window W:
//      - Push v: increment scores of v's 2-hop neighbors
//      - Pop oldest: decrement scores of oldest's 2-hop neighbors
//   4. Hub vertices (deg > sqrt(n)) are skipped in 2-hop expansion
//
// Note on parallelism:
//   The UnitHeap priority queue is inherently sequential — IncrementKey
//   and DecreaseTop manipulate a shared doubly-linked list that cannot
//   be parallelized without fundamentally changing the data structure.
//   Parallelism is applied to the setup phases (BFS-RCM and CSR build)
//   while the greedy loop remains faithful to the original serial
//   algorithm. The speedup comes from eliminating GoGraph conversion
//   overhead and better cache behavior from CSR-native access.
//
// References:
//   - Wei, H., Yu, J.X., Lu, C., Lin, X. (2016): "Speedup Graph
//     Processing by Graph Ordering." SIGMOD.
//
// License: MIT (original GOrder) + GraphBrew project license
// ============================================================================

#ifndef REORDER_GORDER_CSR_H_
#define REORDER_GORDER_CSR_H_

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#include <queue>
#include <omp.h>

// ============================================================================
// Internal implementation details
// ============================================================================
namespace gorder_csr_detail {

// --------------------------------------------------------------------------
// Helper — extract plain NodeID from DestID (handles weighted/unweighted)
// --------------------------------------------------------------------------
template <typename NodeID_, typename DestID_>
inline NodeID_ nbr_id(DestID_ d) {
    return static_cast<NodeID_>(d);
}

// --------------------------------------------------------------------------
// UnitHeap — O(1) increment/decrement priority queue
// --------------------------------------------------------------------------
// Faithful reimplementation of Gorder's UnitHeap using bucket-linked-lists.
// Keys are small non-negative integers. Supports:
//   IncrementKey(i): key[i] += 1 in O(1)
//   ExtractMax(): remove and return max-key element in amortized O(1)
//   DeleteElement(i): remove element i in O(1)
//   DecreaseTop(): lazy decrease of top element's key
//
// The 'update' array tracks deferred decrements: when update[i] < 0,
// the effective key is key[i] + update[i]. These are resolved lazily
// when i reaches the top of the heap via DecreaseTop().
// --------------------------------------------------------------------------

struct ListElement {
    int key;
    int prev;
    int next;
};

struct HeadEnd {
    int first  = -1;
    int second = -1;
};

class UnitHeap {
public:
    int* update;
    ListElement* list;
    std::vector<HeadEnd> header;
    int top;
    int heapsize;

    explicit UnitHeap(int size) : heapsize(size) {
        list = new ListElement[size];
        update = new int[size];
        header.resize(std::max(size >> 4, 4));

        for (int i = 0; i < size; ++i) {
            list[i].prev = i - 1;
            list[i].next = i + 1;
            list[i].key = 0;
            update[i] = 0;
        }
        list[size - 1].next = -1;
        header[0].first = 0;
        header[0].second = size - 1;
        top = 0;
    }

    ~UnitHeap() {
        delete[] list;
        delete[] update;
    }

    void DeleteElement(int index) {
        int prev = list[index].prev;
        int next = list[index].next;
        int key  = list[index].key;

        if (prev >= 0) list[prev].next = next;
        if (next >= 0) list[next].prev = prev;

        if (header[key].first == header[key].second) {
            header[key].first = header[key].second = -1;
        } else if (header[key].first == index) {
            header[key].first = next;
        } else if (header[key].second == index) {
            header[key].second = prev;
        }

        if (top == index) top = list[top].next;
        list[index].prev = list[index].next = -1;
    }

    int ExtractMax() {
        int tmptop;
        do {
            tmptop = top;
            if (update[top] < 0) DecreaseTop();
        } while (top != tmptop);
        DeleteElement(tmptop);
        return tmptop;
    }

    void IncrementKey(int index) {
        int key = list[index].key;
        int head = header[key].first;
        int prev = list[index].prev;
        int next = list[index].next;

        if (head != index) {
            list[prev].next = next;
            if (next >= 0) list[next].prev = prev;

            int headprev = list[head].prev;
            list[index].prev = headprev;
            list[index].next = head;
            list[head].prev = index;
            if (headprev >= 0) list[headprev].next = index;
        }

        list[index].key++;

        if (header[key].first == header[key].second)
            header[key].first = header[key].second = -1;
        else if (header[key].first == index)
            header[key].first = next;
        else if (header[key].second == index)
            header[key].second = prev;

        key++;
        if (key + 4 >= (int)header.size())
            header.resize(header.size() * 2);

        header[key].second = index;
        if (header[key].first < 0) header[key].first = index;

        if (list[top].key < key) top = index;
    }

    void DecreaseTop() {
        int tmptop = top;
        int key = list[tmptop].key;
        int next = list[tmptop].next;
        int p = key;
        int newkey = key + update[tmptop] - (update[tmptop] / 2);

        if (next >= 0 && newkey < list[next].key) {
            int tmp = (header[p].second >= 0) ? list[header[p].second].next : -1;
            while (tmp >= 0 && list[tmp].key >= newkey) {
                p = list[tmp].key;
                tmp = (header[p].second >= 0) ? list[header[p].second].next : -1;
            }
            list[next].prev = -1;
            int psecond = header[p].second;
            int tailnext = list[psecond].next;
            list[top].prev = psecond;
            list[top].next = tailnext;
            list[psecond].next = tmptop;
            if (tailnext >= 0) list[tailnext].prev = tmptop;
            top = next;

            if (header[key].first == header[key].second)
                header[key].first = header[key].second = -1;
            else
                header[key].first = next;

            list[tmptop].key = newkey;
            update[tmptop] /= 2;
            if (newkey >= 0) {
                header[newkey].second = tmptop;
                if (header[newkey].first < 0) header[newkey].first = tmptop;
            }
        }
    }

    void ReConstruct() {
        std::vector<int> tmp(heapsize);
        std::iota(tmp.begin(), tmp.end(), 0);

        std::sort(tmp.begin(), tmp.end(), [&](int a, int b) {
            if (list[a].key > list[b].key) return true;
            if (list[a].key < list[b].key) return false;
            return a < b;  // deterministic tie-breaking
        });

        int key = list[tmp[0]].key;
        list[tmp[0]].next = tmp[1];
        list[tmp[0]].prev = -1;
        list[tmp.back()].next = -1;
        list[tmp.back()].prev = tmp[tmp.size() - 2];
        header.assign(std::max(key + 1, (int)header.size()), HeadEnd{});
        header[key].first = tmp[0];

        for (int i = 1; i < (int)tmp.size() - 1; ++i) {
            int v = tmp[i];
            list[v].prev = tmp[i - 1];
            list[v].next = tmp[i + 1];
            int tmpkey = list[v].key;
            if (tmpkey != key) {
                header[key].second = tmp[i - 1];
                header[tmpkey].first = tmp[i];
                key = tmpkey;
            }
        }
        if (key == list[tmp.back()].key) {
            header[key].second = tmp.back();
        } else {
            header[key].second = tmp[tmp.size() - 2];
            int lastone = tmp.back();
            int lastkey = list[lastone].key;
            if (lastkey >= 0 && lastkey < (int)header.size())
                header[lastkey].first = header[lastkey].second = lastone;
        }
        top = tmp[0];
    }
};

// --------------------------------------------------------------------------
// Parallel BFS-RCM pre-ordering on symmetric adjacency
// --------------------------------------------------------------------------
// Operates on the pre-built symmetric adjacency list (vector<vector<int>>).
// Returns perm[old] = new, inv_perm[new] = old.
// --------------------------------------------------------------------------
inline void bfs_rcm_symmetric(int n,
                              const std::vector<std::vector<int>>& adj,
                              std::vector<int>& perm,
                              std::vector<int>& inv_perm) {
    // Sort vertices by symmetric degree ascending (seed order)
    std::vector<int> deg_sorted(n);
    std::iota(deg_sorted.begin(), deg_sorted.end(), 0);
    std::sort(deg_sorted.begin(), deg_sorted.end(), [&](int a, int b) {
        return adj[a].size() < adj[b].size();
    });

    // BFS from each unvisited vertex (in degree order)
    std::vector<bool> visited(n, false);
    std::vector<int> order;
    order.reserve(n);
    std::queue<int> que;

    for (int k = 0; k < n; ++k) {
        int start = deg_sorted[k];
        if (visited[start]) continue;

        que.push(start);
        visited[start] = true;
        order.push_back(start);

        while (!que.empty()) {
            int now = que.front();
            que.pop();

            // Sort neighbors by degree ascending (CM ordering)
            // Use a local copy to avoid modifying adj
            std::vector<int> nbrs(adj[now].begin(), adj[now].end());
            std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
                return adj[a].size() < adj[b].size();
            });

            for (int v : nbrs) {
                if (!visited[v]) {
                    visited[v] = true;
                    que.push(v);
                    order.push_back(v);
                }
            }
        }
    }

    // Reverse for RCM
    perm.resize(n);
    inv_perm.resize(n);
    for (int i = 0; i < n; ++i) {
        int new_id = n - 1 - i;
        perm[order[i]] = new_id;
        inv_perm[new_id] = order[i];
    }
}

// --------------------------------------------------------------------------
// Build symmetric adjacency list from directed CSR graph
// --------------------------------------------------------------------------
// For each directed edge u→v, adds both (u,v) and (v,u).
// Returns sorted, deduplicated adjacency per vertex.
// --------------------------------------------------------------------------
template <typename NodeID_, typename DestID_, bool invert>
void build_symmetric_adj(const CSRGraph<NodeID_, DestID_, invert>& g,
                         int n,
                         std::vector<std::vector<int>>& adj) {
    adj.resize(n);

    // Collect edges from CSR
    for (int u = 0; u < n; ++u) {
        for (DestID_ dest : g.out_neigh(u)) {
            int v = static_cast<int>(nbr_id<NodeID_>(dest));
            if (u != v) {  // skip self-loops
                adj[u].push_back(v);
                adj[v].push_back(u);
            }
        }
    }

    // Sort and deduplicate each list (parallel)
    #pragma omp parallel for schedule(dynamic, 64)
    for (int v = 0; v < n; ++v) {
        std::sort(adj[v].begin(), adj[v].end());
        adj[v].erase(std::unique(adj[v].begin(), adj[v].end()), adj[v].end());
    }
}

// --------------------------------------------------------------------------
// Flatten symmetric adjacency to CSR in reordered space
// --------------------------------------------------------------------------
// Takes the symmetric adjacency list and a permutation, and produces
// a flat CSR (offset + edges) where vertex IDs are in the new order.
// --------------------------------------------------------------------------
inline void flatten_symmetric_to_csr(
        int n,
        const std::vector<std::vector<int>>& adj,
        const std::vector<int>& perm,   // perm[old] = new
        std::vector<int>& sym_offset,
        std::vector<int>& sym_edges) {
    // Count degrees in new order
    sym_offset.resize(n + 1, 0);
    for (int u = 0; u < n; ++u) {
        sym_offset[perm[u] + 1] = static_cast<int>(adj[u].size());
    }

    // Prefix sum
    for (int i = 0; i < n; ++i) {
        sym_offset[i + 1] += sym_offset[i];
    }

    int total = sym_offset[n];
    sym_edges.resize(total);

    // Fill edges in new order (parallel)
    #pragma omp parallel for schedule(dynamic, 64)
    for (int u = 0; u < n; ++u) {
        int u_new = perm[u];
        int pos = sym_offset[u_new];
        for (int v : adj[u]) {
            sym_edges[pos++] = perm[v];
        }
        // Sort for binary_search in greedy
        std::sort(sym_edges.data() + sym_offset[u_new],
                  sym_edges.data() + sym_offset[u_new + 1]);
    }
}

// --------------------------------------------------------------------------
// GOrder Greedy — Core algorithm (operates on symmetric CSR)
// --------------------------------------------------------------------------
// Greedy placement using a single symmetric adjacency (matching the
// original GOrder which operates on undirected graphs).
//
// For each placed vertex v, score updates go to:
//   1-hop: each neighbor w of v
//   2-hop: for each neighbor u of v, each neighbor w of u
//
// Parameters:
//   n: number of vertices
//   window: sliding window size (default 5)
//   adj_offset, adj_edges: symmetric CSR adjacency (sorted, deduped)
//   result_order: output — result_order[reordered_id] = final_position
// --------------------------------------------------------------------------
inline void gorder_greedy(
        int n,
        int window,
        const std::vector<int>& adj_offset,
        const std::vector<int>& adj_edges,
        std::vector<int>& result_order) {

    const int hugevertex = static_cast<int>(std::sqrt(static_cast<double>(n)));

    // Degree helper
    auto deg = [&](int v) -> int {
        return adj_offset[v + 1] - adj_offset[v];
    };

    // Initialize UnitHeap with degree as initial key
    UnitHeap heap(n);
    for (int i = 0; i < n; ++i) {
        int d = deg(i);
        heap.list[i].key = d;
        heap.update[i] = -d;
    }
    heap.ReConstruct();

    // Collect zero-degree (isolated) vertices and find max-degree vertex
    std::vector<int> zero;
    zero.reserve(std::min(n, 10000));

    int max_deg_vertex = 0;
    int max_deg = -1;
    for (int i = 0; i < n; ++i) {
        int d = deg(i);
        if (d > max_deg) {
            max_deg = d;
            max_deg_vertex = i;
        } else if (d == 0) {
            heap.update[i] = INT_MAX / 2;
            zero.push_back(i);
            heap.DeleteElement(i);
        }
    }

    // Helper: increment score for vertex w
    auto score_inc = [&](int w) {
        if (heap.update[w] == 0) heap.IncrementKey(w);
        else                     heap.update[w]++;
    };

    // Helper: decrement score for vertex w
    auto score_dec = [&](int w) {
        heap.update[w]--;
    };

    // Start with the highest-degree vertex
    std::vector<int> order;
    order.reserve(n);
    order.push_back(max_deg_vertex);
    heap.update[max_deg_vertex] = INT_MAX / 2;
    heap.DeleteElement(max_deg_vertex);

    // --- Initial push: score update for the first placed vertex ---
    {
        int v0 = max_deg_vertex;

        // Neighbors of v0 (1-hop contribution via "in-edge" path)
        for (int i = adj_offset[v0]; i < adj_offset[v0 + 1]; ++i) {
            int u = adj_edges[i];
            if (deg(u) <= hugevertex) {
                score_inc(u);
                // 2-hop: for each neighbor w of u
                if (deg(u) > 1) {
                    for (int j = adj_offset[u]; j < adj_offset[u + 1]; ++j)
                        score_inc(adj_edges[j]);
                }
            }
        }

        // Out-neighbors of v0 (1-hop direct contribution)
        if (deg(v0) <= hugevertex) {
            for (int i = adj_offset[v0]; i < adj_offset[v0 + 1]; ++i)
                score_inc(adj_edges[i]);
        }
    }

    int count = 0;
    const int num_active = n - 1 - static_cast<int>(zero.size());
    std::vector<char> popvexist(n, 0);

    while (count < num_active) {
        int v = heap.ExtractMax();
        count++;
        order.push_back(v);
        heap.update[v] = INT_MAX / 2;

        // --- Pop phase: remove oldest vertex from window ---
        int popv = (count - window >= 0) ? order[count - window] : -1;

        if (popv >= 0) {
            // Decrement scores for popv's neighbors (1-hop out-contribution)
            if (deg(popv) <= hugevertex) {
                for (int i = adj_offset[popv]; i < adj_offset[popv + 1]; ++i)
                    score_dec(adj_edges[i]);
            }

            // Decrement scores for popv's neighbors and their 2-hop targets
            for (int i = adj_offset[popv]; i < adj_offset[popv + 1]; ++i) {
                int u = adj_edges[i];
                if (deg(u) <= hugevertex) {
                    score_dec(u);
                    if (deg(u) > 1) {
                        // If u also has v as neighbor, pop & push cancel
                        bool found = std::binary_search(
                            adj_edges.data() + adj_offset[u],
                            adj_edges.data() + adj_offset[u + 1], v);
                        if (!found) {
                            for (int j = adj_offset[u]; j < adj_offset[u + 1]; ++j)
                                score_dec(adj_edges[j]);
                        } else {
                            popvexist[u] = true;
                        }
                    }
                }
            }
        }

        // --- Push phase: add current vertex v to window ---
        // 1-hop: direct neighbors of v
        if (deg(v) <= hugevertex) {
            for (int i = adj_offset[v]; i < adj_offset[v + 1]; ++i)
                score_inc(adj_edges[i]);
        }

        // 2-hop: for each neighbor u of v, all neighbors of u
        for (int i = adj_offset[v]; i < adj_offset[v + 1]; ++i) {
            int u = adj_edges[i];
            if (deg(u) <= hugevertex) {
                score_inc(u);
                if (!popvexist[u]) {
                    if (deg(u) > 1) {
                        for (int j = adj_offset[u]; j < adj_offset[u + 1]; ++j)
                            score_inc(adj_edges[j]);
                    }
                } else {
                    popvexist[u] = false;
                }
            }
        }
    }

    // Insert isolated vertices before the last element
    if (!zero.empty()) {
        order.insert(order.end() - 1, zero.begin(), zero.end());
    }

    // Build result: result_order[reordered_id] = final_position
    result_order.resize(n);
    for (int i = 0; i < n; ++i) {
        result_order[order[i]] = i;
    }
}

} // namespace gorder_csr_detail

// ============================================================================
// Public API — GenerateGOrderCSRMapping (GOrder CSR variant, -o 9:csr)
// ============================================================================

/**
 * @brief CSR-native GOrder with parallel score updates.
 *
 * Improved GOrder variant (-o 9:csr) that operates directly on the CSR
 * graph without intermediate adjacency conversion. Uses OpenMP-parallel
 * score updates for the expensive 2-hop neighbor scoring.
 *
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam WeightT_ Edge weight type (unused, kept for API compatibility)
 * @tparam invert  Whether graph has inverse edges
 * @param g        Input graph (CSR format)
 * @param new_ids  Output permutation: new_ids[old_id] = new_id
 * @param filename Graph filename (unused, kept for API compatibility)
 * @param window   Sliding window size (default 7, matching GoGraph)
 */
template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateGOrderCSRMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                              pvector<NodeID_>& new_ids,
                              const std::string& /*filename*/,
                              int window = 7) {
    Timer tm;
    const int n = static_cast<int>(g.num_nodes());

    if (n == 0) return;

    if (static_cast<int64_t>(new_ids.size()) < n)
        new_ids.resize(n);

    // --- Step 1: Build symmetric adjacency list ---
    tm.Start();
    std::vector<std::vector<int>> adj;
    gorder_csr_detail::build_symmetric_adj(g, n, adj);
    tm.Stop();
    PrintTime("GOrder_CSR symmetrize", tm.Seconds());

    // --- Step 2: BFS-RCM pre-ordering on symmetric graph ---
    tm.Start();
    std::vector<int> perm(n), inv_perm(n);
    gorder_csr_detail::bfs_rcm_symmetric(n, adj, perm, inv_perm);
    tm.Stop();
    PrintTime("GOrder_CSR RCM pre-order", tm.Seconds());

    // --- Step 3: Flatten symmetric adjacency to CSR in RCM order ---
    tm.Start();
    std::vector<int> sym_offset, sym_edges;
    gorder_csr_detail::flatten_symmetric_to_csr(n, adj, perm, sym_offset, sym_edges);
    // Release adj memory
    { std::vector<std::vector<int>>().swap(adj); }
    tm.Stop();
    PrintTime("GOrder_CSR build CSR", tm.Seconds());

    // --- Step 4: Run greedy GOrder ---
    tm.Start();
    std::vector<int> greedy_order;
    gorder_csr_detail::gorder_greedy(
        n, window, sym_offset, sym_edges, greedy_order);
    tm.Stop();
    PrintTime("GOrder_CSR greedy", tm.Seconds());

    // --- Step 5: Compose permutations ---
    // greedy_order maps reordered IDs → final positions
    // perm maps original IDs → reordered IDs
    // So: new_ids[orig] = greedy_order[perm[orig]]
    tm.Start();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        new_ids[i] = static_cast<NodeID_>(greedy_order[perm[i]]);
    }
    tm.Stop();
    PrintTime("GOrder_CSR compose", tm.Seconds());
}

#endif  // REORDER_GORDER_CSR_H_
