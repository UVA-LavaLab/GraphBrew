// ============================================================================
// GraphBrew — GOrder CSR-Native: Cache-Optimized Graph Ordering
// ============================================================================
//
// CSR-native variant for GOrder (-o 9:csr / -o 9:sym).  Faithful
// reimplementation of the GOrder greedy algorithm that operates directly
// on the CSR graph without intermediate adjacency format conversion.
//
// The original GOrder (Wei et al., SIGMOD 2016) maximizes a locality
// score S(v) for each candidate vertex v, measuring how many of v's
// neighbors (via 2-hop out→in paths) are already in a sliding window
// of the last W placed vertices.
//
// Two modes:
//   -o 9:csr  (default) — directed CSR, no symmetrization overhead
//   -o 9:sym            — symmetric CSR, matches GoGraph behavior
//
// Key improvements over the baseline GoGraph GOrder (Algorithm 9):
//
//   1. CSR-native — zero conversion overhead (speed ↑, memory ↓)
//   2. BFS-RCM pre-ordering directly on CSR graph
//   3. Parallel CSR construction for reordered graph (speed ↑)
//   4. Directed CSR uses native out/in edges (no symmetrization)
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
// BFS-RCM pre-ordering directly on CSR graph
// --------------------------------------------------------------------------
// Operates directly on the CSR graph using both out and in neighbors
// (when available) for undirected BFS traversal.
// Returns perm[old] = new, inv_perm[new] = old.
// --------------------------------------------------------------------------
template <typename NodeID_, typename DestID_, bool invert>
void bfs_rcm_csr(const CSRGraph<NodeID_, DestID_, invert>& g,
                 int n,
                 std::vector<int>& perm,
                 std::vector<int>& inv_perm) {
    // Total degree: out + in when inverted edges available
    auto total_degree = [&](int v) -> int64_t {
        int64_t d = g.out_degree(v);
        if constexpr (invert) d += g.in_degree(v);
        return d;
    };

    // Sort vertices by total degree ascending (seed order)
    std::vector<int> deg_sorted(n);
    std::iota(deg_sorted.begin(), deg_sorted.end(), 0);
    std::sort(deg_sorted.begin(), deg_sorted.end(), [&](int a, int b) {
        int64_t da = total_degree(a), db = total_degree(b);
        if (da != db) return da < db;
        return a < b;  // deterministic tie-breaking
    });

    // BFS from each unvisited vertex (in degree order)
    std::vector<bool> visited(n, false);
    std::vector<int> order;
    order.reserve(n);
    std::queue<int> que;
    std::vector<int> nbrs;

    for (int k = 0; k < n; ++k) {
        int start = deg_sorted[k];
        if (visited[start]) continue;

        que.push(start);
        visited[start] = true;
        order.push_back(start);

        while (!que.empty()) {
            int now = que.front();
            que.pop();

            // Collect neighbors (both directions for undirected BFS)
            nbrs.clear();
            for (DestID_ dest : g.out_neigh(now))
                nbrs.push_back(static_cast<int>(nbr_id<NodeID_>(dest)));
            if constexpr (invert) {
                for (DestID_ dest : g.in_neigh(now))
                    nbrs.push_back(static_cast<int>(nbr_id<NodeID_>(dest)));
                // Deduplicate (out and in may overlap)
                std::sort(nbrs.begin(), nbrs.end());
                nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
            }

            // Sort by degree ascending (CM ordering), deterministic tie-break
            std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
                int64_t da = total_degree(a), db = total_degree(b);
                if (da != db) return da < db;
                return a < b;
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
// Build directed CSR in reordered space (default, -o 9:csr)
// --------------------------------------------------------------------------
// Builds separate out-edge and in-edge flat CSR arrays using the
// permutation from BFS-RCM.  Uses native directed edges -- no
// symmetrization overhead.
// --------------------------------------------------------------------------
template <typename NodeID_, typename DestID_, bool invert>
void build_directed_csr(const CSRGraph<NodeID_, DestID_, invert>& g,
                        int n,
                        const std::vector<int>& perm,
                        std::vector<int>& out_offset,
                        std::vector<int>& out_edges,
                        std::vector<int>& in_offset,
                        std::vector<int>& in_edges) {
    // --- Out-edges ---
    out_offset.assign(n + 1, 0);
    for (int u = 0; u < n; ++u)
        out_offset[perm[u] + 1] = static_cast<int>(g.out_degree(u));
    for (int i = 0; i < n; ++i)
        out_offset[i + 1] += out_offset[i];

    out_edges.resize(out_offset[n]);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int u = 0; u < n; ++u) {
        int u_new = perm[u];
        int pos = out_offset[u_new];
        for (DestID_ dest : g.out_neigh(u))
            out_edges[pos++] = perm[static_cast<int>(nbr_id<NodeID_>(dest))];
        std::sort(out_edges.data() + out_offset[u_new],
                  out_edges.data() + out_offset[u_new + 1]);
    }

    // --- In-edges ---
    if constexpr (invert) {
        // Graph has inverse edges -- fill directly (parallel)
        in_offset.assign(n + 1, 0);
        for (int v = 0; v < n; ++v)
            in_offset[perm[v] + 1] = static_cast<int>(g.in_degree(v));
        for (int i = 0; i < n; ++i)
            in_offset[i + 1] += in_offset[i];

        in_edges.resize(in_offset[n]);

        #pragma omp parallel for schedule(dynamic, 64)
        for (int v = 0; v < n; ++v) {
            int v_new = perm[v];
            int pos = in_offset[v_new];
            for (DestID_ dest : g.in_neigh(v))
                in_edges[pos++] = perm[static_cast<int>(nbr_id<NodeID_>(dest))];
            std::sort(in_edges.data() + in_offset[v_new],
                      in_edges.data() + in_offset[v_new + 1]);
        }
    } else {
        // No inverse edges -- derive in-edges from out-edges
        in_offset.assign(n + 1, 0);
        for (int u = 0; u < n; ++u)
            for (DestID_ dest : g.out_neigh(u))
                in_offset[perm[static_cast<int>(nbr_id<NodeID_>(dest))] + 1]++;
        for (int i = 0; i < n; ++i)
            in_offset[i + 1] += in_offset[i];

        in_edges.resize(in_offset[n]);
        std::vector<int> in_pos(n);
        for (int i = 0; i < n; ++i) in_pos[i] = in_offset[i];

        for (int u = 0; u < n; ++u) {
            int u_new = perm[u];
            for (DestID_ dest : g.out_neigh(u)) {
                int v_new = perm[static_cast<int>(nbr_id<NodeID_>(dest))];
                in_edges[in_pos[v_new]++] = u_new;
            }
        }

        #pragma omp parallel for schedule(dynamic, 64)
        for (int v = 0; v < n; ++v)
            std::sort(in_edges.data() + in_offset[v],
                      in_edges.data() + in_offset[v + 1]);
    }
}

// --------------------------------------------------------------------------
// Build symmetric CSR in reordered space (for -o 9:sym)
// --------------------------------------------------------------------------
// For each directed edge u->v, adds both (u_new,v_new) and (v_new,u_new).
// Result is sorted and deduplicated per vertex.
// --------------------------------------------------------------------------
template <typename NodeID_, typename DestID_, bool invert>
void build_symmetric_csr(const CSRGraph<NodeID_, DestID_, invert>& g,
                         int n,
                         const std::vector<int>& perm,
                         std::vector<int>& sym_offset,
                         std::vector<int>& sym_edges) {
    // Build adjacency lists in reordered space
    std::vector<std::vector<int>> adj(n);
    for (int u = 0; u < n; ++u) {
        int u_new = perm[u];
        for (DestID_ dest : g.out_neigh(u)) {
            int v = static_cast<int>(nbr_id<NodeID_>(dest));
            if (u != v) {
                int v_new = perm[v];
                adj[u_new].push_back(v_new);
                adj[v_new].push_back(u_new);
            }
        }
    }

    // Sort and deduplicate (parallel)
    #pragma omp parallel for schedule(dynamic, 64)
    for (int v = 0; v < n; ++v) {
        std::sort(adj[v].begin(), adj[v].end());
        adj[v].erase(std::unique(adj[v].begin(), adj[v].end()), adj[v].end());
    }

    // Flatten to CSR
    sym_offset.resize(n + 1, 0);
    for (int v = 0; v < n; ++v)
        sym_offset[v + 1] = sym_offset[v] + static_cast<int>(adj[v].size());

    sym_edges.resize(sym_offset[n]);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int v = 0; v < n; ++v)
        std::copy(adj[v].begin(), adj[v].end(),
                  sym_edges.data() + sym_offset[v]);
}

// --------------------------------------------------------------------------
// GOrder Greedy -- Core algorithm (operates on directed or symmetric CSR)
// --------------------------------------------------------------------------
// Takes separate out-edge and in-edge CSR arrays.  For symmetric mode,
// both out and in parameters reference the same arrays.
//
// For each placed vertex v, score updates go to:
//   Out-neighbors of v: 1-hop direct contribution
//   In-neighbors of v: each in-neighbor u gets +1, then each
//     out-neighbor w of u gets +1 (2-hop path: v <- u -> w)
//
// Hub vertices (outdeg > sqrt(n)) are skipped in 2-hop expansion.
// --------------------------------------------------------------------------
inline void gorder_greedy(
        int n,
        int window,
        const std::vector<int>& out_offset,
        const std::vector<int>& out_edges,
        const std::vector<int>& in_offset,
        const std::vector<int>& in_edges,
        std::vector<int>& result_order) {

    const int hugevertex = static_cast<int>(std::sqrt(static_cast<double>(n)));

    auto outdeg = [&](int v) -> int { return out_offset[v+1] - out_offset[v]; };
    auto indeg  = [&](int v) -> int { return in_offset[v+1] - in_offset[v]; };

    // Initialize UnitHeap with indegree as initial key
    UnitHeap heap(n);
    for (int i = 0; i < n; ++i) {
        int id = indeg(i);
        heap.list[i].key = id;
        heap.update[i] = -id;
    }
    heap.ReConstruct();

    // Collect zero-degree vertices and find max-indegree vertex
    std::vector<int> zero;
    zero.reserve(std::min(n, 10000));

    int max_indeg_vertex = 0;
    int max_id = -1;
    for (int i = 0; i < n; ++i) {
        int id = indeg(i);
        int od = outdeg(i);
        if (id > max_id) {
            max_id = id;
            max_indeg_vertex = i;
        } else if (id + od == 0) {
            heap.update[i] = INT_MAX / 2;
            zero.push_back(i);
            heap.DeleteElement(i);
        }
    }

    auto score_inc = [&](int w) {
        if (heap.update[w] == 0) heap.IncrementKey(w);
        else                     heap.update[w]++;
    };
    auto score_dec = [&](int w) {
        heap.update[w]--;
    };

    // Start with max-indegree vertex
    std::vector<int> order;
    order.reserve(n);
    order.push_back(max_indeg_vertex);
    heap.update[max_indeg_vertex] = INT_MAX / 2;
    heap.DeleteElement(max_indeg_vertex);

    // --- Initial push: score update for first placed vertex ---
    {
        int v0 = max_indeg_vertex;

        // In-neighbors of v0 (u -> v0)
        for (int i = in_offset[v0]; i < in_offset[v0 + 1]; ++i) {
            int u = in_edges[i];
            if (outdeg(u) <= hugevertex) {
                score_inc(u);
                if (outdeg(u) > 1) {
                    for (int j = out_offset[u]; j < out_offset[u + 1]; ++j)
                        score_inc(out_edges[j]);
                }
            }
        }

        // Out-neighbors of v0 (v0 -> w)
        if (outdeg(v0) <= hugevertex) {
            for (int i = out_offset[v0]; i < out_offset[v0 + 1]; ++i)
                score_inc(out_edges[i]);
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
            // Decrement for out-neighbors of popv (1-hop)
            if (outdeg(popv) <= hugevertex) {
                for (int i = out_offset[popv]; i < out_offset[popv + 1]; ++i)
                    score_dec(out_edges[i]);
            }

            // In-neighbors of popv: decrement + 2-hop via out-edges
            for (int i = in_offset[popv]; i < in_offset[popv + 1]; ++i) {
                int u = in_edges[i];
                if (outdeg(u) <= hugevertex) {
                    score_dec(u);
                    if (outdeg(u) > 1) {
                        bool found = std::binary_search(
                            out_edges.data() + out_offset[u],
                            out_edges.data() + out_offset[u + 1], v);
                        if (!found) {
                            for (int j = out_offset[u]; j < out_offset[u + 1]; ++j)
                                score_dec(out_edges[j]);
                        } else {
                            popvexist[u] = true;
                        }
                    }
                }
            }
        }

        // --- Push phase: add current vertex v to window ---
        // Out-neighbors of v (1-hop)
        if (outdeg(v) <= hugevertex) {
            for (int i = out_offset[v]; i < out_offset[v + 1]; ++i)
                score_inc(out_edges[i]);
        }

        // In-neighbors of v: 2-hop via out-edges
        for (int i = in_offset[v]; i < in_offset[v + 1]; ++i) {
            int u = in_edges[i];
            if (outdeg(u) <= hugevertex) {
                score_inc(u);
                if (!popvexist[u]) {
                    if (outdeg(u) > 1) {
                        for (int j = out_offset[u]; j < out_offset[u + 1]; ++j)
                            score_inc(out_edges[j]);
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

    result_order.resize(n);
    for (int i = 0; i < n; ++i) {
        result_order[order[i]] = i;
    }
}

} // namespace gorder_csr_detail

// ============================================================================
// Public API -- GenerateGOrderCSRMapping
// ============================================================================
// GOrder CSR variant with two modes:
//   -o 9:csr  (default) -- directed CSR, no symmetrization overhead
//   -o 9:sym            -- symmetric CSR, matches GoGraph behavior
// ============================================================================

template <typename NodeID_, typename DestID_, typename WeightT_, bool invert>
void GenerateGOrderCSRMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                              pvector<NodeID_>& new_ids,
                              const std::string& /*filename*/,
                              bool symmetric = false,
                              int window = 7) {
    Timer tm;
    const int n = static_cast<int>(g.num_nodes());

    if (n == 0) return;

    if (static_cast<int64_t>(new_ids.size()) < n)
        new_ids.resize(n);

    // --- Step 1: BFS-RCM pre-ordering directly on CSR graph ---
    tm.Start();
    std::vector<int> perm(n), inv_perm(n);
    gorder_csr_detail::bfs_rcm_csr(g, n, perm, inv_perm);
    tm.Stop();
    PrintTime("GOrder_CSR RCM pre-order", tm.Seconds());

    // --- Step 2 & 3: Build reordered CSR and run greedy ---
    std::vector<int> greedy_order;

    if (symmetric) {
        // Symmetric mode: union of out+in edges (matches GoGraph)
        std::vector<int> sym_offset, sym_edges;
        tm.Start();
        gorder_csr_detail::build_symmetric_csr(g, n, perm, sym_offset, sym_edges);
        tm.Stop();
        PrintTime("GOrder_CSR build symmetric CSR", tm.Seconds());

        tm.Start();
        gorder_csr_detail::gorder_greedy(
            n, window, sym_offset, sym_edges, sym_offset, sym_edges, greedy_order);
        tm.Stop();
        PrintTime("GOrder_CSR greedy (sym)", tm.Seconds());
    } else {
        // Directed mode (default): separate out/in edges, no symmetrization
        std::vector<int> out_offset, out_edges, in_offset, in_edges;
        tm.Start();
        gorder_csr_detail::build_directed_csr(
            g, n, perm, out_offset, out_edges, in_offset, in_edges);
        tm.Stop();
        PrintTime("GOrder_CSR build directed CSR", tm.Seconds());

        tm.Start();
        gorder_csr_detail::gorder_greedy(
            n, window, out_offset, out_edges, in_offset, in_edges, greedy_order);
        tm.Stop();
        PrintTime("GOrder_CSR greedy", tm.Seconds());
    }

    // --- Step 4: Compose permutations ---
    // greedy_order maps reordered IDs -> final positions
    // perm maps original IDs -> reordered IDs
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
