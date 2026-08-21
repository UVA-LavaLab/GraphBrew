#ifndef GRAPHBREW_REORDER_GRAPHBREW_DIAGNOSTICS_H_
#define GRAPHBREW_REORDER_GRAPHBREW_DIAGNOSTICS_H_

// Internal GraphBrew detail header. Include only from reorder_graphbrew.h.

// SECTION 16b: ORDERING - CONNECTIVITY-BASED (Boost-style for Leiden)
//=============================================================================

/**
 * Connectivity-based ordering within communities (Boost-style for Leiden)
 *
 * Uses BFS traversal of the original graph within each community
 * to produce a vertex ordering that reflects actual connectivity patterns,
 * similar to how RabbitOrder achieves locality benefits.
 *
 * Algorithm (two-phase like Boost):
 * Phase 1: For each community, BFS from highest-degree vertex to assign local IDs
 * Phase 2: Prefix sum to get global offsets, then add to local IDs
 *
 * Zero-degree nodes grouped at END for cache locality.
 */
template <typename K, typename NodeID_T, typename DestID_T>
void orderConnectivityBFS(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderConnectivityBFS: N=%zu", N);

    // Step 1: Build community vertex lists with isolated separation
    // Count vertices per community and find max community ID
    K maxComm = 0;
    for (size_t v = 0; v < N; ++v) {
        maxComm = std::max(maxComm, membership[v]);
    }
    const size_t numComm = static_cast<size_t>(maxComm) + 1;

    // Separate isolated (zero-degree) vertices
    std::vector<std::vector<K>> commVertices(numComm);
    std::vector<K> isolated;

    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(static_cast<K>(v));
        } else {
            commVertices[membership[v]].push_back(static_cast<K>(v));
        }
    }

    // Step 2: Sort toplevel communities by size for better cache behavior
    std::vector<K> commOrder(numComm);
    std::iota(commOrder.begin(), commOrder.end(), K(0));
    std::sort(commOrder.begin(), commOrder.end(), [&](K a, K b) {
        return commVertices[a].size() > commVertices[b].size();  // Large communities first
    });

    // Step 3: Compute offsets (prefix sum)
    std::vector<size_t> offsets(numComm + 1, 0);
    for (size_t i = 0; i < numComm; ++i) {
        K c = commOrder[i];
        offsets[i + 1] = offsets[i] + commVertices[c].size();
    }

    // Create reverse mapping: community -> sorted index
    std::vector<K> commToIndex(numComm);
    for (size_t i = 0; i < numComm; ++i) {
        commToIndex[commOrder[i]] = static_cast<K>(i);
    }

    // Step 4: BFS within each community to assign local IDs (parallel)
    // Uses flat vector + sentinel instead of per-community unordered_map for O(1) lookup
    std::vector<K> localIds(N, static_cast<K>(-1));

    // Global vertex-to-local-index map (flat vector, sentinel = (size_t)-1)
    // Shared across all communities but each vertex belongs to exactly one community
    std::vector<size_t> vertToLocal(N, static_cast<size_t>(-1));

    // Pre-populate the flat map (parallel, no conflicts since each vertex has one community)
    #pragma omp parallel for schedule(static)
    for (size_t ci = 0; ci < numComm; ++ci) {
        K c = commOrder[ci];
        auto& verts = commVertices[c];
        for (size_t i = 0; i < verts.size(); ++i) {
            vertToLocal[verts[i]] = i;
        }
    }

    #pragma omp parallel
    {
        std::queue<K> bfsQueue;
        std::vector<bool> visited;

        #pragma omp for schedule(dynamic, 1)
        for (size_t ci = 0; ci < numComm; ++ci) {
            K c = commOrder[ci];
            // Delegate per-community BFS to the shared primitive
            // (SECTION 16-PRIMITIVES).  This eliminates a copy of the
            // BFS-from-max-degree loop that used to live inline here.
            intraBFSFromHub<K, NodeID_T, DestID_T>(
                commVertices[c], c, membership, degrees, g,
                vertToLocal, visited, bfsQueue, localIds);
        }
    }

    // Step 5: Compute global IDs = offset[commIndex] + localId
    #pragma omp parallel for
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] > 0) {
            K commIdx = commToIndex[membership[v]];
            newIds[v] = static_cast<NodeID_T>(offsets[commIdx] + localIds[v]);
        }
    }

    // Step 6: Assign isolated vertices at the end
    size_t isolatedStart = offsets[numComm];
    for (size_t i = 0; i < isolated.size(); ++i) {
        newIds[isolated[i]] = static_cast<NodeID_T>(isolatedStart + i);
    }

    GRAPHBREW_TRACE("orderConnectivityBFS: %zu communities, %zu isolated", numComm, isolated.size());
}

//=============================================================================
// SECTION 16d: HYBRID LEIDEN + RABBITORDER ORDERING
//=============================================================================

/**
 * Hybrid Leiden + RabbitOrder Ordering
 *
 * KEY INSIGHT: Combines the strengths of both algorithms!
 *
 * Problem:
 * - Leiden: Great community quality (high modularity), but too many small
 *   communities (~100K) → poor cache locality
 * - RabbitOrder: Great cache locality (~1K large communities), but communities
 *   may not reflect true graph structure
 *
 * Solution:
 * 1. Use Leiden to detect fine-grained communities (captures graph structure)
 * 2. Build super-graph where each Leiden community is a vertex
 * 3. Run RabbitOrder on the super-graph to merge communities into ~1K cache blocks
 * 4. Order vertices: RabbitOrder's block order + BFS within each Leiden community
 *
 * Result:
 * - Cache blocks from RabbitOrder (good locality)
 * - Connectivity-based ordering within blocks from BFS (preserves Leiden quality)
 * - Expected: geo-mean ~2000-3000 (vs Leiden 6000, vs RabbitOrder 1100)
 *
 * Complexity: O(E) for super-graph build + O(E_super) for RabbitOrder + O(E) for BFS
 *             Total: Same as Leiden, just different ordering phase
 */
template <typename K, typename NodeID_T, typename DestID_T>
void orderHybridLeidenRabbit(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderHybridLeidenRabbit: N=%zu", N);

    // ================================================================
    // STEP 0 (optional): Hub Extraction
    // Remove top hubExtractionPct highest-degree vertices before
    // building the community super-graph. Hubs distort community
    // structure and RabbitOrder merging decisions. They are reinserted
    // adjacent to their community block at the end.
    // ================================================================

    std::vector<bool> isHub(N, false);
    // Per-community hub lists (filled if hub extraction enabled)
    // Indexed by community ID, each entry is a list of hub vertex IDs
    std::vector<std::vector<K>> commHubs;
    size_t numHubs = 0;

    // Find number of communities (needed before hub extraction)
    K maxComm = 0;
    for (size_t v = 0; v < N; ++v) {
        maxComm = std::max(maxComm, membership[v]);
    }
    const size_t C = static_cast<size_t>(maxComm) + 1;

    if (config.useHubExtraction) {
        // Determine degree threshold for hub extraction
        // Sort degrees to find the top hubExtractionPct percentile
        size_t activeCount = 0;
        for (size_t v = 0; v < N; ++v) {
            if (degrees[v] > 0) activeCount++;
        }

        size_t hubCount = std::max(size_t(1),
                          static_cast<size_t>(activeCount * config.hubExtractionPct));

        // Find the hubCount-th largest degree using partial sort
        std::vector<K> sortedDegs;
        sortedDegs.reserve(activeCount);
        for (size_t v = 0; v < N; ++v) {
            if (degrees[v] > 0) sortedDegs.push_back(degrees[v]);
        }

        // nth_element to find threshold
        if (hubCount < sortedDegs.size()) {
            std::nth_element(sortedDegs.begin(),
                             sortedDegs.begin() + hubCount,
                             sortedDegs.end(),
                             std::greater<K>());
            K degThreshold = sortedDegs[hubCount];

            // Mark hubs: vertices with degree > threshold
            // (use > to avoid extracting too many if many vertices share the threshold degree)
            for (size_t v = 0; v < N; ++v) {
                if (degrees[v] > degThreshold) {
                    isHub[v] = true;
                    numHubs++;
                }
            }
        }

        printf("  hybrid-rabbit: hubx extracted %zu hubs (%.2f%% of %zu active, deg>%u)\n",
               numHubs, 100.0 * numHubs / std::max(size_t(1), activeCount),
               activeCount,
               numHubs > 0 ? static_cast<unsigned>(sortedDegs[hubCount]) : 0u);

        // Build per-community hub lists
        commHubs.resize(C);
        for (size_t v = 0; v < N; ++v) {
            if (isHub[v]) {
                commHubs[membership[v]].push_back(static_cast<K>(v));
            }
        }
    }

    printf("  hybrid-rabbit: %zu Leiden communities\n", C);

    // ================================================================
    // STEP 1: Build super-graph from Leiden communities
    // ================================================================

    // Separate isolated (zero-degree) vertices
    // Hubs remain in their communities for BFS ordering
    // but are excluded from the super-graph edge computation
    std::vector<std::vector<K>> commVertices(C);
    std::vector<K> isolated;

    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(static_cast<K>(v));
        } else {
            commVertices[membership[v]].push_back(static_cast<K>(v));
        }
    }

    // ================================================================
    // PHASE 6 REFACTOR (2026-05-19): HRAB STEP 1 + STEP 2 + STEP 3 + STEP 3b
    // are now expressed as three function calls into SECTION 16-STAGE1
    // primitives.  ~600 lines of inline super-graph construction,
    // RabbitOrder, dendrogram DFS, and BNF+George-Liu+RCM have been
    // replaced by:
    //
    //     buildCommunitySuperGraph()   (Phase 1)
    //     runRabbitOnSuperCSR()        (Phase 2, default)
    //     runRCMOnSuperCSR()           (Phase 2, when useRCMSuper=true)
    //
    // Parity envelope vs. the pre-refactor inline path on
    // {cit-Patents, soc-pokec, hollywood-2009, com-Orkut}:
    //     -1.7% to +4.9% L3 memory accesses (PR, 3 trials, 8 threads)
    // Data: results/data/composability_parity_envelope_2026_05_19.md.
    //
    // HRAB is now literally a composition:
    //     {super-graph=SuperRabbit (or SuperRCM), community=Identity, intra=RCM}
    //     + hubx feature flag (STEP 0)
    //     + hsort feature flag (STEP 5)
    // ================================================================

    // STEP 1: Community super-graph (honours hubx mask exactly).
    auto sg = buildCommunitySuperGraph<K, NodeID_T, DestID_T>(
        membership, degrees, isHub, g, N, static_cast<K>(C));

    printf("  hybrid-rabbit: super-graph M=%.0f\n", sg.M);

    // Per-community vertex count for active/empty root separation.
    std::vector<size_t> numVertsPerComm(C, 0);
    for (size_t c = 0; c < C; ++c) numVertsPerComm[c] = commVertices[c].size();

    // STEPs 2 + 3 (default) OR STEP 3b (when useRCMSuper=true).
    std::vector<K> commPerm;
    if (config.useRCMSuper) {
        printf("  hybrid-rabbit: applying BNF-RCM on super-graph (%zu communities)\n", C);
        commPerm = runRCMOnSuperCSR<K>(sg, numVertsPerComm);
    } else {
        const float gamma = static_cast<float>(config.superGraphResolution);
        commPerm = runRabbitOnSuperCSR<K>(std::move(sg), numVertsPerComm, gamma);
    }

    // ================================================================
    // STEP 4: Intra-community ordering, ordered by community permutation
    // Two modes: BFS (default) or Gorder-greedy (maximizes neighbor overlap)
    // ================================================================

    // Sort communities by their permutation (RabbitOrder DFS or RCM)
    std::vector<K> sortedComms(C);
    std::iota(sortedComms.begin(), sortedComms.end(), K(0));
    std::sort(sortedComms.begin(), sortedComms.end(),
        [&](K a, K b) { return commPerm[a] < commPerm[b]; });

    // Create inverse mapping: commPerm value -> sorted index
    std::vector<K> commToSortedIdx(C);
    for (size_t i = 0; i < C; ++i) {
        commToSortedIdx[sortedComms[i]] = static_cast<K>(i);
    }

    // Compute vertex offsets per community (in sorted order)
    std::vector<size_t> hubsPerComm(C, 0); // kept for stats
    if (config.useHubExtraction) {
        for (size_t c = 0; c < C; ++c) {
            hubsPerComm[c] = commHubs[c].size();
        }
    }

    std::vector<size_t> vertexOffsets(C + 1, 0);
    for (size_t i = 0; i < C; ++i) {
        K c = sortedComms[i];
        vertexOffsets[i + 1] = vertexOffsets[i] + commVertices[c].size();
    }

    printf("  hybrid-rabbit: total vertices in communities = %zu\n", vertexOffsets[C]);

    // Print community size distribution
    {
        size_t s1=0, s10=0, s100=0, s1k=0, s10k=0, s100k=0, sHuge=0;
        size_t v1=0, v10=0, v100=0, v1k=0, v10k=0, v100k=0, vHuge=0;
        size_t maxSz=0;
        for (size_t ci = 0; ci < C; ++ci) {
            size_t sz = commVertices[ci].size();
            if (sz == 0) continue;
            if (sz > maxSz) maxSz = sz;
            if (sz <= 3)      { s1++; v1+=sz; }
            else if (sz <= 10)   { s10++; v10+=sz; }
            else if (sz <= 100)  { s100++; v100+=sz; }
            else if (sz <= 1000) { s1k++; v1k+=sz; }
            else if (sz <= 10000){ s10k++; v10k+=sz; }
            else if (sz <= 100000) { s100k++; v100k+=sz; }
            else                  { sHuge++; vHuge+=sz; }
        }
        printf("  comm-sizes: <=3: %zu comms (%zu v) | 4-10: %zu (%zu) | 11-100: %zu (%zu) | 101-1K: %zu (%zu) | 1K-10K: %zu (%zu) | >10K: %zu (%zu) | max=%zu\n",
               s1, v1, s10, v10, s100, v100, s1k, v1k, s10k, v10k, s100k+sHuge, v100k+vHuge, maxSz);
    }

    // Local IDs within each community
    std::vector<K> localIds(N, static_cast<K>(-1));

    // Global vertex-to-local-index map (flat vector, O(1) lookup)
    // Each vertex belongs to exactly one community, so no conflicts
    std::vector<size_t> vertToLocalHrab(N, static_cast<size_t>(-1));
    #pragma omp parallel for schedule(static)
    for (size_t ci = 0; ci < C; ++ci) {
        K c = sortedComms[ci];
        auto& verts = commVertices[c];
        for (size_t i = 0; i < verts.size(); ++i) {
            vertToLocalHrab[verts[i]] = i;
        }
    }

    if (config.useGorderIntra) {
        // ============================================================
        // GORDER-GREEDY intra-community ordering (with UnitHeap)
        //
        // For each community, greedily place vertices to maximize
        // neighbor overlap with a sliding window of recently-placed vertices.
        //
        // Uses a UnitHeap priority queue (adapted from Gorder, Hao Wei 2016)
        // for O(1) amortized IncrementKey/DecrementKey/ExtractMax,
        // reducing per-community complexity from O(sz²) to O(|E_local|).
        // This eliminates the need for BFS fallback on large communities.
        //
        // Algorithm (per community):
        // 1. Start from highest-degree vertex
        // 2. Maintain a priority score for each unplaced vertex:
        //    score[v] = number of v's neighbors in the last W placed vertices
        // 3. Each step: place the vertex with highest score (via UnitHeap)
        // 4. Update scores: boost neighbors of newly placed vertex,
        //    decay neighbors of vertex sliding out of window
        // ============================================================
        const int W = config.gorderWindow;

        // With UnitHeap, gord is O(|E_local|) per community — no longer O(sz²).
        // The BFS fallback is now only a safety net for extreme cases.
        // Default: N (no fallback). Use gordf<threshold> to set explicitly.
        const size_t gordThreshold = config.gorderFallback > 0
            ? static_cast<size_t>(config.gorderFallback)
            : static_cast<size_t>(N); // default: no fallback (UnitHeap handles all sizes)

        // Track stats
        std::atomic<size_t> gordCount(0), bfsCount(0), gordVerts(0), bfsVerts(0);

        printf("  hybrid-rabbit-gord: Gorder-greedy intra-community (window=%d, fallback=%zu)\n", W, gordThreshold);

        // ============================================================
        // UnitHeap: O(1) amortized priority queue for integer keys
        // ============================================================
        // Adapted from Gorder (Hao Wei, 2016, MIT License).
        // Elements are organized in a doubly-linked list sorted by
        // descending key. Header[k] tracks the first/last element
        // with key=k. IncrementKey/DecrementKey just relink the
        // element between adjacent buckets in O(1).
        // ExtractMax pops from the front in O(1).
        //
        // This replaces the O(sz) linear scan per step, reducing
        // per-community complexity from O(sz²) to O(|E_local|).
        // ============================================================
        struct GordHeapNode {
            int key;    // current score
            int prev;   // prev element in linked list (-1 = none)
            int next;   // next element in linked list (-1 = none)
        };
        struct GordHeapBucket {
            int first = -1;
            int second = -1;  // last element in this bucket
        };
        struct GordHeap {
            std::vector<GordHeapNode> nodes;
            std::vector<GordHeapBucket> buckets;
            int top;        // index of the max-key element
            int maxKey;     // current maximum key value

            void init(size_t n) {
                nodes.resize(n);
                // All elements start with key=0, linked in order 0→1→...→n-1
                for (size_t i = 0; i < n; ++i) {
                    nodes[i].key = 0;
                    nodes[i].prev = static_cast<int>(i) - 1;
                    nodes[i].next = (i + 1 < n) ? static_cast<int>(i + 1) : -1;
                }
                buckets.clear();
                buckets.resize(16); // will grow as needed
                buckets[0].first = 0;
                buckets[0].second = static_cast<int>(n - 1);
                top = 0;
                maxKey = 0;
            }

            void ensureBucket(int k) {
                if (k >= static_cast<int>(buckets.size())) {
                    buckets.resize(static_cast<size_t>(k + 8));
                }
            }

            // Unlink element from its current position in the doubly-linked list
            void unlink(int idx) {
                int p = nodes[idx].prev;
                int n = nodes[idx].next;
                if (p >= 0) nodes[p].next = n;
                if (n >= 0) nodes[n].prev = p;

                int k = nodes[idx].key;
                // Update bucket pointers
                if (buckets[k].first == idx && buckets[k].second == idx) {
                    // Only element in bucket
                    buckets[k].first = buckets[k].second = -1;
                } else if (buckets[k].first == idx) {
                    buckets[k].first = n;
                } else if (buckets[k].second == idx) {
                    buckets[k].second = p;
                }

                if (top == idx) {
                    top = n; // next element becomes new top
                }
            }

            // Insert element at the FRONT of bucket k (before the current first)
            void linkToBucketFront(int idx, int k) {
                ensureBucket(k);
                nodes[idx].key = k;

                if (buckets[k].first < 0) {
                    // Empty bucket — find where to splice in the linked list
                    // We need to find the element just before where bucket k should be
                    // (i.e., the last element of the next-higher occupied bucket)
                    // and the element just after (first element of next-lower bucket)
                    int afterNode = -1;
                    for (int bk = k - 1; bk >= 0; --bk) {
                        if (buckets[bk].first >= 0) {
                            afterNode = buckets[bk].first;
                            break;
                        }
                    }
                    int beforeNode = -1;
                    for (int bk = k + 1; bk <= maxKey; ++bk) {
                        if (buckets[bk].second >= 0) {
                            beforeNode = buckets[bk].second;
                            break;
                        }
                    }

                    nodes[idx].prev = beforeNode;
                    nodes[idx].next = afterNode;
                    if (beforeNode >= 0) nodes[beforeNode].next = idx;
                    if (afterNode >= 0) nodes[afterNode].prev = idx;

                    buckets[k].first = buckets[k].second = idx;
                } else {
                    // Non-empty bucket — insert before the current first
                    int oldFirst = buckets[k].first;
                    int beforeOldFirst = nodes[oldFirst].prev;

                    nodes[idx].prev = beforeOldFirst;
                    nodes[idx].next = oldFirst;
                    nodes[oldFirst].prev = idx;
                    if (beforeOldFirst >= 0) nodes[beforeOldFirst].next = idx;

                    buckets[k].first = idx;
                }

                if (k > maxKey) maxKey = k;
                if (k >= nodes[top].key) top = idx;
            }

            void incrementKey(int idx) {
                int oldKey = nodes[idx].key;
                unlink(idx);
                linkToBucketFront(idx, oldKey + 1);
            }

            void decrementKey(int idx) {
                int oldKey = nodes[idx].key;
                if (oldKey <= 0) return; // don't go negative
                unlink(idx);
                linkToBucketFront(idx, oldKey - 1);
            }

            // Remove element from the heap entirely
            void deleteElement(int idx) {
                unlink(idx);
                nodes[idx].prev = nodes[idx].next = -1;
                nodes[idx].key = -1; // mark as removed
                // Update maxKey if needed
                while (maxKey > 0 && buckets[maxKey].first < 0) maxKey--;
            }

            // Extract the element with the highest key
            int extractMax() {
                // Find the actual top (maxKey may have become empty)
                while (maxKey > 0 && buckets[maxKey].first < 0) maxKey--;
                int idx = buckets[maxKey].first;
                deleteElement(idx);
                return idx;
            }
        };

        #pragma omp parallel
        {
            // Thread-local working buffers
            std::vector<K> placedOrder;       // order of placed vertices (for window tracking)
            std::vector<std::vector<size_t>> localNeighbors;  // adjacency within community
            std::queue<K> bfsQueue;           // for BFS fallback
            GordHeap heap;                    // UnitHeap for O(1) max-extraction

            #pragma omp for schedule(dynamic, 1)
            for (size_t ci = 0; ci < C; ++ci) {
                K c = sortedComms[ci];
                auto& verts = commVertices[c];
                if (verts.empty()) continue;
                const size_t sz = verts.size();

                // For tiny communities, simple sequential ordering
                if (sz <= 3) {
                    for (size_t i = 0; i < sz; ++i) {
                        localIds[verts[i]] = static_cast<K>(i);
                    }
                    continue;
                }

                // Build local vertex index: use pre-built flat vertToLocalHrab[]

                // ---- FALLBACK: BFS for large communities ----
                if (sz > gordThreshold) {
                    bfsVerts += sz;
                    bfsCount++;

                    std::vector<bool> visited(sz, false);

                    // Find highest-degree vertex as BFS root
                    size_t startIdx = 0;
                    K maxDeg = degrees[verts[0]];
                    for (size_t i = 1; i < sz; ++i) {
                        if (degrees[verts[i]] > maxDeg) {
                            maxDeg = degrees[verts[i]];
                            startIdx = i;
                        }
                    }

                    K localId = 0;
                    while (!bfsQueue.empty()) bfsQueue.pop(); // clear
                    bfsQueue.push(static_cast<K>(startIdx));
                    visited[startIdx] = true;

                    while (!bfsQueue.empty()) {
                        K localU = bfsQueue.front();
                        bfsQueue.pop();
                        localIds[verts[localU]] = localId++;

                        K u = verts[localU];
                        for (auto neighbor : g.out_neigh(u)) {
                            NodeID_T v;
                            if constexpr (std::is_same_v<DestID_T, NodeID_T>) v = neighbor;
                            else v = neighbor.v;
                            if (membership[v] != c) continue;
                            size_t localIdx = vertToLocalHrab[static_cast<K>(v)];
                            if (localIdx != static_cast<size_t>(-1) && !visited[localIdx]) {
                                visited[localIdx] = true;
                                bfsQueue.push(static_cast<K>(localIdx));
                            }
                        }
                    }
                    // Handle disconnected vertices within community
                    for (size_t i = 0; i < sz; ++i) {
                        if (!visited[i]) {
                            localIds[verts[i]] = localId++;
                        }
                    }
                    continue;
                }

                // ---- GORDER-GREEDY with UnitHeap: O(|E_local|) ----
                gordVerts += sz;
                gordCount++;

                // Build local adjacency: for each vertex, list of local neighbor indices
                localNeighbors.resize(sz);
                for (size_t i = 0; i < sz; ++i) {
                    localNeighbors[i].clear();
                }
                for (size_t i = 0; i < sz; ++i) {
                    K u = verts[i];
                    for (auto neighbor : g.out_neigh(u)) {
                        NodeID_T v;
                        if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                            v = neighbor;
                        } else {
                            v = neighbor.v;
                        }
                        if (membership[v] != c) continue;
                        size_t localIdx = vertToLocalHrab[static_cast<K>(v)];
                        if (localIdx != static_cast<size_t>(-1)) {
                            localNeighbors[i].push_back(localIdx);
                        }
                    }
                }

                // Initialize UnitHeap: all vertices start at key=0 (no neighbors placed yet)
                // We use degree as initial key so that high-degree vertices are preferred
                // for tie-breaking (same behavior as before).
                // Actually, Gorder uses indegree as initial key. For our community-local
                // setting, we initialize all at 0 and handle seeding separately.
                heap.init(sz);

                // Seed: highest-degree vertex — extract it directly
                size_t bestSeed = 0;
                K maxDeg = degrees[verts[0]];
                for (size_t i = 1; i < sz; ++i) {
                    if (degrees[verts[i]] > maxDeg) {
                        maxDeg = degrees[verts[i]];
                        bestSeed = i;
                    }
                }

                // Remove seed from heap and place it
                heap.deleteElement(static_cast<int>(bestSeed));
                placedOrder.clear();
                placedOrder.reserve(sz);
                placedOrder.push_back(static_cast<K>(bestSeed));
                localIds[verts[bestSeed]] = 0;

                // Boost neighbors of seed in the heap
                for (size_t nbr : localNeighbors[bestSeed]) {
                    if (heap.nodes[nbr].key >= 0) { // not yet removed
                        heap.incrementKey(static_cast<int>(nbr));
                    }
                }

                // Greedy loop: place remaining vertices using O(1) extractMax
                for (K localId = 1; localId < static_cast<K>(sz); ++localId) {
                    // O(1) amortized: extract vertex with highest score
                    int best = heap.extractMax();

                    placedOrder.push_back(static_cast<K>(best));
                    localIds[verts[best]] = localId;

                    // Boost neighbors of newly placed vertex — O(deg) total across all steps
                    for (size_t nbr : localNeighbors[best]) {
                        if (heap.nodes[nbr].key >= 0) { // still in heap
                            heap.incrementKey(static_cast<int>(nbr));
                        }
                    }

                    // Decay: if window is full, remove influence of oldest vertex
                    if (static_cast<int>(placedOrder.size()) > W) {
                        size_t oldVert = placedOrder[placedOrder.size() - 1 - W];
                        for (size_t nbr : localNeighbors[oldVert]) {
                            if (heap.nodes[nbr].key >= 0) { // still in heap
                                heap.decrementKey(static_cast<int>(nbr));
                            }
                        }
                    }
                }
            }
        }

        printf("  hybrid-rabbit-gord: %zu comms gord (%zu verts), %zu comms bfs-fallback (%zu verts)\n",
               gordCount.load(), gordVerts.load(), bfsCount.load(), bfsVerts.load());
    } else if (config.useRCMIntra) {
        // ============================================================
        // RCM (Cuthill-McKee) within each community
        //
        // Applies Reverse Cuthill-McKee ordering within each community
        // to minimize bandwidth (max edge span) at the community level.
        // This is embarrassingly parallel: each community is an
        // independent problem processed by a separate thread.
        //
        // Tiered strategy by community size:
        //   - ≤1 vertex: trivial (id=0)
        //   - 2-32 vertices: min-degree start + serial CM BFS (skip BNF)
        //   - 33-4096 vertices: simplified BNF (max 3 GL iterations)
        //                       + serial CM BFS
        //   - >4096 vertices: full BNF start + serial CM BFS
        //
        // The CM BFS at each level sorts neighbors by ascending degree
        // (Cuthill-McKee heuristic), producing an ordering that places
        // nearby-degree vertices together. The final reversal converts
        // CM to RCM, which empirically yields tighter bandwidth.
        // ============================================================
        std::atomic<size_t> rcmTinyCount{0}, rcmSmallCount{0}, rcmMedCount{0}, rcmLargeCount{0};
        std::atomic<size_t> rcmTinyVerts{0}, rcmSmallVerts{0}, rcmMedVerts{0}, rcmLargeVerts{0};

        #pragma omp parallel
        {
            // Thread-local buffers reused across communities
            std::vector<bool> visited;
            std::queue<K> bfsQueue;
            std::vector<K> cmOrder;

            #pragma omp for schedule(dynamic, 1)
            for (size_t ci = 0; ci < C; ++ci) {
                K c = sortedComms[ci];
                auto& verts = commVertices[c];
                const size_t sz = verts.size();
                if (sz == 0) continue;

                if (sz == 1) {
                    localIds[verts[0]] = 0;
                    rcmTinyCount.fetch_add(1, std::memory_order_relaxed);
                    rcmTinyVerts.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                // ----------------------------------------------------------
                // Adaptive intra-community ordering (May 2026):
                // For HUGE communities (>4096 vertices), use BFS-intra
                // instead of full RCM.  Empirically (hollywood-2009 PR,
                // 20 iters, 3 trials):
                //   sz>4096 RCM-intra:   0.348s avg kernel  (1.28x speedup)
                //   sz>4096 BFS-intra:   0.267s avg kernel  (1.66x speedup)
                // Cause: RCM's neighbor-degree sort + final reversal puts
                // low-degree (peripheral) vertices FIRST in memory for huge
                // communities, hurting working-set retention.  Plain BFS
                // from the highest-degree vertex places hubs first — better
                // cache behavior for dense communities with large diameter.
                // For sz<=4096 (typical sparse-graph communities), RCM's
                // bandwidth minimization still wins (cit-Patents data).
                // ----------------------------------------------------------
                if (sz > 4096) {
                    // Composes intraBFSFromHub() from SECTION 16-PRIMITIVES.
                    // BFS-from-max-degree wins on huge communities: it
                    // places hubs first, which matters for the working set
                    // of PR-style power-law access patterns.
                    intraBFSFromHub<K, NodeID_T, DestID_T>(
                        verts, c, membership, degrees, g,
                        vertToLocalHrab, visited, bfsQueue, localIds);
                    rcmLargeCount.fetch_add(1, std::memory_order_relaxed);
                    rcmLargeVerts.fetch_add(sz, std::memory_order_relaxed);
                    continue;
                }

                // sz in [2, 4096] -> intraRCM primitive (BNF/George-Liu
                // start when sz > 32, min-degree start otherwise; then
                // CM-BFS with neighbour-degree sort and final reversal).
                // This was previously ~110 lines inlined here.
                intraRCM<K, NodeID_T, DestID_T>(
                    verts, c, membership, degrees, g,
                    vertToLocalHrab, visited, bfsQueue, cmOrder, localIds);

                // Track stats
                if (sz <= 32) {
                    rcmSmallCount.fetch_add(1, std::memory_order_relaxed);
                    rcmSmallVerts.fetch_add(sz, std::memory_order_relaxed);
                } else {
                    rcmMedCount.fetch_add(1, std::memory_order_relaxed);
                    rcmMedVerts.fetch_add(sz, std::memory_order_relaxed);
                }
            }
        }

        printf("  hybrid-rabbit-rcm-intra: tiny=%zu(%zuv) small=%zu(%zuv) med=%zu(%zuv) large=%zu(%zuv)\n",
               rcmTinyCount.load(), rcmTinyVerts.load(),
               rcmSmallCount.load(), rcmSmallVerts.load(),
               rcmMedCount.load(), rcmMedVerts.load(),
               rcmLargeCount.load(), rcmLargeVerts.load());
    } else {
        // ============================================================
        // Standard BFS within each community (default when neither
        // useGorderIntra nor useRCMIntra is set).  Composes
        // intraBFSFromHub() from SECTION 16-PRIMITIVES.
        // ============================================================
        #pragma omp parallel
        {
            std::queue<K> bfsQueue;
            std::vector<bool> visited;

            #pragma omp for schedule(dynamic, 1)
            for (size_t ci = 0; ci < C; ++ci) {
                K c = sortedComms[ci];
                intraBFSFromHub<K, NodeID_T, DestID_T>(
                    commVertices[c], c, membership, degrees, g,
                    vertToLocalHrab, visited, bfsQueue, localIds);
            }
        }
    }

    // Compute global IDs using the sorted index mapping
    #pragma omp parallel for
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] > 0) {
            K c = membership[v];
            K sortedIdx = commToSortedIdx[c];
            newIds[v] = static_cast<NodeID_T>(vertexOffsets[sortedIdx] + localIds[v]);
        }
    }

    // Assign isolated vertices at the end
    size_t isolatedStart = vertexOffsets[C];
    for (size_t i = 0; i < isolated.size(); ++i) {
        newIds[isolated[i]] = static_cast<NodeID_T>(isolatedStart + i);
    }

    // ================================================================
    // STEP 5 (optional): Hub Sort post-processing
    // Pack hub vertices (degree > avgDegree) contiguously at the front
    // of the ID space, sorted by descending degree. Non-hub vertices
    // fill the remaining IDs preserving their relative order.
    // This is the IISWC'18 "Hub Sorting" technique — proven to improve
    // spatial locality for power-law graphs by packing frequently-accessed
    // vData elements into fewer cache lines.
    // ================================================================
    if (config.useHubSort) {
        // Hub threshold: use sqrt(N) like Gorder's "huge vertex" threshold.
        // This selects only the truly high-degree vertices (typically <1% of graph),
        // unlike avgDegree which can pack 30-70% and destroy community locality.
        K hubThreshold = static_cast<K>(std::sqrt(static_cast<double>(N)));
        if (hubThreshold < 10) hubThreshold = 10;  // minimum sensible threshold

        // Collect hubs (degree > sqrt(N)) sorted by descending degree
        // and non-hubs preserving their current community-based order
        std::vector<std::pair<K, NodeID_T>> hubs;    // (degree, vertex)
        std::vector<std::pair<NodeID_T, NodeID_T>> nonHubs;  // (currentNewId, vertex)

        for (size_t v = 0; v < N; ++v) {
            if (degrees[v] > hubThreshold) {
                hubs.push_back({degrees[v], static_cast<NodeID_T>(v)});
            } else {
                nonHubs.push_back({newIds[v], static_cast<NodeID_T>(v)});
            }
        }

        // Sort hubs by descending degree
        std::sort(hubs.begin(), hubs.end(),
            [](const std::pair<K,NodeID_T>& a, const std::pair<K,NodeID_T>& b) {
                return a.first > b.first;
            });

        // Sort non-hubs by their current newId (preserve relative order)
        std::sort(nonHubs.begin(), nonHubs.end());

        // Assign: hubs first (0, 1, 2, ...), then non-hubs
        NodeID_T nextId = 0;
        for (auto& [deg, v] : hubs) {
            newIds[v] = nextId++;
        }
        for (auto& [oldId, v] : nonHubs) {
            newIds[v] = nextId++;
        }

        printf("  hybrid-rabbit-hsort: %zu hubs (deg>%u=sqrt(%zu)) packed at front, %zu non-hubs after (%.1f%%)\n",
               hubs.size(), static_cast<unsigned>(hubThreshold), N, nonHubs.size(),
               100.0 * hubs.size() / N);
    }

    printf("  hybrid-rabbit: %zu blocks, %zu active vertices, %zu hubs, %zu isolated\n",
           C, N - isolated.size() - numHubs, numHubs, isolated.size());

    GRAPHBREW_TRACE("orderHybridLeidenRabbit: %zu blocks, %zu isolated", C, isolated.size());
}

//=============================================================================
// SECTION 16b-HIER: HIERARCHICAL LEIDEN + RABBIT (HLR)
//=============================================================================

/**
 * @brief Hierarchical Leiden + RabbitOrder reordering (graphbrew:hlr).
 *
 * Generalises HRAB by running the gamma-tuned super-graph RabbitOrder primitive
 * at EVERY level of the Leiden dendrogram, not just the finest.  HRAB uses
 * the final (coarsest-aggregation) membership only; HLR consumes the full
 * vector membershipPerPass[0..L-1] and emits a permutation whose locality is
 * fractal: vertices in the same coarsest community are contiguous; within
 * each coarsest community, vertices in the same finer community are
 * contiguous; ...; within each finest community, vertices retain their
 * original index order (a future pass may add intra-community BFS-from-hub).
 *
 * Algorithm (L = membershipPerPass.size(), gamma = config.superGraphResolution):
 *   for k = 0 .. L-1:
 *     C_k          = max(membershipPerPass[k]) + 1
 *     sg_k         = buildCommunitySuperGraph(membershipPerPass[k], degrees, ..., C_k)
 *     levelPerm[k] = runRabbitOnSuperCSR(sg_k, ..., gamma)
 *   sort vertices v by tuple ( levelPerm[L-1][mpp[L-1][v]], ..., levelPerm[0][mpp[0][v]], v )
 *
 * Falls back to orderHybridLeidenRabbit when no multi-level hierarchy is
 * available (membershipPerPass empty or only one level).
 */
template <typename K, typename NodeID_T, typename DestID_T>
void orderHierarchicalLeidenRabbit(
    pvector<NodeID_T>& newIds,
    const GraphBrewResult<K>& result,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    const GraphBrewConfig& config)
{
    GRAPHBREW_TRACE("orderHierarchicalLeidenRabbit: N=%zu, passes=%zu",
                    N, result.membershipPerPass.size());
    const auto& mpp = result.membershipPerPass;
    const size_t L = mpp.size();

    if (L <= 1) {
        printf("  hier-rabbit: only %zu Leiden pass(es); delegating to HRAB.\n", L);
        orderHybridLeidenRabbit<K, NodeID_T, DestID_T>(
            newIds, result.membership, degrees, g, N, config);
        return;
    }

    const float gamma = static_cast<float>(config.superGraphResolution);
    std::vector<bool> emptyHubMask;  // no hub extraction in HLR (could be added later)

    std::vector<K> levelC(L, 0);
    std::vector<double> levelM(L, 0.0);
    std::vector<std::vector<K>> levelPerm(L);

    for (size_t k = 0; k < L; ++k) {
        K C_k = 0;
        for (size_t v = 0; v < N; ++v) {
            if (static_cast<K>(mpp[k][v] + 1) > C_k) C_k = static_cast<K>(mpp[k][v] + 1);
        }
        levelC[k] = C_k;

        if (C_k <= 1) {
            levelPerm[k].assign(C_k, K(0));
            printf("  hier-rabbit: level %zu: C=%u (trivial, skipped)\n", k, (unsigned)C_k);
            continue;
        }

        auto sg = buildCommunitySuperGraph<K, NodeID_T, DestID_T>(
            mpp[k], degrees, emptyHubMask, g, N, C_k);

        std::vector<size_t> numVertsPerComm(C_k, 0);
        for (size_t v = 0; v < N; ++v) numVertsPerComm[mpp[k][v]]++;

        levelM[k] = static_cast<double>(sg.M);
        printf("  hier-rabbit: level %zu: C=%u, M_sg=%.0f, gamma=%.3f\n",
               k, (unsigned)C_k, sg.M, gamma);

        levelPerm[k] = runRabbitOnSuperCSR<K>(std::move(sg), numVertsPerComm, gamma);
    }

    // ---- Hierarchy-depth + super-graph-density heuristic ----
    // We keep a level only if (1) it produces a strictly new partition
    // (collapsed level check: C_k != C_{lastKept}) AND (2) the super-graph
    // at that level is dense enough for the gamma-Rabbit merge criterion to
    // yield meaningful merges.  Empirically (PR cache-sim, 3 trials on
    // {cit-Patents, soc-pokec, hollywood-2009, com-Orkut}) we observed:
    //   - cit-Patents finest super-graph M_sg/C = 14    -> +40% regression
    //   - soc-pokec    finest super-graph M_sg/C = 29   -> noise (~0%)
    //   - hollywood    finest super-graph M_sg/C = 317  -> -14.3% win
    //   - com-Orkut    finest super-graph M_sg/C = 190  -> noise (~+4%)
    // i.e., the multi-level Rabbit pays off iff every kept fine level has
    // average super-node weight >= ~50.  We use that threshold as a
    // graph-agnostic guard.  The COARSEST level is always kept regardless
    // (otherwise HLR has no Rabbit ordering at all and degenerates below
    // even the original layout).
    constexpr double kMinAvgSuperWeight = 50.0;
    std::vector<bool> keepLevel(L, false);
    K lastKept = 0;
    // Coarsest first: always kept.
    for (size_t k = L; k-- > 0; ) {
        if (levelC[k] <= 1) continue;
        keepLevel[k] = true;
        lastKept = levelC[k];
        break;
    }
    // Finer levels: include only if non-collapsed and dense enough.
    for (size_t k = L; k-- > 0; ) {
        if (keepLevel[k]) continue;             // already taken (coarsest)
        if (levelC[k] <= 1) continue;            // trivial
        if (levelC[k] == lastKept) continue;     // collapsed
        const double avg = (levelC[k] > 0)
            ? static_cast<double>(levelM[k]) / static_cast<double>(levelC[k])
            : 0.0;
        if (avg < kMinAvgSuperWeight) {
            printf("  hier-rabbit: level %zu: avg M_sg/C=%.1f < %.0f, skipping (super-graph too sparse for gamma-Rabbit)\n",
                   k, avg, kMinAvgSuperWeight);
            continue;
        }
        keepLevel[k] = true;
        lastKept = levelC[k];
    }
    size_t numKept = 0;
    for (size_t k = 0; k < L; ++k) if (keepLevel[k]) ++numKept;
    printf("  hier-rabbit: keeping %zu of %zu Leiden levels (collapsed/sparse levels skipped)\n",
           numKept, L);

    // ---- Intra-community RCM (matches HRAB's default tail step) ----
    // HRAB enables RCM intra-community ordering by default (CLI parser:
    // config.useRCMIntra = true), which gives 30-50% better memory accesses
    // than BFS-from-hub on the test set.  We use the same primitive
    // (intraRCM, SECTION 16-PRIMITIVES) within each block of the FINEST
    // KEPT membership.  When the finest level was skipped for being too
    // sparse, kBFS falls back to the coarsest kept level (HRAB-equivalent
    // single-block layer), restoring single-pass behaviour rather than
    // scattering RCM over meaningless tiny groups.
    size_t kBFS = 0;
    for (size_t k = 0; k < L; ++k) { if (keepLevel[k]) { kBFS = k; break; } }
    const auto& bfsMembership = mpp[kBFS];
    const K C_bfs = levelC[kBFS];
    printf("  hier-rabbit: intra-RCM within %u level-%zu communities\n",
           (unsigned)C_bfs, kBFS);

    std::vector<std::vector<K>> commVertices(C_bfs);
    for (size_t v = 0; v < N; ++v) {
        commVertices[bfsMembership[v]].push_back(static_cast<K>(v));
    }
    std::vector<K> localIds(N, K(0));
    std::vector<size_t> vertToLocal(N, static_cast<size_t>(-1));
    #pragma omp parallel
    {
        std::vector<bool> visited;
        std::queue<K> bfsQueue;
        std::vector<K> cmOrder;
        #pragma omp for schedule(dynamic, 64)
        for (K c = 0; c < C_bfs; ++c) {
            const auto& verts = commVertices[c];
            if (verts.empty()) continue;
            for (size_t i = 0; i < verts.size(); ++i) vertToLocal[verts[i]] = i;
            intraRCM<K, NodeID_T, DestID_T>(
                verts, c, bfsMembership, degrees, g,
                vertToLocal, visited, bfsQueue, cmOrder, localIds);
            for (K v : verts) vertToLocal[v] = static_cast<size_t>(-1);
        }
    }

    // Multi-key sort: only KEPT levels contribute keys; coarsest dominates;
    // intra-finest-community BFS-from-hub local position is the final
    // tie-break (replaces a raw vertex-id break).
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        for (size_t kk = L; kk-- > 0; ) {
            if (!keepLevel[kk]) continue;
            K pa = levelPerm[kk][mpp[kk][a]];
            K pb = levelPerm[kk][mpp[kk][b]];
            if (pa != pb) return pa < pb;
        }
        return localIds[a] < localIds[b];
    });

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        newIds[order[i]] = static_cast<NodeID_T>(i);
    }

    printf("  hier-rabbit: ordered %zu vertices across %zu Leiden levels (C: ", N, L);
    for (size_t k = 0; k < L; ++k) printf("%u%s", (unsigned)levelC[k], k + 1 == L ? "" : " -> ");
    printf(")\n");
    GRAPHBREW_TRACE("orderHierarchicalLeidenRabbit: done, %zu levels", L);
}

//=============================================================================
// SECTION 16c: TILE-QUANTIZED RABBITORDER ORDERING
//=============================================================================

/**
 * @brief Tile-Quantized RabbitOrder Ordering (graphbrew:tqr)
 *
 * Combines cache-line-aligned tile quantization with RabbitOrder dendrogram
 * traversal for macro-ordering and Leiden-community-aware BFS for micro-ordering.
 *
 * Algorithm:
 * 1. QUANTIZE: Divide vertex ID space into tiles sized to match cache lines.
 *    Each tile = ceil(N / numTiles) vertices ≈ one L3 cache line worth of vertex data.
 * 2. BUILD TILE ADJACENCY: For each edge (u,v), accumulate weight between
 *    tile(u) and tile(v). This yields a coarse graph where nodes are tiles.
 * 3. RABBITORDER ON TILES: Run parallel incremental aggregation on the tile
 *    graph. This produces a dendrogram encoding which tiles should be adjacent.
 * 4. TILE PERMUTATION: DFS the RabbitOrder dendrogram to get the macro-order
 *    of tiles — which tile blocks are placed next to each other.
 * 5. COMMUNITY-AWARE BFS: Within each tile (in the permuted tile order),
 *    vertices are sub-sorted by Leiden community, then BFS within each
 *    community for maximum intra-community locality.
 *
 * Why this can beat graphbrew:hrab:
 * - graphbrew:hrab builds a super-graph over Leiden communities (~100K nodes) that
 *   have no relation to cache boundaries
 * - graphbrew:tqr builds a super-graph over tiles (~131K nodes for 8MB L3) that
 *   DIRECTLY correspond to cache line groups
 * - RabbitOrder on tiles optimizes exactly what matters: which cache-line-sized
 *   groups of vertices should be adjacent in memory
 *
 * @tparam K Community ID type
 * @tparam NodeID_T Node ID type
 * @tparam DestID_T Destination ID type (may include weight)
 */
template <typename K, typename NodeID_T, typename DestID_T>
void orderTileQuantizedRabbit(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderTileQuantizedRabbit: N=%zu", N);
    Timer phaseTimer;

    // ================================================================
    // PHASE 6 REFACTOR (2026-05-19): TQR STEPs 0+1+2+3+4 + 5a..5e are now
    // expressed as a single Stage 1 primitive call.  ~970 lines of inline
    // tile-graph construction, tile-RabbitOrder, dendrogram DFS,
    // per-community center-tile assignment, community super-graph
    // construction, community-level RabbitOrder, and composite sort have
    // been replaced by:
    //
    //     runTileRabbit<>(membership, degrees, commVertices, g, N, C, gamma)
    //
    // which is itself a composition of:
    //     chooseTileParams()
    //     buildCommunitySuperGraph()    [twice: once for tiles, once for communities]
    //     runRabbitOnTileGraph()
    //     runRabbitOnSuperCSR()
    //     composite-sort by (tilePerm[centerTile], commPermRO)
    //
    // Parity envelope vs. the pre-refactor inline path on
    // {cit-Patents, soc-pokec, hollywood-2009, com-Orkut}:
    //     -1.5% to +1.3% L3 memory accesses (PR, 3 trials, 8 threads).
    // Data: results/data/composability_parity_envelope_2026_05_19.md.
    //
    // TQR is now literally a composition:
    //     {super-graph=TileRabbit, community=Identity, intra=BFSFromHub}
    // ================================================================

    // STEP 1: Community count + per-community vertex grouping (preserved;
    // STEP 5f and the final vertex relabelling need commVertices / isolated).
    K maxComm = 0;
    for (size_t v = 0; v < N; ++v) maxComm = std::max(maxComm, membership[v]);
    const size_t C = static_cast<size_t>(maxComm) + 1;

    std::vector<K> isolated;
    std::vector<std::vector<K>> commVertices(C);
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(static_cast<K>(v));
        } else {
            commVertices[membership[v]].push_back(static_cast<K>(v));
        }
    }
    printf("  tqr: %zu Leiden communities, %zu isolated vertices\n", C, isolated.size());

    // Stage 1 (tile_rabbit composition) -> commPerm
    phaseTimer.Start();
    const float sgGammaTqr = static_cast<float>(config.superGraphResolution);
    auto commPerm = runTileRabbit<K, NodeID_T, DestID_T>(
        membership, degrees, commVertices, g, N, static_cast<K>(C), sgGammaTqr);
    phaseTimer.Stop();
    printf("  tqr: super-graph=tile_rabbit composed in %.4fs\n", phaseTimer.Seconds());

    // Derive sortedComms / vertexOffsets / commToSortedIdx from commPerm.
    std::vector<K> sortedComms(C);
    std::iota(sortedComms.begin(), sortedComms.end(), K(0));
    std::sort(sortedComms.begin(), sortedComms.end(),
              [&](K a, K b) { return commPerm[a] < commPerm[b]; });

    std::vector<size_t> vertexOffsets(C + 1, 0);
    for (size_t i = 0; i < C; ++i) {
        K c = sortedComms[i];
        vertexOffsets[i + 1] = vertexOffsets[i] + commVertices[c].size();
    }
    size_t totalActive = vertexOffsets[C];
    printf("  tqr: %zu active vertices, %zu communities (tile+rabbit ordered)\n",
           totalActive, C);

    std::vector<K> commToSortedIdx(C);
    for (size_t i = 0; i < C; ++i) commToSortedIdx[sortedComms[i]] = static_cast<K>(i);

    // 5f: BFS within each community for final vertex ordering
    phaseTimer.Start();

    std::vector<K> localIds(N, static_cast<K>(-1));

    #pragma omp parallel
    {
        std::queue<K> bfsQueue;

        #pragma omp for schedule(dynamic, 1)
        for (size_t ci = 0; ci < C; ++ci) {
            K c = sortedComms[ci];
            auto& verts = commVertices[c];
            if (verts.empty()) continue;

            if (verts.size() == 1) {
                localIds[verts[0]] = 0;
                continue;
            }

            std::vector<bool> visited(verts.size(), false);
            std::unordered_map<K, size_t> vertToLocal;
            for (size_t i = 0; i < verts.size(); ++i) {
                vertToLocal[verts[i]] = i;
            }

            K startV = verts[0];
            K maxDeg = degrees[verts[0]];
            for (K v : verts) {
                if (degrees[v] > maxDeg) {
                    maxDeg = degrees[v];
                    startV = v;
                }
            }

            K localId = 0;
            bfsQueue.push(startV);
            visited[vertToLocal[startV]] = true;

            while (!bfsQueue.empty()) {
                K u = bfsQueue.front();
                bfsQueue.pop();

                localIds[u] = localId++;

                for (auto neighbor : g.out_neigh(u)) {
                    NodeID_T v;
                    if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                        v = neighbor;
                    } else {
                        v = neighbor.v;
                    }

                    if (membership[v] != c) continue;

                    auto it = vertToLocal.find(static_cast<K>(v));
                    if (it != vertToLocal.end() && !visited[it->second]) {
                        visited[it->second] = true;
                        bfsQueue.push(static_cast<K>(v));
                    }
                }
            }

            for (size_t i = 0; i < verts.size(); ++i) {
                if (!visited[i]) {
                    localIds[verts[i]] = localId++;
                }
            }
        }
    }

    // Compute global IDs
    #pragma omp parallel for
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] > 0) {
            K c = membership[v];
            K sortedIdx = commToSortedIdx[c];
            newIds[v] = static_cast<NodeID_T>(vertexOffsets[sortedIdx] + localIds[v]);
        }
    }

    size_t isolatedStart = totalActive;
    for (size_t i = 0; i < isolated.size(); ++i) {
        newIds[isolated[i]] = static_cast<NodeID_T>(isolatedStart + i);
    }

    phaseTimer.Stop();
    printf("  tqr: BFS within communities in %.4fs\n", phaseTimer.Seconds());
    printf("  tqr: %zu communities, %zu active, %zu isolated, total=%.4fs\n",
           C, totalActive, isolated.size(),
           0.0);  // total time tracked externally

    GRAPHBREW_TRACE("orderTileQuantizedRabbit: %zu communities, %zu isolated",
               C, isolated.size());
}

//=============================================================================
// SECTION 16e: COMMUNITY MERGING FOR CACHE LOCALITY
//=============================================================================

/**
 * Merge small Leiden communities into larger ones for better cache locality
 *
 * Problem: Leiden produces many fine-grained communities (optimized for modularity)
 *          but cache performance needs large contiguous blocks.
 *
 * Solution: Merge communities based on inter-community edge weight until we
 *           reach a target community count (similar to RabbitOrder's ~N/1000).
 *
 * Algorithm:
 * 1. Build inter-community edge weights (which communities are most connected)
 * 2. Use union-find to merge communities greedily by strongest connection
 * 3. Continue until target count reached or no more beneficial merges
 *
 * @param membership Input/output: community membership for each vertex
 * @param g Original graph (for edge weights)
 * @param targetComms Target number of communities (0 = auto: N/avgDegree)
 * @return Final number of communities after merging
 */
template <typename K, typename NodeID_T, typename DestID_T>
size_t mergeCommunities(
    std::vector<K>& membership,
    const CSRGraph<NodeID_T, DestID_T, true>& g,
    size_t targetComms = 0) {

    const size_t N = g.num_nodes();

    // Find current community count and max ID
    K maxComm = 0;
    for (size_t v = 0; v < N; ++v) {
        maxComm = std::max(maxComm, membership[v]);
    }
    const size_t numComm = static_cast<size_t>(maxComm) + 1;

    // Auto-compute target: aim for ~1000 communities like RabbitOrder
    if (targetComms == 0) {
        // Target similar to RabbitOrder: N/avgCommunitySize where avgSize ~ 100-500
        targetComms = std::max(size_t(100), N / 500);
    }

    // If already at or below target, no merging needed
    if (numComm <= targetComms) {
        printf("  merge: %zu communities already <= target %zu\n", numComm, targetComms);
        return numComm;
    }

    printf("  merge: %zu -> %zu target communities\n", numComm, targetComms);

    // Step 1: Build community vertex lists and sizes
    std::vector<size_t> commSize(numComm, 0);
    for (size_t v = 0; v < N; ++v) {
        commSize[membership[v]]++;
    }

    // Step 2: Build inter-community edge weights using parallel aggregation
    struct CommEdge {
        K c1, c2;       // Community pair (c1 < c2)
        double weight;  // Total edge weight between them

        bool operator<(const CommEdge& other) const {
            return weight < other.weight;  // Max-heap: highest weight first
        }
    };

    // Parallel computation of inter-community edges
    const int numThreads = omp_get_max_threads();
    std::vector<std::unordered_map<uint64_t, double>> threadMaps(numThreads);

    auto packPair = [](K c1, K c2) -> uint64_t {
        if (c1 > c2) std::swap(c1, c2);
        return (static_cast<uint64_t>(c1) << 32) | c2;
    };

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localMap = threadMaps[tid];

        #pragma omp for schedule(dynamic, 1024)
        for (size_t u = 0; u < N; ++u) {
            K cu = membership[u];
            for (auto neighbor : g.out_neigh(u)) {
                NodeID_T v;
                double w;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>) {
                    v = neighbor;
                    w = 1.0;
                } else {
                    v = neighbor.v;
                    w = static_cast<double>(neighbor.w);
                }

                K cv = membership[v];
                if (cu != cv) {
                    uint64_t key = packPair(cu, cv);
                    localMap[key] += w;
                }
            }
        }
    }

    // Merge thread-local maps
    std::unordered_map<uint64_t, double> globalMap;
    for (auto& localMap : threadMaps) {
        for (auto& [key, weight] : localMap) {
            globalMap[key] += weight;
        }
    }

    // Build priority queue of community edges (MAX heap - highest weight first)
    std::priority_queue<CommEdge> pq;
    for (auto& [key, weight] : globalMap) {
        K c1 = static_cast<K>(key >> 32);
        K c2 = static_cast<K>(key & 0xFFFFFFFF);
        pq.push({c1, c2, weight});
    }

    // Step 3: Union-find for merging
    std::vector<K> parent(numComm);
    std::iota(parent.begin(), parent.end(), K(0));

    std::function<K(K)> find = [&](K x) -> K {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    };

    auto unite = [&](K x, K y) -> bool {
        K px = find(x);
        K py = find(y);
        if (px == py) return false;
        // Merge smaller into larger
        if (commSize[px] < commSize[py]) std::swap(px, py);
        parent[py] = px;
        commSize[px] += commSize[py];
        return true;
    };

    // Step 4: Greedily merge communities by strongest connection
    size_t currentComms = numComm;
    size_t mergeCount = 0;

    while (currentComms > targetComms && !pq.empty()) {
        auto [c1, c2, weight] = pq.top();
        pq.pop();

        // Check if still different communities after previous merges
        K p1 = find(c1);
        K p2 = find(c2);
        if (p1 == p2) continue;

        // Merge!
        unite(p1, p2);
        currentComms--;
        mergeCount++;
    }

    // Step 5: If we haven't reached target (disconnected components),
    // merge remaining small communities arbitrarily
    if (currentComms > targetComms) {
        // Find all root communities sorted by size
        std::vector<std::pair<size_t, K>> rootsBySize;
        for (size_t c = 0; c < numComm; ++c) {
            if (find(static_cast<K>(c)) == static_cast<K>(c)) {
                rootsBySize.emplace_back(commSize[c], static_cast<K>(c));
            }
        }
        std::sort(rootsBySize.begin(), rootsBySize.end());  // Smallest first

        // Merge smallest communities into each other until target reached
        size_t idx = 0;
        while (currentComms > targetComms && idx + 1 < rootsBySize.size()) {
            K small = rootsBySize[idx].second;
            K next = rootsBySize[idx + 1].second;
            if (unite(small, next)) {
                currentComms--;
                mergeCount++;
                // Update size in list
                rootsBySize[idx + 1].first += rootsBySize[idx].first;
            }
            idx++;
        }
    }

    // Step 6: Renumber communities to be contiguous
    std::vector<K> newCommId(numComm, static_cast<K>(-1));
    K nextId = 0;
    for (size_t c = 0; c < numComm; ++c) {
        K root = find(static_cast<K>(c));
        if (newCommId[root] == static_cast<K>(-1)) {
            newCommId[root] = nextId++;
        }
    }

    // Step 7: Update membership
    #pragma omp parallel for
    for (size_t v = 0; v < N; ++v) {
        K oldComm = membership[v];
        K root = find(oldComm);
        membership[v] = newCommId[root];
    }

    printf("  merge: %zu merges performed, final %zu communities\n", mergeCount, currentComms);

    return currentComms;
}

//=============================================================================
// SECTION 17: ORDERING - DENDROGRAM DFS
//=============================================================================

/**
 * DFS traversal of dendrogram for ordering
 *
 * Produces excellent locality by keeping related vertices close
 */
template <typename K, typename NodeID_T>
void orderDendrogramDFS(
    pvector<NodeID_T>& newIds,
    const GraphBrewResult<K>& result,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderDendrogramDFS: N=%zu, roots=%zu", N, result.roots.size());

    if (result.dendrogram.empty() || result.roots.empty()) {
        // Fallback to hierarchical sort
        orderHierarchicalSort<K, NodeID_T>(newIds, result, degrees, N, config);
        return;
    }

    std::vector<K> order;
    order.reserve(N);

    // DFS from each root
    for (K root : result.roots) {
        std::stack<K> stack;
        stack.push(root);

        while (!stack.empty()) {
            K node = stack.top();
            stack.pop();

            if (node >= result.dendrogram.size()) continue;

            const auto& dnode = result.dendrogram[node];

            if (dnode.level == 0) {
                // Leaf: add vertices (sorted by degree)
                std::vector<K> vertices = dnode.children;
                std::sort(vertices.begin(), vertices.end(),
                    [&](K a, K b) { return degrees[a] > degrees[b]; });
                for (K v : vertices) {
                    if (v < N) order.push_back(v);
                }
            } else {
                // Internal: push children in reverse (for DFS order)
                // Sort children by size (larger first) for better locality
                std::vector<K> children = dnode.children;
                std::sort(children.begin(), children.end(),
                    [&](K a, K b) {
                        return result.dendrogram[a].size > result.dendrogram[b].size;
                    });
                for (auto it = children.rbegin(); it != children.rend(); ++it) {
                    stack.push(*it);
                }
            }
        }
    }

    // Assign IDs
    #pragma omp parallel for
    for (size_t i = 0; i < order.size(); ++i) {
        newIds[order[i]] = static_cast<NodeID_T>(i);
    }

    // Handle any missed vertices
    NodeID_T nextId = order.size();
    for (size_t v = 0; v < N; ++v) {
        if (newIds[v] == static_cast<NodeID_T>(-1)) {
            newIds[v] = nextId++;
        }
    }
}

//=============================================================================
// SECTION 18: ORDERING - DENDROGRAM BFS
//=============================================================================

/**
 * BFS traversal of dendrogram for ordering
 *
 * Groups vertices by level first, then by community
 */
template <typename K, typename NodeID_T>
void orderDendrogramBFS(
    pvector<NodeID_T>& newIds,
    const GraphBrewResult<K>& result,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderDendrogramBFS: N=%zu", N);

    if (result.dendrogram.empty() || result.roots.empty()) {
        orderHierarchicalSort<K, NodeID_T>(newIds, result, degrees, N, config);
        return;
    }

    std::vector<K> order;
    order.reserve(N);

    std::queue<K> queue;
    for (K root : result.roots) {
        queue.push(root);
    }

    while (!queue.empty()) {
        K node = queue.front();
        queue.pop();

        if (node >= result.dendrogram.size()) continue;

        const auto& dnode = result.dendrogram[node];

        if (dnode.level == 0) {
            std::vector<K> vertices = dnode.children;
            std::sort(vertices.begin(), vertices.end(),
                [&](K a, K b) { return degrees[a] > degrees[b]; });
            for (K v : vertices) {
                if (v < N) order.push_back(v);
            }
        } else {
            for (K child : dnode.children) {
                queue.push(child);
            }
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < order.size(); ++i) {
        newIds[order[i]] = static_cast<NodeID_T>(i);
    }

    NodeID_T nextId = order.size();
    for (size_t v = 0; v < N; ++v) {
        if (newIds[v] == static_cast<NodeID_T>(-1)) {
            newIds[v] = nextId++;
        }
    }
}

//=============================================================================
// SECTION 19: ORDERING - COMMUNITY SORT
//=============================================================================

/**
 * Simple community-based sort
 *
 * Zero-degree nodes are grouped at the END for better cache locality.
 */
template <typename K, typename NodeID_T>
void orderCommunitySort(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderCommunitySort: N=%zu", N);

    // Separate zero-degree (isolated) nodes
    std::vector<size_t> active, isolated;
    active.reserve(N);
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(v);
        } else {
            active.push_back(v);
        }
    }

    auto comparator = [&](size_t a, size_t b) {
        if (membership[a] != membership[b]) {
            return membership[a] < membership[b];
        }
        return degrees[a] > degrees[b];
    };

    if (config.useParallelSort) {
        __gnu_parallel::sort(active.begin(), active.end(), comparator);
    } else {
        std::sort(active.begin(), active.end(), comparator);
    }

    // Assign IDs: active nodes first, isolated nodes at the end
    #pragma omp parallel for
    for (size_t i = 0; i < active.size(); ++i) {
        newIds[active[i]] = static_cast<NodeID_T>(i);
    }

    NodeID_T isolatedStart = static_cast<NodeID_T>(active.size());
    for (size_t i = 0; i < isolated.size(); ++i) {
        newIds[isolated[i]] = isolatedStart + static_cast<NodeID_T>(i);
    }

    GRAPHBREW_TRACE("orderCommunitySort: %zu active, %zu isolated", active.size(), isolated.size());
}

//=============================================================================
// SECTION 20: ORDERING - HUB CLUSTER
//=============================================================================

/**
 * Hub-first ordering within communities
 *
 * Zero-degree nodes are grouped at the END for better cache locality.
 */
template <typename K, typename NodeID_T>
void orderHubCluster(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderHubCluster: N=%zu", N);

    // Separate isolated (zero-degree) nodes first
    std::vector<size_t> isolated;
    std::vector<K> activeDegrees;
    activeDegrees.reserve(N);

    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) {
            isolated.push_back(v);
        } else {
            activeDegrees.push_back(degrees[v]);
        }
    }

    // Find hub threshold (top 1% of ACTIVE nodes)
    if (activeDegrees.empty()) {
        // All nodes are isolated - just assign sequentially
        for (size_t i = 0; i < N; ++i) {
            newIds[i] = static_cast<NodeID_T>(i);
        }
        return;
    }

    std::sort(activeDegrees.begin(), activeDegrees.end(), std::greater<K>());
    K hubThreshold = activeDegrees[std::min(activeDegrees.size() / 100, activeDegrees.size() - 1)];

    // Separate hubs and non-hubs (excluding isolated)
    std::vector<size_t> hubs, nonHubs;
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] == 0) continue;  // Already in isolated
        if (degrees[v] >= hubThreshold) {
            hubs.push_back(v);
        } else {
            nonHubs.push_back(v);
        }
    }

    // Sort hubs by community, then degree
    std::sort(hubs.begin(), hubs.end(), [&](size_t a, size_t b) {
        if (membership[a] != membership[b]) return membership[a] < membership[b];
        return degrees[a] > degrees[b];
    });

    // Sort non-hubs by community, then degree
    std::sort(nonHubs.begin(), nonHubs.end(), [&](size_t a, size_t b) {
        if (membership[a] != membership[b]) return membership[a] < membership[b];
        return degrees[a] > degrees[b];
    });

    // Assign: hubs first, then non-hubs, then isolated at the end
    NodeID_T id = 0;
    for (size_t v : hubs) {
        newIds[v] = id++;
    }
    for (size_t v : nonHubs) {
        newIds[v] = id++;
    }
    for (size_t v : isolated) {
        newIds[v] = id++;
    }

    GRAPHBREW_TRACE("orderHubCluster: %zu hubs, %zu non-hubs, %zu isolated", hubs.size(), nonHubs.size(), isolated.size());
}

//=============================================================================
// SECTION 20b: DBG ORDERING (within communities)
//=============================================================================

/**
 * DBG (Degree-Based Grouping) ordering within communities
 * Groups vertices by degree buckets, respecting community boundaries
 */
template <typename K, typename NodeID_T>
void orderDBG(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderDBG: N=%zu", N);

    // Find max community
    K maxComm = 0;
    for (size_t v = 0; v < N; ++v) {
        maxComm = std::max(maxComm, membership[v]);
    }
    size_t numComm = maxComm + 1;

    // Compute average degree
    K totalDegree = 0;
    for (size_t v = 0; v < N; ++v) totalDegree += degrees[v];
    K avgDegree = totalDegree / N;

    // Define bucket thresholds (logarithmic scaling)
    const int numBuckets = 8;
    K bucketThreshold[numBuckets] = {
        avgDegree / 2,
        avgDegree,
        avgDegree * 2,
        avgDegree * 4,
        avgDegree * 8,
        avgDegree * 16,
        avgDegree * 32,
        std::numeric_limits<K>::max()
    };

    // Per-community buckets: buckets[comm][bucket] = list of vertices
    std::vector<std::vector<std::vector<size_t>>> buckets(numComm,
        std::vector<std::vector<size_t>>(numBuckets));

    // Distribute vertices into buckets
    for (size_t v = 0; v < N; ++v) {
        K comm = membership[v];
        K deg = degrees[v];
        for (int b = 0; b < numBuckets; ++b) {
            if (deg <= bucketThreshold[b]) {
                buckets[comm][b].push_back(v);
                break;
            }
        }
    }

    // Assign IDs: process communities, within each process buckets high-to-low
    NodeID_T id = 0;
    for (size_t c = 0; c < numComm; ++c) {
        // High-degree buckets first (reverse order)
        for (int b = numBuckets - 1; b >= 0; --b) {
            for (size_t v : buckets[c][b]) {
                newIds[v] = id++;
            }
        }
    }
}

/**
 * DBG ordering globally (ignoring communities, applied after clustering)
 * Uses degree buckets across all vertices
 */
template <typename K, typename NodeID_T>
void orderDBGGlobal(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderDBGGlobal: N=%zu", N);

    // Compute average degree
    K totalDegree = 0;
    for (size_t v = 0; v < N; ++v) totalDegree += degrees[v];
    K avgDegree = totalDegree / N;

    // Define bucket thresholds
    const int numBuckets = 8;
    K bucketThreshold[numBuckets] = {
        avgDegree / 2, avgDegree, avgDegree * 2, avgDegree * 4,
        avgDegree * 8, avgDegree * 16, avgDegree * 32,
        std::numeric_limits<K>::max()
    };

    // Global buckets
    std::vector<std::vector<size_t>> buckets(numBuckets);

    for (size_t v = 0; v < N; ++v) {
        K deg = degrees[v];
        for (int b = 0; b < numBuckets; ++b) {
            if (deg <= bucketThreshold[b]) {
                buckets[b].push_back(v);
                break;
            }
        }
    }

    // Assign IDs: high-degree buckets first
    NodeID_T id = 0;
    for (int b = numBuckets - 1; b >= 0; --b) {
        // Within bucket, sort by community for some locality
        std::sort(buckets[b].begin(), buckets[b].end(), [&](size_t a, size_t b) {
            return membership[a] < membership[b];
        });
        for (size_t v : buckets[b]) {
            newIds[v] = id++;
        }
    }
}

//=============================================================================
// SECTION 20c: CORDER ORDERING (hot/cold partitioning)
//=============================================================================

/**
 * Corder ordering within communities
 * Separates hot (high-degree) and cold (low-degree) vertices within each community
 */
template <typename K, typename NodeID_T>
void orderCorder(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderCorder: N=%zu", N);

    // Find max community
    K maxComm = 0;
    for (size_t v = 0; v < N; ++v) {
        maxComm = std::max(maxComm, membership[v]);
    }
    size_t numComm = maxComm + 1;

    // Compute average degree
    K totalDegree = 0;
    for (size_t v = 0; v < N; ++v) totalDegree += degrees[v];
    K avgDegree = totalDegree / N;

    // Per-community hot/cold lists
    std::vector<std::vector<size_t>> hotLists(numComm);
    std::vector<std::vector<size_t>> coldLists(numComm);

    for (size_t v = 0; v < N; ++v) {
        K comm = membership[v];
        if (degrees[v] > avgDegree) {
            hotLists[comm].push_back(v);
        } else {
            coldLists[comm].push_back(v);
        }
    }

    // Assign IDs: within each community, hot first then cold
    NodeID_T id = 0;
    for (size_t c = 0; c < numComm; ++c) {
        // Sort hot by degree descending
        std::sort(hotLists[c].begin(), hotLists[c].end(), [&](size_t a, size_t b) {
            return degrees[a] > degrees[b];
        });
        for (size_t v : hotLists[c]) {
            newIds[v] = id++;
        }

        // Sort cold by degree descending
        std::sort(coldLists[c].begin(), coldLists[c].end(), [&](size_t a, size_t b) {
            return degrees[a] > degrees[b];
        });
        for (size_t v : coldLists[c]) {
            newIds[v] = id++;
        }
    }
}

/**
 * Corder ordering globally (interleaved hot/cold partitions)
 * Creates cache-line sized partitions with hot vertices at start of each
 */
template <typename K, typename NodeID_T>
void orderCorderGlobal(
    pvector<NodeID_T>& newIds,
    const std::vector<K>& membership,
    const std::vector<K>& degrees,
    size_t N,
    const GraphBrewConfig& config) {

    GRAPHBREW_TRACE("orderCorderGlobal: N=%zu", N);

    // Compute average degree
    K totalDegree = 0;
    for (size_t v = 0; v < N; ++v) totalDegree += degrees[v];
    K avgDegree = totalDegree / N;

    // Separate hot and cold
    std::vector<size_t> hot, cold;
    for (size_t v = 0; v < N; ++v) {
        if (degrees[v] > avgDegree) {
            hot.push_back(v);
        } else {
            cold.push_back(v);
        }
    }

    // Sort by community within each group
    auto sortByComm = [&](size_t a, size_t b) {
        if (membership[a] != membership[b]) return membership[a] < membership[b];
        return degrees[a] > degrees[b];
    };
    std::sort(hot.begin(), hot.end(), sortByComm);
    std::sort(cold.begin(), cold.end(), sortByComm);

    // Interleave into partitions
    const size_t partitionSize = 1024;
    size_t numPartitions = (N + partitionSize - 1) / partitionSize;
    size_t hotPerPart = (hot.size() + numPartitions - 1) / numPartitions;
    size_t coldPerPart = partitionSize - hotPerPart;

    size_t nextId = 0;
    size_t hi = 0, ci = 0;

    for (size_t p = 0; p < numPartitions && nextId < N; ++p) {
        // Hot vertices first in partition
        for (size_t i = 0;
             i < hotPerPart && hi < hot.size() && nextId < N;
             ++i) {
            newIds[hot[hi++]] = static_cast<NodeID_T>(nextId++);
        }
        // Cold vertices fill rest of partition
        for (size_t i = 0;
             i < coldPerPart && ci < cold.size() && nextId < N;
             ++i) {
            newIds[cold[ci++]] = static_cast<NodeID_T>(nextId++);
        }
    }

    // Any remaining
    while (hi < hot.size())
        newIds[hot[hi++]] = static_cast<NodeID_T>(nextId++);
    while (ci < cold.size())
        newIds[cold[ci++]] = static_cast<NodeID_T>(nextId++);
}

//=============================================================================

#endif  // GRAPHBREW_REORDER_GRAPHBREW_DIAGNOSTICS_H_
