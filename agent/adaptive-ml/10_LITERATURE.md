# Literature → GraphBrew Feature Map

This document maps five SOTA papers to concrete code locations in GraphBrew's
AdaptiveOrder-ML. It serves as the evidence base for proposals in
[06_SOTA_IDEAS.md](06_SOTA_IDEAS.md) and ablations in
[05_ABLATIONS_AND_EXPERIMENTS.md](05_ABLATIONS_AND_EXPERIMENTS.md).

---

## Paper Summaries

### 1. IISWC'18 — "When is Graph Reordering an Optimization?"
**Authors:** Balaji & Lucia (CMU), IEEE IISWC 2018

**Core insight:** Graph reordering is not universally beneficial. The paper
introduces a lightweight metric ("packing factor") that measures spatial
locality of a graph's current ordering, and shows it predicts whether
reordering will help — without actually performing the reorder.

**Key contributions:**
- Packing factor: fraction of a hub's neighbours whose IDs are within a
  locality window of the hub's ID. High packing → already well-ordered.
- Hub sorting: relabelling high-degree vertices first is a near-zero-cost
  reordering that captures most of the benefit on some graphs.
- Selective reordering: only reorder if packing factor is low AND the graph
  is large enough to benefit (working set > cache).

**Status in GraphBrew:** ✅ **Fully integrated**
| Concept | Code Location | Implementation |
|---------|--------------|----------------|
| Packing factor | `reorder_types.h` ~L4120 | Sampled from top-degree hubs, `locality_window = max(64, N/100)` |
| `w_packing_factor` | `PerceptronWeights` ~L1552 | Linear weight in perceptron scoring |
| Hub sorting | Algorithm 3 (HubSort) | Available as candidate algorithm |
| `w_pf_x_wsr` | `PerceptronWeights` ~L1565 | Quadratic interaction: packing × working_set |

---

### 2. GoGraph — "Accelerating Graph Processing via Graph Ordering"
**Authors:** Zhou et al., 2024

**Core insight:** For iterative graph algorithms (PageRank, SSSP), vertex ordering
where sources precede destinations (high "forward-edge fraction" M(σ)) reduces
the number of iterations needed for convergence. GoGraph proposes a divide-and-
conquer reordering that maximizes M(σ), achieving 1.83× average speedup.

**Key contributions:**
- Forward-edge metric M(σ): count of edges (u,v) where σ(u) < σ(v) — measures
  how well information flows forward in a single pass.
- Divide-and-conquer optimizer: recursively split graph, order each partition
  to maximize forward edges, compose orderings.
- Convergence acceleration: distinct from cache locality — this is about
  reducing the number of iterations, not cache misses per iteration.

**Status in GraphBrew:** ⚠️ **Feature integrated, algorithm NOT integrated**
| Concept | Code Location | Implementation |
|---------|--------------|----------------|
| Forward edge fraction | `reorder_types.h` ~L4150 | Sampled from 2000 vertices, counts (u,v) where u < v |
| `w_forward_edge_fraction` | `PerceptronWeights` ~L1554 | Linear weight in perceptron |
| `w_fef_convergence` | `PerceptronWeights` ~L1695 | Bonus for PR/SSSP only (convergence-aware) |
| GoGraph reordering algo | — | **NOT implemented** as a candidate algorithm |

**Opportunity:** Implement GoGraph's divide-and-conquer as a new candidate
algorithm. The perceptron can then select it when FEF matters most.
See [06_SOTA_IDEAS.md](06_SOTA_IDEAS.md) §3.4.

---

### 3. Rabbit Order — "Rabbit Order: Just-in-time Parallel Reordering"
**Authors:** Arai, Fujiwara, Taura (U. Tokyo), IEEE IPDPS 2016

**Core insight:** Hierarchical community structure maps naturally to hierarchical
caches. Rabbit Order uses Louvain-like agglomerative clustering to build a
community dendrogram, then assigns vertex IDs so that vertices in the same
community get contiguous IDs. Parallelised via incremental aggregation.

**Key contributions:**
- Hierarchical community ordering: communities at each dendrogram level map
  to cache levels (L1 inner, L2 outer, L3 graph-level).
- Parallel incremental aggregation: avoids full graph contraction at each level;
  instead incrementally merges communities.
- JIT reordering: fast enough to amortize in a single graph traversal on
  many benchmarks.

**Status in GraphBrew:** ✅ **Fully integrated as Algorithm 8**
| Concept | Code Location | Implementation |
|---------|--------------|----------------|
| RabbitOrder | Algorithm 8 (`-o 8`) | Full C++ implementation via Boost dependency |
| Boost variant | `rabbit:boost` CSR variant | Available as candidate in AdaptiveOrder |
| Community-based ordering concept | `reorder_adaptive.h` | AdaptiveOrder uses Leiden (not Louvain) for community detection, but the hierarchical principle is the same |

---

### 4. P-OPT — "P-OPT: Practical Optimal Cache Replacement for Graph Analytics"
**Authors:** Balaji et al. (CMU), IEEE HPCA 2021

**Core insight:** For pull-based graph algorithms, the graph transpose encodes
future reuse information: if vertex v has an edge to vertex u in the transpose,
then u's data will be needed when processing v. This enables near-optimal
(Belady's MIN) cache replacement without lookahead — just follow the transpose.

**Key contributions:**
- Working set ratio: `graph_bytes / LLC_size` — when > 1, cache replacement
  policy matters significantly. When >> 3, even good replacement can't help
  much and reordering becomes essential.
- Transpose-guided replacement: use CSC (transpose) to predict future reuse
  distance for each cache line.
- Epoch-quantized rereference matrix: practical approximation of Belady's MIN.

**Status in GraphBrew:** ⚠️ **Feature integrated, replacement policy NOT integrated**
| Concept | Code Location | Implementation |
|---------|--------------|----------------|
| Working set ratio | `reorder_types.h` ~L4170 | `graph_bytes / LLC_size`, LLC detected via `sysconf` |
| `w_working_set_ratio` | `PerceptronWeights` ~L1556 | Linear weight in perceptron |
| `w_pf_x_wsr` | `PerceptronWeights` ~L1565 | Quadratic interaction with packing factor |
| P-OPT cache replacement | — | **NOT implemented** (requires hardware support or simulation) |

**Opportunity:** The working_set_ratio feature is already the most useful
takeaway. A secondary feature — sampled average reuse distance from transpose
— could improve perceptron accuracy. See [06_SOTA_IDEAS.md](06_SOTA_IDEAS.md) §3.2.

---

### 5. DON-RL — "Graph Ordering: Towards the Optimal by Learning"
**Authors:** Zhao, Ma, Guo, Zhao (Zhejiang U. + UCR), 2024

**Core insight:** Replace the greedy heuristic in Gorder (which constructs
vertex orderings by greedily maximizing a locality window score Q(S,v)) with a
learned Deep Order Network (DON). The DON uses a DeepSets architecture
(permutation-invariant) to learn Q_θ(S,v;Θ) from partial orderings. Training
uses episodic RL where state = window contents + vertex features, action = next
vertex to place, reward = incremental locality gain ΔF(σ).

**Key contributions:**
- **Deep Order Network (DON):** DeepSets-based Q-function. Per-vertex features
  [degree, max/avg/min-neighbour-degree, neighbours-in-window] are embedded by
  φ (MLP), SUM-pooled over the window, then decoded by ρ (MLP) → Q-value.
  Guarantees permutation invariance over the window set.
- **Policy Network for sampling:** Critical innovation — uniform training vertex
  sampling wastes time on unimportant vertices. The Policy Network dynamically
  focuses training on high-significance vertices. Without it, DON barely beats
  Gorder. With it, 5-35% improvement on F(σ).
- **Vertex significance skewness:** Key finding — on many graphs, >2/3 of vertices
  contribute ≤0 to F(σ) improvement. High-degree hubs and bridge vertices
  dominate ordering quality. This skewness varies across graph families.
- **Transfer learning:** DON trained on small graphs (1K-10K vertices) transfers
  to larger graphs because degree-based features are scale-invariant. Validates
  GraphBrew's type-transfer design.
- **Results:** DON-RL outperforms Gorder by 1.8-35% on F(σ), and approaches
  optimal on small graphs. The improvement is largest on social networks
  (high hub dominance) and smallest on road networks (uniform structure).

**Status in GraphBrew:** ⚠️ **Insights partially applicable, direct integration impractical**
| Concept | Code Location | Implementation |
|---------|--------------|----------------|
| Learned ordering (DON) | — | Not implemented (O(V²) runtime prohibitive) |
| RL training | — | Not implemented (needs PyTorch, days of training) |
| Greedy locality heuristic | Gorder (Algorithm 9, `-o 9`) | Gorder available as candidate |
| Vertex significance insight | — | **Actionable** as new perceptron feature `w_significance_skew` |
| Window neighbor overlap | — | **Actionable** as new feature `w_window_overlap` |
| Significance-weighted training | — | **Actionable** in Python training pipeline |

**Why direct DON integration is impractical:**
1. Runtime: DON evaluates MLP per candidate per step → O(V²) for ordering.
   Gorder is already the slowest algorithm (9.5s on cit-Patents). DON would
   be 10-100× slower.
2. Dependencies: GraphBrew is header-only C++17 with zero ML framework deps.
3. Marginal gain: DON-RL's 1.8-5% F(σ) improvement on large graphs translates
   to <1% cache-miss improvement on top of Gorder.

**Practical opportunities (3 DON-RL-inspired features, no arch changes):**
1. `w_significance_skew`: Coefficient of variation of per-vertex locality
   contributions. High → hub-dominated → favour HubSort/HubCluster.
2. `w_window_overlap`: Mean fraction of a vertex's neighbours within a
   locality window. Generalizes packing_factor with explicit window.
3. Significance-weighted training: Weight perceptron training examples by
   how much algorithm choice matters (speedup range).

See [../docs/DON-RL_Analysis.md](../docs/DON-RL_Analysis.md) for the full
deep analysis and [06_SOTA_IDEAS.md](06_SOTA_IDEAS.md) §3.1 for proposals.

---

## Integration Summary

| Paper | Year | Feature | Algorithm | Used in Perceptron |
|-------|------|---------|-----------|-------------------|
| IISWC'18 | 2018 | ✅ packing_factor | ✅ HubSort (algo 3) | ✅ w_packing_factor, w_pf_x_wsr |
| GoGraph | 2024 | ✅ forward_edge_fraction | ❌ not implemented | ✅ w_forward_edge_fraction, w_fef_convergence |
| Rabbit Order | 2016 | — | ✅ RabbitOrder (algo 8) | — (selected as candidate) |
| P-OPT | 2021 | ✅ working_set_ratio | ❌ hardware-level | ✅ w_working_set_ratio, w_pf_x_wsr |
| DON-RL | 2024 | ⚠️ insights actionable | ❌ runtime prohibitive | ⚠️ 3 new features proposed |

**Key takeaway:** GraphBrew integrates features from 3 of 5 papers
(IISWC'18, GoGraph, P-OPT) and algorithms from 1 of 5 (Rabbit Order).
DON-RL's direct integration is impractical (O(V²) runtime, PyTorch dependency),
but three DON-RL-inspired perceptron features are low-effort P0 enhancements.
See [../docs/DON-RL_Analysis.md](../docs/DON-RL_Analysis.md).
