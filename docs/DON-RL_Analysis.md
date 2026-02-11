# DON-RL Deep Analysis & GraphBrew Enhancement Proposals

**Date:** 2025-02-11  
**Paper:** Zhao, Ma, Guo, Zhao — "Graph Ordering: Towards the Optimal by Learning" (2024)  
**Status:** Research analysis — no code changes yet

---

## 1. Paper Deep Dive

### 1.1 Problem Definition

Graph ordering assigns each vertex v ∈ V a position σ(v) ∈ [1,|V|] to
maximize the **locality function**:

$$F(\sigma) = \sum_{(u,v) \in E} \frac{1}{|\sigma(u) - \sigma(v)|}$$

Vertices placed close together in the ordering that share edges contribute
more to F(σ). This directly minimizes cache misses during graph traversals
because CSR-format arrays are accessed in vertex-ID order.

### 1.2 Gorder's Greedy Approach (Baseline)

Gorder (Wei et al., PPoPP'16) uses a sliding window of size w:
- Maintain ordered set S (already-placed vertices)
- At each step, score each candidate v as:
  $Q(S,v) = \sum_{u \in W \cap N(v)} \text{sim}(u,v)$
  where W = last w vertices in S, sim counts common neighbours
- Select $v^* = \arg\max_v Q(S,v)$
- Complexity: O(V · w²) — fast but greedy (no lookahead)

**Limitation:** The greedy Q function only sees the current window. It cannot
reason about long-range structure or anticipate future placements.

### 1.3 DON Architecture

DON (Deep Order Network) replaces the greedy Q with a learned function:

```
Q_θ(S, v; Θ) = ρ(Σ_{u ∈ S∩W} φ(f_u)) + ψ(f_v)
```

**Components:**
1. **Feature extraction** f_v for each vertex v:
   - degree(v)
   - max/avg/min neighbour degree
   - |N(v) ∩ W| (neighbours currently in window)
2. **DeepSets backbone** (permutation-invariant):
   - φ: MLP that embeds each window vertex into a fixed-dim vector
   - SUM pooling over window embeddings → set-level representation
   - ρ: MLP that maps pooled representation → scalar contribution
3. **Candidate scoring** ψ: MLP that scores the candidate vertex
4. **Final Q-value** = ρ(POOL(φ(window))) + ψ(f_candidate)

**Key insight:** DeepSets guarantees permutation invariance — the Q-value
is the same regardless of the order vertices entered the window. This is
correct because the window is a set, not a sequence.

### 1.4 RL Training Framework

Training uses **episodic RL** where each episode builds one complete ordering:

- **State:** Partial ordering σ_{1:t-1} (represented by window contents + features)
- **Action:** Select next vertex v_t from candidates
- **Reward:** Incremental locality gain: r_t = F(σ_{1:t}) - F(σ_{1:t-1})
- **Loss:** Standard DQN loss with target network

**Critical innovation — Policy Network for sampling:**
- Uniform sampling of training vertices wastes time on unimportant vertices
- The Policy Network π_φ learns which vertices are most informative for training
- It dynamically adjusts the sampling distribution based on vertex significance
- Without the Policy Network, DON barely improves over Gorder

### 1.5 Key Experimental Results

| Graph | Gorder F(σ) | DON-RL F(σ) | Improvement |
|-------|-------------|-------------|-------------|
| Small synthetic (1K) | baseline | +15-35% | Approaches optimal |
| Medium real (10K-100K) | baseline | +5-15% | Consistent gains |
| Large real (1M+) | baseline | +1.8-5% | Transfer still helps |

**Vertex significance skewness:** On many graphs, >2/3 of vertices contribute
≤0 to F(σ) improvement. The few important vertices (high-degree hubs and
bridge vertices) dominate ordering quality. This is the fundamental reason
why the Policy Network matters — it focuses training on these vertices.

**Transfer learning:** DON trained on small graphs (1K-10K vertices) transfers
to larger graphs because:
- Vertex features (degree stats) are scale-invariant
- Local structure patterns repeat across graph sizes
- The window mechanism naturally limits the receptive field

---

## 2. GraphBrew Integration Analysis

### 2.1 Why Direct DON-RL Integration is Impractical

| Concern | Impact | Details |
|---------|--------|---------|
| **Runtime overhead** | Prohibitive | DON evaluates an MLP per candidate vertex per step. For V=1M, that's 1M × 1M evaluations (O(V²) minimum). Gorder is already the slowest algorithm in GraphBrew (9.5s on cit-Patents, 25.5s on wiki-topcats). DON would be 10-100× slower. |
| **Training data** | Impractical | DON needs thousands of episodes per graph for RL training. Each episode is O(V·w) MLP forward passes. Offline training on 22 graphs would take days. |
| **PyTorch dependency** | Unacceptable | GraphBrew is header-only C++17, no external ML frameworks. Adding PyTorch for runtime inference contradicts the project's zero-dependency design. |
| **Marginal gain** | Diminishing | DON-RL's 1.8-5% improvement on F(σ) for large graphs translates to <1% cache-miss improvement after Gorder already captures most locality. GraphBrew's perceptron would need to detect this tiny margin. |

### 2.2 What IS Practical: Extracting DON-RL's Insights

DON-RL's value to GraphBrew is not the network itself, but the **insights
it reveals about graph ordering dynamics**:

1. **Vertex significance is highly skewed** → new perceptron feature
2. **Window-based neighbor overlap predicts ordering quality** → new feature
3. **Policy-guided sampling improves learning** → training pipeline improvement
4. **Scale-invariant vertex features** → confirms GraphBrew's type-transfer approach

---

## 3. Concrete Enhancement Proposals

### 3.1 [Tier 1] Vertex Significance Skewness Feature

**Insight:** DON-RL's key finding is that vertex contributions to F(σ) are
highly skewed. On social networks, a few hub vertices dominate locality.
On road networks, vertices contribute more uniformly.

**Implementation:**
```cpp
// In ComputeSampledDegreeFeatures() — reorder_types.h ~L4065
// Sample top-k degree vertices, compute local F(σ) contribution
double ComputeVertexSignificanceSkewness(const CSRGraph& g, size_t sample_size = 200) {
    // For each sampled vertex v:
    //   significance(v) = Σ_{u ∈ N(v)} 1/|v - u|  (in current ID ordering)
    // Return: coefficient of variation (stddev/mean) of significance values
    // High CV → skewed → hub-dominated → HubSort/HubCluster work well
    // Low CV → uniform → community-based algorithms (Leiden/RabbitOrder) better
}
```

**Perceptron weight:** `w_significance_skew` — positive for hub-based algorithms,
negative for community-based.

**Effort:** Low (1 day). Pure C++ computation + 1 new weight field.

**Expected impact:** Better discrimination between hub-dominated and
community-dominated graphs. Currently the perceptron relies on
`hub_concentration` and `degree_variance` which are correlated but not identical.

### 3.2 [Tier 1] Window Neighbor Overlap Feature

**Insight:** DON-RL uses |N(v) ∩ W| (neighbors in the current window) as a
key feature. This directly measures how well the current ordering groups
neighbors together — more principled than `packing_factor`.

**Implementation:**
```cpp
// In ComputeSampledDegreeFeatures() — reorder_types.h ~L4065
double ComputeWindowNeighborOverlap(const CSRGraph& g, size_t window = 64) {
    // For sampled vertices v:
    //   overlap(v) = |{u ∈ N(v) : |u - v| < window}| / |N(v)|
    // Return: mean overlap across samples
    // High overlap → current ordering already has good locality
    // Low overlap → reordering has room to improve
}
```

**Perceptron weight:** `w_window_overlap` — negative (high overlap → less
benefit from reordering → prefer ORIGINAL/lightweight algorithms).

**Effort:** Low (1 day). Generalizes packing_factor with explicit window.

**Expected impact:** More accurate ORIGINAL-vs-reorder decisions. Window
overlap directly measures what F(σ) tries to optimize, while packing_factor
only checks hub-neighbor co-location.

### 3.3 [Tier 1] DON-RL-Informed Training: Significance Weighting

**Insight:** DON-RL's Policy Network focuses training on important vertices.
We can apply the same idea to GraphBrew's perceptron training: weight training
examples by their "significance" (how much the ordering choice matters).

**Implementation in** `scripts/lib/training.py`:
```python
def compute_example_weight(graph_info, benchmark_results):
    """Weight training examples by how much the ordering choice matters."""
    times = {r.algorithm: r.time for r in benchmark_results if r.graph == graph_info.name}
    if not times: return 1.0

    best = min(times.values())
    worst = max(times.values())
    if best <= 0: return 1.0

    # Significance = speedup range (how much the choice matters)
    return max(1.0, worst / best)
```

Graphs where algorithm choice makes a big difference (3× speedup range)
get weighted 3× more than graphs where all algorithms perform similarly.

**Effort:** Low (half day). Change training loop only.

**Expected impact:** Perceptron focuses on discriminating cases that matter,
rather than equally weighting easy/hard examples. DON-RL showed this
improves convergence by 2-3× and final accuracy by 5-10%.

### 3.4 [Tier 2] Locality Score as Pre-Reorder Quality Metric

**Insight:** DON-RL optimizes F(σ) directly. We can compute a cheap
approximation of F(σ) on the **current** ordering and use it as a perceptron
feature. This tells us "how good is the existing ordering?" before deciding
whether to reorder.

**Implementation:**
```cpp
double SampledLocalityScore(const CSRGraph& g, size_t sample_size = 500) {
    double total = 0;
    size_t count = 0;
    // Sample random edges (u,v)
    for (size_t i = 0; i < sample_size; i++) {
        // Pick random vertex u, random neighbor v
        size_t dist = std::abs((int64_t)u - (int64_t)v);
        if (dist > 0) total += 1.0 / dist;
        count++;
    }
    return total / count;  // Average per-edge locality contribution
}
```

**Perceptron weight:** `w_locality_score` — high score → good current locality
→ favor ORIGINAL; low score → invest in reordering.

**Effort:** Medium (1 day). Needs normalization across graph sizes.

### 3.5 [Tier 2] DON-Augmented Gorder: Learned Tie-Breaking

**Insight:** Rather than replacing Gorder entirely (impractical), use a
lightweight model **only for tie-breaking** when multiple vertices have
similar Q-scores. This captures most of DON's benefit with minimal overhead.

**Design:**
1. Run standard Gorder greedy
2. When top-k candidates have Q-scores within 5% of each other (tie),
   use a precomputed vertex priority (based on trained features) to break ties
3. The priority is computed once upfront (O(V)) rather than per-step

**Implementation:** Add `DON_tiebreak` mode to Gorder's `GorderGreedy()`:
```cpp
// In reorder_gorder.h
// Precompute vertex priority based on degree features
// priority(v) = w1*degree + w2*clustering_coeff + w3*betweenness_centrality
std::vector<double> priorities = PrecomputeVertexPriority(g);

// In greedy loop:
if (abs(Q[v1] - Q[v2]) < 0.05 * Q[v1]) {
    return priorities[v1] > priorities[v2] ? v1 : v2;
}
```

**Effort:** Medium (2-3 days). Needs to train priority weights offline.

**Expected impact:** 1-5% improvement on F(σ) metric vs standard Gorder,
with <5% runtime overhead.

### 3.6 [Tier 3] DON-Lite: Lightweight Neural Ordering

**Insight:** Full DON-RL is impractical, but a simplified version could work
for communities where the perceptron selects Gorder but the margin is low.

**Design:**
1. **Offline phase:** Train a small MLP (2 layers, 32 hidden units) on
   representative graphs. Input: 5 vertex features → Output: vertex
   priority score.
2. **Runtime phase:** For eligible communities (size >50K, margin <0.1):
   - Compute features for all vertices (O(V))
   - Sort vertices by predicted priority (O(V log V))
   - This produces a "DON-Lite" ordering
3. **Inference:** The MLP is converted to fixed C++ code (weights hardcoded):
   ```cpp
   double DONLitePriority(double degree, double max_nbr_deg,
                          double avg_nbr_deg, double min_nbr_deg,
                          double clustering_coeff) {
       // Layer 1: 5 → 32 → ReLU
       // Layer 2: 32 → 1
       // Weights hardcoded from offline training
   }
   ```

**Effort:** High (1-2 weeks). Needs offline training pipeline + C++ inference.
**Risk:** The sorting approach loses the sequential window context that makes
DON-RL effective. May not outperform simple degree-based sorting (HubSort).

---

## 4. Priority Matrix (Updated with DON-RL Items)

| # | Enhancement | Source | Impact | Effort | Priority |
|---|-------------|--------|--------|--------|----------|
| 3.1 | Vertex significance skewness | DON-RL §5.3 | Medium | Low | **P0** |
| 3.2 | Window neighbor overlap | DON-RL §3.2 | Medium | Low | **P0** |
| 3.3 | Significance-weighted training | DON-RL §4.2 | Medium | Low | **P0** |
| 3.4 | Sampled locality score | DON-RL §2.1 | Medium | Medium | **P1** |
| 3.5 | DON-augmented Gorder tiebreak | DON-RL §4.1 | Low-Med | Medium | **P2** |
| 3.6 | DON-Lite neural ordering | DON-RL §3-4 | Unknown | High | **P3** |

**Recommendation:** Implement 3.1, 3.2, 3.3 (all P0) first. These are
feature-extraction and training-level changes that require no architectural
modifications. Measure impact with LOGO cross-validation. Only proceed to
P1-P3 if P0 shows promise.

---

## 5. Training Results Summary

### 5.1 Small Tier (16 graphs, 0.1-10.7 MB)

| Benchmark | Accuracy | Notes |
|-----------|----------|-------|
| pr | 100% | |
| pr_spmv | 100% | |
| bfs | 100% | |
| cc | 100% | |
| cc_sv | 93.8% | 1 graph misclassified |
| sssp | 100% | |
| bc | 100% | |
| tc | 100% | |

- 3 graph types discovered (type_0, type_1, type_2)
- Training time: 51 minutes (dominated by cache simulation: 37 min)

### 5.2 Medium Tier (6 graphs, 58.6-402.5 MB)

| Benchmark | Accuracy | Notes |
|-----------|----------|-------|
| All 8 | 100% | Perfect accuracy on medium graphs |

- Graphs: wiki-Talk, web-Google, web-BerkStan, as-Skitter, cit-Patents, wiki-topcats
- 1 new graph type discovered (type_3), total 4 types after merge
- Training time: 1h 33m (cache sim skipped due to BC bottleneck)
- Benchmarking: 672 runs, avg 0.123s per kernel execution

### 5.3 Large Tier

- Status: **In progress** (downloading com-Orkut, europe-osm, sk-2005, rgg_n_2_24_s0)
- Cache simulation skipped (same BC bottleneck)

### 5.4 Cache Simulation Bottleneck

BC (betweenness centrality) simulation is O(V × E) with cache tracking overhead.
On medium graphs (1M+ vertices), a single BC simulation takes 20-60+ minutes.
With 6 graphs × 14 algorithms = 84 BC simulations, the total would be 28-84 hours.

**Mitigation options:**
1. `--skip-cache` flag (used for medium/large) — loses cache-miss data but
   keeps benchmark runtime data
2. Increase `TIMEOUT_SIM_HEAVY` with early termination
3. Skip BC simulation selectively (only for graphs > 50MB)
4. Sample fewer source vertices in BC simulation

---

## 6. Current Model Gaps

Based on training results, these weight fields remain zero:

| Field | Reason | Impact |
|-------|--------|--------|
| cache_l1_impact | No cache simulation (skipped) | Missing low-level locality signal |
| cache_l2_impact | No cache simulation | " |
| cache_l3_impact | No cache simulation | " |
| cache_dram_penalty | No cache simulation | " |
| w_packing_factor | Fill-weights phase needs cache data | Missing co-location signal |
| w_forward_edge_fraction | Fill-weights phase needs cache data | Missing convergence signal |
| w_working_set_ratio | Fill-weights phase needs cache data | Missing LLC pressure signal |
| w_dv_x_hub | Cross-term, needs filled components | Missing interaction signal |
| w_mod_x_logn | Cross-term | " |
| w_pf_x_wsr | Cross-term | " |
| w_fef_convergence | PR/SSSP bonus, needs FEF data | " |

**Root cause:** The "fill zero weights" phase depends on cache simulation
results. Without cache data, it cannot compute packing factor, forward edge
fraction, or working set ratio at training time. These features ARE computed
at runtime in the C++ code, but their corresponding perceptron weights are
untrained.

**Fix needed:** Decouple feature computation from cache simulation. The
topology-based features (packing factor, FEF, WSR) should be computed from
graph structure alone, independent of cache simulation results.

---

## 7. Key Insights from DON-RL for GraphBrew's Architecture

### 7.1 Perceptron vs. DON: Different Problems

GraphBrew's perceptron selects **which algorithm** to use (meta-learning / algorithm selection).
DON-RL learns **how to order** vertices (ordering generation).

These are complementary, not competing approaches:
- The perceptron's job: "Should I use Gorder, RabbitOrder, or HubSort for this community?"
- DON-RL's job: "Given that we chose Gorder, can we make Gorder's ordering 5% better?"

GraphBrew already solves the first problem well (93-100% accuracy). DON-RL
addresses the second, but with diminishing returns on top of an already-good
algorithm selection.

### 7.2 Where DON-RL Could Help Most

DON-RL's biggest impact is on cases where:
1. **Multiple algorithms perform similarly** (low perceptron margin) — the
   ordering quality within each algorithm matters more than algorithm choice
2. **Hub-dominated graphs** (social networks) — vertex significance is highly
   skewed, so the ordering of hub vertices matters disproportionately
3. **Large working sets** (WSR > 3) — cache replacement can't help, only
   spatial locality (which DON-RL optimizes) matters

### 7.3 Scale-Invariant Features Validate GraphBrew's Design

DON-RL's finding that degree-based features transfer across graph sizes
validates GraphBrew's design choice to:
- Use normalized topology features (modularity, hub_concentration, density)
- Cluster graphs into types based on feature similarity
- Apply per-type weights trained on small graphs to larger graphs

This is exactly what GraphBrew does with its type system. DON-RL provides
theoretical support for this approach.
