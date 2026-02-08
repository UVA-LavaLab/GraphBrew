# SOTA-Inspired Improvement Ideas (propose + test)

All proposals are grounded in five papers and mapped to actual code locations.
See [10_LITERATURE.md](10_LITERATURE.md) for paper summaries and feature mappings.

---

## Context: What AdaptiveOrder Already Does

AdaptiveOrder (Algorithm 14) lives in `reorder_adaptive.h` (651 lines) and
`reorder_types.h` (4682 lines):

1. **Leiden → communities** (GVE-Leiden CSR-native)
2. **Feature extraction** per community (`ComputeSampledDegreeFeatures`, reorder_types.h ~L4065):
   - Core: modularity, degree_variance, hub_concentration, avg_degree, density
   - IISWC'18-derived: `packing_factor` (hub neighbour co-location)
   - GoGraph-derived: `forward_edge_fraction` (edges where src < dst)
   - P-OPT-derived: `working_set_ratio` (graph_bytes / LLC_size)
3. **Type matching** → nearest centroid in feature space → load per-type `PerceptronWeights`
4. **Perceptron scoring** (`PerceptronWeights::score()`, reorder_types.h ~L1691):
   - Linear: bias + Σ(w × feature) for 14 features
   - Quadratic: `w_dv_x_hub`, `w_mod_x_logn`, `w_pf_x_wsr` (3 cross-terms)
   - Convergence bonus: `w_fef_convergence × forward_edge_fraction` for PR/SSSP
   - Per-benchmark multipliers: bench_pr/bfs/cc/sssp/bc/tc
5. **Fallbacks**: OOD (type distance > threshold → ORIGINAL), margin (mode-aware)
6. **Selection modes**: fastest-reorder (0), fastest-execution (1), best-endtoend (2), best-amortization (3)

---

## Tier 1 — Minimal Risk (implement first)

### 1.1 Explicit "SKIP reorder" cost model

**Paper:** Balaji & Lucia, IISWC'18 — show that reordering only helps a subset
of graph×algorithm combinations. They propose low-overhead metrics to predict
when reordering is beneficial.

**Current state:** `SelectBestReorderingForCommunity` (reorder_types.h ~L3971) returns
ORIGINAL for small communities but doesn't model reorder cost vs benefit explicitly.
`MODE_BEST_ENDTOEND` boosts `w_reorder_time` by 2× but this is a fixed multiplier.

**Proposal:** Add a proper cost model:
```
predicted_gain = perceptron_score(algo) - perceptron_score(ORIGINAL)
predicted_cost = avg_reorder_time × (community_nodes / total_nodes)
net_benefit = predicted_gain - α × predicted_cost
if net_benefit < threshold: return ORIGINAL
```

**Code locations:**
- Add to `SelectReorderingWithMode()` in reorder_types.h (~L3746)
- Use `PerceptronWeights::avg_reorder_time` already stored per-algo
- Threshold `α` as a tunable constant (start at 1.0)

**Test:** A/B brute-force on MEDIUM tier: compare end-to-end time with/without
cost model. Success = fewer graphs where reorder cost exceeds benefit.

### 1.2 Packing factor as reorder-need short-circuit

**Paper:** IISWC'18 — "packing factor" measures fraction of hub neighbours
already co-located in memory. High packing = current ordering already has good locality.

**Current state:** Computed as `SampledDegreeFeatures::packing_factor` (~L4120)
and used as a linear perceptron feature (`w_packing_factor`, ~L1552). Not used
as a short-circuit.

**Proposal:** Before the full perceptron, check:
```
if packing_factor > 0.7 && working_set_ratio < 2.0:
    return ORIGINAL  // already well-packed, small working set
```
This can skip Leiden + per-community feature extraction entirely.

**Code location:** Top of `SelectBestReorderingForCommunity()` (~L3971)

**Test:** Count how many graphs in SMALL/MEDIUM tiers hit this short-circuit
and whether their ORIGINAL time is within 5% of oracle.

### 1.3 Forward-edge-fraction convergence strengthening

**Paper:** Zhou et al., GoGraph (2024) — prove that vertex ordering where
sources precede destinations (high forward-edge fraction M(σ)) reduces async
iteration rounds. Their divide-and-conquer optimizer achieves 1.83× avg speedup.

**Current state:** `forward_edge_fraction` computed (~L4150), used via
`w_fef_convergence` for PR/SSSP bonus (~L1695). Correct but could be stronger.

**Proposal:**
1. **Post-selection validation:** After choosing an algo, estimate its expected
   FEF improvement. For convergence-sensitive benchmarks (PR/SSSP), if the
   chosen algo doesn't improve FEF over ORIGINAL, reconsider runner-up.
2. **FEF delta as training signal:** In `training.py` (~L179), include
   FEF delta (post-reorder − pre-reorder) in the reward, not just speedup.

**Code location:** Extend `score()` (~L1691) and training loop in `training.py`.

**Test:** Compare PR iteration counts (not just wall time) before/after
reordering. GoGraph shows 2-4× iteration reduction on suitable graphs.

### 1.4 Calibrate perceptron scores with Platt scaling

**Current state:** `score()` returns uncalibrated real number.
The argmax picks the winner, but the margin is compared to a fixed threshold.

**Proposal:** After training, fit logistic regression (Platt scaling) from
`score_margin = score(best) - score(runner_up)` → `P(best actually wins)`.
Use this to:
1. Set a data-driven margin threshold (replace hardcoded value)
2. Report confidence in the selection log

**Test:** Measure calibration: does P(win | margin > 0.3) ≈ 0.85?
Compare vs current fixed threshold on Medium tier.

---

## Tier 2 — Moderate (after Tier 1 validated)

### 2.1 Contextual bandit for low-margin cases

**Paper:** DON-RL (Zhao et al.) — key insight beyond the neural network is that
exploration under uncertainty drives learning. When the model is unsure, it
should explore rather than commit.

**Current state:** Low margin → falls back to ORIGINAL. Safe but leaves
performance on the table.

**Proposal:** Replace static margin fallback with ε-greedy bandit:
```
if margin < threshold:
    with prob ε: try runner_up algorithm
    with prob 1-ε: use best (or ORIGINAL if margin < min_threshold)
```
Log choice + outcome for online weight updates.

**Constraints:**
- ε starts at 0.1, decays with log(evaluations)
- Never explore with algos that have `avg_reorder_time > budget`
- Fall back to ORIGINAL if ALL candidates have low predicted gain

**Code location:** New `SelectWithExploration()` wrapping `SelectReorderingWithMode()`.

**Test:** On MEDIUM tier, measure regret distribution with/without exploration.
Success = lower p95 regret AND no worse median.

### 2.2 Per-type OOD radius

**Current state:** Single global `UNKNOWN_TYPE_DISTANCE_THRESHOLD` for all types.
Road-network types (tight clusters) should have smaller radii than social-network
types (diffuse clusters).

**Proposal:** Store `radius` per type (p95 distance of training graphs to centroid).
At runtime: `distance_to_centroid / type_radius > 1.5 → OOD`.

**Code location:**
- Python: `weights.py` — compute radius during k-means type clustering
- C++: `FindBestTypeWithDistance()` in reorder_types.h — use per-type radius

**Test:** Inject 5 unseen graph families. Success = correctly identifies OOD
without false-flagging known families.

### 2.3 Overhead-class algorithm filtering (IISWC'18)

**Paper:** Balaji & Lucia categorize reordering algorithms as:
- Lightweight (<1s): HubSort, DBG, HubClusterDBG, Sort
- Moderate (1-30s): RabbitOrder, RCM, LeidenCSR:gveopt2
- Heavyweight (>30s): Gorder, LeidenCSR:gve

**Proposal:** Add overhead class as input feature or hard filter:
```
if expected_reorder_time / expected_kernel_time > 0.5:
    skip heavyweight candidates from the scoring pool
```

**Code location:** Add to `SelectReorderingFromWeights()` in reorder_types.h.

**Test:** Measure end-to-end time with/without filtering on single-traversal
benchmarks (BFS, SSSP with 1 trial).

---

## Tier 3 — Heavy (research-grade, after Tiers 1-2)

### 3.1 DON-RL-inspired learned ordering

**Paper:** Zhao et al. ("Graph Ordering: Towards the Optimal by Learning") —
train a Deep Order Network (DON) to replace the greedy heuristic in Gorder.
Their RL-trained DON consistently outperforms Gorder on the locality metric.
Key architecture: state = vertex features + window, action = vertex selection,
reward = locality function F(σ).

**Why relevant:** For large communities (>100k nodes) where all perceptron-selected
algorithms give similar speedups, a learned ordering that directly optimizes
the locality score could outperform any fixed algorithm.

**Current state:** Not implemented. AdaptiveOrder selects FROM existing algorithms;
it does not learn a new ordering.

**Proposal (long-term):**
1. Train a small MLP on subgraph structure → vertex ordering
2. Use only for communities where: (a) margin is very low between top-3 algos,
   (b) community size > 100k, (c) expected traversal count > 10
3. Fall back to perceptron-selected algo if learned ordering is slower

**Risk:** Very high implementation effort. Only justified if Tier 1-2 plateau.

### 3.2 P-OPT-inspired transpose reuse feature

**Paper:** Balaji et al. (HPCA'21) — show that the graph transpose encodes
future reuse distance. Their P-OPT reduces LLC misses by 35%.

**Proposal:**
1. **Feature only (feasible):** Sample ~1000 vertices, compute average
   "next-reuse distance" from transpose CSC. Add as perceptron feature
   `w_avg_reuse_distance` in `PerceptronWeights`.
2. **Full P-OPT (impractical in software):** Requires hardware-level support.

**Code location:** Add to `ComputeSampledDegreeFeatures()` (~L4065).

**Test:** Measure correlation between reuse distance feature and actual LLC
miss improvement after reordering. Success = r² > 0.3.

### 3.3 Hierarchical model (Rabbit Order insight)

**Paper:** Arai et al. (IPDPS'16) — key insight: hierarchical communities map
to hierarchical caches. L1 ← inner communities, L2 ← outer, L3 ← graph-level.

**Proposal:** Two-level model:
1. **Graph-level gate:** Classify into broad strategy (community-based vs
   degree-based vs BFS-based) using global features
2. **Community-level expert:** Separate perceptron per regime

This avoids "one perceptron fits all" — separate experts for separate regimes.

**Risk:** Moderate effort, requires sufficient training data per regime.
Start with 2-3 experts max.

### 3.4 GoGraph's divide-and-conquer as candidate algorithm

**Paper:** GoGraph — implements a divide-and-conquer reordering that explicitly
maximizes forward-edge fraction M(σ). Currently GraphBrew only uses the
metric as a feature, not the actual reordering algorithm.

**Proposal:** Implement GoGraph's reordering as a new candidate algorithm
(Algorithm N). The perceptron can then select it when FEF matters most
(convergence-sensitive benchmarks on well-structured graphs).

**Risk:** Medium implementation effort. Need to add the algorithm to the
candidate pool AND retrain all perceptron weights with the new option.

---

## Priority Matrix

| Idea | Impact | Effort | Risk | Priority |
|------|--------|--------|------|----------|
| 1.1 Cost model (skip reorder) | High | Low | Low | **P0** |
| 1.2 Packing factor short-circuit | Medium | Low | Low | **P0** |
| 1.3 FEF convergence strengthening | Medium | Low | Low | **P1** |
| 1.4 Calibrated margins | Medium | Medium | Low | **P1** |
| 2.1 Bandit exploration | Medium | Medium | Medium | **P2** |
| 2.2 Per-type OOD radius | Medium | Medium | Low | **P2** |
| 2.3 Overhead class filtering | Medium | Low | Low | **P1** |
| 3.1 DON-RL learned ordering | High | Very High | High | **P3** |
| 3.2 Transpose reuse feature | Medium | High | Medium | **P3** |
| 3.3 Hierarchical gating | High | High | Medium | **P3** |
| 3.4 GoGraph candidate algo | Medium | Medium | Medium | **P3** |

Start with P0 items. Measure. Only proceed to P1 if P0 shows improvement.
Never jump to P3 without exhausting P1-P2.
