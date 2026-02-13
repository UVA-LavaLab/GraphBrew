# AdaptiveOrder-ML — Mental Model

> Phase 1 deliverable. All claims verified against code, not docs.

---

## 1  Algorithm Summary

**AdaptiveOrder-ML** (Algorithm 14) is a per-community meta-algorithm that:

1. Partitions the graph into communities using GVE-Leiden
2. Extracts structural features from each community
3. Uses a trained perceptron model to select the best reordering algorithm per community
4. Applies the selected algorithm to each community subgraph
5. Stitches per-community permutations into a global permutation

---

## 2  Inputs / Outputs

### Inputs

| Input | Source | Description |
|-------|--------|-------------|
| CSR Graph `g` | Loaded from `.sg` file | `CSRGraph<NodeID_, DestID_, invert>` |
| `new_ids` | Caller (builder.h) | `pvector<NodeID_>` sized `num_nodes`, output buffer |
| `useOutdeg` | CLI flag | Whether to use out-degree or in-degree for ordering |
| `reordering_options` | CLI `-o 14:tok0:tok1:tok2:tok3:tok4` | Up to 5 tokens: `[max_depth, resolution, min_recurse_size, selection_mode, graph_name]` |
| Weight files | `results/weights/type_X/weights.json` | Pre-trained perceptron weights per graph type |
| Type registry | `results/weights/registry.json` | 11 type centroids for graph matching |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `new_ids[v]` | `pvector<NodeID_>` | Permutation: vertex `v` → new ID. Bijection `[0, N)` → `[0, N)` |
| stdout timers | text | `PrintTime("Adaptive Map Time", ...)` for profiling |

---

## 3  Major Steps (with code locations)

### Step 1: Option Parsing
**File:** `reorder_adaptive.h:558–609`

```
opts[0] → max_depth     (int, 0–10)
opts[1] → resolution    (double, for Leiden)
opts[2] → min_recurse   (size_t, minimum community size)
opts[3] → mode          (0=fastest_reorder, 1=fastest_exec [default], 2=best_e2e, 3=best_amort, 100=legacy_fullgraph)
opts[4] → graph_name    (string, for future .time file support)
```

### Step 2: Leiden Community Detection
**File:** `reorder_adaptive.h:432–460`

- Uses `graphbrew::runGraphBrew<K>(g, config)` from `reorder_graphbrew.h`
- Config: `resolution` (auto-computed if 0), `maxIterations=30`, `maxPasses=30`, `ordering=COMMUNITY_SORT`
- Output: `membership[]` (community ID per vertex), `modularity` (scalar)
- Auto-resolution: `γ = clip(0.5 + 0.25 × log₁₀(avg_degree + 1), 0.5, 1.2)` with CV guardrail

### Step 3: Global Feature Extraction
**File:** `reorder_types.h:4030–4120` (`ComputeSampledDegreeFeatures`)

Samples up to 10,000 vertices evenly across the graph to compute:
- `degree_variance` (coefficient of variation: σ/μ)
- `hub_concentration` (edge fraction from top 10% degree nodes)
- `clustering_coeff` (sampled triangle ratio)
- `packing_factor` (IISWC'18: fraction of hub neighbors already co-located)
- `forward_edge_fraction` (GoGraph: fraction of edges u→v where u < v)
- `working_set_ratio` (P-OPT: graph_bytes / LLC_size)
- `estimated_modularity` (rough estimate from degree distribution)

### Step 4: Graph Type Detection
**File:** `reorder_types.h` (`DetectGraphType`)

Heuristic classification into: `SOCIAL`, `WEB`, `ROAD`, `KRON`, `MESH`, `GENERIC`
Used for semantic weight file fallback path (not the primary type-matching system).

### Step 5: Dynamic Thresholds
**File:** `reorder_types.h:3844–3900`

- `MIN_FEATURES_SIZE = max(50, min(avg_comm/4, sqrt(N)))` — capped at 2000
- `MIN_LOCAL_REORDER = min(MIN_FEATURES × 2, 5000)` — minimum size for per-community reordering

### Step 6: Small Community Handling
**File:** `reorder_adaptive.h:508–565`

Communities with `size < MIN_LOCAL_REORDER` are merged:
1. `ComputeMergedCommunityFeatures()` → aggregate features
2. `SelectAlgorithmForSmallGroup()` → typically Sort or ORIGINAL
3. Either degree-sort (<100 nodes) or `ReorderCommunitySubgraphStandalone()`

### Step 7: Per-Community Algorithm Selection (Large Communities)
**File:** `reorder_adaptive.h:568–600`

For each community ≥ `MIN_LOCAL_REORDER`:

1. `ComputeCommunityFeaturesStandalone()` → local features (`reorder_types.h:4357`)
   - Builds induced subgraph, counts internal edges
   - Computes: density, avg_degree, degree_var, hub_conc, clustering, diameter, path_len
2. `SelectBestReorderingForCommunity()` → algorithm choice (`reorder_types.h:3910`)
   - Size guard: `< MIN_COMMUNITY_SIZE → ORIGINAL`
   - Delegates to `SelectReorderingWithMode()` (see Step 8)

### Step 8: Mode-Aware Selection Pipeline
**File:** `reorder_types.h:3686–3820`

```
SelectReorderingWithMode(feat, mode, ...)
  │
  ├── FindBestTypeWithDistance(features) → type_name, distance
  │     Normalize 7D features → Euclidean distance to each centroid
  │     Pick closest type (type_0..type_10)
  │
  ├── OOD GUARD: distance > 1.5 → return ORIGINAL
  │   (except MODE_FASTEST_REORDER, which doesn't depend on graph features)
  │
  ├── LoadPerceptronWeightsForFeatures(features)
  │     Priority: env var > type_X.json > semantic type file > default
  │     Returns: map<Algorithm, PerceptronWeights>
  │
  └── MODE dispatch:
        0 (FASTEST_REORDER): argmax(w_reorder_time) across algorithms
        1 (FASTEST_EXEC):    SelectReorderingFromWeights(feat, weights, bench)
        2 (BEST_E2E):        score + 2× reorder_time penalty
        3 (BEST_AMORT):      argmin(iterationsToAmortize)
```

### Step 9: Perceptron Scoring
**File:** `reorder_types.h:1601–1671`

For MODE_FASTEST_EXECUTION (default):
1. For each candidate algorithm, compute `score(feat, bench)`:
   ```
   s = bias + Σ(weight_i × feature_i)  [15 linear terms]
       + w_dv_x_hub × dv × hub          [3 quadratic terms]  
       + cache_l1×0.5 + cache_l2×0.3 + cache_l3×0.2 + dram
       + w_reorder_time × reorder_time
       IF bench ∈ {PR,SSSP}: s += w_fef_convergence × fef
   s *= benchmark_multiplier[bench]
   ```
2. `argmax(score)` → candidate `best_algo`
3. **ORIGINAL margin check**: `if (best_score - original_score < 0.05) → ORIGINAL`

### Step 10: Per-Community Reordering
**File:** `reorder.h:252–315` (`ReorderCommunitySubgraphStandalone`)

1. Build `global_to_local` / `local_to_global` mappings
2. Extract induced subgraph edge list (only internal edges)
3. Guard: no internal edges → assign in original order
4. Build local CSR graph (`MakeLocalGraphFromELStandalone`)
5. Apply selected algorithm (`ApplyBasicReorderingStandalone`)
6. Map local permutation back to global: `reordered_nodes[sub_new_ids[i]] = local_to_global[i]`
7. Assign final IDs: `new_ids[node] = current_id++`

### Step 11: ID Stitching
**File:** `reorder_adaptive.h:508–600`

IDs are assigned contiguously via `current_id` counter:
1. First: all small communities (merged, degree-sorted or reordered together)
2. Then: each large community in descending size order
3. `current_id` starts at 0 and increments through small → large communities
4. Net result: `new_ids[]` is a complete permutation `[0, N)`

---

## 4  Key Data Structures

### `CommunityFeatures` (`reorder_types.h:1103`)
```cpp
struct CommunityFeatures {
    size_t num_nodes, num_edges;
    double internal_density, avg_degree, degree_variance;
    double hub_concentration, modularity, clustering_coeff;
    double avg_path_length, diameter_estimate, community_count;
    double reorder_time;
    double packing_factor, forward_edge_fraction, working_set_ratio;
};
```

### `PerceptronWeights` (`reorder_types.h:1490`)
```cpp
struct PerceptronWeights {
    double bias;                        // bias term
    // 11 core + extended weights
    double w_modularity, w_log_nodes, w_log_edges, w_density, w_avg_degree;
    double w_degree_variance, w_hub_concentration, w_clustering_coeff;
    double w_avg_path_length, w_diameter, w_community_count;
    // 3 locality weights  
    double w_packing_factor, w_forward_edge_fraction, w_working_set_ratio;
    // 3 quadratic interactions
    double w_dv_x_hub, w_mod_x_logn, w_pf_x_wsr;
    // 1 conditional bonus
    double w_fef_convergence;
    // 4 cache weights
    double cache_l1_impact, cache_l2_impact, cache_l3_impact, cache_dram_penalty;
    // time penalty
    double w_reorder_time;
    // metadata
    double avg_speedup, avg_reorder_time;
    // 6 benchmark multipliers
    double bench_pr, bench_bfs, bench_cc, bench_sssp, bench_bc, bench_tc;
    
    double scoreBase(CommunityFeatures) const;
    double score(CommunityFeatures, BenchmarkType) const;
    double iterationsToAmortize() const;
};
```

### `SampledDegreeFeatures` (`reorder_types.h:3980`)
```cpp
struct SampledDegreeFeatures {
    double degree_variance, hub_concentration, avg_degree;
    double clustering_coeff, estimated_modularity;
    double packing_factor, forward_edge_fraction, working_set_ratio;
};
```

---

## 5  State Persistence

| What | Where | Format | When written | When read |
|------|-------|--------|-------------|-----------|
| Type centroids | `results/weights/registry.json` | JSON: `{type_0: {centroid: [7 floats], graphs: [...]}, ...}` | Training via `graphbrew_experiment.py --train` | C++ at runtime (`FindBestTypeWithDistance`) |
| Per-type weights | `results/weights/type_X/weights.json` | JSON: `{algo_id: {bias, w_modularity, ...}, ...}` | Training | C++ at runtime (`LoadPerceptronWeightsForFeatures`) |
| Graph features cache | `results/weights/graph_properties_cache.json` | JSON: per-graph cached features | Feature computation | Python training code |
| Label maps | `results/mappings/{graph}/AdaptiveOrder.lo` | Binary permutation file | Reorder phase | Benchmark phase (via `-o 13:path.lo`) |
| Reorder timing | `results/mappings/{graph}/AdaptiveOrder.time` | Text: seconds | Reorder phase | Analysis scripts |

---

## 6  Truth Table: Fallbacks

### OOD Guardrail

| Type Distance | Mode | Action | Code Location |
|:---:|:---:|--------|---------------|
| ≤ 1.5 | any | Normal perceptron selection | — |
| > 1.5 | MODE_FASTEST_REORDER (0) | **Proceed normally** (reorder speed is feature-independent) | `reorder_types.h:3717` |
| > 1.5 | MODE_FASTEST_EXECUTION (1) | **Return ORIGINAL** | `reorder_types.h:3719` |
| > 1.5 | MODE_BEST_ENDTOEND (2) | **Return ORIGINAL** | `reorder_types.h:3719` |
| > 1.5 | MODE_BEST_AMORTIZATION (3) | **Return ORIGINAL** | `reorder_types.h:3719` |
| no registry found | any non-0 | **Return ORIGINAL** (best_type is empty) | `reorder_types.h:3715` |

### ORIGINAL Margin Check

| Margin (best − original) | Action | Code Location |
|:---:|--------|---------------|
| ≥ 0.05 | Use best algorithm | `reorder_types.h:3600–3614` |
| < 0.05 | **Return ORIGINAL** (reorder overhead likely exceeds benefit) | `reorder_types.h:3608` |
| ORIGINAL has no score (no weights) | Use best algorithm (no margin check) | `reorder_types.h:3606` |

### Small Community Size

| Community Size | Action | Code Location |
|:---:|--------|---------------|
| < MIN_COMMUNITY_SIZE (dynamic, 50–2000) | **Return ORIGINAL** | `reorder_types.h:3930` |
| ≥ MIN_COMMUNITY_SIZE | Proceed to perceptron selection | — |

### RabbitOrder Availability

| `RABBIT_ENABLE` defined? | Selected = RabbitOrder? | Action | Code Location |
|:---:|:---:|--------|---------------|
| Yes | Yes | Use RabbitOrder | — |
| No | Yes | Heuristic fallback: dv>0.8→HubClusterDBG, hub>0.3→HubSort, else→DBG | `reorder_types.h:3951–3960` |
| No | No | No change | — |

### Empty / Trivial Cases

| Condition | Action | Code Location |
|-----------|--------|---------------|
| `num_nodes == 0` | Identity mapping, return | `reorder_adaptive.h:330` |
| Community has 0 nodes | Skip | `reorder.h:262` |
| Subgraph has 0 internal edges | Assign in original order | `reorder.h:296–299` |
| Community size < 100 (in small group) | Degree-sort only | `reorder_adaptive.h:554` |

### What "ORIGINAL" Means

`ORIGINAL` (enum value 0) = **keep original vertex IDs** = identity permutation for that community. The nodes are assigned contiguous IDs in their original order within the community. This is the "do nothing" option — no reordering benefit, but also no reordering cost.

---

## 7  Candidate Algorithm Set

The perceptron model can select from these algorithms:

| ID | Name | Standalone? | Notes |
|:--:|------|:-----------:|-------|
| 0 | ORIGINAL | ✅ | Identity — no reorder |
| 1 | Random | ✅ | Random permutation |
| 2 | Sort | ✅ | Degree-sort |
| 3 | HubSort | ✅ | Hub-sort (high-degree first) |
| 4 | HubCluster | ✅ | Hub-cluster grouping |
| 5 | DBG | ✅ | Degree-based grouping |
| 6 | HubSortDBG | ✅ | HubSort + DBG hybrid |
| 7 | HubClusterDBG | ✅ | HubCluster + DBG hybrid |
| 8 | RabbitOrder | conditional | Requires `RABBIT_ENABLE` compile flag |
| 9 | GOrder | ✅ | Graph ordering (window-based) |
| 10 | COrder | ✅ | Community ordering |
| 11 | RCMOrder | ✅ | Reverse Cuthill-McKee |

Algorithms 12 (GraphBrewOrder), 13 (MAP), 14 (AdaptiveOrder), 15 (LeidenOrder) are **not** in the candidate set — they are meta-algorithms themselves.
