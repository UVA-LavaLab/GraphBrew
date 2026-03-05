# GraphBrew — Bug Fix & Training Plan

> Generated: March 4, 2026
> Purpose: Track verified bugs, fixes needed, and ML improvements before next training run.

---

## Status Legend

- [ ] Not started
- [x] Done
- [~] In progress

---

## 1. Confirmed True Bugs (Code-Verified)

### Bug #2 — Thread-Unsafe `std::rand()` in `GenerateRandomMapping_v2` [HIGH]

- **File**: `bench/include/graphbrew/reorder/reorder_basic.h` ~L145–170
- **Problem**: `std::rand()` is called inside `#pragma omp parallel for`. The C++ standard states `std::rand()` is not thread-safe — concurrent calls produce undefined behavior (data races on internal PRNG state).
- **Impact**: Potential infinite loops (all threads generate same sequence → massive collisions on `claimedVtxs`), duplicate IDs, or silently wrong orderings.
- **Fix**: Replace with per-thread `std::mt19937`:
  ```cpp
  #pragma omp parallel
  {
      std::mt19937 gen(42 + omp_get_thread_num());
      std::uniform_int_distribution<NodeID_> dis(0, g.num_nodes() - 1);
      #pragma omp for
      for (NodeID_ v = 0; v < g.num_nodes(); ++v) {
          while (true) {
              NodeID_ randID = dis(gen);
              if (claimedVtxs[randID] != 1) {
                  if (compare_and_swap(claimedVtxs[randID], NodeID_(0), NodeID_(1))) {
                      new_ids[v] = randID;
                      break;
                  }
              }
          }
      }
  }
  ```
- **Verification**: Run `run-bfs -g 15 -o 1` (Random reorder) with `OMP_NUM_THREADS=8`. Should complete without hanging.
- [ ] Fix applied
- [ ] Verified

---

### Bug #3 — No Mapping Bijection Validation [HIGH]

- **File**: `bench/include/external/gapbs/builder.h` ~L862–869
- **Problem**: After any reorder algorithm produces `new_ids`, the code never validates that it is a valid permutation (bijection). If an algorithm produces duplicate IDs (e.g., `[0, 1, 1, 3]`), the graph is silently corrupted downstream.
- **Impact**: Hard-to-diagnose incorrect benchmark results. Any reorder bug silently propagates.
- **Fix**: Add a debug-mode validation:
  ```cpp
  #ifndef NDEBUG
  {
      std::vector<bool> seen(g.num_nodes(), false);
      for (NodeID_ v = 0; v < g.num_nodes(); ++v) {
          assert(new_ids[v] >= 0 && new_ids[v] < g.num_nodes()
                 && "Mapping ID out of range");
          assert(!seen[new_ids[v]] && "Duplicate mapping ID — not a permutation");
          seen[new_ids[v]] = true;
      }
  }
  #endif
  ```
- **Verification**: Build with assertions enabled (`-UNDEBUG`). Run all reorder algorithms on a small graph (`-g 10`). No assertions should fire.
- [ ] Fix applied
- [ ] Verified

---

## 2. Evaluation / Metrics Bugs

### Bug #4 — Geometric Mean Filters Out Slowdowns [MEDIUM]

- **File**: `scripts/lib/analysis/metrics.py` ~L161
- **Problem**: `geo_mean_kernel_speedup` filters entries with `speedup <= 0`, which also drops legitimate slowdowns (speedup between 0 and 1). This inflates geometric mean results.
- **Fix**: Only filter truly invalid entries (speedup <= 0 means measurement error, speedup in (0,1) means slowdown and should be kept):
  ```python
  vals = [e.kernel_speedup for e in self.entries if e.kernel_speedup > 0]
  ```
  Ensure the current filter is `> 0` (not `>= 1`). If it currently removes values < 1, change it.
- [ ] Verified logic
- [ ] Fix applied if needed

### Bug #5 — Speedup Returns 0 for Zero Reorder Time [MEDIUM]

- **File**: `scripts/lib/analysis/metrics.py` ~L67
- **Problem**: When `e2e_reord` is 0, speedup is set to `0.0` instead of `inf` (or skipped). This is indistinguishable from a 2x slowdown.
- **Fix**:
  ```python
  speedup = e2e_orig / e2e_reord if e2e_reord > 0 else float('inf')
  ```
- [ ] Fix applied

---

## 3. Confirmed False Positives (No Fix Needed)

These were investigated and confirmed correct:

| Claimed Issue | Verdict | Why |
|---|---|---|
| Atomic fill-in race in `RelabelByMapping` | ✅ Correct | `__sync_fetch_and_add` returns unique old values; no collision possible |
| Z-score normalization mismatch (Python vs C++) | ✅ Correct | Both use identical 21-feature vector with same transforms and z-score formula |
| Data leakage in LOGO cross-validation | ✅ Correct | Weights are retrained from scratch per fold; held-out graph is excluded from training |
| `avg_degree / 100` normalization | ✅ Correct | Both Python and C++ divide by 100 consistently |
| Edge direction inversion in `RelabelByMapping` | ✅ Correct | Naming is confusing, but `inv_index/inv_neighs` (out-edges) → `out_index_` and `index/neighs` (in-edges) → `in_index_`. CSRGraph constructor args are in the right order. |

---

## 4. ML Model & Theory Improvements

### 4.1 Decision Tree Feature Gap — FIXED

- **File**: `scripts/lib/ml/model_tree.py` + `bench/include/graphbrew/reorder/reorder_types.h`
- **Problem**: C++ `MODEL_TREE_N_FEATURES` was 14, but Python DT now trains with 21 features (14 linear + 2 DON-RL + 5 quadratic cross-terms). Feature indices 14–20 in trained JSON trees would hit the C++ fallback branch (`go right`), silently corrupting predictions.
- **Fix applied**: Updated C++ `MODEL_TREE_N_FEATURES` from 14 → 21. Added 7 missing features to `ModelTree::extract_features()`: `vertex_significance_skewness`, `window_neighbor_overlap`, and 5 quadratic terms (`dv×hc`, `mod×logn`, `pf×wsr`, `vss×hc`, `wno×pf`). Feature ordering verified identical to Python `extract_dt_features()`.
- [x] DT features aligned with perceptron (C++ and Python both 21 features, same order)
- [ ] Ablation run completed

### 4.2 Consider Gradient Boosting (XGBoost / LightGBM)

- **Current**: Single perceptron per algorithm (linear boundary) + optional DT
- **Alternative**: XGBoost or LightGBM with 100-500 trees
  - Handles non-linear decision boundaries automatically
  - Built-in feature importance
  - Typical 5–15% accuracy improvement on tabular data
  - Tradeoff: harder to embed in C++ (perceptron inference is O(n) fast)
- **Recommendation**: Try XGBoost in the Python training pipeline as an experiment. If accuracy gain is significant, consider exporting tree ensemble for C++ inference.
- [ ] XGBoost experiment run
- [ ] Results compared to perceptron

### 4.3 Feature Importance / SHAP Analysis

- **Problem**: 21 features × 17 algorithms = 357 weight parameters with no visibility into which features actually matter.
- **Recommendation**: After next training run, compute permutation importance or SHAP values per algorithm. Prune features contributing < 1% to predictions.
- [ ] Feature importance computed
- [ ] Low-impact features identified

### 4.4 Amortization Model Assumes Linear Savings

- **Current formula**: `break_even_iters = reorder_time / time_saved_per_iter`
- **Problem**: Assumes constant per-iteration savings. For convergence-based algorithms (PR, SSSP), early iterations save more than later ones as caches warm up.
- **Recommendation**: For iterative algorithms, model savings as exponential decay rather than constant. Low priority — current linear model is a reasonable first approximation.
- [ ] Investigate convergence-aware amortization

### 4.5 Per-Benchmark Accuracy Reporting

- **Problem**: Aggregated accuracy over all benchmarks masks per-workload failures (e.g., 90% on PR but 50% on SSSP averages to 70%).
- **Recommendation**: Report LOGO accuracy separately per benchmark type.
- [ ] Per-benchmark reporting added

---

## 5. Execution Order

**Before next training run (blocking):**

1. ~~Fix Bug #1 (edge inversion)~~ — Confirmed false positive; constructor args are correct
2. Fix Bug #2 (thread-unsafe rand) — affects Random reorder reliability
3. Fix Bug #3 (add bijection validation in debug mode) — catches future bugs early
4. Fix Bug #4 and #5 (metrics) — ensures training labels are computed correctly

**Done:**

5. [x] Align DT features C++ ↔ Python (§4.1) — MODEL_TREE_N_FEATURES 14→21, extract_features() updated

**After training run (improvements):**

6. Run XGBoost experiment (§4.2)
7. Compute feature importance (§4.3)
8. Add per-benchmark reporting (§4.5)

**Low priority / future work:**

9. Convergence-aware amortization (§4.4)

---

## 6. Impact Assessment

| Fix | Directed Graphs | Undirected Graphs | Training Data | ML Accuracy |
|-----|:---:|:---:|:---:|:---:|
| Bug #2 (rand thread safety) | ✅ | ✅ | Random reorder data may change | Minimal |
| Bug #3 (bijection check) | ✅ | ✅ | Catches bad data early | Prevents garbage-in |
| Bug #4 (geomean) | — | — | — | More honest metrics |
| Bug #5 (speedup=0) | — | — | — | Correct edge cases |
| DT features (§4.1) | — | — | — | +5-15% DT accuracy |
| XGBoost (§4.2) | — | — | — | +5-15% overall |

---

## 7. Questions to Resolve

- [ ] Is `GenerateRandomMapping_v2` actually called in production, or only v1? (Determines Bug #2 urgency)
- [ ] Re-run training after Bug #2 and #3 fixes to establish new baseline accuracy
