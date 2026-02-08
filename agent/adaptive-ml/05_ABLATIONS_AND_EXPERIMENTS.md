# Ablations & Experiments

---

## A) Core Ablations (must implement toggles)

1) No-types: force type_id=0 for all graphs
2) No-Leiden: treat whole graph as one community
3) No-OOD fallback: always pick best-scoring algo
4) No-margin fallback: always apply chosen algo
5) Feature ablations: drop one feature group at a time
6) Partition depth: 1-level vs 2-level recursion (if supported)

---

## A2) Paper-Informed Feature Ablations

These ablations test features derived from specific SOTA papers.
Each ablation zeroes one feature group and measures the impact.

### Packing Factor (IISWC'18)
**Feature:** `w_packing_factor` in `PerceptronWeights` (~L1552 of reorder_types.h)
**Computed at:** `ComputeSampledDegreeFeatures` (~L4120) — samples top-degree hubs,
checks if neighbours have nearby IDs (within `locality_window = max(64, N/100)`).

**Ablation:** Set `w_packing_factor = 0` in all type weight files.
**Expected outcome:** Accuracy drops on graphs with high existing locality
(mesh, road networks) where ORIGINAL should be selected but isn't.
**Run:**
```bash
# baseline
python3 scripts/graphbrew_experiment.py --full --size medium --auto --skip-cache \
  --benchmarks pr bfs cc sssp --trials 5
# ablation: edit weight files to zero packing_factor, then re-run same command
```

### Forward Edge Fraction (GoGraph)
**Features:** `w_forward_edge_fraction` (~L1554) and `w_fef_convergence` (~L1695)
**Computed at:** ~L4150 — samples 2000 vertices, counts edges (u,v) where u < v.

**Ablation:** Zero both `w_forward_edge_fraction` and `w_fef_convergence`.
**Expected outcome:** Accuracy drops primarily on convergence-sensitive benchmarks
(PR, SSSP) where ordering that places sources before destinations matters.
**Separate sub-ablation:** Zero only `w_fef_convergence` (keep linear term) to
isolate the convergence bonus from the general locality signal.

### Working Set Ratio (P-OPT)
**Feature:** `w_working_set_ratio` (~L1556)
**Computed at:** ~L4170 — `graph_bytes / LLC_size` (LLC detected via `sysconf`).

**Ablation:** Set `w_working_set_ratio = 0`.
**Expected outcome:** Accuracy drops on large graphs that exceed LLC (working
set ratio > 3.0) where aggressive reordering is most beneficial.

### Quadratic Interaction Terms
**Features:** `w_dv_x_hub`, `w_mod_x_logn`, `w_pf_x_wsr`
These capture non-linear interactions between feature pairs.

**Ablation:** Zero all three interaction terms simultaneously, then individually.
**Expected outcome:** Removing `w_pf_x_wsr` (packing × working-set) should hurt
most on graphs near the LLC boundary where locality matters conditionally.

### Ablation Protocol
For each ablation above:
1. Record baseline accuracy on MEDIUM tier (28 graphs × 4 benchmarks × 5 trials)
2. Zero the target weight(s) in all type files
3. Re-run the same experiment (same graphs, same seeds)
4. Compare: overall accuracy, per-category accuracy, geo-mean speedup
5. Restore weights and proceed to next ablation

The ablation results directly inform which features to invest in strengthening
(see [06_SOTA_IDEAS.md](06_SOTA_IDEAS.md) Tier 1 proposals).

---

## B) Evaluation Speed Tiers

Pick the tier that matches how fast you need feedback. **Every tier auto-downloads
missing graphs** when using `--full`.

### 🧪 Test — ~2 min (sanity check after a code change)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size small --auto --skip-cache \
  --graph-list ca-GrQc email-Enron soc-Slashdot0902 \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs \
  --trials 2
```
- **3 tiny graphs** (<10 MB each, already present or downloaded in seconds)
- 2 benchmarks × 2 trials — enough to confirm nothing crashes
- Use after every compile to catch regressions fast

### 🔹 Small — ~10 min (quick correctness + rough perf)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size small --auto --skip-cache \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs cc sssp \
  --trials 3
```
- **16 graphs** (SMALL catalog, ~62 MB total download)
- All 4 key benchmarks, 3 trials — good signal on correctness
- Geo-means across 16 graphs already show trends

### 🔸 Medium — ~30-60 min (solid performance signal)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size medium --auto --skip-cache --skip-slow \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs cc sssp \
  --trials 5
```
- **28 graphs** (MEDIUM catalog, ~1.1 GB — includes web-Google, cit-Patents, roadNet-CA, cnr-2000)
- 5 trials per benchmark — stable enough to trust geo-means
- Use for **primary development loop** decisions

### 🔷 Large — ~2-4 hours (production-quality evaluation)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size large --auto --skip-cache --skip-slow \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs cc sssp \
  --trials 5
```
- **37 graphs** (LARGE catalog, ~25 GB — includes soc-LiveJournal1, hollywood-2009, kron_g500-logn20, uk-2002)
- Run as **background process** — this is the definitive evaluation
- Results here drive ship/no-ship decisions
- **First run downloads ~25 GB**, subsequent runs reuse cached graphs + label maps

### 🔶 XLarge — ~8+ hours (stress test at scale)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size xlarge --auto --skip-cache --skip-slow \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs cc sssp \
  --trials 5
```
- **6 massive graphs** (>2 GB each, ~63 GB total — com-Friendster, twitter7, uk-2005, etc.)
- Only run after Large tier passes
- Requires ~64 GB RAM; `--auto` will skip graphs that don't fit

### 🎯 Focused — any duration (investigate specific graphs)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --auto --skip-cache \
  --graph-list cit-Patents web-BerkStan soc-LiveJournal1 \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks sssp \
  --trials 10
```
- Cherry-pick specific graphs and benchmarks
- High trial count (10) for investigating outliers
- Auto-downloads any missing graphs from the list

---

## C) Graph Categories & Expected Behavior

| Category | Example Graphs | Expected Winner |
|----------|--------|-----------------|
| **Social** | soc-Epinions1, soc-Slashdot0902, soc-LiveJournal1 | `vibe:rabbit` (PR), `vibe:hrab` (SSSP) |
| **Web** | web-Google, web-BerkStan, cnr-2000, wiki-topcats | `vibe:hrab:gordi` (locality), `vibe:hrab` (PR) |
| **Road** | roadNet-CA, roadNet-PA, roadNet-TX | boost or `vibe:hrab:gordi` (structured) |
| **Mesh** | delaunay_n17–n24, rgg_n_2_17–24 | `vibe:hrab:gordi` (CC/BFS) |
| **Synthetic** | kron_g500-logn16–21, hollywood-2009 | `vibe:rabbit` (SSSP), `vibe:hrab:gordi` (BFS) |
| **Citation** | cit-Patents, cit-HepPh, cit-HepTh | `vibe:hrab` (SSSP) |
| **XLarge Web** | uk-2005, it-2004, webbase-2001 | `vibe:hrab:gordi` (locality at scale) |
| **XLarge Social** | com-Friendster, twitter7 | `vibe:rabbit` (speed), `vibe:hrab` (SSSP) |

---

## D) Experimental Protocol

- Use a fixed benchmark suite + fixed seeds.
- Report:
  - Kernel time and end-to-end time
  - Reorder cost breakdown
  - Regret distribution vs oracle
- Provide:
  - Per-graph summary table
  - Per-community summary histograms (if easy)

---

## E) Success Criteria

- No correctness regressions.
- End-to-end speedup > baseline on median graphs.
- Tail safety: p95 slowdown ≤ small threshold (define).
- Overhead bounded: reorder overhead ≤ X% of total time.

---

## F) Interpreting Results

### Outlier Rules
- Speedups **>10×** must be verified with 10+ trials on that specific graph
- Speedups **2-5×** are typical for well-matched reorderings
- **SSSP/BFS** are source-dependent — variance is expected, use 5+ trials minimum
- Use `--graph-list <problem_graph> --trials 10` to investigate outliers

### Trial Count Guidelines

| Tier | Trials | Sufficient For |
|------|:---:|---|
| Test | 2 | Crash detection only |
| Small | 3 | Trend detection |
| Medium | 5 | **Trustworthy geo-means** |
| Large | 5 | Ship/no-ship decisions |
| Outlier investigation | 10+ | Confirming or debunking extreme speedups |
