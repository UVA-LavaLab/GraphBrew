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
