# Workflow (follow in order)

---

## Single Tool Rule

**Always use `scripts/graphbrew_experiment.py`** for experiments. It provides:

| Capability | How |
|---|---|
| **Auto-download graphs** | `--full` or `--auto-setup` downloads missing graphs from SuiteSparse before running |
| **Format conversion** | Downloads `.mtx`, auto-converts to `.sg` via the `converter` binary |
| **Deterministic reordering** | `--full` auto-enables `--precompute` which generates `.lo` label maps; subsequent runs reuse them |
| **Consistent logs** | All output goes to `results/logs/{graph}/benchmark_{algo}_{bench}.log` |
| **Structured results** | `results/benchmark_{timestamp}.json`, `results/reorder_{timestamp}.json` |
| **Resource safety** | `--auto` detects RAM/disk, skips graphs that don't fit |
| **Resumable** | Re-running with same params skips completed `.lo`/`.time` files |
| **Variant expansion** | `--csr-variants` / `--rabbit-variants` select exactly what to test |
| **Phase control** | `--phase reorder|benchmark|cache|weights` or `--full` for everything |

**Never use standalone shell scripts or raw `./bench/bin/*` calls for evaluation.**
Raw binary calls are acceptable **only** for 30-second smoke tests.

---

## Phase 0 — Orientation (mandatory)

- Read the wiki page on AdaptiveOrder-ML (Algorithm 14).
- Then validate against code:
  - Find algorithm enum/ID mapping
  - Locate AdaptiveOrder-ML implementation

## Phase 1 — Build the mental model from code

- Produce a written spec:
  - Inputs / outputs
  - Major steps
  - Data structures
  - Where state is persisted (weights, types)
- Produce a truth table of fallbacks:
  - When OOD triggers
  - When margin triggers
  - What "ORIGINAL" means

## Phase 2 — Correctness & safety audit

- Verify permutation is a bijection.
- Verify cross-community edge remap correctness.
- Verify determinism (seeds, stable sorting).
- Add assertions/tests where cheap.

## Phase 3 — Performance accounting

- Identify and time each stage:
  - Leiden / partitioning
  - Feature extraction
  - Scoring
  - Reorder build
  - Relabel / apply
- Add optional profiling hooks guarded by flags.

## Phase 4 — Model quality evaluation

- Define "oracle best" for:
  - Whole-graph
  - Per-community
- Implement analysis scripts to compute:
  - Selection accuracy
  - Regret distribution
  - Confusion matrix

## Phase 5 — Ablations

- Add experiment toggles to isolate:
  - Types vs no-types
  - Leiden vs no-Leiden
  - OOD/margin vs none
  - Feature ablations
  - 1-level vs 2-level partitioning

## Phase 6 — Improvements (SOTA-inspired)

- Propose upgrades in tiers:
  - Tier 1: minimal-risk (cost model, calibration)
  - Tier 2: moderate (bandit on low-margin)
  - Tier 3: heavy (learned reordering option)
- For each: define tests + success criteria

---

## Rules

- Never change behavior without a test or measurement plan.
- Prefer PRs that are easy to revert.

---

## Mandatory Development Protocol

After **any** code change to reordering code:

### Step 1: Build
```bash
cd /home/ab/Documents/00_github_repos/02_GraphBrew
make clean && make -j$(nproc)
```
Abort if build fails. Fix compilation errors first.

### Step 2: Smoke Test (30 seconds)
Quick sanity — raw binary calls allowed here only:
```bash
for v in "17:vibe:rabbit" "17:vibe:hrab" "17:vibe:hrab:gordi" "8:boost"; do
  echo "=== $v ===" && \
  ./bench/bin/pr -f results/graphs/soc-Epinions1/soc-Epinions1.sg -o "$v" -n 2 2>&1 | tail -3
done
```
All 4 must complete without error. Abort if any crash or timeout.

### Step 3: Test Tier (~2 min)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size small --auto --skip-cache \
  --graph-list ca-GrQc email-Enron soc-Slashdot0902 \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs --trials 2
```
Verify no crashes, no timeouts, results JSON is generated.

### Step 4: Medium Evaluation
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size medium --auto --skip-cache --skip-slow \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs cc sssp --trials 5
```
Wait for completion. Evaluate results (see Step 5).

### Step 5: Evaluate Results
```bash
# Find the latest result files
ls -t results/benchmark_*.json | head -1
ls -t results/reorder_*.json | head -1
```

Load the JSON and compute:
1. **Geo-mean speedup** of each VIBE variant vs `RABBITORDER_boost` per benchmark
2. **Per-graph-type breakdown**: social, web, road, mesh, synthetic, citation
3. **Outlier detection**: any benchmark where a VIBE variant is >5× slower than boost
4. **Reorder time ratio**: `vibe:rabbit` should be ≤1.5× boost reorder time

### Step 6: Pass/Fail Criteria

**PASS** — ship the change:
- Geo-mean `vibe:rabbit` vs `boost` is ≥ 0.90× on PR (within 10%)
- No benchmark where ANY VIBE variant is >3× slower than boost
- `vibe:hrab` or `vibe:hrab:gordi` wins on SSSP geo-mean
- All variants complete on all graphs without crash/timeout

**FAIL** — investigate and iterate:
- Any variant crashes or times out on any graph
- `vibe:rabbit` is >20% slower than boost on PR geo-mean
- More than 3 graph×algo combinations where VIBE is >3× slower
- SSSP geo-mean of best VIBE hybrid is worse than boost

### Step 7: If FAIL — Debug & Iterate
1. Read specific log files: `results/logs/{graph}/benchmark_LeidenCSR_{variant}_{bench}.log`
2. Compare reorder times from `results/reorder_*.json`
3. Run focused investigation with high trials:
   ```bash
   python3 scripts/graphbrew_experiment.py --full --auto --skip-cache \
     --graph-list <problem_graph> \
     --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
     --rabbit-variants boost \
     --benchmarks <problem_bench> --trials 10
   ```
4. Check community detection quality (set `VIBE_TRACE=1` env var for debug output)
5. Make targeted fix
6. **GOTO Step 1** — rebuild, smoke test, test tier, medium evaluation

### Step 8: If PASS — Promote to Large
Run the **Large** tier as final validation before committing:
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size large --auto --skip-cache --skip-slow \
  --csr-variants vibe:rabbit vibe:hrab vibe:hrab:gordi \
  --rabbit-variants boost \
  --benchmarks pr bfs cc sssp --trials 5
```
