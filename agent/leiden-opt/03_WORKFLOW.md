# Workflow — Leiden Optimization Iteration Loop

Every change follows this protocol. No exceptions.

---

## Phase 0: Understand the Baseline

Before changing anything, establish where RabbitOrder wins and where Leiden
variants are close.

```bash
# Baseline: RabbitOrder vs current best Leiden variants (MEDIUM tier)
python3 scripts/graphbrew_experiment.py \
  --full --size medium --auto --skip-cache \
  --csr-variants original rabbit:csr rabbit:boost \
    vibe:rabbit vibe:hrab vibe:hrab:gordi \
    leiden:gveopt2 \
  --benchmarks pr bfs cc sssp \
  --trials 5
```

Record:
- Per-graph kernel times for each variant
- Reorder times for each variant
- End-to-end times (reorder + kernel)
- Identify graphs where RabbitOrder wins by >10%

---

## Phase 1: Form a Hypothesis

Pick ONE thing to try. Examples:
- "Increasing resolution from 0.75 to 1.2 on social graphs will produce
  smaller communities that fit L2 cache better"
- "Enabling Gorder intra-community ordering (gordi) on vibe:hrab will
  improve locality within communities"
- "Disabling refinement on graphs with packing_factor > 0.6 saves time
  without hurting quality"

Write the hypothesis down BEFORE running experiments.

---

## Phase 2: Implement the Change

### If tuning parameters (no code change):
Pass parameters directly via CLI:
```bash
# Example: test resolution 1.2 on vibe:hrab
python3 scripts/graphbrew_experiment.py \
  --full --auto --skip-cache \
  --graph-list web-Google cit-Patents soc-Slashdot0902 \
  --csr-variants vibe:hrab "leiden:gveopt2:1.2" \
  --benchmarks pr bfs \
  --trials 5
```

### If modifying code:
1. Create a git branch: `git checkout -b leiden/try-<hypothesis>`
2. Edit the relevant file (usually `reorder_vibe.h` or `reorder_leiden.h`)
3. Compile: `make`
4. Smoke test: run Test tier (3 graphs, 2 benchmarks, 2 trials)
5. If smoke passes, run Small tier

### Mandatory development protocol:
- **Compile after every edit** — never batch multiple untested changes
- **Test tier after compile** — catches crashes and correctness bugs fast
- **One variable at a time** — if you change resolution AND iterations
  simultaneously, you learn nothing

---

## Phase 3: Measure

Run the same benchmark configuration as Phase 0 with your change:

```bash
python3 scripts/graphbrew_experiment.py \
  --full --size medium --auto --skip-cache \
  --csr-variants original rabbit:csr <your_variant> \
  --benchmarks pr bfs cc sssp \
  --trials 5
```

Compare against baseline:
- Did kernel time improve on the graphs where RabbitOrder was winning?
- Did reorder time stay within budget (≤3× RabbitOrder)?
- What's the geo-mean speedup vs RabbitOrder?

---

## Phase 4: Decide

| Outcome | Action |
|---------|--------|
| Geo-mean improved, no regressions | **KEEP** — commit to main |
| Improved on some graphs, regressed on others | **ANALYZE** — which graph categories? Conditional application? |
| No change or worse | **REVERT** — `git checkout main`, document what was tried |
| Unclear | **MORE DATA** — run Large tier or add trials |

### If keeping:
```bash
git add -f <changed files>
git commit -m "leiden: <what changed> — <measured improvement>"
```

### If reverting:
Document in a scratchpad what you tried and why it didn't work.
This prevents re-trying the same thing later.

---

## Phase 5: Next Iteration

Go back to Phase 1. Pick the next hypothesis based on:
1. Where does RabbitOrder still win? Focus there.
2. What parameter hasn't been explored yet?
3. What variant combination hasn't been tested?

---

## Iteration Examples

### Example A: Resolution Sweep
```
Hypothesis: Resolution 1.5 produces smaller communities → better L1 hit rate
Phase 2: Run with --csr-variants "leiden:gveopt2:1.5"
Phase 3: Compare L1 miss rate vs default resolution (0.75)
Phase 4: If L1 improves but kernel time doesn't, the bottleneck is elsewhere → try L3
```

### Example B: Ordering Strategy Comparison
```
Hypothesis: DFS dendrogram ordering beats BFS for road networks
Phase 2: Run vibe:dfs vs vibe:bfs on roadNet-CA, roadNet-PA, roadNet-TX
Phase 3: Compare kernel times across BFS/SSSP/CC
Phase 4: If DFS wins on roads but loses on social, add conditional logic
```

### Example C: Refinement Ablation
```
Hypothesis: Leiden refinement adds reorder cost but doesn't improve locality
Phase 2: Modify VibeConfig to set useRefinement = false
Phase 3: Compare reorder time AND cache miss rate
Phase 4: If refinement saves 30% reorder time with <2% cache quality loss, disable it
```
