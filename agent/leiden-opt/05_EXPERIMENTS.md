# Experiments — Leiden vs RabbitOrder

---

## A) Benchmark Tiers

Same tiers as the adaptive-ml pack. Use the tier that matches feedback speed.

### 🧪 Test — ~2 min
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size small --auto --skip-cache \
  --graph-list ca-GrQc email-Enron soc-Slashdot0902 \
  --csr-variants original rabbit:csr vibe:hrab \
  --benchmarks pr bfs \
  --trials 2
```

### 🔹 Small — ~10 min
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size small --auto --skip-cache \
  --csr-variants original rabbit:csr rabbit:boost \
    vibe:rabbit vibe:hrab vibe:hrab:gordi leiden:gveopt2 \
  --benchmarks pr bfs cc sssp \
  --trials 3
```

### 🔸 Medium — ~30-60 min (primary evaluation)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size medium --auto --skip-cache --skip-slow \
  --csr-variants original rabbit:csr rabbit:boost \
    vibe:rabbit vibe:hrab vibe:hrab:gordi leiden:gveopt2 \
  --benchmarks pr bfs cc sssp \
  --trials 5
```

### 🔷 Large — ~2-4 hours (definitive)
```bash
python3 scripts/graphbrew_experiment.py \
  --full --size large --auto --skip-cache --skip-slow \
  --csr-variants original rabbit:csr rabbit:boost \
    vibe:rabbit vibe:hrab vibe:hrab:gordi leiden:gveopt2 \
  --benchmarks pr bfs cc sssp \
  --trials 5
```

---

## B) Head-to-Head Comparison Protocol

When comparing a new Leiden variant against RabbitOrder:

```bash
python3 scripts/graphbrew_experiment.py \
  --full --size medium --auto --skip-cache \
  --csr-variants original rabbit:csr <your_variant> \
  --benchmarks pr bfs cc sssp \
  --trials 5
```

Report:
1. **Kernel speedup** — `your_variant / rabbit:csr` per graph, then geo-mean
2. **Reorder time ratio** — `your_variant_reorder / rabbit_reorder`
3. **End-to-end ratio** — `(reorder + kernel) ratio`
4. **Win/loss count** — how many graphs does your variant beat rabbit?

---

## C) Resolution Sweep Experiment

Test how resolution affects community size and cache performance:

```bash
for R in 0.25 0.5 0.75 1.0 1.5 2.0; do
  python3 scripts/graphbrew_experiment.py \
    --full --auto --skip-cache \
    --graph-list web-Google cit-Patents soc-Slashdot0902 roadNet-CA \
    --csr-variants "leiden:gveopt2:${R}" rabbit:csr \
    --benchmarks pr bfs \
    --trials 5
done
```

Track: community count, avg community size, kernel time, reorder time.

---

## D) Ordering Strategy Comparison

Test all ordering strategies within VIBE:

```bash
python3 scripts/graphbrew_experiment.py \
  --full --auto --skip-cache \
  --graph-list web-Google cit-Patents soc-Slashdot0902 roadNet-CA \
  --csr-variants vibe vibe:dfs vibe:bfs vibe:rabbit \
    vibe:hrab vibe:hrab:gordi vibe:dbg vibe:corder \
  --benchmarks pr bfs cc sssp \
  --trials 5
```

---

## E) Cache Simulation Experiment

When you need to measure actual cache miss rates (not just kernel time):

```bash
python3 scripts/graphbrew_experiment.py \
  --full --auto \
  --graph-list web-Google soc-Slashdot0902 \
  --csr-variants original rabbit:csr vibe:hrab vibe:hrab:gordi \
  --benchmarks pr bfs \
  --trials 1 \
  --cache-sim
```

Compare: L1/L2/L3 miss rates per variant.

---

## F) Graph Categories & Expected Behaviour

| Category | Graphs | RabbitOrder Strength | Leiden Opportunity |
|----------|--------|:---:|:---:|
| **Social** | soc-Epinions1, soc-Slashdot0902 | Strong (natural communities) | Refinement may find better communities |
| **Web** | web-Google, web-BerkStan, cnr-2000 | Strong (link structure) | Hierarchical ordering could help |
| **Road** | roadNet-CA/PA/TX | Moderate (grid-like) | BFS/DFS ordering may dominate |
| **Mesh** | delaunay_n17-20, rgg_n_2_17-19 | Weak (no communities) | Locality-based ordering needed |
| **Citation** | cit-Patents, cit-HepPh | Moderate | Community-aware ordering |
| **Synthetic** | preferentialAttachment, smallworld | Variable | Resolution tuning |

**Focus on:** Social and Web — these are where RabbitOrder is strongest and
where beating it matters most. Road and Mesh are secondary targets.

---

## G) Refinement Ablation

Test whether Leiden refinement (the key differentiator from Louvain/RabbitOrder)
actually improves cache locality:

```bash
# Need to modify VibeConfig.useRefinement in code, then:
python3 scripts/graphbrew_experiment.py \
  --full --size medium --auto --skip-cache \
  --csr-variants vibe:hrab <vibe:hrab:no-refine> rabbit:csr \
  --benchmarks pr bfs cc sssp \
  --trials 5
```

This answers: "Is Leiden refinement worth its cost for reordering?"
