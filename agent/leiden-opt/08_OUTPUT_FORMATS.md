# Output Formats — Leiden Optimization

---

## Experiment Report Format

After every measurement phase, produce a summary table:

```
## Experiment: <description>
Date: YYYY-MM-DD
Tier: Small / Medium / Large
Graphs: <count> (<list or "MEDIUM catalog">)
Trials: <N>

### Kernel Time (seconds, geo-mean across graphs)

| Variant | PR | BFS | CC | SSSP | Geo-Mean |
|---------|:--:|:---:|:--:|:----:|:--------:|
| original | X.XX | X.XX | X.XX | X.XX | X.XX |
| rabbit:csr | X.XX | X.XX | X.XX | X.XX | X.XX |
| <your variant> | X.XX | X.XX | X.XX | X.XX | X.XX |

### Reorder Time (seconds, geo-mean)

| Variant | Reorder | Kernel | End-to-End |
|---------|:-------:|:------:|:----------:|
| rabbit:csr | X.XX | X.XX | X.XX |
| <your variant> | X.XX | X.XX | X.XX |

### Speedup vs RabbitOrder

| Variant | Kernel Speedup | E2E Speedup | Win/Loss |
|---------|:-:|:-:|:-:|
| <your variant> | X.XXx | X.XXx | W-L |

### Conclusion
<one paragraph: hypothesis confirmed/rejected, what to try next>
```

---

## Hypothesis Log Format

Keep a running log of what was tried:

```
### Hypothesis #N: <title>
Date: YYYY-MM-DD
Parameter changed: <what>
Expected: <prediction>
Result: <actual outcome>
Decision: KEEP / REVERT / INVESTIGATE
Evidence: <link to experiment report or commit>
```

---

## Cache Simulation Report Format

When using `--cache-sim`:

```
### Cache Miss Rates

| Variant | L1 Hit% | L2 Hit% | L3 Hit% | DRAM% |
|---------|:-------:|:-------:|:-------:|:-----:|
| original | XX.X | XX.X | XX.X | XX.X |
| rabbit:csr | XX.X | XX.X | XX.X | XX.X |
| <your variant> | XX.X | XX.X | XX.X | XX.X |
```

---

## Commit Message Format

```
leiden: <what changed> — <measured impact>

<optional body with experiment details>
```

Examples:
```
leiden: increase default resolution to 1.0 — 3% kernel speedup on MEDIUM tier
leiden: add hub extraction to vibe:hrab — 5% L3 miss reduction on social graphs
leiden: disable refinement for small communities — 25% faster reorder, <1% quality loss
```
