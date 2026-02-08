# Mission: Beat RabbitOrder with Leiden Variants

## Goal

Make Leiden-based reordering (LeidenCSR variants + VIBE framework) consistently
outperform standalone RabbitOrder (`-o 8`) on:

1. **Cache quality** — lower L1/L2/L3 miss rates measured by cache simulation
2. **Benchmark speedup** — faster PR, BFS, CC, SSSP kernel execution times
3. **End-to-end time** — including reorder cost (reorder time + kernel time)

## Why This Matters

RabbitOrder (Arai et al., IPDPS'16) is the current gold standard for fast,
high-quality graph reordering. It uses hierarchical Louvain clustering +
parallel incremental aggregation to produce locality-preserving orderings
fast enough to amortize in a single traversal.

Leiden (Traag et al., 2019) is theoretically superior to Louvain — it guarantees
well-connected communities via a refinement phase. GraphBrew has an extensive
Leiden implementation with many variants. The question is whether this
theoretical advantage translates to better cache locality and benchmark
performance in practice.

## Constraints

- **No correctness regressions** — graph algorithm outputs must be identical
- **Reorder time budget** — Leiden variants must not be >3× slower to reorder
  than RabbitOrder unless the kernel speedup justifies it
- **Reproducibility** — every experiment uses `graphbrew_experiment.py` with
  fixed seeds and ≥3 trials
- **One variable at a time** — change one parameter/variant per experiment

## The Competition

| Algorithm | CLI | Reorder Speed | Locality Quality |
|-----------|-----|:---:|:---:|
| RabbitOrder (Boost) | `-o 8:boost` | Fast | High (Louvain) |
| RabbitOrder (CSR) | `-o 8:csr` | Fast | High (Louvain) |
| LeidenCSR:gveopt2 | `-o 17:gveopt2` | Moderate | High (Leiden) |
| VIBE:rabbit | `vibe:rabbit` | Fast | Medium-High |
| VIBE:hrab | `vibe:hrab` | Moderate | Very High |
| VIBE:hrab:gordi | `vibe:hrab:gordi` | Slow | Highest |

## Success Criteria

1. **Primary:** At least one Leiden variant beats RabbitOrder on geo-mean
   kernel speedup across MEDIUM tier (28 graphs × 4 benchmarks)
2. **Secondary:** That variant also wins on end-to-end time (including
   reorder cost) on ≥60% of graphs
3. **Stretch:** Find a variant that beats RabbitOrder on cache miss rate
   (L3 miss reduction > 10%) while being within 2× reorder time

## Non-Goals

- Modifying the AdaptiveOrder perceptron (that's the `adaptive-ml` pack)
- Adding entirely new reordering algorithms
- Optimising for graphs < 10K vertices (too small to benefit from reordering)
