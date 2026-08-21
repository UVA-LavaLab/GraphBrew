# FAQ

Short answers to common questions. For full guides see
[Getting-Started](Getting-Started), [Reordering-Algorithms](Reordering-Algorithms),
and [Troubleshooting](Troubleshooting).

## Which reordering should I try first?

| Situation | Try |
|---|---|
| Establish a baseline | `-o 0` (INPUT-SHUFFLED/ORIGINAL) |
| Very low reorder cost | `-o 5` (DBG) |
| Strong general baseline | `-o 8:csr` (RabbitOrder CSR) |
| Expensive locality reference | `-o 9:csr` (exact Gorder) |
| Road / mesh diagnostic | `-o 11:bnf` |
| Reproducing the frozen study | [Reproducible-Experiments](Reproducible-Experiments) covers the full matrix |

There is no universal winner. Compare complete preprocessing cost, kernel
time, iteration/work changes, and expected mapping reuse on your workload.

## How much speedup should I expect?

It depends on (a) how cache-unfriendly the original ordering is and
(b) how many iterations your benchmark runs.

Report speedup against the exact input-label baseline and include complete
reorder, validation, and CSR-application cost. For iterative kernels, report
iteration count and time per iteration separately.

## Why does reordering sometimes hurt?

Three common reasons:

1. **The original ordering is already good.** Many graphs ship from
   their source in a near-optimal layout (e.g. citation networks
   crawled chronologically). Reordering only adds overhead.
2. **You ran too few iterations to amortize reorder cost.** Reorder
   time is paid once; kernel speedup is paid back per trial. With
   `-n 1` you're seeing reorder + 1 kernel, which is often slower
   than ORIGINAL × 1.
3. **The reordering doesn't match the access pattern.** Degree, bandwidth,
   community, and window-locality methods optimize different structural
   signals. Measure the kernel and graph combination you intend to use.

## Where do my benchmark results land?

| Output | Location |
|---|---|
| Standard runs | stdout (`Read Time`, `Build Time`, `Average Time`) |
| Explicit C++ self-recording (`-D DIR`) | `DIR/benchmarks.json` |
| Pipeline (`graphbrew_experiment.py`) | `results/data/benchmarks.json` |
| Frozen-study runs | the configured `--paper-artifact-root` |
| Historical offline models | `results/data/adaptive_models.json` |

## Does AdaptiveOrder require a trained model?

No. The validated `allkernel-lowreuse-rule` is a frozen deterministic
predicate and does not load `adaptive_models.json`. Historical model
experiments can still read that file. See [AdaptiveOrder](AdaptiveOrder).

## Does the low-reuse rule need an earlier kernel run?

No. It samples the new graph directly and uses kernel identity, LLC capacity,
and declared reuse. It does not execute candidate orderings or consult prior
benchmark rows. See
[All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector).

## Why does the policy fall back to Rabbit?

The promoted non-Rabbit composition is not a universal winner. On fallback
graphs the policy ties the always-Rabbit baseline rather than claiming a
GraphBrew win. Selected GraphBrew graphs contribute the portfolio gains. The
current policy therefore beats always-Rabbit in aggregate but is not
Rabbit-free.

## How do I add a new algorithm or benchmark?

See [Contributing](Contributing).

## What graph formats are supported?

`.sg` (GAPBS binary, fastest), `.el` (edge list, text), `.wel`
(weighted edge list), `.mtx` (Matrix Market). The first run on a
new graph creates a `.sg` cache alongside the input.
See [Supported-Graph-Formats](Supported-Graph-Formats).

## What's the difference between LeidenOrder (15) and GraphBrew-Leiden (12:leiden)?

- `-o 15` runs GVE-Leiden and an explicit GraphBrew post-layout. Use the
  layout option to distinguish hierarchy-degree, final-stable, and
  final-degree behavior.
- `-o 12:leiden` is the composable GraphBrew pipeline with Leiden as one
  partitioning choice. It is a different pipeline, not a guaranteed
  improvement over Algorithm 15.

## When should I use DBG vs HUBCLUSTER?

DBG buckets vertices by `log2(degree)`. HUBCLUSTER does a binary
split (hubs vs non-hubs) and only sorts the hub partition.

- **HUBCLUSTER** preserves non-hub spatial structure, so it composes
  well as a layer on top of a community-aware ordering. This is why
  `12:hubcluster` works.
- **DBG** redistributes everything by degree bucket. Use it
  standalone (`-o 5`) for fast experiments, or chained as a refinement
  step (`-o 12:leiden -o 5`) when both community structure and hub
  temporal locality matter.

## How do I cite GraphBrew?

Cite the repository:

```bibtex
@misc{graphbrew,
  title  = {GraphBrew: Multilayered Graph Reordering for Accelerated Graph Processing},
  author = {Mughrabi, Abdullah T. and Baradaran, Morteza and Ibrahim,
            Mohannad M. and Byrd, Gregory T. and Skadron, Kevin},
  year   = 2026,
  howpublished = {\url{https://github.com/UVA-LavaLab/GraphBrew}},
}
```

## Where is the runtime selection documentation?

[AdaptiveOrder](AdaptiveOrder).

## Common errors

| Error | Fix |
|---|---|
| `fatal error: boost/range/algorithm.hpp` | `sudo apt-get install libboost-all-dev` |
| `-fopenmp not supported` | `sudo apt-get install libomp-dev` |
| `g++ unrecognized command line option '-std=c++17'` | install GCC 7+ |
| `Cannot allocate memory` while building | `make -j2` instead of `-j` |
| `*.sg file not found` after rebuild | re-run with `-f graph.el`; the binary will regenerate `.sg` |
| The low-reuse rule always picks Rabbit | Confirm the exact algorithm-14 string, supported kernel, reuse 1 or 2, and graph size; fallback is expected when the frozen predicate is false |

More in [Troubleshooting](Troubleshooting).
