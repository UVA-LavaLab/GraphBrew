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
| Confirmed GraphBrew quality point | `12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8` |
| Road / mesh diagnostic | `-o 11:bnf` |
| Reproducing the frozen study | [Reproducible-Experiments](Reproducible-Experiments) covers the full matrix |

There is no universal winner. Compare complete preprocessing cost, kernel
time, iteration/work changes, and expected mapping reuse on your workload.

## Why can composition beat one uniform reordering?

Different intra-block layouts optimize different kernel behavior while the
community membership stays fixed:

- LocalGorder8 is 1.143x faster than HubSort for BFS through a resolved
  per-edge locality/throughput advantage.
- LocalGorder8 is 1.248x faster than HubSort and 1.514x faster than RCMpp for
  Afforest CC because it triggers fewer compression steps.
- HubSort is 1.332x faster than LocalGorder8 for PR, but the available
  single-thread cache model does not explain that hardware result.

SizeDesc versus DegreeDesc has no universal main effect. See
[Evidence and Claims](Evidence-and-Claims#why-selected-compositions-work).

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

## What does the paper claim?

The paper claims:

- one fixed GraphBrew composition that is 1.052x faster in kernel GM and
  24.8% cheaper to map than GORDER_csr;
- a workload-dependent composition space in which all seven sealed candidates
  win cells and the oracle is 1.229x faster than the best fixed GraphBrew arm;
  and
- Compact-and-Emit as a permutation-preserving construction optimization.

It does not claim a universal winner, a Rabbit-cost-balanced arm, or an
automatic generator: the frozen family+kernel rule reaches only 0.896x
against the fastest comparator.
The 4% Rabbit per-cell GM does not offset 17–19x mapping cost, and summed
Rabbit seconds never lose. See
[Evidence and Claims](Evidence-and-Claims).

## What is AdaptiveOrder's status?

Algorithm 14 remains available for compatibility and historical reproduction.
Its earlier low-reuse rule falls back to Boost Rabbit and is not a headline
paper contribution. New scientific comparisons should use explicit
Algorithm-12 compositions and include ORIGINAL.

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
  title  = {GraphBrew: Composable and Explainable Vertex Layouts for Graph Analytics},
  author = {Mughrabi, Abdullah T. and Baradaran, Morteza and Ibrahim,
            Mohannad M. and Byrd, Gregory T. and Skadron, Kevin},
  year   = 2026,
  howpublished = {\url{https://github.com/UVA-LavaLab/GraphBrew}},
}
```

## What is Compact-and-Emit?

It compacts active one-pass community IDs and writes final IDs during
intra-community BFS. The final permutation is unchanged; only construction
work is removed. See [GraphBrewOrder](GraphBrewOrder#compact-and-emit).

## Common errors

| Error | Fix |
|---|---|
| `fatal error: boost/range/algorithm.hpp` | `sudo apt-get install libboost-all-dev` |
| `-fopenmp not supported` | `sudo apt-get install libomp-dev` |
| `g++ unrecognized command line option '-std=c++17'` | install GCC 7+ |
| `Cannot allocate memory` while building | `make -j2` instead of `-j` |
| `*.sg file not found` after rebuild | re-run with `-f graph.el`; the binary will regenerate `.sg` |
| Algorithm 14 selects an unexpected arm | Treat it as a compatibility surface; reproduce the exact historical policy or use an explicit Algorithm-12 composition |

More in [Troubleshooting](Troubleshooting).
