# AdaptiveOrder

AdaptiveOrder is GraphBrew's runtime selection boundary. The current validated
deployment path is a **frozen deterministic rule**, not a machine-learning
model.

It does not require a previous kernel run. For a new graph, it samples graph
structure, combines that sample with kernel identity, LLC capacity, and
declared reuse, and applies the frozen predicate once.

## Validated interface

```bash
# One kernel invocation reusing one materialized mapping
./bench/bin/pr -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:1' \
  -n 3

# Two kernel invocations reusing one materialized mapping
./bench/bin/bfs -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:2' \
  -n 3
```

Reuse is mandatory and must be at most 2.

## Runtime inputs

The validated v2 rule uses:

- graph size;
- sampled average degree and degree coefficient of variation;
- kernel-specific property working set relative to LLC;
- kernel identity; and
- declared mapping reuse.

It may not use:

- graph filename or canonical graph name;
- benchmark-database lookup;
- runtime training;
- runtime k-nearest-neighbor search; or
- trial execution of multiple reorderers.

## Decision

The rule chooses between:

1. **FastLeiden-SizeDesc-Gorder8**

   ```text
   12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:
   cd_parallel:sgmb4096:gordf5000:norefine:2:2
   ```

2. **Boost Rabbit** fallback.

The exact frozen predicate and validation figures are documented in
[All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector).

## Supported scope

- PR
- PR-SpMV
- BFS
- Afforest CC
- CC-SV
- BC
- SSSP
- reuse 1 or 2

Unsupported kernels and reuse above 2 use the fallback.

The decision is deterministic. The selected GraphBrew mapping can still vary
because its `cd_parallel` community detection is schedule-sensitive.

## Evidence accounting

The public portfolio ratios account for chosen mapping cost plus reused kernel
time. They do not store algorithm-14 feature-extraction time. The deployable
binary prints `Adaptive Feature Time`, which must be included in final
deployment timing.

Legacy offline-model modes remain for research compatibility but are not the
validated contribution.

## Evidence

The final rule was derived on 18 graphs, frozen, and tested on 12 additional
graphs. All seven selected holdouts won at reuse 1 and 2; five graphs used
Boost Rabbit fallback.

Public evidence:

- [`docs/allkernel-lowreuse-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/allkernel-lowreuse-evidence.json)
- [`docs/recommendation-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/recommendation-evidence.json)
