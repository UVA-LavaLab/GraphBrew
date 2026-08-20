# AdaptiveOrder

AdaptiveOrder is GraphBrew's runtime selection boundary. The current validated
deployment path is a **frozen deterministic rule**, not a machine-learning
model.

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

The rule may use:

- graph size;
- sampled average degree and degree coefficient of variation;
- sampled hub concentration;
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

## Historical offline-model code

The repository retains perceptron, decision-tree, hybrid, and model-emulation
code for experiments and compatibility. Those paths are not the validated
result described by the README or paper. Benchmark binaries never train a
model at runtime.

The older PR-only `budgeted-rule` also remains available as a separately
scoped historical rule. New deployment claims should use
`allkernel-lowreuse-rule`.

## Evidence

The final rule was derived on 18 graphs, frozen, and tested on 12 additional
graphs. All seven selected holdouts won at reuse 1 and 2; five graphs used
Boost Rabbit fallback.

Public evidence:

- [`docs/allkernel-lowreuse-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/allkernel-lowreuse-evidence.json)
- [`docs/recommendation-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/recommendation-evidence.json)

