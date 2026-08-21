# AdaptiveOrder

AdaptiveOrder (`-o 14`) is GraphBrew's runtime selection boundary. The
validated path is a **frozen deterministic rule**, not a machine-learning
model.

For the complete decision, examples, and evidence, see
[All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector).

## Interface

```bash
# Reuse one materialized mapping once
./bench/bin/pr -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:1' \
  -n 3

# Reuse one materialized mapping twice
./bench/bin/bfs -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:2' \
  -n 3
```

Reuse is mandatory and must be `1` or `2`.

## Decision boundary

AdaptiveOrder does not require a previous kernel run. It:

1. samples degree structure from the new graph;
2. models the kernel property footprint relative to machine LLC;
3. reads kernel identity and declared reuse;
4. applies the frozen predicate once; and
5. chooses the promoted GraphBrew composition or Boost Rabbit.

It does not use graph names, benchmark-database lookup, runtime training,
nearest-neighbor search, or trial execution of candidate reorderers.

## Outcomes

GraphBrew branch:

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:
cd_parallel:sgmb4096:gordf5000:norefine:2:2
```

Fallback branch:

```text
8:boost
```

The branch decision is deterministic. A selected GraphBrew mapping can still
vary because `cd_parallel` is schedule-sensitive.

## Validated scope

- PR and PR-SpMV
- BFS
- Afforest CC and CC-SV
- BC
- SSSP
- reuse 1 or 2
- graphs with at least 1000 vertices

Unsupported contexts use the fallback.

## Timing boundary

The public portfolio evidence accounts for:

```text
chosen mapping cost + reuse x chosen kernel time
```

It does not store `Adaptive Feature Time`. The deployable binary prints that
value separately, and final deployment timing must add it.

Legacy offline-model modes remain for research compatibility but are not the
validated contribution.

## Evidence

- [Low-Reuse Selector](All-Kernel-Low-Reuse-Selector)
- [`docs/allkernel-lowreuse-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/allkernel-lowreuse-evidence.json)
- [`docs/recommendation-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/recommendation-evidence.json)
