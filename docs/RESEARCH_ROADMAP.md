# GraphBrew Research Roadmap

## Objective

Build a generic, reproducible graph-reordering research platform around a
GraphBrew-native ordering algorithm that is not a RabbitOrder, Gorder, or
Leiden derivative. The new algorithm must beat RabbitOrder on controlled
kernel performance while retaining lightweight preprocessing, and must beat
Gorder on both kernel performance and reorder cost. Adaptive selection follows
only after independent GraphBrew-native Pareto points exist.

The normal experiment entry point is `scripts/graphbrew_experiment.py`.
Reusable experiment, parsing, storage, and analysis logic belongs under
`scripts/lib/`. Paper-replication paths remain isolated consumers of those
shared contracts.

## Current Evidence

The completed corpus establishes the following controlled-work frontier:

| Ordering | Controlled kernel GM | All-kernel GM | Reorder GM |
|---|---:|---:|---:|
| DBG | 1.147x | 1.043x | 0.80 s |
| Rabbit-Blocks-DBG | 1.643x | 1.672x | 9.31 s |
| Rabbit-Dendrogram-DFS | 1.782x | 1.691x | 9.39 s |
| RabbitOrder CSR | 1.758x | 1.713x | 10.77 s |
| Gorder | 1.695x | 1.687x | 259.17 s |
| LeidenGVE-Blocks-Rabbit | 1.837x | 1.718x | 301.16 s |

DBG amortizes quickly because its logarithmic degree bucketing is extremely
cheap, not because it has the strongest kernel speedup. It preserves input
order inside coarse degree buckets, so labeling remains a controlled factor.
The full GraphBrew Leiden pipeline is kernel-competitive with Rabbit but pays
for global Leiden partitioning, block construction, per-block layout,
validation, and CSR relocation.

## Novelty Boundary

RabbitOrder, Gorder, and Leiden remain external baselines, diagnostic anchors,
and controlled ablations. They are not the implementation substrate of the
headline algorithm.

The new GraphBrew-native ordering must not use:

- Rabbit community assignments, aggregation, or dendrogram traversal;
- Gorder's window objective or a restricted approximation of its scoring loop;
- Leiden communities as the primary partitioning decision;
- an adaptive selector over existing algorithms as a substitute for a new
  ordering contribution.

Generic graph primitives such as CSR scans, degree statistics, sampling,
prefix sums, stable compaction, and BFS are allowed, but the algorithmic
objective and placement decisions must be independently specified.

Before implementation, freeze a novelty matrix comparing the proposed method
against RabbitOrder, Gorder, Leiden-based ordering, DBG/HubSort, Corder,
GoGraph, and recent learned/partitioner-selection work. The matrix records:

- information consumed;
- objective optimized;
- preprocessing complexity and memory;
- partition/block/vertex decisions;
- deterministic behavior;
- whether decisions are global, block-local, or sampled;
- the exact source-code/paper mechanism that overlaps.

If the core mechanism reduces to an existing method under renamed stages, the
proposal is rejected before performance work.

## Phase 0: Data-Integrity and SSOT Cleanup

These items must complete before another broad campaign:

1. Use one Python module identity (`scripts.*`) and one project-root import
   convention.
2. Make result identity condition-aware (`labeling`, `measurement_mode`,
   thread/process policy) and remove min-wins deduplication across conditions.
3. Centralize benchmark lists, ordered evaluation subsets, cache capacities,
   cache iterations, and GraphBrew realized-config validation under
   `scripts/lib/`.
4. Fail closed on retired/non-nested evaluation paths.
5. Make the top-level harness the documented generic front door and keep
   replication-specific runners as isolated consumers.
6. Add one unified build/test entry point and current CI coverage.

Cleanup must preserve existing measurement artifacts and numerical claims.
Mapping-cache unification is deferred until after the next campaign because
the existing artifact tree is frozen evidence.

## Phase 1: Fast-Quality Pareto Pilot

Use the rapid path on twitter7, webbase-2001, and hollywood-2009 before any
full corpus run. Measure load, feature, mapping, validation, apply, kernel,
work, iteration, and peak-RSS phases separately.

Required controls:

- ORIGINAL and DBG cost floors;
- RabbitOrder CSR as the primary performance/cost baseline;
- Rabbit-Dendrogram-DFS and Rabbit-Blocks-DBG as diagnostic derivatives only;
- Gorder as the expensive locality reference;
- LeidenGVE-Blocks-Rabbit as the expensive quality reference.
- independently specified GraphBrew-native candidate(s).

Initial success targets:

- controlled kernel GM at least 1.80x;
- all-kernel GM at least 1.73x;
- paired new-method/Rabbit-CSR controlled-work ratio above 1.03x with the
  graph-bootstrap 95% interval excluding one;
- paired new-method/Gorder kernel ratio above one with the 95% interval
  excluding one;
- reorder GM at most 10 seconds;
- median reorder time at most 8 seconds;
- no graph with more than a 5% kernel regression versus Rabbit CSR;
- deterministic source and mapping contracts;
- lower end-to-end regret than Rabbit CSR at reuse counts 5, 10, and 20.

## Phase 2: Fast Reordering Research

Design one independent algorithm family rather than another composition matrix.
The initial proposal is a bounded-cost topology-sketch and block-packing
framework:

1. **Topology sketch:** deterministic sampled edge-span, degree-distribution,
   hub concentration, and local neighbor-overlap summaries.
2. **Native block formation:** create bounded-size blocks directly from the
   sketch and CSR neighborhoods, without community labels from Rabbit or
   Leiden.
3. **Placement objective:** explicitly minimize a GraphBrew-defined estimate
   of property-line dispersion and cross-block traffic under a fixed
   preprocessing budget.
4. **Stable vertex placement:** order vertices inside blocks with an
   independently specified degree/locality rule; retain stable original-ID
   tie breaks for determinism.
5. **Budget enforcement:** cap sampled edges, block-growth work, passes, and
   auxiliary memory before running the algorithm.
6. **Reusable primitives:** share degree buckets, sketch storage, prefix sums,
   and CSR relocation buffers without importing baseline algorithm state.
7. **Memory discipline:** release sketch/block state before CSR relocation and
   preserve one-base-plus-one-derived-arm residency.

Rabbit-derived and Gorder-derived experiments may diagnose why the independent
candidate wins or loses, but they cannot become the claimed method.

Each optimization must report complexity, mapping determinism, kernel speedup,
reorder cost, apply cost, peak RSS, and amortization. A faster implementation
that loses the controlled-work kernel target does not advance.

## Phase 3: Portfolio and Headroom Gate

Freeze a small portfolio only after Phase 2 identifies genuine Pareto points.
The initial candidate set is:

- ORIGINAL;
- DBG;
- the best independent GraphBrew-native candidate(s).

RabbitOrder CSR, Gorder, and Leiden remain mandatory evaluation baselines and
oracle references, but they are not deployable selector arms in the headline
GraphBrew-native portfolio.

Before model work, require the cross-fitted portfolio oracle to beat the best
static policy by at least 6% overall and not only on road/mesh graphs. If the
headroom gate fails, report the null result and keep the best static policy.

## Phase 4: Adaptive Selection

After the cleanup and headroom gates:

1. collect fresh source-driven and reuse-aware measurements;
2. train only on runtime-emitted Tier-0 features;
3. use nested leave-one-topology-out evaluation with fold-local portfolio
   pruning and OOD calibration;
4. optimize end-to-end regret, not classification accuracy;
5. deploy only a deterministic compact artifact with exact Python/C++ parity;
6. charge feature, model, mapping, validation, and apply costs to selection.

The selector must never consume graph identity or exact-name oracle data.
Report a no-abstention selector, an abstaining selector, the best static
GraphBrew-native policy, Rabbit CSR, Gorder, and the cross-fitted portfolio
oracle.

## Phase 5: Full Evaluation

Only after the pilot, precision, memory, and headroom gates:

- run the frozen graph/kernel/source/reuse matrix through the harness;
- retain rapid and final execution modes as explicit policies;
- regenerate every result, table, PNG, SVG, and report from structured
  manifests;
- preserve a separate, reproducible paper-number replication command without
  duplicating generic registries or parsers.
