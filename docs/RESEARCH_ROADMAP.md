# GraphBrew Research Roadmap

## Objective

Build a generic, reproducible graph-reordering research platform that can
deliver Gorder/Leiden-class kernel quality with Rabbit/DBG-class preprocessing
cost, then select the appropriate Pareto-optimal ordering from lightweight
graph and workload features.

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
- RabbitOrder CSR and Rabbit-Dendrogram-DFS;
- Rabbit-Blocks-DBG and Rabbit-Identity-HubSort;
- Gorder as the expensive locality reference;
- LeidenGVE-Blocks-Rabbit as the expensive quality reference.

Initial success targets:

- controlled kernel GM at least 1.78x;
- reorder GM at most 12 seconds;
- median reorder time at most 8 seconds;
- no graph with more than a 5% kernel regression versus Rabbit CSR;
- deterministic source and mapping contracts;
- lower end-to-end regret than Rabbit CSR at reuse counts 5, 10, and 20.

## Phase 2: Fast Reordering Research

Prioritize changes that attack measured critical paths rather than adding
another broad variant matrix:

1. **Rabbit critical path:** reduce aggregation and dendrogram/layout overhead
   while preserving the current 1.78x controlled speedup.
2. **Selective block refinement:** apply HubSort, DBG, RCM, or small-window
   Gorder only to blocks whose degree/locality features justify the cost.
3. **Budgeted community detection:** test deterministic sampled or pass-limited
   community discovery with explicit quality-versus-cost curves.
4. **Reusable primitives:** share degree buckets, block metadata, and CSR
   relocation buffers across partition and layout stages.
5. **Fast Gorder approximation:** restrict window scoring to sampled hot
   vertices or selected blocks instead of scanning the full graph.
6. **Memory discipline:** release intermediate community/supergraph state
   before CSR relocation and preserve one-base-plus-one-derived-arm residency.

Each optimization must report complexity, mapping determinism, kernel speedup,
reorder cost, apply cost, peak RSS, and amortization. A faster implementation
that loses the controlled-work kernel target does not advance.

## Phase 3: Portfolio and Headroom Gate

Freeze a small portfolio only after Phase 2 identifies genuine Pareto points.
The initial candidate set is:

- ORIGINAL;
- DBG;
- RabbitOrder CSR;
- Rabbit-Dendrogram-DFS;
- Rabbit-Blocks-DBG;
- the best new fast-quality candidate(s).

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
policy, Rabbit CSR, and the cross-fitted portfolio oracle.

## Phase 5: Full Evaluation

Only after the pilot, precision, memory, and headroom gates:

- run the frozen graph/kernel/source/reuse matrix through the harness;
- retain rapid and final execution modes as explicit policies;
- regenerate every result, table, PNG, SVG, and report from structured
  manifests;
- preserve a separate, reproducible paper-number replication command without
  duplicating generic registries or parsers.
