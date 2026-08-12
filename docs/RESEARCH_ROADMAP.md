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

## Novelty Landscape Gate

The first primary-source review rejects the Phase-2 proposal **as originally
worded**. "Sample topology, form cache-sized blocks, and minimize cross-block
traffic/property dispersion" is not a defensible novelty statement by itself.

Highest-risk prior art:

| Method | Verified mechanism | Collision with the original proposal |
|---|---|---|
| [FrontOrder](https://github.com/RecoderChris/FrontOrder) | Random BFS-frontier sampling, learned vertex features, K-means locality clusters, cache-sized partitions | Directly anticipates sampled topology signals followed by cache-sized vertex blocks |
| [Rebo](https://www.cs.nthu.edu.tw/~ychung/Conference/2026-IPDPS.pdf) | In-degree/structure hot-cold-frozen reordering, dense/sparse matrix extraction, cache-resident 2-D blocking, sparse cache-line arrays | Directly anticipates the combined "reordering and blocking" framing |
| [Gorder](https://github.com/datourat/Gorder) | Greedy maximization of a sibling/neighbor locality score inside a sliding window | Any pairwise or windowed neighbor-co-location objective is derivative |
| [DON-RL](https://arxiv.org/abs/2001.06631) | Learns a policy that maximizes the Gorder locality objective | Learning or approximating the same score does not escape the Gorder boundary |
| [Recursive Graph Bisection](https://arxiv.org/abs/1602.08820) | Recursive balanced bisection for minimum-logarithmic-arrangement-style compression/locality | A recursive cross-block or log-gap objective is already occupied |
| [Hypergraph communication-volume partitioning](https://faculty.cc.gatech.edu/~umit/assets/pdf/Catalyurek99.pdf) | Row/column neighborhoods are hyperedges; connectivity `lambda(net)-1` counts distinct parts touched | A sum of distinct cache-line groups per neighborhood is this standard connectivity metric |
| [Corder](https://github.com/yuang-chen/Corder-TPDS-21) | Degree hot/cold classification and workload-balanced cache-sized segments | Hot/cold cache-segment block packing is already occupied |
| [Lightweight reordering](https://github.com/CMUAbstract/Graph-Reordering-IISWC18) and [DBG](https://github.com/faldupriyank/dbg) | Packing-factor analysis, HubSort/HubCluster, stable degree grouping | Degree placement and packing-factor rationale are components, not novelty |
| [RabbitOrder](https://github.com/araij/rabbit_order) / [Leiden](https://doi.org/10.1038/s41598-019-41695-z) | Modularity communities, aggregation, and hierarchy/refinement | Community-derived blocks remain forbidden |
| [GoGraph](https://arxiv.org/abs/2407.14544) | Maximizes forward edges to reduce asynchronous convergence rounds | Must remain a convergence baseline, not a cache-locality building block |

The follow-up proof review also rejects the proposed cache-line
set-cardinality objective:

```text
J_line(sigma) =
  sum_v w_v * |{ floor(sigma(u) / L) : u in S(v) }|
  + lambda * sum_b overfill(b)
```

Treat each sampled neighborhood `S(v)` as a hyperedge and each cache-line group
as a part. Then the first term, minus the constant `sum_v w_v`, is exactly the
weighted hypergraph connectivity (`lambda-1`) metric used to model
communication volume. Sampling and hard budgets change cost, not the
underlying objective. Implementing this under a GraphBrew name would therefore
be a bounded hypergraph-partitioning approximation, not an independent
ordering objective.

**No Phase-2 objective is currently frozen. Implementation remains blocked.**
The next landscape pass must examine cache-line transition, reuse-distance,
cache-oblivious layout, hypergraph ordering, and multi-kernel property-access
objectives. A candidate advances only after:

1. its objective is written mathematically;
2. reductions to Gorder, graph/hypergraph partitioning, minimum linear/log
   arrangement, FrontOrder, Rebo, and Corder are explicitly attempted;
3. a concrete counterexample demonstrates non-equivalence to each closest
   objective;
4. the novelty-bearing mechanism is algorithmic, not only deterministic
   sampling, work caps, or engineering integration;
5. a domain expert reviews the full closest papers and source.

The second objective-family pass found the single-kernel locality space
saturated by Gorder, MinLA/MLogA, bandwidth/profile, graph/hypergraph
cut/connectivity, community, degree/hub, and direct reuse-distance objectives.
A minimax layout across kernel-specific access graphs does not collapse to a
weighted single graph, but it overlaps robust/multi-objective hypergraph and
multilayer partitioning (for example,
[Deveci et al.](https://research.sabanciuniv.edu/27693/1/paper.pdf)) and is not
yet a defensible headline contribution.

The next contribution gate must choose one of these routes:

1. a genuinely new deterministic near-linear optimizer or approximation
   guarantee for a known hard layout objective;
2. a mathematically new joint locality/convergence objective whose reductions
   and closest multi-objective prior art survive review;
3. an explicit pivot to a systems/characterization/adaptive contribution,
   acknowledging that it no longer satisfies the current independent-ordering
   headline.

Route 1 or 2 requires a theorem-level or algorithm-mechanism contribution, not
only a faster implementation. Route 3 requires a deliberate scope decision;
the selector cannot silently substitute for the rejected ordering objective.

Baseline hygiene findings discovered by the same review:

- Algorithm 10 is a historical 1K-vertex Corder-style approximation, not the
  canonical 1 MiB/L2-sized Corder configuration;
- the GoGraph header cites TPDS 2024, while the verified paper is ICDE 2024;
- DON-Lite cites ICDE 2024, while the identifiable DON-RL paper is WISE 2021,
  and the fixed-weight heuristic is not a faithful DON-RL implementation.
- Algorithm 16 omits the published GoGraph Rabbit-clustering/supernode stage;
  on symmetric graphs its M objective is constant, so it is a core diagnostic
  rather than a faithful standalone baseline.
- Bare Algorithm 11 composes two MIND-start RCM passes. It is frozen historical
  evidence, not a canonical single-pass baseline; `11:mind` and `11:bnf`
  provide explicit future controls.

The source identities are now corrected, Algorithm 10 is deterministic, and
`10:canonical` exposes the upstream-sized baseline without changing frozen
bare-`10` evidence. A faithful GoGraph baseline remains blocked on the
published Rabbit-cluster stage and directed/asymmetric evaluation. Rebo and
FrontOrder still require a final domain-expert review before any novelty
claim. This is technical prior-art analysis, not legal advice.

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
Both the generic sketch/block proposal and `J_line` are rejected by the
novelty gate. Phase 2 is a research-definition phase before it is an
implementation phase:

1. survey cache-line transition, reuse-distance, cache-oblivious graph layout,
   hypergraph ordering, and multi-kernel property-access objectives;
2. write each candidate objective and optimizer as a minimal mathematical
   specification;
3. attempt explicit reductions to the closest prior objectives and construct
   non-equivalence counterexamples;
4. reject candidates whose novelty is only sampling, determinism, bounded
   work, or a composition of known stages;
5. freeze one independently reviewed objective/optimizer pair before adding a
   C++ algorithm ID or experiment cell;
6. then enforce stable source-ID tie breaks, hard work/memory/pass caps, and
   one-base-plus-one-derived-arm residency.

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
