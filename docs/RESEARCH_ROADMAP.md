# GraphBrew Research Roadmap

## Objective

Design, implement, and publish a GraphBrew-native ordering that is
algorithmically independent of RabbitOrder, Gorder, and Leiden. It must beat
RabbitOrder and Gorder on controlled kernel performance while keeping complete
reorder overhead at or below RabbitOrder's level. Exact preprocessing,
mapping, kernel, work, cache, memory, and amortization accounting remain
mandatory. Adaptive selection is a downstream extension only after the new
ordering establishes a genuine Pareto point.

The normal experiment entry point is `scripts/graphbrew_experiment.py`.
Reusable experiment, parsing, storage, and analysis logic belongs under
`scripts/lib/`. Paper-replication paths remain isolated consumers of those
shared contracts.

## Full Sprint Execution Contract

The active sprint runs end to end without skipping gates:

1. **Novelty landscape closure:** retain the completed primary-source survey
   and rejected mechanisms as hard constraints.
2. **Independent mechanism search:** develop a new position-sensitive
   objective or a genuinely new near-linear optimizer for a known objective.
3. **Specification freeze:** write the objective, decision rule, complexity,
   memory, determinism, option semantics, and non-equivalence evidence.
4. **Native implementation and correctness:** integrate through existing
   registries and the orchestrator; close mapping, weighted, source, timing,
   work, memory, and provenance contracts.
5. **Rapid Pareto pilot:** run the new candidate on twitter7, webbase-2001,
   and hollywood-2009 against the frozen audited baselines.
6. **Measured optimization:** optimize only demonstrated bottlenecks without
   changing the candidate's objective or mapping semantics.
7. **Independent-ordering Pareto gate:** require statistically supported
   kernel gains over RabbitOrder and Gorder with Rabbit-level reorder cost.
8. **Adaptive extension:** only after Gate 7, test whether graph/workload
   features can select among the new ordering and cheap static controls.
9. **Final evaluation:** verification manifest first, then the frozen full
   matrix, figures, tables, and reproducibility package.

At every stage boundary, run layered `rubber-duck` review with
`claude-opus-5` followed by `gpt-5.6-sol`. Resolve every blocking finding
before advancing and update this roadmap when evidence changes the plan.

## Historical Evidence

The completed pre-pivot corpus established the following controlled-work
frontier:

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

These values were produced by the frozen campaign rooted at `a5193e14` and
packaged through `767703e1`, before the current baseline-semantics corrections.
They are historical motivation only and are claim-ineligible for the new
ordering campaign. No
row, shuffled graph, mapping, or timing from that campaign may be mixed into
fresh measurements.

## Active Novelty Boundary

RabbitOrder, Gorder, and Leiden remain external baselines, diagnostic anchors,
and controlled ablations. They cannot provide communities, hierarchy, scoring,
or placement decisions to the headline ordering.

The GraphBrew-native ordering must not use:

- Rabbit community assignments, aggregation, or dendrogram traversal;
- Gorder's window objective or a restricted approximation of its scoring loop;
- Leiden communities as the primary partitioning decision;
- an adaptive selector over existing algorithms as evidence of a new ordering
  contribution.

Generic graph primitives such as CSR scans, degree statistics, sampling,
prefix sums, stable compaction, and BFS are allowed, but the algorithmic
objective and placement decisions must be independently specified.

Before independent-ordering implementation, freeze a novelty matrix
comparing the proposed method
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

**No independent-ordering objective is currently frozen. Its implementation
remains blocked.**
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

The contribution gate considered these routes:

1. a genuinely new deterministic near-linear optimizer or approximation
   guarantee for a known hard layout objective;
2. a mathematically new joint locality/convergence objective whose reductions
   and closest multi-objective prior art survive review;
3. an explicit pivot to a systems/characterization/adaptive contribution,
   acknowledging that it no longer satisfies the current independent-ordering
   headline.

Route 1 is active again after the user reaffirmed the headline objective.
Route 2 remains admissible if its objective survives reduction and prior-art
review. Route 3 is fallback infrastructure only and requires a separate,
explicit scope decision; it is not the current research contribution.

### Closed Route-1 Hypothesis: Reservation-Nesting Layout

The optimizer survey produced one theory-first hypothesis over the known
minimum-logarithmic-arrangement objective:

```text
J_log(sigma) = sum_(u,v in E) w_uv * log2(max(1, |sigma(u)-sigma(v)|))

LB_deg(G) =
  1/2 * sum_v sum_(i=1..degree(v)) log2(max(1, ceil(i/2)))
```

`LB_deg` is a degree-only lower bound because the nearest distinct positions
around any vertex occur at distances `1,1,2,2,...`; the factor one-half
corrects double-counted undirected edges.

The proposed optimizer, **Reservation-Nesting Layout (RNL)**, would have:

1. count degrees and `LB_deg`;
2. process degree buckets from high to low with source-ID ties;
3. reserve bounded laminar intervals for high-demand anchors using prefix sums;
4. assign remaining vertices to the smallest-capacity feasible host interval
   selected by exact adjacency gain, with deterministic conflict resolution;
5. place overflow vertices by DBG order and report their explicit objective
   charge;
6. emit exact `J_log`, `LB_deg`, and the a-posteriori ratio
   `rho_hat = J_log / LB_deg`.

Layered Opus/Sol theory review rejected RNL before implementation:

- `LB_deg` requires an explicitly simple, unit-weight, loop-free topology
  policy and is zero on useful degree-two cases. `J_log / LB_deg` is therefore
  only a generic diagnostic for existing mappings, not a novelty-bearing
  certificate; report `NA` when the bound is zero.
- The proposed upper bound omits cross-interval edges. Charging their
  least-common interval repairs the expression only by moving toward standard
  hierarchical-layout analysis.
- The proposed `(b, Delta)` packable class is circular. Singleton intervals
  make every maximum-degree graph `(1, Delta)`-packable, including expander
  families for which the claimed constant-factor conclusion is false.
- Disjoint reservations are infeasible even on a matching when requested
  capacity sums above `n`; nested reservations cannot be assigned by an
  independent prefix sum because child capacity must come from its parent.
- Degree anchors without topology collapse to DBG. Adding adjacency gain or
  frontier expansion makes host assignment a capacitated LDG/Fennel or
  BFS/RCM-style region-growing decision. Laminar placement additionally
  overlaps established recursive/hierarchical MLogA methods.
- The claimed three-pass implementation cannot also emit exact `J_log` unless
  it buffers edge state or supplies a separately proved endpoint-charging
  scheme.
- Closest MLogA work already includes star lower bounds, ex-post `UB/LB`
  reporting, recursive contiguous sublayouts, and linear-complexity
  hierarchical optimization. The proposed mechanism does not leave an
  independently defensible decision rule.

RNL, its circular graph-class guarantee, and the artificial irrevocable
placement fallback are closed. Implementing any of them would violate the
non-collapse gate. This rejection applies to RNL, not to the project goal. The
active task is to find a genuinely new mechanism and pass a new layered theory
review before implementation.

### Secondary Characterization and Adaptive Infrastructure

The reviewed characterization infrastructure is retained to:

1. a reproducible characterization of reorder cost, apply cost, peak RSS,
   mapping identity, cache/working-set behavior, convergence work, kernel time,
   end-to-end time, and reuse-count amortization for audited orderings;
2. provide fair, attributed baselines for the new ordering;
3. support a later adaptive extension after the independent-ordering gate.

It must not trigger a broad corpus run, selector training, or a replacement
headline before the new ordering passes the Phase-4 Pareto gate.

#### Protocol SSOT and Supersession

This roadmap is the decision and stage-order SSOT.
`research/ADAPTIVE_SELECTOR_SPRINT.md` is the deferred measurement/model
protocol for the downstream adaptive extension.
Shared benchmark, variant, naming, and cache registries remain under
`scripts/lib/`.

The 14-ordering characterization set is a baseline matrix, not the headline
portfolio. `adaptive_portfolio.def` remains a claim-ineligible Sprint-0
compatibility contract. No old model or authorization artifact may be promoted
by relabeling it.

Reviewer-facing output must print the exact `-o` specification and an
attributed display name. In particular:

- `12:rabbit:compose:sg_none:comm_identity:intra_hubsort` is
  **Rabbit communities + HubSort (GraphBrew implementation)**;
- `12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort` is
  **Rabbit communities + SuperRabbit block layout + HubSort (GraphBrew
  implementation)**.

Neither is named or described as a GraphBrew-native ordering.

#### Fresh-Campaign Artifact Eligibility

All randomized campaign graphs are regenerated transactionally under the
current converter and `graphbrew-reorder/v2` semantics in the external graph
root. Every row must bind:

- canonical source identity and graph-content fingerprint;
- requested and resolved labeling specification, including seed;
- converter and benchmark binary SHA-256 identities;
- repository revision and reorder semantics version;
- exact requested/resolved ordering specification;
- mapping permutation fingerprint and generation policy;
- source, process-block, thread, CPU, and measurement-mode identity.

Pre-correction shuffled graphs, mappings, observations, model artifacts, and
authorization files fail closed. After candidate theory/correctness approval,
only the three pilot graphs may be regenerated for Phase 3. Full 11-graph
corpus refresh, broad mapping generation, adaptive timing, and final timing
remain blocked until the Phase-4 Pareto gate passes.

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
- Gorder `9:gograph` and `9:csr` are exact mapping-equivalent implementations;
  `9:fast` is a distinct relaxed mapping. Its batch/window semantics and
  frontier merge are now deterministic across thread counts.
- Every reorder pass now emits and persists an exact permutation fingerprint;
  standalone Rabbit variants are explicitly marked schedule-sensitive.
- Algorithm 15 is GVE-Leiden community detection plus a GraphBrew post-layout,
  not a native Leiden ordering. Historical `hierarchy-degree` and minimal
  final-community layouts are now explicit, strictly parsed controls.
- RANDOM seed 0 is now a specified thread-independent permutation. HubSort and
  HubCluster restore upstream non-hub ID preservation, while IDs 6/7 remain
  the distinct compact DBG variants. Degree thresholds use stored adjacency
  degree rather than half the symmetric edge count.

The source identities are now corrected, Algorithm 10 is deterministic, and
`10:canonical` exposes the upstream-sized baseline without changing frozen
bare-`10` evidence. A faithful GoGraph baseline remains blocked on the
published Rabbit-cluster stage and directed/asymmetric evaluation. Rebo and
FrontOrder still require a final domain-expert review before any novelty
claim. This is technical prior-art analysis, not legal advice.

## Phase 0: Data-Integrity and SSOT Cleanup (Completed)

Completed items:

1. [x] Use one Python module identity (`scripts.*`) and one project-root import
   convention.
2. [x] Make result identity condition-aware (`labeling`, `measurement_mode`,
   thread/process policy) and remove min-wins deduplication across conditions.
3. [x] Centralize benchmark lists, ordered evaluation subsets, cache capacities,
   cache iterations, and GraphBrew realized-config validation under
   `scripts/lib/`.
4. [x] Fail closed on retired/non-nested evaluation paths.
5. [x] Make the top-level harness the documented generic front door and keep
   replication-specific runners as isolated consumers.
6. [x] Add one unified build/test entry point and current CI coverage.

Cleanup must preserve existing measurement artifacts and numerical claims.
Mapping-cache unification is deferred until after the next campaign because
the existing artifact tree is frozen evidence.

## Phase 1: Independent Ordering Design II

The next candidate must contribute a new decision mechanism, not a renamed
combination of known stages. Before code is added, freeze:

- the exact objective and why it predicts property/adjacency locality;
- a deterministic optimizer with expected `O(m)` or `O(m log n)` work,
  practical memory bounds, and no hidden all-pairs/wedge explosion;
- a proof, guarantee, or independently checkable invariant that carries the
  algorithmic insight;
- explicit non-equivalence examples against RabbitOrder, Gorder, Leiden,
  MinLA/MLogA solvers, graph/hypergraph partitioning, RCM/profile methods,
  DBG/Corder, FrontOrder, Rebo, HashOrder, BOBA, and streaming LDG/Fennel;
- a reason the mechanism should beat Rabbit locality without exceeding
  Rabbit's complete preprocessing cost.

Promising directions may optimize a known objective only when the optimizer
itself is genuinely new. Time caps, sampling, determinism, or engineering
integration alone are insufficient novelty.

The first reopened three-thread search produced no survivor:

- anytime/budgeted refinement reduces to FM/KL, multilevel refinement, or
  ordinary incumbent tracking; a work cap is not a new optimizer;
- exact CSR line transitions reduce to hypergraph connectivity, while
  convergence terms become GoGraph-like, scheduler-specific, or cache-blind;
- deadline, swap, and practical MLogA sketches lack both a global guarantee and
  a credible Rabbit-cost path.

These are rejections, not a scope pivot. The next bounded theory pass examines:

1. whether a low-stretch tree or low-diameter metric decomposition can transfer
   a provable tree-layout guarantee to general-graph MLogA without collapsing
   to recursive partitioning/HST prior art;
2. whether a deterministic near-linear approximation is possible for a
   practically relevant graph class such as bounded arboricity, bounded
   expansion, or low doubling dimension, with an explicit out-of-class
   behavior;
3. whether parallel conflict-independent batch decisions can provide a
   nontrivial approximation for a position-sensitive objective rather than
   merely stale-score Gorder or FM refinement.

Each route must close low-stretch-tree, FRT/HST, separator-layout,
parameterized-layout, and parallel greedy prior art before becoming a
candidate.

The second bounded theory pass also produced no survivor:

- low-stretch trees provide only an upper transfer; no reverse inequality
  relates the weighted tree-layout optimum to general-graph MLogA, and concrete
  implementations become HST/recursive partitioners;
- bounded arboricity/expansion is too weak, low doubling dimension and
  separator guarantees cover only narrow geometric cases, and power-law or
  copied-neighborhood assumptions collapse to hub or shingle methods;
- conflict-independent parallel batches can be arbitrarily bad for MinLA and
  profile, while exact conflict graphs serialize on hubs.

Further paper-only objective invention is paused until a **mechanism discovery
study** identifies a concrete structural failure shared by RabbitOrder and
Gorder. Use the rapid path and controlled synthetic families to compare exact
mappings and kernel behavior for:

- community bridges versus dense interiors;
- hub-and-spoke, biclique, copied-neighborhood, chain, grid, and expander
  structures;
- property-line occupancy, row span/gap distribution, working-set fit,
  convergence work, and time per iteration;
- Rabbit/Gorder/DBG decisions at the vertices and regions where one baseline
  wins and the others fail.

This study is for hypothesis generation only. It must run through the existing
harness, add reusable analysis under `scripts/lib/` if necessary, and produce
no broad corpus or paper claim. A new candidate must explain an observed
failure with a decision rule that is absent from the rejected prior-art
families.

The study is pre-registered as follows:

- implement deterministic generators in
  `scripts/lib/pipeline/synthetic_graphs.py` and reusable analysis in
  `scripts/lib/analysis/mechanism_discovery.py`; do not add a standalone
  runner;
- store generated graphs under
  `/media/Data/00_GraphDatasets/GraphBrew/synthetic/` and structured results
  under
  `/media/Data/00_GraphDatasets/GraphBrew/artifacts/mechanism_discovery/`;
- identify every graph by family, parameters, seed, content fingerprint,
  converter identity, and generation-policy version;
- use `measurement_mode=diagnostic-synthetic`; these rows are permanently
  claim-ineligible and cannot enter adaptive or final matrices;
- use at most 48 family configurations and 24 dedicated node-hours;
- use 42 small mapping-screen configurations
  (7 families x 2 sizes x 3 label/topology seeds) and reserve the remaining
  6 configurations for exactly one promoted family at two sizes spanning below
  and above the property-working-set/LLC fit boundary;
- execute three Rabbit CSR mapping draws per screen graph and require any
  promotion signal to exceed their observed metric spread;
- provide an analytic layout for families with a known optimum structure, or
  label the reference explicitly as a heuristic or control. A baseline that
  beats a proven/heuristic reference is a reference defect, not a no-headroom
  result. A control reference records baseline advantage separately and can
  never qualify;
- use positive-bit MLogA per edge as the primary mapping-screen statistic. The
  reference must improve by at least 5% over median Rabbit CSR and exact Gorder
  at both screen sizes and all seeds before one family may advance to the
  reserved WSR-scale configurations;
- call a case a shared baseline failure only when the reference is at least 5%
  faster in controlled kernel time than both Rabbit CSR and exact Gorder across
  both sizes, with a seed/bootstrap interval excluding parity. A structural
  metric gap alone is insufficient;
- require a cheap sampled statistic detecting the implicated structure in at
  least three frozen corpus graphs spanning at least two topology classes
  before it may motivate a general candidate;
- emit a per-family decision-divergence table keyed by exact mapping
  fingerprints, showing the vertices/regions where Rabbit, Gorder, DBG, and the
  reference make different placements.

Freeze the small-instance screen with:

```bash
python3 scripts/graphbrew_experiment.py \
  --mechanism-discovery-plan
```

After implementation review, execute only that frozen plan with:

```bash
python3 scripts/graphbrew_experiment.py \
  --mechanism-discovery-screen
```

Commit the reviewed generator/analysis implementation before freezing the
plan; its repository-state binding deliberately invalidates pre-commit plans.

If the cap expires with no qualifying, prevalent failure, terminate the study
with a written negative result and ask the user for an explicit scope decision.
Do not automatically launch another theory pass, corpus run, or adaptive study.

**Gate:** Opus then Sol theory review. Do not assign an algorithm ID or launch
timing before both approve.

## Phase 2: Native Implementation and Correctness

Implement only the frozen candidate through the existing C++ dispatch,
algorithm-name SSOT, mapping sidecars, and top-level orchestrator. Require:

- exact permutation validity and weighted relabeling;
- deterministic mappings across repeated runs and declared thread policies;
- complete core, validation, apply, total, work, and peak-RSS accounting;
- requested/resolved option identity and exact mapping fingerprints;
- no silent fallback to RabbitOrder, Gorder, Leiden, DBG, or ORIGINAL.

## Phase 3: Three-Graph Pareto Pilot

Use the rapid path on twitter7, webbase-2001, and hollywood-2009 before any
full corpus run. Measure load, feature, mapping, validation, apply, kernel,
work, iteration, and peak-RSS phases separately.

Required controls:

- ORIGINAL and DBG cost floors;
- RabbitOrder `8:csr` and `8:boost` as primary performance/cost baselines;
- exact Gorder `9:csr` as the expensive locality reference;
- canonical Corder `10:canonical`;
- explicit single-pass RCM controls `11:mind` and `11:bnf`;
- Leiden `15:1.0:10:10:hierarchy-degree`,
  `15:1.0:10:10:final-stable`, and
  `15:1.0:10:10:final-degree`;
- Rabbit-derived composites only as clearly labeled diagnostics;
- the newly reviewed GraphBrew-native candidate.

The pilot is a characterization gate, not a winner-announcement run. It must:

- use exact requested/resolved specs and mapping fingerprints;
- separate iteration count, total kernel time, and time per iteration;
- report cache-to-working-set ratio rather than raw capacity alone;
- compare paired candidate/Rabbit and candidate/Gorder kernel performance;
- retain failures, timeouts, retries, and memory-limit outcomes;
- remain ineligible for final paper claims.

Before authorization, emit a high/low projected-hours table for every
`(graph, kernel, ordering, mode)` and freeze command-level caps. The 168
dedicated-node-hour target remains binding unless a reviewed pre-data budget
amendment is recorded. "Diagnostic" limits novelty claims, not measurement
quality: any arm considered for later deployment must use the same timing,
source, and provenance protocol.

Initial candidate gates use fresh baselines from the same pilot:

- paired candidate/Rabbit-CSR kernel ratio above `1.03x` with the graph
  bootstrap 95% interval excluding one;
- paired candidate/Gorder kernel ratio above one with the interval excluding
  one;
- all-kernel candidate/Rabbit-CSR ratio above one with the interval excluding
  one;
- complete reorder GM no greater than Rabbit CSR's fresh measured GM and an
  initial target at or below `10 s`;
- median reorder time at most `8 s`;
- no graph more than 5% slower than Rabbit CSR on controlled work.

The historical `1.80x` controlled and `1.73x` all-kernel values are provisional
planning references only. Re-derive any absolute reporting target from the
fresh audited baselines before authorizing the pilot.

## Phase 4: Measured Optimization and Pareto Gate

Optimize only bottlenecks observed in Phase 3 while preserving the frozen
objective and mapping semantics. If the candidate misses either locality or
cost targets, diagnose, iterate through a new reviewed specification, or
reject it. Do not compensate by switching to adaptive selection.

Only an independent candidate that meets the cost gate and shows supported
kernel improvement advances to the full corpus.

## Phase 5: Adaptive Selection

Only after the independent-ordering Pareto gate:

1. collect fresh source-driven and reuse-aware measurements;
2. train only on runtime-emitted Tier-0 features;
3. use nested leave-one-topology-out evaluation with fold-local portfolio
   pruning and OOD calibration;
4. optimize end-to-end regret, not classification accuracy;
5. deploy only a deterministic compact artifact with exact Python/C++ parity;
6. charge feature, model, mapping, validation, and apply costs to selection.

The selector must never consume graph identity or exact-name oracle data.
Report a no-abstention selector, an abstaining selector, the best static
GraphBrew-native ordering, Rabbit CSR, Gorder, and the cross-fitted portfolio
oracle. The new ordering must be a deployable arm.

Phase-5 entry requires the existing fail-closed graph-identity and
database/kNN guards plus executable regression tests. A non-empty graph name,
database/kNN model, or non-Tier-0 deployable artifact must abort rather than
fall back.

## Phase 6: Full Evaluation

Only after the independent candidate, pilot, precision, memory, and optional
adaptive gates:

- run the frozen graph/kernel/source/reuse matrix through the harness;
- retain rapid and final execution modes as explicit policies;
- regenerate every result, table, PNG, SVG, and report from structured
  manifests;
- preserve a separate, reproducible paper-number replication command without
  duplicating generic registries or parsers.
