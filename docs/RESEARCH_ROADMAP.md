# GraphBrew Research Roadmap

## Objective

The original objective was to design and publish a GraphBrew-native ordering
that was algorithmically independent of RabbitOrder, Gorder, and Leiden while
beating their locality at Rabbit-level complete reorder cost. The bounded
theory, synthetic, and real-mapping searches below closed without a
novelty-approved mechanism.

The active objective is therefore a cost-aware adaptive GraphBrew system that
selects among fully attributed, audited orderings using only lightweight
topology and workload features. It must beat the best static deployable policy
on cross-fitted end-to-end regret while charging feature extraction,
selection, mapping, validation, application, and kernel cost. It does not
claim a new vertex-ordering algorithm. Exact preprocessing, mapping, kernel,
work, cache, memory, and amortization accounting remain mandatory.

The normal experiment entry point is `scripts/graphbrew_experiment.py`.
Reusable experiment, parsing, storage, and analysis logic belongs under
`scripts/lib/`. Paper-replication paths remain isolated consumers of those
shared contracts.

## Research Execution Gates

The research program advances without skipping gates:

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

The 2026-08-14 scope reset supersedes the original dependency between Gates 7
and 8. Gates 2--7 closed with no independent candidate; the active program
resumes at a revised adaptive Gate 8 and makes no new-ordering claim.

At every stage boundary, obtain two independent technical reviews. Resolve
every blocking finding before advancing and update this roadmap when evidence
changes the plan.

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

Independent theory reviews rejected RNL before implementation:

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

This roadmap is the decision and stage-order SSOT. Detailed adaptive study
notes remain private until the independent-ordering gate passes.
Shared benchmark, variant, naming, and cache registries remain under
`scripts/lib/`.

The 14-ordering characterization set is a baseline matrix, not the headline
portfolio. `adaptive_portfolio.def` remains a claim-ineligible legacy
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

### Mechanism-Discovery Screen Result

The frozen screen at plan
`d54e75e665b78119418110a81484614115d5d2d05f41f4e60928dd658005a0da`
completed all 210 commands over 42 configurations under committed policy
`c65a2ae8`. It consumed 0.00277 measured hours; all rows remain
`diagnostic-synthetic` and claim-ineligible.

The primary positive-bit MLogA screen nominated two families:

- **grid:** the Morton reference was 20--23% below Rabbit CSR and exact Gorder;
- **hub-spoke:** the centered-hub reference was 15--18% below Rabbit and
  54--58% below Gorder.

Independent mechanism reviews rejected both:

- grid uses hidden generator coordinates destroyed by the input-label shuffle,
  and its Morton/SFC or coordinate-free recursive equivalent is occupied
  cache-oblivious mesh/recursive-partitioning prior art. Only the road and mesh
  corpus graphs plausibly exhibit the structure, below the three-graph
  prevalence requirement;
- Rabbit already packs hub-spoke groups nearly perfectly. Centering the hub
  changes MLogA bits but leaves same-line and row-line demand essentially
  unchanged; the large structural packing failure is Gorder-only. Median
  neighbor placement is also established MinLA/local-search behavior.

Block-biclique missed the Rabbit threshold, chain and community-bridge were
matched by exact Gorder, copied-neighborhood exposed a defective reference, and
the expander control behaved as a negative control.

**Negative result:** no recoverable, novelty-safe, shared Rabbit/Gorder
cache-locality failure qualified. The six reserved WSR-scale configurations
remain unused. This closes the synthetic-family route only.

### Route F: Frozen Real-Graph Mapping Forensics

The user explicitly authorizes one final input-read-only, timing-free
real-graph forensic pass. It remains `diagnostic-forensic`, claim-ineligible,
and cannot establish proxy validity or runtime improvement.

Discovery graphs are `cit-Patents`, `soc-pokec`, `USA-road-d.USA`,
`soc-LiveJournal1`, `delaunay_n24`, `com-Orkut`, `wikipedia_link_en`, and
`Gong-gplus`. `hollywood-2009`, `webbase-2001`, and `twitter7` are a locked
confirmation cohort. Confirmation adjacency and mapping contents remain
unopened until one class, detector, and candidate-mechanism specification are
frozen and hashed.

Use only each graph's exact `<graph>/<graph>.sg`, `5.lo`,
`8_csr.draw{0,1,2}.lo`, and `9_csr.lo` artifacts. `8_csr.lo` is the selected
draw-0 alias and is not a fourth Rabbit observation. Derive
**INPUT-SHUFFLED** from SG internal order. Source-ID order is a separately
labelled diagnostic, not a natural or neutral baseline. Never select
`hollywood-2009.natural.sg`.

Campaign `.lo` files use original/source-ID coordinates:

```text
file[new_id] = source_id
new_id[sg_id] = source_to_new[org_ids[sg_id]]
```

Any Python analysis that omits this composition is invalid. Implement a
separate numeric path in `scripts/lib/analysis/mapping_forensics.py`; do not
reuse the synthetic text-graph analyzer.

The artifact gates require:

1. exact 17-byte SG header, file-size, dimension, and layout validation;
2. valid `.lo` and `org_ids` permutations;
3. correct source-space composition;
4. sidecar algorithm/draw identity and Rabbit draw-0 alias equality;
5. Gorder byte-equivalence evidence for promoted legacy mappings where
   available;
6. SHA-256 and permutation fingerprints before and after every pass;
7. exact DBG bucket-order semantic validation;
8. an analytic random-permutation line null reported as diagnostic only.

Historical `reorder_meta/v4` inputs are labelled `legacy-forensic` and are
never promoted to fresh-campaign evidence. Ten exact Gorder mappings are
promoted legacy outputs with later byte-equivalence evidence; twitter7 is a
direct generation. Record the remaining unquantified semantics-drift risk.
No campaign SG, mapping, or mapping-sidecar path may be written. A SHA-256
tripwire over all admitted inputs is frozen before reading their contents.

Let:

```text
b_sigma(e) = 1 + floor(log2(max(1, |sigma(u) - sigma(v)|)))
```

Report exactly:

- **M1:** exact mean positive-bit MLogA per undirected edge;
- **M2:** exact excess-bit mass above gaps `8`, `64`, `4096`, and `262144`,
  plus exact 8-byte-property same-line fraction;
- **M3:** sampled distinct property lines per degree, diagnostic-only and
  pre-rejected as a candidate objective because it is hypergraph
  connectivity;
- **M4:** source- and reflection-safe class-conditioned Rabbit/Gorder bit-bin
  disagreement;
- **M5:** min/median/max over exactly three Rabbit mapping draws;
- **M6:** the one-sided optimistic class-headroom bound obtained by clipping
  class-edge charges at gap 64. M6 is a falsifier, not a relocation
  certificate or optimizer.

M1, M2, M4, M5, and M6 use exact scans with no iid edge bootstrap. M3 samples
at most 65,536 degree-stratified vertices with a frozen hash seed and a
256-bucket cluster bootstrap. Across graphs report fixed-corpus counts and
topology coverage, not population confidence intervals.

Before discovery, freeze at most 64 source-label-free, topology-only class
predicates using degree, neighbor-degree quantiles, bounded-sample local
clustering, and a core-number proxy, plus exactly one nomination score.
Nominate at most one class `C`. Thresholds cannot change after confirmation is
unsealed.

The gates execute in order:

- **H0-Sanity:** on at least seven of eight discovery graphs,
  `min(M1(Rabbit median), M1(Gorder)) < M1(INPUT-SHUFFLED)`. Passing has no
  positive runtime meaning; failure stops Route F.
- **H1:** one frozen class `C` occurs on at least three discovery graphs
  spanning at least two topology types with at least 0.1% incident-edge
  support.
- **H2:** both Rabbit and Gorder carry line-scale/excess-bit charge on `C`,
  and their M4 disagreement exceeds the maximum Rabbit/Rabbit draw
  disagreement by at least 0.05.
- **H3:** M6 is at least 0.05 for both median Rabbit and Gorder on the same
  three or more graphs after accounting for Rabbit draw spread.
- **H4:** `C` is detected deterministically in `O(m)` work and `O(n)` memory
  without source IDs, timings, communities, or baseline mappings.

Passing H0--H4 only nominates one signature. Before an algorithm ID or timing:

- **N0:** write the signature as a decision-rule predicate, not a metric;
- **N1:** freeze its mathematical objective/rule, invariant, deterministic
  semantics, label-equivariance, and Rabbit-cost work/memory argument;
- **N2:** attempt explicit reductions against median/insertion/swap,
  FM/KL and multilevel refinement, Gorder windows, MinLA/MLogA local search,
  DBG/HubSort/Corder, RCM/profile, graph/hypergraph partitioning, FrontOrder,
  Rebo, SFC/Morton, and recursive layouts;
- **N3:** provide a concrete non-equivalence graph for every closest family;
- **N4:** pass two independent reviews with at most one revision, then
  replicate H1--H4 unchanged on at least two of the three locked confirmation
  graphs.

Auto-reject any rule reducing to median/centroid placement of already placed
neighbors, insertion/swap/FM refinement, recursive contiguous bisection, or
distinct-lines-per-neighborhood minimization.

Implement reusable logic in `scripts/lib/analysis/mapping_forensics.py` and
expose `--mapping-forensics-plan`, `--mapping-forensics-discovery`, and
`--mapping-forensics-confirmation` only through
`scripts/graphbrew_experiment.py`.

Implementation requirements:

- mmap/chunked NumPy readers with `int32` position arrays;
- one SG edge pass processing all layouts;
- no `read_text().split()`, Python full adjacency lists, or per-edge Python
  loops on campaign graphs;
- a 56-GiB internal RSS ceiling, four-wall-hour cap including hashing, and an
  observed-throughput projection that aborts before exceeding the cap;
- no persisted composed-permutation caches; store only structured outputs
  under
  `/media/Data/00_GraphDatasets/GraphBrew/artifacts/mapping_forensics/<plan-hash>/`.

Historical timing and observation rows are excluded as analysis inputs.
Historical timing may appear only as a labelled
`falsifier-only, claim-ineligible` reason to discard a proxy that contradicts
known behavior; it cannot support H1--H4 or a candidate.

If any artifact, H, or N gate fails, append a
**Real-Graph Mapping Forensics Result** with the plan hash, failed gate,
per-graph evidence, and consumed caps, then stop. No new theory pass, mapping
generation, corpus refresh, timing, synthetic extension, or adaptive work is
authorized automatically.

**Gate:** two independent theory reviews. Do not assign an algorithm ID or
launch timing before both approve.

### Real-Graph Mapping Forensics Result

The frozen discovery plan
`8d9f1c6e1bc47a6957511d5e9342311264b3f09001e195e883d7e84308421b4a`
completed all eight discovery graphs. The threshold hash was
`f4389f126ff2ceefb0c07f54632db80e2e0851a7efc261c6108b74e6eface70b`,
the class-bank hash was
`0785aa04e6ae637c20062b52e55c8152c1088a250363ab0b50b826dc28f6a38e`,
and the discovery-decision hash was
`05fc68adcca9f6db07fed385c11e51715684bcb64efc3972cc6dfa009c94c0e2`.
The pass consumed 7,341 seconds of analysis time and 7,394 seconds elapsed,
with 8,346,853,376 bytes peak RSS. It generated no mappings and ran no kernel
or cache timing.

H0 passed on all eight graphs. Fifteen frozen classes met H1--H4, and the
frozen nomination score selected class 2, `degree:q2`, with score 2.07966.
The class is the graph-relative degree band `Q50 <= degree < Q75`; H1 held on
all eight graphs, while H2 and H3 held on the six skewed
citation/social/content graphs and failed on the road and mesh graphs:

| Graph | Type | H0 best/input M1 | `degree:q2` support | H2 | H3 |
|---|---|---:|---:|:---:|:---:|
| `cit-Patents` | citation | 0.551 | 42.00% | pass | pass |
| `soc-pokec` | social | 0.759 | 36.02% | pass | pass |
| `USA-road-d.USA` | road | 0.107 | 40.72% | fail | fail |
| `soc-LiveJournal1` | social | 0.609 | 25.15% | pass | pass |
| `delaunay_n24` | mesh | 0.181 | 50.23% | fail | fail |
| `com-Orkut` | social | 0.762 | 36.47% | pass | pass |
| `wikipedia_link_en` | content | 0.664 | 13.33% | pass | pass |
| `Gong-gplus` | social | 0.679 | 21.27% | pass | pass |

The mandatory Opus-then-Sol novelty review rejected the nominee at N1--N3:

- **N1 failed:** the predicate is deterministic and label-equivariant, but it
  does not define an invariant placement rule, objective, or Rabbit-cost
  algorithm. Its optimistic M6 headroom is primarily class support multiplied
  by graph-wide excess, not a localized shared defect. On five of the six
  qualifying graphs, Rabbit's class excess per edge was only 0.727--0.963 of
  its global excess; Gorder was 1.011--1.050. Only
  `wikipedia_link_en` showed material enrichment for both baselines.
- **N2 failed:** a contiguous placement of `degree:q2` is a percentile-cut
  degree segment, reducing to the occupied degree-bucketing space of DBG,
  HubSort/HubCluster, Corder, and hub-peeling layouts. Residual-aware variants
  reduce to weighted MLogA/MinLA, median/insertion refinement,
  distinct-neighborhood-line hypergraph objectives, or partitioning.
- **N3 failed:** without a distinct placement operator, no non-equivalence
  graph can separate the rule from its closest prior families. Changing
  average-relative degree boundaries to graph-relative quantiles is parameter
  selection, not a new mechanism.

No revision was authorized because the frozen evidence contains no
novelty-safe operator to revise. The confirmation cohort remains sealed and
unused, no candidate or algorithm ID is created, and Phases 2--4 remain
blocked. This is the final negative result for Route F and closes the currently
authorized independent-ordering search. No new theory pass, confirmation run,
mapping generation, corpus refresh, timing, synthetic extension, or adaptive
work follows without an explicit scope decision.

### Scope Reset and Automorphism-Safe Contract

The user authorized continuation on 2026-08-14. That authorization opens a
new adaptive/characterization scope; it does not reopen Route F or convert its
failed signature into a candidate.

The scope review corrected an impossible requirement in the earlier novelty
contract. A deterministic total permutation cannot be strictly
label-equivariant on a graph with a nontrivial automorphism. For a relabeling
`pi`, concrete mappings are now compared modulo automorphisms:

```text
sigma_(pi G) o pi = sigma_G o alpha,  alpha in Aut(G)
```

Topology-derived decisions must remain relabeling-equivariant. A concrete ID
tie-break is permitted only inside a certified automorphism class, and the
certificate must include stored edge multiplicity, direction, weights,
self-loops, and any source/color attributes used by the workload. Given a
proposed `alpha`, membership in `Aut(G)` is checked in `O(m)` by verifying the
stored edge multiset and attributes. Topology-isomorphism-invariant metrics
are then identical automatically; source-ID diagnostics remain outside this
equivalence.

The reopened novelty landscape produced no new ordering:

- doubly lexical/vicinal-preorder seriation is established matrix-ordering
  prior art, has unresolved non-unique ties, and its sparse refinement cost is
  unlikely to meet Rabbit's budget;
- exact twin contiguity reduces to depth-one modular decomposition, duplicate
  neighborhood compression, and an extremal case of Gorder's sibling score.
  It is also suboptimal on stars, where the center belongs inside the
  false-twin leaf interval;
- cache-set-aware vertex placement is cache-conscious data-placement prior
  art and is a no-op for dense contiguous property arrays unless padding or
  line repacking is introduced, reducing respectively to coloring/bin packing
  or the already occupied spatial-locality objectives;
- bounded-pass community aggregation plus cache-aware emission overlaps
  2PS-L, size-constrained clustering, RabbitOrder, Corder/Cagra, Rebo, and
  existing fast Leiden implementations. Engineering fusion alone does not
  satisfy the independent-ordering novelty gate.

This closes the independent-ordering headline under the current constraints.
Phases 2--4 below are retained as the historical candidate path but are not
active.

### Active Adaptive Pilot

The active contribution is an attributed cost-aware selector, not a relabeled
ordering. The existing Sprint-1 pilot remains `diagnostic-adaptive` and
claim-ineligible. Its frozen manifest contains 443 serial commands plus four
page-cache priming commands, projects 9.37 buffered node-hours at the pilot
high estimate, and has a 92.56-hour bound if every retry cap binds. The rapid
path contains only the five deployable selector arms; the 14-arm attributed
characterization matrix remains frozen for later evaluation and is not
repeated in every pilot kernel/process cell.

The three graphs formerly reserved as Route-F confirmation may be used only by
this separately frozen adaptive manifest. No Route-F confirmation analysis,
signature replication, or threshold reuse is permitted. Their release to the
new scope must be recorded by the content-bound execution authorization.

Pilot entry requires:

1. the frozen source, natural-label, Tier-0 weight, graph, binary, and command
   bindings to validate byte-for-byte;
2. the executor validation artifact to pass on the current host;
3. an explicit authorization bound to the complete execution-manifest hash;
4. serial exclusive execution with failures, retries, timeouts, censoring,
   wall caps, and peak RSS retained;
5. no graph identity, database/kNN lookup, runtime oracle, or non-Tier-0
   deployable feature.

After execution, analyze only through
`python3 scripts/graphbrew_experiment.py --adaptive-sprint1-analyze`.
The analyzer replays every authorized retry digest, validates priming and
completion counts, suppresses headroom under censoring or incomplete arm
coverage, and uses odd/even process-block oracle cross-fitting plus
leave-one-topology-out static-policy evaluation over reuse counts
`1, 5, 10, 20, 50, 100`, and the kernel-only limit.

After collection, promotion requires nested topology-held-out evaluation and:

- lower end-to-end regret than the best static deployable arm after charging
  feature, selection, reorder, validation, apply, and kernel costs;
- exact Python/C++ policy parity and deterministic tie semantics;
- explicit no-abstention and abstaining policies;
- comparison with ORIGINAL, DBG, Rabbit CSR, exact Gorder, canonical Corder,
  explicit RCM, and attributed Leiden/GraphBrew controls;
- no broad corpus or final claim if the three-graph pilot does not beat the
  best static deployable policy.

## Phase 2: Native Implementation and Correctness

Phase 2 remains blocked until Route F gates N0--N4 pass and the frozen
candidate completes layered novelty review.

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

This phase is active under the 2026-08-14 scope reset. The former
independent-ordering prerequisite is superseded; the portfolio contains only
fully attributed existing methods and the contribution is selection,
measurement, and cost-aware deployment.

The active work must:

1. collect fresh source-driven and reuse-aware measurements;
2. train only on runtime-emitted Tier-0 features;
3. use nested leave-one-topology-out evaluation with fold-local portfolio
   pruning and OOD calibration;
4. optimize end-to-end regret, not classification accuracy;
5. deploy only a deterministic compact artifact with exact Python/C++ parity;
6. charge feature, model, mapping, validation, and apply costs to selection.

The selector must never consume graph identity or exact-name oracle data.
Report a no-abstention selector, an abstaining selector, the best static
deployable ordering, Rabbit CSR, Gorder, and the cross-fitted portfolio oracle.
No arm may be described as a new GraphBrew-native ordering.

Phase-5 entry requires the existing fail-closed graph-identity and
database/kNN guards plus executable regression tests. A non-empty graph name,
database/kNN model, or non-Tier-0 deployable artifact must abort rather than
fall back.

## Phase 6: Full Evaluation

Only after the adaptive pilot, precision, memory, abstention, and end-to-end
regret gates:

- run the frozen graph/kernel/source/reuse matrix through the harness;
- retain rapid and final execution modes as explicit policies;
- regenerate every result, table, PNG, SVG, and report from structured
  manifests;
- preserve a separate, reproducible paper-number replication command without
  duplicating generic registries or parsers.
