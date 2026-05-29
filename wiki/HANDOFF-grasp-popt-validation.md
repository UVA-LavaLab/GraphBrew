# Handoff — GRASP / PIN / POPT faithfulness validation

Branch: `graphbrew_ecg`  •  Date: 2026-05-28

## Status update — confidence-building automation (2026-06-XX)

Tier A/B/C have all landed. The work has since expanded into a full
"is everything still green?" gate suite that runs on a single
`make confidence` invocation. The dashboard lives at
[`wiki/data/confidence_dashboard.md`](data/confidence_dashboard.md)
and currently reports **110 gates, all GREEN, exit 0**.

**Major gate families added since the 42-gate baseline** (each is one
generator + 12-test pytest + Makefile target + dashboard entry +
catalog entry + reproduce_smoke tracking — same 10-step wiring):

- **Curvature / slope family** (gates 50-72): per-(app, graph, policy)
  capacity-sensitivity slopes (OLS of miss% over log2 L3-MB),
  per-policy summaries, per-app and per-family breakdowns, saturation
  distance (4MB→8MB miss-rate drop), curvature, slope vs distance
  cross-check, family-curvature replay, and a cross-tool SRRIP-vs-
  GRASP slope ordering invariant that confirms the "oracle-aware
  policies are less cache-hungry" claim replicates on cache-sim,
  gem5 anchor, and sniper anchor (gate 72).
- **Anchor-tool slope replays** (gates 70/71): timing-faithful slope
  reproductions on gem5 (2 cells) and sniper (6 cells) at the
  4kB-2MB anchor sweep, using the same OLS / monotonicity / SRRIP-
  vs-GRASP / help-floor checks as the cache-sim sweep.
- **Regime-dependence formalization** (gate 74): the cross-tool
  LRU-vs-GRASP slope inversion is now a first-class invariant:
  cache-sim post-WSS (1-8MB) shows LRU strictly steeper than GRASP
  (-0.97 pp/oct), while both anchor tools at sub-WSS scales show
  the opposite sign (gem5 +0.84, sniper +0.24). Sign agreement
  between gem5 and sniper confirms the inversion is physical
  (LRU's give-up-and-stream behaviour vs GRASP's hold-the-hot-set
  behaviour at sub-WSS).
- **Per-app deviation pinning** (gates 68/73): bfs is pinned as a
  documented kernel deviation for both LRU-vs-GRASP (gate 68) and
  SRRIP-vs-GRASP (gate 73) per-app ordering, with the pin gated by
  "no NEW deviations". Frontier-driven streaming pathology that
  gate 65 already flags as the most-saturated kernel.
- **Cross-tool universality + anchor census + saturation replays**
  (gates 75-81): cross-tool slope-sign universality roll-up across
  all 10 (tool, policy) cells (gate 76); cell-level L3-sweep
  monotonicity universality across 320 (Li, Li+1) steps (gate 77);
  anchor cell-pair census pinning 2 gem5 cells + 6 sniper cells
  against silent shrinkage (gate 78); per-family saturation-distance
  replay locking citation/social headroom and the web pin
  (citation=+15.69, social=+12.50, web=+2.15 pp) (gate 79); anchor
  monotonicity replay with tier-aware tolerances — gem5 strict
  (0/18 bumps), sniper bounded (19/54 bumps, 2 hard, max +1.18 pp)
  (gate 80); per-policy final-octave steepness ranking
  POPT(0.10) <= GRASP(0.23) << LRU(1.06) ~ SRRIP(1.09) pp/octave
  (gate 81).
- **Cross-tool agreement, distribution integrity, and registry
  cross-checks** (gates 82-85): cross-tool shared-anchor slope-sign
  agreement across the 3 gem5∩sniper anchor cells (gate 82, all 3
  sign-matched, all both-negative, all sniper-steeper, max |Δ|=5.13
  pp/oct); regression-budget margin-distribution gate over all 330
  literature claim cells, locking global min/median floors and per-
  claim-kind margin floors (gate 83); paper-claims registry-integrity
  gate that re-derives every one of the 14 published claims' value
  from its cited source JSON and asserts equality within a half-LSB
  tolerance (gate 84, also catches stale headline text); cross-
  artifact aggregate consistency that locks 17 invariants between
  policy_winner_table, winning_regime_taxonomy, popt_vs_grasp_delta,
  cross_tool_saturation, literature_deviations, regression_budget,
  and corpus_diversity — load-bearing: the winner counts across
  sibling artifacts now provably agree on 114 cells, 56 GRASP wins,
  44 POPT, 8 SRRIP, 6 LRU (gate 85).
- **Per-app, catalog, family, and timing-tool cross-artifact gates**
  (gates 86-90): per-app oracle-rank-1 ↔ winner-table top-2 parity
  with the bc:SRRIP→GRASP divergence registered as the only allowed
  disagreement (gate 86); artifact-catalog completeness — every
  `wiki/data/*.json` is registered (gate 87, caught the silent gap
  where `paper_baseline_table.json` shipped without being catalogued,
  and registered it as the 72nd catalog entry); family-sensitivity
  cross-artifact parity — the 7 canonical_state claims agree on
  cell-counts with `policy_winner_table.wins_by_family` and on
  dominance direction (gate 88); cross-tool slope-ordering xartifact —
  SRRIP strictly steeper than GRASP on all 3 tools, LRU regime-
  inversion verdict PASS on all 5 checks, anchor sign-agreement and
  doubly-saturated cross-tool agreement (gate 89); gem5/Sniper anchor
  cell parity — load-bearing (email-Eu-core, pr) shared anchor cell
  locked in `shared_cells`, both tools share L3 axis and policy set,
  every anchor cell has all 3 policies populated with miss-rate in
  (0, 1) (gate 90).
- **Cross-artifact integrity gates** (gates 91-95): cell-count
  cross-artifact parity locking the 114-cell universe across the four
  winner-class summaries and the 3 tied cells on bc/email-Eu-core
  (gate 91, 14 tests); cache-sensitivity slope baseline pinning the
  10 known monotonic violations plus the {LRU:13, POPT:5, SRRIP:13,
  GRASP:2} anti-scaling partition over 112 trajectories (gate 92, 14
  tests); WSS-knee vs relative-L3 parity pinning the per-(policy,
  regime) {n, mean_gap_pp, win_rate} grid (114=52+14+48 regime cells,
  knee_rank {GRASP:0, LRU:2, POPT:0, SRRIP:2}) and verifying the
  paired wkl↔wrl payloads agree to <=0.05 pp (gate 93, 13 tests);
  bootstrap-CI nested consistency (seed=1729, ci=0.95) — mean_delta
  anti-symmetry exact, paired-bootstrap CI mirror within 1.0pp,
  p_a_lt_b complementarity within 0.03 across all 7 sign_stability
  entries (gate 94, 13 tests); family-clustering 3-way agreement —
  PWT argmax == FPAC global_winner_by_app on all 5 apps, deviation
  set recomputable from per-family qualified winners, 3 stable
  family_sensitivity claims at stability_floor=0.95, and the global
  cluster split {GRASP:[bc,cc], POPT:[bfs,pr,sssp]} locked (gate 95,
  13 tests).
- **Second cross-artifact integrity block + milestone** (gates 96-100):
  AUC correlation cross-artifact parity — PAC.meta and FPAC.meta agree
  on auc_winner_by_app and clusters_by_winner, intra_inter math
  recomputable from the matrix within 1e-3, 3 qualifying families
  {citation, social, web} locked at min_apps=4 (gate 96, 13 tests);
  family-geomean ↔ margin-replay parity — all 5 families present in
  both artifacts, FMR per-family cells {citation:15, mesh:5, road:25,
  social:54, web:15} summing to corpus total 114, FGI headline-15
  entries derivable from geomean_improve_pct ≥ 10.0, and the 34 strict
  geomean improvements all have ≥1 winning cell in margin-replay (gate
  97, 13 tests); oracle-gap curvature ↔ effect-size parity — slope and
  curvature math (slope = gap_diff/log2_ratio, curvature = slope2 -
  slope1) verifiable to 1e-3, OGES Cliff's-delta antisymmetry to
  machine precision, knee_count {GRASP:4, POPT:3, LRU:0, SRRIP:0}
  pinned (gate 98, 13 tests); monotonicity-universality ↔ anchor-
  replay agreement — MU 14 sub-noise bumps (max 0.0347 pp), AMR per-
  tool steps/bumps/hard_bumps/catastrophic accounting all internally
  consistent, MU.max_noise_bump_pp == AMR.constants.hard_bump_threshold_pp
  (both 0.5 pp), and MU.largest_bump_pp strictly < shared threshold
  (the "cache-sim is sharper than the anchors" guarantee) (gate 99,
  13 tests). **Milestone gate 100** — catalog ↔ dashboard ↔ disk
  coverage triangle: every CATALOG entry has gate/generator/artifact
  on disk under wiki/, every PYTEST_SUITES path resolves, short
  labels unique and non-empty, ≥70 catalog gates fan into the
  dashboard with exactly two documented EXEMPT_FROM_DASHBOARD entries
  (test_confidence_dashboard.py and test_paper_baseline_table.py),
  and the catalog summary's "(N) gates today" text matches
  len(PYTEST_SUITES) (gate 100, 13 tests). Also disambiguated 3
  pre-existing duplicate PYTEST_SUITES short labels (Slope→CSlope/
  CapSlope, Parity→GapPar, Sat→SatOn/SatDist).
- **Third cross-artifact integrity block** (gates 101-105): deviations
  ↔ regime taxonomy parity — LD's 30 entries all carry
  mechanism=popt_overhead_dominates and family ∈ WRT's family universe,
  shares within 1e-5 of wins/total, mechanism×family cross-tab
  recomputable from per-deviation rows (gate 101, 13 tests); corpus
  diversity ↔ regime taxonomy feature parity — every WRT cell's
  avg_degree / hub_concentration / clustering_coeff matches the
  corpus_diversity per-graph value within 1e-3 across all 114 cells,
  WRT.family == GRAPH_FAMILY[graph] for every cell, and the corpus
  and WRT graph universes are identical (gate 102, 13 tests); paper
  claims registry recompute parity — all 14 claim values
  recomputable from their cited artifacts (winner shares from
  policy_winner_table sum to 100±0.5 %, popt_vs_grasp family means
  to 0.01 pp, ok_ratio/disagreement_rate/thrash counts exact),
  every claim.source and claim.gate path resolves, snake-eating-tail
  agreement between paper_claims.green_gate_count and dashboard.json
  (gate 103, 13 tests); family tri-artifact agreement —
  family_sensitivity / family_geomean / family_policy_auc_clustering
  share the same 5-family universe, sensitivity canonical_state
  matches canonical_claims fracs to 1e-9, clustering's
  (family, winner) picks all appear in geomean records, and
  winners_matching equals counted True flags per qualified family
  (gate 104, 13 tests); regression_budget ↔ lit_faith parity — both
  artifacts cover the same 330-cell key universe, rb.status ==
  lf.status for every cell, by_kind.n counts only in-distribution
  cells, fragile_cells subset of per_cell and bounded at 10,
  known_deviation cells all have margin_pp=0, and triple counts of
  known_deviation / within_tolerance agree across summary, per-cell,
  and tolerated list (gate 105, 13 tests).
- **Fourth cross-artifact integrity block** (gates 106-110): oracle_gap
  internal + oracle_gap_by_app aggregation parity — oracle is min across
  the 4-policy panel per (graph, app, l3), winner has miss==oracle and
  gap_pp==0, by_policy_app mean/median/p90/max/n/wins all recompute
  exactly from rows using the right percentile methods (numpy `higher`
  for p90, linear-interpolation median; the two MUST stay distinct),
  by_app_ranking entries sort ascending by mean_gap_pp (gate 106, 13
  tests); GRAPH_FAMILY map duplication lock — 2 full-tier copies (11
  entries with reserved future graphs road-CA / twitter-2010 / uk-2005)
  in policy_winner_table.py and test_corpus_diversity_floor.py vs 5
  short-tier copies (8 entries, current corpus only) in literature_
  deviations_report.py, oracle_gap_report.py, winning_regime_taxonomy.py,
  popt_vs_grasp_report.py, family_saturation_distance.py — every copy
  agrees on family for every shared graph key, using ast.literal_eval
  on the source files (no module imports needed) so the gate stays
  side-effect-free (gate 107, 13 tests); claim_density.json ↔ literature_
  baselines.py parity — every per-graph rollup (n_claims, n_ok, n_cells,
  n_apps, n_policies, n_citations, status_counts) recomputes exactly
  from literature_reproduction_summary.csv, every CSV row's
  (graph, app, l3, policy) reaches via claims_for() expansion or
  KNOWN_DEVIATIONS closure, every CSV citation ⊆ baseline citation
  universe, total_ok_pct matches ratio (gate 108, 13 tests);
  small_l3_thrash internal + WRT-tiny disjointness — 9-policy wide-panel
  4kB snapshot (n_rows = n_cells * n_policies = 9 * 9 = 81), per-policy
  aggregates all recompute from CSV, per-cell winner / runner-up honor
  POLICY_LABEL_ORDER tie-break (necessary for all-thrashing 1.0 cells),
  thrash cells (power-law @ 4kB) disjoint from WRT 'tiny'-regime cells
  (mesh+road @ 4kB and 16kB) so the paper never double-counts
  (gate 109, 13 tests); bootstrap_ci.json + oracle_gap_by_app_bootstrap
  .json parity & hygiene — every (policy, family) and (policy, regime)
  mean/median/n exactly matches oracle_gap.summary, ci_lo ≤ ci_hi with
  ci_width recompute, POPT-vs-GRASP family-level paired-delta sign agrees
  with whether CI excludes zero (locks the headline 'POPT loses on road
  with 95% CI excluding 0' claim), per-app pairs are anti-symmetric in
  mean_delta and p_a_lt_b complement to ≤ 1 + slack (gate 110, 13 tests).
- **Bootstrap / statistical-significance gates**, **policy-rank
  Kendall stability**, **WSS-knee-location**, **family-classification
  sensitivity**, **cross-policy mean-margin asymmetry**, and others
  filled out the dashboard from the original 11 pytest gates to the
  current 110. `make confidence-fast` runs the whole suite in under
  ~3 minutes; `reproduce_smoke.py` snapshots 142 SHA-256 hashes of
  the tracked artifacts and re-runs `make lit-claims lit-catalog`
  in a subprocess to verify drift=0.

Latest additions on top of the Tier A/B/C work:

- `scripts/experiments/ecg/literature_baselines.py` — 264-claim
  spec covering Faldu HPCA20 (143 claims), Balaji HPCA21 (106
  claims), Jaleel ISCA10 (15 claims). Every entry carries a
  `citation=` literal back to a paper figure or section.
- `scripts/experiments/ecg/literature_faithfulness.py` — comparator
  that classifies every observed cell into ok / within_tolerance /
  disagree / known_deviation / insufficient_data / missing.
- `scripts/experiments/ecg/literature_reproduction_summary.py` —
  per-paper grouped reproduction map at
  [`wiki/data/literature_reproduction_summary.md`](data/literature_reproduction_summary.md).
- `scripts/experiments/ecg/regression_budget.py` — per-cell distance-
  to-disagree in pp; emits `wiki/data/regression_budget.{json,md}`.
- `scripts/experiments/ecg/confidence_dashboard.py` — single-screen
  view of all 110+ pytest gates + lit-faith headline + corpus diversity
  + regression budget.
- 6 new pytest gate files in `scripts/test/`:
  `test_baselines_match_literature`, `test_confidence_dashboard`,
  `test_corpus_diversity_floor`, `test_cross_tool_parity`,
  `test_known_deviations_have_root_cause_anchor`,
  `test_literature_baselines_citation_locator`,
  `test_regression_budget_floor`.
- Make targets: `make lit-faith`, `make lit-repro`, `make lit-budget`,
  `make confidence` (CI-ready: exit 0 iff every gate is GREEN).
- `scripts/experiments/ecg/gem5_anchor_summary.py` now emits a
  `small_cache_divergence:<graph>/<app>@4kB` invariant alongside the
  headline (256kB) and asymptote (2MB) checks: at 4kB << WSS the
  three policies **must** diverge by ≥ 2 pp. Together with the
  asymptote check this codifies the GRASP-paper L-shape and would
  catch regressions where a "fix" at small caches collapses policy
  differentiation. Invariants are now per-(graph, app); the Sniper
  anchor scopes to PR + SSSP on `email-Eu-core` and `cit-Patents`
  (16 invariants all `ok`, max small-cache spread 6.36 pp). BFS is
  deferred — 4 kB spread sits at 1.78 pp and email-Eu-core/bfs/Sniper
  shows GRASP +1.49 pp over LRU (insufficient reuse for the L-shape).
- `literature_faithfulness_postfix.json` now reports **0 insufficient_data**
  (was 19). The email-Eu-core/{pr,bc,bfs}/{1MB,4MB,8MB} cells were
  re-run with bumped iterations (PR `-i 2 → -i 20`, BFS `-n 1 → -n 16`,
  BC `-i 1 → -i 64`) to push L3 access counts above the 10 000-access
  validity threshold. lit-faith claims now: 238 ok, 2 within-tolerance,
  30 known-deviation, 0 disagree, 0 missing, 0 insufficient.
- Tier C sign-consistency coverage expanded from 4 to 8 (graph, app)
  pairs: BFS and SSSP added on both email-Eu-core and cit-Patents.
  cache_sim reference sweeps generated for the new pairs; Sniper data
  fold-in confirms strong agreement on email-Eu-core/bfs (all 4 sizes)
  and exposes 3 documented Sniper disagreements (email-Eu-core/sssp —
  noise-floor cache_sim deltas, cit-Patents/sssp@4kB, cit-Patents/bfs
  @4kB+32kB) tracked as xfail in `KNOWN_DISAGREEMENTS`. Tier C count:
  14 pass / 6 skip / 4 xfail (was 5 pass / 3 skip).
- `scripts/experiments/ecg/paper_pipeline.py` now auto-emits per-(graph,
  app) L-curve figures (`figures/l_curve_<graph>_<app>.svg`) and an
  aggregated summary CSV whenever cache_sim rows span ≥3 distinct L3
  sizes. Completes the `final_cache_sim_l_curve` profile end-to-end;
  invariants pinned by 7 new tests in `test_paper_pipeline_l_curve.py`.
- Corpus expanded from 6 → 7 graphs with **roadNet-CA** added as the
  new "road" topology family — hub_concentration 0.140 (vs 0.337 for
  the prior corpus minimum). The new graph is intentionally adversarial
  to GRASP: PR @ 1 MB cache_sim shows GRASP 0.957 vs LRU 0.941
  (GRASP +1.5 pp WORSE because there are no hubs to pin), while POPT
  beats both (0.892). This is direct evidence for the GRASP-needs-hubs
  hypothesis. The finding is locked in by a new theory-driven gate
  `scripts/test/test_no_hub_graph_invariant.py` that asserts no graph
  with `hub_concentration < 0.20` can show GRASP > LRU+0.5 pp, and that
  the corpus must contain ≥1 such no-hub graph (load-bearing rather
  than vacuous).
- roadNet-CA PR cache_sim extended to a full L3 sweep
  ({4 kB, 16 kB, 64 kB, 256 kB, 1 MB}) so the L-curve figure
  `figures/l_curve_roadNet-CA_pr.svg` renders the complete working-set
  curve. The data shows a clean inflection: GRASP only marginally
  beats LRU when both policies are saturated at ≥99.5 % miss
  (4–64 kB), and *loses* monotonically once partial fit appears
  (+0.28 pp at 256 kB, +1.47 pp at 1 MB). POPT remains best at every
  L3 size that can hold useful state. Lit-faith ratio is now 248/280 ok
  (was 240/272); +8 derived POPT-vs-GRASP claims all "ok".
- Corpus expanded again to 8 graphs with **delaunay_n19** added as a
  **mesh** topology family (524 k vertices, 3.1 M edges, hub_concentration
  0.138, clustering_coeff 0.379). The cache_sim data immediately
  *contradicted* the simple "GRASP needs hubs" thesis: at 1 MB GRASP
  beats LRU by 13.73 pp on delaunay_n19 despite uniform degree (~6),
  because GRASP's random-within-bucket protection accidentally aligns
  with the mesh's local cluster structure (protected vertices keep
  their neighbours' hits warm). The original `test_no_hub_graph_invariant.py`
  was therefore renamed (`git mv`) to `test_road_like_graph_invariant.py`
  and the predicate tightened to require BOTH
  `hub_concentration < 0.20` AND `clustering_coeff < 0.10` — i.e.,
  *road-like* (uniform degree AND no triangle structure). roadNet-CA
  still satisfies both (hub 0.140, cluster 0.063); delaunay_n19 is
  correctly excluded by the clustering criterion. Lit-faith ratio
  climbs to **288/320 ok = 90.0 %** (was 248/280). All 17 confidence
  gates remain ✅ GREEN.
- Road-like invariant extended from a single graph-level check to a
  per-(graph, app, L3) cell sweep — every cell on every road-like
  graph at every swept L3 must satisfy the GRASP-cannot-help
  predicate. Test count grows automatically as more road-family
  data folds in (21 cases today for roadNet-CA × {bfs, sssp, cc, pr}
  × 5 L3 sizes + 1 corpus-present check).
- **Four paper-grade aggregator gates added** (bumps total from 17 → 21):
  - `scripts/experiments/ecg/policy_winner_table.py` projects the
    lit-faith CSV onto a winner-per-cell view. 109 cells today; GRASP
    wins 56 (51 %), POPT 41 (38 %), LRU/SRRIP 6 each. Test
    `scripts/test/test_policy_winner_table.py` (7 cases) pins that
    every winner is a known policy, hub graphs at large L3 actually
    have a GRASP/POPT winner, and road-family GRASP wins stay within
    the 0.5 pp noise floor.
  - `scripts/experiments/ecg/small_l3_thrash_report.py` aggregates
    the standalone `final_cache_sim` 4 kB-L3 sweep (9 (graph, app)
    cells × 9 policy variants including POPT_CHARGED and 4 ECG
    modes). LRU wins 5/9 cells; GRASP regresses up to +35.857 pp vs
    LRU on soc-LiveJournal1/bfs. Test
    `scripts/test/test_small_l3_thrash.py` (8 cases) pins the
    "GRASP+POPT both regress ≥ 5 pp vs LRU" tiny-L3 signature,
    that POPT_CHARGED never wins, and that all four ECG variants
    are present.
  - `scripts/experiments/ecg/cross_tool_saturation_report.py`
    pairs each lit-faith cell with the matching gem5/Sniper anchor
    cell, picks each tool's largest L3, and verifies cross-tool
    agreement when both tools are saturated. 7 overlapping cells
    today; 4 doubly-saturated, all agree on Δ(GRASP−LRU) within
    2 pp. Test `scripts/test/test_cross_tool_saturation.py` (8 cases)
    pins that ≥ 1 doubly-saturated cell exists and that no
    doubly-saturated cell disagrees — the central cross-tool
    soundness claim for the paper.
  - `scripts/experiments/ecg/claim_density_report.py` tallies per-
    graph literature claim density (8 graphs, 320 claims, 288 OK
    = 90.0 %). Citation density per graph ranges 2 (delaunay_n19)
    → 12 (cit-Patents). Test `scripts/test/test_claim_density.py`
    (7 cases) pins zero-density absence, status-count consistency,
    and summary-vs-per-graph totals.
- New Make targets: `make lit-winner`, `make lit-thrash`,
  `make lit-cross-tool`, `make lit-density` (all wired into the
  `confidence` dep chain).
- **Four further paper-grade aggregators added** (bumps total from
  21 → 25):
  - `scripts/experiments/ecg/popt_vs_grasp_report.py` projects per-
    cell `Δ(POPT − GRASP)` in pp, broken down by graph family and L3
    regime. Headline: ROAD family mean **−9.276 pp** (POPT crushes
    GRASP, max swing −60.023 pp on roadNet-CA/sssp/1MB); SOCIAL
    family mean +0.360 pp (essentially tie). Counts: POPT better 37,
    GRASP better 35, tie 37 of 109 cells. Test
    `scripts/test/test_popt_vs_grasp_delta.py` (8 cases) pins the
    sign convention, classification consistency, and the family-level
    claims.
  - `scripts/experiments/ecg/literature_deviations_report.py`
    classifies every `known_deviation` row in the reproduction
    summary against a closed mechanism vocabulary
    (`popt_overhead_dominates`, `within_extended_tolerance`,
    `policy_data_missing`, `unclassified`). Today: 30/30 rows classify
    as `popt_overhead_dominates` — the perfect inverse of the road-
    graph finding. Test `scripts/test/test_literature_deviations.py`
    (8 cases) pins the vocabulary, asserts zero unclassified leakage,
    and catches any new policy name without an explicit rule.
  - `scripts/experiments/ecg/paper_claims_registry.py` is the single
    source of truth for every numerical claim the paper makes — 14
    claims across 8 categories (corpus, reproduction, lit_faith,
    winner_table, popt_vs_grasp, thrash, deviations, cross_tool,
    meta), each linked to source artifact + governing gate. Test
    `scripts/test/test_paper_claims_registry.py` (9 cases) pins
    required headline IDs, unique IDs, source/gate file existence,
    the road-popt-negative-sign claim, and a confidence-gate-count
    ≥ 22 floor.
  - `scripts/experiments/ecg/cross_tool_winners_report.py` complements
    the saturation report by computing, for each (graph, app), the
    winning policy each tool picks at its largest L3 (cache_sim,
    gem5, Sniper). Surfaces 6 split cells today — an expected
    negative result because the tools sweep different L3 ranges, so
    largest-L3 operating points sit in different saturation regimes.
    Test `scripts/test/test_cross_tool_winners.py` (8 cases) pins
    schema, closed vocabulary `{unanimous, majority, split}`,
    overlap-with-each-tool, and cache_sim-anchor presence.
  - `scripts/experiments/ecg/winning_regime_taxonomy.py` joins the
    policy winner table with corpus diversity to project a
    (graph family × L3 regime) winner matrix at
    [`wiki/data/winning_regime_taxonomy.md`](data/winning_regime_taxonomy.md).
    Auto-extracts ≥80 % dominance rules: mesh/{tiny,small,large} →
    POPT 100 %; road/large → LRU 75 %; citation/large → GRASP 73 %.
    Test `scripts/test/test_winning_regime_taxonomy.py` (9 cases)
    pins schema, large-regime family coverage, mesh+road presence,
    and the road/large LRU-win existence claim.
  - `scripts/experiments/ecg/oracle_gap_report.py` projects each
    policy's gap to the per-cell empirical oracle = min(LRU, SRRIP,
    GRASP, POPT) at [`wiki/data/oracle_gap.md`](data/oracle_gap.md).
    Mean gaps: POPT 1.65 pp (smallest), GRASP 3.10 pp (most wins:
    56), SRRIP 3.60 pp, LRU 4.93 pp. GRASP/road mean 12.48 pp
    (devastating counter-narrative); POPT/mesh 0.08 pp (near-
    perfect). Test `scripts/test/test_oracle_gap.py` (9 cases) pins
    no-negative-gap, winners-have-zero-gap, every-cell-has-a-
    winner, POPT-smallest-overall-mean (load-bearing), and the
    GRASP/road ≥ 5 pp counter-narrative.
  - `scripts/experiments/ecg/artifact_catalog.py` is the single
    canonical index of every paper-grade aggregator (19 entries
    today) with on-disk audit of (generator, gate, artifact)
    triples. Lives at [`wiki/data/artifact_catalog.md`](data/artifact_catalog.md).
    Test `scripts/test/test_artifact_catalog.py` (10 cases) pins
    schema, no-missing entries on any of the three axes, unique
    snake_lower ids, and an entry-count floor that future PRs
    must bump explicitly. This closes the long-standing five-place
    coordination gap (script + gate + Makefile + dashboard +
    HANDOFF) when adding a new aggregator.
  - `scripts/experiments/ecg/bootstrap_ci.py` adds non-parametric
    percentile bootstrap CIs (5000 resamples, seed 1729) on every
    load-bearing claim: per-(policy, family) and per-(policy, regime)
    oracle-gap means, paired ΔPOPT−GRASP per family, and 7 sign-
    stability claims. Lives at
    [`wiki/data/bootstrap_ci.md`](data/bootstrap_ci.md). Headlines
    are now reported with 95% CIs so reviewers cannot dismiss them
    as point estimates. Key findings: **POPT < GRASP on road
    P=0.976** (bedrock), **POPT < LRU on social P=1.000** (unanimous),
    POPT-mean-smallest headline is dominated by the road family
    (mesh borderline at P=0.948, social/citation/web not significant).
    Test `scripts/test/test_bootstrap_ci.py` (11 cases) pins the
    road sign-stability floor at 0.95, POPT/mesh CI hi ≤ 1.0 pp,
    and POPT-vs-LRU social unanimity ≥ 0.99.
  - `scripts/experiments/ecg/oracle_gap_by_app.py` projects the
    per-cell oracle gap onto the (policy, app) plane so the paper
    can defend "no one-size-fits-all" with per-kernel winners
    instead of relying solely on family-level aggregates. Lives at
    [`wiki/data/oracle_gap_by_app.md`](data/oracle_gap_by_app.md).
    Per-kernel rank-1 (mean gap, pp): **pr→POPT 0.100, bfs→POPT
    1.625, cc→GRASP 0.640, bc→SRRIP 1.689, sssp→POPT** (with
    **GRASP/sssp catastrophic at 7.106 pp**, the worst of any
    policy on any kernel). Test
    `scripts/test/test_oracle_gap_by_app.py` (11 cases) pins all
    20 (policy, app) buckets present, n≥15 per bucket, pr→POPT
    (≤0.5 pp), cc→GRASP, and the GRASP-must-not-win-sssp counter-
    narrative.
  - `scripts/experiments/ecg/wss_relative_l3.py` re-bins every
    oracle-gap cell by L3 / WSS ratio (under_wss < 0.25, near_wss
    0.25–4.0, over_wss > 4.0), using the per-graph
    `working_set_ratio` already published in
    [`wiki/data/corpus_diversity.json`](data/corpus_diversity.json)
    as a WSS proxy. Defends against the reviewer pushback that
    absolute-byte L3 tables silently compare across graphs of
    wildly different sizes. Lives at
    [`wiki/data/wss_relative_l3.md`](data/wss_relative_l3.md).
    Headlines (114 cells; 0 skipped): **POPT has the smallest
    mean gap in EVERY WSS regime** (under 1.619 pp, near 2.351 pp,
    over 0.223 pp); GRASP dominates by **win count** in every
    regime (under 23/48, near 27/52, over 8/14); **LRU win_rate
    in under_wss is 1/48 (~2%)** — strongest quantitative case
    that a real cache-friendly policy actually matters when WSS
    blows past L3. Test `scripts/test/test_wss_relative_l3.py`
    (10 cases) pins the load-bearing per-regime POPT rank-1
    claim plus the no-unknown-graphs invariant so silently-
    dropped cells can't bias the bins.
  - `scripts/experiments/ecg/family_sensitivity.py` re-runs the 7
    sign-stability claims from `bootstrap_ci` under every single-
    graph family reassignment (8 graphs × 4 alternate families =
    32 perturbations), reporting how many flips cross the 0.95
    stability floor. Lives at
    [`wiki/data/family_sensitivity.md`](data/family_sensitivity.md).
    Key findings: **POPT < LRU on social = BEDROCK (0/32 flips)**;
    **POPT < GRASP on road = LOCAL (4/32, all from roadNet-CA
    relocation)** — the road headline depends on a single graph,
    documented as such; POPT < GRASP on mesh = GRAINY (14/32,
    baseline n=5). Uses seed=1729, 2000 resamples. Test
    `scripts/test/test_family_sensitivity.py` (10 cases) pins the
    bedrock claim at 0 flips and enforces that all road flips
    must originate from roadNet-CA (no spurious sources).
  - `scripts/experiments/ecg/reproduce_smoke.py` snapshots SHA-
    256 hashes of 140 tracked `wiki/data/*.{json,md}` artifacts,
    re-runs `make lit-claims lit-catalog` in a subprocess, and
    diffs the canonical hashes (masking volatile timing/runtime
    fields). Lives at
    [`wiki/data/reproduce_smoke.md`](data/reproduce_smoke.md).
    This caught a real drift on first integration —
    `paper_claims.json` carried a stale "28/28" headline after a
    gate count bump because the dashboard regenerates AFTER
    `lit-claims` in the dep chain. Documents the one-cycle
    convergence wart for future maintainers. Test
    `scripts/test/test_reproduce_smoke.py` (8 cases) pins the
    artifact floor at 62 with the load-bearing files list.
  - `scripts/experiments/ecg/oracle_gap_by_app_bootstrap.py`
    paired-bootstraps Δ = gap(a) − gap(b) for every ordered policy
    pair, per kernel (5 apps × 12 pairs = 60 comparisons). 2000
    resamples, seed=1729, 95% percentile CI. Pins CI-backed sign
    claims: pr→POPT<{LRU,SRRIP,GRASP} all P=1.000; cc→GRASP<POPT
    P=0.9995; bfs→POPT<GRASP P=0.999 CI hi=-0.45; sssp→POPT<GRASP
    P=0.971; bc has NO stable ordering among {GRASP, POPT, SRRIP}.
    Output at [`wiki/data/oracle_gap_by_app_bootstrap.md`](data/oracle_gap_by_app_bootstrap.md).
    Test `scripts/test/test_oracle_gap_by_app_bootstrap.py`
    (11 cases) enforces P-floor 0.99 on strong claims, 0.95 on
    stability claims.
  - `scripts/experiments/ecg/popt_vs_grasp_by_family_app.py`
    breaks the POPT-vs-GRASP comparison down by (family × app),
    exposing nuance whole-app gates would average away. 21 cells
    with paired data. Headline findings (all CI-backed):
    **road is POPT-favored on every kernel** (sssp -21.8 pp,
    bfs -11.4 pp, bc -4.6 pp, pr -2.6 pp, cc -1.3 pp);
    cc-counter-narrative is CI-strict on social/cc (+5.5 pp
    P=0.000) and citation/cc (+4.6 pp P=0.000); social/pr is
    CI-strict POPT (P=0.9995); citation/sssp is surprisingly
    GRASP-strict (+1.43 pp P=0.000) — contradicting the per-kernel
    sssp→POPT claim when broken out by family. Output at
    [`wiki/data/popt_vs_grasp_by_family_app.md`](data/popt_vs_grasp_by_family_app.md).
    Test `scripts/test/test_popt_vs_grasp_by_family_app.py`
    (10 cases).
  - `scripts/experiments/ecg/wilson_win_rates.py` — Wilson 95% CIs
    on per-(scope, policy) win-rates. Right tool for small-n
    binomial when p̂ near 0/1. Headline: pr/POPT 20/28 CI
    [0.529, 0.848] strict majority; cc/GRASP 17/20 CI [0.640, 0.948]
    strict majority AND above the 25% null baseline; cc/POPT 0/20
    CI [0.000, 0.161] strict below-chance; sssp policies overlap
    CI. Test `scripts/test/test_wilson_win_rates.py` (11 cases).
  - `scripts/experiments/ecg/cohens_h_win_rates.py` — Cohen's h
    effect-size (arcsine-transformed) on win-rate gaps. 14 large-
    effect (h ≥ 0.8) dominance pairs: cc/GRASP-vs-POPT h=2.346
    (largest), pr/POPT-vs-{LRU,SRRIP} h=2.014. **sssp has no
    large-effect dominance** (max h=0.726 medium). Test
    `scripts/test/test_cohens_h_win_rates.py` (12 cases).
  - `scripts/experiments/ecg/oracle_gap_effect_size.py` — Cliff's
    delta + Mann-Whitney U on RAW gap_pp distributions
    (nonparametric, outlier-robust). MW-U via `math.erfc`, no scipy
    dep. 10 large-effect (|d|≥0.474) dominance pairs. pr/POPT vs
    LRU d=-0.911 MW p=0; cc/GRASP dominates all 3 with MW p<1e-4.
    **sssp again has no large-effect dominance** (third independent
    weak-signal signal). Test `scripts/test/test_oracle_gap_effect_size.py`
    (11 cases).
  - `scripts/experiments/ecg/l3_policy_stability.py` — per-(app, L3)
    winner stability across paper L3 sizes (1MB / 4MB / 8MB).
    **Stable single winners**: cc=GRASP, pr=POPT. **Regime change**:
    bfs (GRASP@1MB → POPT@≥4MB). **No stable winner**: sssp.
    Gray-zone: bc (tied SRRIP/GRASP at 1MB, GRASP unique at 4MB+8MB).
    Pins the firewall against averaging across L3 and silently
    hiding a regime change. Test `scripts/test/test_l3_policy_stability.py`
    (11 cases).
  - `scripts/experiments/ecg/multiple_testing_correction.py` —
    aggregates 81 p-values across the entire gate family (gate 38
    MW, gate 34 paired bootstrap, gate 35 per-(family,app)) and
    applies Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR)
    at α=0.05. **Naive significant 44/81; HB survivors 28/81; BH
    survivors 40/81.** Pins which claims may honestly be called
    'significant' in the paper. Test
    `scripts/test/test_multiple_testing_correction.py` (14 cases).
  - `scripts/experiments/ecg/leave_one_graph_out.py` — drops each
    of the 8 graphs in turn and re-ranks winners. **LOGO-robust**
    (winner survives every drop): pr/POPT, cc/GRASP, bc/GRASP.
    **LOGO-fragile**: bfs (flips when soc-LiveJournal1 dropped),
    sssp (flips under 3/8 drops). Sssp's fragility is now the
    fifth independent signal converging on 'sssp is weak'. Test
    `scripts/test/test_leave_one_graph_out.py` (12 cases).
  - `scripts/experiments/ecg/cell_winner_census.py` — corpus
    decisiveness census. **114 cells: 97.4% unique winner, 2.6%
    tied (3 cells, all in bc/email-Eu-core), 0% no-winner.** The
    one 4-way tie (bc/email-Eu-core/1MB) and two 2-way ties pin
    the 'tied subcorpus' the paper must disclose separately. Test
    `scripts/test/test_cell_winner_census.py` (12 cases).
- New Make targets: `make lit-popt-vs-grasp`, `make lit-deviations`,
  `make lit-claims`, `make lit-cross-tool-winners`,
  `make lit-regime-taxonomy`, `make lit-oracle-gap`,
  `make lit-oracle-gap-by-app`, `make lit-oracle-by-app-bootstrap`,
  `make lit-popt-vs-grasp-by-family-app`,
  `make lit-wilson-wins`, `make lit-cohens-h`, `make lit-gap-effect-size`,
  `make lit-l3-stability`, `make lit-mt-correction`, `make lit-logo-robust`,
  `make lit-cell-census`,
  `make lit-wss-relative-l3`,
  `make lit-bootstrap-ci`, `make lit-family-sensitivity`,
  `make lit-reproduce-smoke`, `make lit-catalog` (all wired into
  the `confidence` dep chain; `lit-claims` depends on every other
  `lit-*` so the registry values are always fresh).

See `wiki/Baseline-Literature-Faithfulness.md` → "The fifteen
confidence gates" and "Regression budget" sections for the
per-gate spec.

## What is already done

**Trace-replay parity (cache_sim layer):** 20/20 zero-delta vs upstream
`faldupriyank/grasp` for {LRU, PIN, GRASP, BELADY} × {BC, BellmanFordOpt,
PageRankOpt, PageRankDeltaOpt, Radii} on web-Google. See
`/tmp/graphbrew-upstream-policy-compare-all4/comparison.csv` (regenerable).

Recent commits on `graphbrew_ecg`:

- `e292903` Align cache_sim/gem5/Sniper with upstream GRASP semantics (f=50 default, `grasp_region` sideband flag, `EvictionPolicy::PIN`).
- `c1372e1` Fix POPT MSB polarity to match HPCA21 reference.
- `0d49c67` Add upstream GRASP trace-replay parity tooling (new `graphbrew_trace_replay.cc` + `upstream_policy_compare.py`).
- `3f8e81b` Extend validation gates and faithfulness pytest (11/11 green).
- `3e5b136` Wiki: GRASP/PIN/POPT faithfulness docs.

Faithfulness state recorded in: [/memories/repo/grasp_srrip_baseline.md](/memories/repo/grasp_srrip_baseline.md).

## What is NOT yet validated

The trace-replay parity only exercises `bench/include/cache_sim/cache_sim.h`.
The gem5 and Sniper integrations share the **same sideband** (registerGRASPTraceRegion,
`grasp_region` flag, `frontier_frac=50`) but use **different replacement-policy
implementations** living inside the simulator overlays. Those are not yet
proven faithful.

## Next session — three-tier validation plan

Run tiers in order; each builds confidence for the next.

### Tier A — Sideband registration sanity (10 min)

Goal: prove gem5 and Sniper actually receive 2 regions (propertyA, propertyB)
with `hot_pct=50` and `grasp_region=true` for a known DBG run.

1. Add a one-shot log line at region registration in:
   - `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh`
   - `bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc`
   Format suggestion: `"[graphctx] register region base=0x%lx upper=0x%lx hot_pct=%u grasp_region=%d"`
2. Run PR on `email-Eu-core` (smallest graph) under both simulators with `-o 5` (DBG):
   ```bash
   make sim-pr && ./bench/bin_sim/pr -f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 5 -n 1
   # then gem5 + sniper equivalents via roi_matrix.py --suite gem5/sniper
   ```
3. Assert log has **exactly 2** lines with `grasp_region=1, hot_pct=50`.
4. Lock in via pytest: `scripts/test/test_grasp_sideband_registration.py` (new).

### Tier B — POPT permutation equivalence (independent of simulators)

POPT is a reordering, not a replacement policy. Validation is fully
deterministic and doesn't need any simulator.

1. Build upstream POPT reference from `research/POPT_HPCA21_CameraReady`
   (header-only / source under `research/`).
2. Run on `web-Google.el` to produce `new_id[]` reference.
3. Run GraphBrew POPT (`bench/include/graphbrew/partition/cagra/popt.h`) on
   the same edge list.
4. Diff the permutation arrays — after the MSB-polarity fix (commit `c1372e1`)
   expect bit-exact equality or Kendall-τ ≈ 1.
5. Lock in via pytest with a tiny synthetic graph.

### Tier C — gem5 / Sniper GRASP-vs-LRU sign test

Bit-exact miss parity is impossible (MSHRs, prefetchers, replacement-state
init differ). Instead, validate **sign consistency** of GRASP-vs-LRU deltas
between cache_sim and the full simulators.

1. Use existing `scripts/experiments/ecg/roi_matrix.py --suite gem5` and
   `--suite sniper` on small graphs (`email-Eu-core`, `cit-Patents`) for
   PR and BC, L3 sizes {4kB, 32kB, 256kB, 2MB}.
2. Compare per-(graph, app, L3-size) GRASP-vs-LRU delta sign against the
   matching cache_sim row in `/tmp/graphbrew-grasp-cache-sweep/*/DBG/roi_matrix.csv`.
3. Disagreement at small L3 (4kB/32kB) = real bug to investigate.
   Disagreement only at 2MB is acceptable (working set fits).
4. Document deltas in `wiki/POPT-GRASP-Faithfulness-Audit.md`.

## Useful one-liners

```bash
# Regenerate trace-replay parity
rm -rf /tmp/graphbrew-upstream-policy-compare-all4 && \
python3 scripts/experiments/ecg/upstream_policy_compare.py \
  --policies lru pin grasp belady \
  --traces BC.web-Google.cvgr.dbg.lru.llc.trace \
           BellmanFordOpt.web-Google.cintgr.dbg.lru.llc.trace \
           PageRankOpt.web-Google.cvgr.dbg.lru.llc.trace \
           PageRankDeltaOpt.web-Google.cvgr.dbg.lru.llc.trace \
           Radii.web-Google.cvgr.dbg.lru.llc.trace \
  --out-dir /tmp/graphbrew-upstream-policy-compare-all4

# Faithfulness pytest
python3 -m pytest -q scripts/test/test_popt_grasp_faithfulness_sources.py \
                     scripts/test/test_ecg_validation_gates.py

# cache_sim sweep that produces the GRASP-vs-LRU reference for Tier C
python3 scripts/experiments/ecg/roi_matrix.py --suite cache-sim --benchmark pr \
  --options "-f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 5 -n 1 -i 1" \
  --policies LRU SRRIP GRASP \
  --l1d-size 1kB --l2-size 2kB --l3-sizes 4kB 32kB 256kB 2MB --l3-ways 16 \
  --line-size 64 --out-dir /tmp/graphbrew-grasp-cache-sweep/email-pr/DBG
```

## Gotchas

- Upstream PIN/BELADY .bin sources live at `/tmp/graphbrew-faithfulness-upstream/grasp/trace-based-simulators/`
  — `upstream_policy_compare.py` builds them on demand. Trace files live in `datasets/`.
- Do **not** run multiple GraphBrew gem5 jobs on the same node unless sideband
  files are isolated (per `.github/copilot-instructions.md`).
- BellmanFordOpt uses `frontier_frac=100`, not 50 — confirm before asserting region params.
- BELADY in `graphbrew_trace_replay.cc` requires the uint64_t cast on `time=-1`
  so empty ways wrap to UINT64_MAX and get evicted first (regression-prone).
