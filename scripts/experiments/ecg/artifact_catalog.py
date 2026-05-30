#!/usr/bin/env python3
"""Paper-artifact catalog — single authoritative index of every
paper-grade aggregator in ``wiki/data/`` plus its governing pytest
gate, source generator, citation purpose, and current headline
finding.

Why this exists
---------------
The paper-grade evidence chain now has 27 confidence gates spread
across 17 JSON aggregators. Onboarding a co-author (or a reviewer
asking "where does claim X come from?") requires walking the
Makefile + gate dashboard + per-aggregator scripts to find the
canonical chain. This script collapses that walk into one
machine-readable + paper-ready index.

The catalog answers, per artifact:

* What it computes (one-line summary).
* The script that generates it.
* The pytest module that guards it.
* The headline number(s) it contributes to the paper.

It is intentionally self-contained — no joins with the data files,
just metadata + a sanity check that each listed source script /
gate / artifact file exists on disk.

Outputs
-------
* ``wiki/data/artifact_catalog.json`` — machine-readable index.
* ``wiki/data/artifact_catalog.md``   — paper-ready table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


# Each entry: (key, label, generator script, pytest gate path,
# json artifact path relative to repo root, one-line headline).
# Keep this sorted by canonical evidence-chain ordering so readers
# see corpus -> reproduction -> per-policy -> per-graph -> meta.
CATALOG = [
    # ---------- Corpus + reproduction foundation ----------
    {
        "id":        "corpus_diversity",
        "label":     "Corpus structural diversity",
        "generator": "scripts/experiments/ecg/corpus_diversity.py",
        "gate":      "scripts/test/test_corpus_diversity_floor.py",
        "artifact":  "wiki/data/corpus_diversity.json",
        "summary":   "Per-graph structural features (hub_concentration, clustering_coeff, avg_degree, working_set_ratio) for every corpus graph; underpins family classification used by every other report.",
    },
    {
        "id":        "literature_reproduction",
        "label":     "Per-paper reproduction summary",
        "generator": "scripts/experiments/ecg/literature_reproduction_summary.py",
        "gate":      "scripts/test/test_baselines_match_literature.py",
        "artifact":  "wiki/data/literature_reproduction_summary.csv",
        "summary":   "Per-paper grouped reproduction map (Faldu HPCA20, Balaji HPCA21, Jaleel ISCA10) classifying every claim into ok/within_tolerance/disagree/known_deviation/missing.",
    },
    {
        "id":        "literature_faithfulness",
        "label":     "Literature faithfulness comparator",
        "generator": "scripts/experiments/ecg/literature_faithfulness.py",
        "gate":      "scripts/test/test_lit_faith_no_disagree.py",
        "artifact":  "wiki/data/literature_faithfulness_postfix.json",
        "summary":   "Cell-level comparator: 288/320 ok (90.0 %), 0 disagree, 30 known_deviation. The single load-bearing aggregate behind every other paper-grade finding.",
    },
    {
        "id":        "paper_baseline_table",
        "label":     "Paper-ready baseline table",
        "generator": "scripts/experiments/ecg/paper_baseline_table.py",
        "gate":      "scripts/test/test_paper_baseline_table.py",
        "artifact":  "wiki/data/paper_baseline_table.json",
        "summary":   "Cross-tabulated (graph, app, L3) table of LRU miss-rate plus SRRIP/GRASP/POPT Δ in pp and literature claim verdicts; mirrors literature_faithfulness so the paper text and the lit-gate cannot disagree.",
    },
    {
        "id":        "regression_budget",
        "label":     "Regression budget floor",
        "generator": "scripts/experiments/ecg/regression_budget.py",
        "gate":      "scripts/test/test_regression_budget_floor.py",
        "artifact":  "wiki/data/regression_budget.json",
        "summary":   "Per-cell distance-to-disagree in pp; the smallest margin defines the corpus-wide regression budget. Fails if any cell's budget collapses below the floor.",
    },
    # ---------- Cross-tool simulator soundness ----------
    {
        "id":        "gem5_anchor",
        "label":     "gem5 literature anchor",
        "generator": "scripts/experiments/ecg/gem5_anchor_summary.py",
        "gate":      "scripts/test/test_gem5_anchor.py",
        "artifact":  "wiki/data/gem5_anchor.json",
        "summary":   "GRASP-paper L-shape invariants codified per (graph, app) at 4 kB / 256 kB / 2 MB. 16 invariants today, all ok.",
    },
    {
        "id":        "sniper_anchor",
        "label":     "Sniper literature anchor",
        "generator": "scripts/experiments/ecg/gem5_anchor_summary.py",
        "gate":      "scripts/test/test_sniper_anchor.py",
        "artifact":  "wiki/data/sniper_anchor.json",
        "summary":   "Sniper L-shape mirror for PR + SSSP on email-Eu-core + cit-Patents (16 invariants ok; max small-cache spread 6.36 pp). Uses the shared gem5_anchor_summary.py generator.",
    },
    {
        "id":        "cross_tool_saturation",
        "label":     "Cross-tool saturation soundness",
        "generator": "scripts/experiments/ecg/cross_tool_saturation_report.py",
        "gate":      "scripts/test/test_cross_tool_saturation.py",
        "artifact":  "wiki/data/cross_tool_saturation.json",
        "summary":   "Pairs each lit-faith cell with its gem5/Sniper anchor at each tool's largest L3 and verifies Δ(GRASP−LRU) sign agreement when doubly saturated.",
    },
    {
        "id":        "cross_tool_winners",
        "label":     "Cross-tool winner agreement",
        "generator": "scripts/experiments/ecg/cross_tool_winners_report.py",
        "gate":      "scripts/test/test_cross_tool_winners.py",
        "artifact":  "wiki/data/cross_tool_winners.json",
        "summary":   "At each tool's largest L3 per (graph, app), do simulators pick the same winning policy? Surfaces 6 split cells (expected; tools sweep disjoint L3 ranges).",
    },
    # ---------- Per-policy / per-cell analyses ----------
    {
        "id":        "policy_winner_table",
        "label":     "Policy winner table",
        "generator": "scripts/experiments/ecg/policy_winner_table.py",
        "gate":      "scripts/test/test_policy_winner_table.py",
        "artifact":  "wiki/data/policy_winner_table.json",
        "summary":   "Per-cell winner projection: GRASP 56 wins, POPT 41, LRU 6, SRRIP 6 (n=109). Top-5 fragile cells flagged when margin ≤ 1 pp.",
    },
    {
        "id":        "popt_vs_grasp_delta",
        "label":     "POPT-vs-GRASP delta",
        "generator": "scripts/experiments/ecg/popt_vs_grasp_report.py",
        "gate":      "scripts/test/test_popt_vs_grasp_delta.py",
        "artifact":  "wiki/data/popt_vs_grasp_delta.json",
        "summary":   "Per-cell Δ(POPT − GRASP) in pp by family/regime. Road family mean −9.276 pp (POPT crushes GRASP); social family mean +0.360 pp (tie).",
    },
    {
        "id":        "oracle_gap",
        "label":     "Per-policy oracle gap",
        "generator": "scripts/experiments/ecg/oracle_gap_report.py",
        "gate":      "scripts/test/test_oracle_gap.py",
        "artifact":  "wiki/data/oracle_gap.json",
        "summary":   "Each policy's gap to per-cell empirical oracle (min across 4 policies). Mean gaps: POPT 1.78 pp, GRASP 3.37 pp, SRRIP 3.46 pp, LRU 4.76 pp.",
    },
    {
        "id":        "oracle_gap_by_app",
        "label":     "Per-kernel oracle gap",
        "generator": "scripts/experiments/ecg/oracle_gap_by_app.py",
        "gate":      "scripts/test/test_oracle_gap_by_app.py",
        "artifact":  "wiki/data/oracle_gap_by_app.json",
        "summary":   "Per-(policy, app) oracle-gap matrix exposing per-kernel winners: POPT crushes pr (0.100 pp); GRASP wins cc (0.640 pp); GRASP catastrophic on sssp (7.106 pp). No one-size-fits-all.",
    },
    {
        "id":        "oracle_gap_by_app_bootstrap",
        "label":     "Per-kernel oracle-gap bootstrap CIs",
        "generator": "scripts/experiments/ecg/oracle_gap_by_app_bootstrap.py",
        "gate":      "scripts/test/test_oracle_gap_by_app_bootstrap.py",
        "artifact":  "wiki/data/oracle_gap_by_app_bootstrap.json",
        "summary":   "CI-backed sign claims per kernel: pr→POPT<all P=1.0; cc→GRASP<POPT P=0.9995; bfs→POPT<GRASP P=0.999; sssp→POPT<GRASP P=0.971; bc has no stable ordering among GRASP/POPT/SRRIP.",
    },
    {
        "id":        "popt_vs_grasp_by_family_app",
        "label":     "POPT-vs-GRASP per (family x app) CIs",
        "generator": "scripts/experiments/ecg/popt_vs_grasp_by_family_app.py",
        "gate":      "scripts/test/test_popt_vs_grasp_by_family_app.py",
        "artifact":  "wiki/data/popt_vs_grasp_by_family_app.json",
        "summary":   "21 (family,app) cells bootstrapped. Road is POPT-favored on every kernel (sssp -21.8pp, bfs -11.4pp). cc-counter-narrative is CI-strict on social/cc and citation/cc (P≈0.000). social/pr is CI-strict POPT (P=0.9995).",
    },
    {
        "id":        "wilson_win_rates",
        "label":     "Wilson CIs on policy win-counts",
        "generator": "scripts/experiments/ecg/wilson_win_rates.py",
        "gate":      "scripts/test/test_wilson_win_rates.py",
        "artifact":  "wiki/data/wilson_win_rates.json",
        "summary":   "Wilson 95% CIs on per-(scope,policy) win-rates. CI-strict majority: pr/POPT [0.529,0.848], cc/GRASP [0.640,0.948]. CI-strict below-chance: cc/POPT [0.000,0.161]. sssp POPT/GRASP not CI-distinguishable.",
    },
    {
        "id":        "cohens_h_win_rates",
        "label":     "Cohen's h on policy win-rate gaps",
        "generator": "scripts/experiments/ecg/cohens_h_win_rates.py",
        "gate":      "scripts/test/test_cohens_h_win_rates.py",
        "artifact":  "wiki/data/cohens_h_win_rates.json",
        "summary":   "14 large-effect (h≥0.8) dominance pairs. cc/GRASP-vs-POPT h=2.346 (largest). pr/POPT-vs-{LRU,SRRIP} h=2.014. sssp has NO large effect (max h=0.726); pinned as the kernel with weakest policy-ordering signal.",
    },
    {
        "id":        "oracle_gap_effect_size",
        "label":     "Cliff's delta + Mann-Whitney on gap distributions",
        "generator": "scripts/experiments/ecg/oracle_gap_effect_size.py",
        "gate":      "scripts/test/test_oracle_gap_effect_size.py",
        "artifact":  "wiki/data/oracle_gap_effect_size.json",
        "summary":   "10 large-effect (|d|≥0.474) dominance pairs on raw gap_pp distributions. pr/POPT vs LRU d=-0.911 p=0. cc/GRASP dominates all 3 at |d|≥0.474 with MW p<1e-4. Confirms gates 36–37 nonparametrically.",
    },
    {
        "id":        "l3_policy_stability",
        "label":     "Per-L3-size policy stability",
        "generator": "scripts/experiments/ecg/l3_policy_stability.py",
        "gate":      "scripts/test/test_l3_policy_stability.py",
        "artifact":  "wiki/data/l3_policy_stability.json",
        "summary":   "Stable single winners at 1MB/4MB/8MB: cc=GRASP, pr=POPT. Regime change: bfs (GRASP@1MB → POPT@≥4MB). No stable cross-L3 winner: sssp. Pinned so paper can never silently average across L3 and hide a regime change.",
    },
    {
        "id":        "multiple_testing_correction",
        "label":     "Multiple-testing correction (Holm-Bonferroni + Benjamini-Hochberg)",
        "generator": "scripts/experiments/ecg/multiple_testing_correction.py",
        "gate":      "scripts/test/test_multiple_testing_correction.py",
        "artifact":  "wiki/data/multiple_testing_correction.json",
        "summary":   "81 p-values across gates 34/38 + per-(family,app) combined; HB (FWER) retains 28, BH (FDR) retains 40 at α=0.05. Pins which findings can honestly be called 'significant' after correcting for the family of tests we actually run.",
    },
    {
        "id":        "leave_one_graph_out",
        "label":     "Leave-one-graph-out (LOGO) winner robustness",
        "generator": "scripts/experiments/ecg/leave_one_graph_out.py",
        "gate":      "scripts/test/test_leave_one_graph_out.py",
        "artifact":  "wiki/data/leave_one_graph_out.json",
        "summary":   "LOGO-robust apps (winner survives every single-graph drop): pr/POPT, cc/GRASP, bc/GRASP. LOGO-fragile: bfs (flips when soc-LiveJournal1 dropped), sssp (flips under com-orkut, roadNet-CA, web-Google drops). Triangulates the cross-gate weak-signal finding for sssp.",
    },
    {
        "id":        "cell_winner_census",
        "label":     "Cell-winner census (corpus decisiveness)",
        "generator": "scripts/experiments/ecg/cell_winner_census.py",
        "gate":      "scripts/test/test_cell_winner_census.py",
        "artifact":  "wiki/data/cell_winner_census.json",
        "summary":   "114 cells: 97.4% have unique winner, 2.6% tied (3 cells all in bc/email-Eu-core: one 4-way tie + two 2-way ties), 0% no-winner. Pins corpus decisiveness; paper must report the tied subcorpus separately.",
    },
    {
        "id":        "family_geomean_improvement",
        "label":     "Family geomean improvement vs LRU (bootstrap CIs)",
        "generator": "scripts/experiments/ecg/family_geomean_improvement.py",
        "gate":      "scripts/test/test_family_geomean_improvement.py",
        "artifact":  "wiki/data/family_geomean_improvement.json",
        "summary":   "63 (family, app, policy != LRU) records with bootstrap CIs on the geomean miss-rate ratio. 34/63 CI-strict improvements vs LRU; 0 CI-strict regressions ('do no harm' check). Marquee: citation/pr/POPT geomean 0.68 (-32% miss-rate, CI [-43%, -13%]); citation/cc/GRASP -26%; social/cc/GRASP -23%; social/pr/POPT -21%.",
    },
    {
        "id":        "per_graph_app_stability",
        "label":     "Per-(graph, app) winner stability across L3",
        "generator": "scripts/experiments/ecg/per_graph_app_stability.py",
        "gate":      "scripts/test/test_per_graph_app_stability.py",
        "artifact":  "wiki/data/per_graph_app_stability.json",
        "summary":   "34 (graph, app) cells at paper L3 (1MB/4MB/8MB): 13 stable-unique winner, 1 stable-partial (email-Eu-core/bc multi-tie), 14 regime-change, 6 insufficient-L3 (roadNet-CA + delaunay_n19/pr only at 1MB). web-Google maximally volatile (5/5 apps flip); soc-LiveJournal1 + cit-Patents most reliable (4/5 stable each).",
    },
    {
        "id":        "corpus_balance",
        "label":     "Corpus tier/family balance audit",
        "generator": "scripts/experiments/ecg/corpus_balance.py",
        "gate":      "scripts/test/test_corpus_balance.py",
        "artifact":  "wiki/data/corpus_balance.json",
        "summary":   "8 graphs across 5 families. Social dominates: 4/8 graphs (50%), 216/360 paper-L3 cells (60%). Pielou evenness 0.86, Simpson's D 0.66. road + mesh capped below 4MB. Apps balanced within +/-30% of mean. Defends against 'unbalanced corpus' reviewer pushback with exact numbers + per-family L3 coverage matrix.",
    },
    {
        "id":        "distribution_diagnostics",
        "label":     "Per-policy miss-rate distribution diagnostics",
        "generator": "scripts/experiments/ecg/distribution_diagnostics.py",
        "gate":      "scripts/test/test_distribution_diagnostics.py",
        "artifact":  "wiki/data/distribution_diagnostics.json",
        "summary":   "Validates bootstrap CI assumptions (gates 35, 43) by pinning per-policy skewness + excess kurtosis at paper L3. Worst observed |skewness| ~ 1.3 (bfs/GRASP), worst |excess kurtosis| ~ 1.4 (sssp/GRASP); both well inside Hesterberg 2015's |skew|<2, |kurt|<7 envelope. Verdict: PASS. Future drift past envelope auto-fails this gate.",
    },
    {
        "id":        "lofo_robustness",
        "label":     "Leave-one-family-out (LOFO) winner robustness",
        "generator": "scripts/experiments/ecg/lofo_robustness.py",
        "gate":      "scripts/test/test_lofo_robustness.py",
        "artifact":  "wiki/data/lofo_robustness.json",
        "summary":   "Strictly stronger sibling of gate 41 (LOGO): drops an entire family at a time (1-4 graphs) and re-ranks per app. 3/5 apps (bc, cc, pr) are LOFO-robust; bfs is honestly disclosed as social-sensitive (drop social -> POPT loses to GRASP) and sssp as citation-sensitive (drop citation -> GRASP loses to POPT).",
    },
    {
        "id":        "winner_margin_gradient",
        "label":     "Per-(app, L3) winner-margin gradient",
        "generator": "scripts/experiments/ecg/winner_margin_gradient.py",
        "gate":      "scripts/test/test_winner_margin_gradient.py",
        "artifact":  "wiki/data/winner_margin_gradient.json",
        "summary":   "Classifies all 15 (app, L3) paper-scope cells by top-vs-runner-up margin: 6 decisive (margin>=4), 6 moderate (2<=margin<4), 1 weak (sssp/1MB), 2 tied (bc/1MB + sssp/8MB). 12/15 = 80% strong cells. Honest disclosure of weak/tied cells defends against 'your winner is one cell from flipping' pushback.",
    },
    {
        "id":        "oracle_gap_auc",
        "label":     "Per-(app, policy) oracle-gap AUC across L3 sweep",
        "generator": "scripts/experiments/ecg/oracle_gap_auc.py",
        "gate":      "scripts/test/test_oracle_gap_auc.py",
        "artifact":  "wiki/data/oracle_gap_auc.json",
        "summary":   "Collapses each (app, policy) trajectory across paper L3 (1MB->4MB->8MB) into one trapezoidal AUC score (gap_pp x log2(MB), smaller = closer to oracle). AUC winners: bc=GRASP, bfs=POPT, cc=GRASP, pr=POPT (AUC<1, basically tracks oracle), sssp=POPT. sssp AUC winner POPT differs from cell-vote winner GRASP — honestly disclosed.",
    },
    {
        "id":        "policy_auc_correlation",
        "label":     "Cross-app policy-AUC correlation matrix",
        "generator": "scripts/experiments/ecg/policy_auc_correlation.py",
        "gate":      "scripts/test/test_policy_auc_correlation.py",
        "artifact":  "wiki/data/policy_auc_correlation.json",
        "summary":   "Reads gate 49 AUC vectors, z-normalizes within each app, and runs pairwise Pearson correlation across 4 policy dimensions. Apps cluster into two AUC-winner groups (GRASP=[bc,cc], POPT=[bfs,pr,sssp]) with intra-cluster mean r positive for every app and 4/5 apps showing intra > inter (paper headline). bfs+sssp is the strongest pair (r=+0.855) — both frontier-bound traversals.",
    },
    {
        "id":        "policy_stability",
        "label":     "Per-policy AUC stability across apps",
        "generator": "scripts/experiments/ecg/policy_stability.py",
        "gate":      "scripts/test/test_policy_stability.py",
        "artifact":  "wiki/data/policy_stability.json",
        "summary":   "Computes coefficient of variation of AUC across 5 paper apps per policy. LRU has lowest CV (0.42) — predictably bad; POPT has highest CV (0.72) — high-reward/high-variance; SRRIP is the 'safe runner-up' (never wins, never finishes last, always rank 2 or 3). POPT wins 3/5 apps, GRASP is bimodal (wins bc+cc, finishes last on bfs+sssp).",
    },
    {
        "id":        "cache_sensitivity_slope",
        "label":     "Per-(app, policy) cache-sensitivity slope across L3 octaves",
        "generator": "scripts/experiments/ecg/cache_sensitivity_slope.py",
        "gate":      "scripts/test/test_cache_sensitivity_slope.py",
        "artifact":  "wiki/data/cache_sensitivity_slope.json",
        "summary":   "Per-(app, policy) gap_pp shrinkage per L3 octave. GRASP has largest mean slope (2.29 pp/octave, shrinks fastest as L3 grows); LRU has near-zero mean slope (0.15) and the most cells with anti-scaling. Key invariant: significant anti-scaling (gap GROWS by >=1.0 pp at any octave) is exclusively LRU/SRRIP — the oracle-aware GRASP and POPT NEVER regress as L3 grows.",
    },
    {
        "id":        "per_graph_cache_slope",
        "label":     "Per-graph oracle-gap cache-sensitivity slope (refines gate 52)",
        "generator": "scripts/experiments/ecg/per_graph_cache_slope.py",
        "gate":      "scripts/test/test_per_graph_cache_slope.py",
        "artifact":  "wiki/data/per_graph_cache_slope.json",
        "summary":   "Drills the gate-52 anti-scaling story down to individual graphs. Of 112 full (graph, app, policy) trajectories at paper L3, 33 cells show >=1.0 pp single-octave gap growth. LRU+SRRIP dominate (26 of 33 = 79%). The corpus-averaged 'oracle-aware never regress' claim has 7 per-graph exceptions: GRASP regresses on cit-Patents/pr (+7.2 pp) and web-Google/bfs; POPT regresses predominantly on bc (4 graphs). Worst single cell: web-Google/bfs/LRU at +14.7 pp. email-Eu-core (PR pilot) has zero anti-scaling.",
    },
    {
        "id":        "cross_generator_gap_parity",
        "label":     "Cross-generator gap_pp parity (oracle_gap vs auc vs slope)",
        "generator": "scripts/experiments/ecg/cross_generator_gap_parity.py",
        "gate":      "scripts/test/test_cross_generator_gap_parity.py",
        "artifact":  "wiki/data/cross_generator_gap_parity.json",
        "summary":   "Reconciles the three load-bearing aggregators that all expose per-(app, policy, L3) gap_pp via different schemas (raw rows / AUC trajectory / slope decoration). Verifies all 60 paper triples (5 apps x 4 policies x 3 L3) report identical gap_pp values within 1e-3 pp. Zero mismatches today. Catches silent staleness, aggregation drift, or rounding mismatches that would invisibly break the paper narrative.",
    },
    {
        "id":        "cache_saturation_onset",
        "label":     "Cache-saturation onset detection",
        "generator": "scripts/experiments/ecg/cache_saturation_onset.py",
        "gate":      "scripts/test/test_cache_saturation_onset.py",
        "artifact":  "wiki/data/cache_saturation_onset.json",
        "summary":   "Per-(app, policy), the L3 size beyond which extra cache buys <0.5 pp/octave gap improvement. Saturation ordering today: POPT > GRASP > LRU > SRRIP. POPT saturates on 3/5 apps within paper L3 (bc, pr, sssp); GRASP on 2 (bc, sssp); LRU and SRRIP on just 1 each (bc). bfs is universally unsaturated — its working set far exceeds 8MB. Paper-grade mechanism story: oracle-aware policies hit diminishing returns earlier because they are already close to oracle.",
    },
    {
        "id":        "gap_distribution_shape",
        "label":     "Per-cell gap-distribution shape envelope",
        "generator": "scripts/experiments/ecg/gap_distribution_shape.py",
        "gate":      "scripts/test/test_gap_distribution_shape.py",
        "artifact":  "wiki/data/gap_distribution_shape.json",
        "summary":   "Per-(app, L3, policy) skew/excess-kurtosis envelope for the paper L3 grid. Extends gate 46 from pooled marginals to the cell level. 14 cells exceed the Hesterberg textbook envelope (|skew|<2, |kurt|<7); all 14 are oracle-aware (GRASP or POPT) and exhibit the same discrete pattern: many oracle-tight near-zero gaps plus one mesh/road outlier (typically roadNet-CA or web-Google). Pinned-exception set guards against any NEW cell entering the offending set. Recommended remedy for the 14 pinned cells: BCa or studentized-t bootstrap.",
    },
    {
        "id":        "family_policy_auc_clustering",
        "label":     "Per-family policy-AUC clustering replay",
        "generator": "scripts/experiments/ecg/family_policy_auc_clustering.py",
        "gate":      "scripts/test/test_family_policy_auc_clustering.py",
        "artifact":  "wiki/data/family_policy_auc_clustering.json",
        "summary":   "Re-derives the AUC winner per app inside each qualifying graph family (citation, social, web — the three with full 1MB/4MB/8MB coverage) and replays gate 50's intra/inter cluster correlation. social (n=4 graphs) replays the global winners perfectly with intra-r 0.92 vs inter-r 0.60. web (n=1) replays 5/5. citation (cit-Patents only) flips bfs and sssp from POPT to GRASP — pinned as expected for a low-out-degree-skew network. Pin guards against any NEW family/app pair drifting.",
    },
    {
        "id":        "oracle_gap_curvature",
        "label":     "Oracle-gap trajectory curvature (knee)",
        "generator": "scripts/experiments/ecg/oracle_gap_curvature.py",
        "gate":      "scripts/test/test_oracle_gap_curvature.py",
        "artifact":  "wiki/data/oracle_gap_curvature.json",
        "summary":   "Per-(app, policy) discrete second derivative of the oracle-gap trajectory on a log2-MB L3 axis. GRASP shows the sharpest knee (mean curv 2.87 pp/oct^2, 4/5 apps); POPT shows a milder knee (0.69, 3/5 apps); LRU and SRRIP both show negative mean curvature (still accelerating their descent at 4→8MB, 0 knees each). Disagreement with gate 55 on lead policy is itself informative: saturation-by-threshold favors POPT (already flat), curvature favors GRASP (the dramatic plateau emerges). Both agree oracle-aware policies dominate non-oracle on plateau evidence.",
    },
    {
        "id":        "policy_rank_kendall",
        "label":     "Policy-rank Kendall-tau across L3 octave",
        "generator": "scripts/experiments/ecg/policy_rank_kendall.py",
        "gate":      "scripts/test/test_policy_rank_kendall.py",
        "artifact":  "wiki/data/policy_rank_kendall.json",
        "summary":   "Per-(app, graph) Kendall-tau rank correlation between the four policies' miss-rate rankings at 1MB vs 4MB vs 8MB. Median tau 1MB↔8MB is 0.50 (positive — rank is generally predictive across capacity), but six cells flip: three GRASP-thrash-at-1MB cells (bc/cit-Patents, bc/web-Google, pr/email-Eu-core where GRASP ranks 4th at 1MB and 1st at 4MB), two large-cache-fit-in-WSS cells (cc/web-Google, sssp/soc-pokec where oracle-pinning hurts when there is no pressure), one pico-corpus noise cell (bfs/email-Eu-core). Verdict pins those six and fails on any new flip.",
    },
    {
        "id":        "wss_knee_location",
        "label":     "WSS-relative knee location",
        "generator": "scripts/experiments/ecg/wss_knee_location.py",
        "gate":      "scripts/test/test_wss_knee_location.py",
        "artifact":  "wiki/data/wss_knee_location.json",
        "summary":   "Per-policy plateau location on the under_wss → near_wss → over_wss regime ladder. Knee = first regime where median gap-to-oracle drops at-or-below 0.5pp. Today GRASP and POPT both plateau at under_wss (rank 0); LRU and SRRIP only at over_wss (rank 2) — a full two ladder steps of separation. Verdict PASS iff every oracle-aware policy plateaus strictly earlier than every non-oracle policy. Combines gates 41 (WSS-relative L3), 55 (saturation onset), and 58 (curvature) into a single ladder-rank invariant.",
    },
    {
        "id":        "family_curvature_replay",
        "label":     "Per-family oracle-gap curvature replay",
        "generator": "scripts/experiments/ecg/family_curvature_replay.py",
        "gate":      "scripts/test/test_family_curvature_replay.py",
        "artifact":  "wiki/data/family_curvature_replay.json",
        "summary":   "Re-runs the gate 58 curvature signal one graph family at a time on the three families with full 1MB/4MB/8MB coverage (citation, social, web). All three replay the global pattern: at least one oracle-aware policy has positive mean curvature (web: GRASP +0.683, social: GRASP +0.169, citation: POPT +0.044) while every non-oracle policy stays non-positive. Verdict PASS iff at least one family replays the pattern AND no NEW family deviates from the pin set. This is the family-level analog of gate 57 (per-family AUC replay) for the curvature metric.",
    },
    {
        "id":        "winner_margin_by_regime",
        "label":     "Winner-margin distribution by WSS regime",
        "generator": "scripts/experiments/ecg/winner_margin_by_regime.py",
        "gate":      "scripts/test/test_winner_margin_by_regime.py",
        "artifact":  "wiki/data/winner_margin_by_regime.json",
        "summary":   "Per-(policy, regime) distribution of winner margin over second-best (in pp of miss-rate). Tests the paper's central claim that oracle-aware payoff is biggest under pressure: GRASP's median margin shrinks from 0.86pp at under_wss to 0.001pp at over_wss; POPT's from 1.16pp at under_wss to 0.02pp at over_wss. Oracle-aware policies (GRASP+POPT) win 100 of 114 classified cells; non-oracle policies (LRU+SRRIP) win only 14. Verdict PASS iff every regime has at least one win AND at least one oracle-aware policy's under_wss median margin strictly exceeds its over_wss median margin.",
    },
    {
        "id":        "family_margin_replay",
        "label":     "Per-family winner-margin replay",
        "generator": "scripts/experiments/ecg/family_margin_replay.py",
        "gate":      "scripts/test/test_family_margin_replay.py",
        "artifact":  "wiki/data/family_margin_replay.json",
        "summary":   "Re-runs gate 62 one graph family at a time. A family qualifies iff at least one oracle-aware policy wins cells in BOTH under_wss and over_wss regimes; only the social family (com-orkut, email-Eu-core, soc-LiveJournal1, soc-pokec) has the WSS diversity to qualify today. social replays with both GRASP (2.54pp -> 0.001pp) and POPT (1.89pp -> 0.000pp) showing the global shrink pattern. Verdict PASS iff at least one family replays AND no NEW family deviates beyond the pinned set. Family-level analog of gate 62, sibling to gates 57 and 61.",
    },
    {
        "id":        "cross_policy_asymmetry",
        "label":     "Cross-policy mean-margin asymmetry",
        "generator": "scripts/experiments/ecg/cross_policy_asymmetry.py",
        "gate":      "scripts/test/test_cross_policy_asymmetry.py",
        "artifact":  "wiki/data/cross_policy_asymmetry.json",
        "summary":   "Head-to-head between every (A,B) policy pair: when A wins H2H against B, by how much pp; and when B wins, by how much. Records 6 unordered pairs with a-wins, b-wins, ties, a/b mean-margin, and asymmetry ratio. GRASP vs POPT is near-balanced (57 vs 55 wins, 2.67x ratio); GRASP vs LRU is lopsided (92 vs 20 wins, 14.05pp LRU mean loss); POPT vs SRRIP is most symmetric (1.04x). Max asymmetry today is 3.20x. Verdict PASS iff every pair has at least one win on each side AND the largest observed asymmetry ratio stays under the 20x sanity ceiling — catches future regressions where one policy's losing-magnitude explodes.",
    },
    {
        "id":        "saturation_distance",
        "label":     "Per-app saturation distance (4MB->8MB)",
        "generator": "scripts/experiments/ecg/saturation_distance.py",
        "gate":      "scripts/test/test_saturation_distance.py",
        "artifact":  "wiki/data/saturation_distance.json",
        "summary":   "Per-(app, graph) gap between best-policy miss rate at 4MB and at 8MB; per-app median quantifies how much an application is still cache-bound at 8MB. bc is the least saturated (17.45pp median, still streaming after 8MB); bfs is the most saturated (4.64pp median, frontier reuse takes over); pr/cc/sssp sit in between. email-Eu-core (pico-sentinel) is saturated for every app within 0.05pp. App-level diversity range = 12.81pp. Verdict PASS iff (1) every cell with WSS > 4MB has non-negative 4MB->8MB best-policy improvement (cache monotonicity), (2) email-Eu-core saturates everywhere, and (3) per-app median range >= 3pp (corpus retains app-level signal).",
    },
    {
        "id":        "capacity_sensitivity",
        "label":     "Per-policy capacity-sensitivity slope",
        "generator": "scripts/experiments/ecg/capacity_sensitivity.py",
        "gate":      "scripts/test/test_capacity_sensitivity.py",
        "artifact":  "wiki/data/capacity_sensitivity.json",
        "summary":   "Per-(app, graph, policy) OLS slope of miss_rate (pp) versus log2(L3 MB) on the {1MB, 4MB, 8MB} axis. Per-policy median slope ranks policies by how much they benefit from cache scaling: LRU is steepest (median -15.62 pp/octave, most cache-hungry); GRASP is shallowest (median -14.65 pp/octave, extracts the most at small caches). POPT (-14.76) and SRRIP (-15.60) sit between. The 0.97 pp/octave gap between LRU and GRASP medians is the dual of gate 62's central finding: oracle-aware policies harvest more value when capacity is tight, so they have less marginal value to gain as cache grows. Verdict PASS iff (1) every policy's median slope is strictly < -5 pp/octave, (2) LRU has the steepest median slope, and (3) GRASP is strictly shallower than LRU.",
    },
    {
        "id":        "family_slope_replay",
        "label":     "Per-family capacity-sensitivity slope replay",
        "generator": "scripts/experiments/ecg/family_slope_replay.py",
        "gate":      "scripts/test/test_family_slope_replay.py",
        "artifact":  "wiki/data/family_slope_replay.json",
        "summary":   "Per-family replay of the gate 66 slope ordering. For each graph family with at least one (graph, app) cell scored across all four policies at 1MB/4MB/8MB, recompute the per-policy median OLS slope and check that LRU and SRRIP are strictly steeper than GRASP and that every policy still beats the -5 pp/octave help floor. Today citation (-13.68/-16.10/-15.52/-14.79) and web (-16.02/-16.48/-18.30/-17.02) replay cleanly; social is pinned as a known deviation because email-Eu-core saturates at every L3 size (WSS ~4.5 kB) and its near-zero slopes wash out the GRASP-vs-LRU ordering once mixed with the other social graphs. Verdict PASS iff at least one family replays and no NEW family deviates beyond the pinned set.",
    },
    {
        "id":        "per_app_capacity_slope",
        "label":     "Per-app capacity-sensitivity slope",
        "generator": "scripts/experiments/ecg/per_app_capacity_slope.py",
        "gate":      "scripts/test/test_per_app_capacity_slope.py",
        "artifact":  "wiki/data/per_app_capacity_slope.json",
        "summary":   "Per-(app, policy) median OLS slope of miss_rate (pp) vs log2(L3 MB), aggregated across all graphs that scored. sssp is most cache-hungry (median-of-medians -19.47 pp/octave); bfs is least (-5.23). Per-app median-of-medians range = 14.24 pp/octave — the corpus spans a wide cache-sensitivity spectrum. bfs is pinned as a known kernel deviation: its frontier-driven access pattern inverts the global GRASP-vs-LRU ordering (LRU median -3.99 > GRASP -6.41 pp/octave), consistent with gate 65 flagging bfs as the most-saturated kernel. Verdict PASS iff (1) every (app, policy) median slope < 0, (2) no app outside the pinned set has GRASP more than 1.0 pp/octave steeper than LRU, and (3) at least one app has every policy median below the help floor.",
    },
    {
        "id":        "slope_saturation_xcheck",
        "label":     "Slope vs saturation-distance cross-check",
        "generator": "scripts/experiments/ecg/slope_saturation_xcheck.py",
        "gate":      "scripts/test/test_slope_saturation_xcheck.py",
        "artifact":  "wiki/data/slope_saturation_xcheck.json",
        "summary":   "Within-policy cross-check between gate 65 (saturation distance, miss(4MB)-miss(8MB)) and gate 66 (capacity-sensitivity slope, OLS over {1MB,4MB,8MB}). Both metrics derive from the same per-(app, graph, policy) miss curve, so they must be positively correlated. Observed: 102 non-flat cells (10 flat cells excluded via |slope|<0.05 pp), Pearson r=0.51, Spearman rho=0.45, median(distance/|slope|)=0.96. The correlation is moderate-not-strong because heterogeneous curve shapes (convex vs concave) decouple the upper-octave drop from the average per-octave drop. This is a regression test for the gate 65 and gate 66 generators: a sign flip in either would push the correlation negative, a scaling error would push the ratio outside [0.70, 1.30]. Verdict PASS iff (1) >= 80 matched cells, (2) Pearson r >= 0.40, (3) Spearman rho >= 0.40, (4) median ratio in [0.70, 1.30].",
    },
    {
        "id":        "gem5_slope_replay",
        "label":     "Gem5 anchor slope sanity",
        "generator": "scripts/experiments/ecg/gem5_slope_replay.py",
        "gate":      "scripts/test/test_gem5_slope_replay.py",
        "artifact":  "wiki/data/gem5_slope_replay.json",
        "summary":   "Computes OLS slope of miss_rate (pp) vs log2(L3 kB) across the four gem5 anchor sizes (4kB/32kB/256kB/2MB) per (app, graph, policy). On the current bc+pr@email-Eu-core anchor corpus, GRASP median = -5.96 pp/oct, LRU = -5.12, SRRIP = -7.21. Verdict PASS iff (1) cache monotonicity holds in every cell (miss(4kB) > miss(2MB)), (2) every per-policy median slope is negative, (3) SRRIP median <= GRASP median (SRRIP is more cache-hungry than the oracle-aware GRASP), (4) GRASP median is below the help-floor (-1.0 pp/oct). The LRU-vs-GRASP delta is reported as INFORMATIONAL and explicitly NOT gated: sub-WSS anchor scales (4kB << email-Eu-core WSS ~4.5kB) put PR into a small-cache regime where LRU's give-up-and-stream behavior beats GRASP's hold-the-hot-set behavior, which inverts the LRU>GRASP ordering observed at 1-8MB cache-sim sizes.",
    },
    {
        "id":        "sniper_slope_replay",
        "label":     "Sniper anchor slope sanity",
        "generator": "scripts/experiments/ecg/sniper_slope_replay.py",
        "gate":      "scripts/test/test_sniper_slope_replay.py",
        "artifact":  "wiki/data/sniper_slope_replay.json",
        "summary":   "Mirror of gate 70 for the Sniper anchor sweep. Six (app, graph) cells (bfs/pr/sssp at cit-Patents and email-Eu-core); same verdict checks. On the current corpus, GRASP median = -7.64 pp/oct, LRU = -7.40, SRRIP = -7.96. SRRIP-GRASP gap = -0.32 pp/oct (gated, want <= 0); LRU-GRASP gap = +0.24 (INFORMATIONAL — sub-WSS at 4kB inverts the ordering). All verdict checks pass: cache monotonicity holds in every cell, every per-policy median negative, SRRIP <= GRASP, GRASP below help-floor (-1.0).",
    },
    {
        "id":        "cross_tool_slope_ordering",
        "label":     "Cross-tool SRRIP-vs-GRASP slope ordering",
        "generator": "scripts/experiments/ecg/cross_tool_slope_ordering.py",
        "gate":      "scripts/test/test_cross_tool_slope_ordering.py",
        "artifact":  "wiki/data/cross_tool_slope_ordering.json",
        "summary":   "Reads gate 66 (cache-sim capacity_sensitivity), gate 70 (gem5 anchor slope), and gate 71 (sniper anchor slope) per-policy medians and verifies the 'oracle-aware policies are less cache-hungry' claim is REPLICATED across all three tools: SRRIP median <= GRASP median in every tool, with at least 2 of 3 tools showing a strict gap >= 0.05 pp/octave. Current state: all 3/3 tools show strict steeper. Cache-sim SRRIP-GRASP = -0.95 pp/oct; gem5 = -1.25; sniper = -0.32. LRU-vs-GRASP delta is reported per tool but explicitly NOT gated — gates 70/71 documented that sub-WSS anchor scales invert the LRU>GRASP ordering observed at 1-8MB cache-sim sizes. Verdict PASS iff (1) all three tools' artifacts present and valid, (2) every tool shows SRRIP <= GRASP, (3) at least 2 of 3 tools show strict gap >= 0.05 pp/octave.",
    },
    {
        "id":        "per_app_srrip_vs_grasp",
        "label":     "Per-app SRRIP-vs-GRASP slope ordering",
        "generator": "scripts/experiments/ecg/per_app_srrip_vs_grasp.py",
        "gate":      "scripts/test/test_per_app_srrip_vs_grasp.py",
        "artifact":  "wiki/data/per_app_srrip_vs_grasp.json",
        "summary":   "Per-app companion to gate 72. Reads gate 68's per_app_capacity_slope artifact and verifies that for every non-pinned app, SRRIP median slope is no more than 1.0 pp/octave shallower than GRASP. Current state: bc -0.27, cc -0.71, pr -0.70, sssp -0.12 (all SRRIP steeper or near-tied). bfs is pinned as a known kernel deviation (+2.35 pp/oct shallower SRRIP) — same frontier-driven streaming pathology that gates 65/68 already document for LRU-vs-GRASP. Verdict PASS iff (1) no missing GRASP or SRRIP medians, (2) no NEW deviating apps beyond {bfs}, (3) at least one app has strictly steeper SRRIP, (4) majority of apps show steeper SRRIP (matches global gate 72 winner).",
    },
    {
        "id":        "cross_tool_lru_regime",
        "label":     "Cross-tool LRU-vs-GRASP regime inversion",
        "generator": "scripts/experiments/ecg/cross_tool_lru_regime.py",
        "gate":      "scripts/test/test_cross_tool_lru_regime.py",
        "artifact":  "wiki/data/cross_tool_lru_regime.json",
        "summary":   "Formalizes the regime-dependent LRU-vs-GRASP slope finding that gates 70/71/72 surfaced as INFORMATIONAL. cache-sim sweep (1-8MB, post-WSS) shows LRU strictly steeper than GRASP (-0.97 pp/oct); both anchor tools at sub-WSS scales (4kB-2MB) show the opposite ordering (gem5 +0.84, sniper +0.24). PASS confirms (1) cache-sim post-WSS LRU is steeper by at least 0.30 pp/oct, (2) both anchor tools sub-WSS show LRU at-most negligibly steeper (within 0.20 pp/oct slack), (3) the inversion sign holds across tools, (4) regime labels match documented L3 ranges. Cross-tool sign agreement between gem5 and sniper confirms the inversion is a physical regime effect (LRU's give-up-and-stream behaviour extracts nothing from extra capacity when no policy fits the hot set; GRASP's hold-the-hot-set behaviour still secures partial reuse) rather than a tool artifact.",
    },
    {
        "id":        "saturation_slope_extremum",
        "label":     "Per-app saturation-vs-slope extremum corroboration",
        "generator": "scripts/experiments/ecg/saturation_slope_extremum.py",
        "gate":      "scripts/test/test_saturation_slope_extremum.py",
        "artifact":  "wiki/data/saturation_slope_extremum.json",
        "summary":   "Reads gate 65 (saturation_distance per_app) and gate 68 (per_app_capacity_slope per-app medians) and verifies that bfs is the UNIQUE extremum on BOTH metrics: smallest 4MB->8MB distance AND shallowest OLS slope across the 1MB-8MB sweep. Pins the bfs-is-least-cache-sensitive finding with two independent measures. The most-cache-hungry app DISAGREES across metrics (sssp by slope -19.4 pp/oct; bc by distance +15.6 pp) — the regime-vs-aggregate distinction in action (distance captures upper-octave headroom while slope averages per-octave drop) — and is reported as INFORMATIONAL rather than gated. PASS iff (1) bfs is argmin(distance), (2) bfs is argmin(|slope|), (3) bfs is unique extremum on both, (4) at least one app has slope steepness >= 3x bfs, (5) at least one app has distance >= 2.5x bfs.",
    },
    {
        "id":        "cross_tool_slope_universality",
        "label":     "Cross-tool slope-sign universality",
        "generator": "scripts/experiments/ecg/cross_tool_slope_universality.py",
        "gate":      "scripts/test/test_cross_tool_slope_universality.py",
        "artifact":  "wiki/data/cross_tool_slope_universality.json",
        "summary":   "Roll-up invariant across all 10 (tool, policy) cells: cache-sim {GRASP,LRU,POPT,SRRIP} + gem5 {GRASP,LRU,SRRIP} + sniper {GRASP,LRU,SRRIP}. PASS iff (1) every (tool, policy) median capacity-sensitivity slope is negative (extra cache never makes a policy worse on average), (2) every median lies in the documented physical band [-25, -0.5] pp/oct (catches both runaway-steep regressions and near-zero collapse), (3) per-tool steepness span (max - min across policies) does not exceed 5 pp/oct (catches partial regressions where one policy on one tool stops scaling while siblings stay healthy). Centralizes the within-tool sign checks from gates 66/70/71 and adds two cross-tool roll-up invariants those gates cannot. All 10 cells in band: cache-sim [-15.62, -14.65], gem5 [-7.21, -5.12], sniper [-7.96, -7.40].",
    },
    {
        "id":        "monotonicity_universality",
        "label":     "L3-sweep monotonicity universality",
        "generator": "scripts/experiments/ecg/monotonicity_universality.py",
        "gate":      "scripts/test/test_monotonicity_universality.py",
        "artifact":  "wiki/data/monotonicity_universality.json",
        "summary":   "Cell-level cache-monotonicity guard. Reads the load-bearing cache-sim sweep (oracle_gap.json, 456 rows, 7-point L3 axis 4kB-8MB), groups by (graph, app, policy), and checks every consecutive L3 step. PASS iff (1) zero hard violations (no step has miss-rate increase >= 0.5 pp), (2) at most 10% of steps are bumps of any size, (3) the largest observed bump stays below 0.05 pp (current worst is 0.035 pp on email-Eu-core/pr/POPT 1MB->4MB). Current state: 136 cells, 320 steps, 14 bumps (4.4%), all bumps < 0.05 pp. Foundational soundness invariant that downstream slope/distance/sensitivity gates (65-68, 70-76) all assume.",
    },
    {
        "id":        "anchor_cell_census",
        "label":     "Anchor cell-pair census",
        "generator": "scripts/experiments/ecg/anchor_cell_census.py",
        "gate":      "scripts/test/test_anchor_cell_census.py",
        "artifact":  "wiki/data/anchor_cell_census.json",
        "summary":   "Pins gem5 and Sniper anchor coverage against silent shrinkage that would invalidate downstream cross-tool gates (70/71/72/74/76) without an obvious failure. gem5 baseline: 2 cells (email-Eu-core/bc, email-Eu-core/pr) x 3 policies (GRASP, LRU, SRRIP) x 4 L3 sizes (4kB, 32kB, 256kB, 2MB) = 6 records. Sniper baseline: 6 cells (cit-Patents and email-Eu-core, each x {bfs, pr, sssp}) x 3 policies x 4 L3 sizes = 18 records. Cross-anchor invariants: shared L3 axis, shared policy set, at least one shared (graph, app) cell (currently email-Eu-core/pr) so per-cell parity spot-checks have a foothold. PASS iff all 13 of these structural invariants hold simultaneously.",
    },
    {
        "id":        "family_saturation_distance",
        "label":     "Per-family saturation-distance replay",
        "generator": "scripts/experiments/ecg/family_saturation_distance.py",
        "gate":      "scripts/test/test_family_saturation_distance.py",
        "artifact":  "wiki/data/family_saturation_distance.json",
        "summary":   "Mirror of gate 67 for the saturation-distance metric (4MB->8MB miss-rate drop in pp). Aggregates gate 65's per_cell data by graph family on the high-WSS qualifying subset (pico sentinels dropped). PASS iff (1) every family has non-negative median distance (family-level dual of gate 77's cell-level monotonicity guard), (2) every family has min distance >= 0, (3) citation and social meet the 5 pp high-headroom floor, (4) web is pinned as the low-headroom outlier with median < 5 pp (web-Google is much closer to its saturation point at 4MB than the hub-heavy families), (5) family ordering citation >= social >= web holds within a 1 pp slack. Current state: citation median +15.69 pp, social +12.50 pp, web +2.15 pp. Makes the regime story explicit at the family level so a regression that flattens citation/social to web's level (or pushes web's median above the ceiling) is caught at exactly one named check.",
    },
    {
        "id":        "anchor_monotonicity_replay",
        "label":     "Anchor monotonicity replay (gem5+sniper)",
        "generator": "scripts/experiments/ecg/anchor_monotonicity_replay.py",
        "gate":      "scripts/test/test_anchor_monotonicity_replay.py",
        "artifact":  "wiki/data/anchor_monotonicity_replay.json",
        "summary":   "Cross-tool anchor-cell monotonicity sister of gate 77. Walks every (graph, app, policy) anchor cell in gem5_slope_replay.json and sniper_slope_replay.json and asserts that miss_pp_by_size is non-increasing as L3 grows, with TIER-AWARE per-tool tolerances locked to the current state: gem5 (high-fidelity) must be strictly monotone (0 bumps allowed in 18 steps across 6 cells); sniper (lower-fidelity, 18 cells x 3 steps = 54 steps) is permitted bounded bumps under three per-tool ceilings (bump_rate <= 40%, hard bumps (>=0.5 pp) <= 5, max bump < 2 pp). No tool may exhibit a catastrophic (>=3 pp) bump at any L3 step. Current state: gem5 = 0/18 bumps (0.0%); sniper = 19/54 bumps (35.2%), 2 hard, max +1.18 pp. Locks the simulator-quality divergence story --- gem5 anchors stay perfectly monotone, sniper noise stays bounded --- so any regression in either simulator's plumbing (broken policy, miswired cache sizing, fresh sniper jitter) trips exactly one named check.",
    },
    {
        "id":        "policy_steepness_ranking",
        "label":     "Per-policy final-octave steepness ranking",
        "generator": "scripts/experiments/ecg/policy_steepness_ranking.py",
        "gate":      "scripts/test/test_policy_steepness_ranking.py",
        "artifact":  "wiki/data/policy_steepness_ranking.json",
        "summary":   "Absolute-magnitude inversion of gate 64's saturation ranking. Aggregates |final-octave slope| (4MB->8MB) per policy across all 5 apps from cache_saturation_onset.json and locks the headline ordering POPT(0.10) < GRASP(0.23) << LRU(1.06) ~ SRRIP(1.09) pp/octave. Seven checks: (1) POPT median <= GRASP median, (2) GRASP median <= LRU median, (3) POPT median < SRRIP median, (4) oracle-aware (POPT/GRASP) medians <= 0.5 pp/oct ceiling, (5) non-oracle (LRU/SRRIP) medians >= 0.5 pp/oct floor, (6) oracle-aware median < half non-oracle median (currently 0.16 vs 1.07 = 0.15x), (7) POPT min slope <= 0.2 (at least one app fully saturates with POPT). Locks the 'oracle-aware policies are less cache-hungry' story as a quantitative ordering, complementing gate 64's binary onset count and gate 72's cross-tool slope-ordering check.",
    },
    {
        "id":        "anchor_cross_tool_agreement",
        "label":     "Cross-tool shared-anchor slope agreement",
        "generator": "scripts/experiments/ecg/anchor_cross_tool_agreement.py",
        "gate":      "scripts/test/test_anchor_cross_tool_agreement.py",
        "artifact":  "wiki/data/anchor_cross_tool_agreement.json",
        "summary":   "Cross-tool physical-replication lock for the (graph, app, policy) anchor cells present in BOTH gem5_slope_replay.json and sniper_slope_replay.json. Currently 3 shared cells: (email-Eu-core, pr) x {GRASP, LRU, SRRIP}. Five invariants: (1) shared-cell floor >= 3; (2) 100% slope-sign agreement across the two simulators; (3) 100% both-negative slopes (the 'bigger cache helps' invariant on the shared cells); (4) 100% sniper-steeper-than-gem5 magnitudes (sniper's wider L3 sweep + lower fidelity); (5) per-cell |sniper-gem5| <= 8 pp/oct ceiling. Current observation: gem5 = {-5.06, -2.88, -6.13}; sniper = {-7.61, -8.00, -8.00}; max|diff| = 5.13 pp/oct. Defends against a sniper or gem5 plumbing regression that would flip the slope sign on the shared cells (where 'L3 helps' becomes 'L3 hurts'), and against a future overhaul that shrinks the shared cell set below 3.",
    },
    {
        "id":        "wss_relative_l3",
        "label":     "WSS-relative L3 axis",
        "generator": "scripts/experiments/ecg/wss_relative_l3.py",
        "gate":      "scripts/test/test_wss_relative_l3.py",
        "artifact":  "wiki/data/wss_relative_l3.json",
        "summary":   "Bins each cell by L3/WSS ratio (under/near/over WSS). POPT has smallest mean gap in EVERY WSS regime (1.62/2.35/0.22 pp); GRASP wins by count in every regime. Defends against absolute-byte cross-graph comparison pushback.",
    },
    {
        "id":        "family_sensitivity",
        "label":     "Family-classification sensitivity",
        "generator": "scripts/experiments/ecg/family_sensitivity.py",
        "gate":      "scripts/test/test_family_sensitivity.py",
        "artifact":  "wiki/data/family_sensitivity.json",
        "summary":   "Enumerates every (graph, alternative family) relabeling (32 total) and reruns the 7 sign-stability claims. Bedrock: POPT < LRU on social survives 32/32 relabelings. Road claim only loses stability when roadNet-CA (sole road member) is relocated, as expected.",
    },
    {
        "id":        "reproduce_smoke",
        "label":     "Reproducibility smoke",
        "generator": "scripts/experiments/ecg/reproduce_smoke.py",
        "gate":      "scripts/test/test_reproduce_smoke.py",
        "artifact":  "wiki/data/reproduce_smoke.json",
        "summary":   "SHA-256-snapshots every tracked wiki/data aggregator, re-runs lit-claims + lit-catalog, asserts byte-identity (modulo declared volatile fields like dashboard runtime_s). Catches silent staleness across the 40-file artifact surface.",
    },
    {
        "id":        "winning_regime_taxonomy",
        "label":     "Winning-regime taxonomy",
        "generator": "scripts/experiments/ecg/winning_regime_taxonomy.py",
        "gate":      "scripts/test/test_winning_regime_taxonomy.py",
        "artifact":  "wiki/data/winning_regime_taxonomy.json",
        "summary":   "(graph_family × L3 regime) winner matrix with auto-extracted ≥80 % dominance rules. Headline rules: mesh/all → POPT 100 %; road/large → LRU 75 %.",
    },
    # ---------- Edge regime + diagnostics ----------
    {
        "id":        "small_l3_thrash",
        "label":     "Small-L3 thrash report",
        "generator": "scripts/experiments/ecg/small_l3_thrash_report.py",
        "gate":      "scripts/test/test_small_l3_thrash.py",
        "artifact":  "wiki/data/small_l3_thrash.json",
        "summary":   "4 kB-L3 sweep (9 (graph, app) cells × 9 policy variants). LRU wins 5/9 cells; GRASP regresses up to +35.857 pp vs LRU on soc-LiveJournal1/bfs.",
    },
    {
        "id":        "literature_deviations",
        "label":     "Literature deviations inventory",
        "generator": "scripts/experiments/ecg/literature_deviations_report.py",
        "gate":      "scripts/test/test_literature_deviations.py",
        "artifact":  "wiki/data/literature_deviations.json",
        "summary":   "Closed-vocab mechanism classifier for known_deviation rows: 30/30 classify as popt_overhead_dominates — the exact inverse of road-graph finding.",
    },
    {
        "id":        "claim_density",
        "label":     "Per-graph claim density",
        "generator": "scripts/experiments/ecg/claim_density_report.py",
        "gate":      "scripts/test/test_claim_density.py",
        "artifact":  "wiki/data/claim_density.json",
        "summary":   "Per-graph literature claim density (8 graphs, 320 claims, 288 OK = 90.0 %). Density per graph: 2 (delaunay_n19) → 12 (cit-Patents).",
    },
    {
        "id":        "bootstrap_ci",
        "label":     "Bootstrap CIs on load-bearing claims",
        "generator": "scripts/experiments/ecg/bootstrap_ci.py",
        "gate":      "scripts/test/test_bootstrap_ci.py",
        "artifact":  "wiki/data/bootstrap_ci.json",
        "summary":   "Percentile bootstrap (5000 resamples, seed 1729) on every (policy, family) and (policy, regime) oracle-gap bucket + paired ΔPOPT−GRASP per family + sign-stability fractions. Road POPT < GRASP survives 97.6 % of resamples; social/citation/web do not.",
    },
    # ---------- Meta artifacts ----------
    {
        "id":        "paper_claims",
        "label":     "Paper claims registry",
        "generator": "scripts/experiments/ecg/paper_claims_registry.py",
        "gate":      "scripts/test/test_paper_claims_registry.py",
        "artifact":  "wiki/data/paper_claims.json",
        "summary":   "Single source of truth for every numerical claim the paper makes (14 claims across 8 categories), each linked to its source artifact + governing gate.",
    },
    {
        "id":        "confidence_dashboard",
        "label":     "Confidence dashboard",
        "generator": "scripts/experiments/ecg/confidence_dashboard.py",
        "gate":      "scripts/test/test_confidence_dashboard.py",
        "artifact":  "wiki/data/confidence_dashboard.json",
        "summary":   "Single-screen verdict (208 gates today, all GREEN). The dashboard this catalog sits next to.",
    },
]


def _audit(entries: list[dict]) -> list[dict]:
    """Annotate each entry with on-disk presence flags so the gate
    can verify nothing has gone stale."""
    out: list[dict] = []
    for e in entries:
        gen = REPO_ROOT / e["generator"]
        gate = REPO_ROOT / e["gate"]
        art = REPO_ROOT / e["artifact"]
        out.append({
            **e,
            "generator_exists": gen.exists(),
            "gate_exists":      gate.exists(),
            "artifact_exists":  art.exists(),
        })
    return out


def _write_json(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_entries":             len(entries),
        "missing_generators":    [e["id"] for e in entries if not e["generator_exists"]],
        "missing_gates":         [e["id"] for e in entries if not e["gate_exists"]],
        "missing_artifacts":     [e["id"] for e in entries if not e["artifact_exists"]],
    }
    path.write_text(json.dumps({
        "summary":  summary,
        "entries":  entries,
    }, indent=2, sort_keys=True))


def _write_md(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Paper-artifact catalog")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/artifact_catalog.py`. "
        "This is the single canonical index reviewers should consult "
        "to trace every paper claim back to its source artifact + "
        "governing pytest gate._"
    )
    lines.append("")
    n = len(entries)
    missing_gen = sum(1 for e in entries if not e["generator_exists"])
    missing_gate = sum(1 for e in entries if not e["gate_exists"])
    missing_art = sum(1 for e in entries if not e["artifact_exists"])
    lines.append(
        f"**{n} entries.** Missing generators: {missing_gen}; "
        f"missing gates: {missing_gate}; "
        f"missing artifacts: {missing_art}."
    )
    lines.append("")
    lines.append("| # | id | label | summary |")
    lines.append("|---:|---|---|---|")
    for i, e in enumerate(entries, 1):
        lines.append(f"| {i} | `{e['id']}` | {e['label']} | {e['summary']} |")
    lines.append("")
    lines.append("## Source chain per entry")
    lines.append("")
    for e in entries:
        lines.append(f"### `{e['id']}` — {e['label']}")
        lines.append("")
        lines.append(f"- **Generator:** `{e['generator']}`"
                     + ("" if e["generator_exists"] else "  **❌ MISSING**"))
        lines.append(f"- **Gate:** `{e['gate']}`"
                     + ("" if e["gate_exists"] else "  **❌ MISSING**"))
        lines.append(f"- **Artifact:** `{e['artifact']}`"
                     + ("" if e["artifact_exists"] else "  **❌ MISSING**"))
        lines.append(f"- **Headline:** {e['summary']}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "artifact_catalog.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "artifact_catalog.md",
    )
    args = parser.parse_args()

    entries = _audit(CATALOG)
    _write_json(entries, args.json_out)
    _write_md(entries, args.md_out)
    missing_gen = [e["id"] for e in entries if not e["generator_exists"]]
    missing_gate = [e["id"] for e in entries if not e["gate_exists"]]
    missing_art = [e["id"] for e in entries if not e["artifact_exists"]]
    print(
        f"[catalog] {len(entries)} entries; "
        f"missing gen={missing_gen} gate={missing_gate} art={missing_art}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
