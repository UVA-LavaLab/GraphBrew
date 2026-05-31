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
        "id":        "lit_faith_diversity",
        "label":     "Literature-faithfulness diversity audit",
        "generator": "scripts/experiments/ecg/lit_faith_diversity.py",
        "gate":      "scripts/test/test_lit_faith_diversity.py",
        "artifact":  "wiki/data/lit_faith_diversity.json",
        "summary":   "Coverage cube of literature-faithfulness claims by (family × app × L3 × paper × policy) plus cross-paper triangulation cells; locks per-axis breadth floors so a regen cannot silently drop a graph family or a cited paper.",
    },
    {
        "id":        "lit_faith_margin",
        "label":     "Literature-faithfulness margin audit",
        "generator": "scripts/experiments/ecg/lit_faith_margin.py",
        "gate":      "scripts/test/test_lit_faith_margin.py",
        "artifact":  "wiki/data/lit_faith_margin.json",
        "summary":   "Per-claim distance to the nearest disagree boundary (median ~5.5 pp, 47 fragile cells under 1 pp today). Caps the fragile-cell count and locks per-family / per-status median floors so corpus drift cannot silently erode confidence in lit-faith verdicts.",
    },
    {
        "id":        "lit_faith_signmass",
        "label":     "Literature-faithfulness sign-mass concentration",
        "generator": "scripts/experiments/ecg/lit_faith_signmass.py",
        "gate":      "scripts/test/test_lit_faith_signmass.py",
        "artifact":  "wiki/data/lit_faith_signmass.json",
        "summary":   "Per-bucket (expected_sign × policy) sign-mass concentration with Wilson 95 % lower bounds and binomial sign-test p-values. Today: GRASP 13/13 correctly signed (p=0.0002, median -4.83 pp), POPT 7/7 (p=0.0156, median -10.74 pp), POPT≥GRASP 55/88 with 2 ties (p=0.0127, median -0.33 pp). Locks per-bucket Wilson LB / binomial-p / median-delta floors so a regression erasing the sign signal trips the gate.",
    },
    {
        "id":        "lit_faith_citations",
        "label":     "Literature-faithfulness citation locator integrity",
        "generator": "scripts/experiments/ecg/lit_faith_citations.py",
        "gate":      "scripts/test/test_lit_faith_citations.py",
        "artifact":  "wiki/data/lit_faith_citations.json",
        "summary":   "Bijection between the 15 unique citation strings in lit-faith and `literature_baselines.py` (zero orphans / unused), per-anchor inventory across the 3 canonical papers (Faldu HPCA 2020, Balaji & Lucia HPCA 2021, Jaleel ISCA 2010), and per-citation well-formedness check (venue tag + year + §/Fig locator + known anchor). All 15 citations well-formed today; 2/3 anchors have DOI/URL in module docstring (Jaleel ISCA 2010 intentionally unlinked).",
    },
    {
        "id":        "lit_faith_deviations",
        "label":     "Literature-faithfulness known-deviation completeness",
        "generator": "scripts/experiments/ecg/lit_faith_deviations.py",
        "gate":      "scripts/test/test_lit_faith_deviations.py",
        "artifact":  "wiki/data/lit_faith_deviations.json",
        "summary":   "Completeness audit on `literature_baselines.KNOWN_DEVIATIONS` (the whitelist that downgrades live lit-faith `disagree` rows to `known_deviation`). All 34 entries today are well-formed: ≥ 80-char reason, contain a quantitative phrase (pp / MB / %), and mention at least one anchor (paper §, design term, or algorithmic root-cause vocabulary like `PR-rank`, `frontier`, `union-find`, `Phase 1`). Bijection enforced: zero orphan entries, zero live `known_deviation` rows without a documented explanation. Coverage: 2 policies × 5 graphs × 5 apps × 3 L3 sizes; CC + BC dominate (82 % of KDs — they're the apps whose algorithmic mismatch with POPT's PR ranking is most documented).",
    },
    {
        "id":        "lit_faith_tolerance",
        "label":     "Literature-faithfulness tolerance calibration",
        "generator": "scripts/experiments/ecg/lit_faith_tolerance.py",
        "gate":      "scripts/test/test_lit_faith_tolerance.py",
        "artifact":  "wiki/data/lit_faith_tolerance.json",
        "summary":   "For every literature claim whose comparator-asserted bound actually fires (202 rows today; 30 known_deviations and 98 BIG_GAP non-triggered rows excluded), computes the per-row `slack_pp` = how many pp the observed `|delta_pct|` could move toward the disagree boundary. Corpus median slack 4.94 pp; 16 fragile rows (slack < 1 pp, 7.9 %), all but one in POPT_GE_GRASP (the tightest bucket). Strict policies (GRASP, POPT, SRRIP) have zero fragile rows and min slack ≥ 1.30 pp. Negative-slack count = 0 (slack formula matches the classifier branches). Histogram + per-policy + per-app aggregates + top-15 most-fragile list emitted for reviewer triage.",
    },
    {
        "id":        "lit_faith_accesses",
        "label":     "Literature-faithfulness accesses-floor audit",
        "generator": "scripts/experiments/ecg/lit_faith_accesses.py",
        "gate":      "scripts/test/test_lit_faith_accesses.py",
        "artifact":  "wiki/data/lit_faith_accesses.json",
        "summary":   "Warmup-noise guard for the lit-faith corpus. For every (graph, app, policy, l3) row, audits the `accesses` count emitted by cache_sim and floors it against per-app thresholds — 1M for BC/CC/PR, 500k for BFS/SSSP on production graphs, with a separate looser table for the email-Eu-core dev-smoke (20k bfs / 200k pr / 2M bc). Today: 311 production rows + 19 smoke rows, production min 735,934, median 15.95M, zero floor violations. Production buckets: 0 tiny, ~25 % large+huge — well above the 25 % gate floor. Catches silent regressions where a workload silently truncates to a warmup-only trace.",
    },
    {
        "id":        "lit_faith_citexapp",
        "label":     "Literature-faithfulness cross-app rationale coherence",
        "generator": "scripts/experiments/ecg/lit_faith_citexapp.py",
        "gate":      "scripts/test/test_lit_faith_citexapp.py",
        "artifact":  "wiki/data/lit_faith_citexapp.json",
        "summary":   "For every (citation, expected_sign) group in the lit-faith corpus (17 groups today across 5 papers), audits the per-cell rationales for mutual coherence. Four checks: (1) **zero contradictions** — no rationale pair within a group may mix opposing-direction vocabulary (e.g., \"dominates\" vs \"underperforms\"), with negation-context handling so phrasings like \"must NOT regress\" don't falsely trigger; (2) **sign-vocabulary alignment** — every rationale must mention at least one term consistent with its `expected_sign` band; (3) **common-kernel** — every multi-rationale group shares at least one anchor token across all member rationales (large groups ≥ 5 rationales: ≥ 2 kernel terms); (4) **length-span** — `max_len/min_len ≤ 3.0` per group. Today: 0 contradictions, 0 sign misses, 0 kernel failures, 0 span failures across 35 unique rationales.",
    },
    {
        "id":        "lit_faith_monotonicity",
        "label":     "Literature-faithfulness cache-size monotonicity audit",
        "generator": "scripts/experiments/ecg/lit_faith_monotonicity.py",
        "gate":      "scripts/test/test_lit_faith_monotonicity.py",
        "artifact":  "wiki/data/lit_faith_monotonicity.json",
        "summary":   "Locks the physical invariant that miss rate is non-increasing in LLC size for every (graph, app, policy) triple in the lit-faith corpus (tolerance 0.5 pp). For every triple with ≥ 2 L3 samples, computes the slope per log2(L3) doubling and flags any pair where the larger cache shows ≥ 0.5 pp higher miss rate. Also flags saturated triples (< 1 pp total drop across the full sweep) — those samples sit in the capacity-bound regime where the policy can't move the needle. Today: 30 triples audited across 17 graphs / 4 apps / {GRASP, SRRIP}, 0 LRU violations, 0 policy violations, 1 saturated triple (com-orkut/bfs/SRRIP at 1MB→8MB — expected, bfs on orkut is capacity-bound below 16MB). Median slope ≈ 0.16 miss-rate-points per L3 doubling.",
    },
    {
        "id":        "lit_faith_stat",
        "label":     "Literature-faithfulness statistical-sanity audit",
        "generator": "scripts/experiments/ecg/lit_faith_stat.py",
        "gate":      "scripts/test/test_lit_faith_stat.py",
        "artifact":  "wiki/data/lit_faith_stat.json",
        "summary":   "Re-derives `delta_pct` from the two miss-rate columns each per_claim row compares (LRU-vs-policy, POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP) and locks: zero NaN/inf, zero out-of-bounds miss rates, zero delta-rounding mismatches (> 0.001 pp), zero sign flips (> 0.01 pp noise floor), zero signed-delta inconsistencies, zero unknown row kinds, zero bad status labels, zero status-vs-delta inconsistencies (with the POPT_NEAR phase-transition-regime exception folded in: assertion only fires when grasp_gain_vs_lru > 10 pp AND POPT is worse than GRASP). Locks status vocabulary to {ok, within_tolerance, disagree, known_deviation, missing, insufficient_data}. Today: 330 rows, 102 LRU-vs-policy, 114 POPT_GE_GRASP, 114 POPT_NEAR_GRASP, 298 ok / 30 known_deviation / 2 within_tolerance / 0 disagree.",
    },
    {
        "id":        "lit_faith_polyord",
        "label":     "Literature-faithfulness policy-ordering audit",
        "generator": "scripts/experiments/ecg/lit_faith_polyord.py",
        "gate":      "scripts/test/test_lit_faith_polyord.py",
        "artifact":  "wiki/data/lit_faith_polyord.json",
        "summary":   "Per (graph_family x app) bucket: hub-bearing families (social/citation/web) must respect the literature ordering POPT/GRASP ≤ LRU (median POPT − LRU ≤ +0.5 pp, median GRASP − LRU ≤ +1.0 pp, POPT improve-frac ≥ 0.50 when n ≥ 5), while hub-less families (road/mesh) are documented exceptions and only assert classification stability. A per-app global hub-aggregate floor of 0.55 catches corpus shifts toward weakly-improving cells. Today: 114 cells across 21 (family,app) buckets (15 hub, 6 no-hub), per-app hub-aggregate POPT improve-frac runs 0.72 (bc) → 0.94 (pr), 0 violations.",
    },
    {
        "id":        "lit_faith_devexp",
        "label":     "Literature-faithfulness deviation-explanation audit",
        "generator": "scripts/experiments/ecg/lit_faith_devexp.py",
        "gate":      "scripts/test/test_lit_faith_devexp.py",
        "artifact":  "wiki/data/lit_faith_devexp.json",
        "summary":   "Per `status == 'known_deviation'` row: the `known_deviation_reason` text must name at least one algorithmic mechanism (PR-rank, frontier, hub, union-find, ordering, capacity, look-ahead, ...), exceed a 60-char length floor, carry a non-empty citation, resolve any cross-references against another known_deviation row, and the same reason text may not cover more than 50 % of all known_deviation rows. Today: 30 known_deviation rows, median reason length 247 chars, median mechanism hits 5, 30/30 unique texts, 16 cross-referenced, 0 violations.",
    },
    {
        "id":        "lit_faith_ratgrid",
        "label":     "Literature-faithfulness rationale-grid audit",
        "generator": "scripts/experiments/ecg/lit_faith_ratgrid.py",
        "gate":      "scripts/test/test_lit_faith_ratgrid.py",
        "artifact":  "wiki/data/lit_faith_ratgrid.json",
        "summary":   "Per (policy, graph, app) cell: rationale text must be unique within the cell (theorem-class policies POPT_GE_GRASP / POPT_NEAR_GRASP_IF_BIG_GAP / SRRIP must carry exactly 1 rationale per (policy, app); point policies GRASP / POPT / LRU may carry up to 2 per (policy, graph, app) to accommodate L3-regime variants). Theorem-class policies are exempt from the citation-token rule (they state algorithmic-class theorems, not Fig references). Point-policy rationales must contain a citation token (HPCA20 / HPCA21 / Fig 9-12 / §3-7). Today: 330 rows, 115 cells, 5 policies, GRASP=19 / POPT=8 / POPT_GE_GRASP=5 / POPT_NEAR=1 / SRRIP=5 distinct rationales, 0 violations.",
    },
    {
        "id":        "lit_faith_cellcomp",
        "label":     "Literature-faithfulness cell-completeness audit",
        "generator": "scripts/experiments/ecg/lit_faith_cellcomp.py",
        "gate":      "scripts/test/test_lit_faith_cellcomp.py",
        "artifact":  "wiki/data/lit_faith_cellcomp.json",
        "summary":   "Per (graph, app, l3) cell in the per_observation table: canonical policy roster {LRU, GRASP, POPT} must be present, LRU baseline row must exist, `delta_vs_lru_pct` must equal `(miss_rate - lru_miss_rate) * 100` within 0.001 pp, every non-LRU policy must cover at least 3 L3 sizes per (graph, app), and every present policy must share the same L3 axis within (graph, app). Today: 456 per_observation rows, 114 cells, 8 graphs, 5 apps, 4 policies (LRU + GRASP + POPT + SRRIP), 0 violations.",
    },
    {
        "id":        "lit_faith_appfreq",
        "label":     "Literature-faithfulness app-frequency audit",
        "generator": "scripts/experiments/ecg/lit_faith_appfreq.py",
        "gate":      "scripts/test/test_lit_faith_appfreq.py",
        "artifact":  "wiki/data/lit_faith_appfreq.json",
        "summary":   "Per-app axis-coverage check on per_observation: each of the 5 apps must touch >= 6 graphs / >= 3 L3 sizes / >= 3 policies (canonical roster {LRU, GRASP, POPT} per app) / >= 60 rows, every (app, graph) pair must cover >= 3 L3 sizes, and the anchor app (pr) must cover the full corpus. Today: 5 apps, 8 corpus graphs; pr=8 graphs/112 rows (full sweep), bc=bfs=7 graphs/92 rows, cc=sssp=6 graphs/80 rows. 0 violations.",
    },
    {
        "id":        "lit_faith_regimesign",
        "label":     "Literature-faithfulness regime-sign audit",
        "generator": "scripts/experiments/ecg/lit_faith_regimesign.py",
        "gate":      "scripts/test/test_lit_faith_regimesign.py",
        "artifact":  "wiki/data/lit_faith_regimesign.json",
        "summary":   "Per (graph_family, app, advice-policy) bucket on per_observation: regime-aware sign-tally + magnitude ceiling. Hub families {social, citation, web} must not show majority regression (pos_cells > neg_cells AND median > +0.5 pp simultaneously) and bucket median must be <= +0.5 pp. No-hub families {road, mesh} may exhibit L-curve sign-flipping but the bucket median must stay within ±8 pp. No individual cell may exceed |delta_vs_lru| > 80 pp. Today: 72 buckets (45 hub + 18 no-hub), 0 extreme cells (worst is roadNet-CA / sssp / 1MB / GRASP = +70.12 pp), 0 violations across all 4 rules.",
    },
    {
        "id":        "lit_faith_citdate",
        "label":     "Literature-faithfulness citation/date audit",
        "generator": "scripts/experiments/ecg/lit_faith_citdate.py",
        "gate":      "scripts/test/test_lit_faith_citdate.py",
        "artifact":  "wiki/data/lit_faith_citdate.json",
        "summary":   "Per_claim citation-field structural audit. Every citation must (D1) be non-empty, (D2) parse to (author, venue, year), (D3) name a top-tier architecture venue {HPCA, ISCA, MICRO, ASPLOS, SC}, (D4) name a year in [2005, 2026], (D5) reference the policy's originator publication (GRASP→Faldu HPCA 2020, POPT[+derived]→Balaji & Lucia HPCA 2021, SRRIP→Jaleel ISCA 2010) OR be an explicit cross-attribution where the policy name appears in the citation string, (D6) contain a locator (§N | Fig N | Tab N | Section N), and (D7) the corpus as a whole must use ≥ 10 distinct citation strings. Today: 330 rows, 100% parsed, 15 distinct citations, venues={HPCA, ISCA}, years∈{2010, 2020, 2021}, 0 violations.",
    },
    {
        "id":        "lit_faith_ecg_parity",
        "label":     "ECG substrate-parity audit",
        "generator": "scripts/experiments/ecg/lit_faith_ecg_parity.py",
        "gate":      "scripts/test/test_lit_faith_ecg_parity.py",
        "artifact":  "wiki/data/lit_faith_ecg_parity.json",
        "summary":   "ECG cache_sim component-proof matrix faithfulness audit (gate 238). (E1) every required ablation {LRU, SRRIP, GRASP_DBG, POPT, ECG_DBG_only, ECG_POPT_primary, PFX_degree_only, PFX_POPT_only, DBG_PFX, POPT_PFX} present with status=ok per benchmark. (E2) |miss_rate(ECG_DBG_only) - miss_rate(GRASP_DBG_only)| ≤ 5e-4 per benchmark. (E3) |miss_rate(ECG_POPT_primary) - miss_rate(POPT_only)| ≤ 5e-4 per benchmark. (E4) every PFX ablation has ecg_runtime_issued ≥ 1 per benchmark. (E5) PFX ablations have prefetch_useful > 0 on PR AND prefetch_useful ≤ prefetch_requests on every benchmark. (E6) ecg_pfx_encoded ≤ ecg_pfx_candidates and every PFX counter ≥ 0. (E7) baselines have memory_accesses > 0 and l3_misses > 0. (E8) distinct backend count ≥ 1. Today: 54 observations on email-Eu-core × {pr, bfs, sssp} × 18 ablations, DBG parity = 0.0 / 0.0 / 0.0 (exact-bitwise), POPT parity = 0.0 / 2.45e-4 / 0.0, PFX issued ∈ [20, 36642], PR useful ≥ 625 across PFX_*/DBG_PFX/POPT_PFX, 0 violations. Confidence floor before any cluster-scale ECG sweep.",
    },
    {
        "id":        "lit_faith_ecg_gem5_parity",
        "label":     "ECG substrate-parity audit (gem5)",
        "generator": "scripts/experiments/ecg/lit_faith_ecg_gem5_parity.py",
        "gate":      "scripts/test/test_lit_faith_ecg_gem5_parity.py",
        "artifact":  "wiki/data/lit_faith_ecg_gem5_parity.json",
        "summary":   "ECG substrate-parity audit on gem5 (gate 239) — POPT-arm only. (G1) every required policy {LRU, POPT, ECG_POPT_PRIMARY} present with status=ok per (benchmark, section, L3) cell. (G2) |miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)| ≤ 2e-3 per cell (2× headroom over observed gem5 drift max 1.09e-3, looser than cache_sim's 5e-4 to absorb timing noise). (G3) backend=gem5 and simulator=gem5 on every row (no silent cache_sim ingestion). (G4) sim_ticks ≥ 1 and ipc > 0 on every row. (G5) LRU baseline has l3_accesses > 0 and l3_misses > 0 on every cell. (G6) l3_misses ≤ l3_accesses and l3_miss_rate ∈ [0, 1] everywhere. (G7) ≥ 2 distinct sections (cold-start + re-warmed) present. Today: 12 observations on email-Eu-core PR × L3∈{16MB, 32MB} × sections∈{1,2}, POPT-arm drift = 1.09e-3 (worst) / 3.21e-4 (best), 0 violations. Out-of-scope: DBG arm (needs ECG_DBG gem5 run), PFX activation (prefetcher=none everywhere), DROPLET comparison.",
    },
    {
        "id":        "lit_faith_ecg_sniper_parity",
        "label":     "ECG substrate-parity audit (Sniper)",
        "generator": "scripts/experiments/ecg/lit_faith_ecg_sniper_parity.py",
        "gate":      "scripts/test/test_lit_faith_ecg_sniper_parity.py",
        "artifact":  "wiki/data/lit_faith_ecg_sniper_parity.json",
        "summary":   "ECG substrate-parity audit on Sniper (gate 240) — SCAFFOLD/DEFERRED today. No matched-proof Sniper ECG sweep is available yet (/tmp/graphbrew-grasp-sniper-sweep provides LRU/SRRIP/GRASP only, no ECG_DBG_ONLY or ECG_POPT_PRIMARY rows). The postfix declares status=deferred with the expected source pattern (/tmp/graphbrew-ecg-sniper-matched-proof-*/...) and minimum observation floor (6). Audit logic is implemented end-to-end with 9 rules — when activated: (G1) roster {LRU, POPT, ECG_POPT_PRIMARY} per cell; (G1b) GRASP paired with ECG_DBG_ONLY when DBG arm present; (G2) |miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)| ≤ 2e-3 (mirrors gem5); (G2b) optional DBG parity at ε=2e-3; (G3) backend=sniper and simulator=sniper; (G4) ipc > 0 and instructions ≥ 1; (G5) LRU baseline non-zero; (G6) L3 hierarchy sanity; (G7) observation-floor from postfix. Activates automatically when ecg_sniper_parity_postfix.json transitions to status=active.",
    },
    {
        "id":        "lit_faith_ecg_pfx_vs_droplet",
        "label":     "ECG PFX vs DROPLET head-to-head",
        "generator": "scripts/experiments/ecg/lit_faith_ecg_pfx_vs_droplet.py",
        "gate":      "scripts/test/test_lit_faith_ecg_pfx_vs_droplet.py",
        "artifact":  "wiki/data/lit_faith_ecg_pfx_vs_droplet.json",
        "summary":   "ECG PFX prefetcher vs DROPLET head-to-head audit (gate 241) — SCAFFOLD/DEFERRED today. No matched-proof ECG_PFX vs DROPLET sweep is available: across /tmp/graphbrew-* the droplet_*/ecg_pfx_* columns are config-only (degrees, table sizes, delivery mode) — all runtime counters (droplet_indirect_issued, droplet_stride_issued, ecg_pfx_issued, ecg_pfx_useful) are zero. Postfix declares status=deferred with expected source pattern (/tmp/graphbrew-ecg-pfx-vs-droplet-*/...) and minimum 8 observations; arms {LRU, DROPLET, ECG_PFX}. When activated the audit emits 6 rules: (G1) arm completeness per cell; (G2) baseline neutrality |miss_rate(LRU) - miss_rate(LRU,paired)| ≤ 0.005 across pairs; (G3) useful-prefetch floor ecg_useful_frac ≥ 0.05 and droplet_useful_frac ≥ 0.05 (anything lower is essentially random); (G4) per-arm policy_label hygiene; (G5) backend identity stable across arms within a cell; (G6) observation-floor from postfix. Sibling family to the substrate-parity trinity (gates 238/239/240) but a different concern: not 'ECG mode ≡ stock mode on the same backend?' but 'ECG PFX beats/matches DROPLET on the same baseline?'.",
    },
    {
        "id":        "lit_faith_paper_label_map",
        "label":     "Paper label-map integrity",
        "generator": "scripts/experiments/ecg/lit_faith_paper_label_map.py",
        "gate":      "scripts/test/test_lit_faith_paper_label_map.py",
        "artifact":  "wiki/data/lit_faith_paper_label_map.json",
        "summary":   "Paper label-map integrity audit (gate 242) — ALWAYS ACTIVE (no deferred mode; source-of-truth is the code, not a curated fixture). Audits the canonical POLICY_LABELS / POLICY_DESCRIPTIONS / POLICY_COLORS triple in paper_pipeline.py against (a) the committed paper_pipeline_*/policy_label_map.csv (byte-for-byte equality), (b) every tracked source artifact (8 JSON files: 4 ECG postfix gates + 2 lit-faith ECG audits + literature_faithfulness_postfix + policy_winner_table) plus 5 paper_pipeline CSVs (roi_matrix_all, roi_policy_summary, ecg_mode_overhead_summary, popt_storage_overhead_summary, popt_charged_overhead). 5 rules: (G1) every POLICY_LABELS key has matching description and color (no partial additions); (G2) every figure_label is unique across the map (no collisions); (G3) every policy_label found in tracked sources is in POLICY_LABELS or in NON_PAPER_LABELS allowlist (CACHE rollup label, DROPLET / ECG_PFX prefetcher-arm names, POPT_GE_GRASP / POPT_NEAR_GRASP_IF_BIG_GAP theorem-class virtual labels); (G4) latest paper_pipeline_*/policy_label_map.csv matches code byte-for-byte (catches stale committed snapshot); (G5) every POLICY_LABELS key appears in at least one tracked source (no orphan paper labels). Catches 'added a new policy without updating the paper legend' regressions. Today: 9 labels, 13 sources, 0 violations.",
    },
    {
        "id":        "lit_faith_color_distinguishability",
        "label":     "POLICY_COLORS perceptual distinguishability",
        "generator": "scripts/experiments/ecg/lit_faith_color_distinguishability.py",
        "gate":      "scripts/test/test_lit_faith_color_distinguishability.py",
        "artifact":  "wiki/data/lit_faith_color_distinguishability.json",
        "summary":   "POLICY_COLORS perceptual-distinguishability audit (gate 243) — ALWAYS ACTIVE (no deferred mode; source-of-truth is paper_pipeline.py POLICY_COLORS / POLICY_HATCHES dicts loaded via importlib). Companion to gate 242: where 242 audits the policy vocabulary, 243 audits the visual quality of the palette — can a reader (or a B&W printer) actually tell the policies apart on the paper figures? 6 rules: (C1) every POLICY_LABELS key has a well-formed 7-char hex color; (C2) no two POLICY_COLORS values are exactly equal; (C3) every pair has CIE76 ΔE ≥ 12 in CIE Lab (D65) — visibly distinguishable on color print; (C4) pairs with CIE Lab lightness delta ΔL < 10 must use a POLICY_HATCHES entry, modulo the ACKNOWLEDGED_BW_PAIRS allowlist (10 grandfathered close-lightness pairs, each with rationale ≥ 60 chars) — keeps the B&W-printable contract honest while documenting current state; (C5) every color has ΔE ≥ 18 from #FFFFFF (no near-invisible policies on a white page); (C6) POLICY_HATCHES keys are a subset of POLICY_LABELS keys (no orphan hatches). Pure-stdlib sRGB → CIE Lab implementation (no third-party deps). Today: 9 colors, 36 pairs, 0 violations.",
    },
    {
        "id":        "lit_faith_paper_snapshot",
        "label":     "Paper-figure data snapshot integrity",
        "generator": "scripts/experiments/ecg/lit_faith_paper_snapshot.py",
        "gate":      "scripts/test/test_lit_faith_paper_snapshot.py",
        "artifact":  "wiki/data/lit_faith_paper_snapshot.json",
        "summary":   "Paper-figure data snapshot integrity audit (gate 244) — ALWAYS ACTIVE (no deferred mode; source-of-truth is paper_pipeline.py POLICY_LABELS loaded via importlib + the committed wiki/data/paper_pipeline_YYYYMMDD/ snapshot dir itself). Third gate in the always-active paper-snapshot trio: 242 audits the policy vocabulary, 243 audits the visual quality of the palette, and 244 audits the actual figure-data snapshot directory. 6 rules: (F1) exactly one paper_pipeline_YYYYMMDD/ dir exists in wiki/data (no stale duplicates that confuse readers or break gate 242's latest-dir lookup); (F2) the snapshot dir name parses to a valid YYYYMMDD date AND is within MAX_SNAPSHOT_AGE_DAYS (365 today; tighter later); (F3) every row in roi_matrix_all.csv has non-empty values for pipeline_source_csv, pipeline_run_dir, and pipeline_run_name — full referential provenance so anyone can re-run the source; (F4) single-run cohesion — every row shares the same pipeline_run_dir, ruling out Frankenstein snapshots stitched from multiple runs; (F5) coverage rectangle — per (benchmark, graph, l3_size) cell the set of policy_labels equals the canonical POLICY_LABELS palette (no missing/extra bars in any paper bar chart); (F6) value hygiene — l3_miss_rate ∈ [0.0, 1.0] universally, and total_accesses ≥ 1 for HIGH_ACTIVITY_BENCHMARKS = {'pr'} (BFS/SSSP can legitimately log total_accesses=0 on short-walk ROIs and are carved out with documented rationale). Today: 1 snapshot dir (paper_pipeline_20260528), 108 rows × 9 policies × 4 graphs × 3 benchmarks × 1 L3-size, 0 violations.",
    },
    {
        "id":        "lit_faith_regime_classifier",
        "label":     "L3 regime-classifier consistency",
        "generator": "scripts/experiments/ecg/lit_faith_regime_classifier.py",
        "gate":      "scripts/test/test_lit_faith_regime_classifier.py",
        "artifact":  "wiki/data/lit_faith_regime_classifier.json",
        "summary":   "L3 regime-classifier consistency audit (gate 245) — ALWAYS ACTIVE (no deferred mode; source-of-truth is a hand-curated REGIME_REGISTRY of every regime-classifying function in scripts/experiments/ecg/, each loaded via importlib). Catches the subtle bug where an author tweaks one regime-boundary copy and the paper's per-regime bar groupings silently diverge between figures. 5 rules: (R1) every registered classifier resolves to an importable callable; (R2) every byte-input classifier returns only labels from its declared vocabulary when fed the canonical L3 grid (1kB..16MB); (R3) within each taxonomy family, all byte-input members agree on every canonical L3 label (catches in-family drift between siblings like policy_winner_table._l3_regime and popt_vs_grasp_report._l3_regime); (R4) source-pattern scan of scripts/experiments/ecg/*.py finds no unregistered regime classifiers (defensive — catches new drift-prone classifiers added without registration); (R5) non-byte-label classifiers (ratio-input, range-input) carry an explanatory note describing what they actually classify. Today: 5 classifiers in 4 families — tiny_small_large_v1 (policy_winner_table + popt_vs_grasp_report, identical sibling pair); tiny_small_large_v2_oracle_gap (oracle_gap_report._regime, intentionally separate because it uses <= boundaries and 256 kB small/large split instead of 1 MB); wss_range (cross_tool_lru_regime, classifies an L3-size range); wss_ratio (wss_relative_l3, classifies L3/WSS ratio). 0 violations.",
    },
    {
        "id":        "lit_faith_citation_registry",
        "label":     "lit-faith citation registry purity",
        "generator": "scripts/experiments/ecg/lit_faith_citation_registry.py",
        "gate":      "scripts/test/test_lit_faith_citation_registry.py",
        "artifact":  "wiki/data/lit_faith_citation_registry.json",
        "summary":   "lit-faith citation registry purity audit (gate 246) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the per_claim array of wiki/data/literature_faithfulness_postfix.json AND a hand-curated CITATION_REGISTRY in the generator listing every canonical literature work the lit-faith table is allowed to cite, with structured {key, title, venue, year, patterns, note} metadata). Catches three classes of bug invisible to numerical tests: (a) an author edits a per_claim row to cite a paper that does not actually carry the claim; (b) a registry entry slowly accretes typos in its prose-citation form until pattern matching breaks; (c) within a (policy, app, expected_sign) bucket — the unit the paper actually quotes — different rows silently drift apart in which canonical work they attribute the expected sign to. 5 rules: (C1) every per_claim citation matches the substring patterns of >=1 registered canonical work; (C2) every registered canonical work is referenced >=1 time in per_claim (no dead-letter registry entries); (C3) within each (policy, app, expected_sign) bucket, all rows share >=1 canonical citation key (the paper-quote anchor stays consistent); (C4) every registry entry has non-empty venue + year (keeps the registry mineable for bibliography generation); (C5) every per_claim row carries a non-empty citation string. Today: 3 canonical works (Faldu-HPCA-2020, Balaji-HPCA-2021, Jaleel-ISCA-2010); 330 per_claim rows; 24 (policy, app, sign) buckets; coverage Balaji=252, Faldu=177, Jaleel=75; 0 violations.",
    },
    {
        "id":        "lit_faith_paper_tables",
        "label":     "Paper LaTeX-table emit invariant",
        "generator": "scripts/experiments/ecg/lit_faith_paper_tables.py",
        "gate":      "scripts/test/test_lit_faith_paper_tables.py",
        "artifact":  "wiki/data/lit_faith_paper_tables.json",
        "summary":   "Paper LaTeX-table emit-invariant audit (gate 247) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the .tex files inside the latest wiki/data/paper_pipeline_YYYYMMDD/ snapshot dir AND a hand-curated TABLE_REGISTRY in the generator listing every shipped paper table with its expected caption, tabular column-spec, and header tuple). Catches silent drift between the paper's prose and the auto-generated tables: a developer can edit a table-generation script, change the column count or header, and the .tex file regenerates with no test failure even though the paper text now mismatches the table. 7 rules: (T1) every registered table file exists in the latest paper_pipeline dir; (T2) registered caption matches the in-file \\caption{...} exactly; (T3) registered tabular column-spec matches the in-file \\begin{tabular}{...} spec; (T4) registered column-header tuple matches the in-file header row; (T5) every data row has exactly len(columns) cells AND no cell is the literal string 'nan'/'NaN'/'inf'/'-inf' (NaN/Inf are sentinels that look fine in PDF but are scientifically meaningless); (T6) there is no .tex file in paper_pipeline dir that is NOT registered (defensive — catches new tables added without registration); (T7) every table ends with the \\bottomrule\\end{tabular}\\end{table} closing trio (catches truncated LaTeX). Today: 5 registered tables (ecg_mode_overhead_summary, faithfulness_summary, popt_charged_overhead, popt_storage_overhead_summary, roi_policy_summary); 78 total data rows; 0 violations.",
    },
    {
        "id":        "lit_faith_sideband_schema",
        "label":     "Sideband-schema registry",
        "generator": "scripts/experiments/ecg/lit_faith_sideband_schema.py",
        "gate":      "scripts/test/test_lit_faith_sideband_schema.py",
        "artifact":  "wiki/data/lit_faith_sideband_schema.json",
        "summary":   "gem5/Sniper/cache_sim sideband-schema registry audit (gate 248) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the three C++ emit sites — graph_cache_context_gem5.hh, graph_cache_context_sniper.cc, graph_cache_context.h — AND a hand-curated SCHEMA_REGISTRY in the generator declaring the canonical field order, printf specifier, and C++ parameter type for every key=value token in the [graphctx] register region log line). Catches silent wire-format drift: a developer can re-order, rename, or drop a field in one overlay and the Tier-A sideband-registration parser (gate 1) stops matching that overlay's lines without any test failing for the obviously wrong reason. 7 rules: (S1) every registered emit-site file exists; (S2) the printf format string in each file matches the canonical schema field order byte-for-byte (after concatenating adjacent C string literals); (S3) the C++ logGraphCtxRegistration() parameter type list matches the canonical type tuple; (S4) every emit-site contains the canonical literal prefix '[graphctx] register region'; (S5) every schema field uses a printf specifier from the documented allow-list (%s, 0x%lx, %u, %d); (S6) the Tier-A parser regex anchor compiles AND round-trips a sample line built from the canonical schema (named groups match schema field names); (S7) each emit-site contains exactly one register-region fprintf (catches divergent backup emit paths added by mistake). Today: 6 schema fields (source, name, base, upper, hot_pct, grasp_region) × 3 emit sites; 0 violations.",
    },
    {
        "id":        "lit_faith_graph_family",
        "label":     "Graph-family map full-coverage",
        "generator": "scripts/experiments/ecg/lit_faith_graph_family.py",
        "gate":      "scripts/test/test_lit_faith_graph_family.py",
        "artifact":  "wiki/data/lit_faith_graph_family.json",
        "summary":   "Graph-family map full-coverage audit (gate 249) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are every .py file under scripts/experiments/ecg and scripts/test AND a CANONICAL_GRAPH_FAMILY map in the generator declaring the 8-graph shipped corpus). Gate 107 (test_graph_family_map_duplication) already locks the topology of 7 known copies (2 full + 5 short). This gate hardens that by AST-harvesting every module-level dict literal whose keys are known graph names and whose values are known family labels, then asserting that every copy agrees with the canonical map on every shared key. Catches new modules added by a future contributor that ship their own GRAPH_FAMILY copy and silently diverge from the canonical mapping. 6 rules: (F1) harvester picks up every module-level dict that looks like a GRAPH_FAMILY map; (F2) every harvested copy is a subset of canonical + reserved-future keys (no unknown graph→family pairs); (F3) every harvested copy agrees with canonical on every shared key (no value drift); (F4) canonical is non-empty AND every value is in the documented family allow-list (social, web, citation, road, mesh); (F5) no harvested copy that contains reserved-future graph tags is missing from the gate-107 FULL_SOURCES universe; (F6) out-of-universe copies are tracked for visibility. Today: canonical=8 graphs (4 social + 1 web + 1 citation + 1 road + 1 mesh); 12 harvested copies across 12 files; 7 out-of-universe but all subset-clean; 0 violations.",
    },
    {
        "id":        "lit_faith_paper_provenance",
        "label":     "Paper-table CSV provenance",
        "generator": "scripts/experiments/ecg/lit_faith_paper_provenance.py",
        "gate":      "scripts/test/test_lit_faith_paper_provenance.py",
        "artifact":  "wiki/data/lit_faith_paper_provenance.json",
        "summary":   "Paper-table CSV provenance audit (gate 250) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are every .tex + sibling .csv pair under wiki/data/paper_pipeline_YYYYMMDD/ AND a hand-curated PROVENANCE_REGISTRY in the generator pairing each shipped paper table with its underlying CSV and declaring the key-column tuples that must trace 1:1). The shipped .tex tables are intentionally truncated by paper_pipeline.py at [:20]/[:24] for paper layout, so the CSV is allowed to be a strict superset — but the .tex must NEVER carry a row that the CSV does not hold, and every paper row's key tuple (policy, benchmark, prefetcher, check, charged, oracle, candidate) must trace to at least one CSV row after LaTeX-escape normalization (ECG\\_DBG\\_ONLY ⇄ ECG_DBG_ONLY). Catches silent splits where paper_pipeline.py regenerates one file but not the other, or where a column rename desynchronizes the .tex header from the CSV header. 7 rules: (P1) every registered .tex AND .csv file exists in the latest paper_pipeline dir; (P2) tex_rows ≤ csv_rows (subset row count, no orphan paper rows past the truncation cap); (P3) for each (tex_header, csv_header) key-column pair, the multiset of tex values is a sub-multiset of the csv values after normalize; (P4) every declared key column exists in the corresponding tex/csv header tuple; (P5) no value in any tracked CSV key column is the empty string (every paper row must trace to non-empty CSV cells); (P6) no unregistered .csv sibling of a registered .tex stem (defensive — catches new paired CSVs added without registration); (P7) every registered CSV has a non-empty header row. Today: 5 pairs, 78 tex rows ↔ 85 csv rows, 13 tracked key columns; 0 violations.",
    },
    {
        "id":        "lit_faith_l3_registry",
        "label":     "L3 cache-size registry",
        "generator": "scripts/experiments/ecg/lit_faith_l3_registry.py",
        "gate":      "scripts/test/test_lit_faith_l3_registry.py",
        "artifact":  "wiki/data/lit_faith_l3_registry.json",
        "summary":   "L3 cache-size registry audit (gate 251) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are every .py module under scripts/experiments/ecg + scripts/test AND a CANONICAL_L3_TIERS map in the generator declaring 11 tokens 4kB..32MB with role + sub_tier + byte size + MB size for each). AST-harvests every module-level PAPER_L3 / PAPER_L3_SIZES / L3_SIZES tuple of string tokens AND every L3_MB / L3_BYTES dict whose keys are token strings, then enforces canonical agreement across the universe. Catches silent drift where a contributor adds a new L3 sweep size (e.g. '2MB') in one module but forgets the byte arithmetic, or where the anchor triplet (1MB, 4MB, 8MB) is reordered in a copy. 7 rules: (L1) every harvested token is in CANONICAL_L3_TIERS (no rogue size labels); (L2) every PAPER_L3-shaped tuple equals ANCHOR_TRIPLET exactly (order and content); (L3) every L3_MB dict value matches the canonical MB scaling (1MB→1.0, 4MB→4.0, 8MB→8.0, with float-division tolerated); (L4) every L3_BYTES dict value matches canonical byte arithmetic via AST constant-folding (so `4 * 1024` ≡ 4096 ≡ canonical 4kB); (L5) the canonical registry itself uses only declared roles + sub_tiers (no typo'd taxonomy); (L6) every anchor token (1MB/4MB/8MB) appears in at least one harvested PAPER_L3 tuple (defensive — catches accidental anchor removal); (L7) PAPER_L3-shaped constants do not disagree across files (cross-file consistency). Today: canonical=11 tokens, 54 files harvested, 43 PAPER_L3-shaped tuples, 9 L3_MB dicts, 7 L3_BYTES dicts, 1 subset tuple (MANDATORY_L3_SIZES); 0 violations.",
    },
    {
        "id":        "lit_faith_slurm_schema",
        "label":     "Slurm SBATCH schema registry",
        "generator": "scripts/experiments/ecg/lit_faith_slurm_schema.py",
        "gate":      "scripts/test/test_lit_faith_slurm_schema.py",
        "artifact":  "wiki/data/lit_faith_slurm_schema.json",
        "summary":   "Slurm SBATCH schema registry audit (gate 252) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are every *.sbatch file under scripts/experiments/ecg and scripts/experiments/vldb AND a CANONICAL_SBATCH_DIRECTIVES map in the generator declaring 14 directive names with required/optional flags + value regex patterns). Parses every shipped sbatch line-by-line (long-form --key=value tokens, multi-token lines like '--nodes=1 --ntasks=1 --cpus-per-task=16' supported), and enforces canonical agreement across the corpus. Catches silent drift like a contributor adding a typo'd directive (--mem-per-node), or shipping an sbatch missing --output, or using a non-Slurm time format. 9 rules: (S1) every #SBATCH line parses to --key[=value] (no garbage tokens); (S2) every required directive present (--job-name, --time, --nodes, --ntasks, --cpus-per-task, --mem, --output); (S3) every directive used appears in CANONICAL_SBATCH_DIRECTIVES (no rogue names); (S4) --mem / --mem-per-cpu values match \\d+[GMK]?; (S5) --time values match HH:MM:SS or D-HH:MM:SS; (S6) every shipped sbatch is single-node single-task (--nodes=1 AND --ntasks=1); (S7) --output and --error templates contain %x+%j or %A+%a (no anonymous logs); (S8) --job-name starts with gbrew- or ecg- (project prefix); (S9) --mem and --mem-per-cpu never co-occur in same file (Slurm forbids both). Today: 9 sbatch files (2 in ecg/, 7 in vldb/), 14 canonical directives, 7 required directives; 0 violations.",
    },
    {
        "id":        "lit_faith_handoff_xref",
        "label":     "HANDOFF gate-reference registry",
        "generator": "scripts/experiments/ecg/lit_faith_handoff_xref.py",
        "gate":      "scripts/test/test_lit_faith_handoff_xref.py",
        "artifact":  "wiki/data/lit_faith_handoff_xref.json",
        "summary":   "HANDOFF gate-reference registry audit (gate 253) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are wiki/HANDOFF-grasp-popt-validation.md AND scripts/experiments/ecg/confidence_dashboard.py:PYTEST_SUITES). A meta-gate that locks the narrative HANDOFF to the live dashboard so the user-visible storytelling cannot silently lag behind the gate suite. Catches the silent-drift case where a new gate lands in the dashboard but its paragraph is never added here, where the '**N gates, all GREEN**' headline falls behind, or where the refresh cadence drifts. 7 rules: (H1) every 'gate N' / 'gates N-M' token in HANDOFF parses to a positive integer (or positive range); (H2) every PYTEST_SUITES label carrying '(gate N)' is mentioned in HANDOFF (no orphan dashboard labels); (H3) the headline '**N gates, all GREEN, exit 0**' equals len(PYTEST_SUITES); (H4) 'Refresh complete at gate N' equals len(PYTEST_SUITES); (H5) 'Next refresh due at gate M' equals refresh-at + 5 (declared cadence); (H6) no duplicate (gate N) token in dashboard labels (each gate number labels at most one suite); (H7) max(labeled_dashboard_gates) == len(PYTEST_SUITES) (the newest labeled gate equals the live count — so a new gate cannot land in the dashboard without an explicit (gate N) label). Today: 139 HANDOFF gate-refs, 12 labeled dashboard gates (gates 242..253), 253 PYTEST_SUITES total; 0 violations.",
    },
    {
        "id":        "lit_faith_wiki_registry",
        "label":     "wiki/data bidirectional registry",
        "generator": "scripts/experiments/ecg/lit_faith_wiki_registry.py",
        "gate":      "scripts/test/test_lit_faith_wiki_registry.py",
        "artifact":  "wiki/data/lit_faith_wiki_registry.json",
        "summary":   "wiki/data bidirectional registry audit (gate 254) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are every wiki/data/*.json file on disk AND scripts/experiments/ecg/artifact_catalog.py entries AND an in-generator ALLOWED_AUXILIARY allow-list of postfix companion artifacts). Symmetric to gate 253 (HANDOFF↔PYTEST_SUITES): gate 253 binds narrative ↔ live suite count; gate 254 binds raw artifact filesystem ↔ catalog entries ↔ pytest coverage so a generator cannot ship a new wiki/data/*.json without explicit catalog + sibling .md + pytest accounting. 8 rules: (W1) every wiki/data/*.json file is accounted for (catalog entry OR auxiliary allow-list OR self-referential set); (W2) no ghost catalog entries (every artifact file referenced exists on disk); (W3) every catalog entry has non-empty generator/gate/artifact strings; (W4) every catalog entry's generator/gate/artifact path exists in the working tree; (W5) every .json artifact in the catalog has a sibling .md summary at the same stem; (W6) catalog ids are unique; (W7) catalog artifact paths are unique (no two entries claim the same file); (W8) every auxiliary allow-list entry references a real catalog id as its parent_id. Today: 110 wiki/data/*.json files, 106 catalog entries (incl. the 254-th lit-faith generator itself), 4 auxiliary allow-list entries (3 ECG-parity postfix files + 1 ECG substrate-parity per-observation companion), 1 self-referential (artifact_catalog.json); 0 violations.",
    },
    {
        "id":        "lit_faith_policy_registry",
        "label":     "Cache-policy vocabulary registry",
        "generator": "scripts/experiments/ecg/lit_faith_policy_registry.py",
        "gate":      "scripts/test/test_lit_faith_policy_registry.py",
        "artifact":  "wiki/data/lit_faith_policy_registry.json",
        "summary":   "Cache-policy vocabulary registry audit (gate 255) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are scripts/experiments/ecg/config.py:ALL_POLICIES AND every POLICIES/ALL_POLICIES/BASELINE_POLICIES/GRAPH_AWARE_POLICIES tuple harvested by AST from scripts/experiments/ecg/*.py AND an in-generator CANONICAL_POLICY_NAMES + CANONICAL_ECG_ARMS allow-list). Symmetric to gate 251 (L3 byte literals) and gate 248 (paper-label map): gate 251 locks the cache-size universe; gate 248 locks the per-policy paper labels; gate 255 locks the policy-token vocabulary itself so a new misspelled `POPT_charged` or rogue `srrip` lowercase cannot enter the codebase without an explicit canonical entry or ECG-arm declaration. 9 rules: (P1) every harvested policy token is in CANONICAL_POLICY_NAMES (8: LRU/FIFO/RANDOM/LFU/SRRIP/GRASP/POPT/ECG) OR CANONICAL_ECG_ARMS (9 documented variants: POPT_CHARGED + 8 ECG:* operational arms); (P2) POLICIES tuples have no duplicates; (P3) config.py ALL_POLICIES strictly equals BASELINE_POLICIES + GRAPH_AWARE_POLICIES with no extras, while extended ALL_POLICIES in other modules (roi_matrix.py) only add canonical-or-arm tokens; (P4) every canonical token has a valid family ∈ {baseline,graph_aware} + non-empty paper_label; (P5) no two canonical tokens share the same paper_label; (P6) every harvested 4-tuple POLICIES is a permutation of CANONICAL_FOUR_TUPLE (LRU,SRRIP,GRASP,POPT) — this is the paper's canonical anchor 4-tuple, so any 4-tuple that drifts to {LRU,SRRIP,GRASP,POPT_PRIMARY} or similar trips the gate; (P7) every harvested 3-tuple POLICIES is a subset of CANONICAL (catches narrative-anchor triplets like (GRASP,LRU,SRRIP)); (P8) no harvested token is a documented forbidden alias (e.g. lowercase `lru`, `Lru`, `srrip`); (P9) every CANONICAL_ECG_ARMS entry declares a real parent ∈ {POPT, ECG} plus a non-empty purpose string explaining why the arm exists. Today: 8 canonical tokens, 9 ECG arms, 43 harvested POLICIES tuples, 2 ALL_POLICIES sites; 0 violations.",
    },
    {
        "id":        "lit_faith_profile_registry",
        "label":     "ECG final-paper-run profile registry",
        "generator": "scripts/experiments/ecg/lit_faith_profile_registry.py",
        "gate":      "scripts/test/test_lit_faith_profile_registry.py",
        "artifact":  "wiki/data/lit_faith_profile_registry.json",
        "summary":   "ECG final-paper-run profile registry audit (gate 256) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are scripts/experiments/ecg/final_paper_manifest.json AND every --profile / profiles= citation harvested from scripts/experiments/ecg/*.py + scripts/test/*.py + scripts/experiments/README.md). Companion to gate 252 (Slurm SBATCH schema) and gate 255 (policy vocab): gate 252 locks the cluster-job vocabulary, gate 255 locks the cache-policy vocabulary, and gate 256 locks the run-profile vocabulary so a stage cannot silently reference a renamed profile, a manifest profile cannot rot without a stage / pytest / README user, and a typo'd `--profile fianl_replacement` in a published walkthrough is caught at gate time rather than at experiment time. 7 rules: (R1) every stage.profiles[*] token resolves to a key in manifest.profiles; (R2) every manifest.profiles key has a non-empty description (the descriptor that --list-profiles emits); (R3) every manifest.profiles key is referenced by at least one stage, pytest fixture, helper script, or README walkthrough — unless the description starts with `Placeholder` (the documented escape hatch for upcoming work explicitly flagged as such); (R4) every --profile / profiles= literal harvested outside the manifest resolves to a known manifest.profiles key (catches typos in published walkthroughs and test fixtures); (R5) profile names match `^[a-z][a-z0-9_]*$` (snake_case ASCII, no leading digit); (R6) stage names match `^[0-9]+[a-z0-9]*_[a-z][a-z0-9_]*$` (digit prefix → optional alnum suffix → underscore → snake_case body — the prefix preserves natural-sort execution order); (R7) each stage's profiles list is non-empty and duplicate-free. Today: 30 manifest profiles, 30 stages, 25 external citations across 12 distinct tokens (3 manifest profiles intentionally flagged Placeholder); 0 violations.",
    },
    {
        "id":        "lit_faith_backend_registry",
        "label":     "Backend/tool vocabulary registry",
        "generator": "scripts/experiments/ecg/lit_faith_backend_registry.py",
        "gate":      "scripts/test/test_lit_faith_backend_registry.py",
        "artifact":  "wiki/data/lit_faith_backend_registry.json",
        "summary":   "Backend/tool vocabulary registry audit (gate 257) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are an in-generator CANONICAL_BACKENDS allow-list of 7 entries AND every backend/tool string literal harvested by AST from scripts/experiments/ecg/*.py + scripts/test/*.py AND every --backend / --source-backend / --tool / --tool-name / --suite argparse choices+default). Vocabulary-lock companion to gate 252 (Slurm SBATCH schema), gate 255 (cache-policy vocab), and gate 256 (run-profile vocab): gate 257 locks WHICH simulator backend / tool every result row, anchor pickup, cross-tool aggregator, and report-label site references — so a contributor cannot silently rename `cache_sim` to `cache_simulator`, drop the hyphen in `gem5-riscv` to `gem5riscv`, or introduce a rogue `sniperx` / `GEM5` uppercase variant without an explicit canonical entry. CANONICAL_BACKENDS today: cache_sim + its kebab-case display sibling cache-sim (analytical LRU-stack sim), gem5 + gem5-riscv + gem5-x86 (cycle-accurate, three frontends), sniper + sniper-sift (interval-sim + SIFT trace frontend). 7 rules: (R1) every harvested literal is in CANONICAL_BACKEND_NAMES; (R2) every canonical has non-empty family ∈ {cache_sim,gem5,sniper} + non-empty paper_label; (R3) no two canonicals share a paper_label unless they are declared mutual punctuation_variants (cache_sim ⇄ cache-sim is the only such pair today); (R4) every canonical's punctuation_variants tuple includes its own name (catches lonely-variant declarations); (R5) every harvested --backend/--source-backend/--tool/--tool-name/--suite argparse choices list AND default ⊆ CANONICAL_BACKEND_NAMES ∪ {'both'} (the 'both' literal is reserved for --suite to mean cache_sim+gem5 together); (R6) every canonical name matches `^[a-z][a-z0-9_-]*$` (lowercase ASCII; hyphen/underscore allowed; no leading digit) — catches uppercase typos like `Gem5` and prefix-clash typos like `3gem5`; (R7) every canonical token has at least one in-tree literal reference (no dead canonicals — the registry follows real usage). Today: 7 canonical tokens, 3 families, 437 literal sites, 7 distinct harvested literals, 4 argparse sites; 0 violations.",
    },
    {
        "id":        "lit_faith_graph_registry",
        "label":     "Graph-name canonical map",
        "generator": "scripts/experiments/ecg/lit_faith_graph_registry.py",
        "gate":      "scripts/test/test_lit_faith_graph_registry.py",
        "artifact":  "wiki/data/lit_faith_graph_registry.json",
        "summary":   "Graph-name canonical map audit (gate 258) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are an in-generator CANONICAL_GRAPHS allow-list of 26 entries AND every graph string literal harvested by AST from scripts/experiments/ecg/*.py + scripts/test/*.py AND every per-source GRAPH_FAMILY / FAMILY_OF / GRAPH_FAMILIES dict AND the EVAL_GRAPHS list in scripts/experiments/ecg/config.py). Vocabulary-lock companion to gates 252 (Slurm SBATCH schema), 255 (cache-policy vocab), 256 (run-profile vocab), and 257 (backend/tool vocab): gate 258 locks WHICH benchmark graph every per-row record, anchor-cell census, family-classifier dict, and cross-paper baseline table references — so a contributor cannot silently misspell `soc-LiveJournal1` as `soc-livejournal1` / `soc-LJ1` in one helper while every other generator still uses the SNAP casing, drop the hyphen in `cit-Patents` to `citPatents`, introduce a new graph (`com-friendster`, `twitter7`) to one stage without adding the matching family-classifier entry, or shorten `delaunay_n19` to `delaunay19` and break the SNAP-vs-synthetic provenance carried by the underscore. CANONICAL_GRAPHS today: 26 entries across 8 families — social (email-Eu-core, soc-pokec, soc-LiveJournal1, com-orkut, com-orkut-undir, twitter-2010, twitter7, soc-LJ), web (web-Google, web-BerkStan, uk-2005, web-uk), road (roadNet-CA, road-CA, roadNet-PA, roadNet-TX, USA-Road), mesh (delaunay_n18, delaunay_n19, delaunay_n20), citation (cit-Patents), kronecker (kron21, kron22, kron23), p2p (p2p-Gnutella31), content (wikipedia_link_en). 8 rules: (R1) every harvested graph literal is in CANONICAL_GRAPHS; (R2) every canonical has non-empty family + paper_label; (R3) every canonical has source ∈ {SNAP, GAP, DIMACS, KONECT, WebGraph, synthetic, test}; (R4) every per-source family-classifier dict's keys are canonical AND map to the SAME family the canonical entry declares (catches the silent drift where roadNet-CA is 'road' in one helper and 'roads' in another); (R5) every harvested literal has at least one non-family-dict site (real corpus use, not just metadata) — unless documented_future=True for entries in RESERVED_FUTURE_KEYS lists; (R6) every canonical name matches `^[A-Za-z][A-Za-z0-9_-]*$` (alphanumeric + hyphen + underscore; no leading digit); (R7) every config.EVAL_GRAPHS entry is canonical (locks the config-driven evaluation corpus to the registry); (R8) every canonical family matches `^[a-z][a-z0-9_]*$` (lowercase ASCII with digits allowed for `p2p`). Today: 26 canonical graphs, 8 families, ~564 literal sites, ~8 family-classifier dicts, 6 EVAL_GRAPHS entries; 0 violations.",
    },
    {
        "id":        "lit_faith_build_registry",
        "label":     "Build-target registry",
        "generator": "scripts/experiments/ecg/lit_faith_build_registry.py",
        "gate":      "scripts/test/test_lit_faith_build_registry.py",
        "artifact":  "wiki/data/lit_faith_build_registry.json",
        "summary":   "SCons/Make build-target registry audit (gate 259) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are an in-generator CANONICAL_BUILD_TARGETS allow-list of 51 (backend, kernel, variant) entries AND the root Makefile's four KERNELS_<BACKEND> lists + four CXXFLAGS_<BACKEND> declarations AND every .cc file on disk under the canonical SRC_DIRS). Vocabulary-lock companion to gates 252 (Slurm SBATCH schema), 255 (cache-policy vocab), 256 (run-profile vocab), 257 (backend/tool vocab), and 258 (graph canonical map): gate 259 locks WHICH compile targets actually produce the binaries that the downstream gates measure — so a contributor cannot silently rename `bench/bin_sim/pr` to `bench/bin_sim/pr_cs`, swap an `-O3` for `-O2` in `CXXFLAGS_GAP` (would break every cache-sim apples-to-apples comparison against the literature baseline), drop the `-fopenmp` that `ecg_preprocess` requires, introduce an undocumented gem5 frontend variant beyond {base, m5ops, riscv_m5ops}, or add a `pr_kernel_smoke` orphan to bench/src_sniper that no canonical entry tracks. CANONICAL_BUILD_TARGETS today: 4 backends (native, cache_sim, gem5, sniper) × kernel × variant: 9 native production kernels (bc, bfs, cc, cc_sv, pr, pr_spmv, sssp, tc, tc_p) + converter, 9 cache_sim kernels (GAP roster + ecg_preprocess), 8 gem5 kernels × 3 variants (base, m5ops, riscv_m5ops) = 24 gem5 targets, 5 sniper Phase-0 smokes (hello_roi, sg_kernel, pr_kernel_smoke, bfs_kernel_smoke, sssp_kernel_smoke) + 3 in-flight real kernels (pr, bfs, sssp). 8 rules: (R1) every (backend, kernel) in CANONICAL_BUILD_TARGETS has a matching source file under the canonical SRC_DIR; (R2) every kernel listed in the Makefile KERNELS_<BACKEND> variable is canonical for that backend; (R3) every canonical backend has a CXXFLAGS_<BACKEND> block containing all required tokens (-std=c++17, -fopenmp, -DNDEBUG, plus -DNO_M5OPS for gem5 default and -I$(SNIPER_INCLUDE) for sniper); (R4) every canonical backend maps to the canonical SRC_DIR / BIN_DIR path; (R5) every canonical kernel has a non-empty graph-algorithm family classification ∈ {pagerank, traversal, shortest-path, connected-component, centrality, triangle, preprocess, smoke}; (R6) every backend's CXXFLAGS carries its documented optimisation level (-O3 native+sim, -O1 gem5, -O2 sniper) — apples-to-apples lock; (R7) every .cc file under the canonical SRC_DIRs maps to a canonical (backend, kernel) entry (no orphan sources); (R8) every canonical backend has a documented ROI mechanism ∈ {m5ops, sift, sim-callback, none}. Today: 4 backends, 51 canonical targets, 26 Makefile-harvested kernels, 0 orphan sources; 0 violations.",
    },
    {
        "id":        "lit_faith_cli_registry",
        "label":     "GAPBS CLI registry",
        "generator": "scripts/experiments/ecg/lit_faith_cli_registry.py",
        "gate":      "scripts/test/test_lit_faith_cli_registry.py",
        "artifact":  "wiki/data/lit_faith_cli_registry.json",
        "summary":   "GAPBS kernel CLI-vocabulary registry audit (gate 260) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are an in-generator CL_CLASSES allow-list of 6 declared inheritance entries AND the live `class CL*` + `get_args_` getopt-extension declarations in bench/include/external/gapbs/command_line.h AND every per-backend kernel `.cc` instantiation of its CL class under bench/src{,_sim,_gem5}/). Vocabulary-lock companion to gates 252 (Slurm SBATCH schema), 255 (cache-policy vocab), 256 (run-profile vocab), 257 (backend/tool vocab), 258 (graph canonical map), and 259 (build-target registry): gate 260 locks WHICH command-line flags each GAPBS-derived kernel binary actually accepts (per its CL-class inheritance chain) — so a contributor cannot silently iterate `-i 16 -i 32` for both pr AND bfs (bfs uses CLApp, which has no -i; the sweep collapses to a single bfs measurement repeated N times), pass `-t 1e-7` to sssp (CLDelta intercepts -t as unknown and bails after the run starts → wasted SBATCH time), invent a long-form `--num-trials 5` when the getopt loop only recognises short `-n 5`, or claim per-kernel arg tables like `args['pr_spmv']['--tolerance']` when the binary only accepts `-t`. CL_CLASSES today: CLBase (parent=None, getopt `f:g:hk:su:m:o:zj:SlD:` — 13 flags: file/scale/help/degree/symmetrize/uniform-random/mem-friendly/ordering/indegree/segmentation/keep-self/log/db-dir), CLApp ⊂ CLBase (+ `an:r:v` — analysis/trials/start-vertex/verify → 17 flags), CLIterApp ⊂ CLApp (+ `i:` → 18 flags, used by bc), CLPageRank ⊂ CLApp (+ `i:t:` → 19 flags, used by pr/pr_spmv), CLDelta ⊂ CLApp (+ `d:` → 18 flags, used by sssp), CLConvert ⊂ CLBase (+ `e:b:x:q:p:y:V:w` → 21 flags, used by converter). KERNEL_CL_CLASS: pr→CLPageRank, pr_spmv→CLPageRank, bfs→CLApp, sssp→CLDelta, bc→CLIterApp, cc→CLApp, cc_sv→CLApp, tc→CLApp, tc_p→CLApp, ecg_preprocess→CLApp, converter→CLConvert. 7 rules: (R1) every declared CL class has a non-empty getopt-extension matching the live header; (R2) every canonical kernel source instantiates the canonical CL class (verified across all 3 backends → 27/27 OK); (R3) every canonical kernel's flag-set is the union of its inheritance chain (no orphan kernel-only flags); (R4) no within-chain getopt conflicts (same letter at two levels of one chain); (R5) every canonical flag matches `^[A-Za-z]$`; (R6) every canonical flag has a documented FLAG_PURPOSE entry (28 entries covering all distinct flags); (R7) flag-arity (takes value / no value) is consistent across every class that declares the flag. Today: 6 CL classes, 11 kernels, 28 distinct flags, 27/27 source-instantiation checks pass; 0 violations.",
    },
    {
        "id":        "lit_faith_arm_catalog",
        "label":     "ECG arm catalog",
        "generator": "scripts/experiments/ecg/lit_faith_arm_catalog.py",
        "gate":      "scripts/test/test_lit_faith_arm_catalog.py",
        "artifact":  "wiki/data/lit_faith_arm_catalog.json",
        "summary":   "ECG arm catalog registry audit (gate 261) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the live paper_pipeline.POLICY_ORDER + POLICY_LABELS + POLICY_DESCRIPTIONS + POLICY_COLORS + POLICY_HATCHES tables, proof_matrix.ABLATIONS + ADAPTIVE_SELECTORS dataclass lists, and lit_faith_policy_registry.CANONICAL_ECG_ARMS). Eighth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog): gate 261 locks the cross-file consistency of every ECG arm + paper-shipping policy across the THREE namespaces they appear in — the registry side (ECG: prefixed, 9 entries), the paper side (underscore-namespace POLICY_ORDER + per-policy label/description/color/hatch tables, 9 entries), and the measurement side (proof_matrix mixed-case ABLATIONS pointing back at registry keys, 16 entries) — so a contributor cannot silently add a new `ECG_DBG_PRIMARY_CHARGED` ablation without bumping POLICY_DESCRIPTIONS (figure with blank legend caption), rename `ECG:DBG_PRIMARY` → `ECG:DBG_HEAD` without updating `ECG_DBG_POPT.policy` in proof_matrix (silently-missing bar in grid plots), or duplicate an ABLATIONS label (silent row-merging in the rollup CSV). 7 rules A1-A7: (A1) every paper_pipeline.POLICY_ORDER non-baseline entry has a CANONICAL_ECG_ARMS entry after namespace translation (ECG_X → ECG:X; POPT_CHARGED is itself); (A2) every POLICY_ORDER entry has a POLICY_LABELS + POLICY_DESCRIPTIONS + POLICY_COLORS entry; (A3) every paper policy with suffix `_CHARGED` has a POLICY_HATCHES entry (required for grayscale legibility); (A4) every proof_matrix.ABLATIONS.policy is a canonical baseline ∈ {LRU,SRRIP,GRASP,POPT} or a CANONICAL_ECG_ARMS key; (A5) every proof_matrix.ADAPTIVE_SELECTORS.candidates entry references a real ablation label; (A6) proof_matrix.ABLATIONS has no duplicate labels; (A7) every CANONICAL_ECG_ARMS entry with parent='ECG' has at least one ABLATIONS row, OR (for _CHARGED arms) its uncharged parent has one (since _CHARGED arms are post-hoc projections of their uncharged parent, see paper_pipeline.py PAIRS). Today: 9 paper policies (2 charged: POPT_CHARGED + ECG_DBG_PRIMARY_CHARGED), 9 registry arms (POPT_CHARGED + 8 ECG:* arms), 16 ablations (4 cache_alone + 7 ecg_replacement + 2 pfx_only + 3 combined), 2 adaptive selectors (ECG_ADAPTIVE_ORACLE, ECG_ADAPTIVE_NO_FULL_POPT); 0 violations.",
    },
    {
        "id":        "lit_faith_cross_tool_schema",
        "label":     "ECG cross-tool aggregator schema",
        "generator": "scripts/experiments/ecg/lit_faith_cross_tool_schema.py",
        "gate":      "scripts/test/test_lit_faith_cross_tool_schema.py",
        "artifact":  "wiki/data/lit_faith_cross_tool_schema.json",
        "summary":   "ECG cross-tool aggregator schema registry audit (gate 262) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the in-generator CROSS_TOOL_AGGREGATORS table — 6 aggregator schema declarations today — and the live on-disk JSON shape of each named aggregator artifact under wiki/data/). Ninth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool aggregator schema): gate 262 locks the on-disk JSON shape of every cross-tool aggregator artifact (cross_tool_lru_regime, cross_tool_saturation, cross_tool_slope_ordering, cross_tool_slope_universality, cross_tool_winners, anchor_cross_tool_agreement) — every report that joins cache_sim ↔ gem5 ↔ sniper at the per-cell or per-policy level — so a contributor cannot silently rename a top-level key in one aggregator (downstream rollup scripts KeyError at the next confidence-fast run, but only after a 6-minute pytest start-up), delete the per-cell `cells` list and replace it with a scalar summary (every cross-artifact parity test passes trivially with 0 rows), introduce a new aggregator that picks a third name for the same concept (`meta` vs `summary` vs `schema` — this gate enforces both shape AND naming), or shadow a canonical tool name (`sniper-v8` instead of `sniper`) in the tools field (dashboard parser silently ignores half the data). 7 rules S1-S7: (S1) every CROSS_TOOL_AGGREGATORS entry's artifact exists on disk; (S2) every artifact is valid JSON with a top-level object; (S3) every declared top-level key is present in the on-disk JSON (extra keys allowed — schemas grow but cannot silently shrink); (S4) every declared evidence-bearing path (cells/per_tool/checks/shared_cells/tool_results) resolves to a non-empty list or dict; (S5) every declared per-row required key (app/graph/policy/...) is present in every evidence row; (S6) every aggregator's declared tools-path value is a subset of the canonical tool vocabulary (gate 257 CANONICAL_BACKENDS — 7 entries including both punctuation variants of cache_sim); (S7) every declared verdict-path value has the declared Python type (bool/str/int/list/dict). Today: 6 aggregators, 25 evidence rows total; 0 violations.",
    },
    {
        "id":        "lit_faith_config_matrix",
        "label":     "ECG configuration matrix",
        "generator": "scripts/experiments/ecg/lit_faith_config_matrix.py",
        "gate":      "scripts/test/test_lit_faith_config_matrix.py",
        "artifact":  "wiki/data/lit_faith_config_matrix.json",
        "summary":   "ECG configuration matrix registry audit (gate 263) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the live scripts/experiments/ecg/config.py module-level tables — BASELINE_POLICIES, GRAPH_AWARE_POLICIES, PREVIEW_POLICIES, ALL_POLICIES, ECG_MODES, ACCURACY_PAIRS, REORDER_POLICY_PAIRS, BENCHMARKS, EVAL_GRAPHS, DEFAULT_CACHE, CACHE_SIZES_SWEEP — and the canonical registries gates 251 (L3 tiers), 255 (policy names), 258 (graph names), 260 (kernel CLI classes) lock). Tenth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool schema, 263 config matrix): gate 263 locks the central ecg/config.py module — the one source-of-truth that the entire ECG sweep / accuracy / proof-matrix / final-paper-run pipeline reads cache sizes, benchmark lists, graph corpora, policy vocab, and (reorder, policy) pair lists from — against silent vocabulary drift relative to the canonical registries AND against internal self-consistency. Catches: a contributor adds `'BIP'` to BASELINE_POLICIES without registering it in gate 255 CANONICAL_POLICY_NAMES (downstream parsers silently drop BIP rows); a graph is renamed `com-orkut` → `orkut` in EVAL_GRAPHS without a corresponding alias entry in gate 258 CANONICAL_GRAPHS (sweep runs land in results/orkut/ but no plot script looks there); DEFAULT_CACHE['CACHE_L3_SIZE'] is bumped to 10485760 (10 MB) which matches no canonical L3 tier (every L3-anchored plot picks an arbitrary closest tier and the 8 MB anchor claim silently breaks); a benchmark is renamed `cc_sv` → `conn-comp-sv` without a corresponding gate 260 KERNEL_CL_CLASS edit (every sweep invokes a non-existent binary); an `('-o 99', 'MAGIC')` pair is added to REORDER_POLICY_PAIRS where MAGIC isn't in ALL_POLICIES (bench runner emits zero rows but pipeline reports OK). 7 rules C1-C7: (C1) every policy in BASELINE_POLICIES ∪ GRAPH_AWARE_POLICIES ∪ PREVIEW_POLICIES is in gate 255 CANONICAL_POLICY_NAMES; (C2) every EVAL_GRAPHS.name is the canonical name of some gate 258 CANONICAL_GRAPHS entry; (C3) every BENCHMARKS entry is a key of gate 260 KERNEL_CL_CLASS; (C4) DEFAULT_CACHE.CACHE_L3_SIZE parses to a byte count that equals the bytes field of exactly one tier in gate 251 CANONICAL_L3_TIERS (today: 8388608 = 8MB anchor); (C5) every CACHE_SIZES_SWEEP entry is a power-of-2 byte count; the sweep brackets the L3 anchor (min ≤ DEFAULT_L3 ≤ max); entries strictly increasing; (C6) every (reorder, policy, ...) tuple across ACCURACY_PAIRS ∪ REORDER_POLICY_PAIRS uses a policy in ALL_POLICIES (the union of BASELINE_POLICIES and GRAPH_AWARE_POLICIES); (C7) every ECG_MODE value observed inside an env dict of ACCURACY_PAIRS is also a member of ECG_MODES (no orphan modes); every ECG_MODES value is either paper-pipeline-recognised as ECG_<mode> in POLICY_ORDER, or in the documented ECG_PRIVATE_MODES allow-list (ECG_EMBEDDED — valid ECG runtime mode but not a paper-figure bar). Today: 8 ALL_POLICIES (5 baseline LRU/FIFO/RANDOM/LFU/SRRIP + 3 graph-aware GRASP/POPT/ECG), 5 PREVIEW_POLICIES, 7 BENCHMARKS (pr/pr_spmv/bfs/cc/cc_sv/sssp/bc), 6 EVAL_GRAPHS, 4 ECG_MODES (DBG_PRIMARY/POPT_PRIMARY/DBG_ONLY/ECG_EMBEDDED) + 1 private mode, 12-entry cache-sweep (32kB → 64MB powers-of-2), 21 (reorder, policy) pairs across the two pair-tables; 0 violations.",
    },
    {
        "id":        "lit_faith_filename_grammar",
        "label":     "wiki/data filename grammar",
        "generator": "scripts/experiments/ecg/lit_faith_filename_grammar.py",
        "gate":      "scripts/test/test_lit_faith_filename_grammar.py",
        "artifact":  "wiki/data/lit_faith_filename_grammar.json",
        "summary":   "wiki/data artifact filename grammar registry audit (gate 264) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the live wiki/data/ filesystem walk + scripts/experiments/ecg/artifact_catalog.py CATALOG list). Eleventh in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool schema, 263 config matrix, 264 filename grammar): gate 264 locks the file-system shape of wiki/data/ — the single shipping surface for every generator artifact in the paper-confidence pipeline — against silent drift in filename casing, extension, trio-pairing, catalog-presence, and subdirectory layout. Catches: a contributor adds `DiscoveryResults.json` (CamelCase) to wiki/data/ — every consumer that does `json.load(open(stem.lower() + '.json'))` FileNotFoundErrors and the rollup script silently treats the report as missing; the trio convention is broken by emitting only `foo.json` without `foo.md` — the per-artifact preview-link doc template renders a broken markdown link in HANDOFF; a generator drops an artifact under `wiki/data/results/<stem>.json` instead of `wiki/data/<stem>.json` — the catalog presence-check passes (artifact path is literal), but the dashboard's `wiki/data/*.json` glob misses it and the gate goes un-scored; a contributor adds a `.txt` or `.tex` file directly to wiki/data/ — the trio pattern is ambiguous, the artifact catalog grows by zero entries, and the dashboard's mtime-based regen check picks an arbitrary cutoff; a renamed catalog entry's `artifact` field points at a path that doesn't exist on disk — the catalog has `missing art=[]` but a single typo can flip it to `missing art=[wiki/data/<stem>.json]` and the entire reproduce smoke gate trips. 7 rules F1-F7: (F1) every regular file in wiki/data/ matches `^[a-z][a-z0-9_]*\\.(json|md|csv)$` — lower_snake_case stem, no leading digit, no whitespace, no dash, exactly one of three approved extensions; (F2) every .json under wiki/data/ has a matching .md sibling, UNLESS the stem is in the documented MD_OPTIONAL_STEMS allow-list (paired-table postfix companions whose narrative lives in the parent table's .md — 4 entries today: ecg_gem5_parity_postfix, ecg_pfx_vs_droplet_postfix, ecg_sniper_parity_postfix, ecg_substrate_parity_postfix); (F3) every .md has a matching .json sibling OR is in JSON_OPTIONAL_STEMS (today: literature_reproduction_summary — free-form text report); (F4) every artifact-catalog entry's artifact field points at a file that exists on disk; catalog id and artifact path are unique; (F5) every catalog.artifact extension is in {.json, .csv} (the catalog tracks data artifacts, not narrative artifacts; narrative .md files are implicit trio-siblings of their .json parent); (F6) every wiki/data stem is accounted for: either declared in artifact_catalog as the artifact (modulo .md/.csv sibling), or in the documented IMPLICIT_PAPER_PIPELINE_STEMS allow-list (MD_OPTIONAL_STEMS ∪ JSON_OPTIONAL_STEMS ∪ META_ARTIFACT_STEMS — the meta-artifact stems today is just `artifact_catalog`, which is the catalog of itself and thus cannot self-reference without bootstrap concerns); (F7) every wiki/data/ subdirectory name matches DOCUMENTED_SUBDIR_RE (`paper_pipeline_<YYYYMMDD>...` — today: 1 subdir `paper_pipeline_20260528`; ad-hoc `results/` or `archive/` subdirs are blocked). Today: 121 stems, 296 files, 1 subdir; 116/116 catalog artifacts on disk; 0 violations.",
    },
    {
        "id":        "lit_faith_sideband_grammar",
        "label":     "gem5/Sniper sideband filename grammar",
        "generator": "scripts/experiments/ecg/lit_faith_sideband_grammar.py",
        "gate":      "scripts/test/test_lit_faith_sideband_grammar.py",
        "artifact":  "wiki/data/lit_faith_sideband_grammar.json",
        "summary":   "gem5/Sniper sideband filename + env-var grammar registry audit (gate 265) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the live bench/include/gem5_sim/gem5_harness.h, the three Sniper cache-set sources (cache_set_popt.cc, cache_set_grasp.cc, cache_set_ecg.cc), the two Sniper prefetcher sources (ecg_pfx_prefetcher.cc, droplet_prefetcher.cc), and scripts/experiments/ecg/roi_matrix.py's gem5_sideband_paths + sniper_sideband_paths functions). Twelfth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool schema, 263 config matrix, 264 wiki/data filename grammar, 265 sideband filename grammar): gate 265 locks the FILENAME + ENV-VAR vocabulary that gem5 and Sniper overlays use to refer to the four runtime sideband artifacts (context JSON, popt matrix bin, out-edges bin, in-edges bin) — the wire-format BETWEEN the benchmark process and the simulator overlay. Where gate 248 locks the *internal schema* of the [graphctx] register region log line (field names + printf specifiers + parser regex), gate 265 locks the complementary surface area: the names by which those artifacts are referred to at every emit-site (C++ overlay default-paths via gem5_env_or_default / envOrDefault) and every parse-site (Python sideband-path dicts). Catches: a contributor renames `gem5_graphbrew_ctx.json` to `gem5_graphbrew_context.json` in the C++ default-path string of gem5_harness.h but forgets to update the Python parse-site gem5_sideband_paths dict — runs succeed but Tier-A parsing silently sees zero registered regions and pivot tests RED for unrelated-looking reasons; a contributor flips `SNIPER_POPT_MATRIX` to `SNIPER_PMATRIX` in one of three Sniper cache-set sources (popt/grasp/ecg) but not the others — runs with the changed cache-set fall back to /tmp/sniper_popt_matrix.bin (the unchanged literal default) while the runner sets SNIPER_PMATRIX (the new env name), and the POPT vector never loads; a contributor adds a fifth sideband artifact under a path like `/tmp/gem5_popt_pfx.bin` without registering it here — runs succeed but the proof-matrix audit doesn't know to clean it up between runs and stale state from prior runs silently contaminates the next run; the `graphbrew_sidebands/` subdirectory convention used by the Python parsers is changed to `sidebands/` — runs succeed but Python parsers look in the wrong dir and silently treat all artifacts as missing. 7 rules S1-S7: (S1) every registry entry filename matches `^(gem5|sniper)_[a-z0-9_]+\\.(json|bin)$` — tool prefix, lower_snake_case stem, approved extension; (S2) every registry filename has tool-prefix matching its tool field; role ∈ {context, popt_matrix, out_edges, in_edges}; context → .json, others → .bin; (S3) env_var = TOOL_STEM (uppercase + underscores derived from filename), default_path = /tmp/<filename> — bijection between filename and (env_var, default_path); (S4) every `gem5_env_or_default(NAME, PATH)` call in gem5_harness.h for a sideband role uses a registry-declared (NAME, PATH) pair; (S5) every Sniper `envOrDefault(NAME, PATH)` call in the cache-set + prefetcher sources for a sideband role uses a registry-declared (NAME, PATH) pair; (S6) every Python sideband-path dict entry in roi_matrix.py's gem5_sideband_paths and sniper_sideband_paths functions points at `<sideband_dir>/<canonical-filename>` for some role in the registry — keys and filenames bijective with registry; (S7) every sideband filename literal found anywhere under bench/include/{gem5,sniper}_sim/ and roi_matrix.py is declared in the registry — no orphan (gem5|sniper)_[a-z_]+.(json|bin) literals (modulo FILENAME_NON_SIDEBAND_ALLOW for non-runtime config files like `.sniper_overlays.json`). Today: 8 registry entries (4 gem5 + 4 sniper), 2 tools, 4 roles, 1 sideband subdir `graphbrew_sidebands/`; 6 emit-sites audited (gem5_harness.h + 3 Sniper cache-sets + 2 Sniper prefetchers); 1 parse-site audited (roi_matrix.py); 0 violations. Together with gate 248 (schema content) and gate 264 (wiki/data filename shape) this completes the filename/vocabulary lock on every shipping surface in the pipeline.",
    },
    {
        "id":        "lit_faith_overlay_tracker",
        "label":     "Sniper overlay-installation tracker",
        "generator": "scripts/experiments/ecg/lit_faith_overlay_tracker.py",
        "gate":      "scripts/test/test_lit_faith_overlay_tracker.py",
        "artifact":  "wiki/data/lit_faith_overlay_tracker.json",
        "summary":   "Sniper overlay-installation tracker registry audit (gate 266) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the live bench/include/sniper_sim/overlays/ tree, the on-disk bench/include/sniper_sim/.sniper_overlays.json tracker, and scripts/setup_sniper.py's write_overlay_status + patch_*_overlay functions). Thirteenth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool schema, 263 config matrix, 264 wiki/data filename grammar, 265 sideband grammar, 266 overlay tracker): gate 266 locks the OVERLAY INSTALLATION CONTRACT that scripts/setup_sniper.py applies when copying GraphBrew overlay files into the Sniper checkout and applying wiring patches. Where gate 265 locks the runtime filename + env-var vocabulary, gate 266 locks the BUILD-TIME contract: what files get copied where, which policies/prefetchers exist, and which patches get applied. Catches: a contributor adds a new cache-set source `cache_set_xyz.cc` under overlays/ but forgets to add 'xyz' to the POLICIES list in setup_sniper.py — the file gets copied into Sniper but the patches/cache_set_factory wiring doesn't know about it and the new policy is silent; the contributor adds a new prefetcher `xyz_prefetcher.cc` but the prefetcher_factory_droplet patch doesn't get extended to wire it in; a contributor removes a copied_files entry from setup_sniper.py without removing the on-disk overlays/ file — subsequent `setup_sniper.py --apply-overlays` runs leave the orphan file unreferenced; a contributor renames an overlay file under overlays/ but the .sniper_overlays.json tracker still lists the old name — the dashboard's overlay-installed check passes (the tracker file is present and valid JSON) but the actual file isn't in the new Sniper checkout. 7 rules O1-O7: (O1) every copied_files entry has valid grammar (lower_snake_case path + .cc/.h); (O2) every copied_files entry exists on disk under overlays/; (O3) every policy token has both cache_set_<pol>.cc + cache_set_<pol>.h in copied_files; (O4) every prefetcher token has both <pf>_prefetcher.cc + <pf>_prefetcher.h in copied_files; (O5) every patches token has a patch_<token>_overlay function in setup_sniper.py OR is in PATCH_NON_FUNCTION_ALLOW (4 entries today — bundled patches applied through other patch_*_overlay functions); (O6) the on-disk .sniper_overlays.json matches the canonical registry (copied_files + policies + prefetchers + patches sorted-set equality); (O7) copied_files is exhaustive — every regular file under overlays/ with a tracked extension is listed (modulo OVERLAY_README_ALLOW for README.md files). Today: 12 copied files (8 cache + 4 prefetcher), 3 policies (grasp, popt, ecg), 2 prefetchers (droplet, ecg_pfx), 5 patches; 0 violations. Together with gate 265 (runtime filename/env-var grammar) and gate 248 (schema content) this completes the FULL coverage on the gem5/Sniper overlay surface: build-time installation + runtime filename + runtime schema.",
    },
    {
        "id":        "lit_faith_gem5_overlay_tracker",
        "label":     "gem5 overlay-installation tracker",
        "generator": "scripts/experiments/ecg/lit_faith_gem5_overlay_tracker.py",
        "gate":      "scripts/test/test_lit_faith_gem5_overlay_tracker.py",
        "artifact":  "wiki/data/lit_faith_gem5_overlay_tracker.json",
        "summary":   "gem5 overlay-installation tracker registry audit (gate 267) — ALWAYS ACTIVE (no deferred mode; sources-of-truth are the live bench/include/gem5_sim/overlays/ tree and scripts/setup_gem5.py's OVERLAY_FILE_MAP source→destination dict + PATCH_FILES list + apply_overlays + apply_patches functions). Fourteenth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool schema, 263 config matrix, 264 wiki/data filename grammar, 265 sideband grammar, 266 Sniper overlay tracker, 267 gem5 overlay tracker): gate 267 is the gem5 mirror of gate 266 — it locks the gem5 OVERLAY INSTALLATION CONTRACT applied by `python scripts/setup_gem5.py` when copying GraphBrew overlay files (grasp_rp / popt_rp / ecg_rp replacement policies; droplet / ecg_pfx prefetchers; GraphReplacementPolicies.py and GraphPrefetchers.py SimObject declarations; graph_cache_context_gem5.hh; arch/riscv/isa/formats/ecg.isa) into the cloned gem5 source tree and applying SConscript patches that register the new SimObjects with gem5's build system. Where gate 266 covers Sniper's build-time contract through its .sniper_overlays.json status file + patch_*_overlay functions, gate 267 covers gem5's contract through OVERLAY_FILE_MAP + PATCH_FILES + the live importable setup_gem5 module. Catches: a contributor adds a new replacement policy source file `mem/cache/replacement_policies/xyz_rp.cc` under overlays/ but forgets to add it to OVERLAY_FILE_MAP — the file gets shipped in the overlays/ tree but never copied into the gem5 checkout, so `--cpu-type DerivO3CPU --caches --l1d_rp=GraphAwareRP_XYZ` fails with 'unknown replacement policy'; a contributor adds GraphReplacementPolicies.py entry but forgets the .cc/.hh pair (or vice versa) and gem5's scons build fails with link errors at deploy-time; a contributor renames a prefetcher source under overlays/ without updating OVERLAY_FILE_MAP — subsequent `setup_gem5.py` runs silently skip the renamed file and gem5 builds without the prefetcher; a contributor adds a new SConscript.patch under a sibling directory but forgets to extend PATCH_FILES — the patch is shipped under overlays/ but never applied, so new SimObjects don't get registered. 7 rules G1-G7: (G1) every OVERLAY_FILE_MAP source path matches grammar (`[A-Za-z][A-Za-z0-9_/.]+\\.(cc|hh|py|isa)$`); (G2) every OVERLAY_FILE_MAP source exists on disk under bench/include/gem5_sim/overlays/; (G3) every policy token has both `mem/cache/replacement_policies/<pol>_rp.cc` AND `_rp.hh` in OVERLAY_FILE_MAP; (G4) every prefetcher token has both `mem/cache/prefetch/<pf>.cc` AND `.hh` in OVERLAY_FILE_MAP; (G5) every PATCH_FILES entry exists on disk under overlays/; (G6) OVERLAY_FILE_MAP + PATCH_FILES is exhaustive over overlays/ — every regular file with a tracked extension or .patch suffix is listed (modulo OVERLAY_EXTRA_ALLOW for staging/dev files like arch/riscv/isa/decoder_ecg_extract.isa); (G7) the LIVE setup_gem5.OVERLAY_FILE_MAP keys + PATCH_FILES list exactly match the canonical registry derived in this gate AND every (src, dst) tuple satisfies the identity invariant src==dst (deviations require explicit audit). Today: 14 overlay sources (8 replacement_policies + 5 prefetch + 1 isa), 3 policies (grasp, popt, ecg), 2 prefetchers (droplet, ecg_pfx), 2 SConscript patches; 0 violations. Together with gate 266 (Sniper) this completes the BUILD-TIME overlay-installation lock on both simulator backends.",
    },
    {
        "id":        "lit_faith_setup_script_registry",
        "label":     "Setup-script invariant registry",
        "generator": "scripts/experiments/ecg/lit_faith_setup_script_registry.py",
        "gate":      "scripts/test/test_lit_faith_setup_script_registry.py",
        "artifact":  "wiki/data/lit_faith_setup_script_registry.json",
        "summary":   "Setup-script invariant registry audit (gate 268) — ALWAYS ACTIVE (sources-of-truth are the live scripts/setup_gem5.py and scripts/setup_sniper.py files). Fifteenth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool schema, 263 config matrix, 264 filename grammar, 265 sideband grammar, 266 Sniper overlay tracker, 267 gem5 overlay tracker, 268 setup-script invariants): gate 268 sits above gates 266+267 by locking the CONTAINING SETUP SCRIPTS themselves. Where gates 266+267 ensure the overlay payload + installation contract is well-formed, gate 268 ensures the scripts that orchestrate that installation cannot silently lose entry points, change which upstream repository they clone from, drop the directory-constant skeleton, or grow undocumented helpers. Catches: a refactor accidentally deletes `def apply_overlays(` from setup_sniper.py and the cleanup goes unnoticed because no test directly invokes it; a contributor switches GEM5_REPO_URL from `https://github.com/gem5/gem5.git` to a personal fork URL while debugging and forgets to revert before commit; a refactor renames PROJECT_ROOT to ROOT_DIR in setup_gem5.py and breaks every downstream tool that imports it; a contributor adds a new top-level helper function but forgets to register it in the canonical inventory, allowing the script's public surface to grow without review. 7 rules S1-S7: (S1) GEM5_REPO_URL == `https://github.com/gem5/gem5.git` AND SNIPER_REPO_URL == `https://github.com/snipersim/snipersim.git` (constant must exist, value must match canonical pin); (S2) every required directory constant (SCRIPT_DIR, PROJECT_ROOT, *_SIM_DIR, *_DIR, OVERLAYS_DIR — 6 per script) is present at module level; (S3) every canonical gem5 entry-point function (14 today: run_cmd, check_prerequisites, clone_gem5, apply_overlays, apply_patches, apply_current_vertex_pseudo_inst_patch, insert_once, apply_riscv_ecg_extract_patch, build_gem5, verify_build, install_riscv_toolchain, clean_gem5, print_summary, main) is present in setup_gem5.py; (S4) every canonical sniper entry-point function (27 today: utc_now, command_text, run_cmd, git_head, clone_or_update, write_version, build_sniper, replace_once, overlay_source_files, copy_overlay_sources, install_graphbrew_configs, write_overlay_status, patch_grasp_overlay, patch_popt_overlay, patch_ecg_overlay, patch_droplet_overlay, patch_graphbrew_simuser_overlay, patch_ecg_pfx_prefetcher_overlay, apply_overlays, compiler_for_checks, header_available, check_host_dependencies, smoke_test, graphbrew_smoke_test, clean, parse_args, main) is present in setup_sniper.py; (S5) both scripts define `def main(` (CLI entry-point invariant); (S6) both scripts define `def apply_overlays(` (overlay-install contract entry point — pairs with gates 266/267); (S7) actual top-level def set EXACTLY equals canonical registry — no unregistered helpers (would require adding to SETUP_GEM5_EXTRA_ALLOW or SETUP_SNIPER_EXTRA_ALLOW), no missing canonical entries (which would also trigger S3/S4). Today: 2 repo URLs, 6+6 = 12 directory constants, 14+27 = 41 canonical functions; 0 violations. Together with gates 266+267 this completes the FULL build-time installation surface lock: payload (266 Sniper overlay tracker, 267 gem5 overlay tracker) + orchestrator (268 setup-script invariants).",
    },
    {
        "id":        "lit_faith_config_deep_lock",
        "label":     "ECG config deep-lock",
        "generator": "scripts/experiments/ecg/lit_faith_config_deep_lock.py",
        "gate":      "scripts/test/test_lit_faith_config_deep_lock.py",
        "artifact":  "wiki/data/lit_faith_config_deep_lock.json",
        "summary":   "ECG config deep-lock registry audit (gate 269) — ALWAYS ACTIVE (source-of-truth is the live scripts/experiments/ecg/config.py module loaded via importlib.spec_from_file_location). Sixteenth in the vocabulary-lock series (252 SBATCH, 255 policy, 256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261 arm-catalog, 262 cross-tool schema, 263 config matrix, 264 filename grammar, 265 sideband grammar, 266 Sniper overlay tracker, 267 gem5 overlay tracker, 268 setup-script invariants, 269 ECG config deep-lock): gate 269 sits beside gate 263 (cross-tool config matrix) but goes deeper — where 263 locks (graph × backend × cache-size × app × policy) coverage matrix shape, gate 269 locks the CONTENTS of ecg/config.py that the runner.py + paper_pipeline.py + reproduce_smoke.py all import at runtime. Catches: a contributor lowers DEFAULT_CACHE['CACHE_L3_SIZE'] from 8 MiB to 4 MiB while debugging a single graph and forgets to revert — every subsequent confidence run silently uses the wrong baseline anchor and every gem5/Sniper paper number drifts off the literature pin; a contributor adds a new entry to CACHE_SIZES_SWEEP that breaks the power-of-2 ladder (e.g., 48 KiB between 32 KiB and 64 KiB) and the per-cache-size slope-extraction tests start producing non-monotonic curves; a contributor moves bc from TRAVERSAL_BENCHMARKS into ITERATIVE_BENCHMARKS, breaking every per-app regime classifier downstream; a contributor introduces a new ECG_MODES value ('NEW_MODE') without updating any of the 50+ downstream gates and runners crash with KeyError mid-simulation; a contributor renames a graph in EVAL_GRAPHS or removes a required schema key (name/short/type/vertices_m/edges_m); a contributor introduces a typo'd ACCURACY_PAIRS relation label that breaks the paper-claims-registry cross-reference. 8 rules C1-C8: (C1) DEFAULT_CACHE has all 7 canonical anchors (CACHE_L1_SIZE=32768, L2_SIZE=262144, L3_SIZE=8388608, L1_WAYS=8, L2_WAYS=4, L3_WAYS=16, LINE_SIZE=64) — exact string match; (C2) CACHE_SIZES_SWEEP is exactly 12 ascending power-of-2 entries from 32 KiB to 64 MiB; (C3) BENCHMARKS = ITERATIVE_BENCHMARKS ∪ TRAVERSAL_BENCHMARKS with no overlap; (C4) ALL_POLICIES = BASELINE_POLICIES ∪ GRAPH_AWARE_POLICIES with no overlap; (C5) ECG_MODES exactly equals the canonical 4-set {DBG_PRIMARY, POPT_PRIMARY, DBG_ONLY, ECG_EMBEDDED}; (C6) every EVAL_GRAPHS entry has all 5 required keys (name, short, type, vertices_m, edges_m) AND its type is one of {Social, Citation, Road, Web, Content, Mesh}; (C7) every REORDER_VARIANTS flag starts with `-o ` and is in the recognized vocabulary; (C8) every ACCURACY_PAIRS relation label is snake_case and in the canonical relation vocabulary. Today: 7 cache anchors, 12 sweep points (32 KiB..64 MiB), 7 benchmarks (3 iterative + 3 traversal + 1 reserved-bcsr), 8 policies (5 baseline + 3 graph-aware), 4 ECG modes, 6 EVAL_GRAPHS; 0 violations. Together with gate 263 (matrix shape) this completes the LIT-CONFIG lock at both schema and value layers.",
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
        "summary":   "Single-screen verdict (269 gates today, all GREEN). The dashboard this catalog sits next to.",
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
    }, indent=2, sort_keys=True) + "\n")


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
