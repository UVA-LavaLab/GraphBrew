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
        "summary":   "Single-screen verdict (53 gates today, all GREEN). The dashboard this catalog sits next to.",
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
