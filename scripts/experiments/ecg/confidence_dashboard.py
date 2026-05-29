#!/usr/bin/env python3
"""Confidence dashboard for the GraphBrew literature-faithfulness pipeline.

Why this exists
---------------
At any moment a reviewer (or the maintainer) needs to know "are we still
green?" without re-reading 6 separate reports. This script aggregates the
existing gates into one screen so the answer is unambiguous.

What it reports
---------------
* Tier A — sideband registration sanity         (test_grasp_sideband_registration)
* Tier B — POPT permutation equivalence         (test_popt_permutation_equivalence)
* Tier C — GRASP-vs-LRU sign test               (test_grasp_sign_consistency)
* Structural lit-baseline test                  (test_literature_baselines_structure)
* Data-driven lit-baseline test                 (test_baselines_match_literature)
* Corpus diversity profile parity test          (test_corpus_diversity)
* Literature-faithfulness comparator headline   (literature_faithfulness JSON summary)
* Corpus diversity coverage                     (corpus_diversity JSON)

Usage
-----
    python -m scripts.experiments.ecg.confidence_dashboard \\
        --lit-faith-json wiki/data/literature_faithfulness_postfix.json \\
        --corpus-diversity-json wiki/data/corpus_diversity.json \\
        --markdown wiki/data/confidence_dashboard.md

Tests can run in `--fast` mode (default) which skips heavyweight suites
(fill_weights_variants) — the dashboard only needs the literature /
sideband / sign tests.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]

PYTEST_SUITES: dict[str, tuple[str, str]] = {
    "Tier A — GRASP sideband registration":
        ("scripts/test/test_grasp_sideband_registration.py", "Tier A"),
    "Tier B — POPT permutation equivalence":
        ("scripts/test/test_popt_permutation_equivalence.py", "Tier B"),
    "Tier C — GRASP vs LRU sign test":
        ("scripts/test/test_grasp_sign_consistency.py", "Tier C"),
    "Lit-baseline data-driven gate":
        ("scripts/test/test_baselines_match_literature.py", "Lit-data"),
    "Lit-baseline structural gate":
        ("scripts/test/test_literature_baselines_structure.py", "Lit-struct"),
    "Lit-faith no-disagree gate":
        ("scripts/test/test_lit_faith_no_disagree.py", "Lit-faith"),
    "ECG validation gate catalog":
        ("scripts/test/test_ecg_validation_gates_catalog.py", "ECG-cat"),
    "Corpus diversity floor":
        ("scripts/test/test_corpus_diversity_floor.py", "Corpus-floor"),
    "Cross-tool report parity":
        ("scripts/test/test_cross_tool_parity.py", "Parity"),
    "Regression budget floor":
        ("scripts/test/test_regression_budget_floor.py", "Budget"),
    "Regression budget margin distribution":
        ("scripts/test/test_regression_budget.py", "Budget-dist"),
    "Paper-claims registry integrity":
        ("scripts/test/test_paper_claims_integrity.py", "Claims-Int"),
    "Cross-artifact aggregate consistency":
        ("scripts/test/test_cross_artifact_consistency.py", "X-Art"),
    "Per-app oracle-vs-winner parity":
        ("scripts/test/test_oracle_winner_per_app_parity.py", "Per-App-Par"),
    "Artifact-catalog completeness":
        ("scripts/test/test_catalog_completeness.py", "Cat-Comp"),
    "Family-sensitivity cross-artifact":
        ("scripts/test/test_family_sensitivity_cross_artifact.py", "Fam-Sens-X"),
    "Cross-tool slope-ordering cross-artifact":
        ("scripts/test/test_cross_tool_slope_ordering_xartifact.py", "Slope-Ord-X"),
    "Gem5/Sniper anchor cell parity":
        ("scripts/test/test_gem5_sniper_anchor_cell_parity.py", "Anch-Par"),
    "Cell-count cross-artifact parity":
        ("scripts/test/test_cell_count_cross_artifact_parity.py", "Cell-Par"),
    "Cache-sensitivity slope baseline":
        ("scripts/test/test_cache_sensitivity_slope_baseline.py", "Slope-Base"),
    "WSS knee vs relative-L3 parity":
        ("scripts/test/test_wss_knee_vs_relative_l3_parity.py", "WSS-Par"),
    "Bootstrap CI nested consistency":
        ("scripts/test/test_bootstrap_ci_nested_consistency.py", "Boot-CI"),
    "Family clustering 3-way agreement":
        ("scripts/test/test_family_clustering_3way_agreement.py", "Fam-3way"),
    "AUC correlation cross-artifact parity":
        ("scripts/test/test_auc_correlation_cross_artifact_parity.py", "AUC-Par"),
    "Family geomean vs margin replay parity":
        ("scripts/test/test_family_geomean_vs_margin_replay_parity.py", "Geo-Marg"),
    "Oracle-gap curvature vs effect-size parity":
        ("scripts/test/test_oracle_gap_curvature_vs_effect_size.py", "Curv-Eff"),
    "Monotonicity universality vs anchor replay":
        ("scripts/test/test_monotonicity_universality_vs_anchor_replay.py", "Mono-Anch"),
    "Catalog ↔ dashboard coverage milestone":
        ("scripts/test/test_catalog_dashboard_coverage_milestone.py", "Cat-Dash"),
    "Deviations vs regime taxonomy parity":
        ("scripts/test/test_deviations_vs_regime_taxonomy.py", "Dev-Reg"),
    "Corpus diversity vs regime taxonomy parity":
        ("scripts/test/test_corpus_diversity_vs_regime_taxonomy.py", "Cor-Reg"),
    "Paper claims registry recompute parity":
        ("scripts/test/test_paper_claims_recompute.py", "PC-Recom"),
    "Family tri-artifact agreement":
        ("scripts/test/test_family_tri_artifact_agreement.py", "Fam-Tri"),
    "Regression budget vs lit_faith parity":
        ("scripts/test/test_regression_budget_vs_lit_faith.py", "Reg-Faith"),
    "Oracle-gap internal and by-app aggregation":
        ("scripts/test/test_oracle_gap_internal_and_by_app.py", "Ora-Ag"),
    "GRAPH_FAMILY map duplication":
        ("scripts/test/test_graph_family_map_duplication.py", "GF-Dup"),
    "Claim density vs literature baselines":
        ("scripts/test/test_claim_density_vs_baselines.py", "CD-Base"),
    "small_l3_thrash internal + WRT-tiny disjointness":
        ("scripts/test/test_small_l3_thrash_consistency.py", "Thr-Tiny"),
    "bootstrap_ci + oracle_gap_by_app_bootstrap parity":
        ("scripts/test/test_bootstrap_ci_consistency.py", "Boot-Par"),
    "oracle_gap_curvature arithmetic + knee rule":
        ("scripts/test/test_oracle_gap_curvature_arithmetic.py", "OG-Curv"),
    "gap_distribution_shape Hesterberg envelope":
        ("scripts/test/test_gap_distribution_shape_envelope.py", "GDS-Env"),
    "distribution_diagnostics envelope + marginals":
        ("scripts/test/test_distribution_diagnostics_envelope.py", "DD-Env"),
    "cross_tool_winners classification arithmetic":
        ("scripts/test/test_cross_tool_winners_classification.py", "CTW-Cls"),
    "cohens_h ↔ wilson win-rate parity":
        ("scripts/test/test_cohens_h_wilson_parity.py", "CH-Wil"),
    "lofo ↔ leave_one_graph_out robustness parity":
        ("scripts/test/test_lofo_logo_robustness_parity.py", "LOFO-Par"),
    "multiple_testing_correction HB+BH arithmetic":
        ("scripts/test/test_multiple_testing_correction_arithmetic.py", "MTC-Lad"),
    "cache_saturation_onset step-down rule + ranking":
        ("scripts/test/test_cache_saturation_onset_arithmetic.py", "CSO-Step"),
    "cross_tool_slope_universality verdict + medians":
        ("scripts/test/test_cross_tool_slope_universality_arithmetic.py", "CTSU-Med"),
    "gem5_slope_replay OLS + verdict":
        ("scripts/test/test_gem5_slope_replay_arithmetic.py", "GSR-OLS"),
    "family_slope_replay arithmetic + verdict":
        ("scripts/test/test_family_slope_replay_arithmetic.py", "FSR-Inv"),
    "policy_steepness_ranking checks + arithmetic":
        ("scripts/test/test_policy_steepness_ranking_arithmetic.py", "PSR-Chk"),
    "cross_policy_asymmetry head-to-head arithmetic":
        ("scripts/test/test_cross_policy_asymmetry_arithmetic.py", "CPA-H2H"),
    "saturation_slope_extremum arithmetic + verdict":
        ("scripts/test/test_saturation_slope_extremum_arithmetic.py", "SSE-Ext"),
    "winner_margin_gradient arithmetic + classification":
        ("scripts/test/test_winner_margin_gradient_arithmetic.py", "WMG-Grad"),
    "per_app_srrip_vs_grasp arithmetic + verdict":
        ("scripts/test/test_per_app_srrip_vs_grasp_arithmetic.py", "PAS-vG"),
    "corpus_balance arithmetic + diversity metrics":
        ("scripts/test/test_corpus_balance_arithmetic.py", "CB-Div"),
    "family_curvature_replay arithmetic + verdict":
        ("scripts/test/test_family_curvature_replay_arithmetic.py", "FCR-Curv"),
    "cross_generator_gap_parity arithmetic + spread":
        ("scripts/test/test_cross_generator_gap_parity_arithmetic.py", "CGGP-Par"),
    "slope_saturation_xcheck arithmetic + statistics":
        ("scripts/test/test_slope_saturation_xcheck_arithmetic.py", "SSX-Stat"),
    "per_graph_app_stability arithmetic + classification":
        ("scripts/test/test_per_graph_app_stability_arithmetic.py", "PGAS-Stab"),
    "paper_baseline_table arithmetic + parity":
        ("scripts/test/test_paper_baseline_table_arithmetic.py", "PBT-Par"),
    "literature_faithfulness_postfix arithmetic + parity":
        ("scripts/test/test_literature_faithfulness_postfix_arithmetic.py", "LFP-Par"),
    "cross-artifact miss + delta parity":
        ("scripts/test/test_cross_artifact_miss_delta_parity.py", "XAP-Par"),
    "winner identification parity (oracle ↔ stability)":
        ("scripts/test/test_winner_identification_parity.py", "WID-Par"),
    "oracle_gap arithmetic + is_winner + summary rollups":
        ("scripts/test/test_oracle_gap_arithmetic.py", "OGA-Stat"),
    "oracle_gap_auc derivation parity (trapezoidal on log2)":
        ("scripts/test/test_oracle_gap_auc_derivation_parity.py", "AUC-Der"),
    "cache_sensitivity_slope derivation parity (auc → slope)":
        ("scripts/test/test_cache_sensitivity_slope_derivation_parity.py", "CSS-Der"),
    "per_graph_rollup arithmetic + meta counts":
        ("scripts/test/test_per_graph_rollup_arithmetic.py", "PGR-Math"),
    "policy_auc_correlation derivation parity (z-score + Pearson)":
        ("scripts/test/test_policy_auc_correlation_derivation_parity.py", "PAC-Der"),
    "family_policy_auc_clustering derivation parity (pooled → trapezoid → corr)":
        ("scripts/test/test_family_policy_auc_clustering_derivation_parity.py", "FPC-Der"),
    "cohens_h_win_rates derivation parity (arcsine + magnitude buckets)":
        ("scripts/test/test_cohens_h_win_rates_derivation_parity.py", "CHW-Der"),
    "wilson_win_rates derivation parity (z + score interval + 3 scopes)":
        ("scripts/test/test_wilson_win_rates_derivation_parity.py", "WWR-Der"),
    "bootstrap_ci derivation parity (aggregation + width + sign logic)":
        ("scripts/test/test_bootstrap_ci_derivation_parity.py", "BCI-Der"),
    "gap_distribution_shape derivation parity (g1/g2 moments + envelope)":
        ("scripts/test/test_gap_distribution_shape_derivation_parity.py", "GDS-Der"),
    "cell_winner_census derivation parity (classification + tied breakdown)":
        ("scripts/test/test_cell_winner_census_derivation_parity.py", "CWC-Der"),
    "policy_winner_table derivation parity (argmin + summary aggregation)":
        ("scripts/test/test_policy_winner_table_derivation_parity.py", "PWT-Der"),
    "monotonicity_universality derivation parity (bump count + verdict logic)":
        ("scripts/test/test_monotonicity_universality_derivation_parity.py", "MUN-Der"),
    "popt_vs_grasp_delta derivation parity (delta + classification + tails)":
        ("scripts/test/test_popt_vs_grasp_delta_derivation_parity.py", "PVG-Der"),
    "winning_regime_taxonomy derivation parity (bins + extracted rules)":
        ("scripts/test/test_winning_regime_taxonomy_derivation_parity.py", "WRT-Der"),
    "cross_tool_winners derivation parity (max-L3 collapse + classification)":
        ("scripts/test/test_cross_tool_winners_derivation_parity.py", "CTW-Der"),
    "regression_budget derivation parity (per-cell distance to disagree)":
        ("scripts/test/test_regression_budget_derivation_parity.py", "RBD-Der"),
    "family_geomean_improvement derivation parity (geomean ratio + bootstrap CI)":
        ("scripts/test/test_family_geomean_improvement_derivation_parity.py", "FGI-Der"),
    "policy_rank_kendall derivation parity (Kendall-tau across L3 octave)":
        ("scripts/test/test_policy_rank_kendall_derivation_parity.py", "PRK-Der"),
    "claim_density derivation parity (per-graph claim aggregation)":
        ("scripts/test/test_claim_density_derivation_parity.py", "CLD-Der"),
    "paper_baseline_table derivation parity (per_observation + PER_GRAPH_CLAIMS)":
        ("scripts/test/test_paper_baseline_table_derivation_parity.py", "PBT-Der"),
    "cross_tool_lru_regime derivation parity (per-tool slope medians + regime classifier)":
        ("scripts/test/test_cross_tool_lru_regime_derivation_parity.py", "CTLR-Der"),
    "cross_tool_slope_universality derivation parity (per-(tool,policy) slope sign + physical band)":
        ("scripts/test/test_cross_tool_slope_universality_derivation_parity.py", "CTSU-Der"),
    "cross_tool_slope_ordering derivation parity (SRRIP-vs-GRASP gap_floor + strict-tool count)":
        ("scripts/test/test_cross_tool_slope_ordering_derivation_parity.py", "CTSO-Der"),
    "policy_steepness_ranking derivation parity (final-octave |slope| reducers + ordering checks)":
        ("scripts/test/test_policy_steepness_ranking_derivation_parity.py", "PSR-Der"),
    "winner_margin_by_regime derivation parity (per-(policy, wss_regime) margin reducers + shrink-evidence)":
        ("scripts/test/test_winner_margin_by_regime_derivation_parity.py", "WMR-Der"),
    "Corpus diversity profile parity":
        ("scripts/test/test_corpus_diversity.py", "Corpus"),
    "Paper-pipeline literature pre-flight gate":
        ("scripts/test/test_paper_pipeline_lit_gate.py", "Preflight"),
    "gem5 literature anchor":
        ("scripts/test/test_gem5_anchor.py", "Gem5-anchor"),
    "Sniper literature anchor":
        ("scripts/test/test_sniper_anchor.py", "Sniper-anchor"),
    "Lit-preflight shared helper":
        ("scripts/test/test_literature_preflight.py", "Lit-helper"),
    "GRASP road-like graph invariant":
        ("scripts/test/test_road_like_graph_invariant.py", "Road-like"),
    "L-curve monotonicity gate":
        ("scripts/test/test_l_curve_monotonicity.py", "L-mono"),
    "Policy-winner table sanity":
        ("scripts/test/test_policy_winner_table.py", "Winner"),
    "Small-L3 thrash sanity":
        ("scripts/test/test_small_l3_thrash.py", "Thrash"),
    "Cross-tool saturation soundness":
        ("scripts/test/test_cross_tool_saturation.py", "X-tool"),
    "Cross-tool winner agreement":
        ("scripts/test/test_cross_tool_winners.py", "X-win"),
    "Per-graph claim density":
        ("scripts/test/test_claim_density.py", "Density"),
    "POPT-vs-GRASP delta":
        ("scripts/test/test_popt_vs_grasp_delta.py", "P-vs-G"),
    "Literature deviations inventory":
        ("scripts/test/test_literature_deviations.py", "Lit-dev"),
    "Paper claims registry":
        ("scripts/test/test_paper_claims_registry.py", "Claims"),
    "Winning-regime taxonomy":
        ("scripts/test/test_winning_regime_taxonomy.py", "Regime"),
    "Oracle gap":
        ("scripts/test/test_oracle_gap.py", "Oracle"),
    "Per-kernel oracle gap":
        ("scripts/test/test_oracle_gap_by_app.py", "OracleApp"),
    "Per-kernel bootstrap CIs":
        ("scripts/test/test_oracle_gap_by_app_bootstrap.py", "AppBoot"),
    "POPT-vs-GRASP per (family x app) CIs":
        ("scripts/test/test_popt_vs_grasp_by_family_app.py", "FamApp"),
    "Wilson CIs on win-counts":
        ("scripts/test/test_wilson_win_rates.py", "Wilson"),
    "Cohen's h on win-rate gaps":
        ("scripts/test/test_cohens_h_win_rates.py", "Cohen"),
    "Cliff's delta on gap distributions":
        ("scripts/test/test_oracle_gap_effect_size.py", "Cliff"),
    "Per-L3 policy stability":
        ("scripts/test/test_l3_policy_stability.py", "L3-Stab"),
    "Multiple-testing correction":
        ("scripts/test/test_multiple_testing_correction.py", "MT-Corr"),
    "Leave-one-graph-out robustness":
        ("scripts/test/test_leave_one_graph_out.py", "LOGO"),
    "Cell-winner census":
        ("scripts/test/test_cell_winner_census.py", "Census"),
    "Family geomean improvement vs LRU":
        ("scripts/test/test_family_geomean_improvement.py", "FamGeo"),
    "Per-(graph, app) L3 stability":
        ("scripts/test/test_per_graph_app_stability.py", "GAStab"),
    "Corpus tier/family balance audit":
        ("scripts/test/test_corpus_balance.py", "Balance"),
    "Per-policy miss-rate distribution diagnostics":
        ("scripts/test/test_distribution_diagnostics.py", "Dist"),
    "Leave-one-family-out (LOFO) robustness":
        ("scripts/test/test_lofo_robustness.py", "LOFO"),
    "Per-(app, L3) winner-margin gradient":
        ("scripts/test/test_winner_margin_gradient.py", "Margin"),
    "Per-(app, policy) oracle-gap AUC":
        ("scripts/test/test_oracle_gap_auc.py", "AUC"),
    "Cross-app policy-AUC correlation matrix":
        ("scripts/test/test_policy_auc_correlation.py", "Corr"),
    "Per-policy stability index":
        ("scripts/test/test_policy_stability.py", "Stab"),
    "Per-(app, policy) cache-sensitivity slope":
        ("scripts/test/test_cache_sensitivity_slope.py", "CSlope"),
    "Per-graph oracle-gap cache-sensitivity slope":
        ("scripts/test/test_per_graph_cache_slope.py", "PGSlope"),
    "Cross-generator gap_pp parity":
        ("scripts/test/test_cross_generator_gap_parity.py", "GapPar"),
    "Cache-saturation onset detection":
        ("scripts/test/test_cache_saturation_onset.py", "SatOn"),
    "Per-cell gap-distribution shape envelope":
        ("scripts/test/test_gap_distribution_shape.py", "Shape"),
    "Per-family policy-AUC clustering replay":
        ("scripts/test/test_family_policy_auc_clustering.py", "FamAUC"),
    "Oracle-gap trajectory curvature":
        ("scripts/test/test_oracle_gap_curvature.py", "Knee"),
    "Policy-rank Kendall-tau across L3 octave":
        ("scripts/test/test_policy_rank_kendall.py", "Tau"),
    "WSS-relative knee location":
        ("scripts/test/test_wss_knee_location.py", "WssKn"),
    "Per-family curvature replay":
        ("scripts/test/test_family_curvature_replay.py", "FamKn"),
    "Winner-margin distribution by WSS regime":
        ("scripts/test/test_winner_margin_by_regime.py", "WMrgn"),
    "Per-family winner-margin replay":
        ("scripts/test/test_family_margin_replay.py", "FMrgn"),
    "Cross-policy mean-margin asymmetry":
        ("scripts/test/test_cross_policy_asymmetry.py", "Asym"),
    "Per-app saturation distance (4MB->8MB)":
        ("scripts/test/test_saturation_distance.py", "SatDist"),
    "Per-policy capacity-sensitivity slope":
        ("scripts/test/test_capacity_sensitivity.py", "CapSlope"),
    "Per-family capacity-sensitivity slope replay":
        ("scripts/test/test_family_slope_replay.py", "FSlope"),
    "Per-app capacity-sensitivity slope":
        ("scripts/test/test_per_app_capacity_slope.py", "ASlope"),
    "Slope vs saturation-distance cross-check":
        ("scripts/test/test_slope_saturation_xcheck.py", "Xchk"),
    "Gem5 anchor slope sanity":
        ("scripts/test/test_gem5_slope_replay.py", "G5slope"),
    "Sniper anchor slope sanity":
        ("scripts/test/test_sniper_slope_replay.py", "SNslope"),
    "Cross-tool SRRIP-vs-GRASP slope ordering":
        ("scripts/test/test_cross_tool_slope_ordering.py", "XTslope"),
    "Per-app SRRIP-vs-GRASP slope ordering":
        ("scripts/test/test_per_app_srrip_vs_grasp.py", "AppSGR"),
    "Cross-tool LRU-vs-GRASP regime inversion":
        ("scripts/test/test_cross_tool_lru_regime.py", "XTregm"),
    "Per-app saturation-vs-slope extremum":
        ("scripts/test/test_saturation_slope_extremum.py", "SatExt"),
    "Cross-tool slope-sign universality":
        ("scripts/test/test_cross_tool_slope_universality.py", "XTsign"),
    "L3-sweep monotonicity universality":
        ("scripts/test/test_monotonicity_universality.py", "Monoton"),
    "Anchor cell-pair census":
        ("scripts/test/test_anchor_cell_census.py", "Anchors"),
    "Per-family saturation-distance replay":
        ("scripts/test/test_family_saturation_distance.py", "FamSat"),
    "Anchor monotonicity replay (gem5+sniper)":
        ("scripts/test/test_anchor_monotonicity_replay.py", "AnchMono"),
    "Per-policy final-octave steepness ranking":
        ("scripts/test/test_policy_steepness_ranking.py", "Steepness"),
    "Cross-tool shared-anchor slope agreement":
        ("scripts/test/test_anchor_cross_tool_agreement.py", "CrossAnch"),
    "WSS-relative L3 axis":
        ("scripts/test/test_wss_relative_l3.py", "WSS-L3"),
    "Bootstrap CIs on load-bearing claims":
        ("scripts/test/test_bootstrap_ci.py", "Bootstrap"),
    "Family-classification sensitivity":
        ("scripts/test/test_family_sensitivity.py", "FamSens"),
    "Reproducibility smoke":
        ("scripts/test/test_reproduce_smoke.py", "Repro"),
    "Artifact catalog completeness":
        ("scripts/test/test_artifact_catalog.py", "Catalog"),
}


@dataclass
class SuiteResult:
    label: str
    short: str
    path: str
    passed: int
    failed: int
    skipped: int
    xfailed: int
    xpassed: int
    errors: int
    runtime_s: float
    raw_tail: str

    @property
    def is_green(self) -> bool:
        return self.failed == 0 and self.errors == 0


SUMMARY_RE = re.compile(
    r"(?P<passed>\d+)\s+passed"
    r"(?:,\s+(?P<skipped>\d+)\s+skipped)?"
    r"(?:,\s+(?P<xfailed>\d+)\s+xfailed)?"
    r"(?:,\s+(?P<xpassed>\d+)\s+xpassed)?"
    r"(?:,\s+(?P<failed>\d+)\s+failed)?"
    r"(?:,\s+(?P<errors>\d+)\s+errors?)?"
)


def _parse_pytest_summary(text: str) -> dict[str, int]:
    """Pull pass/fail/skipped/xfailed/xpassed/errors from a pytest summary line."""
    out = {k: 0 for k in ("passed", "failed", "skipped", "xfailed", "xpassed", "errors")}
    for line in text.splitlines()[::-1]:
        # The summary line looks like:  "== 6 passed, 1 skipped in 0.12s =="
        # or "== 6 passed, 1 failed in 0.12s =="; numbers may appear in any
        # order so we scan for individual keywords.
        if " passed" not in line and " failed" not in line and " error" not in line:
            continue
        for key in out:
            m = re.search(rf"(\d+)\s+{key}", line)
            if m:
                out[key] = int(m.group(1))
        if any(out.values()):
            break
    return out


def _run_suite(label: str, short: str, path: str, pytest_args: Sequence[str]) -> SuiteResult:
    cmd = [sys.executable, "-m", "pytest", path, "-q", "--no-header", "--tb=no"] + list(pytest_args)
    started = time.time()
    completed = subprocess.run(  # noqa: S603 — fixed argv
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - started
    text = completed.stdout + completed.stderr
    counts = _parse_pytest_summary(text)
    tail = "\n".join(text.splitlines()[-15:])
    return SuiteResult(
        label=label, short=short, path=path,
        passed=counts["passed"], failed=counts["failed"], skipped=counts["skipped"],
        xfailed=counts["xfailed"], xpassed=counts["xpassed"], errors=counts["errors"],
        runtime_s=elapsed, raw_tail=tail,
    )


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _lit_faith_section(lit) -> list[str]:
    out = ["## Literature-faithfulness comparator", ""]
    if not isinstance(lit, dict):
        out += ["_No literature_faithfulness JSON found — run "
                "`python -m scripts.experiments.ecg.literature_faithfulness ...` first._", ""]
        return out
    s = lit.get("summary", {})
    total = s.get("claims_total", 0)
    ok = s.get("ok", 0)
    disagree = s.get("disagree", 0)
    known = s.get("known_deviation", 0)
    within = s.get("within_tolerance", 0)
    insuf = s.get("insufficient_data", 0)
    missing = s.get("missing", 0)
    ok_pct = (100.0 * ok / total) if total else 0.0
    verdict = "✅ green" if disagree == 0 else f"⛔ {disagree} unexplained disagreements"
    out += [
        f"**Verdict:** {verdict}  ({ok}/{total} ok = {ok_pct:.1f}%)",
        "",
        "| status | count |",
        "|---|---:|",
        f"| ok | {ok} |",
        f"| within_tolerance | {within} |",
        f"| **DISAGREE** | **{disagree}** |",
        f"| known_deviation | {known} |",
        f"| insufficient_data | {insuf} |",
        f"| missing | {missing} |",
        f"| **total claims** | **{total}** |",
        "",
    ]
    return out


def _budget_section(budget) -> list[str]:
    out = ["## Regression budget — distance to disagree", ""]
    if not isinstance(budget, dict):
        out += ["_No regression_budget JSON found — run "
                "`make lit-budget` to generate one._", ""]
        return out
    s = budget.get("summary", {})
    by_kind = s.get("by_kind", {})
    out += [
        f"- Cells in distribution: **{s.get('cells_in_distribution', 0)}**",
        f"- Min margin (any kind): **{s.get('min_margin_pp', 0):.3f} pp**",
        f"- Median margin: {s.get('median_margin_pp', 0):.3f} pp",
        f"- p90 margin: {s.get('p90_margin_pp', 0):.3f} pp",
        "",
        "| claim kind | n | min margin (pp) | median margin (pp) |",
        "|---|---:|---:|---:|",
    ]
    for k, v in by_kind.items():
        out.append(
            f"| {k} | {v.get('n', 0)} | {v.get('min_pp', 0):.3f} "
            f"| {v.get('median_pp', 0):.3f} |"
        )
    out.append("")
    fragile = (budget.get("fragile_cache_policy_cells") or [])[:5]
    if fragile:
        out += ["**5 most fragile cache-policy cells:**", ""]
        out += ["| graph | app | l3 | policy | Δ (pp) | margin (pp) |"]
        out += ["|---|---|---|---|---:|---:|"]
        for r in fragile:
            out.append(
                f"| {r['graph']} | {r['app']} | {r['l3_size']} "
                f"| {r['policy']} | {r['delta_pct']:+.3f} "
                f"| {r['margin_pp']:.3f} |"
            )
        out.append("")
    return out


def _corpus_section(corpus) -> list[str]:
    out = ["## Corpus diversity coverage", ""]
    if corpus is None:
        out += ["_No corpus_diversity JSON found — run "
                "`python -m scripts.experiments.ecg.corpus_diversity ...` first._", ""]
        return out
    # corpus_diversity emits a top-level list; older callers may wrap it.
    if isinstance(corpus, dict):
        cards = corpus.get("graphs") or corpus.get("rows") or []
    else:
        cards = corpus or []
    if not cards:
        out += ["_corpus_diversity JSON has no graph cards._", ""]
        return out
    out += [f"**Graphs profiled:** {len(cards)}", ""]
    out += ["| graph | nodes | edges | hub_conc | avg_deg | clustering_sampled | working_set_ratio |"]
    out += ["|---|---:|---:|---:|---:|---:|---:|"]
    for card in cards:
        name = card.get("graph", "?")
        feats = card.get("features", card)

        def _g(k: str, src=feats, top=card):
            v = src.get(k) if isinstance(src, dict) else None
            if v is None and isinstance(top, dict):
                v = top.get(k)
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        out.append(
            "| " + " | ".join([
                name,
                _g("nodes", src=card),
                _g("edges", src=card),
                _g("hub_concentration"),
                _g("avg_degree"),
                _g("clustering_coeff"),
                _g("working_set_ratio"),
            ]) + " |"
        )
    out.append("")
    return out


def _headline_verdict(results: list[SuiteResult], lit) -> str:
    red = [r.label for r in results if not r.is_green]
    if red:
        return "⛔ RED — " + ", ".join(red) + " failing"
    if isinstance(lit, dict) and lit.get("summary", {}).get("disagree", 0) > 0:
        n = lit["summary"]["disagree"]
        return f"⛔ RED — {n} unexplained disagreements in lit-faith comparator"
    return "✅ GREEN — every tier + gate + comparator is within tolerance"


def _pytest_section(results: list[SuiteResult]) -> list[str]:
    out = ["## Tier & gate pytest results", ""]
    out += ["| gate | pass | skip | xfail | fail | err | runtime | verdict |"]
    out += ["|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for r in results:
        verdict = "✅" if r.is_green else "⛔"
        out.append(
            f"| {r.label} | {r.passed} | {r.skipped} | {r.xfailed} | "
            f"{r.failed} | {r.errors} | {r.runtime_s:.1f}s | {verdict} |"
        )
    out.append("")
    failing = [r for r in results if not r.is_green]
    if failing:
        out += ["### ⛔ Failing tail (last 15 lines per failing gate)", ""]
        for r in failing:
            out += [f"#### {r.label}", "```", r.raw_tail, "```", ""]
    return out


def render(results: list[SuiteResult], lit, corpus, budget=None) -> str:
    out = [
        "# GraphBrew literature-faithfulness confidence dashboard",
        "",
        "_Generated by `scripts/experiments/ecg/confidence_dashboard.py`._",
        "",
        f"## Headline: {_headline_verdict(results, lit)}",
        "",
    ]
    out += _pytest_section(results)
    out += _lit_faith_section(lit)
    out += _budget_section(budget)
    out += _corpus_section(corpus)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lit-faith-json",
        default=str(REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"),
        help="Path to literature_faithfulness JSON output.",
    )
    parser.add_argument(
        "--corpus-diversity-json",
        default=str(REPO_ROOT / "wiki" / "data" / "corpus_diversity.json"),
        help="Path to corpus_diversity JSON output.",
    )
    parser.add_argument(
        "--regression-budget-json",
        default=str(REPO_ROOT / "wiki" / "data" / "regression_budget.json"),
        help="Path to regression_budget JSON output.",
    )
    parser.add_argument(
        "--markdown", default=None,
        help="Optional path to write the rendered dashboard markdown.",
    )
    parser.add_argument(
        "--skip-pytest", action="store_true",
        help="Skip running pytest; only render the data sections.",
    )
    parser.add_argument(
        "--include-slow", action="store_true",
        help="Include slow tier suites (none currently; reserved for future).",
    )
    parser.add_argument(
        "--pytest-arg", action="append", default=[],
        help="Extra arg to forward to pytest (repeatable).",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="Optional path to dump a machine-readable summary.",
    )
    args = parser.parse_args()

    results: list[SuiteResult] = []
    if not args.skip_pytest:
        for label, (path, short) in PYTEST_SUITES.items():
            results.append(_run_suite(label, short, path, args.pytest_arg))

    lit = _read_json(Path(args.lit_faith_json))
    corpus = _read_json(Path(args.corpus_diversity_json))
    budget = _read_json(Path(args.regression_budget_json))

    rendered = render(results, lit, corpus, budget)
    print(rendered)
    if args.markdown:
        Path(args.markdown).write_text(rendered.rstrip("\n") + "\n")
        print(f"[dashboard] markdown -> {args.markdown}", file=sys.stderr)

    if args.json_out:
        graph_count = 0
        if isinstance(corpus, dict):
            graph_count = len(corpus.get("graphs", []) or corpus.get("rows", []))
        elif isinstance(corpus, list):
            graph_count = len(corpus)
        Path(args.json_out).write_text(json.dumps({
            "headline": _headline_verdict(results, lit),
            "suites": [r.__dict__ for r in results],
            "lit_faith_summary": (lit or {}).get("summary") if isinstance(lit, dict) else None,
            "regression_budget_summary": (budget or {}).get("summary") if isinstance(budget, dict) else None,
            "corpus_graph_count": graph_count,
        }, indent=2) + "\n")
        print(f"[dashboard] json -> {args.json_out}", file=sys.stderr)

    return 0 if "GREEN" in _headline_verdict(results, lit) else 1


if __name__ == "__main__":
    sys.exit(main())
