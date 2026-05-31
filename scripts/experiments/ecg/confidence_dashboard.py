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
    "wss_relative_l3 derivation parity (WSS proxy + L3/WSS regime aggregation + ranking)":
        ("scripts/test/test_wss_relative_l3_derivation_parity.py", "WSL-Der"),
    "slope_saturation_xcheck derivation parity (OLS slope vs upper-octave distance + correlation)":
        ("scripts/test/test_slope_saturation_xcheck_derivation_parity.py", "SSX-Der"),
    "winner_margin_gradient derivation parity (per-(app, L3) win counts + classifier)":
        ("scripts/test/test_winner_margin_gradient_derivation_parity.py", "WMG-Der"),
    "saturation_distance derivation parity (4MB->8MB best-policy gap + pico-sentinel)":
        ("scripts/test/test_saturation_distance_derivation_parity.py", "SD-Der"),
    "per_app_capacity_slope derivation parity (per-(app, policy) OLS slope + cache-hungriness ranking)":
        ("scripts/test/test_per_app_capacity_slope_derivation_parity.py", "PACS-Der"),
    "oracle_gap_by_app derivation parity (per-(policy, app) bucket + per-app ranking)":
        ("scripts/test/test_oracle_gap_by_app_derivation_parity.py", "OGA-Der"),
    "cross_tool_saturation derivation parity (per-cell classifier + summary reducers)":
        ("scripts/test/test_cross_tool_saturation_derivation_parity.py", "CTS-Der"),
    "per_app_srrip_vs_grasp derivation parity (per-app delta + pinned bfs)":
        ("scripts/test/test_per_app_srrip_vs_grasp_derivation_parity.py", "PSG-Der"),
    "cross_policy_asymmetry derivation parity (head-to-head margins + ratio ceiling)":
        ("scripts/test/test_cross_policy_asymmetry_derivation_parity.py", "CPA-Der"),
    "saturation_slope_extremum derivation parity (bfs argmin both axes + corpus floors)":
        ("scripts/test/test_saturation_slope_extremum_derivation_parity.py", "SSE-Der"),
    "oracle_gap_effect_size derivation parity (Cliff's delta + Mann-Whitney U)":
        ("scripts/test/test_oracle_gap_effect_size_derivation_parity.py", "OGE-Der"),
    "oracle_gap_curvature derivation parity (log2-octave second derivative + knee threshold)":
        ("scripts/test/test_oracle_gap_curvature_derivation_parity.py", "OGC-Der"),
    "oracle_gap_by_app_bootstrap derivation parity (paired Δ seeded bootstrap + 95% CI)":
        ("scripts/test/test_oracle_gap_by_app_bootstrap_derivation_parity.py", "OGB-Der"),
    "cross_generator_gap_parity derivation parity (three-source gap_pp reconciliation)":
        ("scripts/test/test_cross_generator_gap_parity_derivation_parity.py", "CGP-Der"),
    "lofo_robustness derivation parity (leave-one-family-out winner partition)":
        ("scripts/test/test_lofo_robustness_derivation_parity.py", "LFR-Der"),
    "leave_one_graph_out derivation parity (LOGO per-app re-rank + fragile partition)":
        ("scripts/test/test_leave_one_graph_out_derivation_parity.py", "LGO-Der"),
    "popt_vs_grasp_by_family_app derivation parity (per-(family,app) paired bootstrap CIs)":
        ("scripts/test/test_popt_vs_grasp_by_family_app_derivation_parity.py", "PGF-Der"),
    "cache_saturation_onset derivation parity (per-(app,policy) octave walker + per-policy rank)":
        ("scripts/test/test_cache_saturation_onset_derivation_parity.py", "CSO-Der"),
    "oracle_gap derivation parity (THE upstream: per-cell rows + summary stat block from CSV)":
        ("scripts/test/test_oracle_gap_derivation_parity.py", "OGR-Der"),
    "policy_stability derivation parity (per-policy CV across apps + rank stability + headlines)":
        ("scripts/test/test_policy_stability_derivation_parity.py", "PST-Der"),
    "l3_policy_stability derivation parity (per-(app, l3) winner aggregation + cross-L3 stability summary)":
        ("scripts/test/test_l3_policy_stability_derivation_parity.py", "L3S-Der"),
    "multiple_testing_correction derivation parity (HB step-down + BH step-up + p-value collection)":
        ("scripts/test/test_multiple_testing_correction_derivation_parity.py", "MTC-Der"),
    "corpus_balance derivation parity (Shannon entropy + Pielou evenness + Simpson + dominance + L3 coverage)":
        ("scripts/test/test_corpus_balance_derivation_parity.py", "CBL-Der"),
    "capacity_sensitivity derivation parity (per-cell OLS slope + per-policy distribution + 3-clause verdict)":
        ("scripts/test/test_capacity_sensitivity_derivation_parity.py", "CSE-Der"),
    "per_graph_cache_slope derivation parity (per-octave slope + anti-scaling cells + per-policy/graph counters)":
        ("scripts/test/test_per_graph_cache_slope_derivation_parity.py", "PCS-Der"),
    "per_graph_app_stability derivation parity (4-class winner stability + per-graph rollup + tie tolerance)":
        ("scripts/test/test_per_graph_app_stability_derivation_parity.py", "PGS-Der"),
    "distribution_diagnostics derivation parity (Fisher g1/g2 skewness/kurtosis + bootstrap validity envelope)":
        ("scripts/test/test_distribution_diagnostics_derivation_parity.py", "DDG-Der"),
    "family_curvature_replay derivation parity (log2-axis 2nd-derivative + per-family replay sign test)":
        ("scripts/test/test_family_curvature_replay_derivation_parity.py", "FCR-Der"),
    "family_margin_replay derivation parity (WSS-regime classifier + winner-margin distribution + shrink evidence)":
        ("scripts/test/test_family_margin_replay_derivation_parity.py", "FMR-Der"),
    "family_slope_replay derivation parity (per-family OLS slope replay + pinned social deviation)":
        ("scripts/test/test_family_slope_replay_derivation_parity.py", "FSR-Der"),
    "family_sensitivity derivation parity (relabel sweep + sign-claim flip detector + seed-stable bootstrap)":
        ("scripts/test/test_family_sensitivity_derivation_parity.py", "FSE-Der"),
    "family_saturation_distance derivation parity (per-family upper-octave headroom + pinned web outlier)":
        ("scripts/test/test_family_saturation_distance_derivation_parity.py", "FSD-Der"),
    "anchor_cell_census derivation parity (gem5+Sniper anchor coverage + 13-check verdict matrix)":
        ("scripts/test/test_anchor_cell_census_derivation_parity.py", "ACC-Der"),
    "anchor_cross_tool_agreement derivation parity (shared-anchor slope-sign agreement + sniper-steeper invariant)":
        ("scripts/test/test_anchor_cross_tool_agreement_derivation_parity.py", "ACT-Der"),
    "anchor_monotonicity_replay derivation parity (tier-aware bump-rate/hard-bump/max-bump ceilings)":
        ("scripts/test/test_anchor_monotonicity_replay_derivation_parity.py", "AMR-Der"),
    "gem5_slope_replay derivation parity (anchor OLS slope + 4-clause verdict matrix)":
        ("scripts/test/test_gem5_slope_replay_derivation_parity.py", "GSR-Der"),
    "sniper_slope_replay derivation parity (anchor OLS slope + 4-clause verdict matrix)":
        ("scripts/test/test_sniper_slope_replay_derivation_parity.py", "SSR-Der"),
    "literature_deviations_report derivation parity (mechanism-label classifier + cross-tab summary)":
        ("scripts/test/test_literature_deviations_report_derivation_parity.py", "LDR-Der"),
    "wss_knee_location derivation parity (regime-ladder knee walk + strict ordering invariant)":
        ("scripts/test/test_wss_knee_location_derivation_parity.py", "WKL-Der"),
    "paper_claims registry derivation parity (single-source-of-truth aggregator + JSON byte parity)":
        ("scripts/test/test_paper_claims_registry_derivation_parity.py", "PCR-Der"),
    "small_l3_thrash derivation parity (POLICY_LABEL_ORDER tie-break + per-policy aggregates)":
        ("scripts/test/test_small_l3_thrash_derivation_parity.py", "SLT-Der"),
    "corpus_diversity derivation parity (GAPBS log scrape + GRAPH_ORDER + _coerce rules)":
        ("scripts/test/test_corpus_diversity_derivation_parity.py", "CDV-Der"),
    "literature_faithfulness evaluator derivation parity (_classify branches + POPT_GE/NEAR_GRASP dispatch)":
        ("scripts/test/test_literature_faithfulness_evaluator_derivation.py", "LFE-Der"),
    "gem5/sniper anchor summary derivation parity (invariant predicates + tolerance constants)":
        ("scripts/test/test_gem5_anchor_summary_derivation.py", "GAS-Der"),
    "sign_consistency derivation parity (_pick canonical-ROI + _sign zero band + bucket dispatch)":
        ("scripts/test/test_sign_consistency_derivation.py", "SCD-Der"),
    "literature_reproduction_summary derivation parity (verdict glyph + paper rollup + render byte parity)":
        ("scripts/test/test_literature_reproduction_summary_derivation.py", "LRS-Der"),
    "local_cache_screen_summary derivation parity (number coercion + LRU-base delta + rank tie-break)":
        ("scripts/test/test_local_cache_screen_summary_derivation.py", "LCS-Der"),
    "cross-artifact integrity (CATALOG ↔ PYTEST_SUITES ↔ paper_claims graph consistency)":
        ("scripts/test/test_cross_artifact_integrity.py", "XAI-Int"),
    "reproduce_smoke coverage (every catalogued artifact is in TRACKED_ARTIFACTS audit)":
        ("scripts/test/test_reproduce_smoke_coverage.py", "RSC-Cov"),
    "Makefile coverage integrity (every CATALOG generator invoked or on documented allow-list)":
        ("scripts/test/test_makefile_coverage_integrity.py", "MFC-Int"),
    "wiki/data/ coverage (every on-disk .json/.md/.csv tracked or exempt)":
        ("scripts/test/test_wiki_data_coverage.py", "WDC-Cov"),
    "paper claims source-value parity (every claim re-derived from source artifact)":
        ("scripts/test/test_paper_claims_value_parity.py", "PCV-Src"),
    "paper claims schema integrity (required fields + controlled vocabularies + path resolution)":
        ("scripts/test/test_paper_claims_schema.py", "PCS-Sch"),
    "per-suite test-count floor (every PYTEST_SUITES entry has >=1 AST test fn + >=1 passed test + no errors)":
        ("scripts/test/test_pytest_suite_minimum.py", "PST-Min"),
    "catalog generator --help signature (every CATALOG generator responds to --help with exit 0 + usage marker)":
        ("scripts/test/test_catalog_generator_help.py", "CGH-Sig"),
    "wiki text quality (every tracked .md: no bad ws + no CRLF + single final newline + no double blanks + H1 first)":
        ("scripts/test/test_wiki_text_quality.py", "WTQ-Fmt"),
    "wiki JSON format (every tracked .json: valid utf-8, parseable, no CRLF, single final newline, no trailing ws)":
        ("scripts/test/test_wiki_json_format.py", "WJF-Fmt"),
    "PYTEST_SUITES label parity (path/short/label uniqueness + format + dashboard JSON cross-source parity)":
        ("scripts/test/test_pytest_suites_label_parity.py", "PSL-Par"),
    "wiki .md/.json companion pair (bijection + existence + non-empty + sibling-reference resolves)":
        ("scripts/test/test_wiki_md_json_pair.py", "WMP-Pair"),
    "literature-faithfulness diversity coverage (family/app/L3/paper floors + cross-paper triangulation)":
        ("scripts/test/test_lit_faith_diversity.py", "LIT-Cov"),
    "literature-faithfulness margin distribution (median/per-family floors + fragile ceiling + classifier parity)":
        ("scripts/test/test_lit_faith_margin.py", "LIT-Mar"),
    "literature-faithfulness sign-mass concentration (per-bucket Wilson LB + binomial sign-test + median delta_pct floors)":
        ("scripts/test/test_lit_faith_signmass.py", "LIT-Sig"),
    "literature-faithfulness citation locator integrity (bijection + anchor inventory + venue/year/locator well-formedness + docstring URL grounding)":
        ("scripts/test/test_lit_faith_citations.py", "LIT-Cite"),
    "literature-faithfulness known-deviation completeness (reason length + quantitative phrase + anchor + bijection with live faith corpus)":
        ("scripts/test/test_lit_faith_deviations.py", "LIT-Dev"),
    "literature-faithfulness tolerance calibration (per-row distance-to-disagree slack distribution + per-policy fragility floors)":
        ("scripts/test/test_lit_faith_tolerance.py", "LIT-Tol"),
    "literature-faithfulness accesses-floor audit (warmup-noise guard + per-axis distribution + zero floor violations)":
        ("scripts/test/test_lit_faith_accesses.py", "LIT-Acc"),
    "literature-faithfulness cross-app rationale coherence (per-citation contradiction + sign-vocabulary alignment + common kernel + length-span)":
        ("scripts/test/test_lit_faith_citexapp.py", "LIT-CXApp"),
    "literature-faithfulness cache-size monotonicity audit (per-triple non-increasing miss-rate + slope-per-doubling floors + saturated-triple ceiling)":
        ("scripts/test/test_lit_faith_monotonicity.py", "LIT-Mono"),
    "literature-faithfulness statistical-sanity audit (delta-arithmetic + sign-flip + miss-rate bounds + status-vocabulary + status-vs-delta consistency)":
        ("scripts/test/test_lit_faith_stat.py", "LIT-Stat"),
    "literature-faithfulness policy-ordering audit (per family x app: hub families respect POPT/GRASP <= LRU bounds + per-app global improve-frac floor)":
        ("scripts/test/test_lit_faith_polyord.py", "LIT-PolyOrd"),
    "literature-faithfulness deviation-explanation audit (every known_deviation reason names a mechanism + length/citation floors + reuse ceiling)":
        ("scripts/test/test_lit_faith_devexp.py", "LIT-DevExp"),
    "literature-faithfulness rationale-grid audit (per (policy, graph, app) rationale uniqueness + theorem-class invariance + per-policy citation floor)":
        ("scripts/test/test_lit_faith_ratgrid.py", "LIT-RatGrid"),
    "literature-faithfulness cell-completeness audit (per (graph, app, l3) canonical roster + LRU baseline + delta arithmetic + L3 sweep + axis parity)":
        ("scripts/test/test_lit_faith_cellcomp.py", "LIT-CellComp"),
    "literature-faithfulness app-frequency audit (per-app graph / L3 / policy / row floors + canonical roster per app + anchor-app full corpus sweep)":
        ("scripts/test/test_lit_faith_appfreq.py", "LIT-AppFreq"),
    "literature-faithfulness regime-sign audit (hub-family sign-tally no-regression + hub median ceiling + no-hub median radius + extreme magnitude cap)":
        ("scripts/test/test_lit_faith_regimesign.py", "LIT-RegimeSign"),
    "literature-faithfulness citation/date audit (per_claim citation parseability + venue whitelist + year range + per-policy originator-publication match + locator presence + distinct-citation floor)":
        ("scripts/test/test_lit_faith_citdate.py", "LIT-CitDate"),
    "ECG substrate-parity audit (cache_sim component-proof matrix: ECG_DBG_only ≡ GRASP_DBG_only and ECG_POPT_primary ≡ POPT_only on miss-rate, PFX activation+useful floors per benchmark, encoding hygiene, baseline non-zero floor, backend coverage)":
        ("scripts/test/test_lit_faith_ecg_parity.py", "ECG-Parity"),
    "ECG substrate-parity audit on gem5 (POPT-arm: ECG_POPT_PRIMARY ≡ POPT in cycle-accurate timing on matched-proof bracket sweep; section + L3 coverage floors; backend identity; LRU baseline non-zero; L3 hierarchy sanity)":
        ("scripts/test/test_lit_faith_ecg_gem5_parity.py", "ECG-Gem5-Parity"),
    "ECG substrate-parity audit on Sniper (scaffold/deferred today; activates on matched-proof Sniper ECG sweep — POPT-arm ECG_POPT_PRIMARY ≡ POPT and DBG-arm ECG_DBG_ONLY ≡ GRASP at ε=2e-3; backend identity; IPC/instructions/LRU floors)":
        ("scripts/test/test_lit_faith_ecg_sniper_parity.py", "ECG-Sniper-Parity"),
    "ECG PFX prefetcher vs DROPLET head-to-head on matched baseline (scaffold/deferred today — no /tmp sweep has nonzero pf_issued/pf_useful; rules: arm completeness, baseline neutrality 0.5pp, useful floor 5%, observation floor)":
        ("scripts/test/test_lit_faith_ecg_pfx_vs_droplet.py", "ECG-Pfx-vs-DROPLET"),
    "Paper label-map integrity (gate 242) — POLICY_LABELS/DESCRIPTIONS/COLORS in paper_pipeline.py vs committed paper_pipeline_*/policy_label_map.csv + every policy_label in tracked sources mapped + figure_labels unique + no orphan labels":
        ("scripts/test/test_lit_faith_paper_label_map.py", "PaperLabelMap"),
    "POLICY_COLORS perceptual distinguishability (gate 243) — hex format + dedup + pairwise CIE76 ΔE ≥ 12 + B&W lightness-delta ≥ 10 or hatch fallback (grandfathered allowlist) + ΔE ≥ 18 from white + POLICY_HATCHES ⊆ POLICY_LABELS":
        ("scripts/test/test_lit_faith_color_distinguishability.py", "ColorDistinguish"),
    "Paper-figure data snapshot (gate 244) — one paper_pipeline_YYYYMMDD/ dir within MAX_SNAPSHOT_AGE_DAYS, rows provenanced + single run_dir, rectangular (bench×graph×L3) palette coverage, miss_rate ∈ [0,1], total_accesses ≥ 1 for high-activity benches":
        ("scripts/test/test_lit_faith_paper_snapshot.py", "PaperSnapshot"),
    "L3 regime-classifier consistency (gate 245) — hand-curated REGIME_REGISTRY: every regime-classifier in ECG dir registered + within-family agreement on canonical L3 grid + vocab purity + cross-family divergence (oracle_gap_report vs v1) documented":
        ("scripts/test/test_lit_faith_regime_classifier.py", "RegimeClassifier"),
    "lit-faith citation registry purity (gate 246) — hand-curated CITATION_REGISTRY anchors every per_claim citation to a canonical work (Faldu HPCA20, Balaji HPCA21, Jaleel ISCA10) + per-bucket citation cohesion + no dead-letter registry entries":
        ("scripts/test/test_lit_faith_citation_registry.py", "CitationRegistry"),
    "Paper LaTeX-table emit invariant (gate 247) — hand-curated TABLE_REGISTRY locks caption + tabular col-spec + column-header tuple for every .tex in paper_pipeline dir; per-row column-count + no NaN/Inf cells + clean closing trio":
        ("scripts/test/test_lit_faith_paper_tables.py", "PaperTables"),
    "Sideband-schema registry (gate 248) — hand-curated SCHEMA_REGISTRY locks the [graphctx] register region wire-format across gem5/Sniper/cache_sim emit sites: field order, printf specifiers, C++ param types, regex round-trip + single emit per file":
        ("scripts/test/test_lit_faith_sideband_schema.py", "SidebandSchema"),
    "Graph-family map full-coverage (gate 249) — AST-harvests every GRAPH_FAMILY dict literal in scripts/experiments/ecg + scripts/test, asserts each agrees with canonical on every shared key; catches copies the gate-107 dup test does not yet guard":
        ("scripts/test/test_lit_faith_graph_family.py", "GraphFamily"),
    "Paper-table CSV provenance (gate 250) — PROVENANCE_REGISTRY pairs every shipped .tex with its sibling .csv; subset row-count and subset key-column multiset (LaTeX-normalized); every paper row traces to a CSV row past the [:20]/[:24] truncation cap":
        ("scripts/test/test_lit_faith_paper_provenance.py", "PaperProvenance"),
    "L3 cache-size registry (gate 251) — CANONICAL_L3_TIERS locks 11 tokens (4kB..32MB); AST-harvests every PAPER_L3/L3_SIZES/L3_MB/L3_BYTES constant in ecg+test; rules: byte arithmetic + MB scaling + ANCHOR_TRIPLET (1MB,4MB,8MB) + cross-file agreement":
        ("scripts/test/test_lit_faith_l3_registry.py", "L3Registry"),
    "Slurm SBATCH schema registry (gate 252) — CANONICAL_SBATCH_DIRECTIVES locks 14 directive names + 7 required; parses every *.sbatch under scripts/experiments/; rules: syntax, vocab, mem/time regex, log %x+%j or %A+%a, prefix, mem×mem-per-cpu":
        ("scripts/test/test_lit_faith_slurm_schema.py", "SlurmSchema"),
    "HANDOFF gate-reference registry (gate 253) — locks the contract between PYTEST_SUITES and the HANDOFF narrative; rules: no orphan (gate N) labels, headline+refresh-at == len(PYTEST_SUITES), refresh-due == refresh-at+5, max labeled == suite count":
        ("scripts/test/test_lit_faith_handoff_xref.py", "HandoffXref"),
    "wiki/data bidirectional registry (gate 254) — locks wiki/data/*.json ↔ catalog ↔ pytest; rules W1-W8: every json catalogued/aux/self-ref, no ghost entries, non-empty fields, paths exist, sibling .md, unique ids+paths, valid aux parent_id":
        ("scripts/test/test_lit_faith_wiki_registry.py", "WikiRegistry"),
    "Cache-policy vocab registry (gate 255) — AST-harvests POLICIES/ALL/BASELINE/GRAPH_AWARE tuples; 8 canonical + 9 ECG arms; rules P1-P9: rogue tokens, dup, config decomposition, paper_label, 4-tuple permutation lock, alias trap, ECG-arm parent+purpose":
        ("scripts/test/test_lit_faith_policy_registry.py", "PolicyRegistry"),
    "ECG profile registry (gate 256) — locks manifest profiles ↔ stages ↔ helpers ↔ README; 30 profiles, 30 stages, 25 citations; R1-R7: stage tokens resolve, descriptions, no dead profiles, citation typos, name snake_case, stage well-formed, unique":
        ("scripts/test/test_lit_faith_profile_registry.py", "ProfileRegistry"),
    "Backend vocab registry (gate 257) — AST-harvests backend literals in ecg+test; 7 canonical (cache_sim+kebab, gem5/riscv/x86, sniper/sift); R1-R7: in-canon, family+label, no dup labels, self-variant, argparse subset, name regex, all-referenced":
        ("scripts/test/test_lit_faith_backend_registry.py", "BackendRegistry"),
    "Graph canonical map (gate 258) — AST-harvests graph literals + GRAPH_FAMILY dicts + EVAL_GRAPHS; 26 canon, 8 families; R1-R8: in-canon, family+label, source provenance, dict-keys+family-match, non-dict use site, name+family regex, EVAL subset":
        ("scripts/test/test_lit_faith_graph_registry.py", "GraphRegistry"),
    "Build target registry (gate 259) — 4-backend × kernel × variant compile matrix vs Makefile KERNELS_* + CXXFLAGS_*; 51 targets, 0 orphans; R1-R8: src on disk, canonical, required flags, dirs, family, opt lock (-O3/-O1/-O2), ROI mechanism":
        ("scripts/test/test_lit_faith_build_registry.py", "BuildRegistry"),
    "GAPBS CLI registry (gate 260) — 6 CL classes (CLBase/App/IterApp/PageRank/Delta/Convert), 11 kernels, 28 flags; R1-R7: getopt live-vs-canonical, src class instantiation, full-flag = chain union, no within-chain conflicts, flag regex, purpose, arity":
        ("scripts/test/test_lit_faith_cli_registry.py", "CLIRegistry"),
    "ECG arm catalog (gate 261) — locks 9 paper policies + 9 registry arms + 16 ablations + 2 selectors across paper_pipeline/proof_matrix/policy_registry; A1-A7: paper→reg map, label/desc/color, _CHARGED hatch, ablation+selector refs, no dups":
        ("scripts/test/test_lit_faith_arm_catalog.py", "ArmCatalog"),
    "ECG cross-tool aggregator schema (gate 262) — locks on-disk JSON shape of 6 cross_tool* aggregators; S1-S7: exists, valid JSON, top-keys present, evidence non-empty, row keys present, tools ⊆ canonical, verdict type":
        ("scripts/test/test_lit_faith_cross_tool_schema.py", "CrossToolSchema"),
    "ECG config matrix (gate 263) — locks ecg/config.py vs canonical registries (gates 251/255/258/260); C1-C7: policies+graphs+kernels canonical, L3 anchor matches tier, sweep pow2 brackets anchor, pairs use ALL_POLICIES, ECG_MODE exhaustive":
        ("scripts/test/test_lit_faith_config_matrix.py", "ConfigMatrix"),
    "wiki/data filename grammar (gate 264) — locks FS shape of wiki/data/; F1-F7: lower_snake_case+ext∈{json,md,csv}, json↔md trio (mod allow-lists), catalog→disk, catalog ext∈{json,csv}, stem declared or META allow-list, subdir matches DOC_SUBDIR_RE":
        ("scripts/test/test_lit_faith_filename_grammar.py", "FilenameGrammar"),
    "gem5/Sniper sideband filename+env-var grammar (gate 265) — 8 entries (4 gem5+4 sniper); S1-S7: filename grammar, role↔ext, env↔default_path bijection, gem5_harness/sniper cache_set+prefetcher emit-sites, roi_matrix parse-sites, no orphan literals":
        ("scripts/test/test_lit_faith_sideband_grammar.py", "SidebandGrammar"),
    "Sniper overlay-installation tracker (gate 266) — locks setup_sniper.py overlay contract (.sniper_overlays.json); O1-O7: copied_files grammar, on-disk presence, policies↔cc+h, prefetchers↔cc+h, patches↔fn, canonical match, no orphans":
        ("scripts/test/test_lit_faith_overlay_tracker.py", "OverlayTracker"),
    "gem5 overlay-installation tracker (gate 267) — locks setup_gem5.py OVERLAY_FILE_MAP+PATCH_FILES; G1-G7: source grammar, on-disk presence, policies↔cc+hh, prefetchers↔cc+hh, patches on-disk, exhaustive, live↔canonical parity+identity":
        ("scripts/test/test_lit_faith_gem5_overlay_tracker.py", "G5OverlayTracker"),
    "Setup-script invariant registry (gate 268) — locks setup_gem5.py+setup_sniper.py against drift in repo URLs, dir-const skeleton, top-level fn inventory (14 gem5+27 sniper); S1-S7: URL, dir-const, gem5 fn, sniper fn, main, apply_overlays, exhaustive":
        ("scripts/test/test_lit_faith_setup_script_registry.py", "SetupScriptRegistry"),
    "ECG config deep-lock (gate 269) — locks ecg/config.py: 7 DEFAULT_CACHE anchors, 12-pt CACHE_SIZES_SWEEP, bench partition, policy partition, 4 ECG_MODES, 6-graph EVAL_GRAPHS, REORDER vocab, ACCURACY_PAIRS rels; C1-C8":
        ("scripts/test/test_lit_faith_config_deep_lock.py", "ConfigDeepLock"),
    "gem5 overlay-file hash registry (gate 270) — locks SHA-256 byte content of all 17 overlay sources under bench/include/gem5_sim/overlays/; M1-M7: hash parity, size bounds, markers, exhaustive, ext whitelist, SimObject, patches":
        ("scripts/test/test_lit_faith_gem5_overlay_hash_registry.py", "G5OverlayHash"),
    "Sniper overlay-file hash registry (gate 271) — locks SHA-256 byte content of all 12 overlay sources under bench/include/sniper_sim/overlays/; N1-N6: hash parity, size bounds, markers, exhaustive, ext whitelist, class decls":
        ("scripts/test/test_lit_faith_sniper_overlay_hash_registry.py", "SnpOverlayHash"),
    "Setup-script fn signature registry (gate 272) — locks positional arg names + defaults count of all 41 top-level public fns in setup_gem5.py + setup_sniper.py; F1-F7: presence, args, defaults, exhaustive, no varargs, no async, returns-annot":
        ("scripts/test/test_lit_faith_setup_fn_signature_registry.py", "SetupFnSig"),
    "Runner CLI registry (gate 273) — locks argparse surface of ecg/runner.py (6 flags) + vldb/runner.py (12 flags); R1-R6: module importable, main() present, flag presence, action match, nargs match, exhaustive":
        ("scripts/test/test_lit_faith_runner_cli_registry.py", "RunnerCLI"),
    "Orchestrator CLI registry (gate 274) — locks parse_args() of paper_pipeline.py (14) + final_paper_run.py (29); O1-O7: ast-parse, fn present, flag presence, action match (BooleanOptionalAction-aware), nargs match, exhaustive, required-shape":
        ("scripts/test/test_lit_faith_orchestrator_cli_registry.py", "OrchCLI"),
    "Paper-pipeline stage registry (gate 275) — locks public top-level fn signatures of paper_pipeline.py (50) + final_paper_run.py (37); 87 fns; S1-S7: ast-parse, fn-kind, args, defaults, return annot, vararg/kwarg/kwonly, exhaustive":
        ("scripts/test/test_lit_faith_paper_stage_registry.py", "StageReg"),
    "Manifest schema registry (gate 276) — locks final_paper_manifest.json shape: version, top-level keys, defaults vocab, stage-kind set, graph-entry shape, options_key xref, stage→profile xref; M1-M6 over 30 profiles, 30 stages, 8 graph_sets":
        ("scripts/test/test_lit_faith_manifest_schema.py", "ManifestSchema"),
    "Subprocess argv registry (gate 277) — locks --flag literals built by run_profile + make_proof_job + make_roi_job (3 builders, 56 flags); A1-A6: module+fn present, target const referenced, no removals, no additions, exhaustive, hygiene":
        ("scripts/test/test_lit_faith_subprocess_argv_registry.py", "ArgvReg"),
    "Receiver CLI registry (gate 278) — locks parse_args() of proof_matrix (17) + roi_matrix (48); 65 flags + 2 cross-pairs; E1-E7: ast-parse, parse_args present, flag presence, action, nargs, exhaustive, cross-side parity with gate 277 senders":
        ("scripts/test/test_lit_faith_receiver_cli_registry.py", "RecvCLI"),
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
