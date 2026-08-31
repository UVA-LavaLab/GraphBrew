"""Bind public GraphBrew claims to frozen external artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / "docs/recommendation-evidence.json"
ARTIFACT_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_PAPER_ARTIFACT_ROOT",
        "/media/NVMeData/00_GraphDatasets/GraphBrew/artifacts",
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(record: dict) -> dict:
    path = ARTIFACT_ROOT / record["path"]
    if not path.is_file():
        pytest.skip(f"Frozen artifact unavailable: {path}")
    assert _sha256(path) == record["sha256"]
    return json.loads(path.read_text())


def test_public_claims_match_frozen_evidence():
    evidence = json.loads(EVIDENCE_PATH.read_text())
    assert evidence["schema"] == "graphbrew-public-evidence/v2"
    sources = evidence["source_artifacts"]
    quality = _load_source(sources["quality_confirmation"])
    compact_mapping = _load_source(sources["compact_and_emit_mapping"])
    compact_development = _load_source(
        sources["compact_and_emit_development"]
    )
    _load_source(sources["compact_and_emit_original_audit"])
    atlas = _load_source(sources["composition_atlas"])
    composability = _load_source(sources["composability_certificate"])
    sealed = _load_source(sources["sealed_composability_confirmation"])
    mechanisms = _load_source(sources["mechanism_factorial"])
    invalidation = _load_source(sources["invalidated_terminal_timing"])

    claims = evidence["confirmed_claims"]
    quality_claim = claims["quality_arm"]
    assert quality_claim["kernel_gm"] == {
        key: quality["comparisons"][source_key][
            "baseline_over_graphbrew_kernel_gm"
        ]
        for key, source_key in (
            ("rabbit_csr_over_graphbrew", "rabbit_csr"),
            ("rabbit_boost_over_graphbrew", "rabbit_boost"),
            ("gorder_csr_over_graphbrew", "gorder_csr"),
        )
    }
    assert quality_claim["mapping_gm"] == {
        key: quality["mapping"][source_key][
            "graphbrew_over_baseline_complete_mapping_gm"
        ]
        for key, source_key in (
            ("graphbrew_over_rabbit_csr", "rabbit_csr"),
            ("graphbrew_over_rabbit_boost", "rabbit_boost"),
            ("graphbrew_over_gorder_csr", "gorder_csr"),
        )
    }
    assert quality_claim["rabbit_cell_gm_break_even_reuse"] == {
        "csr": quality["end_to_end"]["rabbit_csr"][
            "cell_gm_crossover"
        ]["reuse"],
        "boost": quality["end_to_end"]["rabbit_boost"][
            "cell_gm_crossover"
        ]["reuse"],
    }
    gorder_claim = quality_claim["primary_gorder_claim"]
    gorder_e2e = quality["end_to_end"]["gorder_csr"]
    assert gorder_claim[
        "gorder_csr_over_graphbrew_reuse1_end_to_end_cell_gm"
    ] == pytest.approx(
        gorder_e2e["cell_gm_baseline_over_graphbrew"]["1"]
    )
    assert gorder_claim[
        "gorder_csr_over_graphbrew_reuse1_summed_seconds"
    ] == pytest.approx(
        gorder_e2e["summed_seconds_shared_mapping_once_per_graph"][
            "crossover"
        ]["ratio_at_reuse"]
    )
    assert gorder_claim[
        "graphbrew_over_gorder_csr_summed_kernel_seconds"
    ] == pytest.approx(
        quality["comparisons"]["gorder_csr"]["summed_kernel_seconds"][
            "graphbrew_over_baseline"
        ]
    )

    rabbit_limit = quality_claim["rabbit_pareto_limit"]
    assert rabbit_limit[
        "rabbit_csr_over_graphbrew_without_cc_gm"
    ] == pytest.approx(
        quality["verdict"]["leave_cc_out_gm"]["rabbit_csr"]
    )
    assert rabbit_limit[
        "rabbit_boost_over_graphbrew_without_cc_gm"
    ] == pytest.approx(
        quality["verdict"]["leave_cc_out_gm"]["rabbit_boost"]
    )
    assert rabbit_limit[
        "graphbrew_over_rabbit_csr_summed_kernel_seconds"
    ] == pytest.approx(
        quality["comparisons"]["rabbit_csr"]["summed_kernel_seconds"][
            "graphbrew_over_baseline"
        ]
    )
    assert rabbit_limit[
        "graphbrew_over_rabbit_boost_summed_kernel_seconds"
    ] == pytest.approx(
        quality["comparisons"]["rabbit_boost"]["summed_kernel_seconds"][
            "graphbrew_over_baseline"
        ]
    )

    compact_claim = claims["compact_and_emit"]["mapping_only"]
    assert compact_claim["wiki_talk_complete_seconds"] == pytest.approx(
        compact_mapping["gate_results"][
            "wiki_talk_compact_direct_complete_seconds"
        ]["value"]
    )
    assert compact_claim["three_graph_mapping_reduction"] == pytest.approx(
        compact_mapping["gate_results"]["selection_graph_reduction_gm"][
            "value"
        ]
    )
    assert compact_claim[
        "five_graph_candidate_over_min_rabbit_mapping_gm"
    ] == pytest.approx(
        compact_development["gate_results"]["mapping_gm"]["value"]
    )

    atlas_claim = claims["composition_atlas"]
    assert atlas_claim[
        "in_sample_oracle_over_fastest_comparator_gm"
    ] == pytest.approx(
        atlas["historical_in_sample_oracle"][
            "competitor_over_oracle_gm"
        ]
    )
    assert atlas_claim["best_fixed_over_fastest_comparator_gm"] == (
        pytest.approx(
            max(
                row["competitor_over_arm_gm"]
                for row in atlas["fixed_arms"].values()
            )
        )
    )
    assert atlas_claim[
        "graph_held_out_over_fastest_comparator_gm"
    ] == pytest.approx(
        atlas["leave_one_graph_out_kernel_policy"][
            "competitor_over_policy_gm"
        ]
    )
    dependence = atlas_claim["workload_dependence"]
    diversity = composability["winner_diversity"]
    policies = composability["policy_results"]
    assert dependence["distinct_winning_compositions"] == (
        diversity["distinct_winning_compositions"]
    )
    assert dependence["candidate_compositions"] == (
        composability["scope"]["candidate_compositions"]
    )
    assert dependence["minimum_distinct_winners_per_graph"] == (
        diversity["minimum_distinct_winners_per_graph"]
    )
    assert dependence["maximum_distinct_winners_per_graph"] == (
        diversity["maximum_distinct_winners_per_graph"]
    )
    assert dependence["minimum_distinct_winners_per_kernel"] == (
        diversity["minimum_distinct_winners_per_kernel"]
    )
    assert dependence["maximum_distinct_winners_per_kernel"] == (
        diversity["maximum_distinct_winners_per_kernel"]
    )
    assert dependence[
        "cell_oracle_over_best_fixed_graphbrew_gm"
    ] == pytest.approx(
        policies["cell_oracle"]["best_fixed_graphbrew_over_policy_gm"]
    )
    assert dependence[
        "postselected_graph_type_kernel_over_fastest_comparator_gm"
    ] == pytest.approx(
        policies["graph_type_kernel_conditioned"][
            "fastest_comparator_over_policy_gm"
        ]
    )
    assert dependence[
        "held_out_graph_type_kernel_over_fastest_comparator_gm"
    ] == pytest.approx(
        policies["leave_one_graph_out_family_kernel"][
            "fastest_comparator_over_policy_gm"
        ]
    )

    sealed_claim = claims["sealed_composability"]
    sealed_diversity = sealed["winner_diversity"]
    sealed_policies = sealed["policy_results"]
    assert sealed_claim["distinct_winning_compositions"] == (
        sealed_diversity["distinct_winning_compositions"]
    )
    assert sealed_claim["candidate_compositions"] == (
        sealed["scope"]["candidate_compositions"]
    )
    assert sealed_claim[
        "cell_oracle_over_best_fixed_graphbrew_gm"
    ] == pytest.approx(
        sealed_policies["cell_oracle"][
            "best_fixed_graphbrew_over_policy_gm"
        ]
    )
    assert sealed_claim[
        "fastest_comparator_over_cell_oracle_gm"
    ] == pytest.approx(
        sealed_policies["cell_oracle"][
            "fastest_comparator_over_policy_gm"
        ]
    )
    assert sealed_claim[
        "frozen_family_kernel_over_fastest_comparator_gm"
    ] == pytest.approx(
        sealed_policies["frozen_family_kernel"][
            "fastest_comparator_over_policy_gm"
        ]
    )
    sealed_confirmation = sealed["confirmation"]
    assert sealed_claim[
        "rapid_fixed_over_frozen_family_kernel_gm"
    ] == pytest.approx(
        sealed_confirmation["controls"]["rapid_best_fixed"][
            "control_over_frozen_policy_kernel_gm"
        ]
    )
    assert sealed_claim[
        "gorder_csr_over_frozen_family_kernel_gm"
    ] == pytest.approx(
        sealed_confirmation["gorder_csr"]["kernel_gm"]
    )
    assert sealed_claim[
        "gorder_csr_over_frozen_family_kernel_reuse1_end_to_end_gm"
    ] == pytest.approx(
        sealed_confirmation["gorder_csr"]["end_to_end"]["1"][
            "gorder_over_frozen_policy_gm"
        ]
    )
    assert sealed_claim["gorder_csr_cell_gm_crossover_reuse"] == (
        sealed_confirmation["gorder_csr"]["cell_gm_crossover_reuse"]
    )
    assert sealed_confirmation["passes_all_gates"] is False

    mechanism_claim = claims["mechanism_factorial"]
    mechanism_timing = mechanisms["timing_factorial"]
    mechanism_work = mechanisms["dynamic_work"]
    assert mechanism_claim["membership_equivalent"] is (
        mechanisms["factorial_contract"]["membership_equivalent"]
    )
    assert mechanism_claim["block_order"][
        "size_desc_over_degree_desc_speedup"
    ] == pytest.approx(
        1
        / mechanism_timing["block_order"]["marginal"][
            "left_over_right_gm"
        ]
    )
    assert mechanism_claim["bfs"][
        "localgorder8_over_hubsort_speedup"
    ] == pytest.approx(
        mechanism_timing["intra_order_pairs"][
            "hubsort_over_gorder8"
        ]["by_kernel_graph_block"]["bfs"]["left_over_right_gm"]
    )
    assert mechanism_claim["bfs"][
        "hubsort_over_localgorder8_examined_edges"
    ] == pytest.approx(
        mechanism_work["bfs"]["work_factorial"][
            "intra_order_pairs"
        ]["hubsort_over_gorder8"]["left_over_right_gm"]
    )
    assert mechanism_claim["cc"][
        "localgorder8_over_rcmpp_speedup"
    ] == pytest.approx(
        1
        / mechanism_timing["intra_order_pairs"][
            "gorder8_over_rcmpp"
        ]["by_kernel_graph_block"]["cc"]["left_over_right_gm"]
    )
    assert mechanism_claim["cc"][
        "localgorder8_over_rcmpp_compression_steps"
    ] == pytest.approx(
        mechanism_work["cc"]["work_factorial"][
            "intra_order_pairs"
        ]["gorder8_over_rcmpp"]["left_over_right_gm"]
    )
    assert mechanism_claim["unresolved"][
        "pr_hubsort_over_localgorder8_speedup"
    ] == pytest.approx(
        1
        / mechanism_timing["intra_order_pairs"][
            "hubsort_over_gorder8"
        ]["by_kernel_graph_block"]["pr"]["left_over_right_gm"]
    )
    assert mechanisms["scope"]["hardware_counters"]["available"] is False
    assert invalidation["schema"].startswith(
        "graphbrew-native-midreuse-terminal-invalidation"
    )
