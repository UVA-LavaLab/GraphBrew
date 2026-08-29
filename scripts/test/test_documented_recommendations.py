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
    assert invalidation["schema"].startswith(
        "graphbrew-native-midreuse-terminal-invalidation"
    )
