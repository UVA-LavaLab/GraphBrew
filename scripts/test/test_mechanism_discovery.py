"""Mechanism-discovery plan and mapping metric tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.lib.analysis import mechanism_discovery
from scripts.lib.analysis.mechanism_discovery import (
    MEASUREMENT_MODE,
    analyze_mapping,
    build_mapping_screen_plan,
    classify_reference,
    compare_mappings,
    _canonical_digest,
    invert_mapping,
    load_inverse_mapping,
)
from scripts.lib.pipeline.benchmark import (
    file_sha256,
    mapping_permutation_fingerprint,
)
from scripts.lib.pipeline.synthetic_graphs import (
    SyntheticGraphSpec,
    generate_synthetic_graph,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mapping_inversion_round_trip(tmp_path):
    path = tmp_path / "mapping.lo"
    path.write_text("2\n0\n1\n")
    inverse = load_inverse_mapping(path, expected_nodes=3)
    assert inverse == [2, 0, 1]
    assert invert_mapping(inverse) == [1, 2, 0]


def test_chain_reference_beats_shuffled_original_structurally(tmp_path):
    artifact = generate_synthetic_graph(
        SyntheticGraphSpec("chain", 256, 1),
        tmp_path,
    )
    original = tmp_path / "ORIGINAL.lo"
    original.write_text("\n".join(map(str, range(256))) + "\n")

    reference_metrics = analyze_mapping(
        artifact,
        artifact.reference_mapping_path,
        algorithm_spec="REFERENCE",
        reference_mapping_path=artifact.reference_mapping_path,
    )
    original_metrics = analyze_mapping(
        artifact,
        original,
        algorithm_spec="0",
        reference_mapping_path=artifact.reference_mapping_path,
    )

    assert reference_metrics["positive_bit_mloga"] < (
        original_metrics["positive_bit_mloga"]
    )
    assert reference_metrics["same_line_edge_fraction"] > (
        original_metrics["same_line_edge_fraction"]
    )
    assert reference_metrics["decision_divergence"][
        "normalized_footrule_frame_invariant"
    ] == 0.0
    assert original_metrics["measurement_mode"] == MEASUREMENT_MODE
    assert original_metrics["claim_eligible"] is False


def test_screen_plan_is_capped_and_exact(tmp_path):
    graph_root = tmp_path / "graphs"
    artifact_root = tmp_path / "artifacts"
    specs = (
        SyntheticGraphSpec("chain", 64, 0),
        SyntheticGraphSpec("grid", 64, 0),
    )
    plan = build_mapping_screen_plan(
        graph_root,
        artifact_root,
        threads=2,
        specs=specs,
        require_clean_implementation=False,
        require_full_screen=False,
    )
    assert plan["configuration_count"] == 2
    assert plan["command_count"] == 10
    assert plan["rabbit_repeats_per_graph"] == 3
    assert plan["planned_total_configurations"] == 8
    assert plan["configuration_cap"] == 48
    assert plan["defined_cap_hours"] <= plan["node_hour_cap"]
    assert plan["measurement_mode"] == MEASUREMENT_MODE
    assert {
        command["algorithm_spec"] for command in plan["commands"]
    } == {"5", "8:csr", "9:csr"}
    assert all(
        command["environment_mode"] == "clean-allowlist/v1"
        and command["environment"]["RABBIT_RESOLUTION"] == "1"
        and command["environment"]["GORDER_WINDOW"] == "5"
        and command["environment"]["OMP_THREAD_LIMIT"] == "2"
        for command in plan["commands"]
    )
    assert all(
        Path(record["reference_mapping_path"]).is_file()
        for record in plan["graphs"]
    )
    assert all(
        record["metadata_sha256"]
        and record["vertex_metadata_sha256"]
        for record in plan["graphs"]
    )
    assert plan["promotion_rule"][
        "minimum_improvement_baselines"
    ] == ["8:csr", "9:csr"]
    assert plan["promotion_rule"][
        "defect_scan_baselines"
    ] == ["0", "5", "8:csr", "9:csr"]


def test_top_level_orchestrator_exposes_separate_discovery_stages():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/graphbrew_experiment.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--mechanism-discovery-plan" in result.stdout
    assert "--mechanism-discovery-screen" in result.stdout
    assert "--mechanism-discovery-no-resume" in result.stdout
    assert "--mechanism-discovery-refreeze-plan" in result.stdout
    assert "--mechanism-discovery-refreeze-graphs" in result.stdout


def test_biclique_reference_interleaves_partitions(tmp_path):
    artifact = generate_synthetic_graph(
        SyntheticGraphSpec("block-biclique", 64, 0),
        tmp_path,
    )
    metrics = analyze_mapping(
        artifact,
        artifact.reference_mapping_path,
        algorithm_spec="REFERENCE",
    )
    assert metrics["same_line_edge_fraction"] > 0.0


def test_decision_divergence_is_reversal_invariant(tmp_path):
    artifact = generate_synthetic_graph(
        SyntheticGraphSpec("chain", 64, 0),
        tmp_path,
    )
    reference = load_inverse_mapping(
        artifact.reference_mapping_path, expected_nodes=64)
    reversed_path = tmp_path / "reversed.lo"
    reversed_path.write_text(
        "\n".join(map(str, reversed(reference))) + "\n")
    comparison = compare_mappings(
        artifact,
        artifact.reference_mapping_path,
        reversed_path,
        left_spec="REFERENCE",
        right_spec="REVERSED",
    )
    assert comparison["placement_divergence"][
        "normalized_footrule_frame_invariant"
    ] == 0.0


def test_control_reference_never_defects_or_qualifies():
    def row(value, fingerprints=("a",)):
        return {
            "positive_bit_mloga_per_edge": {
                "median": value,
                "relative_spread": 0.0,
            },
            "mapping_fingerprints": list(fingerprints),
        }

    classification = classify_reference({
        "REFERENCE": row(10.0),
        "0": row(9.0),
        "5": row(8.0),
        "8:csr": row(7.0, ("a", "b", "c")),
        "9:csr": row(6.0),
    }, "control-frame-not-oracle")
    assert classification["reference_defects"] == []
    assert classification["mapping_screen_qualifies"] is False
    assert set(classification["control_baseline_advantage"]) == {
        "0", "5", "8:csr", "9:csr",
    }


def test_original_beating_heuristic_is_reference_defect():
    def row(value, fingerprints=("a",)):
        return {
            "positive_bit_mloga_per_edge": {
                "median": value,
                "relative_spread": 0.0,
            },
            "mapping_fingerprints": list(fingerprints),
        }

    classification = classify_reference({
        "REFERENCE": row(10.0),
        "0": row(9.0),
        "5": row(11.0),
        "8:csr": row(12.0, ("a", "b", "c")),
        "9:csr": row(12.0),
    }, "heuristic-test")
    assert classification["reference_defects"] == ["0"]
    assert classification["mapping_screen_qualifies"] is False


def test_plan_requires_committed_clean_implementation(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        mechanism_discovery,
        "repository_scope_state",
        lambda *_args, **_kwargs: {
            "revision": "deadbeef",
            "relevant_diff_sha256": "not-clean",
            "relevant_untracked": ["synthetic_graphs.py"],
        },
    )
    with pytest.raises(RuntimeError, match="Commit the reviewed"):
        build_mapping_screen_plan(
            tmp_path / "graphs",
            tmp_path / "artifacts",
            threads=1,
            specs=(SyntheticGraphSpec("chain", 64, 0),),
            require_full_screen=False,
        )


def test_execute_rejects_vertex_metadata_change(tmp_path):
    plan = build_mapping_screen_plan(
        tmp_path / "graphs",
        tmp_path / "artifacts",
        threads=1,
        specs=(SyntheticGraphSpec("chain", 64, 0),),
        require_clean_implementation=False,
        require_full_screen=False,
    )
    from scripts.lib.analysis.mechanism_discovery import (
        execute_mapping_screen,
        write_mapping_screen_plan,
    )

    plan_path = write_mapping_screen_plan(
        plan, tmp_path / "artifacts")
    vertex_path = Path(plan["graphs"][0]["vertex_metadata_path"])
    vertex_path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="input changed"):
        execute_mapping_screen(plan_path)


def test_resume_rejects_result_from_different_plan(tmp_path):
    plan = build_mapping_screen_plan(
        tmp_path / "graphs",
        tmp_path / "artifacts",
        threads=1,
        specs=(SyntheticGraphSpec("chain", 64, 0),),
        require_clean_implementation=False,
        require_full_screen=False,
    )
    from scripts.lib.analysis.mechanism_discovery import (
        execute_mapping_screen,
        write_mapping_screen_plan,
    )

    plan_path = write_mapping_screen_plan(
        plan, tmp_path / "artifacts")
    command = plan["commands"][0]
    mapping_path = Path(command["mapping_path"])
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text("\n".join(map(str, range(64))) + "\n")
    result_path = (
        tmp_path / "artifacts" / "mapping_screen_results"
        / (command["command_id"].replace("|", "__") + ".json")
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({
        "schema": "mechanism-discovery-command/v2",
        "plan_sha256": "different-plan",
        "command_sha256": _canonical_digest(command),
        "graph_sha256": command["graph_sha256"],
        "mapping_sha256": file_sha256(mapping_path),
        "measurement_mode": MEASUREMENT_MODE,
        "claim_eligible": False,
        "success": True,
    }))
    with pytest.raises(RuntimeError, match="different plan"):
        execute_mapping_screen(plan_path)


def test_converter_mapping_coordinate_contract(tmp_path):
    converter = PROJECT_ROOT / "bench" / "bin" / "converter"
    if not converter.is_file():
        pytest.skip("converter binary is not built")
    artifact = generate_synthetic_graph(
        SyntheticGraphSpec("chain", 64, 2),
        tmp_path / "graphs",
    )
    mapping = tmp_path / "DBG.lo"
    result = subprocess.run(
        [
            str(converter),
            "-f", str(artifact.graph_path),
            "-s",
            "-o", "5",
            "-q", str(mapping),
        ],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert len(load_inverse_mapping(mapping, expected_nodes=64)) == 64
    assert mapping_permutation_fingerprint(mapping)
