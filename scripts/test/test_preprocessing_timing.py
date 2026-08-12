#!/usr/bin/env python3
"""Tests for the explicit preprocessing timing contract."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.graphbrew_experiment import _save_reorder_time
from scripts.lib.core.utils import BenchmarkResult
from scripts.lib.pipeline.benchmark import (
    apply_pregenerated_reorder_cost,
    parse_benchmark_output,
)
from scripts.lib.pipeline.benchmark import mapping_permutation_fingerprint
from scripts.lib.pipeline.reorder import parse_reorder_time_from_converter
from scripts.lib.pipeline.reorder_timing import (
    metadata_path as reorder_time_metadata_path,
    read_reorder_time,
    write_reorder_time,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PR_BINARY = PROJECT_ROOT / "bench" / "bin" / "pr"
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"


def test_parser_exposes_explicit_preprocessing_boundaries():
    output = """
Representation Build Time: 1.00000
Reorder Core Time: 0.20000
Reorder Time: 0.20000
Mapping Fingerprint: abcdef0123456789
Reorder Validation Time: 0.01000
Reorder Apply Time: 0.03000
Reorder End-to-End Time: 0.24000
Total Preprocessing Time: 1.30000
Average Time: 0.50000
"""
    average, reorder, timing = parse_benchmark_output(output)

    assert average == pytest.approx(0.5)
    assert reorder == pytest.approx(0.24)
    assert timing["representation_build_time"] == pytest.approx(1.0)
    assert timing["reorder_core_time"] == pytest.approx(0.2)
    assert timing["mapping_generation_time"] == pytest.approx(0.2)
    assert timing["mapping_fingerprint"] == "abcdef0123456789"
    assert timing["reorder_validation_time"] == pytest.approx(0.01)
    assert timing["reorder_apply_time"] == pytest.approx(0.03)
    assert timing["complete_reorder_time"] == pytest.approx(0.24)
    assert timing["total_preprocessing_time"] == pytest.approx(1.3)


def test_parser_preserves_legacy_reorder_time_fallback():
    average, reorder, timing = parse_benchmark_output(
        "Reorder Time: 0.12500\nAverage Time: 1.00000\n"
    )

    assert average == pytest.approx(1.0)
    assert reorder == pytest.approx(0.125)
    assert timing["reorder_core_time"] == pytest.approx(0.125)
    assert timing["mapping_generation_time"] == pytest.approx(0.125)
    assert timing["complete_reorder_time"] == pytest.approx(0.125)


def test_parser_rejects_inconsistent_explicit_phase_totals():
    with pytest.raises(
        ValueError,
        match="Complete reorder time disagrees",
    ):
        parse_benchmark_output(
            "Reorder Core Time: 0.20000\n"
            "Reorder Time: 0.20000\n"
            "Reorder Validation Time: 0.10000\n"
            "Reorder Apply Time: 0.30000\n"
            "Reorder End-to-End Time: 0.70000\n"
        )


def test_generic_time_sidecar_uses_complete_reorder_cost(tmp_path):
    time_path = tmp_path / "DBG.time"
    output = """
Reorder Core Time: 0.20000
Reorder Time: 0.20000
Reorder Validation Time: 0.10000
Reorder Apply Time: 0.30000
Reorder End-to-End Time: 0.60000
Mapping Fingerprint: abcdef0123456789
Composed Mapping Fingerprint: abcdef0123456789
Resolved Reorder Spec: 5:degree=out
"""

    _save_reorder_time(output, str(time_path), str(tmp_path))

    assert not time_path.exists()
    assert reorder_time_metadata_path(time_path).is_file()
    assert read_reorder_time(time_path) == pytest.approx(0.6)


def test_reorder_pipeline_uses_shared_complete_cost():
    output = """
RabbitOrder Map Time: 0.05000
Reorder Core Time: 0.20000
Reorder Time: 0.20000
Reorder Validation Time: 0.10000
Reorder Apply Time: 0.30000
Reorder End-to-End Time: 0.60000
"""

    assert parse_reorder_time_from_converter(output) == pytest.approx(0.6)


def test_legacy_time_sidecars_are_opt_in(tmp_path):
    legacy = tmp_path / "DBG.time"
    legacy.write_text("1.25\n")

    assert read_reorder_time(legacy) is None
    assert read_reorder_time(legacy, allow_legacy=True) == pytest.approx(
        1.25
    )


def test_versioned_time_sidecar_is_bound_to_mapping(tmp_path):
    path = tmp_path / "DBG.time"
    write_reorder_time(
        path,
        complete_reorder_time=2.5,
        mapping_fingerprint="aaaaaaaaaaaaaaaa",
        algorithm_spec="5:degree=out",
    )
    assert read_reorder_time(
        path,
        expected_mapping_fingerprint="aaaaaaaaaaaaaaaa",
    ) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="mapping mismatch"):
        read_reorder_time(
            path,
            expected_mapping_fingerprint="bbbbbbbbbbbbbbbb",
        )


def test_pregenerated_result_charges_generation_not_replay(tmp_path):
    mappings = tmp_path / "mappings" / "g"
    mappings.mkdir(parents=True)
    mapping = mappings / "DBG.lo"
    mapping.write_text("0\n1\n")
    fingerprint = mapping_permutation_fingerprint(mapping)
    write_reorder_time(
        mappings / "DBG.time",
        complete_reorder_time=4.0,
        mapping_fingerprint=fingerprint,
        algorithm_spec="5:degree=out",
    )
    result = BenchmarkResult(
        graph="g",
        algorithm="DBG",
        algorithm_id=5,
        benchmark="pr",
        time_seconds=1.0,
        reorder_time=0.25,
    )

    apply_pregenerated_reorder_cost(
        result,
        graph_name="g",
        algo_name="DBG",
        mappings_dir=str(mappings),
    )

    assert result.reorder_time == pytest.approx(4.0)
    assert result.mapping_replay_time == pytest.approx(0.25)
    assert result.extra["mapping_replay_time"] == pytest.approx(0.25)

def test_cpp_self_recording_persists_preprocessing_boundaries(tmp_path):
    if not PR_BINARY.exists():
        pytest.skip("PR binary is not built")

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    env = os.environ.copy()
    env.pop("GRAPHBREW_DB_DIR", None)
    env["GRAPHBREW_TOPOLOGY_ANALYSIS"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-f",
            str(TINY_GRAPH),
            "-s",
            "-o",
            "5",
            "-n",
            "1",
            "-D",
            str(db_dir) + "/",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    _average, reorder, timing = parse_benchmark_output(result.stdout)
    assert timing["representation_build_time"] >= 0
    assert timing["reorder_core_time"] >= 0
    assert timing["reorder_validation_time"] >= 0
    assert timing["reorder_apply_time"] >= 0
    assert timing["total_preprocessing_time"] + 5e-5 >= (
        timing["representation_build_time"] + reorder
    )

    rows = json.loads((db_dir / "benchmarks.json").read_text())
    assert len(rows) == 1
    row = rows[0]
    for field in (
        "representation_build_time",
        "reorder_core_time",
        "reorder_validation_time",
        "reorder_apply_time",
        "total_preprocessing_time",
    ):
        assert row[field] == pytest.approx(timing[field], abs=5e-5)
    assert row["reorder_time"] == pytest.approx(reorder, abs=5e-5)

    detail = row["reorder_details"][0]
    assert detail["reorder_core_time"] == pytest.approx(
        timing["reorder_core_time"], abs=5e-5
    )
    assert detail["validation_time"] == pytest.approx(
        timing["reorder_validation_time"], abs=5e-5
    )
    assert detail["apply_time"] == pytest.approx(
        timing["reorder_apply_time"], abs=5e-5
    )


def test_chained_fingerprint_matches_written_and_map_replay(tmp_path):
    if not CONVERTER.exists():
        pytest.skip("converter is not built")
    mapping = tmp_path / "chain.lo"
    env = {
        **os.environ,
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": "1",
    }
    direct = subprocess.run(
        [
            str(CONVERTER),
            "-f",
            str(TINY_GRAPH),
            "-s",
            "-o",
            "2",
            "-o",
            "5",
            "-q",
            str(mapping),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert direct.returncode == 0, direct.stderr
    _average, _reorder, direct_timing = parse_benchmark_output(
        direct.stdout
    )
    expected = mapping_permutation_fingerprint(mapping)
    assert len(direct_timing["mapping_fingerprints"]) == 2
    assert direct_timing["mapping_fingerprint"] == expected

    replay = subprocess.run(
        [
            str(CONVERTER),
            "-f",
            str(TINY_GRAPH),
            "-s",
            "-o",
            f"13:{mapping}",
            "-q",
            str(tmp_path / "replay.lo"),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replay.returncode == 0, replay.stderr
    _average, _reorder, replay_timing = parse_benchmark_output(
        replay.stdout
    )
    assert replay_timing["mapping_fingerprint"] == expected
    assert replay_timing["resolved_algorithm_spec"] == (
        f"13:fingerprint={expected}"
    )

    db_dir = tmp_path / "map-db"
    db_dir.mkdir()
    self_record = subprocess.run(
        [
            str(PR_BINARY),
            "-f",
            str(TINY_GRAPH),
            "-s",
            "-o",
            f"13:{mapping}",
            "-n",
            "1",
            "-D",
            str(db_dir) + "/",
        ],
        cwd=PROJECT_ROOT,
        env={
            key: value for key, value in env.items()
            if key != "GRAPHBREW_DB_DIR"
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert self_record.returncode == 0, self_record.stderr
    row = json.loads((db_dir / "benchmarks.json").read_text())[0]
    assert row["mapping_identity_id"] == f"map:{expected}"
    assert row["mapping_fingerprint"] == expected
