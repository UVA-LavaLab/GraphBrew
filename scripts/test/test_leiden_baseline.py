#!/usr/bin/env python3
"""Leiden community-detection and post-layout baseline contracts."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.lib.pipeline.benchmark import parse_benchmark_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
PR_BINARY = PROJECT_ROOT / "bench" / "bin" / "pr"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} is not built")


def _mapping(
    tmp_path: Path,
    option: str,
    threads: int,
    name: str,
) -> tuple[bytes, str]:
    mapping = tmp_path / f"{name}.lo"
    env = {
        **os.environ,
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": str(threads),
    }
    result = subprocess.run(
        [
            str(CONVERTER),
            "-g",
            "10",
            "-s",
            "-o",
            option,
            "-q",
            str(mapping),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return mapping.read_bytes(), result.stdout


def test_historical_layout_is_explicitly_reproducible(tmp_path):
    _require(CONVERTER)
    implicit, _ = _mapping(tmp_path, "15:1.0", 1, "implicit")
    explicit, output = _mapping(
        tmp_path,
        "15:1.0:10:10:hierarchy-degree",
        1,
        "explicit",
    )

    assert implicit == explicit
    assert "hierarchy-degree" in output
    assert "Leiden Seed:" in output


def test_post_layout_policies_are_distinct(tmp_path):
    _require(CONVERTER)
    hierarchy, _ = _mapping(
        tmp_path,
        "15:1.0:10:10:hierarchy-degree",
        1,
        "hierarchy",
    )
    stable, _ = _mapping(
        tmp_path,
        "15:1.0:10:10:final-stable",
        1,
        "stable",
    )
    degree, _ = _mapping(
        tmp_path,
        "15:1.0:10:10:final-degree",
        1,
        "degree",
    )

    assert len({hierarchy, stable, degree}) == 3


def test_fixed_thread_policy_is_repeat_deterministic(tmp_path):
    _require(CONVERTER)
    first, _ = _mapping(tmp_path, "15:1.0", 4, "first")
    second, _ = _mapping(tmp_path, "15:1.0", 4, "second")

    assert first == second


@pytest.mark.parametrize("option", [
    "15:dynamic",
    "15:not-a-resolution",
    "15:4.0",
    "15:1.0:0",
    "15:1.0:10:0",
    "15:1.0:10:10:unknown",
    "15:1.0:10:10:final-stable:extra",
])
def test_invalid_leiden_options_fail_closed(tmp_path, option):
    _require(CONVERTER)
    mapping = tmp_path / "invalid.lo"
    env = {
        **os.environ,
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            str(CONVERTER),
            "-g",
            "8",
            "-s",
            "-o",
            option,
            "-q",
            str(mapping),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert not mapping.exists()


def test_parser_exposes_layout_and_thread_policy(tmp_path):
    _require(CONVERTER)
    _mapping_bytes, output = _mapping(
        tmp_path,
        "15:0.75:10:10:final-stable",
        1,
        "parsed",
    )
    _average, _reorder, timing = parse_benchmark_output(output)

    assert timing["leiden_layout"] == "final-stable"
    assert timing["leiden_seed"] == "0"
    assert timing["reorder_thread_policy_sensitive"] is True
    assert timing["mapping_fingerprint"]


def test_cpp_self_recording_preserves_spec_and_layout(tmp_path):
    _require(PR_BINARY)
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    option = "15:0.75:10:10:final-stable"
    env = {
        **os.environ,
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": "1",
    }
    env.pop("GRAPHBREW_DB_DIR", None)
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g",
            "8",
            "-s",
            "-o",
            option,
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

    row = json.loads((db_dir / "benchmarks.json").read_text())[0]
    assert row["algorithm"] == "LeidenOrder"
    assert row["algorithm_spec"] == option
    assert row["reorder_thread_policy_sensitive"] is True
    assert row["reorder_details"][0]["layout"] == "final-stable"
