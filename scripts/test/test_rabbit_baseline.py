#!/usr/bin/env python3
"""RabbitOrder variant identity and mapping-provenance contracts."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts.lib.pipeline.benchmark import parse_benchmark_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
PR_BINARY = PROJECT_ROOT / "bench" / "bin" / "pr"
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} is not built")


def _rabbit_build_enabled() -> bool:
    config = PROJECT_ROOT / "bench" / "obj" / ".build-config"
    return not config.is_file() or "RABBIT_ENABLE=1" in config.read_text()


def _run_converter(tmp_path: Path, option: str, name: str):
    mapping = tmp_path / f"{name}.lo"
    env = {
        **os.environ,
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            str(CONVERTER),
            "-f",
            str(TINY_GRAPH),
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
    return result, mapping


def _fingerprint_from_lo(mapping: Path) -> str:
    new_to_source = [
        int(value) for value in mapping.read_text().split()
    ]
    source_to_new = [0] * len(new_to_source)
    for new_id, source_id in enumerate(new_to_source):
        source_to_new[source_id] = new_id

    value = 1469598103934665603
    mask = (1 << 64) - 1
    for source_id, new_id in enumerate(source_to_new):
        value ^= source_id
        value = (value * 1099511628211) & mask
        value ^= new_id
        value = (value * 1099511628211) & mask
    return f"{value:016x}"


def test_bare_rabbit_is_csr_variant(tmp_path):
    _require(CONVERTER)
    _bare_result, bare = _run_converter(tmp_path, "8", "bare")
    _csr_result, csr = _run_converter(tmp_path, "8:csr", "csr")

    assert bare.read_bytes() == csr.read_bytes()


def test_mapping_fingerprint_matches_written_permutation(tmp_path):
    _require(CONVERTER)
    result, mapping = _run_converter(tmp_path, "8:csr", "csr")
    _average, _reorder, timing = parse_benchmark_output(result.stdout)

    assert timing["mapping_fingerprint"] == _fingerprint_from_lo(mapping)
    assert timing["reorder_schedule_sensitive"] is True


def test_boost_variant_is_explicit_when_available(tmp_path):
    _require(CONVERTER)
    if not _rabbit_build_enabled():
        pytest.skip("Boost RabbitOrder is disabled in this build")
    result, mapping = _run_converter(tmp_path, "8:boost", "boost")

    assert mapping.stat().st_size > 0
    assert "Algorithm:           RabbitOrder" in result.stdout


@pytest.mark.parametrize("option,expected,resolved", [
    ("8", "RABBITORDER_csr", "8:csr:degree-sort=out-in"),
    ("8:csr", "RABBITORDER_csr", "8:csr:degree-sort=out-in"),
    ("8:boost", "RABBITORDER_boost", "8:boost:degree-sort=out-in"),
])
def test_cpp_self_recording_preserves_rabbit_variant_and_fingerprint(
    tmp_path,
    option,
    expected,
    resolved,
):
    _require(PR_BINARY)
    if option.endswith("boost") and not _rabbit_build_enabled():
        pytest.skip("Boost RabbitOrder is disabled in this build")

    db_dir = tmp_path / expected
    db_dir.mkdir()
    env = {
        **os.environ,
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": "1",
    }
    env.pop("GRAPHBREW_DB_DIR", None)
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-f",
            str(TINY_GRAPH),
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

    rows = json.loads((db_dir / "benchmarks.json").read_text())
    row = rows[0]
    detail = row["reorder_details"][0]
    assert row["algorithm"] == expected
    assert row["requested_algorithm_spec"] == option
    assert row["algorithm_spec"] == resolved
    assert re.fullmatch(r"[0-9a-f]{16}", row["mapping_fingerprint"])
    assert row["reorder_schedule_sensitive"] is True
    assert detail["mapping_fingerprint"] == row["mapping_fingerprint"]
    assert detail["schedule_sensitive"] is True


def test_graphbrew_rabbit_propagates_schedule_sensitivity(tmp_path):
    _require(PR_BINARY)
    db_dir = tmp_path / "graphbrew-rabbit"
    db_dir.mkdir()
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
            "12:rabbit",
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
    assert row["reorder_schedule_sensitive"] is True
