#!/usr/bin/env python3
"""COrder baseline identity and determinism contracts."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.lib.core.utils import (
    CORDER_DEFAULT_VARIANT,
    CORDER_VARIANTS,
    algo_converter_opt,
    canonical_name_from_converter_opt,
)
from scripts.lib.pipeline.reorder import expand_algorithms_with_variants


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
PR_BINARY = PROJECT_ROOT / "bench" / "bin" / "pr"


def _require_binary(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} is not built")


def _generate_mapping(
    tmp_path: Path,
    option: str,
    threads: int,
    suffix: str,
) -> bytes:
    mapping = tmp_path / f"{suffix}.lo"
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
            "12",
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
    assert mapping.is_file() and mapping.stat().st_size > 0
    return mapping.read_bytes()


def test_corder_variant_naming_contract():
    assert CORDER_VARIANTS == ("legacy", "canonical")
    assert CORDER_DEFAULT_VARIANT == "legacy"
    assert canonical_name_from_converter_opt("10") == "CORDER"
    assert canonical_name_from_converter_opt("10:legacy") == "CORDER"
    assert (
        canonical_name_from_converter_opt("10:canonical")
        == "CORDER_canonical"
    )
    assert algo_converter_opt(10, "canonical") == "10:canonical"
    configs = expand_algorithms_with_variants(
        [10],
        expand_leiden_variants=True,
    )
    assert [(config.name, config.option_string) for config in configs] == [
        ("CORDER", "10:legacy"),
        ("CORDER_canonical", "10:canonical"),
    ]


@pytest.mark.parametrize("option", ["10", "10:canonical"])
def test_corder_mapping_is_thread_deterministic(tmp_path, option):
    _require_binary(CONVERTER)
    one = _generate_mapping(tmp_path, option, 1, "one")
    four = _generate_mapping(tmp_path, option, 4, "four")
    repeat = _generate_mapping(tmp_path, option, 4, "repeat")

    assert one == four == repeat


def test_canonical_corder_is_distinct_from_historical_1k(tmp_path):
    _require_binary(CONVERTER)
    legacy = _generate_mapping(tmp_path, "10", 1, "legacy")
    canonical = _generate_mapping(
        tmp_path, "10:canonical", 1, "canonical"
    )

    assert legacy != canonical


def test_cpp_self_recording_keeps_canonical_corder_identity(tmp_path):
    _require_binary(PR_BINARY)
    db_dir = tmp_path / "db"
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
            "10",
            "-s",
            "-o",
            "10:canonical",
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
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "CORDER_canonical"
