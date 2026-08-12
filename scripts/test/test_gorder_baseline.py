#!/usr/bin/env python3
"""Gorder implementation identity, equivalence, and determinism tests."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.lib.core.utils import canonical_name_from_converter_opt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
PR_BINARY = PROJECT_ROOT / "bench" / "bin" / "pr"
DIRECTED_EDGES = """\
0 1
0 2
1 2
1 3
1 4
2 1
2 3
3 0
3 4
3 5
4 3
4 5
5 0
5 3
"""


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} is not built")


def _graph(tmp_path: Path) -> Path:
    graph = tmp_path / "gorder-directed.el"
    graph.write_text(DIRECTED_EDGES)
    return graph


def _mapping(
    tmp_path: Path,
    option: str,
    threads: int,
    name: str,
    symmetric: bool = False,
) -> bytes:
    graph = _graph(tmp_path)
    output = tmp_path / f"{name}.lo"
    env = {
        **os.environ,
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": str(threads),
    }
    cmd = [
        str(CONVERTER),
        "-f",
        str(graph),
    ]
    if symmetric:
        cmd.append("-s")
    cmd.extend(["-o", option, "-q", str(output)])
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return output.read_bytes()


def _generated_mapping(
    tmp_path: Path,
    option: str,
    threads: int,
    name: str,
) -> bytes:
    output = tmp_path / f"{name}.lo"
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
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return output.read_bytes()


def test_gorder_canonical_names_distinguish_implementations():
    assert canonical_name_from_converter_opt("9") == "GORDER"
    assert (
        canonical_name_from_converter_opt("9:gograph")
        == "GORDER_gograph"
    )
    assert canonical_name_from_converter_opt("9:csr") == "GORDER_csr"
    assert canonical_name_from_converter_opt("9:sym") == "GORDER_csr"
    assert canonical_name_from_converter_opt("9:fast") == "GORDER_fast"


@pytest.mark.parametrize("symmetric", [False, True])
def test_gograph_and_csr_are_mapping_equivalent(tmp_path, symmetric):
    _require(CONVERTER)
    legacy = _mapping(
        tmp_path, "9:gograph", 1, f"legacy-{symmetric}", symmetric
    )
    csr = _mapping(
        tmp_path, "9:csr", 1, f"csr-{symmetric}", symmetric
    )
    bare = _mapping(
        tmp_path, "9", 1, f"bare-{symmetric}", symmetric
    )

    assert legacy == csr == bare


@pytest.mark.parametrize("option", ["9:gograph", "9:csr", "9:fast"])
def test_gorder_variants_are_repeat_and_thread_deterministic(
    tmp_path,
    option,
):
    _require(CONVERTER)
    one = _mapping(tmp_path, option, 1, f"{option}-one")
    four = _mapping(tmp_path, option, 4, f"{option}-four")
    repeat = _mapping(tmp_path, option, 4, f"{option}-repeat")

    assert one == four == repeat


def test_fast_variant_is_a_distinct_relaxed_mapping(tmp_path):
    _require(CONVERTER)
    exact = _mapping(tmp_path, "9:csr", 1, "exact")
    fast = _mapping(tmp_path, "9:fast", 1, "fast")

    assert fast != exact


def test_fast_variant_is_deterministic_across_parallel_rounds(tmp_path):
    _require(CONVERTER)
    one = _generated_mapping(tmp_path, "9:fast", 1, "one")
    four = _generated_mapping(tmp_path, "9:fast", 4, "four")
    repeat = _generated_mapping(tmp_path, "9:fast", 4, "repeat")

    assert one == four == repeat


@pytest.mark.parametrize("option,expected", [
    ("9", "GORDER"),
    ("9:gograph", "GORDER_gograph"),
    ("9:csr", "GORDER_csr"),
    ("9:fast", "GORDER_fast"),
])
def test_cpp_self_recording_preserves_gorder_implementation(
    tmp_path,
    option,
    expected,
):
    _require(PR_BINARY)
    graph = _graph(tmp_path)
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
            str(graph),
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
    assert rows[0]["algorithm"] == expected
    assert rows[0]["algorithm_spec"] == option
