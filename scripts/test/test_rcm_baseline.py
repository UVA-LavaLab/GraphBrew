#!/usr/bin/env python3
"""RCM variant identity, determinism, and baseline-semantics tests."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.lib.core.utils import (
    RCM_VARIANTS,
    canonical_name_from_converter_opt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
PR_BINARY = PROJECT_ROOT / "bench" / "bin" / "pr"

# A path whose source IDs have poor bandwidth.
PATH_ORDER = (0, 5, 1, 6, 2, 7, 3, 8, 4, 9)
PATH_EDGES = tuple(zip(PATH_ORDER, PATH_ORDER[1:]))


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} is not built")


def _graph(tmp_path: Path) -> Path:
    path = tmp_path / "scrambled-path.el"
    path.write_text(
        "".join(f"{source} {target}\n" for source, target in PATH_EDGES)
    )
    return path


def _mapping(
    tmp_path: Path,
    option: str,
    threads: int,
    name: str,
) -> tuple[int, ...]:
    graph = _graph(tmp_path)
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
            "-f",
            str(graph),
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
    return tuple(int(value) for value in output.read_text().split())


def _bandwidth(new_to_source: tuple[int, ...]) -> int:
    source_to_new = [0] * len(new_to_source)
    for new_id, source_id in enumerate(new_to_source):
        source_to_new[source_id] = new_id
    return max(
        abs(source_to_new[source] - source_to_new[target])
        for source, target in PATH_EDGES
    )


def test_rcm_variant_naming_contract():
    assert RCM_VARIANTS == (
        "default", "mind", "bnf", "wavefront",
    )
    assert canonical_name_from_converter_opt("11") == "RCM_default"
    assert canonical_name_from_converter_opt("11:mind") == "RCM_mind"
    assert canonical_name_from_converter_opt("11:bnf") == "RCM_bnf"
    assert (
        canonical_name_from_converter_opt("11:wavefront")
        == "RCM_wavefront"
    )


@pytest.mark.parametrize(
    "option",
    ["11", "11:mind", "11:bnf", "11:wavefront"],
)
def test_rcm_variants_are_thread_deterministic(tmp_path, option):
    _require(CONVERTER)
    one = _mapping(tmp_path, option, 1, f"{option}-one")
    four = _mapping(tmp_path, option, 4, f"{option}-four")
    repeat = _mapping(tmp_path, option, 4, f"{option}-repeat")

    assert one == four == repeat


@pytest.mark.parametrize(
    "option",
    ["11:mind", "11:bnf", "11:wavefront"],
)
def test_explicit_single_pass_rcm_variants_reduce_path_bandwidth(
    tmp_path,
    option,
):
    _require(CONVERTER)
    mapping = _mapping(tmp_path, option, 1, option)

    assert _bandwidth(mapping) == 1


def test_historical_default_is_distinct_from_single_pass_mind(tmp_path):
    _require(CONVERTER)
    historical = _mapping(tmp_path, "11", 1, "historical")
    mind = _mapping(tmp_path, "11:mind", 1, "mind")

    assert historical != mind


@pytest.mark.parametrize("option,expected", [
    ("11", "RCM_default"),
    ("11:mind", "RCM_mind"),
    ("11:bnf", "RCM_bnf"),
    ("11:wavefront", "RCM_wavefront"),
])
def test_cpp_self_recording_preserves_rcm_variant(
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
    assert rows[0]["algorithm"] == expected
    assert rows[0]["requested_algorithm_spec"] == option
    expected_resolved = (
        "11:default" if option == "11" else option
    )
    assert rows[0]["algorithm_spec"] == expected_resolved
