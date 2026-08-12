#!/usr/bin/env python3
"""GoGraph core variant and provenance contracts."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.lib.core.datastore import BenchmarkStore
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


def _directed_graph(tmp_path: Path) -> Path:
    graph = tmp_path / "gograph-directed.el"
    graph.write_text(DIRECTED_EDGES)
    return graph


def _mapping(tmp_path: Path, option: str, name: str) -> bytes:
    output = tmp_path / f"{name}.lo"
    graph = _directed_graph(tmp_path)
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
            str(graph),
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


def test_gograph_canonical_names_are_variant_specific():
    assert (
        canonical_name_from_converter_opt("16")
        == "GOGRAPHORDER_default"
    )
    assert (
        canonical_name_from_converter_opt("16:naive")
        == "GOGRAPHORDER_naive"
    )
    assert (
        canonical_name_from_converter_opt("16:fast")
        == "GOGRAPHORDER_fast"
    )


def test_default_and_naive_share_the_m_maximizing_mapping(tmp_path):
    _require(CONVERTER)
    assert _mapping(tmp_path, "16", "default") == _mapping(
        tmp_path, "16:naive", "naive"
    )


def test_fast_variant_is_not_silently_ignored(tmp_path):
    _require(CONVERTER)
    default = _mapping(tmp_path, "16", "default")
    fast = _mapping(tmp_path, "16:fast", "fast")

    assert fast != default


def test_symmetric_input_reports_constant_objective(tmp_path):
    _require(CONVERTER)
    graph = _directed_graph(tmp_path)
    output = tmp_path / "symmetric.lo"
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
            str(graph),
            "-s",
            "-o",
            "16",
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
    assert "constant-on-symmetric-graph" in result.stdout


@pytest.mark.parametrize("option,expected", [
    ("16", "GOGRAPHORDER_default"),
    ("16:naive", "GOGRAPHORDER_naive"),
    ("16:fast", "GOGRAPHORDER_fast"),
])
def test_cpp_self_recording_preserves_variant_identity(
    tmp_path,
    option,
    expected,
):
    _require(PR_BINARY)
    db_dir = tmp_path / expected
    db_dir.mkdir()
    graph = _directed_graph(tmp_path)
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
    store = BenchmarkStore(db_dir / "benchmarks.json")
    assert store.observations()[0]["algorithm"] == expected
