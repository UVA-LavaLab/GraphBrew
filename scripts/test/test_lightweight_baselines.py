#!/usr/bin/env python3
"""Deterministic shuffled-control and lightweight baseline contracts."""

import os
import subprocess
from pathlib import Path

import pytest

from scripts.lib.pipeline.benchmark import parse_benchmark_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
GRAPH_EDGES = """\
6 0
6 1
6 2
7 3
7 4
7 5
"""


def _require_converter():
    if not CONVERTER.exists():
        pytest.skip("converter is not built")


def _mapping(
    tmp_path: Path,
    option: str,
    threads: int,
    name: str,
) -> tuple[tuple[int, ...], str]:
    graph = tmp_path / "lightweight.el"
    graph.write_text(GRAPH_EDGES)
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
    mapping = tuple(int(value) for value in output.read_text().split())
    return mapping, result.stdout


@pytest.mark.parametrize("option", [str(value) for value in range(8)])
def test_lightweight_mappings_are_thread_deterministic(tmp_path, option):
    _require_converter()
    one, _ = _mapping(tmp_path, option, 1, f"{option}-one")
    four, _ = _mapping(tmp_path, option, 4, f"{option}-four")
    repeat, _ = _mapping(tmp_path, option, 4, f"{option}-repeat")

    assert one == four == repeat


def test_shuffled_control_has_pinned_seed_zero_mapping(tmp_path):
    _require_converter()
    mapping, output = _mapping(tmp_path, "1", 4, "random")
    _average, _reorder, timing = parse_benchmark_output(output)

    assert mapping == (2, 6, 0, 3, 4, 1, 5, 7)
    assert timing["random_seed"] == "0"


def test_hubsort_and_hubcluster_preserve_nonhub_ids_when_possible(
    tmp_path,
):
    _require_converter()
    hubsort, _ = _mapping(tmp_path, "3", 1, "hubsort")
    hubcluster, _ = _mapping(tmp_path, "4", 1, "hubcluster")

    assert hubsort == (7, 6, 2, 3, 4, 5, 1, 0)
    assert hubcluster == (6, 7, 2, 3, 4, 5, 0, 1)


def test_dbg_variants_remain_distinct_compact_mappings(tmp_path):
    _require_converter()
    hubsort, _ = _mapping(tmp_path, "3", 1, "hubsort")
    hubcluster, _ = _mapping(tmp_path, "4", 1, "hubcluster")
    hubsort_dbg, _ = _mapping(tmp_path, "6", 1, "hubsort-dbg")
    hubcluster_dbg, _ = _mapping(
        tmp_path, "7", 1, "hubcluster-dbg")

    assert hubsort_dbg == (7, 6, 0, 1, 2, 3, 4, 5)
    assert hubcluster_dbg == (6, 7, 0, 1, 2, 3, 4, 5)
    assert hubsort != hubsort_dbg
    assert hubcluster != hubcluster_dbg
