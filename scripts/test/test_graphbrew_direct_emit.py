import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "bench/bin/converter"
TINY_GRAPH = ROOT / "scripts/test/data/tiny.el"
GRAPH_CASES = (
    "tiny.el",
    "disconnected.el",
    "isolated.el",
    "star.el",
    "path.el",
)
TREATMENTS = (
    ("intra_bfs_direct", "bfs-direct"),
    ("intra_bfs_compact", "bfs-compact"),
    ("intra_bfs_compact_direct", "bfs-compact-direct"),
)


def run_mapping(
    tmp_path: Path,
    intra: str,
    graph_name: str,
    threads: int,
) -> tuple[list[int], str]:
    if not CONVERTER.is_file():
        pytest.fail("converter binary is not built")
    output_path = tmp_path / (
        f"{Path(graph_name).stem}-{intra}-t{threads}.lo"
    )
    spec = (
        "12:leiden:compose:sg_none:comm_identity:"
        f"{intra}:cd_serial:norefine:1:1"
    )
    completed = subprocess.run(
        [
            str(CONVERTER),
            "-f", str(TINY_GRAPH.parent / graph_name),
            "-s",
            "-o", spec,
            "-q", str(output_path),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(threads),
            "OMP_PLACES": "cores",
            "OMP_PROC_BIND": "close",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    mapping = [
        int(value)
        for value in output_path.read_text().split()
    ]
    assert sorted(mapping) == list(range(len(mapping)))
    return mapping, completed.stdout


@pytest.mark.parametrize("graph_name", GRAPH_CASES)
@pytest.mark.parametrize("threads", (1, 4))
@pytest.mark.parametrize(
    ("treatment", "realized_name"),
    TREATMENTS,
)
def test_bfs_direct_preserves_bfs_mapping(
    tmp_path,
    graph_name,
    threads,
    treatment,
    realized_name,
):
    baseline, baseline_output = run_mapping(
        tmp_path,
        "intra_bfs",
        graph_name,
        threads,
    )
    direct, direct_output = run_mapping(
        tmp_path,
        treatment,
        graph_name,
        threads,
    )
    assert direct == baseline
    assert '"intra_community_order":"bfs"' in baseline_output
    assert (
        f'"intra_community_order":"{realized_name}"'
        in direct_output
    )
