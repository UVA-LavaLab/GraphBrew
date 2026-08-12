"""Integration tests for deterministic SSSP weights and weighted relabeling."""

from __future__ import annotations

import os
import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.experiments.vldb import runner
from scripts.experiments.vldb.config import (
    ABLATION_CONFIGS,
    COMPOSE_VARIANTS,
    GRAPHBREW_VARIANTS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = PROJECT_ROOT / "bench" / "bin" / "converter"
SSSP = PROJECT_ROOT / "bench" / "bin" / "sssp"
PR_BINARIES = [
    PROJECT_ROOT / "bench" / "bin" / "pr",
    PROJECT_ROOT / "bench" / "bin" / "pr_spmv",
]
WORK_BFS = PROJECT_ROOT / "bench" / "bin_work" / "bfs"
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"


def run_converter(args: list[str], *, threads: int = 4) -> str:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["GRAPHBREW_DB_DIR"] = ""
    env["GRAPHBREW_TOPOLOGY_ANALYSIS"] = "0"
    result = subprocess.run(
        [str(CONVERTER), *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def read_weighted_edges(path: Path, labels: list[int]) -> list[tuple[int, int, int]]:
    edges = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        source, destination, weight = map(int, line.split())
        edges.append((labels[source], labels[destination], weight))
    return sorted(edges)


@pytest.fixture()
def base_graph(tmp_path: Path) -> Path:
    if not CONVERTER.exists():
        pytest.skip("converter binary is not built")
    graph = tmp_path / "tiny.sg"
    run_converter(["-f", str(TINY_GRAPH), "-s", "-b", str(graph)])
    return graph


@pytest.fixture()
def config_graph(tmp_path: Path) -> Path:
    if not CONVERTER.exists():
        pytest.skip("converter binary is not built")
    graph = tmp_path / "config.sg"
    run_converter(["-g", "10", "-b", str(graph)])
    return graph


def emit_weighted(
    graph: Path,
    tmp_path: Path,
    name: str,
    ordering: list[str],
    *,
    threads: int = 4,
) -> list[tuple[int, int, int]]:
    edges = tmp_path / f"{name}.wel"
    labels = tmp_path / f"{name}.lo"
    run_converter(
        [
            "-f", str(graph),
            "-W", "hash",
            "-w",
            *ordering,
            "-e", str(edges),
            "-q", str(labels),
        ],
        threads=threads,
    )
    label_values = [int(value) for value in labels.read_text().splitlines()]
    return read_weighted_edges(edges, label_values)


def test_hash_weights_are_thread_independent(base_graph: Path, tmp_path: Path):
    one = emit_weighted(base_graph, tmp_path, "one", ["-o", "0"], threads=1)
    four = emit_weighted(base_graph, tmp_path, "four", ["-o", "0"], threads=4)
    assert one == four
    assert len({weight for _, _, weight in one}) > 1


def test_reorder_timing_includes_validation_and_apply(
    base_graph: Path,
):
    output = run_converter([
        "-f", str(base_graph),
        "-o", "5",
    ])
    parsed = runner.parse_timing(output)
    assert parsed["reorder_time"] == pytest.approx(
        parsed["mapping_generation_time"]
        + parsed["reorder_validation_time"]
        + parsed["reorder_apply_time"],
        abs=2e-5,
    )


@pytest.mark.parametrize(
    "name,ordering",
    [
        ("original", ["-o", "0"]),
        ("dbg", ["-o", "5"]),
        ("rabbit", ["-o", "8:csr"]),
        ("graphbrew", ["-o", "12:rabbit:hubcluster"]),
        ("chain", ["-o", "12:rabbit", "-o", "5"]),
    ],
)
def test_weighted_edges_survive_reordering(
    base_graph: Path,
    tmp_path: Path,
    name: str,
    ordering: list[str],
):
    expected = emit_weighted(
        base_graph, tmp_path, "expected", ["-o", "0"],
    )
    observed = emit_weighted(base_graph, tmp_path, name, ordering)
    assert observed == expected


def run_sssp(graph: Path, ordering: list[str]) -> tuple[str, list[int], list[str]]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["GRAPHBREW_DB_DIR"] = ""
    env["GRAPHBREW_TOPOLOGY_ANALYSIS"] = "0"
    result = subprocess.run(
        [
            str(SSSP),
            "-f", str(graph),
            "-W", "hash",
            "-d", "16",
            "-n", "2",
            *ordering,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    checksum = re.search(
        r"Weight Checksum:\s*([0-9a-f]+)", result.stdout,
    )
    assert checksum is not None
    sources = [
        int(value)
        for value in re.findall(
            r"Source Original:\s*(-?\d+)", result.stdout,
        )
    ]
    fingerprints = re.findall(
        r"Distance Fingerprint:\s*([0-9a-f]+)", result.stdout,
    )
    assert len(sources) == 2
    assert len(fingerprints) == 2
    return checksum.group(1), sources, fingerprints


@pytest.mark.parametrize(
    "ordering",
    [
        ["-o", "5"],
        ["-o", "8:csr"],
        ["-o", "12:rabbit:hubcluster"],
        ["-o", "12:hrab"],
        [
            "-o",
            "12:leiden:compose:"
            "sg_none:comm_identity:intra_hubsort",
        ],
    ],
)
def test_sssp_fingerprints_are_reordering_invariant(
    base_graph: Path,
    ordering: list[str],
):
    if not SSSP.exists():
        pytest.skip("sssp binary is not built")
    expected = run_sssp(base_graph, ["-o", "0"])
    assert run_sssp(base_graph, ordering) == expected


def test_sssp_fingerprints_survive_map_loading(
    base_graph: Path,
    tmp_path: Path,
):
    if not SSSP.exists():
        pytest.skip("sssp binary is not built")
    mapping = tmp_path / "rabbit.lo"
    run_converter(
        [
            "-f", str(base_graph),
            "-o", "8:csr",
            "-q", str(mapping),
        ],
    )
    expected = run_sssp(base_graph, ["-o", "8:csr"])
    observed = run_sssp(base_graph, ["-o", f"13:{mapping}"])
    assert observed == expected


def test_sssp_source_repeats_form_consecutive_blocks(base_graph: Path):
    if not SSSP.exists():
        pytest.skip("sssp binary is not built")
    result = subprocess.run(
        [
            str(SSSP),
            "-f", str(base_graph),
            "-W", "hash",
            "-d", "16",
            "-n", "6",
            "-R", "2",
            "-o", "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    sources = [
        int(value)
        for value in re.findall(
            r"Source Original:\s*(-?\d+)", result.stdout,
        )
    ]
    assert len(sources) == 6
    assert sources[0] == sources[1]
    assert sources[2] == sources[3]
    assert sources[4] == sources[5]
    assert len({sources[0], sources[2], sources[4]}) == 3


@pytest.mark.parametrize("delta", ["True", "0", "-1", "1junk"])
def test_sssp_rejects_malformed_delta(base_graph: Path, delta: str):
    if not SSSP.exists():
        pytest.skip("sssp binary is not built")
    result = subprocess.run(
        [
            str(SSSP),
            "-f", str(base_graph),
            "-W", "hash",
            "-d", delta,
            "-n", "1",
            "-o", "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("binary", PR_BINARIES)
def test_pagerank_fixed_work_executes_exact_iterations(
    base_graph: Path,
    binary: Path,
):
    if not binary.exists():
        pytest.skip(f"{binary.name} binary is not built")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["GRAPHBREW_DB_DIR"] = ""
    env["GRAPHBREW_TOPOLOGY_ANALYSIS"] = "0"
    result = subprocess.run(
        [
            str(binary),
            "-f", str(base_graph),
            "-F",
            "-i", "5",
            "-t", "0.0001",
            "-n", "2",
            "-o", "0",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert re.search(r"PR Mode:\s*fixed-work", result.stdout)
    assert [
        int(value)
        for value in re.findall(r"Iterations:\s*(\d+)", result.stdout)
    ] == [5, 5]
    assert len(re.findall(r"Final Error:\s*[\d.eE+-]+", result.stdout)) == 2


def test_direct_chain_records_full_identity_and_cost(
    base_graph: Path,
    tmp_path: Path,
):
    binary = PR_BINARIES[0]
    if not binary.exists():
        pytest.skip("pr binary is not built")
    database = tmp_path / "db"
    database.mkdir()
    env = os.environ.copy()
    env["GRAPHBREW_DB_DIR"] = ""
    env["GRAPHBREW_TOPOLOGY_ANALYSIS"] = "0"
    result = subprocess.run(
        [
            str(binary),
            "-f", str(base_graph),
            "-F", "-i", "2",
            "-n", "1",
            "-D", str(database),
            "-o", "2",
            "-o", "5",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    records = json.loads((database / "benchmarks.json").read_text())
    record = records[-1]
    assert record["algorithm"] == "Sort+DBG"
    assert record["algorithm_id"] == -1
    details = record["reorder_details"]
    assert [item["algorithm"] for item in details] == ["Sort", "DBG"]
    assert record["reorder_time"] == pytest.approx(
        sum(item["reorder_time"] for item in details),
    )


def test_work_metrics_use_separate_binary(base_graph: Path):
    normal = PROJECT_ROOT / "bench" / "bin" / "bfs"
    if not normal.exists() or not WORK_BFS.exists():
        pytest.skip("normal/work BFS binaries are not built")
    env = os.environ.copy()
    env["GRAPHBREW_DB_DIR"] = ""
    env["GRAPHBREW_TOPOLOGY_ANALYSIS"] = "0"
    normal_result = subprocess.run(
        [str(normal), "-f", str(base_graph), "-n", "1", "-o", "0"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
        timeout=120,
    )
    work_result = subprocess.run(
        [str(WORK_BFS), "-f", str(base_graph), "-n", "1", "-o", "0"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert normal_result.returncode == 0
    assert work_result.returncode == 0
    assert "BFS Edges Examined" not in normal_result.stdout
    assert "BFS Edges Examined" in work_result.stdout


def test_verifier_failure_has_distinct_exit_code(base_graph: Path):
    binary = PR_BINARIES[0]
    if not binary.exists():
        pytest.skip("pr binary is not built")
    env = os.environ.copy()
    env["GRAPHBREW_DB_DIR"] = ""
    env["GRAPHBREW_TOPOLOGY_ANALYSIS"] = "0"
    result = subprocess.run(
        [
            str(binary),
            "-f", str(base_graph),
            "-F", "-i", "1",
            "-t", "1e-12",
            "-n", "1",
            "-o", "0",
            "-v",
        ],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert result.returncode == 3
    assert "Verification Failure" in result.stdout


PUBLISHED_GRAPHBREW_SPECS = sorted({
    *(f"12:{variant}" for variant in GRAPHBREW_VARIANTS),
    *(spec for _label, spec in COMPOSE_VARIANTS),
    *(
        entry["algo"]
        for entry in ABLATION_CONFIGS
        if entry["algo"].startswith("12:")
    ),
})


@pytest.mark.parametrize("spec", PUBLISHED_GRAPHBREW_SPECS)
def test_published_graphbrew_specs_match_real_effective_config(
    config_graph: Path,
    tmp_path: Path,
    spec: str,
):
    mapping = tmp_path / f"{spec.replace(':', '_')}.lo"
    output = run_converter(
        [
            "-f", str(config_graph),
            "-o", spec,
            "-q", str(mapping),
        ],
        threads=4,
    )
    configs = runner.parse_graphbrew_effective_configs(output)
    runner.validate_graphbrew_effective_configs(["-o", spec], configs)
    realized = runner.parse_graphbrew_realized_configs(output)
    runner.validate_graphbrew_realized_configs(
        ["-o", spec], configs, realized,
    )
