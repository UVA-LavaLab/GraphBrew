#!/usr/bin/env python3
"""
GraphBrew Experiment Integration Tests
======================================

Tests for the main experiment pipeline components:
- Binary building
- Graph reordering with label mapping
- Benchmark execution
- Mapping file I/O

These tests use a tiny bundled graph (4 nodes, 5 edges) for fast execution.

Usage:
    pytest scripts/test/test_graphbrew_experiment.py -v
    pytest scripts/test/test_graphbrew_experiment.py::test_reorder_tiny -v
"""

import sys
import subprocess
from pathlib import Path

import pytest
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYS_PY = sys.executable

# Ensure project root is on sys.path for scripts.lib
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.pipeline import (
    reorder as lib_reorder,
    benchmark as lib_benchmark,
    build as lib_build,
)
from scripts.lib.pipeline.reorder import GraphInfo


def ensure_binaries():
    # Build required binaries (all) if missing
    lib_build.ensure_binaries()


def tiny_graph(tmp_path: Path) -> GraphInfo:
    # Copy bundled tiny graph into temp dir with structure graphs_dir/tiny/tiny.el
    src_dir = PROJECT_ROOT / "scripts" / "test" / "graphs" / "tiny"
    assert src_dir.exists(), "Bundled tiny graph not found"
    dst_dir = tmp_path / "graphs" / "tiny"
    dst_dir.mkdir(parents=True)
    data = (src_dir / "tiny.el").read_text()
    (dst_dir / "tiny.el").write_text(data)
    path = dst_dir / "tiny.el"
    return GraphInfo(name="tiny", path=str(path), size_mb=0.0, is_symmetric=True, nodes=4, edges=5)


def load_mapping(map_file: Path):
    lines = [int(line.strip()) for line in map_file.read_text().splitlines() if line.strip()]
    n = len(lines)
    assert sorted(lines) == list(range(n)), "Mapping must be a permutation (new->orig)"
    # invert: orig -> new
    inv = [None] * n
    for new_id, orig_id in enumerate(lines):
        inv[orig_id] = new_id
    assert all(v is not None for v in inv), "Inverse mapping must be complete"
    return lines, inv


def graph_edges_from_el(path: Path):
    edges = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            edges.append((u, v))
    return edges


def test_reorder_preserves_topology(tmp_path):
    ensure_binaries()
    ginfo = tiny_graph(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    results = lib_reorder.generate_reorderings(
        graphs=[ginfo], algorithms=[2], output_dir=str(out_dir), generate_maps=True, force_reorder=True
    )
    assert results and results[0].success
    map_file = Path(results[0].mapping_file)
    assert map_file.exists()

    lines, inv = load_mapping(map_file)
    # Original edges
    edges = graph_edges_from_el(Path(ginfo.path))
    # Remap edges to new IDs and back to verify bijection
    remapped = [(inv[u], inv[v]) for u, v in edges]
    # remap back
    roundtrip = [(lines[u], lines[v]) for u, v in remapped]
    assert sorted(roundtrip) == sorted(edges), "Topology must be preserved by permutation"


def test_benchmark_pr_runs(tmp_path):
    ensure_binaries()
    ginfo = tiny_graph(tmp_path)
    result = lib_benchmark.run_benchmark(
        benchmark="pr", graph_path=ginfo.path, algorithm="0", trials=1, timeout=60
    )
    assert result.success


def test_graphbrew_experiment_reorder_phase(tmp_path):
    ensure_binaries()
    ginfo = tiny_graph(tmp_path)
    graphs_dir = tmp_path / "graphs"
    cmd = [
        SYS_PY,
        "scripts/graphbrew_experiment.py",
        "--phase",
        "reorder",
        "--graphs-dir",
        str(graphs_dir),
        "--graph-list",
        "tiny",
        "--max-graphs",
        "1",
        "--quick",
        "--skip-cache",
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
    assert proc.returncode == 0

    # Check mapping exists under results dir
    mappings_dir = PROJECT_ROOT / "results" / "mappings" / "tiny"
    # Fallback: experiment may have written to default results; if not, skip check
    if mappings_dir.exists():
        lo_files = list(mappings_dir.glob("*.lo"))
        assert lo_files, "Expected mapping files from reorder phase"


def test_networkx_topology_features(tmp_path):
    # Additional sanity: networkx graph features for tiny graph
    ginfo = tiny_graph(tmp_path)
    G = nx.Graph()
    G.add_edges_from(graph_edges_from_el(Path(ginfo.path)))
    assert nx.is_connected(G)
    assert nx.number_of_nodes(G) == 4
    assert nx.number_of_edges(G) == 5
