#!/usr/bin/env python3
"""
Cache Simulation Integration Tests
==================================

Tests for the cache simulation pipeline:
- Running cache simulations with different configurations
- Parsing simulation output
- Verifying cache metrics

These tests use a tiny bundled graph (4 nodes, 5 edges) for fast execution.
Requires cache simulation binaries to be built (make sim).

Usage:
    pytest scripts/test/test_cache_simulation.py -v
    pytest scripts/test/test_cache_simulation.py -k "test_sim" -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.pipeline.cache import run_cache_simulation
from scripts.lib.pipeline.reorder import GraphInfo
from scripts.lib.pipeline import build as lib_build


def tiny_graph(tmp_path: Path) -> GraphInfo:
    src_dir = PROJECT_ROOT / "scripts" / "test" / "graphs" / "tiny"
    assert src_dir.exists(), "Bundled tiny graph not found"
    dst_dir = tmp_path / "graphs" / "tiny"
    dst_dir.mkdir(parents=True)
    data = (src_dir / "tiny.el").read_text()
    (dst_dir / "tiny.el").write_text(data)
    path = dst_dir / "tiny.el"
    return GraphInfo(name="tiny", path=str(path), size_mb=0.0, is_symmetric=True, nodes=4, edges=5)


def sim_binary_exists():
    return (PROJECT_ROOT / "bench" / "bin_sim" / "pr").exists()


def ensure_pr_binary():
    # Build pr binary if missing (for label map usage); cache sim uses pr_sim
    lib_build.ensure_binaries()


def test_cache_simulation_runs_or_skips(tmp_path):
    ensure_pr_binary()
    ginfo = tiny_graph(tmp_path)

    if not sim_binary_exists():
        pytest.skip("bench/bin_sim/pr not built; skipping cache simulation test")

    result = run_cache_simulation(
        benchmark="pr",
        graph_path=ginfo.path,
        algorithm=0,
        label_map_path=None,
        symmetric=True,
        timeout=60,
    )
    assert result.success, f"Cache simulation failed: {result.error}"
    assert result.l1_miss_rate >= 0
    assert result.l2_miss_rate >= 0
    assert result.l3_miss_rate >= 0
