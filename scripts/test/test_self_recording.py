#!/usr/bin/env python3
"""
Self-Recording Database Verification Tests (v2.1)
==================================================

Verify that C++ binaries correctly write benchmark results and graph
properties to JSON files when self-recording is enabled via --db-dir
or the GRAPHBREW_DB_DIR environment variable.

Tests:
  V1: Smoke test — PR benchmark writes benchmarks.json
  V2: Graph properties — builder writes graph_properties.json
  V3: Dual-write consistency — C++ data matches Python stdout parsing
  V4: Multi-benchmark append — sequential runs accumulate records
  V5: Env var fallback — GRAPHBREW_DB_DIR works without --db-dir flag

Usage:
    pytest scripts/test/test_self_recording.py -v
    pytest scripts/test/test_self_recording.py::test_v1_smoke_benchmarks_json -v
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"


def _require_binary(name: str) -> Path:
    """Return path to a benchmark binary, skipping if not built."""
    binary = BIN_DIR / name
    if not binary.exists():
        pytest.skip(f"Binary {name} not built; run 'make bench/bin/{name}' first")
    return binary


def _run(binary: Path, graph: Path, extra_args: list[str] | None = None,
         env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a benchmark binary and return the CompletedProcess."""
    cmd = [str(binary), "-f", str(graph), "-s", "-n", "1"]
    if extra_args:
        cmd.extend(extra_args)
    run_env = os.environ.copy()
    # Remove GRAPHBREW_DB_DIR from inherited env unless explicitly provided
    run_env.pop("GRAPHBREW_DB_DIR", None)
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=run_env,
    )


# ============================================================================
# V1: Smoke Test — benchmarks.json written
# ============================================================================

def test_v1_smoke_benchmarks_json(tmp_path):
    """PR benchmark with -D writes benchmarks.json with correct schema."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    result = _run(pr, TINY_GRAPH, ["-D", str(db_dir) + "/"])
    assert result.returncode == 0, f"PR failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    bench_file = db_dir / "benchmarks.json"
    assert bench_file.exists(), f"benchmarks.json not created in {db_dir}"

    data = json.loads(bench_file.read_text())
    assert isinstance(data, list), "benchmarks.json should be a JSON array"
    assert len(data) >= 1, "Expected at least 1 record"

    rec = data[0]
    # Required fields
    for field in ["graph", "algorithm", "benchmark", "time_seconds",
                   "trials", "nodes", "edges", "success"]:
        assert field in rec, f"Missing field '{field}' in record"

    assert rec["graph"] == "tiny", f"Expected graph='tiny', got '{rec['graph']}'"
    assert rec["benchmark"] == "pr", f"Expected benchmark='pr', got '{rec['benchmark']}'"
    assert rec["time_seconds"] > 0, "time_seconds should be > 0"
    assert rec["success"] is True
    assert rec["nodes"] > 0
    assert rec["edges"] > 0

    # v2.1: trial_details should exist
    if "trial_details" in rec:
        assert isinstance(rec["trial_details"], list)
        assert len(rec["trial_details"]) >= 1
        trial = rec["trial_details"][0]
        assert "time_seconds" in trial
        assert trial["time_seconds"] > 0


# ============================================================================
# V2: Graph Properties — graph_properties.json written
# ============================================================================

def test_v2_graph_properties_json(tmp_path):
    """Builder writes graph_properties.json with topology features."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    result = _run(pr, TINY_GRAPH, ["-D", str(db_dir) + "/"])
    assert result.returncode == 0

    props_file = db_dir / "graph_properties.json"
    assert props_file.exists(), f"graph_properties.json not created in {db_dir}"

    data = json.loads(props_file.read_text())
    assert isinstance(data, dict), "graph_properties.json should be a JSON object"
    assert "tiny" in data, f"Expected 'tiny' key; got keys: {list(data.keys())}"

    props = data["tiny"]

    # Basic fields always recorded (even for small graphs < 100 nodes)
    for field in ["nodes", "edges", "avg_degree", "density"]:
        assert field in props, f"Missing field '{field}' in graph properties"

    assert props["nodes"] > 0, "nodes should be > 0"
    assert props["edges"] > 0, "edges should be > 0"
    assert props["avg_degree"] > 0, "avg_degree should be > 0"

    # Tiny graph: 4 nodes, 5 undirected edges → 10 directed edges
    assert props["nodes"] == 4, f"Expected 4 nodes, got {props['nodes']}"

    # For graphs ≥ 100 nodes, full topology fields are also present:
    # modularity, degree_variance, hub_concentration, clustering_coefficient,
    # packing_factor, forward_edge_fraction, working_set_ratio, etc.


# ============================================================================
# V3: Dual-Write Consistency — C++ data matches Python stdout parsing
# ============================================================================

def test_v3_dual_write_consistency(tmp_path):
    """C++ self-recorded time matches Python-parsed stdout time (±10%)."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    result = _run(pr, TINY_GRAPH, ["-D", str(db_dir) + "/"])
    assert result.returncode == 0

    # Parse average time from stdout (same regex Python uses)
    output = result.stdout + result.stderr
    avg_match = re.search(r"Average Time:\s*([\d.eE+-]+)", output)
    assert avg_match, f"Could not parse 'Average Time' from stdout:\n{output}"
    stdout_avg = float(avg_match.group(1))

    # Read C++ self-recorded time
    bench_file = db_dir / "benchmarks.json"
    data = json.loads(bench_file.read_text())
    assert len(data) >= 1
    cpp_avg = data[0]["time_seconds"]

    # Both should be the same value (exact match, since it's the same computation)
    # Allow small floating-point tolerance
    if stdout_avg > 0:
        rel_diff = abs(cpp_avg - stdout_avg) / stdout_avg
        assert rel_diff < 0.10, (
            f"Timing mismatch: C++={cpp_avg:.6f}, stdout={stdout_avg:.6f}, "
            f"rel_diff={rel_diff:.4f}"
        )
    else:
        # Both should be zero or very small
        assert cpp_avg < 0.1, f"stdout=0 but C++ recorded {cpp_avg}"


# ============================================================================
# V4: Multi-Benchmark Append — sequential runs accumulate records
# ============================================================================

def test_v4_multi_benchmark_append(tmp_path):
    """Running pr, bfs, cc sequentially appends all to same benchmarks.json."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    db_arg = str(db_dir) + "/"

    benchmarks_run = []
    for name in ["pr", "bfs", "cc"]:
        binary = _require_binary(name)
        result = _run(binary, TINY_GRAPH, ["-D", db_arg])
        assert result.returncode == 0, (
            f"{name} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        benchmarks_run.append(name)

    bench_file = db_dir / "benchmarks.json"
    assert bench_file.exists()

    data = json.loads(bench_file.read_text())
    assert isinstance(data, list)

    # All three benchmarks should be present
    recorded_benchmarks = {rec["benchmark"] for rec in data}
    for name in benchmarks_run:
        assert name in recorded_benchmarks, (
            f"Benchmark '{name}' missing from records. "
            f"Found: {recorded_benchmarks}"
        )

    # At least 3 records (one per benchmark, all with algorithm "Original")
    assert len(data) >= len(benchmarks_run), (
        f"Expected ≥{len(benchmarks_run)} records, got {len(data)}"
    )

    # Verify no corruption — all records have required fields
    for rec in data:
        assert "graph" in rec
        assert "benchmark" in rec
        assert "time_seconds" in rec
        assert rec["success"] is True

    # Graph properties should also exist (builder runs once per binary)
    props_file = db_dir / "graph_properties.json"
    assert props_file.exists()
    props_data = json.loads(props_file.read_text())
    assert "tiny" in props_data


# ============================================================================
# V5: Environment Variable Fallback — GRAPHBREW_DB_DIR works without -D
# ============================================================================

def test_v5_env_var_fallback(tmp_path):
    """GRAPHBREW_DB_DIR env var enables self-recording without -D flag."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "env_db"
    db_dir.mkdir()

    # Run WITHOUT -D, but WITH GRAPHBREW_DB_DIR set
    result = _run(
        pr, TINY_GRAPH,
        extra_args=None,  # No -D flag
        env={"GRAPHBREW_DB_DIR": str(db_dir) + "/"},
    )
    assert result.returncode == 0, f"PR failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    bench_file = db_dir / "benchmarks.json"
    assert bench_file.exists(), (
        f"benchmarks.json not created via GRAPHBREW_DB_DIR. "
        f"Dir contents: {list(db_dir.iterdir())}"
    )

    data = json.loads(bench_file.read_text())
    assert len(data) >= 1
    assert data[0]["graph"] == "tiny"


def test_v5_no_env_no_flag_no_recording(tmp_path):
    """Without -D or GRAPHBREW_DB_DIR, self-recording stays disabled."""
    pr = _require_binary("pr")

    # Ensure no env var, no -D flag
    result = _run(pr, TINY_GRAPH)
    assert result.returncode == 0

    # The default data dir should NOT have a new benchmarks.json created
    # (We can't easily check the default dir, but we verify the binary
    # ran successfully without errors related to recording)
    assert "Cannot write" not in result.stderr, (
        f"Unexpected write error in stderr: {result.stderr}"
    )


# ============================================================================
# V6: File Locking — concurrent writes don't corrupt
# ============================================================================

def test_v6_concurrent_writes(tmp_path):
    """Two benchmark binaries running simultaneously don't corrupt JSON."""
    pr = _require_binary("pr")
    bfs = _require_binary("bfs")
    db_dir = tmp_path / "concurrent"
    db_dir.mkdir()
    db_arg = str(db_dir) + "/"

    env = os.environ.copy()
    env.pop("GRAPHBREW_DB_DIR", None)

    # Launch both simultaneously
    cmd_pr = [str(pr), "-f", str(TINY_GRAPH), "-s", "-n", "1", "-D", db_arg]
    cmd_bfs = [str(bfs), "-f", str(TINY_GRAPH), "-s", "-n", "1", "-D", db_arg]

    proc_pr = subprocess.Popen(cmd_pr, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    proc_bfs = subprocess.Popen(cmd_bfs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    proc_pr.wait(timeout=120)
    proc_bfs.wait(timeout=120)

    assert proc_pr.returncode == 0, f"PR failed: {proc_pr.stderr.read()}"
    assert proc_bfs.returncode == 0, f"BFS failed: {proc_bfs.stderr.read()}"

    # benchmarks.json should be valid JSON with both records
    bench_file = db_dir / "benchmarks.json"
    assert bench_file.exists()

    data = json.loads(bench_file.read_text())
    assert isinstance(data, list)
    benchmarks = {rec["benchmark"] for rec in data}
    assert "pr" in benchmarks, f"PR record missing. Found: {benchmarks}"
    assert "bfs" in benchmarks, f"BFS record missing. Found: {benchmarks}"

    # graph_properties.json should also be valid
    props_file = db_dir / "graph_properties.json"
    if props_file.exists():
        props_data = json.loads(props_file.read_text())
        assert isinstance(props_data, dict)


# ============================================================================
# C4: Integration Tests — Orchestrator Helpers & Pipeline
# ============================================================================

def test_dispatch_tool_restores_argv():
    """_dispatch_tool restores sys.argv even if the tool raises."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.graphbrew_experiment import _dispatch_tool

    original_argv = sys.argv[:]

    # Successful dispatch — verify argv restored
    called = []

    def _fake_main():
        called.append(sys.argv[:])

    _dispatch_tool(_fake_main, ["--foo", "bar"])
    assert called == [[sys.argv[0], "--foo", "bar"]]
    assert sys.argv == original_argv, "sys.argv not restored after successful dispatch"

    # Failing dispatch — verify argv still restored
    def _failing_main():
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError, match="intentional"):
        _dispatch_tool(_failing_main)
    assert sys.argv == original_argv, "sys.argv not restored after failed dispatch"


def test_graphprops_store_reads_cpp_data(tmp_path):
    """GraphPropsStore can read what C++ self-recording writes."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    # Run C++ binary to produce graph_properties.json
    result = _run(pr, TINY_GRAPH, ["-D", str(db_dir) + "/"])
    assert result.returncode == 0

    # Verify Python GraphPropsStore reads the same data
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.lib.core.datastore import GraphPropsStore

    store = GraphPropsStore(db_dir / "graph_properties.json")
    props = store.get("tiny")
    assert props is not None, "GraphPropsStore.get('tiny') returned None"
    assert props["nodes"] == 4
    assert props["edges"] > 0
    assert props["avg_degree"] > 0


def test_benchmark_store_reads_cpp_data(tmp_path):
    """BenchmarkStore can read what C++ self-recording writes."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    result = _run(pr, TINY_GRAPH, ["-D", str(db_dir) + "/"])
    assert result.returncode == 0

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.lib.core.datastore import BenchmarkStore

    store = BenchmarkStore(db_dir / "benchmarks.json")
    records = store.query(graph="tiny")
    assert len(records) >= 1, "BenchmarkStore found no records for 'tiny'"
    assert records[0]["benchmark"] == "pr"
    assert records[0]["time_seconds"] > 0


def test_phase6_reads_from_graphprops_store(tmp_path):
    """Phase 6 can read topology features from C++-written graph_properties.json.

    This verifies the C5 refactor: instead of running a binary and parsing
    stdout with regex, Phase 6 reads from GraphPropsStore.
    """
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    # Produce data via C++ self-recording
    result = _run(pr, TINY_GRAPH, ["-D", str(db_dir) + "/"])
    assert result.returncode == 0

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.lib.core.datastore import GraphPropsStore

    store = GraphPropsStore(db_dir / "graph_properties.json")
    stored = store.get("tiny")
    assert stored is not None

    # Build features dict the same way Phase 6 does
    features = {
        'modularity': 0.5,  # default
        'degree_variance': stored.get('degree_variance', 1.0),
        'hub_concentration': stored.get('hub_concentration', 0.3),
        'avg_degree': stored.get('avg_degree', 10.0),
        'density': stored.get('density', 0.0),
        'nodes': stored.get('nodes', 0),
        'edges': stored.get('edges', 0),
    }

    # Verify features are populated (not all defaults)
    assert features['nodes'] == 4
    assert features['edges'] > 0
    assert features['avg_degree'] > 0
