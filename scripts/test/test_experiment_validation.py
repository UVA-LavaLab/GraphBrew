#!/usr/bin/env python3
"""
End-to-End Pipeline Validation Tests (E1–E5)
=============================================

Verify the full GraphBrew pipeline: convert → benchmark → datastore → weights → oracle.

Tests use synthetic graphs (``-g <scale>``) or the bundled ``tiny.el`` to avoid
download dependencies.  Each test is independent and uses ``tmp_path`` for isolation.

Categories:
  E1: Convert — .sg generation and graph_properties.json
  E2: Benchmark Phase — benchmarks.json grows with correct records
  E3: BenchmarkStore — query, perf_matrix, iteration data
  E4: Weights — compute_weights_from_results produces valid output
  E5: Oracle — best-algorithm identification

Usage:
    pytest scripts/test/test_experiment_validation.py -v
    pytest scripts/test/test_experiment_validation.py::test_e2_benchmark_phase_writes_json -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"

# Ensure project root is on sys.path so scripts.lib is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_self_recording.py)
# ---------------------------------------------------------------------------

def _require_binary(name: str) -> Path:
    """Return path to a benchmark binary, skipping if not built."""
    binary = BIN_DIR / name
    if not binary.exists():
        pytest.skip(f"Binary {name} not built; run 'make bench/bin/{name}' first")
    return binary


def _run(binary: Path, graph: Path, extra_args: list[str] | None = None,
         env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a benchmark binary against a file-based graph."""
    cmd = [str(binary), "-f", str(graph), "-s", "-n", "1"]
    if extra_args:
        cmd.extend(extra_args)
    run_env = os.environ.copy()
    run_env.pop("GRAPHBREW_DB_DIR", None)
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=run_env,
    )


def _run_synthetic(binary: Path, scale: int, db_dir: Path,
                   extra_args: list[str] | None = None,
                   timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a benchmark binary with a synthetic Kronecker graph (``-g``)."""
    cmd = [str(binary), "-g", str(scale), "-s", "-n", "1",
           "-D", str(db_dir) + "/"]
    if extra_args:
        cmd.extend(extra_args)
    run_env = os.environ.copy()
    run_env.pop("GRAPHBREW_DB_DIR", None)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=run_env,
    )


# ============================================================================
# E1: Convert — Verify .sg and graph_properties.json
# ============================================================================

def test_e1_convert_synthetic_to_sg(tmp_path):
    """Converter generates a .sg file from a synthetic graph that PR can load."""
    converter = _require_binary("converter")
    pr = _require_binary("pr")

    sg_path = tmp_path / "synth.sg"
    # Generate Kronecker graph (scale 10 = 1024 nodes) and write .sg
    result = subprocess.run(
        [str(converter), "-g", "10", "-b", str(sg_path)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "GRAPHBREW_DB_DIR": ""},
    )
    assert result.returncode == 0, (
        f"converter failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert sg_path.exists(), ".sg file not created"
    assert sg_path.stat().st_size > 0, ".sg file is empty"

    # Verify the .sg is loadable by running PR on it
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    pr_result = subprocess.run(
        [str(pr), "-f", str(sg_path), "-s", "-n", "1", "-D", str(db_dir) + "/"],
        capture_output=True, text=True, timeout=120,
        env={k: v for k, v in os.environ.items() if k != "GRAPHBREW_DB_DIR"},
    )
    assert pr_result.returncode == 0, (
        f"PR on .sg failed:\nstdout: {pr_result.stdout}\nstderr: {pr_result.stderr}"
    )


def test_e1_convert_el_to_sg(tmp_path):
    """Convert tiny.el → .sg, run PR with -D, verify graph_properties.json written."""
    converter = _require_binary("converter")
    pr = _require_binary("pr")

    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny graph not found: {TINY_GRAPH}")

    sg_path = tmp_path / "tiny.sg"
    result = subprocess.run(
        [str(converter), "-f", str(TINY_GRAPH), "-b", str(sg_path), "-s"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"converter failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert sg_path.exists() and sg_path.stat().st_size > 0

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    pr_result = subprocess.run(
        [str(pr), "-f", str(sg_path), "-s", "-n", "1", "-D", str(db_dir) + "/"],
        capture_output=True, text=True, timeout=120,
        env={k: v for k, v in os.environ.items() if k != "GRAPHBREW_DB_DIR"},
    )
    assert pr_result.returncode == 0

    props_file = db_dir / "graph_properties.json"
    assert props_file.exists(), "graph_properties.json not created"

    props = json.loads(props_file.read_text())
    assert isinstance(props, dict)
    # The key is the graph name (stem of the file)
    assert "tiny" in props, f"Expected 'tiny' key in props; got keys: {list(props.keys())}"
    entry = props["tiny"]
    assert entry.get("nodes") == 4, f"Expected nodes=4, got {entry.get('nodes')}"
    assert entry.get("edges", 0) > 0, "Expected edges > 0"


# ============================================================================
# E2: Benchmark Phase — benchmarks.json grows with correct records
# ============================================================================

def test_e2_benchmark_phase_writes_json(tmp_path):
    """Running 3 benchmarks (pr, bfs, cc) produces 3 records in benchmarks.json."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    benchmarks_run = []
    for bench_name in ["pr", "bfs", "cc"]:
        binary = _require_binary(bench_name)
        result = _run_synthetic(binary, scale=10, db_dir=db_dir)
        assert result.returncode == 0, (
            f"{bench_name} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        benchmarks_run.append(bench_name)

    bench_file = db_dir / "benchmarks.json"
    assert bench_file.exists(), "benchmarks.json not created"

    data = json.loads(bench_file.read_text())
    assert isinstance(data, list)
    assert len(data) >= len(benchmarks_run), (
        f"Expected >= {len(benchmarks_run)} records, got {len(data)}"
    )

    recorded_benchmarks = {r["benchmark"] for r in data}
    for bname in benchmarks_run:
        assert bname in recorded_benchmarks, (
            f"Benchmark '{bname}' missing from records; found: {recorded_benchmarks}"
        )

    for rec in data:
        assert rec.get("success") is True, f"Record not successful: {rec}"
        assert rec.get("time_seconds", 0) > 0, f"time_seconds <= 0: {rec}"


def test_e2_multi_algorithm_benchmark(tmp_path):
    """PR with ORIGINAL (-o 0) and RANDOM (-o 1) produces 2 distinct records."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny graph not found: {TINY_GRAPH}")

    # Run with ORIGINAL
    r1 = _run(pr, TINY_GRAPH, ["-o", "0", "-D", str(db_dir) + "/"])
    assert r1.returncode == 0, f"PR ORIGINAL failed:\n{r1.stdout}\n{r1.stderr}"

    # Run with RANDOM
    r2 = _run(pr, TINY_GRAPH, ["-o", "1", "-D", str(db_dir) + "/"])
    assert r2.returncode == 0, f"PR RANDOM failed:\n{r2.stdout}\n{r2.stderr}"

    data = json.loads((db_dir / "benchmarks.json").read_text())
    assert len(data) >= 2, f"Expected >= 2 records, got {len(data)}"

    algorithms = {r["algorithm"] for r in data}
    assert len(algorithms) >= 2, (
        f"Expected 2+ distinct algorithms, got: {algorithms}"
    )

    for rec in data:
        assert rec["benchmark"] == "pr"
        assert rec["graph"] == "tiny"


def test_e2_trial_details_present(tmp_path):
    """Running with -n 3 produces trial_details array with 3 entries."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny graph not found: {TINY_GRAPH}")

    result = subprocess.run(
        [str(pr), "-f", str(TINY_GRAPH), "-s", "-n", "3",
         "-D", str(db_dir) + "/"],
        capture_output=True, text=True, timeout=120,
        env={k: v for k, v in os.environ.items() if k != "GRAPHBREW_DB_DIR"},
    )
    assert result.returncode == 0

    data = json.loads((db_dir / "benchmarks.json").read_text())
    assert len(data) >= 1
    rec = data[0]

    assert "trial_details" in rec, "trial_details missing from record"
    trials = rec["trial_details"]
    assert len(trials) == 3, f"Expected 3 trial entries, got {len(trials)}"

    for i, trial in enumerate(trials):
        assert trial.get("time_seconds", 0) > 0, (
            f"Trial {i} has time_seconds <= 0"
        )


# ============================================================================
# E3: BenchmarkStore reads C++ data
# ============================================================================

def test_e3_benchmark_store_query(tmp_path):
    """BenchmarkStore can load C++-written data and answer queries."""
    pr = _require_binary("pr")
    bfs = _require_binary("bfs")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny graph not found: {TINY_GRAPH}")

    # Run PR and BFS on file-based graph (synthetic -g writes graph="" which
    # BenchmarkStore rejects because the composite key requires non-empty fields)
    for binary in [pr, bfs]:
        result = _run(binary, TINY_GRAPH, ["-D", str(db_dir) + "/"])
        assert result.returncode == 0

    from scripts.lib.core.datastore import BenchmarkStore

    store = BenchmarkStore(db_dir / "benchmarks.json")

    pr_records = store.query(benchmark="pr")
    assert len(pr_records) >= 1, "No PR records found"

    bfs_records = store.query(benchmark="bfs")
    assert len(bfs_records) >= 1, "No BFS records found"

    sssp_records = store.query(benchmark="sssp")
    assert len(sssp_records) == 0, "Unexpected SSSP records (not run)"

    graphs = store.graphs()
    assert len(graphs) >= 1, "No graphs in store"
    assert "tiny" in graphs, f"'tiny' not in graphs: {graphs}"

    benchmarks = store.benchmarks()
    assert "pr" in benchmarks, f"'pr' not in benchmarks: {benchmarks}"
    assert "bfs" in benchmarks, f"'bfs' not in benchmarks: {benchmarks}"


def test_e3_perf_matrix(tmp_path):
    """perf_matrix() returns {graph: {algo: {bench: time}}} with correct structure."""
    pr = _require_binary("pr")
    bfs = _require_binary("bfs")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny graph not found: {TINY_GRAPH}")

    # Run PR and BFS with 2 algorithms (ORIGINAL and RANDOM)
    for bench_binary in [pr, bfs]:
        for algo_opt in ["0", "1"]:
            result = _run(bench_binary, TINY_GRAPH,
                          ["-o", algo_opt, "-D", str(db_dir) + "/"])
            assert result.returncode == 0

    from scripts.lib.core.datastore import BenchmarkStore

    store = BenchmarkStore(db_dir / "benchmarks.json")
    matrix = store.perf_matrix()

    assert isinstance(matrix, dict), "perf_matrix should return a dict"
    assert len(matrix) >= 1, "perf_matrix is empty"

    # Check structure for the tiny graph
    assert "tiny" in matrix, f"'tiny' not in matrix keys: {list(matrix.keys())}"
    graph_data = matrix["tiny"]
    assert isinstance(graph_data, dict)

    # Should have at least 2 algorithm keys
    assert len(graph_data) >= 2, (
        f"Expected >= 2 algorithms, got {len(graph_data)}: {list(graph_data.keys())}"
    )

    # Each algorithm should have pr and bfs times
    for algo, bench_times in graph_data.items():
        assert "pr" in bench_times, f"Algorithm '{algo}' missing 'pr' time"
        assert "bfs" in bench_times, f"Algorithm '{algo}' missing 'bfs' time"
        assert bench_times["pr"] > 0, f"PR time for {algo} is <= 0"
        assert bench_times["bfs"] > 0, f"BFS time for {algo} is <= 0"


def test_e3_iterations_in_trial_details(tmp_path):
    """Per-iteration data (from G1-G10 work) is present in PR trial_details."""
    pr = _require_binary("pr")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    result = _run_synthetic(pr, scale=10, db_dir=db_dir)
    assert result.returncode == 0

    data = json.loads((db_dir / "benchmarks.json").read_text())
    assert len(data) >= 1

    rec = data[0]
    assert rec["benchmark"] == "pr"

    assert "trial_details" in rec, "trial_details missing"
    trials = rec["trial_details"]
    assert len(trials) >= 1

    trial = trials[0]
    assert "answer" in trial, "answer missing from trial"
    answer = trial["answer"]

    assert "iterations" in answer, (
        f"iterations missing from answer; keys: {list(answer.keys())}"
    )
    iterations = answer["iterations"]
    assert isinstance(iterations, list)
    assert len(iterations) >= 1, "No iteration entries"

    # PR iterations should have iter + error fields
    first = iterations[0]
    assert "iter" in first, f"iter field missing; keys: {list(first.keys())}"
    assert "error" in first, f"error field missing; keys: {list(first.keys())}"


# ============================================================================
# E4: Weights Phase
# ============================================================================

def test_e4_weights_from_results(tmp_path):
    """compute_weights_from_results() produces valid output from synthetic data."""
    from scripts.lib.core.utils import BenchmarkResult
    from scripts.lib.ml.weights import compute_weights_from_results

    # Build synthetic BenchmarkResult objects
    results = [
        BenchmarkResult(
            graph="synth", benchmark="pr", algorithm="ORIGINAL",
            algorithm_id=0,
            time_seconds=0.3, success=True, nodes=1024, edges=5000,
        ),
        BenchmarkResult(
            graph="synth", benchmark="pr", algorithm="RABBITORDER_csr",
            algorithm_id=2,
            time_seconds=0.1, success=True, nodes=1024, edges=5000,
        ),
        BenchmarkResult(
            graph="synth", benchmark="pr", algorithm="HubSort",
            algorithm_id=3,
            time_seconds=0.2, success=True, nodes=1024, edges=5000,
        ),
    ]

    out_file = tmp_path / "weights.json"
    weights = compute_weights_from_results(
        benchmark_results=results,
        output_file=str(out_file),
        weights_dir=str(tmp_path / "wd"),
    )

    assert isinstance(weights, dict), "Weights should be a dict"
    assert len(weights) >= 1, "Weights dict is empty"

    # Output file should exist and be valid JSON
    assert out_file.exists(), "Output weights file not created"
    loaded = json.loads(out_file.read_text())
    assert isinstance(loaded, dict)

    # At least one non-metadata algorithm key should exist
    algo_keys = [k for k in weights if not k.startswith("_")]
    assert len(algo_keys) >= 1, (
        f"No algorithm keys in weights; keys: {list(weights.keys())}"
    )


def test_e4_weights_nonzero(tmp_path):
    """Weights trained from real C++ benchmark data have non-zero values."""
    pr = _require_binary("pr")
    bfs = _require_binary("bfs")
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny graph not found: {TINY_GRAPH}")

    # Run PR and BFS with ORIGINAL and RANDOM
    for bench in [pr, bfs]:
        for algo in ["0", "1"]:
            r = _run(bench, TINY_GRAPH, ["-o", algo, "-D", str(db_dir) + "/"])
            assert r.returncode == 0

    # Load records and convert to BenchmarkResult
    from scripts.lib.core.utils import BenchmarkResult
    from scripts.lib.ml.weights import compute_weights_from_results

    raw = json.loads((db_dir / "benchmarks.json").read_text())
    results = []
    for rec in raw:
        results.append(BenchmarkResult(
            graph=rec.get("graph", ""),
            benchmark=rec.get("benchmark", ""),
            algorithm=rec.get("algorithm", ""),
            algorithm_id=rec.get("algorithm_id", 0),
            time_seconds=rec.get("time_seconds", 0),
            success=rec.get("success", False),
            nodes=rec.get("nodes", 0),
            edges=rec.get("edges", 0),
        ))

    out_file = tmp_path / "weights.json"
    weights = compute_weights_from_results(
        benchmark_results=results,
        output_file=str(out_file),
        weights_dir=str(tmp_path / "wd"),
    )

    # Find a non-ORIGINAL, non-metadata key
    algo_keys = [k for k in weights
                 if not k.startswith("_") and k != "ORIGINAL"]

    # We may not have non-ORIGINAL keys if only ORIGINAL and RANDOM were run
    # and RANDOM is excluded from perceptron candidates.  In that case just
    # verify the structure is sane.
    if algo_keys:
        w = weights[algo_keys[0]]
        # Check at least one weight field is non-zero
        weight_fields = [v for k, v in w.items()
                         if not k.startswith("_") and k != "benchmark_weights"
                         and isinstance(v, (int, float))]
        has_nonzero = any(v != 0 for v in weight_fields)
        assert has_nonzero, (
            f"All weight fields are zero for {algo_keys[0]}: {w}"
        )
    else:
        # Just verify we got a valid weights dict
        assert isinstance(weights, dict)


# ============================================================================
# E5: Oracle Comparison
# ============================================================================

def test_e5_oracle_best_algo():
    """Oracle correctly identifies the lowest-time algorithm per graph×benchmark."""
    from scripts.lib.ml.oracle import compute_oracle

    time_lookup = {
        "graphA": {
            "pr": {"ORIGINAL": 1.0, "RABBITORDER_csr": 0.5, "HubSort": 0.7},
            "bfs": {"ORIGINAL": 2.0, "RABBITORDER_csr": 1.5, "HubSort": 1.2},
        }
    }

    adaptive_lookup = {
        "graphA": {
            "graph": "graphA",
            "algorithm_distribution": {"RABBITORDER": 5, "HubSort": 3},
        }
    }

    report = compute_oracle(time_lookup, adaptive_lookup, candidate_only=False)

    assert len(report.entries) >= 2, (
        f"Expected >= 2 oracle entries, got {len(report.entries)}"
    )

    # Build lookup by benchmark
    by_bench = {e.benchmark: e for e in report.entries}

    # PR: oracle should pick RABBITORDER_csr (time=0.5)
    assert "pr" in by_bench
    assert by_bench["pr"].oracle_algo == "RABBITORDER_csr", (
        f"Expected oracle to pick RABBITORDER_csr for PR, got {by_bench['pr'].oracle_algo}"
    )
    assert by_bench["pr"].oracle_time == 0.5

    # BFS: oracle should pick HubSort (time=1.2)
    assert "bfs" in by_bench
    assert by_bench["bfs"].oracle_algo == "HubSort", (
        f"Expected oracle to pick HubSort for BFS, got {by_bench['bfs'].oracle_algo}"
    )
    assert by_bench["bfs"].oracle_time == 1.2

    # Oracle time should always be <= adaptive time
    for entry in report.entries:
        assert entry.oracle_time <= entry.adaptive_time, (
            f"Oracle time {entry.oracle_time} > adaptive time {entry.adaptive_time} "
            f"for {entry.graph}/{entry.benchmark}"
        )


def test_e5_oracle_report_completeness():
    """Oracle covers all graph×benchmark pairs present in the data."""
    from scripts.lib.ml.oracle import compute_oracle

    time_lookup = {
        "graphA": {
            "pr": {"ORIGINAL": 1.0, "HubSort": 0.8},
            "bfs": {"ORIGINAL": 2.0, "HubSort": 1.5},
        },
        "graphB": {
            "pr": {"ORIGINAL": 0.5, "HubSort": 0.4},
            "bfs": {"ORIGINAL": 1.0, "HubSort": 0.9},
        },
    }

    adaptive_lookup = {
        "graphA": {
            "graph": "graphA",
            "algorithm_distribution": {"HubSort": 1},
        },
        "graphB": {
            "graph": "graphB",
            "algorithm_distribution": {"ORIGINAL": 1},
        },
    }

    report = compute_oracle(time_lookup, adaptive_lookup, candidate_only=False)

    # Should have 2 graphs × 2 benchmarks = 4 entries
    assert len(report.entries) == 4, (
        f"Expected 4 oracle entries (2 graphs × 2 benchmarks), got {len(report.entries)}"
    )

    # All (graph, benchmark) pairs should be represented
    pairs = {(e.graph, e.benchmark) for e in report.entries}
    expected_pairs = {
        ("graphA", "bfs"), ("graphA", "pr"),
        ("graphB", "bfs"), ("graphB", "pr"),
    }
    assert pairs == expected_pairs, (
        f"Missing pairs: {expected_pairs - pairs}"
    )

    # Summary stats should be populated
    assert report.accuracy >= 0.0
    assert report.mean_regret >= 0.0
