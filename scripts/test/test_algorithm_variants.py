#!/usr/bin/env python3
"""
Algorithm Variant Integration Tests
====================================

Comprehensive parametrized tests covering ALL algorithm variant combinations.
Each test runs the binary with a given -o option and verifies it exits cleanly
(exit code 0, no crash, no assertion failure).

Tiers:
  1. Basic algorithms (0-11, 15) — no variants
  2. RabbitOrder (8) variants — csr, boost
  3. GraphBrewOrder (12) presets — leiden, rabbit, hubcluster
  4. GraphBrewOrder (12) preset + positional overrides
  5. GraphBrewOrder (12) token mode — ordering strategies
  6. GraphBrewOrder (12) token combinations — multi-token
  7. GraphBrewOrder (12) resolution & feature flags
  8. LeidenOrder (15) resolution variants
  9. Edge cases — legacy aliases, old format, boundary values

Usage:
    pytest scripts/test/test_algorithm_variants.py -v
    pytest scripts/test/test_algorithm_variants.py -k "tier1" -v
    pytest scripts/test/test_algorithm_variants.py -k "graphbrew" -v
    pytest scripts/test/test_algorithm_variants.py --timeout=120 -v

Requires:
    - bench/bin/pr binary (run `make pr` first)
    - A test graph (uses bundled tiny.el, or soc-Epinions1.sg if available)
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts.lib.ml.portfolio import (
    DEPLOYABLE_ARM_CANONICAL_NAMES,
    DEPLOYABLE_ARM_SPECS,
)
from scripts.lib.ml.feature_schema import (
    TIER0_FEATURE_NAMES,
    TIER0_WEIGHT_NAMES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
PR_BINARY = BIN_DIR / "pr"
BFS_BINARY = BIN_DIR / "bfs"

# Test graphs: prefer a real graph for community algos, fall back to tiny
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "data" / "tiny.el"
REAL_GRAPH = PROJECT_ROOT / "results" / "graphs" / "soc-Epinions1" / "soc-Epinions1.sg"

# Timeout for each binary invocation (seconds)
BINARY_TIMEOUT = 120


def adaptive_weight_entry(bias=0.0, **overrides):
    entry = {
        "bias": float(bias),
        **{f"w_t0_{name}": 0.0 for name in TIER0_FEATURE_NAMES},
    }
    entry.update(overrides)
    return entry


def adaptive_env_weight_payload(weights):
    return {
        "_schema": "adaptive-tier0/v1",
        "_note": "0",
        "weights": weights,
    }

# Algorithms that need a real graph (community detection is degenerate on 4 nodes)
NEEDS_REAL_GRAPH = {8, 9, 10, 11, 12, 14, 15}

# Algorithms we skip entirely in CI (need special setup or are too slow)
SKIP_ALGOS = {
    13,  # MAP — needs a mapping file
}


def get_graph_path(algo_id: int) -> str:
    """Return best available graph path for the given algorithm."""
    if algo_id in NEEDS_REAL_GRAPH and REAL_GRAPH.exists():
        return str(REAL_GRAPH)
    return str(TINY_GRAPH)


def run_pr(option: str, graph_path: str = None, timeout: int = BINARY_TIMEOUT) -> subprocess.CompletedProcess:
    """Run the pr binary with -o option and return the result."""
    if graph_path is None:
        # Extract algo ID from option to choose graph
        algo_id = int(option.split(":")[0])
        graph_path = get_graph_path(algo_id)

    cmd = [str(PR_BINARY), "-f", graph_path, "-s", "-o", option, "-n", "1"]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


@pytest.fixture(scope="session", autouse=True)
def check_prerequisites():
    """Verify pr binary exists before running any tests."""
    if not PR_BINARY.exists():
        pytest.skip(f"pr binary not found at {PR_BINARY}. Run 'make pr' first.")
    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny test graph not found at {TINY_GRAPH}.")


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Basic algorithms (0-11, 15) — no variants
# ═══════════════════════════════════════════════════════════════════════════

TIER1_BASIC = [
    ("0",  "ORIGINAL"),
    ("1",  "RANDOM"),
    ("2",  "SORT"),
    ("3",  "HUBSORT"),
    ("4",  "HUBCLUSTER"),
    ("5",  "DBG"),
    ("6",  "HUBSORTDBG"),
    ("7",  "HUBCLUSTERDBG"),
    ("8",  "RABBITORDER_default"),
    ("9",  "GORDER"),
    ("10", "CORDER"),
    ("11", "RCMORDER"),
    ("12", "GraphBrewOrder_default"),
    ("15", "LeidenOrder_default"),
]


@pytest.mark.parametrize("option,name", TIER1_BASIC, ids=[t[1] for t in TIER1_BASIC])
def test_tier1_basic(option, name):
    """Tier 1: Each basic algorithm runs and exits cleanly."""
    algo_id = int(option.split(":")[0])
    if algo_id in SKIP_ALGOS:
        pytest.skip(f"Algorithm {algo_id} skipped (needs special setup)")
    result = run_pr(option)
    assert result.returncode == 0, (
        f"Algorithm {name} (-o {option}) failed with exit code {result.returncode}.\n"
        f"stderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: RabbitOrder (8) variants
# ═══════════════════════════════════════════════════════════════════════════

TIER2_RABBIT = [
    ("8:csr",   "RABBITORDER_csr"),
    ("8:boost", "RABBITORDER_boost"),
]


@pytest.mark.parametrize("option,name", TIER2_RABBIT, ids=[t[1] for t in TIER2_RABBIT])
def test_tier2_rabbitorder_variants(option, name):
    """Tier 2: RabbitOrder csr/boost variants."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"RabbitOrder variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


@pytest.mark.parametrize("arm", DEPLOYABLE_ARM_SPECS)
def test_adaptive_exact_deployable_arms(tmp_path, arm):
    """Algorithm 14 must apply the exact portfolio arm, not a family proxy."""
    weights = tmp_path / "weights.json"
    payload = {
        spec: adaptive_weight_entry()
        for spec in DEPLOYABLE_ARM_SPECS
    }
    payload[arm] = adaptive_weight_entry(bias=1000.0)
    weights.write_text(json.dumps(adaptive_env_weight_payload(payload)))
    env = {
        **os.environ,
        "PERCEPTRON_WEIGHTS_FILE": str(weights),
        "GRAPHBREW_DB_DIR": "",
        "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert re.search(
        rf"Adaptive Predicted:\s+{re.escape(arm)}",
        result.stdout,
    )
    assert re.search(
        rf"Adaptive Applied:\s+{re.escape(arm)}",
        result.stdout,
    )
    assert f"=== Selected Algorithm: {arm} ===" in result.stdout
    if arm.startswith("12:"):
        assert "RabbitOrder: modularity resolution=n/a" in result.stdout


def test_adaptive_runtime_loads_only_model_artifact(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        spec: adaptive_weight_entry(
            bias=1000.0 if spec == "5" else 0.0
        )
        for spec in DEPLOYABLE_ARM_SPECS
    }
    (data_dir / "adaptive_models.json").write_text(json.dumps({
        "perceptron": {
            "schema": "adaptive-tier0/v1",
            "tier0_trained": False,
            "weights": payload,
            "per_benchmark": {},
        },
    }))
    (data_dir / "benchmarks.json").write_text(json.dumps([{
        "graph": "known",
        "algorithm": "DBG",
        "benchmark": "pr",
        "time_seconds": 1.0,
        "success": True,
    }]))
    (data_dir / "graph_properties.json").write_text(json.dumps({
        "known": {"nodes": 1024, "edges": 4096},
    }))
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "GRAPHBREW_DB_DIR": str(data_dir) + "/",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert "[MODEL] Loaded adaptive_models.json" in result.stdout
    assert "[DATABASE] Loaded" not in result.stdout
    assert "Adaptive Weight Source: averaged" in result.stdout
    assert "Adaptive Tier0 Trained: false" in result.stdout
    assert re.search(r"Adaptive Applied:\s+5", result.stdout)


def test_adaptive_missing_portfolio_fails_closed(tmp_path):
    data_dir = tmp_path / "empty-data"
    data_dir.mkdir()
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "GRAPHBREW_DB_DIR": str(data_dir) + "/",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "Deployable adaptive model artifact is unavailable" in combined


@pytest.mark.parametrize("omitted_arm", DEPLOYABLE_ARM_SPECS)
def test_adaptive_partial_portfolio_fails_closed(
    tmp_path, omitted_arm,
):
    weights = tmp_path / "weights.json"
    payload = {
        spec: adaptive_weight_entry()
        for spec in DEPLOYABLE_ARM_SPECS
        if spec != omitted_arm
    }
    weights.write_text(json.dumps(adaptive_env_weight_payload(payload)))
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PERCEPTRON_WEIGHTS_FILE": str(weights),
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode != 0
    assert omitted_arm in result.stdout + result.stderr


def test_adaptive_legacy_feature_artifact_fails_closed(tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({
        "_schema": "adaptive-tier0/v1",
        "weights": {
            spec: {
                "bias": 0.0,
                "w_modularity": 1.0,
                "w_log_nodes": 1.0,
            }
            for spec in DEPLOYABLE_ARM_SPECS
        },
    }))
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PERCEPTRON_WEIGHTS_FILE": str(weights),
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode != 0
    assert "missing Tier-0 weight" in result.stdout + result.stderr


def test_adaptive_schema_value_cannot_be_spoofed_by_note(tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({
        "_schema": "adaptive-legacy/v0",
        "_note": "adaptive-tier0/v1",
        "weights": {
            spec: adaptive_weight_entry()
            for spec in DEPLOYABLE_ARM_SPECS
        },
    }))
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PERCEPTRON_WEIGHTS_FILE": str(weights),
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode != 0
    assert "not adaptive-tier0/v1" in result.stdout + result.stderr


def test_adaptive_conflicting_aliases_fail_closed(tmp_path):
    weights = tmp_path / "weights.json"
    payload = {
        spec: adaptive_weight_entry()
        for spec in DEPLOYABLE_ARM_SPECS
    }
    payload[DEPLOYABLE_ARM_CANONICAL_NAMES[0]] = (
        adaptive_weight_entry(bias=1.0)
    )
    weights.write_text(json.dumps(adaptive_env_weight_payload(payload)))
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PERCEPTRON_WEIGHTS_FILE": str(weights),
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode != 0
    assert "conflicting aliases" in result.stdout + result.stderr


def test_adaptive_no_margin_ablation_is_offline_only(tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps(adaptive_env_weight_payload({
        spec: adaptive_weight_entry()
        for spec in DEPLOYABLE_ARM_SPECS
    })))
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PERCEPTRON_WEIGHTS_FILE": str(weights),
            "ADAPTIVE_NO_MARGIN": "1",
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode != 0
    assert "offline-only" in result.stdout + result.stderr


def test_trainer_export_runtime_contract_roundtrip(
    tmp_path, monkeypatch,
):
    from scripts.lib.core import datastore
    from scripts.lib.core.utils import BenchmarkResult
    from scripts.lib.ml import features as feature_module
    from scripts.lib.ml.weights import compute_weights_from_results

    graph_properties = {
        "synthetic": {
            "nodes": 1024,
            "edges": 4096,
            "avg_degree": 4.0,
            "degree_variance": 1.0,
            "hub_concentration": 0.25,
            "clustering_coefficient": 0.1,
            "normalized_edge_span": 0.2,
            "window_neighbor_overlap": 0.3,
        },
    }
    monkeypatch.setattr(
        feature_module,
        "load_graph_properties_cache",
        lambda *_args, **_kwargs: graph_properties,
    )
    results = [
        BenchmarkResult(
            graph="synthetic",
            algorithm="DBG",
            algorithm_id=5,
            benchmark="pr",
            time_seconds=1.0,
        ),
        BenchmarkResult(
            graph="synthetic",
            algorithm="RABBITORDER_csr",
            algorithm_id=8,
            benchmark="pr",
            time_seconds=1.1,
        ),
    ]
    weights_dir = tmp_path / "models" / "perceptron"
    compute_weights_from_results(
        results,
        weights_dir=str(weights_dir),
    )
    model_path = tmp_path / "data" / "adaptive_models.json"
    datastore.export_unified_models(
        model_path,
        weights_dir=weights_dir,
    )
    model = json.loads(model_path.read_text())
    assert set(model["perceptron"]["weights"]) == set(
        DEPLOYABLE_ARM_SPECS
    )
    exported_sets = [model["perceptron"]["weights"]]
    exported_sets.extend(
        model["perceptron"]["per_benchmark"].values()
    )
    for exported in exported_sets:
        for entry in exported.values():
            assert all(
                entry[name] == 0.0
                for name in TIER0_WEIGHT_NAMES[1:]
            )

    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "10",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "GRAPHBREW_DB_DIR": str(model_path.parent) + "/",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert "[MODEL] Loaded adaptive_models.json" in result.stdout
    assert "Adaptive Weight Source: per-benchmark:pr" in result.stdout
    assert "Adaptive Tier0 Trained: false" in result.stdout


def test_adaptive_tier0_features_change_selected_arm(tmp_path):
    weights = tmp_path / "weights.json"
    payload = {
        spec: adaptive_weight_entry(bias=-100.0)
        for spec in DEPLOYABLE_ARM_SPECS
    }
    payload["0"] = adaptive_weight_entry(bias=4.0)
    payload["5"] = adaptive_weight_entry(
        w_t0_log10_nodes=1.0,
    )
    weights.write_text(json.dumps(adaptive_env_weight_payload(payload)))

    def run(scale):
        return subprocess.run(
            [
                str(PR_BINARY),
                "-g", str(scale),
                "-o", "14",
                "-n", "1",
                "-i", "2",
            ],
            capture_output=True,
            text=True,
            timeout=BINARY_TIMEOUT,
            cwd=str(PROJECT_ROOT),
            env={
                **os.environ,
                "PERCEPTRON_WEIGHTS_FILE": str(weights),
                "GRAPHBREW_DB_DIR": "",
                "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
                "OMP_NUM_THREADS": "1",
            },
        )

    small = run(10)
    large = run(20)
    assert small.returncode == 0, small.stderr[-1000:]
    assert large.returncode == 0, large.stderr[-1000:]
    assert re.search(r"Adaptive Applied:\s+0", small.stdout)
    assert re.search(r"Adaptive Applied:\s+5", large.stdout)


def test_adaptive_tier0_json_ignores_prior_stream_precision(tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({
        "_schema": "adaptive-tier0/v1",
        "weights": {
            spec: adaptive_weight_entry(
                bias=1000.0 if spec == "5" else 0.0
            )
            for spec in DEPLOYABLE_ARM_SPECS
        },
    }))
    result = subprocess.run(
        [
            str(PR_BINARY),
            "-g", "12",
            "-o", "8:csr",
            "-o", "14",
            "-n", "1",
            "-i", "2",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PERCEPTRON_WEIGHTS_FILE": str(weights),
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stderr[-1000:]
    match = re.search(
        r"Adaptive Tier0 Features:\s*(\{.*\})",
        result.stdout,
    )
    assert match is not None
    features = json.loads(match.group(1))
    assert features["property_wsr_llc"] > 0
    assert features["log10_nodes"] > 3


@pytest.mark.parametrize("arm", [
    "0",
    "5",
    "8:csr",
    "12:rabbit:compose:sg_none:comm_identity:intra_hubsort",
    "12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort",
])
def test_deployable_arm_source_roundtrip(arm):
    """Every deployable arm preserves and resolves explicit original IDs."""
    if not BFS_BINARY.exists():
        pytest.skip("bfs binary is required for source round-trip tests")
    result = subprocess.run(
        [
            str(BFS_BINARY),
            "-f", str(TINY_GRAPH),
            "-s",
            "-o", arm,
            "-r", "0,1",
            "-v",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, (
        f"Source round-trip failed for {arm}:\n{result.stderr[-1000:]}"
    )
    originals = [
        int(value)
        for value in re.findall(
            r"Source Original:\s+(\d+)",
            result.stdout,
        )
    ]
    assert originals == [0, 1]
    assert result.stdout.count("Source Out Degree:") == 2
    assert result.stdout.count("Verification:") == 2
    assert result.stdout.count("PASS") >= 2


def test_sssp_single_source_repeat_contract():
    sssp = BIN_DIR / "sssp"
    if not sssp.exists():
        pytest.skip("sssp binary is required")
    result = subprocess.run(
        [
            str(sssp),
            "-f", str(TINY_GRAPH),
            "-s",
            "-W", "hash",
            "-d", "1",
            "-r", "0",
            "-R", "3",
            "-v",
        ],
        capture_output=True,
        text=True,
        timeout=BINARY_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert re.findall(
        r"Source Original:\s+(\d+)",
        result.stdout,
    ) == ["0", "0", "0"]



# ═══════════════════════════════════════════════════════════════════════════
# TIER 2b: RCM (11) variants
# ═══════════════════════════════════════════════════════════════════════════

TIER2B_RCM = [
    ("11",     "RCM_default"),
    ("11:bnf", "RCM_bnf"),
]


@pytest.mark.parametrize("option,name", TIER2B_RCM, ids=[t[1] for t in TIER2B_RCM])
def test_tier2b_rcm_variants(option, name):
    """Tier 2b: RCM default/bnf variants."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"RCM variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2c: GOrder (9) variants
# ═══════════════════════════════════════════════════════════════════════════

TIER2C_GORDER = [
    ("9",         "GORDER_default"),
    ("9:gograph", "GORDER_gograph"),
    ("9:csr",     "GORDER_csr"),
    ("9:fast",    "GORDER_fast"),
]


@pytest.mark.parametrize("option,name", TIER2C_GORDER, ids=[t[1] for t in TIER2C_GORDER])
def test_tier2c_gorder_variants(option, name):
    """Tier 2c: GOrder auto, forced legacy, CSR, and relaxed variants."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GOrder variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: GraphBrewOrder (12) presets
# ═══════════════════════════════════════════════════════════════════════════

TIER3_PRESETS = [
    ("12:leiden",      "GraphBrew_leiden"),
    ("12:rabbit",      "GraphBrew_rabbit"),
    ("12:hubcluster",  "GraphBrew_hubcluster"),
]


@pytest.mark.parametrize("option,name", TIER3_PRESETS, ids=[t[1] for t in TIER3_PRESETS])
def test_tier3_graphbrew_presets(option, name):
    """Tier 3: GraphBrewOrder named presets."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew preset {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 4: GraphBrewOrder (12) preset + positional overrides
# ═══════════════════════════════════════════════════════════════════════════

TIER4_POSITIONAL = [
    ("12:leiden:0",        "leiden_final_ORIGINAL"),
    ("12:leiden:6",        "leiden_final_HUBSORTDBG"),
    ("12:leiden:7",        "leiden_final_HUBCLUSTERDBG"),
    ("12:leiden:8",        "leiden_final_RABBITORDER"),
    ("12:leiden:8:0.75",   "leiden_res_0.75"),
    ("12:leiden:8:1.5",    "leiden_res_1.5"),
    ("12:leiden:8:auto",   "leiden_res_auto"),
    ("12:leiden:8:dynamic", "leiden_res_dynamic"),
    ("12:rabbit:7",        "rabbit_final_HUBCLUSTERDBG"),
    ("12:rabbit:8:0.5",    "rabbit_res_0.5"),
    ("12:hubcluster:6",    "hubcluster_final_HUBSORTDBG"),
]


@pytest.mark.parametrize("option,name", TIER4_POSITIONAL, ids=[t[1] for t in TIER4_POSITIONAL])
def test_tier4_preset_positional(option, name):
    """Tier 4: Preset + positional override (final_algo, resolution)."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew positional {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 5: GraphBrewOrder (12) token mode — ordering strategies
# ═══════════════════════════════════════════════════════════════════════════

TIER5_TOKENS = [
    ("12:hrab",         "token_hrab"),
    ("12:dfs",          "token_dfs"),
    ("12:bfs",          "token_bfs"),
    ("12:conn",         "token_conn"),
    ("12:dbg",          "token_dbg"),
    ("12:corder",       "token_corder"),
    ("12:dbg-global",   "token_dbg_global"),
    ("12:corder-global","token_corder_global"),
    ("12:community",    "token_community"),
    ("12:hierarchical", "token_hierarchical"),
    ("12:hcache",       "token_hcache"),
    ("12:tqr",          "token_tqr"),
]


@pytest.mark.parametrize("option,name", TIER5_TOKENS, ids=[t[1] for t in TIER5_TOKENS])
def test_tier5_token_ordering(option, name):
    """Tier 5: Token-mode ordering strategies."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew token {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 6: GraphBrewOrder (12) token combinations
# ═══════════════════════════════════════════════════════════════════════════

TIER6_COMBOS = [
    ("12:hrab:gvecsr",            "hrab_gvecsr"),
    ("12:hrab:gvecsr:totalm",     "hrab_gvecsr_totalm"),
    ("12:dfs:streaming",          "dfs_streaming"),
    ("12:dbg:streaming",          "dbg_streaming"),
    ("12:hrab:0.75",              "hrab_res_0.75"),
    ("12:graphbrew",              "graphbrew_mode"),
    ("12:graphbrew:final8",       "graphbrew_final8"),
    ("12:graphbrew:final6",       "graphbrew_final6"),
    ("12:graphbrew:depth2",       "graphbrew_depth2"),
    ("12:graphbrew:subauto",      "graphbrew_subauto"),
]


@pytest.mark.parametrize("option,name", TIER6_COMBOS, ids=[t[1] for t in TIER6_COMBOS])
def test_tier6_token_combos(option, name):
    """Tier 6: Multi-token combinations."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew combo {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 7: GraphBrewOrder (12) resolution & feature flags
# ═══════════════════════════════════════════════════════════════════════════

TIER7_FLAGS = [
    ("12:hrab:auto",              "auto_resolution"),
    ("12:hrab:dynamic",           "dynamic_resolution"),
    ("12:hrab:0.5",               "low_resolution"),
    ("12:hrab:2.0",               "high_resolution"),
    ("12:hrab:merge",             "community_merging"),
    ("12:hrab:verify",            "topology_verify"),
    ("12:hrab:norefine",          "no_refinement"),
    ("12:hrab:refine0",           "refine_pass0_only"),
    ("12:hrab:lazyupdate",        "lazy_updates"),
    ("12:hrab:gord",              "gorder_intra"),
    ("12:hrab:hsort",             "hub_sort_post"),
]


@pytest.mark.parametrize("option,name", TIER7_FLAGS, ids=[t[1] for t in TIER7_FLAGS])
def test_tier7_feature_flags(option, name):
    """Tier 7: Feature flags and resolution modes."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew flag {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 8: LeidenOrder (15) resolution variants
# ═══════════════════════════════════════════════════════════════════════════

TIER8_LEIDEN = [
    ("15:0.5",  "LeidenOrder_res_0.5"),
    ("15:0.75", "LeidenOrder_res_0.75"),
    ("15:1.0",  "LeidenOrder_res_1.0"),
    ("15:1.5",  "LeidenOrder_res_1.5"),
]


@pytest.mark.parametrize("option,name", TIER8_LEIDEN, ids=[t[1] for t in TIER8_LEIDEN])
def test_tier8_leiden_resolution(option, name):
    """Tier 8: LeidenOrder with different resolution values."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"LeidenOrder variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 9: Edge cases — legacy aliases, old format, boundary values
# ═══════════════════════════════════════════════════════════════════════════

TIER9_EDGE = [
    # Token parser quality preset
    ("12:quality",                "token_quality_preset"),
    # graphbrew prefix (backward compat)
    ("12:graphbrew:hrab",         "graphbrew_prefix_hrab"),
    # Multiple feature flags combined
    ("12:hrab:gvecsr:totalm:refine0:0.75", "full_combo"),
]


@pytest.mark.parametrize("option,name", TIER9_EDGE, ids=[t[1] for t in TIER9_EDGE])
def test_tier9_edge_cases(option, name):
    """Tier 9: Edge cases — legacy aliases, old format, combined flags."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"Edge case {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test: verify all tiers have expected count
# ═══════════════════════════════════════════════════════════════════════════

def test_variant_count():
    """Verify we have ~70+ test cases across all tiers."""
    total = (
        len(TIER1_BASIC) + len(TIER2_RABBIT) + len(TIER3_PRESETS) +
        len(TIER4_POSITIONAL) + len(TIER5_TOKENS) + len(TIER6_COMBOS) +
        len(TIER7_FLAGS) + len(TIER8_LEIDEN) + len(TIER9_EDGE)
    )
    assert total >= 70, f"Expected ≥70 test cases, got {total}"
