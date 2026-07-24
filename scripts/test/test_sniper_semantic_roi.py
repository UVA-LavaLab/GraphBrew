from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.ecg import roi_matrix  # noqa: E402
from scripts.experiments.ecg.flows import paper_pipeline  # noqa: E402


def test_sg_kernel_counts_static_edge_visits_in_all_kernels():
    source = (ROOT / "bench/src_sniper/sg_kernel.cc").read_text()
    assert "class SemanticEdgeBudget" in source
    assert 'std::getenv("SNIPER_SEMANTIC_EDGE_LIMIT")' in source
    for benchmark in ("pr", "bfs", "sssp", "bc", "cc"):
        assert f'semantic_edges.report("{benchmark}")' in source
    assert source.count("SemanticEdgeBudget semantic_edges;") == 5
    assert source.count("catch (const SemanticEdgeLimitReached&)") == 5
    assert source.count("consume_edge();") == 19
    assert source.count("execute_roi([] {}, [] {});") == 5
    assert source.count("semantic_edges.finish_roi();") == 5


def test_runner_exposes_semantic_edge_limit_and_marker_gate():
    runner = (ROOT / "scripts/experiments/ecg/roi_matrix.py").read_text()
    paper_run = (
        ROOT / "scripts/experiments/ecg/flows/paper_run.py").read_text()
    assert "--sniper-semantic-edge-limit" in runner
    assert 'env["SNIPER_SEMANTIC_EDGE_LIMIT"]' in runner
    assert "Sniper semantic edge-limit marker missing" in runner
    assert "semantic_work_matched" in runner
    assert "--sniper-semantic-edge-limit" in paper_run


def test_instruction_and_semantic_caps_are_mutually_exclusive():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        roi_matrix.main([
            "--suite", "sniper",
            "--dry-run",
            "--sniper-roi-icount", "100",
            "--sniper-semantic-edge-limit", "100",
        ])


def test_semantic_cap_requires_single_core_sg_kernel():
    with pytest.raises(SystemExit, match="requires --sniper-workload sg_kernel"):
        roi_matrix.main([
            "--suite", "sniper",
            "--dry-run",
            "--sniper-workload", "pr_kernel_smoke",
            "--sniper-semantic-edge-limit", "100",
        ])
    with pytest.raises(SystemExit, match="requires --ecg-isa-variant mask"):
        roi_matrix.main([
            "--suite", "sniper",
            "--dry-run",
            "--sniper-workload", "sg_kernel",
            "--ecg-isa-variant", "indexed",
            "--sniper-semantic-edge-limit", "100",
        ])
    with pytest.raises(SystemExit, match="requires --sniper-cores 1"):
        roi_matrix.main([
            "--suite", "sniper",
            "--dry-run",
            "--sniper-workload", "sg_kernel",
            "--ecg-isa-variant", "mask",
            "--sniper-cores", "2",
            "--sniper-semantic-edge-limit", "100",
        ])
    with pytest.raises(SystemExit, match="every --threads value"):
        roi_matrix.main([
            "--suite", "sniper",
            "--dry-run",
            "--sniper-workload", "sg_kernel",
            "--ecg-isa-variant", "mask",
            "--threads", "1", "2",
            "--sniper-semantic-edge-limit", "100",
        ])


def test_semantic_work_is_certified_only_after_cross_policy_match():
    args = SimpleNamespace(
        suite="sniper",
        sniper_semantic_edge_limit=100,
    )
    policies = [
        SimpleNamespace(label="LRU"),
        SimpleNamespace(label="ECG_K2"),
    ]
    rows = [
        {
            "simulator": "sniper",
            "benchmark": "pr",
            "options": "-g 12",
            "l3_size": "128kB",
            "l3_ways": 16,
            "threads": 1,
            "sniper_cores": 1,
            "policy": policy,
            "policy_label": policy,
            "status": "ok",
            "sniper_semantic_edge_limit": 100,
            "sniper_semantic_edge_visits": 100,
            "sniper_semantic_truncated": 1,
            "sniper_semantic_result": "same",
        }
        for policy in ("LRU", "ECG_K2")
    ]
    roi_matrix.certify_sniper_semantic_work(rows, args, policies)
    assert all(row["semantic_work_matched"] == 1 for row in rows)

    rows[1]["sniper_semantic_edge_visits"] = 99
    rows[1]["status"] = "ok"
    rows[0]["status"] = "ok"
    roi_matrix.certify_sniper_semantic_work(rows, args, policies)
    assert all(row["semantic_work_matched"] == 0 for row in rows)
    assert all(row["status"] == "error" for row in rows)


def test_single_policy_shard_waits_for_aggregate_certification(monkeypatch):
    args = SimpleNamespace(
        suite="sniper",
        sniper_semantic_edge_limit=100,
    )
    policies = [SimpleNamespace(label="LRU")]
    rows = [{
        "simulator": "sniper",
        "benchmark": "pr",
        "options": "-g 12",
        "l3_size": "128kB",
        "l3_ways": 16,
        "threads": 1,
        "sniper_cores": 1,
        "policy": "LRU",
        "policy_label": "LRU",
        "status": "ok",
        "sniper_semantic_edge_limit": 100,
        "sniper_semantic_edge_visits": 100,
        "sniper_semantic_truncated": 1,
        "sniper_semantic_result": "same",
        "semantic_work_matched": 0,
    }]
    monkeypatch.setenv(
        "GRAPHBREW_EXPECTED_POLICY_LABELS", '["LRU","ECG_K2"]')
    roi_matrix.certify_sniper_semantic_work(rows, args, policies)
    assert rows[0]["semantic_work_matched"] == 0

    merged = [
        dict(rows[0]),
        {
            **rows[0],
            "policy": "ECG",
            "policy_label": "ECG_K2",
        },
    ]
    assert paper_pipeline.semantic_work_group_matches(merged)
    assert all(row["semantic_work_matched"] == "1" for row in merged)
