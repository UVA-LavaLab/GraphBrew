#!/usr/bin/env python3
"""Regression tests for ECG Slurm shard generation."""

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = PROJECT_ROOT / "scripts" / "experiments" / "ecg"
sys.path.insert(0, str(ECG_DIR))
SHARDS_PATH = ECG_DIR / "make_slurm_shards.py"
spec = importlib.util.spec_from_file_location("make_slurm_shards", SHARDS_PATH)
make_slurm_shards = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["make_slurm_shards"] = make_slurm_shards
spec.loader.exec_module(make_slurm_shards)


def tiny_manifest() -> dict:
    return {
        "defaults": {
            "graph_set": "graphs",
        },
        "graph_sets": {
            "graphs": [
                {"name": "gA", "path": "missing/gA.sg", "options_key": "file_dbg"},
                {"name": "gB", "path": "missing/gB.sg", "options_key": "file_dbg"},
            ],
        },
        "stages": [
            {
                "name": "20_gem5_large_replacement",
                "kind": "roi_matrix",
                "profiles": ["final_replacement"],
                "suite": "gem5",
                "graph_set": "graphs",
                "benchmarks": ["pr", "bfs"],
                "policies": ["LRU", "ECG:DBG_PRIMARY"],
            },
            {
                "name": "ignored_proof",
                "kind": "proof_matrix",
                "profiles": ["final_replacement"],
            },
        ],
    }


def test_generate_shards_expands_manifest_rows_with_filters(tmp_path):
    rows = make_slurm_shards.generate_shards(
        manifest=tiny_manifest(),
        profiles=["final_replacement"],
        run_tag="run1",
        graph_dir=tmp_path,
        stage_filters=[],
        graph_filters=["gA"],
        benchmark_filters=["pr"],
        policy_filters=["ECG_DBG_PRIMARY"],
        allow_missing_graphs=True,
    )

    assert [row.to_tsv() for row in rows] == [
        "final_replacement\t20_gem5_large_replacement\tgA\tpr\tECG:DBG_PRIMARY\trun1"
    ]


def test_write_shards_uses_headerless_tsv(tmp_path):
    rows = [make_slurm_shards.ShardRow("profile", "stage", "graph", "pr", "LRU", "tag")]
    out_path = tmp_path / "shards.tsv"

    make_slurm_shards.write_shards(out_path, rows)

    assert out_path.read_text() == "profile\tstage\tgraph\tpr\tLRU\ttag\n"