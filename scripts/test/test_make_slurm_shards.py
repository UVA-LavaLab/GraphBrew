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


def smoke_manifest() -> dict:
    return {
        "defaults": {
            "graph_set": "graphs",
        },
        "graph_sets": {
            "graphs": [
                {"name": "cit-Patents", "path": "missing/cit-Patents.sg", "options_key": "file_dbg"},
                {"name": "soc-pokec", "path": "missing/soc-pokec.sg", "options_key": "file_dbg"},
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


def test_smoke_defaults_select_canonical_one_row(tmp_path):
    class Args:
        smoke = True
        profile = []
        graph = []
        benchmark = []
        policy = []

    args = Args()
    make_slurm_shards.apply_smoke_defaults(args)
    rows = make_slurm_shards.generate_shards(
        manifest=smoke_manifest(),
        profiles=args.profile,
        run_tag="smoke1",
        graph_dir=tmp_path,
        stage_filters=[],
        graph_filters=args.graph,
        benchmark_filters=args.benchmark,
        policy_filters=args.policy,
        allow_missing_graphs=True,
    )

    assert [row.to_tsv() for row in rows] == [
        "final_replacement\t20_gem5_large_replacement\tcit-Patents\tpr\tLRU\tsmoke1"
    ]


def test_smoke_defaults_preserve_explicit_overrides():
    class Args:
        smoke = True
        profile = ["final_replacement"]
        graph = ["soc-pokec"]
        benchmark = ["bfs"]
        policy = ["ECG_DBG_PRIMARY"]

    args = Args()
    make_slurm_shards.apply_smoke_defaults(args)

    assert args.profile == ["final_replacement"]
    assert args.graph == ["soc-pokec"]
    assert args.benchmark == ["bfs"]
    assert args.policy == ["ECG_DBG_PRIMARY"]