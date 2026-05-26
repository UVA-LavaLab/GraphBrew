#!/usr/bin/env python3
"""Tests for the ECG cluster preflight helper."""

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cluster_preflight_accepts_scale_shards(tmp_path):
    shards = tmp_path / "scale.tsv"
    shards.write_text("11\t0\tsniper\t/tmp/out\n")

    result = subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_cluster_preflight.py",
            "--skip-binaries",
            "--scale-shards", str(shards),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "scale-shards" in result.stdout
    assert "1 rows" in result.stdout


def test_cluster_preflight_allows_missing_graphs(tmp_path):
    shards = tmp_path / "final.tsv"
    shards.write_text("final_replacement\t20_gem5_large_replacement\tsoc-pokec\tpr\tLRU\trun\n")

    result = subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_cluster_preflight.py",
            "--skip-binaries",
            "--allow-missing-graphs",
            "--shards", str(shards),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "graph:soc-pokec" in result.stdout
    assert "missing allowed" in result.stdout


def test_cluster_preflight_profile_staging_check():
    result = subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_cluster_preflight.py",
            "--skip-binaries",
            "--allow-missing-graphs",
            "--profile", "final_replacement",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "staging:soc-pokec" in result.stdout
    assert "staging:cit-Patents" in result.stdout