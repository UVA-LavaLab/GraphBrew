#!/usr/bin/env python3
"""Tests for ECG graph staging status helper."""

import csv
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_graph_staging_status_lists_final_graphs(tmp_path):
    out = tmp_path / "graphs.csv"
    subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_graph_staging_status.py",
            "--profile", "final_replacement", "final_droplet",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    rows = list(csv.DictReader(out.open()))
    names = {row["graph"] for row in rows}

    assert {"soc-pokec", "soc-LiveJournal1", "com-orkut", "cit-Patents"}.issubset(names)
    assert all(row["expected_path"].endswith(f"/{row['graph']}.sg") for row in rows)


def test_graph_staging_status_fail_on_missing(tmp_path):
    out = tmp_path / "graphs.csv"
    result = subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_graph_staging_status.py",
            "--profile", "final_replacement",
            "--graph-dir", str(tmp_path / "missing_graphs"),
            "--out", str(out),
            "--fail-on-missing",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert result.returncode == 2
    assert out.exists()