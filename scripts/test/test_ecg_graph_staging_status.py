#!/usr/bin/env python3
"""Tests for ECG graph staging status helper."""

import csv
import json
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
    by_name = {row["graph"]: row for row in rows}
    assert "scripts.lib.pipeline.download" in by_name["soc-pokec"]["download_command"]
    assert by_name["com-orkut"]["catalog_graph"] == "com-Orkut"
    assert "--graph com-Orkut" in by_name["com-orkut"]["download_command"]
    assert "results/graphs/com-Orkut" in by_name["com-orkut"]["convert_command"]
    assert "results/graphs/com-orkut/com-orkut.sg" in by_name["com-orkut"]["convert_command"]
    assert "bench/bin/converter" in by_name["com-orkut"]["convert_command"]
    assert by_name["com-orkut"]["source_category"] == "social"
    assert by_name["soc-LiveJournal1"]["source_category"] == "social"


def test_graph_staging_status_fail_on_missing(tmp_path):
    out = tmp_path / "graphs.csv"
    # Build a manifest whose final_replacement graphs point at non-existent
    # files inside tmp_path. This makes the test robust to repo checkouts
    # that already have results/graphs/*.sg locally staged (which would
    # otherwise make every graph appear "ok" and short-circuit
    # --fail-on-missing).
    real_manifest = json.loads(
        (PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "final_paper_manifest.json").read_text()
    )
    for graph_set in real_manifest.get("graph_sets", {}).values():
        for graph in graph_set:
            if "path" in graph:
                graph["path"] = str(tmp_path / "missing_graphs" / f"{graph['name']}.sg")
    fake_manifest = tmp_path / "fake_manifest.json"
    fake_manifest.write_text(json.dumps(real_manifest))

    result = subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_graph_staging_status.py",
            "--profile", "final_replacement",
            "--graph-dir", str(tmp_path / "missing_graphs"),
            "--manifest", str(fake_manifest),
            "--out", str(out),
            "--fail-on-missing",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert result.returncode == 2
    assert out.exists()