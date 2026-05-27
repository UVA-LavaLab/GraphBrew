#!/usr/bin/env python3
"""Tests for local cache_sim diversity summary helper."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = PROJECT_ROOT / "scripts" / "experiments" / "ecg"
sys.path.insert(0, str(ECG_DIR))
SUMMARY_PATH = ECG_DIR / "local_cache_screen_summary.py"
spec = importlib.util.spec_from_file_location("local_cache_screen_summary", SUMMARY_PATH)
local_cache_screen_summary = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["local_cache_screen_summary"] = local_cache_screen_summary
spec.loader.exec_module(local_cache_screen_summary)


def test_summarize_rows_ranks_and_computes_rates():
    rows = [
        {"benchmark": "pr", "prefetcher": "none", "policy_label": "LRU", "status": "ok", "l3_misses": "100", "timing_valid_for_speedup": "1"},
        {"benchmark": "pr", "prefetcher": "none", "policy_label": "POPT", "status": "ok", "l3_misses": "75", "timing_valid_for_speedup": "1"},
        {"benchmark": "pr", "prefetcher": "ECG_PFX", "policy_label": "LRU", "status": "ok", "l3_misses": "90", "prefetch_requests": "10", "prefetch_useful": "4", "timing_valid_for_speedup": "1"},
    ]

    summary = local_cache_screen_summary.summarize_rows("unit", rows)
    by_key = {(row["benchmark"], row["prefetcher"], row["policy_label"]): row for row in summary}

    assert by_key[("pr", "none", "POPT")]["l3_rank"] == 1
    assert by_key[("pr", "none", "POPT")]["l3_delta_vs_lru"] == "0.25"
    assert by_key[("pr", "ECG_PFX", "LRU")]["prefetch_useful_per_request"] == "0.4"


def test_summary_cli_writes_csv(tmp_path):
    combined = tmp_path / "combined_roi_matrix.csv"
    combined_2 = tmp_path / "combined_roi_matrix_2.csv"
    with combined.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["benchmark", "prefetcher", "policy_label", "status", "l3_misses", "timing_valid_for_speedup"])
        writer.writeheader()
        writer.writerow({"benchmark": "cc", "prefetcher": "none", "policy_label": "LRU", "status": "ok", "l3_misses": "63", "timing_valid_for_speedup": "1"})
    with combined_2.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["benchmark", "prefetcher", "policy_label", "status", "l3_misses", "timing_valid_for_speedup"])
        writer.writeheader()
        writer.writerow({"benchmark": "bfs", "prefetcher": "none", "policy_label": "LRU", "status": "ok", "l3_misses": "10", "timing_valid_for_speedup": "1"})
    out = tmp_path / "summary.csv"

    subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/local_cache_screen_summary.py",
            "--input", f"unit={combined}",
            "--input", f"unit2={combined_2}",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    rows = list(csv.DictReader(out.open()))
    assert [(row["source"], row["benchmark"]) for row in rows] == [("unit", "cc"), ("unit2", "bfs")]
