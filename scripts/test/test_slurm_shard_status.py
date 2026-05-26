#!/usr/bin/env python3
"""Regression tests for ECG Slurm shard status summaries."""

import csv
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = PROJECT_ROOT / "scripts" / "experiments" / "ecg"
sys.path.insert(0, str(ECG_DIR))
STATUS_PATH = ECG_DIR / "slurm_shard_status.py"
spec = importlib.util.spec_from_file_location("slurm_shard_status", STATUS_PATH)
slurm_shard_status = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["slurm_shard_status"] = slurm_shard_status
spec.loader.exec_module(slurm_shard_status)


def write_jobs(path: Path, status: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["job_id", "stage", "kind", "status", "detail"])
        writer.writeheader()
        writer.writerow({
            "job_id": "job",
            "stage": "stage",
            "kind": "roi_matrix",
            "status": status,
            "detail": detail,
        })


def test_read_shards_accepts_headerless_rows_and_comments(tmp_path):
    path = tmp_path / "shards.tsv"
    path.write_text("# comment\n\nprofile\tstage\tgraph\tpr\tLRU\ttag\n")

    rows = slurm_shard_status.read_shards(path)

    assert len(rows) == 1
    assert rows[0].profile == "profile"
    assert rows[0].run_tag == "tag"


def test_summarize_shards_reports_ok_failed_and_missing(tmp_path):
    ok = slurm_shard_status.ShardRow("profile", "stage", "graph", "pr", "LRU", "tag")
    failed = slurm_shard_status.ShardRow("profile", "stage", "graph", "pr", "ECG:DBG_PRIMARY", "tag")
    missing = slurm_shard_status.ShardRow("profile", "stage", "graph", "pr", "POPT", "tag")
    runs_root = tmp_path / "runs"

    write_jobs(slurm_shard_status.shard_run_dir(ok, runs_root) / "jobs.csv", "ok", "1 ok row")
    write_jobs(slurm_shard_status.shard_run_dir(failed, runs_root) / "jobs.csv", "failed", "statuses=['error'] rows=1")

    rows = slurm_shard_status.summarize_shards([ok, failed, missing], runs_root)
    by_policy = {row["policy"]: row for row in rows}

    assert by_policy["LRU"]["status"] == "ok"
    assert by_policy["ECG:DBG_PRIMARY"]["status"] == "failed"
    assert by_policy["ECG:DBG_PRIMARY"]["job_status_counts"] == "failed:1"
    assert by_policy["POPT"]["status"] == "not_started"


def test_write_status_writes_stable_columns(tmp_path):
    out_path = tmp_path / "status.csv"
    slurm_shard_status.write_status(out_path, [{field: "" for field in slurm_shard_status.STATUS_FIELDNAMES}])

    header = out_path.read_text().splitlines()[0]
    assert header == ",".join(slurm_shard_status.STATUS_FIELDNAMES)