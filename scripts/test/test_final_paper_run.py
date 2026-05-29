#!/usr/bin/env python3
"""Regression tests for final-paper run orchestration."""

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_RUN_PATH = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "final_paper_run.py"
spec = importlib.util.spec_from_file_location("final_paper_run", FINAL_RUN_PATH)
final_paper_run = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["final_paper_run"] = final_paper_run
spec.loader.exec_module(final_paper_run)


def test_combined_outputs_preserve_failed_roi_rows(tmp_path):
    out_dir = tmp_path / "matrices" / "cit" / "pr"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "roi_matrix.csv"
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["status", "error", "benchmark", "policy_label"])
        writer.writeheader()
        writer.writerow({
            "status": "error",
            "error": "exit_code=124",
            "benchmark": "pr",
            "policy_label": "LRU",
        })

    job = final_paper_run.Job(
        job_id="cit_pr",
        stage="09g_sniper_sift_cit_patents_smoke",
        kind="roi_matrix",
        command=["roi_matrix.py"],
        out_dir=out_dir,
        log_path=tmp_path / "logs" / "cit_pr.log",
        metadata={"graph": "cit-Patents", "graph_path": "/graphs/cit-Patents.sg"},
    )

    final_paper_run.write_combined_outputs(tmp_path, [job])

    rows = list(csv.DictReader((tmp_path / "combined_roi_matrix.csv").open(newline="")))
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "exit_code=124"
    assert rows[0]["final_output_status"] == "failed"
    assert rows[0]["final_output_detail"] == "statuses=['error'] rows=1"
    assert rows[0]["final_graph"] == "cit-Patents"
    assert rows[0]["final_graph_path"] == "/graphs/cit-Patents.sg"


def test_run_job_interrupt_records_status_and_terminates_group(tmp_path, monkeypatch):
    class InterruptingProcess:
        pid = 12345

        def __init__(self):
            self.returncode = None
            self.waits = 0

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            self.waits += 1
            self.returncode = -15
            return self.returncode

    process = InterruptingProcess()
    killed = []
    monkeypatch.setattr(final_paper_run.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(final_paper_run.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    job = final_paper_run.Job(
        job_id="interrupt_job",
        stage="stage",
        kind="roi_matrix",
        command=["roi_matrix.py"],
        out_dir=tmp_path / "missing-output",
        log_path=tmp_path / "logs" / "interrupt.log",
    )
    args = Namespace(dry_run=False, force=True, resume=True, skip_failed=False)

    code = final_paper_run.run_job(job, tmp_path, args)

    assert code == 130
    assert killed == [(12345, final_paper_run.signal.SIGTERM)]
    log_text = job.log_path.read_text()
    assert "[final_paper_run_interrupted] KeyboardInterrupt" in log_text
    assert "[interrupt_action] SIGTERM process group" in log_text
    status_lines = (tmp_path / "run_status.jsonl").read_text().splitlines()
    assert '"event": "interrupted"' in status_lines[-1]
    assert '"exit_code": 130' in status_lines[-1]


def test_print_run_status_reports_open_start_as_running(tmp_path, capsys):
    with (tmp_path / "jobs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["job_id", "stage", "kind", "status", "detail"])
        writer.writeheader()
        writer.writerow({
            "job_id": "job1",
            "stage": "stage",
            "kind": "roi_matrix",
            "status": "missing",
            "detail": "output CSV missing",
        })
    with (tmp_path / "run_status.jsonl").open("w") as fh:
        fh.write(json.dumps({
            "utc": "2026-05-25T00:00:00+00:00",
            "job_id": "job1",
            "event": "start",
        }) + "\n")

    assert final_paper_run.print_run_status(tmp_path) == 0
    output = capsys.readouterr().out

    assert "counts={'running': 1}" in output
    assert "job1: running" in output


def test_print_run_status_reports_nonzero_finish_as_failed(tmp_path, capsys):
    with (tmp_path / "jobs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["job_id", "stage", "kind", "status", "detail"])
        writer.writeheader()
        writer.writerow({
            "job_id": "job1",
            "stage": "stage",
            "kind": "roi_matrix",
            "status": "missing",
            "detail": "output CSV missing",
        })
    with (tmp_path / "run_status.jsonl").open("w") as fh:
        fh.write(json.dumps({
            "utc": "2026-05-25T00:00:00+00:00",
            "job_id": "job1",
            "event": "finish",
            "exit_code": 124,
            "detail": "manual timeout cleanup",
        }) + "\n")

    assert final_paper_run.print_run_status(tmp_path) == 0
    output = capsys.readouterr().out

    assert "counts={'failed': 1}" in output
    assert "job1: failed (manual timeout cleanup)" in output


def test_print_run_status_uses_finish_output_status(tmp_path, capsys):
    with (tmp_path / "jobs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["job_id", "stage", "kind", "status", "detail"])
        writer.writeheader()
        writer.writerow({
            "job_id": "job1",
            "stage": "stage",
            "kind": "roi_matrix",
            "status": "missing",
            "detail": "output CSV missing",
        })
    with (tmp_path / "run_status.jsonl").open("w") as fh:
        fh.write(json.dumps({
            "utc": "2026-05-25T00:00:00+00:00",
            "job_id": "job1",
            "event": "finish",
            "exit_code": 0,
            "output_status": "ok",
            "detail": "2 ok rows",
        }) + "\n")

    assert final_paper_run.print_run_status(tmp_path) == 0
    output = capsys.readouterr().out

    assert "counts={'ok': 1}" in output


def test_ecg_pfx_profile_passes_explicit_runner_knobs(tmp_path):
    args = final_paper_run.parse_args([
        "--profile", "available_cache_sim_ecg_pfx",
        "--run-dir", str(tmp_path),
        "--dry-run",
        "--allow-missing-graphs",
    ])
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)
    command = jobs[0].command

    assert "--prefetcher" in command
    assert command[command.index("--prefetcher") + 1] == "ECG_PFX"
    assert command[command.index("--ecg-pfx-mode") + 1] == "popt"
    assert command[command.index("--ecg-pfx-window") + 1] == "16"
    assert command[command.index("--ecg-pfx-lookahead") + 1] == "4"
    assert command[command.index("--ecg-pfx-hint-filter") + 1] == "16"
    assert command[command.index("--ecg-pfx-delivery") + 1] == "explicit-hint"


def test_gem5_ecg_pfx_smoke_profile_passes_allow_flag(tmp_path):
    args = final_paper_run.parse_args([
        "--profile", "gem5_ecg_pfx_tiny_smoke",
        "--run-dir", str(tmp_path),
        "--dry-run",
    ])
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)

    assert len(jobs) == 3
    for job in jobs:
        assert "--prefetcher" in job.command
        assert job.command[job.command.index("--prefetcher") + 1] == "ECG_PFX"
        assert job.command[job.command.index("--ecg-pfx-delivery") + 1] == "instruction"
        assert "--allow-gem5-ecg-pfx" in job.command


def test_local_cache_sim_diversity_profiles_expand_extra_kernels(tmp_path):
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)
    args = final_paper_run.parse_args([
        "--profile", "local_cache_sim_diversity_smoke",
        "--run-dir", str(tmp_path),
        "--dry-run",
        "--no-build",
    ])

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)
    benchmarks = {job.metadata["benchmark"] for job in jobs}

    assert {"pr_spmv", "cc", "cc_sv", "bc", "tc"}.issubset(benchmarks)
    assert len(jobs) == 8


def test_local_cache_sim_medium_diversity_avoids_expensive_kernels(tmp_path):
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)
    args = final_paper_run.parse_args([
        "--profile", "local_cache_sim_diversity_medium",
        "--run-dir", str(tmp_path),
        "--dry-run",
        "--no-build",
    ])

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)
    benchmarks = {job.metadata["benchmark"] for job in jobs}

    assert {"pr", "pr_spmv", "bfs", "sssp", "cc", "cc_sv"} == benchmarks
    assert "bc" not in benchmarks
    assert "tc" not in benchmarks


def test_local_cache_sim_pfx_diversity_uses_ecg_pfx(tmp_path):
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)
    args = final_paper_run.parse_args([
        "--profile", "local_cache_sim_pfx_diversity_smoke",
        "--run-dir", str(tmp_path),
        "--dry-run",
        "--no-build",
    ])

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)

    assert len(jobs) == 6
    for job in jobs:
        assert job.command[job.command.index("--suite") + 1] == "cache-sim"
        assert job.command[job.command.index("--prefetcher") + 1] == "ECG_PFX"


def test_sniper_ecg_pfx_smoke_profile_uses_bounded_sift(tmp_path):
    args = final_paper_run.parse_args([
        "--profile", "sniper_sift_ecg_pfx_smoke",
        "--run-dir", str(tmp_path),
        "--dry-run",
    ])
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)

    assert len(jobs) == 3
    for job in jobs:
        command = job.command
        assert command[command.index("--suite") + 1] == "sniper"
        assert command[command.index("--prefetcher") + 1] == "ECG_PFX"
        assert command[command.index("--sniper-workload") + 1] == "benchmark"
        assert command[command.index("--sniper-frontend") + 1] == "sift"
        assert command[command.index("--sniper-memory-limit-gb") + 1] == "4"
        assert "--allow-sniper-benchmark-workload" in command


def test_sniper_file_ecg_pfx_smoke_profile_uses_benchmark_lookahead(tmp_path):
    args = final_paper_run.parse_args([
        "--profile", "sniper_sift_file_ecg_pfx_smoke",
        "--run-dir", str(tmp_path),
        "--dry-run",
        "--allow-missing-graphs",
    ])
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)

    assert len(jobs) == 3
    by_benchmark = {job.metadata["benchmark"]: job.command for job in jobs}
    assert {"pr", "bfs", "sssp"} == set(by_benchmark)
    for benchmark in ("pr", "bfs"):
        command = by_benchmark[benchmark]
        assert command[command.index("--prefetcher") + 1] == "ECG_PFX"
        assert command[command.index("--ecg-pfx-lookahead") + 1] == "4"
        assert command[command.index("--sniper-frontend") + 1] == "sift"
        assert "--allow-sniper-benchmark-workload" in command
    assert by_benchmark["sssp"][by_benchmark["sssp"].index("--ecg-pfx-lookahead") + 1] == "0"


def test_sniper_file_droplet_smoke_profile_uses_tuned_knobs(tmp_path):
    args = final_paper_run.parse_args([
        "--profile", "sniper_sift_file_droplet_smoke",
        "--run-dir", str(tmp_path),
        "--dry-run",
        "--allow-missing-graphs",
    ])
    manifest = final_paper_run.load_manifest(final_paper_run.DEFAULT_MANIFEST)

    jobs = final_paper_run.expand_jobs(args, manifest, tmp_path)

    assert len(jobs) == 3
    for job in jobs:
        command = job.command
        assert command[command.index("--prefetcher") + 1] == "DROPLET"
        assert command[command.index("--droplet-prefetch-degree") + 1] == "2"
        assert command[command.index("--droplet-indirect-degree") + 1] == "4"
        assert command[command.index("--droplet-stride-table-size") + 1] == "16"
        assert command[command.index("--sniper-frontend") + 1] == "sift"
        assert "--allow-sniper-benchmark-workload" in command


def test_roi_job_can_set_omp_threads_env(tmp_path):
    args = final_paper_run.parse_args(["--profile", "rehearsal", "--run-dir", str(tmp_path)])
    manifest = {
        "benchmark_options": {"synthetic_g12": {"pr": "-g 1"}},
    }
    settings = {
        "name": "stage",
        "suite": "cache-sim",
        "l1d_size": "1kB",
        "l2_size": "2kB",
        "l3_ways": "16",
        "line_size": "64",
        "timeout_cache": 1,
        "timeout_gem5": 1,
        "timeout_sniper": 1,
        "policies": ["LRU"],
        "omp_threads": 1,
    }

    job = final_paper_run.make_roi_job(
        args,
        manifest,
        tmp_path,
        settings,
        {"name": "synthetic_g12", "options_key": "synthetic_g12"},
        None,
        "pr",
    )

    assert job.env == {"OMP_NUM_THREADS": "1"}


def test_synthetic_option_sets_do_not_require_graph_files(tmp_path):
    graph = {"name": "synthetic_ecg_pfx_tiny", "options_key": "synthetic_ecg_pfx_tiny"}

    assert final_paper_run.graph_uses_synthetic_options(graph)


def test_final_cache_sim_l_curve_profile_shape():
    """Lock the L-curve sweep design: bounded scope, ≥4 L3 sizes, baseline policies.

    This invariant exists because the L-curve figure must show the full L-shape
    (divergent at small L3 → convergent at large L3). If a future edit shrinks
    the L3 sweep or removes a baseline policy, this test should fail so we notice.
    """
    manifest_path = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "final_paper_manifest.json"
    manifest = final_paper_run.load_manifest(manifest_path)

    assert "final_cache_sim_l_curve" in manifest["profiles"], (
        "L-curve profile description must be declared in manifest.profiles"
    )

    stages = [s for s in manifest["stages"] if "final_cache_sim_l_curve" in s.get("profiles", [])]
    assert len(stages) == 1, f"expected exactly one L-curve stage, got {[s['name'] for s in stages]}"
    stage = stages[0]

    assert stage["suite"] == "cache-sim", "L-curve uses cache_sim for speed"
    assert stage["kind"] == "roi_matrix"

    graph_set = manifest["graph_sets"][stage["graph_set"]]
    assert 1 <= len(graph_set) <= 2, (
        f"L-curve must stay bounded (<=2 graphs); got {[g['name'] for g in graph_set]}"
    )

    l3_sizes = stage["l3_sizes"]
    assert len(l3_sizes) >= 4, f"L-curve needs >=4 L3 points to show the L-shape; got {l3_sizes}"
    assert "4kB" in l3_sizes, "L-curve must include the small-cache divergence point"
    assert any(sz in l3_sizes for sz in ("1MB", "2MB", "4MB", "8MB")), (
        f"L-curve must include a large-cache convergence point; got {l3_sizes}"
    )

    assert set(stage["benchmarks"]).issubset({"pr", "bfs", "sssp", "bc"}), (
        f"unexpected benchmark in L-curve: {stage['benchmarks']}"
    )
    assert stage["benchmarks"], "L-curve needs at least one benchmark"

    baselines = {"LRU", "SRRIP", "GRASP"}
    assert baselines.issubset(set(stage["policies"])), (
        f"L-curve must include the three GRASP-paper baselines (LRU,SRRIP,GRASP); got {stage['policies']}"
    )