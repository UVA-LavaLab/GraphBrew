#!/usr/bin/env python3
"""Tests for the ECG_PFX scale-proof runner."""

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ecg_pfx_scale_proof_dry_run_emits_both_backends(tmp_path):
    result = subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_pfx_scale_proof.py",
            "--scale", "10",
            "--roots", "0",
            "--backend", "both",
            "--out-root", str(tmp_path / "proof"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "build/RISCV/gem5.opt" in result.stdout
    assert "--ecg-pfx-delivery instruction" in result.stdout
    assert "roi_matrix.py" in result.stdout
    assert "--suite sniper" in result.stdout
    assert "-g 10 -k 8 -o 2 -n 1 -r 0" in result.stdout


def test_slurm_scale_proof_wrapper_uses_runner():
    text = (PROJECT_ROOT / "scripts/experiments/ecg/slurm_ecg_pfx_scale_proof.sbatch").read_text()

    assert "ecg_pfx_scale_proof.py" in text
    assert "SCALE" in text
    assert "ROOT" in text
    assert "BACKEND" in text


def test_make_ecg_pfx_scale_shards_expands_ranges(tmp_path):
    out = tmp_path / "scale.tsv"
    subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/make_ecg_pfx_scale_shards.py",
            "--scale", "11:12",
            "--root", "0:2",
            "--backend", "sniper", "gem5-riscv",
            "--run-tag", "unit",
            "--out-root", "results/ecg_experiments/ecg_pfx_scale_proof",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    rows = out.read_text().splitlines()
    assert len(rows) == 12
    assert rows[0] == "11\t0\tsniper\tresults/ecg_experiments/ecg_pfx_scale_proof/unit/g11_r0_sniper"
    assert rows[-1] == "12\t2\tgem5-riscv\tresults/ecg_experiments/ecg_pfx_scale_proof/unit/g12_r2_gem5-riscv"


def test_ecg_pfx_scale_status_reports_ok_and_missing(tmp_path):
    ok_root = tmp_path / "ok"
    ok_root.mkdir()
    (ok_root / "summary.csv").write_text(
        "backend,scale,root,section,status,pf_issued,pf_useful,hints,ecg_pfx_issued\n"
        "sniper-sift,11,0,1,ok,2,1,8,8\n"
    )
    shards = tmp_path / "shards.tsv"
    shards.write_text(
        f"11\t0\tsniper\t{ok_root}\n"
        f"11\t1\tsniper\t{tmp_path / 'missing'}\n"
    )
    status = tmp_path / "status.csv"
    combined = tmp_path / "combined.csv"

    subprocess.run(
        [
            "python3",
            "scripts/experiments/ecg/ecg_pfx_scale_status.py",
            "--shards", str(shards),
            "--out", str(status),
            "--combined", str(combined),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    text = status.read_text()
    assert "ok" in text
    assert "not_started" in text
    assert "pf_useful_total" in text
    assert "sniper-sift" in combined.read_text()