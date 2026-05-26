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