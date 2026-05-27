#!/usr/bin/env python3
"""Regression tests for ECG Slurm wrapper guardrails."""

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_WRAPPER = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "slurm_final_shard.sbatch"
SCALE_WRAPPER = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "slurm_ecg_pfx_scale_proof.sbatch"


def run_wrapper(
    wrapper: Path,
    shards: Path,
    array_task_id: str | None = "0",
    cwd: Path = PROJECT_ROOT,
    shards_value: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GRAPHBREW_ROOT"] = str(PROJECT_ROOT)
    env["SHARDS"] = shards_value if shards_value is not None else str(shards)
    if array_task_id is None:
        env.pop("SLURM_ARRAY_TASK_ID", None)
    else:
        env["SLURM_ARRAY_TASK_ID"] = array_task_id
    return subprocess.run(["bash", str(wrapper)], cwd=cwd, env=env, text=True, capture_output=True, check=False)


def test_final_wrapper_rejects_malformed_shard_row(tmp_path):
    shards = tmp_path / "bad_final.tsv"
    shards.write_text("final_replacement\tstage\tcit-Patents\tpr\tLRU\n")

    result = run_wrapper(FINAL_WRAPPER, shards)

    assert result.returncode == 4
    assert "expected 6 tab-separated fields" in result.stderr


def test_final_wrapper_requires_array_task_id(tmp_path):
    shards = tmp_path / "one.tsv"
    shards.write_text("final_replacement\tstage\tcit-Patents\tpr\tLRU\trun\n")

    result = run_wrapper(FINAL_WRAPPER, shards, array_task_id=None)

    assert result.returncode == 2
    assert "SLURM_ARRAY_TASK_ID must be set" in result.stderr



def test_scale_wrapper_rejects_malformed_shard_row(tmp_path):
    shards = tmp_path / "bad_scale.tsv"
    shards.write_text("11\t0\tsniper\n")

    result = run_wrapper(SCALE_WRAPPER, shards)

    assert result.returncode == 4
    assert "expected 4 tab-separated fields" in result.stderr


def test_final_wrapper_resolves_relative_shards_after_graphbrew_root_cd(tmp_path):
    shards = tmp_path / "bad_relative.tsv"
    shards.write_text("final_replacement\tstage\tcit-Patents\tpr\tLRU\n")
    relative_to_root = os.path.relpath(shards, PROJECT_ROOT)

    result = run_wrapper(FINAL_WRAPPER, shards, cwd=tmp_path, shards_value=relative_to_root)

    assert result.returncode == 4
    assert "expected 6 tab-separated fields" in result.stderr


def test_scale_wrapper_requires_array_task_id_with_shards(tmp_path):
    shards = tmp_path / "scale.tsv"
    shards.write_text("11\t0\tsniper\t/tmp/out\n")

    result = run_wrapper(SCALE_WRAPPER, shards, array_task_id=None)

    assert result.returncode == 2
    assert "SLURM_ARRAY_TASK_ID must be set" in result.stderr
