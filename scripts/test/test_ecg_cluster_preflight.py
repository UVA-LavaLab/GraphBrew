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
    assert "matches scale-proof contract" in result.stdout
    assert "sbatch:slurm_ecg_pfx_scale_proof.sbatch" in result.stdout
    assert "syntax ok" in result.stdout
    assert "sbatch:slurm_final_shard.sbatch:exclusive" in result.stdout


def test_cluster_preflight_rejects_invalid_scale_backend(tmp_path):
    shards = tmp_path / "scale.tsv"
    shards.write_text("11\t0\tbad-backend\t/tmp/out\n")

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
        check=False,
    )

    assert result.returncode == 1
    assert "bad-backend" in result.stdout
    assert "must be one of" in result.stdout


def test_cluster_preflight_rejects_invalid_scale_root(tmp_path):
    shards = tmp_path / "scale.tsv"
    shards.write_text("11\t-1\tsniper\t/tmp/out\n")

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
        check=False,
    )

    assert result.returncode == 1
    assert "root -1" in result.stdout
    assert "must be nonnegative" in result.stdout


def test_cluster_preflight_rejects_comment_rows_in_final_shards(tmp_path):
    shards = tmp_path / "final.tsv"
    shards.write_text("# comment\nfinal_replacement\t20_gem5_large_replacement\tcit-Patents\tpr\tLRU\trun\n")

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
        check=False,
    )

    assert result.returncode != 0
    assert "comments are not allowed" in result.stderr


def test_cluster_preflight_rejects_blank_rows_in_scale_shards(tmp_path):
    shards = tmp_path / "scale.tsv"
    shards.write_text("\n11\t0\tsniper\t/tmp/out\n")

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
        check=False,
    )

    assert result.returncode != 0
    assert "blank lines are not allowed" in result.stderr


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
    assert "matches manifest" in result.stdout


def test_cluster_preflight_rejects_invalid_shard_policy(tmp_path):
    shards = tmp_path / "final.tsv"
    shards.write_text("final_replacement\t20_gem5_large_replacement\tcit-Patents\tpr\tNOT_A_POLICY\trun\n")

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
        check=False,
    )

    assert result.returncode == 1
    assert "NOT_A_POLICY" in result.stdout
    assert "not valid for stage" in result.stdout


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


def test_cluster_preflight_reports_sbatch_syntax_errors(tmp_path):
    bad_script = tmp_path / "bad.sbatch"
    bad_script.write_text("#!/bin/bash\nif [[ 1 ]]; then\n")

    import importlib.util
    import sys

    path = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "ecg_cluster_preflight.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ecg_cluster_preflight", path)
    ecg_cluster_preflight = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ecg_cluster_preflight"] = ecg_cluster_preflight
    spec.loader.exec_module(ecg_cluster_preflight)

    result = ecg_cluster_preflight.check_sbatch_script(bad_script)

    assert not result.ok
    assert result.name == "sbatch:bad.sbatch"


def test_cluster_preflight_reports_missing_exclusive_directive(tmp_path):
    script = tmp_path / "noexclusive.sbatch"
    script.write_text("#!/bin/bash\n#SBATCH --job-name=test\n")

    import importlib.util
    import sys

    path = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "ecg_cluster_preflight.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ecg_cluster_preflight", path)
    ecg_cluster_preflight = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ecg_cluster_preflight"] = ecg_cluster_preflight
    spec.loader.exec_module(ecg_cluster_preflight)

    result = ecg_cluster_preflight.check_sbatch_directive(script, "--exclusive", "exclusive")
    waived = ecg_cluster_preflight.check_sbatch_directive(script, "--exclusive", "exclusive", waived=True)

    assert not result.ok
    assert "missing #SBATCH --exclusive" in result.detail
    assert waived.ok
    assert "waived" in waived.detail


def test_cluster_preflight_checks_python_environment(tmp_path):
    import importlib.util
    import sys

    path = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "ecg_cluster_preflight.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ecg_cluster_preflight", path)
    ecg_cluster_preflight = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ecg_cluster_preflight"] = ecg_cluster_preflight
    spec.loader.exec_module(ecg_cluster_preflight)

    checks = ecg_cluster_preflight.check_python_environment(tmp_path)
    assert {check.name for check in checks} == {
        "python-env:activate",
        "python-env:python3",
        "python-env:requirements",
    }
    assert not any(check.ok for check in checks)

    activate = tmp_path / ".venv" / "bin" / "activate"
    python = tmp_path / ".venv" / "bin" / "python3"
    requirements = tmp_path / "scripts" / "requirements.txt"
    activate.parent.mkdir(parents=True)
    requirements.parent.mkdir(parents=True)
    activate.write_text("# test activate\n")
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    requirements.write_text("pytest\n")

    assert all(check.ok for check in ecg_cluster_preflight.check_python_environment(tmp_path))
