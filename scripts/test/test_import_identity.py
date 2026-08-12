"""Regression guards for one canonical Python package identity."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def test_research_runners_do_not_load_unprefixed_package_aliases():
    code = """
import sys
import scripts.experiments.adaptive.runner
import scripts.experiments.vldb.runner
import scripts.experiments.vldb.figures
import scripts.experiments.ecg.runner
import scripts.lib.analysis.cold_start_sim
bad = sorted(
    name for name in sys.modules
    if name == "lib" or name.startswith("lib.")
    or name == "experiments" or name.startswith("experiments.")
)
if bad:
    raise SystemExit("unprefixed module identities: " + ", ".join(bad))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "command",
    [
        ["scripts/experiments/vldb/runner.py", "--help"],
        ["scripts/experiments/vldb/figures.py", "--help"],
        ["scripts/experiments/vldb/stages/01_prep.py", "--help"],
        ["scripts/experiments/vldb/stages/02_reorder.py", "--help"],
        ["scripts/experiments/vldb/stages/03_cpu_perf.py", "--help"],
        ["scripts/experiments/vldb/stages/04_cache_sim.py", "--help"],
        ["scripts/experiments/vldb/stages/05_aggregate.py", "--help"],
        ["scripts/experiments/ecg/runner.py", "--help"],
    ],
)
def test_direct_research_entrypoints_keep_working(command):
    result = subprocess.run(
        [sys.executable, *command],
        cwd=PROJECT_ROOT,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cold_start_module_entrypoint_keeps_working():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.lib.analysis.cold_start_sim",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
