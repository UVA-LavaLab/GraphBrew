#!/usr/bin/env python3
"""The legacy non-nested LOGO protocol must fail closed."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.experiments.adaptive_ml.exp3_model_ablation import run_logo_cv
from scripts.lib.analysis.cold_start_sim import (
    run_cold_start_experiment,
    simulate_cold_start,
)
from scripts.lib.ml.weights import (
    cross_validate_logo,
    cross_validate_logo_grouped,
)
from scripts.lib.tools.evaluate_all_modes import (
    eval_logo_all_models,
    eval_logo_family_ablation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("evaluator", [
    cross_validate_logo,
    cross_validate_logo_grouped,
    simulate_cold_start,
    run_cold_start_experiment,
    run_logo_cv,
    eval_logo_all_models,
    eval_logo_family_ablation,
])
def test_retired_library_evaluators_reject_execution(evaluator):
    with pytest.raises(RuntimeError, match="non-nested LOGO"):
        evaluator()


@pytest.mark.parametrize("module,args", [
    ("scripts.lib.analysis.cold_start_sim", []),
    ("scripts.lib.tools.evaluate_all_modes", ["--logo"]),
    ("scripts.experiments.adaptive_ml.exp3_model_ablation", []),
])
def test_retired_tool_entrypoints_fail_before_loading_data(module, args):
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "non-nested LOGO" in result.stdout + result.stderr


@pytest.mark.parametrize("command", [
    [sys.executable, "scripts/graphbrew_experiment.py", "--logo"],
    [sys.executable, "-m", "scripts.lib.ml.eval_weights", "--logo"],
])
def test_official_clis_no_longer_accept_logo_flag(command):
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --logo" in result.stderr
