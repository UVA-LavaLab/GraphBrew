"""Prevent retired campaign and generated-artifact bloat from returning."""

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_retired_cleanup_surfaces_stay_removed():
    removed = (
        "bench/include/graphbrew/reorder/reorder_don_lite.h",
        "scripts/lib/analysis/amortise.py",
        "scripts/lib/analysis/figures.py",
        "scripts/experiments/adaptive/runner.py",
        "scripts/experiments/adaptive/cpu_sprint.py",
        "scripts/lib/analysis/adaptive_pilot.py",
        "scripts/lib/pipeline/adaptive_pilot_contract.py",
        "docs/figures/logo.svg",
    )
    assert not [
        relative
        for relative in removed
        if (PROJECT_ROOT / relative).exists()
    ]


def test_public_logo_and_sssp_snapshot_are_compact():
    logo = PROJECT_ROOT / "docs/figures/logo.png"
    assert logo.stat().st_size < 256 * 1024

    snapshot_path = (
        PROJECT_ROOT
        / "scripts/experiments/vldb/sssp_delta_tuning.json"
    )
    assert snapshot_path.stat().st_size < 64 * 1024
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["artifact_kind"] == "policy-validation-snapshot"
    assert all(
        set(record) == {"graph_info"}
        for record in snapshot["graphs"].values()
    )


def test_public_orchestrator_hides_retired_campaign_flags():
    result = subprocess.run(
        [sys.executable, "scripts/graphbrew_experiment.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    for retired in (
        "--adaptive-sprint1",
        "--adaptive-cpu",
        "--generate-figures",
    ):
        assert retired not in result.stdout
