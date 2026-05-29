"""Pytest gate for monotonicity_universality.json (gate 77).

Locks the cache-monotonicity invariant: every (graph, app, policy)
cell's miss_rate must be non-increasing as L3 grows within a
documented measurement-noise tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "wiki" / "data" / "monotonicity_universality.json"

EXPECTED_L3_AXIS_LABELS = ["4kB", "16kB", "64kB", "256kB", "1MB", "4MB", "8MB"]
MIN_EXPECTED_CELL_COUNT = 130
MIN_EXPECTED_STEP_COUNT = 300
MAX_NOISE_BUMP_PP = 0.5
BUMP_PCT_CEILING = 0.10
LARGEST_BUMP_OBSERVED_PP = 0.05  # current worst is 0.035; gate trips above 0.05


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        pytest.skip(f"missing {JSON_PATH}; run `make lit-monotonicity-universality`")
    return json.loads(JSON_PATH.read_text())


def test_verdict_pass(payload: dict) -> None:
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_no_hard_violations(payload: dict) -> None:
    assert payload["meta"]["hard_violation_count"] == 0


def test_largest_bump_within_noise_tolerance(payload: dict) -> None:
    assert payload["meta"]["largest_bump_pp"] < MAX_NOISE_BUMP_PP


def test_largest_bump_under_observed_floor(payload: dict) -> None:
    """Tighter than the noise tolerance: catch regression if largest
    bump grows beyond current empirical worst-case (~0.035 pp) plus
    a small slack to 0.05 pp."""
    assert payload["meta"]["largest_bump_pp"] < LARGEST_BUMP_OBSERVED_PP


def test_bump_pct_under_ceiling(payload: dict) -> None:
    assert payload["meta"]["bump_pct"] <= BUMP_PCT_CEILING


def test_l3_axis_labels_match(payload: dict) -> None:
    assert payload["meta"]["l3_axis_labels"] == EXPECTED_L3_AXIS_LABELS


def test_cell_count_at_or_above_floor(payload: dict) -> None:
    assert payload["meta"]["cell_count"] >= MIN_EXPECTED_CELL_COUNT


def test_step_count_at_or_above_floor(payload: dict) -> None:
    assert payload["meta"]["total_step_count"] >= MIN_EXPECTED_STEP_COUNT


def test_documented_thresholds_match_module(payload: dict) -> None:
    assert payload["meta"]["max_noise_bump_pp"] == MAX_NOISE_BUMP_PP
    assert payload["meta"]["bump_pct_ceiling"] == BUMP_PCT_CEILING


def test_largest_bump_cell_has_required_fields(payload: dict) -> None:
    if payload["meta"]["bump_count"] == 0:
        pytest.skip("no bumps to inspect")
    cell = payload["meta"]["largest_bump_cell"]
    for field in ("graph", "app", "policy", "l3_from", "l3_to", "delta_pp"):
        assert field in cell, cell


def test_bumps_sorted_by_magnitude(payload: dict) -> None:
    if payload["meta"]["bump_count"] < 2:
        pytest.skip("not enough bumps to check ordering")
    deltas = [b["delta_pp"] for b in payload["meta"]["bumps"]]
    assert deltas == sorted(deltas, reverse=True)


def test_verdict_checks_all_true(payload: dict) -> None:
    checks = payload["meta"]["verdict_checks"]
    assert all(checks.values()), checks
