"""Pytest gate for anchor_cell_census.json (gate 78).

Pins the gem5/Sniper anchor cell coverage against silent shrinkage
that would invalidate downstream cross-tool gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "wiki" / "data" / "anchor_cell_census.json"

EXPECTED_L3_AXIS = ["4kB", "32kB", "256kB", "2MB"]
EXPECTED_POLICIES = ["GRASP", "LRU", "SRRIP"]
EXPECTED_GEM5_CELLS = [["email-Eu-core", "bc"], ["email-Eu-core", "pr"]]
EXPECTED_SNIPER_CELLS = [
    ["cit-Patents",   "bfs"],
    ["cit-Patents",   "pr"],
    ["cit-Patents",   "sssp"],
    ["email-Eu-core", "bfs"],
    ["email-Eu-core", "pr"],
    ["email-Eu-core", "sssp"],
]
EXPECTED_GEM5_RECORDS = 6   # 2 cells * 3 policies
EXPECTED_SNIPER_RECORDS = 18  # 6 cells * 3 policies


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        pytest.skip(f"missing {JSON_PATH}; run `make lit-anchor-cell-census`")
    return json.loads(JSON_PATH.read_text())


def test_verdict_pass(payload: dict) -> None:
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_gem5_cell_count_floor(payload: dict) -> None:
    assert payload["meta"]["gem5"]["n_cells"] >= len(EXPECTED_GEM5_CELLS)


def test_sniper_cell_count_floor(payload: dict) -> None:
    assert payload["meta"]["sniper"]["n_cells"] >= len(EXPECTED_SNIPER_CELLS)


def test_gem5_has_all_expected_cells(payload: dict) -> None:
    actual = {tuple(c) for c in payload["meta"]["gem5"]["cells"]}
    expected = {tuple(c) for c in EXPECTED_GEM5_CELLS}
    assert expected.issubset(actual), expected - actual


def test_sniper_has_all_expected_cells(payload: dict) -> None:
    actual = {tuple(c) for c in payload["meta"]["sniper"]["cells"]}
    expected = {tuple(c) for c in EXPECTED_SNIPER_CELLS}
    assert expected.issubset(actual), expected - actual


def test_gem5_l3_axis_matches(payload: dict) -> None:
    assert payload["meta"]["gem5"]["l3_axis"] == EXPECTED_L3_AXIS


def test_sniper_l3_axis_matches(payload: dict) -> None:
    assert payload["meta"]["sniper"]["l3_axis"] == EXPECTED_L3_AXIS


def test_gem5_policies_match(payload: dict) -> None:
    assert payload["meta"]["gem5"]["policies"] == EXPECTED_POLICIES


def test_sniper_policies_match(payload: dict) -> None:
    assert payload["meta"]["sniper"]["policies"] == EXPECTED_POLICIES


def test_anchors_share_at_least_one_cell(payload: dict) -> None:
    assert payload["meta"]["shared_cell_count"] >= 1


def test_anchors_share_l3_and_policies(payload: dict) -> None:
    assert payload["meta"]["shared_l3_axis"] is True
    assert payload["meta"]["shared_policies"] is True


def test_record_counts_match_baseline(payload: dict) -> None:
    assert payload["meta"]["gem5"]["cell_policy_records"]   == EXPECTED_GEM5_RECORDS
    assert payload["meta"]["sniper"]["cell_policy_records"] == EXPECTED_SNIPER_RECORDS


def test_verdict_checks_all_true(payload: dict) -> None:
    checks = payload["meta"]["verdict_checks"]
    assert all(checks.values()), checks
