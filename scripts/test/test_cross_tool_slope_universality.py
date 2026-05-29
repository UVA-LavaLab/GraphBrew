"""Pytest gate for cross_tool_slope_universality.json (gate 76).

Locks the cross-tool roll-up invariant: every (tool, policy) median
slope must be negative, in-band, and the per-tool steepness span must
not exceed the documented ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "wiki" / "data" / "cross_tool_slope_universality.json"

EXPECTED_TOOLS = ["cache-sim", "gem5", "sniper"]
EXPECTED_CACHE_SIM_POLICIES = ["GRASP", "LRU", "POPT", "SRRIP"]
EXPECTED_GEM5_POLICIES      = ["GRASP", "LRU", "SRRIP"]
EXPECTED_SNIPER_POLICIES    = ["GRASP", "LRU", "SRRIP"]
EXPECTED_IN_BAND_COUNT = 10  # 4 + 3 + 3
MIN_SLOPE_PP_OCT = -25.0
MAX_SLOPE_PP_OCT = -0.5
STEEPNESS_SPAN_CEILING_PP_OCT = 5.0


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        pytest.skip(f"missing {JSON_PATH}; run `make lit-cross-tool-slope-universality`")
    return json.loads(JSON_PATH.read_text())


def test_verdict_pass(payload: dict) -> None:
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_all_tools_present(payload: dict) -> None:
    assert payload["meta"]["tools"] == EXPECTED_TOOLS


def test_cache_sim_policies_complete(payload: dict) -> None:
    assert payload["meta"]["tool_policies"]["cache-sim"] == EXPECTED_CACHE_SIM_POLICIES


def test_gem5_policies_complete(payload: dict) -> None:
    assert payload["meta"]["tool_policies"]["gem5"] == EXPECTED_GEM5_POLICIES


def test_sniper_policies_complete(payload: dict) -> None:
    assert payload["meta"]["tool_policies"]["sniper"] == EXPECTED_SNIPER_POLICIES


def test_every_tool_policy_median_is_negative(payload: dict) -> None:
    bad = [
        (t, p, v)
        for t, med in payload["meta"]["medians"].items()
        for p, v in med.items()
        if v >= 0.0
    ]
    assert not bad, f"non-negative medians: {bad}"


def test_every_tool_policy_median_in_band(payload: dict) -> None:
    bad = [
        (t, p, v)
        for t, med in payload["meta"]["medians"].items()
        for p, v in med.items()
        if v < MIN_SLOPE_PP_OCT or v > MAX_SLOPE_PP_OCT
    ]
    assert not bad, f"out-of-band medians: {bad}"


def test_in_band_count_matches_expected(payload: dict) -> None:
    assert payload["meta"]["in_band_count"] == EXPECTED_IN_BAND_COUNT
    assert payload["meta"]["expected_in_band_count"] == EXPECTED_IN_BAND_COUNT


def test_no_tool_exceeds_steepness_span_ceiling(payload: dict) -> None:
    bad = [
        (t, s)
        for t, s in payload["meta"]["steepness_spans"].items()
        if s > STEEPNESS_SPAN_CEILING_PP_OCT
    ]
    assert not bad, f"span > {STEEPNESS_SPAN_CEILING_PP_OCT}: {bad}"


def test_no_violations(payload: dict) -> None:
    assert payload["meta"]["violations"] == []


def test_documented_thresholds_match_module(payload: dict) -> None:
    assert payload["meta"]["min_slope_pp_oct"] == MIN_SLOPE_PP_OCT
    assert payload["meta"]["max_slope_pp_oct"] == MAX_SLOPE_PP_OCT
    assert payload["meta"]["steepness_span_ceiling_pp_oct"] == STEEPNESS_SPAN_CEILING_PP_OCT


def test_verdict_checks_all_true(payload: dict) -> None:
    checks = payload["meta"]["verdict_checks"]
    assert all(checks.values()), checks
