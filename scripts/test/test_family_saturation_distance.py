"""Pytest gate for family_saturation_distance.json (gate 79).

Locks the per-family saturation-distance invariants: every family
has non-negative median distance, citation/social meet the
high-headroom floor (>= 5 pp), web is pinned as the low-headroom
outlier (< 5 pp), and family ordering citation >= social >= web
holds within a 1 pp slack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "wiki" / "data" / "family_saturation_distance.json"

EXPECTED_FAMILIES = {"citation", "social", "web"}
EXPECTED_HIGH_HEADROOM = ("citation", "social")
EXPECTED_LOW_HEADROOM = ("web",)
HIGH_HEADROOM_FLOOR_PP = 5.0
LOW_HEADROOM_CEILING_PP = 5.0
ORDERING_SLACK_PP = 1.0


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        pytest.skip(f"missing {JSON_PATH}; run `make lit-family-saturation-distance`")
    return json.loads(JSON_PATH.read_text())


def test_verdict_pass(payload: dict) -> None:
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_expected_families_present(payload: dict) -> None:
    assert EXPECTED_FAMILIES.issubset(set(payload["meta"]["families"]))


def test_all_family_medians_nonneg(payload: dict) -> None:
    bad = [
        (f, v["median_pp"])
        for f, v in payload["meta"]["per_family"].items()
        if v["median_pp"] < 0.0
    ]
    assert not bad, bad


def test_all_family_mins_nonneg(payload: dict) -> None:
    bad = [
        (f, v["min_pp"])
        for f, v in payload["meta"]["per_family"].items()
        if v["min_pp"] < 0.0
    ]
    assert not bad, bad


def test_citation_meets_high_headroom_floor(payload: dict) -> None:
    cit = payload["meta"]["per_family"]["citation"]["median_pp"]
    assert cit >= HIGH_HEADROOM_FLOOR_PP, cit


def test_social_meets_high_headroom_floor(payload: dict) -> None:
    soc = payload["meta"]["per_family"]["social"]["median_pp"]
    assert soc >= HIGH_HEADROOM_FLOOR_PP, soc


def test_web_under_low_headroom_ceiling(payload: dict) -> None:
    web = payload["meta"]["per_family"]["web"]["median_pp"]
    assert web < LOW_HEADROOM_CEILING_PP, web


def test_family_ordering_citation_social_web(payload: dict) -> None:
    pf = payload["meta"]["per_family"]
    cit, soc, web = pf["citation"]["median_pp"], pf["social"]["median_pp"], pf["web"]["median_pp"]
    assert cit + ORDERING_SLACK_PP >= soc, (cit, soc)
    assert soc + ORDERING_SLACK_PP >= web, (soc, web)


def test_documented_thresholds_match_module(payload: dict) -> None:
    m = payload["meta"]
    assert m["high_headroom_floor_pp"]  == HIGH_HEADROOM_FLOOR_PP
    assert m["low_headroom_ceiling_pp"] == LOW_HEADROOM_CEILING_PP
    assert m["ordering_slack_pp"]       == ORDERING_SLACK_PP


def test_high_headroom_families_listed_correctly(payload: dict) -> None:
    assert tuple(payload["meta"]["high_headroom_families"]) == EXPECTED_HIGH_HEADROOM


def test_pinned_low_headroom_families_listed_correctly(payload: dict) -> None:
    assert tuple(payload["meta"]["pinned_low_headroom_families"]) == EXPECTED_LOW_HEADROOM


def test_verdict_checks_all_true(payload: dict) -> None:
    checks = payload["meta"]["verdict_checks"]
    assert all(checks.values()), checks
