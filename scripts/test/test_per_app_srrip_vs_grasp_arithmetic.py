"""Gate 126 — per_app_srrip_vs_grasp.json arithmetic + verdict.

Companion to gate 73 — locks the per-app SRRIP-vs-GRASP slope ordering
that the global gate 72 already ensures at the corpus median. Every
per-app entry must reproduce from the upstream per_app_capacity_slope
medians, the SRRIP-shallower-than-GRASP slack (1.0 pp/oct) must be
applied exactly, and the bfs pin (frontier-streaming kernel) must
remain the only deviating app.

Source: wiki/data/per_app_capacity_slope.json — block
    meta.per_app[app][POLICY].median_pp

Per-app arithmetic:
    grasp = block[GRASP].median_pp
    srrip = block[SRRIP].median_pp
    delta = srrip - grasp  (NOT abs — sign matters)
    deviates = delta > 1.0   (SRRIP is shallower than GRASP by more
                              than the slack)
    if either is None → missing; entry has None deltas.

Verdict (PASS) requires all three checks:
    no_missing_apps
    no_new_deviating_apps (deviating minus pinned must be empty)
    every_app_has_both_grasp_and_srrip

Invariants (14 tests, 4 groups):
- meta + constants (3): source filename, apps = sorted upstream keys,
  allow_srrip_shallower_by_pp=1.0, pinned_deviating_apps=['bfs'].
- per_app stats from source (4): grasp/srrip medians match upstream
  rounded to 4dp; delta = srrip - grasp rounded; deviates flag
  matches (delta > 1.0); missing handling on None.
- deviating + new_deviating + missing lists (3): deviating_apps =
  [a for a if per_app[a].deviates is True], in iteration order;
  new_deviating = deviating minus pinned; missing = apps with None.
- three verdict checks + verdict (4): each check.ok matches
  documented invariant; verdict = PASS iff conjunction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/per_app_srrip_vs_grasp.json")
SOURCE = Path("wiki/data/per_app_capacity_slope.json")

ALLOW_SHALLOW_PP = 1.0
PINNED = ()
ROUND_TOL = 5e-4


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists()
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def src():
    assert SOURCE.exists()
    return json.loads(SOURCE.read_text())["meta"]["per_app"]


# ── group 1: meta + constants ────────────────────────────────────────────


def test_meta_source_filename(data):
    assert data["meta"]["source"] == "per_app_capacity_slope.json"


def test_apps_sorted_upstream_keys(data, src):
    assert data["meta"]["apps"] == sorted(src.keys())


def test_constants_match(data):
    m = data["meta"]
    assert m["allow_srrip_shallower_by_pp"] == ALLOW_SHALLOW_PP
    assert tuple(m["pinned_deviating_apps"]) == PINNED


# ── group 2: per_app from source ─────────────────────────────────────────


def test_per_app_grasp_median_from_source(data, src):
    for app, entry in data["meta"]["per_app"].items():
        expected = src[app].get("GRASP", {}).get("median_pp")
        if expected is None:
            assert entry["grasp_median_pp_oct"] is None
        else:
            assert math.isclose(
                entry["grasp_median_pp_oct"], round(expected, 4), abs_tol=ROUND_TOL
            ), app


def test_per_app_srrip_median_from_source(data, src):
    for app, entry in data["meta"]["per_app"].items():
        expected = src[app].get("SRRIP", {}).get("median_pp")
        if expected is None:
            assert entry["srrip_median_pp_oct"] is None
        else:
            assert math.isclose(
                entry["srrip_median_pp_oct"], round(expected, 4), abs_tol=ROUND_TOL
            ), app


def test_per_app_delta_equals_srrip_minus_grasp(data):
    for app, entry in data["meta"]["per_app"].items():
        g, s = entry["grasp_median_pp_oct"], entry["srrip_median_pp_oct"]
        delta = entry["srrip_minus_grasp_pp_oct"]
        if g is None or s is None:
            assert delta is None
        else:
            assert math.isclose(delta, round(s - g, 4), abs_tol=ROUND_TOL), app


def test_per_app_deviates_flag(data):
    for app, entry in data["meta"]["per_app"].items():
        delta = entry["srrip_minus_grasp_pp_oct"]
        if delta is None:
            assert entry["deviates"] is None
        else:
            assert entry["deviates"] is (delta > ALLOW_SHALLOW_PP), app


# ── group 3: deviating + missing lists ───────────────────────────────────


def test_deviating_apps_list(data):
    expected = [
        a for a in data["meta"]["apps"]
        if data["meta"]["per_app"][a]["deviates"] is True
    ]
    assert data["meta"]["deviating_apps"] == expected


def test_new_deviating_excludes_pinned(data):
    expected = [a for a in data["meta"]["deviating_apps"] if a not in PINNED]
    assert data["meta"]["new_deviating_apps"] == expected


def test_missing_apps_list(data):
    expected = [
        a for a in data["meta"]["apps"]
        if data["meta"]["per_app"][a]["srrip_minus_grasp_pp_oct"] is None
    ]
    assert data["meta"]["missing_apps"] == expected


# ── group 4: verdict checks + verdict ────────────────────────────────────


def test_check_no_missing_apps(data):
    m = data["meta"]
    expected = len(m["missing_apps"]) == 0
    assert m["verdict_checks"]["no_missing_apps"] is expected


def test_check_no_new_deviating_apps(data):
    m = data["meta"]
    expected = len(m["new_deviating_apps"]) == 0
    assert m["verdict_checks"]["no_new_deviating_apps"] is expected


def test_check_every_app_has_both(data):
    m = data["meta"]
    expected = all(
        m["per_app"][a]["srrip_minus_grasp_pp_oct"] is not None for a in m["apps"]
    )
    assert m["verdict_checks"]["every_app_has_both_grasp_and_srrip"] is expected


def test_verdict_is_conjunction(data):
    m = data["meta"]
    expected = "PASS" if all(m["verdict_checks"].values()) else "FAIL"
    assert m["verdict"] == expected


def test_pinned_bfs_actually_deviates_in_current_run(data):
    bfs = data["meta"]["per_app"].get("bfs")
    assert bfs is not None, "bfs must be in per_app to validate the pin"
    if bfs["srrip_minus_grasp_pp_oct"] is not None:
        assert bfs["deviates"] is False, (
            "bfs now OBEYS the SRRIP-vs-GRASP ordering at array-relative GRASP "
            "0.15 (single-thread); it is no longer a pinned deviation (the "
            "multi-thread frontier-flatness deviation is gone)."
        )
