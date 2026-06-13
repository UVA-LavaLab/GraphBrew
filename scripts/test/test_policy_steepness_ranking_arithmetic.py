"""Gate 122 — policy_steepness_ranking.json arithmetic + checks.

Locks the per-policy final-octave (4MB→8MB) absolute-steepness ranking
that is the headline cache-sensitivity invariant under the faithful
1-way-charged-P-OPT corpus:

    GRASP is FLATTEST (median <= every other policy)
    LRU   is STEEPEST (median >= every other policy)
    GRASP <= LRU, with LRU median >= 1.5x GRASP median
    POPT  min slope <= 0.2 (at least one app fully saturates).

The artifact pulls per-app final-octave slope magnitudes from
cache_saturation_onset.json (already locked by gate 118) and runs
five explicit checks; this gate reproduces every per-policy stat
and every check from raw data, so any regression in the
cache-sensitivity story surfaces immediately.

NOTE: the earlier "oracle-aware {POPT,GRASP} both flat / < half of
non-oracle" model was a multi-thread + uncharged-P-OPT artifact. Under
the faithful 1-way RRM charge, P-OPT is a practical mid-pack policy;
GRASP (degree heuristic) is the only flat policy.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/policy_steepness_ranking.json")
SOURCE = Path("wiki/data/cache_saturation_onset.json")

FLATTEST_POLICY = "GRASP"
STEEPEST_POLICY = "LRU"
ALL_POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")

LRU_OVER_GRASP_SPREAD = 1.5
POPT_MIN_CEILING = 0.2

EPS = 1e-9
STAT_TOL = 1e-4


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def source():
    assert SOURCE.exists(), f"missing source artifact: {SOURCE}"
    return json.loads(SOURCE.read_text())


# ── group 1: schema + meta ───────────────────────────────────────────────


def test_schema_and_source(data):
    assert data["schema"] == "policy_steepness_ranking/v1"
    assert data["source"].endswith("cache_saturation_onset.json")


def test_meta_policy_anchors(data):
    meta = data["meta"]
    assert meta["flattest_policy"] == FLATTEST_POLICY
    assert meta["steepest_policy"] == STEEPEST_POLICY
    assert set(meta["policies"]) == set(ALL_POLICIES)


def test_meta_thresholds(data):
    t = data["meta"]["thresholds"]
    assert t["lru_over_grasp_spread"] == LRU_OVER_GRASP_SPREAD
    assert t["popt_min_slope_ceiling_pp"] == POPT_MIN_CEILING


# ── group 2: per_policy stats from source ────────────────────────────────


def test_per_policy_per_app_from_source(data, source):
    apps = source["meta"]["apps"]
    src_per_app = source["per_app"]
    for pol in ALL_POLICIES:
        entry = data["per_policy"][pol]
        for app in apps:
            expected = abs(float(src_per_app[app][pol]["final_octave_slope_pp"]))
            actual = entry["per_app"][app]
            assert math.isclose(actual, expected, abs_tol=STAT_TOL), (
                f"{pol}/{app}: per_app={actual} expected={expected:.6f}"
            )


def test_per_policy_n_matches_app_count(data, source):
    n_apps = len(source["meta"]["apps"])
    for pol in ALL_POLICIES:
        assert data["per_policy"][pol]["n"] == n_apps, f"{pol}: n mismatch"


def test_per_policy_min_median_mean_max(data):
    for pol in ALL_POLICIES:
        entry = data["per_policy"][pol]
        vals = list(entry["per_app"].values())
        assert math.isclose(entry["min"], min(vals), abs_tol=STAT_TOL), f"{pol}: min"
        assert math.isclose(entry["max"], max(vals), abs_tol=STAT_TOL), f"{pol}: max"
        assert math.isclose(
            entry["median"], statistics.median(vals), abs_tol=STAT_TOL
        ), f"{pol}: median"
        assert math.isclose(
            entry["mean"], sum(vals) / len(vals), abs_tol=STAT_TOL
        ), f"{pol}: mean"


# ── group 3: medians + ranking ───────────────────────────────────────────


def test_medians_pp_matches_per_policy(data):
    for pol in ALL_POLICIES:
        assert math.isclose(
            data["medians_pp"][pol], data["per_policy"][pol]["median"], abs_tol=STAT_TOL
        ), f"{pol}: medians_pp mismatch"


def test_ranking_by_median_sorted_ascending(data):
    expected = sorted(ALL_POLICIES, key=lambda p: data["medians_pp"][p])
    assert data["ranking_by_median"] == expected, (
        f"ranking={data['ranking_by_median']} expected={expected}"
    )


# ── group 4: five checks + verdict ───────────────────────────────────────


def test_check_grasp_is_flattest(data):
    m = data["medians_pp"]
    chk = data["checks"]["grasp_is_flattest"]
    others = [p for p in ALL_POLICIES if p != FLATTEST_POLICY]
    assert chk["ok"] is all(m[FLATTEST_POLICY] <= m[p] + EPS for p in others)
    assert math.isclose(chk["grasp"], m[FLATTEST_POLICY], abs_tol=STAT_TOL)


def test_check_lru_is_steepest(data):
    m = data["medians_pp"]
    chk = data["checks"]["lru_is_steepest"]
    others = [p for p in ALL_POLICIES if p != STEEPEST_POLICY]
    assert chk["ok"] is all(m[STEEPEST_POLICY] >= m[p] - EPS for p in others)
    assert math.isclose(chk["lru"], m[STEEPEST_POLICY], abs_tol=STAT_TOL)


def test_check_grasp_le_lru_median(data):
    m = data["medians_pp"]
    chk = data["checks"]["grasp_le_lru_median"]
    assert chk["ok"] is (m[FLATTEST_POLICY] <= m[STEEPEST_POLICY] + EPS)


def test_check_steepness_spread(data):
    m = data["medians_pp"]
    chk = data["checks"]["steepness_spread"]
    expected = m[STEEPEST_POLICY] >= LRU_OVER_GRASP_SPREAD * m[FLATTEST_POLICY] - EPS
    assert chk["ok"] is expected
    if m[FLATTEST_POLICY] != 0:
        assert math.isclose(
            chk["actual_ratio"], m[STEEPEST_POLICY] / m[FLATTEST_POLICY], abs_tol=STAT_TOL
        )


def test_check_popt_min_saturates(data):
    pp = data["per_policy"]
    chk = data["checks"]["popt_min_saturates"]
    assert chk["ok"] is (pp["POPT"]["min"] <= POPT_MIN_CEILING + EPS)
    assert math.isclose(chk["popt_min"], pp["POPT"]["min"], abs_tol=STAT_TOL)


def test_verdict_ok_is_conjunction_of_checks(data):
    expected = all(chk["ok"] for chk in data["checks"].values())
    assert data["verdict_ok"] is expected
