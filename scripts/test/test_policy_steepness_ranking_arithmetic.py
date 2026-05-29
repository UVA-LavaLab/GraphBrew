"""Gate 122 — policy_steepness_ranking.json arithmetic + checks.

Locks the per-policy final-octave (4MB→8MB) absolute-steepness ranking
that is the headline ordering invariant for the saturation story:

    POPT <= GRASP <= LRU  AND  POPT < SRRIP

with explicit ceiling/floor thresholds on each band:
    oracle-aware ceiling = 0.5 pp/oct
    non-oracle floor      = 0.5 pp/oct
    oracle median must be < half non-oracle median
    POPT min slope must reach <= 0.2 (at least one app fully saturates).

The artifact pulls per-app final-octave slope magnitudes from
cache_saturation_onset.json (already locked by gate 118) and runs
seven explicit checks; this gate reproduces every per-policy stat
and every check from raw data, so any regression in the saturation
story surfaces immediately.

Invariants (13 tests, 4 groups):
- schema + meta (3): schema='policy_steepness_ranking/v1', source ends
  with cache_saturation_onset.json, oracle_aware=(POPT,GRASP),
  non_oracle=(LRU,SRRIP), thresholds match documented values.
- per_policy stats from source (3): n=5, per_app values are
  |final_octave_slope_pp| from cache_saturation_onset.per_app;
  min/median/mean/max recomputed from per_app values.
- aggregates + ranking (3): medians_pp == {pol: per_policy[pol].median};
  oracle_median_pp = median(medians[POPT], medians[GRASP]);
  non_oracle_median_pp = median(medians[LRU], medians[SRRIP]);
  ranking_by_median = sorted(policies, key=median asc).
- seven checks + verdict (4): each of the seven check.ok values matches
  the documented inequality with 1e-9 tolerance; verdict_ok = all
  check.ok values.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/policy_steepness_ranking.json")
SOURCE = Path("wiki/data/cache_saturation_onset.json")

ORACLE_AWARE = ("POPT", "GRASP")
NON_ORACLE = ("LRU", "SRRIP")
ALL_POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")

ORACLE_CEILING = 0.5
NON_ORACLE_FLOOR = 0.5
HALF_RATIO = 0.5
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


def test_meta_policy_bands(data):
    meta = data["meta"]
    assert tuple(meta["oracle_aware"]) == ORACLE_AWARE
    assert tuple(meta["non_oracle"]) == NON_ORACLE
    assert set(meta["policies"]) == set(ALL_POLICIES)


def test_meta_thresholds(data):
    t = data["meta"]["thresholds"]
    assert t["oracle_aware_ceiling_pp"] == ORACLE_CEILING
    assert t["non_oracle_floor_pp"] == NON_ORACLE_FLOOR
    assert t["oracle_aware_half_of_non_oracle"] == HALF_RATIO
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


def test_oracle_and_non_oracle_aggregate_medians(data):
    medians = data["medians_pp"]
    oracle_expected = statistics.median([medians[p] for p in ORACLE_AWARE])
    non_oracle_expected = statistics.median([medians[p] for p in NON_ORACLE])
    assert math.isclose(data["oracle_median_pp"], oracle_expected, abs_tol=STAT_TOL)
    assert math.isclose(data["non_oracle_median_pp"], non_oracle_expected, abs_tol=STAT_TOL)


def test_ranking_by_median_sorted_ascending(data):
    expected = sorted(ALL_POLICIES, key=lambda p: data["medians_pp"][p])
    assert data["ranking_by_median"] == expected, (
        f"ranking={data['ranking_by_median']} expected={expected}"
    )


# ── group 4: seven checks + verdict ──────────────────────────────────────


def test_check_popt_le_grasp_median(data):
    m = data["medians_pp"]
    chk = data["checks"]["popt_le_grasp_median"]
    assert chk["ok"] is (m["POPT"] <= m["GRASP"] + EPS)
    assert math.isclose(chk["popt"], m["POPT"], abs_tol=STAT_TOL)
    assert math.isclose(chk["grasp"], m["GRASP"], abs_tol=STAT_TOL)


def test_check_grasp_le_lru_median(data):
    m = data["medians_pp"]
    chk = data["checks"]["grasp_le_lru_median"]
    assert chk["ok"] is (m["GRASP"] <= m["LRU"] + EPS)


def test_check_popt_lt_srrip_median(data):
    m = data["medians_pp"]
    chk = data["checks"]["popt_lt_srrip_median"]
    assert chk["ok"] is (m["POPT"] < m["SRRIP"] - EPS)


def test_check_ceilings_floors_and_half_ratio(data):
    m = data["medians_pp"]
    pp = data["per_policy"]

    ceil_chk = data["checks"]["oracle_aware_ceiling"]
    expected = all(m[p] <= ORACLE_CEILING + EPS for p in ORACLE_AWARE)
    assert ceil_chk["ok"] is expected

    floor_chk = data["checks"]["non_oracle_floor"]
    expected = all(m[p] >= NON_ORACLE_FLOOR - EPS for p in NON_ORACLE)
    assert floor_chk["ok"] is expected

    half_chk = data["checks"]["oracle_half_of_non_oracle"]
    expected = data["oracle_median_pp"] < data["non_oracle_median_pp"] * HALF_RATIO + EPS
    assert half_chk["ok"] is expected
    if data["non_oracle_median_pp"] != 0:
        expected_ratio = data["oracle_median_pp"] / data["non_oracle_median_pp"]
        assert math.isclose(half_chk["actual_ratio"], expected_ratio, abs_tol=STAT_TOL)

    popt_chk = data["checks"]["popt_min_saturates"]
    expected = pp["POPT"]["min"] <= POPT_MIN_CEILING + EPS
    assert popt_chk["ok"] is expected
    assert math.isclose(popt_chk["popt_min"], pp["POPT"]["min"], abs_tol=STAT_TOL)


def test_verdict_ok_is_conjunction_of_checks(data):
    expected = all(chk["ok"] for chk in data["checks"].values())
    assert data["verdict_ok"] is expected
