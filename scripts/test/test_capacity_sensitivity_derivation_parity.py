"""Gate 185 — capacity_sensitivity derivation parity.

Reconstruct ``wiki/data/capacity_sensitivity.json`` from scratch by walking
``wiki/data/oracle_gap.json#rows`` and re-deriving every per-cell OLS slope
and per-policy summary stat. Pin the OLS math (closed form), the bespoke
median/percentile (which differ from numpy.percentile), the steepest /
shallowest argmin/argmax tie-break, and the three-clause verdict against
the published artifact.

Load-bearing rules being locked:

* L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0} — rows with any other
  l3_size are dropped silently (NOT counted toward n_points).
* miss_rate is multiplied by 100 to convert to PERCENTAGE POINTS — slope
  output unit is pp / log2(MB) (= pp/octave on the L3 axis).
* OLS slope = (n·Σxy − Σx·Σy) / (n·Σxx − (Σx)²); returns None if n < 2
  OR denominator is 0 (collinear-x — won't happen with our L3 axis).
* Per-cell iteration order: sorted(cells.items()) where key is
  (app, graph, policy) tuple — alphabetical/ASCII order is load-bearing
  for per_cell list ordering.
* slope_pp rounded to 4dp.
* per_cell entries have field set {app, graph, policy, slope_pp, n_points}.
* policy_summary built in POLICIES order: ("GRASP", "LRU", "POPT", "SRRIP")
  — alphabetical, NOT canonical (LRU, SRRIP, GRASP, POPT). Load-bearing
  for any dict-iteration order assumption.
* _median: standard pair-average for even n; for odd n uses s[n // 2].
* _pct: bespoke formula `s[max(0, min(n-1, int(round(p · (n-1)))))]` — this
  is NOT numpy.percentile (which uses linear interpolation).
* p10_pp uses p=0.10; p90_pp uses p=0.90.
* All summary stats rounded to 4dp.
* steepest_policy = min over medians (most-negative); shallowest_policy =
  max (least-negative); both use dict.items + key=lambda lookup so ties
  break by Python's min/max tie-break = first encountered = POLICIES order.
* median_steepness_gap_pp = round(|steepest_med| - |shallowest_med|, 4).
* invariant_all_help: every policy's median_pp < -5.0 (STRICT <).
* invariant_lru_steepest: steepest == "LRU" exactly.
* invariant_grasp_shallower_lru: medians["GRASP"] > medians["LRU"]
  (STRICT >) AND both present.
* verdict = PASS iff all three invariants True else FAIL.

The whole gate runs offline against committed JSON.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
ARTIFACT = WIKI_DATA / "capacity_sensitivity.json"
ORACLE = WIKI_DATA / "oracle_gap.json"

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
HELP_FLOOR = -5.0


# ---------------------------------------------------------------------------
# Reference rebuilders
# ---------------------------------------------------------------------------


def _ols_slope(pts: list[tuple[float, float]]) -> float | None:
    n = len(pts)
    if n < 2:
        return None
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    return (n * sxy - sx * sy) / den


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    k = max(0, min(n - 1, int(round(p * (n - 1)))))
    return s[k]


def _rederive_per_cell(rows: list[dict]) -> list[dict]:
    cells: dict = defaultdict(list)
    for r in rows:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        cells[(r["app"], r["graph"], r["policy"])].append(
            (L3_LOG2_MB[r["l3_size"]], float(r["miss_rate"]) * 100.0)
        )
    out = []
    for (app, graph, pol), pts in sorted(cells.items()):
        slope = _ols_slope(pts)
        if slope is None:
            continue
        out.append({
            "app": app,
            "graph": graph,
            "policy": pol,
            "slope_pp": round(slope, 4),
            "n_points": len(pts),
        })
    return out


def _rederive_summary(per_cell: list[dict]) -> dict:
    slopes_by_pol: dict[str, list] = defaultdict(list)
    for e in per_cell:
        slopes_by_pol[e["policy"]].append(e["slope_pp"])
    summary: dict[str, dict] = {}
    medians: dict[str, float] = {}
    for pol in POLICIES:
        xs = slopes_by_pol.get(pol, [])
        if not xs:
            continue
        med = round(_median(xs), 4)
        summary[pol] = {
            "n_cells": len(xs),
            "median_pp": med,
            "mean_pp": round(sum(xs) / len(xs), 4),
            "p10_pp": round(_pct(xs, 0.10), 4),
            "p90_pp": round(_pct(xs, 0.90), 4),
            "min_pp": round(min(xs), 4),
            "max_pp": round(max(xs), 4),
        }
        medians[pol] = med
    return {"summary": summary, "medians": medians}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oracle_rows() -> list[dict]:
    return json.loads(ORACLE.read_text())["rows"]


@pytest.fixture(scope="module")
def expected_per_cell(oracle_rows) -> list[dict]:
    return _rederive_per_cell(oracle_rows)


@pytest.fixture(scope="module")
def expected_summary(expected_per_cell) -> dict:
    return _rederive_summary(expected_per_cell)


# ---------------------------------------------------------------------------
# Group 1 — Schema
# ---------------------------------------------------------------------------


def test_top_keys(published):
    assert set(published.keys()) == {"meta", "per_cell"}


def test_meta_field_set(published):
    assert set(published["meta"].keys()) == {
        "cell_count",
        "help_floor_pp_octave",
        "invariant_all_help",
        "invariant_grasp_shallower_lru",
        "invariant_lru_steepest",
        "l3_axis",
        "median_steepness_gap_pp",
        "policies",
        "policy_summary",
        "shallowest_policy",
        "steepest_policy",
        "verdict",
        "verdict_invariant",
    }


def test_meta_constants(published):
    m = published["meta"]
    assert m["policies"] == list(POLICIES)
    assert m["l3_axis"] == list(L3_LOG2_MB.keys())
    assert m["help_floor_pp_octave"] == HELP_FLOOR


def test_per_cell_entry_field_set(published):
    expected = {"app", "graph", "policy", "slope_pp", "n_points"}
    for entry in published["per_cell"]:
        assert set(entry.keys()) == expected, entry


def test_policy_summary_field_set(published):
    expected = {
        "n_cells",
        "median_pp",
        "mean_pp",
        "p10_pp",
        "p90_pp",
        "min_pp",
        "max_pp",
    }
    for pol, s in published["meta"]["policy_summary"].items():
        assert set(s.keys()) == expected, pol


# ---------------------------------------------------------------------------
# Group 2 — Per-cell OLS reconstruction
# ---------------------------------------------------------------------------


def test_per_cell_full_rederive(published, expected_per_cell):
    assert published["per_cell"] == expected_per_cell


def test_per_cell_count_matches_meta(published):
    assert published["meta"]["cell_count"] == len(published["per_cell"])


def test_per_cell_sorted_by_app_graph_policy(published):
    keys = [(e["app"], e["graph"], e["policy"]) for e in published["per_cell"]]
    assert keys == sorted(keys)


def test_slope_pp_rounded_4dp(published):
    for entry in published["per_cell"]:
        assert entry["slope_pp"] == round(entry["slope_pp"], 4), entry


def test_n_points_at_least_2(published):
    for entry in published["per_cell"]:
        assert entry["n_points"] >= 2, entry


def test_n_points_within_paper_l3_axis(published):
    for entry in published["per_cell"]:
        assert entry["n_points"] <= len(L3_LOG2_MB), entry


def test_per_cell_only_uses_paper_l3_rows(published, oracle_rows):
    # No (app, graph, policy) in per_cell should have all its oracle rows
    # outside the paper L3 axis.
    seen = {(e["app"], e["graph"], e["policy"]) for e in published["per_cell"]}
    for key in seen:
        app, graph, pol = key
        rows = [
            r for r in oracle_rows
            if r["app"] == app and r["graph"] == graph and r["policy"] == pol
            and r["l3_size"] in L3_LOG2_MB
        ]
        assert len(rows) >= 2, key


def test_slope_sign_matches_monotonicity(published, oracle_rows):
    # If miss rate strictly decreases as L3 grows, slope MUST be negative.
    # We assert this on cells with n_points == 3 (unambiguous monotone).
    by_key: dict = defaultdict(dict)
    for r in oracle_rows:
        if r["l3_size"] in L3_LOG2_MB:
            by_key[(r["app"], r["graph"], r["policy"])][r["l3_size"]] = float(
                r["miss_rate"]
            )
    for entry in published["per_cell"]:
        if entry["n_points"] != 3:
            continue
        mr = by_key[(entry["app"], entry["graph"], entry["policy"])]
        vals = [mr["1MB"], mr["4MB"], mr["8MB"]]
        if vals[0] > vals[1] > vals[2]:
            assert entry["slope_pp"] <= 0, (entry, vals)


# ---------------------------------------------------------------------------
# Group 3 — Per-policy summary
# ---------------------------------------------------------------------------


def test_policy_summary_full_rederive(published, expected_summary):
    assert published["meta"]["policy_summary"] == expected_summary["summary"]


def test_policy_summary_keys_subset_of_policies(published):
    assert set(published["meta"]["policy_summary"].keys()).issubset(set(POLICIES))


def test_summary_n_cells_sum_matches_per_cell_count(published):
    total = sum(s["n_cells"] for s in published["meta"]["policy_summary"].values())
    assert total == len(published["per_cell"])


def test_summary_min_le_p10_le_median_le_p90_le_max(published):
    for pol, s in published["meta"]["policy_summary"].items():
        # bespoke percentile may equal endpoints for small n; allow LE chain
        assert s["min_pp"] <= s["p10_pp"] <= s["median_pp"] <= s["p90_pp"] <= s["max_pp"], (
            pol,
            s,
        )


def test_summary_stats_rounded_4dp(published):
    for pol, s in published["meta"]["policy_summary"].items():
        for k in ("median_pp", "mean_pp", "p10_pp", "p90_pp", "min_pp", "max_pp"):
            assert s[k] == round(s[k], 4), (pol, k, s)


def test_bespoke_percentile_matches_index_formula(published, expected_per_cell):
    by_pol: dict = defaultdict(list)
    for e in expected_per_cell:
        by_pol[e["policy"]].append(e["slope_pp"])
    for pol, xs in by_pol.items():
        s = sorted(xs)
        n = len(s)
        for p, key in ((0.10, "p10_pp"), (0.90, "p90_pp")):
            k = max(0, min(n - 1, int(round(p * (n - 1)))))
            assert published["meta"]["policy_summary"][pol][key] == round(s[k], 4), (
                pol,
                key,
            )


# ---------------------------------------------------------------------------
# Group 4 — Steepest/shallowest + gap
# ---------------------------------------------------------------------------


def test_steepest_is_argmin_of_medians(published, expected_summary):
    medians = expected_summary["medians"]
    steepest = min(medians, key=lambda p: medians[p])
    assert published["meta"]["steepest_policy"] == steepest


def test_shallowest_is_argmax_of_medians(published, expected_summary):
    medians = expected_summary["medians"]
    shallowest = max(medians, key=lambda p: medians[p])
    assert published["meta"]["shallowest_policy"] == shallowest


def test_steepness_gap_formula(published, expected_summary):
    medians = expected_summary["medians"]
    st = published["meta"]["steepest_policy"]
    sh = published["meta"]["shallowest_policy"]
    expected = round(abs(medians[st]) - abs(medians[sh]), 4)
    assert published["meta"]["median_steepness_gap_pp"] == expected


def test_steepness_gap_non_negative(published):
    assert published["meta"]["median_steepness_gap_pp"] >= 0


def test_steepest_and_shallowest_distinct_when_multiple_policies(published):
    if len(published["meta"]["policy_summary"]) > 1:
        assert (
            published["meta"]["steepest_policy"]
            != published["meta"]["shallowest_policy"]
        )


# ---------------------------------------------------------------------------
# Group 5 — Three-clause verdict
# ---------------------------------------------------------------------------


def test_invariant_all_help_strict_less_than_floor(published):
    expected = all(
        s["median_pp"] < HELP_FLOOR
        for s in published["meta"]["policy_summary"].values()
    )
    assert published["meta"]["invariant_all_help"] == expected


def test_invariant_lru_steepest_exact_string(published):
    assert published["meta"]["invariant_lru_steepest"] == (
        published["meta"]["steepest_policy"] == "LRU"
    )


def test_invariant_grasp_shallower_lru_strict(published):
    summary = published["meta"]["policy_summary"]
    if "GRASP" in summary and "LRU" in summary:
        expected = summary["GRASP"]["median_pp"] > summary["LRU"]["median_pp"]
    else:
        expected = False
    assert published["meta"]["invariant_grasp_shallower_lru"] == expected


def test_verdict_is_pass_iff_all_invariants_true(published):
    m = published["meta"]
    all_true = (
        m["invariant_all_help"]
        and m["invariant_lru_steepest"]
        and m["invariant_grasp_shallower_lru"]
    )
    assert m["verdict"] == ("PASS" if all_true else "FAIL")


def test_verdict_invariant_text_carries_help_floor(published):
    assert str(HELP_FLOOR) in published["meta"]["verdict_invariant"]


def test_current_artifact_passes_all_three_invariants(published):
    # Sanity: corpus today actually passes — fail loud if a future regen
    # silently violates one of the structural sanity rules.
    m = published["meta"]
    assert m["verdict"] == "PASS"
    assert m["invariant_all_help"]
    assert m["invariant_lru_steepest"]
    assert m["invariant_grasp_shallower_lru"]
