"""Derivation parity gate for ``wiki/data/per_app_capacity_slope.json``.

Locks the per-app capacity-sensitivity slope report (gate 68)
against its single upstream — ``oracle_gap.json#rows`` — so any
silent drift in the OLS slope reducer, the (1MB/4MB/8MB) L3 axis
filter, the per-(app, policy) median aggregator, the median-of-
medians "cache-hungriness" ranking, the bfs deviation pin, or the
verdict predicates trips a test before the dashboard re-publishes
the "which kernel benefits most from cache scaling" story.

    oracle_gap.json#rows    (filtered to L3 in {1MB, 4MB, 8MB})
                  │
        per_app_capacity_slope.py:build()
                  │
                  ▼
    wiki/data/per_app_capacity_slope.json    ← gate target

The gated claim: per-(app, policy) median slope is strictly negative
on every cell (cache scaling never hurts on the medianed view), the
corpus contains at least one app whose every-policy median is below
−5 pp/oct (genuinely cache-sensitive kernels exist), and no new app
outside the (now-empty) pinned set shows GRASP > 1 pp/oct steeper than LRU
(the oracle-aware ordering is preserved).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "per_app_capacity_slope.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
HELP_FLOOR_PP_OCTAVE = -5.0
ALLOW_LRU_SHALLOWER_BY_PP = 1.0
PINNED_DEVIATING_APPS = ()


def _ols_slope(pts):
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


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_doc() -> dict:
    if not ORACLE_PATH.exists():
        pytest.skip(f"missing {ORACLE_PATH}")
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def reconstructed(oracle_doc) -> dict:
    """Mirror build() end-to-end against the same upstream rows."""
    rows = oracle_doc["rows"]
    cells = defaultdict(list)
    for r in rows:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        cells[(r["app"], r["graph"], r["policy"])].append(
            (L3_LOG2_MB[r["l3_size"]], float(r["miss_rate"]) * 100.0)
        )

    per_cell, slopes_by_app_pol, apps_seen = [], defaultdict(list), set()
    for (app, graph, pol), pts in sorted(cells.items()):
        slope = _ols_slope(pts)
        if slope is None:
            continue
        slope = round(slope, 4)
        per_cell.append({
            "app": app, "graph": graph, "policy": pol,
            "slope_pp": slope, "n_points": len(pts),
        })
        slopes_by_app_pol[(app, pol)].append(slope)
        apps_seen.add(app)

    apps = sorted(apps_seen)
    per_app = {}
    medians_of_medians = {}
    for app in apps:
        block = {}
        for pol in POLICIES:
            xs = slopes_by_app_pol.get((app, pol), [])
            if not xs:
                continue
            block[pol] = {
                "n_cells": len(xs),
                "median_pp": round(_median(xs), 4),
                "mean_pp": round(sum(xs) / len(xs), 4),
                "min_pp": round(min(xs), 4),
                "max_pp": round(max(xs), 4),
            }
        per_app[app] = block
        meds = [s["median_pp"] for s in block.values()]
        if meds:
            medians_of_medians[app] = round(_median(meds), 4)

    most_hungry = (
        min(medians_of_medians, key=lambda a: medians_of_medians[a])
        if medians_of_medians else None
    )
    least_hungry = (
        max(medians_of_medians, key=lambda a: medians_of_medians[a])
        if medians_of_medians else None
    )
    range_pp = (
        round(
            medians_of_medians[least_hungry] - medians_of_medians[most_hungry],
            4,
        ) if most_hungry and least_hungry else 0.0
    )

    deviating = []
    for app, block in per_app.items():
        if "GRASP" in block and "LRU" in block:
            if (block["LRU"]["median_pp"] - block["GRASP"]["median_pp"]
                    > ALLOW_LRU_SHALLOWER_BY_PP):
                deviating.append(app)
    new_dev = [a for a in deviating if a not in PINNED_DEVIATING_APPS]

    return {
        "per_cell": per_cell, "apps": apps, "per_app": per_app,
        "medians_of_medians": medians_of_medians,
        "most_hungry": most_hungry, "least_hungry": least_hungry,
        "range_pp": range_pp,
        "deviating_apps": deviating, "new_deviating_apps": new_dev,
    }


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_cell"}


def test_meta_carries_canonical_fields(artifact):
    expected = {
        "apps", "policies", "l3_axis", "help_floor_pp_octave",
        "allow_lru_shallower_by_pp", "per_app",
        "per_app_median_of_medians_pp",
        "most_cache_hungry_app", "least_cache_hungry_app",
        "per_app_median_range_pp",
        "invariant_all_negative",
        "deviating_apps", "pinned_deviating_apps", "new_deviating_apps",
        "invariant_no_new_deviating_apps",
        "invariant_at_least_one_cache_sensitive_app",
        "verdict", "verdict_invariant",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_pinned_constants(artifact):
    m = artifact["meta"]
    assert tuple(m["policies"]) == POLICIES
    assert tuple(m["l3_axis"]) == tuple(L3_LOG2_MB.keys())
    assert m["help_floor_pp_octave"] == HELP_FLOOR_PP_OCTAVE
    assert m["allow_lru_shallower_by_pp"] == ALLOW_LRU_SHALLOWER_BY_PP
    assert tuple(m["pinned_deviating_apps"]) == PINNED_DEVIATING_APPS


def test_verdict_invariant_string_pinned(artifact):
    expected = (
        "PASS iff (1) every (app, policy) median slope < 0, "
        "(2) no app outside the pinned set has GRASP more than "
        f"{ALLOW_LRU_SHALLOWER_BY_PP} pp/octave steeper than LRU, "
        "and (3) at least one app has every policy median below "
        f"{HELP_FLOOR_PP_OCTAVE} pp/octave."
    )
    assert artifact["meta"]["verdict_invariant"] == expected


def test_per_cell_entry_shape(artifact):
    expected = {"app", "graph", "policy", "slope_pp", "n_points"}
    for r in artifact["per_cell"]:
        missing = expected - set(r.keys())
        assert not missing, f"per_cell entry missing {missing}"


def test_per_app_block_shape(artifact):
    expected = {"n_cells", "median_pp", "mean_pp", "min_pp", "max_pp"}
    for app, block in artifact["meta"]["per_app"].items():
        for pol, s in block.items():
            assert pol in POLICIES, (
                f"per_app[{app}]: unknown policy {pol!r}"
            )
            missing = expected - set(s.keys())
            assert not missing, (
                f"per_app[{app}][{pol}] missing {missing}"
            )


# ----------------------------------------------------------------------
# Group B: per-cell cross-source parity
# ----------------------------------------------------------------------

def test_per_cell_count_matches_recomputation(artifact, reconstructed):
    assert len(artifact["per_cell"]) == len(reconstructed["per_cell"])


def test_per_cell_keyset_matches_recomputation(artifact, reconstructed):
    a = {(r["app"], r["graph"], r["policy"]) for r in artifact["per_cell"]}
    e = {(r["app"], r["graph"], r["policy"]) for r in reconstructed["per_cell"]}
    assert a == e


def test_per_cell_records_match_recomputation(artifact, reconstructed):
    expected = {
        (r["app"], r["graph"], r["policy"]): r
        for r in reconstructed["per_cell"]
    }
    for r in artifact["per_cell"]:
        key = (r["app"], r["graph"], r["policy"])
        e = expected[key]
        assert r["slope_pp"] == e["slope_pp"], (
            f"{key}: slope_pp drift — {r['slope_pp']!r} vs {e['slope_pp']!r}"
        )
        assert r["n_points"] == e["n_points"]


def test_n_points_only_three(artifact):
    """Every per-cell slope must come from exactly 3 L3 points
    (1MB, 4MB, 8MB) — the build filter restricts to L3_LOG2_MB."""
    for r in artifact["per_cell"]:
        assert r["n_points"] == 3, (
            f"{r}: n_points ≠ 3 — L3 axis filter drifted"
        )


def test_per_cell_policies_pinned(artifact):
    for r in artifact["per_cell"]:
        assert r["policy"] in POLICIES, (
            f"unknown policy {r['policy']!r} surfaced through reducer"
        )


# ----------------------------------------------------------------------
# Group C: per-(app, policy) reducer cross-source parity
# ----------------------------------------------------------------------

def test_apps_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["apps"] == reconstructed["apps"]


def test_per_app_keyset_matches_recomputation(artifact, reconstructed):
    assert set(artifact["meta"]["per_app"].keys()) == (
        set(reconstructed["per_app"].keys())
    )


def test_per_app_records_match_recomputation(artifact, reconstructed):
    expected = reconstructed["per_app"]
    for app, block in artifact["meta"]["per_app"].items():
        e_block = expected[app]
        assert set(block.keys()) == set(e_block.keys()), (
            f"per_app[{app}]: policy set drift — {set(block)} vs {set(e_block)}"
        )
        for pol, s in block.items():
            e_s = e_block[pol]
            for k in ("n_cells", "median_pp", "mean_pp", "min_pp", "max_pp"):
                assert s[k] == e_s[k], (
                    f"per_app[{app}][{pol}].{k} drift — "
                    f"{s[k]!r} vs {e_s[k]!r}"
                )


def test_per_app_n_cells_matches_per_cell_count(artifact):
    by_app_pol = defaultdict(int)
    for r in artifact["per_cell"]:
        by_app_pol[(r["app"], r["policy"])] += 1
    for app, block in artifact["meta"]["per_app"].items():
        for pol, s in block.items():
            assert s["n_cells"] == by_app_pol[(app, pol)], (
                f"per_app[{app}][{pol}].n_cells ≠ count of per_cell rows"
            )


def test_per_app_min_le_median_le_max(artifact):
    for app, block in artifact["meta"]["per_app"].items():
        for pol, s in block.items():
            assert s["min_pp"] <= s["median_pp"] <= s["max_pp"], (
                f"per_app[{app}][{pol}]: min ≤ median ≤ max ordering broken"
            )


# ----------------------------------------------------------------------
# Group D: cache-hungriness ranking + deviation pin parity
# ----------------------------------------------------------------------

def test_per_app_median_of_medians_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["per_app_median_of_medians_pp"] == (
        reconstructed["medians_of_medians"]
    )


def test_per_app_median_of_medians_formula(artifact):
    """For each app: median-of-medians = round(_median(policy medians), 4)."""
    for app, block in artifact["meta"]["per_app"].items():
        meds = [s["median_pp"] for s in block.values()]
        if not meds:
            continue
        expected = round(_median(meds), 4)
        assert artifact["meta"]["per_app_median_of_medians_pp"][app] == expected, (
            f"per_app_median_of_medians_pp[{app}] formula drift"
        )


def test_most_cache_hungry_app_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["most_cache_hungry_app"] == (
        reconstructed["most_hungry"]
    )


def test_least_cache_hungry_app_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["least_cache_hungry_app"] == (
        reconstructed["least_hungry"]
    )


def test_most_hungry_has_smallest_median(artifact):
    mom = artifact["meta"]["per_app_median_of_medians_pp"]
    if not mom:
        pytest.skip("no apps observed")
    assert artifact["meta"]["most_cache_hungry_app"] == (
        min(mom, key=lambda a: mom[a])
    )


def test_least_hungry_has_largest_median(artifact):
    mom = artifact["meta"]["per_app_median_of_medians_pp"]
    if not mom:
        pytest.skip("no apps observed")
    assert artifact["meta"]["least_cache_hungry_app"] == (
        max(mom, key=lambda a: mom[a])
    )


def test_per_app_median_range_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["per_app_median_range_pp"] == reconstructed["range_pp"]


def test_per_app_median_range_formula(artifact):
    m = artifact["meta"]
    mom = m["per_app_median_of_medians_pp"]
    most, least = m["most_cache_hungry_app"], m["least_cache_hungry_app"]
    if most and least:
        expected = round(mom[least] - mom[most], 4)
    else:
        expected = 0.0
    assert m["per_app_median_range_pp"] == expected


def test_deviating_apps_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["deviating_apps"] == reconstructed["deviating_apps"]


def test_new_deviating_apps_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["new_deviating_apps"] == (
        reconstructed["new_deviating_apps"]
    )


def test_new_deviating_apps_excludes_pinned(artifact):
    pinned = set(artifact["meta"]["pinned_deviating_apps"])
    for app in artifact["meta"]["new_deviating_apps"]:
        assert app not in pinned, (
            f"{app!r} listed as new_deviating_app but is in pinned set"
        )


# ----------------------------------------------------------------------
# Group E: invariants + verdict parity
# ----------------------------------------------------------------------

def test_invariant_all_negative_matches_recomputation(artifact):
    expected = all(
        s["median_pp"] < 0.0
        for block in artifact["meta"]["per_app"].values()
        for s in block.values()
    )
    assert artifact["meta"]["invariant_all_negative"] == expected


def test_invariant_no_new_deviating_apps_matches_recomputation(artifact):
    expected = (not artifact["meta"]["new_deviating_apps"])
    assert artifact["meta"]["invariant_no_new_deviating_apps"] == expected


def test_invariant_at_least_one_cache_sensitive_app_matches_recomputation(artifact):
    expected = any(
        all(s["median_pp"] < HELP_FLOOR_PP_OCTAVE for s in block.values())
        for block in artifact["meta"]["per_app"].values()
    )
    assert (
        artifact["meta"]["invariant_at_least_one_cache_sensitive_app"]
        == expected
    )


def test_verdict_matches_and_of_invariants(artifact):
    m = artifact["meta"]
    expected = "PASS" if (
        m["invariant_all_negative"]
        and m["invariant_no_new_deviating_apps"]
        and m["invariant_at_least_one_cache_sensitive_app"]
    ) else "FAIL"
    assert m["verdict"] == expected


def test_current_verdict_is_pass(artifact):
    assert artifact["meta"]["verdict"] == "PASS", (
        "per_app_capacity_slope regressed to FAIL — the corpus has "
        "lost either uniform-negative per-(app, policy) median slope, "
        "the bfs-only deviation invariant, or the existence of at "
        "least one cache-sensitive kernel (every-policy median < -5 "
        "pp/octave)."
    )
