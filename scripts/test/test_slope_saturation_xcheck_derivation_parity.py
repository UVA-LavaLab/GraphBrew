"""Derivation parity gate for ``wiki/data/slope_saturation_xcheck.json``.

Locks the saturation-distance × capacity-slope cross-check (gate 69)
against its single upstream — oracle_gap.json#rows — so any silent
drift in the OLS slope reducer, the bespoke Pearson/Spearman
helpers, the per-cell flat-cell partitioner, or the verdict
predicates trips a test before the dashboard re-publishes the
'distance and slope are two cuts of the same signal' story.

    oracle_gap.json#rows
                  │
       slope_saturation_xcheck.py:build()
                  │
                  ▼
    wiki/data/slope_saturation_xcheck.json   ← gate target

The gated claim: at the paper's L3 axis {1MB, 4MB, 8MB}, the per-
cell upper-octave drop (distance_pp = mr(4MB) − mr(8MB)) and the
full-axis OLS slope (pp/octave) move together with moderate
positive correlation (Pearson ≥ 0.40, Spearman ≥ 0.35), on the same
per-octave scale (median ratio in [0.70, 1.30]). If either signal's
reducer breaks (e.g. slope sign flips, distance scaling drifts),
the correlation or ratio band fails and the gate trips.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "slope_saturation_xcheck.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"

# Pinned mirror of generator constants.
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
MIN_MATCHED_CELLS = 80
PEARSON_FLOOR = 0.40
SPEARMAN_FLOOR = 0.35
RATIO_MIN = 0.70
RATIO_MAX = 1.30
SLOPE_EPSILON = 0.05


# ----------------------------------------------------------------------
# Mirror helpers (verbatim from generator)
# ----------------------------------------------------------------------

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


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _ranks(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


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
def reconstructed(oracle_doc):
    """Mirror ``build()`` end-to-end against the same upstream."""
    rows = oracle_doc["rows"]
    by: dict = defaultdict(dict)
    for r in rows:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        by[(r["app"], r["graph"], r["policy"])][r["l3_size"]] = (
            float(r["miss_rate"]) * 100.0
        )
    matched, flat = [], []
    for key, vals in sorted(by.items()):
        if not all(l in vals for l in L3_SIZES):
            continue
        pts = [(L3_LOG2_MB[l], vals[l]) for l in L3_SIZES]
        slope = _ols_slope(pts)
        if slope is None:
            continue
        slope = round(slope, 4)
        abs_slope = abs(slope)
        distance = round(vals["4MB"] - vals["8MB"], 4)
        ratio = (distance / abs_slope) if abs_slope > 0 else None
        rec = {
            "app": key[0], "graph": key[1], "policy": key[2],
            "distance_pp": distance, "slope_pp": slope,
            "abs_slope_pp": round(abs_slope, 4),
            "ratio_dist_slope": round(ratio, 4) if ratio is not None else None,
        }
        if abs_slope < SLOPE_EPSILON:
            flat.append(rec)
        else:
            matched.append(rec)
    return {"matched": matched, "flat": flat}


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_cell", "flat_cells"}


def test_meta_carries_canonical_fields(artifact):
    expected = {
        "cells_matched", "cells_flat_excluded", "slope_epsilon_pp",
        "pearson_r", "spearman_rho",
        "median_distance_pp", "median_abs_slope_pp",
        "median_ratio_distance_to_slope",
        "min_matched_cells", "pearson_floor", "spearman_floor",
        "ratio_band",
        "invariant_match_count", "invariant_pearson_floor",
        "invariant_spearman_floor", "invariant_ratio_band",
        "verdict", "verdict_invariant",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_thresholds_pinned(artifact):
    m = artifact["meta"]
    assert m["min_matched_cells"] == MIN_MATCHED_CELLS
    assert m["pearson_floor"] == PEARSON_FLOOR
    assert m["spearman_floor"] == SPEARMAN_FLOOR
    assert m["ratio_band"] == [RATIO_MIN, RATIO_MAX]
    assert m["slope_epsilon_pp"] == SLOPE_EPSILON


def test_verdict_invariant_string_pinned(artifact):
    expected = (
        f"PASS iff (1) >= {MIN_MATCHED_CELLS} non-flat per-(app, "
        f"graph, policy) cells matched, (2) Pearson r >= "
        f"{PEARSON_FLOOR}, (3) Spearman rho >= {SPEARMAN_FLOOR}, "
        f"(4) median (distance_pp / |slope_pp|) in "
        f"[{RATIO_MIN}, {RATIO_MAX}]. Cells with |slope_pp| < "
        f"{SLOPE_EPSILON} are reported as flat_cells and excluded."
    )
    assert artifact["meta"]["verdict_invariant"] == expected


def test_per_cell_entry_shape(artifact):
    expected = {
        "app", "graph", "policy", "distance_pp",
        "slope_pp", "abs_slope_pp", "ratio_dist_slope",
    }
    for r in artifact["per_cell"]:
        missing = expected - set(r.keys())
        assert not missing, f"per_cell entry missing fields: {missing}"


def test_flat_cells_entry_shape(artifact):
    expected = {
        "app", "graph", "policy", "distance_pp",
        "slope_pp", "abs_slope_pp", "ratio_dist_slope",
    }
    for r in artifact["flat_cells"]:
        missing = expected - set(r.keys())
        assert not missing, f"flat_cells entry missing fields: {missing}"


# ----------------------------------------------------------------------
# Group B: cell counts cross-source parity
# ----------------------------------------------------------------------

def test_cells_matched_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["cells_matched"] == len(reconstructed["matched"])


def test_cells_flat_excluded_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["cells_flat_excluded"] == len(reconstructed["flat"])


def test_per_cell_list_size_matches_meta(artifact):
    assert len(artifact["per_cell"]) == artifact["meta"]["cells_matched"]


def test_flat_cells_list_size_matches_meta(artifact):
    assert len(artifact["flat_cells"]) == artifact["meta"]["cells_flat_excluded"]


def test_per_cell_and_flat_cells_disjoint(artifact):
    matched_keys = {
        (r["app"], r["graph"], r["policy"]) for r in artifact["per_cell"]
    }
    flat_keys = {
        (r["app"], r["graph"], r["policy"]) for r in artifact["flat_cells"]
    }
    assert matched_keys.isdisjoint(flat_keys), (
        "per_cell and flat_cells share at least one (app, graph, policy) "
        "key — the |slope| < SLOPE_EPSILON partition is non-exclusive"
    )


def test_policies_present_only_known(artifact):
    for r in artifact["per_cell"] + artifact["flat_cells"]:
        assert r["policy"] in POLICIES + ("OPT", "POPT"), (
            f"unknown policy {r['policy']!r} surfaced through reducer"
        )


# ----------------------------------------------------------------------
# Group C: per-cell reducer cross-source parity
# ----------------------------------------------------------------------

def test_per_cell_distance_and_slope_match_recomputation(artifact, reconstructed):
    """Each artifact record matches a mirror record byte-exact."""
    expected = {
        (r["app"], r["graph"], r["policy"]): r
        for r in reconstructed["matched"]
    }
    for r in artifact["per_cell"]:
        key = (r["app"], r["graph"], r["policy"])
        assert key in expected, f"unexpected matched cell {key}"
        e = expected[key]
        assert r["distance_pp"] == e["distance_pp"], (
            f"{key}: distance_pp drift — {r['distance_pp']!r} vs {e['distance_pp']!r}"
        )
        assert r["slope_pp"] == e["slope_pp"], (
            f"{key}: slope_pp drift — {r['slope_pp']!r} vs {e['slope_pp']!r}"
        )
        assert r["abs_slope_pp"] == e["abs_slope_pp"]
        assert r["ratio_dist_slope"] == e["ratio_dist_slope"], (
            f"{key}: ratio_dist_slope drift — "
            f"{r['ratio_dist_slope']!r} vs {e['ratio_dist_slope']!r}"
        )


def test_flat_cells_match_recomputation(artifact, reconstructed):
    expected = {
        (r["app"], r["graph"], r["policy"]): r
        for r in reconstructed["flat"]
    }
    for r in artifact["flat_cells"]:
        key = (r["app"], r["graph"], r["policy"])
        assert key in expected, f"unexpected flat cell {key}"
        e = expected[key]
        assert r["abs_slope_pp"] == e["abs_slope_pp"]
        assert r["slope_pp"] == e["slope_pp"]
        assert r["distance_pp"] == e["distance_pp"]


def test_abs_slope_equals_abs_of_slope(artifact):
    for r in artifact["per_cell"] + artifact["flat_cells"]:
        assert r["abs_slope_pp"] == round(abs(r["slope_pp"]), 4), (
            f"{r}: abs_slope_pp ≠ round(abs(slope_pp), 4)"
        )


def test_flat_cells_partitioned_by_epsilon(artifact):
    """Every flat cell satisfies abs_slope < SLOPE_EPSILON."""
    for r in artifact["flat_cells"]:
        assert r["abs_slope_pp"] < SLOPE_EPSILON, (
            f"flat cell {r}: abs_slope={r['abs_slope_pp']} not below "
            f"{SLOPE_EPSILON}"
        )


def test_matched_cells_above_epsilon(artifact):
    for r in artifact["per_cell"]:
        assert r["abs_slope_pp"] >= SLOPE_EPSILON, (
            f"matched cell {r}: abs_slope={r['abs_slope_pp']} below "
            f"{SLOPE_EPSILON} should be in flat_cells"
        )


# ----------------------------------------------------------------------
# Group D: aggregate stats + verdict cross-source parity
# ----------------------------------------------------------------------

def test_pearson_r_matches_recomputation(artifact, reconstructed):
    xs = [r["distance_pp"] for r in reconstructed["matched"]]
    ys = [r["abs_slope_pp"] for r in reconstructed["matched"]]
    expected = round(_pearson(xs, ys), 4) if reconstructed["matched"] else 0.0
    assert artifact["meta"]["pearson_r"] == expected, (
        f"pearson_r drift — recomputed {expected!r}, "
        f"got {artifact['meta']['pearson_r']!r}"
    )


def test_spearman_rho_matches_recomputation(artifact, reconstructed):
    xs = [r["distance_pp"] for r in reconstructed["matched"]]
    ys = [r["abs_slope_pp"] for r in reconstructed["matched"]]
    expected = round(_spearman(xs, ys), 4) if reconstructed["matched"] else 0.0
    assert artifact["meta"]["spearman_rho"] == expected


def test_median_distance_pp_matches_recomputation(artifact, reconstructed):
    xs = [r["distance_pp"] for r in reconstructed["matched"]]
    expected = round(_median(xs), 4) if xs else 0.0
    assert artifact["meta"]["median_distance_pp"] == expected


def test_median_abs_slope_pp_matches_recomputation(artifact, reconstructed):
    xs = [r["abs_slope_pp"] for r in reconstructed["matched"]]
    expected = round(_median(xs), 4) if xs else 0.0
    assert artifact["meta"]["median_abs_slope_pp"] == expected


def test_median_ratio_distance_to_slope_matches_recomputation(artifact, reconstructed):
    xs = [
        r["ratio_dist_slope"] for r in reconstructed["matched"]
        if r["ratio_dist_slope"] is not None
    ]
    expected = round(_median(xs), 4) if xs else 0.0
    assert artifact["meta"]["median_ratio_distance_to_slope"] == expected


def test_invariant_match_count_matches_recomputation(artifact):
    expected = artifact["meta"]["cells_matched"] >= MIN_MATCHED_CELLS
    assert artifact["meta"]["invariant_match_count"] == expected


def test_invariant_pearson_floor_matches_recomputation(artifact):
    expected = artifact["meta"]["pearson_r"] >= PEARSON_FLOOR
    assert artifact["meta"]["invariant_pearson_floor"] == expected


def test_invariant_spearman_floor_matches_recomputation(artifact):
    expected = artifact["meta"]["spearman_rho"] >= SPEARMAN_FLOOR
    assert artifact["meta"]["invariant_spearman_floor"] == expected


def test_invariant_ratio_band_matches_recomputation(artifact):
    r = artifact["meta"]["median_ratio_distance_to_slope"]
    expected = RATIO_MIN <= r <= RATIO_MAX
    assert artifact["meta"]["invariant_ratio_band"] == expected


def test_verdict_matches_and_of_invariants(artifact):
    m = artifact["meta"]
    expected = "PASS" if (
        m["invariant_match_count"] and m["invariant_pearson_floor"]
        and m["invariant_spearman_floor"] and m["invariant_ratio_band"]
    ) else "FAIL"
    assert m["verdict"] == expected


def test_current_verdict_is_pass(artifact):
    assert artifact["meta"]["verdict"] == "PASS", (
        "slope_saturation_xcheck regressed to FAIL — distance_pp and "
        "|slope_pp| are two cuts of the same per-octave miss-rate "
        "signal, and breaking their correlation or rescaling either "
        "side without rescaling the other indicates a reducer bug."
    )
