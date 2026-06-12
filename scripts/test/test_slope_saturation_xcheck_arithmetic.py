"""Gate 130 — arithmetic + statistics audit of `slope_saturation_xcheck.json`.

Independently reproduces every per-cell distance / slope / ratio value
and the meta-level Pearson r, Spearman rho, medians, and verdict logic
from the raw `oracle_gap.json` rows. Asserts the published artifact
matches our independent computation to a tight numeric tolerance.

The artifact is the cross-check that links two complementary capacity-
sensitivity metrics (saturation distance = miss_rate(4MB) - miss_rate(8MB)
in pp, and OLS slope of miss_rate vs log2(MB) across 1MB/4MB/8MB) on the
same miss curve. A silent drift in either generator (different averaging
window, different sign convention, swapped scale) would invalidate every
downstream cliff-rank / cache-hungry narrative the paper rests on.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
XCHECK_PATH = REPO_ROOT / "wiki" / "data" / "slope_saturation_xcheck.json"

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}

SLOPE_EPSILON = 0.05
MIN_MATCHED_CELLS = 80
PEARSON_FLOOR = 0.40
SPEARMAN_FLOOR = 0.35
RATIO_MIN, RATIO_MAX = 0.70, 1.30

TOL = 5e-4  # generator rounds to 4 dp


def _ols_slope(pts):
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def _ranks(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[idx[k]] = avg
        i = j + 1
    return out


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(XCHECK_PATH.read_text())


@pytest.fixture(scope="module")
def expected() -> dict:
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    by: dict = defaultdict(dict)
    for r in rows:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        by[(r["app"], r["graph"], r["policy"])][r["l3_size"]] = float(r["miss_rate"]) * 100.0

    matched, flat = [], []
    for key, vals in sorted(by.items()):
        if not all(l in vals for l in L3_SIZES):
            continue
        pts = [(L3_LOG2_MB[l], vals[l]) for l in L3_SIZES]
        slope = round(_ols_slope(pts), 4)
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


def _key(r):
    return (r["app"], r["graph"], r["policy"])


# ---------- Group 1: meta scope + constants ----------

def test_meta_slope_epsilon_is_locked(artifact):
    assert artifact["meta"]["slope_epsilon_pp"] == SLOPE_EPSILON


def test_meta_thresholds_locked(artifact):
    m = artifact["meta"]
    assert m["min_matched_cells"] == MIN_MATCHED_CELLS
    assert m["pearson_floor"] == PEARSON_FLOOR
    assert m["spearman_floor"] == SPEARMAN_FLOOR
    assert m["ratio_band"] == [RATIO_MIN, RATIO_MAX]


def test_meta_cell_counts_match_lists(artifact):
    assert artifact["meta"]["cells_matched"] == len(artifact["per_cell"])
    assert artifact["meta"]["cells_flat_excluded"] == len(artifact["flat_cells"])


# ---------- Group 2: per-cell arithmetic reproduction ----------

def test_per_cell_count_matches_expected(artifact, expected):
    assert len(artifact["per_cell"]) == len(expected["matched"])


def test_flat_cells_count_matches_expected(artifact, expected):
    assert len(artifact["flat_cells"]) == len(expected["flat"])


def test_per_cell_distance_pp_reproduces(artifact, expected):
    exp_by_key = {_key(r): r for r in expected["matched"]}
    for r in artifact["per_cell"]:
        e = exp_by_key[_key(r)]
        assert abs(r["distance_pp"] - e["distance_pp"]) < TOL, _key(r)


def test_per_cell_slope_pp_reproduces(artifact, expected):
    exp_by_key = {_key(r): r for r in expected["matched"]}
    for r in artifact["per_cell"]:
        e = exp_by_key[_key(r)]
        assert abs(r["slope_pp"] - e["slope_pp"]) < TOL, _key(r)


def test_per_cell_abs_slope_consistency(artifact):
    for r in artifact["per_cell"]:
        assert abs(r["abs_slope_pp"] - abs(r["slope_pp"])) < TOL, _key(r)
        assert r["abs_slope_pp"] >= SLOPE_EPSILON - TOL, _key(r)


def test_per_cell_ratio_reproduces(artifact, expected):
    exp_by_key = {_key(r): r for r in expected["matched"]}
    for r in artifact["per_cell"]:
        e = exp_by_key[_key(r)]
        if r["ratio_dist_slope"] is None:
            assert e["ratio_dist_slope"] is None, _key(r)
            continue
        assert abs(r["ratio_dist_slope"] - e["ratio_dist_slope"]) < TOL, _key(r)


def test_flat_cells_all_below_epsilon(artifact):
    for r in artifact["flat_cells"]:
        assert r["abs_slope_pp"] < SLOPE_EPSILON, _key(r)


# ---------- Group 3: meta statistics reproduction ----------

def test_meta_pearson_r_reproduces(artifact):
    distances = [r["distance_pp"] for r in artifact["per_cell"]]
    abs_slopes = [r["abs_slope_pp"] for r in artifact["per_cell"]]
    want = round(_pearson(distances, abs_slopes), 4) if artifact["per_cell"] else 0.0
    assert abs(artifact["meta"]["pearson_r"] - want) < TOL


def test_meta_spearman_rho_reproduces(artifact):
    distances = [r["distance_pp"] for r in artifact["per_cell"]]
    abs_slopes = [r["abs_slope_pp"] for r in artifact["per_cell"]]
    want = round(_spearman(distances, abs_slopes), 4) if artifact["per_cell"] else 0.0
    assert abs(artifact["meta"]["spearman_rho"] - want) < TOL


def test_meta_medians_reproduce(artifact):
    distances = [r["distance_pp"] for r in artifact["per_cell"]]
    abs_slopes = [r["abs_slope_pp"] for r in artifact["per_cell"]]
    ratios = [r["ratio_dist_slope"] for r in artifact["per_cell"]
              if r["ratio_dist_slope"] is not None]
    assert abs(artifact["meta"]["median_distance_pp"] - round(_median(distances), 4)) < TOL
    assert abs(artifact["meta"]["median_abs_slope_pp"] - round(_median(abs_slopes), 4)) < TOL
    assert abs(artifact["meta"]["median_ratio_distance_to_slope"]
               - round(_median(ratios), 4)) < TOL


# ---------- Group 4: verdict invariants ----------

def test_invariant_match_count(artifact):
    m = artifact["meta"]
    assert m["invariant_match_count"] == (m["cells_matched"] >= MIN_MATCHED_CELLS)


def test_invariant_pearson_floor(artifact):
    m = artifact["meta"]
    assert m["invariant_pearson_floor"] == (m["pearson_r"] >= PEARSON_FLOOR)


def test_invariant_spearman_floor(artifact):
    m = artifact["meta"]
    assert m["invariant_spearman_floor"] == (m["spearman_rho"] >= SPEARMAN_FLOOR)


def test_invariant_ratio_band(artifact):
    m = artifact["meta"]
    assert m["invariant_ratio_band"] == (
        RATIO_MIN <= m["median_ratio_distance_to_slope"] <= RATIO_MAX
    )


def test_meta_verdict_is_conjunction(artifact):
    m = artifact["meta"]
    expected = "PASS" if all([
        m["invariant_match_count"], m["invariant_pearson_floor"],
        m["invariant_spearman_floor"], m["invariant_ratio_band"],
    ]) else "FAIL"
    assert m["verdict"] == expected
