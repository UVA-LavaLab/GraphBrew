"""Gate 124 — saturation_slope_extremum.json arithmetic + verdict.

Locks the per-app least/most cache-sensitive extremum corroboration:
distance metric (4MB→8MB drop) and slope metric (OLS over 1MB-8MB)
must agree on the LEAST-sensitive app (bfs), but are explicitly
allowed to disagree on the MOST-hungry app (regime-vs-aggregate).

Source artifacts:
    distance: wiki/data/saturation_distance.json (per_app[app].mean_pp)
    slope:    wiki/data/per_app_capacity_slope.json
              (meta.per_app[app][policy].median_pp; aggregated by
               median across policies)

Per-app derivations:
    distance_pp = round(saturation_distance.per_app[app].mean_pp, 4)
    slope_pp_oct = round(median(per_policy median_pp), 4)
    slope_steepness = round(abs(slope_pp_oct), 4)
    distance_rank = position after sort by distance_pp ascending
    slope_rank = position after sort by slope_steepness ascending

Verdict (PASS) requires all 5 checks:
    bfs_is_argmin_distance (least distance == bfs)
    bfs_is_shallowest_slope (least steepness == bfs)
    bfs_unique_extremum_on_both_metrics (every other app has BOTH
        steeper slope AND larger distance than bfs)
    corpus_has_slope_steeper_than_3x_bfs
        (at least one app's steepness >= 3.0× bfs)
    corpus_has_distance_larger_than_2_5x_bfs
        (at least one app's distance >= 2.5× bfs)

Invariants (14 tests, 4 groups):
- meta + sources (3): distance/slope source filenames; apps list
  = sorted intersection of distance and slope per_app keys;
  thresholds match documented constants (3.0, 2.5).
- per_app stats from source (4): distance_pp, slope_pp_oct,
  slope_steepness recomputed from upstream artifacts; ranks
  reproduce ascending sort positions.
- least/most extremum identifiers (3): least_by_distance = argmin
  on distance_pp; least_by_slope = argmin on slope_steepness;
  most_by_distance = argmax on distance_pp; most_by_slope =
  argmin on raw slope_pp_oct (most-negative).
- five verdict checks + verdict + note (4): each check.ok matches
  documented inequality; verdict_checks values match recomputed
  invariants; verdict = PASS iff all checks; note string mentions
  both metrics.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/saturation_slope_extremum.json")
DIST_SRC = Path("wiki/data/saturation_distance.json")
SLOPE_SRC = Path("wiki/data/per_app_capacity_slope.json")

EXPECTED_BFS = "bfs"
SLOPE_RATIO_FLOOR = 3.0
DIST_RATIO_FLOOR = 2.5
ROUND_TOL = 5e-4


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists()
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def dist_src():
    assert DIST_SRC.exists()
    return json.loads(DIST_SRC.read_text())["per_app"]


@pytest.fixture(scope="module")
def slope_src():
    assert SLOPE_SRC.exists()
    return json.loads(SLOPE_SRC.read_text())["meta"]["per_app"]


def _expected_apps(dist_src, slope_src):
    return sorted(set(dist_src) & set(slope_src))


def _expected_slope_median(slope_src, app):
    medians = [b["median_pp"] for b in slope_src[app].values()]
    return statistics.median(medians)


# ── group 1: meta + sources ──────────────────────────────────────────────


def test_source_filenames(data):
    m = data["meta"]
    assert m["distance_source"] == "saturation_distance.json"
    assert m["slope_source"] == "per_app_capacity_slope.json"


def test_apps_is_sorted_intersection(data, dist_src, slope_src):
    assert data["meta"]["apps"] == _expected_apps(dist_src, slope_src)


def test_threshold_floors(data):
    m = data["meta"]
    assert m["slope_steepness_ratio_floor"] == SLOPE_RATIO_FLOOR
    assert m["distance_ratio_floor"] == DIST_RATIO_FLOOR


# ── group 2: per_app stats from source ───────────────────────────────────


def test_per_app_distance_from_source(data, dist_src):
    for app, entry in data["meta"]["per_app"].items():
        expected = round(float(dist_src[app]["mean_pp"]), 4)
        assert math.isclose(entry["distance_pp"], expected, abs_tol=ROUND_TOL), app


def test_per_app_slope_from_source(data, slope_src):
    for app, entry in data["meta"]["per_app"].items():
        expected_slope = round(_expected_slope_median(slope_src, app), 4)
        assert math.isclose(entry["slope_pp_oct"], expected_slope, abs_tol=ROUND_TOL), app
        assert math.isclose(
            entry["slope_steepness"], round(abs(expected_slope), 4), abs_tol=ROUND_TOL
        ), app


def test_distance_rank_is_ascending_sort_position(data):
    per_app = data["meta"]["per_app"]
    ordered = sorted(per_app.items(), key=lambda kv: kv[1]["distance_pp"])
    for rank, (app, _) in enumerate(ordered, 1):
        assert per_app[app]["distance_rank"] == rank, f"{app}: distance_rank"


def test_slope_rank_is_ascending_steepness_position(data):
    per_app = data["meta"]["per_app"]
    ordered = sorted(per_app.items(), key=lambda kv: kv[1]["slope_steepness"])
    for rank, (app, _) in enumerate(ordered, 1):
        assert per_app[app]["slope_rank"] == rank, f"{app}: slope_rank"


# ── group 3: least/most extremum identifiers ─────────────────────────────


def test_least_extremum_identifiers(data):
    m = data["meta"]
    per_app = m["per_app"]
    least_dist = min(per_app.items(), key=lambda kv: kv[1]["distance_pp"])[0]
    least_slope = min(per_app.items(), key=lambda kv: kv[1]["slope_steepness"])[0]
    assert m["least_cache_sensitive_app_by_distance"] == least_dist
    assert m["least_cache_sensitive_app_by_slope"] == least_slope


def test_most_hungry_by_distance_is_argmax(data):
    per_app = data["meta"]["per_app"]
    most_dist = max(per_app.items(), key=lambda kv: kv[1]["distance_pp"])[0]
    assert data["meta"]["most_cache_hungry_app_by_distance"] == most_dist


def test_most_hungry_by_slope_is_argmin_raw_slope(data):
    per_app = data["meta"]["per_app"]
    most_slope = min(per_app.items(), key=lambda kv: kv[1]["slope_pp_oct"])[0]
    assert data["meta"]["most_cache_hungry_app_by_slope"] == most_slope


# ── group 4: five verdict checks + verdict + note ────────────────────────


def test_check_bfs_is_argmin_distance(data):
    m = data["meta"]
    expected = m["least_cache_sensitive_app_by_distance"] == EXPECTED_BFS
    assert m["verdict_checks"]["bfs_is_argmin_distance"] is expected


def test_check_bfs_is_shallowest_slope(data):
    m = data["meta"]
    expected = m["least_cache_sensitive_app_by_slope"] == EXPECTED_BFS
    assert m["verdict_checks"]["bfs_is_shallowest_slope"] is expected


def test_check_bfs_unique_extremum(data):
    per_app = data["meta"]["per_app"]
    bfs = per_app.get(EXPECTED_BFS)
    expected = bool(bfs) and all(
        e["distance_pp"] > bfs["distance_pp"] and e["slope_steepness"] > bfs["slope_steepness"]
        for a, e in per_app.items()
        if a != EXPECTED_BFS
    )
    assert data["meta"]["verdict_checks"]["bfs_unique_extremum_on_both_metrics"] is expected


def test_corpus_floor_checks(data):
    per_app = data["meta"]["per_app"]
    bfs = per_app[EXPECTED_BFS]
    slope_ratios = (
        [e["slope_steepness"] / bfs["slope_steepness"] for a, e in per_app.items() if a != EXPECTED_BFS]
        if bfs["slope_steepness"] > 0
        else []
    )
    dist_ratios = (
        [e["distance_pp"] / bfs["distance_pp"] for a, e in per_app.items() if a != EXPECTED_BFS]
        if bfs["distance_pp"] > 0
        else []
    )
    expected_slope = any(r >= SLOPE_RATIO_FLOOR for r in slope_ratios) if slope_ratios else False
    expected_dist = any(r >= DIST_RATIO_FLOOR for r in dist_ratios) if dist_ratios else False
    checks = data["meta"]["verdict_checks"]
    assert checks["corpus_has_slope_steeper_than_3x_bfs"] is expected_slope
    assert checks["corpus_has_distance_larger_than_2_5x_bfs"] is expected_dist


def test_verdict_is_all_checks(data):
    m = data["meta"]
    expected = "PASS" if all(m["verdict_checks"].values()) else "FAIL"
    assert m["verdict"] == expected


def test_disagreement_note_mentions_both_metrics(data):
    note = data["meta"]["most_hungry_app_disagreement_note"]
    assert "INFORMATIONAL" in note
    assert "distance" in note.lower()
    assert "slope" in note.lower()
    assert "bfs" in note.lower()
