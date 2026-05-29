"""Derivation parity gate for ``wiki/data/saturation_slope_extremum.json``.

Locks the per-app saturation-vs-slope extremum corroboration report
(gate 75) against its two upstreams — ``saturation_distance.json``
(per_app mean 4MB→8MB drop) and ``per_app_capacity_slope.json``
(per_(app, policy) median OLS slope) — so any silent drift in the
per-app distance/slope readout, the bespoke `_median` reducer for
per-app slope (median over policy medians, NOT mean), the
{distance, slope} ranking, the bfs-extremum uniqueness invariant,
or the 5-invariant AND verdict (bfs argmin both metrics ∧
strict-greater-than for all other apps ∧ corpus_has_slope_3x_bfs
∧ corpus_has_distance_2_5x_bfs) trips a test before the dashboard
re-publishes the "bfs is the unique least-cache-sensitive kernel"
claim.

The gate fully mirrors `scripts/experiments/ecg/saturation_slope_extremum.py`'s
`build()` against the same upstream JSONs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

DISTANCE_PATH = WIKI_DATA / "saturation_distance.json"
SLOPE_PATH = WIKI_DATA / "per_app_capacity_slope.json"
ARTIFACT_PATH = WIKI_DATA / "saturation_slope_extremum.json"

EXPECTED_BFS = "bfs"
SLOPE_STEEPNESS_RATIO_FLOOR = 3.0
DISTANCE_RATIO_FLOOR = 2.5


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
def distance_doc() -> dict:
    if not DISTANCE_PATH.exists():
        pytest.skip(f"missing {DISTANCE_PATH}")
    return json.loads(DISTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def slope_doc() -> dict:
    if not SLOPE_PATH.exists():
        pytest.skip(f"missing {SLOPE_PATH}")
    return json.loads(SLOPE_PATH.read_text())


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta"}


def test_meta_fields(artifact):
    expected = {
        "distance_source", "slope_source", "apps", "per_app",
        "least_cache_sensitive_app_by_distance",
        "least_cache_sensitive_app_by_slope",
        "most_cache_hungry_app_by_distance",
        "most_cache_hungry_app_by_slope",
        "most_hungry_app_disagreement_note",
        "slope_steepness_ratio_floor", "distance_ratio_floor",
        "verdict_checks", "verdict",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_distance_source_is_saturation_distance(artifact):
    assert artifact["meta"]["distance_source"] == DISTANCE_PATH.name


def test_slope_source_is_per_app_capacity_slope(artifact):
    assert artifact["meta"]["slope_source"] == SLOPE_PATH.name


def test_thresholds_match_constants(artifact):
    assert artifact["meta"]["slope_steepness_ratio_floor"] == \
        SLOPE_STEEPNESS_RATIO_FLOOR
    assert artifact["meta"]["distance_ratio_floor"] == DISTANCE_RATIO_FLOOR


def test_verdict_is_pass_or_fail(artifact):
    assert artifact["meta"]["verdict"] in {"PASS", "FAIL"}


def test_per_app_entry_shape(artifact):
    expected = {
        "distance_pp", "slope_pp_oct", "slope_steepness",
        "distance_rank", "slope_rank",
    }
    for app, entry in artifact["meta"]["per_app"].items():
        missing = expected - set(entry.keys())
        assert not missing, f"per_app[{app}] missing fields: {missing}"


def test_verdict_checks_has_five_keys(artifact):
    expected = {
        "bfs_is_argmin_distance",
        "bfs_is_shallowest_slope",
        "bfs_unique_extremum_on_both_metrics",
        "corpus_has_slope_steeper_than_3x_bfs",
        "corpus_has_distance_larger_than_2_5x_bfs",
    }
    assert set(artifact["meta"]["verdict_checks"].keys()) == expected


# ----------------------------------------------------------------------
# Group B: per-app cross-source parity
# ----------------------------------------------------------------------

def test_apps_match_intersection_of_upstreams(artifact, distance_doc, slope_doc):
    dist_apps = set(distance_doc["per_app"].keys())
    slope_apps = set(slope_doc["meta"]["per_app"].keys())
    expected = sorted(dist_apps & slope_apps)
    assert artifact["meta"]["apps"] == expected


def test_per_app_keys_match_apps(artifact):
    assert sorted(artifact["meta"]["per_app"].keys()) == \
        sorted(artifact["meta"]["apps"])


def test_per_app_distance_matches_upstream(artifact, distance_doc):
    src = distance_doc["per_app"]
    for app in artifact["meta"]["apps"]:
        dist_up = float(src[app]["mean_pp"])
        dist_out = artifact["meta"]["per_app"][app]["distance_pp"]
        assert dist_out == round(dist_up, 4), (
            f"app={app}: distance_pp {dist_out} ≠ round({dist_up}, 4)"
        )


def test_per_app_slope_matches_median_of_policy_medians(artifact, slope_doc):
    """Per-app slope = bespoke `_median` over the per-policy median_pp
    values (NOT a mean — explicit choice of central tendency)."""
    src = slope_doc["meta"]["per_app"]
    for app in artifact["meta"]["apps"]:
        pol_slopes = [b["median_pp"] for b in src[app].values()]
        expected = round(_median(pol_slopes), 4)
        actual = artifact["meta"]["per_app"][app]["slope_pp_oct"]
        assert actual == expected, (
            f"app={app}: slope_pp_oct {actual} ≠ "
            f"round(_median({pol_slopes}), 4) = {expected}"
        )


def test_per_app_slope_steepness_matches_abs_slope(artifact):
    for app, entry in artifact["meta"]["per_app"].items():
        expected = round(abs(entry["slope_pp_oct"]), 4)
        assert entry["slope_steepness"] == expected, (
            f"app={app}: slope_steepness {entry['slope_steepness']} ≠ "
            f"round(abs({entry['slope_pp_oct']}), 4)"
        )


# ----------------------------------------------------------------------
# Group C: ranking parity
# ----------------------------------------------------------------------

def test_distance_rank_matches_sort(artifact):
    per_app = artifact["meta"]["per_app"]
    rows = sorted(per_app.items(), key=lambda kv: kv[1]["distance_pp"])
    expected = {app: i + 1 for i, (app, _) in enumerate(rows)}
    actual = {a: e["distance_rank"] for a, e in per_app.items()}
    assert actual == expected, f"distance_rank drift: {actual} ≠ {expected}"


def test_slope_rank_matches_sort_by_steepness(artifact):
    """slope_rank sorted ASC by slope_steepness (NOT signed slope)."""
    per_app = artifact["meta"]["per_app"]
    rows = sorted(per_app.items(), key=lambda kv: kv[1]["slope_steepness"])
    expected = {app: i + 1 for i, (app, _) in enumerate(rows)}
    actual = {a: e["slope_rank"] for a, e in per_app.items()}
    assert actual == expected, f"slope_rank drift: {actual} ≠ {expected}"


def test_least_by_distance_matches_argmin(artifact):
    per_app = artifact["meta"]["per_app"]
    expected = min(per_app.items(), key=lambda kv: kv[1]["distance_pp"])[0]
    assert artifact["meta"]["least_cache_sensitive_app_by_distance"] == expected


def test_least_by_slope_matches_argmin_steepness(artifact):
    per_app = artifact["meta"]["per_app"]
    expected = min(per_app.items(), key=lambda kv: kv[1]["slope_steepness"])[0]
    assert artifact["meta"]["least_cache_sensitive_app_by_slope"] == expected


def test_most_by_distance_matches_argmax(artifact):
    per_app = artifact["meta"]["per_app"]
    expected = max(per_app.items(), key=lambda kv: kv[1]["distance_pp"])[0]
    assert artifact["meta"]["most_cache_hungry_app_by_distance"] == expected


def test_most_by_slope_matches_signed_min(artifact):
    """most_cache_hungry_app_by_slope = sorted(rows, key=slope_pp_oct)[0]
    — i.e., the MOST NEGATIVE signed slope (NOT argmax steepness)."""
    per_app = artifact["meta"]["per_app"]
    expected = sorted(per_app.items(),
                      key=lambda kv: kv[1]["slope_pp_oct"])[0][0]
    assert artifact["meta"]["most_cache_hungry_app_by_slope"] == expected


# ----------------------------------------------------------------------
# Group D: verdict reducer parity
# ----------------------------------------------------------------------

def test_check_bfs_is_argmin_distance_matches_recompute(artifact):
    expected = (
        artifact["meta"]["least_cache_sensitive_app_by_distance"]
        == EXPECTED_BFS
    )
    assert artifact["meta"]["verdict_checks"]["bfs_is_argmin_distance"] \
        == expected


def test_check_bfs_is_shallowest_slope_matches_recompute(artifact):
    expected = (
        artifact["meta"]["least_cache_sensitive_app_by_slope"]
        == EXPECTED_BFS
    )
    assert artifact["meta"]["verdict_checks"]["bfs_is_shallowest_slope"] \
        == expected


def test_check_bfs_unique_extremum_matches_recompute(artifact):
    per_app = artifact["meta"]["per_app"]
    bfs = per_app.get(EXPECTED_BFS)
    if bfs is None:
        expected = False
    else:
        expected = all(
            e["distance_pp"] > bfs["distance_pp"]
            and e["slope_steepness"] > bfs["slope_steepness"]
            for a, e in per_app.items() if a != EXPECTED_BFS
        )
    assert artifact["meta"]["verdict_checks"][
        "bfs_unique_extremum_on_both_metrics"] == expected


def test_check_corpus_slope_sensitive_matches_recompute(artifact):
    per_app = artifact["meta"]["per_app"]
    bfs = per_app.get(EXPECTED_BFS)
    if bfs and bfs["slope_steepness"] > 0:
        ratios = [
            e["slope_steepness"] / bfs["slope_steepness"]
            for a, e in per_app.items() if a != EXPECTED_BFS
        ]
        expected = any(r >= SLOPE_STEEPNESS_RATIO_FLOOR for r in ratios) \
            if ratios else False
    else:
        expected = False
    assert artifact["meta"]["verdict_checks"][
        "corpus_has_slope_steeper_than_3x_bfs"] == expected


def test_check_corpus_distance_sensitive_matches_recompute(artifact):
    per_app = artifact["meta"]["per_app"]
    bfs = per_app.get(EXPECTED_BFS)
    if bfs and bfs["distance_pp"] > 0:
        ratios = [
            e["distance_pp"] / bfs["distance_pp"]
            for a, e in per_app.items() if a != EXPECTED_BFS
        ]
        expected = any(r >= DISTANCE_RATIO_FLOOR for r in ratios) \
            if ratios else False
    else:
        expected = False
    assert artifact["meta"]["verdict_checks"][
        "corpus_has_distance_larger_than_2_5x_bfs"] == expected


def test_verdict_is_pass_iff_all_checks_pass(artifact):
    checks = artifact["meta"]["verdict_checks"]
    expected = "PASS" if all(checks.values()) else "FAIL"
    assert artifact["meta"]["verdict"] == expected


# ----------------------------------------------------------------------
# Group E: end-to-end sanity
# ----------------------------------------------------------------------

def test_bfs_has_rank_one_when_present(artifact):
    """bfs is the pinned least-cache-sensitive extremum on both axes,
    so its distance_rank and slope_rank must both be 1 when present
    AND when the gate is PASS."""
    if artifact["meta"]["verdict"] != "PASS":
        pytest.skip("verdict not PASS — bfs ranking may legitimately differ")
    bfs = artifact["meta"]["per_app"].get(EXPECTED_BFS)
    if bfs is None:
        pytest.skip("bfs not in corpus")
    assert bfs["distance_rank"] == 1
    assert bfs["slope_rank"] == 1


def test_disagreement_note_present(artifact):
    """The informational note is load-bearing for the dashboard's
    'regime-vs-aggregate distinction' reviewer explanation — must
    not vanish silently."""
    note = artifact["meta"]["most_hungry_app_disagreement_note"]
    assert isinstance(note, str) and len(note) > 50, (
        f"disagreement note is missing or trivially short: {note!r}"
    )


def test_thresholds_are_positive(artifact):
    assert artifact["meta"]["slope_steepness_ratio_floor"] > 0
    assert artifact["meta"]["distance_ratio_floor"] > 0


def test_ranks_form_permutation_of_one_to_n(artifact):
    per_app = artifact["meta"]["per_app"]
    n = len(per_app)
    expected = set(range(1, n + 1))
    dist_ranks = {e["distance_rank"] for e in per_app.values()}
    slope_ranks = {e["slope_rank"] for e in per_app.values()}
    assert dist_ranks == expected, (
        f"distance ranks not a permutation 1..{n}: {dist_ranks}"
    )
    assert slope_ranks == expected, (
        f"slope ranks not a permutation 1..{n}: {slope_ranks}"
    )
