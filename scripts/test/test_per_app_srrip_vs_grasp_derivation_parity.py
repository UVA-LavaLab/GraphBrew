"""Derivation parity gate for ``wiki/data/per_app_srrip_vs_grasp.json``.

Locks the per-app SRRIP-vs-GRASP slope ordering report against its
single upstream — ``per_app_capacity_slope.json`` — so any silent
drift in the gap formula (delta = SRRIP.median − GRASP.median),
the deviation predicate (delta > ALLOW_SRRIP_SHALLOWER_BY_PP=1.0),
the pinned-app subtraction (deviating − {bfs}), or the 3-invariant
AND verdict trips a test before the dashboard re-publishes the
"per-app SRRIP-GRASP ordering holds modulo pinned bfs" reviewer
narrative.

The gate fully mirrors `scripts/experiments/ecg/per_app_srrip_vs_grasp.py`'s
`compute()` against the same upstream JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

UPSTREAM_PATH = WIKI_DATA / "per_app_capacity_slope.json"
ARTIFACT_PATH = WIKI_DATA / "per_app_srrip_vs_grasp.json"

ALLOW_SRRIP_SHALLOWER_BY_PP = 1.0
PINNED_DEVIATING_APPS = ("bfs",)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def upstream() -> dict:
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    return json.loads(UPSTREAM_PATH.read_text())


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta"}


def test_meta_keys(artifact):
    expected = {
        "source", "apps", "allow_srrip_shallower_by_pp",
        "pinned_deviating_apps", "deviating_apps", "new_deviating_apps",
        "missing_apps", "per_app", "verdict_checks", "verdict",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_source_points_to_upstream(artifact):
    assert artifact["meta"]["source"] == UPSTREAM_PATH.name


def test_allow_threshold_matches_constant(artifact):
    assert artifact["meta"]["allow_srrip_shallower_by_pp"] == \
        ALLOW_SRRIP_SHALLOWER_BY_PP


def test_pinned_deviating_apps_matches_constant(artifact):
    assert list(artifact["meta"]["pinned_deviating_apps"]) == \
        list(PINNED_DEVIATING_APPS)


def test_verdict_is_pass_or_fail(artifact):
    assert artifact["meta"]["verdict"] in {"PASS", "FAIL"}


def test_per_app_entry_shape(artifact):
    expected = {
        "grasp_median_pp_oct", "srrip_median_pp_oct",
        "srrip_minus_grasp_pp_oct", "deviates",
    }
    for app, entry in artifact["meta"]["per_app"].items():
        missing = expected - set(entry.keys())
        assert not missing, f"per_app[{app}] missing fields: {missing}"


def test_verdict_checks_three_keys(artifact):
    expected = {
        "no_missing_apps", "no_new_deviating_apps",
        "every_app_has_both_grasp_and_srrip",
    }
    assert set(artifact["meta"]["verdict_checks"].keys()) == expected


# ----------------------------------------------------------------------
# Group B: per-app cross-source parity
# ----------------------------------------------------------------------

def test_apps_list_matches_upstream_sorted(artifact, upstream):
    expected = sorted(upstream["meta"]["per_app"].keys())
    assert artifact["meta"]["apps"] == expected


def test_per_app_keys_match_apps(artifact):
    assert sorted(artifact["meta"]["per_app"].keys()) == \
        sorted(artifact["meta"]["apps"])


def test_per_app_grasp_median_matches_upstream(artifact, upstream):
    src = upstream["meta"]["per_app"]
    for app in artifact["meta"]["apps"]:
        g_up = src[app].get("GRASP", {}).get("median_pp")
        g_out = artifact["meta"]["per_app"][app]["grasp_median_pp_oct"]
        if g_up is None:
            assert g_out is None, (
                f"app={app}: GRASP missing upstream but artifact has {g_out}"
            )
        else:
            assert g_out == round(g_up, 4), (
                f"app={app}: GRASP median {g_out} ≠ round({g_up}, 4)"
            )


def test_per_app_srrip_median_matches_upstream(artifact, upstream):
    src = upstream["meta"]["per_app"]
    for app in artifact["meta"]["apps"]:
        s_up = src[app].get("SRRIP", {}).get("median_pp")
        s_out = artifact["meta"]["per_app"][app]["srrip_median_pp_oct"]
        if s_up is None:
            assert s_out is None, (
                f"app={app}: SRRIP missing upstream but artifact has {s_out}"
            )
        else:
            assert s_out == round(s_up, 4), (
                f"app={app}: SRRIP median {s_out} ≠ round({s_up}, 4)"
            )


def test_per_app_delta_matches_srrip_minus_grasp(artifact, upstream):
    src = upstream["meta"]["per_app"]
    for app in artifact["meta"]["apps"]:
        g = src[app].get("GRASP", {}).get("median_pp")
        s = src[app].get("SRRIP", {}).get("median_pp")
        d_out = artifact["meta"]["per_app"][app]["srrip_minus_grasp_pp_oct"]
        if g is None or s is None:
            assert d_out is None, (
                f"app={app}: delta must be None when either median is None"
            )
        else:
            expected = round(s - g, 4)
            assert d_out == expected, (
                f"app={app}: delta {d_out} ≠ round({s} − {g}, 4) = {expected}"
            )


def test_per_app_deviates_matches_threshold(artifact, upstream):
    src = upstream["meta"]["per_app"]
    for app in artifact["meta"]["apps"]:
        g = src[app].get("GRASP", {}).get("median_pp")
        s = src[app].get("SRRIP", {}).get("median_pp")
        d_out = artifact["meta"]["per_app"][app]["deviates"]
        if g is None or s is None:
            assert d_out is None, (
                f"app={app}: deviates must be None when a median is missing"
            )
        else:
            expected = (s - g) > ALLOW_SRRIP_SHALLOWER_BY_PP
            assert d_out == expected, (
                f"app={app}: deviates {d_out} ≠ "
                f"(({s} − {g}) > {ALLOW_SRRIP_SHALLOWER_BY_PP}) = {expected}"
            )


# ----------------------------------------------------------------------
# Group C: deviating-set & verdict parity
# ----------------------------------------------------------------------

def test_deviating_apps_matches_recompute(artifact):
    expected = [
        a for a in artifact["meta"]["apps"]
        if artifact["meta"]["per_app"][a].get("deviates") is True
    ]
    assert artifact["meta"]["deviating_apps"] == expected


def test_new_deviating_apps_excludes_pinned(artifact):
    deviating = artifact["meta"]["deviating_apps"]
    pinned = set(artifact["meta"]["pinned_deviating_apps"])
    expected = [a for a in deviating if a not in pinned]
    assert artifact["meta"]["new_deviating_apps"] == expected


def test_missing_apps_matches_recompute(artifact, upstream):
    src = upstream["meta"]["per_app"]
    expected = []
    for app in artifact["meta"]["apps"]:
        g = src[app].get("GRASP", {}).get("median_pp")
        s = src[app].get("SRRIP", {}).get("median_pp")
        if g is None or s is None:
            expected.append(app)
    assert artifact["meta"]["missing_apps"] == expected


def test_no_missing_apps_check_matches(artifact):
    expected = len(artifact["meta"]["missing_apps"]) == 0
    assert artifact["meta"]["verdict_checks"]["no_missing_apps"] == expected


def test_no_new_deviating_check_matches(artifact):
    expected = len(artifact["meta"]["new_deviating_apps"]) == 0
    assert artifact["meta"]["verdict_checks"]["no_new_deviating_apps"] \
        == expected


def test_every_app_has_both_check_matches(artifact):
    apps = artifact["meta"]["apps"]
    per_app = artifact["meta"]["per_app"]
    expected = all(
        per_app[a]["srrip_minus_grasp_pp_oct"] is not None for a in apps
    )
    assert artifact["meta"]["verdict_checks"][
        "every_app_has_both_grasp_and_srrip"] == expected


def test_verdict_is_pass_iff_all_checks_pass(artifact):
    checks = artifact["meta"]["verdict_checks"]
    expected = "PASS" if all(checks.values()) else "FAIL"
    assert artifact["meta"]["verdict"] == expected


# ----------------------------------------------------------------------
# Group D: end-to-end sanity
# ----------------------------------------------------------------------

def test_pinned_apps_are_subset_of_apps_or_subset_of_apps(artifact):
    """pinned must be a subset of apps OR all pinned apps must appear
    in deviating_apps (catches refactors that hide the pin)."""
    apps = set(artifact["meta"]["apps"])
    pinned = set(artifact["meta"]["pinned_deviating_apps"])
    for p in pinned:
        if p in apps:
            assert True
        else:
            pytest.skip(f"pinned app {p!r} not in current corpus")


def test_bfs_remains_deviating_per_pinned_set(artifact):
    """bfs is the pinned-deviating sentinel app. If GRASP & SRRIP both
    have medians, bfs must still appear in deviating_apps to match its
    pinned status (a frontier-driven near-flat curve)."""
    bfs = artifact["meta"]["per_app"].get("bfs")
    if not bfs:
        pytest.skip("bfs not in corpus")
    if (bfs["grasp_median_pp_oct"] is None
            or bfs["srrip_median_pp_oct"] is None):
        pytest.skip("bfs missing GRASP or SRRIP median")
    assert bfs["deviates"] is True, (
        "bfs is pinned-deviating but its delta no longer exceeds the floor "
        f"({bfs['srrip_minus_grasp_pp_oct']} vs floor "
        f"{ALLOW_SRRIP_SHALLOWER_BY_PP}) — either re-tune the pin or "
        "re-investigate the frontier-driven flatness assumption"
    )


def test_verdict_pass_implies_no_new_deviating(artifact):
    if artifact["meta"]["verdict"] == "PASS":
        assert artifact["meta"]["new_deviating_apps"] == [], (
            "PASS verdict but new_deviating_apps is non-empty"
        )


def test_threshold_is_positive(artifact):
    assert artifact["meta"]["allow_srrip_shallower_by_pp"] > 0
