"""Derivation parity gate for ``wiki/data/policy_steepness_ranking.json``.

Locks the per-policy final-octave steepness ranking artifact (gate 81)
against its single upstream input — ``cache_saturation_onset.json`` —
so any silent drift in the reducer aggregates, the ordering checks,
the pinned thresholds, or the headline ranking trips a test before
the dashboard re-publishes:

    cache_saturation_onset.json#per_app[app][policy].final_octave_slope_pp
                  │
            abs() + statistics.{median,mean,min,max}
                  │
                  ▼
    wiki/data/policy_steepness_ranking.json   ← gate target

The gated claim (charge-invariant): GRASP (degree heuristic) is the
FLATTEST policy (cache-insensitive hot-set pinning) and LRU (blind
recency) is the STEEPEST (most cache-sensitive), with a material spread
(LRU median ≥ 1.5× GRASP median) and at least one POPT app saturating
fully (min slope ≤ 0.2 pp/oct). Charged P-OPT is mid-pack — no longer
flat — so the gate no longer asserts POPT-is-flat (a multi-thread +
uncharged-P-OPT artifact).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "policy_steepness_ranking.json"
ONSET_PATH = WIKI_DATA / "cache_saturation_onset.json"

# Pinned mirror of generator constants.
FLATTEST_POLICY = "GRASP"
STEEPEST_POLICY = "LRU"
LRU_OVER_GRASP_SPREAD = 1.5
POPT_MIN_SLOPE_CEILING_PP = 0.2
EXPECTED_SCHEMA = "policy_steepness_ranking/v1"
EXPECTED_CHECK_KEYS = {
    "grasp_is_flattest", "lru_is_steepest", "grasp_le_lru_median",
    "steepness_spread", "popt_min_saturates",
}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def onset() -> dict:
    if not ONSET_PATH.exists():
        pytest.skip(f"missing {ONSET_PATH}")
    return json.loads(ONSET_PATH.read_text())


@pytest.fixture(scope="module")
def upstream_slopes(onset) -> dict[str, dict[str, float]]:
    """Recompute the |final-octave slope| matrix from upstream."""
    out: dict[str, dict[str, float]] = {}
    per_app = onset["per_app"]
    for app, by_policy in per_app.items():
        for pol, blob in by_policy.items():
            out.setdefault(pol, {})[app] = abs(float(blob["final_octave_slope_pp"]))
    return out


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    expected = {
        "schema", "source", "meta", "per_policy", "medians_pp",
        "ranking_by_median", "checks", "verdict_ok",
    }
    missing = expected - set(artifact.keys())
    assert not missing, f"missing top-level keys: {missing}"


def test_schema_version_pinned(artifact):
    assert artifact["schema"] == EXPECTED_SCHEMA


def test_source_path_pinned(artifact):
    assert artifact["source"] == "wiki/data/cache_saturation_onset.json"


def test_meta_has_canonical_fields(artifact):
    meta = artifact["meta"]
    expected = {"apps", "policies", "flattest_policy", "steepest_policy", "thresholds"}
    missing = expected - set(meta.keys())
    assert not missing, f"missing meta fields: {missing}"


def test_anchor_policies_pinned(artifact):
    meta = artifact["meta"]
    assert meta["flattest_policy"] == FLATTEST_POLICY, (
        "Flattest-policy anchor drifted — gate 81 expects GRASP "
        "(degree heuristic) to be the cache-insensitive policy."
    )
    assert meta["steepest_policy"] == STEEPEST_POLICY, (
        "Steepest-policy anchor drifted — gate 81 expects LRU "
        "(blind recency) to be the most cache-sensitive policy."
    )


def test_thresholds_pinned(artifact):
    th = artifact["meta"]["thresholds"]
    assert th["lru_over_grasp_spread"] == LRU_OVER_GRASP_SPREAD
    assert th["popt_min_slope_ceiling_pp"] == POPT_MIN_SLOPE_CEILING_PP


def test_apps_and_policies_match_upstream(artifact, onset):
    assert artifact["meta"]["apps"] == list(onset["meta"]["apps"])
    assert artifact["meta"]["policies"] == list(onset["meta"]["policies"])


# ----------------------------------------------------------------------
# Group B: per-policy aggregates cross-source parity
# ----------------------------------------------------------------------

def test_per_policy_keys_match_upstream_policies(artifact):
    assert sorted(artifact["per_policy"].keys()) == sorted(artifact["meta"]["policies"])


def test_per_policy_n_matches_app_count(artifact):
    n_apps = len(artifact["meta"]["apps"])
    for pol, e in artifact["per_policy"].items():
        assert e["n"] == n_apps, (
            f"{pol}: per_policy.n={e['n']} ≠ len(apps)={n_apps}"
        )


def test_per_app_slopes_match_upstream(artifact, upstream_slopes):
    """Each per_policy[p].per_app[a] equals round(|upstream|, 6)."""
    for pol, e in artifact["per_policy"].items():
        for app, val in e["per_app"].items():
            raw = upstream_slopes[pol][app]
            expected = round(raw, 6)
            assert val == expected, (
                f"{pol}/{app}: per_app drift — raw |final_octave_slope|={raw!r} "
                f"round6={expected!r} artifact={val!r}"
            )


def test_per_policy_min_matches_upstream(artifact, upstream_slopes):
    for pol, e in artifact["per_policy"].items():
        slopes = [upstream_slopes[pol][a] for a in artifact["meta"]["apps"]]
        assert e["min"] == round(min(slopes), 6), (
            f"{pol}: min drift — recomputed={round(min(slopes), 6)!r} "
            f"artifact={e['min']!r}"
        )


def test_per_policy_max_matches_upstream(artifact, upstream_slopes):
    for pol, e in artifact["per_policy"].items():
        slopes = [upstream_slopes[pol][a] for a in artifact["meta"]["apps"]]
        assert e["max"] == round(max(slopes), 6)


def test_per_policy_median_matches_upstream(artifact, upstream_slopes):
    for pol, e in artifact["per_policy"].items():
        slopes = [upstream_slopes[pol][a] for a in artifact["meta"]["apps"]]
        assert e["median"] == round(statistics.median(slopes), 6), (
            f"{pol}: median drift — recomputed="
            f"{round(statistics.median(slopes), 6)!r} artifact={e['median']!r}"
        )


def test_per_policy_mean_matches_upstream(artifact, upstream_slopes):
    for pol, e in artifact["per_policy"].items():
        slopes = [upstream_slopes[pol][a] for a in artifact["meta"]["apps"]]
        assert e["mean"] == round(statistics.mean(slopes), 6)


# ----------------------------------------------------------------------
# Group C: family medians + ranking
# ----------------------------------------------------------------------

def test_medians_pp_mirrors_per_policy_medians(artifact):
    for pol, mr in artifact["medians_pp"].items():
        assert mr == artifact["per_policy"][pol]["median"], (
            f"{pol}: medians_pp drift from per_policy.median"
        )


def test_ranking_is_sorted_by_median_ascending(artifact):
    policies = artifact["meta"]["policies"]
    expected = sorted(policies, key=lambda p: artifact["per_policy"][p]["median"])
    assert artifact["ranking_by_median"] == expected


def test_headline_ranking_grasp_first_lru_last(artifact):
    """The dashboard story is 'GRASP is flattest (top slot), LRU is
    steepest (bottom slot)'. Lock that today.
    """
    ranking = artifact["ranking_by_median"]
    assert ranking[0] == FLATTEST_POLICY, (
        f"GRASP lost the flattest slot in steepness ranking: top = {ranking[0]}"
    )
    assert ranking[-1] == STEEPEST_POLICY, (
        f"LRU lost the steepest slot in steepness ranking: bottom = {ranking[-1]}"
    )


# ----------------------------------------------------------------------
# Group D: checks dict — mirror of every generator predicate
# ----------------------------------------------------------------------

def test_checks_carries_canonical_keys(artifact):
    assert set(artifact["checks"].keys()) == EXPECTED_CHECK_KEYS


def test_grasp_is_flattest_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    others = [p for p in artifact["meta"]["policies"] if p != FLATTEST_POLICY]
    expected = all(meds[FLATTEST_POLICY] <= meds[p] + 1e-9 for p in others)
    assert artifact["checks"]["grasp_is_flattest"]["ok"] == expected


def test_lru_is_steepest_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    others = [p for p in artifact["meta"]["policies"] if p != STEEPEST_POLICY]
    expected = all(meds[STEEPEST_POLICY] >= meds[p] - 1e-9 for p in others)
    assert artifact["checks"]["lru_is_steepest"]["ok"] == expected


def test_grasp_le_lru_median_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    expected = meds[FLATTEST_POLICY] <= meds[STEEPEST_POLICY] + 1e-9
    assert artifact["checks"]["grasp_le_lru_median"]["ok"] == expected


def test_steepness_spread_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    expected = (
        meds[STEEPEST_POLICY] >= LRU_OVER_GRASP_SPREAD * meds[FLATTEST_POLICY] - 1e-9
    )
    assert artifact["checks"]["steepness_spread"]["ok"] == expected


def test_popt_min_saturates_matches_recomputation(artifact):
    popt_min = artifact["per_policy"]["POPT"]["min"]
    expected = popt_min <= POPT_MIN_SLOPE_CEILING_PP + 1e-9
    assert artifact["checks"]["popt_min_saturates"]["ok"] == expected


def test_check_payloads_match_artifact_state(artifact):
    """Side-band fields embedded in each check payload must match the
    artifact's own medians/thresholds.
    """
    ch = artifact["checks"]
    meds = artifact["medians_pp"]
    assert ch["grasp_is_flattest"]["grasp"] == meds[FLATTEST_POLICY]
    assert ch["lru_is_steepest"]["lru"] == meds[STEEPEST_POLICY]
    assert ch["grasp_le_lru_median"]["grasp"] == meds[FLATTEST_POLICY]
    assert ch["grasp_le_lru_median"]["lru"] == meds[STEEPEST_POLICY]
    assert ch["steepness_spread"]["required_ratio"] == LRU_OVER_GRASP_SPREAD
    assert ch["popt_min_saturates"]["ceiling"] == POPT_MIN_SLOPE_CEILING_PP


# ----------------------------------------------------------------------
# Group E: verdict
# ----------------------------------------------------------------------

def test_verdict_ok_is_and_of_all_checks(artifact):
    expected = all(c["ok"] for c in artifact["checks"].values())
    assert artifact["verdict_ok"] == expected


def test_current_verdict_is_pass(artifact):
    assert artifact["verdict_ok"] is True, (
        "policy_steepness_ranking regressed to FAIL — the GRASP-flattest "
        "vs LRU-steepest steepness story is the headline cache-sensitivity "
        "claim, and a regression here means an upstream slope reducer "
        "or saturation onset moved unexpectedly."
    )
