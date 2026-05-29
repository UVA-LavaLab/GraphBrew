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

The gated claim: oracle-aware policies (POPT, GRASP) saturate to
near-zero final-octave slope, while non-oracle policies (LRU, SRRIP)
remain steep — concretely, ``POPT ≤ GRASP ≤ LRU``, ``POPT < SRRIP``,
all oracle medians ≤ 0.5 pp/oct ceiling, both non-oracle medians ≥
0.5 pp/oct floor, oracle median < 50% of non-oracle median, and at
least one POPT app saturates fully (min slope ≤ 0.2 pp/oct).
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
ORACLE_AWARE = ("POPT", "GRASP")
NON_ORACLE = ("LRU", "SRRIP")
ORACLE_AWARE_CEILING_PP = 0.5
NON_ORACLE_FLOOR_PP = 0.5
ORACLE_AWARE_HALF_OF_NON_ORACLE = 0.5
POPT_MIN_SLOPE_CEILING_PP = 0.2
EXPECTED_SCHEMA = "policy_steepness_ranking/v1"
EXPECTED_CHECK_KEYS = {
    "popt_le_grasp_median", "grasp_le_lru_median", "popt_lt_srrip_median",
    "oracle_aware_ceiling", "non_oracle_floor",
    "oracle_half_of_non_oracle", "popt_min_saturates",
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
        "oracle_median_pp", "non_oracle_median_pp",
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
    expected = {"apps", "policies", "oracle_aware", "non_oracle", "thresholds"}
    missing = expected - set(meta.keys())
    assert not missing, f"missing meta fields: {missing}"


def test_policy_families_pinned(artifact):
    meta = artifact["meta"]
    assert tuple(meta["oracle_aware"]) == ORACLE_AWARE, (
        "Oracle-aware family drifted — gate 81 reorders around "
        "ORACLE_AWARE=(POPT, GRASP)."
    )
    assert tuple(meta["non_oracle"]) == NON_ORACLE, (
        "Non-oracle family drifted — gate 81 reorders around "
        "NON_ORACLE=(LRU, SRRIP)."
    )


def test_thresholds_pinned(artifact):
    th = artifact["meta"]["thresholds"]
    assert th["oracle_aware_ceiling_pp"] == ORACLE_AWARE_CEILING_PP
    assert th["non_oracle_floor_pp"] == NON_ORACLE_FLOOR_PP
    assert th["oracle_aware_half_of_non_oracle"] == ORACLE_AWARE_HALF_OF_NON_ORACLE
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


def test_oracle_family_median_matches_recomputation(artifact):
    meds = [artifact["per_policy"][p]["median"] for p in ORACLE_AWARE]
    expected = round(statistics.median(meds), 6)
    assert artifact["oracle_median_pp"] == expected, (
        f"oracle_median_pp drift — recomputed {expected!r}, "
        f"got {artifact['oracle_median_pp']!r}"
    )


def test_non_oracle_family_median_matches_recomputation(artifact):
    meds = [artifact["per_policy"][p]["median"] for p in NON_ORACLE]
    expected = round(statistics.median(meds), 6)
    assert artifact["non_oracle_median_pp"] == expected


def test_ranking_is_sorted_by_median_ascending(artifact):
    policies = artifact["meta"]["policies"]
    expected = sorted(policies, key=lambda p: artifact["per_policy"][p]["median"])
    assert artifact["ranking_by_median"] == expected


def test_headline_ranking_oracle_first(artifact):
    """The dashboard story is 'POPT and GRASP take the top two slots,
    LRU and SRRIP the bottom'. Lock that today.
    """
    top_two = set(artifact["ranking_by_median"][:2])
    assert top_two == set(ORACLE_AWARE), (
        f"oracle-aware family lost top-2 in steepness ranking: "
        f"top two = {top_two}"
    )


# ----------------------------------------------------------------------
# Group D: checks dict — mirror of every generator predicate
# ----------------------------------------------------------------------

def test_checks_carries_canonical_keys(artifact):
    assert set(artifact["checks"].keys()) == EXPECTED_CHECK_KEYS


def test_popt_le_grasp_median_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    expected = meds["POPT"] <= meds["GRASP"] + 1e-9
    assert artifact["checks"]["popt_le_grasp_median"]["ok"] == expected


def test_grasp_le_lru_median_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    expected = meds["GRASP"] <= meds["LRU"] + 1e-9
    assert artifact["checks"]["grasp_le_lru_median"]["ok"] == expected


def test_popt_lt_srrip_median_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    expected = meds["POPT"] < meds["SRRIP"] - 1e-9
    assert artifact["checks"]["popt_lt_srrip_median"]["ok"] == expected


def test_oracle_ceiling_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    expected = all(meds[p] <= ORACLE_AWARE_CEILING_PP + 1e-9 for p in ORACLE_AWARE)
    assert artifact["checks"]["oracle_aware_ceiling"]["ok"] == expected


def test_non_oracle_floor_matches_recomputation(artifact):
    meds = artifact["medians_pp"]
    expected = all(meds[p] >= NON_ORACLE_FLOOR_PP - 1e-9 for p in NON_ORACLE)
    assert artifact["checks"]["non_oracle_floor"]["ok"] == expected


def test_half_check_matches_recomputation(artifact):
    expected = (
        artifact["oracle_median_pp"]
        < artifact["non_oracle_median_pp"] * ORACLE_AWARE_HALF_OF_NON_ORACLE + 1e-9
    )
    assert artifact["checks"]["oracle_half_of_non_oracle"]["ok"] == expected


def test_popt_min_saturates_matches_recomputation(artifact):
    popt_min = artifact["per_policy"]["POPT"]["min"]
    expected = popt_min <= POPT_MIN_SLOPE_CEILING_PP + 1e-9
    assert artifact["checks"]["popt_min_saturates"]["ok"] == expected


def test_check_payloads_match_artifact_state(artifact):
    """Side-band fields embedded in each check payload (e.g. popt=…,
    grasp=…, ceiling=…) must match the artifact's own medians/thresholds.
    """
    ch = artifact["checks"]
    meds = artifact["medians_pp"]
    assert ch["popt_le_grasp_median"]["popt"] == meds["POPT"]
    assert ch["popt_le_grasp_median"]["grasp"] == meds["GRASP"]
    assert ch["grasp_le_lru_median"]["grasp"] == meds["GRASP"]
    assert ch["grasp_le_lru_median"]["lru"] == meds["LRU"]
    assert ch["popt_lt_srrip_median"]["popt"] == meds["POPT"]
    assert ch["popt_lt_srrip_median"]["srrip"] == meds["SRRIP"]
    assert ch["oracle_aware_ceiling"]["ceiling"] == ORACLE_AWARE_CEILING_PP
    assert ch["non_oracle_floor"]["floor"] == NON_ORACLE_FLOOR_PP
    assert ch["oracle_half_of_non_oracle"]["required_ratio"] == ORACLE_AWARE_HALF_OF_NON_ORACLE
    assert ch["popt_min_saturates"]["ceiling"] == POPT_MIN_SLOPE_CEILING_PP


# ----------------------------------------------------------------------
# Group E: verdict
# ----------------------------------------------------------------------

def test_verdict_ok_is_and_of_all_checks(artifact):
    expected = all(c["ok"] for c in artifact["checks"].values())
    assert artifact["verdict_ok"] == expected


def test_current_verdict_is_pass(artifact):
    assert artifact["verdict_ok"] is True, (
        "policy_steepness_ranking regressed to FAIL — the oracle-aware "
        "vs non-oracle steepness story is the headline universality "
        "claim, and a regression here means an upstream slope reducer "
        "or saturation onset moved unexpectedly."
    )
