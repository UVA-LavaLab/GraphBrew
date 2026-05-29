"""
Confidence gate 115 — cohens_h_win_rates ↔ wilson_win_rates parity + arithmetic.

The two effect-size views must agree on the underlying counts:
- Same source (wiki/data/oracle_gap.json), same apps, policies, n_rows.
- For every (app, policy): same wins, same total, same p_hat — Wilson and
  Cohen's h read identical 0/1 win indicators, so their per-cell rates
  must coincide exactly.

The gate also recomputes Cohen's h, delta_p, favored direction, and
magnitude bucket from the per-cell rates, so any drift between the
displayed effect size and the formula in the paper text is caught
immediately. Magnitude buckets follow Cohen 1988 (small=0.2, medium=0.5,
large=0.8 — see meta.thresholds).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "wiki" / "data"
COHENS_PATH = DATA / "cohens_h_win_rates.json"
WILSON_PATH = DATA / "wilson_win_rates.json"

EXPECTED_SOURCE = "wiki/data/oracle_gap.json"
P_HAT_TOL = 1e-4          # both artifacts round to 4 dp
H_TOL = 1e-3              # arcsin recomputation tolerance
DELTA_TOL = 1e-3
EXPECTED_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_MAGNITUDES = {"large", "medium", "small", "negligible"}


@pytest.fixture(scope="module")
def cohens() -> dict:
    assert COHENS_PATH.exists(), f"missing artifact: {COHENS_PATH}"
    return json.loads(COHENS_PATH.read_text())


@pytest.fixture(scope="module")
def wilson() -> dict:
    assert WILSON_PATH.exists(), f"missing artifact: {WILSON_PATH}"
    return json.loads(WILSON_PATH.read_text())


def _h(p_a: float, p_b: float) -> float:
    return 2.0 * abs(math.asin(math.sqrt(p_a)) - math.asin(math.sqrt(p_b)))


def _magnitude(h: float, thresholds: dict) -> str:
    if h >= thresholds["large"]:
        return "large"
    if h >= thresholds["medium"]:
        return "medium"
    if h >= thresholds["small"]:
        return "small"
    return "negligible"


# ---------------------------------------------------------------------------
# Group A — meta + universe parity
# ---------------------------------------------------------------------------

def test_both_artifacts_share_source(cohens, wilson):
    assert cohens["meta"]["source"] == EXPECTED_SOURCE
    assert wilson["meta"]["source"] == EXPECTED_SOURCE


def test_meta_apps_match(cohens, wilson):
    assert set(cohens["meta"]["apps"]) == EXPECTED_APPS
    assert set(wilson["meta"]["apps"]) == EXPECTED_APPS
    assert set(cohens["meta"]["apps"]) == set(wilson["meta"]["apps"])


def test_meta_policies_match(cohens, wilson):
    assert set(cohens["meta"]["policies"]) == EXPECTED_POLICIES
    assert set(wilson["meta"]["policies"]) == EXPECTED_POLICIES
    assert set(cohens["meta"]["policies"]) == set(wilson["meta"]["policies"])


def test_meta_n_rows_matches(cohens, wilson):
    assert cohens["meta"]["n_rows"] == wilson["meta"]["n_rows"]
    assert cohens["meta"]["n_rows"] > 0


def test_cohens_h_thresholds_follow_cohen_1988(cohens):
    t = cohens["meta"]["thresholds"]
    assert t["large"] == 0.8
    assert t["medium"] == 0.5
    assert t["small"] == 0.2


def test_wilson_ci_level_is_95pct(wilson):
    assert wilson["meta"]["ci_level"] == 0.95
    assert wilson["meta"]["method"] == "wilson_score"
    # z for 95% two-sided CI ≈ 1.96
    assert abs(wilson["meta"]["z"] - 1.959963984540054) < 1e-9


# ---------------------------------------------------------------------------
# Group B — per-app cell parity (THE key invariant)
# ---------------------------------------------------------------------------

def test_per_app_app_set_matches(cohens, wilson):
    assert set(cohens["per_app"]) == set(wilson["per_app"]) == EXPECTED_APPS


def test_per_app_policy_set_matches(cohens, wilson):
    for app in cohens["per_app"]:
        c_pols = set(cohens["per_app"][app]["rates"])
        w_pols = set(wilson["per_app"][app])
        assert c_pols == w_pols == EXPECTED_POLICIES, f"{app}: policy set mismatch"


def test_per_app_wins_total_p_hat_match_between_artifacts(cohens, wilson):
    for app in cohens["per_app"]:
        crates = cohens["per_app"][app]["rates"]
        wrates = wilson["per_app"][app]
        for pol in crates:
            assert crates[pol]["wins"] == wrates[pol]["wins"], (
                f"{app}/{pol}: wins mismatch cohens={crates[pol]['wins']} wilson={wrates[pol]['wins']}"
            )
            assert crates[pol]["total"] == wrates[pol]["total"], (
                f"{app}/{pol}: total mismatch"
            )
            assert abs(crates[pol]["p_hat"] - wrates[pol]["p_hat"]) < P_HAT_TOL, (
                f"{app}/{pol}: p_hat drift cohens={crates[pol]['p_hat']} wilson={wrates[pol]['p_hat']}"
            )


# ---------------------------------------------------------------------------
# Group C — Cohen's h arithmetic
# ---------------------------------------------------------------------------

def test_cohens_h_per_comparison_recomputes_from_raw_counts(cohens):
    for app, payload in cohens["per_app"].items():
        rates = payload["rates"]
        for cmp in payload["comparisons"]:
            a, b = cmp["a"], cmp["b"]
            p_a_raw = rates[a]["wins"] / rates[a]["total"]
            p_b_raw = rates[b]["wins"] / rates[b]["total"]
            expected = _h(p_a_raw, p_b_raw)
            assert abs(cmp["h"] - expected) < H_TOL, (
                f"{app}/{a}-{b}: h={cmp['h']} expected≈{expected:.4f}"
            )


def test_cohens_h_delta_p_matches_p_a_minus_p_b(cohens):
    for app, payload in cohens["per_app"].items():
        rates = payload["rates"]
        for cmp in payload["comparisons"]:
            expected = rates[cmp["a"]]["p_hat"] - rates[cmp["b"]]["p_hat"]
            assert abs(cmp["delta_p"] - expected) < DELTA_TOL, (
                f"{app}/{cmp['a']}-{cmp['b']}: delta_p={cmp['delta_p']} expected={expected:.4f}"
            )


def test_cohens_h_favors_is_higher_p_hat(cohens):
    for app, payload in cohens["per_app"].items():
        rates = payload["rates"]
        for cmp in payload["comparisons"]:
            p_a = rates[cmp["a"]]["p_hat"]
            p_b = rates[cmp["b"]]["p_hat"]
            if p_a == p_b:
                # tied — artifact uses the sentinel 'tie' or picks either side
                assert cmp["favors"] in (cmp["a"], cmp["b"], "tie"), (
                    f"{app}/{cmp['a']}-{cmp['b']}: tied p_hat but favors={cmp['favors']!r}"
                )
            else:
                expected = cmp["a"] if p_a > p_b else cmp["b"]
                assert cmp["favors"] == expected, (
                    f"{app}/{cmp['a']}-{cmp['b']}: favors={cmp['favors']} expected={expected}"
                )


def test_cohens_h_magnitude_bucket_matches_thresholds(cohens):
    thresholds = cohens["meta"]["thresholds"]
    for app, payload in cohens["per_app"].items():
        for cmp in payload["comparisons"]:
            assert cmp["magnitude"] in EXPECTED_MAGNITUDES
            expected = _magnitude(cmp["h"], thresholds)
            assert cmp["magnitude"] == expected, (
                f"{app}/{cmp['a']}-{cmp['b']}: magnitude={cmp['magnitude']} "
                f"expected={expected} (h={cmp['h']})"
            )


def test_cohens_h_comparisons_cover_full_permutations(cohens):
    # P(4, 2) = 12 ordered pairs per app
    for app, payload in cohens["per_app"].items():
        seen = {(cmp["a"], cmp["b"]) for cmp in payload["comparisons"]}
        expected = {(a, b) for a in EXPECTED_POLICIES for b in EXPECTED_POLICIES if a != b}
        assert seen == expected, f"{app}: comparison set mismatch"
        assert len(payload["comparisons"]) == 12
