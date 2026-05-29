"""Confidence gate 98 — oracle-gap curvature vs effect-size cross-artifact
consistency.

Two per-app summaries of oracle-gap behaviour should each be internally
well-formed AND agree on which (app, policy) pairs show a real shift:

  - ``wiki/data/oracle_gap_curvature.json`` (OGC) — per-app per-policy
    gap_at_1MB / gap_at_4MB / gap_at_8MB plus their octave-normalized
    slopes (slope_1MB_to_4MB, slope_4MB_to_8MB), a derived curvature
    (slope_4MB_to_8MB - slope_1MB_to_4MB), and a knee_present flag when
    curvature exceeds 0.05 pp/oct^2.
  - ``wiki/data/oracle_gap_effect_size.json`` (OGES) — per-app exhaustive
    ordered-pair comparisons across the 4 policies giving Cliff's delta,
    its magnitude class (large / medium / small / negligible), the
    Mann-Whitney p-value (two-sided), n_a / n_b, and which policy is
    stochastically_smaller.

This gate locks 13 invariants split across four groups:

  OGC internal book-keeping (4):
    1. sum of per_app cells equals meta.cells_total (= 20 = 5 apps x 4
       policies)
    2. per_policy_summary[pol].knee_count equals the recomputed knee
       count across per_app[*][pol]
    3. meta.cells_with_knee equals the total knee_present count across
       all per_app cells
    4. knee_rank_by_policy lists every policy with knees BEFORE every
       policy without knees AND respects descending knee_count

  OGC slope/curvature math (2):
    5. slope_1MB_to_4MB == (gap_at_4MB - gap_at_1MB) / log2(4/1) AND
       slope_4MB_to_8MB == (gap_at_8MB - gap_at_4MB) / log2(8/4) for
       every (app, policy) cell
    6. curvature_at_4MB == slope_4MB_to_8MB - slope_1MB_to_4MB for every
       (app, policy) cell

  OGES internal book-keeping (4):
    7. per_app keys equal meta.apps set AND every app has exactly 12
       comparisons (4 policies x 3 ordered-pair partners)
    8. cliffs_delta_a_minus_b is antisymmetric: cliffs(a,b) ==
       -cliffs(b,a) for every (app, a, b) pair
    9. mannwhitney_p is symmetric: p(a,b) == p(b,a) AND magnitude
       classification matches the meta.cliffs_thresholds bands
   10. large_negative_deltas equals exactly the set of comparisons with
       magnitude == 'large' AND cliffs_delta_a_minus_b < 0 across all
       per_app

  Cross-artifact (OGC <-> OGES) (3):
   11. app universes agree: set(OGC.per_app) == set(OGES.per_app) ==
       set(OGES.meta.apps) == {bc, bfs, cc, pr, sssp}
   12. policy universes agree: every OGC per_app cell uses the same 4
       policies as OGES.meta.policies (GRASP, LRU, POPT, SRRIP)
   13. knee implies stochastic-smaller-vs-LRU: for every (app, policy)
       cell with knee_present == True and policy != LRU, the OGES
       comparison (policy, LRU) for the same app must declare
       stochastically_smaller == policy. (If a policy's oracle gap
       collapses fast enough to register as a knee, it must also
       distributionally beat LRU's oracle gap.)

If any one invariant breaks, the curvature-based "knee policies" claim
loses its statistical backing (or vice versa: the effect-size table no
longer guarantees that the geometric knee we report corresponds to a
real distributional advantage).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OGC_PATH = PROJECT_ROOT / "wiki" / "data" / "oracle_gap_curvature.json"
OGES_PATH = PROJECT_ROOT / "wiki" / "data" / "oracle_gap_effect_size.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"GRASP", "LRU", "POPT", "SRRIP"}
N_POLICIES = 4
N_COMPARISONS_PER_APP = N_POLICIES * (N_POLICIES - 1)  # 12 ordered pairs

SLOPE_TOL = 1e-3
CURV_TOL = 1e-3
CLIFFS_ANTISYM_TOL = 1e-6
P_SYM_TOL = 1e-9


@pytest.fixture(scope="module")
def ogc() -> dict:
    assert OGC_PATH.exists(), f"missing oracle_gap_curvature.json at {OGC_PATH}"
    return json.loads(OGC_PATH.read_text())


@pytest.fixture(scope="module")
def oges() -> dict:
    assert OGES_PATH.exists(), f"missing oracle_gap_effect_size.json at {OGES_PATH}"
    return json.loads(OGES_PATH.read_text())


# ---------------------------------------------------------------------------
# OGC internal (4)
# ---------------------------------------------------------------------------


def test_ogc_per_app_total_cells_matches_meta(ogc: dict) -> None:
    declared = ogc["meta"]["cells_total"]
    actual = sum(len(p) for p in ogc["per_app"].values())
    assert declared == actual, (
        f"meta.cells_total={declared} but per_app sum={actual}"
    )


def test_ogc_per_policy_knee_count_matches_per_app(ogc: dict) -> None:
    recomputed: dict[str, int] = {}
    for app, p in ogc["per_app"].items():
        for pol, payload in p.items():
            recomputed[pol] = recomputed.get(pol, 0) + (1 if payload.get("knee_present") else 0)
    declared = {pol: ogc["per_policy_summary"][pol]["knee_count"] for pol in ogc["per_policy_summary"]}
    assert recomputed == declared, (
        f"per_policy_summary knee_count != per_app recomputation:\n"
        f"  declared: {declared}\n  actual:   {recomputed}"
    )


def test_ogc_cells_with_knee_matches_sum(ogc: dict) -> None:
    declared = ogc["meta"]["cells_with_knee"]
    actual = sum(
        1
        for app, p in ogc["per_app"].items()
        for pol, payload in p.items()
        if payload.get("knee_present")
    )
    assert declared == actual, (
        f"meta.cells_with_knee={declared} but per_app knee_present sum={actual}"
    )


def test_ogc_knee_rank_consistent_with_counts(ogc: dict) -> None:
    rank = ogc["meta"]["knee_rank_by_policy"]
    counts = {pol: ogc["per_policy_summary"][pol]["knee_count"] for pol in rank}
    # Every policy with knee_count > 0 must precede every policy with knee_count == 0
    seen_zero = False
    for pol in rank:
        if counts[pol] == 0:
            seen_zero = True
        elif seen_zero:
            raise AssertionError(
                f"knee_rank_by_policy has policy with knees ({pol}, n={counts[pol]}) after a zero-knee policy: {rank}"
            )
    # Among policies with knees, counts must be non-increasing
    nonzero_counts = [counts[p] for p in rank if counts[p] > 0]
    assert nonzero_counts == sorted(nonzero_counts, reverse=True), (
        f"knee_rank_by_policy not in descending knee_count order for nonzero policies: {dict(zip(rank, nonzero_counts))}"
    )


# ---------------------------------------------------------------------------
# OGC slope/curvature math (2)
# ---------------------------------------------------------------------------


def test_ogc_slope_math_matches_gap_octaves(ogc: dict) -> None:
    bad: list = []
    log2_4_over_1 = math.log2(4.0)
    log2_8_over_4 = math.log2(2.0)
    for app, p in ogc["per_app"].items():
        for pol, payload in p.items():
            g1 = payload["gap_at_1MB"]
            g4 = payload["gap_at_4MB"]
            g8 = payload["gap_at_8MB"]
            expected_s14 = (g4 - g1) / log2_4_over_1
            expected_s48 = (g8 - g4) / log2_8_over_4
            actual_s14 = payload["slope_1MB_to_4MB"]
            actual_s48 = payload["slope_4MB_to_8MB"]
            if not math.isclose(expected_s14, actual_s14, abs_tol=SLOPE_TOL):
                bad.append((app, pol, "slope_1MB_to_4MB", expected_s14, actual_s14))
            if not math.isclose(expected_s48, actual_s48, abs_tol=SLOPE_TOL):
                bad.append((app, pol, "slope_4MB_to_8MB", expected_s48, actual_s48))
    assert not bad, f"OGC slope math violations: {bad}"


def test_ogc_curvature_math_matches_slope_diff(ogc: dict) -> None:
    bad: list = []
    for app, p in ogc["per_app"].items():
        for pol, payload in p.items():
            expected = payload["slope_4MB_to_8MB"] - payload["slope_1MB_to_4MB"]
            actual = payload["curvature_at_4MB"]
            if not math.isclose(expected, actual, abs_tol=CURV_TOL):
                bad.append((app, pol, expected, actual))
    assert not bad, f"OGC curvature math violations: {bad}"


# ---------------------------------------------------------------------------
# OGES internal (4)
# ---------------------------------------------------------------------------


def test_oges_per_app_keys_and_comparison_count(oges: dict) -> None:
    declared_apps = set(oges["meta"]["apps"])
    actual_apps = set(oges["per_app"].keys())
    assert declared_apps == actual_apps, (
        f"meta.apps != per_app keys:\n  meta: {declared_apps}\n  per_app: {actual_apps}"
    )
    bad = [
        (app, len(p["comparisons"]))
        for app, p in oges["per_app"].items()
        if len(p["comparisons"]) != N_COMPARISONS_PER_APP
    ]
    assert not bad, f"per_app comparison count != {N_COMPARISONS_PER_APP}: {bad}"


def test_oges_cliffs_delta_antisymmetric(oges: dict) -> None:
    bad: list = []
    for app, p in oges["per_app"].items():
        pair_map = {(c["a"], c["b"]): c["cliffs_delta_a_minus_b"] for c in p["comparisons"]}
        for (a, b), d_ab in pair_map.items():
            d_ba = pair_map.get((b, a))
            if d_ba is None:
                bad.append((app, a, b, "no_reverse_pair"))
                continue
            if abs(d_ab + d_ba) > CLIFFS_ANTISYM_TOL:
                bad.append((app, a, b, d_ab, d_ba))
    assert not bad, f"cliffs_delta antisymmetry violations: {bad}"


def test_oges_mannwhitney_symmetric_and_magnitude_correct(oges: dict) -> None:
    thresholds = oges["meta"]["cliffs_thresholds"]

    def expected_mag(abs_d: float) -> str:
        if abs_d >= thresholds["large"]:
            return "large"
        if abs_d >= thresholds["medium"]:
            return "medium"
        if abs_d >= thresholds["small"]:
            return "small"
        return "negligible"

    sym_bad: list = []
    mag_bad: list = []
    for app, p in oges["per_app"].items():
        pair_map = {(c["a"], c["b"]): c for c in p["comparisons"]}
        for (a, b), c1 in pair_map.items():
            c2 = pair_map.get((b, a))
            if c2 is not None:
                if abs(c1["mannwhitney_p"] - c2["mannwhitney_p"]) > P_SYM_TOL:
                    sym_bad.append((app, a, b, c1["mannwhitney_p"], c2["mannwhitney_p"]))
            expected = expected_mag(abs(c1["cliffs_delta_a_minus_b"]))
            if expected != c1["magnitude"]:
                mag_bad.append(
                    (app, a, b, c1["cliffs_delta_a_minus_b"], c1["magnitude"], expected)
                )
    assert not sym_bad, f"mannwhitney_p symmetry violations: {sym_bad}"
    assert not mag_bad, f"magnitude classification violations: {mag_bad}"


def test_oges_large_negative_deltas_is_correct_subset(oges: dict) -> None:
    expected = {
        (app, c["a"], c["b"])
        for app, p in oges["per_app"].items()
        for c in p["comparisons"]
        if c["magnitude"] == "large" and c["cliffs_delta_a_minus_b"] < 0
    }
    actual = {(x["app"], x["a"], x["b"]) for x in oges["large_negative_deltas"]}
    assert expected == actual, (
        f"large_negative_deltas mismatch:\n  in_actual_only: {actual - expected}\n  in_expected_only: {expected - actual}"
    )


# ---------------------------------------------------------------------------
# Cross-artifact (3)
# ---------------------------------------------------------------------------


def test_ogc_oges_app_universe_agreement(ogc: dict, oges: dict) -> None:
    ogc_apps = set(ogc["per_app"].keys())
    oges_apps = set(oges["per_app"].keys())
    oges_meta_apps = set(oges["meta"]["apps"])
    assert ogc_apps == oges_apps == oges_meta_apps == EXPECTED_APPS, (
        f"app universes differ:\n  OGC.per_app:  {ogc_apps}\n  OGES.per_app: {oges_apps}\n"
        f"  OGES.meta.apps: {oges_meta_apps}\n  expected: {EXPECTED_APPS}"
    )


def test_ogc_oges_policy_universe_agreement(ogc: dict, oges: dict) -> None:
    ogc_policies: set = set()
    for app, p in ogc["per_app"].items():
        ogc_policies.update(p.keys())
    oges_policies = set(oges["meta"]["policies"])
    assert ogc_policies == oges_policies == EXPECTED_POLICIES, (
        f"policy universes differ:\n  OGC: {ogc_policies}\n  OGES.meta.policies: {oges_policies}\n  expected: {EXPECTED_POLICIES}"
    )


def test_ogc_knee_implies_oges_stochastically_smaller_vs_lru(
    ogc: dict, oges: dict
) -> None:
    bad: list = []
    for app, ogc_pa in ogc["per_app"].items():
        oges_pa = oges["per_app"].get(app, {"comparisons": []})
        pair_map = {(c["a"], c["b"]): c for c in oges_pa["comparisons"]}
        for pol, payload in ogc_pa.items():
            if not payload.get("knee_present"):
                continue
            if pol == "LRU":
                continue
            c = pair_map.get((pol, "LRU"))
            if c is None:
                bad.append((app, pol, "no_oges_pair_vs_LRU"))
                continue
            if c["stochastically_smaller"] != pol:
                bad.append(
                    (app, pol, c["stochastically_smaller"], c["cliffs_delta_a_minus_b"])
                )
    assert not bad, (
        "OGC knee_present but OGES does not show policy stochastically_smaller "
        f"vs LRU: {bad}"
    )
