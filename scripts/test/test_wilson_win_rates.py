"""Gate: Wilson 95% CIs on policy win-counts produce expected sign claims.

Win-count narratives ("GRASP wins 17/20 cc cells", "POPT wins 20/28 pr
cells") are evidence-by-counting arguments. This gate pins the
**CI-strict** subset of those claims — the ones whose Wilson 95% CIs
exclude either the 0.5 (majority) or the 0.25 (random-baseline-with-
four-policies) thresholds.

The CI-loose comparisons (e.g. bc/GRASP CI hi above 0.5) are
explicitly tested as *not* strict-majority, so future runs that
silently elevate them to "GRASP wins bc" will trip the gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "wilson_win_rates.json"

MAJORITY = 0.5
CHANCE_4POL = 0.25

REQUIRED_APPS = ("pr", "bc", "cc", "bfs", "sssp")
REQUIRED_POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(
            f"missing {PAYLOAD}; run `make lit-wilson-wins` "
            f"(or `make lit-oracle-gap lit-wilson-wins`)."
        )
    return json.loads(PAYLOAD.read_text())


def _cell(payload: dict, scope: str, key: str, policy: str) -> dict:
    bucket = payload[scope]
    if scope == "overall":
        return bucket[policy]
    return bucket[key][policy]


def test_schema_has_three_scopes(payload):
    for scope in ("overall", "per_app", "per_family"):
        assert scope in payload, f"missing scope {scope}"
    for app in REQUIRED_APPS:
        assert app in payload["per_app"], f"missing per_app[{app}]"


def test_every_cell_has_wilson_fields(payload):
    for scope_name in ("per_app", "per_family"):
        for key, pols in payload[scope_name].items():
            for pol, stats in pols.items():
                for field in ("wins", "total", "p_hat", "ci_lo", "ci_hi"):
                    assert field in stats, (
                        f"{scope_name}[{key}][{pol}] missing {field}"
                    )
                assert 0.0 <= stats["ci_lo"] <= stats["p_hat"] + 1e-9, (
                    f"CI lo > p_hat: {scope_name}[{key}][{pol}] {stats}"
                )
                assert stats["p_hat"] - 1e-9 <= stats["ci_hi"] <= 1.0, (
                    f"CI hi < p_hat: {scope_name}[{key}][{pol}] {stats}"
                )


def test_pr_popt_is_ci_strict_majority(payload):
    """POPT on pr: 20/28 wins. CI lower bound must exceed 0.5."""
    stats = _cell(payload, "per_app", "pr", "POPT")
    assert stats["wins"] >= 18, f"pr/POPT wins regressed: {stats}"
    assert stats["ci_lo"] > MAJORITY, (
        f"pr/POPT lost CI-strict majority: CI lo {stats['ci_lo']} ≤ {MAJORITY}"
    )


def test_cc_grasp_is_ci_strict_majority_and_above_chance(payload):
    """GRASP on cc: 17/20 wins. CI must exclude both 0.5 and 0.25."""
    stats = _cell(payload, "per_app", "cc", "GRASP")
    assert stats["wins"] >= 15, f"cc/GRASP wins regressed: {stats}"
    assert stats["ci_lo"] > MAJORITY, (
        f"cc/GRASP CI lo {stats['ci_lo']} ≤ {MAJORITY}"
    )
    assert stats["ci_lo"] > CHANCE_4POL, (
        f"cc/GRASP CI lo {stats['ci_lo']} ≤ {CHANCE_4POL}"
    )


def test_cc_popt_is_ci_strict_below_chance(payload):
    """POPT on cc: 0/20. CI upper bound must be well below 0.25."""
    stats = _cell(payload, "per_app", "cc", "POPT")
    assert stats["wins"] == 0, f"cc/POPT had wins; data drift: {stats}"
    assert stats["ci_hi"] < CHANCE_4POL, (
        f"cc/POPT CI hi {stats['ci_hi']} ≥ chance baseline {CHANCE_4POL}"
    )


def test_bc_grasp_is_above_chance_but_not_strict_majority(payload):
    """GRASP on bc: 15/23. Above-chance (CI lo > 0.25) but CI hi spans 0.5.

    This pins the *honest* status of the "GRASP wins bc by count" claim:
    real above-random-noise effect, but does not survive a strict-
    majority Wilson test. If future runs strengthen the claim to strict
    majority the test will need updating — that is a deliberate
    forcing-function.
    """
    stats = _cell(payload, "per_app", "bc", "GRASP")
    assert stats["ci_lo"] > CHANCE_4POL, (
        f"bc/GRASP no longer above chance: {stats}"
    )
    if stats["ci_lo"] > MAJORITY:
        pytest.fail(
            f"bc/GRASP NEWLY strict-majority (CI lo {stats['ci_lo']}). "
            f"Update gate + claim docs."
        )


def test_lru_is_never_ci_strict_majority(payload):
    """LRU CI upper bound stays below 0.5 on every kernel (no scope in
    which LRU is plausibly a majority winner). The looser bound is
    correct: small-n cells (e.g. LRU 2/20 on cc) can include 0.25 in
    their CI, but never 0.5.
    """
    for app in REQUIRED_APPS:
        stats = _cell(payload, "per_app", app, "LRU")
        assert stats["ci_hi"] < MAJORITY, (
            f"LRU on {app} CI hi {stats['ci_hi']} ≥ {MAJORITY}: "
            f"{stats['wins']}/{stats['total']}"
        )


def test_sssp_popt_vs_grasp_ties_are_not_ci_separable(payload):
    """sssp POPT (8/20) vs GRASP (7/20): CIs overlap heavily.

    Pin the *non-claim* explicitly so future code that elevates this
    near-tie to a winner sets off the gate.
    """
    popt = _cell(payload, "per_app", "sssp", "POPT")
    grasp = _cell(payload, "per_app", "sssp", "GRASP")
    assert popt["ci_lo"] < grasp["ci_hi"], (
        f"sssp POPT/GRASP CIs no longer overlap: POPT {popt}, GRASP {grasp}"
    )
    assert popt["ci_lo"] < MAJORITY and grasp["ci_lo"] < MAJORITY, (
        f"sssp got a CI-strict majority: POPT {popt}, GRASP {grasp}"
    )


def test_road_family_popt_dominates(payload):
    """Per-family scope: POPT on road family wins by an obvious margin."""
    if "road" not in payload["per_family"]:
        pytest.skip("road family absent from corpus this run")
    popt = payload["per_family"]["road"].get("POPT")
    if popt is None:
        pytest.skip("POPT absent from road family this run")
    # road family is the cleanest POPT signal in the corpus
    assert popt["p_hat"] >= 0.45, (
        f"road/POPT win-rate dropped to {popt['p_hat']}: {popt}"
    )


def test_wilson_formula_matches_known_reference(payload):
    """Independent re-derivation: 17/20 must give CI ≈ [0.640, 0.948]."""
    z = 1.959963984540054
    wins, total = 17, 20
    p = wins / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denom
    lo = center - margin
    hi = center + margin
    stats = _cell(payload, "per_app", "cc", "GRASP")
    assert stats["wins"] == 17 and stats["total"] == 20, (
        f"input changed: {stats}"
    )
    assert abs(stats["ci_lo"] - lo) < 1e-3, (
        f"CI lo {stats['ci_lo']} ≠ reference {lo:.4f}"
    )
    assert abs(stats["ci_hi"] - hi) < 1e-3, (
        f"CI hi {stats['ci_hi']} ≠ reference {hi:.4f}"
    )


def test_overall_grasp_above_chance_with_ci(payload):
    """Overall GRASP wins 56/114 — CI lo must exceed 0.25 (chance)."""
    stats = _cell(payload, "overall", "", "GRASP")
    assert stats["wins"] >= 50, f"overall GRASP wins regressed: {stats}"
    assert stats["ci_lo"] > CHANCE_4POL, (
        f"overall GRASP CI lo {stats['ci_lo']} ≤ {CHANCE_4POL}"
    )
