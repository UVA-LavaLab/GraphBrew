"""Gate 142+ — wilson_win_rates derivation parity.

wilson_win_rates.json is derived from oracle_gap.json by:

  1. Aggregate per-(scope, policy) win counts using is_winner flag
     across three scopes: overall (all rows), per_app, per_family
  2. For each (scope, policy) cell with total > 0, compute the Wilson
     score interval at 95% (z = 1.959963984540054):

        p_hat   = wins / total
        center  = (p_hat + z²/(2n)) / (1 + z²/n)
        margin  = z·sqrt(p_hat(1-p_hat)/n + z²/(4n²)) / (1 + z²/n)
        ci_lo   = max(0, center - margin)
        ci_hi   = min(1, center + margin)

  3. Round p_hat / ci_lo / ci_hi to 4 decimals on output

Why this gate matters
---------------------
Wilson intervals underpin EVERY CI-strict claim in the paper. If the
aggregation rule (what counts as a 'win'), the z constant, the
clamping at [0,1], or the rounding policy ever drifts, then every
downstream claim that says 'policy A's CI separates from policy B's
CI' will become incorrect without any other gate noticing.

This test pins:

  - the canonical z constant (1.959963984540054, the standard normal
    quantile for two-sided 95%, NOT the often-quoted 1.96)
  - the Wilson score formula with continuity-correction-FREE form
  - the [0,1] clamping rule
  - the three-scope aggregation (overall, per_app, per_family)
  - the relationship to cohens_h_win_rates (same wins/total)

Invariants (19 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, overall, per_app, per_family
  2. meta.source == oracle_gap.json, method == 'wilson_score'
  3. meta.z == 1.959963984540054 (NOT 1.96)
  4. meta.ci_level == 0.95
  5. Scope keys match meta.apps / meta.families

Group B — Aggregation (oracle_gap → wins/total)
  6. overall[pol] aggregates ALL rows (across apps and families)
  7. per_app[app][pol] aggregates only rows for that app
  8. per_family[fam][pol] aggregates only rows for that family
  9. wins/total match cohens_h_win_rates per_app (cross-gate consistency)

Group C — Wilson formula reproduction
  10. p_hat == round(wins/total, 4) at 1e-6
  11. ci_lo / ci_hi match Wilson formula at 1e-3 across overall scope
  12. ci_lo / ci_hi match Wilson formula at 1e-3 across per_app scope
  13. ci_lo / ci_hi match Wilson formula at 1e-3 across per_family scope

Group D — Numerical sanity
  14. ci_lo <= p_hat <= ci_hi for every cell
  15. ci_lo >= 0.0 and ci_hi <= 1.0 (clamping invariant)
  16. ci_hi > ci_lo when 0 < total
  17. For wins == 0: p_hat == 0 and ci_lo == 0; for wins == total: ci_hi == 1

Group E — Cross-scope conservation
  18. Sum of per_app totals[pol] == overall[pol] total (every row in
      exactly one (app) bucket)
  19. Sum of per_family totals[pol] == overall[pol] total (every row
      in exactly one (family) bucket)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

Z_95 = 1.959963984540054
EPS_RATE = 1e-6
EPS_CI = 1e-3


def _is_winner(row: dict) -> bool:
    val = row.get("is_winner")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return int(val) == 1
    if isinstance(val, str):
        return val.strip() in {"1", "true", "True"}
    return False


def _wilson(wins: int, total: int, z: float = Z_95) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 1.0
    p = wins / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap.json").read_text())


@pytest.fixture(scope="module")
def ww() -> dict:
    return json.loads((WIKI_DATA / "wilson_win_rates.json").read_text())


@pytest.fixture(scope="module")
def expected(og) -> dict:
    overall = defaultdict(lambda: [0, 0])
    by_app = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_fam = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in og["rows"]:
        w = 1 if _is_winner(r) else 0
        overall[r["policy"]][0] += w
        overall[r["policy"]][1] += 1
        by_app[r["app"]][r["policy"]][0] += w
        by_app[r["app"]][r["policy"]][1] += 1
        by_fam[r["family"]][r["policy"]][0] += w
        by_fam[r["family"]][r["policy"]][1] += 1
    return {
        "overall": {p: tuple(v) for p, v in overall.items()},
        "per_app": {a: {p: tuple(v) for p, v in d.items()} for a, d in by_app.items()},
        "per_family": {f: {p: tuple(v) for p, v in d.items()} for f, d in by_fam.items()},
    }


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(ww):
    assert set(ww.keys()) >= {"meta", "overall", "per_app", "per_family"}


def test_meta_source_and_method(ww):
    assert ww["meta"]["source"] == "wiki/data/oracle_gap.json"
    assert ww["meta"]["method"] == "wilson_score"


def test_meta_z_is_canonical_95(ww):
    """z == standard-normal 0.975 quantile, NOT 1.96 (load-bearing precision)."""
    assert ww["meta"]["z"] == Z_95


def test_meta_ci_level(ww):
    assert ww["meta"]["ci_level"] == 0.95


def test_per_app_per_family_keys_match_oracle(ww, expected):
    assert set(ww["per_app"].keys()) == set(expected["per_app"].keys())
    assert set(ww["per_family"].keys()) == set(expected["per_family"].keys())


# ─── Group B — Aggregation ───────────────────────────────────────────


def test_overall_wins_total_match(ww, expected):
    mism = []
    for pol, st in ww["overall"].items():
        exp_w, exp_t = expected["overall"][pol]
        if st["wins"] != exp_w or st["total"] != exp_t:
            mism.append((pol, (st["wins"], st["total"]), (exp_w, exp_t)))
    assert not mism, mism


def test_per_app_wins_total_match(ww, expected):
    mism = []
    for app, by_pol in ww["per_app"].items():
        for pol, st in by_pol.items():
            exp_w, exp_t = expected["per_app"][app][pol]
            if st["wins"] != exp_w or st["total"] != exp_t:
                mism.append((app, pol, (st["wins"], st["total"]), (exp_w, exp_t)))
    assert not mism, mism


def test_per_family_wins_total_match(ww, expected):
    mism = []
    for fam, by_pol in ww["per_family"].items():
        for pol, st in by_pol.items():
            exp_w, exp_t = expected["per_family"][fam][pol]
            if st["wins"] != exp_w or st["total"] != exp_t:
                mism.append((fam, pol, (st["wins"], st["total"]), (exp_w, exp_t)))
    assert not mism, mism


def test_cross_gate_consistency_with_cohens_h(ww):
    """Wilson per_app wins/total must equal cohens_h_win_rates per_app rates."""
    ch = json.loads((WIKI_DATA / "cohens_h_win_rates.json").read_text())
    mism = []
    for app, by_pol in ww["per_app"].items():
        if app not in ch["per_app"]:
            continue
        for pol, st in by_pol.items():
            ch_st = ch["per_app"][app]["rates"].get(pol)
            if ch_st is None:
                continue
            if ch_st["wins"] != st["wins"] or ch_st["total"] != st["total"]:
                mism.append((app, pol, (st["wins"], st["total"]),
                             (ch_st["wins"], ch_st["total"])))
    assert not mism, mism


# ─── Group C — Wilson formula ────────────────────────────────────────


def test_p_hat_is_rounded_quotient(ww, expected):
    mism = []
    for scope_name, scope in (
        ("overall", ww["overall"]),
    ):
        for pol, st in scope.items():
            exp_w, exp_t = expected["overall"][pol]
            exp_p = round(exp_w / exp_t, 4) if exp_t > 0 else 0.0
            if abs(st["p_hat"] - exp_p) > EPS_RATE:
                mism.append((scope_name, pol, st["p_hat"], exp_p))
    assert not mism, mism


def _check_wilson_for_scope(ww_scope: dict, exp_scope: dict, scope_name: str):
    mism = []
    for key, by_pol in ww_scope.items():
        for pol, st in by_pol.items():
            exp_w, exp_t = exp_scope[key][pol]
            _, lo, hi = _wilson(exp_w, exp_t)
            exp_lo = round(lo, 4)
            exp_hi = round(hi, 4)
            if abs(st["ci_lo"] - exp_lo) > EPS_CI or abs(st["ci_hi"] - exp_hi) > EPS_CI:
                mism.append((scope_name, key, pol,
                             (st["ci_lo"], st["ci_hi"]), (exp_lo, exp_hi)))
    return mism


def test_ci_overall_matches_wilson(ww, expected):
    mism = []
    for pol, st in ww["overall"].items():
        exp_w, exp_t = expected["overall"][pol]
        _, lo, hi = _wilson(exp_w, exp_t)
        if abs(st["ci_lo"] - round(lo, 4)) > EPS_CI or abs(st["ci_hi"] - round(hi, 4)) > EPS_CI:
            mism.append((pol, st["ci_lo"], st["ci_hi"], round(lo, 4), round(hi, 4)))
    assert not mism, mism


def test_ci_per_app_matches_wilson(ww, expected):
    mism = _check_wilson_for_scope(ww["per_app"], expected["per_app"], "per_app")
    assert not mism, mism[:5]


def test_ci_per_family_matches_wilson(ww, expected):
    mism = _check_wilson_for_scope(ww["per_family"], expected["per_family"], "per_family")
    assert not mism, mism[:5]


# ─── Group D — Numerical sanity ──────────────────────────────────────


def test_ci_brackets_p_hat(ww):
    bad = []
    for scope in ("overall", "per_app", "per_family"):
        if scope == "overall":
            for pol, st in ww[scope].items():
                if not (st["ci_lo"] - EPS_CI <= st["p_hat"] <= st["ci_hi"] + EPS_CI):
                    bad.append((scope, pol, st))
        else:
            for k, by_pol in ww[scope].items():
                for pol, st in by_pol.items():
                    if not (st["ci_lo"] - EPS_CI <= st["p_hat"] <= st["ci_hi"] + EPS_CI):
                        bad.append((scope, k, pol, st))
    assert not bad, bad[:5]


def test_ci_clamped_to_unit_interval(ww):
    bad = []
    for scope in ("overall", "per_app", "per_family"):
        if scope == "overall":
            iters = [(None, ww[scope])]
        else:
            iters = [(k, by_pol) for k, by_pol in ww[scope].items()]
        for k, by_pol in iters:
            for pol, st in by_pol.items():
                if st["ci_lo"] < 0.0 - EPS_CI or st["ci_hi"] > 1.0 + EPS_CI:
                    bad.append((scope, k, pol, st))
    assert not bad, bad[:5]


def test_ci_hi_gt_ci_lo_when_total_positive(ww):
    bad = []
    for scope_name in ("overall", "per_app", "per_family"):
        scope = ww[scope_name]
        if scope_name == "overall":
            for pol, st in scope.items():
                if st["total"] > 0 and st["ci_hi"] <= st["ci_lo"]:
                    bad.append((scope_name, pol, st))
        else:
            for k, by_pol in scope.items():
                for pol, st in by_pol.items():
                    if st["total"] > 0 and st["ci_hi"] <= st["ci_lo"]:
                        bad.append((scope_name, k, pol, st))
    assert not bad, bad[:5]


def test_edge_cases_zero_wins_and_all_wins(ww):
    """For wins=0: p_hat=0 and ci_lo=0. For wins=total: ci_hi=1."""
    bad = []
    for scope_name in ("overall", "per_app", "per_family"):
        scope = ww[scope_name]
        items = (
            [(None, scope)] if scope_name == "overall" else list(scope.items())
        )
        for k, by_pol in items:
            for pol, st in by_pol.items():
                if st["wins"] == 0 and st["total"] > 0:
                    if abs(st["p_hat"]) > EPS_RATE or abs(st["ci_lo"]) > EPS_CI:
                        bad.append(("wins=0", scope_name, k, pol, st))
                if st["wins"] == st["total"] and st["total"] > 0:
                    if abs(st["ci_hi"] - 1.0) > EPS_CI:
                        bad.append(("wins=total", scope_name, k, pol, st))
    assert not bad, bad[:5]


# ─── Group E — Cross-scope conservation ──────────────────────────────


def test_per_app_totals_sum_to_overall(ww):
    mism = []
    for pol, st in ww["overall"].items():
        per_app_total = sum(
            by_pol.get(pol, {"total": 0})["total"]
            for by_pol in ww["per_app"].values()
        )
        per_app_wins = sum(
            by_pol.get(pol, {"wins": 0})["wins"]
            for by_pol in ww["per_app"].values()
        )
        if per_app_total != st["total"] or per_app_wins != st["wins"]:
            mism.append((pol, (per_app_wins, per_app_total),
                         (st["wins"], st["total"])))
    assert not mism, mism


def test_per_family_totals_sum_to_overall(ww):
    mism = []
    for pol, st in ww["overall"].items():
        per_fam_total = sum(
            by_pol.get(pol, {"total": 0})["total"]
            for by_pol in ww["per_family"].values()
        )
        per_fam_wins = sum(
            by_pol.get(pol, {"wins": 0})["wins"]
            for by_pol in ww["per_family"].values()
        )
        if per_fam_total != st["total"] or per_fam_wins != st["wins"]:
            mism.append((pol, (per_fam_wins, per_fam_total),
                         (st["wins"], st["total"])))
    assert not mism, mism
