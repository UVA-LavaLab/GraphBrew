"""Gate 141+ — cohens_h_win_rates derivation parity.

cohens_h_win_rates.json is derived from oracle_gap.json by:

  1. Aggregate per-(app, policy) win counts using is_winner flag
     (treated as boolean: True / "1" / 1 → win)
  2. For each app and each ordered pair (a, b) of policies present
     in that app, compute Cohen's h = |2*arcsin(√p_a) - 2*arcsin(√p_b)|
  3. Classify magnitude using cumulative thresholds:
     - large    if h >= 0.8
     - medium   if h >= 0.5 (and < 0.8)
     - small    if h >= 0.2 (and < 0.5)
     - negligible otherwise
  4. favors = a if p_a > p_b else (b if p_b > p_a else 'tie')
  5. Headlines:
     - largest_per_app[app] = comparison with max h
     - large_effects = all comparisons with magnitude=='large' AND
       p_a > p_b (dominance direction), sorted desc by h

Why this gate matters
---------------------
Cohen's h is the headline effect-size statistic used to argue that
policy win-rate gaps are large in *practical* (not just statistical)
terms. Future refactors of the win-rate aggregation could silently
change which rows count as wins (e.g. by adding ties to the winner
side), which would shift p_hat values and change magnitude buckets
without any other gate noticing. This test pins:

  - the is_winner aggregation rule (raw flag, not derived from gap_pp)
  - the arcsine-transform math (vs raw Δp)
  - the magnitude bucketing
  - the largest_per_app / large_effects filter logic

Invariants (19 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, per_app, largest_per_app, large_effects
  2. meta.source == wiki/data/oracle_gap.json
  3. meta.thresholds matches Cohen 1988 (small=0.2, medium=0.5, large=0.8)
  4. meta.policies and meta.apps match the canonical 4-policy / 5-app set
  5. per_app keys subset of meta.apps

Group B — Win-rate aggregation (oracle_gap → rates)
  6. For every (app, pol) entry in rates: wins == sum(is_winner)
     and total == count
  7. p_hat == round(wins/total, 4) at 1e-6
  8. wins <= total, p_hat in [0, 1]

Group C — Cohen's h reproduction
  9. h == round(|2*arcsin(√p_a) - 2*arcsin(√p_b)|, 4) at 1e-3
     (rounding of p_hat then h compounds ≤ 1e-4 typically)
  10. h is symmetric in (a, b)
  11. h(p,p) == 0 for the implicit diagonal (and a==b is skipped)
  12. delta_p == round(p_a - p_b, 4) at 1e-6

Group D — Magnitude bucketing
  13. magnitude assignment follows cumulative thresholds exactly
  14. favors == a if p_a > p_b, b if p_b > p_a, else 'tie'

Group E — Headline collections
  15. largest_per_app[app] == argmax(comparisons, key=h) for each app
      with non-empty comparisons
  16. large_effects contains every (app, comparison) with
      magnitude=='large' AND p_a > p_b
  17. large_effects sorted descending by h
  18. No row in large_effects has p_a <= p_b (dominance direction enforced)
  19. Every large_effect entry's h >= 0.8 (large floor)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
APPS = ("pr", "bc", "cc", "bfs", "sssp")

EPS_RATE = 1e-6   # p_hat is exact division, rounded to 4dp
EPS_H = 1e-3      # h compounds rounding of p_hat then h


def _is_winner(row: dict) -> bool:
    val = row.get("is_winner")
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return int(val) == 1
    if isinstance(val, str):
        return val.strip() in {"1", "true", "True"}
    return False


def _cohens_h(p1: float, p2: float) -> float:
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return abs(2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2)))


def _magnitude(h: float) -> str:
    if h >= 0.8:
        return "large"
    if h >= 0.5:
        return "medium"
    if h >= 0.2:
        return "small"
    return "negligible"


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap.json").read_text())


@pytest.fixture(scope="module")
def ch() -> dict:
    return json.loads((WIKI_DATA / "cohens_h_win_rates.json").read_text())


@pytest.fixture(scope="module")
def expected_rates(og) -> dict:
    """{app: {pol: (wins, total, p_hat_raw)}} from oracle_gap rows."""
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in og["rows"]:
        w = 1 if _is_winner(r) else 0
        by[r["app"]][r["policy"]][0] += w
        by[r["app"]][r["policy"]][1] += 1
    out = {}
    for app, pol_map in by.items():
        out[app] = {
            pol: (wins, total, wins / total)
            for pol, (wins, total) in pol_map.items()
        }
    return out


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(ch):
    assert set(ch.keys()) >= {"meta", "per_app", "largest_per_app", "large_effects"}


def test_meta_source(ch):
    assert ch["meta"]["source"] == "wiki/data/oracle_gap.json"


def test_meta_thresholds_pin_cohen_1988(ch):
    th = ch["meta"]["thresholds"]
    assert th["small"] == 0.2
    assert th["medium"] == 0.5
    assert th["large"] == 0.8


def test_meta_policies_and_apps_canonical(ch):
    assert sorted(ch["meta"]["policies"]) == sorted(POLICIES)
    assert sorted(ch["meta"]["apps"]) == sorted(APPS)


def test_per_app_subset_of_meta_apps(ch):
    assert set(ch["per_app"].keys()) <= set(APPS)


# ─── Group B — Win-rate aggregation ──────────────────────────────────


def test_rates_wins_total_match_oracle_gap(ch, expected_rates):
    mism = []
    for app, payload in ch["per_app"].items():
        for pol, st in payload["rates"].items():
            exp_w, exp_t, _ = expected_rates[app][pol]
            if st["wins"] != exp_w or st["total"] != exp_t:
                mism.append((app, pol, (st["wins"], st["total"]), (exp_w, exp_t)))
    assert not mism, mism


def test_p_hat_is_rounded_quotient(ch, expected_rates):
    mism = []
    for app, payload in ch["per_app"].items():
        for pol, st in payload["rates"].items():
            _, _, exp = expected_rates[app][pol]
            if abs(st["p_hat"] - round(exp, 4)) > EPS_RATE:
                mism.append((app, pol, st["p_hat"], round(exp, 4)))
    assert not mism, mism


def test_wins_le_total_and_p_hat_in_unit_interval(ch):
    bad = []
    for app, payload in ch["per_app"].items():
        for pol, st in payload["rates"].items():
            if st["wins"] > st["total"] or not (0.0 <= st["p_hat"] <= 1.0):
                bad.append((app, pol, st))
    assert not bad, bad


# ─── Group C — Cohen's h reproduction ────────────────────────────────


def test_h_reproduces_arcsine_delta(ch, expected_rates):
    mism = []
    for app, payload in ch["per_app"].items():
        for c in payload["comparisons"]:
            _, _, p_a_raw = expected_rates[app][c["a"]]
            _, _, p_b_raw = expected_rates[app][c["b"]]
            expected = round(_cohens_h(p_a_raw, p_b_raw), 4)
            if abs(c["h"] - expected) > EPS_H:
                mism.append((app, c["a"], c["b"], c["h"], expected))
    assert not mism, mism[:5]


def test_h_is_symmetric(ch):
    by = defaultdict(dict)
    for app, payload in ch["per_app"].items():
        for c in payload["comparisons"]:
            by[app][(c["a"], c["b"])] = c["h"]
    mism = []
    for app, pairs in by.items():
        for (a, b), h_ab in pairs.items():
            h_ba = pairs.get((b, a))
            if h_ba is None or abs(h_ab - h_ba) > EPS_H:
                mism.append((app, a, b, h_ab, h_ba))
    assert not mism, mism[:5]


def test_no_self_pair_present(ch):
    bad = []
    for app, payload in ch["per_app"].items():
        for c in payload["comparisons"]:
            if c["a"] == c["b"]:
                bad.append((app, c))
    assert not bad, bad


def test_delta_p_is_rounded_difference(ch, expected_rates):
    mism = []
    for app, payload in ch["per_app"].items():
        for c in payload["comparisons"]:
            _, _, p_a_raw = expected_rates[app][c["a"]]
            _, _, p_b_raw = expected_rates[app][c["b"]]
            expected = round(p_a_raw - p_b_raw, 4)
            if abs(c["delta_p"] - expected) > EPS_RATE:
                mism.append((app, c, expected))
    assert not mism, mism[:5]


# ─── Group D — Magnitude bucketing ───────────────────────────────────


def test_magnitude_bucketing_is_cumulative(ch):
    mism = []
    for app, payload in ch["per_app"].items():
        for c in payload["comparisons"]:
            expected = _magnitude(c["h"])
            if c["magnitude"] != expected:
                mism.append((app, c["a"], c["b"], c["h"], c["magnitude"], expected))
    assert not mism, mism[:5]


def test_favors_field_follows_p_a_vs_p_b(ch):
    mism = []
    for app, payload in ch["per_app"].items():
        for c in payload["comparisons"]:
            if c["p_a"] > c["p_b"]:
                exp = c["a"]
            elif c["p_b"] > c["p_a"]:
                exp = c["b"]
            else:
                exp = "tie"
            if c["favors"] != exp:
                mism.append((app, c, exp))
    assert not mism, mism[:5]


# ─── Group E — Headline collections ──────────────────────────────────


def test_largest_per_app_is_argmax_h(ch):
    mism = []
    for app, payload in ch["per_app"].items():
        if not payload["comparisons"]:
            continue
        expected_h = max(c["h"] for c in payload["comparisons"])
        got = ch["largest_per_app"].get(app)
        assert got is not None, f"missing largest_per_app[{app}]"
        if abs(got["h"] - expected_h) > EPS_H:
            mism.append((app, got["h"], expected_h))
    assert not mism, mism


def test_large_effects_match_filter(ch):
    """large_effects = every (app, comparison) with magnitude==large AND p_a > p_b."""
    expected = []
    for app, payload in ch["per_app"].items():
        for c in payload["comparisons"]:
            if c["magnitude"] == "large" and c["p_a"] > c["p_b"]:
                expected.append((app, c["a"], c["b"], c["h"]))
    got = [(r["app"], r["a"], r["b"], r["h"]) for r in ch["large_effects"]]
    assert sorted(got) == sorted(expected)


def test_large_effects_sorted_desc_by_h(ch):
    hs = [r["h"] for r in ch["large_effects"]]
    assert hs == sorted(hs, reverse=True)


def test_large_effects_enforce_dominance_direction(ch):
    bad = [r for r in ch["large_effects"] if r["p_a"] <= r["p_b"]]
    assert not bad, bad[:5]


def test_large_effects_meet_large_floor(ch):
    bad = [r for r in ch["large_effects"] if r["h"] < 0.8 - EPS_H]
    assert not bad, bad[:5]
