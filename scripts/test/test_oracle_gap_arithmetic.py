"""Gate 136 — oracle_gap.json arithmetic + is_winner + summary rollups.

oracle_gap is the canonical observation pipeline that nearly every
downstream artifact depends on (paper_baseline_table, lit-faithfulness,
per_graph_app_stability, oracle_gap_auc, cache_sensitivity_slope, ...).
Gates 130/132/133/134/135 already lock various downstream artifacts'
parity WITH oracle_gap, but no gate locks oracle_gap's OWN arithmetic
end-to-end: gap_pp derivation, oracle field correctness, is_winner
flag, n_policies_in_cell, and the rich `summary` statistical rollups
(mean/median/max/p90/wins per policy × family × regime).

If oracle_gap drifts, every downstream artifact silently drifts with
it — and the per-artifact gates would still pass because the drift
propagates uniformly. This gate is the bedrock.

Invariants (21 tests, 5 groups):

structural —
* every row carries the required fields
* every (graph, app, l3) cell has exactly the same policy set
* policies drawn from the documented set
* families drawn from the documented set
* regimes drawn from the documented set

gap_pp + oracle field —
* gap_pp == round((miss_rate - min(miss_rate per cell)) * 100, 3)
* oracle field == min(miss_rate per cell)
* gap_pp >= 0 for every row (oracle is the minimum)
* n_policies_in_cell == actual policy count per cell

is_winner derivation —
* is_winner='1' iff miss_rate == min(miss_rate per cell) strictly
  (NOT gap_pp == 0.0 — small-graph cells have multiple policies that
  round to gap_pp=0.0 but only the strict-min is_winner). Gate 135
  is the broader TIE_TOL semantic; here we lock the strict semantic.
* every cell has exactly one strict winner (no ties in raw miss_rate)
* every cell has at least one is_winner='1' row

summary count rollups —
* summary.n_rows == len(rows)
* summary.n_cells == #unique (graph, app, l3)
* summary.overall_by_policy[pol].n == count of rows with policy=pol
* summary.overall_by_policy[pol].wins == #is_winner rows for policy
* per-(policy, family) and per-(policy, regime) counts roll up

summary statistical rollups —
* mean/median/max/p90 reproduce from gap_pp values per group, to
  1e-3 tolerance (matches the 3-decimal rounding in oracle_gap)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

REQUIRED_FIELDS = frozenset({
    "app", "family", "gap_pp", "graph", "is_winner", "l3_size",
    "miss_rate", "n_policies_in_cell", "oracle", "policy", "regime",
})
POLICIES = frozenset({"LRU", "SRRIP", "GRASP", "POPT"})
FAMILIES = frozenset({"citation", "mesh", "road", "social", "web"})
REGIMES = frozenset({"tiny", "small", "large"})
EPS_PP = 1e-3       # matches 3-decimal rounding in gap_pp
EPS_MISS = 1e-9     # raw miss_rate equality tolerance


def _p90_nearest(vals: list[float]) -> float:
    """oracle_gap.summary uses nearest-rank percentile: round(0.9*(n-1))."""
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n == 1:
        return s[0]
    idx = round(0.9 * (n - 1))
    return s[idx]


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def rows(og) -> list:
    return og["rows"]


@pytest.fixture(scope="module")
def by_cell(rows) -> dict:
    out = defaultdict(dict)
    for r in rows:
        out[(r["graph"], r["app"], r["l3_size"])][r["policy"]] = r
    return dict(out)


# ---------- Group 1: structural ----------

def test_every_row_has_required_fields(rows):
    bad = [(i, sorted(REQUIRED_FIELDS - set(r.keys())))
           for i, r in enumerate(rows)
           if not REQUIRED_FIELDS.issubset(set(r.keys()))]
    assert not bad, bad[:3]


def test_every_cell_has_same_policy_set(by_cell):
    """Every cell must observe all 4 raw policies — partial observation
    would make winner identification ambiguous."""
    bad = [(cell, sorted(policies.keys()))
           for cell, policies in by_cell.items()
           if set(policies.keys()) != POLICIES]
    assert not bad, bad[:3]


def test_policies_drawn_from_documented_set(rows):
    bad = sorted({r["policy"] for r in rows} - POLICIES)
    assert not bad, bad


def test_families_drawn_from_documented_set(rows):
    bad = sorted({r["family"] for r in rows} - FAMILIES)
    assert not bad, bad


def test_regimes_drawn_from_documented_set(rows):
    bad = sorted({r["regime"] for r in rows} - REGIMES)
    assert not bad, bad


# ---------- Group 2: gap_pp + oracle field arithmetic ----------

def test_gap_pp_reproduces_from_miss_rate(by_cell):
    mism = []
    for cell, pol_rows in by_cell.items():
        miss_by = {p: float(r["miss_rate"]) for p, r in pol_rows.items()}
        oracle_miss = min(miss_by.values())
        for p, r in pol_rows.items():
            want = round((miss_by[p] - oracle_miss) * 100.0, 3)
            got = float(r["gap_pp"])
            if abs(got - want) > EPS_PP:
                mism.append((cell, p, got, want))
    assert not mism, mism[:3]


def test_oracle_field_is_per_cell_min_miss_rate(by_cell):
    mism = []
    for cell, pol_rows in by_cell.items():
        miss_by = {p: float(r["miss_rate"]) for p, r in pol_rows.items()}
        oracle_miss = min(miss_by.values())
        for p, r in pol_rows.items():
            got = float(r["oracle"])
            if abs(got - oracle_miss) > EPS_MISS:
                mism.append((cell, p, got, oracle_miss))
    assert not mism, mism[:3]


def test_gap_pp_nonnegative(rows):
    bad = [(r["graph"], r["app"], r["l3_size"], r["policy"], r["gap_pp"])
           for r in rows if float(r["gap_pp"]) < 0]
    assert not bad, bad[:3]


def test_n_policies_in_cell_matches_actual(rows, by_cell):
    mism = []
    for r in rows:
        cell = (r["graph"], r["app"], r["l3_size"])
        actual = len(by_cell[cell])
        reported = int(r["n_policies_in_cell"])
        if actual != reported:
            mism.append((cell, r["policy"], actual, reported))
    assert not mism, mism[:3]


# ---------- Group 3: is_winner derivation (strict) ----------

def test_is_winner_strictly_equals_min_miss_rate(by_cell):
    """is_winner='1' iff miss_rate == min(miss_rate per cell) strictly.
    Note: this is the STRICT semantic — multiple policies may have
    gap_pp == 0.0 when miss rates differ in their 7th decimal, but
    only the strict-min policy gets is_winner='1'. Gate 135 covers
    the broader TIE_TOL semantic used by per_graph_app_stability."""
    mism = []
    for cell, pol_rows in by_cell.items():
        miss_by = {p: float(r["miss_rate"]) for p, r in pol_rows.items()}
        oracle_miss = min(miss_by.values())
        for p, r in pol_rows.items():
            is_w = str(r["is_winner"]) == "1"
            want = (miss_by[p] == oracle_miss)
            if is_w != want:
                mism.append((cell, p, is_w, want, miss_by[p], oracle_miss))
    assert not mism, mism[:3]


def test_every_cell_has_at_least_one_winner(by_cell):
    bad = [cell for cell, pol_rows in by_cell.items()
           if not any(str(r["is_winner"]) == "1" for r in pol_rows.values())]
    assert not bad, bad[:3]


def test_every_winner_has_gap_pp_zero(rows):
    """is_winner='1' implies gap_pp == 0 (strict consequence of the
    is_winner derivation)."""
    bad = [(r["graph"], r["app"], r["l3_size"], r["policy"], r["gap_pp"])
           for r in rows
           if str(r["is_winner"]) == "1" and float(r["gap_pp"]) != 0.0]
    assert not bad, bad[:3]


# ---------- Group 4: summary count rollups ----------

def test_summary_n_rows(og, rows):
    assert og["summary"]["n_rows"] == len(rows)


def test_summary_n_cells(og, by_cell):
    assert og["summary"]["n_cells"] == len(by_cell)


def test_summary_overall_by_policy_n_counts(og, rows):
    """summary.overall_by_policy[pol].n == count of rows with that policy."""
    actual = defaultdict(int)
    for r in rows:
        actual[r["policy"]] += 1
    for pol, stats in og["summary"]["overall_by_policy"].items():
        assert stats["n"] == actual[pol], (pol, stats["n"], actual[pol])


def test_summary_overall_by_policy_wins(og, rows):
    """summary.overall_by_policy[pol].wins == count of is_winner='1' rows
    for that policy (paper headline: which policy wins how many cells)."""
    actual = defaultdict(int)
    for r in rows:
        if str(r["is_winner"]) == "1":
            actual[r["policy"]] += 1
    for pol, stats in og["summary"]["overall_by_policy"].items():
        assert stats["wins"] == actual[pol], (pol, stats["wins"], actual[pol])


def test_summary_by_policy_family_n_counts(og, rows):
    actual = defaultdict(int)
    for r in rows:
        actual[f"{r['policy']}/{r['family']}"] += 1
    for key, stats in og["summary"]["by_policy_family"].items():
        assert stats["n"] == actual[key], (key, stats["n"], actual[key])


def test_summary_by_policy_regime_n_counts(og, rows):
    actual = defaultdict(int)
    for r in rows:
        actual[f"{r['policy']}/{r['regime']}"] += 1
    for key, stats in og["summary"]["by_policy_regime"].items():
        assert stats["n"] == actual[key], (key, stats["n"], actual[key])


# ---------- Group 5: summary statistical rollups ----------

def test_summary_overall_by_policy_stats_reproduce(og, rows):
    """mean/median/max/p90 reproduce from gap_pp per policy, to 1e-3
    (matches the 3-decimal rounding of gap_pp). p90 uses the
    nearest-rank method: round(0.9 * (n-1))."""
    gap_by_pol = defaultdict(list)
    for r in rows:
        gap_by_pol[r["policy"]].append(float(r["gap_pp"]))
    mism = []
    for pol, stats in og["summary"]["overall_by_policy"].items():
        vals = gap_by_pol[pol]
        for stat_name, py_val in (("mean", sum(vals) / len(vals)),
                                  ("median", median(vals)),
                                  ("max", max(vals)),
                                  ("p90", _p90_nearest(vals))):
            got = stats[stat_name]
            if abs(got - py_val) > EPS_PP:
                mism.append((pol, stat_name, got, py_val))
    assert not mism, mism[:3]


def test_summary_by_policy_family_stats_reproduce(og, rows):
    gap_by = defaultdict(list)
    for r in rows:
        gap_by[f"{r['policy']}/{r['family']}"].append(float(r["gap_pp"]))
    mism = []
    for key, stats in og["summary"]["by_policy_family"].items():
        vals = gap_by[key]
        for stat_name, py_val in (("mean", sum(vals) / len(vals)),
                                  ("median", median(vals)),
                                  ("max", max(vals)),
                                  ("p90", _p90_nearest(vals))):
            got = stats[stat_name]
            if abs(got - py_val) > EPS_PP:
                mism.append((key, stat_name, got, py_val))
    assert not mism, mism[:3]


def test_summary_by_policy_regime_stats_reproduce(og, rows):
    gap_by = defaultdict(list)
    for r in rows:
        gap_by[f"{r['policy']}/{r['regime']}"].append(float(r["gap_pp"]))
    mism = []
    for key, stats in og["summary"]["by_policy_regime"].items():
        vals = gap_by[key]
        for stat_name, py_val in (("mean", sum(vals) / len(vals)),
                                  ("median", median(vals)),
                                  ("max", max(vals)),
                                  ("p90", _p90_nearest(vals))):
            got = stats[stat_name]
            if abs(got - py_val) > EPS_PP:
                mism.append((key, stat_name, got, py_val))
    assert not mism, mism[:3]
