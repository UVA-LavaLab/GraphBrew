"""Cross-artifact integrity tests for the oracle-gap source artifacts.

Gate 106 locks two invariants:

1.  ``wiki/data/oracle_gap.json`` is internally consistent. The 456 rows cover
    114 unique (graph, app, l3_size) cells with all four policies present, the
    per-cell ``oracle`` value equals the minimum policy miss-rate, the
    ``is_winner`` flag identifies exactly the rows that match the oracle, and
    ``gap_pp`` recomputes from ``(miss_rate - oracle) * 100`` everywhere.

2.  ``wiki/data/oracle_gap_by_app.json`` is a faithful aggregation of the
    oracle_gap rows. Both the per-(policy, app) summary stats and the per-app
    ranked tables match what we can recompute from the source rows.

These artifacts feed everything downstream (regression_budget, popt_vs_grasp,
family_geomean_improvement). Locking the source here keeps every later gate
honest.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_GAP_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
ORACLE_GAP_BY_APP_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap_by_app.json"

EXPECTED_ROW_COUNT = 456
EXPECTED_CELL_COUNT = 114
EXPECTED_POLICIES = {"GRASP", "LRU", "POPT", "SRRIP"}
EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_FAMILIES = {"citation", "mesh", "road", "social", "web"}
EXPECTED_REGIMES = {"tiny", "small", "large"}
EXPECTED_L3_SIZES = {"4kB", "16kB", "64kB", "256kB", "1MB", "4MB", "8MB"}

CELL_KEYS = ("graph", "app", "l3_size")
ROW_REQUIRED_FIELDS = (
    "app",
    "family",
    "gap_pp",
    "graph",
    "is_winner",
    "l3_size",
    "miss_rate",
    "n_policies_in_cell",
    "oracle",
    "policy",
    "regime",
)
BY_POLICY_APP_REQUIRED = ("max", "mean", "median", "n", "p90", "wins")
BY_APP_RANKING_REQUIRED = (
    "max_gap_pp",
    "mean_gap_pp",
    "median_gap_pp",
    "n",
    "p90_gap_pp",
    "policy",
    "wins",
)

GAP_RECOMPUTE_TOL = 1e-3
ORACLE_MIN_TOL = 1e-9
MEAN_TOL = 0.001
PCTL_TOL = 0.001


@pytest.fixture(scope="module")
def oracle_gap() -> dict:
    return json.loads(ORACLE_GAP_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_gap_by_app() -> dict:
    return json.loads(ORACLE_GAP_BY_APP_PATH.read_text())


@pytest.fixture(scope="module")
def og_rows(oracle_gap: dict) -> list:
    return oracle_gap["rows"]


@pytest.fixture(scope="module")
def og_by_cell(og_rows: list) -> dict:
    by_cell: dict = defaultdict(list)
    for row in og_rows:
        by_cell[tuple(row[k] for k in CELL_KEYS)].append(row)
    return dict(by_cell)


def _median_linear(values: list[float]) -> float:
    """Standard median (average of the middle two values for even-length lists),
    matching ``statistics.median`` and numpy's linear interpolation at q=0.5."""
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    mid = (n - 1) / 2
    lo = int(math.floor(mid))
    hi = int(math.ceil(mid))
    if lo == hi:
        return s[lo]
    return (s[lo] + s[hi]) / 2.0


def _p90_higher(values: list[float]) -> float:
    """Numpy's ``method='higher'`` percentile at q=0.9, i.e. the smallest value
    whose rank is at least ``ceil(0.9 * (n - 1))``."""
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(math.ceil((len(s) - 1) * 0.9))
    return s[idx]


# ---------------------------------------------------------------------------
# Group A — oracle_gap internal structural (4)
# ---------------------------------------------------------------------------


def test_oracle_gap_row_and_cell_counts(oracle_gap: dict, og_rows: list, og_by_cell: dict) -> None:
    summary = oracle_gap["summary"]
    assert len(og_rows) == EXPECTED_ROW_COUNT, (
        f"oracle_gap.rows len={len(og_rows)} expected={EXPECTED_ROW_COUNT}"
    )
    assert summary["n_rows"] == EXPECTED_ROW_COUNT, (
        f"summary.n_rows={summary['n_rows']} expected={EXPECTED_ROW_COUNT}"
    )
    assert len(og_by_cell) == EXPECTED_CELL_COUNT, (
        f"unique (graph,app,l3_size) cells={len(og_by_cell)} expected={EXPECTED_CELL_COUNT}"
    )
    assert summary["n_cells"] == EXPECTED_CELL_COUNT, (
        f"summary.n_cells={summary['n_cells']} expected={EXPECTED_CELL_COUNT}"
    )
    # Every row has all required fields.
    for row in og_rows:
        missing = [f for f in ROW_REQUIRED_FIELDS if f not in row]
        assert not missing, f"row missing fields {missing}: {row}"


def test_oracle_gap_each_cell_has_all_four_policies(og_by_cell: dict) -> None:
    bad = []
    for cell, rows in og_by_cell.items():
        policies = sorted(r["policy"] for r in rows)
        if len(rows) != 4 or set(policies) != EXPECTED_POLICIES:
            bad.append((cell, policies))
        # n_policies_in_cell field must also be 4 on every row of the cell.
        for r in rows:
            if r["n_policies_in_cell"] != 4:
                bad.append((cell, r["policy"], r["n_policies_in_cell"]))
    assert not bad, f"cells without exactly 4 distinct policies: {bad[:5]}"


def test_oracle_gap_by_policy_family_counts(oracle_gap: dict, og_rows: list) -> None:
    summary_pf = oracle_gap["summary"]["by_policy_family"]
    counted: Counter = Counter(f"{r['policy']}/{r['family']}" for r in og_rows)
    bad = []
    for key, stats in summary_pf.items():
        if stats["n"] != counted.get(key, 0):
            bad.append((key, stats["n"], counted.get(key, 0)))
    extras = set(counted.keys()) - set(summary_pf.keys())
    assert not bad, f"by_policy_family.n mismatches: {bad[:5]}"
    assert not extras, f"counted (policy,family) pairs not in summary: {sorted(extras)[:5]}"


def test_oracle_gap_by_policy_regime_counts(oracle_gap: dict, og_rows: list) -> None:
    summary_pr = oracle_gap["summary"]["by_policy_regime"]
    counted: Counter = Counter(f"{r['policy']}/{r['regime']}" for r in og_rows)
    bad = []
    for key, stats in summary_pr.items():
        if stats["n"] != counted.get(key, 0):
            bad.append((key, stats["n"], counted.get(key, 0)))
    extras = set(counted.keys()) - set(summary_pr.keys())
    assert not bad, f"by_policy_regime.n mismatches: {bad[:5]}"
    assert not extras, f"counted (policy,regime) pairs not in summary: {sorted(extras)[:5]}"


# ---------------------------------------------------------------------------
# Group B — oracle_gap winner math (4)
# ---------------------------------------------------------------------------


def test_oracle_value_is_min_miss_rate_per_cell(og_by_cell: dict) -> None:
    """For every cell, the ``oracle`` field is constant and equals the minimum
    miss_rate observed across the four policies."""
    bad = []
    for cell, rows in og_by_cell.items():
        oracle_vals = {float(r["oracle"]) for r in rows}
        if len(oracle_vals) != 1:
            bad.append((cell, "non-unique oracle", oracle_vals))
            continue
        oracle = next(iter(oracle_vals))
        min_mr = min(float(r["miss_rate"]) for r in rows)
        if abs(oracle - min_mr) > ORACLE_MIN_TOL:
            bad.append((cell, oracle, min_mr))
    assert not bad, f"oracle != min(miss_rate) cells: {bad[:5]}"


def test_is_winner_matches_oracle_match(og_by_cell: dict) -> None:
    """``is_winner == "1"`` iff that row's ``miss_rate`` equals the oracle
    within ``ORACLE_MIN_TOL`` and every cell has at least one winner."""
    bad = []
    cells_without_winner = []
    for cell, rows in og_by_cell.items():
        oracle = float(rows[0]["oracle"])
        n_winners = 0
        for r in rows:
            is_winner = r["is_winner"] == "1"
            is_min = abs(float(r["miss_rate"]) - oracle) < ORACLE_MIN_TOL
            if is_winner != is_min:
                bad.append((cell, r["policy"], r["is_winner"], float(r["miss_rate"]) - oracle))
            if is_winner:
                n_winners += 1
        if n_winners == 0:
            cells_without_winner.append(cell)
    assert not bad, f"is_winner != (miss_rate == oracle): {bad[:5]}"
    assert not cells_without_winner, f"cells with no winner: {cells_without_winner[:5]}"


def test_gap_pp_recomputes_from_miss_rate_minus_oracle(og_rows: list) -> None:
    """``gap_pp`` recomputes as ``(miss_rate - oracle) * 100`` within tolerance,
    and ``oracle <= miss_rate`` everywhere (gap is non-negative)."""
    bad = []
    negative_gap = []
    for r in og_rows:
        mr = float(r["miss_rate"])
        ora = float(r["oracle"])
        stated_gap = float(r["gap_pp"])
        computed = (mr - ora) * 100.0
        if abs(stated_gap - computed) > GAP_RECOMPUTE_TOL:
            bad.append((r["graph"], r["app"], r["l3_size"], r["policy"], stated_gap, computed))
        if mr < ora - ORACLE_MIN_TOL:
            negative_gap.append((r["graph"], r["app"], r["l3_size"], r["policy"], mr, ora))
    assert not bad, f"gap_pp recompute mismatches: {bad[:5]}"
    assert not negative_gap, f"miss_rate < oracle rows: {negative_gap[:5]}"


def test_overall_by_policy_wins_and_counts(oracle_gap: dict, og_rows: list) -> None:
    """``summary.overall_by_policy`` records 114 rows per policy and a wins
    tally that matches counted winner rows. Total wins equals the sum of
    ``is_winner == "1"`` rows (which can exceed cell count when ties exist)."""
    summary_op = oracle_gap["summary"]["overall_by_policy"]
    assert set(summary_op.keys()) == EXPECTED_POLICIES, (
        f"overall_by_policy keys={set(summary_op.keys())} expected={EXPECTED_POLICIES}"
    )
    counted_n = Counter(r["policy"] for r in og_rows)
    counted_wins = Counter(r["policy"] for r in og_rows if r["is_winner"] == "1")
    for pol, stats in summary_op.items():
        assert stats["n"] == counted_n[pol] == EXPECTED_CELL_COUNT, (
            f"overall_by_policy[{pol}].n={stats['n']} counted={counted_n[pol]} expected={EXPECTED_CELL_COUNT}"
        )
        assert stats["wins"] == counted_wins[pol], (
            f"overall_by_policy[{pol}].wins={stats['wins']} counted={counted_wins[pol]}"
        )
    total_wins = sum(s["wins"] for s in summary_op.values())
    counted_total = sum(1 for r in og_rows if r["is_winner"] == "1")
    assert total_wins == counted_total, (
        f"sum(overall_by_policy.wins)={total_wins} counted total winners={counted_total}"
    )


# ---------------------------------------------------------------------------
# Group C — cross-artifact (oracle_gap ↔ oracle_gap_by_app) (4)
# ---------------------------------------------------------------------------


def test_by_policy_app_aggregates_match_rows(
    oracle_gap_by_app: dict, og_rows: list
) -> None:
    """``oracle_gap_by_app.by_policy_app`` n/mean/max/wins recompute exactly
    from oracle_gap rows when grouped by (policy, app); median/p90 match within
    a small numeric tolerance."""
    bpa = oracle_gap_by_app["by_policy_app"]
    # Pre-aggregate rows by (policy, app).
    grouped: dict = defaultdict(list)
    wins_count: Counter = Counter()
    for r in og_rows:
        key = f"{r['policy']}/{r['app']}"
        grouped[key].append(float(r["gap_pp"]))
        if r["is_winner"] == "1":
            wins_count[key] += 1
    # Every (policy, app) combination present in og rows must appear in bpa.
    assert set(bpa.keys()) == set(grouped.keys()), (
        f"by_policy_app keys differ: only-in-summary={set(bpa) - set(grouped)} "
        f"only-in-rows={set(grouped) - set(bpa)}"
    )
    bad: list = []
    for key, stats in bpa.items():
        for field in BY_POLICY_APP_REQUIRED:
            assert field in stats, f"by_policy_app[{key}] missing field {field}"
        gaps = grouped[key]
        if stats["n"] != len(gaps):
            bad.append((key, "n", stats["n"], len(gaps)))
        mean = sum(gaps) / len(gaps)
        if abs(stats["mean"] - round(mean, 4)) > MEAN_TOL:
            bad.append((key, "mean", stats["mean"], mean))
        if abs(stats["max"] - max(gaps)) > MEAN_TOL:
            bad.append((key, "max", stats["max"], max(gaps)))
        if abs(stats["median"] - _median_linear(gaps)) > PCTL_TOL:
            bad.append((key, "median", stats["median"], _median_linear(gaps)))
        if abs(stats["p90"] - _p90_higher(gaps)) > PCTL_TOL:
            bad.append((key, "p90", stats["p90"], _p90_higher(gaps)))
        if stats["wins"] != wins_count[key]:
            bad.append((key, "wins", stats["wins"], wins_count[key]))
    assert not bad, f"by_policy_app aggregation mismatches: {bad[:8]}"


def test_by_app_ranking_shape_and_policies(oracle_gap_by_app: dict) -> None:
    """``by_app_ranking[app]`` has 4 entries with the canonical policy set and
    every entry exposes the required keys."""
    bar = oracle_gap_by_app["by_app_ranking"]
    assert set(bar.keys()) == EXPECTED_APPS, (
        f"by_app_ranking keys={set(bar.keys())} expected={EXPECTED_APPS}"
    )
    bad: list = []
    for app, entries in bar.items():
        if len(entries) != 4:
            bad.append((app, "len", len(entries)))
            continue
        policies = sorted(e["policy"] for e in entries)
        if set(policies) != EXPECTED_POLICIES:
            bad.append((app, "policies", policies))
        for entry in entries:
            for field in BY_APP_RANKING_REQUIRED:
                if field not in entry:
                    bad.append((app, entry.get("policy"), "missing-field", field))
    assert not bad, f"by_app_ranking shape issues: {bad[:5]}"


def test_by_app_ranking_sorted_by_mean_ascending(oracle_gap_by_app: dict) -> None:
    """Within each app, the ranking is sorted by ``mean_gap_pp`` ascending so
    the best (smallest gap) policy appears first."""
    bad = []
    for app, entries in oracle_gap_by_app["by_app_ranking"].items():
        means = [e["mean_gap_pp"] for e in entries]
        if means != sorted(means):
            bad.append((app, means))
    assert not bad, f"by_app_ranking not sorted ascending by mean_gap_pp: {bad}"


def test_by_app_ranking_matches_by_policy_app(oracle_gap_by_app: dict) -> None:
    """For every (app, entry) in the ranking the per-policy stats agree with
    ``by_policy_app[policy/app]`` (which we already verified against rows)."""
    bpa = oracle_gap_by_app["by_policy_app"]
    bar = oracle_gap_by_app["by_app_ranking"]
    bad: list = []
    for app, entries in bar.items():
        for entry in entries:
            key = f"{entry['policy']}/{app}"
            ref = bpa.get(key)
            assert ref is not None, f"by_app_ranking[{app}] references missing key {key}"
            pairs = (
                ("n", entry["n"], ref["n"]),
                ("mean", entry["mean_gap_pp"], ref["mean"]),
                ("median", entry["median_gap_pp"], ref["median"]),
                ("max", entry["max_gap_pp"], ref["max"]),
                ("p90", entry["p90_gap_pp"], ref["p90"]),
                ("wins", entry["wins"], ref["wins"]),
            )
            for label, a, b in pairs:
                if label == "n" or label == "wins":
                    if a != b:
                        bad.append((app, entry["policy"], label, a, b))
                else:
                    if abs(a - b) > MEAN_TOL:
                        bad.append((app, entry["policy"], label, a, b))
    assert not bad, f"by_app_ranking ↔ by_policy_app mismatches: {bad[:5]}"


# ---------------------------------------------------------------------------
# Group D — universe + math hygiene (1)
# ---------------------------------------------------------------------------


def test_universe_enumerations_and_numeric_hygiene(og_rows: list) -> None:
    """All categorical fields stay inside their expected universes and every
    numeric-bearing field parses to a finite float."""
    policies = {r["policy"] for r in og_rows}
    apps = {r["app"] for r in og_rows}
    families = {r["family"] for r in og_rows}
    regimes = {r["regime"] for r in og_rows}
    l3_sizes = {r["l3_size"] for r in og_rows}
    assert policies <= EXPECTED_POLICIES, f"unexpected policies: {policies - EXPECTED_POLICIES}"
    assert apps <= EXPECTED_APPS, f"unexpected apps: {apps - EXPECTED_APPS}"
    assert families <= EXPECTED_FAMILIES, f"unexpected families: {families - EXPECTED_FAMILIES}"
    assert regimes <= EXPECTED_REGIMES, f"unexpected regimes: {regimes - EXPECTED_REGIMES}"
    assert l3_sizes <= EXPECTED_L3_SIZES, f"unexpected l3_sizes: {l3_sizes - EXPECTED_L3_SIZES}"
    # Every numeric-bearing field must parse to a finite float.
    bad: list = []
    for r in og_rows:
        for field in ("gap_pp", "miss_rate", "oracle"):
            try:
                val = float(r[field])
            except (TypeError, ValueError):
                bad.append((r["graph"], r["app"], r["l3_size"], r["policy"], field, r[field]))
                continue
            if not math.isfinite(val):
                bad.append((r["graph"], r["app"], r["l3_size"], r["policy"], field, val))
    assert not bad, f"non-finite numeric fields: {bad[:5]}"
