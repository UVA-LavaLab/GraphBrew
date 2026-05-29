"""Gate 109: small_l3_thrash.json self-consistency + winner derivation + tiny-regime disjointness.

This gate locks the small-L3 thrash artifact, which underlies the paper's
"tiny-cache regime where ECG/GRASP loses" claim. The artifact is a 4kB-only
snapshot derived from the paper pipeline's combined_roi_matrix.csv with a
fixed 9-policy panel; it must satisfy:

  * Internal arithmetic: n_rows = n_cells * n_policies, two distinct win
    accounting structures (policy_stats.wins and win_counts) agree with
    each other and with n_cells.
  * Winner derivation: the per-cell winner / runner-up / margin_pp must
    follow strictly from the CSV miss rates (no editorial overrides).
  * Showdown panel: the LRU/POPT/GRASP cross-policy gap fields match the
    underlying miss-rate arithmetic exactly.
  * Cross-artifact disjointness with winning_regime_taxonomy.json: the
    thrash cells and WRT 'tiny'-regime cells partition the 4kB regime
    (thrash covers power-law graphs; WRT-tiny covers mesh+road graphs).
    No overlap is permitted.

The 9-policy panel of small_l3_thrash is broader than the 4-policy panel
used by winning_regime_taxonomy / oracle_gap; the two artifacts intentionally
serve different roles (thrash = wide-panel detective work; WRT/oracle =
canonical 4-policy comparator). This gate verifies the contract.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
THRASH_JSON = REPO_ROOT / "wiki" / "data" / "small_l3_thrash.json"
THRASH_CSV = REPO_ROOT / "wiki" / "data" / "small_l3_thrash.csv"
WRT_JSON = REPO_ROOT / "wiki" / "data" / "winning_regime_taxonomy.json"

EXPECTED_N_CELLS = 9
EXPECTED_N_POLICIES = 9
EXPECTED_N_ROWS = EXPECTED_N_CELLS * EXPECTED_N_POLICIES  # 81
EXPECTED_L3_SIZE = "4kB"
EXPECTED_THRASH_GRAPHS = {"cit-Patents", "com-orkut", "soc-LiveJournal1"}
EXPECTED_THRASH_APPS = {"bfs", "pr", "sssp"}
EXPECTED_POLICY_LABELS = {
    "ECG_DBG_ONLY",
    "ECG_DBG_PRIMARY",
    "ECG_DBG_PRIMARY_CHARGED",
    "ECG_POPT_PRIMARY",
    "GRASP",
    "LRU",
    "POPT",
    "POPT_CHARGED",
    "SRRIP",
}

# Deterministic tie-breaking order from
# scripts/experiments/ecg/small_l3_thrash_report.py:POLICY_LABEL_ORDER.
# Sorted by (miss_rate, index_in_this_list) so when multiple policies tie
# in miss rate, the earlier one in this list wins. Locking the order here
# pins the artifact's deterministic-tie contract.
POLICY_LABEL_ORDER = (
    "LRU",
    "SRRIP",
    "GRASP",
    "POPT",
    "POPT_CHARGED",
    "ECG_DBG_ONLY",
    "ECG_DBG_PRIMARY",
    "ECG_DBG_PRIMARY_CHARGED",
    "ECG_POPT_PRIMARY",
)

# Power-law graphs that thrash subset must come from (excluding reserved-future
# entries from POWER_LAW_GRAPHS in literature_baselines.py).
POWER_LAW_CORPUS_GRAPHS = {
    "cit-Patents",
    "soc-pokec",
    "soc-LiveJournal1",
    "com-orkut",
    "web-Google",
}

# Mesh + road families (tiny-regime in WRT) must be disjoint from thrash power-law
# graphs at the 4kB cut.
TINY_REGIME_FAMILIES = {"mesh", "road"}

MARGIN_TOL = 1e-3
MISS_RATE_TOL = 1e-9


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def thrash():
    assert THRASH_JSON.exists(), f"missing artifact: {THRASH_JSON}"
    return json.loads(THRASH_JSON.read_text())


@pytest.fixture(scope="module")
def thrash_csv():
    assert THRASH_CSV.exists(), f"missing artifact: {THRASH_CSV}"
    with THRASH_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    return rows


@pytest.fixture(scope="module")
def wrt():
    assert WRT_JSON.exists(), f"missing artifact: {WRT_JSON}"
    return json.loads(WRT_JSON.read_text())


# ---------- Group A: internal arithmetic (4) ----------


def test_thrash_n_rows_matches_cells_times_policies(thrash):
    s = thrash["summary"]
    assert s["n_cells"] == EXPECTED_N_CELLS, s["n_cells"]
    assert len(s["policy_stats"]) == EXPECTED_N_POLICIES, len(s["policy_stats"])
    assert s["n_rows"] == EXPECTED_N_ROWS, s["n_rows"]
    assert s["n_rows"] == s["n_cells"] * len(s["policy_stats"])


def test_thrash_policy_stats_wins_sum_to_n_cells(thrash):
    total = sum(p["wins"] for p in thrash["summary"]["policy_stats"].values())
    assert total == thrash["summary"]["n_cells"], total


def test_thrash_win_counts_consistent_with_policy_stats(thrash):
    s = thrash["summary"]
    wc_total = sum(s["win_counts"].values())
    assert wc_total == s["n_cells"]
    for policy, wins in s["win_counts"].items():
        assert policy in s["policy_stats"], policy
        assert s["policy_stats"][policy]["wins"] == wins, (
            policy, wins, s["policy_stats"][policy]["wins"]
        )
    # any policy not in win_counts must have zero wins in policy_stats
    for policy, stats in s["policy_stats"].items():
        if policy not in s["win_counts"]:
            assert stats["wins"] == 0, (policy, stats["wins"])


def test_thrash_each_policy_appears_n_cells_times_in_csv(thrash_csv, thrash):
    counts = Counter(r["policy_label"] for r in thrash_csv)
    assert set(counts) == EXPECTED_POLICY_LABELS, set(counts) ^ EXPECTED_POLICY_LABELS
    n_cells = thrash["summary"]["n_cells"]
    for policy, c in counts.items():
        assert c == n_cells, (policy, c, n_cells)


# ---------- Group B: winner derivation from CSV (4) ----------


def _winner_key(policy: str, miss_rate: float) -> tuple[float, int]:
    try:
        idx = POLICY_LABEL_ORDER.index(policy)
    except ValueError:
        idx = len(POLICY_LABEL_ORDER)
    return (miss_rate, idx)


def test_thrash_winner_is_argmin_miss_rate(thrash_csv, thrash):
    by_cell: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for r in thrash_csv:
        key = (r["graph"], r["app"], r["l3_size"])
        by_cell.setdefault(key, []).append((r["policy_label"], float(r["l3_miss_rate"])))
    for c in thrash["cells"]:
        key = (c["graph"], c["app"], c["l3_size"])
        rows = by_cell[key]
        rows_sorted = sorted(rows, key=lambda x: _winner_key(x[0], x[1]))
        winner = rows_sorted[0]
        assert c["winner"] == winner[0], (key, c["winner"], winner)
        assert abs(c["winner_miss_rate"] - winner[1]) < MISS_RATE_TOL, (
            key, c["winner_miss_rate"], winner[1]
        )


def test_thrash_runner_up_is_second_min(thrash_csv, thrash):
    by_cell: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for r in thrash_csv:
        key = (r["graph"], r["app"], r["l3_size"])
        by_cell.setdefault(key, []).append((r["policy_label"], float(r["l3_miss_rate"])))
    for c in thrash["cells"]:
        key = (c["graph"], c["app"], c["l3_size"])
        rows_sorted = sorted(by_cell[key], key=lambda x: _winner_key(x[0], x[1]))
        runner = rows_sorted[1]
        assert c["runner_up"] == runner[0], (key, c["runner_up"], runner)
        assert abs(c["runner_up_miss_rate"] - runner[1]) < MISS_RATE_TOL, (
            key, c["runner_up_miss_rate"], runner[1]
        )


def test_thrash_margin_pp_matches_runner_up_minus_winner(thrash):
    for c in thrash["cells"]:
        expected = (c["runner_up_miss_rate"] - c["winner_miss_rate"]) * 100
        assert abs(c["margin_pp"] - expected) < MARGIN_TOL, (c, expected)
        assert c["margin_pp"] >= -MARGIN_TOL, c  # never negative


def test_thrash_policy_stats_aggregates_recompute_from_csv(thrash_csv, thrash):
    by_policy: dict[str, list[float]] = {}
    for r in thrash_csv:
        by_policy.setdefault(r["policy_label"], []).append(float(r["l3_miss_rate"]))
    for policy, stats in thrash["summary"]["policy_stats"].items():
        rates = sorted(by_policy[policy])
        assert stats["n_cells"] == len(rates), policy
        assert abs(stats["min_miss_rate"] - min(rates)) < MISS_RATE_TOL, policy
        assert abs(stats["max_miss_rate"] - max(rates)) < MISS_RATE_TOL, policy
        assert abs(stats["mean_miss_rate"] - sum(rates) / len(rates)) < MISS_RATE_TOL, policy
        n = len(rates)
        if n % 2 == 0:
            median = (rates[n // 2 - 1] + rates[n // 2]) / 2
        else:
            median = rates[n // 2]
        assert abs(stats["median_miss_rate"] - median) < MISS_RATE_TOL, policy


# ---------- Group C: cross-artifact disjointness with WRT (4) ----------


def test_thrash_cells_all_at_4kb_and_powerlaw(thrash):
    for c in thrash["cells"]:
        assert c["l3_size"] == EXPECTED_L3_SIZE, c
        assert c["graph"] in POWER_LAW_CORPUS_GRAPHS, c["graph"]
        assert c["graph"] in EXPECTED_THRASH_GRAPHS, c["graph"]
        assert c["app"] in EXPECTED_THRASH_APPS, c["app"]


def test_thrash_disjoint_from_wrt_tiny_regime(thrash, wrt):
    thrash_keys = {(c["graph"], c["app"], c["l3_size"]) for c in thrash["cells"]}
    tiny_keys = {
        (c["graph"], c["app"], c["l3_size"])
        for c in wrt["cells"]
        if c["regime"] == "tiny"
    }
    assert thrash_keys.isdisjoint(tiny_keys), thrash_keys & tiny_keys


def test_wrt_tiny_only_mesh_road_families(wrt):
    tiny = [c for c in wrt["cells"] if c["regime"] == "tiny"]
    assert tiny, "expected at least one tiny-regime cell"
    families = {c["family"] for c in tiny}
    assert families <= TINY_REGIME_FAMILIES, families - TINY_REGIME_FAMILIES


def test_thrash_no_wrt_powerlaw_overlap_at_4kb(thrash, wrt):
    thrash_graphs = {c["graph"] for c in thrash["cells"]}
    wrt_4kb_graphs = {c["graph"] for c in wrt["cells"] if c["l3_size"] == EXPECTED_L3_SIZE}
    # WRT@4kB graphs are mesh/road; thrash@4kB graphs are power-law; no overlap.
    assert thrash_graphs.isdisjoint(wrt_4kb_graphs), thrash_graphs & wrt_4kb_graphs


# ---------- Group D: showdown panel arithmetic (1) ----------


def test_thrash_showdown_gap_fields_match_miss_rate_arithmetic(thrash):
    showdown = thrash["showdown"]
    assert len(showdown) == thrash["summary"]["n_cells"], len(showdown)
    for r in showdown:
        # lru_minus_X_pp encodes (X_miss - lru_miss) * 100 in the artifact;
        # positive => X is worse than LRU; negative => X beats LRU.
        expected_grasp = (r["grasp_miss"] - r["lru_miss"]) * 100
        expected_popt = (r["popt_miss"] - r["lru_miss"]) * 100
        assert abs(r["lru_minus_grasp_pp"] - expected_grasp) < MARGIN_TOL, (r, expected_grasp)
        assert abs(r["lru_minus_popt_pp"] - expected_popt) < MARGIN_TOL, (r, expected_popt)
        assert r["l3_size"] == EXPECTED_L3_SIZE, r
        assert r["graph"] in EXPECTED_THRASH_GRAPHS, r
        assert r["app"] in EXPECTED_THRASH_APPS, r
