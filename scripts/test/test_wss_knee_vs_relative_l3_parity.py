"""Gate 93 — WSS knee location vs WSS-relative-L3 parity.

Two WSS-regime artifacts measure the same per-(graph, app, L3-size) cells
from different angles:

- ``wss_knee_location`` looks at the smallest WSS regime where each policy
  crosses a 0.5pp oracle-gap threshold (the "knee"). It ranks policies by
  knee_rank along the under_wss → near_wss → over_wss ladder and asserts
  that oracle-aware policies (GRASP/POPT) have a strictly smaller knee
  than non-oracle policies (LRU/SRRIP).
- ``wss_relative_l3`` partitions the 114 cells by L3-vs-WSS regime and
  publishes a per-policy ranking inside each regime.

The two share the same underlying cell partition (per_regime_cell_count)
and per-(policy, regime) summary statistics (mean_gap_pp, win_rate). If
they drift apart silently, every WSS-based claim in the paper becomes
internally inconsistent.

This gate locks the knee-rank/regime values, the regime cell-count
distribution, the verdict invariant, and the per-(policy, regime)
parity across both artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WKL_PATH = REPO / "wiki/data/wss_knee_location.json"
WRL_PATH = REPO / "wiki/data/wss_relative_l3.json"

EXPECTED_POLICIES = {"GRASP", "LRU", "POPT", "SRRIP"}
EXPECTED_ORACLE_AWARE = {"GRASP", "POPT"}
EXPECTED_NON_ORACLE = {"LRU", "SRRIP"}
EXPECTED_REGIME_LADDER = ["under_wss", "near_wss", "over_wss"]

EXPECTED_KNEE_THRESHOLD_PP = 0.5
EXPECTED_KNEE_RANK = {"GRASP": 0, "LRU": 2, "POPT": 0, "SRRIP": 2}
EXPECTED_KNEE_REGIME = {
    "GRASP": "under_wss",
    "LRU": "over_wss",
    "POPT": "under_wss",
    "SRRIP": "over_wss",
}

EXPECTED_REGIME_CELL_COUNT = {"near_wss": 52, "over_wss": 14, "under_wss": 48}
EXPECTED_TOTAL_CELLS = 114
EXPECTED_WSS_PROXY_COUNT = 8
EXPECTED_WSS_REFERENCE_BYTES = 1048576

FLOAT_EPS = 1e-6


def _load(path: Path) -> dict:
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# wss_knee_location.json — single-artifact invariants
# ---------------------------------------------------------------------------


def test_wkl_meta_scope_is_locked():
    meta = _load(WKL_PATH)["meta"]
    assert set(meta["policies"]) == EXPECTED_POLICIES
    assert meta["regime_ladder"] == EXPECTED_REGIME_LADDER
    assert math.isclose(meta["knee_threshold_pp"], EXPECTED_KNEE_THRESHOLD_PP)
    assert set(meta["oracle_aware_policies"]) == EXPECTED_ORACLE_AWARE
    assert set(meta["non_oracle_policies"]) == EXPECTED_NON_ORACLE


def test_wkl_oracle_aware_partition_is_disjoint_and_complete():
    meta = _load(WKL_PATH)["meta"]
    oa = set(meta["oracle_aware_policies"])
    no = set(meta["non_oracle_policies"])
    assert oa.isdisjoint(no), "oracle-aware and non-oracle sets overlap"
    assert oa | no == EXPECTED_POLICIES, "oracle partition is missing policies"


def test_wkl_knee_rank_by_policy_matches_baseline():
    meta = _load(WKL_PATH)["meta"]
    assert meta["knee_rank_by_policy"] == EXPECTED_KNEE_RANK


def test_wkl_knee_regime_by_policy_matches_baseline():
    meta = _load(WKL_PATH)["meta"]
    assert meta["knee_regime_by_policy"] == EXPECTED_KNEE_REGIME
    for pol, regime in meta["knee_regime_by_policy"].items():
        assert regime in EXPECTED_REGIME_LADDER, f"{pol} regime '{regime}' not in ladder"


def test_wkl_verdict_invariant_holds():
    meta = _load(WKL_PATH)["meta"]
    max_oa = max(EXPECTED_KNEE_RANK[p] for p in EXPECTED_ORACLE_AWARE)
    min_no = min(EXPECTED_KNEE_RANK[p] for p in EXPECTED_NON_ORACLE)
    assert meta["max_oracle_aware_knee_rank"] == max_oa
    assert meta["min_non_oracle_knee_rank"] == min_no
    assert max_oa < min_no, (
        f"verdict invariant broken: max_oracle_aware_knee_rank={max_oa} "
        f"not < min_non_oracle_knee_rank={min_no}"
    )
    assert meta["verdict"] == "PASS"


def test_wkl_per_policy_oracle_aware_flag_matches_partition():
    per_policy = _load(WKL_PATH)["per_policy"]
    assert set(per_policy) == EXPECTED_POLICIES
    for pol, payload in per_policy.items():
        expected = pol in EXPECTED_ORACLE_AWARE
        assert payload["is_oracle_aware"] is expected, (
            f"{pol} is_oracle_aware={payload['is_oracle_aware']} expected {expected}"
        )
        assert payload["knee_rank"] == EXPECTED_KNEE_RANK[pol]
        assert payload["knee_regime"] == EXPECTED_KNEE_REGIME[pol]


# ---------------------------------------------------------------------------
# wss_relative_l3.json — single-artifact invariants
# ---------------------------------------------------------------------------


def test_wrl_classification_covers_full_corpus():
    meta = _load(WRL_PATH)["meta"]
    assert meta["n_cells_classified"] == EXPECTED_TOTAL_CELLS
    assert meta["n_cells_skipped"] == 0
    assert meta["unknown_graphs"] == []
    assert len(meta["wss_proxies"]) == EXPECTED_WSS_PROXY_COUNT
    assert meta["wss_reference_bytes"] == EXPECTED_WSS_REFERENCE_BYTES


def test_wrl_per_regime_cell_count_matches_baseline_and_sums_to_total():
    counts = _load(WRL_PATH)["per_regime_cell_count"]
    assert counts == EXPECTED_REGIME_CELL_COUNT
    assert sum(counts.values()) == EXPECTED_TOTAL_CELLS


def test_wrl_per_regime_ranking_is_sorted_by_mean_gap_ascending():
    rankings = _load(WRL_PATH)["per_regime_ranking"]
    assert set(rankings) == set(EXPECTED_REGIME_LADDER)
    for regime, ranking in rankings.items():
        assert {p["policy"] for p in ranking} == EXPECTED_POLICIES
        means = [p["mean_gap_pp"] for p in ranking]
        assert means == sorted(means), (
            f"per_regime_ranking[{regime}] not sorted ascending by mean_gap_pp: {means}"
        )


def test_wrl_per_regime_total_wins_equal_n():
    rankings = _load(WRL_PATH)["per_regime_ranking"]
    for regime, ranking in rankings.items():
        total_wins = sum(p["wins"] for p in ranking)
        n = ranking[0]["n"]
        assert total_wins == n, (
            f"regime '{regime}': sum(wins)={total_wins} != n={n}"
        )
        for p in ranking:
            assert p["n"] == n, f"regime '{regime}' policy '{p['policy']}' has n={p['n']} != {n}"


def test_wrl_by_policy_regime_grid_is_complete_and_matches_counts():
    wrl = _load(WRL_PATH)
    bpr = wrl["by_policy_regime"]
    expected_keys = {
        f"{pol}/{reg}" for pol in EXPECTED_POLICIES for reg in EXPECTED_REGIME_LADDER
    }
    assert set(bpr) == expected_keys
    for key, stats in bpr.items():
        pol, regime = key.split("/")
        assert stats["policy"] == pol
        assert stats["wss_regime"] == regime
        assert stats["n_cells_in_regime"] == EXPECTED_REGIME_CELL_COUNT[regime]
        assert stats["n"] == EXPECTED_REGIME_CELL_COUNT[regime]


# ---------------------------------------------------------------------------
# Cross-artifact parity
# ---------------------------------------------------------------------------


def test_xartifact_per_policy_regime_stats_agree():
    wkl = _load(WKL_PATH)
    wrl = _load(WRL_PATH)
    per_policy = wkl["per_policy"]
    bpr = wrl["by_policy_regime"]
    regime_counts = wrl["per_regime_cell_count"]

    for pol, payload in per_policy.items():
        for regime, stats in payload["per_regime"].items():
            wrl_stats = bpr[f"{pol}/{regime}"]
            assert stats["n"] == regime_counts[regime] == wrl_stats["n_cells_in_regime"] == wrl_stats["n"], (
                f"{pol}/{regime}: cell-count drift between wkl/wrl"
            )
            assert abs(stats["mean_gap_pp"] - wrl_stats["mean_gap_pp"]) < FLOAT_EPS, (
                f"{pol}/{regime}: mean_gap_pp drift {stats['mean_gap_pp']} vs {wrl_stats['mean_gap_pp']}"
            )
            assert abs(stats["win_rate"] - wrl_stats["win_rate"]) < FLOAT_EPS, (
                f"{pol}/{regime}: win_rate drift {stats['win_rate']} vs {wrl_stats['win_rate']}"
            )


def test_xartifact_regime_ladder_is_shared():
    wkl_ladder = _load(WKL_PATH)["meta"]["regime_ladder"]
    wrl_regimes = set(_load(WRL_PATH)["per_regime_cell_count"].keys())
    assert set(wkl_ladder) == wrl_regimes, (
        f"wkl ladder {wkl_ladder} disagrees with wrl regimes {wrl_regimes}"
    )
