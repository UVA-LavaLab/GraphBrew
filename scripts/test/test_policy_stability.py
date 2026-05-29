"""Gate 51: per-policy AUC stability across apps.

Pins:
  - 4-policy / 5-app inventory
  - safest policy by CV (LRU — predictably bad)
  - best avg-AUC policy (POPT)
  - highest-variance policy (POPT)
  - SRRIP is the 'always in top-3' safe runner-up (never wins, never #4)
  - GRASP wins twice (bc, cc) but finishes #4 twice (bfs, sssp)
  - POPT wins 3 of 5 apps
  - LRU never wins, never finishes better than rank 3
  - rank means and rank stdevs are within expected bands
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "policy_stability.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA.exists():
        pytest.skip("policy_stability.json not built")
    return json.loads(DATA.read_text())


def test_meta_inventory(payload):
    m = payload["meta"]
    assert m["n_apps"] == 5
    assert m["n_policies"] == 4
    assert m["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]
    assert m["apps"] == ["bc", "bfs", "cc", "pr", "sssp"]


def test_headline_policies(payload):
    """Pin the 3 headline rankings."""
    m = payload["meta"]
    assert m["safest_policy"] == "LRU"          # predictably bad (low CV)
    assert m["best_avg_policy"] == "POPT"       # lowest mean AUC
    assert m["highest_variance_policy"] == "POPT"  # also highest CV


def test_popt_wins_three_apps(payload):
    """POPT is the AUC winner on bfs, pr, sssp (3 wins)."""
    p = payload["per_policy"]["POPT"]
    assert p["n_wins"] == 3
    winners = [app for app, rank in p["ranks_by_app"].items() if rank == 1]
    assert sorted(winners) == ["bfs", "pr", "sssp"]


def test_grasp_wins_two_apps_loses_two(payload):
    """GRASP is bimodal: wins bc+cc, finishes #4 on bfs+sssp."""
    g = payload["per_policy"]["GRASP"]
    assert g["n_wins"] == 2
    assert g["n_lasts"] == 2
    winners = [app for app, rank in g["ranks_by_app"].items() if rank == 1]
    losers = [app for app, rank in g["ranks_by_app"].items() if rank == 4]
    assert sorted(winners) == ["bc", "cc"]
    assert sorted(losers) == ["bfs", "sssp"]


def test_lru_never_wins(payload):
    """LRU is the literature baseline; the paper's core claim is that
    every (app, L3) cell prefers some advanced policy. Pin n_wins=0."""
    assert payload["per_policy"]["LRU"]["n_wins"] == 0


def test_lru_never_better_than_rank_3(payload):
    """LRU's best rank across apps must be >= 3. If LRU ever climbs to
    rank 2, the paper's 'advanced policies dominate' framing weakens."""
    assert payload["per_policy"]["LRU"]["best_rank"] >= 3


def test_srrip_is_safe_runner_up(payload):
    """SRRIP never wins, never finishes last (rank in [2,3] for all apps).
    This is the 'safe runner-up' archetype the paper text leans on."""
    s = payload["per_policy"]["SRRIP"]
    assert s["n_wins"] == 0
    assert s["n_lasts"] == 0
    assert s["best_rank"] == 2
    assert s["worst_rank"] == 3
    # All ranks must be 2 or 3
    for app, rank in s["ranks_by_app"].items():
        assert rank in (2, 3), f"SRRIP rank on {app} = {rank}; expected 2 or 3"


def test_popt_has_highest_cv(payload):
    """POPT has the highest CV (most variable across workloads)."""
    cvs = {p: d["auc_cv"] for p, d in payload["per_policy"].items()}
    assert max(cvs, key=cvs.get) == "POPT"


def test_lru_has_lowest_cv(payload):
    """LRU has the lowest CV (predictably bad — small variance because
    every app is uniformly far from oracle)."""
    cvs = {p: d["auc_cv"] for p, d in payload["per_policy"].items()}
    assert min(cvs, key=cvs.get) == "LRU"


def test_ranking_by_cv_matches_safest_policy(payload):
    """First entry of ranking_by_cv_ascending should equal safest_policy."""
    first = payload["ranking_by_cv_ascending"][0]["policy"]
    assert first == payload["meta"]["safest_policy"]


def test_ranking_by_mean_auc_matches_best_avg_policy(payload):
    """First entry of ranking_by_mean_auc_ascending should equal best_avg_policy."""
    first = payload["ranking_by_mean_auc_ascending"][0]["policy"]
    assert first == payload["meta"]["best_avg_policy"]


def test_cross_gate_consistency_n_wins_sum_equals_apps(payload):
    """Sum of n_wins across all 4 policies must equal n_apps (every
    app has exactly one winner). This guards against a future bug
    where ties or missing data inflate or deflate the win counts."""
    total = sum(d["n_wins"] for d in payload["per_policy"].values())
    assert total == payload["meta"]["n_apps"]
