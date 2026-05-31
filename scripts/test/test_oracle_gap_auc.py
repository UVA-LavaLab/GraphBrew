"""Gate 49: per-(app, policy) oracle-gap AUC across L3 sweep.

Pins the trapezoidal AUC winners per app, the AUC dominance ratios,
and surfaces where the AUC winner disagrees with the cell-vote winner
(an honest-disclosure flag for the paper text).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "oracle_gap_auc.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA.exists():
        pytest.skip("oracle_gap_auc.json not built")
    return json.loads(DATA.read_text())


def test_meta_pins_scope(payload):
    m = payload["meta"]
    assert m["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert m["x_axis"] == "log2(L3 size in MB)"
    assert m["auc_units"] == "gap_pp × log2(MB)"
    assert m["n_apps"] == 5
    assert m["n_policies"] == 4
    assert m["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]


def test_auc_winner_by_app_exact(payload):
    """Pin AUC winners: bc=SRRIP, bfs=POPT, cc=GRASP, pr=POPT, sssp=POPT.

    Post cache_sim ECG sweep: bc/AUC flipped GRASP→SRRIP after the
    binary-fix refresh; SRRIP edges GRASP on Pearson-clustered AUC by
    a tiny margin while GRASP still wins on cell-count. Both classifiers
    are now exposed as borderline for bc.
    Note: sssp/POPT (AUC) differs from sssp/GRASP (cell-vote) — see test below."""
    assert payload["meta"]["auc_winner_by_app"] == {
        "bc": "SRRIP",
        "bfs": "POPT",
        "cc": "GRASP",
        "pr": "POPT",
        "sssp": "POPT",
    }


def test_every_app_has_4_policies_in_ranking(payload):
    for app, p in payload["per_app"].items():
        assert len(p["ranking"]) == 4, f"{app} ranking has {len(p['ranking'])} entries"
        pols = [r["policy"] for r in p["ranking"]]
        assert sorted(pols) == ["GRASP", "LRU", "POPT", "SRRIP"]


def test_ranking_is_sorted_ascending(payload):
    for app, p in payload["per_app"].items():
        aucs = [r["auc"] for r in p["ranking"]]
        assert aucs == sorted(aucs), (
            f"{app} ranking not ascending by AUC: {aucs}"
        )


def test_winner_auc_is_minimum(payload):
    for app, p in payload["per_app"].items():
        all_aucs = [r["auc"] for r in p["ranking"]]
        assert p["winner_auc"] == min(all_aucs)
        assert p["winner"] == p["ranking"][0]["policy"]


def test_pr_popt_dominance(payload):
    """pr/POPT must have AUC < 1.0 (essentially tracks oracle) and < 5% of LRU AUC."""
    p = payload["per_app"]["pr"]
    assert p["winner"] == "POPT"
    assert p["winner_auc"] < 1.0, (
        f"pr/POPT AUC = {p['winner_auc']}; expected < 1.0 (paper headline)"
    )
    assert p["auc_ratio_winner_over_lru"] < 0.05, (
        f"pr/POPT vs LRU ratio = {p['auc_ratio_winner_over_lru']}; expected < 5%"
    )


def test_cc_grasp_dominance(payload):
    """cc/GRASP must have AUC < 2.0 and < 10% of LRU AUC."""
    p = payload["per_app"]["cc"]
    assert p["winner"] == "GRASP"
    assert p["winner_auc"] < 2.0
    assert p["auc_ratio_winner_over_lru"] < 0.10


def test_lru_is_never_winner(payload):
    """LRU is the literature baseline; no app should pick LRU as AUC winner.
    If this ever flips, the paper claim 'every app beats LRU' is dead."""
    for app, p in payload["per_app"].items():
        assert p["winner"] != "LRU", f"{app} now picks LRU as AUC winner"


def test_winner_beats_lru_for_all_apps(payload):
    """AUC savings vs LRU must be positive for every app."""
    for app, p in payload["per_app"].items():
        assert p["auc_pp_savings_winner_vs_lru"] > 0, (
            f"{app} winner AUC > LRU AUC; LRU now beats the proposed winner"
        )


def test_trajectory_has_3_l3_points_per_policy(payload):
    """Each (app, policy) trajectory must have exactly 3 L3 entries (1MB/4MB/8MB)."""
    for app, p in payload["per_app"].items():
        for pol, traj in p["trajectory_by_policy"].items():
            assert set(traj.keys()) == {"1MB", "4MB", "8MB"}, (
                f"{app}/{pol} trajectory missing L3 sizes: {sorted(traj.keys())}"
            )


def test_cross_gate_consistency_sssp_auc_vs_cell_vote_disagrees(payload):
    """sssp's AUC winner is POPT but the cell-vote winner (gate 47/48) is
    GRASP. This is an *expected, honestly disclosed* disagreement —
    POPT tracks oracle more closely on average but GRASP wins more cells.
    The paper text must mention BOTH framings. If this disagreement
    disappears in a future run, the test will fire and we'll re-evaluate."""
    p = payload["per_app"]["sssp"]
    assert p["winner"] == "POPT", (
        "sssp AUC winner flipped — re-check paper text for the AUC story"
    )

    # Cross-check: cell-vote winner must still be GRASP (per gate 47/48)
    lofo_path = REPO_ROOT / "wiki" / "data" / "lofo_robustness.json"
    if lofo_path.exists():
        lofo = json.loads(lofo_path.read_text())
        sssp_lofo_full = lofo["per_app"]["sssp"]["full_corpus"]["top_policy"]
        assert sssp_lofo_full == "GRASP", (
            "sssp cell-vote winner flipped; the AUC-vs-cell-vote"
            " disagreement story needs revision"
        )
