"""Tests for gate 64 — cross-policy mean-margin asymmetry."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "cross_policy_asymmetry.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "cross_policy_asymmetry.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_well_formed():
    p = _payload()
    assert "meta" in p and "per_pair" in p
    assert p["meta"]["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]


def test_six_unordered_pairs():
    # C(4,2) = 6 unordered pairs across the four policies.
    p = _payload()
    assert p["meta"]["pair_count"] == 6
    assert len(p["per_pair"]) == 6


def test_every_pair_has_both_sides_winning():
    p = _payload()
    assert p["meta"]["every_pair_both_win"] is True


def test_max_asymmetry_under_ceiling():
    p = _payload()
    assert p["meta"]["max_ratio_under_ceiling"] is True
    assert p["meta"]["max_asymmetry_ratio"] < p["meta"]["ratio_ceiling"]


def test_grasp_vs_lru_grasp_dominates_winrate():
    # GRASP must win head-to-head against LRU on at least 75% of cells
    # (a soft proxy for "oracle-aware beats LRU broadly").
    p = _payload()
    e = p["per_pair"]["GRASP_vs_LRU"]
    total = e["a_wins"] + e["b_wins"] + e["ties"]
    assert e["a_wins"] / total >= 0.75


def test_grasp_vs_popt_balanced_winrate():
    # GRASP vs POPT must be close to balanced (each within 25% of half
    # of the non-tie cells).
    p = _payload()
    e = p["per_pair"]["GRASP_vs_POPT"]
    total = e["a_wins"] + e["b_wins"]
    assert total > 0
    a_share = e["a_wins"] / total
    assert 0.25 <= a_share <= 0.75


def test_popt_vs_lru_popt_dominates_winrate():
    p = _payload()
    e = p["per_pair"]["LRU_vs_POPT"]
    total = e["a_wins"] + e["b_wins"] + e["ties"]
    # Charged corpus: POPT remains a broad winner over LRU, but the faithful
    # capacity charge lowers the win share to ~0.71.
    assert e["b_wins"] / total >= 0.70


def test_all_means_non_negative():
    p = _payload()
    for e in p["per_pair"].values():
        assert e["a_mean_margin_pp"] >= 0.0
        assert e["b_mean_margin_pp"] >= 0.0


def test_asymmetry_ratios_finite_and_sane():
    p = _payload()
    for e in p["per_pair"].values():
        r = e["asymmetry_ratio"]
        assert r is not None, (
            f"pair {e['a_policy']}/{e['b_policy']} has no finite ratio "
            f"(one side never won)"
        )
        assert r >= 1.0  # by construction, max/min >= 1
        assert r < 20.0


def test_oracle_pair_is_close_to_symmetric():
    # GRASP and POPT are both oracle-aware; we expect their mean-loss
    # magnitudes to be closer to each other than the worst pair.
    # Today GRASP vs POPT ratio is 2.668, ceiling for this test 5.0.
    p = _payload()
    r = p["per_pair"]["GRASP_vs_POPT"]["asymmetry_ratio"]
    assert r is not None and r < 5.0, (
        f"GRASP-vs-POPT asymmetry must remain under 5x; got {r}"
    )


def test_total_cells_consistent_across_pairs():
    # Every pair sees the same population of cells (every (app, graph,
    # L3) cell that has both policies). The corpus contains all 4
    # policies on the same cells today, so all 6 pairs see identical
    # totals.
    p = _payload()
    totals = {
        e["a_wins"] + e["b_wins"] + e["ties"]
        for e in p["per_pair"].values()
    }
    assert len(totals) == 1, (
        f"pairs see inconsistent cell totals: {totals}"
    )


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["verdict"] == "PASS"
