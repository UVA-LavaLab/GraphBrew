"""Tests for gate 62 — winner-margin distribution per WSS regime."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "winner_margin_by_regime.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "winner_margin_by_regime.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_well_formed():
    p = _payload()
    assert "meta" in p and "per_policy_regime" in p
    assert p["meta"]["regimes"] == ["under_wss", "near_wss", "over_wss"]
    assert p["meta"]["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]


def test_no_cells_skipped():
    p = _payload()
    assert p["meta"]["cells_skipped"] == 0, (
        f"skipped cells should be 0; got {p['meta']['cells_skipped']}"
    )


def test_full_classification_count():
    # 8 graphs * 5 apps * various L3 sizes; today the full classified
    # count is exactly 114.
    p = _payload()
    assert p["meta"]["cells_classified"] == 114


def test_every_regime_has_wins():
    p = _payload()
    assert all(p["meta"]["regime_has_wins"].values()), (
        f"some regime has zero wins: {p['meta']['regime_has_wins']}"
    )


def test_grasp_median_margin_shrinks_with_capacity():
    p = _payload()
    u = p["per_policy_regime"]["GRASP/under_wss"]["median_margin_pp"]
    o = p["per_policy_regime"]["GRASP/over_wss"]["median_margin_pp"]
    assert u > o, (
        f"GRASP median margin must shrink under -> over; got {u} vs {o}"
    )


def test_popt_median_margin_shrinks_with_capacity():
    p = _payload()
    u = p["per_policy_regime"]["POPT/under_wss"]["median_margin_pp"]
    o = p["per_policy_regime"]["POPT/over_wss"]["median_margin_pp"]
    assert u > o, (
        f"POPT median margin must shrink under -> over; got {u} vs {o}"
    )


def test_oracle_aware_dominate_total_wins():
    # GRASP+POPT must win more cells than LRU+SRRIP across all regimes.
    p = _payload()
    oracle_wins = sum(
        p["per_policy_regime"][f"{pol}/{rg}"]["cells_won"]
        for pol in ("GRASP", "POPT")
        for rg in ("under_wss", "near_wss", "over_wss")
    )
    non_wins = sum(
        p["per_policy_regime"][f"{pol}/{rg}"]["cells_won"]
        for pol in ("LRU", "SRRIP")
        for rg in ("under_wss", "near_wss", "over_wss")
    )
    assert oracle_wins > 5 * non_wins, (
        f"oracle-aware wins ({oracle_wins}) must dominate non-oracle "
        f"({non_wins}) by at least 5x"
    )


def test_grasp_wins_at_least_20_in_under_wss():
    p = _payload()
    assert p["per_policy_regime"]["GRASP/under_wss"]["cells_won"] >= 20


def test_popt_wins_at_least_20_in_under_wss():
    p = _payload()
    assert p["per_policy_regime"]["POPT/under_wss"]["cells_won"] >= 20


def test_at_least_one_shrink_evidence_oracle_aware():
    p = _payload()
    sh = p["meta"]["shrink_evidence"]
    assert len(sh) >= 1
    for ev in sh:
        assert ev["policy"] in ("GRASP", "POPT")
        assert ev["under_median"] > ev["over_median"]


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["verdict"] == "PASS"


def test_margin_values_are_non_negative():
    # Winner margin is "second - winner" so must be >= 0 always.
    p = _payload()
    for cell in p["per_policy_regime"].values():
        for field in (
            "median_margin_pp",
            "mean_margin_pp",
            "p90_margin_pp",
            "max_margin_pp",
        ):
            assert cell[field] >= 0.0, (
                f"{cell['policy']}/{cell['wss_regime']} {field} negative"
            )
