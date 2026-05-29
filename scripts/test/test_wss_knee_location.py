"""Tests for gate 60 — WSS-relative knee location."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "wss_knee_location.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "wss_knee_location.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_present_and_well_formed():
    p = _payload()
    assert "meta" in p and "per_policy" in p
    assert p["meta"]["regime_ladder"] == ["under_wss", "near_wss", "over_wss"]
    assert p["meta"]["policies"] == ["GRASP", "LRU", "POPT", "SRRIP"]
    assert set(p["meta"]["oracle_aware_policies"]) == {"GRASP", "POPT"}
    assert set(p["meta"]["non_oracle_policies"]) == {"LRU", "SRRIP"}


def test_knee_threshold_is_sane():
    p = _payload()
    assert 0.0 < p["meta"]["knee_threshold_pp"] <= 5.0


def test_every_policy_has_three_regimes():
    p = _payload()
    for pol, info in p["per_policy"].items():
        assert set(info["per_regime"].keys()) == {
            "under_wss",
            "near_wss",
            "over_wss",
        }


def test_grasp_plateaus_at_under_wss():
    p = _payload()
    assert p["per_policy"]["GRASP"]["knee_regime"] == "under_wss"
    assert p["per_policy"]["GRASP"]["knee_rank"] == 0


def test_popt_plateaus_at_under_wss():
    p = _payload()
    assert p["per_policy"]["POPT"]["knee_regime"] == "under_wss"
    assert p["per_policy"]["POPT"]["knee_rank"] == 0


def test_lru_only_plateaus_at_over_wss():
    p = _payload()
    assert p["per_policy"]["LRU"]["knee_regime"] == "over_wss"
    assert p["per_policy"]["LRU"]["knee_rank"] == 2


def test_srrip_only_plateaus_at_over_wss():
    p = _payload()
    assert p["per_policy"]["SRRIP"]["knee_regime"] == "over_wss"
    assert p["per_policy"]["SRRIP"]["knee_rank"] == 2


def test_oracle_aware_strictly_earlier_than_non_oracle():
    p = _payload()
    m = p["meta"]
    assert m["max_oracle_aware_knee_rank"] < m["min_non_oracle_knee_rank"]


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["verdict"] == "PASS"


def test_non_oracle_median_gap_at_under_wss_exceeds_threshold():
    # LRU and SRRIP must NOT meet the knee threshold at under_wss
    # (this is the whole point of the gate: they need over_wss).
    p = _payload()
    thr = p["meta"]["knee_threshold_pp"]
    for pol in ("LRU", "SRRIP"):
        m = p["per_policy"][pol]["per_regime"]["under_wss"]["median_gap_pp"]
        assert m > thr, (
            f"{pol} median gap at under_wss must exceed knee threshold "
            f"({thr:.2f} pp); got {m:.3f}"
        )


def test_oracle_aware_median_gap_at_under_wss_meets_threshold():
    # GRASP and POPT must meet knee threshold even under heavy pressure.
    p = _payload()
    thr = p["meta"]["knee_threshold_pp"]
    for pol in ("GRASP", "POPT"):
        m = p["per_policy"][pol]["per_regime"]["under_wss"]["median_gap_pp"]
        assert m <= thr, (
            f"{pol} median gap at under_wss must meet knee threshold "
            f"({thr:.2f} pp); got {m:.3f}"
        )


def test_every_policy_reaches_threshold_at_over_wss():
    # When the WSS comfortably fits, every policy should plateau.
    p = _payload()
    thr = p["meta"]["knee_threshold_pp"]
    for pol in ("GRASP", "LRU", "POPT", "SRRIP"):
        m = p["per_policy"][pol]["per_regime"]["over_wss"]["median_gap_pp"]
        assert m <= thr, (
            f"{pol} median gap at over_wss must meet knee threshold "
            f"({thr:.2f} pp); got {m:.3f}"
        )
