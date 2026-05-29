"""Pytest gate for anchor cell-level L3-sweep monotonicity replay.

Locks the current behaviour of wiki/data/anchor_monotonicity_replay.json:
gem5 is strictly monotone, sniper has bounded bumps under tier-aware
ceilings, and no tool exhibits a catastrophic (>=3 pp) regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

JSON_PATH = Path("wiki/data/anchor_monotonicity_replay.json")
MD_PATH = Path("wiki/data/anchor_monotonicity_replay.md")


def _load():
    if not JSON_PATH.exists():
        pytest.skip(f"{JSON_PATH} not generated yet")
    return json.loads(JSON_PATH.read_text())


# ---------- structural ----------


def test_schema_tag():
    blob = _load()
    assert blob.get("schema") == "anchor_monotonicity_replay/v1"


def test_both_tools_present():
    blob = _load()
    assert set(blob["per_tool"]) == {"gem5", "sniper"}


def test_expected_sizes_match_anchor_axis():
    blob = _load()
    expected = ["4kB", "32kB", "256kB", "2MB"]
    for tool, payload in blob["per_tool"].items():
        assert payload["expected_sizes"] == expected, tool


def test_cells_floor():
    blob = _load()
    # gem5 anchors: 2 cells x 3 policies = 6;  sniper anchors: 6 cells x 3 = 18.
    assert blob["per_tool"]["gem5"]["cells"] >= 6
    assert blob["per_tool"]["sniper"]["cells"] >= 18


# ---------- gem5 (strict) ----------


def test_gem5_zero_bumps():
    blob = _load()
    g = blob["per_tool"]["gem5"]
    assert g["bumps"] == 0, g["worst_bumps"]


def test_gem5_zero_hard_bumps():
    blob = _load()
    assert blob["per_tool"]["gem5"]["hard_bumps"] == 0


# ---------- sniper (tier-aware) ----------


def test_sniper_bump_rate_within_ceiling():
    blob = _load()
    s = blob["per_tool"]["sniper"]
    tol = s["evaluation"]["tolerances"]
    assert s["bump_rate_pct"] <= tol["bump_rate_max_pct"] + 1e-9


def test_sniper_hard_bumps_within_ceiling():
    blob = _load()
    s = blob["per_tool"]["sniper"]
    tol = s["evaluation"]["tolerances"]
    assert s["hard_bumps"] <= tol["hard_bumps_max"]


def test_sniper_max_bump_within_ceiling():
    blob = _load()
    s = blob["per_tool"]["sniper"]
    tol = s["evaluation"]["tolerances"]
    assert s["max_bump_pp"] <= tol["max_bump_pp_max"] + 1e-9


# ---------- overall + invariants ----------


def test_no_catastrophic_bumps_any_tool():
    blob = _load()
    assert blob["overall"]["catastrophic_bumps"] == []


def test_overall_verdict_pass():
    blob = _load()
    assert blob["overall"]["verdict_ok"] is True


def test_md_renders_verdict():
    if not MD_PATH.exists():
        pytest.skip(f"{MD_PATH} not generated yet")
    txt = MD_PATH.read_text()
    assert "Anchor cell L3-sweep monotonicity replay" in txt
    assert "**Verdict:** PASS" in txt
