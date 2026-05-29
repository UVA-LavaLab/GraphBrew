"""Pytest gate for cross-tool shared-anchor slope-sign agreement (gate 82).

Locks the physical replication invariant on the (graph, app, policy)
cells present in BOTH gem5 and sniper anchor sweeps: 100% sign
agreement, 100% both-negative, 100% sniper-steeper, bounded |slope|
difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

JSON_PATH = Path("wiki/data/anchor_cross_tool_agreement.json")
MD_PATH = Path("wiki/data/anchor_cross_tool_agreement.md")


def _load():
    if not JSON_PATH.exists():
        pytest.skip(f"{JSON_PATH} not generated yet")
    return json.loads(JSON_PATH.read_text())


def test_schema_tag():
    blob = _load()
    assert blob["schema"] == "anchor_cross_tool_agreement/v1"


def test_shared_cells_floor():
    blob = _load()
    assert blob["meta"]["shared_cells"] >= 3


def test_shared_cells_keyset():
    blob = _load()
    keys = {(c["graph"], c["app"], c["policy"]) for c in blob["shared_cells"]}
    # Current shared cells: (email-Eu-core, pr) x {GRASP, LRU, SRRIP}.
    expected = {
        ("email-Eu-core", "pr", "GRASP"),
        ("email-Eu-core", "pr", "LRU"),
        ("email-Eu-core", "pr", "SRRIP"),
    }
    assert expected <= keys, sorted(keys)


def test_all_cells_sign_match():
    blob = _load()
    assert all(c["sign_match"] for c in blob["shared_cells"])


def test_all_cells_both_negative():
    blob = _load()
    assert all(c["both_negative"] for c in blob["shared_cells"])


def test_all_cells_sniper_steeper():
    blob = _load()
    assert all(c["sniper_steeper"] for c in blob["shared_cells"])


def test_sign_agreement_check_ok():
    blob = _load()
    assert blob["checks"]["sign_agreement"]["ok"]


def test_both_negative_check_ok():
    blob = _load()
    assert blob["checks"]["both_negative"]["ok"]


def test_sniper_steeper_check_ok():
    blob = _load()
    assert blob["checks"]["sniper_steeper"]["ok"]


def test_abs_diff_within_ceiling():
    blob = _load()
    assert blob["checks"]["abs_diff_ceiling"]["ok"]


def test_per_cell_slope_magnitudes_nontrivial():
    """Both tools must report |slope| >= 1 pp/oct on every shared cell --
    rules out a degenerate 'both zero, both match' false-pass."""
    blob = _load()
    for c in blob["shared_cells"]:
        assert abs(c["gem5_slope_pp"]) >= 1.0, c
        assert abs(c["sniper_slope_pp"]) >= 1.0, c


def test_verdict_pass():
    blob = _load()
    assert blob["verdict_ok"] is True


def test_md_renders_verdict():
    if not MD_PATH.exists():
        pytest.skip(f"{MD_PATH} not generated yet")
    txt = MD_PATH.read_text()
    assert "Cross-tool shared-anchor slope-sign agreement" in txt
    assert "**Verdict:** PASS" in txt
