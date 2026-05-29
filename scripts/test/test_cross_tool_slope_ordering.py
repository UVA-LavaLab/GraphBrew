"""Gate 72 — cross-tool SRRIP-vs-GRASP slope ordering invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "cross_tool_slope_ordering.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "cross_tool_slope_ordering.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload and "per_tool" in payload
    for k in ("tools", "gap_floor_pp_octave", "required_strict_tools",
              "n_strict_tools", "verdict_checks", "verdict"):
        assert k in payload["meta"], f"missing meta.{k}"


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_three_tools_compared(payload):
    """Cache-sim, gem5, sniper — exactly three tools."""
    assert payload["meta"]["tools"] == ["cache_sim", "gem5", "sniper"]


def test_all_tools_present_and_valid(payload):
    assert payload["meta"]["verdict_checks"]["all_tools_present_and_valid"]
    for tool in payload["meta"]["tools"]:
        e = payload["per_tool"][tool]
        assert e.get("present"), f"{tool} not present"
        assert "error" not in e, f"{tool} has error: {e.get('error')}"


def test_all_tools_srrip_at_least_as_steep_as_grasp(payload):
    """Hard gate — every tool must show SRRIP ≤ GRASP."""
    assert payload["meta"]["verdict_checks"]["all_tools_srrip_le_grasp"]
    for tool in payload["meta"]["tools"]:
        e = payload["per_tool"][tool]
        assert e["srrip_steeper"], (
            f"{tool}: SRRIP {e['srrip_median']} not <= GRASP "
            f"{e['grasp_median']}"
        )


def test_required_strict_tools_met(payload):
    """At least 2 of 3 tools must show strictly steeper SRRIP."""
    m = payload["meta"]
    assert m["n_strict_tools"] >= m["required_strict_tools"], (
        f"only {m['n_strict_tools']} of 3 tools strictly steeper; "
        f"need >= {m['required_strict_tools']}"
    )


def test_per_tool_medians_negative(payload):
    """Every per-tool GRASP/SRRIP median must be negative (cache helps)."""
    for tool in payload["meta"]["tools"]:
        e = payload["per_tool"][tool]
        assert e["grasp_median"] < 0, (
            f"{tool} GRASP median non-negative: {e['grasp_median']}"
        )
        assert e["srrip_median"] < 0, (
            f"{tool} SRRIP median non-negative: {e['srrip_median']}"
        )


def test_gap_floor_is_positive(payload):
    """Sanity: the gap floor that defines 'strictly steeper' must be > 0."""
    assert payload["meta"]["gap_floor_pp_octave"] > 0


def test_lru_minus_grasp_reported_per_tool(payload):
    """Each tool must report the LRU-vs-GRASP delta (informational; not
    gated). Catches a future silent removal of the diagnostic."""
    for tool in payload["meta"]["tools"]:
        e = payload["per_tool"][tool]
        assert "lru_minus_grasp_pp_oct" in e, (
            f"{tool} missing lru_minus_grasp_pp_oct"
        )
        assert e["lru_minus_grasp_pp_oct"] is not None, (
            f"{tool} has null lru_minus_grasp_pp_oct (LRU median absent?)"
        )


def test_srrip_minus_grasp_arithmetic_is_consistent(payload):
    """srrip_minus_grasp must equal srrip_median - grasp_median to 4dp."""
    for tool in payload["meta"]["tools"]:
        e = payload["per_tool"][tool]
        expected = round(e["srrip_median"] - e["grasp_median"], 4)
        assert abs(e["srrip_minus_grasp_pp_oct"] - expected) < 1e-3, (
            f"{tool}: arithmetic inconsistency: "
            f"srrip-grasp={e['srrip_minus_grasp_pp_oct']!r}, "
            f"srrip={e['srrip_median']!r}, grasp={e['grasp_median']!r}"
        )


def test_cache_sim_uses_capacity_sensitivity_artifact(payload):
    """The cache-sim slot must read from gate 66's capacity_sensitivity
    artifact (not a stale alternative)."""
    e = payload["per_tool"]["cache_sim"]
    assert "capacity_sensitivity" in e.get("source", ""), e


def test_anchor_sources_are_slope_replays(payload):
    """gem5 and sniper slots must read from the slope replay artifacts
    of gates 70 and 71."""
    assert "gem5_slope_replay" in payload["per_tool"]["gem5"]["source"]
    assert "sniper_slope_replay" in payload["per_tool"]["sniper"]["source"]
