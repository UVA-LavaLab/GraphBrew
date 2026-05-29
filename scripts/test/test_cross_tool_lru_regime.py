"""Gate 74 — cross-tool LRU-vs-GRASP regime inversion invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "cross_tool_lru_regime.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "cross_tool_lru_regime.json"
CACHE_SIM_JSON = REPO_ROOT / "wiki" / "data" / "capacity_sensitivity.json"
GEM5_JSON      = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
SNIPER_JSON    = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"

EXPECTED_TOOLS = {"cache-sim", "gem5", "sniper"}
POSTWSS_GAP_FLOOR_PP_OCT = 0.30
SUBWSS_TOLERANCE_PP = 0.20


@pytest.fixture(scope="module")
def payload() -> dict:
    for sib, name in (
        (CACHE_SIM_JSON, "capacity_sensitivity.json (gate 66)"),
        (GEM5_JSON,      "gem5_slope_replay.json (gate 70)"),
        (SNIPER_JSON,    "sniper_slope_replay.json (gate 71)"),
    ):
        if not sib.exists():
            pytest.skip(f"{name} missing — run upstream gate first")
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload
    meta = payload["meta"]
    for k in (
        "tools",
        "postwss_gap_floor_pp_oct",
        "subwss_tolerance_pp",
        "tool_results",
        "regime_inversion_holds",
        "verdict_checks",
        "verdict",
    ):
        assert k in meta, f"missing meta.{k}"


def test_all_three_tools_present(payload):
    tools = set(payload["meta"]["tool_results"].keys())
    assert tools == EXPECTED_TOOLS, (
        f"tool set drift: {tools} != {EXPECTED_TOOLS}"
    )


def test_each_tool_has_full_block(payload):
    for tool, t in payload["meta"]["tool_results"].items():
        for k in (
            "grasp_pp_oct",
            "lru_pp_oct",
            "lru_minus_grasp_pp_oct",
            "l3_min_kb",
            "l3_max_kb",
            "regime",
        ):
            assert k in t, f"tool {tool!r}: missing {k}"
        assert t["lru_minus_grasp_pp_oct"] is not None, (
            f"tool {tool!r}: LRU-GRASP delta is None"
        )


def test_cache_sim_classified_postwss(payload):
    cs = payload["meta"]["tool_results"]["cache-sim"]
    assert cs["regime"] == "post-WSS", (
        f"cache-sim regime should be post-WSS (1MB-8MB), got {cs['regime']!r}"
    )


def test_anchor_tools_classified_subwss(payload):
    for tool in ("gem5", "sniper"):
        t = payload["meta"]["tool_results"][tool]
        assert t["regime"] == "sub-WSS", (
            f"{tool} regime should be sub-WSS (4kB-2MB), got {t['regime']!r}"
        )


def test_cache_sim_lru_strictly_steeper(payload):
    cs = payload["meta"]["tool_results"]["cache-sim"]
    delta = cs["lru_minus_grasp_pp_oct"]
    assert delta <= -POSTWSS_GAP_FLOOR_PP_OCT, (
        f"cache-sim LRU-GRASP delta {delta:+.4f} not below "
        f"-{POSTWSS_GAP_FLOOR_PP_OCT} (post-WSS LRU should be strictly steeper)"
    )


def test_anchor_tools_not_strictly_steeper(payload):
    for tool in ("gem5", "sniper"):
        t = payload["meta"]["tool_results"][tool]
        delta = t["lru_minus_grasp_pp_oct"]
        assert delta >= -SUBWSS_TOLERANCE_PP, (
            f"{tool} LRU-GRASP delta {delta:+.4f} below "
            f"-{SUBWSS_TOLERANCE_PP} (sub-WSS LRU not expected to be "
            f"strictly steeper than GRASP)"
        )


def test_regime_inversion_holds(payload):
    assert payload["meta"]["regime_inversion_holds"] is True, (
        "regime_inversion_holds is False: signs of cache-sim and both "
        "anchor deltas do not show the expected inversion"
    )


def test_sign_inversion_explicit(payload):
    cs = payload["meta"]["tool_results"]["cache-sim"]["lru_minus_grasp_pp_oct"]
    g5 = payload["meta"]["tool_results"]["gem5"]["lru_minus_grasp_pp_oct"]
    sn = payload["meta"]["tool_results"]["sniper"]["lru_minus_grasp_pp_oct"]
    assert cs < 0.0, f"cache-sim delta {cs:+.4f} not negative (post-WSS)"
    assert g5 >= 0.0 and sn >= 0.0, (
        f"anchor deltas not >= 0: gem5={g5:+.4f} sniper={sn:+.4f}"
    )


def test_anchor_tools_sign_agreement(payload):
    g5 = payload["meta"]["tool_results"]["gem5"]["lru_minus_grasp_pp_oct"]
    sn = payload["meta"]["tool_results"]["sniper"]["lru_minus_grasp_pp_oct"]
    assert (g5 >= 0.0) == (sn >= 0.0), (
        f"anchor tools disagree on sub-WSS LRU-GRASP sign: "
        f"gem5={g5:+.4f} sniper={sn:+.4f}"
    )


def test_all_verdict_checks_green(payload):
    failed = [k for k, v in payload["meta"]["verdict_checks"].items() if not v]
    assert not failed, f"failed verdict checks: {failed}"


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", (
        f"verdict={payload['meta']['verdict']}, "
        f"checks={payload['meta']['verdict_checks']}"
    )
