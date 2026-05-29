"""Gate 70 — gem5 anchor slope sanity invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "gem5_slope_replay.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not JSON_PATH.exists():
        subprocess.check_call([sys.executable, str(GEN)], cwd=str(REPO_ROOT))
    return json.loads(JSON_PATH.read_text())


def test_payload_well_formed(payload):
    assert "meta" in payload and "per_cell" in payload
    meta = payload["meta"]
    for k in (
        "anchor_source",
        "n_cells",
        "n_cell_policy_records",
        "l3_axis_log2_kb",
        "expected_sizes",
        "policies",
        "per_policy",
        "lru_minus_grasp_pp_oct",
        "srrip_minus_grasp_pp_oct",
        "help_floor_pp_octave",
        "monotonic_violations",
        "verdict_checks",
        "verdict",
    ):
        assert k in meta, f"missing meta.{k}"


def test_anchor_source_is_gem5(payload):
    assert "gem5" in payload["meta"]["anchor_source"]


def test_verdict_pass(payload):
    assert payload["meta"]["verdict"] == "PASS", payload["meta"]


def test_has_at_least_one_cell(payload):
    """The gem5 anchor must contribute at least one (app, graph) cell."""
    assert payload["meta"]["n_cells"] >= 1, payload["meta"]


def test_axis_uses_four_anchor_sizes(payload):
    """The axis must be exactly the four documented anchor L3 sizes."""
    expected = ["4kB", "32kB", "256kB", "2MB"]
    assert payload["meta"]["expected_sizes"] == expected


def test_axis_is_log2_kb(payload):
    """Sanity-check the log2(kB) values to catch axis-flip regressions."""
    axis = payload["meta"]["l3_axis_log2_kb"]
    assert axis["4kB"] == 2.0
    assert axis["32kB"] == 5.0
    assert axis["256kB"] == 8.0
    assert axis["2MB"] == 11.0


def test_all_per_policy_medians_negative(payload):
    """Cache must materially help every policy at anchor scales."""
    per_policy = payload["meta"]["per_policy"]
    for p in ("GRASP", "LRU", "SRRIP"):
        med = per_policy[p]["median"]
        assert med is not None, f"{p} median missing"
        assert med < 0, f"{p} median slope is non-negative: {med!r}"


def test_srrip_at_least_as_steep_as_grasp(payload):
    """SRRIP is not oracle-aware and is more cache-hungry than GRASP."""
    p = payload["meta"]["per_policy"]
    assert p["SRRIP"]["median"] <= p["GRASP"]["median"], (
        f"SRRIP median {p['SRRIP']['median']!r} should be <= GRASP median "
        f"{p['GRASP']['median']!r}"
    )


def test_grasp_below_help_floor(payload):
    """GRASP median slope must be below the help-floor — cache helps it."""
    med = payload["meta"]["per_policy"]["GRASP"]["median"]
    floor = payload["meta"]["help_floor_pp_octave"]
    assert med < floor, f"GRASP median {med!r} not below help floor {floor!r}"


def test_cache_monotonicity_every_cell(payload):
    """Every cell must have miss(4kB) > miss(2MB)."""
    viols = payload["meta"]["monotonic_violations"]
    assert viols == [], f"monotonicity violations: {viols}"


def test_per_cell_slopes_internally_consistent(payload):
    """Per-cell slope must agree with miss(4kB)-vs-miss(2MB) sign — both
    must be non-positive when normal."""
    for r in payload["per_cell"]:
        ms = r["miss_pp_by_size"]
        assert r["slope_pp_per_octave"] <= 0, (
            f"per-cell positive slope detected: {r}"
        )
        assert ms["4kB"] >= ms["2MB"], (
            f"per-cell miss(4kB) < miss(2MB), monotonicity broken: {r}"
        )


def test_lru_minus_grasp_is_reported(payload):
    """The LRU-GRASP delta must be reported (even though it's INFORMATIONAL)
    so the data is available for downstream analyses and to catch a future
    silent removal of the diagnostic."""
    assert payload["meta"]["lru_minus_grasp_pp_oct"] is not None
    assert "lru_minus_grasp_note" in payload["meta"]
    note = payload["meta"]["lru_minus_grasp_note"]
    assert "INFORMATIONAL" in note
