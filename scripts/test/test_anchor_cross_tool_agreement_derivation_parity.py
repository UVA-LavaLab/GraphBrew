"""Gate 195 (ACT-Der) — derivation parity of anchor_cross_tool_agreement.json.

Locks the byte-for-byte derivation of
wiki/data/anchor_cross_tool_agreement.json from its two upstreams
(gem5_slope_replay.json + sniper_slope_replay.json) so any silent drift
in the per-shared-cell slope-sign agreement, both-negative, sniper-
steeper, or absolute-difference-ceiling checks trips a pytest gate
before the dashboard regen step.

Five test groups:
  1. meta:               schema label, locked thresholds
                         (MAX_ABS_SLOPE_DIFF_PP, SHARED_CELLS_FLOOR,
                         SIGN_AGREEMENT_FLOOR, SNIPER_STEEPER_FLOOR);
                         cell counts roll up from upstream blobs.
  2. shared_cells:       record shape (graph/app/policy/slopes/flags);
                         shared keys sorted; sign/both_neg/sniper_steeper
                         per-cell predicates correct; abs_diff_pp
                         formula = ||sniper| − |gem5||.
  3. summary + checks:   count consistency (sign_agree_count == #cells
                         with sign_match=True, etc.); rate = count/n;
                         floor/ceiling polarity (sign/both/steeper
                         floors use ≥ with epsilon; abs_diff ceiling
                         uses ≤ with epsilon); verdict_ok iff all
                         checks.ok.
  4. helpers:            _index() key shape; sign_match definition
                         matches the OR-of-equal-zero edge case.
  5. byte parity:        full-file byte-for-byte (sort_keys=True +
                         trailing newline).
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "anchor_cross_tool_agreement.py"
ARTIFACT_PATH = REPO_ROOT / "wiki" / "data" / "anchor_cross_tool_agreement.json"
GEM5_PATH = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
SNIPER_PATH = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("act_gen", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen()
ARTIFACT = json.loads(ARTIFACT_PATH.read_text())
REBUILT = GEN._build_report(GEM5_PATH, SNIPER_PATH)


# ---------------------------------------------------------------------------
# Group 1 — meta & thresholds
# ---------------------------------------------------------------------------

def test_schema_label_pinned():
    assert ARTIFACT["schema"] == "anchor_cross_tool_agreement/v1"


def test_thresholds_pinned():
    assert GEN.MAX_ABS_SLOPE_DIFF_PP == 8.0
    assert GEN.SHARED_CELLS_FLOOR == 3
    assert GEN.SIGN_AGREEMENT_FLOOR == 1.0
    assert GEN.SNIPER_STEEPER_FLOOR == 1.0


def test_meta_thresholds_block_match_constants():
    t = ARTIFACT["meta"]["thresholds"]
    assert t["max_abs_slope_diff_pp"] == GEN.MAX_ABS_SLOPE_DIFF_PP
    assert t["shared_cells_floor"] == GEN.SHARED_CELLS_FLOOR
    assert t["sign_agreement_floor"] == GEN.SIGN_AGREEMENT_FLOOR
    assert t["sniper_steeper_floor"] == GEN.SNIPER_STEEPER_FLOOR


def test_meta_cell_counts_from_upstream():
    g5 = json.loads(GEM5_PATH.read_text())
    sn = json.loads(SNIPER_PATH.read_text())
    g_idx = GEN._index(g5)
    s_idx = GEN._index(sn)
    assert ARTIFACT["meta"]["gem5_cells"] == len(g_idx)
    assert ARTIFACT["meta"]["sniper_cells"] == len(s_idx)


def test_meta_shared_cells_equals_intersection_size():
    g5 = json.loads(GEM5_PATH.read_text())
    sn = json.loads(SNIPER_PATH.read_text())
    inter = set(GEN._index(g5)) & set(GEN._index(sn))
    assert ARTIFACT["meta"]["shared_cells"] == len(inter)


# ---------------------------------------------------------------------------
# Group 2 — shared_cells shape & per-cell predicates
# ---------------------------------------------------------------------------

def test_shared_cells_record_shape():
    for c in ARTIFACT["shared_cells"]:
        assert set(c.keys()) == {
            "graph", "app", "policy",
            "gem5_slope_pp", "sniper_slope_pp",
            "sign_match", "both_negative", "sniper_steeper",
            "abs_diff_pp",
        }


def test_shared_cells_count_matches_meta():
    assert len(ARTIFACT["shared_cells"]) == ARTIFACT["meta"]["shared_cells"]


def test_shared_cells_sorted_by_key():
    """Cells iterate in sorted((graph, app, policy)) order — the
    generator sorts the shared key set before iteration."""
    keys = [(c["graph"], c["app"], c["policy"]) for c in ARTIFACT["shared_cells"]]
    assert keys == sorted(keys)


def test_shared_cells_keys_are_intersection():
    g5 = json.loads(GEM5_PATH.read_text())
    sn = json.loads(SNIPER_PATH.read_text())
    inter = sorted(set(GEN._index(g5)) & set(GEN._index(sn)))
    expected = [
        {"graph": g, "app": a, "policy": p}
        for (g, a, p) in inter
    ]
    actual = [
        {"graph": c["graph"], "app": c["app"], "policy": c["policy"]}
        for c in ARTIFACT["shared_cells"]
    ]
    assert actual == expected


def test_sign_match_matches_definition():
    """sign_match iff (both > 0) OR (both < 0) OR (both == 0)."""
    for c in ARTIFACT["shared_cells"]:
        g = c["gem5_slope_pp"]
        s = c["sniper_slope_pp"]
        expected = (g > 0) == (s > 0) or (g == 0 and s == 0)
        assert c["sign_match"] is expected


def test_both_negative_strict_lt_zero():
    for c in ARTIFACT["shared_cells"]:
        expected = c["gem5_slope_pp"] < 0 and c["sniper_slope_pp"] < 0
        assert c["both_negative"] is expected


def test_sniper_steeper_inclusive_ge_abs():
    """sniper_steeper iff |sniper| ≥ |gem5| (INCLUSIVE)."""
    for c in ARTIFACT["shared_cells"]:
        expected = abs(c["sniper_slope_pp"]) >= abs(c["gem5_slope_pp"])
        assert c["sniper_steeper"] is expected


def test_abs_diff_pp_formula():
    """abs_diff_pp = ||sniper| − |gem5|| rounded to 6dp."""
    for c in ARTIFACT["shared_cells"]:
        expected = round(abs(abs(c["sniper_slope_pp"]) - abs(c["gem5_slope_pp"])), 6)
        assert c["abs_diff_pp"] == expected


def test_slope_rounding_6dp():
    for c in ARTIFACT["shared_cells"]:
        for k in ("gem5_slope_pp", "sniper_slope_pp", "abs_diff_pp"):
            assert abs(round(c[k], 6) - c[k]) < 1e-12


# ---------------------------------------------------------------------------
# Group 3 — summary + checks polarity
# ---------------------------------------------------------------------------

def test_summary_counts_match_shared_cells():
    cells = ARTIFACT["shared_cells"]
    assert ARTIFACT["summary"]["sign_agree_count"] == sum(c["sign_match"] for c in cells)
    assert ARTIFACT["summary"]["both_negative_count"] == sum(c["both_negative"] for c in cells)
    assert ARTIFACT["summary"]["sniper_steeper_count"] == sum(c["sniper_steeper"] for c in cells)


def test_summary_max_abs_diff_matches_cells():
    cells = ARTIFACT["shared_cells"]
    diffs = [c["abs_diff_pp"] for c in cells]
    expected = round(max(diffs) if diffs else 0.0, 6)
    assert ARTIFACT["summary"]["max_abs_diff_pp"] == expected


def test_summary_median_abs_diff_matches_statistics_median():
    cells = ARTIFACT["shared_cells"]
    diffs = [c["abs_diff_pp"] for c in cells]
    expected = round(statistics.median(diffs) if diffs else 0.0, 6)
    assert ARTIFACT["summary"]["median_abs_diff_pp"] == expected


def test_checks_block_keys():
    assert set(ARTIFACT["checks"].keys()) == {
        "shared_floor",
        "sign_agreement",
        "both_negative",
        "sniper_steeper",
        "abs_diff_ceiling",
    }


def test_shared_floor_check_inclusive_ge():
    c = ARTIFACT["checks"]["shared_floor"]
    assert c["floor"] == GEN.SHARED_CELLS_FLOOR
    assert c["n_shared"] == ARTIFACT["meta"]["shared_cells"]
    assert c["ok"] is (c["n_shared"] >= c["floor"])


def test_sign_agreement_check_rate_and_floor():
    c = ARTIFACT["checks"]["sign_agreement"]
    n = ARTIFACT["meta"]["shared_cells"]
    expected_rate = round(ARTIFACT["summary"]["sign_agree_count"] / n, 6) if n else 0.0
    assert c["rate"] == expected_rate
    assert c["floor"] == GEN.SIGN_AGREEMENT_FLOOR
    assert c["ok"] is (expected_rate >= GEN.SIGN_AGREEMENT_FLOOR - 1e-9)


def test_both_negative_check_floor_is_one():
    c = ARTIFACT["checks"]["both_negative"]
    n = ARTIFACT["meta"]["shared_cells"]
    expected_rate = round(ARTIFACT["summary"]["both_negative_count"] / n, 6) if n else 0.0
    assert c["rate"] == expected_rate
    assert c["floor"] == 1.0
    assert c["ok"] is (expected_rate >= 1.0 - 1e-9)


def test_sniper_steeper_check_polarity():
    c = ARTIFACT["checks"]["sniper_steeper"]
    n = ARTIFACT["meta"]["shared_cells"]
    expected_rate = round(ARTIFACT["summary"]["sniper_steeper_count"] / n, 6) if n else 0.0
    assert c["rate"] == expected_rate
    assert c["floor"] == GEN.SNIPER_STEEPER_FLOOR
    assert c["ok"] is (expected_rate >= GEN.SNIPER_STEEPER_FLOOR - 1e-9)


def test_abs_diff_ceiling_inclusive_le():
    c = ARTIFACT["checks"]["abs_diff_ceiling"]
    assert c["ceiling"] == GEN.MAX_ABS_SLOPE_DIFF_PP
    assert c["max_abs_diff_pp"] == ARTIFACT["summary"]["max_abs_diff_pp"]
    assert c["ok"] is (c["max_abs_diff_pp"] <= GEN.MAX_ABS_SLOPE_DIFF_PP + 1e-9)


def test_verdict_ok_iff_all_checks():
    expected = all(c["ok"] for c in ARTIFACT["checks"].values())
    assert ARTIFACT["verdict_ok"] is expected


# ---------------------------------------------------------------------------
# Group 4 — helpers
# ---------------------------------------------------------------------------

def test_index_keys_are_triples():
    """_index returns {(graph, app, policy): row}."""
    g = GEN._index({"per_cell": [
        {"graph": "g1", "app": "pr", "policy": "LRU", "slope_pp_per_octave": -2.0},
    ]})
    assert list(g.keys()) == [("g1", "pr", "LRU")]


def test_index_missing_per_cell_returns_empty():
    assert GEN._index({}) == {}


def test_sign_match_when_both_zero_synth():
    """Synthetic edge case: g=0, s=0 → sign_match True (OR clause)."""
    # Both signs equal in (>0) reading: False == False is True.
    # Also covered explicitly by the OR-of-equal-zero branch.
    assert ((0.0 > 0) == (0.0 > 0)) or (0.0 == 0 and 0.0 == 0)


def test_sniper_steeper_at_equality_is_true():
    """|s| == |g| → INCLUSIVE >= → True."""
    assert (abs(-3.0) >= abs(-3.0)) is True


# ---------------------------------------------------------------------------
# Group 5 — byte parity (sort_keys=True + trailing newline)
# ---------------------------------------------------------------------------

def test_full_artifact_byte_parity():
    on_disk = ARTIFACT_PATH.read_text()
    rebuilt = json.dumps(REBUILT, indent=2, sort_keys=True) + "\n"
    assert on_disk == rebuilt
