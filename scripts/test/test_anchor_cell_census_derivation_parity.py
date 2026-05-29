"""Gate 194 (ACC-Der) — derivation parity of anchor_cell_census.json.

Locks the byte-for-byte derivation of wiki/data/anchor_cell_census.json
from its TWO upstreams (gem5_slope_replay.json + sniper_slope_replay.json)
so any silent shrinkage in either anchor's cell coverage trips a pytest
gate before the dashboard regen step. Pins the EXPECTED_L3_AXIS /
EXPECTED_POLICIES / EXPECTED_GEM5_CELLS / EXPECTED_SNIPER_CELLS baseline
constants and the 13-check verdict matrix.

Five test groups:
  1. meta:               pinned baseline constants (axes, policies,
                         cell tuples).
  2. summary:            _summarize() round-trip; cells/policies sorted;
                         JSON-list cell encoding; cell_policy_records ==
                         len(per_cell).
  3. verdict_checks:     all 13 checks reproduce the documented
                         predicates and polarity; verdict = PASS iff
                         all True.
  4. shared:             intersection sorted; share-counter consistency;
                         expected baseline includes shared (email-Eu-core,
                         pr) by construction.
  5. byte parity:        JSON layout `json.dumps(..., indent=2) + "\\n"`
                         WITHOUT sort_keys.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "anchor_cell_census.py"
ARTIFACT_PATH = REPO_ROOT / "wiki" / "data" / "anchor_cell_census.json"
GEM5_PATH = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
SNIPER_PATH = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("acc_gen", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen()
ARTIFACT = json.loads(ARTIFACT_PATH.read_text())
REBUILT = GEN.build(GEM5_PATH, SNIPER_PATH)


# ---------------------------------------------------------------------------
# Group 1 — meta constants
# ---------------------------------------------------------------------------

def test_expected_l3_axis_pinned():
    assert GEN.EXPECTED_L3_AXIS == ["4kB", "32kB", "256kB", "2MB"]


def test_expected_policies_pinned():
    assert GEN.EXPECTED_POLICIES == ["GRASP", "LRU", "SRRIP"]


def test_expected_gem5_cells_pinned():
    assert GEN.EXPECTED_GEM5_CELLS == [
        ("email-Eu-core", "bc"),
        ("email-Eu-core", "pr"),
    ]


def test_expected_sniper_cells_pinned():
    assert GEN.EXPECTED_SNIPER_CELLS == [
        ("cit-Patents",   "bfs"),
        ("cit-Patents",   "pr"),
        ("cit-Patents",   "sssp"),
        ("email-Eu-core", "bfs"),
        ("email-Eu-core", "pr"),
        ("email-Eu-core", "sssp"),
    ]


def test_min_cell_count_constants():
    assert GEN.GEM5_MIN_CELLS == len(GEN.EXPECTED_GEM5_CELLS)
    assert GEN.SNIPER_MIN_CELLS == len(GEN.EXPECTED_SNIPER_CELLS)


def test_meta_expected_blocks_match_constants():
    m = ARTIFACT["meta"]
    assert m["expected_l3_axis"] == GEN.EXPECTED_L3_AXIS
    assert m["expected_policies"] == GEN.EXPECTED_POLICIES
    assert m["expected_gem5_cells"] == [list(c) for c in GEN.EXPECTED_GEM5_CELLS]
    assert m["expected_sniper_cells"] == [list(c) for c in GEN.EXPECTED_SNIPER_CELLS]


def test_meta_n_cells_minimum_match():
    assert ARTIFACT["meta"]["gem5"]["n_cells_minimum"] == GEN.GEM5_MIN_CELLS
    assert ARTIFACT["meta"]["sniper"]["n_cells_minimum"] == GEN.SNIPER_MIN_CELLS


# ---------------------------------------------------------------------------
# Group 2 — _summarize() shape
# ---------------------------------------------------------------------------

def test_summarize_gem5_record_shape():
    s = GEN._summarize(GEM5_PATH)
    assert set(s.keys()) == {
        "anchor_source", "cells", "n_cells",
        "policies", "l3_axis", "cell_policy_records",
    }


def test_summarize_cells_sorted_unique():
    for anchor in ("gem5", "sniper"):
        cells = ARTIFACT["meta"][anchor]["cells"]
        # Cells are lists in JSON; sort tuple-equivalents
        as_tuples = [tuple(c) for c in cells]
        assert as_tuples == sorted(set(as_tuples))


def test_summarize_policies_sorted_unique():
    for anchor in ("gem5", "sniper"):
        policies = ARTIFACT["meta"][anchor]["policies"]
        assert policies == sorted(set(policies))


def test_summarize_cells_encoded_as_lists():
    for anchor in ("gem5", "sniper"):
        for c in ARTIFACT["meta"][anchor]["cells"]:
            assert isinstance(c, list)
            assert len(c) == 2
            assert all(isinstance(x, str) for x in c)


def test_summarize_records_match_per_cell_length():
    gem5_doc = json.loads(GEM5_PATH.read_text())
    sniper_doc = json.loads(SNIPER_PATH.read_text())
    assert ARTIFACT["meta"]["gem5"]["cell_policy_records"] == len(gem5_doc["per_cell"])
    assert ARTIFACT["meta"]["sniper"]["cell_policy_records"] == len(sniper_doc["per_cell"])


def test_summarize_n_cells_matches_cell_count():
    for anchor in ("gem5", "sniper"):
        assert ARTIFACT["meta"][anchor]["n_cells"] == len(ARTIFACT["meta"][anchor]["cells"])


def test_summarize_l3_axis_from_expected_sizes():
    """l3_axis is taken from meta.expected_sizes of the upstream doc."""
    gem5_doc = json.loads(GEM5_PATH.read_text())
    sniper_doc = json.loads(SNIPER_PATH.read_text())
    assert ARTIFACT["meta"]["gem5"]["l3_axis"] == list(gem5_doc["meta"].get("expected_sizes", []))
    assert ARTIFACT["meta"]["sniper"]["l3_axis"] == list(sniper_doc["meta"].get("expected_sizes", []))


# ---------------------------------------------------------------------------
# Group 3 — verdict_checks polarity (all 13 keys)
# ---------------------------------------------------------------------------

def test_verdict_checks_keys_exact_thirteen():
    assert set(ARTIFACT["meta"]["verdict_checks"].keys()) == {
        "gem5_cell_count_at_or_above_baseline",
        "sniper_cell_count_at_or_above_baseline",
        "gem5_has_expected_cells",
        "sniper_has_expected_cells",
        "gem5_l3_axis_matches",
        "sniper_l3_axis_matches",
        "gem5_policy_set_matches",
        "sniper_policy_set_matches",
        "anchors_share_l3_axis",
        "anchors_share_policy_set",
        "anchors_share_at_least_one_cell",
        "gem5_cell_policy_records_match",
        "sniper_cell_policy_records_match",
    }


def test_min_count_inclusive_ge_baseline():
    g = ARTIFACT["meta"]["gem5"]
    s = ARTIFACT["meta"]["sniper"]
    assert ARTIFACT["meta"]["verdict_checks"]["gem5_cell_count_at_or_above_baseline"] is (
        g["n_cells"] >= g["n_cells_minimum"]
    )
    assert ARTIFACT["meta"]["verdict_checks"]["sniper_cell_count_at_or_above_baseline"] is (
        s["n_cells"] >= s["n_cells_minimum"]
    )


def test_has_expected_cells_is_subset_check():
    g_cells = {tuple(c) for c in ARTIFACT["meta"]["gem5"]["cells"]}
    s_cells = {tuple(c) for c in ARTIFACT["meta"]["sniper"]["cells"]}
    assert ARTIFACT["meta"]["verdict_checks"]["gem5_has_expected_cells"] is (
        set(GEN.EXPECTED_GEM5_CELLS).issubset(g_cells)
    )
    assert ARTIFACT["meta"]["verdict_checks"]["sniper_has_expected_cells"] is (
        set(GEN.EXPECTED_SNIPER_CELLS).issubset(s_cells)
    )


def test_l3_axis_check_is_exact_list_equality():
    assert ARTIFACT["meta"]["verdict_checks"]["gem5_l3_axis_matches"] is (
        ARTIFACT["meta"]["gem5"]["l3_axis"] == GEN.EXPECTED_L3_AXIS
    )
    assert ARTIFACT["meta"]["verdict_checks"]["sniper_l3_axis_matches"] is (
        ARTIFACT["meta"]["sniper"]["l3_axis"] == GEN.EXPECTED_L3_AXIS
    )


def test_policy_set_check_is_exact_list_equality():
    assert ARTIFACT["meta"]["verdict_checks"]["gem5_policy_set_matches"] is (
        ARTIFACT["meta"]["gem5"]["policies"] == GEN.EXPECTED_POLICIES
    )
    assert ARTIFACT["meta"]["verdict_checks"]["sniper_policy_set_matches"] is (
        ARTIFACT["meta"]["sniper"]["policies"] == GEN.EXPECTED_POLICIES
    )


def test_shared_l3_axis_check_consistency():
    expected = ARTIFACT["meta"]["gem5"]["l3_axis"] == ARTIFACT["meta"]["sniper"]["l3_axis"]
    assert ARTIFACT["meta"]["verdict_checks"]["anchors_share_l3_axis"] is expected
    assert ARTIFACT["meta"]["shared_l3_axis"] is expected


def test_shared_policies_check_consistency():
    expected = ARTIFACT["meta"]["gem5"]["policies"] == ARTIFACT["meta"]["sniper"]["policies"]
    assert ARTIFACT["meta"]["verdict_checks"]["anchors_share_policy_set"] is expected
    assert ARTIFACT["meta"]["shared_policies"] is expected


def test_record_count_match_product_of_cells_x_policies():
    g_expected = len(GEN.EXPECTED_GEM5_CELLS) * len(GEN.EXPECTED_POLICIES)
    s_expected = len(GEN.EXPECTED_SNIPER_CELLS) * len(GEN.EXPECTED_POLICIES)
    assert ARTIFACT["meta"]["verdict_checks"]["gem5_cell_policy_records_match"] is (
        ARTIFACT["meta"]["gem5"]["cell_policy_records"] == g_expected
    )
    assert ARTIFACT["meta"]["verdict_checks"]["sniper_cell_policy_records_match"] is (
        ARTIFACT["meta"]["sniper"]["cell_policy_records"] == s_expected
    )


def test_verdict_iff_all_checks_true():
    expected = "PASS" if all(ARTIFACT["meta"]["verdict_checks"].values()) else "FAIL"
    assert ARTIFACT["meta"]["verdict"] == expected


# ---------------------------------------------------------------------------
# Group 4 — shared cell intersection
# ---------------------------------------------------------------------------

def test_shared_cells_intersection_match():
    g = {tuple(c) for c in ARTIFACT["meta"]["gem5"]["cells"]}
    s = {tuple(c) for c in ARTIFACT["meta"]["sniper"]["cells"]}
    expected = sorted(g & s)
    actual = [tuple(c) for c in ARTIFACT["meta"]["shared_cells"]]
    assert actual == expected


def test_shared_cell_count_matches_list_length():
    assert ARTIFACT["meta"]["shared_cell_count"] == len(ARTIFACT["meta"]["shared_cells"])


def test_shared_includes_email_eu_core_pr():
    """The documented baseline guarantees this anchor exists in both."""
    shared = {tuple(c) for c in ARTIFACT["meta"]["shared_cells"]}
    assert ("email-Eu-core", "pr") in shared


def test_share_at_least_one_check_consistency():
    expected = ARTIFACT["meta"]["shared_cell_count"] >= 1
    assert ARTIFACT["meta"]["verdict_checks"]["anchors_share_at_least_one_cell"] is expected


# ---------------------------------------------------------------------------
# Group 5 — byte parity (no sort_keys; trailing newline)
# ---------------------------------------------------------------------------

def test_full_artifact_byte_parity():
    """Generator uses json.dumps(..., indent=2) + "\\n" WITHOUT sort_keys —
    insertion order of meta keys (gem5, sniper, expected_*, shared_*,
    verdict_checks, verdict) is load-bearing."""
    on_disk = ARTIFACT_PATH.read_text()
    rebuilt = json.dumps(REBUILT, indent=2) + "\n"
    assert on_disk == rebuilt
