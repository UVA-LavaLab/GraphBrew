"""Gate 193 (FSD-Der) — derivation parity of family_saturation_distance.json.

Locks the byte-for-byte derivation of wiki/data/family_saturation_distance.json
from its single upstream (saturation_distance.json#per_cell) so any
silent drift in the per-family saturation-distance replay generator
trips a pytest gate before the dashboard regen step.

Five test groups:
  1. meta:               pinned constants (GRAPH_FAMILY 8-graph map,
                         thresholds, pinned/high-headroom tuples).
  2. per_family:         record shape; n_cells/min/median/p90/max
                         consistency; graphs sorted+unique;
                         pico-sentinel filter applied at upstream.
  3. classification:     six verdict_checks reproduce the documented
                         predicates with the exact strict/inclusive
                         polarity (high floor inclusive >=, low ceiling
                         strict <, ordering inclusive >= with slack).
  4. helpers:            _median (even/odd/empty/single) and _p90
                         (bespoke percentile formula sanity).
  5. byte parity:        JSON layout is indent=2 + trailing newline
                         WITHOUT sort_keys (insertion-order load-bearing
                         for top-level meta keys).
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_saturation_distance.py"
ARTIFACT_PATH = REPO_ROOT / "wiki" / "data" / "family_saturation_distance.json"
DISTANCE_PATH = REPO_ROOT / "wiki" / "data" / "saturation_distance.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("fsd_gen", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen()
ARTIFACT = json.loads(ARTIFACT_PATH.read_text())
DISTANCE = json.loads(DISTANCE_PATH.read_text())
REBUILT = GEN.build(DISTANCE_PATH)


# ---------------------------------------------------------------------------
# Group 1 — meta constants & pinned invariants
# ---------------------------------------------------------------------------

def test_meta_graph_family_pinned_eight_graphs():
    assert GEN.GRAPH_FAMILY == {
        "email-Eu-core":    "social",
        "web-Google":       "web",
        "cit-Patents":      "citation",
        "soc-pokec":        "social",
        "soc-LiveJournal1": "social",
        "com-orkut":        "social",
        "roadNet-CA":       "road",
        "delaunay_n19":     "mesh",
    }


def test_meta_thresholds_pinned():
    assert GEN.HIGH_HEADROOM_FLOOR_PP == 5.0
    assert GEN.LOW_HEADROOM_CEILING_PP == 5.0
    assert GEN.ORDERING_SLACK_PP == 1.0


def test_meta_high_headroom_families_tuple():
    assert GEN.HIGH_HEADROOM_FAMILIES == ("citation", "social")
    assert isinstance(GEN.HIGH_HEADROOM_FAMILIES, tuple)


def test_meta_pinned_low_headroom_is_web_tuple():
    assert GEN.PINNED_LOW_HEADROOM == ("web",)
    assert isinstance(GEN.PINNED_LOW_HEADROOM, tuple)


def test_meta_high_low_partition_disjoint():
    assert set(GEN.HIGH_HEADROOM_FAMILIES) & set(GEN.PINNED_LOW_HEADROOM) == set()


def test_meta_families_alphabetical():
    fs = ARTIFACT["meta"]["families"]
    assert fs == sorted(fs)


def test_meta_families_match_rebuild():
    assert ARTIFACT["meta"]["families"] == REBUILT["meta"]["families"]


def test_meta_thresholds_appear_in_artifact():
    m = ARTIFACT["meta"]
    assert m["high_headroom_floor_pp"] == 5.0
    assert m["low_headroom_ceiling_pp"] == 5.0
    assert m["ordering_slack_pp"] == 1.0


def test_meta_source_artifact_relative_string():
    """source_artifact records the upstream path as a relative string."""
    assert ARTIFACT["meta"]["source_artifact"] == "wiki/data/saturation_distance.json"


def test_meta_high_low_lists_match_pinned_tuples():
    assert ARTIFACT["meta"]["high_headroom_families"] == list(GEN.HIGH_HEADROOM_FAMILIES)
    assert ARTIFACT["meta"]["pinned_low_headroom_families"] == list(GEN.PINNED_LOW_HEADROOM)


# ---------------------------------------------------------------------------
# Group 2 — per_family record shape & sentinel filter
# ---------------------------------------------------------------------------

def test_per_family_record_shape():
    for fam, rec in ARTIFACT["meta"]["per_family"].items():
        assert set(rec.keys()) == {
            "n_cells", "min_pp", "median_pp", "p90_pp", "max_pp", "graphs",
        }
        assert rec["n_cells"] >= 1
        assert rec["min_pp"] <= rec["median_pp"] <= rec["max_pp"] + 1e-9
        assert rec["min_pp"] <= rec["p90_pp"] <= rec["max_pp"] + 1e-9


def test_per_family_graphs_sorted_unique():
    for fam, rec in ARTIFACT["meta"]["per_family"].items():
        assert rec["graphs"] == sorted(set(rec["graphs"]))


def test_per_family_graphs_subset_of_pinned_mapping():
    for fam, rec in ARTIFACT["meta"]["per_family"].items():
        for g in rec["graphs"]:
            assert g in GEN.GRAPH_FAMILY
            assert GEN.GRAPH_FAMILY[g] == fam


def test_per_family_n_cells_matches_upstream_after_sentinel_filter():
    """Cell count per family equals upstream rows minus pico sentinels."""
    by_fam: dict[str, int] = defaultdict(int)
    for r in DISTANCE["per_cell"]:
        if r.get("is_pico_sentinel"):
            continue
        fam = GEN.GRAPH_FAMILY.get(r["graph"], "unknown")
        by_fam[fam] += 1
    for fam, rec in ARTIFACT["meta"]["per_family"].items():
        assert rec["n_cells"] == by_fam[fam]


def test_per_family_pico_sentinels_are_excluded():
    """No family aggregates over pico-sentinel rows."""
    sentinel_graphs = {
        r["graph"]
        for r in DISTANCE["per_cell"]
        if r.get("is_pico_sentinel")
    }
    # If sentinels exist, the sentinel-only graphs must NOT contribute
    # to any family's cell count beyond what the non-sentinel rows allow.
    if sentinel_graphs:
        non_sentinel_graphs = {
            r["graph"]
            for r in DISTANCE["per_cell"]
            if not r.get("is_pico_sentinel")
        }
        sentinel_only = sentinel_graphs - non_sentinel_graphs
        for fam, rec in ARTIFACT["meta"]["per_family"].items():
            for g in rec["graphs"]:
                assert g not in sentinel_only


def test_per_family_rounding_4dp():
    """All emitted floats round to 4 decimal places."""
    for fam, rec in ARTIFACT["meta"]["per_family"].items():
        for k in ("min_pp", "median_pp", "p90_pp", "max_pp"):
            assert abs(round(rec[k], 4) - rec[k]) < 1e-12


# ---------------------------------------------------------------------------
# Group 3 — verdict_checks polarity (strict vs inclusive)
# ---------------------------------------------------------------------------

def test_verdict_checks_keys():
    assert set(ARTIFACT["meta"]["verdict_checks"].keys()) == {
        "all_family_medians_nonneg",
        "all_family_mins_nonneg",
        "high_headroom_families_meet_floor",
        "pinned_low_headroom_under_ceiling",
        "family_ordering_citation_social_web",
        "at_least_three_families_present",
    }


def test_medians_nonneg_iff_all_medians_ge_zero():
    expected = all(
        rec["median_pp"] >= 0.0
        for rec in ARTIFACT["meta"]["per_family"].values()
    )
    assert ARTIFACT["meta"]["verdict_checks"]["all_family_medians_nonneg"] is expected


def test_mins_nonneg_iff_all_mins_ge_zero():
    expected = all(
        rec["min_pp"] >= 0.0
        for rec in ARTIFACT["meta"]["per_family"].values()
    )
    assert ARTIFACT["meta"]["verdict_checks"]["all_family_mins_nonneg"] is expected


def test_high_floor_is_inclusive_ge():
    pf = ARTIFACT["meta"]["per_family"]
    present = [f for f in GEN.HIGH_HEADROOM_FAMILIES if f in pf]
    expected = all(pf[f]["median_pp"] >= GEN.HIGH_HEADROOM_FLOOR_PP for f in present)
    assert ARTIFACT["meta"]["verdict_checks"]["high_headroom_families_meet_floor"] is expected


def test_low_ceiling_is_strict_lt():
    pf = ARTIFACT["meta"]["per_family"]
    present = [f for f in GEN.PINNED_LOW_HEADROOM if f in pf]
    expected = all(pf[f]["median_pp"] < GEN.LOW_HEADROOM_CEILING_PP for f in present)
    assert ARTIFACT["meta"]["verdict_checks"]["pinned_low_headroom_under_ceiling"] is expected


def test_family_ordering_inclusive_with_slack():
    pf = ARTIFACT["meta"]["per_family"]
    if all(f in pf for f in ("citation", "social", "web")):
        cit = pf["citation"]["median_pp"]
        soc = pf["social"]["median_pp"]
        web = pf["web"]["median_pp"]
        expected = (
            (cit + GEN.ORDERING_SLACK_PP >= soc)
            and (soc + GEN.ORDERING_SLACK_PP >= web)
        )
    else:
        expected = True  # default-true path
    assert ARTIFACT["meta"]["verdict_checks"]["family_ordering_citation_social_web"] is expected


def test_three_families_inclusive_ge_three():
    n = sum(
        1 for rec in ARTIFACT["meta"]["per_family"].values()
        if rec["n_cells"] >= 1
    )
    expected = n >= 3
    assert ARTIFACT["meta"]["verdict_checks"]["at_least_three_families_present"] is expected


def test_verdict_iff_all_checks_true():
    expected = "PASS" if all(ARTIFACT["meta"]["verdict_checks"].values()) else "FAIL"
    assert ARTIFACT["meta"]["verdict"] == expected


# ---------------------------------------------------------------------------
# Group 4 — helpers
# ---------------------------------------------------------------------------

def test_median_empty_returns_zero():
    assert GEN._median([]) == 0.0


def test_median_single_returns_element():
    assert GEN._median([3.5]) == 3.5


def test_median_odd_returns_middle():
    assert GEN._median([5.0, 1.0, 3.0]) == 3.0


def test_median_even_returns_pair_average():
    assert GEN._median([1.0, 3.0, 5.0, 7.0]) == 4.0


def test_p90_empty_returns_zero():
    assert GEN._p90([]) == 0.0


def test_p90_single_returns_element():
    assert GEN._p90([42.0]) == 42.0


def test_p90_bespoke_index_formula():
    """idx = max(0, min(n-1, int(round(0.9·(n-1))))) over sorted xs."""
    xs = list(range(1, 11))  # 1..10 → n=10, idx = round(0.9·9) = 8 → s[8] = 9
    assert GEN._p90(xs) == 9


def test_p90_two_element_picks_max():
    """n=2 → idx = round(0.9·1) = 1 → s[1] (max element)."""
    assert GEN._p90([1.0, 9.0]) == 9.0


# ---------------------------------------------------------------------------
# Group 5 — byte parity (insertion-order JSON layout)
# ---------------------------------------------------------------------------

def test_full_artifact_byte_parity():
    """Generator uses json.dumps(..., indent=2) + "\\n" WITHOUT
    sort_keys, so insertion order of meta keys is load-bearing."""
    on_disk = ARTIFACT_PATH.read_text()
    rebuilt = json.dumps(REBUILT, indent=2) + "\n"
    assert on_disk == rebuilt
