"""Gate 191 (FSR-Der) — derivation parity of family_slope_replay.json.

Locks the byte-for-byte derivation of wiki/data/family_slope_replay.json
from its single upstream (oracle_gap.json#rows) so any silent drift in
the per-family OLS-slope replay generator trips a pytest gate before
the dashboard regen step.

Five test groups:
  1. meta:                pinned constants (POLICIES, ORACLE_AWARE,
                          NON_ORACLE, HELP_FLOOR_PP_OCTAVE, L3 axis,
                          PINNED_DEVIATING_FAMILIES = ("social",)).
  2. per-family:          record shape + invariant equivalences
                          (qualifying, replays, deviating).
  3. classification:      three independent invariants combined by AND
                          (lru < grasp, srrip < grasp, all < help_floor).
  4. helpers:             _ols_slope edge cases (single point, vertical
                          axis, n=2) and _median (even/odd/empty).
  5. byte parity:         full-file byte-for-byte vs build(...).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_slope_replay.py"
ARTIFACT_PATH = REPO_ROOT / "wiki" / "data" / "family_slope_replay.json"
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("fsr_gen", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen()
ARTIFACT = json.loads(ARTIFACT_PATH.read_text())
ORACLE = json.loads(ORACLE_PATH.read_text())
REBUILT = GEN.build(ORACLE)


# ---------------------------------------------------------------------------
# Group 1 — meta constants & pinned invariants
# ---------------------------------------------------------------------------

def test_meta_policies_alpha_quadruple():
    assert GEN.POLICIES == ("GRASP", "LRU", "POPT", "SRRIP")


def test_meta_oracle_aware_tuple():
    assert isinstance(GEN.ORACLE_AWARE, tuple)
    assert GEN.ORACLE_AWARE == ("GRASP", "POPT")


def test_meta_non_oracle_tuple():
    assert isinstance(GEN.NON_ORACLE, tuple)
    assert GEN.NON_ORACLE == ("LRU", "SRRIP")


def test_meta_oracle_non_oracle_partition():
    assert set(GEN.ORACLE_AWARE) | set(GEN.NON_ORACLE) == set(GEN.POLICIES)
    assert set(GEN.ORACLE_AWARE) & set(GEN.NON_ORACLE) == set()


def test_meta_l3_axis_non_uniform():
    assert GEN.L3_LOG2_MB == {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
    assert GEN.L3_SIZES == ("1MB", "4MB", "8MB")


def test_meta_help_floor_pinned():
    assert GEN.HELP_FLOOR_PP_OCTAVE == -5.0


def test_meta_pinned_deviating_is_social_tuple():
    assert GEN.PINNED_DEVIATING_FAMILIES == ("social",)
    assert isinstance(GEN.PINNED_DEVIATING_FAMILIES, tuple)


def test_meta_qualifying_alphabetical():
    qs = ARTIFACT["meta"]["qualifying_families"]
    assert qs == sorted(qs)


def test_meta_qualifying_match():
    assert ARTIFACT["meta"]["qualifying_families"] == REBUILT["meta"]["qualifying_families"]


def test_meta_deviating_match():
    assert ARTIFACT["meta"]["deviating_families"] == REBUILT["meta"]["deviating_families"]


def test_meta_new_deviating_formula():
    deviating = ARTIFACT["meta"]["deviating_families"]
    pinned = ARTIFACT["meta"]["pinned_deviating_families"]
    expected = [f for f in deviating if f not in pinned]
    assert ARTIFACT["meta"]["new_deviating_families"] == expected


def test_meta_replay_count_consistency():
    qs = ARTIFACT["meta"]["qualifying_families"]
    actual = sum(1 for f in qs if ARTIFACT["per_family"][f]["replays_pattern"])
    assert actual == ARTIFACT["meta"]["replay_count"]


def test_meta_verdict_closed_form():
    rc = ARTIFACT["meta"]["replay_count"]
    new_dev = ARTIFACT["meta"]["new_deviating_families"]
    expected = "PASS" if (rc >= 1 and not new_dev) else "FAIL"
    assert ARTIFACT["meta"]["verdict"] == expected


# ---------------------------------------------------------------------------
# Group 2 — per-family record shape
# ---------------------------------------------------------------------------

def test_per_family_record_shape():
    for fam, rec in ARTIFACT["per_family"].items():
        assert set(rec.keys()) == {
            "per_policy",
            "replays_pattern",
            "lru_steeper_than_grasp",
            "srrip_steeper_than_grasp",
            "all_policies_helped",
        }


def test_per_family_per_policy_subset_of_policies():
    for fam, rec in ARTIFACT["per_family"].items():
        assert set(rec["per_policy"].keys()) <= set(GEN.POLICIES)


def test_per_family_per_policy_record_shape():
    for fam, rec in ARTIFACT["per_family"].items():
        for pol, pp in rec["per_policy"].items():
            assert set(pp.keys()) == {
                "n_cells", "median_pp", "mean_pp",
                "min_pp", "max_pp", "is_oracle_aware",
            }
            assert pp["n_cells"] >= 1
            assert pp["is_oracle_aware"] == (pol in GEN.ORACLE_AWARE)
            assert pp["min_pp"] <= pp["median_pp"] <= pp["max_pp"] + 1e-9
            assert pp["min_pp"] <= pp["mean_pp"] <= pp["max_pp"] + 1e-9


def test_per_family_qualifying_iff_nonempty_per_policy():
    """Qualifying families always have at least one per_policy entry."""
    for fam in ARTIFACT["meta"]["qualifying_families"]:
        assert len(ARTIFACT["per_family"][fam]["per_policy"]) >= 1


def test_per_family_deviating_iff_not_replays():
    qs = ARTIFACT["meta"]["qualifying_families"]
    deviating = ARTIFACT["meta"]["deviating_families"]
    expected = [f for f in qs if not ARTIFACT["per_family"][f]["replays_pattern"]]
    assert deviating == expected


# ---------------------------------------------------------------------------
# Group 3 — classification invariants
# ---------------------------------------------------------------------------

def test_replays_implies_three_invariants():
    for fam, rec in ARTIFACT["per_family"].items():
        if rec["replays_pattern"]:
            assert rec["lru_steeper_than_grasp"] is True
            assert rec["srrip_steeper_than_grasp"] is True
            assert rec["all_policies_helped"] is True


def test_lru_steeper_iff_median_lt_grasp():
    for fam, rec in ARTIFACT["per_family"].items():
        pp = rec["per_policy"]
        if "LRU" in pp and "GRASP" in pp:
            expected = pp["LRU"]["median_pp"] < pp["GRASP"]["median_pp"]
            assert rec["lru_steeper_than_grasp"] is expected
        else:
            assert rec["lru_steeper_than_grasp"] is False


def test_srrip_steeper_iff_median_lt_grasp():
    for fam, rec in ARTIFACT["per_family"].items():
        pp = rec["per_policy"]
        if "SRRIP" in pp and "GRASP" in pp:
            expected = pp["SRRIP"]["median_pp"] < pp["GRASP"]["median_pp"]
            assert rec["srrip_steeper_than_grasp"] is expected
        else:
            assert rec["srrip_steeper_than_grasp"] is False


def test_all_policies_helped_strict_lt_help_floor():
    for fam, rec in ARTIFACT["per_family"].items():
        pp = rec["per_policy"]
        if not pp:
            assert rec["all_policies_helped"] is True
            continue
        expected = all(
            entry["median_pp"] < GEN.HELP_FLOOR_PP_OCTAVE
            for entry in pp.values()
        )
        assert rec["all_policies_helped"] is expected


def test_replays_is_pure_conjunction():
    for fam, rec in ARTIFACT["per_family"].items():
        expected = (
            rec["lru_steeper_than_grasp"]
            and rec["srrip_steeper_than_grasp"]
            and rec["all_policies_helped"]
        )
        assert rec["replays_pattern"] is expected


def test_social_family_pinned_deviation_stays_pinned():
    """The known-pinned social family must continue to deviate.

    If this ever flips to replays_pattern=True, the pin can be removed
    — but until then the gate locks the documented deviation in place.
    """
    if "social" in ARTIFACT["meta"]["qualifying_families"]:
        assert ARTIFACT["per_family"]["social"]["replays_pattern"] is False
        assert "social" in ARTIFACT["meta"]["deviating_families"]
        assert "social" in ARTIFACT["meta"]["pinned_deviating_families"]
        assert "social" not in ARTIFACT["meta"]["new_deviating_families"]


# ---------------------------------------------------------------------------
# Group 4 — helpers
# ---------------------------------------------------------------------------

def test_ols_slope_single_point_returns_none():
    assert GEN._ols_slope([(0.0, 1.0)]) is None


def test_ols_slope_two_points_returns_simple_slope():
    # (0,0) and (2,4) → slope = 2
    assert GEN._ols_slope([(0.0, 0.0), (2.0, 4.0)]) == pytest.approx(2.0)


def test_ols_slope_vertical_axis_returns_none():
    # All x equal → denominator 0 → None
    assert GEN._ols_slope([(1.0, 0.0), (1.0, 5.0), (1.0, 7.0)]) is None


def test_ols_slope_matches_hand_value_three_points():
    # Hand-fit OLS on points along log2 axis 0/2/3:
    # miss_rate_pp 30, 12, 6 → slope ~ -7.71
    pts = [(0.0, 30.0), (2.0, 12.0), (3.0, 6.0)]
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    expected = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    assert GEN._ols_slope(pts) == pytest.approx(expected)


def test_median_empty_returns_zero():
    assert GEN._median([]) == 0.0


def test_median_odd_returns_middle():
    assert GEN._median([3.0, 1.0, 2.0]) == 2.0


def test_median_even_returns_pair_average():
    assert GEN._median([1.0, 3.0, 5.0, 7.0]) == 4.0


def test_median_single_returns_element():
    assert GEN._median([42.0]) == 42.0


# ---------------------------------------------------------------------------
# Group 5 — byte parity
# ---------------------------------------------------------------------------

def test_full_artifact_byte_parity():
    on_disk = ARTIFACT_PATH.read_text()
    rebuilt = json.dumps(REBUILT, indent=2, sort_keys=True)
    assert on_disk == rebuilt
