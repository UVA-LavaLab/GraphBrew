"""Derivation parity gate for family_curvature_replay.json (gate 189).

Regenerates per-family discrete second-derivative (curvature) of the
oracle-gap trajectory directly from oracle_gap.json#rows and asserts
byte-equality with the committed artifact. Validates the per-family
replay of the global concavity signal that gate 58 establishes.

Load-bearing rules:

- L3_SIZES = ("1MB", "4MB", "8MB"); L3_LOG2_MB = {1MB:0, 4MB:2, 8MB:3}
- POLICIES = ("GRASP", "LRU", "POPT", "SRRIP") — alphabetical order
  (tuple ordering is load-bearing for emission)
- ORACLE_AWARE = {GRASP, POPT}; NON_ORACLE = {LRU, SRRIP}
- CURVATURE_THRESHOLD = 0.0 (sign test, strict > for oracle, <= for non)
- PINNED_DEVIATING_FAMILIES = () (empty tuple)
- Curvature formula on log2 axis:
    slope_lo = (g4-g1) / (2-0) = (g4-g1)/2
    slope_hi = (g8-g4) / (3-2) = (g8-g4)/1
    span = 0.5*((2-0)+(3-2)) = 1.5
    curvature = (slope_hi - slope_lo) / 1.5
- (Graph, app) qualifies iff ALL 4 policies present AND each policy has
  ALL 3 L3 sizes (NOT a subset check)
- Family qualifies iff at least one (graph, app) qualifies
- Iteration order: sorted(by.items()) — families alphabetical
- per_policy[P] contains EVERY policy in POLICIES (zero-fill on empty)
- mean_curvature rounded to 4 decimal places
- replays_pattern = oracle_pos AND non_oracle_nonpos:
    oracle_pos      = any(curv > 0 for p in ORACLE_AWARE)
    non_oracle_nonp = all(curv <= 0 for p in NON_ORACLE)
- deviating_families preserves qualifying order (NOT sorted again)
- new_dev = deviating - PINNED (preserves order)
- verdict = PASS iff replay_count >= 1 AND len(new_dev) == 0
- JSON written sort_keys=True
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "family_curvature_replay.json"
ORACLE = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = {"GRASP", "POPT"}
NON_ORACLE = {"LRU", "SRRIP"}
CURVATURE_THRESHOLD = 0.0
PINNED_DEVIATING_FAMILIES: tuple[str, ...] = ()


def _curvature(gaps):
    g1, g4, g8 = gaps[0], gaps[1], gaps[2]
    slope_lo = (g4 - g1) / (L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])
    slope_hi = (g8 - g4) / (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"])
    span = 0.5 * (
        (L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])
        + (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"])
    )
    return (slope_hi - slope_lo) / span


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT.exists(), f"missing {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oracle():
    assert ORACLE.exists(), f"missing {ORACLE}"
    return json.loads(ORACLE.read_text())


@pytest.fixture(scope="module")
def derived(oracle):
    rows = oracle["rows"]
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in rows:
        by[r["family"]][r["graph"]][r["app"]][r["policy"]][r["l3_size"]] = (
            float(r["gap_pp"])
        )

    per_family = {}
    qualifying = []
    for fam, by_graph in sorted(by.items()):
        per_pol = defaultdict(list)
        any_full = False
        for graph, by_app in by_graph.items():
            for app, by_pol in by_app.items():
                if not all(p in by_pol for p in POLICIES):
                    continue
                if not all(all(l in by_pol[p] for l in L3_SIZES) for p in POLICIES):
                    continue
                any_full = True
                for pol in POLICIES:
                    gaps = [by_pol[pol][l] for l in L3_SIZES]
                    per_pol[pol].append(_curvature(gaps))
        if not any_full:
            continue
        qualifying.append(fam)
        per_policy = {}
        for pol in POLICIES:
            xs = per_pol[pol]
            per_policy[pol] = {
                "mean_curvature":   round(sum(xs) / len(xs), 4) if xs else 0.0,
                "n_app_graph_cells": len(xs),
                "is_oracle_aware":  pol in ORACLE_AWARE,
            }
        oracle_pos = any(
            per_policy[p]["mean_curvature"] > CURVATURE_THRESHOLD for p in ORACLE_AWARE
        )
        non_oracle_nonpos = all(
            per_policy[p]["mean_curvature"] <= CURVATURE_THRESHOLD for p in NON_ORACLE
        )
        replays = oracle_pos and non_oracle_nonpos
        per_family[fam] = {
            "per_policy": per_policy,
            "replays_pattern": replays,
            "any_oracle_aware_positive": oracle_pos,
            "all_non_oracle_nonpositive": non_oracle_nonpos,
        }

    deviating = [f for f in qualifying if not per_family[f]["replays_pattern"]]
    new_dev = [f for f in deviating if f not in PINNED_DEVIATING_FAMILIES]
    replay_count = sum(1 for f in qualifying if per_family[f]["replays_pattern"])
    verdict = "PASS" if (replay_count >= 1 and not new_dev) else "FAIL"

    return {
        "meta": {
            "qualifying_families": qualifying,
            "curvature_threshold_pp_oct2": CURVATURE_THRESHOLD,
            "replay_count": replay_count,
            "deviating_families": deviating,
            "pinned_deviating_families": list(PINNED_DEVIATING_FAMILIES),
            "new_deviating_families": new_dev,
            "policies": list(POLICIES),
            "oracle_aware_policies": sorted(ORACLE_AWARE),
            "non_oracle_policies": sorted(NON_ORACLE),
            "verdict": verdict,
        },
        "per_family": per_family,
    }


# ---------------------------------------------------------------------------
# Group A — meta constants & verdict
# ---------------------------------------------------------------------------


def test_meta_curvature_threshold(artifact):
    assert artifact["meta"]["curvature_threshold_pp_oct2"] == CURVATURE_THRESHOLD


def test_meta_policies_order(artifact):
    """POLICIES order is alphabetical and load-bearing for emission."""
    assert artifact["meta"]["policies"] == list(POLICIES)


def test_meta_oracle_aware_sorted(artifact):
    assert artifact["meta"]["oracle_aware_policies"] == sorted(ORACLE_AWARE)


def test_meta_non_oracle_sorted(artifact):
    assert artifact["meta"]["non_oracle_policies"] == sorted(NON_ORACLE)


def test_meta_qualifying_families_alpha_sorted(artifact):
    qf = artifact["meta"]["qualifying_families"]
    assert qf == sorted(qf)


def test_meta_qualifying_families_match_derived(artifact, derived):
    assert artifact["meta"]["qualifying_families"] == derived["meta"]["qualifying_families"]


def test_meta_replay_count_matches(artifact, derived):
    assert artifact["meta"]["replay_count"] == derived["meta"]["replay_count"]


def test_meta_deviating_matches(artifact, derived):
    assert artifact["meta"]["deviating_families"] == derived["meta"]["deviating_families"]


def test_meta_pinned_matches(artifact):
    assert artifact["meta"]["pinned_deviating_families"] == list(PINNED_DEVIATING_FAMILIES)


def test_meta_new_deviating_subset(artifact):
    """new_deviating ⊆ deviating; preserves order."""
    new = artifact["meta"]["new_deviating_families"]
    dev = artifact["meta"]["deviating_families"]
    pinned = set(artifact["meta"]["pinned_deviating_families"])
    assert new == [f for f in dev if f not in pinned]


def test_meta_verdict_closed_form(artifact):
    m = artifact["meta"]
    expected = "PASS" if (m["replay_count"] >= 1 and not m["new_deviating_families"]) else "FAIL"
    assert m["verdict"] == expected


def test_meta_replay_count_consistency(artifact):
    qf = artifact["meta"]["qualifying_families"]
    dev = artifact["meta"]["deviating_families"]
    assert artifact["meta"]["replay_count"] == len(qf) - len(dev)


# ---------------------------------------------------------------------------
# Group B — per_family shape
# ---------------------------------------------------------------------------


def test_per_family_keys_match_qualifying(artifact):
    assert set(artifact["per_family"].keys()) == set(artifact["meta"]["qualifying_families"])


def test_per_family_record_shape(artifact):
    for fam, rec in artifact["per_family"].items():
        assert set(rec.keys()) == {
            "per_policy",
            "replays_pattern",
            "any_oracle_aware_positive",
            "all_non_oracle_nonpositive",
        }


def test_per_family_per_policy_has_all_policies(artifact):
    for fam, rec in artifact["per_family"].items():
        assert set(rec["per_policy"].keys()) == set(POLICIES)


def test_per_family_per_policy_record_shape(artifact):
    for fam, rec in artifact["per_family"].items():
        for pol, pp in rec["per_policy"].items():
            assert set(pp.keys()) == {
                "mean_curvature",
                "n_app_graph_cells",
                "is_oracle_aware",
            }


def test_per_family_is_oracle_aware_flag(artifact):
    for fam, rec in artifact["per_family"].items():
        for pol, pp in rec["per_policy"].items():
            assert pp["is_oracle_aware"] == (pol in ORACLE_AWARE)


def test_per_family_mean_curvature_4dp(artifact):
    for fam, rec in artifact["per_family"].items():
        for pol, pp in rec["per_policy"].items():
            mc = pp["mean_curvature"]
            assert abs(mc - round(mc, 4)) <= 1e-9


def test_per_family_n_cells_consistent_across_policies(artifact):
    """All 4 policies must have identical n_app_graph_cells (gating is on
    (graph, app), not per-policy)."""
    for fam, rec in artifact["per_family"].items():
        ns = {pp["n_app_graph_cells"] for pp in rec["per_policy"].values()}
        assert len(ns) == 1, f"{fam} policies disagree: {ns}"


# ---------------------------------------------------------------------------
# Group C — curvature math
# ---------------------------------------------------------------------------


def test_curvature_formula_constants():
    """span = 0.5*(2+1) = 1.5; slope_lo divisor = 2; slope_hi divisor = 1."""
    assert L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"] == 2.0
    assert L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"] == 1.0
    span = 0.5 * ((L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"]) + (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"]))
    assert span == 1.5


def test_curvature_known_value():
    """Hand-computed: gaps (10, 6, 5) → slope_lo=-2, slope_hi=-1, curv=2/3 ≈ 0.6667."""
    assert abs(_curvature([10.0, 6.0, 5.0]) - (2.0 / 3.0)) < 1e-12


def test_curvature_zero_for_linear():
    """Linear in log2 axis → curvature == 0."""
    # gap = a + b*log2(MB): values at log2 [0,2,3] = [a, a+2b, a+3b]
    a, b = 5.0, -1.5
    gaps = [a, a + 2 * b, a + 3 * b]
    assert abs(_curvature(gaps)) < 1e-12


def test_curvature_negative_for_accelerating_descent():
    """Trajectory that drops faster at the high end → curvature < 0."""
    # gap_1 > gap_4 > gap_8 with bigger drop on high octave
    gaps = [10.0, 8.0, 4.0]  # slope_lo=-1, slope_hi=-4 → curv=-2 < 0
    assert _curvature(gaps) < 0


def test_curvature_positive_for_decelerating_descent():
    """Bend toward plateau → curvature > 0."""
    gaps = [10.0, 4.0, 3.0]  # slope_lo=-3, slope_hi=-1 → curv≈1.33 > 0
    assert _curvature(gaps) > 0


# ---------------------------------------------------------------------------
# Group D — replays_pattern logic
# ---------------------------------------------------------------------------


def test_replays_pattern_is_conjunction(artifact):
    for fam, rec in artifact["per_family"].items():
        expected = rec["any_oracle_aware_positive"] and rec["all_non_oracle_nonpositive"]
        assert rec["replays_pattern"] == expected


def test_any_oracle_aware_positive_logic(artifact):
    for fam, rec in artifact["per_family"].items():
        expected = any(
            rec["per_policy"][p]["mean_curvature"] > CURVATURE_THRESHOLD
            for p in ORACLE_AWARE
        )
        assert rec["any_oracle_aware_positive"] == expected


def test_all_non_oracle_nonpositive_logic(artifact):
    for fam, rec in artifact["per_family"].items():
        expected = all(
            rec["per_policy"][p]["mean_curvature"] <= CURVATURE_THRESHOLD
            for p in NON_ORACLE
        )
        assert rec["all_non_oracle_nonpositive"] == expected


def test_replay_count_matches_per_family_flags(artifact):
    n = sum(1 for f in artifact["meta"]["qualifying_families"]
            if artifact["per_family"][f]["replays_pattern"])
    assert n == artifact["meta"]["replay_count"]


# ---------------------------------------------------------------------------
# Group E — full byte parity + qualification rule
# ---------------------------------------------------------------------------


def test_qualification_requires_full_4policy_3l3_cells(oracle, artifact):
    """For each qualifying family, there must exist at least one (graph, app)
    cell with all 4 policies × all 3 L3 sizes in the oracle."""
    family_full_cells = defaultdict(int)
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for r in oracle["rows"]:
        by[r["family"]][(r["graph"], r["app"])][r["policy"]].add(r["l3_size"])
    for fam, cells in by.items():
        for (g, a), pols in cells.items():
            if set(pols.keys()) >= set(POLICIES) and all(
                set(L3_SIZES).issubset(pols[p]) for p in POLICIES
            ):
                family_full_cells[fam] += 1

    qualifying = set(artifact["meta"]["qualifying_families"])
    expected = {f for f, n in family_full_cells.items() if n >= 1}
    assert qualifying == expected


def test_full_artifact_byte_parity(artifact, derived):
    """End-to-end: artifact byte-equal to derived (excluding verdict_invariant string)."""
    a = dict(artifact)
    a_meta = dict(a["meta"])
    a_meta.pop("verdict_invariant", None)
    a["meta"] = a_meta
    d = dict(derived)
    d_meta = dict(d["meta"])
    d_meta.pop("verdict_invariant", None)
    d["meta"] = d_meta
    assert a == d
