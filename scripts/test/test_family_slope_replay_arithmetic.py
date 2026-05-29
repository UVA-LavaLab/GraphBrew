"""Gate 121 — family_slope_replay.json arithmetic + verdict.

Locks the per-family slope-replay analysis built from oracle_gap.json
rows. For each graph-family that has at least one (graph, app) cell
with all 4 policies populated at every L3 size in {1MB, 4MB, 8MB},
the artifact computes per-policy OLS slopes of miss_pp vs log2(L3_MB)
and aggregates them into per-family per-policy summary stats. A
family 'replays the pattern' iff:

    (1) LRU median slope is strictly steeper (more negative) than GRASP;
    (2) SRRIP median slope is strictly steeper than GRASP;
    (3) every policy median is strictly < HELP_FLOOR_PP_OCTAVE = -5.0.

The verdict is PASS iff at least one family replays AND no NEW
deviating family appears beyond PINNED_DEVIATING_FAMILIES=('social',).

This gate locks the per-family OLS arithmetic and the four-step
verdict logic so any regression in oracle_gap source data or the
replay computation surfaces immediately.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/family_slope_replay.json")
SOURCE = Path("wiki/data/oracle_gap.json")

L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = ("GRASP", "POPT")
NON_ORACLE = ("LRU", "SRRIP")
HELP_FLOOR = -5.0
PINNED_DEVIATING = ("social",)

SLOPE_TOL = 1e-3


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def source():
    assert SOURCE.exists(), f"missing source artifact: {SOURCE}"
    return json.loads(SOURCE.read_text())


def _ols_slope(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def _compute_per_pol_slopes(source: dict):
    by = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for r in source["rows"]:
        if r["l3_size"] not in L3_LOG2_MB:
            continue
        by[r["family"]][r["graph"]][r["app"]][r["policy"]][r["l3_size"]] = (
            float(r["miss_rate"]) * 100.0
        )

    per_family: dict[str, dict[str, list[float]]] = {}
    for fam, by_graph in by.items():
        per_pol: dict[str, list[float]] = defaultdict(list)
        any_full = False
        for graph, by_app in by_graph.items():
            for app, by_pol in by_app.items():
                if not all(p in by_pol for p in POLICIES):
                    continue
                if not all(all(l in by_pol[p] for l in L3_SIZES) for p in POLICIES):
                    continue
                any_full = True
                for pol in POLICIES:
                    pts = [(L3_LOG2_MB[l], by_pol[pol][l]) for l in L3_SIZES]
                    per_pol[pol].append(_ols_slope(pts))
        if any_full:
            per_family[fam] = per_pol
    return per_family


# ── group 1: meta constants ──────────────────────────────────────────────


def test_meta_constants(data):
    meta = data["meta"]
    assert meta["l3_axis"] == list(L3_SIZES)
    assert meta["policies"] == list(POLICIES)
    assert meta["oracle_aware_policies"] == list(ORACLE_AWARE)
    assert meta["non_oracle_policies"] == list(NON_ORACLE)
    assert meta["help_floor_pp_octave"] == HELP_FLOOR
    assert meta["pinned_deviating_families"] == list(PINNED_DEVIATING)
    assert meta["verdict"] in ("PASS", "FAIL")


def test_qualifying_families_match_source(data, source):
    expected = sorted(_compute_per_pol_slopes(source).keys())
    assert data["meta"]["qualifying_families"] == expected


# ── group 2: per_family.per_policy stats from OLS slopes ─────────────────


def test_per_family_per_policy_stats(data, source):
    recomputed = _compute_per_pol_slopes(source)
    for fam in data["meta"]["qualifying_families"]:
        for pol in POLICIES:
            vals = recomputed[fam].get(pol, [])
            entry = data["per_family"][fam]["per_policy"].get(pol)
            if not vals:
                assert entry is None or entry == {}, (
                    f"{fam}/{pol}: empty source but non-empty entry"
                )
                continue
            assert entry is not None, f"{fam}/{pol}: missing entry"
            assert entry["n_cells"] == len(vals), f"{fam}/{pol}: n_cells"
            assert math.isclose(
                entry["median_pp"], statistics.median(vals), abs_tol=SLOPE_TOL
            ), f"{fam}/{pol}: median_pp mismatch"
            assert math.isclose(
                entry["mean_pp"], sum(vals) / len(vals), abs_tol=SLOPE_TOL
            ), f"{fam}/{pol}: mean_pp mismatch"
            assert math.isclose(
                entry["min_pp"], min(vals), abs_tol=SLOPE_TOL
            ), f"{fam}/{pol}: min_pp mismatch"
            assert math.isclose(
                entry["max_pp"], max(vals), abs_tol=SLOPE_TOL
            ), f"{fam}/{pol}: max_pp mismatch"


def test_is_oracle_aware_flag(data):
    for fam, blob in data["per_family"].items():
        for pol, entry in blob["per_policy"].items():
            expected = pol in ORACLE_AWARE
            assert entry["is_oracle_aware"] is expected, (
                f"{fam}/{pol}: is_oracle_aware={entry['is_oracle_aware']} expected={expected}"
            )


# ── group 3: replay invariants per family ────────────────────────────────


def test_lru_steeper_than_grasp(data):
    for fam, blob in data["per_family"].items():
        pp = blob["per_policy"]
        if "LRU" in pp and "GRASP" in pp:
            expected = pp["LRU"]["median_pp"] < pp["GRASP"]["median_pp"]
        else:
            expected = False
        assert blob["lru_steeper_than_grasp"] is expected, (
            f"{fam}: lru_steeper_than_grasp mismatch"
        )


def test_srrip_steeper_than_grasp(data):
    for fam, blob in data["per_family"].items():
        pp = blob["per_policy"]
        if "SRRIP" in pp and "GRASP" in pp:
            expected = pp["SRRIP"]["median_pp"] < pp["GRASP"]["median_pp"]
        else:
            expected = False
        assert blob["srrip_steeper_than_grasp"] is expected, (
            f"{fam}: srrip_steeper_than_grasp mismatch"
        )


def test_all_policies_helped(data):
    for fam, blob in data["per_family"].items():
        expected = all(
            entry["median_pp"] < HELP_FLOOR for entry in blob["per_policy"].values()
        )
        assert blob["all_policies_helped"] is expected, (
            f"{fam}: all_policies_helped mismatch"
        )


def test_replays_pattern_is_conjunction(data):
    for fam, blob in data["per_family"].items():
        expected = (
            blob["lru_steeper_than_grasp"]
            and blob["srrip_steeper_than_grasp"]
            and blob["all_policies_helped"]
        )
        assert blob["replays_pattern"] is expected, (
            f"{fam}: replays_pattern={blob['replays_pattern']} expected={expected}"
        )


# ── group 4: meta aggregates + verdict ───────────────────────────────────


def test_replay_count_matches(data):
    expected = sum(
        1
        for fam in data["meta"]["qualifying_families"]
        if data["per_family"][fam]["replays_pattern"]
    )
    assert data["meta"]["replay_count"] == expected


def test_deviating_families_partition(data):
    quals = data["meta"]["qualifying_families"]
    expected_dev = [f for f in quals if not data["per_family"][f]["replays_pattern"]]
    assert data["meta"]["deviating_families"] == expected_dev
    expected_new = [f for f in expected_dev if f not in PINNED_DEVIATING]
    assert data["meta"]["new_deviating_families"] == expected_new


def test_verdict_invariant(data):
    meta = data["meta"]
    expected = (
        "PASS"
        if (meta["replay_count"] >= 1 and not meta["new_deviating_families"])
        else "FAIL"
    )
    assert meta["verdict"] == expected
