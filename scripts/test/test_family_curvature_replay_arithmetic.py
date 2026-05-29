"""Gate 128 — arithmetic + verdict audit of `family_curvature_replay.json`.

Independently reproduces every per-family curvature value, replay-pattern
flag, and meta verdict from the raw `oracle_gap.json` rows, asserting they
match the published artifact to a tight numeric tolerance.

The artifact computes a per-(family, app, graph, policy) discrete second
derivative of the oracle-gap trajectory across L3 sizes 1MB/4MB/8MB on a
log2-MB axis (1→0, 4→2, 8→3), then takes the mean across (app, graph) cells.
A family REPLAYS the global pattern iff at least one oracle-aware policy
(GRASP or POPT) has mean_curvature > 0 AND every non-oracle policy
(LRU, SRRIP) has mean_curvature <= 0.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
REPLAY_PATH = REPO_ROOT / "wiki" / "data" / "family_curvature_replay.json"

L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = frozenset({"GRASP", "POPT"})
NON_ORACLE = frozenset({"LRU", "SRRIP"})
CURVATURE_THRESHOLD = 0.0
TOL = 5e-4  # generator rounds to 4 dp


def _curvature(g1: float, g4: float, g8: float) -> float:
    slope_lo = (g4 - g1) / (L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])  # /2
    slope_hi = (g8 - g4) / (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"])  # /1
    span = 0.5 * ((L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])
                  + (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"]))  # 1.5
    return (slope_hi - slope_lo) / span


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(REPLAY_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_rows() -> list:
    return json.loads(ORACLE_PATH.read_text())["rows"]


@pytest.fixture(scope="module")
def expected(oracle_rows) -> dict:
    """Recompute the entire artifact from oracle_gap rows."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in oracle_rows:
        by[r["family"]][r["graph"]][r["app"]][r["policy"]][r["l3_size"]] = float(r["gap_pp"])

    per_family: dict[str, dict] = {}
    qualifying = []
    for fam, by_graph in sorted(by.items()):
        per_pol = defaultdict(list)
        any_full = False
        for _graph, by_app in by_graph.items():
            for _app, by_pol in by_app.items():
                if not all(pol in by_pol for pol in POLICIES):
                    continue
                if not all(all(l in by_pol[pol] for l in L3_SIZES) for pol in POLICIES):
                    continue
                any_full = True
                for pol in POLICIES:
                    per_pol[pol].append(_curvature(
                        by_pol[pol]["1MB"], by_pol[pol]["4MB"], by_pol[pol]["8MB"]
                    ))
        if not any_full:
            continue
        qualifying.append(fam)
        per_policy = {}
        for pol in POLICIES:
            xs = per_pol[pol]
            per_policy[pol] = {
                "mean_curvature_raw": sum(xs) / len(xs) if xs else 0.0,
                "n_app_graph_cells": len(xs),
                "is_oracle_aware": pol in ORACLE_AWARE,
            }
        oracle_pos = any(per_policy[p]["mean_curvature_raw"] > CURVATURE_THRESHOLD
                         for p in ORACLE_AWARE)
        non_oracle_nonpos = all(per_policy[p]["mean_curvature_raw"] <= CURVATURE_THRESHOLD
                                for p in NON_ORACLE)
        per_family[fam] = {
            "per_policy": per_policy,
            "replays_pattern": oracle_pos and non_oracle_nonpos,
            "any_oracle_aware_positive": oracle_pos,
            "all_non_oracle_nonpositive": non_oracle_nonpos,
        }
    return {"qualifying": qualifying, "per_family": per_family}


# ---------- Group 1: meta scope + sanity ----------

def test_meta_threshold_is_sign_test(artifact):
    assert artifact["meta"]["curvature_threshold_pp_oct2"] == CURVATURE_THRESHOLD


def test_meta_policies_full_set(artifact):
    assert sorted(artifact["meta"]["policies"]) == sorted(POLICIES)


def test_meta_oracle_aware_set(artifact):
    assert sorted(artifact["meta"]["oracle_aware_policies"]) == sorted(ORACLE_AWARE)


def test_meta_non_oracle_set(artifact):
    assert sorted(artifact["meta"]["non_oracle_policies"]) == sorted(NON_ORACLE)


# ---------- Group 2: qualifying families + per-family curvatures ----------

def test_qualifying_families_match(artifact, expected):
    assert artifact["meta"]["qualifying_families"] == expected["qualifying"]


def test_per_family_keys_match_qualifying(artifact):
    assert sorted(artifact["per_family"].keys()) == sorted(artifact["meta"]["qualifying_families"])


def test_per_family_mean_curvature_arithmetic(artifact, expected):
    for fam, exp in expected["per_family"].items():
        for pol in POLICIES:
            got = artifact["per_family"][fam]["per_policy"][pol]["mean_curvature"]
            want = round(exp["per_policy"][pol]["mean_curvature_raw"], 4)
            assert abs(got - want) < TOL, f"{fam}/{pol}: artifact={got} expected={want}"


def test_per_family_cell_counts_match(artifact, expected):
    for fam, exp in expected["per_family"].items():
        for pol in POLICIES:
            got = artifact["per_family"][fam]["per_policy"][pol]["n_app_graph_cells"]
            assert got == exp["per_policy"][pol]["n_app_graph_cells"], f"{fam}/{pol}"


def test_per_family_is_oracle_aware_flag(artifact):
    for fam, info in artifact["per_family"].items():
        for pol, p in info["per_policy"].items():
            assert p["is_oracle_aware"] == (pol in ORACLE_AWARE), f"{fam}/{pol}"


# ---------- Group 3: replay-pattern booleans ----------

def test_any_oracle_aware_positive_matches(artifact, expected):
    for fam, exp in expected["per_family"].items():
        assert artifact["per_family"][fam]["any_oracle_aware_positive"] == \
               exp["any_oracle_aware_positive"], fam


def test_all_non_oracle_nonpositive_matches(artifact, expected):
    for fam, exp in expected["per_family"].items():
        assert artifact["per_family"][fam]["all_non_oracle_nonpositive"] == \
               exp["all_non_oracle_nonpositive"], fam


def test_replays_pattern_matches(artifact, expected):
    for fam, exp in expected["per_family"].items():
        assert artifact["per_family"][fam]["replays_pattern"] == exp["replays_pattern"], fam


def test_replays_pattern_is_conjunction(artifact):
    for fam, info in artifact["per_family"].items():
        assert info["replays_pattern"] == (
            info["any_oracle_aware_positive"] and info["all_non_oracle_nonpositive"]
        ), fam


# ---------- Group 4: meta verdict logic ----------

def test_replay_count_matches(artifact):
    n = sum(1 for f in artifact["meta"]["qualifying_families"]
            if artifact["per_family"][f]["replays_pattern"])
    assert artifact["meta"]["replay_count"] == n


def test_deviating_families_complement(artifact):
    qual = artifact["meta"]["qualifying_families"]
    dev = [f for f in qual if not artifact["per_family"][f]["replays_pattern"]]
    assert artifact["meta"]["deviating_families"] == dev


def test_new_deviating_excludes_pinned(artifact):
    pinned = set(artifact["meta"]["pinned_deviating_families"])
    dev = artifact["meta"]["deviating_families"]
    expected_new = [f for f in dev if f not in pinned]
    assert artifact["meta"]["new_deviating_families"] == expected_new


def test_verdict_logic(artifact):
    m = artifact["meta"]
    expected = "PASS" if (m["replay_count"] >= 1 and not m["new_deviating_families"]) else "FAIL"
    assert m["verdict"] == expected
