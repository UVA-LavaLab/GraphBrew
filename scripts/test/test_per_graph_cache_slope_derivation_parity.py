"""Derivation parity gate for per_graph_cache_slope.json (gate 186).

Regenerates per-(graph, app, policy) per-octave slopes and anti-scaling counters
directly from oracle_gap.json#rows and asserts byte-equality against the
committed artifact.

Load-bearing rules (must hold for the paper anti-scaling claim):

- PAPER_L3 = ("1MB", "4MB", "8MB"); L3_MB = {"1MB":1.0, "4MB":4.0, "8MB":8.0}
- SIGNIFICANT_PP = 1.0 with INCLUSIVE threshold (d_gap >= 1.0)
- Only (graph, app, policy) cells with ALL THREE L3 sizes present participate
  (strict set equality, not subset)
- Two octaves: (1MB→4MB), (4MB→8MB); slope_pp_per_octave = -d_gap / d_log
- families dict overwrites on later rows (last-write-wins)
- per_graph_policy_anti_scaling_count keys = "graph|policy" string
- anti_scaling_cells sorted by max_pp_growth DESC
- n_oracle_aware_anti_scaling = GRASP + POPT only (LRU + SRRIP excluded)
- All emitted floats rounded to 4 decimal places
- meta.policies / apps / graphs sorted ascii-lexicographically over
  full_trajectories key set (NOT all rows)
- per_policy_counts and per_graph_counts only carry policies/graphs with ≥1
  flagged cell (no zero entries)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "per_graph_cache_slope.json"
ORACLE = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

PAPER_L3 = ("1MB", "4MB", "8MB")
L3_MB = {"1MB": 1.0, "4MB": 4.0, "8MB": 8.0}
SIGNIFICANT_PP = 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oracle() -> dict:
    assert ORACLE.exists(), f"missing {ORACLE}"
    return json.loads(ORACLE.read_text())


@pytest.fixture(scope="module")
def derived(oracle) -> dict:
    """Independent reproduction of build_payload() from generator source."""
    grid: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    families: dict[str, str] = {}
    for row in oracle["rows"]:
        if row["l3_size"] not in PAPER_L3:
            continue
        key = (row["graph"], row["app"], row["policy"])
        grid[key][row["l3_size"]] = float(row["gap_pp"])
        families[row["graph"]] = row["family"]

    full = {k: v for k, v in grid.items() if set(v.keys()) == set(PAPER_L3)}

    anti_cells: list[dict] = []
    per_policy: dict[str, int] = defaultdict(int)
    per_graph: dict[str, int] = defaultdict(int)
    per_gp: dict[tuple[str, str], int] = defaultdict(int)
    cell_records: list[dict] = []

    for (graph, app, policy), traj in full.items():
        octaves = []
        for src, dst in (("1MB", "4MB"), ("4MB", "8MB")):
            d_log = math.log2(L3_MB[dst]) - math.log2(L3_MB[src])
            d_gap = traj[dst] - traj[src]
            slope = -d_gap / d_log if d_log > 0 else 0.0
            octaves.append({
                "from": src,
                "to": dst,
                "gap_from": round(traj[src], 4),
                "gap_to": round(traj[dst], 4),
                "delta_gap_pp": round(d_gap, 4),
                "slope_pp_per_octave": round(slope, 4),
                "significant_anti_scaling": d_gap >= SIGNIFICANT_PP,
            })
        any_anti = any(o["significant_anti_scaling"] for o in octaves)
        if any_anti:
            per_policy[policy] += 1
            per_graph[graph] += 1
            per_gp[(graph, policy)] += 1
            anti_cells.append({
                "graph": graph,
                "family": families.get(graph, "unknown"),
                "app": app,
                "policy": policy,
                "octaves": octaves,
                "max_pp_growth": round(
                    max(o["delta_gap_pp"] for o in octaves), 4
                ),
            })
        cell_records.append({
            "graph": graph,
            "family": families.get(graph, "unknown"),
            "app": app,
            "policy": policy,
            "octaves": octaves,
            "any_significant_anti_scaling": any_anti,
            "total_shrinkage_pp": round(traj["1MB"] - traj["8MB"], 4),
        })

    anti_cells.sort(key=lambda d: d["max_pp_growth"], reverse=True)

    policies = sorted({p for (_, _, p) in full.keys()})
    apps = sorted({a for (_, a, _) in full.keys()})
    graphs = sorted({g for (g, _, _) in full.keys()})
    g_anti = per_policy.get("GRASP", 0)
    p_anti = per_policy.get("POPT", 0)

    return {
        "meta": {
            "scope_l3_sizes": list(PAPER_L3),
            "n_apps": len(apps),
            "n_policies": len(policies),
            "n_graphs_with_full_trajectory": len(graphs),
            "apps": apps,
            "policies": policies,
            "graphs": graphs,
            "n_full_trajectories": len(full),
            "significant_pp_threshold": SIGNIFICANT_PP,
            "n_cells_with_significant_anti_scaling": len(anti_cells),
            "n_oracle_aware_anti_scaling": g_anti + p_anti,
        },
        "per_policy_anti_scaling_count": dict(per_policy),
        "per_graph_anti_scaling_count": dict(per_graph),
        "per_graph_policy_anti_scaling_count": {
            f"{g}|{p}": n for (g, p), n in per_gp.items()
        },
        "anti_scaling_cells": anti_cells,
        "all_trajectories": cell_records,
    }


# ---------------------------------------------------------------------------
# Group A — meta scope + counts
# ---------------------------------------------------------------------------


def test_meta_scope_l3_sizes(artifact):
    assert artifact["meta"]["scope_l3_sizes"] == list(PAPER_L3)


def test_meta_significant_pp_threshold(artifact):
    assert artifact["meta"]["significant_pp_threshold"] == SIGNIFICANT_PP


def test_meta_n_full_trajectories(artifact, derived):
    assert artifact["meta"]["n_full_trajectories"] == derived["meta"]["n_full_trajectories"]


def test_meta_n_cells_with_significant_anti_scaling(artifact, derived):
    assert (
        artifact["meta"]["n_cells_with_significant_anti_scaling"]
        == derived["meta"]["n_cells_with_significant_anti_scaling"]
    )


def test_meta_n_oracle_aware_anti_scaling(artifact, derived):
    assert (
        artifact["meta"]["n_oracle_aware_anti_scaling"]
        == derived["meta"]["n_oracle_aware_anti_scaling"]
    )


def test_meta_apps_sorted(artifact, derived):
    assert artifact["meta"]["apps"] == derived["meta"]["apps"]
    assert artifact["meta"]["apps"] == sorted(artifact["meta"]["apps"])


def test_meta_policies_sorted(artifact, derived):
    assert artifact["meta"]["policies"] == derived["meta"]["policies"]
    assert artifact["meta"]["policies"] == sorted(artifact["meta"]["policies"])


def test_meta_graphs_sorted(artifact, derived):
    assert artifact["meta"]["graphs"] == derived["meta"]["graphs"]
    assert artifact["meta"]["graphs"] == sorted(artifact["meta"]["graphs"])


def test_meta_n_counts_match_list_lens(artifact):
    m = artifact["meta"]
    assert m["n_apps"] == len(m["apps"])
    assert m["n_policies"] == len(m["policies"])
    assert m["n_graphs_with_full_trajectory"] == len(m["graphs"])


# ---------------------------------------------------------------------------
# Group B — per-policy / per-graph counter parity
# ---------------------------------------------------------------------------


def test_per_policy_anti_scaling_count_matches(artifact, derived):
    assert (
        artifact["per_policy_anti_scaling_count"]
        == derived["per_policy_anti_scaling_count"]
    )


def test_per_graph_anti_scaling_count_matches(artifact, derived):
    assert (
        artifact["per_graph_anti_scaling_count"]
        == derived["per_graph_anti_scaling_count"]
    )


def test_per_graph_policy_anti_scaling_count_matches(artifact, derived):
    assert (
        artifact["per_graph_policy_anti_scaling_count"]
        == derived["per_graph_policy_anti_scaling_count"]
    )


def test_per_policy_count_has_no_zero_entries(artifact):
    for pol, n in artifact["per_policy_anti_scaling_count"].items():
        assert n >= 1, f"{pol} entry should not be present with 0 count"


def test_per_graph_count_has_no_zero_entries(artifact):
    for g, n in artifact["per_graph_anti_scaling_count"].items():
        assert n >= 1, f"{g} entry should not be present with 0 count"


def test_per_graph_policy_key_format(artifact):
    for key in artifact["per_graph_policy_anti_scaling_count"].keys():
        assert "|" in key
        g, p = key.split("|", 1)
        assert g in artifact["meta"]["graphs"]
        assert p in artifact["meta"]["policies"]


def test_per_policy_sum_equals_total_anti_cells(artifact):
    total = sum(artifact["per_policy_anti_scaling_count"].values())
    assert total == artifact["meta"]["n_cells_with_significant_anti_scaling"]


def test_oracle_aware_count_is_grasp_plus_popt(artifact):
    counts = artifact["per_policy_anti_scaling_count"]
    expected = counts.get("GRASP", 0) + counts.get("POPT", 0)
    assert artifact["meta"]["n_oracle_aware_anti_scaling"] == expected


# ---------------------------------------------------------------------------
# Group C — octave math (closed form)
# ---------------------------------------------------------------------------


def test_octave_slope_closed_form(artifact):
    """slope_pp_per_octave = -(gap_to - gap_from) / log2(L3_to / L3_from)."""
    for cell in artifact["all_trajectories"]:
        for octv in cell["octaves"]:
            d_log = math.log2(L3_MB[octv["to"]]) - math.log2(L3_MB[octv["from"]])
            d_gap = octv["gap_to"] - octv["gap_from"]
            expected = round(-d_gap / d_log, 4)
            # gap_from / gap_to are themselves rounded to 4dp, so allow ±5e-5
            assert abs(octv["slope_pp_per_octave"] - expected) <= 5e-5


def test_octave_delta_gap_closed_form(artifact):
    for cell in artifact["all_trajectories"]:
        for octv in cell["octaves"]:
            expected = round(octv["gap_to"] - octv["gap_from"], 4)
            assert abs(octv["delta_gap_pp"] - expected) <= 5e-5


def test_octave_significant_threshold_inclusive(artifact):
    """SIGNIFICANT_PP threshold is >= (inclusive), NOT > (strict)."""
    for cell in artifact["all_trajectories"]:
        for octv in cell["octaves"]:
            expected = octv["delta_gap_pp"] >= SIGNIFICANT_PP
            assert octv["significant_anti_scaling"] == expected


def test_octave_pair_ordering(artifact):
    """Every cell has exactly two octaves: 1MB→4MB then 4MB→8MB."""
    for cell in artifact["all_trajectories"]:
        assert len(cell["octaves"]) == 2
        assert (cell["octaves"][0]["from"], cell["octaves"][0]["to"]) == ("1MB", "4MB")
        assert (cell["octaves"][1]["from"], cell["octaves"][1]["to"]) == ("4MB", "8MB")


def test_octave_floats_4dp(artifact):
    for cell in artifact["all_trajectories"]:
        for octv in cell["octaves"]:
            for k in ("gap_from", "gap_to", "delta_gap_pp", "slope_pp_per_octave"):
                v = octv[k]
                assert abs(v - round(v, 4)) <= 1e-9


# ---------------------------------------------------------------------------
# Group D — all_trajectories shape + total_shrinkage
# ---------------------------------------------------------------------------


def test_all_trajectories_count_matches_full(artifact, derived):
    assert len(artifact["all_trajectories"]) == derived["meta"]["n_full_trajectories"]


def test_all_trajectories_unique_keys(artifact):
    keys = [(c["graph"], c["app"], c["policy"]) for c in artifact["all_trajectories"]]
    assert len(keys) == len(set(keys))


def test_all_trajectories_total_shrinkage_closed_form(artifact):
    for cell in artifact["all_trajectories"]:
        gap_1mb = cell["octaves"][0]["gap_from"]
        gap_8mb = cell["octaves"][1]["gap_to"]
        expected = round(gap_1mb - gap_8mb, 4)
        assert abs(cell["total_shrinkage_pp"] - expected) <= 5e-5


def test_all_trajectories_any_anti_consistency(artifact):
    for cell in artifact["all_trajectories"]:
        expected = any(o["significant_anti_scaling"] for o in cell["octaves"])
        assert cell["any_significant_anti_scaling"] == expected


def test_all_trajectories_family_is_str(artifact):
    for cell in artifact["all_trajectories"]:
        assert isinstance(cell["family"], str)
        assert cell["family"]


# ---------------------------------------------------------------------------
# Group E — anti_scaling_cells subset + sort + max_pp_growth
# ---------------------------------------------------------------------------


def test_anti_scaling_cells_count_matches_meta(artifact):
    assert (
        len(artifact["anti_scaling_cells"])
        == artifact["meta"]["n_cells_with_significant_anti_scaling"]
    )


def test_anti_scaling_cells_subset_of_all_trajectories(artifact):
    flagged_all = {
        (c["graph"], c["app"], c["policy"])
        for c in artifact["all_trajectories"]
        if c["any_significant_anti_scaling"]
    }
    anti_keys = {
        (c["graph"], c["app"], c["policy"])
        for c in artifact["anti_scaling_cells"]
    }
    assert anti_keys == flagged_all


def test_anti_scaling_cells_sorted_desc_by_growth(artifact):
    growths = [c["max_pp_growth"] for c in artifact["anti_scaling_cells"]]
    assert growths == sorted(growths, reverse=True)


def test_anti_scaling_cells_max_pp_growth_formula(artifact):
    for cell in artifact["anti_scaling_cells"]:
        expected = round(max(o["delta_gap_pp"] for o in cell["octaves"]), 4)
        assert abs(cell["max_pp_growth"] - expected) <= 5e-5


def test_anti_scaling_cells_each_meets_threshold(artifact):
    """Every entry in anti_scaling_cells has at least one octave at/above 1.0pp."""
    for cell in artifact["anti_scaling_cells"]:
        assert any(o["delta_gap_pp"] >= SIGNIFICANT_PP for o in cell["octaves"])


def test_anti_scaling_cells_derived_parity(artifact, derived):
    """The full anti_scaling_cells list (including sort order) matches derivation."""
    assert artifact["anti_scaling_cells"] == derived["anti_scaling_cells"]


def test_full_artifact_byte_parity(artifact, derived):
    """End-to-end: artifact == derived (excluding source path in meta)."""
    a = dict(artifact)
    a_meta = dict(a["meta"])
    a_meta.pop("source", None)
    a["meta"] = a_meta
    assert a == derived
