"""Derivation parity gate for ``wiki/data/oracle_gap_curvature.json``.

Locks the discrete second-derivative ("knee") detector against its
single upstream — ``oracle_gap_auc.json#per_app[app].trajectory_by_policy``
— so any drift in the log2-octave slope arithmetic
(slope_01 = (g4 − g1) / 2, slope_12 = (g8 − g4) / 1), the curvature
combiner (s12 − s01), the knee threshold (≥ 0.05 pp/oct²,
non-strict), the cells_total gate (must have ALL three L3 sizes),
the per-policy {n_cells, knee_count, mean_curvature, median_curvature}
reducers, the saturation-rank ordering (sort by −knee_count then
−mean_curvature), the cross-gate-55 lead_agrees flag, or the verdict
("min(knee_count) over oracle-aware > max(knee_count) over non-oracle")
trips a test before the dashboard re-publishes the
"oracle-aware policies plateau earlier than LRU/SRRIP" knee claim.

Mirrors `build_payload()` from
`scripts/experiments/ecg/oracle_gap_curvature.py` verbatim.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

UPSTREAM_PATH = WIKI_DATA / "oracle_gap_auc.json"
ARTIFACT_PATH = WIKI_DATA / "oracle_gap_curvature.json"
GATE55_PATH = WIKI_DATA / "cache_saturation_onset.json"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
L3_LOG2_MB = {"1MB": 0.0, "4MB": 2.0, "8MB": 3.0}
KNEE_THRESHOLD = 0.05
ORACLE_AWARE = ("GRASP", "POPT")
NON_ORACLE = ("LRU", "SRRIP")


def _curvature(g1, g4, g8):
    s01 = (g4 - g1) / (L3_LOG2_MB["4MB"] - L3_LOG2_MB["1MB"])
    s12 = (g8 - g4) / (L3_LOG2_MB["8MB"] - L3_LOG2_MB["4MB"])
    return s01, s12, s12 - s01


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def upstream():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    return json.loads(UPSTREAM_PATH.read_text())


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_policy_summary", "per_app"}


def test_meta_fields(artifact):
    expected = {
        "source", "scope_l3_sizes", "x_axis",
        "knee_curvature_threshold_pp_per_oct2",
        "cells_total", "cells_with_knee", "knee_rank_by_policy",
        "cross_gate_consistency", "knee_lead_verdict",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing: {missing}"


def test_meta_scope_l3_sizes_match_constant(artifact):
    assert tuple(artifact["meta"]["scope_l3_sizes"]) == PAPER_L3_SIZES


def test_meta_threshold_matches_constant(artifact):
    assert artifact["meta"]["knee_curvature_threshold_pp_per_oct2"] == KNEE_THRESHOLD


def test_meta_x_axis_is_log2_octave(artifact):
    assert artifact["meta"]["x_axis"] == "log2(L3 / 1MB)"


def test_per_app_keys_match_upstream_sorted(artifact, upstream):
    """Generator does `apps = sorted(meta['apps'])`."""
    assert sorted(artifact["per_app"].keys()) == sorted(upstream["meta"]["apps"])


def test_per_policy_summary_keys_match_upstream_sorted(artifact, upstream):
    """Generator does `policies = sorted(meta['policies'])`."""
    assert (
        sorted(artifact["per_policy_summary"].keys())
        == sorted(upstream["meta"]["policies"])
    )


def test_per_cell_fields(artifact):
    expected = {
        "gap_at_1MB", "gap_at_4MB", "gap_at_8MB",
        "slope_1MB_to_4MB", "slope_4MB_to_8MB",
        "curvature_at_4MB", "knee_present",
    }
    for app, pols in artifact["per_app"].items():
        for pol, cell in pols.items():
            assert set(cell.keys()) == expected, (
                f"{app}/{pol}: cell field drift"
            )


def test_per_policy_summary_fields(artifact):
    expected = {"n_cells", "knee_count", "mean_curvature", "median_curvature"}
    for pol, s in artifact["per_policy_summary"].items():
        assert set(s.keys()) == expected


# ----------------------------------------------------------------------
# Group B: per-cell arithmetic parity (mirror _curvature)
# ----------------------------------------------------------------------

def test_per_cell_gaps_match_upstream_trajectory(artifact, upstream):
    for app, pols in artifact["per_app"].items():
        traj = upstream["per_app"][app]["trajectory_by_policy"]
        for pol, cell in pols.items():
            t = traj[pol]
            assert cell["gap_at_1MB"] == round(t["1MB"], 4)
            assert cell["gap_at_4MB"] == round(t["4MB"], 4)
            assert cell["gap_at_8MB"] == round(t["8MB"], 4)


def test_per_cell_slopes_match_log2_arithmetic(artifact, upstream):
    """slope_01 divides by (2 − 0)=2 octaves; slope_12 divides by 1."""
    for app, pols in artifact["per_app"].items():
        traj = upstream["per_app"][app]["trajectory_by_policy"]
        for pol, cell in pols.items():
            s01, s12, _ = _curvature(
                traj[pol]["1MB"], traj[pol]["4MB"], traj[pol]["8MB"],
            )
            assert cell["slope_1MB_to_4MB"] == round(s01, 4)
            assert cell["slope_4MB_to_8MB"] == round(s12, 4)


def test_per_cell_curvature_matches_s12_minus_s01(artifact, upstream):
    for app, pols in artifact["per_app"].items():
        traj = upstream["per_app"][app]["trajectory_by_policy"]
        for pol, cell in pols.items():
            _, _, curv = _curvature(
                traj[pol]["1MB"], traj[pol]["4MB"], traj[pol]["8MB"],
            )
            assert cell["curvature_at_4MB"] == round(curv, 4)


def test_per_cell_knee_present_uses_non_strict_ge(artifact):
    """Generator uses `curv >= 0.05` — boundary inclusive."""
    for app, pols in artifact["per_app"].items():
        for pol, cell in pols.items():
            expected = cell["curvature_at_4MB"] >= KNEE_THRESHOLD
            assert cell["knee_present"] == expected, (
                f"{app}/{pol}: knee_present drift"
            )


def test_cells_gate_requires_all_three_l3s(artifact, upstream):
    """Generator skips cells where any of {1MB, 4MB, 8MB} is missing."""
    for app, pols in artifact["per_app"].items():
        traj = upstream["per_app"][app]["trajectory_by_policy"]
        for pol in pols.keys():
            assert all(s in traj[pol] for s in PAPER_L3_SIZES), (
                f"{app}/{pol} appears in artifact but upstream traj is "
                f"missing one of {PAPER_L3_SIZES}"
            )


# ----------------------------------------------------------------------
# Group C: per-policy reducers
# ----------------------------------------------------------------------

def test_per_policy_n_cells_matches_count(artifact):
    for pol, s in artifact["per_policy_summary"].items():
        n = sum(1 for pols in artifact["per_app"].values() if pol in pols)
        assert s["n_cells"] == n, f"{pol}: n_cells {s['n_cells']} ≠ {n}"


def test_per_policy_knee_count_matches_per_app(artifact):
    for pol, s in artifact["per_policy_summary"].items():
        n_knee = sum(
            1 for pols in artifact["per_app"].values()
            if pol in pols and pols[pol]["knee_present"]
        )
        assert s["knee_count"] == n_knee, (
            f"{pol}: knee_count {s['knee_count']} ≠ {n_knee}"
        )


def test_per_policy_mean_curvature_matches_fmean(artifact):
    """Uses statistics.fmean; 0.0 if no cells."""
    for pol, s in artifact["per_policy_summary"].items():
        curvs = [
            pols[pol]["curvature_at_4MB"]
            for pols in artifact["per_app"].values()
            if pol in pols
        ]
        expected = round(statistics.fmean(curvs), 4) if curvs else 0.0
        assert s["mean_curvature"] == expected


def test_per_policy_median_curvature_matches_statistics_median(artifact):
    """Uses statistics.median (averages two middle elements for even n)."""
    for pol, s in artifact["per_policy_summary"].items():
        curvs = [
            pols[pol]["curvature_at_4MB"]
            for pols in artifact["per_app"].values()
            if pol in pols
        ]
        expected = round(statistics.median(curvs), 4) if curvs else 0.0
        assert s["median_curvature"] == expected


def test_meta_cells_totals_match_per_app(artifact):
    total = sum(len(p) for p in artifact["per_app"].values())
    knee = sum(
        1 for p in artifact["per_app"].values()
        for c in p.values() if c["knee_present"]
    )
    assert artifact["meta"]["cells_total"] == total
    assert artifact["meta"]["cells_with_knee"] == knee


# ----------------------------------------------------------------------
# Group D: rank & verdict & cross-gate consistency
# ----------------------------------------------------------------------

def test_knee_rank_by_policy_sorted_by_minus_count_then_minus_mean(artifact):
    """Generator sorts by (-knee_count, -mean_curvature) — ties broken
    by higher mean_curvature first."""
    items = list(artifact["per_policy_summary"].items())
    expected = [
        pol for pol, _ in sorted(
            items, key=lambda kv: (-kv[1]["knee_count"], -kv[1]["mean_curvature"]),
        )
    ]
    assert artifact["meta"]["knee_rank_by_policy"] == expected


def test_knee_rank_is_permutation_of_policies(artifact):
    assert (
        sorted(artifact["meta"]["knee_rank_by_policy"])
        == sorted(artifact["per_policy_summary"].keys())
    )


def test_knee_lead_verdict_uses_oracle_vs_non_oracle_min_max(artifact):
    pps = artifact["per_policy_summary"]
    min_oracle = min(pps[p]["knee_count"] for p in ORACLE_AWARE)
    max_nonoracle = max(pps[p]["knee_count"] for p in NON_ORACLE)
    expected = "PASS" if min_oracle > max_nonoracle else "FAIL"
    assert artifact["meta"]["knee_lead_verdict"] == expected


def test_cross_gate_consistency_lead_agrees_matches(artifact):
    cgc = artifact["meta"].get("cross_gate_consistency")
    if cgc is None:
        pytest.skip("cross_gate_consistency absent (gate55 artifact missing)")
    sat_rank = cgc.get("saturation_rank_gate55") or []
    knee_rank = cgc.get("knee_rank_gate58") or []
    expected = bool(sat_rank and knee_rank and sat_rank[0] == knee_rank[0])
    assert cgc["lead_agrees"] is expected


def test_cross_gate_consistency_mirrors_artifacts(artifact):
    """When both gate55 and gate58 artifacts exist, the cross-gate
    block must mirror their current contents (no stale snapshot)."""
    cgc = artifact["meta"].get("cross_gate_consistency")
    if cgc is None or not GATE55_PATH.exists():
        pytest.skip("gate55 artifact missing — cross-gate block optional")
    gate55 = json.loads(GATE55_PATH.read_text())
    expected_sat = gate55["meta"].get("saturation_rank_by_policy", [])
    assert cgc["saturation_rank_gate55"] == expected_sat
    assert cgc["knee_rank_gate58"] == artifact["meta"]["knee_rank_by_policy"]


# ----------------------------------------------------------------------
# Group E: end-to-end sanity
# ----------------------------------------------------------------------

def test_per_policy_knee_count_le_n_cells(artifact):
    for pol, s in artifact["per_policy_summary"].items():
        assert 0 <= s["knee_count"] <= s["n_cells"]


def test_meta_cells_with_knee_le_total(artifact):
    assert (
        0
        <= artifact["meta"]["cells_with_knee"]
        <= artifact["meta"]["cells_total"]
    )


def test_per_cell_knee_implies_curvature_at_or_above_threshold(artifact):
    for app, pols in artifact["per_app"].items():
        for pol, cell in pols.items():
            if cell["knee_present"]:
                assert cell["curvature_at_4MB"] >= KNEE_THRESHOLD, (
                    f"{app}/{pol}: knee_present True but curv "
                    f"{cell['curvature_at_4MB']} < {KNEE_THRESHOLD}"
                )
