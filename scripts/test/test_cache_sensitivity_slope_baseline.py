"""Gate 92 — cache-sensitivity slope baseline lock.

Two slope artifacts read the same oracle gap data from different angles:
``cache_sensitivity_slope`` aggregates per (app, policy) across all graphs,
``per_graph_cache_slope`` keeps the (graph, app, policy) trajectories so the
anti-scaling cells can be enumerated.  The "all monotonic" assumption
(miss-rate / oracle-gap shrinks as the L3 grows) is **violated** in the
current data; both artifacts agree on the exact violation budget.

This gate locks the violation budget so a silent generator change that
hides anti-scaling cells (or, inversely, accidentally inflates them) is
caught immediately:

- cache_sensitivity_slope: 10 monotonic violations across 20 (app, policy)
  cells, on a known 10-pair set.
- per_graph_cache_slope: 44 cells with significant anti-scaling across
  112 full trajectories, partitioned exactly as
  ``{LRU: 16, POPT: 8, SRRIP: 12, GRASP: 8}`` (sum 44); 16 of those 44 are
  oracle-aware (rank-1 oracle policy still anti-scales).

  Baseline re-pinned 2026-06-12 to the reproducible single-thread,
  array-relative-GRASP (0.15) corpus; the prior figures (9/33/8) were
  measured under the non-deterministic multi-threaded cache_sim runs.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSS_PATH = REPO / "wiki/data/cache_sensitivity_slope.json"
PGS_PATH = REPO / "wiki/data/per_graph_cache_slope.json"

EXPECTED_APPS = ["bc", "bfs", "cc", "pr", "sssp"]
EXPECTED_POLICIES = ["GRASP", "LRU", "POPT", "SRRIP"]
EXPECTED_L3_OCTAVES = ["1MB", "4MB", "8MB"]
EXPECTED_GRAPHS = [
    "cit-Patents",
    "com-orkut",
    "email-Eu-core",
    "soc-LiveJournal1",
    "soc-pokec",
    "web-Google",
]

# The 10 (app, policy) pairs where the gap shrinkage is NOT monotonic.
EXPECTED_VIOLATION_PAIRS = frozenset(
    {
        ("bc", "GRASP"),
        ("bc", "LRU"),
        ("bc", "POPT"),
        ("bc", "SRRIP"),
        ("bfs", "GRASP"),
        ("bfs", "LRU"),
        ("bfs", "SRRIP"),
        ("cc", "POPT"),
        ("pr", "GRASP"),
        ("pr", "POPT"),
    }
)

EXPECTED_VIOLATION_COUNT = 10
EXPECTED_FULL_TRAJECTORIES = 112
EXPECTED_ANTI_SCALING_CELLS = 44
EXPECTED_ORACLE_AWARE_ANTI_SCALING = 16
EXPECTED_POLICY_ANTI_SCALING_COUNT = {"LRU": 16, "POPT": 8, "SRRIP": 12, "GRASP": 8}


def _load(path: Path) -> dict:
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# cache_sensitivity_slope.json — aggregate-level invariants
# ---------------------------------------------------------------------------


def test_css_meta_scope_is_locked():
    meta = _load(CSS_PATH)["meta"]
    assert meta["apps"] == EXPECTED_APPS
    assert sorted(meta["policies"]) == sorted(EXPECTED_POLICIES)
    assert meta["l3_octaves"] == EXPECTED_L3_OCTAVES
    assert meta["n_apps"] == len(EXPECTED_APPS)
    assert meta["n_policies"] == len(EXPECTED_POLICIES)


def test_css_violation_count_matches_known_baseline():
    meta = _load(CSS_PATH)["meta"]
    assert meta["n_monotonic_violations"] == EXPECTED_VIOLATION_COUNT
    assert meta["all_monotonic"] is False


def test_css_violation_list_length_agrees_with_meta():
    css = _load(CSS_PATH)
    assert len(css["monotonic_violations"]) == css["meta"]["n_monotonic_violations"]


def test_css_violation_pairs_match_known_set():
    css = _load(CSS_PATH)
    actual = frozenset(
        (v["app"], v["policy"]) for v in css["monotonic_violations"]
    )
    assert actual == EXPECTED_VIOLATION_PAIRS, (
        f"violation pairs drifted: only-in-actual={actual - EXPECTED_VIOLATION_PAIRS} "
        f"only-in-expected={EXPECTED_VIOLATION_PAIRS - actual}"
    )


def test_css_per_app_covers_full_app_policy_grid():
    per_app = _load(CSS_PATH)["per_app"]
    assert set(per_app) == set(EXPECTED_APPS)
    for app in EXPECTED_APPS:
        assert set(per_app[app]) == set(EXPECTED_POLICIES), (
            f"missing policies in per_app[{app}]: have={set(per_app[app])}"
        )


def test_css_per_app_monotonic_flag_inverts_violation_set():
    per_app = _load(CSS_PATH)["per_app"]
    n_mono = 0
    n_total = 0
    for app, by_pol in per_app.items():
        for pol, summary in by_pol.items():
            n_total += 1
            mono = summary["monotonic_decreasing"]
            in_violations = (app, pol) in EXPECTED_VIOLATION_PAIRS
            assert mono is (not in_violations), (
                f"({app},{pol}) monotonic_decreasing={mono} but in_violations={in_violations}"
            )
            if mono:
                n_mono += 1
    assert n_total == len(EXPECTED_APPS) * len(EXPECTED_POLICIES)
    assert n_mono == n_total - EXPECTED_VIOLATION_COUNT


def test_css_per_app_octave_count_is_two():
    per_app = _load(CSS_PATH)["per_app"]
    expected_octaves = len(EXPECTED_L3_OCTAVES) - 1  # N-1 segments
    for app, by_pol in per_app.items():
        for pol, summary in by_pol.items():
            assert len(summary["octaves"]) == expected_octaves, (
                f"({app},{pol}) has {len(summary['octaves'])} octave segments"
            )


def test_css_per_policy_summary_covers_all_policies_with_full_apps():
    pps = _load(CSS_PATH)["per_policy_summary"]
    assert set(pps) == set(EXPECTED_POLICIES)
    for pol, summary in pps.items():
        assert summary["n_apps"] == len(EXPECTED_APPS), (
            f"per_policy_summary[{pol}].n_apps={summary['n_apps']}"
        )


# ---------------------------------------------------------------------------
# per_graph_cache_slope.json — per-graph anti-scaling invariants
# ---------------------------------------------------------------------------


def test_pgs_meta_scope_is_locked():
    meta = _load(PGS_PATH)["meta"]
    assert sorted(meta["apps"]) == sorted(EXPECTED_APPS)
    assert sorted(meta["policies"]) == sorted(EXPECTED_POLICIES)
    assert sorted(meta["graphs"]) == sorted(EXPECTED_GRAPHS)
    assert meta["scope_l3_sizes"] == EXPECTED_L3_OCTAVES


def test_pgs_full_trajectory_count_matches_baseline():
    meta = _load(PGS_PATH)["meta"]
    assert meta["n_full_trajectories"] == EXPECTED_FULL_TRAJECTORIES
    assert meta["n_graphs_with_full_trajectory"] == len(EXPECTED_GRAPHS)


def test_pgs_anti_scaling_cell_baseline_matches():
    pgs = _load(PGS_PATH)
    meta = pgs["meta"]
    assert meta["n_cells_with_significant_anti_scaling"] == EXPECTED_ANTI_SCALING_CELLS
    assert len(pgs["anti_scaling_cells"]) == EXPECTED_ANTI_SCALING_CELLS


def test_pgs_per_policy_anti_scaling_partition_matches():
    pgs = _load(PGS_PATH)
    assert pgs["per_policy_anti_scaling_count"] == EXPECTED_POLICY_ANTI_SCALING_COUNT
    assert sum(pgs["per_policy_anti_scaling_count"].values()) == EXPECTED_ANTI_SCALING_CELLS


def test_pgs_per_graph_anti_scaling_sums_to_total():
    pgs = _load(PGS_PATH)
    per_graph = pgs["per_graph_anti_scaling_count"]
    per_graph_policy = pgs["per_graph_policy_anti_scaling_count"]
    assert sum(per_graph.values()) == EXPECTED_ANTI_SCALING_CELLS
    assert sum(per_graph_policy.values()) == EXPECTED_ANTI_SCALING_CELLS


def test_pgs_oracle_aware_anti_scaling_count_matches():
    meta = _load(PGS_PATH)["meta"]
    assert meta["n_oracle_aware_anti_scaling"] == EXPECTED_ORACLE_AWARE_ANTI_SCALING
    assert (
        EXPECTED_ORACLE_AWARE_ANTI_SCALING <= EXPECTED_ANTI_SCALING_CELLS
    ), "oracle-aware count must be a subset of total anti-scaling cells"
