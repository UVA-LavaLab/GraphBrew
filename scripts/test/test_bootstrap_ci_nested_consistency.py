"""Gate 94 — bootstrap CI nested-claim consistency.

Two bootstrap artifacts publish confidence intervals over the same
underlying oracle-gap measurements from different aggregation angles:

- ``bootstrap_ci.json`` aggregates by (policy, family), by (policy,
  regime), per-family POPT minus GRASP deltas, and per-(family,
  policy_a, policy_b) sign-stability fractions. n_resamples=5000.
- ``oracle_gap_by_app_bootstrap.json`` publishes the full 4×3
  ordered-pair (policy_a vs policy_b) CI per app. n_resamples=2000.

The two share the bootstrap seed (1729) and the 95% confidence level.
They are derived from the same 456 (policy, cell) measurements, so
the per-app n_paired values must aggregate to the corpus cell count
(23+23+20+28+20=114 cells × 4 policies = 456 rows).

Within ``oracle_gap_by_app_bootstrap`` each (A_vs_B, B_vs_A) pair
must have exactly anti-symmetric mean_delta values and roughly
mirrored CI bounds (modulo bootstrap-quantile noise).

This gate locks the bootstrap configuration, the per-family/per-app
cell counts, the pair-symmetry within per_app_pairs, and the
sign_stability roster.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BCI_PATH = REPO / "wiki/data/bootstrap_ci.json"
OBA_PATH = REPO / "wiki/data/oracle_gap_by_app_bootstrap.json"

EXPECTED_SEED = 1729
EXPECTED_CI_LEVEL = 0.95
EXPECTED_BCI_RESAMPLES = 5000
EXPECTED_OBA_RESAMPLES = 2000

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"GRASP", "LRU", "POPT", "SRRIP"}
EXPECTED_FAMILIES = {"citation", "mesh", "road", "social", "web"}
EXPECTED_REGIMES = {"large", "small", "tiny"}

EXPECTED_FAMILY_CELL_COUNT = {
    "citation": 15,
    "mesh": 5,
    "road": 25,
    "social": 54,
    "web": 15,
}
EXPECTED_TOTAL_CELLS = 114

EXPECTED_APP_N_PAIRED = {"bc": 23, "bfs": 23, "cc": 20, "pr": 28, "sssp": 20}
EXPECTED_TOTAL_ROWS = 456  # 114 cells × 4 policies

EXPECTED_PAIRS_PER_APP = 12  # 4 policies × 3 = ordered (A,B) with A≠B

# 7 known sign_stability entries: 5 (POPT, GRASP) per family + 2 socials
EXPECTED_SIGN_STABILITY_PAIRS = frozenset(
    {
        ("road", "POPT", "GRASP"),
        ("social", "POPT", "GRASP"),
        ("mesh", "POPT", "GRASP"),
        ("citation", "POPT", "GRASP"),
        ("web", "POPT", "GRASP"),
        ("social", "POPT", "LRU"),
        ("social", "GRASP", "LRU"),
    }
)

# Bootstrap-quantile slop: paired-bootstrap CI bounds aren't exactly
# anti-symmetric because each direction independently samples its own
# percentile, especially when the distribution is skewed.
CI_SYMMETRY_TOL_PP = 1.0
MEAN_ANTISYMMETRY_TOL_PP = 1e-6
CI_WIDTH_TOL_PP = 1e-3


def _load(path: Path) -> dict:
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# bootstrap_ci.json — single-artifact invariants
# ---------------------------------------------------------------------------


def test_bci_meta_locks_seed_ci_resamples():
    meta = _load(BCI_PATH)["meta"]
    assert meta["seed"] == EXPECTED_SEED
    assert math.isclose(meta["ci_level"], EXPECTED_CI_LEVEL)
    assert meta["n_resamples"] == EXPECTED_BCI_RESAMPLES


def test_bci_by_policy_family_covers_full_4x5_grid():
    bpf = _load(BCI_PATH)["oracle_gap_by_policy_family"]
    expected_keys = {
        f"{pol}/{fam}" for pol in EXPECTED_POLICIES for fam in EXPECTED_FAMILIES
    }
    assert set(bpf) == expected_keys
    for key, stats in bpf.items():
        pol, fam = key.split("/")
        assert stats["n"] == EXPECTED_FAMILY_CELL_COUNT[fam], (
            f"{key}: n={stats['n']} != expected {EXPECTED_FAMILY_CELL_COUNT[fam]}"
        )
        assert math.isclose(stats["ci_level"], EXPECTED_CI_LEVEL)
        assert stats["ci_hi"] >= stats["ci_lo"]
        assert math.isclose(stats["ci_width"], stats["ci_hi"] - stats["ci_lo"], abs_tol=CI_WIDTH_TOL_PP)


def test_bci_family_cell_counts_sum_to_corpus_total():
    assert sum(EXPECTED_FAMILY_CELL_COUNT.values()) == EXPECTED_TOTAL_CELLS


def test_bci_popt_minus_grasp_by_family_complete_and_self_consistent():
    pmg = _load(BCI_PATH)["popt_minus_grasp_by_family"]
    assert set(pmg) == EXPECTED_FAMILIES
    for fam, stats in pmg.items():
        assert stats["n"] == EXPECTED_FAMILY_CELL_COUNT[fam]
        excludes_zero = stats["ci_excludes_zero"]
        spans_zero = stats["ci_lo"] < 0.0 < stats["ci_hi"]
        assert excludes_zero is (not spans_zero), (
            f"family {fam}: ci_excludes_zero={excludes_zero} disagrees with "
            f"ci=[{stats['ci_lo']}, {stats['ci_hi']}]"
        )
        sign = stats["sign"]
        if spans_zero:
            assert sign == "0", f"family {fam}: ci spans zero but sign={sign}"
        else:
            expected_sign = "+" if stats["ci_lo"] > 0 else "-"
            assert sign == expected_sign, (
                f"family {fam}: sign={sign} disagrees with ci=[{stats['ci_lo']}, {stats['ci_hi']}]"
            )


def test_bci_sign_stability_roster_matches_known_set():
    bci = _load(BCI_PATH)
    entries = bci["sign_stability"]
    actual = frozenset((e["family"], e["policy_a"], e["policy_b"]) for e in entries)
    assert actual == EXPECTED_SIGN_STABILITY_PAIRS, (
        f"sign_stability roster drifted: "
        f"only-in-actual={actual - EXPECTED_SIGN_STABILITY_PAIRS} "
        f"only-in-expected={EXPECTED_SIGN_STABILITY_PAIRS - actual}"
    )
    for e in entries:
        assert e["family"] in EXPECTED_FAMILIES
        assert e["policy_a"] in EXPECTED_POLICIES
        assert e["policy_b"] in EXPECTED_POLICIES
        assert e["n_a"] == e["n_b"] == EXPECTED_FAMILY_CELL_COUNT[e["family"]]
        assert 0.0 <= e["frac_a_lt_b"] <= 1.0


def test_bci_by_policy_regime_covers_full_4x3_grid():
    bpr = _load(BCI_PATH)["oracle_gap_by_policy_regime"]
    expected_keys = {
        f"{pol}/{reg}" for pol in EXPECTED_POLICIES for reg in EXPECTED_REGIMES
    }
    assert set(bpr) == expected_keys
    for key, stats in bpr.items():
        assert stats["ci_hi"] >= stats["ci_lo"]
        assert math.isclose(stats["ci_level"], EXPECTED_CI_LEVEL)


# ---------------------------------------------------------------------------
# oracle_gap_by_app_bootstrap.json — single-artifact invariants
# ---------------------------------------------------------------------------


def test_oba_meta_locks_seed_ci_resamples():
    meta = _load(OBA_PATH)["meta"]
    assert meta["seed"] == EXPECTED_SEED
    assert math.isclose(meta["ci_level"], EXPECTED_CI_LEVEL)
    assert meta["n_resamples"] == EXPECTED_OBA_RESAMPLES
    assert set(meta["apps"]) == EXPECTED_APPS
    assert set(meta["policies"]) == EXPECTED_POLICIES
    assert meta["n_total_rows"] == EXPECTED_TOTAL_ROWS


def test_oba_per_app_pairs_full_ordered_grid_and_n_paired():
    pap = _load(OBA_PATH)["per_app_pairs"]
    assert set(pap) == EXPECTED_APPS
    for app, pairs in pap.items():
        # 4 × 3 ordered pairs (A vs B, A ≠ B)
        assert len(pairs) == EXPECTED_PAIRS_PER_APP, (
            f"app {app}: {len(pairs)} pairs, expected {EXPECTED_PAIRS_PER_APP}"
        )
        for key, stats in pairs.items():
            a, b = key.split("_vs_")
            assert a in EXPECTED_POLICIES and b in EXPECTED_POLICIES and a != b
            assert stats["n_paired"] == EXPECTED_APP_N_PAIRED[app], (
                f"{app}/{key}: n_paired={stats['n_paired']} != expected {EXPECTED_APP_N_PAIRED[app]}"
            )
            assert stats["ci_hi"] >= stats["ci_lo"]
            assert 0.0 <= stats["p_a_lt_b"] <= 1.0


def test_oba_per_app_pairs_mean_delta_is_anti_symmetric():
    pap = _load(OBA_PATH)["per_app_pairs"]
    for app, pairs in pap.items():
        for key, stats in pairs.items():
            a, b = key.split("_vs_")
            mirror_key = f"{b}_vs_{a}"
            mirror = pairs[mirror_key]
            delta_sum = stats["mean_delta"] + mirror["mean_delta"]
            assert abs(delta_sum) < MEAN_ANTISYMMETRY_TOL_PP, (
                f"{app}: mean_delta {key}={stats['mean_delta']} + {mirror_key}={mirror['mean_delta']} = {delta_sum}"
            )


def test_oba_per_app_pairs_ci_bounds_are_near_mirror():
    pap = _load(OBA_PATH)["per_app_pairs"]
    for app, pairs in pap.items():
        for key, stats in pairs.items():
            a, b = key.split("_vs_")
            mirror = pairs[f"{b}_vs_{a}"]
            # ci_hi(A,B) ≈ -ci_lo(B,A); ci_lo(A,B) ≈ -ci_hi(B,A)
            assert abs(stats["ci_hi"] + mirror["ci_lo"]) < CI_SYMMETRY_TOL_PP, (
                f"{app}/{key}: ci_hi={stats['ci_hi']}, mirror ci_lo={mirror['ci_lo']}"
            )
            assert abs(stats["ci_lo"] + mirror["ci_hi"]) < CI_SYMMETRY_TOL_PP, (
                f"{app}/{key}: ci_lo={stats['ci_lo']}, mirror ci_hi={mirror['ci_hi']}"
            )


def test_oba_per_app_pairs_p_a_lt_b_is_near_complementary():
    """For paired bootstrap, p(A<B) + p(B<A) should sum to 1 modulo tie-mass."""
    pap = _load(OBA_PATH)["per_app_pairs"]
    P_TOL = 0.035  # charged-corpus sssp/POPT-LRU tie-mass / quantile rounding
    for app, pairs in pap.items():
        seen = set()
        for key, stats in pairs.items():
            a, b = key.split("_vs_")
            mirror_key = f"{b}_vs_{a}"
            pair_id = frozenset({a, b})
            if pair_id in seen:
                continue
            seen.add(pair_id)
            mirror = pairs[mirror_key]
            tot = stats["p_a_lt_b"] + mirror["p_a_lt_b"]
            assert abs(tot - 1.0) <= P_TOL, (
                f"{app}/{key}+{mirror_key}: p_a_lt_b sum {tot} not ≈ 1"
            )


# ---------------------------------------------------------------------------
# Cross-artifact parity
# ---------------------------------------------------------------------------


def test_xartifact_seed_and_ci_level_match():
    bci_meta = _load(BCI_PATH)["meta"]
    oba_meta = _load(OBA_PATH)["meta"]
    assert bci_meta["seed"] == oba_meta["seed"] == EXPECTED_SEED
    assert math.isclose(bci_meta["ci_level"], oba_meta["ci_level"])
    assert math.isclose(bci_meta["ci_level"], EXPECTED_CI_LEVEL)


def test_xartifact_total_rows_equals_per_app_cells_times_policies():
    oba = _load(OBA_PATH)
    total_cells = sum(EXPECTED_APP_N_PAIRED.values())
    assert total_cells == EXPECTED_TOTAL_CELLS, (
        f"per-app cell count sum {total_cells} != corpus total {EXPECTED_TOTAL_CELLS}"
    )
    assert oba["meta"]["n_total_rows"] == total_cells * len(EXPECTED_POLICIES)
    # Verify directly against the live per_app_pairs n_paired values.
    pap = oba["per_app_pairs"]
    live_total = sum(next(iter(pairs.values()))["n_paired"] for pairs in pap.values())
    assert live_total == EXPECTED_TOTAL_CELLS
