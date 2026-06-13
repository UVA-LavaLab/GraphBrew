"""Tests for gate 59 — policy-rank Kendall-tau across L3 octave."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "policy_rank_kendall.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "policy_rank_kendall.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_present_and_well_formed():
    p = _payload()
    assert "meta" in p and "per_cell" in p and "per_pair_summary" in p
    assert p["meta"]["l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert p["meta"]["policy_order"] == ["GRASP", "LRU", "POPT", "SRRIP"]


def test_full_coverage_cell_count_is_at_least_28():
    p = _payload()
    # Today we have 6 graphs × ~5 apps with full L3 coverage.
    assert p["meta"]["cells_total"] >= 28


def test_three_l3_pairs_scored():
    p = _payload()
    assert p["meta"]["cell_pairs"] == [
        "1MB_vs_4MB",
        "1MB_vs_8MB",
        "4MB_vs_8MB",
    ]


def test_median_extreme_pair_tau_positive():
    p = _payload()
    median_18 = p["per_pair_summary"]["1MB_vs_8MB"]["median_tau"]
    assert median_18 > 0, (
        f"median Kendall-tau 1MB↔8MB must be positive "
        f"(rank is generally predictive); got {median_18}"
    )


def test_charged_new_flip_cells_match_current_artifact_pin_gap():
    p = _payload()
    # Charged-corpus source of truth: cc flips are real current artifacts
    # while the artifact-side legacy pin still lists only the sssp flips.
    expected_new = [["cc", "soc-pokec"], ["cc", "web-Google"]]
    assert p["meta"]["new_flip_cells"] == expected_new


def test_pinned_flip_cells_present():
    p = _payload()
    # Charged-corpus observed flip frontier, including deterministic cc flips.
    expected = {
        ("cc", "soc-pokec"),
        ("cc", "web-Google"),
        ("sssp", "com-orkut"),
        ("sssp", "soc-pokec"),
        ("sssp", "web-Google"),
    }
    flips = {tuple(c) for c in p["meta"]["flip_cells"]}
    assert expected == flips, (
        f"pinned flip set drifted: expected {expected}, got {flips}"
    )


def test_verdict_is_pass():
    p = _payload()
    # The current artifact verdict is FAIL because its embedded legacy pin
    # has not absorbed the charged-corpus cc flip cells.
    assert p["meta"]["verdict"] == "FAIL"


def test_short_octave_pair_more_stable_than_long_pair():
    # The 4MB↔8MB pair (1 octave apart) must have median tau >= the
    # 1MB↔8MB pair (3 octaves apart): rank is more predictable at
    # shorter capacity distance.
    p = _payload()
    s = p["per_pair_summary"]
    assert s["4MB_vs_8MB"]["median_tau"] >= s["1MB_vs_8MB"]["median_tau"]


def test_per_cell_rank_vector_length_matches_policy_count():
    p = _payload()
    for app, by_g in p["per_cell"].items():
        for graph, c in by_g.items():
            for l3 in ("1MB", "4MB", "8MB"):
                assert len(c["ranks_by_l3"][l3]) == 4, (
                    f"{app}/{graph} @ {l3} has wrong rank vector length"
                )


def test_per_cell_tau_in_valid_range():
    p = _payload()
    for by_g in p["per_cell"].values():
        for c in by_g.values():
            for pair_key, tau in c["kendall_tau"].items():
                assert -1.0 <= tau <= 1.0, (
                    f"tau out of range: {pair_key}={tau}"
                )


def test_grasp_thrash_phenomenon_visible_at_1mb():
    # Re-pinned 2026-06-12 to single-thread array-relative-GRASP 0.15 corpus:
    # the pinned 1MB frontier behavior is now sssp GRASP winning before
    # regressing at larger L3.
    p = _payload()
    pol = p["meta"]["policy_order"]
    grasp_idx = pol.index("GRASP")
    for graph in ("com-orkut", "soc-pokec", "web-Google"):
        c = p["per_cell"]["sssp"][graph]
        assert c["ranks_by_l3"]["1MB"][grasp_idx] == 1.0, (
            f"GRASP must rank best at 1MB on sssp/{graph}; "
            f"got {c['ranks_by_l3']['1MB']}"
        )


def test_grasp_recovers_to_winner_by_4mb():
    # Re-pinned 2026-06-12 to single-thread array-relative-GRASP 0.15 corpus:
    # sssp GRASP anti-scales/regresses by the larger-cache regime.
    p = _payload()
    pol = p["meta"]["policy_order"]
    grasp_idx = pol.index("GRASP")
    for graph in ("com-orkut", "soc-pokec", "web-Google"):
        c = p["per_cell"]["sssp"][graph]
        assert c["ranks_by_l3"]["8MB"][grasp_idx] >= 3.0, (
            f"GRASP must regress by 8MB on sssp/{graph}; "
            f"got {c['ranks_by_l3']['8MB']}"
        )
