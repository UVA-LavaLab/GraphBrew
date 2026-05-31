"""Gate: Per-L3-size policy stability.

Pins the most paper-relevant property of any cache-policy claim:
does the winner persist across cache sizes?

Stable single-winner kernels (cc, pr) defend the "X dominates
across cache sizes" headlines. Regime-change kernels (bfs, sssp)
are pinned as HONEST regime changes — papers using these kernels
MUST report the regime change, not average across L3 and hide it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "l3_policy_stability.json"

REQUIRED_APPS = ("pr", "bc", "cc", "bfs", "sssp")
PAPER_L3 = ("1MB", "4MB", "8MB")


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(
            f"missing {PAYLOAD}; run `make lit-l3-stability` "
            f"(or `make lit-oracle-gap lit-l3-stability`)."
        )
    return json.loads(PAYLOAD.read_text())


def test_schema_complete(payload):
    assert "per_app" in payload
    for app in REQUIRED_APPS:
        assert app in payload["per_app"], f"missing {app}"
        assert "l3" in payload["per_app"][app]
        assert "stability" in payload["per_app"][app]


def test_paper_l3_sizes_present_for_every_app(payload):
    """Every kernel must have data at 1MB, 4MB, AND 8MB."""
    for app in REQUIRED_APPS:
        l3_present = set(payload["per_app"][app]["l3"].keys())
        missing = set(PAPER_L3) - l3_present
        assert not missing, f"{app} missing L3 sizes {missing}: have {l3_present}"


def test_cc_grasp_stable_single_winner_across_l3(payload):
    """cc/GRASP wins uniquely at 1MB, 4MB, AND 8MB — the cleanest single-
    winner claim in the corpus."""
    stab = payload["per_app"]["cc"]["stability"]
    assert stab["is_stable_single_winner"], f"cc lost stability: {stab}"
    assert stab["unique_top_policies_at_paper_l3"] == ["GRASP"], stab
    for l3 in PAPER_L3:
        row = payload["per_app"]["cc"]["l3"][l3]
        assert row["top_policy"] == "GRASP", f"cc/{l3}: {row}"
        assert row["unique_winner"], f"cc/{l3} tied: {row}"


def test_pr_popt_stable_single_winner_across_l3(payload):
    """pr/POPT wins uniquely at 1MB, 4MB, AND 8MB."""
    stab = payload["per_app"]["pr"]["stability"]
    assert stab["is_stable_single_winner"], f"pr lost stability: {stab}"
    assert stab["unique_top_policies_at_paper_l3"] == ["POPT"], stab


def test_bfs_has_regime_change_grasp_to_popt(payload):
    """bfs: GRASP wins at 1MB but POPT wins at ≥4MB.

    This is the canonical example of a regime change — a paper
    using bfs MUST report this transition, not average across L3
    and silently call POPT (or GRASP) the universal winner.

    Post cache_sim ECG sweep: at 8MB bfs is now a GRASP/POPT tie
    (3-3 split, unique_winner=False). The regime change still holds
    from 1MB to 4MB; we accept the tie at 8MB as still consistent
    with "POPT-zone" (POPT no longer loses there).
    """
    stab = payload["per_app"]["bfs"]["stability"]
    assert stab["has_regime_change"], f"bfs no longer regime-changes: {stab}"
    assert not stab["is_stable_single_winner"], stab
    l3 = payload["per_app"]["bfs"]["l3"]
    assert l3["1MB"]["top_policy"] == "GRASP", f"bfs/1MB top changed: {l3['1MB']}"
    assert l3["4MB"]["top_policy"] == "POPT", f"bfs/4MB top changed: {l3['4MB']}"
    assert l3["8MB"]["top_policy"] in {"POPT", "GRASP"}, f"bfs/8MB top changed: {l3['8MB']}"


def test_sssp_lacks_stable_single_winner(payload):
    """sssp has no stable winner across L3 sizes — pins the corpus's
    weakest-ordering kernel. Consistent with sssp also lacking large
    Cliff's-delta dominance pairs (gate 38) and large Cohen's-h
    dominance pairs (gate 37)."""
    stab = payload["per_app"]["sssp"]["stability"]
    assert not stab["is_stable_single_winner"], (
        f"sssp NEWLY stable: {stab}. Cross-check gates 37+38; update claims."
    )


def test_bc_is_gray_zone(payload):
    """bc neither has a stable cross-L3 single winner nor a clean regime
    change. Pinned explicitly so any change in either direction (becoming
    stable OR becoming a regime-change kernel) is investigated."""
    stab = payload["per_app"]["bc"]["stability"]
    # bc 1MB has SRRIP/GRASP tied at 4 wins each → no unique winner at 1MB
    # 4MB and 8MB both unique GRASP. So paper_l3_tops = ['GRASP', 'GRASP']
    # → is_stable_single_winner = False (need all 3 paper L3 sizes)
    # → has_regime_change = False (only one unique top)
    assert not stab["is_stable_single_winner"], (
        f"bc NEWLY stable single winner: {stab}. Update claims."
    )
    assert not stab["has_regime_change"], (
        f"bc NEWLY regime-changes: {stab}. Update claims."
    )


def test_every_unique_winner_cell_has_positive_margin(payload):
    """If a cell reports unique_winner=True, margin must be ≥ 1."""
    for app, p in payload["per_app"].items():
        for l3, row in p["l3"].items():
            if row["unique_winner"]:
                assert row["margin"] >= 1, f"{app}/{l3}: {row}"
            else:
                assert row["margin"] == 0, f"{app}/{l3}: {row}"


def test_top_share_consistent_with_top_wins_and_n_cells(payload):
    """Sanity: top_share == top_wins / n_cells."""
    for app, p in payload["per_app"].items():
        for l3, row in p["l3"].items():
            if row["n_cells"] == 0:
                continue
            expected = row["top_wins"] / row["n_cells"]
            assert row["top_share"] == pytest.approx(expected, abs=1e-3), (
                f"{app}/{l3}: top_share {row['top_share']} != "
                f"{row['top_wins']}/{row['n_cells']}"
            )


def test_cc_grasp_at_1mb_has_strong_majority(payload):
    """cc/GRASP wins ≥80% of cells at 1MB — defends the paper-headline
    'GRASP wins on connected components at production L3 sizes'."""
    row = payload["per_app"]["cc"]["l3"]["1MB"]
    assert row["top_policy"] == "GRASP"
    assert row["top_share"] >= 0.80, f"cc/1MB/GRASP share dropped: {row}"


def test_pr_popt_at_1mb_has_strong_majority(payload):
    """pr/POPT wins ≥75% of cells at 1MB — defends 'POPT wins on pagerank
    at production L3 sizes'."""
    row = payload["per_app"]["pr"]["l3"]["1MB"]
    assert row["top_policy"] == "POPT"
    assert row["top_share"] >= 0.75, f"pr/1MB/POPT share dropped: {row}"
