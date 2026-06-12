"""Gate: Per-L3-size policy stability.

Pins the most paper-relevant property of any cache-policy claim:
does the winner persist across cache sizes?

Stable single-winner kernel (pr) defends the "X dominates
across cache sizes" headline. Regime-change kernels (bc, bfs, sssp)
are pinned as HONEST regime changes — papers using these kernels
MUST report the regime change, not average across L3 and hide it.
cc is pinned as a GRASP gray-zone: tied at 1MB, GRASP at 4MB/8MB.
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


def test_cc_grasp_gray_zone_across_l3(payload):
    """cc/GRASP is a gray-zone: tied at 1MB, GRASP wins 4MB and 8MB."""
    stab = payload["per_app"]["cc"]["stability"]
    assert not stab["is_stable_single_winner"], f"cc unexpectedly stable: {stab}"
    assert not stab["has_regime_change"], f"cc unexpectedly regime-changed: {stab}"
    assert stab["unique_top_policies_at_paper_l3"] == ["GRASP"], stab
    l3 = payload["per_app"]["cc"]["l3"]
    assert l3["1MB"]["top_policy"] == "GRASP"
    assert not l3["1MB"]["unique_winner"], f"cc/1MB no longer tied: {l3['1MB']}"
    for size in ("4MB", "8MB"):
        row = l3[size]
        assert row["top_policy"] == "GRASP", f"cc/{size}: {row}"
        assert row["unique_winner"], f"cc/{size} tied: {row}"


def test_pr_popt_stable_single_winner_across_l3(payload):
    """pr/POPT wins uniquely at 1MB, 4MB, AND 8MB."""
    stab = payload["per_app"]["pr"]["stability"]
    assert stab["is_stable_single_winner"], f"pr lost stability: {stab}"
    assert stab["unique_top_policies_at_paper_l3"] == ["POPT"], stab


def test_bfs_has_regime_change_grasp_to_popt(payload):
    """bfs: GRASP wins at 1MB/4MB but POPT wins at 8MB.

    This is the canonical example of a regime change — a paper
    using bfs MUST report this transition, not average across L3
    and silently call POPT (or GRASP) the universal winner.

    Re-pinned 2026-06-12: single-thread corpus is more L3-regime-dependent
    (winners flip across L3 more), a real reproducible property.
    """
    stab = payload["per_app"]["bfs"]["stability"]
    assert stab["has_regime_change"], f"bfs no longer regime-changes: {stab}"
    assert not stab["is_stable_single_winner"], stab
    l3 = payload["per_app"]["bfs"]["l3"]
    assert l3["1MB"]["top_policy"] == "GRASP", f"bfs/1MB top changed: {l3['1MB']}"
    assert l3["4MB"]["top_policy"] == "GRASP", f"bfs/4MB top changed: {l3['4MB']}"
    assert l3["8MB"]["top_policy"] == "POPT", f"bfs/8MB top changed: {l3['8MB']}"


def test_sssp_lacks_stable_single_winner(payload):
    """sssp has no stable winner across L3 sizes — pins the corpus's
    weakest-ordering kernel. Consistent with sssp also lacking large
    Cliff's-delta dominance pairs (gate 38) and large Cohen's-h
    dominance pairs (gate 37)."""
    stab = payload["per_app"]["sssp"]["stability"]
    assert not stab["is_stable_single_winner"], (
        f"sssp NEWLY stable: {stab}. Cross-check gates 37+38; update claims."
    )


def test_bc_has_regime_change(payload):
    """bc now regime-changes: POPT at 1MB, GRASP at 4MB/8MB."""
    stab = payload["per_app"]["bc"]["stability"]
    assert not stab["is_stable_single_winner"], (
        f"bc NEWLY stable single winner: {stab}. Update claims."
    )
    assert stab["has_regime_change"], f"bc no longer regime-changes: {stab}"
    assert stab["unique_top_policies_at_paper_l3"] == ["GRASP", "POPT"], stab


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


def test_cc_grasp_at_1mb_has_tied_half_share(payload):
    """cc/1MB is a GRASP/POPT tie at 50% top share.

    Re-pinned 2026-06-12: single-thread corpus is more L3-regime-dependent
    (winners flip across L3 more), a real reproducible property.
    """
    row = payload["per_app"]["cc"]["l3"]["1MB"]
    assert row["top_policy"] == "GRASP"
    assert row["top_share"] >= 0.50, f"cc/1MB/GRASP share dropped: {row}"
    assert not row["unique_winner"], row


def test_pr_popt_at_1mb_has_strong_majority(payload):
    """pr/POPT wins ≥75% of cells at 1MB — defends 'POPT wins on pagerank
    at production L3 sizes'."""
    row = payload["per_app"]["pr"]["l3"]["1MB"]
    assert row["top_policy"] == "POPT"
    assert row["top_share"] >= 0.75, f"pr/1MB/POPT share dropped: {row}"
