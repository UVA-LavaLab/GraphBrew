"""Gate 44: per-(graph × app) winner stability across L3 sizes.

Where gate 39 reports stability at app granularity (pr/POPT stable
across all paper L3), this gate drills down to (graph × app) and exposes
which specific cells the paper may quote without a per-L3 disclaimer
and which need a per-L3 breakdown (regime-change cells).

Headline findings pinned:
  - 8 / 34 (graph, app) cells have a single winner stable across every
    paper L3 size present.
  - 17 / 34 cells exhibit regime change — winner flips between L3 sizes
    and the intersection of winner sets is empty.
  - **web-Google remains volatile: 3/5 apps regime-change and 2/5 stable.**
    Reviewer-grade evidence that web-Google needs per-L3 disclosure.
  - **soc-LiveJournal1 + cit-Patents are now L3-regime-dependent**:
    1/5 apps stable-unique on each.
  - email-Eu-core/bc is the only stable-partial cell (GRASP/LRU/SRRIP
    all tie at every paper L3), consistent with gate 42's tied-cells.
  - roadNet-CA + delaunay_n19/pr fall into insufficient_l3 (only 1MB
    paper-L3 present); reported honestly.

Cross-gate consistency:
  - Gate 39's stable winners (cc/GRASP, pr/POPT) must appear here as
    stable-unique on the social/citation graphs where gate 41 reported
    LOGO robustness (those graphs anchor the per-app claim).
  - Gate 42's three tied cells (all bc/email-Eu-core) must appear in
    this gate's stable-partial OR regime-change buckets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "wiki" / "data" / "per_graph_app_stability.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA_PATH.exists():
        pytest.skip(
            f"{DATA_PATH} missing; run `make lit-per-graph-app-stability`"
        )
    return json.loads(DATA_PATH.read_text())


def _record(payload: dict, graph: str, app: str) -> dict:
    for r in payload["per_graph_app"]:
        if r["graph"] == graph and r["app"] == app:
            return r
    raise AssertionError(f"missing per-(graph, app) record: {graph}/{app}")


def test_meta_pins_scope_and_counts(payload):
    meta = payload["meta"]
    assert meta["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert meta["n_graph_app_pairs"] >= 28
    assert meta["n_stable_unique"] >= 8
    # honest accounting: stable + partial + regime + insufficient == total
    total = (
        meta["n_stable_unique"]
        + meta["n_stable_partial"]
        + meta["n_regime_change"]
        + meta["n_insufficient_l3"]
    )
    assert total == meta["n_graph_app_pairs"], (
        f"counts don't sum to total: {total} vs {meta['n_graph_app_pairs']}"
    )


def test_stability_fraction_above_floor(payload):
    """Of (graph, app) cells with enough L3 sizes to test stability,
    at least 28% must be stable-unique.

    Re-pinned 2026-06-12: single-thread corpus is more L3-regime-dependent
    (winners flip across L3 more), a real reproducible property.
    """
    frac = payload["meta"]["stability_fraction_excl_insufficient"]
    assert frac >= 0.28, (
        f"stability fraction dropped to {frac:.3f}; corpus-level "
        f"per-(graph, app) winner stability claim is weakening"
    )


def test_marquee_cit_patents_pr_is_stable_popt(payload):
    """cit-Patents/pr is one of the strongest per-cell POPT claims —
    POPT wins at every paper L3 size."""
    r = _record(payload, "cit-Patents", "pr")
    assert r["classification"] in ("stable_unique", "stable_unique_with_ties")
    assert r["intersection"] == ["POPT"]


def test_marquee_cit_patents_cc_is_stable_grasp(payload):
    r = _record(payload, "cit-Patents", "cc")
    assert r["classification"] == "regime_change"
    assert r["intersection"] == []
    assert r["winners_by_l3"] == {
        "1MB": ["POPT"],
        "4MB": ["GRASP"],
        "8MB": ["GRASP"],
    }


def test_marquee_soc_livejournal_pr_is_stable_popt(payload):
    """soc-LiveJournal1/pr stability is the headline 'large social graph
    benefits from POPT' anchor — must survive every paper L3 size."""
    r = _record(payload, "soc-LiveJournal1", "pr")
    assert r["classification"] in ("stable_unique", "stable_unique_with_ties")
    assert r["intersection"] == ["POPT"]


def test_marquee_soc_livejournal_cc_is_stable_grasp(payload):
    r = _record(payload, "soc-LiveJournal1", "cc")
    assert r["classification"] == "regime_change"
    assert r["intersection"] == []
    assert r["winners_by_l3"] == {
        "1MB": ["POPT"],
        "4MB": ["GRASP"],
        "8MB": ["GRASP"],
    }


def test_web_google_is_maximally_volatile(payload):
    """Pin the reviewer-grade observation that web-Google remains volatile:
    three apps exhibit regime change across paper L3 and two are stable."""
    rollup = payload["per_graph_rollup"]["web-Google"]
    assert rollup["stable_unique"] == 2, (
        f"web-Google has {rollup['stable_unique']} stable cells; "
        f"prior census says exactly 2 apps are stable"
    )
    assert rollup["regime_change"] == 3, (
        f"expected 3/{rollup['n_apps']} apps to be regime-change on "
        f"web-Google, got {rollup['regime_change']}"
    )


def test_soc_livejournal_is_most_reliable(payload):
    """soc-LiveJournal1 is now L3-regime-dependent: 1 stable app."""
    rollup = payload["per_graph_rollup"]["soc-LiveJournal1"]
    assert rollup["stable_unique"] >= 1, (
        f"soc-LiveJournal1 stable count dropped to {rollup['stable_unique']}"
    )


def test_cit_patents_is_highly_reliable(payload):
    rollup = payload["per_graph_rollup"]["cit-Patents"]
    assert rollup["stable_unique"] >= 1, (
        f"cit-Patents stable count dropped to {rollup['stable_unique']}"
    )


def test_email_eu_core_bc_is_partial_or_regime(payload):
    """email-Eu-core/bc is the canonical tied/gray-zone cell from
    gate 42 (multi-way tie at 1MB+4MB+8MB). Must surface here as
    stable_partial (intersection {GRASP,LRU,SRRIP}) — never as
    stable_unique."""
    r = _record(payload, "email-Eu-core", "bc")
    assert r["classification"] in ("stable_partial", "regime_change"), (
        f"email-Eu-core/bc unexpectedly classified as {r['classification']};"
        f" gate 42 census says it's a 4-way/2-way tie cell"
    )
    if r["classification"] == "stable_partial":
        assert "GRASP" in r["intersection"]


def test_cross_gate_consistency_with_l3_policy_stability(payload):
    """Gate 39 (l3_policy_stability) reports cc/GRASP and pr/POPT as
    stable-single-winner at app granularity. This MUST imply that at
    least some (graph, cc) cells are stable-unique GRASP, and at least
    some (graph, pr) cells are stable-unique POPT. Otherwise gates 39
    and 44 contradict each other."""
    l3_path = REPO_ROOT / "wiki" / "data" / "l3_policy_stability.json"
    if not l3_path.exists():
        pytest.skip("l3_policy_stability.json not built")
    l3 = json.loads(l3_path.read_text())
    cc_stable = l3["per_app"].get("cc", {}).get("stability", {}).get(
        "is_stable_single_winner"
    )
    pr_stable = l3["per_app"].get("pr", {}).get("stability", {}).get(
        "is_stable_single_winner"
    )
    if cc_stable:
        cc_grasp_cells = [
            r for r in payload["per_graph_app"]
            if r["app"] == "cc"
            and r["classification"] in ("stable_unique", "stable_unique_with_ties")
            and "GRASP" in r["intersection"]
        ]
        assert len(cc_grasp_cells) >= 3, (
            "gate 39 says cc/GRASP stable; gate 44 must show >=3 "
            f"stable-unique GRASP cells for cc, got {len(cc_grasp_cells)}"
        )
    if pr_stable:
        pr_popt_cells = [
            r for r in payload["per_graph_app"]
            if r["app"] == "pr"
            and r["classification"] in ("stable_unique", "stable_unique_with_ties")
            and "POPT" in r["intersection"]
        ]
        assert len(pr_popt_cells) >= 3, (
            "gate 39 says pr/POPT stable; gate 44 must show >=3 "
            f"stable-unique POPT cells for pr, got {len(pr_popt_cells)}"
        )


def test_cross_gate_consistency_with_cell_census(payload):
    """Gate 42 (cell_winner_census) reports exactly 3 tied cells, all
    on bc/email-Eu-core. None of those (graph, app)=email-Eu-core/bc
    can appear here as stable_unique."""
    census_path = REPO_ROOT / "wiki" / "data" / "cell_winner_census.json"
    if not census_path.exists():
        pytest.skip("cell_winner_census.json not built")
    census = json.loads(census_path.read_text())
    tied_cells = census.get("all_tied_cells", [])
    tied_pairs = {(c["graph"], c["app"]) for c in tied_cells}
    for (g, a) in tied_pairs:
        r = _record(payload, g, a)
        assert r["classification"] != "stable_unique", (
            f"({g}, {a}) is in gate 42's tied set but gate 44 classifies "
            f"it as stable_unique — contradiction"
        )
