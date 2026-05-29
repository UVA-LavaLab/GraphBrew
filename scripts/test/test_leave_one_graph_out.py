"""Gate: Leave-one-graph-out (LOGO) winner robustness.

Pins which kernels' headline winners survive dropping any one graph.
Robust = no single graph is driving the headline. Fragile = the headline
changes if a particular graph is removed; this kernel's claim must
be explicitly qualified in the paper text.

The number of graphs in the corpus may grow over time; tests are
parametrized on the FRAGILE-set (which is what we care about) rather
than the exact n_drops count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "leave_one_graph_out.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(
            f"missing {PAYLOAD}; run `make lit-logo-robust` "
            f"(or `make lit-claims`)."
        )
    return json.loads(PAYLOAD.read_text())


def test_schema_complete(payload):
    assert "meta" in payload and "per_app" in payload
    for key in ("n_graphs", "graphs", "robust_apps", "fragile_apps",
                "n_robust_apps", "n_fragile_apps"):
        assert key in payload["meta"], f"meta missing {key}"
    for app in ("pr", "cc", "bfs", "sssp", "bc"):
        assert app in payload["per_app"], f"missing app {app}"
        ap = payload["per_app"][app]
        for key in ("full_corpus", "drops", "n_drops",
                    "n_robust_drops", "fragile_drops", "is_logo_robust"):
            assert key in ap, f"{app} missing {key}"


def test_minimum_graph_corpus_floor(payload):
    """The LOGO analysis is meaningful only with ≥ 5 graphs (otherwise
    every drop is too much of the corpus). Current corpus has 8; we
    floor at 6 to leave room to remove the smallest 2 if needed but
    catch the case where the corpus shrinks accidentally."""
    assert payload["meta"]["n_graphs"] >= 6, payload["meta"]


def test_pr_popt_is_logo_robust(payload):
    """pr/POPT is the marquee paper claim — MUST survive every LOGO drop."""
    ap = payload["per_app"]["pr"]
    assert ap["full_corpus"]["top_policy"] == "POPT"
    assert ap["is_logo_robust"], (
        f"pr/POPT no longer LOGO-robust — fragile drops: {ap['fragile_drops']}. "
        "Marquee paper claim at risk; STOP and investigate."
    )


def test_cc_grasp_is_logo_robust(payload):
    """cc/GRASP is the other marquee claim — must be LOGO-robust."""
    ap = payload["per_app"]["cc"]
    assert ap["full_corpus"]["top_policy"] == "GRASP"
    assert ap["is_logo_robust"], (
        f"cc/GRASP lost LOGO robustness; fragile drops: {ap['fragile_drops']}."
    )


def test_bc_grasp_is_logo_robust(payload):
    """bc/GRASP is the supporting claim for centrality/triangle workloads."""
    ap = payload["per_app"]["bc"]
    assert ap["full_corpus"]["top_policy"] == "GRASP"
    assert ap["is_logo_robust"], (
        f"bc/GRASP lost LOGO robustness; fragile drops: {ap['fragile_drops']}."
    )


def test_sssp_is_logo_fragile_honest(payload):
    """sssp is the honest negative pin: fragile under MULTIPLE LOGO drops.
    Triangulated with gate 36 (no Wilson CI-strict majority), gate 37
    (no large Cohen's h), gate 38 (no large Cliff's delta), and gate 39
    (no stable cross-L3 winner)."""
    ap = payload["per_app"]["sssp"]
    assert not ap["is_logo_robust"], (
        "sssp NEWLY LOGO-robust — cross-check gates 36/37/38/39 because "
        "this would contradict 4 independent prior signals."
    )
    # Floor: at least 2 drops must change the winner.
    assert len(ap["fragile_drops"]) >= 2, (
        f"sssp fragile_drops dropped below 2 ({ap['fragile_drops']}); "
        "this kernel was previously fragile under 3 drops. Investigate."
    )


def test_bfs_is_logo_fragile(payload):
    """bfs is fragile under exactly the drop of the large social graph,
    consistent with gate 39's regime-change finding (GRASP at small L3,
    POPT at production L3 — fragility around the regime boundary)."""
    ap = payload["per_app"]["bfs"]
    assert not ap["is_logo_robust"], (
        "bfs NEWLY LOGO-robust — but gate 39 pins it as a regime-change "
        "kernel. Re-investigate cross-gate consistency."
    )


def test_n_robust_apps_meets_floor(payload):
    """At least 3 of the 5 apps must survive LOGO. If only 2 or fewer
    are robust, the paper's headline 'these policies work across our
    corpus' is no longer defensible."""
    assert payload["meta"]["n_robust_apps"] >= 3, payload["meta"]


def test_robust_set_includes_pr_cc_bc(payload):
    """The robust set must include exactly the 'GRASP/POPT marquee'
    workloads. Loss of any would tank a headline claim."""
    robust = set(payload["meta"]["robust_apps"])
    expected = {"pr", "cc", "bc"}
    missing = expected - robust
    assert not missing, (
        f"Marquee LOGO-robust apps lost: {missing}. Surviving: {robust}."
    )


def test_per_drop_consistency_for_robust_apps(payload):
    """For every LOGO-robust app, every (app, drop) entry must have
    same_winner_as_full == True. If any single drop disagrees, the
    overall robust flag is lying."""
    for app, ap in payload["per_app"].items():
        if not ap["is_logo_robust"]:
            continue
        for g, d in ap["drops"].items():
            if d.get("missing"):
                continue
            assert d["same_winner_as_full"], (
                f"{app} flagged robust but drop={g} changed winner: {d}"
            )


def test_per_drop_winner_consistency_for_fragile_apps(payload):
    """A fragile app must have at least one drop with same_winner_as_full
    == False. Otherwise the fragile flag is lying."""
    for app, ap in payload["per_app"].items():
        if ap["is_logo_robust"]:
            continue
        any_changed = any(
            not d.get("same_winner_as_full", True)
            for d in ap["drops"].values()
            if not d.get("missing")
        )
        assert any_changed, (
            f"{app} flagged fragile but no drop actually changes winner; "
            f"drops={ap['drops']}"
        )


def test_full_top_policy_matches_first_drop_consistency(payload):
    """For every app, every drop's top_policy must be in the policy
    set {GRASP, POPT, LRU, SRRIP}. Catches typos/silent regressions."""
    valid = {"GRASP", "POPT", "LRU", "SRRIP"}
    for app, ap in payload["per_app"].items():
        top = ap["full_corpus"].get("top_policy")
        assert top in valid, f"{app} full top {top!r} not in {valid}"
        for g, d in ap["drops"].items():
            if d.get("missing"):
                continue
            assert d["top_policy"] in valid, (
                f"{app}/{g} top {d['top_policy']!r} not in {valid}"
            )
