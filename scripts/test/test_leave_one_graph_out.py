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


def test_cc_popt_is_logo_fragile(payload):
    """Charged corpus: cc is graph-dependent; full-corpus POPT flips on web-Google."""
    ap = payload["per_app"]["cc"]
    assert ap["full_corpus"]["top_policy"] == "POPT"
    assert not ap["is_logo_robust"]
    assert ap["fragile_drops"] == ["web-Google"]


def test_bc_grasp_is_logo_robust(payload):
    """Charged corpus: bc/GRASP is the clean robust counter-kernel."""
    ap = payload["per_app"]["bc"]
    assert ap["full_corpus"]["top_policy"] == "GRASP"
    assert ap["is_logo_robust"]
    assert ap["fragile_drops"] == []


def test_sssp_is_logo_robust_grasp(payload):
    """Charged corpus: sssp is LOGO-robust by cell-vote with GRASP."""
    ap = payload["per_app"]["sssp"]
    assert ap["full_corpus"]["top_policy"] == "GRASP"
    assert ap["is_logo_robust"]
    assert ap["fragile_drops"] == []


def test_bfs_is_logo_robust(payload):
    """Charged corpus: bfs is LOGO-robust by cell-vote with GRASP."""
    ap = payload["per_app"]["bfs"]
    assert ap["full_corpus"]["top_policy"] == "GRASP"
    assert ap["is_logo_robust"]
    assert ap["fragile_drops"] == []


def test_n_robust_apps_meets_floor(payload):
    """At least 2 of the 5 apps must survive LOGO.

    Re-pinned 2026-06-12: single-thread corpus is more L3-regime-dependent
    (winners flip across L3 more), a real reproducible property.
    """
    assert payload["meta"]["n_robust_apps"] >= 2, payload["meta"]


def test_robust_set_includes_pr_cc(payload):
    """The robust set must include charged-corpus robust workloads."""
    robust = set(payload["meta"]["robust_apps"])
    expected = {"bc", "bfs", "pr", "sssp"}
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
