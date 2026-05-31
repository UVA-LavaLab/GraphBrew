"""Gate 53: per-graph oracle-gap cache-sensitivity slope invariants.

Companion to gate 52 (corpus-averaged slope). Gate 52 found:
'significant anti-scaling at the corpus-averaged level is exclusively
LRU/SRRIP; GRASP and POPT never regress.'

This gate drills into the per-graph data and verifies that:
  - The headline LRU/SRRIP dominance of anti-scaling still holds
    (most regressions are LRU + SRRIP).
  - The small number of GRASP/POPT regressions at the per-graph level
    is pinned exactly, so the paper can disclose them transparently.
  - email-Eu-core (PR-pilot graph used to develop GRASP) has zero
    anti-scaling cells — the optimisation target is not regressing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "per_graph_cache_slope.json"


@pytest.fixture(scope="module")
def payload():
    if not PAYLOAD.exists():
        pytest.skip(f"missing {PAYLOAD}; run `make lit-per-graph-cache-slope`")
    return json.loads(PAYLOAD.read_text())


def test_meta_paper_l3_scope(payload):
    assert payload["meta"]["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]


def test_meta_minimum_full_trajectories(payload):
    """At least 100 full (graph, app, policy) trajectories at paper L3.

    Locks down the corpus size used in this gate so future graph-add
    or graph-drop runs surface immediately.
    """
    assert payload["meta"]["n_full_trajectories"] >= 100


def test_meta_all_four_policies_present(payload):
    assert set(payload["meta"]["policies"]) == {"LRU", "SRRIP", "GRASP", "POPT"}


def test_meta_paper_apps_present(payload):
    assert set(payload["meta"]["apps"]) >= {"bc", "bfs", "cc", "pr", "sssp"}


def test_oracle_aware_anti_scaling_is_minority(payload):
    """Even at the per-graph level, the bulk of regressions are LRU+SRRIP.

    GRASP+POPT combined should account for strictly less than half of
    all anti-scaling cells. (Currently 7 of 33 = 21 percent.)
    """
    per_pol = payload["per_policy_anti_scaling_count"]
    total = sum(per_pol.values())
    oracle_aware = per_pol.get("GRASP", 0) + per_pol.get("POPT", 0)
    assert total > 0
    assert oracle_aware * 2 < total, (
        f"oracle-aware anti-scaling cells {oracle_aware} should be a minority "
        f"of total anti-scaling cells {total}"
    )


def test_grasp_anti_scaling_cells_are_pinned(payload):
    """GRASP currently regresses on exactly 2 (graph, app) cells.

    Pinning the set so any future regression growth is caught by CI.
    Allowed-listed cells:
      - cit-Patents/pr  (gate-49 already flagged pr/GRASP as
        worst-of-class; not unexpected here)
      - web-Google/bfs  (modest, ~+2 pp single-octave growth)
    """
    grasp_cells = {
        (c["graph"], c["app"])
        for c in payload["anti_scaling_cells"]
        if c["policy"] == "GRASP"
    }
    expected = {("cit-Patents", "pr"), ("web-Google", "bfs")}
    assert grasp_cells == expected, (
        f"GRASP anti-scaling cells drift: {grasp_cells ^ expected}"
    )


def test_popt_anti_scaling_concentrates_in_bc_and_sssp(payload):
    """POPT regresses predominantly on bc and sssp (frontier-bound apps).

    Post cache_sim ECG sweep: POPT anti-scaling redistributed from
    bc-majority to bc(2)+sssp(2)+cc(1)+pr(1). Both bc and sssp are
    traversal-frontier apps where POPT's static rank-schedule
    mis-aligns with dynamic frontier order.
    """
    popt_cells = [
        c
        for c in payload["anti_scaling_cells"]
        if c["policy"] == "POPT"
    ]
    bc_sssp_count = sum(1 for c in popt_cells if c["app"] in ("bc", "sssp"))
    assert len(popt_cells) >= 3
    assert bc_sssp_count >= len(popt_cells) / 2, (
        f"expected bc+sssp to be majority of POPT anti-scaling cells; "
        f"got {bc_sssp_count}/{len(popt_cells)}"
    )


def test_email_eu_core_has_zero_anti_scaling(payload):
    """The PR-pilot graph used to develop GRASP must not regress on anything."""
    if "email-Eu-core" not in payload["meta"]["graphs"]:
        pytest.skip("email-Eu-core not in this run's full-trajectory graphs")
    assert payload["per_graph_anti_scaling_count"].get("email-Eu-core", 0) == 0


def test_lru_and_srrip_dominate_anti_scaling(payload):
    """LRU + SRRIP together account for at least 75% of anti-scaling cells."""
    per_pol = payload["per_policy_anti_scaling_count"]
    total = sum(per_pol.values())
    non_oracle = per_pol.get("LRU", 0) + per_pol.get("SRRIP", 0)
    assert non_oracle * 4 >= total * 3, (
        f"LRU+SRRIP={non_oracle} should be >=75% of total {total}"
    )


def test_worst_cell_is_web_google_bfs(payload):
    """Largest single-octave gap-growth cell pin.

    web-Google/bfs/LRU and web-Google/bfs/SRRIP both grow by ~14.7 pp
    in one octave — the most dramatic anti-scaling in the corpus.
    Useful narrative anchor for the paper.
    """
    if not payload["anti_scaling_cells"]:
        pytest.skip("no anti-scaling cells in current run")
    top = payload["anti_scaling_cells"][0]
    assert top["graph"] == "web-Google"
    assert top["app"] == "bfs"
    assert top["policy"] in {"LRU", "SRRIP"}
    assert top["max_pp_growth"] >= 10.0


def test_all_trajectories_have_two_octaves(payload):
    """Schema invariant — every trajectory record has exactly 2 octave slices."""
    for cell in payload["all_trajectories"]:
        assert len(cell["octaves"]) == 2
        assert {o["from"] for o in cell["octaves"]} == {"1MB", "4MB"}
        assert {o["to"] for o in cell["octaves"]} == {"4MB", "8MB"}


def test_anti_scaling_consistent_with_corpus_view(payload):
    """Cross-gate anchor against gate 52 (cache_sensitivity_slope).

    Gate 52 reports both 'monotonic_violations' (any non-monotonic
    trajectory, including noise) and significant anti-scaling cells.
    We only require corpus-averaged cells with a *significant*
    positive octave delta to appear at the per-graph level — tiny
    noise-floor violations are excluded.
    """
    sibling = REPO_ROOT / "wiki" / "data" / "cache_sensitivity_slope.json"
    if not sibling.exists():
        pytest.skip("gate 52 artifact missing; skipping cross-gate anchor")
    corpus = json.loads(sibling.read_text())
    threshold = payload["meta"]["significant_pp_threshold"]
    corpus_significant = {
        (v["app"], v["policy"])
        for v in corpus["monotonic_violations"]
        if any(o["delta_gap_pp"] >= threshold for o in v["octaves"])
    }
    per_graph_pairs = {
        (c["app"], c["policy"])
        for c in payload["anti_scaling_cells"]
    }
    assert corpus_significant, "expected gate 52 to flag at least one significant anti-scaling cell"
    for app, pol in corpus_significant:
        assert (app, pol) in per_graph_pairs, (
            f"corpus-averaged anti-scaling ({app}, {pol}) has no "
            f"per-graph cell — investigate sign or aggregation"
        )
