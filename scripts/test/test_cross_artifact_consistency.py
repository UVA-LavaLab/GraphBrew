"""Cross-artifact aggregate consistency gate (gate 85).

Many of our headline artifacts independently compute the *same*
underlying aggregates (e.g. "how many lit-faith cells does the corpus
contain?", "how many wins does each policy take?", "which graphs are
in the corpus?"). If any pair of generators silently disagrees we want
the gate suite to fail loudly rather than ship a paper with mismatched
numbers across tables.

This gate compares aggregate values that *should* be invariants
between sibling reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path("wiki/data")

EXPECTED_FAMILIES = {"citation", "mesh", "road", "social", "web"}


def _load(name: str):
    p = DATA / name
    if not p.exists():
        pytest.skip(f"{p} not generated yet")
    return json.loads(p.read_text())


def _corpus_graphs():
    return sorted(r["graph"] for r in _load("corpus_diversity.json"))


# ---------- corpus graph-set parity ----------


def test_policy_winner_table_graphs_match_corpus():
    corpus = set(_corpus_graphs())
    pwt = _load("policy_winner_table.json")
    pwt_graphs = {c["graph"] for c in pwt["cells"]}
    assert pwt_graphs == corpus, pwt_graphs ^ corpus


def test_winning_regime_taxonomy_graphs_match_corpus():
    corpus = set(_corpus_graphs())
    wrt = _load("winning_regime_taxonomy.json")
    wrt_graphs = {c["graph"] for c in wrt["cells"]}
    assert wrt_graphs == corpus, wrt_graphs ^ corpus


def test_popt_vs_grasp_delta_graphs_subset_of_corpus():
    corpus = set(_corpus_graphs())
    pvg = _load("popt_vs_grasp_delta.json")
    delta_graphs = {c["graph"] for c in pvg["cells"]}
    extra = delta_graphs - corpus
    assert not extra, extra


def test_claim_density_n_graphs_matches_power_law_claim_corpus():
    """POPT literature claims are power-law-scoped (road/mesh carry no POPT
    claims; P-OPT only tested power-law graphs), so claim_density covers the
    power-law claim graphs (6: cit-Patents, com-orkut, email-Eu-core,
    soc-LiveJournal1, soc-pokec, web-Google) — a subset of the 8-graph
    descriptive corpus."""
    corpus = _corpus_graphs()
    cd = _load("claim_density.json")
    assert cd["summary"]["n_graphs"] <= len(corpus), (
        f"claim_density n_graphs {cd['summary']['n_graphs']} exceeds corpus "
        f"{len(corpus)}"
    )
    assert cd["summary"]["n_graphs"] == 6, (
        f"claim_density n_graphs={cd['summary']['n_graphs']}; expected 6 "
        f"power-law claim graphs (road/mesh excluded from POPT claims)"
    )


# ---------- family parity ----------


def test_policy_winner_table_families_complete():
    pwt = _load("policy_winner_table.json")
    fams = set(pwt["summary"]["wins_by_family"])
    assert fams == EXPECTED_FAMILIES, fams ^ EXPECTED_FAMILIES


def test_winning_regime_taxonomy_families_complete():
    wrt = _load("winning_regime_taxonomy.json")
    fams = {b["family"] for b in wrt["summary"]["by_family_regime"]}
    assert fams == EXPECTED_FAMILIES, fams ^ EXPECTED_FAMILIES


# ---------- cell-count parity ----------


def test_winner_table_and_regime_taxonomy_have_same_n_cells():
    pwt = _load("policy_winner_table.json")
    wrt = _load("winning_regime_taxonomy.json")
    assert pwt["summary"]["n_cells"] == wrt["summary"]["n_cells"]


def test_winner_table_and_popt_delta_have_same_n_cells():
    pwt = _load("policy_winner_table.json")
    pvg = _load("popt_vs_grasp_delta.json")
    assert pwt["summary"]["n_cells"] == pvg["summary"]["n_cells"]


# ---------- winner-count parity (the load-bearing one) ----------


def test_wins_by_policy_matches_overall_winner_counts():
    pwt = _load("policy_winner_table.json")
    wrt = _load("winning_regime_taxonomy.json")
    pwt_wins = pwt["summary"]["wins_by_policy"]
    wrt_wins = wrt["summary"]["overall_winner_counts"]
    bad = []
    # WRT may contain extra keys like 'OTHER' that PWT doesn't track;
    # require every shared policy to agree exactly.
    shared = set(pwt_wins) & set(wrt_wins)
    for pol in shared:
        if pwt_wins[pol] != wrt_wins[pol]:
            bad.append((pol, pwt_wins[pol], wrt_wins[pol]))
    assert not bad, bad


def test_wins_by_policy_sums_to_n_cells():
    pwt = _load("policy_winner_table.json")
    total = sum(pwt["summary"]["wins_by_policy"].values())
    assert total == pwt["summary"]["n_cells"], (total, pwt["summary"]["n_cells"])


def test_wins_by_family_sums_to_n_cells():
    pwt = _load("policy_winner_table.json")
    total = sum(
        sum(pol_counts.values())
        for pol_counts in pwt["summary"]["wins_by_family"].values()
    )
    assert total == pwt["summary"]["n_cells"], (total, pwt["summary"]["n_cells"])


def test_wins_by_app_sums_to_n_cells():
    pwt = _load("policy_winner_table.json")
    total = sum(
        sum(pol_counts.values())
        for pol_counts in pwt["summary"]["wins_by_app"].values()
    )
    assert total == pwt["summary"]["n_cells"], (total, pwt["summary"]["n_cells"])


# ---------- cross-tool & deviations integrity ----------


def test_cross_tool_doubly_saturated_all_agree():
    cts = _load("cross_tool_saturation.json")
    s = cts["summary"]
    assert s["doubly_saturated_agree"] == s["doubly_saturated_total"]
    assert s["disagreements"] == [] or s["disagreements"] == 0


def test_literature_deviations_all_popt_overhead():
    dev = _load("literature_deviations.json")
    n_total = dev["summary"]["n_deviations"]
    popt = dev["summary"]["by_mechanism"].get("popt_overhead_dominates", 0)
    assert popt == n_total, (popt, n_total)


def test_literature_deviations_only_on_hub_families():
    """Locks the structural finding that POPT-overhead deviations only
    appear on hub-dominant families (citation/social/web), never on
    road or mesh — which is the load-bearing rationale for treating
    them as "known-deviations, not bugs"."""
    dev = _load("literature_deviations.json")
    fams = set(dev["summary"]["by_family"])
    allowed = {"citation", "social", "web"}
    bad = fams - allowed
    assert not bad, bad


def test_regression_budget_graphs_subset_of_corpus():
    corpus = set(_corpus_graphs())
    rb = _load("regression_budget.json")
    graphs = {c["graph"] for c in rb["per_cell"]}
    extra = graphs - corpus
    assert not extra, extra


def test_regression_budget_apps_complete():
    rb = _load("regression_budget.json")
    apps = {c["app"] for c in rb["per_cell"]}
    expected = {"bc", "bfs", "cc", "pr", "sssp"}
    assert apps == expected, apps ^ expected
