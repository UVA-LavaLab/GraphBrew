"""Gate 135 — winner-identification parity:
oracle_gap (is_winner) ↔ per_graph_app_stability (winners_by_l3,
intersection, union, classification, headline cell lists).

Continues the multi-artifact parity push started by gate 134. Where
gate 134 locks numeric agreement (miss_rate, delta_pct) across three
paper-table sources, this gate locks the *winner-identification*
chain: oracle_gap is_winner is the canonical per-cell winner flag,
per_graph_app_stability re-aggregates it into (graph, app)-level
stability classifications, and the same generator emits four
headline cell lists (stable_unique / regime_change / stable_partial
/ insufficient) consumed verbatim by the paper.

Drift between any of these layers would let the paper claim a graph
is "stable on POPT across all L3 sizes" while the per-cell winner
table shows POPT losing at 1MB.

Invariants (18 tests, 5 groups):

structural —
* every (graph, app, l3) cell in oracle_gap has at least 1 winner
* per_graph_app keys cover exactly the (graph, app) pairs present
  in oracle_gap

winners_by_l3 parity —
* for every (graph, app, l3) tuple: set(per_graph_app.winners_by_l3[l3])
  equals the set of oracle_gap.is_winner='1' policies for that cell
* per_graph_app.l3_sizes_present matches winners_by_l3.keys()
* winners_by_l3.keys() ⊆ canonical L3 list

intersection / union derivation —
* intersection == set-intersection of winners_by_l3 values
* union == set-union of winners_by_l3 values
* intersection ⊆ union
* unique_in_intersection == (len(intersection) == 1)

classification decision-tree —
* < 2 L3 sizes present                     ⇒ insufficient
* intersection empty AND union non-empty   ⇒ regime_change
* intersection == union == {p}             ⇒ stable_unique
* intersection == {p} and union has others ⇒ stable_unique_with_ties
* otherwise                                 ⇒ stable_partial

headline cell-list parity —
* stable_unique_cells list contains EXACTLY entries with
  classification ∈ {stable_unique, stable_unique_with_ties},
  formatted as "graph/app -> winner"
* regime_change_cells contains exactly classification==regime_change,
  formatted as "graph/app"
* stable_partial_cells contains exactly classification==stable_partial,
  formatted as "graph/app -> sorted(intersection or union)"
* insufficient_cells contains exactly classification==insufficient,
  formatted as "graph/app"
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
PGAS_PATH = REPO_ROOT / "wiki" / "data" / "per_graph_app_stability.json"

PAPER_L3 = frozenset({"1MB", "4MB", "8MB"})
TIE_TOL = 1e-6  # matches per_graph_app_stability generator winner detection


@pytest.fixture(scope="module")
def og_winners() -> dict:
    """{(graph, app, l3): set(winner policies)} for paper-scope cells.

    The per_graph_app_stability generator's `cell_winners` ties on
    rounded `gap_pp` (3-decimal precision in oracle_gap) within 1e-6
    tolerance, NOT on raw miss_rate. That's why distinct miss-rate
    values like 0.045849/0.045851/0.045853 all tie as winners — their
    gap_pp values all round to 0.000.

    Mirroring the generator semantics here is critical: using raw
    miss_rate would mis-identify winners on small-graph cells where
    every policy effectively ties.
    """
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    by_cell = defaultdict(list)
    for r in rows:
        if r["l3_size"] not in PAPER_L3:
            continue
        by_cell[(r["graph"], r["app"], r["l3_size"])].append(
            (float(r["gap_pp"]), r["policy"]))
    out = {}
    for cell, pairs in by_cell.items():
        m_min = min(g for g, _ in pairs)
        out[cell] = {p for g, p in pairs if abs(g - m_min) < TIE_TOL}
    return out


@pytest.fixture(scope="module")
def og_strict_winners() -> dict:
    """oracle_gap.is_winner strict-equality winner set per cell —
    kept for the subset-relationship invariant below."""
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    out = defaultdict(set)
    for r in rows:
        if r["l3_size"] not in PAPER_L3:
            continue
        if str(r.get("is_winner")) == "1":
            out[(r["graph"], r["app"], r["l3_size"])].add(r["policy"])
    return dict(out)


@pytest.fixture(scope="module")
def og_cells() -> set:
    """All (graph, app, l3) cells observed in oracle_gap, restricted
    to PAPER_L3 — matches the per_graph_app_stability scope."""
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    return {(r["graph"], r["app"], r["l3_size"]) for r in rows
            if r["l3_size"] in PAPER_L3}


@pytest.fixture(scope="module")
def pgas() -> dict:
    return json.loads(PGAS_PATH.read_text())


@pytest.fixture(scope="module")
def pgas_by_pair(pgas) -> dict:
    return {(p["graph"], p["app"]): p for p in pgas["per_graph_app"]}


# ---------- Group 1: structural ----------

def test_every_oracle_cell_has_at_least_one_winner(og_winners, og_cells):
    """Every (graph, app, l3) cell in oracle_gap must mark at least
    one policy as the winner — there's always a min miss rate."""
    cells_with_winners = set(og_winners.keys())
    missing = og_cells - cells_with_winners
    assert not missing, sorted(missing)[:3]


def test_pgas_pairs_cover_oracle_pairs(og_cells, pgas_by_pair):
    """per_graph_app must cover every (graph, app) pair in oracle_gap."""
    og_pairs = {(g, a) for (g, a, _) in og_cells}
    pgas_pairs = set(pgas_by_pair.keys())
    assert pgas_pairs == og_pairs, sorted(og_pairs - pgas_pairs)[:3]


def test_winners_by_l3_uses_canonical_l3_list(pgas_by_pair):
    bad = []
    for k, p in pgas_by_pair.items():
        for l3 in (p.get("winners_by_l3") or {}):
            if l3 not in PAPER_L3:
                bad.append((k, l3))
    assert not bad, bad[:3]


def test_pgas_classification_in_documented_set(pgas_by_pair):
    allowed = {"stable_unique", "stable_unique_with_ties",
               "regime_change", "stable_partial", "insufficient_l3"}
    bad = [(k, p["classification"]) for k, p in pgas_by_pair.items()
           if p["classification"] not in allowed]
    assert not bad, bad[:3]


# ---------- Group 2: winners_by_l3 parity ----------

def test_winners_by_l3_matches_oracle_is_winner(og_winners, pgas_by_pair):
    """For every (graph, app, l3) where per_graph_app has winners_by_l3,
    the set of winners must equal oracle_gap's is_winner='1' policies."""
    mism = []
    for (g, a), p in pgas_by_pair.items():
        for l3, winners in (p.get("winners_by_l3") or {}).items():
            og = og_winners.get((g, a, l3), set())
            if set(winners) != og:
                mism.append(((g, a, l3), set(winners), og))
    assert not mism, mism[:3]


def test_l3_sizes_present_matches_winners_keys(pgas_by_pair):
    mism = []
    for k, p in pgas_by_pair.items():
        present = set(p.get("l3_sizes_present") or [])
        wkeys = set((p.get("winners_by_l3") or {}).keys())
        if present != wkeys:
            mism.append((k, present, wkeys))
    assert not mism, mism[:3]


def test_oracle_cells_for_pair_match_l3_sizes_present(og_cells, pgas_by_pair):
    """l3_sizes_present == set of l3 in oracle_gap for that pair."""
    by_pair_l3 = defaultdict(set)
    for g, a, l3 in og_cells:
        by_pair_l3[(g, a)].add(l3)
    mism = []
    for (g, a), p in pgas_by_pair.items():
        present = set(p.get("l3_sizes_present") or [])
        expect = by_pair_l3[(g, a)]
        if present != expect:
            mism.append(((g, a), present, expect))
    assert not mism, mism[:3]


# ---------- Group 3: intersection / union derivation ----------

def test_oracle_strict_winners_subset_of_tie_tol_winners(og_strict_winners, og_winners):
    """oracle_gap.is_winner uses strict-equality with min miss_rate; the
    stability generator uses TIE_TOL=1e-6. So strict winners ⊆ TIE_TOL
    winners for every cell — the stability generator's broader set
    avoids spurious regime_change classifications caused by 1e-9 noise."""
    bad = []
    for cell, strict in og_strict_winners.items():
        tied = og_winners.get(cell, set())
        if not strict.issubset(tied):
            bad.append((cell, strict, tied))
    assert not bad, bad[:3]


def test_intersection_is_set_intersection_of_winners(pgas_by_pair):
    """For non-insufficient_l3 cases, intersection == set-intersection of
    winners_by_l3 values. insufficient_l3 cases (< 2 L3 sizes) have
    intersection explicitly suppressed to [] (no stability claim
    possible from a single observation)."""
    mism = []
    for k, p in pgas_by_pair.items():
        if p["classification"] == "insufficient_l3":
            if (p.get("intersection") or []) != []:
                mism.append((k, "insufficient_l3 must have empty intersection",
                             p.get("intersection")))
            continue
        wb = p.get("winners_by_l3") or {}
        if not wb:
            continue
        sets = [set(v) for v in wb.values()]
        want = sorted(set.intersection(*sets)) if sets else []
        got = sorted(p.get("intersection") or [])
        if got != want:
            mism.append((k, got, want))
    assert not mism, mism[:3]


def test_union_is_set_union_of_winners(pgas_by_pair):
    mism = []
    for k, p in pgas_by_pair.items():
        wb = p.get("winners_by_l3") or {}
        if not wb:
            continue
        sets = [set(v) for v in wb.values()]
        want = sorted(set.union(*sets)) if sets else []
        got = sorted(p.get("union") or [])
        if got != want:
            mism.append((k, got, want))
    assert not mism, mism[:3]


def test_intersection_subset_of_union(pgas_by_pair):
    bad = []
    for k, p in pgas_by_pair.items():
        i = set(p.get("intersection") or [])
        u = set(p.get("union") or [])
        if not i.issubset(u):
            bad.append((k, i, u))
    assert not bad, bad[:3]


def test_unique_in_intersection_flag(pgas_by_pair):
    mism = []
    for k, p in pgas_by_pair.items():
        want = len(p.get("intersection") or []) == 1
        got = bool(p.get("unique_in_intersection"))
        if got != want:
            mism.append((k, got, want))
    assert not mism, mism[:3]


# ---------- Group 4: classification decision-tree ----------

def _classify(p) -> str:
    """Reimplements the per_graph_app_stability decision tree (lines
    56-93 of scripts/experiments/ecg/per_graph_app_stability.py)."""
    l3_count = len(p.get("l3_sizes_present") or [])
    if l3_count < 2:
        return "insufficient_l3"
    inter = set(p.get("intersection") or [])
    union = set(p.get("union") or [])
    if not inter:
        return "regime_change"
    if len(inter) == 1 and union == inter:
        return "stable_unique"
    if len(inter) == 1:
        return "stable_unique_with_ties"
    return "stable_partial"


def test_classification_decision_tree_reproduces(pgas_by_pair):
    mism = []
    for k, p in pgas_by_pair.items():
        want = _classify(p)
        got = p["classification"]
        if got != want:
            mism.append((k, got, want, p.get("intersection"), p.get("union"),
                         p.get("l3_sizes_present")))
    assert not mism, mism[:3]


def test_regime_change_means_empty_intersection(pgas_by_pair):
    bad = []
    for k, p in pgas_by_pair.items():
        if p["classification"] == "regime_change":
            if p.get("intersection"):
                bad.append((k, p["intersection"]))
    assert not bad, bad[:3]


def test_stable_unique_means_singleton_intersection(pgas_by_pair):
    """Both stable_unique and stable_unique_with_ties imply singleton
    intersection (only one policy is the shared winner across all L3
    sizes); the difference is whether other policies tie at any L3."""
    bad = []
    for k, p in pgas_by_pair.items():
        if p["classification"] in ("stable_unique", "stable_unique_with_ties"):
            i = sorted(p.get("intersection") or [])
            if len(i) != 1:
                bad.append((k, p["classification"], i))
    assert not bad, bad[:3]


def test_stable_partial_means_multi_intersection(pgas_by_pair):
    """stable_partial == |intersection| >= 2 (multiple shared winners)."""
    bad = []
    for k, p in pgas_by_pair.items():
        if p["classification"] == "stable_partial":
            if len(p.get("intersection") or []) < 2:
                bad.append((k, p.get("intersection")))
    assert not bad, bad[:3]


def test_insufficient_l3_means_few_sizes(pgas_by_pair):
    bad = []
    for k, p in pgas_by_pair.items():
        if p["classification"] == "insufficient_l3":
            if len(p.get("l3_sizes_present") or []) >= 2:
                bad.append((k, p.get("l3_sizes_present")))
    assert not bad, bad[:3]


# ---------- Group 5: headline cell-list parity ----------

def test_stable_unique_cells_match(pgas, pgas_by_pair):
    """stable_unique_cells lists both stable_unique AND
    stable_unique_with_ties (the generator merges them into one
    headline list per source line 120), formatted "g/a -> winner"."""
    want = sorted(
        f"{p['graph']}/{p['app']} -> {sorted(p['intersection'])[0]}"
        for p in pgas_by_pair.values()
        if p["classification"] in ("stable_unique", "stable_unique_with_ties")
    )
    got = sorted(pgas.get("stable_unique_cells") or [])
    assert got == want, (set(want) ^ set(got))


def test_regime_change_cells_match(pgas, pgas_by_pair):
    want = sorted(
        f"{p['graph']}/{p['app']}"
        for p in pgas_by_pair.values()
        if p["classification"] == "regime_change"
    )
    got = sorted(pgas.get("regime_change_cells") or [])
    assert got == want, (set(want) ^ set(got))


def test_insufficient_cells_match(pgas, pgas_by_pair):
    want = sorted(
        f"{p['graph']}/{p['app']}"
        for p in pgas_by_pair.values()
        if p["classification"] == "insufficient_l3"
    )
    got = sorted(pgas.get("insufficient_cells") or [])
    assert got == want, (set(want) ^ set(got))


def test_stable_partial_cells_match(pgas, pgas_by_pair):
    """stable_partial_cells uses intersection (shared winners across all
    L3 sizes), not union. Format: "g/a -> sorted(intersection)"."""
    want = sorted(
        f"{p['graph']}/{p['app']} -> {','.join(sorted(p.get('intersection') or []))}"
        for p in pgas_by_pair.values()
        if p["classification"] == "stable_partial"
    )
    got = sorted(pgas.get("stable_partial_cells") or [])
    assert got == want, (set(want) ^ set(got))


def test_meta_counts_match_classifications(pgas, pgas_by_pair):
    """meta.n_* counts roll up from per_graph_app classifications.
    n_stable_unique merges stable_unique + stable_unique_with_ties
    (per generator source lines 162-163)."""
    from collections import Counter
    counts = Counter(p["classification"] for p in pgas_by_pair.values())
    meta = pgas["meta"]
    assert meta["n_graph_app_pairs"] == len(pgas_by_pair)
    assert meta["n_stable_unique"] == (counts.get("stable_unique", 0)
                                       + counts.get("stable_unique_with_ties", 0))
    assert meta["n_regime_change"] == counts.get("regime_change", 0)
    assert meta["n_stable_partial"] == counts.get("stable_partial", 0)
    assert meta["n_insufficient_l3"] == counts.get("insufficient_l3", 0)
