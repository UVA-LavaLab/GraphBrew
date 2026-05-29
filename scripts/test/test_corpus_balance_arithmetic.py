"""Gate 127 — corpus_balance.json arithmetic + diversity metrics.

Pins the actual corpus composition (graphs per family, paper-L3 cells
per family and per app), computes Shannon entropy + Simpson's index
diversity metrics, identifies the dominant family, and emits honest
tier disclosures. This gate reproduces every counter and every
diversity metric from the upstream oracle_gap rows so any corpus
re-shuffling is detected immediately.

Source: wiki/data/oracle_gap.json (every row carries family/graph/
app/l3_size). Paper L3 scope = {1MB, 4MB, 8MB}.

Derivations:
    n_graphs_per_family   : |{r.graph for r in rows if r.family==f}|
    fam_l3_cells[(f,l3)]  : count of paper-L3 rows with that family+l3
    app_l3_cells[(a,l3)]  : count of paper-L3 rows with that app+l3
    Shannon H (bits)      : -sum(p*log2(p)) over family/app counts
    Simpson's D           : 1 - sum(p^2)
    evenness              : H / log2(N)
    dominance fractions   : top family count / total
    families_capped_below_(4|8)MB : families whose L3 coverage misses
        the level (graphs land in 'over' regime before that L3 size)

Invariants (16 tests, 4 groups):
- meta + scope (3): source ends with oracle_gap.json,
  scope_l3_sizes=(1MB,4MB,8MB), n_apps/n_families/n_graphs/n_paper_l3
  match recomputed counts from rows.
- per_family + per_app counts from source (5): graphs lists are
  sorted unique per family; n_graphs and n_paper_l3_cells reproduce
  exactly; per_family_per_l3_cells matrix recomputed per (family,L3);
  per_app_per_l3_cells matrix recomputed per (app,L3); paper_l3_sizes_
  reached + reaches_4mb / reaches_8mb flags match L3 coverage.
- diversity metrics (4): Shannon H per family and per app, evenness
  ratios, Simpson's D recomputed from counter values; rounded to 4dp.
- dominance + honest disclosures (4): dominant family argmax both by
  graph count and by cells; fractions correct; families_capped_below_*
  sorted lists; note string contains 'over regime'.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/corpus_balance.json")
SOURCE = Path("wiki/data/oracle_gap.json")

PAPER_L3 = ("1MB", "4MB", "8MB")
TOL = 5e-4


def _shannon(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n == 0:
            continue
        p = n / total
        h -= p * math.log2(p)
    return h


def _simpson(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((n / total) ** 2 for n in counts.values())


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists()
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def src_rows():
    assert SOURCE.exists()
    return json.loads(SOURCE.read_text())["rows"]


@pytest.fixture(scope="module")
def derived(src_rows):
    g2f = {r["graph"]: r["family"] for r in src_rows if r.get("family")}
    fam_graphs = defaultdict(set)
    for g, f in g2f.items():
        fam_graphs[f].add(g)
    fam_l3 = defaultdict(int)
    app_l3 = defaultdict(int)
    for r in src_rows:
        if r["l3_size"] in PAPER_L3:
            fam_l3[(r["family"], r["l3_size"])] += 1
            app_l3[(r["app"], r["l3_size"])] += 1
    n_graphs_per_family = {f: len(gs) for f, gs in fam_graphs.items()}
    n_cells_per_family = {
        f: sum(c for (ff, _), c in fam_l3.items() if ff == f) for f in fam_graphs
    }
    n_cells_per_app = {
        a: sum(c for (aa, _), c in app_l3.items() if aa == a)
        for a in {r["app"] for r in src_rows}
    }
    fam_l3_coverage = defaultdict(list)
    for (f, l3) in fam_l3:
        fam_l3_coverage[f].append(l3)
    fam_l3_coverage = {k: sorted(set(v)) for k, v in fam_l3_coverage.items()}
    return {
        "fam_graphs": {f: sorted(gs) for f, gs in fam_graphs.items()},
        "n_graphs_per_family": n_graphs_per_family,
        "n_cells_per_family": n_cells_per_family,
        "n_cells_per_app": n_cells_per_app,
        "fam_l3": dict(fam_l3),
        "app_l3": dict(app_l3),
        "fam_l3_coverage": fam_l3_coverage,
    }


# ── group 1: meta + scope ────────────────────────────────────────────────


def test_meta_source_and_scope(data):
    assert data["meta"]["source"].endswith("oracle_gap.json")
    assert tuple(data["meta"]["scope_l3_sizes"]) == PAPER_L3


def test_meta_n_apps_families_graphs(data, derived):
    m = data["meta"]
    assert m["n_apps"] == len(derived["n_cells_per_app"])
    assert m["n_families"] == len(derived["fam_graphs"])
    assert m["n_graphs"] == sum(derived["n_graphs_per_family"].values())


def test_n_paper_l3_cells_total(data, derived):
    assert data["meta"]["n_paper_l3_cells_total"] == sum(derived["n_cells_per_family"].values())


# ── group 2: per_family + per_app counts ─────────────────────────────────


def test_per_family_graphs_sorted_unique(data, derived):
    for f, entry in data["per_family"].items():
        assert entry["graphs"] == derived["fam_graphs"][f], f
        assert entry["n_graphs"] == len(derived["fam_graphs"][f]), f


def test_per_family_n_paper_l3_cells(data, derived):
    for f, entry in data["per_family"].items():
        assert entry["n_paper_l3_cells"] == derived["n_cells_per_family"][f], f


def test_per_family_per_l3_matrix(data, derived):
    matrix = data["per_family_per_l3_cells"]
    for f, sub in matrix.items():
        for l3 in PAPER_L3:
            assert sub[l3] == derived["fam_l3"].get((f, l3), 0), f"{f}/{l3}"


def test_per_app_n_and_per_l3_matrix(data, derived):
    for a, entry in data["per_app"].items():
        assert entry["n_paper_l3_cells"] == derived["n_cells_per_app"][a], a
    matrix = data["per_app_per_l3_cells"]
    for a, sub in matrix.items():
        for l3 in PAPER_L3:
            assert sub[l3] == derived["app_l3"].get((a, l3), 0), f"{a}/{l3}"


def test_paper_l3_sizes_reached_and_flags(data, derived):
    for f, entry in data["per_family"].items():
        cov = derived["fam_l3_coverage"].get(f, [])
        assert entry["paper_l3_sizes_reached"] == cov, f
        assert entry["reaches_4mb"] is ("4MB" in cov), f
        assert entry["reaches_8mb"] is ("8MB" in cov), f


# ── group 3: diversity metrics ───────────────────────────────────────────


def test_shannon_h_per_family(data, derived):
    m = data["meta"]
    expected_h = _shannon(derived["n_graphs_per_family"])
    assert math.isclose(
        m["shannon_entropy_graphs_per_family_bits"], round(expected_h, 4), abs_tol=TOL
    )
    expected_max = math.log2(len(derived["n_graphs_per_family"]))
    assert math.isclose(
        m["shannon_entropy_graphs_per_family_max_bits"], round(expected_max, 4), abs_tol=TOL
    )


def test_evenness_per_family(data, derived):
    m = data["meta"]
    h = _shannon(derived["n_graphs_per_family"])
    h_max = math.log2(len(derived["n_graphs_per_family"]))
    expected = h / h_max if h_max > 0 else 0.0
    assert math.isclose(m["evenness_graphs_per_family"], round(expected, 4), abs_tol=TOL)


def test_simpson_diversity_metrics(data, derived):
    m = data["meta"]
    expected_fam = _simpson(derived["n_graphs_per_family"])
    expected_app = _simpson(derived["n_cells_per_app"])
    assert math.isclose(
        m["simpsons_diversity_graphs_per_family"], round(expected_fam, 4), abs_tol=TOL
    )
    assert math.isclose(
        m["simpsons_diversity_cells_per_app"], round(expected_app, 4), abs_tol=TOL
    )


def test_shannon_h_and_evenness_per_app(data, derived):
    m = data["meta"]
    h = _shannon(derived["n_cells_per_app"])
    h_max = math.log2(len(derived["n_cells_per_app"]))
    assert math.isclose(
        m["shannon_entropy_cells_per_app_bits"], round(h, 4), abs_tol=TOL
    )
    expected_even = h / h_max if h_max > 0 else 0.0
    assert math.isclose(m["evenness_cells_per_app"], round(expected_even, 4), abs_tol=TOL)


# ── group 4: dominance + honest disclosures ──────────────────────────────


def test_dominant_family_by_graph_count(data, derived):
    dom = data["dominance"]
    expected_family, expected_n = max(
        derived["n_graphs_per_family"].items(), key=lambda kv: kv[1]
    )
    assert dom["dominant_family_by_graph_count"] == expected_family
    assert dom["dominant_family_graph_count"] == expected_n
    total = sum(derived["n_graphs_per_family"].values())
    assert math.isclose(
        dom["dominant_family_graph_fraction"], round(expected_n / total, 4), abs_tol=TOL
    )


def test_dominant_family_by_paper_l3_cells(data, derived):
    dom = data["dominance"]
    expected_family, expected_n = max(
        derived["n_cells_per_family"].items(), key=lambda kv: kv[1]
    )
    assert dom["dominant_family_by_paper_l3_cells"] == expected_family
    assert dom["dominant_family_paper_l3_cell_count"] == expected_n
    total = sum(derived["n_cells_per_family"].values())
    assert math.isclose(
        dom["dominant_family_paper_l3_cell_fraction"],
        round(expected_n / total, 4),
        abs_tol=TOL,
    )


def test_honest_disclosure_lists(data, derived):
    hd = data["honest_disclosures"]
    cov = derived["fam_l3_coverage"]
    expected_capped_4 = sorted(f for f, l3s in cov.items() if "4MB" not in l3s)
    expected_capped_8 = sorted(f for f, l3s in cov.items() if "8MB" not in l3s)
    expected_reaching_8 = sorted(f for f, l3s in cov.items() if "8MB" in l3s)
    assert hd["families_capped_below_4MB"] == expected_capped_4
    assert hd["families_capped_below_8MB"] == expected_capped_8
    assert hd["families_reaching_8MB"] == expected_reaching_8


def test_honest_disclosure_note(data):
    note = data["honest_disclosures"]["note"]
    assert "over" in note.lower() and "regime" in note.lower()
