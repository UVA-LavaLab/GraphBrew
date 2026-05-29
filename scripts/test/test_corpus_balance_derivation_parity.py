"""Gate 184 — corpus_balance derivation parity.

Reconstruct ``wiki/data/corpus_balance.json`` from scratch by walking
``wiki/data/oracle_gap.json#rows`` and re-deriving every composition + balance
metric. Pin Shannon entropy (bits), Pielou evenness, Simpson's diversity,
the dominance disclosures, and the per-(family,L3) / per-(app,L3) coverage
matrices against the published artifact.

Load-bearing rules being locked:

* PAPER_L3_SIZES = ("1MB", "4MB", "8MB") — the only L3 sizes counted in
  paper_l3_cells; other sizes are excluded by-design.
* graph→family map built by iterating rows; later rows overwrite earlier
  (dict-set assignment), but in practice each graph has one consistent
  family so the map is unambiguous.
* fam_l3_cells counts ROWS (not unique graphs) — multiple-policy rows
  multiply the cell count. With 4 policies × 5 apps × 8 graphs × 3 paper L3
  sizes = up to 480 cells; missing-policy or missing-cell rows lower this.
* Shannon H computed in BITS (log2, not natural log).
* Max Shannon H = log2(n_families) — used for evenness denominator only.
* Pielou evenness = H / log2(K); returns 0 when K ≤ 1.
* Simpson's D = 1 - sum(p_i²).
* All meta floats rounded to 4dp.
* Dominance: by_graph_count uses MAX of n_graphs per family;
  by_paper_l3_cells uses MAX of n_cells per family; argmax tie-break is
  whatever Python's max() picks (first encountered = dict iteration order).
* per_family / per_app / per_family_per_l3_cells / per_app_per_l3_cells are
  emitted with sorted keys; per_family.graphs sorted alphabetically.
* honest_disclosures.families_capped_below_4MB := sorted list of families
  whose paper_l3_sizes_reached lacks "4MB" (same for 8MB);
  families_reaching_8MB := sorted list with "8MB" present.
* paper_l3_sizes_reached is set-then-sorted (dedup); per-cell zero counts
  do NOT count as "reached" because fam_l3_cells only adds keys for
  rows that exist.

The whole gate runs offline against committed JSON.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
ARTIFACT = WIKI_DATA / "corpus_balance.json"
ORACLE = WIKI_DATA / "oracle_gap.json"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")


# ---------------------------------------------------------------------------
# Reference rebuilders
# ---------------------------------------------------------------------------


def _shannon_bits(counts: dict) -> float:
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


def _simpson(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((n / total) ** 2 for n in counts.values())


def _rederive(rows: list[dict]) -> dict:
    g2f: dict[str, str] = {}
    for r in rows:
        if r.get("family"):
            g2f[r["graph"]] = r["family"]

    fam_graphs: dict[str, set] = defaultdict(set)
    for g, f in g2f.items():
        fam_graphs[f].add(g)

    fam_l3_cells: dict[tuple, int] = defaultdict(int)
    for r in rows:
        if r["l3_size"] in PAPER_L3_SIZES:
            fam_l3_cells[(r.get("family"), r["l3_size"])] += 1

    app_l3_cells: dict[tuple, int] = defaultdict(int)
    for r in rows:
        if r["l3_size"] in PAPER_L3_SIZES:
            app_l3_cells[(r["app"], r["l3_size"])] += 1

    fam_l3_cov: dict[str, list] = defaultdict(list)
    for (fam, l3) in fam_l3_cells:
        fam_l3_cov[fam].append(l3)
    fam_l3_cov = {k: sorted(set(v)) for k, v in fam_l3_cov.items()}

    n_graphs_per_family = {f: len(gs) for f, gs in fam_graphs.items()}
    n_cells_per_family = {
        f: sum(c for (ff, _), c in fam_l3_cells.items() if ff == f)
        for f in fam_graphs
    }
    n_cells_per_app = {
        a: sum(c for (aa, _), c in app_l3_cells.items() if aa == a)
        for a in {r["app"] for r in rows}
    }

    return {
        "g2f": g2f,
        "fam_graphs": fam_graphs,
        "fam_l3_cells": fam_l3_cells,
        "app_l3_cells": app_l3_cells,
        "fam_l3_cov": fam_l3_cov,
        "n_graphs_per_family": n_graphs_per_family,
        "n_cells_per_family": n_cells_per_family,
        "n_cells_per_app": n_cells_per_app,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oracle_rows() -> list[dict]:
    return json.loads(ORACLE.read_text())["rows"]


@pytest.fixture(scope="module")
def rebuild(oracle_rows) -> dict:
    return _rederive(oracle_rows)


# ---------------------------------------------------------------------------
# Group 1 — Schema / top-level keys
# ---------------------------------------------------------------------------


def test_top_keys(published):
    assert set(published.keys()) == {
        "meta",
        "dominance",
        "per_family",
        "per_app",
        "per_family_per_l3_cells",
        "per_app_per_l3_cells",
        "honest_disclosures",
    }


def test_meta_field_set(published):
    assert set(published["meta"].keys()) == {
        "source",
        "scope_l3_sizes",
        "n_graphs",
        "n_families",
        "n_apps",
        "n_paper_l3_cells_total",
        "shannon_entropy_graphs_per_family_bits",
        "shannon_entropy_graphs_per_family_max_bits",
        "evenness_graphs_per_family",
        "simpsons_diversity_graphs_per_family",
        "shannon_entropy_cells_per_app_bits",
        "evenness_cells_per_app",
        "simpsons_diversity_cells_per_app",
    }


def test_dominance_field_set(published):
    assert set(published["dominance"].keys()) == {
        "dominant_family_by_graph_count",
        "dominant_family_graph_count",
        "dominant_family_graph_fraction",
        "dominant_family_by_paper_l3_cells",
        "dominant_family_paper_l3_cell_count",
        "dominant_family_paper_l3_cell_fraction",
    }


def test_honest_disclosures_field_set(published):
    assert set(published["honest_disclosures"].keys()) == {
        "families_capped_below_4MB",
        "families_capped_below_8MB",
        "families_reaching_8MB",
        "note",
    }


def test_per_family_entry_field_set(published):
    expected = {
        "graphs",
        "n_graphs",
        "n_paper_l3_cells",
        "paper_l3_sizes_reached",
        "reaches_4mb",
        "reaches_8mb",
    }
    for fam, e in published["per_family"].items():
        assert set(e.keys()) == expected, fam


# ---------------------------------------------------------------------------
# Group 2 — Meta counters
# ---------------------------------------------------------------------------


def test_meta_source(published):
    assert published["meta"]["source"] == "wiki/data/oracle_gap.json"


def test_meta_scope_l3_sizes(published):
    assert published["meta"]["scope_l3_sizes"] == list(PAPER_L3_SIZES)


def test_meta_n_graphs_matches_unique_graphs(published, rebuild):
    assert published["meta"]["n_graphs"] == len(rebuild["g2f"])


def test_meta_n_families_matches_unique_families(published, rebuild):
    assert published["meta"]["n_families"] == len(rebuild["fam_graphs"])


def test_meta_n_apps_matches_unique_apps(published, rebuild):
    assert published["meta"]["n_apps"] == len(rebuild["n_cells_per_app"])


def test_meta_n_paper_l3_cells_total_matches_sum(published, rebuild):
    total = sum(rebuild["n_cells_per_family"].values())
    assert published["meta"]["n_paper_l3_cells_total"] == total


def test_meta_n_paper_l3_cells_total_equals_app_sum(published, rebuild):
    # Same total should be reachable from the per-app view.
    total = sum(rebuild["n_cells_per_app"].values())
    assert published["meta"]["n_paper_l3_cells_total"] == total


# ---------------------------------------------------------------------------
# Group 3 — Diversity / evenness metrics
# ---------------------------------------------------------------------------


def test_shannon_entropy_graphs_per_family(published, rebuild):
    expected = round(_shannon_bits(rebuild["n_graphs_per_family"]), 4)
    assert published["meta"]["shannon_entropy_graphs_per_family_bits"] == expected


def test_shannon_max_is_log2_n_families(published, rebuild):
    expected = round(
        math.log2(len(rebuild["n_graphs_per_family"]))
        if rebuild["n_graphs_per_family"]
        else 0.0,
        4,
    )
    assert published["meta"]["shannon_entropy_graphs_per_family_max_bits"] == expected


def test_evenness_graphs_per_family(published, rebuild):
    h = _shannon_bits(rebuild["n_graphs_per_family"])
    hmax = (
        math.log2(len(rebuild["n_graphs_per_family"]))
        if rebuild["n_graphs_per_family"]
        else 0.0
    )
    expected = round(h / hmax if hmax > 0 else 0.0, 4)
    assert published["meta"]["evenness_graphs_per_family"] == expected


def test_simpsons_graphs_per_family(published, rebuild):
    expected = round(_simpson(rebuild["n_graphs_per_family"]), 4)
    assert published["meta"]["simpsons_diversity_graphs_per_family"] == expected


def test_shannon_entropy_cells_per_app(published, rebuild):
    expected = round(_shannon_bits(rebuild["n_cells_per_app"]), 4)
    assert published["meta"]["shannon_entropy_cells_per_app_bits"] == expected


def test_evenness_cells_per_app(published, rebuild):
    h = _shannon_bits(rebuild["n_cells_per_app"])
    hmax = (
        math.log2(len(rebuild["n_cells_per_app"]))
        if rebuild["n_cells_per_app"]
        else 0.0
    )
    expected = round(h / hmax if hmax > 0 else 0.0, 4)
    assert published["meta"]["evenness_cells_per_app"] == expected


def test_simpsons_cells_per_app(published, rebuild):
    expected = round(_simpson(rebuild["n_cells_per_app"]), 4)
    assert published["meta"]["simpsons_diversity_cells_per_app"] == expected


def test_evenness_between_0_and_1(published):
    for k in ("evenness_graphs_per_family", "evenness_cells_per_app"):
        v = published["meta"][k]
        assert 0.0 <= v <= 1.0, k


def test_simpson_between_0_and_1(published):
    for k in (
        "simpsons_diversity_graphs_per_family",
        "simpsons_diversity_cells_per_app",
    ):
        v = published["meta"][k]
        assert 0.0 <= v <= 1.0, k


# ---------------------------------------------------------------------------
# Group 4 — Dominance + per-family + per-app aggregation
# ---------------------------------------------------------------------------


def test_dominance_by_graph_count(published, rebuild):
    family, n = max(
        rebuild["n_graphs_per_family"].items(), key=lambda kv: kv[1]
    )
    total_graphs = sum(rebuild["n_graphs_per_family"].values())
    assert published["dominance"]["dominant_family_by_graph_count"] == family
    assert published["dominance"]["dominant_family_graph_count"] == n
    assert published["dominance"]["dominant_family_graph_fraction"] == round(
        n / total_graphs, 4
    )


def test_dominance_by_paper_l3_cells(published, rebuild):
    family, n = max(
        rebuild["n_cells_per_family"].items(), key=lambda kv: kv[1]
    )
    total = sum(rebuild["n_cells_per_family"].values())
    assert published["dominance"]["dominant_family_by_paper_l3_cells"] == family
    assert published["dominance"]["dominant_family_paper_l3_cell_count"] == n
    assert published["dominance"]["dominant_family_paper_l3_cell_fraction"] == round(
        n / total, 4
    )


def test_per_family_full_rederive(published, rebuild):
    expected = {
        f: {
            "graphs": sorted(rebuild["fam_graphs"][f]),
            "n_graphs": rebuild["n_graphs_per_family"][f],
            "n_paper_l3_cells": rebuild["n_cells_per_family"][f],
            "paper_l3_sizes_reached": rebuild["fam_l3_cov"][f],
            "reaches_4mb": "4MB" in rebuild["fam_l3_cov"][f],
            "reaches_8mb": "8MB" in rebuild["fam_l3_cov"][f],
        }
        for f in sorted(rebuild["fam_graphs"])
    }
    assert published["per_family"] == expected


def test_per_app_n_paper_l3_cells(published, rebuild):
    expected = {
        a: {"n_paper_l3_cells": rebuild["n_cells_per_app"][a]}
        for a in sorted(rebuild["n_cells_per_app"])
    }
    assert published["per_app"] == expected


def test_per_family_per_l3_cells_full_rederive(published, rebuild):
    expected = {
        f: {l3: rebuild["fam_l3_cells"].get((f, l3), 0) for l3 in PAPER_L3_SIZES}
        for f in sorted(rebuild["fam_graphs"])
    }
    assert published["per_family_per_l3_cells"] == expected


def test_per_app_per_l3_cells_full_rederive(published, rebuild):
    expected = {
        a: {l3: rebuild["app_l3_cells"].get((a, l3), 0) for l3 in PAPER_L3_SIZES}
        for a in sorted(rebuild["n_cells_per_app"])
    }
    assert published["per_app_per_l3_cells"] == expected


def test_per_family_per_l3_keys_are_all_three_paper_sizes(published):
    for fam, row in published["per_family_per_l3_cells"].items():
        assert set(row.keys()) == set(PAPER_L3_SIZES), fam


def test_per_family_sum_matches_row_total(published):
    for fam, row in published["per_family_per_l3_cells"].items():
        total = sum(row.values())
        assert published["per_family"][fam]["n_paper_l3_cells"] == total, fam


# ---------------------------------------------------------------------------
# Group 5 — Honest disclosures (capped families)
# ---------------------------------------------------------------------------


def test_families_capped_below_4mb_sorted(published):
    capped = published["honest_disclosures"]["families_capped_below_4MB"]
    assert capped == sorted(capped)
    for fam in capped:
        assert "4MB" not in published["per_family"][fam]["paper_l3_sizes_reached"], fam


def test_families_capped_below_8mb_sorted(published):
    capped = published["honest_disclosures"]["families_capped_below_8MB"]
    assert capped == sorted(capped)
    for fam in capped:
        assert "8MB" not in published["per_family"][fam]["paper_l3_sizes_reached"], fam


def test_families_reaching_8mb_sorted_and_consistent(published):
    reaching = published["honest_disclosures"]["families_reaching_8MB"]
    assert reaching == sorted(reaching)
    for fam in reaching:
        assert "8MB" in published["per_family"][fam]["paper_l3_sizes_reached"], fam


def test_4mb_capped_subset_of_8mb_capped(published):
    c4 = set(published["honest_disclosures"]["families_capped_below_4MB"])
    c8 = set(published["honest_disclosures"]["families_capped_below_8MB"])
    # Any family that doesn't reach 4MB also doesn't reach 8MB
    assert c4.issubset(c8), c4 - c8


def test_capped_and_reaching_partition_all_families(published):
    families = set(published["per_family"].keys())
    c8 = set(published["honest_disclosures"]["families_capped_below_8MB"])
    r8 = set(published["honest_disclosures"]["families_reaching_8MB"])
    assert c8 | r8 == families
    assert c8 & r8 == set()
