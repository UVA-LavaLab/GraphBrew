"""Gate 45: corpus tier/family balance audit.

Pins the actual composition of the GraphBrew corpus so reviewer claims
of 'your corpus is unbalanced toward X family' can be answered with
exact numbers and per-family L3 coverage.

Headline observations pinned:
  - 8 graphs across 5 families: social=4, citation=1, mesh=1, road=1, web=1
  - **social dominates by graph count (4/8 = 50.0%) AND by paper-L3 cells**
    (216/360 = 60.0%); the paper must surface per-family findings, not
    only aggregates, or social will dominate the average.
  - mesh and road do NOT reach 4MB/8MB paper-L3; reviewer comparisons at
    4MB+ exclude these families.
  - Pielou evenness across families ~0.86 (max 1.0 = uniform).

Cross-gate consistency:
  - Gates 35, 43, 44 break out per (family) precisely BECAUSE the corpus
    is social-dominated; this gate is the numeric anchor for that
    methodological choice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "wiki" / "data" / "corpus_balance.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} missing; run `make lit-corpus-balance`")
    return json.loads(DATA_PATH.read_text())


def test_meta_pins_corpus_size(payload):
    meta = payload["meta"]
    assert meta["n_graphs"] == 8, (
        f"corpus size changed: {meta['n_graphs']} graphs (was 8). "
        "Update gate floor explicitly with justification."
    )
    assert meta["n_families"] == 5
    assert meta["n_apps"] == 5
    assert meta["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]


def test_family_inventory_is_exact(payload):
    """The 5 families and their graphs must match exactly. Any change is
    a corpus event the paper must discuss."""
    expected = {
        "citation": ["cit-Patents"],
        "mesh": ["delaunay_n19"],
        "road": ["roadNet-CA"],
        "social": [
            "com-orkut",
            "email-Eu-core",
            "soc-LiveJournal1",
            "soc-pokec",
        ],
        "web": ["web-Google"],
    }
    for fam, expected_graphs in expected.items():
        assert fam in payload["per_family"], f"missing family {fam}"
        actual = payload["per_family"][fam]["graphs"]
        assert actual == sorted(expected_graphs), (
            f"family {fam} graphs changed: actual={actual} expected={expected_graphs}"
        )


def test_social_dominance_by_graph_count(payload):
    """Pin the social-family dominance disclosure: 4 of 8 graphs (50%)."""
    dom = payload["dominance"]
    assert dom["dominant_family_by_graph_count"] == "social"
    assert dom["dominant_family_graph_count"] == 4
    assert abs(dom["dominant_family_graph_fraction"] - 0.5) < 0.001


def test_social_dominance_by_paper_l3_cells(payload):
    """social also dominates by paper-L3 cells (3 L3 sizes × 4 graphs ×
    18 (policy, app) combinations roughly)."""
    dom = payload["dominance"]
    assert dom["dominant_family_by_paper_l3_cells"] == "social"
    assert dom["dominant_family_paper_l3_cell_fraction"] >= 0.50


def test_road_and_mesh_capped_below_4mb(payload):
    """road + mesh families MUST NOT reach 4MB paper-L3 (graphs cap out
    before 4MB). If a future corpus expansion adds 4MB road/mesh data,
    this test must be bumped with the new floor and the paper's per-L3
    cross-family comparison disclaimer rewritten."""
    capped = payload["honest_disclosures"]["families_capped_below_4MB"]
    assert "road" in capped
    assert "mesh" in capped


def test_citation_social_web_reach_8mb(payload):
    """The three families that span the full paper L3 axis. Removing any
    of these from 8MB would break gates 39/44 (cross-L3 stability)."""
    reaching = payload["honest_disclosures"]["families_reaching_8MB"]
    for fam in ("citation", "social", "web"):
        assert fam in reaching, (
            f"family {fam} no longer reaches 8MB; gates 39/44 are at risk"
        )


def test_diversity_metrics_above_floor(payload):
    """Pielou evenness across families must stay above 0.80 — below this
    the corpus is meaningfully unbalanced and the paper must add a
    'balanced subcorpus' supplementary analysis."""
    meta = payload["meta"]
    assert meta["evenness_graphs_per_family"] >= 0.80, (
        f"family evenness dropped to {meta['evenness_graphs_per_family']:.3f}; "
        f"corpus balance below floor"
    )
    # Simpson's D <= max for n=5 is 0.8; we expect >= 0.6 with current
    # composition (4 + 1 + 1 + 1 + 1).
    assert meta["simpsons_diversity_graphs_per_family"] >= 0.60


def test_app_cell_distribution_is_well_balanced(payload):
    """The 5 apps should each have similar paper-L3 cell counts (within
    ~25% of mean). If a corpus update silently drops sssp/cc data on a
    graph, this catches it."""
    per_app = payload["per_app"]
    counts = [r["n_paper_l3_cells"] for r in per_app.values()]
    mean = sum(counts) / len(counts)
    for app, r in per_app.items():
        ratio = r["n_paper_l3_cells"] / mean
        assert 0.75 <= ratio <= 1.30, (
            f"app {app} cell count {r['n_paper_l3_cells']} is "
            f"{ratio:.2f}x the mean ({mean:.1f}); coverage skew exceeds floor"
        )


def test_per_family_per_l3_cell_counts_sum_correctly(payload):
    """Internal consistency: per-family paper-L3 cell sums must equal the
    per-family n_paper_l3_cells aggregate."""
    for fam, by_l3 in payload["per_family_per_l3_cells"].items():
        total = sum(by_l3.values())
        agg = payload["per_family"][fam]["n_paper_l3_cells"]
        assert total == agg, (
            f"family {fam} cell counts inconsistent: per-L3 sum {total} "
            f"vs aggregate {agg}"
        )


def test_total_paper_l3_cells_match(payload):
    """Total cells aggregated from per-family vs per-app must agree."""
    fam_total = sum(
        r["n_paper_l3_cells"] for r in payload["per_family"].values()
    )
    app_total = sum(
        r["n_paper_l3_cells"] for r in payload["per_app"].values()
    )
    assert fam_total == app_total, (
        f"per-family total {fam_total} != per-app total {app_total}; "
        f"the rows must be the same set partitioned two ways"
    )
    assert fam_total == payload["meta"]["n_paper_l3_cells_total"]


def test_evenness_within_theoretical_bounds(payload):
    """Sanity bounds on the evenness/entropy reporting."""
    meta = payload["meta"]
    h = meta["shannon_entropy_graphs_per_family_bits"]
    h_max = meta["shannon_entropy_graphs_per_family_max_bits"]
    n_fams = meta["n_families"]
    assert 0.0 <= h <= h_max + 1e-6
    assert abs(h_max - math.log2(n_fams)) < 1e-3
    assert 0.0 <= meta["evenness_graphs_per_family"] <= 1.0 + 1e-6
    assert 0.0 <= meta["simpsons_diversity_graphs_per_family"] <= 1.0


def test_cross_gate_consistency_with_oracle_gap(payload):
    """Total paper-L3 cells reported here must match the row count we'd
    get filtering oracle_gap to paper L3 sizes (sanity that this gate
    is auditing the same canonical source other gates use)."""
    oracle_path = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
    if not oracle_path.exists():
        pytest.skip("oracle_gap.json not available")
    rows = json.loads(oracle_path.read_text())["rows"]
    expected_n = sum(1 for r in rows if r["l3_size"] in ("1MB", "4MB", "8MB"))
    assert payload["meta"]["n_paper_l3_cells_total"] == expected_n, (
        f"audit total {payload['meta']['n_paper_l3_cells_total']} != "
        f"oracle_gap paper-L3 rows {expected_n}; audit drifted from source"
    )
