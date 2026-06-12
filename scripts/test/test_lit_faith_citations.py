"""LIT-Cite — literature-faithfulness citation locator integrity gate.

Locks the invariants in ``wiki/data/lit_faith_citations.json`` so a
regression that:

* Adds a citation in lit-faith but forgets to add the underlying
  ``LiteratureClaim`` to ``literature_baselines.py`` (or vice-versa),
* Drops one of the three canonical anchor papers,
* Removes the DOI/URL from the source-of-truth module docstring,
* Leaves a baseline claim with a placeholder citation string,
* Strips a location qualifier (``§N.N`` / ``Fig N``) from a citation,

trips the gate. This is the literature-grounding twin of LIT-Cov
(coverage), LIT-Mar (margin), and LIT-Sig (sign signal).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "lit_faith_citations.json"

# Sample-size and shape floors.
UNIQUE_CITATIONS_FLOOR = 13
BASELINE_CLAIM_COUNT_FLOOR = 38
FAITH_CLAIM_COUNT_FLOOR = 270
WELL_FORMED_FRACTION_FLOOR = 1.0  # every citation must be well-formed today

# Per-anchor minimum baseline claim counts. If any of these drop, the
# anchor paper has been silently de-emphasized in the corpus.
ANCHOR_BASELINE_FLOORS = {
    "faldu_hpca_2020":   12,   # GRASP — drives most of the per-graph claims
    "balaji_hpca_2021":  10,   # P-OPT — most of the POPT cells
    "jaleel_isca_2010":   5,   # RRIP / SRRIP scan-resistance claims
}

# Anchor papers that must have a DOI/URL anchor in the literature_baselines
# module docstring. Jaleel is intentionally omitted because the current
# docstring doesn't link a stable DOI for the ISCA 2010 RRIP paper.
ANCHORS_REQUIRING_URL = {"faldu_hpca_2020", "balaji_hpca_2021"}


@pytest.fixture(scope="module")
def audit():
    assert ARTIFACT.exists(), (
        f"missing {ARTIFACT} — run `make lit-citations`"
    )
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def anchors_by_key(audit):
    return {a["key"]: a for a in audit["anchors"]}


@pytest.fixture(scope="module")
def docstring_grounding_by_key(audit):
    return {d["key"]: d for d in audit["docstring_grounding"]}


# ────────────────────── shape / coverage ──────────────────────


def test_schema_version(audit):
    assert audit["schema_version"] == 1


def test_unique_citation_count_floor(audit):
    assert (
        audit["summary"]["lit_faith_unique_citations"]
        >= UNIQUE_CITATIONS_FLOOR
    )
    assert (
        audit["summary"]["baseline_unique_citations"]
        >= UNIQUE_CITATIONS_FLOOR
    )


def test_baseline_claim_count_floor(audit):
    assert (
        audit["summary"]["baseline_claim_count"]
        >= BASELINE_CLAIM_COUNT_FLOOR
    )


def test_faith_claim_count_floor(audit):
    assert (
        audit["summary"]["faith_claim_count"]
        >= FAITH_CLAIM_COUNT_FLOOR
    )


# ────────────────────── bijection invariant ──────────────────────


def test_no_citations_only_in_faith(audit):
    """Every citation in lit-faith must resolve to a baseline claim."""
    only_in_faith = audit["only_in_faith"]
    assert only_in_faith == [], (
        f"citations in lit-faith with no baseline backing: "
        f"{only_in_faith}"
    )


def test_no_citations_only_in_baselines(audit):
    """Every baseline citation must be exercised by lit-faith."""
    only_in_baselines = audit["only_in_baselines"]
    assert only_in_baselines == [], (
        f"baseline citations never referenced by lit-faith: "
        f"{only_in_baselines}"
    )


def test_intersection_matches_unique_counts(audit):
    """Bijection sanity: intersection equals both unique counts."""
    s = audit["summary"]
    assert s["intersection_size"] == s["lit_faith_unique_citations"]
    assert s["intersection_size"] == s["baseline_unique_citations"]


# ────────────────────── citation well-formedness ──────────────────────


def test_all_citations_well_formed(audit):
    """Every citation must carry venue + year + locator + known anchor."""
    grounding = audit["citation_grounding"]
    bad = [g for g in grounding if not g["well_formed"]]
    assert not bad, (
        f"{len(bad)} citation(s) ill-formed: "
        f"{[g['citation'] for g in bad]}"
    )


def test_well_formed_fraction_at_ceiling(audit):
    """Fraction of well-formed citations must equal the floor (1.0)."""
    s = audit["summary"]
    total = (
        s["citations_well_formed"] + s["citations_ill_formed_count"]
    )
    assert total > 0
    fraction = s["citations_well_formed"] / total
    assert fraction >= WELL_FORMED_FRACTION_FLOOR, (
        f"well-formed fraction {fraction:.3f} below floor "
        f"{WELL_FORMED_FRACTION_FLOOR}"
    )


def test_no_baseline_short_citations(audit):
    """Baseline claims must not carry placeholder-length citations."""
    short = audit["baseline_short_citations"]
    assert short == [], (
        f"baseline claims with short citation strings: {short}"
    )


# ────────────────────── anchor-paper inventory ──────────────────────


def test_all_anchors_present_in_audit(anchors_by_key):
    for key in ANCHOR_BASELINE_FLOORS:
        assert key in anchors_by_key, f"anchor {key} missing from audit"


@pytest.mark.parametrize("key", sorted(ANCHOR_BASELINE_FLOORS))
def test_anchor_baseline_claim_floor(anchors_by_key, key):
    floor = ANCHOR_BASELINE_FLOORS[key]
    n = anchors_by_key[key]["baseline_claim_count"]
    assert n >= floor, (
        f"anchor {key} backs only {n} baseline claims; floor is {floor}"
    )


@pytest.mark.parametrize("key", sorted(ANCHOR_BASELINE_FLOORS))
def test_anchor_faith_claim_floor(anchors_by_key, key):
    """Each anchor must be used by at least one lit-faith cell."""
    n = anchors_by_key[key]["faith_claim_count"]
    assert n > 0, f"anchor {key} backs 0 lit-faith cells"


@pytest.mark.parametrize("key", sorted(ANCHORS_REQUIRING_URL))
def test_anchor_url_in_docstring(docstring_grounding_by_key, key):
    """Anchor papers expected to be linked must appear by URL prefix in
    the literature_baselines module docstring (source-of-truth grounding)."""
    d = docstring_grounding_by_key[key]
    assert d["url_prefix_present_in_docstring"], (
        f"anchor {key} expected URL prefix "
        f"`{d['url_prefix_expected']}` missing from "
        f"literature_baselines.py docstring"
    )


def test_anchors_present_in_baselines_floor(audit):
    """All canonical anchor papers must back at least one baseline."""
    s = audit["summary"]
    assert s["anchors_present_in_baselines"] == s["anchors_total"]


def test_anchors_present_in_faith_floor(audit):
    """All canonical anchor papers must back at least one lit-faith cell."""
    s = audit["summary"]
    assert s["anchors_present_in_faith"] == s["anchors_total"]


# ────────────────────── grounding-diagnostic completeness ──────────────────────


def test_citation_grounding_size_matches_unique_count(audit):
    """citation_grounding must enumerate one row per unique citation."""
    assert len(audit["citation_grounding"]) == (
        audit["summary"]["lit_faith_unique_citations"]
    )


def test_every_citation_has_venue_tag(audit):
    bad = [g for g in audit["citation_grounding"] if not g["has_venue_tag"]]
    assert not bad, f"citations without venue tag: {[g['citation'] for g in bad]}"


def test_every_citation_has_year(audit):
    bad = [g for g in audit["citation_grounding"] if not g["has_year"]]
    assert not bad, f"citations without year: {[g['citation'] for g in bad]}"


def test_every_citation_has_location_qualifier(audit):
    bad = [
        g for g in audit["citation_grounding"]
        if not g["has_location_qualifier"]
    ]
    assert not bad, (
        f"citations without location qualifier (§/Fig/Sec/Table): "
        f"{[g['citation'] for g in bad]}"
    )


def test_every_citation_has_anchor(audit):
    bad = [g for g in audit["citation_grounding"] if not g["anchors"]]
    assert not bad, (
        f"citations not mapping to any canonical anchor: "
        f"{[g['citation'] for g in bad]}"
    )
