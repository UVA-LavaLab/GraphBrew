"""Pytest gate: family-classification sensitivity sweep.

Defends the paper's headline sign claims against the obvious reviewer
challenge "what if you relabeled graph X as family Y?". The bedrock
claim (POPT < LRU on social) must survive EVERY relabeling. Per-
family POPT < GRASP claims may degrade when the only member of the
family is reassigned (e.g. roadNet-CA out of road), which is
geometrically expected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_JSON = REPO_ROOT / "wiki" / "data" / "family_sensitivity.json"

EXPECTED_RELABELINGS = 32  # 8 graphs × 4 alternative families


@pytest.fixture(scope="module")
def doc() -> dict:
    if not DOC_JSON.exists():
        pytest.skip(f"{DOC_JSON} missing; run `make lit-family-sensitivity`")
    return json.loads(DOC_JSON.read_text())


def test_schema(doc):
    expected = {"meta", "canonical_claims", "relabelings",
                "per_claim_flip_count", "n_relabelings",
                "n_flipping_relabelings", "canonical_state"}
    assert expected.issubset(doc.keys()), f"missing keys: {expected - doc.keys()}"


def test_n_relabelings_is_32(doc):
    assert doc["n_relabelings"] == EXPECTED_RELABELINGS, (
        f"expected {EXPECTED_RELABELINGS} relabelings "
        f"(8 graphs × 4 alt families), got {doc['n_relabelings']}"
    )


def test_canonical_claims_count(doc):
    assert len(doc["canonical_claims"]) == 7, (
        f"expected 7 sign claims, got {len(doc['canonical_claims'])}"
    )


def test_popt_lt_lru_on_social_is_bedrock(doc):
    """The strongest paper claim: POPT < LRU on social. Must survive
    EVERY family relabeling (0 flips out of 32)."""
    n = doc["per_claim_flip_count"].get("POPT < LRU on social")
    assert n == 0, (
        f"POPT < LRU on social is fragile: {n} of 32 relabelings flip "
        "it. The bedrock paper claim no longer survives reviewer pushback "
        "on family taxonomy."
    )


def test_grasp_lt_lru_on_social_robust(doc):
    """Sister claim: GRASP < LRU on social. ≤1 flip allowed (only the
    roadNet-CA→social relabeling poisons social with off-distribution
    data; the resulting flip is geometrically expected)."""
    n = doc["per_claim_flip_count"].get("GRASP < LRU on social")
    assert n is not None and n <= 1, (
        f"GRASP < LRU on social fragile: {n} of 32 relabelings flip it"
    )


def test_road_claim_only_flipped_by_pulling_road_member(doc):
    """POPT < GRASP on road can ONLY lose stability when the sole road
    member (roadNet-CA) is moved out of road. That's 4 flips (one per
    alternative family). Any other source of flip would indicate
    a deeper data issue."""
    n = doc["per_claim_flip_count"].get("POPT < GRASP on road")
    assert n is not None and n <= 4, (
        f"POPT < GRASP on road has {n} flips, expected ≤ 4 "
        "(roadNet-CA → {{citation, mesh, social, web}})"
    )

    # Every road-claim flip should involve roadNet-CA reassignment.
    road_flips = [
        r for r in doc["relabelings"]
        if any(f["claim"] == "POPT < GRASP on road" for f in r["flipped"])
    ]
    bad = [r for r in road_flips if r["graph"] != "roadNet-CA"]
    assert not bad, (
        f"unexpected road-claim flips not caused by roadNet-CA: "
        f"{[(r['graph'], r['new_family']) for r in bad]}"
    )


def test_canonical_state_pins_stable_claims(doc):
    """The canonical bootstrap state for the three claims the paper
    pins must remain above 0.95."""
    state = doc["canonical_state"]
    pinned = {
        "POPT < GRASP on road": 0.95,
        "POPT < LRU on social": 0.99,
        "GRASP < LRU on social": 0.99,
    }
    for claim, floor in pinned.items():
        frac = state.get(claim)
        assert frac is not None and frac >= floor, (
            f"canonical `{claim}` = {frac}, below floor {floor}"
        )


def test_every_relabeling_has_required_fields(doc):
    for r in doc["relabelings"]:
        assert {"graph", "canonical_family", "new_family", "flipped"} <= r.keys()
        assert r["canonical_family"] != r["new_family"]


def test_meta_records_stability_floor(doc):
    assert doc["meta"].get("stability_floor") == 0.95
    assert doc["meta"].get("n_resamples") >= 1000


def test_flip_counts_sum_consistency(doc):
    """Sum of per-claim flip counts equals sum over relabelings of
    len(flipped). Internal consistency check."""
    total_from_per_claim = sum(doc["per_claim_flip_count"].values())
    total_from_rows = sum(len(r["flipped"]) for r in doc["relabelings"])
    assert total_from_per_claim == total_from_rows, (
        f"flip-count books don't balance: "
        f"per-claim sum = {total_from_per_claim}, "
        f"per-row sum = {total_from_rows}"
    )
