"""Confidence gate 103 — paper_claims registry recompute parity.

paper_claims.json is the bridge between the experimental artifact
tree (~70 generators × cross-tool sweeps) and the prose in the paper:
every numeric claim made in the text has one entry here, pointing at
the exact upstream artifact and gate that produced it. If a value in
the registry drifts away from what its cited artifact actually says,
the paper is suddenly making claims that the artifacts no longer
support — and nothing else in the pipeline catches it because every
other gate validates artifacts against themselves, not against the
registry's frozen copy.

This gate ties down 13 invariants so that paper_claims.json is always
recomputable from its sources, all references resolve, and the
registry schema stays internally consistent.

Invariants (3 / 3 / 6 / 1):

  Registry schema (3):
    1. exactly EXPECTED_CLAIM_COUNT claims (14)
    2. every claim has the 7 required keys
       (id/category/source/value/units/gate/text), text is non-empty
    3. claim ids are unique across the registry

  References resolve (3):
    4. every claim.source path exists on disk
    5. every claim.gate path exists on disk
    6. every claim.category is in EXPECTED_CATEGORIES, every
       claim.units is in EXPECTED_UNITS

  Value recompute (6) — for every claim, the value baked into the
  registry must match what its cited artifact currently says, within
  the appropriate tolerance:
    7. corpus + reproduction + lit_faith claims recompute correctly
    8. winner.* shares recompute from policy_winner_table.json
       and sum to ~100 %
    9. thrash + cross_tool claims recompute correctly
   10. popt_vs_grasp.* family means recompute from
       popt_vs_grasp_delta.json
   11. deviations.popt_overhead_share recomputes from
       literature_deviations.json
   12. confidence.green_gate_count recomputes from
       confidence_dashboard.json (snake-eating-tail aware: claim
       value and dashboard.json are regenerated together, so the
       on-disk pair must always agree)

  Math hygiene (1):
   13. every value is finite numeric (no NaN/inf/None leak)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_CLAIMS_PATH = PROJECT_ROOT / "wiki" / "data" / "paper_claims.json"

EXPECTED_CLAIM_COUNT = 14
CLAIM_REQUIRED_KEYS = frozenset({
    "id", "category", "source", "value", "units", "gate", "text",
})
EXPECTED_CATEGORIES = frozenset({
    "corpus", "reproduction", "lit_faith", "winner_table", "thrash",
    "popt_vs_grasp", "deviations", "cross_tool", "meta",
})
EXPECTED_UNITS = frozenset({
    "graphs", "percent", "claims", "cells", "pp", "disagreements", "gates",
})
PERCENT_TOL = 0.1   # registry stores percentages rounded to one decimal
PP_TOL = 0.01       # registry stores per-family means rounded to three decimals
SHARE_SUM_TOL = 0.5 # winner shares sum to 100 % within rounding wobble


@pytest.fixture(scope="module")
def registry() -> dict:
    assert PAPER_CLAIMS_PATH.exists(), f"missing paper_claims.json at {PAPER_CLAIMS_PATH}"
    return json.loads(PAPER_CLAIMS_PATH.read_text())


def _load(rel_path: str) -> object:
    return json.loads((PROJECT_ROOT / rel_path).read_text())


def _by_id(claims: list[dict]) -> dict[str, dict]:
    return {c["id"]: c for c in claims}


# ---------------------------------------------------------------------------
# Registry schema (3)
# ---------------------------------------------------------------------------


def test_claim_count_matches_expected(registry: dict) -> None:
    claims = registry["claims"]
    assert len(claims) == EXPECTED_CLAIM_COUNT, (
        f"paper_claims.json has {len(claims)} claims, expected {EXPECTED_CLAIM_COUNT}; "
        "bump EXPECTED_CLAIM_COUNT in this test if you've intentionally added a claim"
    )
    n_claims_field = registry.get("n_claims")
    assert n_claims_field == len(claims), (
        f"top-level n_claims={n_claims_field} != len(claims)={len(claims)}"
    )


def test_each_claim_has_required_keys(registry: dict) -> None:
    bad: list[tuple[str, list[str], list[str]]] = []
    for c in registry["claims"]:
        keys = set(c.keys())
        missing = sorted(CLAIM_REQUIRED_KEYS - keys)
        if missing:
            bad.append((c.get("id", "?"), missing, sorted(keys - CLAIM_REQUIRED_KEYS)))
        elif not (isinstance(c.get("text"), str) and c["text"].strip()):
            bad.append((c["id"], ["empty text"], []))
    assert not bad, f"claims with bad schema (id, missing, extra): {bad}"


def test_claim_ids_are_unique(registry: dict) -> None:
    seen: dict[str, int] = {}
    for c in registry["claims"]:
        seen[c["id"]] = seen.get(c["id"], 0) + 1
    dupes = [(k, v) for k, v in seen.items() if v > 1]
    assert not dupes, f"paper_claims.json has duplicate claim ids: {dupes}"


# ---------------------------------------------------------------------------
# References resolve (3)
# ---------------------------------------------------------------------------


def test_every_source_path_exists(registry: dict) -> None:
    missing = [
        (c["id"], c["source"])
        for c in registry["claims"]
        if not (PROJECT_ROOT / c["source"]).exists()
    ]
    assert not missing, f"claim.source paths that don't exist: {missing}"


def test_every_gate_path_exists(registry: dict) -> None:
    missing = [
        (c["id"], c["gate"])
        for c in registry["claims"]
        if not (PROJECT_ROOT / c["gate"]).exists()
    ]
    assert not missing, f"claim.gate paths that don't exist: {missing}"


def test_categories_and_units_in_expected_sets(registry: dict) -> None:
    bad_cat = [
        (c["id"], c["category"])
        for c in registry["claims"] if c["category"] not in EXPECTED_CATEGORIES
    ]
    bad_units = [
        (c["id"], c["units"])
        for c in registry["claims"] if c["units"] not in EXPECTED_UNITS
    ]
    assert not bad_cat, f"unexpected categories: {bad_cat} (expected {sorted(EXPECTED_CATEGORIES)})"
    assert not bad_units, f"unexpected units: {bad_units} (expected {sorted(EXPECTED_UNITS)})"


# ---------------------------------------------------------------------------
# Value recompute (6)
# ---------------------------------------------------------------------------


def test_corpus_reproduction_lit_faith_values_recompute(registry: dict) -> None:
    by_id = _by_id(registry["claims"])
    cor = _load("wiki/data/corpus_diversity.json")
    cd = _load("wiki/data/claim_density.json")
    lf = _load("wiki/data/literature_faithfulness_postfix.json")

    assert by_id["corpus.graph_count"]["value"] == len(cor)

    rec_ok_pct = round(cd["summary"]["total_ok_pct"], 1)
    assert math.isclose(
        by_id["reproduction.ok_ratio"]["value"], rec_ok_pct, abs_tol=PERCENT_TOL
    ), f"reproduction.ok_ratio drift: registry={by_id['reproduction.ok_ratio']['value']} vs cd={rec_ok_pct}"

    assert by_id["reproduction.n_graphs_with_claims"]["value"] == cd["summary"]["n_graphs"]
    assert by_id["lit_faith.disagreement_rate"]["value"] == lf["summary"]["disagree"]


def test_winner_shares_recompute_and_sum_to_100(registry: dict) -> None:
    by_id = _by_id(registry["claims"])
    pw = _load("wiki/data/policy_winner_table.json")
    n = pw["summary"]["n_cells"]
    wins = pw["summary"]["wins_by_policy"]
    bad: list[tuple[str, float, float]] = []
    for pol_key, claim_id in [
        ("GRASP", "winner.grasp_share"),
        ("POPT", "winner.popt_share"),
        ("SRRIP", "winner.srrip_share"),
        ("LRU", "winner.lru_share"),
    ]:
        recomputed = round(100.0 * wins[pol_key] / n, 1)
        v = by_id[claim_id]["value"]
        if not math.isclose(v, recomputed, abs_tol=PERCENT_TOL):
            bad.append((claim_id, v, recomputed))
    assert not bad, f"winner share drift: {bad}"

    total = sum(by_id[k]["value"] for k in (
        "winner.grasp_share", "winner.popt_share", "winner.srrip_share", "winner.lru_share"
    ))
    assert math.isclose(total, 100.0, abs_tol=SHARE_SUM_TOL), (
        f"winner shares sum to {total}, not 100 % (within {SHARE_SUM_TOL})"
    )


def test_thrash_and_cross_tool_recompute(registry: dict) -> None:
    by_id = _by_id(registry["claims"])
    th = _load("wiki/data/small_l3_thrash.json")
    cs = _load("wiki/data/cross_tool_saturation.json")

    rec_lru = th["summary"]["win_counts"].get("LRU", 0)
    assert by_id["thrash.lru_wins_at_4kb"]["value"] == rec_lru, (
        f"thrash.lru_wins_at_4kb drift: registry={by_id['thrash.lru_wins_at_4kb']['value']} vs th={rec_lru}"
    )

    rec_disagree = len(cs["summary"]["disagreements"])
    assert by_id["cross_tool.doubly_saturated_agreement"]["value"] == rec_disagree, (
        f"cross_tool.doubly_saturated_agreement drift: "
        f"registry={by_id['cross_tool.doubly_saturated_agreement']['value']} vs cs={rec_disagree}"
    )


def test_popt_vs_grasp_family_means_recompute(registry: dict) -> None:
    by_id = _by_id(registry["claims"])
    pv = _load("wiki/data/popt_vs_grasp_delta.json")
    bf = pv["summary"]["by_family"]
    bad: list[tuple[str, float, float]] = []
    for fam, claim_id in [
        ("road", "popt_vs_grasp.road_family_mean"),
        ("social", "popt_vs_grasp.social_family_mean"),
    ]:
        recomputed = round(bf[fam]["mean_pp"], 3)
        v = by_id[claim_id]["value"]
        if not math.isclose(v, recomputed, abs_tol=PP_TOL):
            bad.append((claim_id, v, recomputed))
    assert not bad, f"popt_vs_grasp family mean drift: {bad}"


def test_deviations_share_recomputes(registry: dict) -> None:
    by_id = _by_id(registry["claims"])
    ld = _load("wiki/data/literature_deviations.json")
    total = ld["summary"]["n_deviations"]
    popt_overhead = ld["summary"]["by_mechanism"].get("popt_overhead_dominates", 0)
    rec = round(100.0 * popt_overhead / total, 1) if total else 0.0
    v = by_id["deviations.popt_overhead_share"]["value"]
    assert math.isclose(v, rec, abs_tol=PERCENT_TOL), (
        f"deviations.popt_overhead_share drift: registry={v} vs ld={rec} "
        f"({popt_overhead}/{total})"
    )


def test_confidence_green_gate_count_recomputes(registry: dict) -> None:
    by_id = _by_id(registry["claims"])
    dash = _load("wiki/data/confidence_dashboard.json")
    suites = dash["suites"]
    # Mirror the snake-eating-tail self-exemption in paper_claims_registry.
    self_ref = {
        "scripts/test/test_paper_claims_integrity.py",
        "scripts/test/test_paper_claims_recompute.py",
        "scripts/test/test_paper_claims_registry_derivation_parity.py",
        "scripts/test/test_paper_claims_value_parity.py",
        "scripts/test/test_catalog_dashboard_coverage_milestone.py",
    }
    rec_green = sum(
        1 for s in suites
        if s.get("path") in self_ref
        or (s["failed"] == 0 and s["errors"] == 0)
    )
    v = by_id["confidence.green_gate_count"]["value"]
    # Snake-eating-tail: paper_claims.value and dashboard.json are regenerated
    # together, so the on-disk pair must always agree exactly. The text field
    # additionally encodes "N gates today" — both must point at the same N.
    assert v == rec_green, (
        f"confidence.green_gate_count drift: registry={v} vs dashboard={rec_green}/{len(suites)}; "
        "if you added a new suite, also rerun lit-claims to refresh paper_claims.json"
    )


# ---------------------------------------------------------------------------
# Math hygiene (1)
# ---------------------------------------------------------------------------


def test_every_value_is_finite_numeric(registry: dict) -> None:
    bad: list[tuple[str, object]] = []
    for c in registry["claims"]:
        v = c.get("value")
        if not isinstance(v, (int, float)):
            bad.append((c["id"], v))
        elif isinstance(v, float) and not math.isfinite(v):
            bad.append((c["id"], v))
    assert not bad, f"non-finite or non-numeric claim values: {bad}"
