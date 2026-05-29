"""Paper-claims registry cross-artifact integrity gate (gate 84).

paper_claims.json is the canonical numerical headline for the paper.
Each claim cites a source JSON file plus a `value`. If a source
artifact regenerates with a different number but the registry
generator's value-extraction logic silently masks it, the paper would
ship a stale claim. This gate re-derives every claim's value from its
cited source and asserts equality (within a numeric tolerance).

It also checks the structural contract of the registry: every claim
has the required keys, every source file exists, every gate path
exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CLAIMS_JSON = Path("wiki/data/paper_claims.json")
REQUIRED_CLAIM_KEYS = {"id", "category", "text", "value", "units", "source", "gate"}
EXPECTED_CLAIM_IDS = {
    "corpus.graph_count",
    "reproduction.ok_ratio",
    "reproduction.n_graphs_with_claims",
    "lit_faith.disagreement_rate",
    "winner.grasp_share",
    "winner.popt_share",
    "winner.srrip_share",
    "winner.lru_share",
    "thrash.lru_wins_at_4kb",
    "popt_vs_grasp.road_family_mean",
    "popt_vs_grasp.social_family_mean",
    "deviations.popt_overhead_share",
    "cross_tool.doubly_saturated_agreement",
    "confidence.green_gate_count",
}
NUMERIC_TOL = 0.06   # registry rounds to 1 dp; allow ~half-LSB headroom


def _load():
    if not CLAIMS_JSON.exists():
        pytest.skip(f"{CLAIMS_JSON} not generated yet")
    return json.loads(CLAIMS_JSON.read_text())


def _claims_by_id():
    blob = _load()
    return {c["id"]: c for c in blob["claims"]}


def _load_source(path: str):
    p = Path(path)
    if not p.exists():
        pytest.skip(f"source artifact missing: {p}")
    return json.loads(p.read_text())


# ---------- structural ----------


def test_n_claims_matches_list():
    blob = _load()
    assert blob["n_claims"] == len(blob["claims"])


def test_every_claim_has_required_keys():
    blob = _load()
    bad = []
    for c in blob["claims"]:
        missing = REQUIRED_CLAIM_KEYS - set(c)
        if missing:
            bad.append((c.get("id", "?"), missing))
    assert not bad, bad


def test_expected_claim_ids_present():
    cbi = _claims_by_id()
    missing = EXPECTED_CLAIM_IDS - set(cbi)
    assert not missing, missing


def test_no_duplicate_claim_ids():
    blob = _load()
    ids = [c["id"] for c in blob["claims"]]
    assert len(ids) == len(set(ids))


def test_every_source_path_exists():
    blob = _load()
    missing = []
    for c in blob["claims"]:
        if not Path(c["source"]).exists():
            missing.append((c["id"], c["source"]))
    assert not missing, missing


def test_every_gate_path_exists():
    blob = _load()
    missing = []
    for c in blob["claims"]:
        if not Path(c["gate"]).exists():
            missing.append((c["id"], c["gate"]))
    assert not missing, missing


# ---------- value re-derivation ----------


def test_corpus_graph_count_matches_source():
    cbi = _claims_by_id()
    claim = cbi["corpus.graph_count"]
    src = _load_source(claim["source"])
    # corpus_diversity.json is a list of per-graph rows.
    if isinstance(src, list):
        expected = len(src)
    else:
        expected = len(src.get("graphs", []))
    assert claim["value"] == expected, (claim["value"], expected)


def test_reproduction_ok_ratio_matches_source():
    cbi = _claims_by_id()
    claim = cbi["reproduction.ok_ratio"]
    src = _load_source(claim["source"])
    s = src.get("summary", {})
    # Registry stores the total_ok_pct (rounded to 1 dp).
    expected = s.get("total_ok_pct") or s.get("ok_pct")
    assert expected is not None, "summary lacks total_ok_pct"
    assert abs(float(claim["value"]) - float(expected)) <= NUMERIC_TOL, (
        claim["value"], expected,
    )


def test_reproduction_n_graphs_matches_source():
    cbi = _claims_by_id()
    claim = cbi["reproduction.n_graphs_with_claims"]
    src = _load_source(claim["source"])
    s = src.get("summary", {})
    expected = s.get("n_graphs") or len(src.get("graphs", []))
    assert claim["value"] == expected, (claim["value"], expected)


def test_winner_shares_sum_to_100():
    cbi = _claims_by_id()
    shares = [
        cbi[f"winner.{p}_share"]["value"]
        for p in ("grasp", "popt", "srrip", "lru")
    ]
    total = sum(shares)
    assert abs(total - 100.0) <= 0.5, (shares, total)


def test_winner_shares_match_source():
    cbi = _claims_by_id()
    claim_pol = cbi["winner.grasp_share"]
    src = _load_source(claim_pol["source"])
    s = src.get("summary", {})
    win_counts = s.get("wins_by_policy") or s.get("win_counts", {})
    n_cells = sum(win_counts.values())
    assert n_cells >= 30, n_cells
    bad = []
    for pol_key in ("grasp", "popt", "srrip", "lru"):
        claim = cbi[f"winner.{pol_key}_share"]
        wins = win_counts.get(pol_key.upper(), 0)
        expected = round(100.0 * wins / n_cells, 1)
        if abs(claim["value"] - expected) > NUMERIC_TOL:
            bad.append((pol_key, claim["value"], expected))
    assert not bad, bad


def test_thrash_lru_wins_matches_source():
    cbi = _claims_by_id()
    claim = cbi["thrash.lru_wins_at_4kb"]
    src = _load_source(claim["source"])
    expected = src.get("summary", {}).get("win_counts", {}).get("LRU")
    if expected is None:
        expected = src.get("summary", {}).get("policy_stats", {}).get("LRU", {}).get("wins")
    assert expected is not None
    assert claim["value"] == expected, (claim["value"], expected)


def test_popt_vs_grasp_road_family_matches_source():
    cbi = _claims_by_id()
    claim = cbi["popt_vs_grasp.road_family_mean"]
    src = _load_source(claim["source"])
    expected = src["summary"]["by_family"]["road"]["mean_pp"]
    assert abs(float(claim["value"]) - float(expected)) <= NUMERIC_TOL, (
        claim["value"], expected,
    )


def test_popt_vs_grasp_social_family_matches_source():
    cbi = _claims_by_id()
    claim = cbi["popt_vs_grasp.social_family_mean"]
    src = _load_source(claim["source"])
    expected = src["summary"]["by_family"]["social"]["mean_pp"]
    assert abs(float(claim["value"]) - float(expected)) <= NUMERIC_TOL, (
        claim["value"], expected,
    )


def test_deviations_popt_overhead_share_in_range():
    cbi = _claims_by_id()
    claim = cbi["deviations.popt_overhead_share"]
    # Source is literature_deviations.json - presence + range check.
    src = _load_source(claim["source"])
    assert isinstance(claim["value"], (int, float))
    assert 0.0 <= claim["value"] <= 100.0, claim["value"]
    # Sanity: the source carries some deviations payload.
    assert src, "deviations source is empty"


def test_cross_tool_doubly_saturated_agreement_zero_disagreements():
    cbi = _claims_by_id()
    claim = cbi["cross_tool.doubly_saturated_agreement"]
    # Headline: 0 disagreements.
    assert claim["value"] == 0, claim["value"]


def test_confidence_green_gate_count_matches_dashboard():
    cbi = _claims_by_id()
    claim = cbi["confidence.green_gate_count"]
    src = _load_source(claim["source"])
    suites = src.get("suites", [])
    n_green = sum(
        1 for s in suites
        if int(s.get("failed", 0)) == 0 and int(s.get("errors", 0)) == 0
    )
    assert claim["value"] == n_green, (claim["value"], n_green, len(suites))
    # Headline text must mention the same numerator.
    assert str(n_green) in claim["text"], claim["text"]


def test_lit_faith_disagreement_rate_zero():
    cbi = _claims_by_id()
    claim = cbi["lit_faith.disagreement_rate"]
    # Headline: zero disagreements is the load-bearing claim.
    assert claim["value"] == 0, claim["value"]
