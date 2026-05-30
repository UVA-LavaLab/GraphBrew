"""Confidence gate 213 — paper claims source-value parity (PCV-Src).

For every claim in wiki/data/paper_claims.json, independently re-derive
the claim's value from the source artifact (using only the artifact's
public schema) and assert that the result matches the claim's recorded
value within a tight numeric tolerance. This is the strongest form of
claim integrity gate: it does NOT trust paper_claims.json's value field;
it re-computes from the source-of-truth artifact.

Why this matters:
* paper_claims.json is the SINGLE registry that the paper introduction
  and abstract cite by ID. If a generator silently changes its output
  (e.g. corpus diversity drops a graph), the paper_claims.json value
  field stays stale unless someone re-runs `make lit-claims`. This gate
  catches that drift by re-deriving each value from the source artifact
  on every confidence run.
* Distinct from gate 209 (XAI-Int) which only checks REGISTRY-GRAPH
  consistency (claim points to existing source/gate). This gate checks
  VALUE consistency (claim.value actually equals what the source says).

One test per paper claim (14 tests today, one per claim). When a claim
is added/removed, the test count auto-updates via PAPER_CLAIMS_DERIVATIONS
mapping below — adding a claim without registering a derivation here
fails test_all_claims_have_derivation.

Derivations cover all 9 categories: corpus, reproduction, lit_faith,
winner, thrash, popt_vs_grasp, deviations, cross_tool, meta.

Special case: confidence.green_gate_count is a META-claim whose source
IS the confidence_dashboard.json output AND whose gate IS the dashboard
generator. The derivation is `len(suites)` where every suite has
failed==0 and errors==0. This claim is only valid AFTER the dashboard
has been regenerated in the current cycle; otherwise it's the count
from the previous run. We assert equality to ±0 (exact integer).

Tolerance rules:
* Integer claims: exact equality (no tolerance).
* Percentage claims: 0.1pp absolute tolerance (rounded display).
* Per-mille / pp signed claims: 0.01 absolute tolerance.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PAPER_CLAIMS = json.loads((WIKI_DATA / "paper_claims.json").read_text())["claims"]
CLAIMS_BY_ID = {c["id"]: c for c in PAPER_CLAIMS}


# ---------------------------------------------------------------------------
# Per-claim derivations. Each function takes no args, opens the source
# artifact directly, and returns the derived numeric value.
# ---------------------------------------------------------------------------


def _derive_corpus_graph_count() -> int:
    d = json.loads((WIKI_DATA / "corpus_diversity.json").read_text())
    assert isinstance(d, list), "corpus_diversity.json root must be a list of graph entries"
    return len(d)


def _derive_reproduction_ok_ratio() -> float:
    d = json.loads((WIKI_DATA / "claim_density.json").read_text())["summary"]
    return float(d["total_ok_pct"])


def _derive_reproduction_n_graphs_with_claims() -> int:
    d = json.loads((WIKI_DATA / "claim_density.json").read_text())["summary"]
    return int(d["n_graphs"])


def _derive_lit_faith_disagreement_rate() -> int:
    d = json.loads((WIKI_DATA / "literature_faithfulness_postfix.json").read_text())["summary"]
    return int(d["disagree"])


def _derive_winner_grasp_share() -> float:
    d = json.loads((WIKI_DATA / "policy_winner_table.json").read_text())["summary"]
    return d["wins_by_policy"]["GRASP"] / d["n_cells"] * 100.0


def _derive_winner_popt_share() -> float:
    d = json.loads((WIKI_DATA / "policy_winner_table.json").read_text())["summary"]
    return d["wins_by_policy"]["POPT"] / d["n_cells"] * 100.0


def _derive_winner_srrip_share() -> float:
    d = json.loads((WIKI_DATA / "policy_winner_table.json").read_text())["summary"]
    return d["wins_by_policy"]["SRRIP"] / d["n_cells"] * 100.0


def _derive_winner_lru_share() -> float:
    d = json.loads((WIKI_DATA / "policy_winner_table.json").read_text())["summary"]
    return d["wins_by_policy"]["LRU"] / d["n_cells"] * 100.0


def _derive_thrash_lru_wins_at_4kb() -> int:
    d = json.loads((WIKI_DATA / "small_l3_thrash.json").read_text())
    return sum(
        1 for c in d["cells"]
        if c.get("l3_size") == "4kB" and c.get("winner") == "LRU"
    )


def _derive_popt_vs_grasp_road_family_mean() -> float:
    d = json.loads((WIKI_DATA / "popt_vs_grasp_delta.json").read_text())["summary"]
    return float(d["by_family"]["road"]["mean_pp"])


def _derive_popt_vs_grasp_social_family_mean() -> float:
    d = json.loads((WIKI_DATA / "popt_vs_grasp_delta.json").read_text())["summary"]
    return float(d["by_family"]["social"]["mean_pp"])


def _derive_deviations_popt_overhead_share() -> float:
    d = json.loads((WIKI_DATA / "literature_deviations.json").read_text())["summary"]
    return d["by_mechanism"]["popt_overhead_dominates"] / d["n_deviations"] * 100.0


def _derive_cross_tool_doubly_saturated_agreement() -> int:
    d = json.loads((WIKI_DATA / "cross_tool_saturation.json").read_text())["summary"]
    return len(d["disagreements"])


def _derive_confidence_green_gate_count() -> int:
    d = json.loads((WIKI_DATA / "confidence_dashboard.json").read_text())
    return sum(
        1 for s in d["suites"]
        if s.get("failed", 0) == 0 and s.get("errors", 0) == 0
    )


PAPER_CLAIMS_DERIVATIONS = {
    "corpus.graph_count": (_derive_corpus_graph_count, 0),
    "reproduction.ok_ratio": (_derive_reproduction_ok_ratio, 0.1),
    "reproduction.n_graphs_with_claims": (_derive_reproduction_n_graphs_with_claims, 0),
    "lit_faith.disagreement_rate": (_derive_lit_faith_disagreement_rate, 0),
    "winner.grasp_share": (_derive_winner_grasp_share, 0.1),
    "winner.popt_share": (_derive_winner_popt_share, 0.1),
    "winner.srrip_share": (_derive_winner_srrip_share, 0.1),
    "winner.lru_share": (_derive_winner_lru_share, 0.1),
    "thrash.lru_wins_at_4kb": (_derive_thrash_lru_wins_at_4kb, 0),
    "popt_vs_grasp.road_family_mean": (_derive_popt_vs_grasp_road_family_mean, 0.01),
    "popt_vs_grasp.social_family_mean": (_derive_popt_vs_grasp_social_family_mean, 0.01),
    "deviations.popt_overhead_share": (_derive_deviations_popt_overhead_share, 0.1),
    "cross_tool.doubly_saturated_agreement": (_derive_cross_tool_doubly_saturated_agreement, 0),
    "confidence.green_gate_count": (_derive_confidence_green_gate_count, 0),
}


# ---------------------------------------------------------------------------
# Self-consistency tests
# ---------------------------------------------------------------------------


def test_all_claims_have_derivation():
    """Adding a claim to paper_claims.json without registering a
    derivation here is a forbidden silent expansion of the claim
    surface. Force the test author to acknowledge new claims.
    """
    claim_ids = {c["id"] for c in PAPER_CLAIMS}
    missing = sorted(claim_ids - set(PAPER_CLAIMS_DERIVATIONS))
    assert not missing, (
        f"Paper claims with no derivation registered in PAPER_CLAIMS_DERIVATIONS: {missing}. "
        f"Every claim must have a re-derivation from its source artifact."
    )


def test_no_orphan_derivations():
    """If a claim is removed from paper_claims.json, also remove its
    derivation here. Catches stale derivations against removed claims.
    """
    claim_ids = {c["id"] for c in PAPER_CLAIMS}
    extras = sorted(set(PAPER_CLAIMS_DERIVATIONS) - claim_ids)
    assert not extras, (
        f"PAPER_CLAIMS_DERIVATIONS entries with no matching claim in registry: {extras}. "
        f"Remove them — they refer to deleted claims."
    )


# ---------------------------------------------------------------------------
# Per-claim parametric test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim_id",
    sorted(PAPER_CLAIMS_DERIVATIONS.keys()),
)
def test_claim_value_matches_source_derivation(claim_id: str):
    claim = CLAIMS_BY_ID[claim_id]
    derive, tol = PAPER_CLAIMS_DERIVATIONS[claim_id]
    declared = claim["value"]
    derived = derive()
    if tol == 0:
        assert int(derived) == int(declared), (
            f"Claim {claim_id}: declared value {declared} != "
            f"derived value {derived} from {claim['source']} (tolerance=exact)"
        )
    else:
        delta = abs(float(derived) - float(declared))
        assert delta <= tol, (
            f"Claim {claim_id}: declared value {declared} ± {tol} != "
            f"derived value {derived:.6f} from {claim['source']} "
            f"(delta={delta:.6f})"
        )


def test_no_nan_or_infinite_claim_values():
    """A claim with non-finite value would silently pass any tolerance
    check. Forbid them at the registry level.
    """
    bad = []
    for c in PAPER_CLAIMS:
        v = c["value"]
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                bad.append((c["id"], v))
    assert not bad, f"Claims with non-finite values: {bad}"


def test_no_nan_or_infinite_derived_values():
    """Mirror of the registry check, but for the derived side."""
    bad = []
    for cid, (derive, _) in PAPER_CLAIMS_DERIVATIONS.items():
        v = derive()
        if isinstance(v, float) and not math.isfinite(v):
            bad.append((cid, v))
    assert not bad, f"Derivations yielding non-finite values: {bad}"
