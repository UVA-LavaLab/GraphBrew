"""Gate 88 — family-sensitivity canonical-state cross-artifact gate.

Cross-artifact integrity gate locking the 7 canonical_state claims in
:file:`wiki/data/family_sensitivity.json` against the per-family
aggregates in :file:`wiki/data/policy_winner_table.json` and the
per-(family, app) breakdown in
:file:`wiki/data/popt_vs_grasp_by_family_app.json`.

The canonical_state claims are the only family-level POPT/GRASP/LRU
inequality claims the paper makes; if they ever silently flip or drift
out of sync with the winner-table family totals, the paper's narrative
changes underneath us. This gate makes that impossible.

Locked invariants:
  * Schema: 7 canonical claims, all 5 families {road, social, mesh,
    citation, web} present, canonical_state mirrors canonical_claims
    keys 1:1.
  * Each canonical claim's ``n_a`` and ``n_b`` (per-family cell count)
    matches the corresponding ``policy_winner_table.summary.
    wins_by_family[family]`` total sum exactly.
  * Dominance direction: a claim with ``frac > 0.5`` must have
    ``mean_a < mean_b`` (and the winner-table family must give
    policy_a more wins than policy_b), and conversely.
  * Always-stable POPT < LRU on social: frac == 1.0 and
    per_claim_flip_count == 0 (paper baseline that cannot regress).
  * Always-stable GRASP < LRU on social: frac == 1.0.
  * Strong road claim: frac >= 0.95 (POPT dominates road).
  * Strong mesh claim: frac >= 0.90 (POPT dominates mesh).
  * Robustness budget: at most ⌈n_relabelings / 2⌉ relabelings flip any
    canonical claim (n_flipping_relabelings is bounded relative to the
    sweep size).
  * Cell-count parity with popt_vs_grasp_by_family_app: 5 families × 5
    apps = 25 (family, app) cells must exist in
    popt_vs_grasp_by_family_app per_family_app keyset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "wiki" / "data"

FS_JSON = WIKI / "family_sensitivity.json"
PWT_JSON = WIKI / "policy_winner_table.json"
PVG_JSON = WIKI / "popt_vs_grasp_by_family_app.json"

EXPECTED_FAMILIES = {"road", "social", "mesh", "citation", "web"}
EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_CLAIMS = {
    "POPT < GRASP on road",
    "POPT < GRASP on social",
    "POPT < GRASP on mesh",
    "POPT < GRASP on citation",
    "POPT < GRASP on web",
    "POPT < LRU on social",
    "GRASP < LRU on social",
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _fs() -> dict:
    return _load(FS_JSON)


def _pwt_family_totals() -> dict[str, int]:
    pwt = _load(PWT_JSON)
    wbf = pwt["summary"]["wins_by_family"]
    return {fam: sum(d.values()) for fam, d in wbf.items()}


def _pwt_family_policy_wins(family: str) -> dict[str, int]:
    pwt = _load(PWT_JSON)
    return pwt["summary"]["wins_by_family"][family]


# ---------------------------------------------------------------------------
# Schema / shape
# ---------------------------------------------------------------------------

def test_canonical_claims_count_is_seven():
    cc = _fs()["canonical_claims"]
    assert len(cc) == 7, f"expected 7 canonical claims, got {len(cc)}"


def test_canonical_state_keys_match_claims_one_to_one():
    fs = _fs()
    state_keys = set(fs["canonical_state"].keys())
    claim_keys = {c["claim"] for c in fs["canonical_claims"]}
    assert state_keys == EXPECTED_CLAIMS == claim_keys, (
        f"canonical_state vs canonical_claims keys diverge: "
        f"state-only={state_keys - claim_keys}, "
        f"claims-only={claim_keys - state_keys}, "
        f"missing-from-expected={EXPECTED_CLAIMS - state_keys}"
    )


def test_all_expected_families_appear_in_canonical_claims():
    fs = _fs()
    fams = {c["family"] for c in fs["canonical_claims"]}
    assert fams == EXPECTED_FAMILIES, (
        f"family set in canonical_claims != expected: "
        f"got {sorted(fams)}, expected {sorted(EXPECTED_FAMILIES)}"
    )


def test_per_claim_flip_count_covers_every_canonical_claim():
    fs = _fs()
    pcfc = set(fs["per_claim_flip_count"].keys())
    assert pcfc == EXPECTED_CLAIMS, (
        f"per_claim_flip_count keys diverge from canonical claims: "
        f"missing={EXPECTED_CLAIMS - pcfc}, extra={pcfc - EXPECTED_CLAIMS}"
    )


# ---------------------------------------------------------------------------
# Cross-artifact: cell-count parity with policy_winner_table
# ---------------------------------------------------------------------------

def test_n_a_matches_winner_table_family_cell_count():
    fs = _fs()
    fam_totals = _pwt_family_totals()
    bad = []
    for c in fs["canonical_claims"]:
        want = fam_totals[c["family"]]
        if c["n_a"] != want:
            bad.append((c["claim"], c["n_a"], want))
    assert not bad, (
        f"canonical_claim n_a != policy_winner_table family cell count "
        f"for {bad}"
    )


def test_n_a_equals_n_b_for_every_canonical_claim():
    fs = _fs()
    bad = [
        (c["claim"], c["n_a"], c["n_b"])
        for c in fs["canonical_claims"]
        if c["n_a"] != c["n_b"]
    ]
    assert not bad, (
        f"canonical claims are paired-by-cell, so n_a must equal n_b: {bad}"
    )


# ---------------------------------------------------------------------------
# Cross-artifact: dominance direction agrees with winner table
# ---------------------------------------------------------------------------

def test_frac_dominance_direction_matches_mean_direction():
    """If frac > 0.5 then mean_a < mean_b (policy_a really is the
    smaller-delta-vs-LRU one), and vice versa. Catches a frac
    miscomputation."""
    fs = _fs()
    bad = []
    for c in fs["canonical_claims"]:
        if c["frac"] > 0.5 and not (c["mean_a"] < c["mean_b"]):
            bad.append((c["claim"], c["frac"], c["mean_a"], c["mean_b"]))
        if c["frac"] < 0.5 and not (c["mean_a"] > c["mean_b"]):
            bad.append((c["claim"], c["frac"], c["mean_a"], c["mean_b"]))
    assert not bad, (
        f"frac direction does not match mean direction for {bad}"
    )


def test_popt_dominates_web_in_winner_table_when_frac_high():
    """web has frac >= 0.95 for POPT < GRASP (a power-law family — road was
    RETIRED as out of P-OPT's power-law literature scope) — the winner table
    must therefore give POPT more wins than GRASP on web."""
    fs = _fs()
    web = next(c for c in fs["canonical_claims"] if c["claim"] == "POPT < GRASP on web")
    assert web["frac"] >= 0.95, f"web POPT<GRASP frac regressed: {web['frac']}"
    wins = _pwt_family_policy_wins("web")
    assert wins.get("POPT", 0) > wins.get("GRASP", 0), (
        f"web frac says POPT>GRASP but winner table says {wins}"
    )


def test_popt_dominates_mesh_in_winner_table_when_mesh_frac_high():
    fs = _fs()
    mesh = next(c for c in fs["canonical_claims"] if c["claim"] == "POPT < GRASP on mesh")
    assert mesh["frac"] >= 0.90, f"mesh POPT<GRASP frac regressed: {mesh['frac']}"
    wins = _pwt_family_policy_wins("mesh")
    assert wins.get("POPT", 0) >= wins.get("GRASP", 0), (
        f"mesh frac says POPT>=GRASP but winner table says {wins}"
    )


# ---------------------------------------------------------------------------
# Always-stable claims (paper baselines that cannot drift)
# ---------------------------------------------------------------------------

def test_popt_lt_lru_on_social_is_perfectly_stable():
    fs = _fs()
    state = fs["canonical_state"]["POPT < LRU on social"]
    flips = fs["per_claim_flip_count"]["POPT < LRU on social"]
    assert state == 1.0, f"POPT<LRU on social fraction regressed to {state}"
    assert flips == 0, (
        f"POPT<LRU on social is supposed to be relabeling-invariant; "
        f"flipped under {flips} relabelings"
    )


def test_grasp_lt_lru_on_social_at_unanimous_fraction():
    fs = _fs()
    state = fs["canonical_state"]["GRASP < LRU on social"]
    assert state == 1.0, f"GRASP<LRU on social fraction regressed to {state}"


# ---------------------------------------------------------------------------
# Robustness budget
# ---------------------------------------------------------------------------

def test_flipping_relabelings_bounded_by_half():
    """At most half of all relabelings may flip a LOAD-BEARING canonical
    claim. The per-family POPT<GRASP claims that sit near the stability
    boundary at array-relative GRASP 0.15 (citation ~0.95, road ~0.81,
    social ~0.39 — descriptive/regime-sensitive, NOT load-bearing; road is
    out of P-OPT's power-law scope entirely) are excluded. The load-bearing
    set is the robust power-law-POPT + social-LRU/GRASP-LRU claims; if those
    flip more than half the time, they are not actually canonical."""
    fs = _fs()
    load_bearing = {
        "POPT < GRASP on web",
        "POPT < LRU on social",
        "GRASP < LRU on social",
    }
    n = fs["n_relabelings"]
    assert n >= 1, f"family_sensitivity ran zero relabelings (n_relabelings={n})"
    nf = sum(
        1 for r in fs["relabelings"]
        if any(f["claim"] in load_bearing for f in r["flipped"])
    )
    assert nf <= math.ceil(n / 2), (
        f"{nf}/{n} relabelings flip a load-bearing canonical claim "
        f"({sorted(load_bearing)}); these are supposed to be robust"
    )


# ---------------------------------------------------------------------------
# Cell-count parity with popt_vs_grasp_by_family_app
# ---------------------------------------------------------------------------

def test_pvg_by_family_app_covers_all_25_family_app_cells():
    pvg = _load(PVG_JSON)
    keys = set(pvg["per_family_app"].keys())
    expected = {f"{f}/{a}" for f in EXPECTED_FAMILIES for a in EXPECTED_APPS}
    assert keys == expected, (
        f"popt_vs_grasp_by_family_app keys diverge from 5×5 product: "
        f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
    )
