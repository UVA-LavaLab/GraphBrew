"""Confidence gate 104 — family-aggregation tri-artifact agreement.

family_sensitivity.json (bootstrap relabeling stability of canonical
claims), family_geomean_improvement.json (geomean miss-ratio improvement
per (family, app, policy) vs LRU with bootstrap CI), and
family_policy_auc_clustering.json (per-family AUC-based winner
selection) are three downstream views of the same (graph, family) corpus
slice. If they fall out of sync — different family sets, different
apps/policies, or counted-vs-stated mismatches inside any one of them —
the paper's per-family policy story rests on inconsistent foundations.

The gate runs 13 invariants split 4/3/3/3 across each artifact's
internal hygiene plus a cross-artifact universe check.

Invariants:

  family_sensitivity internal (4):
    1. canonical_state[claim] equals canonical_claims[*].frac for the
       same claim text (within 1e-9)
    2. per_claim_flip_count keys == canonical_state keys (no orphan
       counts on either side)
    3. n_flipping_relabelings equals count of relabelings with
       flipped=True; sum of per-claim flip counts is at least the
       n_flipping (each relabeling flips ≥1 claim by definition)
    4. len(relabelings) == n_relabelings; every relabeling carries the
       4 required keys (canonical_family / new_family / graph / flipped)

  family_geomean internal (3):
    5. len(records) == meta.n_records
    6. counted records with ci_strict_improvement_vs_lru == True match
       meta.n_ci_strict_improvements; same for regressions
    7. every record has app ⊆ meta.apps, policy ⊆ {GRASP,POPT,SRRIP},
       family ⊆ meta.families; geomean_improve_pct is finite;
       n_cells > 0

  family_clustering internal (3):
    8. per_family keys == set(meta.families); meta.qualifying_families
       ⊆ meta.families and every qualifying family has qualified=True
    9. for every qualified family, winners_matching equals counted
       True values in winner_matches_global
   10. meta.global_winner_by_app keys ⊆ meta.apps and every winning
       policy is in meta.policies

  Cross-artifact universe (3):
   11. all three artifacts share the same family universe
       (citation, mesh, road, social, web)
   12. for every qualified family in clustering, every app/winner
       (family, winner_policy) combo has at least one matching record
       in family_geomean.records (the clustering pick must be a
       measurable point in the geomean panel)
   13. every family in family_sensitivity.canonical_claims is in
       family_clustering.meta.families; every policy referenced by
       canonical_claims (policy_a, policy_b) is in
       family_clustering.meta.policies
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FS_PATH = PROJECT_ROOT / "wiki" / "data" / "family_sensitivity.json"
FG_PATH = PROJECT_ROOT / "wiki" / "data" / "family_geomean_improvement.json"
FC_PATH = PROJECT_ROOT / "wiki" / "data" / "family_policy_auc_clustering.json"

EXPECTED_FAMILIES = frozenset({"citation", "mesh", "road", "social", "web"})
EXPECTED_APPS = frozenset({"bc", "bfs", "cc", "pr", "sssp"})
GEOMEAN_POLICIES = frozenset({"GRASP", "POPT", "SRRIP"})
SENSITIVITY_POLICIES = frozenset({"GRASP", "POPT", "LRU", "SRRIP"})
RELABEL_KEYS = frozenset({"canonical_family", "new_family", "graph", "flipped"})
FRAC_TOL = 1e-9


@pytest.fixture(scope="module")
def fs() -> dict:
    assert FS_PATH.exists(), f"missing {FS_PATH}"
    return json.loads(FS_PATH.read_text())


@pytest.fixture(scope="module")
def fg() -> dict:
    assert FG_PATH.exists(), f"missing {FG_PATH}"
    return json.loads(FG_PATH.read_text())


@pytest.fixture(scope="module")
def fc() -> dict:
    assert FC_PATH.exists(), f"missing {FC_PATH}"
    return json.loads(FC_PATH.read_text())


# ---------------------------------------------------------------------------
# family_sensitivity internal (4)
# ---------------------------------------------------------------------------


def test_fs_state_matches_claims(fs: dict) -> None:
    state = fs["canonical_state"]
    bad: list[tuple[str, float, float]] = []
    for cc in fs["canonical_claims"]:
        s = state.get(cc["claim"])
        if s is None or not math.isclose(s, cc["frac"], abs_tol=FRAC_TOL):
            bad.append((cc["claim"], cc["frac"], s))
    assert not bad, f"canonical_state drift from canonical_claims: {bad}"


def test_fs_flip_count_keys_match_state(fs: dict) -> None:
    state_keys = set(fs["canonical_state"].keys())
    flip_keys = set(fs["per_claim_flip_count"].keys())
    assert state_keys == flip_keys, (
        f"per_claim_flip_count keys != canonical_state keys; "
        f"only_in_state={state_keys - flip_keys}, only_in_flip={flip_keys - state_keys}"
    )


def test_fs_flipping_relabelings_count_matches(fs: dict) -> None:
    n_flip_field = fs["n_flipping_relabelings"]
    counted = sum(1 for r in fs["relabelings"] if r.get("flipped"))
    assert n_flip_field == counted, (
        f"n_flipping_relabelings={n_flip_field} != counted flipped={counted}"
    )
    flip_sum = sum(fs["per_claim_flip_count"].values())
    assert flip_sum >= n_flip_field, (
        f"sum(per_claim_flip_count)={flip_sum} < n_flipping_relabelings={n_flip_field}; "
        "each flipping relabeling must flip at least one claim"
    )


def test_fs_relabelings_length_and_schema(fs: dict) -> None:
    rel = fs["relabelings"]
    assert len(rel) == fs["n_relabelings"], (
        f"len(relabelings)={len(rel)} != n_relabelings={fs['n_relabelings']}"
    )
    bad: list[tuple[int, list[str]]] = []
    for i, r in enumerate(rel):
        missing = sorted(RELABEL_KEYS - set(r.keys()))
        if missing:
            bad.append((i, missing))
    assert not bad, f"relabelings missing keys (index, missing): {bad}"


# ---------------------------------------------------------------------------
# family_geomean internal (3)
# ---------------------------------------------------------------------------


def test_fg_record_count_matches_meta(fg: dict) -> None:
    assert len(fg["records"]) == fg["meta"]["n_records"], (
        f"len(records)={len(fg['records'])} != meta.n_records={fg['meta']['n_records']}"
    )


def test_fg_ci_strict_counts_match_meta(fg: dict) -> None:
    improve_counted = sum(
        1 for r in fg["records"] if r.get("ci_strict_improvement_vs_lru")
    )
    regress_counted = sum(
        1 for r in fg["records"] if r.get("ci_strict_regression_vs_lru")
    )
    assert improve_counted == fg["meta"]["n_ci_strict_improvements"], (
        f"counted improvements={improve_counted} != meta.n_ci_strict_improvements="
        f"{fg['meta']['n_ci_strict_improvements']}"
    )
    assert regress_counted == fg["meta"]["n_ci_strict_regressions"], (
        f"counted regressions={regress_counted} != meta.n_ci_strict_regressions="
        f"{fg['meta']['n_ci_strict_regressions']}"
    )


def test_fg_record_fields_well_typed(fg: dict) -> None:
    bad: list[tuple[str, str]] = []
    for i, r in enumerate(fg["records"]):
        if r["app"] not in EXPECTED_APPS:
            bad.append((f"rec[{i}].app", str(r["app"])))
        if r["policy"] not in GEOMEAN_POLICIES:
            bad.append((f"rec[{i}].policy", str(r["policy"])))
        if r["family"] not in EXPECTED_FAMILIES:
            bad.append((f"rec[{i}].family", str(r["family"])))
        v = r.get("geomean_improve_pct")
        skipped = r.get("skipped_reason")
        if skipped:
            # Skipped records may have geomean_improve_pct=None; that's fine
            if v is not None and not (isinstance(v, (int, float)) and math.isfinite(float(v))):
                bad.append((f"rec[{i}].geomean_improve_pct(skipped)", repr(v)))
        else:
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                bad.append((f"rec[{i}].geomean_improve_pct", repr(v)))
        if not (isinstance(r.get("n_cells"), int) and r["n_cells"] > 0):
            bad.append((f"rec[{i}].n_cells", repr(r.get("n_cells"))))
    assert not bad, f"family_geomean record field issues: {bad}"


# ---------------------------------------------------------------------------
# family_clustering internal (3)
# ---------------------------------------------------------------------------


def test_fc_per_family_matches_meta(fc: dict) -> None:
    pf_keys = set(fc["per_family"].keys())
    meta_fams = set(fc["meta"]["families"])
    assert pf_keys == meta_fams, (
        f"per_family keys != meta.families; only_in_pf={pf_keys - meta_fams}, "
        f"only_in_meta={meta_fams - pf_keys}"
    )
    qf = set(fc["meta"]["qualifying_families"])
    assert qf <= meta_fams, f"qualifying_families {qf} not subset of families {meta_fams}"
    bad_qual = [
        f for f in qf if not fc["per_family"][f].get("qualified")
    ]
    assert not bad_qual, (
        f"qualifying_families includes entries with qualified=False: {bad_qual}"
    )


def test_fc_winners_matching_count(fc: dict) -> None:
    bad: list[tuple[str, int, int]] = []
    for fam, info in fc["per_family"].items():
        if not info.get("qualified"):
            continue
        counted = sum(1 for v in info["winner_matches_global"].values() if v)
        if info["winners_matching"] != counted:
            bad.append((fam, info["winners_matching"], counted))
    assert not bad, f"winners_matching drift (family, stated, counted): {bad}"


def test_fc_global_winner_by_app_well_formed(fc: dict) -> None:
    gw = fc["meta"]["global_winner_by_app"]
    bad_apps = [a for a in gw if a not in EXPECTED_APPS]
    pols = set(fc["meta"]["policies"])
    bad_pols = [
        (a, p) for a, p in gw.items() if p not in pols
    ]
    assert not bad_apps, f"global_winner_by_app has unexpected apps: {bad_apps}"
    assert not bad_pols, f"global_winner_by_app picks unknown policy: {bad_pols}"


# ---------------------------------------------------------------------------
# Cross-artifact universe (3)
# ---------------------------------------------------------------------------


def test_all_three_share_family_universe(fs: dict, fg: dict, fc: dict) -> None:
    fs_fams = {cc["family"] for cc in fs["canonical_claims"]}
    fg_fams = set(fg["meta"]["families"]) if "families" in fg["meta"] else {
        r["family"] for r in fg["records"]
    }
    fc_fams = set(fc["meta"]["families"])
    assert fs_fams <= EXPECTED_FAMILIES, f"family_sensitivity has unexpected family: {fs_fams - EXPECTED_FAMILIES}"
    assert fg_fams == EXPECTED_FAMILIES, f"family_geomean families != EXPECTED: {fg_fams ^ EXPECTED_FAMILIES}"
    assert fc_fams == EXPECTED_FAMILIES, f"family_clustering families != EXPECTED: {fc_fams ^ EXPECTED_FAMILIES}"


def test_clustering_winners_present_in_geomean(fg: dict, fc: dict) -> None:
    geomean_records = {(r["family"], r["app"], r["policy"]) for r in fg["records"]}
    bad: list[tuple[str, str, str]] = []
    for fam, info in fc["per_family"].items():
        if not info.get("qualified"):
            continue
        for app, winner in info["winner_by_app"].items():
            if winner == "LRU":
                continue  # LRU is the reference policy; not present as a record
            key = (fam, app, winner)
            if key not in geomean_records:
                bad.append(key)
    assert not bad, (
        f"clustering winner picks not present in family_geomean.records: {bad}"
    )


def test_sensitivity_claims_universe_consistent(fs: dict, fc: dict) -> None:
    fc_families = set(fc["meta"]["families"])
    fc_policies = set(fc["meta"]["policies"])
    bad_fam: list[tuple[str, str]] = []
    bad_pol: list[tuple[str, str]] = []
    for cc in fs["canonical_claims"]:
        if cc["family"] not in fc_families:
            bad_fam.append((cc["claim"], cc["family"]))
        for k in ("policy_a", "policy_b"):
            pol = cc.get(k)
            if pol not in SENSITIVITY_POLICIES:
                bad_pol.append((cc["claim"], f"{k}={pol}"))
            if pol not in fc_policies:
                bad_pol.append((cc["claim"], f"{k}={pol} not in clustering policies"))
    assert not bad_fam, f"sensitivity claims reference unknown family: {bad_fam}"
    assert not bad_pol, f"sensitivity claims reference unknown policy: {bad_pol}"
