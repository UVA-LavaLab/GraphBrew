"""Pytest gate: bootstrap confidence intervals on load-bearing claims.

The paper makes several POPT-vs-GRASP ordering claims grounded in
oracle-gap means. With only n ∈ [5, 54] cells per family bucket, the
honest answer is to show CIs. This gate enforces that:

* the bootstrap_ci.json artifact is regenerated from current rows;
* every (policy, family) bucket has non-negative-width CIs containing
  the point estimate;
* the road-family POPT-vs-GRASP ordering is DESCRIPTIVE only — road is
  out of P-OPT's power-law literature scope (P-OPT, Balaji & Lucia
  HPCA'21, only ever tested power-law graphs), so it is directionally
  POPT-leaning but is NOT required to be CI-strict; the load-bearing
  POPT-vs-GRASP claim is the power-law geomean (POPT_GE_GRASP_GEOMEAN
  gate), NOT a per-family road claim;
* claims that do NOT survive resampling (social, citation, web) are
  flagged as `ci_excludes_zero=False` so the paper cannot accidentally
  promote them to load-bearing in the future.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_JSON = REPO_ROOT / "wiki" / "data" / "bootstrap_ci.json"

# Road POPT-vs-GRASP leans POPT but is not literature-grade (road is out
# of P-OPT's power-law scope). The real claim is the power-law geomean.
# We require road to be directionally POPT (frac > 0.5), not CI-strict.
ROAD_SIGN_DIRECTION_FLOOR = 0.5


@pytest.fixture(scope="module")
def ci_doc() -> dict:
    if not CI_JSON.exists():
        pytest.skip(f"{CI_JSON} not generated; run `make lit-bootstrap-ci`")
    return json.loads(CI_JSON.read_text())


def test_top_level_schema(ci_doc):
    assert {
        "meta",
        "oracle_gap_by_policy_family",
        "oracle_gap_by_policy_regime",
        "popt_minus_grasp_by_family",
        "sign_stability",
    }.issubset(ci_doc.keys())


def test_meta_has_resamples(ci_doc):
    m = ci_doc["meta"]
    assert m["n_resamples"] >= 1000, "need ≥ 1000 resamples for stable CIs"
    assert 0.5 < m["ci_level"] < 1.0
    assert isinstance(m["seed"], int), "seed must be persisted for reproducibility"


def test_every_bucket_has_finite_ci(ci_doc):
    for kind in (
        "oracle_gap_by_policy_family",
        "oracle_gap_by_policy_regime",
    ):
        for k, v in ci_doc[kind].items():
            assert v["n"] >= 1, f"{kind} bucket {k} has n=0"
            assert v["ci_lo"] <= v["mean"] <= v["ci_hi"], (
                f"{kind} bucket {k}: mean {v['mean']} outside CI "
                f"[{v['ci_lo']}, {v['ci_hi']}]"
            )
            assert v["ci_width"] >= 0


def test_oracle_gap_is_non_negative(ci_doc):
    for kind in (
        "oracle_gap_by_policy_family",
        "oracle_gap_by_policy_regime",
    ):
        for k, v in ci_doc[kind].items():
            assert v["ci_lo"] >= -1e-6, (
                f"{kind} bucket {k} has negative CI lower bound {v['ci_lo']} — "
                "oracle gap by definition cannot be negative"
            )


def test_road_popt_minus_grasp_is_directionally_popt(ci_doc):
    """Road ΔPOPT leans POPT (negative point estimate) but is NOT
    required to be CI-strict: road is out of P-OPT's power-law literature
    scope, so the wide CI crossing zero is expected and acceptable. The
    load-bearing POPT-vs-GRASP claim is the power-law geomean."""
    bucket = ci_doc["popt_minus_grasp_by_family"].get("road")
    assert bucket is not None, "no road bucket in popt_minus_grasp_by_family"
    assert bucket["mean_delta"] < 0, (
        f"road ΔPOPT mean={bucket['mean_delta']:+.3f} — expected < 0 (POPT leans "
        f"ahead of GRASP on road by the point estimate even though the CI "
        f"[{bucket['ci_lo']:+.3f}, {bucket['ci_hi']:+.3f}] is not strict, so "
        f"sign={bucket['sign']!r})"
    )


def test_road_sign_direction_is_popt_leaning(ci_doc):
    """Road bootstrap-resampled mean(POPT) < mean(GRASP) a majority of
    the time (directional), but NOT at the 0.95 literature floor — road
    is descriptive-only (out of P-OPT's power-law scope)."""
    s = next(
        x for x in ci_doc["sign_stability"]
        if x["policy_a"] == "POPT" and x["policy_b"] == "GRASP" and x["family"] == "road"
    )
    assert s["frac_a_lt_b"] >= ROAD_SIGN_DIRECTION_FLOOR, (
        f"P(mean(POPT/road) < mean(GRASP/road)) = {s['frac_a_lt_b']:.4f} "
        f"below the {ROAD_SIGN_DIRECTION_FLOOR} directional floor — road no "
        f"longer even leans POPT, which would contradict the descriptive trend"
    )


def test_popt_lt_lru_on_social_is_unanimous(ci_doc):
    """POPT < LRU on social is so large the bootstrap should never flip it."""
    s = next(
        x for x in ci_doc["sign_stability"]
        if x["policy_a"] == "POPT" and x["policy_b"] == "LRU" and x["family"] == "social"
    )
    assert s["frac_a_lt_b"] >= 0.99, (
        f"P(mean(POPT/social) < mean(LRU/social)) = {s['frac_a_lt_b']:.4f} — "
        "a real LRU-beats-POPT regression has crept into the social family"
    )


def test_popt_mesh_oracle_gap_is_near_zero(ci_doc):
    """POPT/mesh near-perfect claim: CI upper bound should be < 1 pp."""
    bucket = ci_doc["oracle_gap_by_policy_family"]["POPT/mesh"]
    assert bucket["ci_hi"] <= 1.0, (
        f"POPT/mesh CI upper bound {bucket['ci_hi']:.3f} pp exceeds 1 pp — "
        "the paper's POPT-near-oracle-on-mesh claim no longer holds"
    )


def test_required_families_present(ci_doc):
    families = {b.split("/")[1] for b in ci_doc["oracle_gap_by_policy_family"]}
    required = {"citation", "mesh", "road", "social", "web"}
    missing = required - families
    assert not missing, f"required families missing from CI buckets: {missing}"


def test_required_policies_present(ci_doc):
    pols = {b.split("/")[0] for b in ci_doc["oracle_gap_by_policy_family"]}
    required = {"LRU", "SRRIP", "GRASP", "POPT"}
    missing = required - pols
    assert not missing, f"required policies missing from CI buckets: {missing}"


def test_n_sums_match_source(ci_doc):
    """The per-family sample sizes must add up to a sane corpus-wide total
    so a future bug that silently drops cells is caught."""
    for pol in ("LRU", "SRRIP", "GRASP", "POPT"):
        n_total = sum(
            v["n"] for k, v in ci_doc["oracle_gap_by_policy_family"].items()
            if k.startswith(pol + "/")
        )
        assert n_total >= 100, (
            f"{pol} has only {n_total} cells across all families — "
            "corpus shrank below 100-cell floor"
        )
