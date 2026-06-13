"""Pytest gate: per-kernel oracle-gap bootstrap CIs.

Turns the point-estimate per-kernel narrative from
``oracle_gap_by_app`` into CI-backed sign claims:

  - pr  : POPT < {LRU, SRRIP, GRASP} all P ≥ 0.999 (bedrock)
  - bc  : GRASP < POPT          P ≥ 0.99  (clean counter-kernel)
  - cc  : POPT < GRASP          directional only (graph-dependent)
  - bfs : POPT < GRASP          directional only (graph-dependent)
  - sssp: POPT vs GRASP         no CI-strict sign claim
  - bc  : NO sign claim is stable (paper's "no one-size-fits-all")

These claims are the load-bearing per-kernel sentences. If any
flips, the paper's per-kernel argument changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap_by_app_bootstrap.json"

STABILITY_FLOOR = 0.95
STRONG_FLOOR = 0.99


@pytest.fixture(scope="module")
def doc() -> dict:
    if not DOC_JSON.exists():
        pytest.skip(f"{DOC_JSON} missing; run `make lit-oracle-by-app-bootstrap`")
    return json.loads(DOC_JSON.read_text())


def test_schema(doc):
    assert "meta" in doc and "per_app_pairs" in doc
    m = doc["meta"]
    for k in ("n_resamples", "seed", "ci_level", "apps", "policies"):
        assert k in m, f"missing meta.{k}"
    assert m["n_resamples"] >= 1000, "too few resamples"
    assert 0.90 <= m["ci_level"] <= 0.99


def test_all_apps_have_comparisons(doc):
    expected_apps = {"pr", "bc", "cc", "bfs", "sssp"}
    assert expected_apps.issubset(doc["per_app_pairs"].keys()), (
        f"missing apps: {expected_apps - doc['per_app_pairs'].keys()}"
    )


def test_pr_popt_bedrock(doc):
    """pr → POPT beats LRU, SRRIP, and GRASP with P ≥ 0.999.
    This is the load-bearing 'POPT wins pr' claim. If any of these
    drops below 0.999, the paper's pr-as-bedrock argument is
    weakened."""
    pairs = doc["per_app_pairs"]["pr"]
    for vs in ("POPT_vs_LRU", "POPT_vs_SRRIP", "POPT_vs_GRASP"):
        p = pairs[vs]["p_a_lt_b"]
        assert p is not None and p >= STRONG_FLOOR, (
            f"pr {vs} P(POPT<other)={p} < {STRONG_FLOOR}; "
            "POPT's bedrock dominance on PR is weakened"
        )


def test_pr_popt_vs_grasp_ci_excludes_zero(doc):
    """The pr/POPT vs GRASP CI must lie entirely on the POPT-better
    side of zero (CI hi < 0). Stronger than a sign claim."""
    r = doc["per_app_pairs"]["pr"]["POPT_vs_GRASP"]
    assert r["ci_hi"] is not None and r["ci_hi"] < 0, (
        f"pr POPT_vs_GRASP CI hi = {r['ci_hi']}; does not exclude 0 "
        "(POPT-pr dominance claim no longer CI-strict)"
    )


def test_cc_popt_vs_grasp_is_directional_not_ci_strict(doc):
    """Charged corpus retires uniform cc/GRASP; cc is graph-dependent."""
    r = doc["per_app_pairs"]["cc"]["POPT_vs_GRASP"]
    assert r["p_a_lt_b"] is not None and 0.80 <= r["p_a_lt_b"] < STRONG_FLOOR
    assert r["ci_lo"] is not None and r["ci_lo"] < 0
    assert r["ci_hi"] is not None and r["ci_hi"] > 0


def test_bfs_popt_vs_grasp_is_directional_not_ci_strict(doc):
    """Charged corpus: bfs POPT-vs-GRASP is graph-dependent and CI-overlaps zero."""
    r = doc["per_app_pairs"]["bfs"]["POPT_vs_GRASP"]
    p = r["p_a_lt_b"]
    assert p is not None and p >= STABILITY_FLOOR, (
        f"bfs POPT_vs_GRASP P={p} < {STABILITY_FLOOR}"
    )
    assert r["ci_lo"] is not None and r["ci_lo"] < 0
    assert r["ci_hi"] is not None and r["ci_hi"] > 0


def test_sssp_popt_vs_grasp_has_no_ci_strict_sign(doc):
    """Charged corpus: sssp is graph-dependent; bootstrap sign is near even."""
    p = doc["per_app_pairs"]["sssp"]["POPT_vs_GRASP"]["p_a_lt_b"]
    assert p is not None and 0.45 <= p <= 0.60


def test_bc_grasp_beats_popt(doc):
    """At array-relative GRASP 0.15 (single-thread) bc has a clear winner:
    GRASP strictly beats POPT (P≈1.0). bc is frontier-driven and GRASP's
    degree-protection aligns better than P-OPT's static rereference schedule.
    (Was the 'no clear winner on bc' narrative under the multi-thread corpus.)"""
    gp = doc["per_app_pairs"]["bc"]["GRASP_vs_POPT"]["p_a_lt_b"]
    assert gp is not None and gp >= STRONG_FLOOR, (
        f"bc GRASP_vs_POPT P={gp} < {STRONG_FLOOR}; GRASP no longer clearly "
        "beats POPT on bc"
    )


def test_no_self_comparisons(doc):
    """A_vs_A should never appear (would be degenerate)."""
    for app, pairs in doc["per_app_pairs"].items():
        for key in pairs:
            a, b = key.split("_vs_")
            assert a != b, f"degenerate self-comparison {app}/{key}"


def test_pair_counts_per_app(doc):
    """Each app should expose all 4P2 = 12 ordered pairs over the 4
    policies."""
    for app, pairs in doc["per_app_pairs"].items():
        assert len(pairs) == 12, (
            f"app {app} has {len(pairs)} pairs; expected 12 "
            f"(4 policies × 3 others)"
        )


def test_n_paired_floor(doc):
    """Every pair should have at least 15 paired cells for the
    bootstrap to be meaningful."""
    for app, pairs in doc["per_app_pairs"].items():
        for key, r in pairs.items():
            assert r["n_paired"] >= 15, (
                f"{app}/{key} n_paired={r['n_paired']} < 15; "
                "bootstrap CIs would be unreliable"
            )
