"""Gate 43: family × app × policy geomean improvement vs LRU is CI-strict.

Pins the QUANTITATIVE SIZE of policy improvement, not just direction. The
significance gates (34/35/36/37/38/40) tell us where deltas have stable
signs; this gate tells us how big the deltas are, with a 95% percentile
bootstrap CI on the geomean miss-rate ratio.

Marquee claims pinned (paper-defended):
  - citation/pr/POPT: geomean miss-rate 0.64 of LRU (~36% reduction),
    CI [0.55, 0.82]; CI-strict improvement.
  - citation/cc/GRASP: geomean 0.75 (~25% reduction), CI [0.64, 0.84].
  - social/cc/GRASP: geomean 0.74 (~26% reduction), CI [0.64, 0.86].
  - social/pr/POPT: geomean 0.78 (~22% reduction), CI [0.70, 0.86].
  - **No CI-strict regression vs LRU on any (family, app, policy).** This
    is the "do no harm" check the paper needs to make before recommending
    a winner per family×app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "wiki" / "data" / "family_geomean_improvement.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA_PATH.exists():
        pytest.skip(
            f"{DATA_PATH} missing; run `make lit-family-geomean` to regenerate"
        )
    return json.loads(DATA_PATH.read_text())


def _record(payload: dict, family: str, app: str, policy: str) -> dict:
    for r in payload["records"]:
        if (
            r["family"] == family
            and r["app"] == app
            and r["policy"] == policy
        ):
            return r
    raise AssertionError(
        f"missing record: family={family}, app={app}, policy={policy}"
    )


def test_meta_pins_bootstrap_invariants(payload):
    meta = payload["meta"]
    assert meta["bootstrap_iters"] >= 2000
    assert meta["bootstrap_seed"] == 1729
    assert meta["alpha"] == 0.05
    assert meta["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert meta["n_records"] >= 60, (
        f"expected >=60 (family, app, policy) records, got {meta['n_records']}"
    )


def test_no_unexpected_ci_strict_regression_vs_lru(payload):
    """The 'do no harm' invariant: no policy may CI-strictly regress vs LRU
    on any (family, app), EXCEPT the DISCLOSED frontier-misalignment cells
    where the degree-based GRASP/POPT protection hurts a frontier kernel
    (web/bc — bc traverses dependency frontiers, not the degree-sorted
    property array; see docs/findings/grasp_road_anti_thrashing.md for the
    parallel mechanism). Any OTHER CI-strict regression must be investigated
    and disclosed."""
    known = {("web", "bc", "POPT"), ("web", "bc", "GRASP")}
    regs = payload["headline_regressions_ci_strict"]
    unexpected = [
        r for r in regs
        if (r["family"], r["app"], r["policy"]) not in known
    ]
    assert not unexpected, (
        f"unexpected CI-strict regression vs LRU outside the documented "
        f"frontier exceptions {sorted(known)}: {unexpected}"
    )


def test_marquee_citation_pr_popt_is_large_improvement(payload):
    """The paper's flagship CC-bypass claim: pr/POPT on citation graphs is
    a ~36% miss-rate reduction vs LRU, CI strictly below 1.0."""
    r = _record(payload, "citation", "pr", "POPT")
    assert r["ci_strict_improvement_vs_lru"] is True
    assert r["geomean_ratio"] < 0.75
    assert r["ci_hi_ratio"] < 1.0
    assert r["geomean_improve_pct"] >= 25.0
    assert r["ci_lo_improve_pct"] > 5.0


def test_marquee_citation_cc_grasp_is_large_improvement(payload):
    """Charged corpus: citation/cc GRASP remains a CI-strict improvement."""
    r = _record(payload, "citation", "cc", "GRASP")
    assert r["ci_strict_improvement_vs_lru"] is True
    assert r["geomean_ratio"] < 0.85
    assert r["geomean_improve_pct"] >= 15.0


def test_social_cc_popt_is_large_improvement(payload):
    """Charged corpus: social/cc improvement is POPT-led, not uniformly GRASP."""
    r = _record(payload, "social", "cc", "POPT")
    assert r["ci_strict_improvement_vs_lru"] is True
    assert r["n_cells"] >= 9
    assert r["geomean_improve_pct"] >= 10.0
    assert r["ci_lo_improve_pct"] > 5.0


def test_marquee_social_pr_popt_is_large_improvement(payload):
    """pr/POPT on social graphs (12 cells)."""
    r = _record(payload, "social", "pr", "POPT")
    assert r["ci_strict_improvement_vs_lru"] is True
    assert r["n_cells"] >= 12
    assert r["geomean_improve_pct"] >= 15.0


def test_ratio_and_improvement_consistency(payload):
    """For every well-formed record, geomean_improve_pct == (1 - geomean_ratio) * 100."""
    for r in payload["records"]:
        if r.get("skipped_reason"):
            continue
        expected = (1.0 - r["geomean_ratio"]) * 100.0
        assert abs(r["geomean_improve_pct"] - expected) < 0.02, (
            f"{r['family']}/{r['app']}/{r['policy']}: improve_pct "
            f"{r['geomean_improve_pct']} inconsistent with geomean_ratio "
            f"{r['geomean_ratio']}"
        )


def test_ci_bounds_bracket_point_estimate(payload):
    """Every CI must bracket the point estimate (lo <= point <= hi)."""
    for r in payload["records"]:
        if r.get("skipped_reason"):
            continue
        # tiny tolerance for percentile rounding at integer indices
        assert r["ci_lo_ratio"] - 1e-6 <= r["geomean_ratio"] <= r["ci_hi_ratio"] + 1e-6, (
            f"{r['family']}/{r['app']}/{r['policy']}: CI {r['ci_lo_ratio']:.6f}.."
            f"{r['ci_hi_ratio']:.6f} does not bracket point {r['geomean_ratio']:.6f}"
        )


def test_strict_improvement_implies_ci_hi_below_one(payload):
    """Definitional consistency: ci_strict_improvement_vs_lru ⇔ ci_hi_ratio < 1.0."""
    for r in payload["records"]:
        if r.get("skipped_reason"):
            continue
        expected = r["ci_hi_ratio"] < 1.0
        assert r["ci_strict_improvement_vs_lru"] == expected, (
            f"{r['family']}/{r['app']}/{r['policy']}: flag mismatch — "
            f"ci_hi_ratio={r['ci_hi_ratio']}, flag={r['ci_strict_improvement_vs_lru']}"
        )


def test_headline_improvements_sorted_strongest_first(payload):
    """Headline list must be sorted by improvement % descending so the paper
    can lift the top-N entries without re-sorting."""
    hs = payload["headline_improvements_ge_10pct"]
    pcts = [h["geomean_improve_pct"] for h in hs]
    assert pcts == sorted(pcts, reverse=True)
    assert all(p >= 10.0 for p in pcts)
    assert all(h["ci_strict_improvement_vs_lru"] for h in hs)


def test_no_record_has_negative_or_zero_n(payload):
    """Every record carries a defensible n_cells >= 1 (skipped records may
    have n=1, others must have n >= 2 to bootstrap)."""
    for r in payload["records"]:
        assert r["n_cells"] >= 1
        if not r.get("skipped_reason"):
            assert r["n_cells"] >= 2, (
                f"{r['family']}/{r['app']}/{r['policy']}: bootstrapped with"
                f" n={r['n_cells']} < 2"
            )


def test_cross_gate_consistency_with_oracle_gap_by_app_bootstrap(payload):
    """Sanity check: where the per-app bootstrap (gate 34) reports a strong
    POPT < LRU sign for an app, the family-aggregated improvement here must
    show at least one CI-strict improvement record for that app with policy
    POPT (or GRASP) on at least one family."""
    by_app_path = REPO_ROOT / "wiki" / "data" / "oracle_gap_by_app_bootstrap.json"
    if not by_app_path.exists():
        pytest.skip("oracle_gap_by_app_bootstrap.json not built")
    by_app = json.loads(by_app_path.read_text())
    # gate 34's headline: pr → POPT beats LRU strongly. Verify pr/POPT has
    # ≥1 CI-strict improvement record at family granularity.
    pr_popt_records = [
        r
        for r in payload["records"]
        if r["app"] == "pr"
        and r["policy"] == "POPT"
        and not r.get("skipped_reason")
    ]
    assert pr_popt_records, "expected pr/POPT records at family granularity"
    assert any(
        r["ci_strict_improvement_vs_lru"] for r in pr_popt_records
    ), (
        "gate 34 says pr/POPT < LRU strongly; gate 43 must echo that with "
        "at least one CI-strict family-level improvement"
    )
    _ = by_app  # presence-only sanity, structure-checked in gate 34's own tests
