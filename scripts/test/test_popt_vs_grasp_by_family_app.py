"""Pytest gate: POPT vs GRASP per (family × app) bootstrap CIs.

Pins the per-(family, app) cut of POPT-vs-GRASP on the reproducible
single-thread, array-relative-GRASP (0.15), faithful 1-way-charged
P-OPT corpus. The split is APP-DRIVEN, not family-driven (the older
"road is uniformly POPT-favored" framing was a multi-thread-corpus
artifact and is also out of P-OPT's literature scope, which is
power-law only — see the POPT_GE_GRASP_GEOMEAN gate). Road POPT-vs-GRASP
is retained here only as a descriptive breakdown.

Load-bearing findings (ST, 0.15, charged P-OPT):

* `pr` is POPT-favored on EVERY family by mean (mean Δ < 0 on
  citation/social/web/road/mesh) — the PR bedrock. P-OPT's static
  PR-rank rereference schedule matches PageRank's all-vertex reuse.
  CI-strict (P ≥ 0.99) on citation/web/mesh; the 1-way charge narrowed
  social/pr to a mean-only win (P = 0.87, CI spans 0).
* `bc` is the clean GRASP counter-kernel: GRASP-favored on every family
  (mean Δ > 0), CI-strict on citation/social/web (P ≈ 0, CI lo > 0).
  BC is frontier-driven and misaligns with P-OPT's PR-rank schedule.
* `cc`/`bfs`/`sssp` are GRAPH-DEPENDENT, not uniformly GRASP-favored:
  citation tends GRASP (citation/sssp, social/sssp are GRASP-strict),
  while web tends strongly POPT (web/cc = −11.0 pp, web/bfs = −9.0 pp).
  NOTE: the earlier "cc is uniformly GRASP-favored / edge-driven" claim
  was based on a STALE multi-thread GRASP/cc number (committed
  soc-pokec/cc/1MB GRASP = 0.604; corrected deterministic = 0.651,
  verified by a 3rd byte-identical re-run) and is RETIRED — only
  citation/cc still leans GRASP.
* `web/bfs` (−9.0 pp) and `road/sssp` (−10.1 pp) are the largest POPT
  effects; road/sssp has very high variance (CI spans 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_JSON = REPO_ROOT / "wiki" / "data" / "popt_vs_grasp_by_family_app.json"

STRONG_FLOOR = 0.99
STABILITY_FLOOR = 0.95
ALL_FAMILIES = ("citation", "social", "web", "road", "mesh")
# pr families that remain CI-strict POPT wins after the 1-way charge
# (social/pr narrowed to a mean-only win, P = 0.87).
PR_CI_STRICT_FAMILIES = ("citation", "web", "mesh")
# bc families that are CI-strict GRASP wins (the clean counter-kernel).
BC_GRASP_STRICT_FAMILIES = ("citation", "social", "web")


@pytest.fixture(scope="module")
def doc() -> dict:
    if not DOC_JSON.exists():
        pytest.skip(
            f"{DOC_JSON} missing; run `make lit-popt-vs-grasp-by-family-app`"
        )
    return json.loads(DOC_JSON.read_text())


def test_schema(doc):
    assert "meta" in doc and "per_family_app" in doc
    m = doc["meta"]
    for k in ("n_resamples", "seed", "ci_level", "families", "apps",
              "cells_with_data"):
        assert k in m, f"missing meta.{k}"
    assert m["n_resamples"] >= 1000


def test_min_cells_with_data(doc):
    """Need at least 20 (family, app) cells with paired data — below
    that the matrix becomes too sparse to defend per-family kernel
    breakdowns."""
    assert doc["meta"]["cells_with_data"] >= 20, (
        f"only {doc['meta']['cells_with_data']} cells with data"
    )


def test_pr_is_popt_favored_on_every_family(doc):
    """PR is the bedrock POPT win: every family's pr cell must have
    mean Δ < 0. P-OPT's static PR-rank rereference schedule matches
    PageRank's all-vertex reuse, so POPT wins by mean across
    citation/social/web/road/mesh even under the 1-way charge."""
    rogues = []
    for fam in ALL_FAMILIES:
        key = f"{fam}/pr"
        r = doc["per_family_app"].get(key, {})
        if r.get("n_paired", 0) == 0:
            pytest.fail(f"{key} has no paired data")
        md = r["mean_delta"]
        if md is None or md >= 0:
            rogues.append((key, md))
    assert not rogues, (
        f"pr cells not POPT-favored by mean (mean Δ<0): {rogues}. "
        "The PR-bedrock POPT claim no longer holds across families."
    )


def test_pr_ci_strict_families_after_charge(doc):
    """citation/pr, web/pr and mesh/pr remain CI-strict POPT wins
    (P ≥ 0.95) after the 1-way charge. social/pr narrowed to a
    mean-only win (covered by test_social_pr_is_popt_by_mean)."""
    rogues = []
    for fam in PR_CI_STRICT_FAMILIES:
        key = f"{fam}/pr"
        r = doc["per_family_app"][key]
        p = r["p_popt_lt_grasp"]
        if p is None or p < STABILITY_FLOOR:
            rogues.append((key, p))
    assert not rogues, (
        f"pr CI-strict families dropped below {STABILITY_FLOOR}: {rogues}"
    )


def test_largest_popt_effects_are_huge(doc):
    """web/bfs and road/sssp are the largest POPT effects; pin both
    mean ≤ -8 pp. (road/sssp ≈ -10.1 pp but high-variance: its CI
    spans 0, so we pin the mean magnitude, not CI strictness.)"""
    for key, floor in (("web/bfs", -8.0), ("road/sssp", -8.0)):
        r = doc["per_family_app"][key]
        assert r["mean_delta"] is not None and r["mean_delta"] <= floor, (
            f"{key} mean Δ = {r['mean_delta']}; expected ≤ {floor} pp"
        )


def test_pr_high_signal_cells_are_ci_strict(doc):
    """citation/pr, web/pr and mesh/pr are CI-strict POPT wins
    (P ≥ 0.99 and CI excludes 0 on the POPT-better side)."""
    for key in ("citation/pr", "web/pr", "mesh/pr"):
        r = doc["per_family_app"][key]
        p = r["p_popt_lt_grasp"]
        assert p is not None and p >= STRONG_FLOOR, (
            f"{key} P(POPT<GRASP)={p} < {STRONG_FLOOR}"
        )
        assert r["ci_hi"] is not None and r["ci_hi"] < 0, (
            f"{key} CI hi = {r['ci_hi']}; does not exclude 0"
        )


def test_bc_is_grasp_strict_across_families(doc):
    """BC is the clean GRASP counter-kernel: citation/bc, social/bc and
    web/bc are GRASP-strict (P ≈ 0.000, CI lo > 0). BC is frontier-driven
    and misaligns with P-OPT's PR-rank rereference schedule. This is the
    GRASP counter-narrative under the corrected deterministic corpus
    (it replaces the retired uniform-cc claim)."""
    for fam in BC_GRASP_STRICT_FAMILIES:
        key = f"{fam}/bc"
        r = doc["per_family_app"][key]
        p = r["p_popt_lt_grasp"]
        assert p is not None and p <= (1.0 - STRONG_FLOOR), (
            f"{key} P(POPT<GRASP)={p} > {1.0 - STRONG_FLOOR}; "
            "the bc GRASP counter-narrative is no longer cell-strict"
        )
        assert r["ci_lo"] is not None and r["ci_lo"] > 0, (
            f"{key} CI lo = {r['ci_lo']}; does not exclude 0 "
            "from the GRASP-better side"
        )


def test_cc_is_graph_dependent_not_uniform_grasp(doc):
    """The corrected deterministic GRASP/cc data RETIRES the old
    "cc is uniformly GRASP-favored" claim: only citation/cc still leans
    GRASP while web/cc is strongly POPT (≤ -8 pp). cc is graph-dependent."""
    citation_cc = doc["per_family_app"]["citation/cc"]
    web_cc = doc["per_family_app"]["web/cc"]
    assert citation_cc["mean_delta"] is not None and citation_cc["mean_delta"] > 0, (
        f"citation/cc mean Δ = {citation_cc['mean_delta']}; expected GRASP-leaning (>0)"
    )
    assert web_cc["mean_delta"] is not None and web_cc["mean_delta"] <= -8.0, (
        f"web/cc mean Δ = {web_cc['mean_delta']}; expected strongly POPT (≤ -8 pp)"
    )


def test_social_pr_is_popt_by_mean(doc):
    """social/pr is POPT-favored by mean (mean Δ < 0) but the 1-way
    charge narrowed it below CI-strict (P = 0.87, CI spans 0). This is
    the charge's clearest narrowing of the PR-bedrock on the largest
    pr sample (n = 12 social cells)."""
    r = doc["per_family_app"]["social/pr"]
    md = r["mean_delta"]
    p = r["p_popt_lt_grasp"]
    assert md is not None and md < 0, (
        f"social/pr mean Δ = {md}; expected POPT-favored (<0)"
    )
    assert p is not None and p >= 0.80, (
        f"social/pr P(POPT<GRASP) = {p}; expected POPT-leaning (≥ 0.80)"
    )


def test_no_cell_has_implausible_n_paired(doc):
    """Sanity bound: no cell should report n_paired above 50 (would
    indicate row joins exploded)."""
    rogues = [
        (k, v["n_paired"]) for k, v in doc["per_family_app"].items()
        if v["n_paired"] > 50
    ]
    assert not rogues, f"implausible n_paired: {rogues}"


def test_all_families_appear_in_keys(doc):
    """Every meta.families entry should appear as a key prefix."""
    seen_families = {k.split("/", 1)[0] for k in doc["per_family_app"].keys()}
    missing = set(doc["meta"]["families"]) - seen_families
    assert not missing, f"families missing from keys: {missing}"


def test_all_apps_appear_in_keys(doc):
    """Every meta.apps entry should appear as a key suffix."""
    seen_apps = {k.split("/", 1)[1] for k in doc["per_family_app"].keys()}
    missing = set(doc["meta"]["apps"]) - seen_apps
    assert not missing, f"apps missing from keys: {missing}"
