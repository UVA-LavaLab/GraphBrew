"""Pytest gate: POPT vs GRASP per (family × app) bootstrap CIs.

Pins the per-(family, app) cut of POPT-vs-GRASP on the reproducible
single-thread, array-relative-GRASP (0.15) corpus. The split is
APP-DRIVEN, not family-driven (the older "road is uniformly POPT-
favored" framing was a multi-thread-corpus artifact and is also out of
P-OPT's literature scope, which is power-law only — see the
POPT_GE_GRASP_GEOMEAN gate). Road POPT-vs-GRASP is retained here only as
a descriptive breakdown.

Load-bearing findings (ST, 0.15):

* `pr` is POPT-favored on EVERY family (P(POPT<GRASP) ≥ 0.95 on
  citation/social/web/road/mesh) — the PR bedrock. P-OPT's static
  PR-rank rereference schedule matches PageRank's all-vertex reuse.
* `cc` is GRASP-favored and CI-strict on `social/cc` and `road/cc`
  (P ≈ 0): CC's union-find is edge-driven and misaligns with P-OPT's
  PR-rank schedule (the cc-counter-narrative).
* `bc` is GRASP-favored across families (mean Δ > 0 on
  citation/social/web/road) — frontier-driven, like cc.
* `web/bfs` (−10.6 pp) and `road/sssp` (−11.7 pp) are the largest POPT
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
    mean Δ < 0 AND P(POPT<GRASP) ≥ 0.95. P-OPT's static PR-rank
    rereference schedule matches PageRank's all-vertex reuse, so this
    holds across citation/social/web/road/mesh."""
    rogues = []
    for fam in ALL_FAMILIES:
        key = f"{fam}/pr"
        r = doc["per_family_app"].get(key, {})
        if r.get("n_paired", 0) == 0:
            pytest.fail(f"{key} has no paired data")
        md = r["mean_delta"]
        p = r["p_popt_lt_grasp"]
        if md is None or md >= 0 or p is None or p < STABILITY_FLOOR:
            rogues.append((key, md, p))
    assert not rogues, (
        f"pr cells not POPT-favored (mean Δ<0 AND P≥{STABILITY_FLOOR}): {rogues}. "
        "The PR-bedrock POPT claim no longer factorises across families."
    )


def test_largest_popt_effects_are_huge(doc):
    """web/bfs and road/sssp are the largest POPT effects; pin both
    mean ≤ -8 pp. (road/sssp ≈ -11.7 pp but high-variance: its CI
    spans 0, so we pin the mean magnitude, not CI strictness.)"""
    for key, floor in (("web/bfs", -8.0), ("road/sssp", -8.0)):
        r = doc["per_family_app"][key]
        assert r["mean_delta"] is not None and r["mean_delta"] <= floor, (
            f"{key} mean Δ = {r['mean_delta']}; expected ≤ {floor} pp"
        )


def test_pr_high_signal_cells_are_ci_strict(doc):
    """citation/pr, social/pr and web/pr are CI-strict POPT wins
    (P ≥ 0.99 and CI excludes 0 on the POPT-better side)."""
    for key in ("citation/pr", "social/pr", "web/pr"):
        r = doc["per_family_app"][key]
        p = r["p_popt_lt_grasp"]
        assert p is not None and p >= STRONG_FLOOR, (
            f"{key} P(POPT<GRASP)={p} < {STRONG_FLOOR}"
        )
        assert r["ci_hi"] is not None and r["ci_hi"] < 0, (
            f"{key} CI hi = {r['ci_hi']}; does not exclude 0"
        )


def test_cc_is_grasp_strict_on_social_and_road(doc):
    """social/cc and road/cc are GRASP-strict (P ≈ 0.000, CI lo > 0).
    CC's union-find is edge-driven and misaligns with P-OPT's PR-rank
    rereference schedule — the cc-counter-narrative, cell-by-cell."""
    for key in ("social/cc", "road/cc"):
        r = doc["per_family_app"][key]
        p = r["p_popt_lt_grasp"]
        assert p is not None and p <= (1.0 - STRONG_FLOOR), (
            f"{key} P(POPT<GRASP)={p} > {1.0 - STRONG_FLOOR}; "
            "cc-counter-narrative is no longer cell-by-cell strict"
        )
        assert r["ci_lo"] is not None and r["ci_lo"] > 0, (
            f"{key} CI lo = {r['ci_lo']}; does not exclude 0 "
            "from the GRASP-better side"
        )


def test_social_pr_is_popt_ci_strict(doc):
    """social/pr is the largest single (family, app) cell where
    POPT wins by mean (-2.33 pp, P=0.9995). Pin P ≥ 0.99."""
    r = doc["per_family_app"]["social/pr"]
    p = r["p_popt_lt_grasp"]
    assert p is not None and p >= STRONG_FLOOR, (
        f"social/pr P(POPT<GRASP)={p} < {STRONG_FLOOR}; "
        "the PR-bedrock-on-social claim is no longer CI-strict"
    )
    assert r["ci_hi"] is not None and r["ci_hi"] < 0, (
        f"social/pr CI hi = {r['ci_hi']}; does not exclude 0"
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
