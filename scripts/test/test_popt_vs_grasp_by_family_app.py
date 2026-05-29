"""Pytest gate: POPT vs GRASP per (family × app) bootstrap CIs.

Pins the deepest cut of the paper's core "POPT beats GRASP on
road graphs" claim: which (family, app) cells actually carry the
family-level signal.

Load-bearing findings:

* `road` family is POPT-favored on EVERY kernel (5/5 cells, all
  mean Δ < 0). The road-wins-with-POPT story is not driven by a
  single kernel; it holds across pr, bc, bfs, cc, sssp.
* `road/sssp` is the single biggest POPT effect anywhere
  (mean Δ ≈ -21.8 pp); the CI excludes 0.
* `social/cc` and `citation/cc` are CI-strict GRASP wins
  (P ≈ 0.000), confirming the cc-counter-narrative cell-by-cell.
* `social/pr` is CI-strict POPT (P ≥ 0.99) — pr stays POPT even
  when broken out by family.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_JSON = REPO_ROOT / "wiki" / "data" / "popt_vs_grasp_by_family_app.json"

STRONG_FLOOR = 0.99
STABILITY_FLOOR = 0.95
ROAD_APPS = ("pr", "bc", "bfs", "cc", "sssp")


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


def test_road_is_popt_favored_on_every_kernel(doc):
    """Every road/* cell with paired data must have mean Δ < 0.
    If a road kernel flips to GRASP, the family-level
    'POPT < GRASP on road' claim no longer factorises."""
    rogues = []
    for app in ROAD_APPS:
        key = f"road/{app}"
        r = doc["per_family_app"].get(key, {})
        if r.get("n_paired", 0) == 0:
            pytest.fail(f"road/{app} has no paired data")
        if r["mean_delta"] is None or r["mean_delta"] >= 0:
            rogues.append((key, r["mean_delta"]))
    assert not rogues, (
        f"road kernels with non-negative mean Δ: {rogues}. "
        "Road-family POPT win is no longer carried by every kernel."
    )


def test_road_sssp_is_huge_popt_win_and_ci_strict(doc):
    """road/sssp is the single biggest POPT effect anywhere
    (mean Δ ≈ -21.8 pp). Pin mean ≤ -10 pp and CI hi < 0."""
    r = doc["per_family_app"]["road/sssp"]
    assert r["mean_delta"] is not None and r["mean_delta"] <= -10.0, (
        f"road/sssp mean Δ = {r['mean_delta']}; expected ≤ -10 pp "
        "(was -21.8 pp at gate-write time)"
    )
    assert r["ci_hi"] is not None and r["ci_hi"] < 0, (
        f"road/sssp CI hi = {r['ci_hi']}; does not exclude 0"
    )


def test_road_high_signal_cells_are_ci_strict(doc):
    """For road/{sssp, bc} which have very strong POPT-better
    mean deltas, the bootstrap P must be ≥ 0.95."""
    for app in ("sssp", "bc"):
        key = f"road/{app}"
        p = doc["per_family_app"][key]["p_popt_lt_grasp"]
        assert p is not None and p >= STABILITY_FLOOR, (
            f"{key} P(POPT<GRASP)={p} < {STABILITY_FLOOR}"
        )


def test_cc_is_grasp_strict_outside_road(doc):
    """social/cc and citation/cc are GRASP-strict (P ≈ 0.000).
    Confirms the cc-counter-narrative cell-by-cell, not just at
    the per-kernel-across-families level."""
    for key in ("social/cc", "citation/cc"):
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
