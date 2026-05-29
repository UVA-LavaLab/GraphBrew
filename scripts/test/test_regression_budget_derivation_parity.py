"""Derivation-parity gate for ``wiki/data/regression_budget.json``.

The regression-budget artifact records, per lit-faith cell, the distance
(in pp) that the observed Δ would have to drift in the *adverse*
direction before the cell flips from ``ok`` / ``within_tolerance`` to
``disagree``. It is the only artifact in the dashboard that quantifies
*how close we are to disagreeing with the literature*; a silent change
to the perturbation step, the inclusion set, or the per-kind margin
math would let the headline get more brittle without tripping any
other gate.

This module re-derives the artifact end-to-end from
``literature_faithfulness_postfix.json`` and asserts byte-for-byte
equivalence with the committed JSON, plus the load-bearing invariants
the generator enforces:

* per-cell margin is non-negative and rounded to 4 dp
* status filter ``{ok, within_tolerance}`` defines the headline
  distribution; everything else collapses to ``margin_pp == 0.0``
* the perturbation actually flips the classifier for cache-policy
  cells: ``classify(claim, delta + adverse * (margin + epsilon))``
  must return ``disagree`` (and ``margin - epsilon`` must not)
* percentile math: min / p10 / median / p90 / max are computed via the
  generator's left-floor index on a sorted list
* per-kind summary: n / min_pp / median_pp match recomputation
* fragility rankings: top-10 sorted ascending by margin; cache-only
  list restricted to ``claim_kind == 'cache_policy'``

5 groups, 22 tests total.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "regression_budget.json"
SOURCE = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
GENERATOR = REPO_ROOT / "scripts" / "experiments" / "ecg" / "regression_budget.py"

INCLUDED_STATUSES = {"ok", "within_tolerance"}


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("regression_budget_local", GENERATOR)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_classifier() -> Any:
    path = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_faithfulness.py"
    spec = importlib.util.spec_from_file_location("lit_faith_for_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_baselines() -> Any:
    path = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_baselines.py"
    spec = importlib.util.spec_from_file_location("lit_baselines_for_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def regenerated() -> dict[str, Any]:
    gen = _load_generator()
    return gen.compute(SOURCE)


# ----------------------------------------------------------------------
# Group 1: cross-source byte equivalence
# ----------------------------------------------------------------------

def test_regenerated_matches_committed_artifact(regenerated: dict[str, Any], artifact: dict[str, Any]) -> None:
    """Re-running compute() on the same source must produce the committed JSON."""
    assert json.dumps(regenerated, sort_keys=True) == json.dumps(artifact, sort_keys=True)


def test_committed_artifact_has_expected_top_level_keys(artifact: dict[str, Any]) -> None:
    assert set(artifact.keys()) == {
        "summary",
        "fragile_cells",
        "fragile_cache_policy_cells",
        "per_cell",
    }


# ----------------------------------------------------------------------
# Group 2: per-cell shape & rounding
# ----------------------------------------------------------------------

def test_every_per_cell_row_has_required_fields(artifact: dict[str, Any]) -> None:
    required = {
        "graph", "app", "l3_size", "policy", "status", "delta_pct",
        "expected_sign", "min_abs_delta_pct", "max_abs_delta_pct",
        "tolerance_pct", "margin_pp", "claim_kind", "citation",
    }
    for row in artifact["per_cell"]:
        missing = required - set(row.keys())
        assert not missing, f"row missing fields {missing}: {row}"


def test_margin_pp_is_non_negative(artifact: dict[str, Any]) -> None:
    for row in artifact["per_cell"]:
        assert row["margin_pp"] >= 0.0, f"negative margin in {row}"


def test_margin_pp_rounded_to_4dp(artifact: dict[str, Any]) -> None:
    for row in artifact["per_cell"]:
        m = row["margin_pp"]
        assert round(m, 4) == m, f"margin {m} not rounded to 4dp in {row}"


def test_excluded_status_rows_have_zero_margin(artifact: dict[str, Any]) -> None:
    for row in artifact["per_cell"]:
        if row["status"] not in INCLUDED_STATUSES:
            assert row["margin_pp"] == 0.0, (
                f"row with status={row['status']} must have margin_pp=0; got {row}"
            )


def test_claim_kind_in_known_set(artifact: dict[str, Any]) -> None:
    known = {
        "cache_policy",
        "popt_ge_grasp",
        "popt_near_grasp_active",
        "popt_near_grasp_inactive",
        # generator emits the lowercased policy name when the row is excluded
        # and the policy is not in {LRU,SRRIP,GRASP,POPT}; included for parity.
        "popt_near_grasp_if_big_gap",
    }
    for row in artifact["per_cell"]:
        assert row["claim_kind"] in known, f"unknown claim_kind in {row}"


# ----------------------------------------------------------------------
# Group 3: adverse-perturbation flip semantics (cache_policy cells)
# ----------------------------------------------------------------------

def test_cache_policy_margin_flips_classifier(artifact: dict[str, Any]) -> None:
    """For cache-policy cells, classify(delta + adverse*(margin+ε)) must be 'disagree'.

    This is the load-bearing invariant of the binary search: the margin
    is the *smallest* adverse drift at which the classifier flips.
    """
    faith = _load_classifier()
    lit = _load_baselines()
    classify = faith._classify
    step = 0.005  # _PERTURB_STEP_PCT from the generator
    sample_count = 0
    for row in artifact["per_cell"]:
        if row["claim_kind"] != "cache_policy":
            continue
        if row["status"] not in INCLUDED_STATUSES:
            continue
        if row["margin_pp"] >= 50.0:  # _PERTURB_MAX_PCT — search exhausted
            continue
        claim = None
        for c in lit.claims_for(row["graph"], row["app"], row["l3_size"]):
            if c.policy == row["policy"]:
                claim = c
                break
        assert claim is not None, f"could not find claim for {row}"
        sign = row["expected_sign"]
        directions: tuple[float, ...]
        if sign == "-":
            directions = (+1.0,)
        elif sign == "+":
            directions = (-1.0,)
        else:
            directions = (+1.0, -1.0)
        statuses = []
        for d in directions:
            perturbed = row["delta_pct"] + d * (row["margin_pp"] + step)
            statuses.append(classify(claim, perturbed))
        assert "disagree" in statuses, (
            f"perturbation by margin+step did not flip cell {row} "
            f"(statuses={statuses})"
        )
        sample_count += 1
    assert sample_count >= 50, (
        f"expected at least 50 included cache_policy cells to verify; got {sample_count}"
    )


def test_cache_policy_margin_is_minimal(artifact: dict[str, Any]) -> None:
    """One step *short* of the margin must NOT yet flip the classifier.

    Combined with the previous test, this pins the margin as the
    minimal adverse drift to disagree.
    """
    faith = _load_classifier()
    lit = _load_baselines()
    classify = faith._classify
    step = 0.005
    sample_count = 0
    for row in artifact["per_cell"]:
        if row["claim_kind"] != "cache_policy":
            continue
        if row["status"] not in INCLUDED_STATUSES:
            continue
        if row["margin_pp"] < 2 * step:  # too close to zero to step short
            continue
        if row["margin_pp"] >= 50.0:
            continue
        claim = None
        for c in lit.claims_for(row["graph"], row["app"], row["l3_size"]):
            if c.policy == row["policy"]:
                claim = c
                break
        assert claim is not None
        sign = row["expected_sign"]
        if sign == "-":
            directions = (+1.0,)
        elif sign == "+":
            directions = (-1.0,)
        else:
            directions = (+1.0, -1.0)
        # The minimal-step claim: AT LEAST ONE adverse direction must
        # still classify as ok/within_tolerance at margin - step.
        ok_dirs = 0
        for d in directions:
            perturbed = row["delta_pct"] + d * (row["margin_pp"] - step)
            if classify(claim, perturbed) in INCLUDED_STATUSES:
                ok_dirs += 1
        assert ok_dirs >= 1, (
            f"cell {row} flipped to disagree one step BEFORE its recorded margin"
        )
        sample_count += 1
    assert sample_count >= 50


def test_popt_ge_grasp_margin_matches_tolerance_minus_delta(artifact: dict[str, Any]) -> None:
    """POPT_GE_GRASP cells use the closed-form margin = max(0, tol - delta_pct)."""
    for row in artifact["per_cell"]:
        if row["claim_kind"] != "popt_ge_grasp":
            continue
        if row["status"] not in INCLUDED_STATUSES:
            continue
        expected = round(max(0.0, (row["tolerance_pct"] or 0.0) - row["delta_pct"]), 4)
        assert row["margin_pp"] == expected, (
            f"popt_ge_grasp closed-form mismatch: row={row} expected={expected}"
        )


# ----------------------------------------------------------------------
# Group 4: summary aggregation math
# ----------------------------------------------------------------------

def _pct_left_floor(values: list[float], p: float) -> float:
    """Mirror the generator's percentile selector."""
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(math.floor(p * (len(values) - 1)))))
    return values[idx]


def test_cells_total_matches_per_cell_length(artifact: dict[str, Any]) -> None:
    assert artifact["summary"]["cells_total"] == len(artifact["per_cell"])


def test_cells_in_distribution_matches_included_count(artifact: dict[str, Any]) -> None:
    included = [r for r in artifact["per_cell"] if r["status"] in INCLUDED_STATUSES]
    assert artifact["summary"]["cells_in_distribution"] == len(included)


def test_min_max_margin_match_included_extremes(artifact: dict[str, Any]) -> None:
    margins = sorted(r["margin_pp"] for r in artifact["per_cell"] if r["status"] in INCLUDED_STATUSES)
    s = artifact["summary"]
    assert s["min_margin_pp"] == round(margins[0], 4)
    assert s["max_margin_pp"] == round(margins[-1], 4)


def test_percentile_math_uses_left_floor_index(artifact: dict[str, Any]) -> None:
    margins = sorted(r["margin_pp"] for r in artifact["per_cell"] if r["status"] in INCLUDED_STATUSES)
    s = artifact["summary"]
    assert s["p10_margin_pp"] == round(_pct_left_floor(margins, 0.10), 4)
    assert s["median_margin_pp"] == round(_pct_left_floor(margins, 0.50), 4)
    assert s["p90_margin_pp"] == round(_pct_left_floor(margins, 0.90), 4)


def test_by_kind_summary_matches_recomputation(artifact: dict[str, Any]) -> None:
    by_kind: dict[str, list[float]] = {}
    for r in artifact["per_cell"]:
        if r["status"] not in INCLUDED_STATUSES:
            continue
        by_kind.setdefault(r["claim_kind"], []).append(r["margin_pp"])
    expected = {
        k: {
            "n": len(vs),
            "min_pp": round(min(vs), 4),
            "median_pp": round(sorted(vs)[len(vs) // 2], 4),
        }
        for k, vs in sorted(by_kind.items())
    }
    assert artifact["summary"]["by_kind"] == expected


def test_by_kind_keys_are_sorted(artifact: dict[str, Any]) -> None:
    keys = list(artifact["summary"]["by_kind"].keys())
    assert keys == sorted(keys), f"by_kind keys not sorted: {keys}"


def test_excluded_kinds_absent_from_by_kind(artifact: dict[str, Any]) -> None:
    """Rows in excluded statuses never contribute to by_kind."""
    excluded_only_kinds = set()
    for r in artifact["per_cell"]:
        if r["status"] not in INCLUDED_STATUSES:
            excluded_only_kinds.add(r["claim_kind"])
    included_kinds = set()
    for r in artifact["per_cell"]:
        if r["status"] in INCLUDED_STATUSES:
            included_kinds.add(r["claim_kind"])
    # Any kind that has ONLY excluded rows must not appear in by_kind.
    by_kind_keys = set(artifact["summary"]["by_kind"].keys())
    only_excluded = excluded_only_kinds - included_kinds
    assert only_excluded.isdisjoint(by_kind_keys), (
        f"by_kind contains kinds with no included rows: "
        f"{only_excluded & by_kind_keys}"
    )


# ----------------------------------------------------------------------
# Group 5: fragility ranking invariants
# ----------------------------------------------------------------------

def test_fragile_cells_sorted_ascending_by_margin(artifact: dict[str, Any]) -> None:
    margins = [r["margin_pp"] for r in artifact["fragile_cells"]]
    assert margins == sorted(margins), f"fragile_cells not sorted asc: {margins}"


def test_fragile_cells_has_at_most_10_entries(artifact: dict[str, Any]) -> None:
    assert len(artifact["fragile_cells"]) <= 10


def test_fragile_cells_are_the_global_minima(artifact: dict[str, Any]) -> None:
    included = [r for r in artifact["per_cell"] if r["status"] in INCLUDED_STATUSES]
    expected_margins = sorted(r["margin_pp"] for r in included)[:10]
    actual_margins = [r["margin_pp"] for r in artifact["fragile_cells"]]
    assert actual_margins == expected_margins


def test_fragile_cache_policy_cells_restricted_to_cache_policy_kind(artifact: dict[str, Any]) -> None:
    for r in artifact["fragile_cache_policy_cells"]:
        assert r["claim_kind"] == "cache_policy", (
            f"fragile_cache_policy_cells contains non-cache_policy row: {r}"
        )


def test_fragile_cache_policy_cells_sorted_ascending(artifact: dict[str, Any]) -> None:
    margins = [r["margin_pp"] for r in artifact["fragile_cache_policy_cells"]]
    assert margins == sorted(margins)


def test_fragile_cache_policy_cells_are_cache_minima(artifact: dict[str, Any]) -> None:
    cache_included = [
        r for r in artifact["per_cell"]
        if r["status"] in INCLUDED_STATUSES and r["claim_kind"] == "cache_policy"
    ]
    expected = sorted(r["margin_pp"] for r in cache_included)[:10]
    actual = [r["margin_pp"] for r in artifact["fragile_cache_policy_cells"]]
    assert actual == expected
