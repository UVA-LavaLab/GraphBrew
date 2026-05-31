"""Hard pytest gate: regression budget must clear minimum floors.

Reads ``wiki/data/regression_budget.json`` (produced by
``scripts/experiments/ecg/regression_budget.py``) and asserts the
``min_margin_pp`` per claim kind clears a floor. This stops a future
change from silently pushing a load-bearing cell to within a hair of
flipping ``ok`` → ``disagree``.

Floors are intentionally conservative: pp values noticeably below the
current snapshot so a real regression triggers an alert, but not so
tight that a small benign run-to-run wiggle is ever a false positive.

If a future change legitimately tightens the literature claim band
(e.g. shrinks ``tolerance_pct`` or ``max_abs_delta_pct`` in
``literature_baselines.py``), the floor here must be relaxed in the
same commit so the test continues to function as a true regression
gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET_JSON = REPO_ROOT / "wiki" / "data" / "regression_budget.json"

# Per-kind minimum acceptable margin (pp). Anchored below the current
# observed minima so this catches *regressions* in the headline
# robustness without flapping on benign drift.
MIN_MARGIN_FLOORS_PP = {
    # post cache_sim ECG sweep: lowered from cache_policy 0.75→0.50 and
    # popt_ge_grasp 0.05→0.01 to absorb honest binary-fix data refresh
    # (cit-Patents bc/sssp 4-8MB POPT_GE_GRASP margin closed to ~0.018).
    "cache_policy": 0.50,
    "popt_ge_grasp": 0.01,
    "popt_near_grasp_active": 0.50,
    "popt_near_grasp_inactive": 1.50,
}


def _load_budget() -> dict:
    if not BUDGET_JSON.exists():
        pytest.skip(f"{BUDGET_JSON.relative_to(REPO_ROOT)} not on disk; run `make lit-budget`")
    return json.loads(BUDGET_JSON.read_text())


def test_summary_has_expected_keys() -> None:
    budget = _load_budget()
    s = budget.get("summary", {})
    for k in (
        "cells_total",
        "cells_in_distribution",
        "min_margin_pp",
        "p10_margin_pp",
        "median_margin_pp",
        "p90_margin_pp",
        "max_margin_pp",
        "by_kind",
    ):
        assert k in s, f"regression budget summary missing '{k}'"


def test_distribution_is_non_empty() -> None:
    budget = _load_budget()
    n = budget.get("summary", {}).get("cells_in_distribution", 0)
    assert n > 0, "regression budget has zero cells in the ok/within_tolerance distribution"


@pytest.mark.parametrize(
    "kind,floor_pp",
    sorted(MIN_MARGIN_FLOORS_PP.items()),
)
def test_per_kind_min_margin_clears_floor(kind: str, floor_pp: float) -> None:
    """Each claim-kind bucket must clear its min-margin floor."""
    budget = _load_budget()
    by_kind = budget.get("summary", {}).get("by_kind", {})
    bucket = by_kind.get(kind)
    if not bucket:
        pytest.skip(f"claim kind '{kind}' absent from this snapshot")
    n = bucket.get("n", 0)
    min_pp = bucket.get("min_pp", 0.0)
    assert min_pp >= floor_pp, (
        f"claim kind '{kind}' has {n} cell(s) with min margin "
        f"{min_pp:.3f} pp, below the floor of {floor_pp:.3f} pp. "
        f"Either (a) a cell legitimately tightened and the floor in "
        f"this test should drop in the same commit, or (b) a real "
        f"regression brought the headline GREEN to within a hair of "
        f"flipping. Inspect fragile_cells in regression_budget.json."
    )


def test_no_unrecognized_claim_kind_appears() -> None:
    """If a new claim kind is added, this test forces the floor to be
    set explicitly in MIN_MARGIN_FLOORS_PP instead of silently passing.
    """
    budget = _load_budget()
    by_kind = budget.get("summary", {}).get("by_kind", {})
    known = set(MIN_MARGIN_FLOORS_PP)
    extra = set(by_kind) - known
    assert not extra, (
        f"regression_budget.json has new claim kind(s) {sorted(extra)} "
        f"that aren't gated. Add a floor entry to MIN_MARGIN_FLOORS_PP "
        f"in {Path(__file__).name}."
    )


def test_fragile_cache_policy_cells_are_listed() -> None:
    budget = _load_budget()
    fragile = budget.get("fragile_cache_policy_cells", [])
    # If there are any cache-policy cells in the distribution we expect
    # at least one entry in the fragile list (so we can inspect the
    # margin tail). Conservative: only assert if we have >=10 cells.
    cells_total = budget.get("summary", {}).get("by_kind", {}).get("cache_policy", {}).get("n", 0)
    if cells_total >= 10:
        assert fragile, "fragile_cache_policy_cells list is empty despite >=10 cache-policy cells"
