"""Confidence gate 105 — regression_budget ↔ literature_faithfulness parity.

regression_budget.json (per-cell distance from a claim's tolerance
boundary, computed as margin_pp = how many percentage points of
headroom each cell has before its claim would flip from PASS to
DISAGREE) and literature_faithfulness_postfix.json (the corresponding
PASS / WITHIN / KNOWN_DEVIATION / DISAGREE / INSUFFICIENT / MISSING
state machine output) are two views of the same 330-cell literature
replay. The budget side is the *quantitative* margin; the faithfulness
side is the *categorical* verdict. They must agree on every cell, or
the paper's "X claims pass, Y are known deviations, Z disagree" line
gets out of sync with the per-cell distance distribution that backs
its threshold choices.

The gate runs 13 invariants split 4/4/4/1 across regression_budget
internal hygiene, lit_faith internal hygiene, cross-artifact
key-and-status parity, and one math invariant.

Invariants:

  regression_budget internal (4):
    1. len(per_cell) == summary.cells_total
    2. summary.by_kind[*].n matches counted claim_kind in per_cell for
       every kind present in by_kind
    3. summary.cells_in_distribution == count of per_cell with
       status in {ok, within_tolerance}; in-distribution min/max
       margins match the corresponding summary fields exactly
    4. fragile_cells has at most 10 entries; every entry is in
       per_cell; every entry's margin_pp >= summary.min_margin_pp

  lit_faith internal (4):
    5. len(per_claim) == summary.claims_total
    6. counted status frequencies in per_claim match summary.{ok,
       within_tolerance, known_deviation, disagree, insufficient_data,
       missing}
    7. summary state counters sum to summary.claims_total
    8. len(known_deviations) == summary.known_deviation;
       len(disagreements) == summary.disagree;
       len(tolerated) == summary.within_tolerance

  Cross-artifact key/status parity (4):
    9. rb.per_cell (graph,app,l3,policy) key set equals lf.per_claim
       key set (both 330)
   10. for every shared key, rb.status == lf.status (no semantic drift)
   11. count of (rb.status == "known_deviation") equals
       count of (lf.status == "known_deviation") equals
       lf.summary.known_deviation
   12. count of (rb.status == "within_tolerance") equals
       lf.summary.within_tolerance equals len(lf.tolerated)

  Math hygiene (1):
   13. every margin_pp in per_cell is finite and non-negative;
       every known_deviation cell has margin_pp == 0.0
       (KD cells are excluded from the distance distribution)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RB_PATH = PROJECT_ROOT / "wiki" / "data" / "regression_budget.json"
LF_PATH = PROJECT_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"

CELL_KEY_FIELDS = ("graph", "app", "l3_size", "policy")
IN_DIST_STATUSES = frozenset({"ok", "within_tolerance"})
ALL_LF_STATES = ("ok", "within_tolerance", "known_deviation",
                 "disagree", "insufficient_data", "missing")
FRAGILE_FLOOR = 10  # at most 10 fragile cells in fragile_cells list


@pytest.fixture(scope="module")
def rb() -> dict:
    assert RB_PATH.exists(), f"missing {RB_PATH}"
    return json.loads(RB_PATH.read_text())


@pytest.fixture(scope="module")
def lf() -> dict:
    assert LF_PATH.exists(), f"missing {LF_PATH}"
    return json.loads(LF_PATH.read_text())


def _key(c: dict) -> tuple:
    return tuple(c[k] for k in CELL_KEY_FIELDS)


# ---------------------------------------------------------------------------
# regression_budget internal (4)
# ---------------------------------------------------------------------------


def test_rb_cells_total_matches_per_cell_length(rb: dict) -> None:
    assert len(rb["per_cell"]) == rb["summary"]["cells_total"], (
        f"len(per_cell)={len(rb['per_cell'])} != summary.cells_total="
        f"{rb['summary']['cells_total']}"
    )


def test_rb_by_kind_counts_match_per_cell(rb: dict) -> None:
    # by_kind.n counts only in-distribution cells (status ∈ {ok, within_tolerance});
    # out-of-distribution cells (known_deviation, etc.) are excluded from the
    # per-kind margin distribution.
    counted = Counter(
        c["claim_kind"] for c in rb["per_cell"]
        if c["status"] in IN_DIST_STATUSES
    )
    bad: list[tuple[str, int, int]] = []
    for kind, stats in rb["summary"]["by_kind"].items():
        stated = stats.get("n")
        if counted.get(kind) != stated:
            bad.append((kind, stated, counted.get(kind, 0)))
    assert not bad, (
        f"by_kind in-distribution count drift (kind, stated, counted): {bad}"
    )


def test_rb_in_distribution_count_and_extremes(rb: dict) -> None:
    in_dist = [c for c in rb["per_cell"] if c["status"] in IN_DIST_STATUSES]
    stated_n = rb["summary"]["cells_in_distribution"]
    assert len(in_dist) == stated_n, (
        f"cells_in_distribution={stated_n} != counted in-dist={len(in_dist)}"
    )
    margins = sorted(c["margin_pp"] for c in in_dist)
    assert margins, "no in-distribution cells found"
    assert math.isclose(margins[0], rb["summary"]["min_margin_pp"], abs_tol=1e-6), (
        f"min_margin_pp drift: stated={rb['summary']['min_margin_pp']} counted={margins[0]}"
    )
    assert math.isclose(margins[-1], rb["summary"]["max_margin_pp"], abs_tol=1e-6), (
        f"max_margin_pp drift: stated={rb['summary']['max_margin_pp']} counted={margins[-1]}"
    )


def test_rb_fragile_cells_well_formed(rb: dict) -> None:
    fragile = rb["fragile_cells"]
    assert len(fragile) <= FRAGILE_FLOOR, (
        f"fragile_cells exceeds floor: len={len(fragile)}, floor={FRAGILE_FLOOR}"
    )
    per_cell_keys = {_key(c) for c in rb["per_cell"]}
    missing = [c for c in fragile if _key(c) not in per_cell_keys]
    assert not missing, f"fragile_cells entries not in per_cell: {[_key(c) for c in missing]}"
    floor = rb["summary"]["min_margin_pp"]
    too_small = [
        (_key(c), c["margin_pp"]) for c in fragile if c["margin_pp"] < floor - 1e-9
    ]
    assert not too_small, (
        f"fragile_cells include margins below summary.min_margin_pp={floor}: {too_small}"
    )


# ---------------------------------------------------------------------------
# lit_faith internal (4)
# ---------------------------------------------------------------------------


def test_lf_per_claim_length_matches_summary(lf: dict) -> None:
    assert len(lf["per_claim"]) == lf["summary"]["claims_total"], (
        f"len(per_claim)={len(lf['per_claim'])} != summary.claims_total="
        f"{lf['summary']['claims_total']}"
    )


def test_lf_status_counts_match_summary(lf: dict) -> None:
    counted = Counter(c["status"] for c in lf["per_claim"])
    bad: list[tuple[str, int, int]] = []
    for state in ALL_LF_STATES:
        stated = lf["summary"].get(state, 0)
        if counted.get(state, 0) != stated:
            bad.append((state, stated, counted.get(state, 0)))
    assert not bad, f"lit_faith state count drift (state, stated, counted): {bad}"


def test_lf_summary_states_sum_to_total(lf: dict) -> None:
    s = lf["summary"]
    states_sum = sum(s.get(k, 0) for k in ALL_LF_STATES)
    assert states_sum == s["claims_total"], (
        f"sum of summary states {states_sum} != claims_total {s['claims_total']}"
    )


def test_lf_list_lengths_match_summary(lf: dict) -> None:
    s = lf["summary"]
    bad: list[tuple[str, int, int]] = []
    if len(lf["known_deviations"]) != s["known_deviation"]:
        bad.append(("known_deviations", s["known_deviation"], len(lf["known_deviations"])))
    if len(lf["disagreements"]) != s["disagree"]:
        bad.append(("disagreements", s["disagree"], len(lf["disagreements"])))
    if len(lf["tolerated"]) != s["within_tolerance"]:
        bad.append(("tolerated", s["within_tolerance"], len(lf["tolerated"])))
    assert not bad, f"lf list lengths != summary counters (list, stated, counted): {bad}"


# ---------------------------------------------------------------------------
# Cross-artifact key/status parity (4)
# ---------------------------------------------------------------------------


def test_rb_lf_share_same_key_set(rb: dict, lf: dict) -> None:
    rb_keys = {_key(c) for c in rb["per_cell"]}
    lf_keys = {_key(c) for c in lf["per_claim"]}
    only_rb = rb_keys - lf_keys
    only_lf = lf_keys - rb_keys
    assert not only_rb and not only_lf, (
        f"key set mismatch: only_in_rb={len(only_rb)} (e.g. {list(only_rb)[:3]}), "
        f"only_in_lf={len(only_lf)} (e.g. {list(only_lf)[:3]})"
    )


def test_rb_lf_status_agree_per_cell(rb: dict, lf: dict) -> None:
    lf_status = {_key(c): c["status"] for c in lf["per_claim"]}
    bad: list[tuple[tuple, str, str]] = []
    for c in rb["per_cell"]:
        k = _key(c)
        rb_s, lf_s = c["status"], lf_status.get(k)
        if rb_s != lf_s:
            bad.append((k, rb_s, lf_s))
    assert not bad, f"per-cell status mismatch (key, rb_status, lf_status): {bad[:5]} ({len(bad)} total)"


def test_known_deviation_counts_agree(rb: dict, lf: dict) -> None:
    rb_kd = sum(1 for c in rb["per_cell"] if c["status"] == "known_deviation")
    lf_kd = sum(1 for c in lf["per_claim"] if c["status"] == "known_deviation")
    stated = lf["summary"]["known_deviation"]
    assert rb_kd == lf_kd == stated, (
        f"known_deviation triple-count mismatch: rb={rb_kd}, lf={lf_kd}, summary={stated}"
    )


def test_within_tolerance_counts_agree(rb: dict, lf: dict) -> None:
    rb_wt = sum(1 for c in rb["per_cell"] if c["status"] == "within_tolerance")
    stated = lf["summary"]["within_tolerance"]
    list_len = len(lf["tolerated"])
    assert rb_wt == stated == list_len, (
        f"within_tolerance triple-count mismatch: rb={rb_wt}, summary={stated}, "
        f"len(tolerated)={list_len}"
    )


# ---------------------------------------------------------------------------
# Math hygiene (1)
# ---------------------------------------------------------------------------


def test_margins_well_formed(rb: dict) -> None:
    bad: list[tuple[tuple, str, object]] = []
    for c in rb["per_cell"]:
        m = c.get("margin_pp")
        if not isinstance(m, (int, float)) or not math.isfinite(float(m)):
            bad.append((_key(c), "not finite", m))
        elif m < 0:
            bad.append((_key(c), "negative", m))
        elif c["status"] == "known_deviation" and not math.isclose(m, 0.0, abs_tol=1e-9):
            bad.append((_key(c), "known_deviation with non-zero margin", m))
    assert not bad, f"margin_pp issues: {bad[:5]} ({len(bad)} total)"
