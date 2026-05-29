"""Derivation parity gate for wiki/data/monotonicity_universality.json.

Reproduces the per-(graph, app, policy) L3-sweep monotonicity check from
wiki/data/oracle_gap.json. The generator at
scripts/experiments/ecg/monotonicity_universality.py groups oracle rows
by (graph, app, policy), sorts the L3 sweep by bytes, and counts every
consecutive (L_i, L_{i+1}) step where miss_rate increased. Below
MAX_NOISE_BUMP_PP (0.5 pp) is noise; at-or-above is a hard violation
that fails the verdict.

Load-bearing rules pinned here:
  * cell key is (graph, app, policy); cells with < 2 L3 points are skipped;
  * L3 axis sorted by bytes via _l3_bytes(label);
  * delta_pp = (miss_rate(L_{i+1}) - miss_rate(L_i)) * 100, rounded to 6dp;
  * bumps = all positive deltas, sorted by -delta_pp (largest first);
  * MAX_NOISE_BUMP_PP = 0.5;
  * BUMP_PCT_CEILING = 0.10;
  * verdict_checks combine 3 invariants via all().
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "wiki" / "data" / "monotonicity_universality.json"
ORACLE = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

MAX_NOISE_BUMP_PP = 0.5
BUMP_PCT_CEILING = 0.10


def _l3_bytes(label):
    n = int(label[:-2])
    unit = label[-2:]
    if unit == "kB":
        return n * 1024
    if unit == "MB":
        return n * 1024 * 1024
    raise ValueError(f"unknown L3 unit: {label!r}")


def _build_expected():
    doc = json.loads(ORACLE.read_text())
    cells = defaultdict(list)
    labels_seen = set()
    for r in doc["rows"]:
        key = (r["graph"], r["app"], r["policy"])
        label = r["l3_size"]
        cells[key].append((_l3_bytes(label), label, float(r["miss_rate"])))
        labels_seen.add(label)
    axis_labels = sorted(labels_seen, key=_l3_bytes)
    axis_bytes = [_l3_bytes(l) for l in axis_labels]
    bumps = []
    total_steps = 0
    cells_ok = 0
    largest = 0.0
    largest_cell = None
    for (g, a, p), pts in cells.items():
        if len(pts) < 2:
            continue
        cells_ok += 1
        s = sorted(pts, key=lambda x: x[0])
        for i in range(len(s) - 1):
            total_steps += 1
            delta_pp = (s[i + 1][2] - s[i][2]) * 100.0
            if delta_pp > 0.0:
                entry = {
                    "graph": g, "app": a, "policy": p,
                    "l3_from": s[i][1], "l3_to": s[i + 1][1],
                    "delta_pp": round(delta_pp, 6),
                }
                bumps.append(entry)
                if delta_pp > largest:
                    largest = delta_pp
                    largest_cell = entry
    bumps.sort(key=lambda e: -e["delta_pp"])
    return {
        "axis_labels": axis_labels, "axis_bytes": axis_bytes,
        "cells_ok": cells_ok, "total_steps": total_steps,
        "bumps": bumps, "largest_bump_pp": round(largest, 6),
        "largest_bump_cell": largest_cell,
    }


def _meta():
    return json.loads(ART.read_text())["meta"]


# Group A — meta layer constants ---------------------------------------

def test_max_noise_bump_pp_is_half_pp():
    assert _meta()["max_noise_bump_pp"] == MAX_NOISE_BUMP_PP


def test_bump_pct_ceiling_is_10pct():
    assert _meta()["bump_pct_ceiling"] == BUMP_PCT_CEILING


def test_source_artifact_points_to_oracle_gap():
    assert _meta()["source_artifact"] == "wiki/data/oracle_gap.json"


def test_l3_axis_sorted_ascending_by_bytes():
    m = _meta()
    assert m["l3_axis_bytes"] == sorted(m["l3_axis_bytes"])
    assert m["l3_axis_bytes"] == [_l3_bytes(l) for l in m["l3_axis_labels"]]


# Group B — counts match derivation -------------------------------------

def test_cell_count_matches_derived():
    e = _build_expected()
    assert _meta()["cell_count"] == e["cells_ok"]


def test_total_step_count_matches_derived():
    e = _build_expected()
    assert _meta()["total_step_count"] == e["total_steps"]


def test_bump_count_matches_derived():
    e = _build_expected()
    assert _meta()["bump_count"] == len(e["bumps"])


def test_bump_pct_equals_bump_count_over_total_steps():
    m = _meta()
    if m["total_step_count"] == 0:
        assert m["bump_pct"] == 0.0
    else:
        expected = round(m["bump_count"] / m["total_step_count"], 6)
        assert m["bump_pct"] == expected


def test_hard_violation_count_matches_threshold_filter():
    m = _meta()
    expected = sum(1 for b in m["bumps"] if b["delta_pp"] >= MAX_NOISE_BUMP_PP)
    assert m["hard_violation_count"] == expected


# Group C — per-bump derivation parity ----------------------------------

def test_every_bump_has_positive_delta():
    for b in _meta()["bumps"]:
        assert b["delta_pp"] > 0.0


def test_bumps_sorted_largest_first():
    deltas = [b["delta_pp"] for b in _meta()["bumps"]]
    assert deltas == sorted(deltas, reverse=True)


def test_bumps_set_matches_derived_set():
    """Each bump entry matches one of the derived entries (order-independent)."""
    e = _build_expected()
    derived = {
        (b["graph"], b["app"], b["policy"], b["l3_from"], b["l3_to"], b["delta_pp"])
        for b in e["bumps"]
    }
    actual = {
        (b["graph"], b["app"], b["policy"], b["l3_from"], b["l3_to"], b["delta_pp"])
        for b in _meta()["bumps"]
    }
    assert actual == derived


def test_largest_bump_pp_matches_max_over_bumps():
    m = _meta()
    if not m["bumps"]:
        assert m["largest_bump_pp"] == 0.0
    else:
        max_d = max(b["delta_pp"] for b in m["bumps"])
        assert abs(m["largest_bump_pp"] - max_d) < 1e-6


def test_largest_bump_cell_matches_first_in_sorted_bumps():
    m = _meta()
    if not m["bumps"]:
        assert m["largest_bump_cell"] is None
    else:
        top = m["bumps"][0]
        assert m["largest_bump_cell"]["delta_pp"] == top["delta_pp"]
        # may not be the SAME entry if there's a tie at the top — but
        # the recorded one must have the same delta_pp as the maximum.
        assert m["largest_bump_cell"]["delta_pp"] == m["largest_bump_pp"]


def test_bumps_from_to_labels_are_consecutive_in_axis():
    m = _meta()
    axis = m["l3_axis_labels"]
    for b in m["bumps"]:
        i = axis.index(b["l3_from"])
        # l3_to need not be axis[i+1] globally — a cell may skip axis
        # points that have no row; but l3_to must come AFTER l3_from.
        assert axis.index(b["l3_to"]) > i, b


# Group D — verdict logic -----------------------------------------------

def test_verdict_checks_dict_has_three_known_keys():
    keys = set(_meta()["verdict_checks"].keys())
    assert keys == {
        "no_hard_violations",
        "bump_pct_under_ceiling",
        "largest_bump_within_noise",
    }


def test_verdict_pass_iff_all_checks_true():
    m = _meta()
    expected = "PASS" if all(m["verdict_checks"].values()) else "FAIL"
    assert m["verdict"] == expected


def test_no_hard_violations_check_matches_count_zero():
    m = _meta()
    assert m["verdict_checks"]["no_hard_violations"] == (m["hard_violation_count"] == 0)


def test_bump_pct_under_ceiling_check_matches():
    m = _meta()
    assert m["verdict_checks"]["bump_pct_under_ceiling"] == (m["bump_pct"] <= BUMP_PCT_CEILING)


def test_largest_bump_within_noise_check_matches():
    m = _meta()
    assert m["verdict_checks"]["largest_bump_within_noise"] == (
        m["largest_bump_pp"] < MAX_NOISE_BUMP_PP
    )


# Group E — invariants on the current corpus ----------------------------

def test_currently_zero_hard_violations():
    """Foundational invariant: no real non-monotonicity in cache-sim sweep."""
    assert _meta()["hard_violation_count"] == 0


def test_currently_verdict_pass():
    assert _meta()["verdict"] == "PASS"
