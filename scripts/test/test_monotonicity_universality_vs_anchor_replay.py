"""Confidence gate 99 — monotonicity universality vs anchor monotonicity
replay cross-artifact agreement.

Two parallel cache-monotonicity check artifacts should each be internally
well-formed AND agree on the global "miss-rate is monotone non-increasing
as L3 grows" claim:

  - ``wiki/data/monotonicity_universality.json`` (MU) — cache-sim
    full 7-point L3 sweep (4kB-8MB), 320 (Li, Li+1) steps across 136
    cells. Tight tolerance: max_noise_bump_pp = 0.5 with a separate
    bump_pct_ceiling of 10%. Tracks every individual bump with cell key.
  - ``wiki/data/anchor_monotonicity_replay.json`` (AMR) — gem5 and
    sniper anchor sweeps at 4 sizes (4kB / 32kB / 256kB / 2MB).
    Per-tool checks (bump rate, hard bumps, max bump, catastrophic
    bumps) with looser tolerance (hard_bump_threshold_pp = 0.5,
    catastrophic_bump_pp = 3.0).

This gate locks 13 invariants split across four groups:

  MU internal (5):
    1. meta.bump_count == len(meta.bumps)
    2. meta.bump_pct == meta.bump_count / meta.total_step_count
    3. meta.largest_bump_pp == max(b.delta_pp for b in meta.bumps) AND
       meta.largest_bump_cell payload matches that exact bump
    4. meta.hard_violation_count == count of bumps with delta_pp >=
       meta.max_noise_bump_pp
    5. every bump has 0 < delta_pp < meta.max_noise_bump_pp (i.e., all
       are real bumps but all sub-threshold; the "universality" claim
       hangs on this being non-empty for sub-noise but empty for hard)

  MU axis structure (1):
    6. len(l3_axis_bytes) == len(l3_axis_labels) AND every label decodes
       to its byte value (4kB == 4096, 8MB == 8*1024*1024, etc.)

  AMR internal (5):
    7. for each tool: per_tool[tool].bumps == len(all_bumps) AND
       steps_total == cells * (len(expected_sizes) - 1)
    8. for each tool: max_bump_pp == max(delta_pp for b in all_bumps)
       (or 0 if empty)
    9. for each tool: hard_bumps == count of all_bumps with delta_pp >=
       constants.hard_bump_threshold_pp
   10. for each tool: bump_rate_pct == 100 * bumps / steps_total
   11. for each tool: evaluation.catastrophic_bumps is exactly the
       subset of all_bumps with delta_pp >= constants.catastrophic_bump_pp

  Overall + cross-artifact (2):
   12. AMR.overall.verdict_ok == True AND equals (all per_tool
       evaluation.verdict_ok) AND no catastrophic bumps in any tool
   13. cross-artifact tolerance agreement: MU.max_noise_bump_pp ==
       AMR.constants.hard_bump_threshold_pp (both 0.5 pp) AND MU's
       largest_bump_pp < AMR's hard_bump_threshold_pp (cache-sim is
       strictly tighter than the simulator anchors; this is the
       central "anchors broader, cache-sim sharper" guarantee)

If any one invariant breaks, the universal monotonicity claim either
loses its internal book-keeping or stops agreeing with the simulator
anchors that paper-claims uses to validate cross-tool consistency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MU_PATH = PROJECT_ROOT / "wiki" / "data" / "monotonicity_universality.json"
AMR_PATH = PROJECT_ROOT / "wiki" / "data" / "anchor_monotonicity_replay.json"

BUMP_PCT_TOL = 1e-6
DELTA_TOL = 1e-9
RATE_TOL = 1e-4
SIZE_SUFFIX = {"B": 1, "kB": 1024, "MB": 1024 * 1024, "GB": 1024 * 1024 * 1024}


@pytest.fixture(scope="module")
def mu() -> dict:
    assert MU_PATH.exists(), f"missing monotonicity_universality.json at {MU_PATH}"
    return json.loads(MU_PATH.read_text())


@pytest.fixture(scope="module")
def amr() -> dict:
    assert AMR_PATH.exists(), f"missing anchor_monotonicity_replay.json at {AMR_PATH}"
    return json.loads(AMR_PATH.read_text())


def _decode_size_label(label: str) -> int:
    # Split numeric prefix and unit suffix
    for n in range(len(label), 0, -1):
        prefix = label[:n]
        suffix = label[n:]
        if suffix in SIZE_SUFFIX:
            try:
                return int(prefix) * SIZE_SUFFIX[suffix]
            except ValueError:
                continue
    raise AssertionError(f"could not decode size label: {label!r}")


# ---------------------------------------------------------------------------
# MU internal (5)
# ---------------------------------------------------------------------------


def test_mu_bump_count_matches_bumps_len(mu: dict) -> None:
    declared = mu["meta"]["bump_count"]
    actual = len(mu["meta"]["bumps"])
    assert declared == actual, f"meta.bump_count={declared} but len(bumps)={actual}"


def test_mu_bump_pct_matches_count_over_total(mu: dict) -> None:
    m = mu["meta"]
    expected = m["bump_count"] / m["total_step_count"]
    assert math.isclose(m["bump_pct"], expected, abs_tol=BUMP_PCT_TOL), (
        f"meta.bump_pct={m['bump_pct']} but computed={expected}"
    )


def test_mu_largest_bump_matches_max(mu: dict) -> None:
    m = mu["meta"]
    if not m["bumps"]:
        pytest.skip("no bumps to test largest")
    actual_max = max(m["bumps"], key=lambda b: b["delta_pp"])
    assert math.isclose(m["largest_bump_pp"], actual_max["delta_pp"], abs_tol=DELTA_TOL), (
        f"meta.largest_bump_pp={m['largest_bump_pp']} but max(bumps)={actual_max['delta_pp']}"
    )
    # Validate largest_bump_cell payload matches the recomputed bump
    for k in ("graph", "app", "policy", "l3_from", "l3_to", "delta_pp"):
        assert m["largest_bump_cell"][k] == actual_max[k], (
            f"largest_bump_cell.{k}={m['largest_bump_cell'][k]} but max(bumps).{k}={actual_max[k]}"
        )


def test_mu_hard_violation_count_matches_threshold(mu: dict) -> None:
    m = mu["meta"]
    threshold = m["max_noise_bump_pp"]
    actual = sum(1 for b in m["bumps"] if b["delta_pp"] >= threshold)
    assert m["hard_violation_count"] == actual, (
        f"meta.hard_violation_count={m['hard_violation_count']} but per-bump count={actual} (threshold={threshold})"
    )


def test_mu_all_bumps_below_threshold_and_positive(mu: dict) -> None:
    m = mu["meta"]
    threshold = m["max_noise_bump_pp"]
    bad: list = []
    for b in m["bumps"]:
        if not (b["delta_pp"] > 0.0):
            bad.append((b, "non_positive"))
        if b["delta_pp"] >= threshold:
            bad.append((b, "hard_violation_in_softlist"))
    assert not bad, f"MU bump payload violations: {bad}"


# ---------------------------------------------------------------------------
# MU axis structure (1)
# ---------------------------------------------------------------------------


def test_mu_axis_labels_match_bytes(mu: dict) -> None:
    m = mu["meta"]
    labels = m["l3_axis_labels"]
    sizes = m["l3_axis_bytes"]
    assert len(labels) == len(sizes), f"axis size/label length mismatch: {len(sizes)} vs {len(labels)}"
    bad = []
    for lbl, sz in zip(labels, sizes):
        decoded = _decode_size_label(lbl)
        if decoded != sz:
            bad.append((lbl, sz, decoded))
    assert not bad, f"MU axis label->byte mismatches: {bad}"


# ---------------------------------------------------------------------------
# AMR internal (5)
# ---------------------------------------------------------------------------


def test_amr_per_tool_bumps_and_steps(amr: dict) -> None:
    bad: list = []
    for tool, p in amr["per_tool"].items():
        if p["bumps"] != len(p["all_bumps"]):
            bad.append((tool, "bumps_vs_all_bumps", p["bumps"], len(p["all_bumps"])))
        expected_steps = p["cells"] * (len(p["expected_sizes"]) - 1)
        if p["steps_total"] != expected_steps:
            bad.append(
                (tool, "steps_total", p["steps_total"], expected_steps, p["cells"], p["expected_sizes"])
            )
    assert not bad, f"AMR per_tool bumps/steps mismatches: {bad}"


def test_amr_per_tool_max_bump_matches(amr: dict) -> None:
    bad: list = []
    for tool, p in amr["per_tool"].items():
        if p["all_bumps"]:
            actual = max(b["delta_pp"] for b in p["all_bumps"])
        else:
            actual = 0.0
        if not math.isclose(p["max_bump_pp"], actual, abs_tol=DELTA_TOL):
            bad.append((tool, p["max_bump_pp"], actual))
    assert not bad, f"AMR max_bump_pp mismatches: {bad}"


def test_amr_per_tool_hard_bumps_matches_threshold(amr: dict) -> None:
    thr = amr["constants"]["hard_bump_threshold_pp"]
    bad: list = []
    for tool, p in amr["per_tool"].items():
        actual = sum(1 for b in p["all_bumps"] if b["delta_pp"] >= thr)
        if p["hard_bumps"] != actual:
            bad.append((tool, "hard_bumps", p["hard_bumps"], actual, thr))
    assert not bad, f"AMR hard_bumps mismatches: {bad}"


def test_amr_per_tool_bump_rate_pct_matches(amr: dict) -> None:
    bad: list = []
    for tool, p in amr["per_tool"].items():
        if p["steps_total"] == 0:
            continue
        expected = 100.0 * p["bumps"] / p["steps_total"]
        if not math.isclose(p["bump_rate_pct"], expected, abs_tol=RATE_TOL):
            bad.append((tool, p["bump_rate_pct"], expected))
    assert not bad, f"AMR bump_rate_pct mismatches: {bad}"


def test_amr_per_tool_catastrophic_bumps_matches_threshold(amr: dict) -> None:
    thr = amr["constants"]["catastrophic_bump_pp"]
    bad: list = []
    for tool, p in amr["per_tool"].items():
        expected = [b for b in p["all_bumps"] if b["delta_pp"] >= thr]
        # Compare on (app, graph, policy, l3_from, l3_to, delta_pp)
        def key(b):
            return (b["app"], b["graph"], b["policy"], b["l3_from"], b["l3_to"], b["delta_pp"])
        actual = p["evaluation"]["catastrophic_bumps"]
        if sorted(map(key, expected)) != sorted(map(key, actual)):
            bad.append((tool, len(expected), len(actual), thr))
    assert not bad, f"AMR catastrophic_bumps mismatches: {bad}"


# ---------------------------------------------------------------------------
# Overall + cross-artifact (2)
# ---------------------------------------------------------------------------


def test_amr_overall_verdict_matches_per_tool(amr: dict) -> None:
    overall = amr["overall"]
    assert overall["verdict_ok"] is True, f"AMR overall.verdict_ok={overall['verdict_ok']}; expected True"
    per_tool_verdicts = {tool: p["evaluation"]["verdict_ok"] for tool, p in amr["per_tool"].items()}
    bad_tools = [t for t, v in per_tool_verdicts.items() if v is not True]
    assert not bad_tools, f"AMR per_tool verdict_ok not True: {per_tool_verdicts}"
    # overall.catastrophic_bumps should be the union across tools
    union = []
    for p in amr["per_tool"].values():
        union.extend(p["evaluation"]["catastrophic_bumps"])
    assert overall["catastrophic_bumps"] == [] and union == [], (
        f"AMR overall.catastrophic_bumps={overall['catastrophic_bumps']}, per-tool union={union}"
    )


def test_mu_amr_tolerance_agreement_and_strict_tightness(mu: dict, amr: dict) -> None:
    mu_thr = mu["meta"]["max_noise_bump_pp"]
    amr_thr = amr["constants"]["hard_bump_threshold_pp"]
    assert math.isclose(mu_thr, amr_thr, abs_tol=DELTA_TOL), (
        f"MU.max_noise_bump_pp={mu_thr} != AMR.constants.hard_bump_threshold_pp={amr_thr}"
    )
    # MU's largest bump must be strictly below the shared threshold (the
    # 'cache-sim is sharper than the anchors' guarantee).
    mu_largest = mu["meta"]["largest_bump_pp"]
    assert mu_largest < amr_thr, (
        f"MU.largest_bump_pp={mu_largest} not strictly < shared threshold={amr_thr}; "
        "cache-sim monotonicity has degraded to anchor-tolerance levels"
    )
