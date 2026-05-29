"""Gate 119 — cross_tool_slope_universality.json arithmetic + verdict.

Locks the roll-up that pulls the per-policy median capacity-sensitivity
slope from each tool (cache-sim, gem5, sniper) and enforces three
cross-tool invariants:

    (1) every (tool, policy) median slope is negative (extra cache must
        not, on average, hurt any policy on any tool);
    (2) every median lies within the physical band
        [MIN_SLOPE_PP_OCT, MAX_SLOPE_PP_OCT] (catches both runaway-
        steep regressions and near-zero collapse);
    (3) no tool's steepness span (max - min across its policies)
        exceeds STEEPNESS_SPAN_CEILING_PP_OCT (catches partial
        regressions where one policy collapses while siblings stay
        steep).

This gate is the central trust anchor for the paper's "cache helps
every policy on every simulator" claim — gates 66/70/71 check each
tool individually; this artifact ties them together. Any regression
that flips a sign or collapses a policy on any tool will fail here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/cross_tool_slope_universality.json")
SOURCES = {
    "cache-sim": Path("wiki/data/capacity_sensitivity.json"),
    "gem5":      Path("wiki/data/gem5_slope_replay.json"),
    "sniper":    Path("wiki/data/sniper_slope_replay.json"),
}

MIN_SLOPE_PP_OCT = -25.0
MAX_SLOPE_PP_OCT = -0.5
STEEPNESS_SPAN_CEILING_PP_OCT = 5.0

MEDIAN_TOL = 1e-3
SPAN_TOL = 1e-3


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def sources():
    return {tool: json.loads(path.read_text()) for tool, path in SOURCES.items()}


def _source_median(tool: str, source_blob: dict, policy: str) -> float:
    if tool == "cache-sim":
        return float(source_blob["meta"]["policy_summary"][policy]["median_pp"])
    return float(source_blob["meta"]["per_policy"][policy]["median"])


# ── group 1: meta + thresholds ───────────────────────────────────────────


def test_meta_tools_and_constants(data):
    meta = data["meta"]
    assert meta["tools"] == ["cache-sim", "gem5", "sniper"]
    assert meta["min_slope_pp_oct"] == MIN_SLOPE_PP_OCT
    assert meta["max_slope_pp_oct"] == MAX_SLOPE_PP_OCT
    assert meta["steepness_span_ceiling_pp_oct"] == STEEPNESS_SPAN_CEILING_PP_OCT
    assert meta["verdict"] in ("PASS", "FAIL")


def test_tool_policies_sorted_and_match_medians(data):
    meta = data["meta"]
    for tool in meta["tools"]:
        assert meta["tool_policies"][tool] == sorted(meta["medians"][tool].keys())


# ── group 2: medians copied from source artifacts ─────────────────────────


def test_medians_copy_from_source_artifacts(data, sources):
    for tool in data["meta"]["tools"]:
        for pol, val in data["meta"]["medians"][tool].items():
            expected = _source_median(tool, sources[tool], pol)
            assert math.isclose(val, expected, abs_tol=MEDIAN_TOL), (
                f"{tool}/{pol}: median={val} expected={expected:.4f}"
            )


def test_no_extra_or_missing_policies_per_tool(data, sources):
    for tool, path in SOURCES.items():
        src = sources[tool]
        if tool == "cache-sim":
            src_pols = set(src["meta"]["policy_summary"].keys())
        else:
            src_pols = set(src["meta"]["per_policy"].keys())
        artifact_pols = set(data["meta"]["medians"][tool].keys())
        assert artifact_pols == src_pols, (
            f"{tool}: artifact={sorted(artifact_pols)} source={sorted(src_pols)}"
        )


# ── group 3: derived counts + spans ──────────────────────────────────────


def test_steepness_span_per_tool(data):
    for tool, med in data["meta"]["medians"].items():
        vals = list(med.values())
        expected_span = max(vals) - min(vals) if vals else 0.0
        assert math.isclose(
            data["meta"]["steepness_spans"][tool], expected_span, abs_tol=SPAN_TOL
        ), f"{tool}: span mismatch"


def test_expected_in_band_count_partitions_all_cells(data):
    meta = data["meta"]
    expected = sum(len(p) for p in meta["tool_policies"].values())
    assert meta["expected_in_band_count"] == expected


def test_in_band_count_matches_band_inclusion(data):
    meta = data["meta"]
    in_band = 0
    for tool, med in meta["medians"].items():
        for pol, val in med.items():
            if MIN_SLOPE_PP_OCT <= val <= MAX_SLOPE_PP_OCT:
                in_band += 1
    assert meta["in_band_count"] == in_band


# ── group 4: violations + verdict ────────────────────────────────────────


def test_violations_reproduce_three_categories(data):
    meta = data["meta"]
    expected = []
    for tool, med in meta["medians"].items():
        for pol, val in med.items():
            if val >= 0.0:
                expected.append({
                    "type": "positive_slope",
                    "tool": tool,
                    "policy": pol,
                    "value": round(val, 4),
                })
            if val < MIN_SLOPE_PP_OCT or val > MAX_SLOPE_PP_OCT:
                expected.append({
                    "type": "out_of_band",
                    "tool": tool,
                    "policy": pol,
                    "value": round(val, 4),
                    "band": [MIN_SLOPE_PP_OCT, MAX_SLOPE_PP_OCT],
                })
    for tool, span in meta["steepness_spans"].items():
        if span > STEEPNESS_SPAN_CEILING_PP_OCT:
            expected.append({
                "type": "steepness_span_exceeded",
                "tool": tool,
                "span": span,
                "ceiling": STEEPNESS_SPAN_CEILING_PP_OCT,
            })
    assert meta["violations"] == expected, (
        f"violations mismatch:\nartifact={meta['violations']}\nexpected={expected}"
    )


def test_verdict_checks_reproduce_invariants(data):
    meta = data["meta"]
    all_negative = all(
        v < 0.0 for med in meta["medians"].values() for v in med.values()
    )
    all_in_band = meta["in_band_count"] == meta["expected_in_band_count"]
    no_span_exceeded = all(
        s <= STEEPNESS_SPAN_CEILING_PP_OCT for s in meta["steepness_spans"].values()
    )
    assert meta["verdict_checks"]["all_tool_policy_medians_negative"] is all_negative
    assert meta["verdict_checks"]["all_tool_policy_medians_in_band"] is all_in_band
    assert meta["verdict_checks"]["no_tool_exceeds_steepness_span_ceiling"] is no_span_exceeded


def test_verdict_pass_iff_all_checks_pass(data):
    meta = data["meta"]
    expected = "PASS" if all(meta["verdict_checks"].values()) else "FAIL"
    assert meta["verdict"] == expected


def test_no_violations_when_verdict_pass(data):
    meta = data["meta"]
    if meta["verdict"] == "PASS":
        assert meta["violations"] == [], (
            f"PASS verdict but {len(meta['violations'])} violations present"
        )
