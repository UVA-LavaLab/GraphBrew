"""Derivation parity gate (GSR-Der) for `gem5_slope_replay.json`.

This artifact is the gem5-side anchor for the cross-tool agreement gates
(ACC-Der, ACT-Der, AMR-Der downstream). The generator at
``scripts/experiments/ecg/gem5_slope_replay.py`` walks the gem5 anchor
JSON (`gem5_anchor.json`), reshapes cells by (app, graph, l3_size),
computes per-(app,graph,policy) OLS slope on the log2(kB) L3 axis, and
emits a 4-clause verdict matrix.

The gate locks every load-bearing rule that downstream gates depend on:

* Group A — schema-style pinned constants (L3-axis log2 mapping,
  EXPECTED_SIZES tuple, POLICIES tuple, HELP_FLOOR_PP_OCTAVE).
* Group B — per-cell construction rules (cell skipped iff incomplete,
  policy skipped iff missing in any size, miss-rate × 100 conversion,
  4-dp rounding).
* Group C — per-policy aggregator (manual median with pair-average,
  mean, n counter; rounding pins).
* Group D — verdict block (4 ordered checks with locked polarity,
  monotonic_violation predicate, lru_minus_grasp INFORMATIONAL note,
  PASS conjunction).
* Group E — byte parity with the committed JSON.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "gem5_slope_replay.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
ANCHOR_PATH = REPO_ROOT / "wiki" / "data" / "gem5_anchor.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("gem5_slope_replay", GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def artifact():
    return json.loads(JSON_PATH.read_text())


@pytest.fixture(scope="module")
def anchor():
    return json.loads(ANCHOR_PATH.read_text())


# ---------------------------------------------------------------- Group A
def test_log2_axis_pinned(gen):
    assert gen.ANCHOR_L3_LOG2_KB == {
        "4kB": 2.0,
        "32kB": 5.0,
        "256kB": 8.0,
        "2MB": 11.0,
    }


def test_expected_sizes_is_tuple_in_axis_order(gen):
    assert gen.EXPECTED_SIZES == ("4kB", "32kB", "256kB", "2MB")
    assert isinstance(gen.EXPECTED_SIZES, tuple)


def test_policies_tuple_pinned(gen):
    assert gen.POLICIES == ("GRASP", "LRU", "SRRIP")
    assert isinstance(gen.POLICIES, tuple)


def test_help_floor_pinned(gen):
    assert gen.HELP_FLOOR_PP_OCTAVE == -1.0


def test_meta_axis_block_mirrors_constants(gen, artifact):
    m = artifact["meta"]
    assert m["l3_axis_log2_kb"] == gen.ANCHOR_L3_LOG2_KB
    assert m["expected_sizes"] == list(gen.EXPECTED_SIZES)
    assert m["policies"] == list(gen.POLICIES)
    assert m["help_floor_pp_octave"] == gen.HELP_FLOOR_PP_OCTAVE


def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_cell"}


# ---------------------------------------------------------------- Group B
def _reshape_anchor(anchor):
    by_cell = defaultdict(dict)
    for c in anchor["cells"]:
        by_cell[(c["app"], c["graph"])][c["l3_size"]] = {
            p: float(v) * 100.0 for p, v in c["miss_rate_by_policy"].items()
        }
    return by_cell


def test_complete_cells_count_matches_meta(artifact, anchor):
    by_cell = _reshape_anchor(anchor)
    complete = sum(
        1 for sweep in by_cell.values()
        if all(s in sweep for s in ("4kB", "32kB", "256kB", "2MB"))
    )
    assert artifact["meta"]["n_cells"] == complete


def test_n_cell_policy_records_matches_per_cell_len(artifact):
    assert artifact["meta"]["n_cell_policy_records"] == len(artifact["per_cell"])


def test_per_cell_records_only_complete_policy_coverage(artifact, anchor):
    by_cell = _reshape_anchor(anchor)
    for r in artifact["per_cell"]:
        sweep = by_cell[(r["app"], r["graph"])]
        for s in ("4kB", "32kB", "256kB", "2MB"):
            assert s in sweep, f"missing size {s} for {(r['app'],r['graph'])}"
            assert r["policy"] in sweep[s], (
                f"policy {r['policy']} missing at size {s}"
            )


def test_miss_pp_converted_from_rate_times_100(artifact, anchor):
    by_cell = _reshape_anchor(anchor)
    for r in artifact["per_cell"]:
        sweep = by_cell[(r["app"], r["graph"])]
        for s in ("4kB", "32kB", "256kB", "2MB"):
            expected = round(sweep[s][r["policy"]], 4)
            assert r["miss_pp_by_size"][s] == expected


def test_per_cell_record_keys(artifact):
    for r in artifact["per_cell"]:
        assert set(r.keys()) == {"app", "graph", "policy", "slope_pp_per_octave", "miss_pp_by_size"}
        assert list(r["miss_pp_by_size"].keys()) == ["4kB", "32kB", "256kB", "2MB"]


def test_per_cell_slope_rounded_4dp_matches_recomputed_ols(artifact, anchor, gen):
    by_cell = _reshape_anchor(anchor)
    for r in artifact["per_cell"]:
        sweep = by_cell[(r["app"], r["graph"])]
        xs = [gen.ANCHOR_L3_LOG2_KB[s] for s in ("4kB", "32kB", "256kB", "2MB")]
        ys = [sweep[s][r["policy"]] for s in ("4kB", "32kB", "256kB", "2MB")]
        expected = gen._ols_slope(xs, ys)
        assert expected is not None and not math.isnan(expected)
        assert r["slope_pp_per_octave"] == round(expected, 4)


def test_per_cell_iteration_order_sorted_by_cell_then_policy(artifact):
    """Cells iterated in sorted((app,graph)) order; within each cell policies in POLICIES order."""
    # Build expected ordering by replaying generator's deterministic walk.
    # Distinct (app,graph) pairs appear contiguously in sorted order.
    seen_pairs = []
    expected_policy_order = ("GRASP", "LRU", "SRRIP")
    for r in artifact["per_cell"]:
        seen_pairs.append((r["app"], r["graph"]))
    pair_blocks = []
    i = 0
    while i < len(seen_pairs):
        pair = seen_pairs[i]
        j = i
        while j < len(seen_pairs) and seen_pairs[j] == pair:
            j += 1
        pair_blocks.append(pair)
        i = j
    assert pair_blocks == sorted(set(seen_pairs))
    # Within each contiguous block of records for same pair, policies follow POLICIES order.
    pos = 0
    for pair in pair_blocks:
        block = []
        while pos < len(artifact["per_cell"]) and (
            artifact["per_cell"][pos]["app"] == pair[0]
            and artifact["per_cell"][pos]["graph"] == pair[1]
        ):
            block.append(artifact["per_cell"][pos]["policy"])
            pos += 1
        for p in block:
            assert p in expected_policy_order
        # Block respects POLICIES order (subset of it preserving relative order).
        indices = [expected_policy_order.index(p) for p in block]
        assert indices == sorted(indices)


# ---------------------------------------------------------------- Group C
def test_per_policy_block_keys_match_policies(gen, artifact):
    assert set(artifact["meta"]["per_policy"].keys()) == set(gen.POLICIES)


def test_per_policy_n_equals_count_of_records(artifact):
    for p in ("GRASP", "LRU", "SRRIP"):
        recs = [r for r in artifact["per_cell"] if r["policy"] == p]
        assert artifact["meta"]["per_policy"][p]["n"] == len(recs)


def test_per_policy_median_pair_average_for_even_n(gen):
    assert gen._median([1.0, 2.0]) == 1.5
    assert gen._median([1.0, 2.0, 3.0]) == 2.0
    assert gen._median([]) is None
    assert gen._median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_per_policy_mean_simple_average(gen):
    assert gen._mean([1.0, 2.0, 3.0]) == 2.0
    assert gen._mean([]) is None


def test_per_policy_median_mean_rounded_4dp_matches_recompute(artifact, gen):
    slopes = {p: [] for p in ("GRASP", "LRU", "SRRIP")}
    by_cell = _reshape_anchor(json.loads(ANCHOR_PATH.read_text()))
    for (app, graph), sweep in sorted(by_cell.items()):
        if not all(s in sweep for s in ("4kB", "32kB", "256kB", "2MB")):
            continue
        for policy in ("GRASP", "LRU", "SRRIP"):
            if not all(policy in sweep[s] for s in ("4kB", "32kB", "256kB", "2MB")):
                continue
            xs = [gen.ANCHOR_L3_LOG2_KB[s] for s in ("4kB", "32kB", "256kB", "2MB")]
            ys = [sweep[s][policy] for s in ("4kB", "32kB", "256kB", "2MB")]
            slopes[policy].append(gen._ols_slope(xs, ys))
    for p in ("GRASP", "LRU", "SRRIP"):
        expected_median = gen._median(slopes[p])
        expected_mean = gen._mean(slopes[p])
        assert artifact["meta"]["per_policy"][p]["median"] == (
            round(expected_median, 4) if expected_median is not None else None
        )
        assert artifact["meta"]["per_policy"][p]["mean"] == (
            round(expected_mean, 4) if expected_mean is not None else None
        )


def test_lru_minus_grasp_rounded_4dp(artifact):
    pp = artifact["meta"]["per_policy"]
    expected = pp["LRU"]["median"] - pp["GRASP"]["median"]
    assert artifact["meta"]["lru_minus_grasp_pp_oct"] == round(expected, 4)


def test_srrip_minus_grasp_rounded_4dp(artifact):
    pp = artifact["meta"]["per_policy"]
    expected = pp["SRRIP"]["median"] - pp["GRASP"]["median"]
    assert artifact["meta"]["srrip_minus_grasp_pp_oct"] == round(expected, 4)


def test_lru_minus_grasp_note_is_informational(artifact):
    note = artifact["meta"]["lru_minus_grasp_note"]
    assert "INFORMATIONAL" in note
    assert "not gated" in note


# ---------------------------------------------------------------- Group D
def test_verdict_checks_keys_pinned(artifact):
    assert list(artifact["meta"]["verdict_checks"].keys()) == [
        "cache_monotonic_every_cell",
        "all_per_policy_medians_negative",
        "srrip_at_least_as_steep_as_grasp",
        "grasp_below_help_floor",
    ]


def test_check_monotonic_zero_violations(artifact):
    expected = len(artifact["meta"]["monotonic_violations"]) == 0
    assert artifact["meta"]["verdict_checks"]["cache_monotonic_every_cell"] == expected


def test_monotonic_violation_predicate_is_small_le_large(artifact, anchor, gen):
    """Violation iff miss(4kB) <= miss(2MB) — strict-monotonicity polarity."""
    by_cell = _reshape_anchor(anchor)
    expected = []
    for (app, graph), sweep in sorted(by_cell.items()):
        if not all(s in sweep for s in ("4kB", "32kB", "256kB", "2MB")):
            continue
        for policy in ("GRASP", "LRU", "SRRIP"):
            if not all(policy in sweep[s] for s in ("4kB", "32kB", "256kB", "2MB")):
                continue
            small = sweep["4kB"][policy]
            large = sweep["2MB"][policy]
            if small <= large:
                expected.append({
                    "app": app, "graph": graph, "policy": policy,
                    "miss_small": small, "miss_large": large,
                })
    assert artifact["meta"]["monotonic_violations"] == expected


def test_check_all_medians_negative_strict_lt_zero(artifact):
    pp = artifact["meta"]["per_policy"]
    expected = all(
        pp[p]["median"] is not None and pp[p]["median"] < 0
        for p in ("GRASP", "LRU", "SRRIP")
    )
    assert artifact["meta"]["verdict_checks"]["all_per_policy_medians_negative"] == expected


def test_check_srrip_at_least_as_steep_inclusive_le(artifact):
    pp = artifact["meta"]["per_policy"]
    expected = pp["SRRIP"]["median"] <= pp["GRASP"]["median"]
    assert artifact["meta"]["verdict_checks"]["srrip_at_least_as_steep_as_grasp"] == expected


def test_check_grasp_below_help_floor_strict_lt(artifact, gen):
    expected = artifact["meta"]["per_policy"]["GRASP"]["median"] < gen.HELP_FLOOR_PP_OCTAVE
    assert artifact["meta"]["verdict_checks"]["grasp_below_help_floor"] == expected


def test_verdict_pass_iff_all_checks(artifact):
    expected = "PASS" if all(artifact["meta"]["verdict_checks"].values()) else "FAIL"
    assert artifact["meta"]["verdict"] == expected


def test_anchor_source_name(artifact):
    assert artifact["meta"]["anchor_source"] == "gem5_anchor.json"


# ---------------------------------------------------------------- Group E
def test_full_artifact_byte_parity(tmp_path):
    out_json = tmp_path / "gem5_slope_replay.json"
    out_md = tmp_path / "gem5_slope_replay.md"
    res = subprocess.run(
        [
            sys.executable,
            str(GEN_PATH),
            "--anchor-json", str(ANCHOR_PATH),
            "--json-out", str(out_json),
            "--md-out", str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verdict=PASS" in res.stdout
    assert out_json.read_text() == JSON_PATH.read_text()
