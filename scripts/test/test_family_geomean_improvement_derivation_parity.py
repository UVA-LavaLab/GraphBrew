"""Derivation-parity gate for ``wiki/data/family_geomean_improvement.json``.

The geomean artifact reports the *size* (not just the direction) of each
(family × app × policy) miss-rate improvement vs LRU, with a 95%
percentile-bootstrap CI on the geomean ratio. It is the artifact the
paper cites for headline magnitudes (`+24% on social, +12% on web`,
etc.), so a silent change to:

* the seeded bootstrap (B=2000, seed=1729),
* the percentile index formula (left-floor lo, right-ceil hi - 1, clamped),
* the LRU-baseline-required filter,
* the >= 1MB paper-L3 scope, or
* the (1 - ratio) * 100 → improve_pct conversion (with the lo/hi swap)

would let the paper headlines drift without tripping any other gate.

This module re-derives the artifact end-to-end from ``oracle_gap.json``
and asserts byte-for-byte equivalence with the committed JSON, plus the
load-bearing invariants the generator enforces.

5 groups, 22 tests total.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "family_geomean_improvement.json"
SOURCE = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
GENERATOR = REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_geomean_improvement.py"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 1729
ALPHA = 0.05


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("family_geomean_local", GENERATOR)
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
    return gen.build_payload(SOURCE)


# ----------------------------------------------------------------------
# Group 1: cross-source byte equivalence + meta invariants
# ----------------------------------------------------------------------

def test_regenerated_matches_committed_artifact(regenerated: dict[str, Any], artifact: dict[str, Any]) -> None:
    """Seeded bootstrap means re-running compute() must yield byte-identical JSON."""
    assert json.dumps(regenerated, sort_keys=True) == json.dumps(artifact, sort_keys=True)


def test_top_level_keys_exact(artifact: dict[str, Any]) -> None:
    assert set(artifact.keys()) == {
        "headline_improvements_ge_10pct",
        "headline_regressions_ci_strict",
        "meta",
        "records",
    }


def test_meta_has_load_bearing_constants(artifact: dict[str, Any]) -> None:
    meta = artifact["meta"]
    assert meta["bootstrap_iters"] == BOOTSTRAP_ITERS
    assert meta["bootstrap_seed"] == BOOTSTRAP_SEED
    assert meta["alpha"] == ALPHA
    assert tuple(meta["scope_l3_sizes"]) == PAPER_L3_SIZES
    assert meta["source"] == "wiki/data/oracle_gap.json"


def test_meta_record_counts_match(artifact: dict[str, Any]) -> None:
    records = artifact["records"]
    meta = artifact["meta"]
    assert meta["n_records"] == len(records)
    assert meta["n_ci_strict_improvements"] == sum(
        1 for r in records if r.get("ci_strict_improvement_vs_lru")
    )
    assert meta["n_ci_strict_regressions"] == sum(
        1 for r in records if r.get("ci_strict_regression_vs_lru")
    )


# ----------------------------------------------------------------------
# Group 2: record shape + sort order
# ----------------------------------------------------------------------

def test_records_sorted_by_family_app_policy(artifact: dict[str, Any]) -> None:
    keys = [(r["family"], r["app"], r["policy"]) for r in artifact["records"]]
    assert keys == sorted(keys), f"records not sorted: first deviation in {keys}"


def test_no_lru_records_present(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        assert r["policy"] != "LRU", f"LRU should never be a target policy: {r}"


def test_every_cell_has_paper_l3_size(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        for c in r.get("cells", []):
            assert c["l3_size"] in PAPER_L3_SIZES, (
                f"cell outside paper L3 scope: {c} in record {r['family']}/{r['app']}/{r['policy']}"
            )


def test_every_cell_has_positive_miss_rates(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        for c in r.get("cells", []):
            assert c["mr_lru"] > 0, f"non-positive LRU mr: {c}"
            assert c["mr_policy"] > 0, f"non-positive policy mr: {c}"


def test_cell_ratio_equals_mr_policy_over_mr_lru(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        for c in r.get("cells", []):
            expected = c["mr_policy"] / c["mr_lru"]
            assert math.isclose(c["ratio"], expected, rel_tol=1e-12, abs_tol=1e-12), (
                f"cell ratio mismatch: {c}"
            )


def test_skipped_records_have_under_2_cells(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            assert r["skipped_reason"] == "insufficient_cells_for_bootstrap_min_2"
            assert r["n_cells"] < 2, f"skipped record with n_cells>=2: {r}"
        else:
            assert r["n_cells"] >= 2, f"non-skipped record with n_cells<2: {r}"


# ----------------------------------------------------------------------
# Group 3: geomean + CI math invariants
# ----------------------------------------------------------------------

def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def test_geomean_ratio_matches_cell_geomean(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            continue
        ratios = [c["ratio"] for c in r["cells"]]
        expected = round(_geomean(ratios), 6)
        assert r["geomean_ratio"] == expected, (
            f"geomean mismatch: {r['family']}/{r['app']}/{r['policy']} "
            f"recomputed={expected} stored={r['geomean_ratio']}"
        )


def test_ci_lo_le_geomean_le_ci_hi(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            continue
        # Allow 1 ulp slack from rounding.
        assert r["ci_lo_ratio"] <= r["geomean_ratio"] + 1e-6
        assert r["geomean_ratio"] <= r["ci_hi_ratio"] + 1e-6


def test_improve_pct_ordering_matches_ratio_swap(artifact: dict[str, Any]) -> None:
    """ci_lo_improve_pct ≤ geomean_improve_pct ≤ ci_hi_improve_pct
    (and the bounds are derived from the SWAPPED ratio bounds: high ratio
    → low improvement, low ratio → high improvement)."""
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            continue
        assert r["ci_lo_improve_pct"] <= r["geomean_improve_pct"] + 0.002, r
        assert r["geomean_improve_pct"] <= r["ci_hi_improve_pct"] + 0.002, r
        # Direction-of-derivation: ci_lo_improve_pct comes from ci_hi_ratio,
        # so when ci_hi_ratio > 1 the ci_lo_improve_pct must be <= 0.
        if r["ci_hi_ratio"] > 1.0:
            assert r["ci_lo_improve_pct"] <= 0.0 + 0.002, r
        if r["ci_lo_ratio"] < 1.0:
            assert r["ci_hi_improve_pct"] >= 0.0 - 0.002, r


def test_improve_pct_matches_raw_geomean_reconstruction(artifact: dict[str, Any]) -> None:
    """Exact check against the raw geomean computed from cells: improve_pct =
    round((1 - raw_geomean_from_cells) * 100, 3)."""
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            continue
        ratios = [c["ratio"] for c in r["cells"]]
        raw = _geomean(ratios)
        assert r["geomean_improve_pct"] == round((1.0 - raw) * 100.0, 3), (
            f"{r['family']}/{r['app']}/{r['policy']}: "
            f"expected improve_pct={round((1.0 - raw) * 100.0, 3)} "
            f"got={r['geomean_improve_pct']}"
        )


def test_ci_strict_improvement_iff_ci_hi_ratio_lt_1(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            continue
        expected = r["ci_hi_ratio"] < 1.0
        assert r["ci_strict_improvement_vs_lru"] is expected, r


def test_ci_strict_regression_iff_ci_lo_ratio_gt_1(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            continue
        expected = r["ci_lo_ratio"] > 1.0
        assert r["ci_strict_regression_vs_lru"] is expected, r


def test_no_record_is_both_strict_improvement_and_regression(artifact: dict[str, Any]) -> None:
    for r in artifact["records"]:
        if r.get("skipped_reason"):
            continue
        assert not (
            r["ci_strict_improvement_vs_lru"] and r["ci_strict_regression_vs_lru"]
        ), r


# ----------------------------------------------------------------------
# Group 4: headline list invariants
# ----------------------------------------------------------------------

def test_headline_improvements_filter(artifact: dict[str, Any]) -> None:
    """Headline = ci_strict_improvement AND geomean_improve_pct >= 10."""
    in_headline_keys = {
        (r["family"], r["app"], r["policy"])
        for r in artifact["headline_improvements_ge_10pct"]
    }
    expected_keys = {
        (r["family"], r["app"], r["policy"])
        for r in artifact["records"]
        if not r.get("skipped_reason")
        and r["ci_strict_improvement_vs_lru"]
        and r["geomean_improve_pct"] >= 10.0
    }
    assert in_headline_keys == expected_keys


def test_headline_improvements_sorted_desc_by_improve_pct(artifact: dict[str, Any]) -> None:
    pcts = [r["geomean_improve_pct"] for r in artifact["headline_improvements_ge_10pct"]]
    assert pcts == sorted(pcts, reverse=True), f"not sorted desc: {pcts}"


def test_headline_regressions_filter_and_sort(artifact: dict[str, Any]) -> None:
    in_headline_keys = {
        (r["family"], r["app"], r["policy"])
        for r in artifact["headline_regressions_ci_strict"]
    }
    expected_keys = {
        (r["family"], r["app"], r["policy"])
        for r in artifact["records"]
        if not r.get("skipped_reason") and r["ci_strict_regression_vs_lru"]
    }
    assert in_headline_keys == expected_keys
    # sorted ASC by improve_pct (most negative first)
    pcts = [r["geomean_improve_pct"] for r in artifact["headline_regressions_ci_strict"]]
    assert pcts == sorted(pcts), f"regressions not sorted asc: {pcts}"


def test_headline_lists_drop_cells_field(artifact: dict[str, Any]) -> None:
    """Headlines must not carry the full cells list (compact projection)."""
    for r in artifact["headline_improvements_ge_10pct"]:
        assert "cells" not in r, f"headline improvement leaked cells: {r}"
    for r in artifact["headline_regressions_ci_strict"]:
        assert "cells" not in r, f"headline regression leaked cells: {r}"


# ----------------------------------------------------------------------
# Group 5: cross-source filter parity (LRU-required + L3 scope)
# ----------------------------------------------------------------------

def test_lru_required_filter_holds_against_oracle(artifact: dict[str, Any]) -> None:
    """Every cell in records.cells must have an LRU baseline in oracle_gap.json
    for the same (graph, app, l3_size). Cells lacking LRU must not appear."""
    oracle = json.loads(SOURCE.read_text())
    lru_keys = {
        (r["graph"], r["app"], r["l3_size"])
        for r in oracle["rows"]
        if r["policy"] == "LRU" and float(r["miss_rate"]) > 0
    }
    for r in artifact["records"]:
        for c in r["cells"]:
            assert (c["graph"], r["app"], c["l3_size"]) in lru_keys, (
                f"cell present without LRU baseline: {c} in "
                f"{r['family']}/{r['app']}/{r['policy']}"
            )


def test_no_record_appears_outside_paired_universe(artifact: dict[str, Any]) -> None:
    """Re-derive the paired universe from oracle_gap.json directly and assert
    the artifact's (family, app, policy) keyset is a subset."""
    oracle = json.loads(SOURCE.read_text())
    by_cell: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in oracle["rows"]:
        if row["l3_size"] not in PAPER_L3_SIZES:
            continue
        if not row.get("family"):
            continue
        by_cell[(row["family"], row["app"], row["graph"], row["l3_size"])][row["policy"]] = float(row["miss_rate"])
    paired_keys: set[tuple[str, str, str]] = set()
    for (family, app, _graph, _l3), pmap in by_cell.items():
        lru = pmap.get("LRU")
        if lru is None or lru <= 0:
            continue
        for pol, mr in pmap.items():
            if pol == "LRU" or mr <= 0:
                continue
            paired_keys.add((family, app, pol))
    artifact_keys = {(r["family"], r["app"], r["policy"]) for r in artifact["records"]}
    assert artifact_keys == paired_keys, (
        f"artifact-vs-oracle paired-key drift: "
        f"missing_from_artifact={paired_keys - artifact_keys}, "
        f"extra_in_artifact={artifact_keys - paired_keys}"
    )
