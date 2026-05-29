"""Derivation parity gate for ``wiki/data/cross_generator_gap_parity.json``.

Locks the three-way reconciliation reducer that joins per-(app, policy,
L3) gap_pp triples across:
  - ``oracle_gap.json#rows``               (raw cells)
  - ``oracle_gap_auc.json#per_app.trajectory_by_policy`` (averaged)
  - ``cache_sensitivity_slope.json#per_app[app][pol]``   (octave-decorated)

Any drift in the PAPER_L3 scope filter, the raw aggregator
(`statistics.mean` rounded to 4 dp — NOT fmean and NOT bespoke
median), the AUC pull (raw float, no rounding), the slope pull
priority (top-level ``gap_at_{l3}`` keys take precedence; octave
records are consulted via ``setdefault`` only as a fallback for the
4MB midpoint), the spread reducer (``max(present) − min(present)``
on cells with at least one source), the agree predicate (spread ≤
TOLERANCE AND all three sources present — partial cells are NEVER
``agree=True``), the mismatch filter (``not agree``), or the meta
counters (``n_full_triple_cells`` = all_three_present count) trips
a test before the dashboard re-publishes the "all three aggregators
agree" parity headline.

Mirrors ``build_payload()`` from
``scripts/experiments/ecg/cross_generator_gap_parity.py`` verbatim.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "cross_generator_gap_parity.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"
AUC_PATH = WIKI_DATA / "oracle_gap_auc.json"
SLOPE_PATH = WIKI_DATA / "cache_sensitivity_slope.json"

PAPER_L3 = ("1MB", "4MB", "8MB")
TOLERANCE = 1e-3


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def raw():
    if not ORACLE_PATH.exists():
        pytest.skip(f"missing {ORACLE_PATH}")
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def auc():
    if not AUC_PATH.exists():
        pytest.skip(f"missing {AUC_PATH}")
    return json.loads(AUC_PATH.read_text())


@pytest.fixture(scope="module")
def slope():
    if not SLOPE_PATH.exists():
        pytest.skip(f"missing {SLOPE_PATH}")
    return json.loads(SLOPE_PATH.read_text())


@pytest.fixture(scope="module")
def expected(raw, auc, slope):
    """Re-derive raw_means, auc_vals, slope_vals from the three
    upstreams using the same rules as the generator."""
    raw_acc = defaultdict(list)
    for r in raw["rows"]:
        if r["l3_size"] not in PAPER_L3:
            continue
        raw_acc[(r["app"], r["policy"], r["l3_size"])].append(float(r["gap_pp"]))
    raw_means = {k: round(statistics.mean(vs), 4) for k, vs in raw_acc.items()}
    raw_counts = {k: len(vs) for k, vs in raw_acc.items()}

    auc_vals = {}
    for app, blob in auc["per_app"].items():
        for pol, traj in blob["trajectory_by_policy"].items():
            for l3, v in traj.items():
                if l3 in PAPER_L3:
                    auc_vals[(app, pol, l3)] = float(v)

    slope_vals = {}
    for app, app_blob in slope["per_app"].items():
        for pol, blob in app_blob.items():
            for l3 in PAPER_L3:
                key = f"gap_at_{l3}"
                if key in blob:
                    slope_vals[(app, pol, l3)] = float(blob[key])
            for oct_ in blob.get("octaves", []):
                if oct_.get("from") in PAPER_L3:
                    slope_vals.setdefault(
                        (app, pol, oct_["from"]), float(oct_["gap_from"])
                    )
                if oct_.get("to") in PAPER_L3:
                    slope_vals.setdefault(
                        (app, pol, oct_["to"]), float(oct_["gap_to"])
                    )
    return raw_means, raw_counts, auc_vals, slope_vals


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "mismatches", "cells"}


def test_meta_fields(artifact):
    expected = {
        "sources", "scope_l3_sizes", "tolerance_pp",
        "n_cells_checked", "n_mismatches", "n_full_triple_cells",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing: {missing}"


def test_meta_constants(artifact):
    m = artifact["meta"]
    assert tuple(m["scope_l3_sizes"]) == PAPER_L3
    assert m["tolerance_pp"] == TOLERANCE


def test_meta_sources_keys(artifact):
    assert set(artifact["meta"]["sources"].keys()) == {
        "oracle_gap", "oracle_gap_auc", "cache_sensitivity",
    }


def test_cell_record_fields(artifact):
    expected = {
        "app", "policy", "l3_size",
        "raw_mean_gap_pp", "auc_trajectory_gap_pp", "slope_gap_pp",
        "spread_pp", "n_graphs_in_raw",
        "all_three_present", "agree",
    }
    for c in artifact["cells"]:
        assert set(c.keys()) == expected, "cell field-set drift"


def test_mismatches_are_subset_of_cells(artifact):
    cell_keys = {(c["app"], c["policy"], c["l3_size"]) for c in artifact["cells"]}
    for m in artifact["mismatches"]:
        assert (m["app"], m["policy"], m["l3_size"]) in cell_keys


def test_cells_only_use_paper_l3_sizes(artifact):
    for c in artifact["cells"]:
        assert c["l3_size"] in PAPER_L3, (
            f"non-paper L3 leaked: {c['l3_size']}"
        )


# ----------------------------------------------------------------------
# Group B: per-source value parity
# ----------------------------------------------------------------------

def test_cell_raw_mean_matches_statistics_mean(artifact, expected):
    raw_means, _, _, _ = expected
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        assert c["raw_mean_gap_pp"] == raw_means.get(k), (
            f"{k}: raw_mean drift {c['raw_mean_gap_pp']} ≠ {raw_means.get(k)}"
        )


def test_cell_n_graphs_in_raw_matches_count(artifact, expected):
    _, raw_counts, _, _ = expected
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        assert c["n_graphs_in_raw"] == raw_counts.get(k), (
            f"{k}: n_graphs_in_raw drift"
        )


def test_cell_auc_value_matches_trajectory(artifact, expected):
    _, _, auc_vals, _ = expected
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        assert c["auc_trajectory_gap_pp"] == auc_vals.get(k), (
            f"{k}: auc value drift"
        )


def test_cell_slope_value_matches_priority_pull(artifact, expected):
    """Slope value pulls FIRST from top-level gap_at_{l3} keys, then
    SETDEFAULTs from octave records — the priority order is load-bearing
    for the 4MB midpoint (which isn't a top-level slope field)."""
    _, _, _, slope_vals = expected
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        assert c["slope_gap_pp"] == slope_vals.get(k), (
            f"{k}: slope value drift {c['slope_gap_pp']} "
            f"≠ {slope_vals.get(k)}"
        )


def test_cells_union_covers_every_upstream_key(artifact, expected):
    raw_means, _, auc_vals, slope_vals = expected
    union = set(raw_means) | set(auc_vals) | set(slope_vals)
    cell_keys = {(c["app"], c["policy"], c["l3_size"]) for c in artifact["cells"]}
    assert cell_keys == union, (
        f"missing keys: {sorted(union - cell_keys)[:5]} "
        f"unexpected: {sorted(cell_keys - union)[:5]}"
    )


def test_cells_sorted_by_key(artifact):
    """Generator iterates `for k in sorted(keys)` — emission order is
    lex by (app, policy, l3_size)."""
    keys = [(c["app"], c["policy"], c["l3_size"]) for c in artifact["cells"]]
    assert keys == sorted(keys), "cells not lex-sorted by (app, policy, l3)"


# ----------------------------------------------------------------------
# Group C: reducer parity (spread / agree / all_three_present)
# ----------------------------------------------------------------------

def test_all_three_present_matches_field_nullness(artifact):
    for c in artifact["cells"]:
        expected = (
            c["raw_mean_gap_pp"] is not None
            and c["auc_trajectory_gap_pp"] is not None
            and c["slope_gap_pp"] is not None
        )
        assert c["all_three_present"] is expected, (
            f"{c['app']}/{c['policy']}/{c['l3_size']}: "
            f"all_three_present drift"
        )


def test_spread_matches_max_minus_min_over_present_sources(artifact):
    """spread_pp = max(present) − min(present), rounded to 6 dp."""
    for c in artifact["cells"]:
        present = [
            v for v in (c["raw_mean_gap_pp"], c["auc_trajectory_gap_pp"], c["slope_gap_pp"])
            if v is not None
        ]
        if not present:
            pytest.fail(
                f"{c['app']}/{c['policy']}/{c['l3_size']}: "
                f"cell with no present source emitted"
            )
        expected = round(max(present) - min(present), 6)
        assert c["spread_pp"] == expected, (
            f"{c['app']}/{c['policy']}/{c['l3_size']}: "
            f"spread {c['spread_pp']} ≠ {expected}"
        )


def test_agree_requires_within_tol_and_all_three_present(artifact):
    """agree = (spread ≤ TOLERANCE) AND all_three_present. Cells
    missing any source MUST be agree=False — partial agreement does
    not count."""
    for c in artifact["cells"]:
        expected = c["spread_pp"] <= TOLERANCE and c["all_three_present"]
        assert c["agree"] is expected, (
            f"{c['app']}/{c['policy']}/{c['l3_size']}: "
            f"agree {c['agree']} ≠ {expected} "
            f"(spread {c['spread_pp']}, all3 {c['all_three_present']})"
        )


def test_mismatches_equal_non_agreeing_cells(artifact):
    """mismatches list = [c for c in cells if not c['agree']]."""
    expected = [c for c in artifact["cells"] if not c["agree"]]
    assert artifact["mismatches"] == expected, (
        f"mismatches drift — expected {len(expected)}, "
        f"got {len(artifact['mismatches'])}"
    )


# ----------------------------------------------------------------------
# Group D: meta-counter parity
# ----------------------------------------------------------------------

def test_meta_n_cells_checked_matches(artifact):
    assert artifact["meta"]["n_cells_checked"] == len(artifact["cells"])


def test_meta_n_mismatches_matches(artifact):
    assert artifact["meta"]["n_mismatches"] == len(artifact["mismatches"])


def test_meta_n_full_triple_cells_matches(artifact):
    expected = sum(1 for c in artifact["cells"] if c["all_three_present"])
    assert artifact["meta"]["n_full_triple_cells"] == expected


# ----------------------------------------------------------------------
# Group E: end-to-end sanity / claim invariant
# ----------------------------------------------------------------------

def test_spread_within_tol_implies_agree_when_all_three_present(artifact):
    for c in artifact["cells"]:
        if c["all_three_present"] and c["spread_pp"] <= TOLERANCE:
            assert c["agree"] is True


def test_agree_implies_spread_within_tol(artifact):
    for c in artifact["cells"]:
        if c["agree"]:
            assert c["spread_pp"] <= TOLERANCE


def test_n_mismatches_zero_means_empty_mismatches_list(artifact):
    if artifact["meta"]["n_mismatches"] == 0:
        assert artifact["mismatches"] == []


def test_at_least_one_cell_emitted(artifact):
    """Sanity: parity gate must have run over the corpus."""
    assert artifact["meta"]["n_cells_checked"] > 0
