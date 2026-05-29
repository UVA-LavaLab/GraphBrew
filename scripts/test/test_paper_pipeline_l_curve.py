"""Tests for the L-curve aggregation+plot wiring in paper_pipeline.

The L-curve is the GRASP-paper canonical figure: x-axis = L3 cache size
(log scale), y-axis = miss rate, one line per policy. The figure must
appear automatically whenever cache_sim rows for a given (graph, app)
span at least three distinct L3 sizes (so the L-shape is visible).

These tests pin three invariants:
1. ``l_curve_rows`` groups rows by (graph, app) and drops groups with
   <3 distinct L3 sizes.
2. ``_l_curve_summary_rows`` deduplicates (policy, L3) repeats by
   averaging miss-rate and recording sample count.
3. ``plot_l_curve`` writes an SVG with finite size and ``save_figure``
   semantics (no crash when matplotlib is present; clean skip when not).
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("pp_l_curve_test_module", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pipeline():
    return _load_pipeline()


def _make_row(simulator, graph, app, policy, l3_size, l3_miss_rate, *, options=None):
    """Build a roi_matrix-like row that the aggregator accepts."""
    opt = options or f"-f results/graphs/{graph}/{graph}.sg -s -o 5"
    return {
        "simulator": simulator,
        "benchmark": app,
        "policy_label": policy,
        "policy": policy,
        "l3_size": l3_size,
        "l3_miss_rate": str(l3_miss_rate),
        "options": opt,
    }


def test_l_curve_rows_groups_by_graph_app(pipeline):
    rows = [
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "4kB", 0.40),
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "32kB", 0.30),
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "256kB", 0.20),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "4kB", 0.38),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "32kB", 0.29),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "256kB", 0.19),
        _make_row("cache_sim", "cit-Patents", "bc", "LRU", "4kB", 0.85),
        _make_row("cache_sim", "cit-Patents", "bc", "LRU", "32kB", 0.81),
        _make_row("cache_sim", "cit-Patents", "bc", "LRU", "256kB", 0.78),
    ]
    groups = pipeline.l_curve_rows(rows)
    assert set(groups) == {("email-Eu-core", "pr"), ("cit-Patents", "bc")}
    assert len(groups[("email-Eu-core", "pr")]) == 6
    assert len(groups[("cit-Patents", "bc")]) == 3


def test_l_curve_rows_drops_groups_under_three_sizes(pipeline):
    """An L-shape needs ≥3 L3 points; smaller groups must be filtered out."""
    rows = [
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "4kB", 0.40),
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "32kB", 0.30),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "4kB", 0.38),
    ]
    groups = pipeline.l_curve_rows(rows)
    assert groups == {}


def test_l_curve_rows_ignores_non_cache_sim(pipeline):
    """gem5/Sniper rows must be excluded — the L-curve is a cache_sim plot."""
    rows = [
        _make_row("gem5", "email-Eu-core", "pr", "LRU", "4kB", 0.40),
        _make_row("gem5", "email-Eu-core", "pr", "LRU", "32kB", 0.30),
        _make_row("gem5", "email-Eu-core", "pr", "LRU", "256kB", 0.20),
        _make_row("sniper", "email-Eu-core", "pr", "GRASP", "4kB", 0.38),
        _make_row("sniper", "email-Eu-core", "pr", "GRASP", "32kB", 0.29),
        _make_row("sniper", "email-Eu-core", "pr", "GRASP", "256kB", 0.19),
    ]
    assert pipeline.l_curve_rows(rows) == {}


def test_l_curve_rows_extracts_graph_from_options(pipeline):
    """When ``graph`` field is absent, fall back to parsing ``-f .../graphs/<g>/...``."""
    rows = [
        _make_row("cache_sim", "soc-pokec", "pr", "LRU", "4kB", 0.40,
                  options="-f results/graphs/soc-pokec/soc-pokec.sg -s"),
        _make_row("cache_sim", "soc-pokec", "pr", "LRU", "32kB", 0.30,
                  options="-f results/graphs/soc-pokec/soc-pokec.sg -s"),
        _make_row("cache_sim", "soc-pokec", "pr", "LRU", "256kB", 0.20,
                  options="-f results/graphs/soc-pokec/soc-pokec.sg -s"),
    ]
    # Force `graph` to be absent so the extractor must derive it.
    for r in rows:
        r.pop("graph", None)
    groups = pipeline.l_curve_rows(rows)
    assert set(groups) == {("soc-pokec", "pr")}


def test_l_curve_summary_rows_averages_duplicates(pipeline):
    rows = [
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "4kB", 0.40),
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "4kB", 0.42),  # duplicate
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "32kB", 0.30),
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "256kB", 0.20),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "4kB", 0.38),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "32kB", 0.29),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "256kB", 0.19),
    ]
    groups = pipeline.l_curve_rows(rows)
    summary = pipeline._l_curve_summary_rows(groups)
    lru_4kb = [r for r in summary if r["policy_label"] == "LRU" and r["l3_size"] == "4kB"]
    assert len(lru_4kb) == 1
    assert lru_4kb[0]["samples"] == 2
    assert abs(lru_4kb[0]["l3_miss_rate"] - 0.41) < 1e-9


def test_plot_l_curve_writes_svg(pipeline, tmp_path):
    rows = [
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "4kB", 0.40),
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "32kB", 0.30),
        _make_row("cache_sim", "email-Eu-core", "pr", "LRU", "256kB", 0.20),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "4kB", 0.38),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "32kB", 0.29),
        _make_row("cache_sim", "email-Eu-core", "pr", "GRASP", "256kB", 0.19),
    ]
    groups = pipeline.l_curve_rows(rows)
    key = ("email-Eu-core", "pr")
    out = tmp_path / "l_curve.svg"
    pipeline.plot_l_curve(out, key, groups[key])
    if not pipeline.HAS_MATPLOTLIB:
        pytest.skip("matplotlib not installed in this environment")
    assert out.exists(), "plot_l_curve must produce the SVG when matplotlib is available"
    assert out.stat().st_size > 0, "plot_l_curve SVG must not be empty"


def test_generate_outputs_emits_l_curve_artifacts(pipeline, tmp_path):
    """End-to-end: generate_outputs writes l_curve_miss_rate_by_size.csv and SVGs."""
    rows = [
        _make_row("cache_sim", "email-Eu-core", "pr", policy, l3, miss)
        for policy, points in [
            ("LRU", [("4kB", 0.40), ("32kB", 0.30), ("256kB", 0.20)]),
            ("GRASP", [("4kB", 0.38), ("32kB", 0.29), ("256kB", 0.19)]),
        ]
        for l3, miss in points
    ]
    pipeline.generate_outputs(tmp_path, rows, [], copy_to_paper=False)
    csv_path = tmp_path / "aggregate" / "l_curve_miss_rate_by_size.csv"
    assert csv_path.exists(), "l_curve_miss_rate_by_size.csv must be written for cache_sim L-curve data"
    with open(csv_path) as f:
        out_rows = list(csv.DictReader(f))
    assert {r["policy_label"] for r in out_rows} >= {"LRU", "GRASP"}
    assert {r["l3_size"] for r in out_rows} >= {"4kB", "32kB", "256kB"}
    if pipeline.HAS_MATPLOTLIB:
        svg_path = tmp_path / "figures" / "l_curve_email-Eu-core_pr.svg"
        assert svg_path.exists()
