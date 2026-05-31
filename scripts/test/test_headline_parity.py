#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gate 283 — headline parity proof (paper-table preview).

PROOF GATE — backed by real simulator measurements joined across the
3 platforms (cache_sim, gem5, Sniper). The assertion is the cross-sim
WINNER agreement on cells covered by ≥2 simulators.

Today the overlap is sparse (cache_sim covers 1MB cells but gem5/Sniper
anchors are at different L3 sizes), so this test is structurally
defensive: when no overlap exists the gate vacuously passes; when
overlap exists, the winner-agreement ratio must meet the configured
threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.ecg import headline_parity as mod  # noqa: E402

CACHE_SIM_CSV = REPO_ROOT / "wiki/data/literature_faithfulness_postfix.csv"
GEM5_JSON     = REPO_ROOT / "wiki/data/gem5_anchor.json"
SNIPER_JSON   = REPO_ROOT / "wiki/data/sniper_anchor.json"

# Today's threshold: ZERO disagreements tolerated. We start at 0 because
# if any disagreement DID exist it would be a strong signal that one of
# the simulator setups has a faithfulness bug (different policy winners
# on the same workload). Bump the threshold ONLY with a documented
# rationale + the disagreeing cell list.
MAX_DISAGREEMENT_RATIO = 0.0


# ---------------------------------------------------------------------------
# Scope sanity
# ---------------------------------------------------------------------------

def test_sims_roster_matches_gate_282():
    assert set(mod.SIMS) == {"cache_sim", "gem5", "sniper"}


def test_policy_order_includes_ecg():
    assert "ECG_DBG_PRIMARY" in mod.POLICY_ORDER
    for p in ("LRU", "SRRIP", "GRASP", "POPT"):
        assert p in mod.POLICY_ORDER


def test_literature_concrete_cells_nonempty():
    cells = mod.literature_concrete_cells()
    assert len(cells) >= 10


# ---------------------------------------------------------------------------
# CellMeasurement winner derivation
# ---------------------------------------------------------------------------

def test_winner_lowest_miss_rate():
    cm = mod.CellMeasurement("cache_sim", "g", "pr", "1MB",
                              {"LRU": 0.20, "GRASP": 0.10, "POPT": 0.15})
    assert cm.winner == "GRASP"


def test_winner_tiebreak_by_policy_order():
    """Ties break by the canonical policy order so the winner is
    deterministic across re-runs."""
    cm = mod.CellMeasurement("cache_sim", "g", "pr", "1MB",
                              {"LRU": 0.10, "GRASP": 0.10})
    # LRU comes before GRASP in POLICY_ORDER → LRU wins the tie
    assert cm.winner == "LRU"


def test_winner_none_on_empty():
    cm = mod.CellMeasurement("cache_sim", "g", "pr", "1MB", {})
    assert cm.winner is None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def test_load_cache_sim_real_file_returns_measurements():
    if not CACHE_SIM_CSV.exists():
        pytest.skip("lit-faith CSV not on disk")
    ms = mod.load_cache_sim_measurements(CACHE_SIM_CSV)
    assert len(ms) >= 10
    # Every measurement should have a sim label and non-empty miss-by-policy
    for m in ms[:5]:
        assert m.sim == "cache_sim"
        assert len(m.miss_by_policy) >= 1


def test_load_anchor_json_real_file_returns_measurements():
    if not GEM5_JSON.exists():
        pytest.skip("gem5_anchor.json not on disk")
    ms = mod.load_anchor_measurements(GEM5_JSON, "gem5")
    assert len(ms) >= 1
    for m in ms[:3]:
        assert m.sim == "gem5"


def test_load_handles_missing_file(tmp_path):
    assert mod.load_cache_sim_measurements(tmp_path / "nope.csv") == []
    assert mod.load_anchor_measurements(tmp_path / "nope.json", "gem5") == []


# ---------------------------------------------------------------------------
# Headline table construction
# ---------------------------------------------------------------------------

def test_headline_table_covers_literature_1mb_cells():
    """Each literature 1MB cell must appear in the headline table even
    if no sim reports for it (verdict=empty)."""
    if not CACHE_SIM_CSV.exists():
        pytest.skip("lit-faith CSV not on disk")
    measurements = mod.load_all_measurements(CACHE_SIM_CSV, GEM5_JSON,
                                              SNIPER_JSON)
    rows = mod.compute_headline_table(measurements, scope_l3=("1MB",))
    lit_1mb = mod.literature_concrete_cells(l3_filter=("1MB",))
    rows_keys = {(r.graph, r.app, r.l3_size) for r in rows}
    assert rows_keys == lit_1mb


def test_verdict_semantics():
    """Synthetic — verify all 4 verdict states (agree/disagree/single/empty)."""
    # graph-1, pr, 1MB: cache_sim says GRASP wins, gem5 says GRASP wins → agree
    # graph-2, pr, 1MB: cache_sim GRASP, gem5 LRU → disagree
    # graph-3, pr, 1MB: cache_sim only → single
    # graph-4, pr, 1MB: no measurements → empty
    measurements = [
        mod.CellMeasurement("cache_sim", "graph-1", "pr", "1MB",
                            {"LRU": 0.2, "GRASP": 0.1}),
        mod.CellMeasurement("gem5", "graph-1", "pr", "1MB",
                            {"LRU": 0.18, "GRASP": 0.09}),
        mod.CellMeasurement("cache_sim", "graph-2", "pr", "1MB",
                            {"LRU": 0.2, "GRASP": 0.1}),
        mod.CellMeasurement("gem5", "graph-2", "pr", "1MB",
                            {"LRU": 0.08, "GRASP": 0.18}),
        mod.CellMeasurement("cache_sim", "graph-3", "pr", "1MB",
                            {"LRU": 0.2, "GRASP": 0.1}),
    ]
    # Monkeypatch the literature cell set for this test
    orig = mod.literature_concrete_cells
    mod.literature_concrete_cells = lambda l3_filter=None: {
        ("graph-1", "pr", "1MB"), ("graph-2", "pr", "1MB"),
        ("graph-3", "pr", "1MB"), ("graph-4", "pr", "1MB"),
    }
    try:
        rows = mod.compute_headline_table(measurements, scope_l3=("1MB",))
    finally:
        mod.literature_concrete_cells = orig
    verdicts = {(r.graph, r.app): r.verdict for r in rows}
    assert verdicts[("graph-1", "pr")] == "agree"
    assert verdicts[("graph-2", "pr")] == "disagree"
    assert verdicts[("graph-3", "pr")] == "single"
    assert verdicts[("graph-4", "pr")] == "empty"


# ---------------------------------------------------------------------------
# Summary + load-bearing assertion
# ---------------------------------------------------------------------------

def _live_summary():
    if not CACHE_SIM_CSV.exists():
        pytest.skip("lit-faith CSV not on disk")
    measurements = mod.load_all_measurements(CACHE_SIM_CSV, GEM5_JSON,
                                              SNIPER_JSON)
    rows = mod.compute_headline_table(measurements, scope_l3=("1MB",))
    return mod.summarise(rows), rows


def test_summarise_returns_required_keys():
    summary, _ = _live_summary()
    for k in ("cells_total", "cells_with_overlap", "cells_agree",
              "cells_disagree", "cells_single_sim", "cells_empty",
              "winner_agreement_pct"):
        assert k in summary


def test_disagreement_ratio_below_threshold():
    """The PROOF assertion: across cells where >=2 sims report, the
    fraction with disagreeing winners must not exceed
    MAX_DISAGREEMENT_RATIO. Today the overlap is 0 so the assertion is
    vacuously true; as gem5/Sniper anchors expand to literature L3
    sizes, this gate starts measuring actual cross-sim agreement."""
    summary, rows = _live_summary()
    overlap = summary["cells_with_overlap"]
    disagree = summary["cells_disagree"]
    if overlap == 0:
        # Vacuously true; gate is structurally armed but not yet
        # measuring real disagreement. Document by listing disagreeing
        # cells (none today).
        assert disagree == 0
        return
    ratio = disagree / overlap
    if ratio > MAX_DISAGREEMENT_RATIO:
        disagreeing = [(r.graph, r.app, r.l3_size, r.per_sim_winner)
                       for r in rows if r.verdict == "disagree"]
        pytest.fail(
            f"cross-sim winner disagreement ratio {ratio:.2%} exceeds "
            f"threshold {MAX_DISAGREEMENT_RATIO:.2%}. "
            f"Disagreeing cells: {disagreeing}"
        )


# ---------------------------------------------------------------------------
# Artifact-shape sanity
# ---------------------------------------------------------------------------

def test_main_writes_artifacts(tmp_path):
    rc = mod.main([
        "--json-out", str(tmp_path / "h.json"),
        "--md-out",   str(tmp_path / "h.md"),
        "--csv-out",  str(tmp_path / "h.csv"),
        "--quiet",
    ])
    assert rc == 0
    payload = json.loads((tmp_path / "h.json").read_text())
    assert payload["gate"] == 283
    assert payload["status"] == "active"
    assert "summary" in payload
    assert "rows" in payload
    md = (tmp_path / "h.md").read_text()
    assert "gate 283" in md
    csv_lines = (tmp_path / "h.csv").read_text().splitlines()
    assert csv_lines[0].startswith("graph,app,l3_size,sim,")


def test_live_artifact_present():
    p = REPO_ROOT / "wiki/data/headline_parity.json"
    if not p.exists():
        pytest.skip("headline_parity.json not on disk; "
                    "run `make headline-parity` first")
    data = json.loads(p.read_text())
    assert data["gate"] == 283
    assert "rows" in data
    assert isinstance(data["summary"], dict)
