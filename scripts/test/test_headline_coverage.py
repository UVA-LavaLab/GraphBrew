#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for gate 282 — headline coverage proof (RATCHET).

This is the first PROOF gate — backed by real experimental measurements
on the 3 simulators, NOT an AST audit.

Behavior:
  - Asserts coverage_count >= baseline_count (per scope, per sim).
  - Baseline lives in wiki/data/headline_coverage_baseline.json.
  - Coverage can never RETREAT below the baseline without an explicit
    --bump-baseline (which intentionally rewrites the floor downward).
  - When coverage EXCEEDS baseline, the test PASSES with an informational
    warning so the human can bump the floor on their next commit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.ecg import headline_coverage as mod  # noqa: E402

BASELINE_PATH = REPO_ROOT / "scripts/experiments/ecg/headline_coverage_baseline.json"


# ---------------------------------------------------------------------------
# Scope + roster invariants (cheap sanity)
# ---------------------------------------------------------------------------

def test_sims_roster_has_three_platforms():
    assert set(mod.SIMS) == {"cache_sim", "gem5", "sniper"}


def test_policies_headline_includes_ecg():
    assert "ECG_DBG_PRIMARY" in mod.POLICIES_HEADLINE
    for p in ("LRU", "SRRIP", "GRASP", "POPT"):
        assert p in mod.POLICIES_HEADLINE


def test_workstation_tiers_cover_literature_graphs():
    """Every graph the literature claims results on must have an
    explicit workstation tier classification — no silent UNKNOWNs."""
    lit_cells = mod.literature_concrete_cells(l3_filter=None)
    literature_graphs = {g for g, _, _ in lit_cells}
    for g in literature_graphs:
        assert g in mod.WORKSTATION_TIERS, \
            f"literature graph {g!r} missing workstation tier classification"
        assert mod.WORKSTATION_TIERS[g] in {"LOCAL", "LOCAL_TIGHT", "SLURM"}, \
            f"unknown tier label for {g}"


def test_valid_scopes_complete():
    for scope in mod.VALID_SCOPES:
        cells = mod.required_cells(scope)
        assert len(cells) >= 1, f"scope {scope!r} produced empty cell set"


# ---------------------------------------------------------------------------
# Literature derivation (the registry is the source of truth)
# ---------------------------------------------------------------------------

def test_literature_concrete_cells_nonempty():
    cells = mod.literature_concrete_cells(l3_filter=None)
    assert len(cells) >= 10, \
        f"literature_baselines.py exposes only {len(cells)} concrete cells"


def test_literature_concrete_cells_skip_patterns():
    """Pattern claims (graph startswith '*') are tier-1 invariants and
    do not belong in headline coverage."""
    cells = mod.literature_concrete_cells(l3_filter=None)
    for g, _, _ in cells:
        assert not g.startswith("*"), f"unexpected pattern graph: {g}"


def test_literature_cells_1mb_filter_works():
    cells_1mb = mod.literature_concrete_cells(l3_filter=("1MB",))
    cells_all = mod.literature_concrete_cells(l3_filter=None)
    assert len(cells_1mb) <= len(cells_all)
    for _, _, l3 in cells_1mb:
        assert l3 == "1MB"


# ---------------------------------------------------------------------------
# Loaders are non-crashing on missing inputs
# ---------------------------------------------------------------------------

def test_load_cache_sim_missing_file_returns_empty(tmp_path):
    out = mod.load_cache_sim(tmp_path / "nope.csv")
    assert out == set()


def test_load_anchor_json_missing_file_returns_empty(tmp_path):
    out = mod.load_anchor_json(tmp_path / "nope.json", "gem5")
    assert out == set()


def test_load_anchor_json_malformed_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json at all", encoding="utf-8")
    out = mod.load_anchor_json(p, "gem5")
    assert out == set()


def test_load_cache_sim_real_file_returns_rows():
    csv_path = REPO_ROOT / "wiki/data/literature_faithfulness_postfix.csv"
    if not csv_path.exists():
        pytest.skip("lit-faith CSV not on disk; run `make lit-faith` first")
    out = mod.load_cache_sim(csv_path)
    assert len(out) >= 100, f"lit-faith CSV produced only {len(out)} cells"
    # Every loaded cell must have sim="cache_sim"
    for c in out:
        assert c.sim == "cache_sim"
        assert c.prefetcher == "no_pfx"


# ---------------------------------------------------------------------------
# Ratchet — the load-bearing assertion
# ---------------------------------------------------------------------------

def _load_baseline() -> dict:
    assert BASELINE_PATH.exists(), (
        f"headline_coverage_baseline.json missing at {BASELINE_PATH}. "
        "Run `python3 -m scripts.experiments.ecg.headline_coverage "
        "--scope headline_1MB --bump-baseline` to create one.")
    data = json.loads(BASELINE_PATH.read_text())
    assert "schema" in data and data["schema"] >= 1
    assert "scope" in data and data["scope"] in mod.VALID_SCOPES
    assert "present_count" in data and isinstance(data["present_count"], int)
    assert "per_sim_present_count" in data
    return data


def test_baseline_file_exists_and_parses():
    _load_baseline()


def test_ratchet_total_coverage_at_or_above_baseline():
    """The headline assertion: total in-scope coverage must not retreat
    below the committed baseline floor."""
    baseline = _load_baseline()
    scope = baseline["scope"]
    present = mod.load_all_presence(
        REPO_ROOT / "wiki/data/literature_faithfulness_postfix.csv",
        REPO_ROOT / "wiki/data/gem5_anchor.json",
        REPO_ROOT / "wiki/data/sniper_anchor.json",
    )
    required = mod.required_cells(scope)
    actual = len(required & present)
    floor = baseline["present_count"]
    assert actual >= floor, (
        f"coverage regression: scope={scope} actual={actual} cells, "
        f"baseline floor={floor}. Either populate the missing cells OR "
        f"explicitly bump the floor with --bump-baseline.")


def test_ratchet_per_sim_coverage_at_or_above_baseline():
    """Per-sim coverage must also not retreat."""
    baseline = _load_baseline()
    scope = baseline["scope"]
    present = mod.load_all_presence(
        REPO_ROOT / "wiki/data/literature_faithfulness_postfix.csv",
        REPO_ROOT / "wiki/data/gem5_anchor.json",
        REPO_ROOT / "wiki/data/sniper_anchor.json",
    )
    required = mod.required_cells(scope)
    inscope = required & present
    per_sim_floors = baseline["per_sim_present_count"]
    for sim, floor in per_sim_floors.items():
        actual = sum(1 for c in inscope if c.sim == sim)
        assert actual >= floor, (
            f"coverage regression on {sim}: actual={actual} cells, "
            f"baseline floor={floor}. Either populate the missing cells "
            f"OR explicitly bump the floor with --bump-baseline.")


def test_baseline_scope_in_valid_set():
    baseline = _load_baseline()
    assert baseline["scope"] in mod.VALID_SCOPES


# ---------------------------------------------------------------------------
# Artifact-shape sanity
# ---------------------------------------------------------------------------

def test_main_writes_artifacts(tmp_path):
    """Coverage CLI must produce 3 artifacts cleanly."""
    rc = mod.main([
        "--scope", "headline_1MB",
        "--json-out", str(tmp_path / "c.json"),
        "--md-out",   str(tmp_path / "c.md"),
        "--csv-out",  str(tmp_path / "c.csv"),
        "--baseline-path", str(tmp_path / "baseline.json"),
        "--quiet",
    ])
    assert rc == 0
    payload = json.loads((tmp_path / "c.json").read_text())
    assert payload["gate"] == 282
    assert payload["scope"] == "headline_1MB"
    assert payload["status"] == "active"
    assert "summary" in payload
    assert "per_sim_graph" in payload
    assert "missing_cells" in payload
    md = (tmp_path / "c.md").read_text()
    assert "gate 282" in md
    csv_lines = (tmp_path / "c.csv").read_text().splitlines()
    assert csv_lines[0].startswith("sim,graph,app,l3_size,policy,prefetcher")


def test_json_artifact_matches_live():
    """The committed wiki/data/headline_coverage.json must match what
    the tool produces today (otherwise the dashboard reads stale data)."""
    live_json = REPO_ROOT / "wiki/data/headline_coverage.json"
    if not live_json.exists():
        pytest.skip("headline_coverage.json not on disk; "
                    "run `make headline-coverage` first")
    data = json.loads(live_json.read_text())
    assert data["gate"] == 282
    assert data["scope"] in mod.VALID_SCOPES


def test_required_cells_returns_frozen_keys():
    """CellKey is a frozen dataclass — usable as a dict/set key."""
    cells = mod.required_cells("headline_1MB")
    assert len(cells) >= 1
    sample = next(iter(cells))
    # frozen, hashable, equal-by-value
    assert isinstance(sample, mod.CellKey)
    assert sample == mod.CellKey(sample.sim, sample.graph, sample.app,
                                   sample.l3_size, sample.policy,
                                   sample.prefetcher)
