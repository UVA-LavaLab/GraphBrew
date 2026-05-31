#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for headline-1MB anchor companions (gem5 + Sniper).

These are SEPARATE artifacts from the canonical wiki/data/gem5_anchor.json
and wiki/data/sniper_anchor.json (which gate the stress-config L-shape
+ asymptote invariants and feed many cross-tool parity gates whose
shape must not silently change). The headline-1MB companions hold the
literature-canonical 1MB cells with the LRU/SRRIP/GRASP/POPT/
ECG_DBG_PRIMARY roster, and feed gate 282 (coverage) + gate 283
(parity).

The test is structurally defensive: the companions may not yet exist
on disk (the corresponding gem5/Sniper sweeps are background-launched
and may not have completed). When the file exists, validate its shape
and key invariants.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEM5_HEADLINE_1MB = REPO_ROOT / "wiki/data/gem5_anchor_headline_1mb.json"
SNIPER_HEADLINE_1MB = REPO_ROOT / "wiki/data/sniper_anchor_headline_1mb.json"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Existence — sanity (always pass; companions are workstation-dispatched).
# ---------------------------------------------------------------------------

def test_paths_are_in_wiki_data():
    """The companions must live in wiki/data/ so wiki-registry gates
    can audit them as regular published artifacts."""
    assert GEM5_HEADLINE_1MB.parent.name == "data"
    assert GEM5_HEADLINE_1MB.parent.parent.name == "wiki"
    assert SNIPER_HEADLINE_1MB.parent.name == "data"
    assert SNIPER_HEADLINE_1MB.parent.parent.name == "wiki"


# ---------------------------------------------------------------------------
# Shape — when on disk, must match the canonical anchor schema.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,sim", [
    (GEM5_HEADLINE_1MB, "gem5"),
    (SNIPER_HEADLINE_1MB, "sniper"),
])
def test_companion_schema_when_present(path, sim):
    d = _load(path)
    if d is None:
        pytest.skip(f"{sim} headline-1MB companion not on disk yet")
    # Top-level keys from the shared gem5_anchor_summary schema.
    assert "cells" in d
    assert isinstance(d["cells"], list)
    assert "sweep_root" in d
    assert "sweep_subdir" in d
    assert "counts" in d
    assert d["counts"]["cells"] == len(d["cells"])


@pytest.mark.parametrize("path,sim", [
    (GEM5_HEADLINE_1MB, "gem5"),
    (SNIPER_HEADLINE_1MB, "sniper"),
])
def test_companion_cells_are_at_literature_l3_when_present(path, sim):
    """Cells in the headline-1MB companion must all be at the GRASP
    HPCA20 canonical L3=1MB — that's the entire reason for separating
    these from the stress-config anchor."""
    d = _load(path)
    if d is None:
        pytest.skip(f"{sim} headline-1MB companion not on disk yet")
    for c in d["cells"]:
        assert c["l3_size"] == "1MB", (
            f"{sim} companion contains non-1MB cell: "
            f"{c['graph']}/{c['app']}/{c['l3_size']} — move stress-config "
            f"cells back to canonical anchor"
        )


@pytest.mark.parametrize("path,sim", [
    (GEM5_HEADLINE_1MB, "gem5"),
    (SNIPER_HEADLINE_1MB, "sniper"),
])
def test_companion_uses_literature_policy_roster_when_present(path, sim):
    """Every reporting cell must include at least one of the four
    literature baselines (LRU/SRRIP/GRASP/POPT). Cells that contain ONLY
    ECG variants would be a sweep-script bug — they cannot be used for
    cross-policy comparison."""
    d = _load(path)
    if d is None:
        pytest.skip(f"{sim} headline-1MB companion not on disk yet")
    baselines = {"LRU", "SRRIP", "GRASP", "POPT"}
    for c in d["cells"]:
        pols = set(c.get("miss_rate_by_policy", {}).keys())
        if pols:  # only require if cell has any policy data
            assert pols & baselines, (
                f"{sim} cell {c['graph']}/{c['app']}/{c['l3_size']} has "
                f"no literature baseline policy ({pols})"
            )


@pytest.mark.parametrize("path,sim", [
    (GEM5_HEADLINE_1MB, "gem5"),
    (SNIPER_HEADLINE_1MB, "sniper"),
])
def test_companion_no_error_rows_when_present(path, sim):
    """If sweep produced cells, they should be `ok` — error rows mean
    the sim run crashed and the cell shouldn't be counted toward
    coverage. Surface them loudly so the user can re-dispatch."""
    d = _load(path)
    if d is None:
        pytest.skip(f"{sim} headline-1MB companion not on disk yet")
    for c in d["cells"]:
        assert c.get("error_rows", 0) == 0, (
            f"{sim} cell {c['graph']}/{c['app']}/{c['l3_size']} has "
            f"{c.get('error_rows')} error rows — re-dispatch this cell"
        )
