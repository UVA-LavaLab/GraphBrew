"""Pytest gate: cross-tool winner-agreement report sanity.

This pins the *plumbing* and *vocabulary* of
``wiki/data/cross_tool_winners.json`` rather than enforcing
unanimous agreement, because the three tools sweep different L3
ranges (cache_sim ≥ 1 MB, gem5 / Sniper ≤ 2 MB), so the largest-L3
operating points sit in different saturation regimes and disagreement
is expected. The proper saturation-controlled agreement test lives
in ``test_cross_tool_saturation.py`` (which restricts to cells where
both tools are above their per-tool saturation floor).

What this gate enforces:

* the report exists, has a non-empty ``cells`` list, and a
  ``by_classification`` summary keyed only by the closed vocabulary
  ``{unanimous, majority, split}``;
* every cell records the per-tool L3 it picked (so reviewers can see
  the regime mismatch);
* at least one (graph, app) overlap exists for both
  cache_sim↔gem5 and cache_sim↔Sniper, proving the intersection
  layer is wired end-to-end;
* ``cache_sim_winner`` is populated for every cell (cache_sim is the
  anchor tool — if it is ever empty, the lit-faith CSV failed to
  load, which would silently mask later analyses).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "wiki" / "data" / "cross_tool_winners.json"

CLOSED_VOCAB = {"unanimous", "majority", "split"}


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT} not generated; run `make lit-cross-tool-winners`")
    return json.loads(REPORT.read_text())


def test_report_has_required_top_level_keys(report):
    assert {"summary", "cells"}.issubset(report.keys())


def test_summary_n_cells_matches_cells_length(report):
    assert report["summary"]["n_cells"] == len(report["cells"])


def test_cells_nonempty(report):
    assert report["cells"], "cross-tool overlap is empty — cache_sim, gem5, or Sniper anchor JSON is missing"


def test_classification_vocab_is_closed(report):
    by_cls = report["summary"]["by_classification"]
    extra = set(by_cls) - CLOSED_VOCAB
    assert not extra, f"unexpected classification values: {sorted(extra)}"


def test_every_cell_has_required_fields(report):
    required = {
        "graph", "app",
        "cache_sim_l3", "gem5_l3", "sniper_l3",
        "cache_sim_winner", "gem5_winner", "sniper_winner",
        "n_tools", "classification",
    }
    for cell in report["cells"]:
        missing = required - set(cell)
        assert not missing, f"cell {cell.get('graph')}/{cell.get('app')} missing fields {missing}"


def test_cache_sim_winner_always_populated(report):
    """cache_sim is the anchor — every overlap cell must include it.
    If this fails the lit-faith CSV path is wrong."""
    for cell in report["cells"]:
        assert cell["cache_sim_winner"], (
            f"cache_sim_winner empty for {cell['graph']}/{cell['app']} — "
            "lit-faith CSV may be missing or misnamed"
        )


def test_overlap_with_each_tool_exists(report):
    """At least one cell must overlap with gem5 AND at least one must
    overlap with Sniper. Otherwise downstream cross-tool analyses are
    blind to one of the simulators."""
    has_gem5 = any(c.get("gem5_winner") for c in report["cells"])
    has_sniper = any(c.get("sniper_winner") for c in report["cells"])
    assert has_gem5, "no overlap with gem5 anchor — gem5_anchor.json may be stale"
    assert has_sniper, "no overlap with sniper anchor — sniper_anchor.json may be stale"


def test_n_tools_is_two_or_three(report):
    for cell in report["cells"]:
        n = cell["n_tools"]
        assert n in (2, 3), f"n_tools={n} for {cell['graph']}/{cell['app']} (expected 2 or 3)"
