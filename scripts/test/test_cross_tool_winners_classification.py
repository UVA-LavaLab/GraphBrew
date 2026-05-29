"""
Confidence gate 114 — cross_tool_winners.json arithmetic + classification.

Locks the cross-tool winners table that pivots per-cell (app, graph)
winners across three simulators (cache_sim, gem5, sniper) and
classifies each cell as 'split' (≥2 non-empty winners disagree) or
'majority' (≥2 non-empty winners agree).

The current snapshot has 6 cells, all 'split', which is a strong negative
result for the paper's claim that the winner picked by cache_sim does not
reliably transfer to gem5/sniper. The gate enforces:

- the per-cell n_tools field matches the count of non-empty tool winners,
- absence of a winner implies absence of its l3 and margin_pp fields,
- classification is consistent with the actual winner agreement pattern,
- the summary projection (n_cells, by_classification, split_cells,
  majority_cells) is a faithful rollup of cells.

A future regression that flips even one cell from 'split' to 'majority'
(or vice versa) will fail this gate immediately.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "wiki" / "data" / "cross_tool_winners.json"

TOOLS = ("cache_sim", "gem5", "sniper")
KNOWN_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT", "POPT_CHARGED",
                  "ECG_DBG_ONLY", "ECG_DBG_PRIMARY", "ECG_DBG_PRIMARY_CHARGED",
                  "ECG_POPT_PRIMARY"}
KNOWN_APPS = {"bc", "bfs", "cc", "pr", "sssp", "radii"}
KNOWN_CLASSIFICATIONS = {"split", "majority"}
PROJECTED_FIELDS = ("app", "graph", "cache_sim_winner", "gem5_winner", "sniper_winner")


@pytest.fixture(scope="module")
def doc() -> dict:
    assert ARTIFACT_PATH.exists(), f"missing artifact: {ARTIFACT_PATH}"
    return json.loads(ARTIFACT_PATH.read_text())


def _winners(cell: dict) -> list[str]:
    return [cell[f"{t}_winner"] for t in TOOLS if cell[f"{t}_winner"]]


def _classify(cell: dict) -> str:
    w = _winners(cell)
    if len(w) < 2:
        return "single"            # cannot classify cross-tool agreement
    return "majority" if len(set(w)) == 1 else "split"


# ---------------------------------------------------------------------------
# Group A — top-level structure
# ---------------------------------------------------------------------------

def test_top_level_keys(doc):
    assert set(doc) == {"cells", "summary"}
    assert isinstance(doc["cells"], list)
    assert isinstance(doc["summary"], dict)


def test_summary_required_fields(doc):
    s = doc["summary"]
    assert set(s) == {"n_cells", "by_classification", "split_cells", "majority_cells"}
    assert isinstance(s["n_cells"], int)
    assert isinstance(s["by_classification"], dict)
    assert isinstance(s["split_cells"], list)
    assert isinstance(s["majority_cells"], list)


def test_summary_n_cells_matches_cells_length(doc):
    assert doc["summary"]["n_cells"] == len(doc["cells"])
    assert doc["summary"]["n_cells"] > 0, "must have at least one cell"


def test_cell_required_fields(doc):
    expected = {"app", "graph", "classification", "n_tools"}
    for t in TOOLS:
        expected |= {f"{t}_l3", f"{t}_winner", f"{t}_margin_pp"}
    for c in doc["cells"]:
        missing = expected - set(c)
        assert not missing, f"cell {c.get('app')}/{c.get('graph')} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Group B — per-cell invariants
# ---------------------------------------------------------------------------

def test_cell_app_and_classification_known(doc):
    for c in doc["cells"]:
        assert c["app"] in KNOWN_APPS, f"unknown app: {c['app']}"
        assert c["classification"] in KNOWN_CLASSIFICATIONS, (
            f"{c['app']}/{c['graph']}: unknown classification {c['classification']!r}"
        )


def test_cell_winners_are_known_policies_or_empty(doc):
    for c in doc["cells"]:
        for t in TOOLS:
            w = c[f"{t}_winner"]
            assert w == "" or w in KNOWN_POLICIES, (
                f"{c['app']}/{c['graph']}: {t}_winner={w!r} not in known set"
            )


def test_cell_empty_winner_implies_empty_l3_and_margin(doc):
    for c in doc["cells"]:
        for t in TOOLS:
            if c[f"{t}_winner"] == "":
                assert c[f"{t}_l3"] == "", (
                    f"{c['app']}/{c['graph']}: {t}_winner empty but {t}_l3={c[f'{t}_l3']!r}"
                )
                assert c[f"{t}_margin_pp"] == "", (
                    f"{c['app']}/{c['graph']}: {t}_winner empty but {t}_margin_pp={c[f'{t}_margin_pp']!r}"
                )


def test_cell_present_winner_has_l3_and_parseable_margin(doc):
    for c in doc["cells"]:
        for t in TOOLS:
            if c[f"{t}_winner"]:
                assert c[f"{t}_l3"], (
                    f"{c['app']}/{c['graph']}: {t}_winner={c[f'{t}_winner']!r} but {t}_l3 empty"
                )
                margin_str = c[f"{t}_margin_pp"]
                assert margin_str, (
                    f"{c['app']}/{c['graph']}: {t}_winner present but {t}_margin_pp empty"
                )
                margin = float(margin_str)
                assert margin >= 0.0, f"{c['app']}/{c['graph']}: {t}_margin_pp={margin} negative"


def test_cell_n_tools_matches_nonempty_winner_count(doc):
    for c in doc["cells"]:
        recomp = sum(1 for t in TOOLS if c[f"{t}_winner"])
        assert recomp == c["n_tools"], (
            f"{c['app']}/{c['graph']}: n_tools={c['n_tools']} but {recomp} non-empty winners"
        )
        assert recomp >= 2, (
            f"{c['app']}/{c['graph']}: cross-tool comparison needs ≥2 tools, got {recomp}"
        )


def test_cell_classification_matches_winner_agreement(doc):
    for c in doc["cells"]:
        derived = _classify(c)
        assert derived == c["classification"], (
            f"{c['app']}/{c['graph']}: classification={c['classification']!r} "
            f"but winners={_winners(c)} → derived={derived!r}"
        )


# ---------------------------------------------------------------------------
# Group C — summary rollup parity
# ---------------------------------------------------------------------------

def test_summary_by_classification_matches_cells(doc):
    recomp = dict(collections.Counter(c["classification"] for c in doc["cells"]))
    assert recomp == doc["summary"]["by_classification"]


def test_summary_split_cells_match_projected_cells(doc):
    expected = [
        {k: c[k] for k in PROJECTED_FIELDS}
        for c in doc["cells"]
        if c["classification"] == "split"
    ]
    assert expected == doc["summary"]["split_cells"]


def test_summary_majority_cells_match_projected_cells(doc):
    expected = [
        {k: c[k] for k in PROJECTED_FIELDS}
        for c in doc["cells"]
        if c["classification"] == "majority"
    ]
    assert expected == doc["summary"]["majority_cells"]
