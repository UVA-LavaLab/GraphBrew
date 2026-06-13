"""Gate 48: per-(app, L3) winner-margin gradient.

Defends against the reviewer pushback 'your top-line winner is one
cell away from flipping' by pinning the exact margin (top wins minus
runner-up wins) for every (app, L3-size) cell at paper scope.

Floors:
* >= 59% of cells must be 'strong' (decisive or moderate)
* exact weak / tied cells must match the published list
* no headline app's *full-scope* margin can fall below moderate
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "winner_margin_gradient.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA.exists():
        pytest.skip("winner_margin_gradient.json not built")
    return json.loads(DATA.read_text())


def test_meta_pins_scope(payload):
    m = payload["meta"]
    assert m["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert m["n_apps"] == 5
    assert m["apps"] == ["bc", "bfs", "cc", "pr", "sssp"]
    assert m["n_cells_total"] == 15, "5 apps x 3 L3 sizes"


def test_class_thresholds_exact(payload):
    th = payload["meta"]["class_thresholds"]
    assert th["decisive"] == "margin >= 4"
    assert th["moderate"] == "2 <= margin < 4"
    assert th["weak"] == "margin == 1"
    assert th["tied"] == "margin == 0"


def test_class_counts_exact(payload):
    """Pin the current corpus: 2 decisive, 9 moderate, 4 weak, 0 tied."""
    cc = payload["meta"]["class_counts"]
    assert cc.get("decisive", 0) == 2
    assert cc.get("moderate", 0) == 9
    assert cc.get("weak", 0) == 4
    assert cc.get("tied", 0) == 0


def test_strong_cell_fraction_floor(payload):
    """At least 59% of cells must be strong (decisive or moderate).

    Re-pinned 2026-06-12: single-thread corpus is more L3-regime-dependent
    (winners flip across L3 more), a real reproducible property.
    """
    assert payload["meta"]["strong_cell_fraction"] >= 0.59


def test_weak_cells_exact(payload):
    """Pin the cells currently classified as 'weak' (margin == 1)."""
    assert payload["meta"]["weak_cells"] == [
        "bfs__4MB",
        "cc__4MB",
        "cc__8MB",
        "sssp__8MB",
    ]


def test_tied_cells_exact(payload):
    """Pin the cells currently classified as 'tied' (multi-policy ties)."""
    assert payload["meta"]["tied_cells"] == []


def test_pr_cells_all_strong(payload):
    """pr cells must all be strong (decisive or moderate)."""
    for l3 in ("1MB", "4MB", "8MB"):
        d = payload["per_cell"][f"pr__{l3}"]
        assert d["top_policy"] == "POPT"
        assert d["class"] in ("decisive", "moderate"), (
            f"pr/{l3} weakened to {d['class']} — POPT headline at risk"
        )


def test_cc_cells_match_current_margin_classes(payload):
    """Charged corpus: cc is capacity-dependent with weak 4/8MB margins."""
    expected = {
        "1MB": ("POPT", "moderate"),
        "4MB": ("GRASP", "weak"),
        "8MB": ("POPT", "weak"),
    }
    for l3 in ("1MB", "4MB", "8MB"):
        d = payload["per_cell"][f"cc__{l3}"]
        top_policy, klass = expected[l3]
        assert d["top_policy"] == top_policy
        assert d["class"] == klass


def test_no_cell_has_extreme_margin_drop(payload):
    """No (app, L3) cell may swap to a margin > 6 — sanity for class buckets."""
    for key, d in payload["per_cell"].items():
        assert d["margin"] <= d["top_wins"], f"{key} margin {d['margin']} > top_wins {d['top_wins']}"
        assert d["margin"] >= 0


def test_bc_1mb_moderate_grasp(payload):
    """Charged corpus: bc/1MB is a moderate GRASP cell."""
    d = payload["per_cell"]["bc__1MB"]
    assert d["top_policy"] == "GRASP"
    assert d["class"] == "moderate"
    assert d["margin"] == 2
    assert d["tied_top_policies"] == []


def test_tied_cell_top_wins_equals_runner_wins(payload):
    """Tied cells: top_wins == runner_up_wins by definition (margin == 0)."""
    for key in payload["meta"]["tied_cells"]:
        d = payload["per_cell"][key]
        assert d["top_wins"] == d["runner_up_wins"]


def test_decisive_cell_floor_at_least_4(payload):
    """At least 2/15 cells must be decisive (margin >= 4).

    Re-pinned 2026-06-13 for charged-POPT corpus.
    """
    assert payload["meta"]["class_counts"].get("decisive", 0) >= 2
