"""Gate 47: leave-one-family-out (LOFO) winner robustness.

Strictly stronger than gate 41 (LOGO). Drops a full family at a time
and verifies the top-line policy per app. Pins which apps are LOFO-robust
and which are family-sensitive (publication-honest).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "lofo_robustness.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA.exists():
        pytest.skip("lofo_robustness.json not built")
    return json.loads(DATA.read_text())


def test_meta_pins_scope(payload):
    m = payload["meta"]
    assert m["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert m["n_families"] == 5
    assert m["n_apps"] == 5
    assert m["families"] == ["citation", "mesh", "road", "social", "web"]
    assert m["apps"] == ["bc", "bfs", "cc", "pr", "sssp"]
    assert m["n_rows_in_scope"] == 360


def test_robust_apps_inventory_exact(payload):
    """At current corpus: bc, cc, pr are LOFO-robust."""
    assert payload["meta"]["robust_apps"] == ["bc", "cc", "pr"]


def test_fragile_apps_inventory_exact(payload):
    """At current corpus: bfs, sssp are LOFO-fragile (family-sensitive)."""
    assert payload["meta"]["fragile_apps"] == ["bfs", "sssp"]


def test_robustness_fraction_at_least_60pct(payload):
    """At least 60% of apps must survive LOFO (current 60%)."""
    assert payload["meta"]["robustness_fraction"] >= 0.60


def test_n_robust_apps_floor(payload):
    """At least 3/5 apps must be LOFO-robust."""
    assert payload["meta"]["n_robust_apps"] >= 3


def test_pr_lofo_robust_with_popt(payload):
    """The headline pr/POPT claim must survive every family drop."""
    p = payload["per_app"]["pr"]
    assert p["full_corpus"]["top_policy"] == "POPT"
    assert p["is_lofo_robust"] is True
    assert p["fragile_family_drops"] == []
    for fam, d in p["drops"].items():
        assert d.get("missing") is not True
        assert d["top_policy"] == "POPT", f"pr/drop-{fam} flipped to {d['top_policy']}"


def test_bc_lofo_robust_with_grasp(payload):
    """The headline bc/GRASP claim must survive every family drop."""
    p = payload["per_app"]["bc"]
    assert p["full_corpus"]["top_policy"] == "GRASP"
    assert p["is_lofo_robust"] is True


def test_cc_lofo_robust_with_grasp(payload):
    """The cc/GRASP claim must survive every family drop."""
    p = payload["per_app"]["cc"]
    assert p["full_corpus"]["top_policy"] == "GRASP"
    assert p["is_lofo_robust"] is True


def test_bfs_fragile_drop_is_citation(payload):
    """bfs is honestly disclosed as family-sensitive: drop citation → GRASP ties.

    Post cache_sim ECG sweep: bfs's full-corpus winner is now GRASP (was POPT).
    Dropping citation reduces GRASP to a tied top (no unique winner), so
    the lofo gate marks bfs as fragile.
    """
    p = payload["per_app"]["bfs"]
    assert p["is_lofo_robust"] is False
    assert "citation" in p["fragile_family_drops"]
    assert p["full_corpus"]["top_policy"] == "GRASP"
    # without citation, GRASP and POPT both end up at 7 wins → not unique
    assert p["drops"]["citation"]["top_policy"] == "GRASP"
    assert p["drops"]["citation"]["unique_top"] is False


def test_sssp_fragile_drop_is_citation(payload):
    """sssp is honestly disclosed as family-sensitive: drop citation → POPT."""
    p = payload["per_app"]["sssp"]
    assert p["is_lofo_robust"] is False
    assert "citation" in p["fragile_family_drops"]
    assert p["full_corpus"]["top_policy"] == "GRASP"
    assert p["drops"]["citation"]["top_policy"] == "POPT"


def test_per_app_drop_count_equals_family_count(payload):
    """Every app must have exactly n_families drop entries."""
    n_fams = payload["meta"]["n_families"]
    for app, p in payload["per_app"].items():
        assert p["n_drops"] == n_fams, (
            f"{app} expected {n_fams} drops, got {p['n_drops']}"
        )
        non_missing = [
            f for f, d in p["drops"].items() if not d.get("missing")
        ]
        # At minimum 4/5 family drops must yield non-missing top
        assert len(non_missing) >= n_fams - 1


def test_no_missing_drops_in_current_corpus(payload):
    """Current corpus has every (app, drop) combination — no MISSING cells."""
    for app, p in payload["per_app"].items():
        for fam, d in p["drops"].items():
            assert not d.get("missing"), (
                f"{app}/drop-{fam} returned MISSING — corpus shrank"
                " and reviewers can no longer audit this drop"
            )
