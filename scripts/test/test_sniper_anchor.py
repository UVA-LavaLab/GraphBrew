"""Test the Sniper literature-anchor invariants.

Parallel structure to test_gem5_anchor.py but scoped to the Sniper
sweep on email-Eu-core. Sniper currently only sweeps PR (BC variants
are empty), so the asymptote and headline invariants are only
asserted for PR.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "scripts/experiments/ecg/gem5_anchor_summary.py"
ANCHOR_JSON = REPO_ROOT / "wiki/data/sniper_anchor.json"


def _load_summary_module():
    spec = importlib.util.spec_from_file_location(
        "sniper_anchor_summary_test_mod", SUMMARY_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def anchor_snapshot() -> dict:
    if not ANCHOR_JSON.exists():
        pytest.fail(
            f"Sniper anchor snapshot missing: {ANCHOR_JSON}. "
            "Regenerate with `make sniper-anchor`."
        )
    return json.loads(ANCHOR_JSON.read_text())


def test_snapshot_well_formed(anchor_snapshot: dict) -> None:
    assert "invariants" in anchor_snapshot
    assert "cells" in anchor_snapshot
    assert anchor_snapshot["apps_scope"] == ["pr"], (
        "expected Sniper anchor scoped to PR only; expand --apps once BC "
        "sweeps are populated."
    )


def test_no_invariant_disagrees(anchor_snapshot: dict) -> None:
    disagreements = [
        i for i in anchor_snapshot["invariants"] if i["status"] == "disagree"
    ]
    assert not disagreements, (
        "Sniper anchor invariants in disagree state: "
        + ", ".join(f"{d['name']} ({d['detail']})" for d in disagreements)
    )


def test_no_invariant_missing(anchor_snapshot: dict) -> None:
    missing = [i for i in anchor_snapshot["invariants"] if i["status"] == "missing"]
    assert not missing, (
        "Sniper anchor invariants missing data: "
        + ", ".join(f"{m['name']} ({m['detail']})" for m in missing)
    )


def test_headline_pr_present(anchor_snapshot: dict) -> None:
    name = "GRASP_LE_LRU_headline:pr@256kB"
    matches = [i for i in anchor_snapshot["invariants"] if i["name"] == name]
    assert matches, f"expected invariant {name}"
    assert matches[0]["status"] == "ok", matches[0]


def test_asymptote_pr_present(anchor_snapshot: dict) -> None:
    name_prefix = "asymptote_within_"
    matches = [
        i for i in anchor_snapshot["invariants"]
        if i["name"].startswith(name_prefix) and i["name"].endswith(":pr@2MB")
    ]
    assert matches, "expected asymptote invariant for pr@2MB"
    assert matches[0]["status"] == "ok", matches[0]


def test_apps_filter_isolates_pr(tmp_path: Path) -> None:
    """When invoked with --apps pr, no bc invariants should appear."""
    mod = _load_summary_module()
    # synthetic sweep with both pr and bc data
    pr_rows = [
        {"benchmark": "pr", "policy": "LRU", "l3_size": "256kB",
         "l3_miss_rate": "0.118", "section": "1", "status": "ok"},
        {"benchmark": "pr", "policy": "GRASP", "l3_size": "256kB",
         "l3_miss_rate": "0.121", "section": "1", "status": "ok"},
        {"benchmark": "pr", "policy": "SRRIP", "l3_size": "256kB",
         "l3_miss_rate": "0.122", "section": "1", "status": "ok"},
        {"benchmark": "pr", "policy": "LRU", "l3_size": "2MB",
         "l3_miss_rate": "0.117", "section": "1", "status": "ok"},
        {"benchmark": "pr", "policy": "GRASP", "l3_size": "2MB",
         "l3_miss_rate": "0.119", "section": "1", "status": "ok"},
        {"benchmark": "pr", "policy": "SRRIP", "l3_size": "2MB",
         "l3_miss_rate": "0.114", "section": "1", "status": "ok"},
    ]
    bc_rows = [
        {"benchmark": "bc", "policy": "LRU", "l3_size": "256kB",
         "l3_miss_rate": "", "section": "1", "status": "error"},
    ]
    import csv as _csv

    def _write(app, rows):
        dest = tmp_path / f"email-Eu-core-{app}" / "DBG"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "roi_matrix.csv"
        with path.open("w") as f:
            fns = ["benchmark", "policy", "l3_size", "l3_miss_rate", "section", "status"]
            w = _csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    _write("pr", pr_rows)
    _write("bc", bc_rows)

    cells = mod.load_cells(tmp_path, "DBG", graphs={"email-Eu-core"})
    # with apps=("pr",) the bc invariants should not appear at all
    invariants_pr = mod.evaluate_invariants(cells, apps=("pr",))
    names = [i.name for i in invariants_pr]
    assert not any(":bc@" in n for n in names), names
    # the bc error_row should still be counted by no_error_rows
    no_err = [i for i in invariants_pr if i.name == "no_error_rows"][0]
    assert no_err.status == "disagree", no_err.detail
