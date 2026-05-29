"""Tests for the shared literature-faithfulness snapshot pre-flight.

The same helper is invoked from ``paper_pipeline.py`` and
``final_paper_run.py``. These tests pin the helper's exit-code contract
plus the integration behaviour on both call sites (opt-out semantics,
final_ profile detection, inspection-only short-circuit).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


ECG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "ecg"
HELPER_PATH = ECG_DIR / "literature_preflight.py"


def _load_helper():
    if str(ECG_DIR) not in sys.path:
        sys.path.insert(0, str(ECG_DIR))
    spec = importlib.util.spec_from_file_location(
        "literature_preflight_under_test", HELPER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_snapshot(path: Path, *, ok: int, within: int, kd: int,
                    disagree: int, per_claim=None) -> None:
    payload = {
        "summary": {
            "ok": ok,
            "within_tolerance": within,
            "known_deviation": kd,
            "disagree": disagree,
            "claims_total": ok + within + kd + disagree,
        },
        "per_claim": per_claim or [],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_helper_default_snapshot_points_at_wiki_data():
    mod = _load_helper()
    assert mod.DEFAULT_SNAPSHOT.name == "literature_faithfulness_postfix.json"
    assert mod.DEFAULT_SNAPSHOT.parent.name == "data"
    assert mod.DEFAULT_SNAPSHOT.parent.parent.name == "wiki"


def test_helper_pass_when_no_disagree(tmp_path):
    mod = _load_helper()
    snap = tmp_path / "lit.json"
    _write_snapshot(snap, ok=219, within=2, kd=30, disagree=0)
    out, err = io.StringIO(), io.StringIO()
    rc = mod.snapshot_preflight(snap, out=out, err=err)
    assert rc == 0
    assert "PASS" in out.getvalue()
    assert "219 ok" in out.getvalue()
    assert err.getvalue() == ""


def test_helper_fails_on_disagreement(tmp_path):
    mod = _load_helper()
    snap = tmp_path / "lit.json"
    _write_snapshot(
        snap, ok=10, within=0, kd=0, disagree=2,
        per_claim=[
            {"graph": "g1", "app": "pr", "l3_size": "1MB",
             "policy": "GRASP", "status": "disagree"},
            {"graph": "g2", "app": "bc", "l3_size": "4MB",
             "policy": "SRRIP", "status": "disagree"},
            {"graph": "g3", "app": "pr", "l3_size": "8MB",
             "policy": "LRU", "status": "ok"},
        ],
    )
    out, err = io.StringIO(), io.StringIO()
    rc = mod.snapshot_preflight(snap, out=out, err=err)
    assert rc == 1
    msg = err.getvalue()
    assert "FAIL" in msg
    assert "2 unexplained literature disagreement" in msg
    # only the two disagree rows should appear in examples; ok row excluded
    assert "g1/pr/1MB/GRASP" in msg
    assert "g2/bc/4MB/SRRIP" in msg
    assert "g3/pr/8MB/LRU" not in msg


def test_helper_fails_when_snapshot_missing(tmp_path):
    mod = _load_helper()
    snap = tmp_path / "does-not-exist.json"
    out, err = io.StringIO(), io.StringIO()
    rc = mod.snapshot_preflight(snap, out=out, err=err)
    assert rc == 2
    assert "not found" in err.getvalue()
    assert "make lit-faith" in err.getvalue()


def test_helper_fails_when_snapshot_unparseable(tmp_path):
    mod = _load_helper()
    snap = tmp_path / "garbage.json"
    snap.write_text("{ this is not valid JSON")
    out, err = io.StringIO(), io.StringIO()
    rc = mod.snapshot_preflight(snap, out=out, err=err)
    assert rc == 2
    assert "could not parse" in err.getvalue()


def test_helper_treats_missing_summary_keys_as_zero(tmp_path):
    mod = _load_helper()
    snap = tmp_path / "lit.json"
    snap.write_text(json.dumps({"summary": {}, "per_claim": []}))
    out, err = io.StringIO(), io.StringIO()
    rc = mod.snapshot_preflight(snap, out=out, err=err)
    assert rc == 0
    assert "0 ok" in out.getvalue()


def test_paper_pipeline_imports_shared_helper():
    """Regression guard: paper_pipeline must use the extracted helper,
    not a re-implemented copy that could drift."""
    src = (ECG_DIR / "paper_pipeline.py").read_text()
    assert "from literature_preflight import snapshot_preflight" in src
    # Make sure the old inline implementation is gone — the wrapper may
    # still build the snapshot path and hand it to the helper, but it
    # should not re-implement the parsing/disagree-counting logic.
    assert "data.get(\"summary\"" not in src, (
        "paper_pipeline.py should delegate snapshot parsing to "
        "literature_preflight.snapshot_preflight(); the old inline "
        "implementation appears to have been re-introduced."
    )


def test_final_paper_run_imports_shared_helper():
    """Regression guard: final_paper_run must use the extracted helper."""
    src = (ECG_DIR / "final_paper_run.py").read_text()
    assert "from literature_preflight import snapshot_preflight" in src


def test_final_paper_run_inspection_modes_skip_lit_gate():
    """--dry-run, --list, --check-graphs must not trigger the snapshot
    gate even on final_ profiles, since no real jobs are dispatched."""
    src = (ECG_DIR / "final_paper_run.py").read_text()
    # Ensure the inspection_only short-circuit is present
    assert "inspection_only" in src
    assert "args.dry_run" in src
    assert "args.list" in src
    assert "args.check_graphs" in src
