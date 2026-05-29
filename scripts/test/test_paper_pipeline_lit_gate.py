"""Tests for the literature pre-flight gate in paper_pipeline.

The gate refuses to launch any final_* profile if the on-disk
literature_faithfulness snapshot reports unexplained disagreements.
These tests poke the gate function directly with synthetic JSON
artifacts so we can prove every failure mode is handled and the gate
is bypassable via --skip-literature-gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("paper_pipeline_test_module", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pipeline():
    return _load_pipeline()


def _write_lit_json(tmp_path: Path, *, ok=0, wt=0, kd=0, disagree=0,
                    disagree_cells=None) -> Path:
    """Build a synthetic lit-faith JSON matching the real schema."""
    payload = {
        "summary": {
            "claims_total": ok + wt + kd + disagree,
            "ok": ok,
            "within_tolerance": wt,
            "known_deviation": kd,
            "disagree": disagree,
            "missing": 0,
            "insufficient_data": 0,
        },
        "per_claim": [],
    }
    for e in (disagree_cells or []):
        payload["per_claim"].append({
            "graph": e[0], "app": e[1], "l3_size": e[2], "policy": e[3],
            "status": "disagree", "delta_pct": -99.0,
        })
    out = tmp_path / "lit.json"
    out.write_text(json.dumps(payload))
    return out


def test_preflight_passes_on_zero_disagree(pipeline, tmp_path, monkeypatch):
    lit = _write_lit_json(tmp_path, ok=10, wt=1, kd=2)
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    (tmp_path / "wiki" / "data").mkdir(parents=True)
    (tmp_path / "wiki" / "data" / "literature_faithfulness_postfix.json").write_text(
        lit.read_text())
    assert pipeline._literature_preflight() == 0


def test_preflight_fails_on_any_disagree(pipeline, tmp_path, monkeypatch):
    lit = _write_lit_json(
        tmp_path, ok=10, kd=2, disagree=1,
        disagree_cells=[("com-orkut", "bc", "8MB", "POPT_GE_GRASP")],
    )
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    (tmp_path / "wiki" / "data").mkdir(parents=True)
    (tmp_path / "wiki" / "data" / "literature_faithfulness_postfix.json").write_text(
        lit.read_text())
    rc = pipeline._literature_preflight()
    assert rc == 1, f"expected exit 1 on disagree, got {rc}"


def test_preflight_fails_on_missing_lit_json(pipeline, tmp_path, monkeypatch):
    """Missing snapshot is a hard fail — never silently pass."""
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    assert pipeline._literature_preflight() == 2


def test_preflight_fails_on_malformed_lit_json(pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    (tmp_path / "wiki" / "data").mkdir(parents=True)
    (tmp_path / "wiki" / "data" / "literature_faithfulness_postfix.json").write_text(
        "not json at all")
    assert pipeline._literature_preflight() == 2


def test_real_snapshot_currently_passes(pipeline):
    """The committed lit-faith snapshot must currently pass the gate."""
    lit = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
    if not lit.exists():
        pytest.skip("lit-faith snapshot not built yet")
    assert pipeline._literature_preflight() == 0


def test_skip_literature_gate_flag_is_declared(pipeline):
    """--skip-literature-gate must be a real CLI flag, not silently ignored."""
    parsed = pipeline.parse_args(["--skip-literature-gate"])
    assert parsed.skip_literature_gate is True
    # Default must be False — gate is opt-out, not opt-in.
    parsed_default = pipeline.parse_args([])
    assert parsed_default.skip_literature_gate is False
