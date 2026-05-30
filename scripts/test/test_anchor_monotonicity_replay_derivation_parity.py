"""Derivation parity gate (AMR-Der) for `anchor_monotonicity_replay.json`.

The dashboard treats `wiki/data/anchor_monotonicity_replay.json` as the
single-screen verdict for L3-sweep monotonicity across the two anchor
simulators (gem5 + Sniper). The generator at
``scripts/experiments/ecg/anchor_monotonicity_replay.py`` walks every
(graph, app, policy) cell in each tool's slope-replay artifact, counts
per-step regressions ("bumps"), and applies tier-aware tolerances:

* gem5 is high-fidelity → zero bumps allowed (strict monotone),
* sniper is the noisier tier → bounded bump rate / hard-bump count /
  max-bump magnitude.

This gate locks the load-bearing constants and per-cell predicates so a
future regression in either simulator's plumbing surfaces as a failed
assertion instead of a silent JSON drift:

* Group A — schema + tier-aware tolerances + universal constants.
* Group B — per-cell walk shape (expected_sizes, steps_total,
  bump enumeration, sort/limit of worst_bumps, rounding pins).
* Group C — per-tool checks block (epsilon-relaxed inequalities,
  no_catastrophic gate, verdict_ok conjunction).
* Group D — overall block (verdict conjunction, median aggregator,
  catastrophic accumulator).
* Group E — byte parity with the committed JSON.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "anchor_monotonicity_replay.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "anchor_monotonicity_replay.json"
GEM5_PATH = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
SNIPER_PATH = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("anchor_monotonicity_replay", GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def artifact():
    return json.loads(JSON_PATH.read_text())


@pytest.fixture(scope="module")
def gem5_blob():
    return json.loads(GEM5_PATH.read_text())


@pytest.fixture(scope="module")
def sniper_blob():
    return json.loads(SNIPER_PATH.read_text())


# ---------------------------------------------------------------- Group A
def test_schema_label_pinned(artifact):
    assert artifact["schema"] == "anchor_monotonicity_replay/v1"


def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"schema", "per_tool", "overall", "constants"}


def test_gem5_tolerances_pinned(gen):
    assert gen.TOOL_TOLERANCES["gem5"] == {
        "bump_rate_max_pct": 0.0,
        "hard_bumps_max": 0,
        "max_bump_pp_max": 0.0,
    }


def test_sniper_tolerances_pinned(gen):
    assert gen.TOOL_TOLERANCES["sniper"] == {
        "bump_rate_max_pct": 40.0,
        "hard_bumps_max": 5,
        "max_bump_pp_max": 2.0,
    }


def test_universal_constants_pinned(gen, artifact):
    assert gen.HARD_BUMP_THRESHOLD_PP == 0.5
    assert gen.CATASTROPHIC_BUMP_PP == 3.0
    assert artifact["constants"] == {
        "hard_bump_threshold_pp": 0.5,
        "catastrophic_bump_pp": 3.0,
    }


def test_per_tool_keys_exactly_gem5_and_sniper(artifact):
    assert set(artifact["per_tool"].keys()) == {"gem5", "sniper"}


# ---------------------------------------------------------------- Group B
def test_per_tool_expected_sizes_from_upstream(artifact, gem5_blob, sniper_blob):
    assert artifact["per_tool"]["gem5"]["expected_sizes"] == gem5_blob["meta"]["expected_sizes"]
    assert artifact["per_tool"]["sniper"]["expected_sizes"] == sniper_blob["meta"]["expected_sizes"]


def test_per_tool_cells_count_matches_upstream(artifact, gem5_blob, sniper_blob):
    assert artifact["per_tool"]["gem5"]["cells"] == len(gem5_blob.get("per_cell", []))
    assert artifact["per_tool"]["sniper"]["cells"] == len(sniper_blob.get("per_cell", []))


def _expected_steps(blob):
    expected = blob["meta"]["expected_sizes"]
    total = 0
    for cell in blob.get("per_cell", []) or []:
        mb = cell.get("miss_pp_by_size", {}) or {}
        try:
            _ = [float(mb[s]) for s in expected]
        except (KeyError, TypeError, ValueError):
            continue
        total += len(expected) - 1
    return total


def test_per_tool_steps_total_equals_walked_steps(artifact, gem5_blob, sniper_blob):
    assert artifact["per_tool"]["gem5"]["steps_total"] == _expected_steps(gem5_blob)
    assert artifact["per_tool"]["sniper"]["steps_total"] == _expected_steps(sniper_blob)


def _expected_bumps_payload(blob):
    expected = blob["meta"]["expected_sizes"]
    bumps = []
    hard = 0
    max_pp = 0.0
    for cell in blob.get("per_cell", []) or []:
        mb = cell.get("miss_pp_by_size", {}) or {}
        try:
            seq = [float(mb[s]) for s in expected]
        except (KeyError, TypeError, ValueError):
            continue
        for i in range(len(seq) - 1):
            delta = seq[i + 1] - seq[i]
            if delta > 0:
                bumps.append({
                    "graph": cell.get("graph"),
                    "app": cell.get("app"),
                    "policy": cell.get("policy"),
                    "l3_from": expected[i],
                    "l3_to": expected[i + 1],
                    "delta_pp": round(delta, 6),
                })
                if delta >= 0.5:
                    hard += 1
                if delta > max_pp:
                    max_pp = delta
    return bumps, hard, max_pp


@pytest.mark.parametrize("tool,path", [("gem5", "gem5"), ("sniper", "sniper")])
def test_per_tool_bumps_count_matches_predicate(artifact, gem5_blob, sniper_blob, tool, path):
    blob = gem5_blob if tool == "gem5" else sniper_blob
    expected_bumps, _, _ = _expected_bumps_payload(blob)
    assert artifact["per_tool"][path]["bumps"] == len(expected_bumps)


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_per_tool_hard_bumps_threshold_inclusive_ge(artifact, gem5_blob, sniper_blob, tool):
    blob = gem5_blob if tool == "gem5" else sniper_blob
    _, expected_hard, _ = _expected_bumps_payload(blob)
    assert artifact["per_tool"][tool]["hard_bumps"] == expected_hard


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_per_tool_max_bump_pp_rounded_6dp(artifact, gem5_blob, sniper_blob, tool):
    blob = gem5_blob if tool == "gem5" else sniper_blob
    _, _, expected_max = _expected_bumps_payload(blob)
    assert artifact["per_tool"][tool]["max_bump_pp"] == round(expected_max, 6)


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_per_tool_bump_rate_pct_rounded_4dp(artifact, tool):
    p = artifact["per_tool"][tool]
    expected = (p["bumps"] / p["steps_total"] * 100.0) if p["steps_total"] else 0.0
    assert p["bump_rate_pct"] == round(expected, 4)


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_per_tool_worst_bumps_sorted_desc_and_capped_at_6(artifact, tool):
    p = artifact["per_tool"][tool]
    worst = p["worst_bumps"]
    assert len(worst) <= 6
    deltas = [b["delta_pp"] for b in worst]
    assert deltas == sorted(deltas, reverse=True)
    # And the head equals the sorted prefix of all_bumps.
    expected_head = sorted(p["all_bumps"], key=lambda b: -b["delta_pp"])[:6]
    assert [b["delta_pp"] for b in worst] == [b["delta_pp"] for b in expected_head]


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_bump_entries_carry_tool_field(artifact, tool):
    p = artifact["per_tool"][tool]
    for b in p["all_bumps"]:
        assert b["tool"] == tool
        assert set(b.keys()) >= {"tool", "graph", "app", "policy", "l3_from", "l3_to", "delta_pp"}


# ---------------------------------------------------------------- Group C
@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_check_keys_block_pinned(artifact, tool):
    checks = artifact["per_tool"][tool]["evaluation"]["checks"]
    assert set(checks.keys()) == {
        "bump_rate_ok",
        "hard_bumps_ok",
        "max_bump_pp_ok",
        "no_catastrophic",
    }


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_check_bump_rate_inclusive_with_epsilon(artifact, gen, tool):
    p = artifact["per_tool"][tool]
    ev = p["evaluation"]
    tol = ev["tolerances"]
    assert ev["checks"]["bump_rate_ok"] == (p["bump_rate_pct"] <= tol["bump_rate_max_pct"] + 1e-9)


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_check_hard_bumps_inclusive_le(artifact, tool):
    p = artifact["per_tool"][tool]
    ev = p["evaluation"]
    assert ev["checks"]["hard_bumps_ok"] == (p["hard_bumps"] <= ev["tolerances"]["hard_bumps_max"])


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_check_max_bump_inclusive_with_epsilon(artifact, tool):
    p = artifact["per_tool"][tool]
    ev = p["evaluation"]
    assert ev["checks"]["max_bump_pp_ok"] == (
        p["max_bump_pp"] <= ev["tolerances"]["max_bump_pp_max"] + 1e-9
    )


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_no_catastrophic_iff_no_3pp_bump(artifact, tool):
    p = artifact["per_tool"][tool]
    ev = p["evaluation"]
    catastrophic = [b for b in p["all_bumps"] if b["delta_pp"] >= 3.0]
    assert ev["catastrophic_bumps"] == catastrophic
    assert ev["checks"]["no_catastrophic"] == (not catastrophic)


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_per_tool_verdict_ok_iff_all_checks(artifact, tool):
    ev = artifact["per_tool"][tool]["evaluation"]
    assert ev["verdict_ok"] == all(ev["checks"].values())


@pytest.mark.parametrize("tool", ["gem5", "sniper"])
def test_per_tool_tolerances_block_mirrors_constants(artifact, gen, tool):
    assert artifact["per_tool"][tool]["evaluation"]["tolerances"] == gen.TOOL_TOLERANCES[tool]


# ---------------------------------------------------------------- Group D
def test_overall_verdict_is_conjunction(artifact):
    expected = all(p["evaluation"]["verdict_ok"] for p in artifact["per_tool"].values())
    assert artifact["overall"]["verdict_ok"] == expected


def test_overall_catastrophic_concatenates_per_tool(artifact):
    expected = []
    for tool in ("gem5", "sniper"):
        expected.extend(artifact["per_tool"][tool]["evaluation"]["catastrophic_bumps"])
    assert artifact["overall"]["catastrophic_bumps"] == expected


def test_overall_median_bump_uses_statistics_median(artifact):
    for tool, payload in artifact["per_tool"].items():
        bumps = payload["all_bumps"]
        expected = statistics.median([b["delta_pp"] for b in bumps]) if bumps else 0.0
        assert artifact["overall"]["median_bump_pp"][tool] == round(expected, 6)


def test_overall_median_keys_match_per_tool(artifact):
    assert set(artifact["overall"]["median_bump_pp"].keys()) == set(artifact["per_tool"].keys())


# ---------------------------------------------------------------- Group E
def test_full_artifact_byte_parity(tmp_path):
    out_json = tmp_path / "anchor_monotonicity_replay.json"
    out_md = tmp_path / "anchor_monotonicity_replay.md"
    res = subprocess.run(
        [
            sys.executable,
            str(GEN_PATH),
            "--gem5-json", str(GEM5_PATH),
            "--sniper-json", str(SNIPER_PATH),
            "--json-out", str(out_json),
            "--md-out", str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verdict=PASS" in res.stdout
    assert out_json.read_text() == JSON_PATH.read_text()
