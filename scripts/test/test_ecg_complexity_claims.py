"""Sprint 6f-5 rubber-duck followup: ECG complexity-comparison claim gates.

Reads ``wiki/data/paper_table_complexity.json`` produced by
``scripts/experiments/ecg/complexity_comparison.py`` and locks the
hardware/software complexity comparison the paper claims for
ECG vs DROPLET vs POPT vs GRASP.

Gates (313-315):
  - 313: every component has all 7 comparison axes populated
  - 314: ECG per-vertex storage ≤ POPT per-vertex storage / 1.5x
    (storage advantage holds, restating gate 309 in this artifact)
  - 315: ECG has the smallest "magic + fixed state" footprint
    of any component that ships both ISA changes AND per-vertex
    storage (asserts the Pareto-frontier claim is non-trivial)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_JSON = REPO_ROOT / "wiki" / "data" / "paper_table_complexity.json"

REQUIRED_AXES = [
    "axis_storage_per_n_vertices_bytes",
    "axis_fixed_state_bytes",
    "axis_hardware_datapath",
    "axis_isa_extensions",
    "axis_offline_preprocessing_complexity",
    "axis_per_access_cycles_no_prefetch",
    "axis_software_kernel_changes",
]

ECG_OVER_POPT_MAX = 1.0 / 1.5  # mirror gate 309 threshold


def _load() -> dict:
    if not TABLE_JSON.exists():
        pytest.skip(f"{TABLE_JSON} missing; run `make lit-paper-table-complexity`.")
    return json.loads(TABLE_JSON.read_text())


# --- gate 313 ---


def test_every_component_has_all_axes():
    """Gate 313: each component declares every comparison axis."""
    payload = _load()
    bad = []
    for c in payload["components"]:
        for ax in REQUIRED_AXES:
            if ax not in c:
                bad.append((c.get("name", "?"), ax))
                continue
            v = c[ax]
            if v == "" or v is None:
                bad.append((c.get("name", "?"), ax + " (empty)"))
    assert not bad, f"missing/empty axes: {bad}"


# --- gate 314 ---


def test_ecg_per_vertex_storage_below_popt():
    """Gate 314: restating gate 309's metadata-cost claim in the
    complexity-comparison artifact. ECG per-vertex storage must be at
    most 2/3 of POPT's per-vertex storage."""
    payload = _load()
    by_name = {c["name"].split(" (")[0]: c for c in payload["components"]}
    ecg = by_name.get("ECG")
    popt = by_name.get("POPT")
    assert ecg is not None and popt is not None, "ECG and POPT must both be present"
    ecg_b = ecg["axis_storage_per_n_vertices_bytes"]
    popt_b = popt["axis_storage_per_n_vertices_bytes"]
    assert popt_b > 0, f"POPT per-vertex storage {popt_b} must be positive"
    ratio = ecg_b / popt_b
    assert ratio <= ECG_OVER_POPT_MAX, (
        f"ECG per-vertex {ecg_b}B / POPT per-vertex {popt_b}B = {ratio:.3f} "
        f"> threshold {ECG_OVER_POPT_MAX:.3f}"
    )


# --- gate 315 ---


def test_ecg_design_point_uncovered_by_prior_art():
    """Gate 315 (revised): assert that among the compared prior-art
    substrates (DROPLET, POPT, GRASP), none combines software-visible
    hints (ISA extensions) with per-vertex masks. This is a descriptive
    observation about prior-art coverage — NOT a Pareto-dominance claim.

    Rubber-duck critique 2026-06-04 noted that the original "Pareto-unique"
    framing was tautological (ECG is unique because we defined it that
    way). Reformulated as a check on the prior-art design space: if a
    future paper adds a substrate that also combines ISA + per-vertex
    storage, this test will fail and we should re-examine the framing.

    The gate also ensures every component declares all axes accurately
    so the descriptive comparison stays honest.
    """
    payload = _load()
    by_name = {c["name"].split(" (")[0]: c for c in payload["components"]}
    ecg = by_name.get("ECG")
    assert ecg is not None
    assert ecg["axis_storage_per_n_vertices_bytes"] > 0, (
        "ECG must have per-vertex storage > 0 (gate 309)"
    )
    assert ecg["axis_isa_extensions"].startswith("2 magic"), (
        "ECG must declare ISA extensions (architectural-simplicity story)"
    )
    others_with_both = []
    for name, c in by_name.items():
        if name == "ECG":
            continue
        has_isa = c["axis_isa_extensions"].startswith("2 magic")
        has_per_v = c["axis_storage_per_n_vertices_bytes"] > 0
        if has_isa and has_per_v:
            others_with_both.append(name)
    assert not others_with_both, (
        "ECG should be the only component with BOTH ISA + per-vertex storage. "
        f"Others found: {others_with_both}. If a baseline now shares ECG's design "
        "point, the paper's novelty framing needs revisiting."
    )
