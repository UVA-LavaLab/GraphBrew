"""Sprint 6f-5 P1: ECG metadata-cost claim — primary architectural-simplicity
gate.

This gate locks the headline claim that ECG's per-vertex mask uses
substantially less storage than POPT's re-reference matrix at the same
graph scale. Computed by ``scripts/experiments/ecg/metadata_cost.py``
which is a static calculation from the MaskConfig bit budget defaults
(graph_cache_context.h:213-292) and the POPT matrix dimensions
(bench/src_sim/pr.cc:76 makeOffsetMatrix args).

Reads ``wiki/data/paper_table_metadata_cost.json``.

Claims:
  - gate 309 — ECG mask storage ≤ POPT matrix storage / 1.5 (at least
    33% smaller per graph)
  - gate 310 — ECG mask storage ≤ GRASP + POPT + DROPLET sum / 1.5
    (combined-baseline saving)
  - gate 311 — bit budget integrity: DBG + POPT + prefetch + reserved
    = container width (single source of truth for the mask layout)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_JSON = REPO_ROOT / "wiki" / "data" / "paper_table_metadata_cost.json"

ECG_OVER_POPT_MAX_RATIO = 1.0 / 1.5  # ECG must be at most 67% of POPT
ECG_OVER_BASELINE_MAX_RATIO = 1.0 / 1.5  # ECG must be at most 67% of baseline sum


def _load() -> dict:
    if not TABLE_JSON.exists():
        pytest.skip(
            f"{TABLE_JSON} missing. Run "
            "`make lit-paper-table-metadata-cost` to populate."
        )
    return json.loads(TABLE_JSON.read_text())


# --- gate 309 ---


def test_ecg_mask_smaller_than_popt_matrix():
    """Gate 309: ECG mask storage ≤ 2/3 × POPT matrix storage on every graph.

    ECG packs DBG + POPT quant + prefetch target into one container,
    while POPT requires the full numEpochs × numCacheLines re-reference
    matrix. Threshold is 1.5× looser than the theoretical 2× (container
    bits / POPT entries per vertex) to absorb any per-graph rounding.
    """
    payload = _load()
    bad = []
    for r in payload["corpus"]:
        ratio = r["ecg_vs_popt_ratio"]
        if ratio > ECG_OVER_POPT_MAX_RATIO:
            bad.append((r["graph"], ratio))
    assert not bad, (
        f"ECG mask storage too large vs POPT on {len(bad)} graphs "
        f"(max ratio {ECG_OVER_POPT_MAX_RATIO:.3f}): {bad[:5]}"
    )


# --- gate 310 ---


def test_ecg_mask_smaller_than_combined_baseline():
    """Gate 310: ECG mask storage ≤ 2/3 × (GRASP + POPT + DROPLET) on
    every graph.

    The headline architectural-simplicity claim: ECG replaces 3 separate
    pieces of state with one per-vertex mask, saving substantial storage.
    """
    payload = _load()
    bad = []
    for r in payload["corpus"]:
        ratio = r["ecg_vs_baseline_combined_ratio"]
        if ratio > ECG_OVER_BASELINE_MAX_RATIO:
            bad.append((r["graph"], ratio))
    assert not bad, (
        f"ECG mask storage too large vs baseline-combined on {len(bad)} "
        f"graphs (max ratio {ECG_OVER_BASELINE_MAX_RATIO:.3f}): {bad[:5]}"
    )


# --- gate 311 ---


def test_ecg_mask_bit_budget_integrity():
    """Gate 311: the ECG mask bit budget actually adds up to the container
    width. Catches accidental over-allocation that would silently
    truncate one of the fields.
    """
    payload = _load()
    s = payload["summary"]
    total = (
        s["ecg_dbg_bits_per_vertex"]
        + s["ecg_popt_bits_per_vertex"]
        + s["ecg_pfx_bits_per_vertex"]
        + s["ecg_reserved_bits_per_vertex"]
    )
    assert total == s["ecg_container_bits"], (
        f"ECG mask bits don't sum to container width: "
        f"DBG={s['ecg_dbg_bits_per_vertex']} + "
        f"POPT={s['ecg_popt_bits_per_vertex']} + "
        f"PFX={s['ecg_pfx_bits_per_vertex']} + "
        f"reserved={s['ecg_reserved_bits_per_vertex']} = {total} "
        f"!= container {s['ecg_container_bits']}"
    )
    # Also positive bit budget
    for k in ("ecg_dbg_bits_per_vertex", "ecg_popt_bits_per_vertex",
              "ecg_pfx_bits_per_vertex"):
        assert s[k] > 0, f"{k} = {s[k]} (must be > 0)"
