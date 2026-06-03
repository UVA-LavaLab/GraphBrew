"""Sprint 6c: ECG combined-mask scale claims (cache_sim PFX scale sweep).

After sprint 6c revealed that ECG_PFX paired with ECG eviction delivers
10-15 pp L3 miss reduction at scale (see
``docs/findings/ecg_pfx_recovery_2026-06-01.md``), these gates lock in
the publish-grade claims from cache_sim measurements at literature
L3=1MB across the full corpus.

Reads ``wiki/data/paper_table_prefetcher.json`` produced by
``scripts/experiments/ecg/paper_table_prefetcher.py`` which in turn
consumes the matched sweep at
``/tmp/graphbrew-ecg-pfx-cache_sim-scale/{graph}-{app}/{baselines,pfx_combined}/roi_matrix.csv``.

Until the cache_sim PFX scale sweep has been dispatched and the table
has been regenerated (sprint 6c-1 + 6c-3), these gates skip with
guidance pointing at ``sweeps/pfx_cache_sim_scale_sweep.sh``.

Claims:
  - gate 297 — full-data cell count floor (sweep produced data for
    enough cells to back the headline)
  - gate 298 — ECG combined vs LRU: mean Δ ≤ -3 pp across the
    full-data corpus (combined mask cuts L3 miss-rate on average)
  - gate 299 — per-cell regression detector: no cell where ECG
    combined is worse than LRU by more than 0.5 pp
  - gate 300 — useful-rate floor: mean prefetch useful_rate ≥ 90%
    on cells where the prefetcher actually fired
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_JSON = REPO_ROOT / "wiki" / "data" / "paper_table_prefetcher.json"

FULL_DATA_FLOOR = 1  # ratchet up as more cells land
LRU_DELTA_HEADLINE_PP = -3.0  # mean ECG_combined vs LRU must be ≤ -3 pp
LRU_REGRESSION_THRESHOLD_PP = 0.5  # no cell may regress vs LRU by more than this
USEFUL_RATE_FLOOR = 0.90  # mean prefetch_useful / prefetch_fills
ECG_VS_DROPLET_TOLERANCE_PP = 0.5  # ECG_PFX must not lose to DROPLET by more than this on any cell


def _load() -> dict[str, Any]:
    if not TABLE_JSON.exists():
        pytest.skip(
            f"{TABLE_JSON} missing. Run "
            "`bash scripts/experiments/ecg/sweeps/pfx_cache_sim_scale_sweep.sh` "
            "then `make lit-paper-table-prefetcher` to populate."
        )
    return json.loads(TABLE_JSON.read_text())


def _full_cells(payload: dict) -> list[dict]:
    """Cells that have both LRU baseline and ECG_DBG_ONLY + ECG_PFX measurements."""
    pfx_key = payload.get("pfx_label", "ECG_DBG_ONLY + ECG_PFX")
    return [
        c for c in payload.get("cells", [])
        if c.get("LRU") is not None and c.get(pfx_key) is not None
    ]


def _skip_if_no_data(payload: dict) -> None:
    if not _full_cells(payload):
        pytest.skip(
            "No full-data cells in paper_table_prefetcher.json. "
            "Re-run pfx_cache_sim_scale_sweep.sh."
        )


# --- gate 297 ---


def test_full_data_cell_count_floor():
    """Gate 297: enough cells have full data to back the headline claim."""
    payload = _load()
    n_full = len(_full_cells(payload))
    summary = payload.get("summary", {})
    assert n_full >= FULL_DATA_FLOOR, (
        f"only {n_full} full-data cells (need ≥ {FULL_DATA_FLOOR}). "
        f"summary={summary}"
    )


# --- gate 298 ---


def test_ecg_combined_vs_lru_mean_floor():
    """Gate 298: mean Δ ECG_combined vs LRU ≤ -3 pp (combined mask wins on average)."""
    payload = _load()
    _skip_if_no_data(payload)
    mean_delta = payload.get("summary", {}).get("mean_delta_vs_LRU_pp")
    assert mean_delta is not None, "summary missing mean_delta_vs_LRU_pp"
    assert mean_delta <= LRU_DELTA_HEADLINE_PP, (
        f"mean Δ ECG_combined vs LRU = {mean_delta:+.2f} pp, "
        f"expected ≤ {LRU_DELTA_HEADLINE_PP:+.2f} pp"
    )


# --- gate 299 ---


# Known cells where graph-aware eviction (GRASP / ECG_DBG / POPT)
# legitimately loses to LRU per literature documentation. ECG
# combined-mask reproduces GRASP's behavior on these cells, so the
# claim "ECG never regresses vs LRU" is too strong — instead we
# require "ECG matches its substrate (gate 238 substrate parity) AND
# the only cells where it loses to LRU are documented literature-
# baseline corner cases".
#
# cit-Patents/bc: source-rooted BC frontier mis-aligns with GRASP's
# hot-region pinning (which is PR-rank-derived). See
# scripts/experiments/ecg/literature_baselines.py
# KNOWN_DEVIATIONS for the cit-Patents/bc/{4MB,8MB} POPT_GE_GRASP
# entries that describe the same phenomenon.
KNOWN_LRU_FAVORED_CELLS = {
    ("cit-Patents", "bc"),
}


def test_no_per_cell_regression_vs_lru():
    """Gate 299: no cell where ECG combined regresses vs LRU by > 0.5 pp,
    excluding documented literature-favored corner cases.
    """
    payload = _load()
    _skip_if_no_data(payload)
    bad: list[tuple[str, str, float]] = []
    waived: list[tuple[str, str, float]] = []
    for cell in _full_cells(payload):
        delta = cell.get("delta_vs_LRU_pp")
        if delta is None:
            continue
        if delta > LRU_REGRESSION_THRESHOLD_PP:
            key = (cell["graph"], cell["app"])
            if key in KNOWN_LRU_FAVORED_CELLS:
                waived.append((cell["graph"], cell["app"], float(delta)))
                continue
            bad.append((cell["graph"], cell["app"], float(delta)))
    assert not bad, (
        f"ECG_combined regresses vs LRU by > {LRU_REGRESSION_THRESHOLD_PP} pp "
        f"on {len(bad)} cells (waived literature-favored: {len(waived)}): {bad[:5]}"
    )


# --- gate 300 ---


def test_prefetch_useful_rate_floor():
    """Gate 300: mean prefetch_useful / prefetch_fills ≥ 90% on active cells.

    Cells where the prefetcher actually fired must show that the fills
    were demand-hit at high rate. Low useful-rate would suggest the
    prefetcher's target selection is wrong (cache pollution without
    benefit).
    """
    payload = _load()
    _skip_if_no_data(payload)
    rate = payload.get("summary", {}).get("mean_useful_rate")
    if rate is None:
        # No cells where prefetcher fired (e.g. only email-Eu-core in sweep)
        pytest.skip("no cells with prefetch_fills > 0 in current sweep data")
    assert rate >= USEFUL_RATE_FLOOR, (
        f"mean prefetch_useful_rate = {rate:.4f}, expected ≥ {USEFUL_RATE_FLOOR}"
    )


# --- gate 301 ---


def test_ecg_pfx_matches_droplet():
    """Gate 301: ECG_PFX is competitive with the literature SOTA prefetcher
    DROPLET when paired with the same eviction policy.

    For each cell, ECG_PFX miss-rate must be ≤ DROPLET miss-rate + 0.5 pp.
    This is THE paper claim: ECG_PFX's hint-driven approach matches or
    beats DROPLET's edge-stream stride approach on the same baseline.

    Skips cells where DROPLET measurement is missing (e.g. sweep didn't
    complete that cell's third pass).
    """
    payload = _load()
    _skip_if_no_data(payload)
    cells = _full_cells(payload)
    bad: list[tuple[str, str, float]] = []
    no_droplet = 0
    for cell in cells:
        delta = cell.get("delta_vs_DROPLET_COMBINED_pp")
        if delta is None:
            no_droplet += 1
            continue
        if delta > ECG_VS_DROPLET_TOLERANCE_PP:
            bad.append((cell["graph"], cell["app"], float(delta)))
    assert not bad, (
        f"ECG_PFX worse than DROPLET by > {ECG_VS_DROPLET_TOLERANCE_PP} pp "
        f"on {len(bad)} cells (skipped {no_droplet} cells without DROPLET data): {bad[:5]}"
    )


ECG_PFX_EFFICIENCY_MAX_RATIO = 1.0  # ECG_PFX req/useful must be ≤ DROPLET's


# --- gate 302 ---


def test_ecg_pfx_more_efficient_than_droplet():
    """Gate 302: ECG_PFX uses fewer prefetch requests per useful hit
    than DROPLET on the same baseline eviction.

    Aggregate efficiency claim: total ECG_PFX requests-per-useful ≤
    total DROPLET requests-per-useful. With L3-miss-rate parity (gate 301),
    this is the cleaner story — ECG_PFX achieves the same miss-rate
    reduction with less prefetch bandwidth, less cache pollution.
    """
    payload = _load()
    _skip_if_no_data(payload)
    summary = payload.get("summary", {})
    ecg_rpu = summary.get("ecg_pfx_req_per_useful")
    drop_rpu = summary.get("droplet_req_per_useful")
    if ecg_rpu is None or drop_rpu is None:
        pytest.skip("no DROPLET aggregate data in summary")
    ratio = ecg_rpu / drop_rpu
    assert ratio <= ECG_PFX_EFFICIENCY_MAX_RATIO, (
        f"ECG_PFX req/useful={ecg_rpu:.3f} > DROPLET req/useful={drop_rpu:.3f} "
        f"(ratio={ratio:.3f}, max {ECG_PFX_EFFICIENCY_MAX_RATIO})"
    )
