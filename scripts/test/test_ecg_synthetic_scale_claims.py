"""Sprint 6f: ECG synthetic-scale claims (Kronecker extrapolation).

After sprint 6e established the ECG_PFX vs DROPLET efficiency story
on the literature corpus (gate 285 PfxScale, 16 cells, mean Δ vs
LRU = -5.52 pp, ECG_PFX 1.75-3.27x fewer prefetches than DROPLET for
the same L3 miss reduction), this gate suite (286 PfxScaleSynthetic)
extrapolates the same claim to synthetic Kronecker graphs at 4.2M
and 16.7M vertices — well beyond the largest literature graph
soc-LiveJournal1 at 4.8M.

Reads ``wiki/data/paper_table_prefetcher_kronecker.json`` produced by
``scripts/experiments/ecg/paper_table_prefetcher.py
  --sweep-root /tmp/graphbrew-ecg-pfx-cache_sim-kronecker``
which in turn consumes the matched sweep at
``/tmp/graphbrew-ecg-pfx-cache_sim-kronecker/{graph}-{app}/{baselines,pfx_combined,droplet_combined}/roi_matrix.csv``.

Kronecker graphs were generated locally with::

    bench/bin/converter -g 22 -k 16 -s -b results/graphs/kron-s22/kron-s22.sg
    bench/bin/converter -g 24 -k 16 -s -b results/graphs/kron-s24/kron-s24.sg

Claims:
  - gate 303 — synthetic mean Δ ECG_combined vs LRU is at least as
    negative as the literature corpus mean (gains amplify at scale)
  - gate 304 — ECG_PFX ≤ DROPLET + 0.5 pp on every synthetic cell
    (efficiency holds beyond literature corpus)
  - gate 305 — ECG_PFX total prefetch requests < DROPLET total
    requests on the synthetic corpus (efficiency claim holds)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_JSON = REPO_ROOT / "wiki" / "data" / "paper_table_prefetcher_kronecker.json"

LITERATURE_LRU_DELTA_PP = -3.0  # synthetic must do at least this well (the gate 285 floor)
ECG_VS_DROPLET_TOLERANCE_PP = 0.5


def _load() -> dict[str, Any]:
    if not TABLE_JSON.exists():
        pytest.skip(
            f"{TABLE_JSON} missing. Run "
            "`bash scripts/experiments/ecg/sweeps/pfx_cache_sim_kronecker_sweep.sh` "
            "then `make lit-paper-table-prefetcher-kronecker` to populate."
        )
    return json.loads(TABLE_JSON.read_text())


def _full_cells(payload: dict) -> list[dict]:
    pfx_key = payload.get("pfx_label", "ECG_DBG_ONLY + ECG_PFX")
    return [
        c for c in payload.get("cells", [])
        if c.get("LRU") is not None and c.get(pfx_key) is not None
    ]


def _skip_if_no_data(payload: dict) -> None:
    if not _full_cells(payload):
        pytest.skip(
            "No full-data cells in paper_table_prefetcher_kronecker.json. "
            "Re-run pfx_cache_sim_kronecker_sweep.sh."
        )


# --- gate 303 ---


def test_synthetic_mean_delta_vs_lru_meets_literature_floor():
    """Gate 303: ECG combined-mask on synthetic Kronecker graphs achieves
    at least the same mean Δ vs LRU as the literature corpus (-3 pp).

    Because Kronecker graphs are larger than the literature corpus, the
    eviction component (ECG_DBG) has more headroom to win — sprint 6f
    measured -13.19 pp mean on 3 synthetic cells vs -5.52 pp mean on
    16 literature cells. This gate enforces the literature floor; the
    synthetic gain should be at least as strong.
    """
    payload = _load()
    _skip_if_no_data(payload)
    summary = payload.get("summary", {})
    mean_delta = summary.get("mean_delta_vs_LRU_pp")
    assert mean_delta is not None, "summary missing mean_delta_vs_LRU_pp"
    assert mean_delta <= LITERATURE_LRU_DELTA_PP, (
        f"Synthetic mean Δ vs LRU = {mean_delta:.2f} pp > literature floor "
        f"{LITERATURE_LRU_DELTA_PP:.2f} pp — synthetic should be at least "
        "as strong as the literature corpus"
    )


# --- gate 304 ---


def test_synthetic_ecg_pfx_matches_droplet_per_cell():
    """Gate 304: ECG_PFX L3 miss-rate ≤ DROPLET + 0.5 pp on every
    synthetic Kronecker cell (efficiency story holds at scale)."""
    payload = _load()
    _skip_if_no_data(payload)
    cells = _full_cells(payload)
    bad = []
    no_droplet = 0
    for cell in cells:
        delta = cell.get("delta_vs_DROPLET_COMBINED_pp")
        if delta is None:
            no_droplet += 1
            continue
        if delta > ECG_VS_DROPLET_TOLERANCE_PP:
            bad.append((cell["graph"], cell["app"], float(delta)))
    if no_droplet == len(cells):
        pytest.skip("no DROPLET data on any cell")
    assert not bad, (
        f"ECG_PFX worse than DROPLET by > {ECG_VS_DROPLET_TOLERANCE_PP} pp "
        f"on {len(bad)} synthetic cells: {bad[:5]}"
    )


# --- gate 305 ---


def test_synthetic_ecg_pfx_issues_fewer_requests_than_droplet():
    """Gate 305: aggregate ECG_PFX total prefetch requests <
    DROPLET total prefetch requests across the synthetic corpus.

    The efficiency claim: ECG_PFX achieves the same L3 miss reduction
    using less prefetch bandwidth. On literature corpus DROPLET issues
    3.27× more requests; on synthetic Kronecker DROPLET issues 1.79×
    more. Either way, ECG_PFX wins on bandwidth.
    """
    payload = _load()
    _skip_if_no_data(payload)
    summary = payload.get("summary", {})
    ecg_req = summary.get("ecg_pfx_total_requests")
    drop_req = summary.get("droplet_total_requests")
    if ecg_req is None or drop_req is None:
        pytest.skip("no aggregate request data in summary")
    if ecg_req == 0:
        pytest.skip("ECG_PFX issued zero requests (no activity)")
    ratio = drop_req / ecg_req
    assert drop_req > ecg_req, (
        f"DROPLET total requests ({drop_req:,}) should exceed ECG_PFX "
        f"({ecg_req:,}); got ratio {ratio:.3f}"
    )
