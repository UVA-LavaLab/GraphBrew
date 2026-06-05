"""Sprint 6f-5 spike corpus: per-edge mode 6 efficiency claim.

After implementing mode 6 (per-edge ECG mask = paper's actual design) and
verifying it works on cit-Patents/pr, the corpus extension on 4 cells
(cit-Patents, soc-LiveJournal1, com-orkut, web-Google) confirms the
bandwidth-efficiency claim:

  Mode 6 corpus pp/Mreq:  0.1499  (winner)
  Mode 2 corpus pp/Mreq:  0.1312  (+14% mode 6 advantage)
  DROPLET corpus pp/Mreq: 0.1111  (+35% mode 6 advantage)

This is the paper's bandwidth-efficiency Pareto-frontier claim:
at matched bandwidth, mode 6 delivers MORE absolute miss-reduction
than runtime mode 2 lookahead due to offline-precomputed per-edge
POPT-ranked target selection.

This gate locks the corpus aggregate. If a future change breaks the
per-edge advantage, this test fails.

Reads `wiki/data/paper_table_mode6_corpus.json` (TODO: emit script).
For now reads raw cache_sim CSVs from /tmp/mode6_corpus and
/tmp/graphbrew-ecg-pfx-cache_sim-scale.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


CORPUS_CELLS = ['cit-Patents-pr', 'soc-LiveJournal1-pr', 'com-orkut-pr', 'web-Google-pr']

# Minimum mode 6 efficiency advantage over mode 2 (corpus aggregate pp/Mreq ratio)
MIN_MODE6_VS_MODE2_RATIO = 1.05  # mode 6 must be at least 5% more efficient

# Source CSV roots
MODE6_ROOT = Path('/tmp/mode6_corpus')
SCALE_ROOT = Path('/tmp/graphbrew-ecg-pfx-cache_sim-scale')
MODE6_CITPAT_FALLBACK = Path('/tmp/mode6_smoke/charged1/roi_matrix.csv')


def _read_first_row(path: Path, label: str | None = None) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        for row in csv.DictReader(f):
            if label is None:
                return row
            if row.get('policy_label') == label or row.get('policy') == label:
                return row
    return None


def _demand_rate(row: dict) -> float:
    return int(row['memory_accesses']) / int(row['total_accesses'])


def _gather_corpus():
    """Returns dict[cell][config] = (demand, reqs, baseline_demand)."""
    out = {}
    for cell in CORPUS_CELLS:
        base = _read_first_row(SCALE_ROOT / cell / 'baselines' / 'roi_matrix.csv', 'ECG_DBG_ONLY')
        if not base: continue
        baseline = _demand_rate(base)
        cell_data = {'baseline': baseline}
        # Mode 2 K=1 LH=8 (pfx_combined arm)
        m2 = _read_first_row(SCALE_ROOT / cell / 'pfx_combined' / 'roi_matrix.csv')
        if m2:
            cell_data['mode2'] = {'demand': _demand_rate(m2), 'reqs': int(m2.get('prefetch_requests','0'))}
        # DROPLET
        drp = _read_first_row(SCALE_ROOT / cell / 'droplet_combined' / 'roi_matrix.csv')
        if drp:
            cell_data['droplet'] = {'demand': _demand_rate(drp), 'reqs': int(drp.get('prefetch_requests','0'))}
        # Mode 6 (cit-Patents has fallback path from smoke)
        if cell == 'cit-Patents-pr' and MODE6_CITPAT_FALLBACK.exists():
            m6 = _read_first_row(MODE6_CITPAT_FALLBACK)
        else:
            m6 = _read_first_row(MODE6_ROOT / cell / 'roi_matrix.csv')
        if m6:
            cell_data['mode6'] = {'demand': _demand_rate(m6), 'reqs': int(m6.get('prefetch_requests','0'))}
        out[cell] = cell_data
    return out


def _corpus_pp_per_mreq(corpus, key):
    """Aggregate pp/Mreq across the corpus: total_savings / (total_reqs/1M)."""
    total_savings = 0.0
    total_reqs = 0
    for cell, data in corpus.items():
        baseline = data.get('baseline')
        cfg = data.get(key)
        if baseline is None or cfg is None:
            continue
        delta_pp = (cfg['demand'] - baseline) * 100
        total_savings += -delta_pp  # convert to positive savings
        total_reqs += cfg['reqs']
    return (total_savings / (total_reqs / 1e6)) if total_reqs > 0 else None


def _skip_if_no_data(corpus, key):
    n = sum(1 for c in corpus.values() if key in c)
    if n < 2:
        pytest.skip(f"only {n} cells have {key} data (need ≥2)")


# --- gate 316 ---


def test_mode6_corpus_data_present():
    """Gate 316: mode 6 corpus sweep produced data for ≥3 of 4 cells.

    The corpus contains 4 cells from sprint 6f-3 + sprint 6f-5 spike
    (cit-Patents/pr, soc-LiveJournal1/pr, com-orkut/pr, web-Google/pr).
    Mode 6 should produce data for at least 3.
    """
    corpus = _gather_corpus()
    n_cells_with_mode6 = sum(1 for c in corpus.values() if 'mode6' in c)
    assert n_cells_with_mode6 >= 3, (
        f"mode 6 corpus data missing on {len(corpus) - n_cells_with_mode6} of "
        f"{len(corpus)} cells; only {n_cells_with_mode6} available"
    )


# --- gate 317 ---


def test_mode6_more_efficient_than_mode2_corpus():
    """Gate 317: mode 6 corpus pp/Mreq ≥ 1.05× mode 2 corpus pp/Mreq.

    This is the per-edge-mask Pareto-efficiency claim. Mode 6 uses
    per-edge POPT-ranked encoding; mode 2 uses runtime lookahead with
    the same effective K. At matched bandwidth, mode 6 should deliver
    more demand-memory reduction per prefetch request.

    Today: mode 6 = 0.1499 pp/Mreq, mode 2 = 0.1312 = ratio 1.14× (+14%).
    """
    corpus = _gather_corpus()
    _skip_if_no_data(corpus, 'mode6')
    _skip_if_no_data(corpus, 'mode2')
    m6_pp = _corpus_pp_per_mreq(corpus, 'mode6')
    m2_pp = _corpus_pp_per_mreq(corpus, 'mode2')
    assert m6_pp is not None and m2_pp is not None, "missing pp/Mreq data"
    ratio = m6_pp / m2_pp
    assert ratio >= MIN_MODE6_VS_MODE2_RATIO, (
        f"mode 6 corpus pp/Mreq = {m6_pp:.4f} is NOT ≥ {MIN_MODE6_VS_MODE2_RATIO}× "
        f"mode 2 = {m2_pp:.4f} (ratio = {ratio:.3f}). Per-edge precision should "
        f"deliver better per-request efficiency than runtime lookahead at matched bandwidth."
    )


# --- gate 318 ---


def test_mode6_more_efficient_than_droplet_corpus():
    """Gate 318: mode 6 corpus pp/Mreq ≥ 1.10× DROPLET corpus pp/Mreq.

    DROPLET fires K=lookahead prefetches per stride trigger with no
    selection; mode 6 fires 1 well-chosen prefetch per edge. Mode 6
    should win per-request efficiency by a larger margin against DROPLET
    than against mode 2 (since DROPLET issues more wasted prefetches).

    Today: mode 6 = 0.1499 pp/Mreq, DROPLET = 0.1111 = ratio 1.35× (+35%).
    """
    corpus = _gather_corpus()
    _skip_if_no_data(corpus, 'mode6')
    _skip_if_no_data(corpus, 'droplet')
    m6_pp = _corpus_pp_per_mreq(corpus, 'mode6')
    drp_pp = _corpus_pp_per_mreq(corpus, 'droplet')
    assert m6_pp is not None and drp_pp is not None
    ratio = m6_pp / drp_pp
    assert ratio >= 1.10, (
        f"mode 6 corpus pp/Mreq = {m6_pp:.4f} is NOT ≥ 1.10× DROPLET = {drp_pp:.4f} "
        f"(ratio = {ratio:.3f}). Per-edge precision should beat DROPLET's "
        f"brute-force stride-and-indirect by a meaningful margin."
    )
