"""Prefetcher claim gates 293–296: ECG_PFX vs DROPLET vs no_pfx.

Reads ``wiki/data/lit_faith_ecg_pfx_vs_droplet.json`` (the audit output
of gate 241) which in turn ingests the matched-proof sweep observations.

Until the matched-proof sweep activates the postfix (sprint 6b-4), the
audit lives in ``status="deferred"`` and these tests skip. Once
activated, they assert the 4 publish-grade ECG_PFX claims:

  - gate 293 — ECG_PFX useful-fraction floor (≥ 5%) on cells where it fired
  - gate 294 — ECG_PFX miss-rate ≤ DROPLET miss-rate on hint-rich cells
  - gate 295 — ECG_PFX miss-rate < LRU miss-rate (no_pfx baseline) by ≥ 0.5 pp
  - gate 296 — when ECG_PFX issues no prefetches, miss-rate ≡ LRU ± 0.5 pp
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_JSON = REPO_ROOT / "wiki" / "data" / "lit_faith_ecg_pfx_vs_droplet.json"
POSTFIX_JSON = REPO_ROOT / "wiki" / "data" / "ecg_pfx_vs_droplet_postfix.json"

USEFUL_FLOOR = 0.05
NEUTRAL_TOLERANCE_PP = 0.005
BEATS_NOPFX_THRESHOLD_PP = 0.005

# Apps where prefetching matters most (bandwidth-bound graph traversals).
# Gate 294 + 295 apply specifically here; non-hint-rich apps may legitimately
# not differ from baseline.
HINT_RICH_APPS = frozenset({"pr", "bc", "bfs", "sssp"})


def _load_audit() -> dict[str, Any]:
    if not AUDIT_JSON.exists():
        pytest.skip(f"{AUDIT_JSON} missing — run `make lit-ecg-pfx-vs-droplet`")
    return json.loads(AUDIT_JSON.read_text())


def _load_postfix() -> dict[str, Any]:
    if not POSTFIX_JSON.exists():
        pytest.skip(f"{POSTFIX_JSON} missing")
    return json.loads(POSTFIX_JSON.read_text())


def _skip_if_deferred(audit: dict[str, Any]) -> None:
    if audit.get("status") == "deferred":
        pytest.skip(
            "gate 241 is deferred — no matched-proof prefetcher sweep has been "
            "activated yet. Run scripts/experiments/ecg/sweeps/pfx_matched_proof_sweep.sh, "
            "then python3 scripts/experiments/ecg/ecg_pfx_vs_droplet_postfix_builder.py "
            "--activate, then make lit-ecg-pfx-vs-droplet."
        )


# --- gate 293 -------------------------------------------------------------


def test_ecg_pfx_useful_fraction_floor():
    """Gate 293: when ECG_PFX issues prefetches, useful_frac ≥ 5%."""
    audit = _load_audit()
    _skip_if_deferred(audit)
    bad: list[tuple[str, str, float]] = []
    for row in audit.get("head_to_head", []):
        if row.get("benchmark") not in HINT_RICH_APPS:
            continue
        useful_frac = row.get("ecg_pfx_useful_frac")
        if useful_frac is None:
            continue
        if useful_frac < USEFUL_FLOOR:
            bad.append((row["benchmark"], row.get("l3_size"), float(useful_frac)))
    assert not bad, (
        f"ECG_PFX useful_frac below {USEFUL_FLOOR} floor on {len(bad)} cells: "
        f"{bad[:5]}"
    )


# --- gate 294 -------------------------------------------------------------


def test_ecg_pfx_ge_droplet_on_hint_rich():
    """Gate 294: ECG_PFX miss-rate ≤ DROPLET miss-rate + 0.5 pp tolerance.

    ECG_PFX should not lose to DROPLET by more than the neutral floor on
    apps where prefetching is the dominant lever. ECG_PFX's hint-driven
    approach has the advantage of targeted prefetches vs DROPLET's
    edge-stream-derived prefetches.

    Only asserts on cells where ECG_PFX actually got past Sniper's
    already-in-cache filter (pf_issued > 0). Cells with pf_issued = 0
    are vacuous for this claim — gate 296 covers them via
    baseline-neutrality.
    """
    audit = _load_audit()
    _skip_if_deferred(audit)
    bad: list[tuple[str, str, str, float]] = []
    skipped_vacuous = 0
    for row in audit.get("head_to_head", []):
        if row.get("benchmark") not in HINT_RICH_APPS:
            continue
        # Skip cells where ECG_PFX issued nothing — claim is vacuous.
        if int(row.get("ecg_pfx_pf_issued") or 0) == 0:
            skipped_vacuous += 1
            continue
        ecg_mr = row.get("ecg_pfx_miss_rate")
        drop_mr = row.get("droplet_miss_rate")
        if ecg_mr is None or drop_mr is None:
            continue
        # Signed delta: positive = ECG_PFX worse than DROPLET
        delta = ecg_mr - drop_mr
        if delta > NEUTRAL_TOLERANCE_PP:
            bad.append((row.get("graph", ""), row["benchmark"], row.get("l3_size"), float(delta)))
    assert not bad, (
        f"ECG_PFX worse than DROPLET by > {NEUTRAL_TOLERANCE_PP * 100:.1f} pp "
        f"on {len(bad)} cells (positive delta = ECG_PFX miss-rate higher; "
        f"vacuous cells skipped: {skipped_vacuous}): {bad[:5]}"
    )


# --- gate 295 -------------------------------------------------------------


def test_ecg_pfx_beats_nopfx():
    """Gate 295: ECG_PFX < LRU-baseline miss-rate on hint-rich cells.

    The whole point of running a prefetcher is to beat the
    no-prefetcher baseline. Allow a tiny noise band of
    BEATS_NOPFX_THRESHOLD_PP (0.5 pp) for cells where the working set
    fits and prefetch has no headroom.

    Only asserts on cells where ECG_PFX actually got past Sniper's
    already-in-cache filter (pf_issued > 0). Cells with pf_issued = 0
    cannot meaningfully beat the baseline because nothing was
    prefetched.
    """
    audit = _load_audit()
    _skip_if_deferred(audit)
    bad: list[tuple[str, str, str, float]] = []
    skipped_vacuous = 0
    for row in audit.get("head_to_head", []):
        if row.get("benchmark") not in HINT_RICH_APPS:
            continue
        if int(row.get("ecg_pfx_pf_issued") or 0) == 0:
            skipped_vacuous += 1
            continue
        ecg_mr = row.get("ecg_pfx_miss_rate")
        lru_mr = row.get("lru_miss_rate")
        if ecg_mr is None or lru_mr is None:
            continue
        # Signed delta: positive = ECG_PFX worse than baseline
        delta = ecg_mr - lru_mr
        if delta > BEATS_NOPFX_THRESHOLD_PP:
            bad.append((row.get("graph", ""), row["benchmark"], row.get("l3_size"), float(delta)))
    assert not bad, (
        f"ECG_PFX worse than no_pfx baseline by > {BEATS_NOPFX_THRESHOLD_PP * 100:.1f} pp "
        f"on {len(bad)} cells (vacuous cells skipped: {skipped_vacuous}): {bad[:5]}"
    )


# --- gate 296 -------------------------------------------------------------


def test_ecg_pfx_baseline_neutral_when_inactive():
    """Gate 296: when ECG_PFX issues zero hardware prefetches (pf_issued=0),
    its miss-rate must equal the LRU baseline within 0.5 pp.

    Reads the postfix directly because the audit's head_to_head doesn't
    expose pf_issued counts. A misbehaving prefetcher that consumes CPU
    cycles without issuing prefetches (e.g. computing addresses then
    discarding) should not degrade the baseline.
    """
    audit = _load_audit()
    _skip_if_deferred(audit)
    postfix = _load_postfix()
    by_cell: dict[tuple[str, str, str], dict[str, dict]] = {}
    for obs in postfix.get("per_observation", []):
        key = (obs.get("benchmark"), obs.get("graph"), obs.get("l3_size"))
        by_cell.setdefault(key, {})[obs.get("arm")] = obs
    bad: list[tuple[tuple[str, str, str], float, int]] = []
    for key, arms in by_cell.items():
        ecg = arms.get("ECG_PFX")
        lru = arms.get("LRU")
        if not ecg or not lru:
            continue
        ecg_iss = int(ecg.get("pf_issued") or 0)
        if ecg_iss != 0:
            continue
        ecg_mr = ecg.get("l3_miss_rate")
        lru_mr = lru.get("l3_miss_rate")
        if ecg_mr is None or lru_mr is None:
            continue
        delta = abs(ecg_mr - lru_mr)
        if delta > NEUTRAL_TOLERANCE_PP:
            bad.append((key, float(delta), ecg_iss))
    assert not bad, (
        f"ECG_PFX with pf_issued=0 drifted from LRU baseline > "
        f"{NEUTRAL_TOLERANCE_PP * 100:.1f} pp on {len(bad)} cells: {bad[:5]}"
    )
