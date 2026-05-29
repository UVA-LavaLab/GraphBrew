"""L-curve monotonicity gate.

For every cache replacement policy, the L3 miss rate should be a
non-increasing function of L3 cache size: doubling the cache cannot
make hit rate strictly worse outside of noise.  When this gate trips,
something is genuinely broken — either:

  * the policy has a state-space bug that pollutes larger caches with
    stale victims (e.g. a wrong SRRIP RRPV update, a GRASP bypass that
    leaks into the LLC, a POPT cost-matrix indexing bug), or
  * the simulator is mis-reporting miss counts at one of the sizes.

The gate scans the L-curve summary produced by ``paper_pipeline.py``
(``aggregate/l_curve_miss_rate_by_size.csv``) and asserts that no
adjacent (prev_size → cur_size) transition shows the miss rate
*increasing* by more than the configured tolerance.

Tolerance is generous to accommodate genuine simulator noise without
masking real regressions:

  * Absolute floor: ``ABS_TOLERANCE_PP`` = 0.10 pp
  * Relative floor: ``REL_TOLERANCE_FRAC`` = 0.10 (10 %) of the
    smaller of the two adjacent miss rates
  * Saturation skip: when both miss rates are ≥ ``SATURATION_THRESHOLD``
    (default 0.995, i.e. 99.5 %) we skip the transition. At saturation
    everything misses essentially every access and the differences are
    purely tag-set hash collisions.
  * Sub-noise skip: when both miss rates are ≤ ``SUB_NOISE_THRESHOLD``
    (default 0.001, i.e. 0.1 %) we skip — the working set fits and a
    handful of compulsory misses can flip sign arbitrarily.

The L-curve aggregator only emits groups with ≥3 distinct L3 sizes,
so this gate is automatically vacuous on graphs without an L3 sweep.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
L_CURVE_CSV_CANDIDATES = [
    PROJECT_ROOT
    / "results/ecg_experiments/paper_pipeline/lit_baseline_aggregate"
    / "aggregate/l_curve_miss_rate_by_size.csv",
]

ABS_TOLERANCE_PP = 0.10
REL_TOLERANCE_FRAC = 0.10
SATURATION_THRESHOLD = 0.995
SUB_NOISE_THRESHOLD = 0.001


def _find_l_curve_csv() -> Path | None:
    for cand in L_CURVE_CSV_CANDIDATES:
        if cand.exists():
            return cand
    # Fallback: glob the paper_pipeline tree for any l_curve summary.
    matches = sorted(
        PROJECT_ROOT.glob(
            "results/ecg_experiments/paper_pipeline/**/aggregate/l_curve_miss_rate_by_size.csv"
        )
    )
    return matches[-1] if matches else None


def _load_curves() -> dict[tuple[str, str, str], list[tuple[int, str, float]]]:
    csv_path = _find_l_curve_csv()
    if csv_path is None:
        pytest.skip(
            "No L-curve aggregate CSV found; run `python3 scripts/experiments/ecg/"
            "paper_pipeline.py --skip-run --skip-literature-gate "
            "--input-csv-glob '/tmp/graphbrew-lit-baseline/*/lit/roi_matrix.csv' "
            "--run-root results/ecg_experiments/paper_pipeline/lit_baseline_aggregate` "
            "first."
        )
    groups: dict[tuple[str, str, str], list[tuple[int, str, float]]] = defaultdict(list)
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["graph"], row["benchmark"], row["policy_label"])
            groups[key].append((int(row["l3_size_bytes"]), row["l3_size"], float(row["l3_miss_rate"])))
    for pts in groups.values():
        pts.sort()
    return groups


def _tolerance_pp(prev_miss: float, cur_miss: float) -> float:
    rel = REL_TOLERANCE_FRAC * min(prev_miss, cur_miss) * 100.0
    return max(ABS_TOLERANCE_PP, rel)


def _skip_transition(prev_miss: float, cur_miss: float) -> bool:
    if prev_miss >= SATURATION_THRESHOLD and cur_miss >= SATURATION_THRESHOLD:
        return True
    if prev_miss <= SUB_NOISE_THRESHOLD and cur_miss <= SUB_NOISE_THRESHOLD:
        return True
    return False


def test_l_curve_csv_present_and_non_trivial():
    """Sanity: we have an L-curve aggregate with enough curves to be meaningful."""
    groups = _load_curves()
    assert len(groups) >= 5, (
        f"Expected at least 5 (graph, app, policy) curves in L-curve aggregate, "
        f"found {len(groups)}. The L-curve aggregator only emits groups with "
        "≥3 distinct L3 sizes — re-run cache_sim sweeps if the corpus is empty."
    )


def test_l_curve_miss_rate_is_monotone_non_increasing():
    """Larger L3 must not increase miss rate beyond noise tolerance."""
    groups = _load_curves()

    violations: list[str] = []
    for (graph, app, policy), pts in sorted(groups.items()):
        for i in range(1, len(pts)):
            prev_sz, prev_label, prev_miss = pts[i - 1]
            cur_sz, cur_label, cur_miss = pts[i]
            if cur_miss <= prev_miss:
                continue
            if _skip_transition(prev_miss, cur_miss):
                continue
            jump_pp = (cur_miss - prev_miss) * 100.0
            tol_pp = _tolerance_pp(prev_miss, cur_miss)
            if jump_pp > tol_pp:
                violations.append(
                    f"{graph}/{app}/{policy} {prev_label}→{cur_label}: "
                    f"miss {prev_miss:.5f}→{cur_miss:.5f} (+{jump_pp:.3f}pp, "
                    f"tolerance={tol_pp:.3f}pp)"
                )

    if violations:
        msg = (
            f"L-curve is NOT monotone non-increasing for {len(violations)} "
            "policy×graph×app cell transitions. Larger L3 should never "
            "increase miss rate beyond noise; this typically indicates a "
            "replacement-policy bug:\n  " + "\n  ".join(violations)
        )
        pytest.fail(msg)
