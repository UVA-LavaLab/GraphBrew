"""Gate 144+ — gap_distribution_shape derivation parity.

gap_distribution_shape.json is derived from oracle_gap.json by:

  1. Filter rows to paper L3 set (1MB, 4MB, 8MB)
  2. Group gap_pp by (app, L3, policy) → 60 cells = 5 apps × 3 L3 × 4 policies
  3. For each cell compute:
       n, mean, sd (sample), min, max
       sample skewness (Fisher-Pearson g1, sample-adjusted)
       sample excess kurtosis (Fisher g2, sample-adjusted with bias correction)
  4. Compute observed envelope: worst |skew|, worst |kurt|, list of
     cells outside (|skew|>=2 OR |kurt|>=7), worst-cell labels
  5. Compute pinned-exception delta: new_offenders, gone_offenders vs
     PINNED_EXCEPTION_CELLS
  6. Per-L3 summary: n_cells, worst_abs_skew, worst_abs_kurt per L3
  7. bootstrap_validity_verdict: PASS iff no new_offenders AND
     total cells_outside <= PINNED_EXCEPTION_CELLS_MAX (14)

Why this gate matters
---------------------
This is the only artifact that determines whether plain-percentile
bootstrap is statistically valid for the per-cell sample sizes the
paper relies on. If the moment formulas or the envelope thresholds
ever silently change, every downstream CI claim becomes suspect.

Load-bearing rules:
  - Sample skewness uses Fisher-Pearson g1 (NOT scipy's default biased
    or skew without adjustment): (n/((n-1)(n-2))) * Σ((x-m)/sd)³
  - Sample excess kurtosis uses bias-corrected formula:
    n(n+1)/((n-1)(n-2)(n-3)) * Σ((x-m)/sd)⁴ - 3(n-1)²/((n-2)(n-3))
  - sd is statistics.stdev (sample, n-1 denominator)
  - Envelope thresholds: 2.0 / 7.0 (Hesterberg 2015, Efron/Tibshirani 1993)

Invariants (20 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, per_cell
  2. meta.source == oracle_gap.json
  3. meta.scope_l3_sizes == ('1MB','4MB','8MB')
  4. meta.n_cells == 60 (5 × 3 × 4)
  5. per_cell keys match expected (app__L3__policy) cartesian product
     (allowing missing cells)

Group B — Cell aggregation (oracle_gap → per_cell n/mean/sd/min/max)
  6. n == count of rows for (app, L3, policy)
  7. mean_gap_pp == round(fmean(gap_pp), 4) at 1e-6
  8. sd_gap_pp == round(statistics.stdev(gap_pp), 4) at 1e-3 if n>=2
  9. min/max match exact rounded sample extrema

Group C — Moment reproduction (sample skewness + excess kurtosis)
  10. skewness_g1 == round(Fisher-Pearson g1, 4) at 1e-3
  11. excess_kurtosis_g2 == round(sample-adjusted g2, 4) at 1e-3
  12. For n<3: skewness defaults to 0.0; n<4: kurtosis defaults to 0.0
  13. Constant sample (sd==0) yields skew=0 and kurt=0

Group D — Envelope derivation
  14. observed_envelope.worst_abs_skew_any_cell == max over |skewness_g1|
  15. observed_envelope.worst_abs_kurt_any_cell == max over |excess_kurtosis_g2|
  16. cells_outside_envelope == sorted list of cells with
      |skew|>=2 OR |kurt|>=7 (in app/L3/policy slash form)
  17. n_cells_outside_envelope == len(cells_outside_envelope)
  18. pinned_exception_set.new_offenders == cells_outside - PINNED;
      gone_offenders == PINNED - cells_outside

Group E — Per-L3 summary + verdict
  19. per_l3_worst[L3] reproduces from per-cell aggregation per L3
  20. bootstrap_validity_verdict == 'PASS' iff no new_offenders AND
      n_cells_outside_envelope <= 14
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
EPS_STAT = 1e-6
EPS_MOMENT = 1e-3
MAX_ABS_SKEW = 2.0
MAX_ABS_KURT = 7.0


def _g1(xs: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = sum(xs) / n
    s2 = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(s2)
    if sd == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * sum(((x - m) / sd) ** 3 for x in xs)


def _g2(xs: list[float]) -> float:
    n = len(xs)
    if n < 4:
        return 0.0
    m = sum(xs) / n
    s2 = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(s2)
    if sd == 0:
        return 0.0
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(
        ((x - m) / sd) ** 4 for x in xs
    ) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap.json").read_text())


@pytest.fixture(scope="module")
def gds() -> dict:
    return json.loads((WIKI_DATA / "gap_distribution_shape.json").read_text())


@pytest.fixture(scope="module")
def cells(og) -> dict:
    """{(app, l3, pol): [gap_pp values]} for paper L3 rows."""
    d = defaultdict(list)
    for r in og["rows"]:
        if r["l3_size"] in PAPER_L3_SIZES:
            d[(r["app"], r["l3_size"], r["policy"])].append(float(r["gap_pp"]))
    return d


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(gds):
    assert set(gds.keys()) >= {"meta", "per_cell"}


def test_meta_source(gds):
    assert gds["meta"]["source"] == "wiki/data/oracle_gap.json"


def test_meta_scope_l3_sizes(gds):
    assert tuple(gds["meta"]["scope_l3_sizes"]) == PAPER_L3_SIZES


def test_meta_n_cells_60(gds):
    assert gds["meta"]["n_cells"] == 60
    assert len(gds["per_cell"]) == 60


def test_per_cell_keys_use_double_underscore_separator(gds):
    bad = [k for k in gds["per_cell"] if k.count("__") != 2]
    assert not bad, bad


# ─── Group B — Cell aggregation ──────────────────────────────────────


def test_per_cell_n_matches_oracle(gds, cells):
    mism = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        exp_n = len(cells.get((app, l3, pol), []))
        if info["n"] != exp_n:
            mism.append((key, info["n"], exp_n))
    assert not mism, mism[:5]


def test_per_cell_mean_matches(gds, cells):
    mism = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        xs = cells.get((app, l3, pol), [])
        if not xs:
            continue
        exp = round(sum(xs) / len(xs), 4)
        if abs(info["mean_gap_pp"] - exp) > EPS_STAT:
            mism.append((key, info["mean_gap_pp"], exp))
    assert not mism, mism[:5]


def test_per_cell_sd_matches(gds, cells):
    mism = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        xs = cells.get((app, l3, pol), [])
        if len(xs) < 2:
            continue
        exp = round(statistics.stdev(xs), 4)
        if abs(info["sd_gap_pp"] - exp) > EPS_MOMENT:
            mism.append((key, info["sd_gap_pp"], exp))
    assert not mism, mism[:5]


def test_per_cell_min_max_match(gds, cells):
    mism = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        xs = cells.get((app, l3, pol), [])
        if not xs:
            continue
        if abs(info["min_gap_pp"] - round(min(xs), 4)) > EPS_STAT:
            mism.append(("min", key, info["min_gap_pp"], round(min(xs), 4)))
        if abs(info["max_gap_pp"] - round(max(xs), 4)) > EPS_STAT:
            mism.append(("max", key, info["max_gap_pp"], round(max(xs), 4)))
    assert not mism, mism[:5]


# ─── Group C — Moment reproduction ──────────────────────────────────


def test_skewness_g1_reproduces(gds, cells):
    mism = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        xs = cells.get((app, l3, pol), [])
        exp = round(_g1(xs), 4)
        if abs(info["skewness_g1"] - exp) > EPS_MOMENT:
            mism.append((key, info["skewness_g1"], exp))
    assert not mism, mism[:5]


def test_excess_kurtosis_g2_reproduces(gds, cells):
    mism = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        xs = cells.get((app, l3, pol), [])
        exp = round(_g2(xs), 4)
        if abs(info["excess_kurtosis_g2"] - exp) > EPS_MOMENT:
            mism.append((key, info["excess_kurtosis_g2"], exp))
    assert not mism, mism[:5]


def test_small_n_defaults_to_zero_moments(gds, cells):
    """For n<3 skew defaults to 0.0; for n<4 kurt defaults to 0.0."""
    bad = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        n = len(cells.get((app, l3, pol), []))
        if n < 3 and info["skewness_g1"] != 0.0:
            bad.append(("skew", key, n, info["skewness_g1"]))
        if n < 4 and info["excess_kurtosis_g2"] != 0.0:
            bad.append(("kurt", key, n, info["excess_kurtosis_g2"]))
    assert not bad, bad


def test_constant_samples_yield_zero_moments(gds, cells):
    bad = []
    for key, info in gds["per_cell"].items():
        app, l3, pol = key.split("__")
        xs = cells.get((app, l3, pol), [])
        if xs and len(set(xs)) == 1:
            if info["skewness_g1"] != 0.0 or info["excess_kurtosis_g2"] != 0.0:
                bad.append((key, xs[0], info))
    assert not bad, bad


# ─── Group D — Envelope derivation ──────────────────────────────────


def test_worst_abs_skew_reproduces(gds):
    exp = round(max(abs(c["skewness_g1"]) for c in gds["per_cell"].values()), 4)
    obs = gds["meta"]["observed_envelope"]["worst_abs_skew_any_cell"]
    assert abs(obs - exp) <= EPS_MOMENT, (obs, exp)


def test_worst_abs_kurt_reproduces(gds):
    exp = round(max(abs(c["excess_kurtosis_g2"]) for c in gds["per_cell"].values()), 4)
    obs = gds["meta"]["observed_envelope"]["worst_abs_kurt_any_cell"]
    assert abs(obs - exp) <= EPS_MOMENT, (obs, exp)


def test_cells_outside_envelope_filter(gds):
    expected = sorted(
        f"{c['app']}/{c['l3_size']}/{c['policy']}"
        for c in gds["per_cell"].values()
        if abs(c["skewness_g1"]) >= MAX_ABS_SKEW
        or abs(c["excess_kurtosis_g2"]) >= MAX_ABS_KURT
    )
    got = gds["meta"]["observed_envelope"]["cells_outside_envelope"]
    assert got == expected


def test_n_cells_outside_matches_list_length(gds):
    obs = gds["meta"]["observed_envelope"]
    assert obs["n_cells_outside_envelope"] == len(obs["cells_outside_envelope"])


def test_new_and_gone_offenders_set_diff(gds):
    outs = set(gds["meta"]["observed_envelope"]["cells_outside_envelope"])
    pin = set(gds["meta"]["pinned_exception_set"]["pinned"])
    new_exp = sorted(outs - pin)
    gone_exp = sorted(pin - outs)
    pin_block = gds["meta"]["pinned_exception_set"]
    assert pin_block["new_offenders_vs_pin"] == new_exp
    assert pin_block["gone_offenders_vs_pin"] == gone_exp


# ─── Group E — Per-L3 summary + verdict ──────────────────────────────


def test_per_l3_worst_reproduces(gds):
    mism = []
    for l3 in PAPER_L3_SIZES:
        skews = [
            abs(c["skewness_g1"])
            for c in gds["per_cell"].values()
            if c["l3_size"] == l3
        ]
        kurts = [
            abs(c["excess_kurtosis_g2"])
            for c in gds["per_cell"].values()
            if c["l3_size"] == l3
        ]
        exp = {
            "n_cells": len(skews),
            "worst_abs_skew": round(max(skews), 4) if skews else 0.0,
            "worst_abs_kurt": round(max(kurts), 4) if kurts else 0.0,
        }
        got = gds["meta"]["per_l3_worst"][l3]
        if got["n_cells"] != exp["n_cells"]:
            mism.append(("n", l3, got, exp))
        if abs(got["worst_abs_skew"] - exp["worst_abs_skew"]) > EPS_MOMENT:
            mism.append(("skew", l3, got, exp))
        if abs(got["worst_abs_kurt"] - exp["worst_abs_kurt"]) > EPS_MOMENT:
            mism.append(("kurt", l3, got, exp))
    assert not mism, mism


def test_verdict_logic(gds):
    pin = gds["meta"]["pinned_exception_set"]
    obs = gds["meta"]["observed_envelope"]
    expected = (
        "PASS"
        if (not pin["new_offenders_vs_pin"]
            and obs["n_cells_outside_envelope"] <= pin["max_allowed"])
        else "FAIL"
    )
    assert gds["meta"]["bootstrap_validity_verdict"] == expected
