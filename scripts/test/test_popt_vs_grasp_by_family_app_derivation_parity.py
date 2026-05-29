"""Derivation parity gate for ``wiki/data/popt_vs_grasp_by_family_app.json``.

Per-(family × app) paired bootstrap CI on Δ = gap(POPT) − gap(GRASP).
Sibling of gate 174 OGB-Der (oracle_gap_by_app_bootstrap) but cut by
family×app instead of app-pair.

Locks the deterministic seeded bootstrap so any drift in the iteration
order (families sorted × apps sorted — load-bearing for RNG sequence),
the cell match key ((graph, l3_size)), the delta polarity (POPT minus
GRASP), the percentile-CI index math (``int(alpha·n_res)`` lo,
``int((1-alpha)·n_res)-1`` hi), the STRICT ``m < 0`` predicate for
``p_popt_lt_grasp``, the round-4dp quantization, the coverage counters
(``cells_with_data`` requires n_paired ≥ N_PAIRED_FLOOR=3; partial
``cells_skipped_insufficient`` requires 0 < n_paired < 3), or the
None-passthrough for empty cells trips a test before the dashboard
re-publishes the per-(family, app) verdict.

Mirrors ``_aggregate()`` / ``_bootstrap()`` / ``_paired_deltas()`` from
``scripts/experiments/ecg/popt_vs_grasp_by_family_app.py`` verbatim.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "popt_vs_grasp_by_family_app.json"
UPSTREAM_PATH = WIKI_DATA / "oracle_gap.json"

N_RESAMPLES = 2000
SEED = 1729
CI_LEVEL = 0.95
N_PAIRED_FLOOR = 3


def _load_rows(path):
    raw = json.loads(path.read_text())
    out = []
    for r in raw.get("rows", []):
        try:
            r = {**r, "gap_pp": float(r["gap_pp"])}
            out.append(r)
        except (ValueError, KeyError):
            continue
    return out


def _paired_deltas(rows, family, app):
    a_by_cell, b_by_cell = {}, {}
    for r in rows:
        if r["family"] != family or r["app"] != app:
            continue
        cell = (r["graph"], r["l3_size"])
        if r["policy"] == "POPT":
            a_by_cell[cell] = r["gap_pp"]
        elif r["policy"] == "GRASP":
            b_by_cell[cell] = r["gap_pp"]
    return [a_by_cell[c] - b_by_cell[c] for c in a_by_cell if c in b_by_cell]


def _bootstrap(deltas, rng, n_res):
    if not deltas:
        return {
            "n_paired":  0, "p_popt_lt_grasp": None, "mean_delta": None,
            "ci_lo":     None, "ci_hi": None,
        }
    n = len(deltas)
    means = []
    n_neg = 0
    for _ in range(n_res):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        means.append(m)
        if m < 0:
            n_neg += 1
    means.sort()
    alpha = (1.0 - CI_LEVEL) / 2.0
    return {
        "n_paired":         n,
        "p_popt_lt_grasp":  round(n_neg / n_res, 4),
        "mean_delta":       round(sum(deltas) / n, 4),
        "ci_lo":            round(means[int(alpha * n_res)], 4),
        "ci_hi":            round(means[int((1.0 - alpha) * n_res) - 1], 4),
    }


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def rows():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    return _load_rows(UPSTREAM_PATH)


@pytest.fixture(scope="module")
def families(rows):
    return sorted({r["family"] for r in rows})


@pytest.fixture(scope="module")
def apps(rows):
    return sorted({r["app"] for r in rows})


@pytest.fixture(scope="module")
def deltas_by_cell(rows, families, apps):
    return {(f, a): _paired_deltas(rows, f, a) for f in families for a in apps}


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_family_app"}


def test_meta_fields(artifact):
    expected = {
        "n_resamples", "seed", "ci_level", "families", "apps",
        "n_paired_floor", "cells_with_data", "cells_skipped_insufficient",
    }
    assert set(artifact["meta"].keys()) == expected


def test_meta_constants(artifact):
    m = artifact["meta"]
    assert m["n_resamples"] == N_RESAMPLES
    assert m["seed"] == SEED
    assert m["ci_level"] == CI_LEVEL
    assert m["n_paired_floor"] == N_PAIRED_FLOOR


def test_per_cell_entry_shape(artifact):
    expected = {"n_paired", "p_popt_lt_grasp", "mean_delta", "ci_lo", "ci_hi"}
    for k, v in artifact["per_family_app"].items():
        assert set(v.keys()) == expected, f"{k}: cell field drift"


def test_per_cell_keys_are_family_slash_app(artifact, families, apps):
    expected = {f"{f}/{a}" for f in families for a in apps}
    assert set(artifact["per_family_app"].keys()) == expected


# ----------------------------------------------------------------------
# Group B: scope & meta counters
# ----------------------------------------------------------------------

def test_meta_families_sorted_distinct(artifact, families):
    assert artifact["meta"]["families"] == families


def test_meta_apps_sorted_distinct(artifact, apps):
    assert artifact["meta"]["apps"] == apps


def test_cells_with_data_counter(artifact):
    """cells_with_data: n_paired >= N_PAIRED_FLOOR."""
    expected = sum(
        1 for v in artifact["per_family_app"].values()
        if v["n_paired"] >= N_PAIRED_FLOOR
    )
    assert artifact["meta"]["cells_with_data"] == expected


def test_cells_skipped_insufficient_counter(artifact):
    """cells_skipped_insufficient: 0 < n_paired < N_PAIRED_FLOOR
    (n_paired == 0 cells are NOT counted as skipped — load-bearing)."""
    expected = sum(
        1 for v in artifact["per_family_app"].values()
        if 0 < v["n_paired"] < N_PAIRED_FLOOR
    )
    assert artifact["meta"]["cells_skipped_insufficient"] == expected


# ----------------------------------------------------------------------
# Group C: deterministic arithmetic
# ----------------------------------------------------------------------

def test_n_paired_matches_upstream_pairing(artifact, deltas_by_cell):
    for (f, a), d in deltas_by_cell.items():
        assert artifact["per_family_app"][f"{f}/{a}"]["n_paired"] == len(d)


def test_mean_delta_matches_arithmetic(artifact, deltas_by_cell):
    for (f, a), d in deltas_by_cell.items():
        v = artifact["per_family_app"][f"{f}/{a}"]
        if d:
            assert v["mean_delta"] == round(sum(d) / len(d), 4)
        else:
            assert v["mean_delta"] is None


def test_empty_cells_have_all_none_fields(artifact, deltas_by_cell):
    for (f, a), d in deltas_by_cell.items():
        if d:
            continue
        v = artifact["per_family_app"][f"{f}/{a}"]
        assert v["n_paired"] == 0
        assert v["p_popt_lt_grasp"] is None
        assert v["mean_delta"] is None
        assert v["ci_lo"] is None
        assert v["ci_hi"] is None


def test_p_is_quantized_in_units_of_inv_n_resamples(artifact):
    """p_popt_lt_grasp must be n_neg/N_RESAMPLES rounded 4dp — i.e.
    a multiple of 1/2000 = 0.0005 in {0, 0.0005, 0.001, ..., 1.0}."""
    for v in artifact["per_family_app"].values():
        p = v["p_popt_lt_grasp"]
        if p is None:
            continue
        scaled = p * N_RESAMPLES
        assert abs(scaled - round(scaled)) < 1e-9, f"p={p} not on 1/N grid"
        assert 0.0 <= p <= 1.0


# ----------------------------------------------------------------------
# Group D: byte-exact bootstrap reproduction (RNG seed + iter order)
# ----------------------------------------------------------------------

def test_full_bootstrap_byte_exact_reproduction(artifact, rows, families, apps):
    """Re-running Random(1729) over families × apps in the canonical
    iteration order MUST yield byte-identical per-cell results.
    Iteration order is load-bearing: any reshuffle desyncs the RNG.

    Per-cell ci_lo / ci_hi / p_popt_lt_grasp MUST byte-match the
    artifact (~25 cells × 2000 resamples each = 50,000 RNG draws)."""
    rng = random.Random(SEED)
    for f in families:
        for a in apps:
            d = _paired_deltas(rows, f, a)
            expected = _bootstrap(d, rng, N_RESAMPLES)
            artifact_cell = artifact["per_family_app"][f"{f}/{a}"]
            assert artifact_cell == expected, (
                f"{f}/{a}: byte drift\n  art={artifact_cell}\n  exp={expected}"
            )


def test_ci_lo_le_ci_hi(artifact):
    for v in artifact["per_family_app"].values():
        if v["ci_lo"] is not None:
            assert v["ci_lo"] <= v["ci_hi"]


def test_mean_inside_ci(artifact):
    """Bootstrap mean of resamples should bracket the empirical mean
    — though sample-size sensitive at 4dp rounding, the mean MUST
    lie within [ci_lo, ci_hi] modulo a 1e-4 round-trip tolerance."""
    for k, v in artifact["per_family_app"].items():
        if v["mean_delta"] is None:
            continue
        assert v["ci_lo"] - 1e-4 <= v["mean_delta"] <= v["ci_hi"] + 1e-4, (
            f"{k}: mean={v['mean_delta']} not in CI [{v['ci_lo']}, {v['ci_hi']}]"
        )


# ----------------------------------------------------------------------
# Group E: end-to-end sanity
# ----------------------------------------------------------------------

def test_at_least_one_cell_with_data(artifact):
    assert artifact["meta"]["cells_with_data"] >= 1


def test_families_apps_match_per_cell_keys(artifact):
    fams_in_keys = {k.split("/")[0] for k in artifact["per_family_app"]}
    apps_in_keys = {k.split("/")[1] for k in artifact["per_family_app"]}
    assert fams_in_keys == set(artifact["meta"]["families"])
    assert apps_in_keys == set(artifact["meta"]["apps"])


def test_counters_le_total_cells(artifact):
    total = len(artifact["per_family_app"])
    m = artifact["meta"]
    assert m["cells_with_data"] + m["cells_skipped_insufficient"] <= total
