"""Derivation parity gate for ``wiki/data/oracle_gap_by_app_bootstrap.json``.

Locks the paired Δ bootstrap reducer against its single upstream —
``oracle_gap.json#rows`` — so any drift in the cell-pairing
(by (graph, l3_size), only when BOTH policies have a row), the
delta sign convention (gap_a − gap_b), the iteration order through
apps × ordered policy-pairs (load-bearing for the seeded RNG to
stay reproducible), the deterministic Random(1729) sequence, the
n=2000 resample count, the 95 % CI percentile indices
(lo = int(0.025·n), hi = int(0.975·n) − 1), the p_a_lt_b decision
rule (strict mean < 0), or the rounding (4 dp on mean/CI/p) trips
a test before the dashboard re-publishes the per-kernel oracle-gap
rank CIs.

Mirrors ``_aggregate()`` from
``scripts/experiments/ecg/oracle_gap_by_app_bootstrap.py`` verbatim.
The bootstrap is deterministic (seeded), so this gate re-runs it and
checks BYTE-EXACT equality with the published artifact.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

UPSTREAM_PATH = WIKI_DATA / "oracle_gap.json"
ARTIFACT_PATH = WIKI_DATA / "oracle_gap_by_app_bootstrap.json"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
APPS = ("pr", "bc", "cc", "bfs", "sssp")
N_RESAMPLES = 2000
SEED = 1729
CI_LEVEL = 0.95


def _load_rows(path):
    raw = json.loads(path.read_text())
    rows = raw["rows"]
    out = []
    for r in rows:
        try:
            r = {**r, "gap_pp": float(r["gap_pp"])}
            out.append(r)
        except (ValueError, KeyError):
            continue
    return out


def _paired_deltas(rows, app, a, b):
    by_cell_a, by_cell_b = {}, {}
    for r in rows:
        if r["app"] != app:
            continue
        cell = (r["graph"], r["l3_size"])
        if r["policy"] == a:
            by_cell_a[cell] = r["gap_pp"]
        elif r["policy"] == b:
            by_cell_b[cell] = r["gap_pp"]
    return [by_cell_a[c] - by_cell_b[c] for c in by_cell_a if c in by_cell_b]


def _bootstrap(deltas, rng, n_resamples):
    if not deltas:
        return {
            "n_paired": 0, "p_a_lt_b": None, "mean_delta": None,
            "ci_lo": None, "ci_hi": None,
        }
    n = len(deltas)
    means = []
    n_neg = 0
    for _ in range(n_resamples):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        m = sum(s) / n
        means.append(m)
        if m < 0:
            n_neg += 1
    means.sort()
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo_i = int(alpha * n_resamples)
    hi_i = int((1.0 - alpha) * n_resamples) - 1
    return {
        "n_paired":   n,
        "p_a_lt_b":   round(n_neg / n_resamples, 4),
        "mean_delta": round(sum(deltas) / n, 4),
        "ci_lo":      round(means[lo_i], 4),
        "ci_hi":      round(means[hi_i], 4),
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
def mirror(rows):
    """Re-run the entire bootstrap with the same seed → byte-exact
    mirror of `per_app_pairs`."""
    rng = random.Random(SEED)
    apps_seen = sorted({r["app"] for r in rows}) or list(APPS)
    per_app_pairs = {}
    for app in apps_seen:
        pairs = {}
        for a in POLICIES:
            for b in POLICIES:
                if a == b:
                    continue
                deltas = _paired_deltas(rows, app, a, b)
                pairs[f"{a}_vs_{b}"] = _bootstrap(deltas, rng, N_RESAMPLES)
        per_app_pairs[app] = pairs
    return per_app_pairs


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_app_pairs"}


def test_meta_fields(artifact):
    expected = {
        "apps", "ci_level", "n_resamples",
        "n_total_rows", "policies", "seed",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing: {missing}"


def test_meta_constants(artifact):
    m = artifact["meta"]
    assert m["n_resamples"] == N_RESAMPLES
    assert m["seed"] == SEED
    assert m["ci_level"] == CI_LEVEL
    assert tuple(m["policies"]) == POLICIES


def test_meta_n_total_rows_matches_loaded(artifact, rows):
    assert artifact["meta"]["n_total_rows"] == len(rows)


def test_meta_apps_match_sorted_distinct_from_rows(artifact, rows):
    expected = sorted({r["app"] for r in rows})
    assert artifact["meta"]["apps"] == expected


def test_per_app_keys_match_meta_apps(artifact):
    assert sorted(artifact["per_app_pairs"].keys()) == sorted(artifact["meta"]["apps"])


def test_per_app_pair_keys_are_ordered_pairs_a_ne_b(artifact):
    expected = sorted(
        f"{a}_vs_{b}"
        for a in POLICIES for b in POLICIES if a != b
    )
    for app, pairs in artifact["per_app_pairs"].items():
        assert sorted(pairs.keys()) == expected, (
            f"{app}: pair key set drift"
        )


def test_each_pair_has_expected_fields(artifact):
    expected = {"ci_hi", "ci_lo", "mean_delta", "n_paired", "p_a_lt_b"}
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            assert set(v.keys()) == expected, f"{app}/{k}: field set drift"


# ----------------------------------------------------------------------
# Group B: deterministic n_paired + mean_delta (no RNG needed)
# ----------------------------------------------------------------------

def test_n_paired_matches_paired_deltas_length(artifact, rows):
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            a, b = k.split("_vs_")
            n = len(_paired_deltas(rows, app, a, b))
            assert v["n_paired"] == n, (
                f"{app}/{k}: n_paired {v['n_paired']} ≠ {n}"
            )


def test_mean_delta_matches_paired_arithmetic(artifact, rows):
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            a, b = k.split("_vs_")
            deltas = _paired_deltas(rows, app, a, b)
            expected = (
                round(sum(deltas) / len(deltas), 4) if deltas else None
            )
            assert v["mean_delta"] == expected, (
                f"{app}/{k}: mean_delta {v['mean_delta']} ≠ {expected}"
            )


def test_mean_delta_symmetric_under_pair_swap(artifact):
    """Δ(a vs b) = −Δ(b vs a) cell-by-cell, so means are exact negations."""
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            a, b = k.split("_vs_")
            inv = pairs[f"{b}_vs_{a}"]
            if v["mean_delta"] is None:
                assert inv["mean_delta"] is None
                continue
            assert v["mean_delta"] == round(-inv["mean_delta"], 4), (
                f"{app}/{k}: mean asymmetry "
                f"{v['mean_delta']} vs −{inv['mean_delta']}"
            )


def test_n_paired_symmetric_under_pair_swap(artifact):
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            a, b = k.split("_vs_")
            assert v["n_paired"] == pairs[f"{b}_vs_{a}"]["n_paired"]


# ----------------------------------------------------------------------
# Group C: byte-exact bootstrap reproduction (seeded RNG)
# ----------------------------------------------------------------------

def test_bootstrap_byte_exact_matches_seeded_rerun(artifact, mirror):
    """The bootstrap is seeded — re-running with Random(1729) in the
    same app × ordered-pair iteration order must yield byte-identical
    p_a_lt_b / ci_lo / ci_hi (4dp). Any drift here indicates iteration
    order, seed, n_resamples, or percentile-index arithmetic changed."""
    art = artifact["per_app_pairs"]
    assert set(art.keys()) == set(mirror.keys())
    for app in art:
        assert set(art[app].keys()) == set(mirror[app].keys())
        for k in art[app]:
            assert art[app][k] == mirror[app][k], (
                f"{app}/{k}: bootstrap drift\n  art={art[app][k]}\n"
                f"  mir={mirror[app][k]}"
            )


def test_ci_lo_le_mean_le_ci_hi(artifact):
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            if v["mean_delta"] is None:
                continue
            assert v["ci_lo"] <= v["mean_delta"] <= v["ci_hi"], (
                f"{app}/{k}: mean outside CI"
            )


# ----------------------------------------------------------------------
# Group D: end-to-end sanity
# ----------------------------------------------------------------------

def test_p_a_lt_b_in_unit_band(artifact):
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            if v["p_a_lt_b"] is None:
                continue
            assert 0.0 <= v["p_a_lt_b"] <= 1.0, (
                f"{app}/{k}: p_a_lt_b {v['p_a_lt_b']} outside [0, 1]"
            )


def test_p_quantization_is_2000ths(artifact):
    """n_resamples=2000 → p_a_lt_b is k/2000 for k in [0, 2000].
    Round to 4dp keeps 4 fractional digits which always equals
    int(k*5)/10000 (since 10000 / 2000 = 5).  Strictly: p * 2000
    must be very close to an integer."""
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            p = v["p_a_lt_b"]
            if p is None:
                continue
            scaled = p * N_RESAMPLES
            assert abs(scaled - round(scaled)) < 1e-6, (
                f"{app}/{k}: p_a_lt_b {p} not k/2000"
            )


def test_no_zero_paired_cells_for_present_apps(artifact, rows):
    """Every (app, ordered policy pair) should have ≥1 paired cell
    because the corpus has every (graph, l3_size, app) combination
    populated for all four policies in the current paper grid."""
    for app, pairs in artifact["per_app_pairs"].items():
        for k, v in pairs.items():
            assert v["n_paired"] > 0, (
                f"{app}/{k}: zero paired cells — sparsity regression?"
            )
