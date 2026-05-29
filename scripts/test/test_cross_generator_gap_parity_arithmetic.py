"""Gate 129 — arithmetic + parity audit of `cross_generator_gap_parity.json`.

Independently reproduces the cross-source (oracle_gap → oracle_gap_auc →
cache_sensitivity_slope) parity table from the raw oracle_gap rows + the
AUC trajectory + the slope generator's gap_at_* fields and octave records,
asserting every cell agrees with the published artifact to the documented
1e-3 pp tolerance.

This is a structural-integrity gate: the paper's narrative depends on the
three load-bearing aggregators reporting identical mean gap_pp values for
every (app, policy, L3) triple. The gate also locks the meta accounting
(n_cells_checked / n_full_triple_cells / n_mismatches) and the sort
ordering of cells.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
AUC_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap_auc.json"
SLOPE_PATH = REPO_ROOT / "wiki" / "data" / "cache_sensitivity_slope.json"
PARITY_PATH = REPO_ROOT / "wiki" / "data" / "cross_generator_gap_parity.json"

PAPER_L3 = ("1MB", "4MB", "8MB")
TOL = 1e-3  # generator's published tolerance


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(PARITY_PATH.read_text())


@pytest.fixture(scope="module")
def raw_means_and_counts() -> tuple[dict, dict]:
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    acc: dict = defaultdict(list)
    for r in rows:
        if r["l3_size"] not in PAPER_L3:
            continue
        acc[(r["app"], r["policy"], r["l3_size"])].append(float(r["gap_pp"]))
    means = {k: round(statistics.mean(vs), 4) for k, vs in acc.items()}
    counts = {k: len(vs) for k, vs in acc.items()}
    return means, counts


@pytest.fixture(scope="module")
def auc_vals() -> dict:
    auc = json.loads(AUC_PATH.read_text())
    out: dict = {}
    for app, app_blob in auc["per_app"].items():
        for pol, traj in app_blob["trajectory_by_policy"].items():
            for l3, v in traj.items():
                if l3 in PAPER_L3:
                    out[(app, pol, l3)] = float(v)
    return out


@pytest.fixture(scope="module")
def slope_vals() -> dict:
    slope = json.loads(SLOPE_PATH.read_text())
    out: dict = {}
    for app, app_blob in slope["per_app"].items():
        for pol, blob in app_blob.items():
            for l3 in PAPER_L3:
                key = f"gap_at_{l3}"
                if key in blob:
                    out[(app, pol, l3)] = float(blob[key])
            for oct_ in blob.get("octaves", []):
                if oct_.get("from") in PAPER_L3:
                    out.setdefault((app, pol, oct_["from"]), float(oct_["gap_from"]))
                if oct_.get("to") in PAPER_L3:
                    out.setdefault((app, pol, oct_["to"]), float(oct_["gap_to"]))
    return out


# ---------- Group 1: meta sanity ----------

def test_meta_scope_is_paper_l3(artifact):
    assert tuple(artifact["meta"]["scope_l3_sizes"]) == PAPER_L3


def test_meta_tolerance_matches(artifact):
    assert artifact["meta"]["tolerance_pp"] == TOL


def test_meta_sources_paths_match_expected(artifact):
    s = artifact["meta"]["sources"]
    assert s["oracle_gap"].endswith("wiki/data/oracle_gap.json")
    assert s["oracle_gap_auc"].endswith("wiki/data/oracle_gap_auc.json")
    assert s["cache_sensitivity"].endswith("wiki/data/cache_sensitivity_slope.json")


def test_meta_counts_match_cells(artifact):
    cells = artifact["cells"]
    assert artifact["meta"]["n_cells_checked"] == len(cells)
    assert artifact["meta"]["n_full_triple_cells"] == sum(
        1 for c in cells if c["all_three_present"]
    )
    assert artifact["meta"]["n_mismatches"] == sum(1 for c in cells if not c["agree"])


# ---------- Group 2: per-cell raw mean reproduction ----------

def test_cell_raw_means_match(artifact, raw_means_and_counts):
    means, _ = raw_means_and_counts
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        if c["raw_mean_gap_pp"] is None:
            assert k not in means, f"artifact says no raw for {k} but we got {means.get(k)}"
            continue
        assert abs(c["raw_mean_gap_pp"] - means[k]) < TOL, k


def test_cell_n_graphs_in_raw_matches(artifact, raw_means_and_counts):
    _, counts = raw_means_and_counts
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        expected = counts.get(k)
        assert c["n_graphs_in_raw"] == expected, k


def test_cell_auc_values_match(artifact, auc_vals):
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        if c["auc_trajectory_gap_pp"] is None:
            assert k not in auc_vals, k
            continue
        assert abs(c["auc_trajectory_gap_pp"] - auc_vals[k]) < TOL, k


def test_cell_slope_values_match(artifact, slope_vals):
    for c in artifact["cells"]:
        k = (c["app"], c["policy"], c["l3_size"])
        if c["slope_gap_pp"] is None:
            assert k not in slope_vals, k
            continue
        assert abs(c["slope_gap_pp"] - slope_vals[k]) < TOL, k


# ---------- Group 3: spread + agree derivations ----------

def test_cell_spread_matches_max_minus_min(artifact):
    for c in artifact["cells"]:
        present = [v for v in (c["raw_mean_gap_pp"], c["auc_trajectory_gap_pp"],
                               c["slope_gap_pp"]) if v is not None]
        assert present, "every cell should have at least one source value"
        want = round(max(present) - min(present), 6)
        assert c["spread_pp"] == want, (c["app"], c["policy"], c["l3_size"], want, c["spread_pp"])


def test_cell_all_three_present_matches(artifact):
    for c in artifact["cells"]:
        all_three = all(c[k] is not None for k in
                        ("raw_mean_gap_pp", "auc_trajectory_gap_pp", "slope_gap_pp"))
        assert c["all_three_present"] == all_three


def test_cell_agree_is_conjunction(artifact):
    for c in artifact["cells"]:
        want = (c["spread_pp"] <= TOL) and c["all_three_present"]
        assert c["agree"] == want, (c["app"], c["policy"], c["l3_size"])


def test_mismatches_complement_agree(artifact):
    derived = [c for c in artifact["cells"] if not c["agree"]]
    # mismatches list rows must equal cells where agree is False; compare by key triples
    art_keys = [(m["app"], m["policy"], m["l3_size"]) for m in artifact["mismatches"]]
    der_keys = [(c["app"], c["policy"], c["l3_size"]) for c in derived]
    assert sorted(art_keys) == sorted(der_keys)


# ---------- Group 4: enumeration completeness + sort ----------

def test_cells_sorted_by_app_policy_l3(artifact):
    keys = [(c["app"], c["policy"], c["l3_size"]) for c in artifact["cells"]]
    assert keys == sorted(keys)


def test_cells_keyset_is_union_of_sources(artifact, raw_means_and_counts, auc_vals, slope_vals):
    means, _ = raw_means_and_counts
    expected_keys = sorted(set(means) | set(auc_vals) | set(slope_vals))
    got_keys = [(c["app"], c["policy"], c["l3_size"]) for c in artifact["cells"]]
    assert got_keys == expected_keys


def test_cells_per_l3_only_paper_scope(artifact):
    for c in artifact["cells"]:
        assert c["l3_size"] in PAPER_L3, c
