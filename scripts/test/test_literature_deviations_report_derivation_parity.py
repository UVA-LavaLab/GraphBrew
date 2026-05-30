"""Derivation parity gate (LDR-Der) for `literature_deviations.json`.

The generator at
``scripts/experiments/ecg/literature_deviations_report.py`` consolidates
every row in ``literature_reproduction_summary.csv`` carrying
``status=known_deviation`` and assigns one of four mechanism labels by
cross-referencing miss-rate data in ``literature_faithfulness_postfix.csv``.
The dashboard surfaces this as the paper's defence layer for the
KNOWN_DEVIATIONS table.

This gate locks every load-bearing derivation rule that determines which
cells appear in the inventory and how they are classified:

* Group A — pinned constants (GRAPH_FAMILY 8-graph mapping,
  MECHANISM_ORDER tuple, indexing key tuple).
* Group B — known_deviation row selection and mr_index construction.
* Group C — classification logic (computed-policy branch, real-policy
  branch, extended-tolerance threshold, missing-data branch).
* Group D — summary aggregator (Counter most_common ordering,
  cross-tab key format, n_deviations conservation).
* Group E — byte parity with the committed JSON.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_deviations_report.py"
JSON_PATH = REPO_ROOT / "wiki" / "data" / "literature_deviations.json"
CSV_PATH = REPO_ROOT / "wiki" / "data" / "literature_deviations.csv"
REPRO_CSV = REPO_ROOT / "wiki" / "data" / "literature_reproduction_summary.csv"
LIT_FAITH_CSV = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv"


def _load_gen():
    spec = importlib.util.spec_from_file_location("literature_deviations_report", GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def artifact():
    return json.loads(JSON_PATH.read_text())


@pytest.fixture(scope="module")
def repro_rows():
    with REPRO_CSV.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def lit_faith_rows():
    with LIT_FAITH_CSV.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- Group A
def test_graph_family_pinned(gen):
    assert gen.GRAPH_FAMILY == {
        "email-Eu-core": "social",
        "web-Google": "web",
        "cit-Patents": "citation",
        "soc-pokec": "social",
        "soc-LiveJournal1": "social",
        "com-orkut": "social",
        "roadNet-CA": "road",
        "delaunay_n19": "mesh",
    }


def test_mechanism_order_tuple_pinned(gen):
    assert gen.MECHANISM_ORDER == (
        "popt_overhead_dominates",
        "within_extended_tolerance",
        "policy_data_missing",
        "unclassified",
    )
    assert isinstance(gen.MECHANISM_ORDER, tuple)


def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"summary", "deviations"}


# ---------------------------------------------------------------- Group B
def test_only_known_deviation_rows_selected(repro_rows, artifact):
    expected_count = sum(1 for r in repro_rows if r.get("status") == "known_deviation")
    assert artifact["summary"]["n_deviations"] == expected_count
    assert len(artifact["deviations"]) == expected_count


def test_deviation_record_keys(artifact):
    expected_keys = {
        "citation", "graph", "graph_family", "app", "l3_size", "policy",
        "expected_sign", "tolerance_pct", "delta_pct", "popt_vs_grasp_pp",
        "mechanism",
    }
    for r in artifact["deviations"]:
        assert set(r.keys()) == expected_keys


def test_graph_family_per_record_matches_pinned_map(artifact, gen):
    for r in artifact["deviations"]:
        assert r["graph_family"] == gen.GRAPH_FAMILY.get(r["graph"], "unknown")


def test_mr_index_key_shape_4tuple(gen, lit_faith_rows):
    idx = gen._build_miss_rate_index(lit_faith_rows)
    assert idx, "expected non-empty index from lit-faith CSV"
    for k in list(idx.keys())[:10]:
        assert isinstance(k, tuple) and len(k) == 4


def test_mr_index_skips_non_finite(gen):
    rows = [
        {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "GRASP", "miss_rate": "nan"},
        {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "LRU", "miss_rate": "0.5"},
        {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "SRRIP", "miss_rate": "bad"},
    ]
    idx = gen._build_miss_rate_index(rows)
    assert ("g", "bc", "1MB", "LRU") in idx
    assert ("g", "bc", "1MB", "GRASP") not in idx
    assert ("g", "bc", "1MB", "SRRIP") not in idx


# ---------------------------------------------------------------- Group C
def test_classify_popt_overhead_dominates_when_popt_exceeds_grasp_plus_tol(gen):
    """Computed POPT_GE_GRASP claim, popt_mr > grasp_mr + tol → popt_overhead_dominates."""
    mr_index = {
        ("g", "bc", "1MB", "GRASP"): 0.5,
        ("g", "bc", "1MB", "POPT"): 0.515,  # 1.5pp over GRASP
    }
    row = {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "POPT_GE_GRASP",
           "delta_pct": "1.5", "tolerance_pct": "1.0"}
    label, delta = gen._classify(row, mr_index)
    assert label == "popt_overhead_dominates"
    assert abs(delta - 1.5) < 1e-9


def test_classify_within_extended_tolerance_when_inside_2x_tol(gen):
    """Computed POPT_GE_GRASP, |popt-grasp| within 2×tol but popt not above tol."""
    mr_index = {
        ("g", "bc", "1MB", "GRASP"): 0.5,
        ("g", "bc", "1MB", "POPT"): 0.508,  # 0.8 pp (positive but <= tol)
    }
    row = {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "POPT_GE_GRASP",
           "delta_pct": "0.8", "tolerance_pct": "1.0"}
    label, delta = gen._classify(row, mr_index)
    assert label == "within_extended_tolerance"
    assert abs(delta - 0.8) < 1e-9


def test_classify_policy_data_missing_when_popt_or_grasp_missing(gen):
    """Computed POPT_GE_GRASP, GRASP missing from mr_index → policy_data_missing."""
    mr_index = {}
    row = {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "POPT_GE_GRASP",
           "delta_pct": "0.5", "tolerance_pct": "1.0"}
    label, delta = gen._classify(row, mr_index)
    assert label == "policy_data_missing"
    assert delta is None


def test_classify_real_policy_missing_from_index(gen):
    """Real policy not in index → policy_data_missing."""
    mr_index = {("g", "bc", "1MB", "GRASP"): 0.5}
    row = {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "LRU",
           "delta_pct": "1.0", "tolerance_pct": "1.0"}
    label, delta = gen._classify(row, mr_index)
    assert label == "policy_data_missing"


def test_classify_real_policy_within_2x_tolerance(gen):
    """Real policy present, |delta_pct| <= 2 × tol → within_extended_tolerance."""
    mr_index = {("g", "bc", "1MB", "LRU"): 0.5,
                ("g", "bc", "1MB", "GRASP"): 0.5, ("g", "bc", "1MB", "POPT"): 0.5}
    row = {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "LRU",
           "delta_pct": "1.5", "tolerance_pct": "1.0"}
    label, delta = gen._classify(row, mr_index)
    assert label == "within_extended_tolerance"


def test_classify_unclassified_fallback(gen):
    """Real policy present, |delta_pct| > 2 × tol → unclassified."""
    mr_index = {("g", "bc", "1MB", "LRU"): 0.5,
                ("g", "bc", "1MB", "GRASP"): 0.5, ("g", "bc", "1MB", "POPT"): 0.5}
    row = {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "LRU",
           "delta_pct": "5.0", "tolerance_pct": "1.0"}
    label, delta = gen._classify(row, mr_index)
    assert label == "unclassified"


def test_classify_computed_policy_popt_near_grasp_if_big_gap(gen):
    """POPT_NEAR_GRASP_IF_BIG_GAP routes through computed-policy branch."""
    mr_index = {
        ("g", "bc", "1MB", "GRASP"): 0.5,
        ("g", "bc", "1MB", "POPT"): 0.520,  # +2 pp
    }
    row = {"graph": "g", "app": "bc", "l3_size": "1MB",
           "policy": "POPT_NEAR_GRASP_IF_BIG_GAP",
           "delta_pct": "2.0", "tolerance_pct": "1.0"}
    label, _ = gen._classify(row, mr_index)
    assert label == "popt_overhead_dominates"


def test_popt_vs_grasp_pp_formula(gen):
    """popt_vs_grasp_pp = (popt_mr - grasp_mr) × 100."""
    mr_index = {
        ("g", "bc", "1MB", "GRASP"): 0.500,
        ("g", "bc", "1MB", "POPT"): 0.513,
    }
    row = {"graph": "g", "app": "bc", "l3_size": "1MB", "policy": "POPT_GE_GRASP",
           "delta_pct": "1.3", "tolerance_pct": "1.0"}
    _, delta = gen._classify(row, mr_index)
    assert abs(delta - 1.3) < 1e-6


def test_records_sorted_by_mechanism_then_graph_app_l3(artifact, gen):
    """Records ordered by (MECHANISM_ORDER.index, graph, app, l3_size)."""
    keys = [
        (gen.MECHANISM_ORDER.index(r["mechanism"]) if r["mechanism"] in gen.MECHANISM_ORDER
         else len(gen.MECHANISM_ORDER), r["graph"], r["app"], r["l3_size"])
        for r in artifact["deviations"]
    ]
    assert keys == sorted(keys)


def test_popt_vs_grasp_pp_formatted_3dp_or_empty(artifact):
    for r in artifact["deviations"]:
        if r["popt_vs_grasp_pp"] != "":
            # Parses cleanly as a float with up to 3 decimals.
            v = float(r["popt_vs_grasp_pp"])
            assert abs(v - round(v, 3)) < 1e-9


# ---------------------------------------------------------------- Group D
def test_summary_by_mechanism_counts_match_records(artifact):
    """by_mechanism counts match Counter over records; JSON sort_keys=True
    means keys are alphabetically sorted on write — preserve that as the
    expected layout."""
    counter = Counter(r["mechanism"] for r in artifact["deviations"])
    assert dict(artifact["summary"]["by_mechanism"]) == dict(counter)
    keys = list(artifact["summary"]["by_mechanism"].keys())
    assert keys == sorted(keys)


def test_summary_by_graph_counts_match_records(artifact):
    counter = Counter(r["graph"] for r in artifact["deviations"])
    assert dict(artifact["summary"]["by_graph"]) == dict(counter)
    keys = list(artifact["summary"]["by_graph"].keys())
    assert keys == sorted(keys)


def test_summary_by_family_counts_match_records(artifact):
    counter = Counter(r["graph_family"] for r in artifact["deviations"])
    assert dict(artifact["summary"]["by_family"]) == dict(counter)
    keys = list(artifact["summary"]["by_family"].keys())
    assert keys == sorted(keys)


def test_summary_by_app_counts_match_records(artifact):
    counter = Counter(r["app"] for r in artifact["deviations"])
    assert dict(artifact["summary"]["by_app"]) == dict(counter)
    keys = list(artifact["summary"]["by_app"].keys())
    assert keys == sorted(keys)


def test_summary_by_policy_counts_match_records(artifact):
    counter = Counter(r["policy"] for r in artifact["deviations"])
    assert dict(artifact["summary"]["by_policy"]) == dict(counter)
    keys = list(artifact["summary"]["by_policy"].keys())
    assert keys == sorted(keys)


def test_cross_tab_key_format_mech_pipe_family(artifact):
    cross = defaultdict(int)
    for r in artifact["deviations"]:
        cross[(r["mechanism"], r["graph_family"])] += 1
    expected = {f"{m}|{f}": n for (m, f), n in sorted(cross.items())}
    assert artifact["summary"]["mechanism_family_cross_tab"] == expected


def test_n_deviations_equals_records_len(artifact):
    assert artifact["summary"]["n_deviations"] == len(artifact["deviations"])


def test_summary_counts_sum_equals_total(artifact):
    """Each by_X aggregator sums to n_deviations (no records dropped)."""
    n = artifact["summary"]["n_deviations"]
    for key in ("by_mechanism", "by_graph", "by_family", "by_app", "by_policy"):
        assert sum(artifact["summary"][key].values()) == n


# ---------------------------------------------------------------- Group E
def test_full_artifact_byte_parity(tmp_path):
    out_csv = tmp_path / "literature_deviations.csv"
    out_json = tmp_path / "literature_deviations.json"
    out_md = tmp_path / "literature_deviations.md"
    res = subprocess.run(
        [
            sys.executable,
            str(GEN_PATH),
            "--repro-csv", str(REPRO_CSV),
            "--lit-faith-csv", str(LIT_FAITH_CSV),
            "--csv-out", str(out_csv),
            "--json-out", str(out_json),
            "--md-out", str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[lit-deviations]" in res.stdout
    assert out_json.read_text() == JSON_PATH.read_text()
    assert out_csv.read_text() == CSV_PATH.read_text()
