"""Derivation parity gate for wiki/data/popt_vs_grasp_delta.json.

Reproduces the per-cell Δ(POPT - GRASP) projection from the raw lit-faith CSV.
The generator at scripts/experiments/ecg/popt_vs_grasp_report.py answers the
central paper question: when does POPT actually help GRASP, and by how much?

Load-bearing rules pinned here:

* Filter rows where policy IN {GRASP, POPT}; cells lacking EITHER policy skip;
* delta_pp = (popt_miss_rate - grasp_miss_rate) * 100, formatted to 3dp;
* classification floor = 0.5 pp:
    delta < -0.5 → popt_better, > +0.5 → grasp_better, else tie;
* stats: mean/median use fmean/median; stdev uses **pstdev** (population)
  when n>=2 else 0.0; all rounded to 3dp;
* popt_top5_helps = 5 most negative deltas (sorted ascending);
* grasp_top5_helps = 5 most positive deltas (sorted descending);
* by_family_regime key format: f"{family}|{regime}" (pipe-separated);
* sort key for cells: (graph, app, l3_bytes, l3_size).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "wiki" / "data" / "popt_vs_grasp_delta.json"
CSV_PATH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv"

L3_BYTES = {
    "4kB": 4 * 1024, "16kB": 16 * 1024, "64kB": 64 * 1024,
    "256kB": 256 * 1024, "1MB": 1024 * 1024, "2MB": 2 * 1024 * 1024,
    "4MB": 4 * 1024 * 1024, "8MB": 8 * 1024 * 1024,
}

GRAPH_FAMILY = {
    "email-Eu-core": "social", "web-Google": "web", "cit-Patents": "citation",
    "soc-pokec": "social", "soc-LiveJournal1": "social", "com-orkut": "social",
    "roadNet-CA": "road", "delaunay_n19": "mesh",
}

CLASS_FLOOR_PP = 0.5


def _regime(label):
    b = L3_BYTES.get(label, -1)
    if b < 0:
        return "unknown"
    if b < 64 * 1024:
        return "tiny"
    if b < 1024 * 1024:
        return "small"
    return "large"


def _classify(d):
    if math.isnan(d):
        return "missing"
    if d < -CLASS_FLOOR_PP:
        return "popt_better"
    if d > CLASS_FLOOR_PP:
        return "grasp_better"
    return "tie"


def _stats(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean_pp": round(statistics.fmean(values), 3),
        "median_pp": round(statistics.median(values), 3),
        "min_pp": round(min(values), 3),
        "max_pp": round(max(values), 3),
        "stdev_pp": round(statistics.pstdev(values), 3) if len(values) >= 2 else 0.0,
    }


def _expected_records():
    cells = defaultdict(dict)
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            g = r.get("graph") or ""
            a = r.get("app") or r.get("benchmark") or ""
            l = r.get("l3_size") or ""
            p = (r.get("policy") or "").strip()
            try:
                mr = float(r.get("miss_rate") or r.get("l3_miss_rate") or "nan")
            except ValueError:
                continue
            if not math.isfinite(mr):
                continue
            if p not in {"GRASP", "POPT"}:
                continue
            cells[(g, a, l)][p] = mr
    out = []
    for (g, a, l), by_pol in sorted(cells.items(), key=lambda kv: (
        kv[0][0], kv[0][1], L3_BYTES.get(kv[0][2], -1), kv[0][2],
    )):
        gm, pm = by_pol.get("GRASP"), by_pol.get("POPT")
        if gm is None or pm is None:
            continue
        d = (pm - gm) * 100.0
        out.append({
            "graph": g, "graph_family": GRAPH_FAMILY.get(g, "unknown"),
            "app": a, "l3_size": l, "l3_regime": _regime(l),
            "grasp_mr": gm, "popt_mr": pm, "delta_pp": d,
            "classification": _classify(d),
        })
    return out


def _art():
    return json.loads(ART.read_text())


# Group A — cell-level derivation parity --------------------------------

def test_n_cells_matches_derived():
    art = _art()
    assert art["summary"]["n_cells"] == len(_expected_records())
    assert len(art["cells"]) == art["summary"]["n_cells"]


def test_every_cell_has_both_grasp_and_popt():
    art = _art()
    for c in art["cells"]:
        assert c["grasp_miss_rate"] != ""
        assert c["popt_miss_rate"] != ""


def test_delta_pp_equals_popt_minus_grasp_times_100():
    art = _art()
    for c in art["cells"]:
        grasp = float(c["grasp_miss_rate"])
        popt = float(c["popt_miss_rate"])
        expected = (popt - grasp) * 100.0
        assert abs(float(c["delta_pp"]) - round(expected, 3)) < 5e-4, c


def test_classification_floor_is_half_pp():
    art = _art()
    for c in art["cells"]:
        d = float(c["delta_pp"])
        if d < -CLASS_FLOOR_PP:
            assert c["classification"] == "popt_better"
        elif d > CLASS_FLOOR_PP:
            assert c["classification"] == "grasp_better"
        else:
            assert c["classification"] == "tie"


def test_cells_sorted_by_graph_app_l3bytes_l3label():
    art = _art()
    keys = [(c["graph"], c["app"], L3_BYTES.get(c["l3_size"], -1), c["l3_size"])
            for c in art["cells"]]
    assert keys == sorted(keys)


def test_miss_rates_formatted_to_6dp():
    art = _art()
    for c in art["cells"]:
        assert len(c["grasp_miss_rate"].split(".")[-1]) == 6
        assert len(c["popt_miss_rate"].split(".")[-1]) == 6


# Group B — overall stats parity ---------------------------------------

def test_overall_stats_match_pstdev_formula():
    art = _art()
    deltas = [float(c["delta_pp"]) for c in art["cells"]]
    expected = _stats(deltas)
    assert art["summary"]["overall"] == expected


# Group C — by_family / by_regime / by_app stats parity ----------------

def test_by_family_stats_match_pstdev_per_family():
    art = _art()
    buckets = defaultdict(list)
    for c in art["cells"]:
        buckets[c["graph_family"]].append(float(c["delta_pp"]))
    for fam, vals in buckets.items():
        assert art["summary"]["by_family"][fam] == _stats(vals), fam


def test_by_regime_stats_match_pstdev_per_regime():
    art = _art()
    buckets = defaultdict(list)
    for c in art["cells"]:
        buckets[c["l3_regime"]].append(float(c["delta_pp"]))
    for reg, vals in buckets.items():
        assert art["summary"]["by_regime"][reg] == _stats(vals), reg


def test_by_app_stats_match_pstdev_per_app():
    art = _art()
    buckets = defaultdict(list)
    for c in art["cells"]:
        buckets[c["app"]].append(float(c["delta_pp"]))
    for app, vals in buckets.items():
        assert art["summary"]["by_app"][app] == _stats(vals), app


def test_by_family_regime_key_format_is_pipe_separated():
    art = _art()
    for key in art["summary"]["by_family_regime"]:
        assert "|" in key
        fam, reg = key.split("|")
        assert reg in {"tiny", "small", "large", "unknown"}


def test_by_family_regime_stats_match_pstdev_per_pair():
    art = _art()
    buckets = defaultdict(list)
    for c in art["cells"]:
        buckets[(c["graph_family"], c["l3_regime"])].append(float(c["delta_pp"]))
    for (fam, reg), vals in buckets.items():
        assert art["summary"]["by_family_regime"][f"{fam}|{reg}"] == _stats(vals)


def test_per_family_n_sums_to_n_cells():
    art = _art()
    total = sum(b["n"] for b in art["summary"]["by_family"].values())
    assert total == art["summary"]["n_cells"]


def test_per_regime_n_sums_to_n_cells():
    art = _art()
    total = sum(b["n"] for b in art["summary"]["by_regime"].values())
    assert total == art["summary"]["n_cells"]


def test_per_app_n_sums_to_n_cells():
    art = _art()
    total = sum(b["n"] for b in art["summary"]["by_app"].values())
    assert total == art["summary"]["n_cells"]


# Group D — classification_counts parity --------------------------------

def test_classification_counts_match_per_cell_classification():
    art = _art()
    expected = defaultdict(int)
    for c in art["cells"]:
        expected[c["classification"]] += 1
    assert dict(art["summary"]["classification_counts"]) == dict(expected)


def test_classification_counts_sum_to_n_cells():
    art = _art()
    total = sum(art["summary"]["classification_counts"].values())
    assert total == art["summary"]["n_cells"]


def test_classification_only_three_labels():
    art = _art()
    labels = set(art["summary"]["classification_counts"].keys())
    assert labels <= {"popt_better", "grasp_better", "tie"}


# Group E — top-5 tails parity -----------------------------------------

def test_popt_top5_sorted_most_negative_first():
    art = _art()
    deltas = [float(r["delta_pp"]) for r in art["summary"]["popt_top5_helps"]]
    assert deltas == sorted(deltas)


def test_grasp_top5_sorted_most_positive_first():
    art = _art()
    deltas = [float(r["delta_pp"]) for r in art["summary"]["grasp_top5_helps"]]
    assert deltas == sorted(deltas, reverse=True)


def test_top5_tails_lengths_max_five():
    art = _art()
    assert len(art["summary"]["popt_top5_helps"]) <= 5
    assert len(art["summary"]["grasp_top5_helps"]) <= 5


def test_popt_top5_matches_argmin_over_all_cells():
    """The 5 most negative deltas over all cells, sorted ascending."""
    art = _art()
    all_deltas = sorted(float(c["delta_pp"]) for c in art["cells"])
    expected = all_deltas[:5]
    actual = [float(r["delta_pp"]) for r in art["summary"]["popt_top5_helps"]]
    assert actual == expected


def test_grasp_top5_matches_argmax_over_all_cells():
    """The 5 most positive deltas over all cells, sorted descending."""
    art = _art()
    all_deltas = sorted((float(c["delta_pp"]) for c in art["cells"]), reverse=True)
    expected = all_deltas[:5]
    actual = [float(r["delta_pp"]) for r in art["summary"]["grasp_top5_helps"]]
    assert actual == expected
