"""Gate 131 — arithmetic + classification audit of `per_graph_app_stability.json`.

Independently reproduces, for every (graph, app) pair in the paper L3 scope
(1MB, 4MB, 8MB), the per-L3 winner set (policies tied for min gap_pp), the
intersection-based stability classification (stable_unique, stable_partial,
regime_change, insufficient_l3), and the per-graph rollup + meta accounting,
asserting they match the published artifact exactly.

The artifact is the per-(graph, app) winner-stability disclosure that
pins which cells the paper may quote without a per-L3 disclaimer. A
silent drift in winner detection (tie-tolerance, sign, scope filter) or
classification logic would silently mis-license the paper's cell quotes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
STAB_PATH = REPO_ROOT / "wiki" / "data" / "per_graph_app_stability.json"

PAPER_L3 = ("1MB", "4MB", "8MB")
TIE_TOL = 1e-6


def _winners(rows):
    by_cell = defaultdict(list)
    for r in rows:
        if r["l3_size"] not in PAPER_L3:
            continue
        by_cell[(r["graph"], r["app"], r["l3_size"])].append(
            (float(r["gap_pp"]), r["policy"]))
    out = {}
    for k, pairs in by_cell.items():
        mn = min(g for g, _ in pairs)
        out[k] = sorted(p for g, p in pairs if abs(g - mn) < TIE_TOL)
    return out


def _classify(winners_per_l3):
    l3s = sorted(winners_per_l3.keys())
    union = sorted({p for ws in winners_per_l3.values() for p in ws})
    if len(l3s) < 2:
        return {"classification": "insufficient_l3", "intersection": [],
                "union": union, "unique_in_intersection": False}
    inter = set(winners_per_l3[l3s[0]])
    for l3 in l3s[1:]:
        inter &= set(winners_per_l3[l3])
    inter_s = sorted(inter)
    if not inter:
        cls = "regime_change"
    elif len(inter) == 1 and set(union) == inter:
        cls = "stable_unique"
    elif len(inter) == 1:
        cls = "stable_unique_with_ties"
    else:
        cls = "stable_partial"
    return {"classification": cls, "intersection": inter_s, "union": union,
            "unique_in_intersection": len(inter) == 1}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(STAB_PATH.read_text())


@pytest.fixture(scope="module")
def expected() -> dict:
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    winners = _winners(rows)
    by_ga = defaultdict(dict)
    for (g, a, l), wl in winners.items():
        by_ga[(g, a)][l] = wl
    per_ga = []
    for (graph, app) in sorted(by_ga.keys()):
        rec = _classify(by_ga[(graph, app)])
        rec.update({"graph": graph, "app": app,
                    "l3_sizes_present": sorted(by_ga[(graph, app)].keys()),
                    "winners_by_l3": by_ga[(graph, app)]})
        per_ga.append(rec)
    return {"per_ga": per_ga}


# ---------- Group 1: meta scope ----------

def test_meta_scope_l3_matches_paper(artifact):
    assert tuple(artifact["meta"]["scope_l3_sizes"]) == PAPER_L3


def test_meta_source_path(artifact):
    assert artifact["meta"]["source"].endswith("wiki/data/oracle_gap.json")


def test_meta_n_graph_app_pairs_matches(artifact):
    assert artifact["meta"]["n_graph_app_pairs"] == len(artifact["per_graph_app"])


# ---------- Group 2: per-(graph,app) reproduction ----------

def test_per_graph_app_count_matches_expected(artifact, expected):
    assert len(artifact["per_graph_app"]) == len(expected["per_ga"])


def test_per_graph_app_keys_match(artifact, expected):
    art_keys = sorted((r["graph"], r["app"]) for r in artifact["per_graph_app"])
    exp_keys = sorted((r["graph"], r["app"]) for r in expected["per_ga"])
    assert art_keys == exp_keys


def test_per_graph_app_winners_by_l3_match(artifact, expected):
    exp_by = {(r["graph"], r["app"]): r for r in expected["per_ga"]}
    for r in artifact["per_graph_app"]:
        e = exp_by[(r["graph"], r["app"])]
        assert r["winners_by_l3"] == e["winners_by_l3"], (r["graph"], r["app"])


def test_per_graph_app_l3_sizes_present_match(artifact, expected):
    exp_by = {(r["graph"], r["app"]): r for r in expected["per_ga"]}
    for r in artifact["per_graph_app"]:
        e = exp_by[(r["graph"], r["app"])]
        assert r["l3_sizes_present"] == e["l3_sizes_present"], (r["graph"], r["app"])


def test_per_graph_app_classification_matches(artifact, expected):
    exp_by = {(r["graph"], r["app"]): r for r in expected["per_ga"]}
    for r in artifact["per_graph_app"]:
        e = exp_by[(r["graph"], r["app"])]
        assert r["classification"] == e["classification"], (r["graph"], r["app"])


def test_per_graph_app_intersection_union_match(artifact, expected):
    exp_by = {(r["graph"], r["app"]): r for r in expected["per_ga"]}
    for r in artifact["per_graph_app"]:
        e = exp_by[(r["graph"], r["app"])]
        assert r["intersection"] == e["intersection"], (r["graph"], r["app"])
        assert r["union"] == e["union"], (r["graph"], r["app"])


def test_per_graph_app_unique_in_intersection_matches(artifact):
    for r in artifact["per_graph_app"]:
        assert r["unique_in_intersection"] == (len(r["intersection"]) == 1)


# ---------- Group 3: headline lists ----------

def test_stable_unique_cells_reproduce(artifact):
    expected = [
        f"{r['graph']}/{r['app']} -> {r['intersection'][0]}"
        for r in artifact["per_graph_app"]
        if r["classification"] in ("stable_unique", "stable_unique_with_ties")
    ]
    assert artifact["stable_unique_cells"] == expected


def test_regime_change_cells_reproduce(artifact):
    expected = [
        f"{r['graph']}/{r['app']}"
        for r in artifact["per_graph_app"]
        if r["classification"] == "regime_change"
    ]
    assert artifact["regime_change_cells"] == expected


def test_stable_partial_cells_reproduce(artifact):
    expected = [
        f"{r['graph']}/{r['app']} -> {','.join(r['intersection'])}"
        for r in artifact["per_graph_app"]
        if r["classification"] == "stable_partial"
    ]
    assert artifact["stable_partial_cells"] == expected


def test_insufficient_cells_reproduce(artifact):
    expected = [
        f"{r['graph']}/{r['app']}"
        for r in artifact["per_graph_app"]
        if r["classification"] == "insufficient_l3"
    ]
    assert artifact["insufficient_cells"] == expected


# ---------- Group 4: rollups + meta stats ----------

def test_per_graph_rollup_reproduces(artifact):
    rollup = defaultdict(lambda: {"n_apps": 0, "stable_unique": 0,
                                  "regime_change": 0, "partial": 0})
    for r in artifact["per_graph_app"]:
        b = rollup[r["graph"]]
        b["n_apps"] += 1
        if r["classification"] in ("stable_unique", "stable_unique_with_ties"):
            b["stable_unique"] += 1
        elif r["classification"] == "regime_change":
            b["regime_change"] += 1
        elif r["classification"] == "stable_partial":
            b["partial"] += 1
    assert artifact["per_graph_rollup"] == dict(rollup)


def test_meta_class_counts_match(artifact):
    cls_counts = defaultdict(int)
    for r in artifact["per_graph_app"]:
        cls_counts[r["classification"]] += 1
    m = artifact["meta"]
    assert m["n_stable_unique"] == (cls_counts["stable_unique"]
                                    + cls_counts["stable_unique_with_ties"])
    assert m["n_stable_partial"] == cls_counts["stable_partial"]
    assert m["n_regime_change"] == cls_counts["regime_change"]
    assert m["n_insufficient_l3"] == cls_counts["insufficient_l3"]


def test_meta_stability_fraction_excl_insufficient(artifact):
    m = artifact["meta"]
    denom = max(1, m["n_graph_app_pairs"] - m["n_insufficient_l3"])
    expected = m["n_stable_unique"] / denom
    assert abs(m["stability_fraction_excl_insufficient"] - expected) < 1e-12
