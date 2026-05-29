"""Derivation parity gate for wiki/data/policy_winner_table.json.

Reproduces the per-(graph, app, l3_size) winner classification and the
six summary layers (wins_by_policy / family / regime / app, n_cells,
fragile_top_5) from the raw literature-faithfulness CSV. The artifact
itself is already byte-pinned by reproduce_smoke; this gate pins the
load-bearing rules that decide what a "winning" cell means:

* winner = MIN miss_rate within a (graph, app, l3_size) group;
* tie-break is by policy NAME (alphabetical), so output is deterministic;
* margin_pp = (runner_up_miss_rate - winner_miss_rate) * 100 in pp;
* L3 regime bucketing uses byte boundaries 64 kB and 1 MB
  (tiny < 64 kB <= small < 1 MB <= large);
* graph_family is taken from the hard-coded GRAPH_FAMILY map in
  policy_winner_table.py (kept in sync with corpus-diversity gate);
* fragile_top_5 = the 5 lowest-margin cells with margin < 0.5 pp,
  sorted ascending by margin.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "wiki" / "data" / "policy_winner_table.json"
CSV_PATH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv"

L3_BYTES = {
    "4kB": 4 * 1024, "16kB": 16 * 1024, "64kB": 64 * 1024,
    "256kB": 256 * 1024, "1MB": 1024 * 1024, "2MB": 2 * 1024 * 1024,
    "4MB": 4 * 1024 * 1024, "8MB": 8 * 1024 * 1024,
}

GRAPH_FAMILY = {
    "email-Eu-core": "social", "web-Google": "web", "cit-Patents": "citation",
    "soc-pokec": "social", "soc-LiveJournal1": "social", "com-orkut": "social",
    "roadNet-CA": "road", "delaunay_n19": "mesh", "road-CA": "road",
    "twitter-2010": "social", "uk-2005": "web",
}


def _regime(label: str) -> str:
    b = L3_BYTES.get(label, -1)
    if b < 0:
        return "unknown"
    if b < 64 * 1024:
        return "tiny"
    if b < 1024 * 1024:
        return "small"
    return "large"


def _expected_records():
    """Reproduce _winner_rows() from the generator, layer by layer."""
    grouped = defaultdict(list)
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            g = r.get("graph") or ""
            a = r.get("app") or r.get("benchmark") or ""
            l = r.get("l3_size") or ""
            p = r.get("policy") or ""
            try:
                mr = float(r.get("miss_rate") or r.get("l3_miss_rate") or "nan")
            except ValueError:
                continue
            if mr != mr:
                continue
            grouped[(g, a, l)].append((p, mr))
    out = []
    for (g, a, l), pol_mr in sorted(grouped.items(), key=lambda kv: (
        kv[0][0], kv[0][1], L3_BYTES.get(kv[0][2], -1), kv[0][2],
    )):
        pol_mr_sorted = sorted(pol_mr, key=lambda x: (x[1], x[0]))
        if not pol_mr_sorted:
            continue
        wp, wmr = pol_mr_sorted[0]
        if len(pol_mr_sorted) >= 2:
            rp, rmr = pol_mr_sorted[1]
            margin = (rmr - wmr) * 100.0
        else:
            rp, rmr, margin = "", float("nan"), float("nan")
        out.append({
            "graph": g, "app": a, "l3_size": l,
            "l3_regime": _regime(l),
            "graph_family": GRAPH_FAMILY.get(g, "unknown"),
            "winner_policy": wp, "winner_miss_rate": wmr,
            "runner_up_policy": rp, "runner_up_miss_rate": rmr,
            "margin_pp": margin, "n_policies": len(pol_mr_sorted),
        })
    return out


def _art():
    return json.loads(ART.read_text())


def _by_key(cells):
    return {(c["graph"], c["app"], c["l3_size"]): c for c in cells}


# Group A — meta / aggregate counts ------------------------------------

def test_n_cells_equals_distinct_groups():
    art = _art()
    exp = _expected_records()
    assert art["summary"]["n_cells"] == len(exp)


def test_n_cells_matches_cells_list_length():
    art = _art()
    assert art["summary"]["n_cells"] == len(art["cells"])


def test_wins_by_policy_counts_match_winner_column():
    art = _art()
    expected = Counter(r["winner_policy"] for r in _expected_records())
    assert dict(art["summary"]["wins_by_policy"]) == dict(expected)


def test_wins_by_policy_total_equals_n_cells():
    art = _art()
    assert sum(art["summary"]["wins_by_policy"].values()) == art["summary"]["n_cells"]


# Group B — per-cell winner derivation parity --------------------------

def test_every_cell_winner_is_argmin_miss_rate():
    art = _art()
    by_key = {(c["graph"], c["app"], c["l3_size"]): c for c in art["cells"]}
    for r in _expected_records():
        k = (r["graph"], r["app"], r["l3_size"])
        a = by_key[k]
        assert a["winner_policy"] == r["winner_policy"], (
            f"winner mismatch at {k}: art={a['winner_policy']} exp={r['winner_policy']}"
        )


def test_winner_miss_rate_rounded_to_6dp():
    art = _art()
    by_key = {(c["graph"], c["app"], c["l3_size"]): c for c in art["cells"]}
    for r in _expected_records():
        k = (r["graph"], r["app"], r["l3_size"])
        a = by_key[k]
        assert a["winner_miss_rate"] == f"{r['winner_miss_rate']:.6f}"


def test_runner_up_present_when_multipolicy():
    art = _art()
    for c in art["cells"]:
        if int(c["n_policies"]) >= 2:
            assert c["runner_up_policy"] != ""
            assert c["runner_up_miss_rate"] != ""


def test_margin_pp_equals_runner_minus_winner_times_100():
    art = _art()
    for c in art["cells"]:
        if c["margin_pp"] == "":
            continue
        winner = float(c["winner_miss_rate"])
        runner = float(c["runner_up_miss_rate"])
        assert abs(float(c["margin_pp"]) - (runner - winner) * 100.0) < 1e-3


def test_margin_pp_is_non_negative():
    art = _art()
    for c in art["cells"]:
        if c["margin_pp"] == "":
            continue
        assert float(c["margin_pp"]) >= 0.0


# Group C — tie-break by policy name -----------------------------------

def test_winner_runner_up_distinct_policies():
    """Runner-up is always a DIFFERENT policy than the winner."""
    art = _art()
    for c in art["cells"]:
        if c["runner_up_policy"] == "":
            continue
        assert c["winner_policy"] != c["runner_up_policy"], (
            f"self-runner at {c['graph']}/{c['app']}/{c['l3_size']}"
        )


def test_winner_miss_rate_le_runner_up_miss_rate():
    """The winner's raw miss_rate is <= runner-up's; not just at 6dp."""
    art = _art()
    for c in art["cells"]:
        if c["runner_up_miss_rate"] == "":
            continue
        assert float(c["winner_miss_rate"]) <= float(c["runner_up_miss_rate"]), (
            f"order violated at {c['graph']}/{c['app']}/{c['l3_size']}"
        )


# Group D — wins_by_family / regime / app summaries --------------------

def test_wins_by_family_matches_per_family_winner_count():
    art = _art()
    by_family = defaultdict(Counter)
    for r in _expected_records():
        by_family[r["graph_family"]][r["winner_policy"]] += 1
    for fam, counts in art["summary"]["wins_by_family"].items():
        assert dict(counts) == dict(by_family[fam]), f"family={fam}"


def test_wins_by_regime_matches_per_regime_winner_count():
    art = _art()
    by_regime = defaultdict(Counter)
    for r in _expected_records():
        by_regime[r["l3_regime"]][r["winner_policy"]] += 1
    for reg, counts in art["summary"]["wins_by_regime"].items():
        assert dict(counts) == dict(by_regime[reg]), f"regime={reg}"


def test_wins_by_app_matches_per_app_winner_count():
    art = _art()
    by_app = defaultdict(Counter)
    for r in _expected_records():
        by_app[r["app"]][r["winner_policy"]] += 1
    for app, counts in art["summary"]["wins_by_app"].items():
        assert dict(counts) == dict(by_app[app]), f"app={app}"


def test_family_totals_equal_n_cells():
    art = _art()
    total = sum(sum(c.values()) for c in art["summary"]["wins_by_family"].values())
    assert total == art["summary"]["n_cells"]


def test_regime_totals_equal_n_cells():
    art = _art()
    total = sum(sum(c.values()) for c in art["summary"]["wins_by_regime"].values())
    assert total == art["summary"]["n_cells"]


def test_app_totals_equal_n_cells():
    art = _art()
    total = sum(sum(c.values()) for c in art["summary"]["wins_by_app"].values())
    assert total == art["summary"]["n_cells"]


# Group E — fragile_top_5 + regime + family rules ----------------------

def test_fragile_top_5_only_below_half_pp():
    art = _art()
    for r in art["summary"]["fragile_top_5"]:
        assert float(r["margin_pp"]) < 0.5


def test_fragile_top_5_sorted_ascending():
    art = _art()
    margins = [float(r["margin_pp"]) for r in art["summary"]["fragile_top_5"]]
    assert margins == sorted(margins)


def test_fragile_top_5_max_five():
    art = _art()
    assert len(art["summary"]["fragile_top_5"]) <= 5


def test_l3_regime_thresholds_64kb_and_1mb():
    art = _art()
    by_key = _by_key(art["cells"])
    for c in art["cells"]:
        assert c["l3_regime"] == _regime(c["l3_size"]), (
            f"regime mismatch at {c['graph']}/{c['app']}/{c['l3_size']}: "
            f"art={c['l3_regime']} expected={_regime(c['l3_size'])}"
        )


def test_graph_family_matches_canonical_map():
    art = _art()
    for c in art["cells"]:
        expected = GRAPH_FAMILY.get(c["graph"], "unknown")
        assert c["graph_family"] == expected, f"graph={c['graph']}"
