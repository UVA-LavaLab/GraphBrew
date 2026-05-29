"""Derivation parity gate for wiki/data/cross_tool_winners.json.

Reproduces the cross-tool winner-agreement projection from raw lit-faith
CSV + gem5_anchor + sniper_anchor. Generator at
scripts/experiments/ecg/cross_tool_winners_report.py asks: 'do the
simulators agree on the winning policy for the same (graph, app)?'

Load-bearing rules pinned here:

* TOOLS = (cache_sim, gem5, sniper) — three-way agreement;
* per-tool winner = argmin(miss_rate) over per-tool data, tie-break by
  policy name (alphabetical);
* per-tool cell is collapsed to LARGEST L3 (most saturated) — different
  tools sweep different L3 sizes, so equal-L3 overlap would be ~zero;
* cells where < 2 tools have data are SKIPPED;
* classification: unanimous (all winners equal) / majority (>=2 agree)
  / split (every tool different);
* margin_pp = (runner_up_mr - winner_mr) * 100, formatted as 3dp string;
* winner_miss_rate formatted as 6dp string when finite, else "";
* cell sort key = (graph, app) ascending.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "wiki" / "data" / "cross_tool_winners.json"
LIT_CSV = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv"
GEM5 = REPO_ROOT / "wiki" / "data" / "gem5_anchor.json"
SNIPER = REPO_ROOT / "wiki" / "data" / "sniper_anchor.json"

L3_BYTES = {
    "4kB": 4 * 1024, "16kB": 16 * 1024, "32kB": 32 * 1024,
    "64kB": 64 * 1024, "256kB": 256 * 1024, "1MB": 1024 * 1024,
    "2MB": 2 * 1024 * 1024, "4MB": 4 * 1024 * 1024, "8MB": 8 * 1024 * 1024,
}


def _winner(miss_by_pol):
    items = sorted(
        ((p, mr) for p, mr in miss_by_pol.items() if math.isfinite(mr)),
        key=lambda x: (x[1], x[0]),
    )
    if not items:
        return ("", float("nan"), "", float("nan"))
    if len(items) == 1:
        return (items[0][0], items[0][1], "", float("nan"))
    return (items[0][0], items[0][1], items[1][0], items[1][1])


def _read_anchor(path):
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out = {}
    for c in d.get("cells", []):
        key = (c.get("graph", ""), c.get("app", ""), c.get("l3_size", ""))
        out[key] = dict(c.get("miss_rate_by_policy", {}))
    return out


def _read_lit():
    out = defaultdict(dict)
    with LIT_CSV.open() as f:
        for r in csv.DictReader(f):
            key = (r.get("graph", ""), r.get("app") or r.get("benchmark", ""),
                   r.get("l3_size", ""))
            pol = (r.get("policy") or "").strip()
            try:
                mr = float(r.get("miss_rate") or r.get("l3_miss_rate", "nan"))
            except ValueError:
                continue
            if math.isfinite(mr) and pol:
                out[key][pol] = mr
    return out


def _max_l3(src):
    out = {}
    for (g, a, l), mbp in src.items():
        if not mbp:
            continue
        prev = out.get((g, a))
        if prev is None or L3_BYTES.get(l, -1) > L3_BYTES.get(prev[0], -1):
            out[(g, a)] = (l, mbp)
    return out


def _classify(winners):
    vals = [v for v in winners.values() if v]
    if not vals:
        return "missing"
    most = Counter(vals).most_common(1)[0][1]
    if most == len(vals):
        return "unanimous"
    if most >= 2:
        return "majority"
    return "split"


def _build_expected():
    lf = _max_l3(_read_lit())
    g5 = _max_l3(_read_anchor(GEM5))
    sn = _max_l3(_read_anchor(SNIPER))
    out = []
    for key in sorted(set(lf) | set(g5) | set(sn)):
        srcs = {"cache_sim": lf.get(key), "gem5": g5.get(key), "sniper": sn.get(key)}
        winners = {}
        for tool, entry in srcs.items():
            if not entry:
                winners[tool] = ""
            else:
                wp, _, _, _ = _winner(entry[1])
                winners[tool] = wp
        n = sum(1 for v in winners.values() if v)
        if n < 2:
            continue
        out.append({
            "graph": key[0], "app": key[1], "n_tools": n,
            "winners": winners, "classification": _classify(winners),
        })
    return out


def _art():
    return json.loads(ART.read_text())


# Group A — meta + cell-count parity ----------------------------------

def test_n_cells_matches_derived():
    assert _art()["summary"]["n_cells"] == len(_build_expected())


def test_n_cells_matches_cells_list():
    art = _art()
    assert art["summary"]["n_cells"] == len(art["cells"])


def test_every_cell_has_at_least_two_tool_winners():
    art = _art()
    for c in art["cells"]:
        winners = [c.get(f"{t}_winner", "") for t in ("cache_sim", "gem5", "sniper")]
        assert sum(1 for w in winners if w) >= 2, c


def test_n_tools_matches_nonempty_winner_count():
    art = _art()
    for c in art["cells"]:
        winners = [c.get(f"{t}_winner", "") for t in ("cache_sim", "gem5", "sniper")]
        assert c["n_tools"] == sum(1 for w in winners if w)


# Group B — per-cell winner derivation parity --------------------------

def test_every_cell_winner_matches_derived():
    art = _art()
    exp = {(r["graph"], r["app"]): r for r in _build_expected()}
    for c in art["cells"]:
        e = exp[(c["graph"], c["app"])]
        for tool in ("cache_sim", "gem5", "sniper"):
            assert c[f"{tool}_winner"] == e["winners"][tool], (c, tool)


def test_classification_matches_derived():
    art = _art()
    exp = {(r["graph"], r["app"]): r["classification"] for r in _build_expected()}
    for c in art["cells"]:
        assert c["classification"] == exp[(c["graph"], c["app"])], c


def test_margin_pp_string_3dp_or_empty():
    art = _art()
    for c in art["cells"]:
        for tool in ("cache_sim", "gem5", "sniper"):
            m = c[f"{tool}_margin_pp"]
            if m == "":
                continue
            # Must parse as float with 3dp formatting
            assert "." in m
            assert len(m.split(".")[-1]) == 3


def test_cells_sorted_by_graph_app():
    art = _art()
    keys = [(c["graph"], c["app"]) for c in art["cells"]]
    assert keys == sorted(keys)


def test_l3_by_tool_is_largest_per_cell():
    """l3 picked per tool is the LARGEST L3 size with data."""
    art = _art()
    lf = _max_l3(_read_lit())
    g5 = _max_l3(_read_anchor(GEM5))
    sn = _max_l3(_read_anchor(SNIPER))
    for c in art["cells"]:
        k = (c["graph"], c["app"])
        if c["cache_sim_winner"]:
            assert c["cache_sim_l3"] == lf[k][0]
        if c["gem5_winner"]:
            assert c["gem5_l3"] == g5[k][0]
        if c["sniper_winner"]:
            assert c["sniper_l3"] == sn[k][0]


# Group C — classification logic parity --------------------------------

def test_unanimous_iff_all_nonempty_winners_equal():
    art = _art()
    for c in art["cells"]:
        winners = [c.get(f"{t}_winner") for t in ("cache_sim", "gem5", "sniper")
                   if c.get(f"{t}_winner")]
        if c["classification"] == "unanimous":
            assert len(set(winners)) == 1, c


def test_split_iff_all_nonempty_winners_distinct():
    art = _art()
    for c in art["cells"]:
        winners = [c.get(f"{t}_winner") for t in ("cache_sim", "gem5", "sniper")
                   if c.get(f"{t}_winner")]
        if c["classification"] == "split":
            assert len(set(winners)) == len(winners), c


def test_majority_iff_max_winner_count_at_least_two_but_not_all():
    art = _art()
    for c in art["cells"]:
        winners = [c.get(f"{t}_winner") for t in ("cache_sim", "gem5", "sniper")
                   if c.get(f"{t}_winner")]
        if c["classification"] == "majority":
            assert max(Counter(winners).values()) >= 2
            assert max(Counter(winners).values()) < len(winners)


# Group D — by_classification + split/majority list parity --------------

def test_by_classification_counts_match_cells():
    art = _art()
    expected = Counter(c["classification"] for c in art["cells"])
    assert dict(art["summary"]["by_classification"]) == dict(expected.most_common())


def test_split_cells_match_split_classification():
    art = _art()
    expected = {(c["graph"], c["app"]) for c in art["cells"]
                if c["classification"] == "split"}
    actual = {(c["graph"], c["app"]) for c in art["summary"]["split_cells"]}
    assert actual == expected


def test_majority_cells_match_majority_classification():
    art = _art()
    expected = {(c["graph"], c["app"]) for c in art["cells"]
                if c["classification"] == "majority"}
    actual = {(c["graph"], c["app"]) for c in art["summary"]["majority_cells"]}
    assert actual == expected


def test_split_and_majority_lists_disjoint():
    art = _art()
    split = {(c["graph"], c["app"]) for c in art["summary"]["split_cells"]}
    maj = {(c["graph"], c["app"]) for c in art["summary"]["majority_cells"]}
    assert split & maj == set()


def test_classification_subset_of_known_labels():
    art = _art()
    labels = set(art["summary"]["by_classification"].keys())
    assert labels <= {"unanimous", "majority", "split", "missing"}


# Group E — projection invariants -------------------------------------

def test_split_list_entries_only_have_three_fields_plus_keys():
    """split_cells / majority_cells contain only graph, app, and 3 tool winners."""
    art = _art()
    expected = {"graph", "app", "cache_sim_winner", "gem5_winner", "sniper_winner"}
    for r in art["summary"]["split_cells"] + art["summary"]["majority_cells"]:
        assert set(r.keys()) == expected, r


def test_at_most_three_known_tools_per_cell():
    art = _art()
    for c in art["cells"]:
        winners_set = {c.get(f"{t}_winner") for t in ("cache_sim", "gem5", "sniper")}
        winners_set.discard("")
        assert len(winners_set) <= 3
