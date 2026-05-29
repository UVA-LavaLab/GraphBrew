#!/usr/bin/env python3
"""Cross-tool saturation soundness report.

Why this exists
---------------
The three simulators in this project (``cache_sim``, ``gem5``, and
``Sniper``) all measure the same kernel/graph cells but sweep slightly
different L3 sizes. The lit-faith CSV sweeps ``cache_sim`` from 1 MB
upward; the gem5/Sniper anchors top out at 2 MB. A direct
"do the winners agree" parity check is therefore impossible — each
tool's largest L3 sits in a different region of the L-curve.

The most we can demand cross-tool is **saturation consistency**:

* For every (graph, app) cell present in BOTH the lit-faith CSV and at
  least one anchor, take each tool's *largest* L3.
* Compute the spread = ``max(LRU,SRRIP,GRASP) − min(LRU,SRRIP,GRASP)``
  at that L3 (in percentage points).
* When a cell is "doubly saturated" (spread < ``--sat-floor`` in BOTH
  tools) we assert that the two tools agree on direction within a
  ``--headline-tol`` of pp on ``GRASP − LRU``.
* When a cell is saturated in exactly one tool (typically gem5/Sniper
  saturates first because their max L3 is smaller for small graphs),
  we record but do not assert — these are L-curve-regime artifacts.

The output is a paper-grade artifact answering reviewer questions like
"do gem5 and Sniper say the same thing as cache_sim about email-Eu-core
at saturation?".

Output
------
* ``wiki/data/cross_tool_saturation.csv``  — one row per overlapping cell.
* ``wiki/data/cross_tool_saturation.json`` — summary + per-cell records.
* ``wiki/data/cross_tool_saturation.md``   — paper-ready markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

DEFAULT_LIT_FAITH = WIKI_DATA / "literature_faithfulness_postfix.csv"
DEFAULT_GEM5 = WIKI_DATA / "gem5_anchor.json"
DEFAULT_SNIPER = WIKI_DATA / "sniper_anchor.json"
DEFAULT_CSV_OUT = WIKI_DATA / "cross_tool_saturation.csv"
DEFAULT_JSON_OUT = WIKI_DATA / "cross_tool_saturation.json"
DEFAULT_MD_OUT = WIKI_DATA / "cross_tool_saturation.md"

L3_SIZE_BYTES = {
    "4kB": 4 * 1024, "8kB": 8 * 1024, "16kB": 16 * 1024, "32kB": 32 * 1024,
    "64kB": 64 * 1024, "128kB": 128 * 1024, "256kB": 256 * 1024,
    "512kB": 512 * 1024, "1MB": 1024 * 1024, "2MB": 2 * 1024 * 1024,
    "4MB": 4 * 1024 * 1024, "8MB": 8 * 1024 * 1024,
}


def _l3_bytes(label: str) -> int:
    return L3_SIZE_BYTES.get(label, -1)


def _load_cache_sim(path: Path) -> dict:
    """(graph, app) -> {l3_size -> {policy -> miss_rate}}"""
    by_cell: dict = defaultdict(lambda: defaultdict(dict))
    if not path.exists():
        return by_cell
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                m = float(r["miss_rate"])
            except (KeyError, ValueError, TypeError):
                continue
            by_cell[(r["graph"], r["app"])][r["l3_size"]][r["policy"]] = m
    return by_cell


def _load_anchor(path: Path) -> dict:
    """(graph, app) -> {l3_size -> {policy -> miss_rate}}"""
    by_cell: dict = defaultdict(lambda: defaultdict(dict))
    if not path.exists():
        return by_cell
    d = json.loads(path.read_text())
    for c in d.get("cells", []):
        m = c.get("miss_rate_by_policy") or {}
        if not m:
            continue
        by_cell[(c["graph"], c["app"])][c["l3_size"]] = dict(m)
    return by_cell


def _largest_l3(cells: dict) -> str | None:
    if not cells:
        return None
    return max(cells.keys(), key=lambda x: _l3_bytes(x))


def _spread_pp(pols: dict, restrict: set[str] | None = None) -> float | None:
    if restrict:
        pols = {p: v for p, v in pols.items() if p in restrict}
    if len(pols) < 2:
        return None
    return (max(pols.values()) - min(pols.values())) * 100.0


def _grasp_minus_lru_pp(pols: dict) -> float | None:
    g, l = pols.get("GRASP"), pols.get("LRU")
    if g is None or l is None:
        return None
    return (g - l) * 100.0


def _build_cells(cs: dict, gem: dict, snp: dict) -> list[dict]:
    cells: list[dict] = []
    keys = sorted(set(cs.keys()) & (set(gem.keys()) | set(snp.keys())))
    for k in keys:
        cs_l3 = _largest_l3(cs[k])
        cs_pols = cs[k].get(cs_l3, {}) if cs_l3 else {}
        cs_spread = _spread_pp(cs_pols, {"LRU", "SRRIP", "GRASP"})
        cs_diff = _grasp_minus_lru_pp(cs_pols)

        for tool, anchor in (("gem5", gem), ("sniper", snp)):
            if k not in anchor:
                continue
            a_l3 = _largest_l3(anchor[k])
            a_pols = anchor[k].get(a_l3, {}) if a_l3 else {}
            a_spread = _spread_pp(a_pols, {"LRU", "SRRIP", "GRASP"})
            a_diff = _grasp_minus_lru_pp(a_pols)
            cells.append(
                {
                    "graph": k[0],
                    "app": k[1],
                    "tool": tool,
                    "cache_sim_l3": cs_l3,
                    "anchor_l3": a_l3,
                    "cache_sim_spread_pp": cs_spread,
                    "anchor_spread_pp": a_spread,
                    "cache_sim_grasp_minus_lru_pp": cs_diff,
                    "anchor_grasp_minus_lru_pp": a_diff,
                }
            )
    return cells


def _classify(c: dict, sat_floor: float, headline_tol: float) -> dict:
    cs_s = c["cache_sim_spread_pp"]
    a_s = c["anchor_spread_pp"]
    cs_d = c["cache_sim_grasp_minus_lru_pp"]
    a_d = c["anchor_grasp_minus_lru_pp"]
    if cs_s is None or a_s is None:
        return {"regime": "incomplete", "agree": None, "delta_pp": None}
    cs_sat = cs_s < sat_floor
    a_sat = a_s < sat_floor
    if cs_sat and a_sat:
        regime = "doubly_saturated"
    elif cs_sat or a_sat:
        regime = "single_saturated"
    else:
        regime = "neither_saturated"
    delta = None
    agree: bool | None = None
    if cs_d is not None and a_d is not None:
        delta = abs(cs_d - a_d)
        agree = delta <= headline_tol
    return {"regime": regime, "agree": agree, "delta_pp": delta}


def _emit_csv(cells: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "graph", "app", "tool", "cache_sim_l3", "anchor_l3",
        "cache_sim_spread_pp", "anchor_spread_pp",
        "cache_sim_grasp_minus_lru_pp", "anchor_grasp_minus_lru_pp",
        "regime", "agree", "delta_pp",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in cells:
            row = {k: c.get(k, "") for k in cols}
            for fld in (
                "cache_sim_spread_pp", "anchor_spread_pp",
                "cache_sim_grasp_minus_lru_pp", "anchor_grasp_minus_lru_pp",
                "delta_pp",
            ):
                v = row[fld]
                row[fld] = f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)
            w.writerow(row)


def _emit_json(cells: list[dict], summary: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "summary": summary,
        "cells": cells,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _emit_md(cells: list[dict], summary: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Cross-tool saturation soundness\n")
    lines.append(
        "Generated by "
        "`scripts/experiments/ecg/cross_tool_saturation_report.py`. "
        "Each row pairs a cache_sim lit-faith cell with the matching "
        "gem5 or Sniper anchor cell, picks each tool's largest L3, and "
        "reports the LRU/SRRIP/GRASP spread.\n\n"
    )
    lines.append(
        f"* Sat. floor: **{summary['sat_floor_pp']:.2f} pp** "
        f"(below = considered saturated)\n"
        f"* Headline tolerance: **{summary['headline_tol_pp']:.2f} pp** "
        f"(GRASP−LRU delta must agree within this)\n"
        f"* Overlapping cells: **{summary['n_cells']}** "
        f"({summary['regime_counts']})\n\n"
    )
    if summary["disagreements"]:
        lines.append("## ⚠ Disagreements\n\n")
        lines.append("| graph | app | tool | regime | Δ |\n")
        lines.append("|---|---|---|---|---:|\n")
        for d in summary["disagreements"]:
            lines.append(
                f"| {d['graph']} | {d['app']} | {d['tool']} | "
                f"{d['regime']} | {d['delta_pp']:.3f} |\n"
            )
        lines.append("\n")
    else:
        lines.append("## ✅ No saturated disagreements\n\n")

    lines.append("## All overlapping cells\n\n")
    lines.append(
        "| graph | app | tool | cs L3 | cs spread | anchor L3 | "
        "anchor spread | cs GRASP−LRU | anchor GRASP−LRU | regime |\n"
    )
    lines.append("|---|---|---|---|---:|---|---:|---:|---:|---|\n")
    for c in cells:
        def fmt(v): return "—" if v is None else f"{v:.3f}"
        lines.append(
            f"| {c['graph']} | {c['app']} | {c['tool']} | "
            f"{c['cache_sim_l3']} | {fmt(c['cache_sim_spread_pp'])} | "
            f"{c['anchor_l3']} | {fmt(c['anchor_spread_pp'])} | "
            f"{fmt(c['cache_sim_grasp_minus_lru_pp'])} | "
            f"{fmt(c['anchor_grasp_minus_lru_pp'])} | "
            f"{c['regime']} |\n"
        )
    out.write_text("".join(lines))


def _summarise(cells: list[dict], sat_floor: float, headline_tol: float) -> dict:
    regimes: dict = defaultdict(int)
    disagreements: list[dict] = []
    agreed = 0
    n_doubly = 0
    for c in cells:
        cls = _classify(c, sat_floor, headline_tol)
        c.update(cls)
        regimes[cls["regime"]] += 1
        if cls["regime"] == "doubly_saturated":
            n_doubly += 1
            if cls["agree"] is True:
                agreed += 1
            elif cls["agree"] is False:
                disagreements.append({
                    "graph": c["graph"], "app": c["app"], "tool": c["tool"],
                    "regime": cls["regime"], "delta_pp": cls["delta_pp"],
                })
    return {
        "sat_floor_pp": sat_floor,
        "headline_tol_pp": headline_tol,
        "n_cells": len(cells),
        "regime_counts": dict(regimes),
        "doubly_saturated_agree": agreed,
        "doubly_saturated_total": n_doubly,
        "disagreements": disagreements,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lit-faith-csv", default=str(DEFAULT_LIT_FAITH))
    p.add_argument("--gem5-anchor", default=str(DEFAULT_GEM5))
    p.add_argument("--sniper-anchor", default=str(DEFAULT_SNIPER))
    p.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT))
    p.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    p.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    p.add_argument("--sat-floor", type=float, default=1.0,
                   help="Spread (pp) below which a cell is considered saturated.")
    p.add_argument("--headline-tol", type=float, default=2.0,
                   help="Allowed |Δ(GRASP−LRU)| in pp between tools "
                        "for doubly-saturated cells to agree.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    cs = _load_cache_sim(Path(ns.lit_faith_csv))
    gem = _load_anchor(Path(ns.gem5_anchor))
    snp = _load_anchor(Path(ns.sniper_anchor))
    cells = _build_cells(cs, gem, snp)
    summary = _summarise(cells, ns.sat_floor, ns.headline_tol)
    _emit_csv(cells, Path(ns.csv_out))
    _emit_json(cells, summary, Path(ns.json_out))
    _emit_md(cells, summary, Path(ns.md_out))
    print(
        f"[cross-tool-sat] cells={summary['n_cells']} "
        f"doubly_saturated={summary['doubly_saturated_total']} "
        f"agreed={summary['doubly_saturated_agree']} "
        f"disagreements={len(summary['disagreements'])} → "
        f"{Path(ns.md_out).relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
