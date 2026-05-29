#!/usr/bin/env python3
"""POPT-vs-GRASP delta report.

Why this exists
---------------
The policy-winner table tells us GRASP wins ~51 % of cells and POPT
~38 %, but it does *not* answer the central paper question: **when
does POPT actually improve on GRASP, and by how much?** This script
projects the lit-faith CSV onto a per-cell ``Δ(POPT − GRASP)`` view
(negative = POPT better; positive = POPT worse) and breaks the
distribution down by graph family and L3 regime — the two axes that
matter for the paper's claim that POPT's offline permutation pays off
on hub-heavy graphs at near-saturation but is wasted on diffuse-
locality regimes.

Output
------
* ``wiki/data/popt_vs_grasp_delta.csv`` — one row per (graph, app,
  l3_size) with the GRASP miss rate, POPT miss rate, the delta in pp,
  and a coarse classification.
* ``wiki/data/popt_vs_grasp_delta.json`` — machine-readable summary
  with per-family / per-regime / per-app statistics, plus the
  POPT-wins and GRASP-wins extreme tails.
* ``wiki/data/popt_vs_grasp_delta.md`` — paper-ready markdown.

Usage
-----
    python3 -m scripts.experiments.ecg.popt_vs_grasp_report

The defaults read ``wiki/data/literature_faithfulness_postfix.csv``
and ``wiki/data/corpus_diversity.json`` and write all three artifacts
into ``wiki/data/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]

L3_SIZE_BYTES = {
    "4kB": 4 * 1024,
    "16kB": 16 * 1024,
    "64kB": 64 * 1024,
    "256kB": 256 * 1024,
    "1MB": 1024 * 1024,
    "2MB": 2 * 1024 * 1024,
    "4MB": 4 * 1024 * 1024,
    "8MB": 8 * 1024 * 1024,
}

GRAPH_FAMILY: dict[str, str] = {
    "email-Eu-core": "social",
    "web-Google": "web",
    "cit-Patents": "citation",
    "soc-pokec": "social",
    "soc-LiveJournal1": "social",
    "com-orkut": "social",
    "roadNet-CA": "road",
    "delaunay_n19": "mesh",
}

CLASS_FLOOR_PP = 0.5


def _l3_bytes(label: str) -> int:
    return L3_SIZE_BYTES.get(label, -1)


def _l3_regime(label: str) -> str:
    b = _l3_bytes(label)
    if b < 0:
        return "unknown"
    if b < 64 * 1024:
        return "tiny"
    if b < 1024 * 1024:
        return "small"
    return "large"


def _classify(delta_pp: float) -> str:
    """delta = POPT - GRASP (so negative = POPT helps)."""
    if math.isnan(delta_pp):
        return "missing"
    if delta_pp < -CLASS_FLOOR_PP:
        return "popt_better"
    if delta_pp > CLASS_FLOOR_PP:
        return "grasp_better"
    return "tie"


def _load_corpus(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        graphs = data
    elif isinstance(data, dict):
        graphs = data.get("graphs", []) or []
    else:
        graphs = []
    by_name: dict[str, dict] = {}
    for c in graphs:
        g = c.get("graph") or c.get("name")
        if g:
            by_name[g] = c
    return by_name


def _graph_family(graph: str, corpus_entry: dict | None) -> str:
    if corpus_entry:
        for key in ("family", "category"):
            v = corpus_entry.get(key)
            if v:
                return v
    return GRAPH_FAMILY.get(graph, "unknown")


def _read_lit_faith(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _delta_rows(rows: Iterable[dict], corpus: dict[str, dict]) -> list[dict]:
    """One row per (graph, app, l3_size) with the POPT−GRASP delta in
    percentage points. Cells that lack EITHER GRASP or POPT are
    skipped — they cannot contribute to a head-to-head comparison.
    """
    cells: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        graph = r.get("graph") or ""
        app = r.get("app") or r.get("benchmark") or ""
        l3 = r.get("l3_size") or ""
        pol = (r.get("policy") or "").strip()
        try:
            mr = float(r.get("miss_rate") or r.get("l3_miss_rate") or "nan")
        except ValueError:
            continue
        if not math.isfinite(mr):
            continue
        if pol not in {"GRASP", "POPT"}:
            continue
        cells[(graph, app, l3)][pol] = mr

    out: list[dict] = []
    for (graph, app, l3), by_pol in sorted(cells.items(), key=lambda kv: (
        kv[0][0], kv[0][1], _l3_bytes(kv[0][2]), kv[0][2],
    )):
        grasp_mr = by_pol.get("GRASP")
        popt_mr = by_pol.get("POPT")
        if grasp_mr is None or popt_mr is None:
            continue
        delta_pp = (popt_mr - grasp_mr) * 100.0
        out.append({
            "graph": graph,
            "graph_family": _graph_family(graph, corpus.get(graph)),
            "app": app,
            "l3_size": l3,
            "l3_regime": _l3_regime(l3),
            "grasp_miss_rate": f"{grasp_mr:.6f}",
            "popt_miss_rate": f"{popt_mr:.6f}",
            "delta_pp": f"{delta_pp:.3f}",
            "classification": _classify(delta_pp),
        })
    return out


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean_pp": round(statistics.fmean(values), 3),
        "median_pp": round(statistics.median(values), 3),
        "min_pp": round(min(values), 3),
        "max_pp": round(max(values), 3),
        "stdev_pp": (
            round(statistics.pstdev(values), 3) if len(values) >= 2 else 0.0
        ),
    }


def _summarize(records: list[dict]) -> dict:
    """Bucket deltas by (family, regime, app) and report standard stats.
    Also surface the top-5 POPT-wins and top-5 GRASP-wins.
    """
    by_family: dict[str, list[float]] = defaultdict(list)
    by_regime: dict[str, list[float]] = defaultdict(list)
    by_app: dict[str, list[float]] = defaultdict(list)
    by_family_regime: dict[tuple[str, str], list[float]] = defaultdict(list)
    overall: list[float] = []
    class_counts: dict[str, int] = defaultdict(int)
    rows_with_delta: list[tuple[float, dict]] = []
    for r in records:
        try:
            d = float(r["delta_pp"])
        except (TypeError, ValueError):
            continue
        rows_with_delta.append((d, r))
        overall.append(d)
        by_family[r["graph_family"]].append(d)
        by_regime[r["l3_regime"]].append(d)
        by_app[r["app"]].append(d)
        by_family_regime[(r["graph_family"], r["l3_regime"])].append(d)
        class_counts[r["classification"]] += 1

    # POPT helps most where delta is most negative; GRASP helps most
    # where delta is most positive.
    rows_sorted_neg = sorted(rows_with_delta, key=lambda x: x[0])
    rows_sorted_pos = sorted(rows_with_delta, key=lambda x: -x[0])

    def _slim(r: dict) -> dict:
        return {k: r[k] for k in (
            "graph", "graph_family", "app", "l3_size", "l3_regime",
            "grasp_miss_rate", "popt_miss_rate", "delta_pp",
            "classification",
        )}

    return {
        "n_cells": len(records),
        "overall": _stats(overall),
        "classification_counts": dict(sorted(class_counts.items())),
        "by_family": {
            fam: _stats(v) for fam, v in sorted(by_family.items())
        },
        "by_regime": {
            reg: _stats(v) for reg, v in sorted(by_regime.items())
        },
        "by_app": {
            app: _stats(v) for app, v in sorted(by_app.items())
        },
        "by_family_regime": {
            f"{fam}|{reg}": _stats(v)
            for (fam, reg), v in sorted(by_family_regime.items())
        },
        "popt_top5_helps": [_slim(r) for _, r in rows_sorted_neg[:5]],
        "grasp_top5_helps": [_slim(r) for _, r in rows_sorted_pos[:5]],
    }


def _write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def _write_json(summary: dict, records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "summary": summary,
        "cells": records,
    }, indent=2, sort_keys=True))


def _md_stats_row(label: str, st: dict) -> str:
    if not st or st.get("n", 0) == 0:
        return f"| {label} | 0 | — | — | — | — | — |"
    return (
        f"| {label} | {st['n']} | {st['mean_pp']:+.3f} | "
        f"{st['median_pp']:+.3f} | {st['min_pp']:+.3f} | "
        f"{st['max_pp']:+.3f} | {st['stdev_pp']:.3f} |"
    )


def _write_md(summary: dict, records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# POPT vs GRASP delta")
    lines.append("")
    lines.append(
        "_Generated by `scripts/experiments/ecg/popt_vs_grasp_report.py` "
        "from `wiki/data/literature_faithfulness_postfix.csv`._"
    )
    lines.append("")
    lines.append(
        "**Sign convention:** `delta_pp = miss_rate(POPT) − miss_rate(GRASP)`, "
        "expressed in percentage points. Negative = POPT improves on GRASP; "
        "positive = GRASP wins. Cells within ±0.5 pp are classified `tie`."
    )
    lines.append("")
    n = summary["n_cells"]
    cc = summary["classification_counts"]
    pb = cc.get("popt_better", 0)
    gb = cc.get("grasp_better", 0)
    tie = cc.get("tie", 0)
    lines.append(
        f"**Headline:** {n} (graph, app, L3) cells have both GRASP and "
        f"POPT data. POPT strictly improves on GRASP in {pb} "
        f"({pb/max(1,n)*100:.1f} %), GRASP wins {gb} ({gb/max(1,n)*100:.1f} %), "
        f"and {tie} cells are within the 0.5 pp tie band."
    )
    lines.append("")

    st = summary["overall"]
    lines.append(
        f"Overall delta: mean {st['mean_pp']:+.3f} pp, median "
        f"{st['median_pp']:+.3f} pp, range [{st['min_pp']:+.3f}, "
        f"{st['max_pp']:+.3f}] pp."
    )
    lines.append("")

    lines.append("## Delta by graph family")
    lines.append("")
    lines.append("| family | n | mean | median | min | max | stdev |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, st in summary["by_family"].items():
        lines.append(_md_stats_row(fam, st))
    lines.append("")

    lines.append("## Delta by L3 regime")
    lines.append("")
    lines.append("| regime | n | mean | median | min | max | stdev |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for reg in ("tiny", "small", "large", "unknown"):
        st = summary["by_regime"].get(reg)
        if st and st.get("n", 0) > 0:
            lines.append(_md_stats_row(reg, st))
    lines.append("")

    lines.append("## Delta by application")
    lines.append("")
    lines.append("| app | n | mean | median | min | max | stdev |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for app, st in summary["by_app"].items():
        lines.append(_md_stats_row(app, st))
    lines.append("")

    lines.append("## Delta by (family, regime) intersection")
    lines.append("")
    lines.append("| family | regime | n | mean | median | min | max | stdev |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for key, st in summary["by_family_regime"].items():
        fam, reg = key.split("|", 1)
        if st.get("n", 0) == 0:
            continue
        lines.append(
            f"| {fam} | {reg} | {st['n']} | {st['mean_pp']:+.3f} | "
            f"{st['median_pp']:+.3f} | {st['min_pp']:+.3f} | "
            f"{st['max_pp']:+.3f} | {st['stdev_pp']:.3f} |"
        )
    lines.append("")

    lines.append("## Top 5 cells where POPT helps the most")
    lines.append("")
    lines.append(
        "| graph | family | app | L3 | regime | GRASP | POPT | delta pp |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for r in summary["popt_top5_helps"]:
        lines.append(
            f"| {r['graph']} | {r['graph_family']} | {r['app']} | "
            f"{r['l3_size']} | {r['l3_regime']} | {r['grasp_miss_rate']} | "
            f"{r['popt_miss_rate']} | {r['delta_pp']} |"
        )
    lines.append("")

    lines.append("## Top 5 cells where GRASP helps the most")
    lines.append("")
    lines.append(
        "| graph | family | app | L3 | regime | GRASP | POPT | delta pp |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for r in summary["grasp_top5_helps"]:
        lines.append(
            f"| {r['graph']} | {r['graph_family']} | {r['app']} | "
            f"{r['l3_size']} | {r['l3_regime']} | {r['grasp_miss_rate']} | "
            f"{r['popt_miss_rate']} | {r['delta_pp']} |"
        )
    lines.append("")

    lines.append("## Per-cell deltas (full table)")
    lines.append("")
    lines.append(
        "| graph | family | app | L3 | regime | GRASP | POPT | "
        "delta pp | class |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---|")
    for r in records:
        lines.append(
            f"| {r['graph']} | {r['graph_family']} | {r['app']} | "
            f"{r['l3_size']} | {r['l3_regime']} | {r['grasp_miss_rate']} | "
            f"{r['popt_miss_rate']} | {r['delta_pp']} | "
            f"{r['classification']} |"
        )
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lit-faith-csv",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv",
    )
    parser.add_argument(
        "--corpus-json",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "corpus_diversity.json",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "popt_vs_grasp_delta.csv",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "popt_vs_grasp_delta.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "wiki" / "data" / "popt_vs_grasp_delta.md",
    )
    args = parser.parse_args()

    if not args.lit_faith_csv.exists():
        raise SystemExit(
            f"missing lit-faith CSV at {args.lit_faith_csv}; "
            "run `make lit-faith` first."
        )

    rows = _read_lit_faith(args.lit_faith_csv)
    corpus = _load_corpus(args.corpus_json)
    records = _delta_rows(rows, corpus)
    summary = _summarize(records)
    _write_csv(records, args.csv_out)
    _write_json(summary, records, args.json_out)
    _write_md(summary, records, args.md_out)
    cc = summary["classification_counts"]
    print(
        f"[popt-vs-grasp] {summary['n_cells']} cells; "
        f"POPT better={cc.get('popt_better', 0)}, "
        f"GRASP better={cc.get('grasp_better', 0)}, "
        f"tie={cc.get('tie', 0)}; "
        f"overall mean Δ={summary['overall'].get('mean_pp', 0):+.3f} pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
