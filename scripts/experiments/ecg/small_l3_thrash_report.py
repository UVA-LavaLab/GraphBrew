#!/usr/bin/env python3
"""Build a paper-grade *Small-L3 Thrash Report* from the standalone
``final_cache_sim`` sweep output.

Why this exists
---------------
The main lit-faith CSV sweeps L3 from 1 MB upward — that's the "headline"
regime where GRASP/POPT win by design. The paper also needs to show what
happens *below* the working-set knee, where the L3 cannot hold even the
hot region. The standalone sweep emitted to
``results/ecg_experiments/paper_pipeline/<ts>/final_cache_sim/`` covers
9 (graph, app) cells at a single tiny 4 kB L3 with **9 policy variants**
that the main sweep never exercises:

* LRU, SRRIP, GRASP, POPT, POPT_CHARGED (overhead-charged POPT)
* ECG_DBG_ONLY, ECG_DBG_PRIMARY, ECG_DBG_PRIMARY_CHARGED, ECG_POPT_PRIMARY

This script reads one or more ``combined_roi_matrix.csv`` files from that
stage and emits a focused report answering:

* For each (graph, app), which policy wins at the tiny L3?
* How often does LRU beat GRASP/POPT in this regime? (Reviewers ask.)
* Do ECG variants recover the GRASP/POPT loss when L3 is tiny?
* Per-policy win counts + mean / median / worst miss rate.

Output (paths are configurable; defaults under ``wiki/data/``):

* ``small_l3_thrash.csv``  — one row per (graph, app, policy_label).
* ``small_l3_thrash.json`` — per-cell winner + per-policy aggregates.
* ``small_l3_thrash.md``   — paper-ready markdown with tables.

Usage
-----
    python3 -m scripts.experiments.ecg.small_l3_thrash_report

    # Or with an explicit input glob:
    python3 -m scripts.experiments.ecg.small_l3_thrash_report \\
        --input-glob 'results/ecg_experiments/paper_pipeline/*/final_cache_sim/combined_roi_matrix.csv'
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_GLOB = (
    "results/ecg_experiments/paper_pipeline/*/final_cache_sim/combined_roi_matrix.csv"
)
DEFAULT_CSV_OUT = REPO_ROOT / "wiki" / "data" / "small_l3_thrash.csv"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "small_l3_thrash.json"
DEFAULT_MD_OUT = REPO_ROOT / "wiki" / "data" / "small_l3_thrash.md"

POLICY_LABEL_ORDER = [
    "LRU",
    "SRRIP",
    "GRASP",
    "POPT",
    "POPT_CHARGED",
    "ECG_DBG_ONLY",
    "ECG_DBG_PRIMARY",
    "ECG_DBG_PRIMARY_CHARGED",
    "ECG_POPT_PRIMARY",
]


def _load_rows(input_glob: str) -> list[dict]:
    files = sorted(glob.glob(str(REPO_ROOT / input_glob)))
    if not files:
        raise SystemExit(
            f"[small-l3-thrash] no input CSVs match {input_glob} "
            f"(searched under {REPO_ROOT}). Run the final_cache_sim "
            f"stage of paper_pipeline.py first."
        )
    rows: list[dict] = []
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("l3_miss_rate") in (None, ""):
                    continue
                rows.append(
                    {
                        "source": str(Path(f).relative_to(REPO_ROOT)),
                        "graph": r["final_graph"],
                        "app": r["benchmark"],
                        "policy": r["policy"],
                        "policy_label": r["policy_label"] or r["policy"],
                        "l3_size": r["l3_size"],
                        "l3_ways": r["l3_ways"],
                        "l3_miss_rate": float(r["l3_miss_rate"]),
                        "ecg_mode": r.get("ecg_mode", ""),
                    }
                )
    return rows


def _winner(cell_rows: list[dict]) -> tuple[dict, dict, float]:
    """Return (winner, runner_up, margin_pp) for one (graph,app,l3) cell.

    Ties (within 1e-9 absolute) are broken by ``POLICY_LABEL_ORDER`` so
    the report is deterministic.
    """
    def _key(r: dict) -> tuple[float, int]:
        try:
            idx = POLICY_LABEL_ORDER.index(r["policy_label"])
        except ValueError:
            idx = len(POLICY_LABEL_ORDER)
        return (r["l3_miss_rate"], idx)

    ordered = sorted(cell_rows, key=_key)
    w = ordered[0]
    r = ordered[1] if len(ordered) > 1 else ordered[0]
    return w, r, (r["l3_miss_rate"] - w["l3_miss_rate"]) * 100.0


def _aggregate(rows: list[dict]) -> dict:
    """Compute per-policy_label aggregates and per-cell winners."""
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["graph"], r["app"], r["l3_size"])].append(r)

    winners: list[dict] = []
    win_counts: Counter[str] = Counter()
    for (g, a, l), cell in sorted(by_cell.items()):
        w, ru, margin = _winner(cell)
        winners.append(
            {
                "graph": g,
                "app": a,
                "l3_size": l,
                "winner": w["policy_label"],
                "winner_miss_rate": w["l3_miss_rate"],
                "runner_up": ru["policy_label"],
                "runner_up_miss_rate": ru["l3_miss_rate"],
                "margin_pp": margin,
            }
        )
        win_counts[w["policy_label"]] += 1

    by_policy: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_policy[r["policy_label"]].append(r["l3_miss_rate"])

    policy_stats: dict[str, dict] = {}
    for label in POLICY_LABEL_ORDER:
        vals = by_policy.get(label, [])
        if not vals:
            continue
        policy_stats[label] = {
            "n_cells": len(vals),
            "mean_miss_rate": statistics.mean(vals),
            "median_miss_rate": statistics.median(vals),
            "min_miss_rate": min(vals),
            "max_miss_rate": max(vals),
            "wins": win_counts.get(label, 0),
        }

    # LRU-vs-GRASP and LRU-vs-POPT showdown table.
    showdown: list[dict] = []
    for (g, a, l), cell in sorted(by_cell.items()):
        by_lbl = {r["policy_label"]: r["l3_miss_rate"] for r in cell}
        lru = by_lbl.get("LRU")
        grasp = by_lbl.get("GRASP")
        popt = by_lbl.get("POPT")
        if lru is None:
            continue
        showdown.append(
            {
                "graph": g,
                "app": a,
                "l3_size": l,
                "lru_miss": lru,
                "grasp_miss": grasp,
                "popt_miss": popt,
                "lru_minus_grasp_pp": (grasp - lru) * 100.0 if grasp is not None else None,
                "lru_minus_popt_pp": (popt - lru) * 100.0 if popt is not None else None,
            }
        )

    return {
        "n_cells": len(by_cell),
        "n_rows": len(rows),
        "winners": winners,
        "win_counts": dict(win_counts),
        "policy_stats": policy_stats,
        "showdown": showdown,
    }


def _emit_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "graph", "app", "l3_size", "l3_ways",
        "policy", "policy_label", "ecg_mode", "l3_miss_rate", "source",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(
            rows,
            key=lambda x: (x["graph"], x["app"], x["l3_size"], x["policy_label"]),
        ):
            w.writerow({k: r.get(k, "") for k in cols})


def _emit_json(rows: list[dict], agg: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "sources": sorted({r["source"] for r in rows}),
        "summary": {
            "n_cells": agg["n_cells"],
            "n_rows": agg["n_rows"],
            "win_counts": agg["win_counts"],
            "policy_stats": agg["policy_stats"],
        },
        "cells": agg["winners"],
        "showdown": agg["showdown"],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _emit_md(rows: list[dict], agg: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Small-L3 thrash report\n")
    lines.append(
        "Generated by "
        "`scripts/experiments/ecg/small_l3_thrash_report.py` from the "
        "standalone `final_cache_sim` sweep. This regime (4 kB L3, "
        "tiny L1/L2) intentionally overflows even the hot working set "
        "so we can see how policies behave **below** the saturation "
        "knee that drives the L-curve figures.\n"
    )
    lines.append(f"* Cells: **{agg['n_cells']}**\n")
    lines.append(f"* Rows: **{agg['n_rows']}**\n")
    lines.append(f"* Policy variants: **{len(agg['policy_stats'])}**\n\n")

    lines.append("## Per-cell winner\n\n")
    lines.append("| graph | app | L3 | winner | miss | runner-up | runner-up miss | margin (pp) |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for c in agg["winners"]:
        lines.append(
            f"| {c['graph']} | {c['app']} | {c['l3_size']} | "
            f"`{c['winner']}` | {c['winner_miss_rate']:.4f} | "
            f"`{c['runner_up']}` | {c['runner_up_miss_rate']:.4f} | "
            f"{c['margin_pp']:+.3f} |\n"
        )

    lines.append("\n## Per-policy stats\n\n")
    lines.append("| policy | cells | wins | mean miss | median | min | max |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for label in POLICY_LABEL_ORDER:
        st = agg["policy_stats"].get(label)
        if not st:
            continue
        lines.append(
            f"| `{label}` | {st['n_cells']} | {st['wins']} | "
            f"{st['mean_miss_rate']:.4f} | {st['median_miss_rate']:.4f} | "
            f"{st['min_miss_rate']:.4f} | {st['max_miss_rate']:.4f} |\n"
        )

    lines.append("\n## LRU vs GRASP / POPT showdown (Δ in pp)\n\n")
    lines.append(
        "Positive Δ means the named policy is **worse than LRU at the "
        "tiny L3** — i.e., the pinning strategy thrashes when even the "
        "hot region overflows.\n\n"
    )
    lines.append("| graph | app | L3 | LRU | GRASP | Δ vs LRU | POPT | Δ vs LRU |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|\n")
    for s in agg["showdown"]:
        gd = "—" if s["grasp_miss"] is None else f"{s['grasp_miss']:.4f}"
        pd = "—" if s["popt_miss"] is None else f"{s['popt_miss']:.4f}"
        gdelta = "—" if s["lru_minus_grasp_pp"] is None else f"{s['lru_minus_grasp_pp']:+.3f}"
        pdelta = "—" if s["lru_minus_popt_pp"] is None else f"{s['lru_minus_popt_pp']:+.3f}"
        lines.append(
            f"| {s['graph']} | {s['app']} | {s['l3_size']} | "
            f"{s['lru_miss']:.4f} | {gd} | {gdelta} | {pd} | {pdelta} |\n"
        )

    out.write_text("".join(lines))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help=f"Glob (relative to repo root) for combined_roi_matrix.csv files. "
        f"Default: {DEFAULT_INPUT_GLOB}",
    )
    p.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT))
    p.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    p.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    rows = _load_rows(ns.input_glob)
    agg = _aggregate(rows)
    _emit_csv(rows, Path(ns.csv_out))
    _emit_json(rows, agg, Path(ns.json_out))
    _emit_md(rows, agg, Path(ns.md_out))
    md_path = Path(ns.md_out).resolve()
    try:
        md_display = md_path.relative_to(REPO_ROOT)
    except ValueError:
        md_display = md_path
    print(
        f"[small-l3-thrash] cells={agg['n_cells']} rows={agg['n_rows']} "
        f"policies={len(agg['policy_stats'])} → "
        f"{md_display}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
