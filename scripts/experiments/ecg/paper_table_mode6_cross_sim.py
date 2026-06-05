#!/usr/bin/env python3
"""Paper Table 8 — Sniper cross-simulator mode 6 corroboration.

Sprint 6f-6 deliverable: cross-simulator audit of the cache_sim-derived
mode 6 (per-edge mask) result from sprint 6f-5. Reads Sniper cells
from /tmp/graphbrew-pfx-sniper-mode6 (output root of
pfx_sniper_mode6_sweep.sh), pairs them with the matching cache_sim
mode 6 cells, and emits a side-by-side comparison table.

Status: SCAFFOLD. The Sniper sweep is long-running (~5400s per cell);
this emitter is wired into the cascade so that as cells land, the
table updates incrementally.

Reads:
  --sniper-root   directory tree of <graph>-<app>/<arm>/roi_matrix.csv
  --cache-sim-csv canonical cache_sim mode 6 table
                  (default: wiki/data/paper_table_mode6_corpus.csv)

CLI::

    python3 -m scripts.experiments.ecg.paper_table_mode6_cross_sim \\
        --sniper-root /tmp/graphbrew-pfx-sniper-mode6 \\
        --cache-sim-csv wiki/data/paper_table_mode6_corpus.csv \\
        --json-out wiki/data/paper_table_mode6_cross_sim.json \\
        --md-out   wiki/data/paper_table_mode6_cross_sim.md \\
        --csv-out  wiki/data/paper_table_mode6_cross_sim.csv \\
        --tex-out  docs/paper_tables/paper_table_mode6_cross_sim.tex

The sniper-root layout mirrors pfx_sniper_mode6_sweep.sh's OUT_ROOT:

    <sniper-root>/<graph>-<app>/<arm>/roi_matrix.csv

where <arm> in (none, DROPLET, ECG_PFX). The script extracts the
l3_miss_rate + memory_accesses fields per cell, computes the
pp-savings of ECG_PFX vs none, and reports it alongside the
cache_sim mode 6 pp-savings for that cell. Cells absent from
sniper-root are flagged as "pending."
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _read_first_row(csv_path: Path) -> dict | None:
    """Return the first non-error data row from a roi_matrix CSV.

    Returns None if the file is missing, empty, or all rows have
    status != ok/active_no_fill/inactive.
    """
    if not csv_path.is_file():
        return None
    valid_statuses = {"ok", "active_no_fill", "inactive"}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            status = (row.get("status") or "").strip()
            if status in valid_statuses:
                return row
    return None


def _f(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _i(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _harvest_sniper(sniper_root: Path, cells: list[str]) -> dict[str, dict]:
    """For each "graph-app" cell, return {arm: {l3_miss_rate, mem_accs}}."""
    out: dict[str, dict] = {}
    if not sniper_root.is_dir():
        return out
    for cell in cells:
        cell_dir = sniper_root / cell
        if not cell_dir.is_dir():
            continue
        cell_data: dict = {}
        for arm in ("none", "DROPLET", "ECG_PFX"):
            csv_path = cell_dir / arm / "roi_matrix.csv"
            row = _read_first_row(csv_path)
            if row:
                cell_data[arm] = {
                    "l3_miss_rate": _f(row.get("l3_miss_rate")),
                    "memory_accesses": _i(row.get("memory_accesses")),
                    "total_accesses": _i(row.get("total_accesses")),
                    "status": row.get("status"),
                }
        if cell_data:
            out[cell] = cell_data
    return out


def _read_cache_sim_table(csv_path: Path) -> dict[str, dict]:
    """Parse the canonical cache_sim mode 6 corpus CSV into per-cell dict."""
    out: dict[str, dict] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            cell = row.get("cell") or row.get("Cell") or row.get("graph_app")
            if not cell:
                continue
            out[cell] = row
    return out


def _build_rows(sniper_data: dict[str, dict],
                cache_sim_data: dict[str, dict],
                cells: list[str]) -> list[dict]:
    rows = []
    for cell in cells:
        sn = sniper_data.get(cell, {})
        cs = cache_sim_data.get(cell, {})

        baseline_lmr = sn.get("none", {}).get("l3_miss_rate")
        ecg_pfx_lmr = sn.get("ECG_PFX", {}).get("l3_miss_rate")
        sniper_pp = None
        if baseline_lmr is not None and ecg_pfx_lmr is not None:
            sniper_pp = (baseline_lmr - ecg_pfx_lmr) * 100.0

        rows.append({
            "cell": cell,
            "sniper_arm_none_status": sn.get("none", {}).get("status", "pending"),
            "sniper_arm_droplet_status": sn.get("DROPLET", {}).get("status", "pending"),
            "sniper_arm_ecg_pfx_status": sn.get("ECG_PFX", {}).get("status", "pending"),
            "sniper_baseline_l3_miss_rate": baseline_lmr,
            "sniper_ecg_pfx_l3_miss_rate": ecg_pfx_lmr,
            "sniper_ecg_pfx_pp_savings": sniper_pp,
            # cache_sim companion from the existing mode 6 corpus table
            "cache_sim_mode6_pp_savings": _f(cs.get("ecg_pfx_pp_savings")
                                              or cs.get("mode6_pp_savings")),
        })
    return rows


def _render_md(rows: list[dict]) -> str:
    n_complete = sum(1 for r in rows
                     if r["sniper_arm_ecg_pfx_status"] not in ("pending", "error", None))
    n_total = len(rows)
    lines = [
        "# Paper Table 8 — Sniper cross-simulator mode 6 corroboration",
        "",
        f"**Status: {n_complete}/{n_total} cells with valid Sniper ECG_PFX data.**",
        "",
        ("Sprint 6f-6 cross-sim audit: pairs Sniper mode 6 cells (from "
         "pfx_sniper_mode6_sweep.sh) with their cache_sim mode 6 "
         "counterparts. Cells in 'pending' state are still simulating "
         "or have not yet been launched."),
        "",
        "## Per-cell pp-savings",
        "",
        "| Cell | Sniper baseline | Sniper ECG_PFX | Sniper Δ | cache_sim Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "—"
        def _ppfmt(v):
            return f"{v:+.2f}pp" if isinstance(v, float) else "pending"
        lines.append(
            f"| {r['cell']} "
            f"| {_fmt(r['sniper_baseline_l3_miss_rate'])} "
            f"| {_fmt(r['sniper_ecg_pfx_l3_miss_rate'])} "
            f"| {_ppfmt(r['sniper_ecg_pfx_pp_savings'])} "
            f"| {_ppfmt(r['cache_sim_mode6_pp_savings'])} |"
        )
    return "\n".join(lines) + "\n"


def _render_tex(rows: list[dict]) -> str:
    n_complete = sum(1 for r in rows
                     if r["sniper_arm_ecg_pfx_status"] not in ("pending", "error", None))
    n_total = len(rows)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\footnotesize",
        ("  \\caption{Sniper cross-simulator mode 6 corroboration. "
         "Per-cell ECG\\_PFX pp-savings under Sniper paired with the "
         f"matching cache\\_sim mode 6 result. Status: {n_complete} of "
         f"{n_total} cells complete; remaining cells are simulating or "
         "have timed out at the per-cell wall-clock budget.}"),
        "  \\label{tab:ecg_mode6_cross_sim}",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    Cell & Sniper baseline LMR & Sniper $\\Delta$ pp & cache\\_sim $\\Delta$ pp \\\\",
        "    \\midrule",
    ]
    for r in rows:
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "---"
        def _ppfmt(v):
            return f"{v:+.2f}" if isinstance(v, float) else "pending"
        lines.append(
            f"    {r['cell'].replace('_', '\\_')} & "
            f"{_fmt(r['sniper_baseline_l3_miss_rate'])} & "
            f"{_ppfmt(r['sniper_ecg_pfx_pp_savings'])} & "
            f"{_ppfmt(r['cache_sim_mode6_pp_savings'])} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sniper-root", default="/tmp/graphbrew-pfx-sniper-mode6",
                        help="Sniper sweep OUT_ROOT (cell layout: <graph>-<app>/<arm>/roi_matrix.csv)")
    parser.add_argument("--cache-sim-csv",
                        default="wiki/data/paper_table_mode6_corpus.csv",
                        help="canonical cache_sim mode 6 CSV for cross-reference")
    parser.add_argument("--cells", nargs="*",
                        default=["email-Eu-core-pr", "delaunay_n19-pr",
                                 "roadNet-CA-pr", "web-Google-pr"],
                        help="cells to report on (one per line)")
    parser.add_argument("--json-out",
                        default="wiki/data/paper_table_mode6_cross_sim.json")
    parser.add_argument("--md-out",
                        default="wiki/data/paper_table_mode6_cross_sim.md")
    parser.add_argument("--csv-out",
                        default="wiki/data/paper_table_mode6_cross_sim.csv")
    parser.add_argument("--tex-out",
                        default="docs/paper_tables/paper_table_mode6_cross_sim.tex")
    args = parser.parse_args(argv)

    sniper_root = Path(args.sniper_root)
    cache_sim_csv = Path(args.cache_sim_csv)

    sniper_data = _harvest_sniper(sniper_root, args.cells)
    cache_sim_data = _read_cache_sim_table(cache_sim_csv)
    rows = _build_rows(sniper_data, cache_sim_data, args.cells)

    Path(args.json_out).write_text(json.dumps({
        "source_sniper_root": str(sniper_root),
        "source_cache_sim_csv": str(cache_sim_csv),
        "cells": rows,
    }, indent=2) + "\n")
    Path(args.md_out).write_text(_render_md(rows))

    with Path(args.csv_out).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    Path(args.tex_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tex_out).write_text(_render_tex(rows))

    n_complete = sum(1 for r in rows
                     if r["sniper_arm_ecg_pfx_status"] not in ("pending", "error", None))
    print(f"[mode6-cross-sim] {n_complete} / {len(rows)} cells with valid Sniper data; "
          f"wrote {args.md_out}, {args.json_out}, {args.csv_out}, {args.tex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
