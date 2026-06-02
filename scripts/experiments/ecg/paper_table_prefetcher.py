#!/usr/bin/env python3
"""Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB.

Consumes the matched sweep produced by
``scripts/experiments/ecg/sweeps/pfx_cache_sim_scale_sweep.sh`` under
``/tmp/graphbrew-ecg-pfx-cache_sim-scale/{graph}-{app}/{baselines,pfx_combined}/roi_matrix.csv``
and emits a paper-grade matrix:

    rows = (graph, app)
    cols = LRU, SRRIP, GRASP, POPT, ECG_DBG_ONLY, ECG_DBG_PRIMARY,
           ECG_DBG_ONLY + ECG_PFX, delta_vs_LRU, delta_vs_GRASP

The delta columns are ECG_DBG_ONLY+ECG_PFX miss-rate MINUS the listed
baseline's miss-rate (in pp). Negative = ECG combined wins.

Outputs:
- ``wiki/data/paper_table_prefetcher.csv`` (machine-readable)
- ``wiki/data/paper_table_prefetcher.md`` (rendered table)
- ``wiki/data/paper_table_prefetcher.json`` (structured)
- ``wiki/data/paper_table_prefetcher.tex`` (LaTeX, for the paper directly)

Headline claim emitted into paper_claims.json:
    prefetcher.ecg_combined_vs_lru_mean_pp = (mean Δ miss-rate)

CLI::

    python3 -m scripts.experiments.ecg.paper_table_prefetcher \\
        --sweep-root /tmp/graphbrew-ecg-pfx-cache_sim-scale \\
        --json-out wiki/data/paper_table_prefetcher.json \\
        --md-out wiki/data/paper_table_prefetcher.md \\
        --csv-out wiki/data/paper_table_prefetcher.csv \\
        --tex-out wiki/data/paper_table_prefetcher.tex
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

BASELINE_LABELS = (
    "LRU",
    "SRRIP",
    "GRASP",
    "POPT",
    "ECG_DBG_PRIMARY",
    "ECG_DBG_ONLY",
)
PFX_LABEL = "ECG_DBG_ONLY + ECG_PFX"
DELTA_BASES = ("LRU", "GRASP", "POPT")


def _read_baseline_row(csv_path: Path, policy_label: str) -> dict | None:
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("policy_label") == policy_label or r.get("policy") == policy_label:
                return r
    return None


def _read_pfx_row(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("prefetcher") == "ECG_PFX":
                return r
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_cells(sweep_root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    if not sweep_root.exists():
        return cells
    for cell_dir in sorted(sweep_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        name = cell_dir.name
        if "-" not in name:
            continue
        graph, app = name.rsplit("-", 1)
        base_csv = cell_dir / "baselines" / "roi_matrix.csv"
        pfx_csv = cell_dir / "pfx_combined" / "roi_matrix.csv"
        row: dict[str, Any] = {"graph": graph, "app": app}
        for pol in BASELINE_LABELS:
            br = _read_baseline_row(base_csv, pol)
            row[pol] = _coerce_float(br.get("l3_miss_rate")) if br else None
        pfx_row = _read_pfx_row(pfx_csv)
        row[PFX_LABEL] = _coerce_float(pfx_row.get("l3_miss_rate")) if pfx_row else None
        # Capture activity counters for table footnote / validation
        if pfx_row:
            row["prefetch_fills"] = int(pfx_row.get("prefetch_fills") or 0)
            row["prefetch_useful"] = int(pfx_row.get("prefetch_useful") or 0)
            row["prefetch_requests"] = int(pfx_row.get("prefetch_requests") or 0)
        for base in DELTA_BASES:
            base_mr = row.get(base)
            pfx_mr = row.get(PFX_LABEL)
            if base_mr is not None and pfx_mr is not None:
                row[f"delta_vs_{base}_pp"] = (pfx_mr - base_mr) * 100
            else:
                row[f"delta_vs_{base}_pp"] = None
        cells.append(row)
    return cells


def emit_csv(cells: list[dict], path: Path) -> None:
    cols = ["graph", "app"] + list(BASELINE_LABELS) + [PFX_LABEL] + \
        [f"delta_vs_{b}_pp" for b in DELTA_BASES] + \
        ["prefetch_fills", "prefetch_useful", "prefetch_requests"]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            row = {k: c.get(k) for k in cols}
            # Round miss-rates for readability
            for pol in list(BASELINE_LABELS) + [PFX_LABEL]:
                v = row.get(pol)
                if v is not None:
                    row[pol] = round(v, 4)
            for b in DELTA_BASES:
                v = row.get(f"delta_vs_{b}_pp")
                if v is not None:
                    row[f"delta_vs_{b}_pp"] = round(v, 2)
            w.writerow(row)


def emit_md(cells: list[dict], path: Path, summary: dict) -> None:
    lines: list[str] = []
    lines.append("# Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB")
    lines.append("")
    lines.append("Cache simulator with `ECG_CONTAINER_BITS=64` and runtime")
    lines.append("`ECG_PREFETCH_LOOKAHEAD=8` (`ECG_PREFETCH_MODE=2`, popt-ranked).")
    lines.append("")
    lines.append("## Headline summary")
    lines.append("")
    if summary:
        if summary.get("n_cells_with_full_row") is not None:
            lines.append(f"- Cells with full data: **{summary['n_cells_with_full_row']}** of {len(cells)}")
        if summary.get("mean_delta_vs_LRU_pp") is not None:
            lines.append(f"- Mean Δ ECG_combined vs LRU: **{summary['mean_delta_vs_LRU_pp']:+.2f} pp**")
        if summary.get("mean_delta_vs_GRASP_pp") is not None:
            lines.append(f"- Mean Δ ECG_combined vs GRASP: **{summary['mean_delta_vs_GRASP_pp']:+.2f} pp**")
        if summary.get("mean_delta_vs_POPT_pp") is not None:
            lines.append(f"- Mean Δ ECG_combined vs POPT: **{summary['mean_delta_vs_POPT_pp']:+.2f} pp**")
        if summary.get("mean_useful_rate") is not None:
            lines.append(f"- Mean prefetch useful-rate: **{summary['mean_useful_rate'] * 100:.2f}%**")
    lines.append("")
    lines.append("## Per-cell miss-rates")
    lines.append("")
    head = ["graph", "app", "LRU", "SRRIP", "GRASP", "POPT", "ECG_DBG_ONLY", "ECG+PFX", "Δ vs LRU", "Δ vs GRASP", "Δ vs POPT"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for c in cells:
        def f(key, fmt="{:.4f}", default="—"):
            v = c.get(key)
            return fmt.format(v) if v is not None else default
        row_cells = [
            c["graph"], c["app"],
            f("LRU"), f("SRRIP"), f("GRASP"), f("POPT"),
            f("ECG_DBG_ONLY"), f(PFX_LABEL),
            f("delta_vs_LRU_pp", "{:+.2f} pp"),
            f("delta_vs_GRASP_pp", "{:+.2f} pp"),
            f("delta_vs_POPT_pp", "{:+.2f} pp"),
        ]
        lines.append("| " + " | ".join(str(x) for x in row_cells) + " |")
    lines.append("")
    lines.append("## Prefetcher activity")
    lines.append("")
    lines.append("| graph | app | requests | fills | useful | useful_rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for c in cells:
        req = c.get("prefetch_requests") or 0
        fills = c.get("prefetch_fills") or 0
        useful = c.get("prefetch_useful") or 0
        rate = (useful / fills * 100) if fills else 0.0
        lines.append(f"| {c['graph']} | {c['app']} | {req:,} | {fills:,} | {useful:,} | {rate:.2f}% |")
    path.write_text("\n".join(lines) + "\n")


def emit_tex(cells: list[dict], path: Path, summary: dict) -> None:
    lines: list[str] = []
    lines.append(r"% Auto-generated by paper_table_prefetcher.py")
    lines.append(r"% Do not edit by hand — re-run `make lit-paper-table-prefetcher`")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{ECG combined mask (\textsc{ECG-D} eviction + \textsc{ECG-PFX} prefetcher) vs literature baselines at $L3=1$MB. Negative $\Delta$ = ECG-combined wins. Computed by cache simulator with \texttt{ECG\_CONTAINER\_BITS=64} and \texttt{ECG\_PREFETCH\_LOOKAHEAD=8}.}")
    lines.append(r"  \label{tab:ecg_prefetcher}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{llrrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Graph & App & LRU & SRRIP & GRASP & POPT & ECG-D & ECG+PFX & $\Delta$LRU & $\Delta$GRASP & $\Delta$POPT \\")
    lines.append(r"    \midrule")
    for c in cells:
        def f(key, fmt="{:.3f}", default="--"):
            v = c.get(key)
            return fmt.format(v) if v is not None else default
        row_cells = [
            c["graph"].replace("_", r"\_"), c["app"],
            f("LRU"), f("SRRIP"), f("GRASP"), f("POPT"),
            f("ECG_DBG_ONLY"), f(PFX_LABEL),
            f("delta_vs_LRU_pp", "{:+.2f}"),
            f("delta_vs_GRASP_pp", "{:+.2f}"),
            f("delta_vs_POPT_pp", "{:+.2f}"),
        ]
        lines.append("    " + " & ".join(str(x) for x in row_cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n")


def compute_summary(cells: list[dict]) -> dict:
    full = [c for c in cells if c.get(PFX_LABEL) is not None and c.get("LRU") is not None]
    out: dict[str, Any] = {
        "n_cells_total": len(cells),
        "n_cells_with_full_row": len(full),
    }
    for base in DELTA_BASES:
        deltas = [c[f"delta_vs_{base}_pp"] for c in full if c.get(f"delta_vs_{base}_pp") is not None]
        out[f"mean_delta_vs_{base}_pp"] = mean(deltas) if deltas else None
        out[f"min_delta_vs_{base}_pp"] = min(deltas) if deltas else None
        out[f"max_delta_vs_{base}_pp"] = max(deltas) if deltas else None
    rates: list[float] = []
    for c in cells:
        fills = c.get("prefetch_fills") or 0
        useful = c.get("prefetch_useful") or 0
        if fills > 0:
            rates.append(useful / fills)
    out["mean_useful_rate"] = mean(rates) if rates else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--tex-out", type=Path, required=True)
    args = parser.parse_args()

    cells = load_cells(args.sweep_root)
    summary = compute_summary(cells)

    emit_csv(cells, args.csv_out)
    emit_md(cells, args.md_out, summary)
    emit_tex(cells, args.tex_out, summary)
    args.json_out.write_text(json.dumps(
        {
            "schema_version": 1,
            "source": "scripts.experiments.ecg.paper_table_prefetcher",
            "sweep_root": str(args.sweep_root),
            "baselines": list(BASELINE_LABELS),
            "pfx_label": PFX_LABEL,
            "delta_bases": list(DELTA_BASES),
            "summary": summary,
            "cells": cells,
        },
        indent=2,
    ) + "\n")

    mean_lru = summary.get("mean_delta_vs_LRU_pp")
    mean_grasp = summary.get("mean_delta_vs_GRASP_pp")
    print(f"[paper-table-prefetcher] n_cells={len(cells)} full_rows={summary['n_cells_with_full_row']}"
          f" mean_delta_vs_LRU={mean_lru:.2f}pp" if mean_lru is not None else
          f"[paper-table-prefetcher] n_cells={len(cells)} full_rows={summary['n_cells_with_full_row']} mean_delta_vs_LRU=N/A")
    if mean_grasp is not None:
        print(f"[paper-table-prefetcher] mean_delta_vs_GRASP={mean_grasp:.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
