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


def _dram_requests_from_sim_stats(cell_dir: Path, arm: str) -> int | None:
    """Fallback: parse dram-bank-group-*.num-requests directly from
    sim.stats when the CSV doesn't yet carry the dram_demand_requests
    column (pre-bug-1 sweep output).
    """
    pattern = (cell_dir / arm / "sniper").glob("sniper_*/simulation/sim.stats")
    for stats_path in pattern:
        if not stats_path.is_file():
            continue
        total = 0
        try:
            for line in stats_path.read_text(errors="replace").splitlines():
                if line.startswith("dram-bank-group-") and ".num-requests =" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        try:
                            total += int(float(parts[1].strip()))
                        except ValueError:
                            pass
        except OSError:
            return None
        return total
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
                dram = _i(row.get("dram_demand_requests"))
                if dram is None:
                    dram = _dram_requests_from_sim_stats(cell_dir, arm)
                cell_data[arm] = {
                    "l3_miss_rate": _f(row.get("l3_miss_rate")),
                    "memory_accesses": _i(row.get("memory_accesses")),
                    "total_accesses": _i(row.get("total_accesses")),
                    "dram_demand_requests": dram,
                    "pf_issued": _i(row.get("pf_issued")),
                    "pf_useful": _i(row.get("pf_useful")),
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
            # Normalize the cache_sim corpus columns to our generic keys.
            out[cell] = {
                "ecg_pfx_pp_savings": row.get("mode6_delta_pp"),
                "mode6_pp_savings": row.get("mode6_delta_pp"),
                **row,
            }
    return out


def _build_rows(sniper_data: dict[str, dict],
                cache_sim_data: dict[str, dict],
                cells: list[str]) -> list[dict]:
    rows = []
    for cell in cells:
        sn = sniper_data.get(cell, {})
        cs = cache_sim_data.get(cell, {})

        baseline_lmr = sn.get("none", {}).get("l3_miss_rate")
        droplet_lmr = sn.get("DROPLET", {}).get("l3_miss_rate")
        ecg_pfx_lmr = sn.get("ECG_PFX", {}).get("l3_miss_rate")
        droplet_pp = None
        ecg_pfx_pp = None
        if baseline_lmr is not None and droplet_lmr is not None:
            droplet_pp = (baseline_lmr - droplet_lmr) * 100.0
        if baseline_lmr is not None and ecg_pfx_lmr is not None:
            ecg_pfx_pp = (baseline_lmr - ecg_pfx_lmr) * 100.0

        # DRAM-level demand request count is the §4.3-safe metric.
        # Δ values here are ratios (DROPLET/baseline, ECG/baseline)
        # so values >1 mean "more DRAM traffic" (worse).
        baseline_dram = sn.get("none", {}).get("dram_demand_requests")
        droplet_dram = sn.get("DROPLET", {}).get("dram_demand_requests")
        ecg_pfx_dram = sn.get("ECG_PFX", {}).get("dram_demand_requests")
        droplet_dram_ratio = None
        ecg_pfx_dram_ratio = None
        if baseline_dram and droplet_dram is not None:
            droplet_dram_ratio = droplet_dram / baseline_dram
        if baseline_dram and ecg_pfx_dram is not None:
            ecg_pfx_dram_ratio = ecg_pfx_dram / baseline_dram

        rows.append({
            "cell": cell,
            "sniper_arm_none_status": sn.get("none", {}).get("status", "pending"),
            "sniper_arm_droplet_status": sn.get("DROPLET", {}).get("status", "pending"),
            "sniper_arm_ecg_pfx_status": sn.get("ECG_PFX", {}).get("status", "pending"),
            "sniper_baseline_l3_miss_rate": baseline_lmr,
            "sniper_droplet_l3_miss_rate": droplet_lmr,
            "sniper_ecg_pfx_l3_miss_rate": ecg_pfx_lmr,
            "sniper_droplet_pp_savings": droplet_pp,
            "sniper_ecg_pfx_pp_savings": ecg_pfx_pp,
            # DRAM demand traffic (sprint 6f-6 bug 1 fix)
            "sniper_baseline_dram_requests": baseline_dram,
            "sniper_droplet_dram_requests": droplet_dram,
            "sniper_ecg_pfx_dram_requests": ecg_pfx_dram,
            "sniper_droplet_dram_ratio": droplet_dram_ratio,
            "sniper_ecg_pfx_dram_ratio": ecg_pfx_dram_ratio,
            # Prefetcher activity counters
            "sniper_droplet_pf_useful": sn.get("DROPLET", {}).get("pf_useful"),
            "sniper_ecg_pfx_pf_useful": sn.get("ECG_PFX", {}).get("pf_useful"),
            # cache_sim companions from the existing mode 6 corpus table
            "cache_sim_mode6_pp_savings": _f(cs.get("ecg_pfx_pp_savings")
                                              or cs.get("mode6_pp_savings")),
            "cache_sim_droplet_pp_savings": _f(cs.get("droplet_delta_pp")),
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
         "or have not yet been launched. The DROPLET column is the "
         "Sniper-side baseline-stronger comparator (different "
         "prefetching mechanism than ECG_PFX mode 6)."),
        "",
        "## ⚠️ Metric caveat — l3_miss_rate is NOT a fair cross-arm comparator here",
        "",
        ("The L3 miss-rate metric below is reported because it is the "
         "raw counter the simulator emits, but it is **misleading "
         "across arms** for two independent reasons:"),
        "",
        ("1. **Prefetcher placement**: in our config DROPLET and ECG_PFX "
         "both attach at L2 (`--prefetcher-level l2`); the original "
         "DROPLET paper (Basak HPCA'19) prefetches at L1. The 1,969 "
         "useful prefetches DROPLET issues land at L2 and hide L2 "
         "miss latency, but the L3 still sees the same demand-miss "
         "count, so DROPLET's l3_miss_rate equals the baseline even "
         "though DROPLET genuinely helped."),
        "",
        ("2. **Mask-charged denominator inflation**: ECG_PFX mode 6 "
         "with `ECG_EDGE_MASK_CHARGED=1` (default) reads the per-edge "
         "mask array on every edge access, inflating `l3_accesses` "
         "by ~10× (2,360 → 23,295 on email-Eu-core). The mask reads "
         "are mostly L3 hits, so `l3_misses / l3_accesses` drops "
         "purely by denominator growth, not by actual demand-miss "
         "reduction. This is the same metric trap §4.3 of the paper "
         "documents (sprint 6f-2 finding)."),
        "",
        ("Until the demand-memory rate (`memory_accesses / "
         "total_accesses`) is wired into the Sniper postfix builder, "
         "the headline metric for these Sniper cells is "
         "`pf_useful` (the count of prefetches that hid a demand "
         "miss at the prefetcher's attachment level)."),
        "",
        "## Per-cell Sniper measurements (l3_miss_rate; pp = baseline minus arm)",
        "",
        "| Cell | none | DROPLET | ECG_PFX | DROPLET Δ | ECG_PFX Δ | cache_sim mode 6 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "—"
        def _ppfmt(v):
            return f"{v:+.2f}pp" if isinstance(v, float) else "pending"
        lines.append(
            f"| {r['cell']} "
            f"| {_fmt(r['sniper_baseline_l3_miss_rate'])} "
            f"| {_fmt(r['sniper_droplet_l3_miss_rate'])} "
            f"| {_fmt(r['sniper_ecg_pfx_l3_miss_rate'])} "
            f"| {_ppfmt(r['sniper_droplet_pp_savings'])} "
            f"| {_ppfmt(r['sniper_ecg_pfx_pp_savings'])} "
            f"| {_ppfmt(r['cache_sim_mode6_pp_savings'])} |"
        )
    lines.extend([
        "",
        "## DRAM-level demand traffic (the §4.3-safe metric)",
        "",
        ("Aggregated `dram-bank-group-*.num-requests` counters across "
         "all DRAM banks. Ratio >1 means MORE DRAM traffic than baseline "
         "(worse); ratio <1 means LESS DRAM traffic (better)."),
        "",
        "| Cell | none (req) | DROPLET (req) | ECG_PFX (req) | DROPLET/base | ECG_PFX/base |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in rows:
        def _ifmt(v):
            return f"{v:,}" if isinstance(v, int) else "—"
        def _rfmt(v):
            if not isinstance(v, float):
                return "pending"
            sign = "✓" if v < 0.95 else ("≈" if v <= 1.05 else "✗")
            return f"{v:.2f}× {sign}"
        lines.append(
            f"| {r['cell']} "
            f"| {_ifmt(r['sniper_baseline_dram_requests'])} "
            f"| {_ifmt(r['sniper_droplet_dram_requests'])} "
            f"| {_ifmt(r['sniper_ecg_pfx_dram_requests'])} "
            f"| {_rfmt(r['sniper_droplet_dram_ratio'])} "
            f"| {_rfmt(r['sniper_ecg_pfx_dram_ratio'])} |"
        )
    lines.extend([
        "",
        "## Honest reading of the email-Eu-core row",
        "",
        ("Raw Sniper counters (not in the table above):"),
        "",
        ("- baseline: l3_accesses=2360, l3_misses=2360, DRAM requests=2360, pf_useful=0"),
        ("- DROPLET: l3_accesses=2359, l3_misses=2359, DRAM requests=2359, "
         "**pf_useful=1969** (100% L2-hit accuracy, but doesn't reduce DRAM traffic)"),
        ("- ECG_PFX: l3_accesses=23295, l3_misses=8768, **DRAM requests=15312 (6.5× MORE)**, "
         "pf_issued=38, pf_useful=38"),
        "",
        ("Under the L3-miss-rate metric ECG_PFX appeared to win +62pp; "
         "under the §4.3-safe DRAM-traffic metric, ECG_PFX **increases** "
         "DRAM traffic 6.5× on email-Eu-core because the per-edge mask "
         "reads themselves miss to DRAM. DROPLET's 1,969 useful L2 "
         "prefetches do not reduce DRAM traffic (those lines were "
         "cold-misses regardless) but they do hide L2 miss latency. "
         "email-Eu-core is structurally too small to demonstrate "
         "prefetching at L3 boundary: the entire property array fits "
         "in L1d (4 KB << 32 KB L1d). Cells larger than L1d "
         "(delaunay_n19's 2 MB and above) are needed to measure "
         "prefetcher value cleanly."),
        "",
        ("This is **NOT** a refutation of the cache_sim mode 6 corpus "
         "finding (which uses million-vertex graphs where the property "
         "array exceeds L3). It IS a demonstration that the "
         "convergence story (§5.4 of the paper) holds: when the cache "
         "hierarchy already captures the working set, prefetcher state "
         "adds bandwidth without benefit."),
    ])
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
         "\\textbf{Headline metric is DRAM demand-request ratio} "
         "(arm/baseline; $<$1 = better, $\\approx$1 = neutral, "
         "$>$1 = worse). The L3 miss-rate column is reported but "
         "interpretation requires the caveat in \\S\\ref{sec:methodology:metrics}: "
         "both DROPLET and ECG\\_PFX attach at L2, so DROPLET's "
         "useful prefetches hide L2 latency without reducing L3 "
         "demand; ECG\\_PFX's mask-cost-charged accesses inflate "
         "the L3 miss-rate denominator. "
         f"Status: {n_complete} of {n_total} cells complete; "
         "remaining cells are simulating or have timed out at the "
         "per-cell wall-clock budget.}"),
        "  \\label{tab:ecg_mode6_cross_sim}",
        "  \\begin{tabular}{lrrrrr}",
        "    \\toprule",
        "    \\multirow{2}{*}{Cell} & \\multicolumn{3}{c}{DRAM demand req (\\S\\ref{sec:methodology:metrics})} & \\multicolumn{2}{c}{L3 miss-rate $\\Delta$} \\\\",
        "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-6}",
        "    & none & DROPLET/base & ECG\\_PFX/base & DROPLET & ECG\\_PFX \\\\",
        "    \\midrule",
    ]
    for r in rows:
        def _ifmt(v):
            return f"{v:,}" if isinstance(v, int) else "---"
        def _rfmt(v):
            return f"{v:.2f}$\\times$" if isinstance(v, float) else "pending"
        def _ppfmt(v):
            return f"{v:+.2f}pp" if isinstance(v, float) else "pending"
        lines.append(
            f"    {r['cell'].replace('_', '\\_')} & "
            f"{_ifmt(r['sniper_baseline_dram_requests'])} & "
            f"{_rfmt(r['sniper_droplet_dram_ratio'])} & "
            f"{_rfmt(r['sniper_ecg_pfx_dram_ratio'])} & "
            f"{_ppfmt(r['sniper_droplet_pp_savings'])} & "
            f"{_ppfmt(r['sniper_ecg_pfx_pp_savings'])} \\\\"
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
