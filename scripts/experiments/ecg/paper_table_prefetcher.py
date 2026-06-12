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
DROPLET_LABEL = "ECG_DBG_ONLY + DROPLET"
DELTA_BASES = ("LRU", "GRASP", "POPT", "DROPLET_COMBINED")


def _read_baseline_row(csv_path: Path, policy_label: str) -> dict | None:
    if not csv_path.exists():
        return None
    # Accept short-form aliases used by older sweep scripts that pass
    # `--policies ECG_DBG ECG_PRIMARY` instead of the canonical
    # `--policies ECG_DBG_PRIMARY ECG_DBG_ONLY`. The cache_sim emits
    # policy_label=ECG_DBG for the former and ECG_DBG_ONLY for the latter,
    # so we treat them as the same eviction-only ablation.
    aliases = {
        "ECG_DBG_ONLY": ("ECG_DBG_ONLY", "ECG_DBG"),
        "ECG_DBG_PRIMARY": ("ECG_DBG_PRIMARY", "ECG_PRIMARY"),
    }
    accept = aliases.get(policy_label, (policy_label,))
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            label = r.get("policy_label", "")
            pol = r.get("policy", "")
            if label in accept or pol in accept:
                return r
    return None


def _read_pfx_row(csv_path: Path, prefetcher: str = "ECG_PFX") -> dict | None:
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("prefetcher") == prefetcher:
                return r
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _demand_rate(row: dict | None) -> float | None:
    """Demand-only L3 miss-rate proxy.

    cache_sim counts prefetch-triggered L3 fetches toward l3_misses,
    which masks prefetcher value when the prefetcher is active. The
    correct prefetcher-aware metric is demand memory traffic per L1
    access — cache_sim's ``memory_accesses_++`` counter (see
    bench/include/cache_sim/cache_sim.h line 1450) is only
    incremented on the demand path; ``prefetch()`` explicitly does
    NOT touch it (line 1463 comment).
    """
    if row is None:
        return None
    total = _coerce_float(row.get("total_accesses"))
    mem = _coerce_float(row.get("memory_accesses"))
    if total is None or mem is None or total == 0:
        return None
    return mem / total


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
        drop_csv = cell_dir / "droplet_combined" / "roi_matrix.csv"
        row: dict[str, Any] = {"graph": graph, "app": app}
        for pol in BASELINE_LABELS:
            br = _read_baseline_row(base_csv, pol)
            row[pol] = _coerce_float(br.get("l3_miss_rate")) if br else None
            row[f"{pol}_demand"] = _demand_rate(br)
        pfx_row = _read_pfx_row(pfx_csv, "ECG_PFX")
        row[PFX_LABEL] = _coerce_float(pfx_row.get("l3_miss_rate")) if pfx_row else None
        row[f"{PFX_LABEL}_demand"] = _demand_rate(pfx_row)
        drop_row = _read_pfx_row(drop_csv, "DROPLET")
        row[DROPLET_LABEL] = _coerce_float(drop_row.get("l3_miss_rate")) if drop_row else None
        row[f"{DROPLET_LABEL}_demand"] = _demand_rate(drop_row)
        # Synthetic key for the delta-vs-DROPLET calculation
        row["DROPLET_COMBINED"] = row[DROPLET_LABEL]
        row["DROPLET_COMBINED_demand"] = row[f"{DROPLET_LABEL}_demand"]
        # Capture activity counters for table footnote / validation
        if pfx_row:
            row["pfx_fills"] = int(pfx_row.get("prefetch_fills") or 0)
            row["pfx_useful"] = int(pfx_row.get("prefetch_useful") or 0)
            row["pfx_requests"] = int(pfx_row.get("prefetch_requests") or 0)
            # Backward-compat keys (legacy field names from sprint 6c-1)
            row["prefetch_fills"] = row["pfx_fills"]
            row["prefetch_useful"] = row["pfx_useful"]
            row["prefetch_requests"] = row["pfx_requests"]
        if drop_row:
            row["droplet_fills"] = int(drop_row.get("prefetch_fills") or 0)
            row["droplet_useful"] = int(drop_row.get("prefetch_useful") or 0)
            row["droplet_requests"] = int(drop_row.get("prefetch_requests") or 0)
        for base in DELTA_BASES:
            base_mr = row.get(base)
            pfx_mr = row.get(PFX_LABEL)
            if base_mr is not None and pfx_mr is not None:
                row[f"delta_vs_{base}_pp"] = (pfx_mr - base_mr) * 100
            else:
                row[f"delta_vs_{base}_pp"] = None
            # Demand-only deltas — the prefetcher-aware metric.
            base_dr = row.get(f"{base}_demand")
            pfx_dr = row.get(f"{PFX_LABEL}_demand")
            if base_dr is not None and pfx_dr is not None:
                row[f"demand_delta_vs_{base}_pp"] = (pfx_dr - base_dr) * 100
            else:
                row[f"demand_delta_vs_{base}_pp"] = None
        # Marginal prefetcher gain ON TOP OF ECG_DBG eviction —
        # the honest "is the prefetcher doing anything?" measurement.
        dbg_only_demand = row.get("ECG_DBG_ONLY_demand")
        pfx_combined_demand = row.get(f"{PFX_LABEL}_demand")
        drop_combined_demand = row.get(f"{DROPLET_LABEL}_demand")
        if dbg_only_demand is not None and pfx_combined_demand is not None:
            row["pfx_marginal_demand_pp"] = (pfx_combined_demand - dbg_only_demand) * 100
        else:
            row["pfx_marginal_demand_pp"] = None
        if dbg_only_demand is not None and drop_combined_demand is not None:
            row["droplet_marginal_demand_pp"] = (drop_combined_demand - dbg_only_demand) * 100
        else:
            row["droplet_marginal_demand_pp"] = None
        cells.append(row)
    return cells


def emit_csv(cells: list[dict], path: Path) -> None:
    cols = ["graph", "app"] + list(BASELINE_LABELS) + [PFX_LABEL, DROPLET_LABEL] + \
        [f"{pol}_demand" for pol in BASELINE_LABELS] + \
        [f"{PFX_LABEL}_demand", f"{DROPLET_LABEL}_demand"] + \
        [f"delta_vs_{b}_pp" for b in DELTA_BASES] + \
        [f"demand_delta_vs_{b}_pp" for b in DELTA_BASES] + \
        ["pfx_marginal_demand_pp", "droplet_marginal_demand_pp",
         "pfx_fills", "pfx_useful", "pfx_requests",
         "droplet_fills", "droplet_useful", "droplet_requests"]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            row = {k: c.get(k) for k in cols}
            # Round miss-rates / demand-rates for readability
            for pol in list(BASELINE_LABELS) + [PFX_LABEL, DROPLET_LABEL]:
                v = row.get(pol)
                if v is not None:
                    row[pol] = round(v, 4)
                v = row.get(f"{pol}_demand")
                if v is not None:
                    row[f"{pol}_demand"] = round(v, 4)
            for b in DELTA_BASES:
                v = row.get(f"delta_vs_{b}_pp")
                if v is not None:
                    row[f"delta_vs_{b}_pp"] = round(v, 2)
                v = row.get(f"demand_delta_vs_{b}_pp")
                if v is not None:
                    row[f"demand_delta_vs_{b}_pp"] = round(v, 2)
            for k in ("pfx_marginal_demand_pp", "droplet_marginal_demand_pp"):
                v = row.get(k)
                if v is not None:
                    row[k] = round(v, 2)
            w.writerow(row)


def emit_md(cells: list[dict], path: Path, summary: dict) -> None:
    lines: list[str] = []
    lines.append("# Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB")
    lines.append("")
    lines.append("Cache simulator with `ECG_CONTAINER_BITS=64` and runtime")
    lines.append("`ECG_PREFETCH_LOOKAHEAD=8` (`ECG_PREFETCH_MODE=2`, popt-ranked).")
    lines.append("DROPLET-combined column uses the same lookahead window with")
    lines.append("sequential target selection (`ECG_PREFETCH_MODE=3`) — best-case")
    lines.append("oracle comparator to literature DROPLET (Basak HPCA'19); the")
    lines.append("real DROPLET stride detector would add mis-prediction overhead.")
    lines.append("")
    lines.append("## How to read this table")
    lines.append("")
    lines.append("Two metrics are reported. **The prefetcher-aware metric is the")
    lines.append("`demand-memory` rate**, not L3 miss-rate:")
    lines.append("")
    lines.append("- **`l3_miss_rate`** = `l3.misses / l3.accesses`. cache_sim's")
    lines.append("  l3.misses counter is incremented on every L3 lookup that misses,")
    lines.append("  including prefetch-triggered lookups (`prefetch()` calls")
    lines.append("  `l3->access()` which increments `misses_++`; see")
    lines.append("  bench/include/cache_sim/cache_sim.h:465 + 1480). When a")
    lines.append("  prefetcher is active, the prefetcher itself triggers L3 misses")
    lines.append("  (the fetch from memory IS an L3 miss), so L3 miss-rate barely")
    lines.append("  moves even when the prefetcher eliminates demand misses 1-for-1.")
    lines.append("- **`demand-memory` rate** = `memory_accesses / total_accesses`.")
    lines.append("  `memory_accesses_++` only fires on the demand path (cache_sim.h:1450)")
    lines.append("  and `prefetch()` explicitly does NOT increment it (cache_sim.h:1463")
    lines.append("  comment). This is demand misses to memory per demand access —")
    lines.append("  the metric the DROPLET paper's claims map onto.")
    lines.append("")
    lines.append("## Headline summary — demand-memory metric (prefetcher-aware)")
    lines.append("")
    if summary:
        if summary.get("n_cells_with_full_row") is not None:
            lines.append(f"- Cells with full data: **{summary['n_cells_with_full_row']}** of {len(cells)}")
        if summary.get("mean_demand_delta_vs_LRU_pp") is not None:
            lines.append(f"- Mean Δ ECG_combined demand-memory vs LRU: **{summary['mean_demand_delta_vs_LRU_pp']:+.2f} pp**")
        if summary.get("mean_pfx_marginal_demand_pp") is not None:
            lines.append(f"- **Marginal ECG_PFX gain on top of ECG_DBG eviction: `{summary['mean_pfx_marginal_demand_pp']:+.2f}` pp**  ← the honest prefetcher value")
        if summary.get("mean_droplet_marginal_demand_pp") is not None:
            lines.append(f"- **Marginal DROPLET gain on top of ECG_DBG eviction: `{summary['mean_droplet_marginal_demand_pp']:+.2f}` pp**")
        if summary.get("n_pfx_active_cells") is not None:
            lines.append(f"- Active prefetcher cells (≥1k requests issued): ECG_PFX **{summary['n_pfx_active_cells']}**, DROPLET **{summary['n_droplet_active_cells']}** of {len(cells)}")
        if summary.get("mean_pfx_marginal_demand_active_pp") is not None:
            lines.append(f"- Active-cell mean marginal: ECG_PFX **{summary['mean_pfx_marginal_demand_active_pp']:+.2f}** pp, DROPLET **{summary['mean_droplet_marginal_demand_active_pp']:+.2f}** pp")
        if summary.get("ecg_pfx_pp_per_mreq") is not None and summary.get("droplet_pp_per_mreq") is not None:
            lines.append(f"- Prefetcher efficiency (pp demand-memory reduction per million requests, active cells):")
            lines.append(f"  - ECG_PFX: **{summary['ecg_pfx_pp_per_mreq']:.4f}** pp/Mreq")
            lines.append(f"  - DROPLET: **{summary['droplet_pp_per_mreq']:.4f}** pp/Mreq")
    lines.append("")
    lines.append("## L3 miss-rate (pre-prefetch-aware metric; eviction story only)")
    lines.append("")
    if summary:
        if summary.get("mean_delta_vs_LRU_pp") is not None:
            lines.append(f"- Mean Δ ECG_combined L3 miss vs LRU: **{summary['mean_delta_vs_LRU_pp']:+.2f} pp** ← eviction component dominates")
        if summary.get("mean_delta_vs_GRASP_pp") is not None:
            lines.append(f"- Mean Δ ECG_combined L3 miss vs GRASP: **{summary['mean_delta_vs_GRASP_pp']:+.2f} pp**")
        if summary.get("mean_delta_vs_POPT_pp") is not None:
            lines.append(f"- Mean Δ ECG_combined L3 miss vs POPT: **{summary['mean_delta_vs_POPT_pp']:+.2f} pp**")
        if summary.get("mean_delta_vs_DROPLET_COMBINED_pp") is not None:
            lines.append(f"- Mean Δ ECG_PFX L3 miss vs DROPLET (same baseline): **{summary['mean_delta_vs_DROPLET_COMBINED_pp']:+.2f} pp** ← misleading: see demand-memory metric above")
        if summary.get("mean_useful_rate") is not None:
            lines.append(f"- Mean prefetch useful-rate: **{summary['mean_useful_rate'] * 100:.2f}%**")
        if summary.get("ecg_pfx_total_requests") is not None and summary.get("droplet_total_requests") is not None:
            ecg_r = summary["ecg_pfx_total_requests"]
            drop_r = summary["droplet_total_requests"]
            ratio = drop_r / ecg_r if ecg_r else 0.0
            lines.append(f"- Total prefetch requests issued: ECG_PFX **{ecg_r:,}** vs DROPLET **{drop_r:,}** (DROPLET issues {ratio:.2f}× more)")
    lines.append("")
    lines.append("## Per-cell demand-memory rate (prefetcher-aware)")
    lines.append("")
    head = ["graph", "app", "LRU", "ECG_DBG", "ECG+PFX", "ECG+DROP", "Δ DBG vs LRU", "Marg. PFX", "Marg. DROP"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for c in cells:
        def fd(key, fmt="{:.4f}", default="—"):
            v = c.get(key)
            return fmt.format(v) if v is not None else default
        lru_d = c.get("LRU_demand")
        dbg_d = c.get("ECG_DBG_ONLY_demand")
        dbg_lru_delta_pp = (dbg_d - lru_d) * 100 if (dbg_d is not None and lru_d is not None) else None
        row_cells = [
            c["graph"], c["app"],
            fd("LRU_demand"), fd("ECG_DBG_ONLY_demand"),
            fd(f"{PFX_LABEL}_demand"), fd(f"{DROPLET_LABEL}_demand"),
            f"{dbg_lru_delta_pp:+.2f} pp" if dbg_lru_delta_pp is not None else "—",
            fd("pfx_marginal_demand_pp", "{:+.2f} pp"),
            fd("droplet_marginal_demand_pp", "{:+.2f} pp"),
        ]
        lines.append("| " + " | ".join(str(x) for x in row_cells) + " |")
    lines.append("")
    lines.append("## Per-cell L3 miss-rates (legacy — kept for cross-reference)")
    lines.append("")
    head = ["graph", "app", "LRU", "GRASP", "POPT", "ECG_DBG", "ECG+PFX", "ECG+DROP", "Δ LRU", "Δ GRASP", "Δ POPT", "Δ DROPLET"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for c in cells:
        def f(key, fmt="{:.4f}", default="—"):
            v = c.get(key)
            return fmt.format(v) if v is not None else default
        row_cells = [
            c["graph"], c["app"],
            f("LRU"), f("GRASP"), f("POPT"),
            f("ECG_DBG_ONLY"), f(PFX_LABEL), f(DROPLET_LABEL),
            f("delta_vs_LRU_pp", "{:+.2f} pp"),
            f("delta_vs_GRASP_pp", "{:+.2f} pp"),
            f("delta_vs_POPT_pp", "{:+.2f} pp"),
            f("delta_vs_DROPLET_COMBINED_pp", "{:+.2f} pp"),
        ]
        lines.append("| " + " | ".join(str(x) for x in row_cells) + " |")
    lines.append("")
    lines.append("## Prefetcher efficiency (ECG_PFX vs DROPLET on same baseline)")
    lines.append("")
    lines.append("`req/useful` = total prefetch requests issued per useful prefetch.")
    lines.append("Lower is better (fewer wasted predictions per cache-hit benefit).")
    lines.append("`ratio` = ECG_PFX(req/useful) / DROPLET(req/useful). < 1.0 means ECG_PFX")
    lines.append("is more efficient than DROPLET.")
    lines.append("")
    lines.append("| graph | app | ECG_PFX requests | DROPLET requests | ECG_PFX req/useful | DROPLET req/useful | ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for c in cells:
        pfx_req = c.get("pfx_requests") or 0
        pfx_use = c.get("pfx_useful") or 0
        drop_req = c.get("droplet_requests") or 0
        drop_use = c.get("droplet_useful") or 0
        if not pfx_req and not drop_req:
            continue
        pfx_rpu = (pfx_req / pfx_use) if pfx_use else float("inf")
        drop_rpu = (drop_req / drop_use) if drop_use else float("inf")
        ratio = (pfx_rpu / drop_rpu) if (drop_rpu and drop_rpu != float("inf")) else None
        ratio_str = f"{ratio:.3f}" if ratio is not None else "—"
        pfx_rpu_str = f"{pfx_rpu:.3f}" if pfx_rpu != float("inf") else "—"
        drop_rpu_str = f"{drop_rpu:.3f}" if drop_rpu != float("inf") else "—"
        lines.append(f"| {c['graph']} | {c['app']} | {pfx_req:,} | {drop_req:,} | {pfx_rpu_str} | {drop_rpu_str} | {ratio_str} |")
    lines.append("")
    lines.append("## Prefetcher activity (ECG_PFX)")
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
    # Auto-pick a unique label per output file: kronecker_corpus output gets
    # tab:ecg_kronecker; literature corpus gets tab:ecg_prefetcher.
    label_suffix = "kronecker" if "kronecker" in str(path).lower() else "prefetcher"
    lines.append(r"  \label{tab:ecg_" + label_suffix + r"}")
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
        # Demand-only deltas — the prefetcher-aware metric.
        demand_deltas = [c[f"demand_delta_vs_{base}_pp"] for c in full
                         if c.get(f"demand_delta_vs_{base}_pp") is not None]
        out[f"mean_demand_delta_vs_{base}_pp"] = mean(demand_deltas) if demand_deltas else None
        out[f"min_demand_delta_vs_{base}_pp"] = min(demand_deltas) if demand_deltas else None
        out[f"max_demand_delta_vs_{base}_pp"] = max(demand_deltas) if demand_deltas else None
    # Marginal prefetcher gain on top of ECG_DBG eviction —
    # the "is the prefetcher doing anything?" headline metric.
    pfx_marg = [c["pfx_marginal_demand_pp"] for c in cells
                if c.get("pfx_marginal_demand_pp") is not None]
    drop_marg = [c["droplet_marginal_demand_pp"] for c in cells
                 if c.get("droplet_marginal_demand_pp") is not None]
    out["mean_pfx_marginal_demand_pp"] = mean(pfx_marg) if pfx_marg else None
    out["mean_droplet_marginal_demand_pp"] = mean(drop_marg) if drop_marg else None
    # Active-cell-only summary (cells where prefetcher actually fired) —
    # avoids diluting the prefetcher claim with no-hint cells (e.g. BC kernel
    # which emits zero hints).
    pfx_active = [c["pfx_marginal_demand_pp"] for c in cells
                  if c.get("pfx_marginal_demand_pp") is not None
                  and (c.get("pfx_requests") or 0) >= 1000]
    drop_active = [c["droplet_marginal_demand_pp"] for c in cells
                   if c.get("droplet_marginal_demand_pp") is not None
                   and (c.get("droplet_requests") or 0) >= 1000]
    out["n_pfx_active_cells"] = len(pfx_active)
    out["n_droplet_active_cells"] = len(drop_active)
    out["mean_pfx_marginal_demand_active_pp"] = mean(pfx_active) if pfx_active else None
    out["mean_droplet_marginal_demand_active_pp"] = mean(drop_active) if drop_active else None
    rates: list[float] = []
    for c in cells:
        fills = c.get("prefetch_fills") or 0
        useful = c.get("prefetch_useful") or 0
        if fills > 0:
            rates.append(useful / fills)
    out["mean_useful_rate"] = mean(rates) if rates else None
    # Efficiency aggregates — total requests/fills/useful summed across cells
    # and req/useful ratio (lower = fewer wasted predictions per useful hit).
    pfx_req = sum((c.get("pfx_requests") or 0) for c in cells)
    pfx_fill = sum((c.get("pfx_fills") or 0) for c in cells)
    pfx_useful = sum((c.get("pfx_useful") or 0) for c in cells)
    drop_req = sum((c.get("droplet_requests") or 0) for c in cells)
    drop_fill = sum((c.get("droplet_fills") or 0) for c in cells)
    drop_useful = sum((c.get("droplet_useful") or 0) for c in cells)
    out["ecg_pfx_total_requests"] = pfx_req
    out["ecg_pfx_total_fills"] = pfx_fill
    out["ecg_pfx_total_useful"] = pfx_useful
    out["ecg_pfx_req_per_useful"] = (pfx_req / pfx_useful) if pfx_useful else None
    out["droplet_total_requests"] = drop_req
    out["droplet_total_fills"] = drop_fill
    out["droplet_total_useful"] = drop_useful
    out["droplet_req_per_useful"] = (drop_req / drop_useful) if drop_useful else None
    if drop_req and pfx_req:
        out["droplet_requests_over_ecg_pfx"] = drop_req / pfx_req
    if drop_useful and pfx_useful:
        out["droplet_useful_over_ecg_pfx"] = drop_useful / pfx_useful
    # Prefetcher demand-memory reduction per million requests —
    # honest efficiency metric (pp of demand-memory reduction per Mreq).
    # Aggregate demand-memory savings across active cells, divided by total
    # requests on those same cells.
    pfx_active_savings_pp = sum(-(c["pfx_marginal_demand_pp"]) for c in cells
                                if c.get("pfx_marginal_demand_pp") is not None
                                and (c.get("pfx_requests") or 0) >= 1000)
    drop_active_savings_pp = sum(-(c["droplet_marginal_demand_pp"]) for c in cells
                                 if c.get("droplet_marginal_demand_pp") is not None
                                 and (c.get("droplet_requests") or 0) >= 1000)
    pfx_active_reqs = sum((c.get("pfx_requests") or 0) for c in cells
                          if (c.get("pfx_requests") or 0) >= 1000)
    drop_active_reqs = sum((c.get("droplet_requests") or 0) for c in cells
                           if (c.get("droplet_requests") or 0) >= 1000)
    out["ecg_pfx_pp_per_mreq"] = (
        pfx_active_savings_pp / (pfx_active_reqs / 1_000_000)
        if pfx_active_reqs else None
    )
    out["droplet_pp_per_mreq"] = (
        drop_active_savings_pp / (drop_active_reqs / 1_000_000)
        if drop_active_reqs else None
    )
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
    if not cells:
        # The ECG_PFX/DROPLET prefetcher sweep (/tmp) is ephemeral and separate
        # from the lit-faith cache-replacement corpus; preserve the committed
        # prefetcher table rather than overwrite it with empty data (which would
        # crash the formatters and regress the table + every downstream consumer,
        # e.g. paper_table_grasp_parity). Refresh requires re-running the
        # prefetcher sweep.
        import sys as _sys
        print(f"[paper-table-prefetcher] no prefetcher cells under "
              f"{args.sweep_root}; preserving committed {args.json_out.name} "
              "(re-run the prefetcher sweep to refresh).", file=_sys.stderr)
        return 0
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
