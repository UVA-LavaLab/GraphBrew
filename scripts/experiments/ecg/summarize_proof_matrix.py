#!/usr/bin/env python3
"""Summarize ECG proof_matrix.csv component ablations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


GROUP_ORDER = ["cache_alone", "ecg_replacement", "pfx_only", "combined"]


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "" or value is None:
        return 0
    return int(float(value))


def fmt_delta(value: int) -> str:
    return f"{value:+d}"


def summarize(path: Path) -> str:
    rows = [row for row in csv.DictReader(path.open()) if row.get("status") == "ok"]
    by_benchmark: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_benchmark.setdefault(row["benchmark"], []).append(row)

    lines: list[str] = []
    lines.append(f"# ECG Proof Matrix Summary")
    lines.append("")
    lines.append(f"Source: `{path}`")
    lines.append("")

    for benchmark in sorted(by_benchmark):
        bench_rows = by_benchmark[benchmark]
        baseline = next((row for row in bench_rows if row["ablation"] == "LRU_cache_only"), None)
        if baseline is None:
            continue
        base_mem = as_int(baseline, "memory_accesses")
        base_traffic = as_int(baseline, "total_memory_traffic")
        lines.append(f"## {benchmark}")
        lines.append("")
        lines.append(f"LRU baseline: demand `{base_mem:,}`, traffic `{base_traffic:,}`.")
        lines.append("")
        lines.append("| Group | Best Demand Row | Demand | Delta | Best Traffic Row | Traffic | Delta |")
        lines.append("|-------|-----------------|--------|-------|------------------|---------|-------|")
        for group in GROUP_ORDER:
            group_rows = [row for row in bench_rows if row.get("ablation_group") == group]
            if not group_rows:
                continue
            best_mem = min(group_rows, key=lambda row: as_int(row, "memory_accesses"))
            best_traffic = min(group_rows, key=lambda row: as_int(row, "total_memory_traffic"))
            mem = as_int(best_mem, "memory_accesses")
            traffic = as_int(best_traffic, "total_memory_traffic")
            lines.append(
                "| "
                f"{group} | `{best_mem['ablation']}` | {mem:,} | {fmt_delta(mem - base_mem)} | "
                f"`{best_traffic['ablation']}` | {traffic:,} | {fmt_delta(traffic - base_traffic)} |"
            )
        lines.append("")
        lines.append("| Ablation | Group | Demand | Traffic | Fills | Useful | Useful Rate |")
        lines.append("|----------|-------|--------|---------|-------|--------|-------------|")
        for row in sorted(bench_rows, key=lambda row: as_int(row, "memory_accesses")):
            fills = row.get("prefetch_fills") or "0"
            useful = row.get("prefetch_useful") or "0"
            useful_rate = row.get("prefetch_fill_useful_rate") or ""
            lines.append(
                "| "
                f"`{row['ablation']}` | {row.get('ablation_group', '')} | "
                f"{as_int(row, 'memory_accesses'):,} | "
                f"{as_int(row, 'total_memory_traffic'):,} | "
                f"{fills} | {useful} | {useful_rate} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ECG proof_matrix.csv.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize(args.csv_path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
        print(args.output)
    else:
        print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())