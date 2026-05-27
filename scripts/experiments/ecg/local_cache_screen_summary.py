#!/usr/bin/env python3
"""Summarize local cache_sim diversity screen CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import final_paper_run


FIELDNAMES = [
    "source",
    "benchmark",
    "prefetcher",
    "l3_size",
    "policy_label",
    "status",
    "l3_misses",
    "l3_rank",
    "l3_delta_vs_lru",
    "prefetch_requests",
    "prefetch_useful",
    "prefetch_useful_per_request",
    "timing_valid_for_speedup",
]


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def input_label(path: Path, explicit_label: str | None = None) -> str:
    if explicit_label:
        return explicit_label
    parent = path.parent.name
    if parent:
        return parent
    return path.stem


def parse_input(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        return label, final_paper_run.resolve_path(path_text)
    return None, final_paper_run.resolve_path(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_rows(label: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped[(str(row.get("benchmark", "")), str(row.get("prefetcher", "")), str(row.get("l3_size", "")))].append(row)

    out: list[dict[str, Any]] = []
    for (benchmark, prefetcher, l3_size), group_rows in sorted(grouped.items()):
        lru_row = next((row for row in group_rows if row.get("policy_label") == "LRU"), None)
        lru_misses = number(lru_row.get("l3_misses")) if lru_row else None
        ranked = sorted(
            group_rows,
            key=lambda row: (
                number(row.get("l3_misses")) if number(row.get("l3_misses")) is not None else float("inf"),
                str(row.get("policy_label", "")),
            ),
        )
        ranks = {id(row): index + 1 for index, row in enumerate(ranked)}
        for row in sorted(group_rows, key=lambda item: str(item.get("policy_label", ""))):
            l3_misses = number(row.get("l3_misses"))
            requests = number(row.get("prefetch_requests")) or 0.0
            useful = number(row.get("prefetch_useful")) or 0.0
            delta = None
            if lru_misses not in (None, 0) and l3_misses is not None:
                delta = (lru_misses - l3_misses) / lru_misses
            useful_rate = useful / requests if requests else None
            out.append({
                "source": label,
                "benchmark": benchmark,
                "prefetcher": prefetcher,
                "l3_size": l3_size,
                "policy_label": row.get("policy_label", ""),
                "status": row.get("status", ""),
                "l3_misses": format_number(l3_misses),
                "l3_rank": ranks[id(row)],
                "l3_delta_vs_lru": format_number(delta),
                "prefetch_requests": format_number(requests),
                "prefetch_useful": format_number(useful),
                "prefetch_useful_per_request": format_number(useful_rate),
                "timing_valid_for_speedup": row.get("timing_valid_for_speedup", ""),
            })
    return out


def write_csv(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = path.open("w", newline="")
        close = True
    else:
        fh = sys.stdout
        close = False
    try:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if close:
            fh.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local cache_sim diversity screen CSVs.")
    parser.add_argument("--input", action="append", nargs="+", required=True, help="Input combined_roi_matrix.csv path(s), optionally label=path. Can be repeated.")
    parser.add_argument("--out", default="-", help="Output summary CSV path, or '-' for stdout.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    summaries: list[dict[str, Any]] = []
    input_values = [item for group in args.input for item in group]
    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for value in input_values:
        label, path = parse_input(str(value))
        rows_by_label[input_label(path, label)].extend(read_rows(path))
    for label, rows in sorted(rows_by_label.items()):
        summaries.extend(summarize_rows(label, rows))
    out_path = None if args.out == "-" else final_paper_run.resolve_path(str(args.out))
    write_csv(out_path, summaries)
    if out_path is not None:
        print(f"[write] {out_path} rows={len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
