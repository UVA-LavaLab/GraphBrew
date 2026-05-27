#!/usr/bin/env python3
"""Turn ECG proof-matrix rows into explicit validation gate verdicts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "benchmark",
    "gate",
    "status",
    "metric",
    "candidate",
    "candidate_value",
    "baseline",
    "baseline_value",
    "delta_vs_baseline",
    "tolerance",
    "notes",
]

STATUS_ORDER = {"pass": 0, "activation_only": 1, "fail": 2, "missing": 3}


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def rel_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return (baseline - candidate) / baseline


def abs_rel_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return abs(candidate - baseline) / baseline


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return [row for row in csv.DictReader(fh) if row.get("status") == "ok"]


def by_benchmark(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("benchmark", ""), {})[row.get("ablation", "")] = row
    return grouped


def gate_row(
    benchmark: str,
    gate: str,
    status: str,
    metric: str,
    candidate: str,
    candidate_value: float | None,
    baseline: str,
    baseline_value: float | None,
    tolerance: float | None,
    notes: str,
) -> dict[str, str]:
    delta = rel_delta(candidate_value, baseline_value)
    return {
        "benchmark": benchmark,
        "gate": gate,
        "status": status,
        "metric": metric,
        "candidate": candidate,
        "candidate_value": fmt(candidate_value),
        "baseline": baseline,
        "baseline_value": fmt(baseline_value),
        "delta_vs_baseline": fmt(delta),
        "tolerance": "" if tolerance is None else fmt(tolerance),
        "notes": notes,
    }


def missing_gate(benchmark: str, gate: str, missing: list[str], metric: str) -> dict[str, str]:
    return gate_row(
        benchmark,
        gate,
        "missing",
        metric,
        "",
        None,
        "",
        None,
        None,
        "missing rows: " + ", ".join(missing),
    )


def metric_value(table: dict[str, dict[str, str]], label: str, metric: str) -> float | None:
    row = table.get(label)
    if row is None:
        return None
    return number(row.get(metric))


def require(table: dict[str, dict[str, str]], labels: list[str]) -> list[str]:
    return [label for label in labels if label not in table]


def parity_gate(
    benchmark: str,
    table: dict[str, dict[str, str]],
    gate: str,
    candidate: str,
    baseline: str,
    metric: str,
    tolerance: float,
    notes: str,
) -> dict[str, str]:
    missing = require(table, [candidate, baseline])
    if missing:
        return missing_gate(benchmark, gate, missing, metric)
    candidate_value = metric_value(table, candidate, metric)
    baseline_value = metric_value(table, baseline, metric)
    delta = abs_rel_delta(candidate_value, baseline_value)
    status = "pass" if delta is not None and delta <= tolerance else "fail"
    return gate_row(benchmark, gate, status, metric, candidate, candidate_value, baseline, baseline_value, tolerance, notes)


def benefit_gate(
    benchmark: str,
    table: dict[str, dict[str, str]],
    gate: str,
    candidate: str,
    baseline: str,
    metric: str,
    tolerance: float,
    notes: str,
) -> dict[str, str]:
    missing = require(table, [candidate, baseline])
    if missing:
        return missing_gate(benchmark, gate, missing, metric)
    candidate_value = metric_value(table, candidate, metric)
    baseline_value = metric_value(table, baseline, metric)
    delta = rel_delta(candidate_value, baseline_value)
    status = "pass" if delta is not None and delta >= -tolerance else "fail"
    return gate_row(benchmark, gate, status, metric, candidate, candidate_value, baseline, baseline_value, tolerance, notes)


def best_of(table: dict[str, dict[str, str]], labels: list[str], metric: str) -> tuple[str, float | None]:
    best_label = ""
    best_value: float | None = None
    for label in labels:
        value = metric_value(table, label, metric)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_label = label
            best_value = value
    return best_label, best_value


def hybrid_gate(
    benchmark: str,
    table: dict[str, dict[str, str]],
    metric: str,
    tolerance: float,
) -> dict[str, str]:
    missing = require(table, ["ECG_DBG_POPT", "GRASP_DBG_only", "POPT_only"])
    if missing:
        return missing_gate(benchmark, "ecg_hybrid_value", missing, metric)
    baseline, baseline_value = best_of(table, ["GRASP_DBG_only", "POPT_only"], metric)
    candidate_value = metric_value(table, "ECG_DBG_POPT", metric)
    delta = rel_delta(candidate_value, baseline_value)
    status = "pass" if delta is not None and delta >= -tolerance else "fail"
    return gate_row(
        benchmark,
        "ecg_hybrid_value",
        status,
        metric,
        "ECG_DBG_POPT",
        candidate_value,
        baseline,
        baseline_value,
        tolerance,
        "ECG hybrid should match or beat the stronger prior replacement baseline.",
    )


def pfx_gate(
    benchmark: str,
    table: dict[str, dict[str, str]],
    label: str,
    baseline: str,
    metric: str,
    tolerance: float,
) -> dict[str, str]:
    missing = require(table, [label, baseline])
    if missing:
        return missing_gate(benchmark, f"{label}_pfx", missing, metric)
    useful = number(table[label].get("prefetch_useful")) or 0.0
    fills = number(table[label].get("prefetch_fills")) or 0.0
    candidate_value = metric_value(table, label, metric)
    baseline_value = metric_value(table, baseline, metric)
    delta = rel_delta(candidate_value, baseline_value)
    if useful <= 0:
        status = "fail"
        notes = "PFX did not report useful prefetches."
    elif delta is not None and delta >= -tolerance:
        status = "pass"
        notes = f"PFX useful={fmt(useful)}, fills={fmt(fills)} and demand did not regress beyond tolerance."
    else:
        status = "activation_only"
        notes = f"PFX useful={fmt(useful)}, fills={fmt(fills)} but demand regressed beyond tolerance."
    return gate_row(benchmark, f"{label}_pfx", status, metric, label, candidate_value, baseline, baseline_value, tolerance, notes)


def evaluate(rows: list[dict[str, str]], metric: str, parity_tolerance: float, benefit_tolerance: float,
             embedded_tolerance: float) -> list[dict[str, str]]:
    verdicts: list[dict[str, str]] = []
    for benchmark, table in sorted(by_benchmark(rows).items()):
        verdicts.append(parity_gate(
            benchmark, table, "grasp_parity", "ECG_DBG_only", "GRASP_DBG_only",
            metric, parity_tolerance, "ECG DBG_ONLY should match GRASP.",
        ))
        verdicts.append(parity_gate(
            benchmark, table, "popt_parity", "ECG_POPT_primary", "POPT_only",
            metric, parity_tolerance, "ECG POPT_PRIMARY should match pure P-OPT closely.",
        ))
        verdicts.append(parity_gate(
            benchmark, table, "embedded_quality", "ECG_EMBEDDED", "POPT_only",
            metric, embedded_tolerance, "Embedded stored P-OPT hint should stay near pure P-OPT within quantization tolerance.",
        ))
        verdicts.append(benefit_gate(
            benchmark, table, "combined_insertion_quality", "ECG_COMBINED", "LRU_cache_only",
            metric, benefit_tolerance, "Combined stored DBG+P-OPT insertion should not regress badly versus LRU.",
        ))
        verdicts.append(hybrid_gate(benchmark, table, metric, benefit_tolerance))
        verdicts.append(pfx_gate(benchmark, table, "PFX_POPT_only", "LRU_cache_only", metric, benefit_tolerance))
        verdicts.append(pfx_gate(benchmark, table, "DBG_PFX", "ECG_DBG_only", metric, benefit_tolerance))
        verdicts.append(pfx_gate(benchmark, table, "POPT_PFX", "ECG_POPT_primary", metric, benefit_tolerance))
        verdicts.append(pfx_gate(benchmark, table, "DBG_POPT_PFX", "ECG_DBG_POPT", metric, benefit_tolerance))
    return verdicts


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, str]], source: Path) -> str:
    lines = ["# ECG Validation Gate Report", "", f"Source: `{source}`", ""]
    by_bench: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_bench.setdefault(row["benchmark"], []).append(row)
    for benchmark in sorted(by_bench):
        lines.extend([f"## {benchmark}", "", "| Gate | Status | Candidate | Baseline | Delta | Notes |", "|------|--------|-----------|----------|-------|-------|"])
        for row in sorted(by_bench[benchmark], key=lambda item: (STATUS_ORDER.get(item["status"], 9), item["gate"])):
            lines.append(
                f"| `{row['gate']}` | {row['status']} | `{row['candidate']}` {row['candidate_value']} | "
                f"`{row['baseline']}` {row['baseline_value']} | {row['delta_vs_baseline']} | {row['notes']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ECG validation gate verdicts from proof_matrix.csv.")
    parser.add_argument("proof_csv", type=Path)
    parser.add_argument("--metric", default="memory_accesses", choices=["memory_accesses", "total_memory_traffic", "l3_misses"])
    parser.add_argument("--parity-tolerance", type=float, default=0.05)
    parser.add_argument("--benefit-tolerance", type=float, default=0.0)
    parser.add_argument("--embedded-tolerance", type=float, default=0.10)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--fail-on", choices=["fail", "missing", "activation_only", "never"], default="never")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = evaluate(read_rows(args.proof_csv), args.metric, args.parity_tolerance, args.benefit_tolerance, args.embedded_tolerance)
    if args.out_csv:
        write_csv(args.out_csv, rows)
        print(f"[write] {args.out_csv} rows={len(rows)}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown(rows, args.proof_csv))
        print(f"[write] {args.out_md}")
    if not args.out_csv and not args.out_md:
        print(markdown(rows, args.proof_csv), end="")

    if args.fail_on == "never":
        return 0
    bad_statuses = {"fail", "missing"}
    if args.fail_on == "activation_only":
        bad_statuses.add("activation_only")
    elif args.fail_on == "missing":
        bad_statuses = {"missing"}
    elif args.fail_on == "fail":
        bad_statuses = {"fail", "missing"}
    return 1 if any(row["status"] in bad_statuses for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())