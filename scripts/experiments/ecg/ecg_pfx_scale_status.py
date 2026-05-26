#!/usr/bin/env python3
"""Summarize ECG_PFX scale-proof shard status."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
from typing import Any

import final_paper_run
from make_ecg_pfx_scale_shards import ScaleShardRow


STATUS_FIELDNAMES = [
    "scale",
    "root",
    "backend",
    "out_root",
    "status",
    "detail",
    "summary_rows",
    "backends_seen",
    "pf_issued_total",
    "pf_useful_total",
    "hints_total",
    "ecg_pfx_issued_total",
]


def read_shards(path: Path) -> list[ScaleShardRow]:
    rows: list[ScaleShardRow] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise SystemExit(f"invalid scale shard row {path}:{line_number}: expected 4 tab-separated fields")
        rows.append(ScaleShardRow(scale=int(parts[0]), root=int(parts[1]), backend=parts[2], out_root=parts[3]))
    return rows


def expected_backends(backend: str) -> set[str]:
    if backend == "both":
        return {"gem5-riscv", "sniper-sift"}
    if backend == "sniper":
        return {"sniper-sift"}
    if backend == "gem5-riscv":
        return {"gem5-riscv"}
    return {backend}


def to_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_shard(row: ScaleShardRow) -> tuple[dict[str, Any], list[dict[str, str]]]:
    out_root = final_paper_run.resolve_path(row.out_root)
    summary_path = out_root / "summary.csv"
    record: dict[str, Any] = {
        "scale": row.scale,
        "root": row.root,
        "backend": row.backend,
        "out_root": str(out_root),
        "status": "not_started",
        "detail": "out_root missing",
        "summary_rows": 0,
        "backends_seen": "",
        "pf_issued_total": 0,
        "pf_useful_total": 0,
        "hints_total": 0,
        "ecg_pfx_issued_total": 0,
    }
    if not out_root.exists():
        return record, []
    if not summary_path.exists():
        record.update({"status": "pending", "detail": "summary.csv missing"})
        return record, []

    rows = read_summary(summary_path)
    record["summary_rows"] = len(rows)
    if not rows:
        record.update({"status": "failed", "detail": "summary.csv has no rows"})
        return record, rows

    backends_seen = {str(item.get("backend", "")) for item in rows if item.get("backend")}
    missing = expected_backends(row.backend) - backends_seen
    record["backends_seen"] = ";".join(sorted(backends_seen))
    record["pf_issued_total"] = sum(to_int(item.get("pf_issued")) for item in rows)
    record["pf_useful_total"] = sum(to_int(item.get("pf_useful")) for item in rows)
    record["hints_total"] = sum(to_int(item.get("hints")) for item in rows)
    record["ecg_pfx_issued_total"] = sum(to_int(item.get("ecg_pfx_issued")) for item in rows)

    statuses = Counter(str(item.get("status", "")) or "unknown" for item in rows)
    if missing:
        record.update({"status": "incomplete", "detail": f"missing backends={sorted(missing)}"})
    elif any(status != "ok" for status in statuses):
        record.update({"status": "failed", "detail": ";".join(f"{key}:{value}" for key, value in sorted(statuses.items()))})
    elif int(record["pf_issued_total"]) <= 0:
        record.update({"status": "no_issue", "detail": "no prefetches issued"})
    elif int(record["pf_useful_total"]) <= 0:
        record.update({"status": "no_useful", "detail": "prefetches issued but none useful"})
    else:
        record.update({"status": "ok", "detail": f"{len(rows)} summary row(s)"})
    return record, rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ECG_PFX scale-proof shard status.")
    parser.add_argument("--shards", required=True, help="Scale-proof shard TSV from make_ecg_pfx_scale_shards.py.")
    parser.add_argument("--out", default="-", help="Output status CSV path, or '-' for stdout.")
    parser.add_argument("--combined", default="", help="Optional combined summary CSV output path.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit nonzero for not_started/pending/incomplete shards.")
    parser.add_argument("--fail-on-failed", action="store_true", help="Exit nonzero for failed/no_issue/no_useful shards.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    shard_path = final_paper_run.resolve_path(str(args.shards))
    status_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for shard in read_shards(shard_path):
        status, summary_rows = summarize_shard(shard)
        status_rows.append(status)
        for summary in summary_rows:
            combined = dict(summary)
            combined.update({
                "shard_scale": shard.scale,
                "shard_root": shard.root,
                "shard_backend": shard.backend,
                "shard_out_root": str(final_paper_run.resolve_path(shard.out_root)),
            })
            combined_rows.append(combined)

    counts = Counter(str(row["status"]) for row in status_rows)
    if args.out == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=STATUS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(status_rows)
    else:
        out_path = final_paper_run.resolve_path(str(args.out))
        write_csv(out_path, status_rows, STATUS_FIELDNAMES)
        print(f"[write] {out_path} rows={len(status_rows)} status_counts={dict(sorted(counts.items()))}")

    if args.combined:
        combined_path = final_paper_run.resolve_path(str(args.combined))
        write_csv(combined_path, combined_rows)
        print(f"[write] {combined_path} rows={len(combined_rows)}")

    if args.fail_on_failed and any(counts.get(status, 0) for status in ("failed", "no_issue", "no_useful")):
        return 2
    if args.fail_on_missing and any(counts.get(status, 0) for status in ("not_started", "pending", "incomplete")):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))