#!/usr/bin/env python3
"""Check ECG proof-matrix claims before ISA work."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "" or value is None:
        return 0
    return int(float(value))


def rel_delta(actual: int, expected: int) -> float:
    return abs(actual - expected) / expected if expected else 0.0


def check(path: Path, tolerance: float) -> int:
    rows = [row for row in csv.DictReader(path.open()) if row.get("status") == "ok"]
    by_benchmark: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_benchmark.setdefault(row["benchmark"], {})[row["ablation"]] = row

    failures: list[str] = []
    for benchmark, table in sorted(by_benchmark.items()):
        required = [
            "LRU_cache_only",
            "GRASP_DBG_only",
            "POPT_only",
            "ECG_DBG_only",
            "ECG_POPT_primary",
            "PFX_POPT_only",
        ]
        missing = [name for name in required if name not in table]
        if missing:
            failures.append(f"{benchmark}: missing rows {', '.join(missing)}")
            continue

        lru = as_int(table["LRU_cache_only"], "memory_accesses")
        grasp = as_int(table["GRASP_DBG_only"], "memory_accesses")
        ecg_dbg = as_int(table["ECG_DBG_only"], "memory_accesses")
        popt = as_int(table["POPT_only"], "memory_accesses")
        ecg_popt = as_int(table["ECG_POPT_primary"], "memory_accesses")
        pfx_popt = as_int(table["PFX_POPT_only"], "memory_accesses")
        pfx_useful = as_int(table["PFX_POPT_only"], "prefetch_useful")

        dbg_delta = rel_delta(ecg_dbg, grasp)
        popt_delta = rel_delta(ecg_popt, popt)
        if dbg_delta > tolerance:
            failures.append(
                f"{benchmark}: ECG_DBG_only diverges from GRASP by {dbg_delta:.2%} "
                f"({ecg_dbg} vs {grasp})"
            )
        if popt_delta > tolerance:
            failures.append(
                f"{benchmark}: ECG_POPT_primary diverges from POPT by {popt_delta:.2%} "
                f"({ecg_popt} vs {popt})"
            )
        if pfx_useful <= 0 or pfx_popt >= lru:
            failures.append(
                f"{benchmark}: PFX_POPT_only did not reduce demand misses usefully "
                f"(demand {pfx_popt} vs LRU {lru}, useful {pfx_useful})"
            )

        cache_rows = [row for row in table.values() if row.get("ablation_group") in ("cache_alone", "ecg_replacement")]
        combined_rows = [row for row in table.values() if row.get("ablation_group") == "combined"]
        if combined_rows and cache_rows:
            best_cache = min(as_int(row, "memory_accesses") for row in cache_rows)
            best_combined = min(as_int(row, "memory_accesses") for row in combined_rows)
            if best_combined >= best_cache:
                failures.append(
                    f"{benchmark}: best combined row does not beat best cache-only/replacement row "
                    f"({best_combined} vs {best_cache})"
                )

        print(
            f"{benchmark}: DBG delta={dbg_delta:.2%}, POPT delta={popt_delta:.2%}, "
            f"PFX useful={pfx_useful}, LRU={lru}"
        )

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("All proof claims passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ECG proof-matrix claims.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Relative tolerance for parity checks.")
    args = parser.parse_args()
    return check(args.csv_path, args.tolerance)


if __name__ == "__main__":
    raise SystemExit(main())