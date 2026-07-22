#!/usr/bin/env python3
"""Validate transport-matched Sniper K2-M instruction parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_KERNELS = ("pr", "bfs", "sssp", "bc", "cc")


def validate(
        root: Path, kernels: tuple[str, ...] = DEFAULT_KERNELS,
        tolerance: float = 0.0025) -> list[str]:
    errors: list[str] = []
    for kernel in kernels:
        path = root / kernel / "roi_matrix.csv"
        if not path.exists():
            errors.append(f"{kernel}: missing {path}")
            continue
        rows = list(csv.DictReader(path.open()))
        if len(rows) != 2:
            errors.append(f"{kernel}: expected exactly two rows, got {len(rows)}")
            continue
        by_policy = {row.get("policy_label"): row for row in rows}
        if set(by_policy) != {"LRU", "ECG_K2"}:
            errors.append(
                f"{kernel}: expected LRU/ECG_K2, got {sorted(by_policy)}")
            continue
        lru = by_policy["LRU"]
        k2 = by_policy["ECG_K2"]
        if lru.get("status") != "ok" or k2.get("status") != "ok":
            errors.append(f"{kernel}: non-ok row")
            continue
        if lru.get("sniper_transport_matched") != "1":
            errors.append(f"{kernel}: LRU transport not matched")
        if k2.get("sniper_transport_matched") != "1":
            errors.append(f"{kernel}: K2 transport not matched")
        if k2.get("ecg_isa_variant") != "mask":
            errors.append(f"{kernel}: K2 ISA variant is not mask")
        if lru.get("sniper_workload") != "sg_kernel":
            errors.append(f"{kernel}: LRU workload is not sg_kernel")
        if k2.get("sniper_workload") != "sg_kernel":
            errors.append(f"{kernel}: K2 workload is not sg_kernel")
        if lru.get("sniper_roi_icount") not in ("", "0", None):
            errors.append(f"{kernel}: LRU row is instruction-capped")
        if k2.get("sniper_roi_icount") not in ("", "0", None):
            errors.append(f"{kernel}: K2 row is instruction-capped")
        if lru.get("timing_valid_for_speedup") != "0":
            errors.append(f"{kernel}: LRU matched row is not diagnostic-only")
        if k2.get("timing_valid_for_speedup") != "0":
            errors.append(f"{kernel}: K2 matched row is not diagnostic-only")
        if not lru.get("sniper_workload_sha256"):
            errors.append(f"{kernel}: missing workload hash")
        elif (lru.get("sniper_workload_sha256") !=
              k2.get("sniper_workload_sha256")):
            errors.append(f"{kernel}: workload hashes differ")
        matched_fields = (
            "benchmark", "options", "prefetcher", "l1d_size", "l2_size",
            "l3_size", "l3_ways", "threads", "sniper_cores",
            "sniper_cache_warming", "sniper_transport_record_bytes",
        )
        for field in matched_fields:
            if lru.get(field) != k2.get(field):
                errors.append(f"{kernel}: configuration mismatch in {field}")
        if (not lru.get("sniper_semantic_result") or
                lru.get("sniper_semantic_result") !=
                k2.get("sniper_semantic_result")):
            errors.append(f"{kernel}: semantic results differ or are missing")
        for row in (lru, k2):
            log_path = Path(row.get("log_path", ""))
            text = log_path.read_text(errors="ignore") if log_path.exists() else ""
            if "[K2_TRANSPORT_MATCHED]" not in text:
                errors.append(
                    f"{kernel}/{row.get('policy_label')}: transport marker missing")

        marker_path = root / kernel / "roi_matrix.complete.json"
        json_path = root / kernel / "roi_matrix.json"
        if not marker_path.exists():
            errors.append(f"{kernel}: completion marker missing")
        elif not json_path.exists():
            errors.append(f"{kernel}: roi_matrix.json missing")
        else:
            marker = json.loads(marker_path.read_text())
            if not marker.get("complete") or not marker.get("all_rows_ok"):
                errors.append(f"{kernel}: completion marker is not all-ok")
            json_rows = json.loads(json_path.read_text())
            outputs = marker.get("outputs", {})
            descriptors = (
                ("roi_matrix.csv", path, len(rows)),
                ("roi_matrix.json", json_path, len(json_rows)),
            )
            for name, output_path, row_count in descriptors:
                descriptor = outputs.get(name, {})
                digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
                if (descriptor.get("rows") != row_count or
                        descriptor.get("sha256") != digest):
                    errors.append(
                        f"{kernel}: {name} marker hash/rows mismatch")
        try:
            lru_instructions = int(lru["instructions"])
            k2_instructions = int(k2["instructions"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{kernel}: missing instruction count")
            continue
        ratio = k2_instructions / lru_instructions
        if abs(ratio - 1.0) > tolerance:
            errors.append(
                f"{kernel}: instruction ratio {ratio:.6f} exceeds "
                f"{tolerance:.4%} tolerance")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument("--tolerance", type=float, default=0.0025)
    args = parser.parse_args()
    errors = validate(
        args.root, tuple(args.kernels), float(args.tolerance))
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        f"[PASS] matched K2-M instruction parity: "
        f"{len(args.kernels)} kernels within {args.tolerance:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
