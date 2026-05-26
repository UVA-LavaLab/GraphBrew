#!/usr/bin/env python3
"""Run one-root ECG_PFX BFS scale-proof shards.

This helper turns the validated manual BFS g6-g10 proof commands into a
reusable local/Slurm entry point. It intentionally stays narrow: BFS only,
LRU only, ECG_PFX at L2, and one or more explicit source roots.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEM5_RISCV = PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt"
GEM5_CONFIG = PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "configs" / "graphbrew" / "graph_se.py"
RISCV_BFS = PROJECT_ROOT / "bench" / "bin_gem5" / "bfs_riscv_m5ops"
ROI_MATRIX = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None, dry_run: bool, timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {command_text(command)}")
    if dry_run:
        return 0
    with log_path.open("w") as log:
        log.write(f"$ {command_text(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        log.write(f"\n[ecg_pfx_scale_exit_code] {completed.returncode}\n")
    return completed.returncode


def bfs_options(scale: int, root: int) -> str:
    return f"-g {scale} -k 8 -o 2 -n 1 -r {root}"


def gem5_env(out_dir: Path) -> dict[str, str]:
    sideband_dir = out_dir / "sidebands"
    env = dict(os.environ)
    env.update({
        "GEM5_GRAPHBREW_CTX": str(sideband_dir / "ctx.json"),
        "GEM5_POPT_MATRIX": str(sideband_dir / "popt.bin"),
        "GEM5_GRAPHBREW_OUT_EDGES": str(sideband_dir / "out_edges.bin"),
        "GEM5_GRAPHBREW_IN_EDGES": str(sideband_dir / "in_edges.bin"),
        "ECG_PREFETCH_MODE": "2",
        "ECG_PREFETCH_WINDOW": "16",
    })
    return env


def gem5_command(scale: int, root: int, out_dir: Path) -> list[str]:
    return [
        str(GEM5_RISCV),
        f"--outdir={out_dir / 'm5out'}",
        str(GEM5_CONFIG),
        "--binary", str(RISCV_BFS),
        "--options", bfs_options(scale, root),
        "--policy", "LRU",
        "--prefetcher", "ECG_PFX",
        "--prefetcher-level", "l2",
        "--ecg-pfx-lookahead", "4",
        "--ecg-pfx-hint-filter", "16",
        "--ecg-pfx-delivery", "instruction",
        "--l1d-size", "1kB",
        "--l2-size", "2kB",
        "--l3-size", "32kB",
        "--l3-ways", "16",
    ]


def sniper_command(scale: int, root: int, out_dir: Path, timeout: int) -> list[str]:
    return [
        sys.executable,
        str(ROI_MATRIX),
        "--suite", "sniper",
        "--benchmark", "bfs",
        "--options", bfs_options(scale, root),
        "--policies", "LRU",
        "--prefetcher", "ECG_PFX",
        "--prefetcher-level", "l2",
        "--l1d-size", "1kB",
        "--l2-size", "2kB",
        "--l3-sizes", "32kB",
        "--l3-ways", "16",
        "--ecg-pfx-mode", "popt",
        "--ecg-pfx-window", "16",
        "--ecg-pfx-lookahead", "4",
        "--ecg-pfx-hint-filter", "16",
        "--sniper-workload", "benchmark",
        "--sniper-frontend", "sift",
        "--allow-sniper-benchmark-workload",
        "--sniper-memory-limit-gb", "4",
        "--sniper-cores", "1",
        "--sniper-address-domain", "virtual",
        "--timeout-sniper", str(timeout),
        "--no-build",
        "--out-dir", str(out_dir),
    ]


def parse_gem5_stats(stats_path: Path, scale: int, root: int, out_dir: Path) -> list[dict[str, Any]]:
    if not stats_path.exists():
        return [{"backend": "gem5-riscv", "scale": scale, "root": root, "section": 0, "status": "missing_stats", "run_dir": str(out_dir)}]
    text = stats_path.read_text(errors="ignore")
    sections = text.split("---------- Begin Simulation Statistics ----------")[1:]
    rows: list[dict[str, Any]] = []
    for section_id, section in enumerate(sections, 1):
        def match(pattern: str) -> str:
            found = re.search(pattern, section, re.M)
            return found.group(1) if found else ""
        rows.append({
            "backend": "gem5-riscv",
            "scale": scale,
            "root": root,
            "section": section_id,
            "status": "ok",
            "pf_identified": match(r"^system\.l2cache\.prefetcher\.pfIdentified\s+([0-9.]+)"),
            "pf_issued": match(r"^system\.l2cache\.prefetcher\.pfIssued\s+([0-9.]+)"),
            "pf_useful": match(r"^system\.l2cache\.prefetcher\.pfUseful\s+([0-9.]+)"),
            "hints": "",
            "ecg_pfx_issued": "",
            "l3_accesses": match(r"^system\.l3cache\.overallAccesses::total\s+([0-9.]+)"),
            "l3_misses": match(r"^system\.l3cache\.overallMisses::total\s+([0-9.]+)"),
            "run_dir": str(out_dir),
            "error": "",
        })
    return rows


def parse_sniper_csv(csv_path: Path, scale: int, root: int, out_dir: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return [{"backend": "sniper-sift", "scale": scale, "root": root, "section": 0, "status": "missing_csv", "run_dir": str(out_dir)}]
    with csv_path.open(newline="") as fh:
        source = list(csv.DictReader(fh))
    rows: list[dict[str, Any]] = []
    for row in source:
        rows.append({
            "backend": "sniper-sift",
            "scale": scale,
            "root": root,
            "section": row.get("section", ""),
            "status": row.get("status", ""),
            "pf_identified": "",
            "pf_issued": row.get("pf_issued", ""),
            "pf_useful": row.get("pf_useful", ""),
            "hints": row.get("ecg_pfx_target_hints_seen", ""),
            "ecg_pfx_issued": row.get("ecg_pfx_issued", ""),
            "l3_accesses": row.get("l3_accesses", ""),
            "l3_misses": row.get("l3_misses", ""),
            "run_dir": str(out_dir),
            "error": row.get("error", ""),
        })
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend", "scale", "root", "section", "status", "pf_identified",
        "pf_issued", "pf_useful", "hints", "ecg_pfx_issued", "l3_accesses",
        "l3_misses", "run_dir", "error",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BFS ECG_PFX scale-proof shards.")
    parser.add_argument("--scale", type=int, required=True, help="Synthetic graph scale, e.g. 10.")
    parser.add_argument("--roots", nargs="+", type=int, required=True, help="BFS root(s) to run.")
    parser.add_argument("--backend", choices=["sniper", "gem5-riscv", "both"], default="both")
    parser.add_argument("--out-root", default="", help="Output root; defaults under /tmp.")
    parser.add_argument("--timeout-sniper", type=int, default=1200)
    parser.add_argument("--timeout-gem5", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out_root = resolve_path(args.out_root) if args.out_root else Path(f"/tmp/graphbrew-ecg-pfx-scale-proof-{now_tag()}")
    rows: list[dict[str, Any]] = []
    failures = 0

    for root in args.roots:
        if args.backend in ("gem5-riscv", "both"):
            gem5_out = out_root / f"gem5_riscv_bfs_g{args.scale}_r{root}"
            command = gem5_command(args.scale, root, gem5_out)
            returncode = run_command(command, gem5_out / "gem5.log", gem5_env(gem5_out), args.dry_run, args.timeout_gem5)
            if returncode != 0:
                failures += 1
            if not args.dry_run:
                rows.extend(parse_gem5_stats(gem5_out / "m5out" / "stats.txt", args.scale, root, gem5_out))

        if args.backend in ("sniper", "both"):
            sniper_out = out_root / f"sniper_bfs_g{args.scale}_r{root}"
            command = sniper_command(args.scale, root, sniper_out, args.timeout_sniper)
            returncode = run_command(command, sniper_out / "sniper.log", None, args.dry_run, args.timeout_sniper + 60)
            if returncode != 0:
                failures += 1
            if not args.dry_run:
                rows.extend(parse_sniper_csv(sniper_out / "roi_matrix.csv", args.scale, root, sniper_out))

    if not args.dry_run:
        write_summary(out_root / "summary.csv", rows)
    return failures


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))