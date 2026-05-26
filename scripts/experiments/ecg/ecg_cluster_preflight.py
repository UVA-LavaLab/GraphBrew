#!/usr/bin/env python3
"""Preflight checks for GraphBrew ECG cluster runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any

import final_paper_run
import ecg_graph_staging_status


PROJECT_ROOT = final_paper_run.PROJECT_ROOT
DEFAULT_MANIFEST = final_paper_run.DEFAULT_MANIFEST
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "results" / "graphs"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_path(path: Path, name: str, executable: bool = False) -> CheckResult:
    if not path.exists():
        return CheckResult(name, False, f"missing: {path}")
    if executable and not path.is_file():
        return CheckResult(name, False, f"not a file: {path}")
    if executable and not path.stat().st_mode & 0o111:
        return CheckResult(name, False, f"not executable: {path}")
    return CheckResult(name, True, str(path))


def read_rows(path: Path, columns: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != columns:
            raise SystemExit(f"invalid row {path}:{line_number}: expected {columns} tab-separated fields")
        rows.append(parts)
    return rows


def manifest_graphs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for graphs in manifest.get("graph_sets", {}).values():
        for graph in graphs:
            out[str(graph["name"])] = graph
    return out


def graph_checks(shard_rows: list[list[str]], graph_dir: Path, allow_missing: bool) -> list[CheckResult]:
    manifest = final_paper_run.load_manifest(DEFAULT_MANIFEST)
    graphs = manifest_graphs(manifest)
    names = sorted({row[2] for row in shard_rows})
    checks: list[CheckResult] = []
    for name in names:
        graph = graphs.get(name)
        if graph is None:
            checks.append(CheckResult(f"graph:{name}", False, "not present in manifest graph_sets"))
            continue
        if final_paper_run.graph_uses_synthetic_options(graph):
            checks.append(CheckResult(f"graph:{name}", True, "synthetic graph"))
            continue
        try:
            path = final_paper_run.find_graph_path(graph, graph_dir, allow_missing)
        except SystemExit as exc:
            checks.append(CheckResult(f"graph:{name}", False, str(exc)))
            continue
        exists = path.exists() if path else False
        if exists:
            checks.append(CheckResult(f"graph:{name}", True, str(path)))
        else:
            checks.append(CheckResult(f"graph:{name}", allow_missing, f"missing allowed: {path}"))
    return checks


def staging_checks(profiles: list[str], graph_dir: Path, allow_missing: bool) -> list[CheckResult]:
    manifest = final_paper_run.load_manifest(DEFAULT_MANIFEST)
    checks: list[CheckResult] = []
    for row in ecg_graph_staging_status.staging_rows(manifest, profiles, graph_dir):
        ok = str(row.get("status")) == "ok" or allow_missing
        detail = str(row.get("detail", ""))
        if str(row.get("status")) == "missing" and allow_missing:
            detail = f"missing allowed: {row.get('expected_path')}"
        checks.append(CheckResult(f"staging:{row['graph']}", ok, detail))
    return checks


def check_slurm(require_slurm: bool) -> list[CheckResult]:
    checks = []
    for command in ("sbatch", "squeue", "sacct"):
        found = shutil.which(command)
        checks.append(CheckResult(f"slurm:{command}", bool(found) or not require_slurm, found or "not found"))
    return checks


def check_required_binaries(skip_sniper: bool) -> list[CheckResult]:
    checks = [
        check_path(PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "X86" / "gem5.opt", "gem5-x86", True),
        check_path(PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "RISCV" / "gem5.opt", "gem5-riscv", True),
    ]
    for bench in ("pr", "bfs", "sssp"):
        checks.append(check_path(PROJECT_ROOT / "bench" / "bin_sim" / bench, f"bin_sim:{bench}", True))
        checks.append(check_path(PROJECT_ROOT / "bench" / "bin_gem5" / f"{bench}_m5ops", f"bin_gem5:{bench}_m5ops", True))
        checks.append(check_path(PROJECT_ROOT / "bench" / "bin_gem5" / f"{bench}_riscv_m5ops", f"bin_gem5:{bench}_riscv_m5ops", True))
    if not skip_sniper:
        checks.append(check_path(PROJECT_ROOT / "bench" / "include" / "sniper_sim" / "snipersim" / "run-sniper", "sniper:run-sniper", True))
        checks.append(check_path(PROJECT_ROOT / "bench" / "include" / "sniper_sim" / ".sniper_overlays.json", "sniper:overlays", False))
        for bench in ("pr", "bfs", "sssp"):
            checks.append(check_path(PROJECT_ROOT / "bench" / "bin_sniper" / bench, f"bin_sniper:{bench}", True))
    return checks


def print_checks(checks: list[CheckResult]) -> int:
    failures = 0
    for item in checks:
        status = "ok" if item.ok else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}")
        if not item.ok:
            failures += 1
    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight ECG cluster shard runs.")
    parser.add_argument("--shards", nargs="*", default=[], help="final_paper_run shard TSV(s), 6 columns.")
    parser.add_argument("--scale-shards", nargs="*", default=[], help="ECG_PFX scale-proof shard TSV(s), 4 columns.")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--profile", nargs="+", default=[], help="Profiles for explicit graph staging checks.")
    parser.add_argument("--allow-missing-graphs", action="store_true")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--skip-binaries", action="store_true")
    parser.add_argument("--skip-sniper", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    checks: list[CheckResult] = []
    graph_dir = final_paper_run.resolve_path(str(args.graph_dir))

    if not args.skip_binaries:
        checks.extend(check_required_binaries(bool(args.skip_sniper)))
    checks.extend(check_slurm(bool(args.require_slurm)))

    if args.profile:
        checks.extend(staging_checks([str(profile) for profile in args.profile], graph_dir, bool(args.allow_missing_graphs)))

    for shard in args.shards:
        path = final_paper_run.resolve_path(str(shard))
        checks.append(check_path(path, f"shards:{path.name}"))
        if path.exists():
            rows = read_rows(path, 6)
            checks.append(CheckResult(f"shards:{path.name}:rows", bool(rows), f"{len(rows)} rows"))
            checks.extend(graph_checks(rows, graph_dir, bool(args.allow_missing_graphs)))

    for shard in args.scale_shards:
        path = final_paper_run.resolve_path(str(shard))
        checks.append(check_path(path, f"scale-shards:{path.name}"))
        if path.exists():
            rows = read_rows(path, 4)
            checks.append(CheckResult(f"scale-shards:{path.name}:rows", bool(rows), f"{len(rows)} rows"))

    return 1 if print_checks(checks) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))