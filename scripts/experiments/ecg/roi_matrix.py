#!/usr/bin/env python3
"""
ROI-scoped cache_sim/gem5 policy matrix for ECG validation.

This runner is intentionally small and explicit. It compares the fast,
accurate cache simulator against ROI-scoped gem5 runs at the same policy scope:
L1/L2 stay LRU and the tested replacement policy is applied to L3.

Default workload is the synthetic PR stress point used during validation:
    -g 10 -k 16 -o 5 -n 1 -i 5

Examples:
    python3 scripts/experiments/ecg/roi_matrix.py --dry-run

    python3 scripts/experiments/ecg/roi_matrix.py \
        --suite cache-sim --policies LRU GRASP POPT ECG:POPT_PRIMARY

    python3 scripts/experiments/ecg/roi_matrix.py \
        --suite gem5 --policies LRU GRASP POPT ECG:POPT_PRIMARY \
        --l3-sizes 32kB
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "ecg_experiments" / "roi_matrix"

GEM5_OPT = PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "X86" / "gem5.opt"
GEM5_CONFIG = PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "configs" / "graphbrew" / "graph_se.py"
GEM5_RUNTIME_SIDEBAND_FILES = (
    Path(os.environ.get("GEM5_GRAPHBREW_CTX", "/tmp/gem5_graphbrew_ctx.json")),
    Path(os.environ.get("GEM5_POPT_MATRIX", "/tmp/gem5_popt_matrix.bin")),
    Path(os.environ.get("GEM5_GRAPHBREW_OUT_EDGES", "/tmp/gem5_graphbrew_out_edges.bin")),
    Path(os.environ.get("GEM5_GRAPHBREW_IN_EDGES", "/tmp/gem5_graphbrew_in_edges.bin")),
)

DEFAULT_POLICIES = [
    "LRU", "SRRIP", "GRASP", "POPT_CHARGED", "POPT",
    "ECG:DBG_ONLY", "ECG:DBG_PRIMARY_CHARGED", "ECG:DBG_PRIMARY",
    "ECG:POPT_PRIMARY",
]
ALL_POLICIES = [
    "LRU",
    "SRRIP",
    "GRASP",
    "POPT_CHARGED",
    "POPT",
    "ECG:DBG_ONLY",
    "ECG:DBG_PRIMARY_CHARGED",
    "ECG:DBG_PRIMARY",
    "ECG:POPT_PRIMARY",
    "ECG:ECG_EMBEDDED",
    "ECG:ECG_COMBINED",
]

GEM5_STAT_KEYS = {
    "sim_ticks": "simTicks",
    "ipc": "system.cpu.ipc",
    "l1_miss_rate": "system.cpu.dcache.overallMissRate::total",
    "l2_miss_rate": "system.l2cache.overallMissRate::total",
    "l3_miss_rate": "system.l3cache.overallMissRate::total",
    "l1_misses": "system.cpu.dcache.overallMisses::total",
    "l2_misses": "system.l2cache.overallMisses::total",
    "l3_misses": "system.l3cache.overallMisses::total",
    "l1_accesses": "system.cpu.dcache.overallAccesses::total",
    "l2_accesses": "system.l2cache.overallAccesses::total",
    "l3_accesses": "system.l3cache.overallAccesses::total",
}

GEM5_PREFETCH_STAT_KEYS = {
    "pf_issued": "pfIssued",
    "pf_useful": "pfUseful",
    "pf_useful_but_miss": "pfUsefulButMiss",
    "pf_unused": "pfUnused",
    "pf_late": "pfLate",
    "pf_identified": "pfIdentified",
    "pf_hit_in_cache": "pfHitInCache",
    "pf_hit_in_mshr": "pfHitInMSHR",
    "pf_hit_in_wb": "pfHitInWB",
    "pf_in_cache": "pfInCache",
    "pf_removed_demand": "pfRemovedDemand",
    "pf_removed_full": "pfRemovedFull",
    "pf_span_page": "pfSpanPage",
    "pf_useful_span_page": "pfUsefulSpanPage",
}


def parse_gem5_number(text: str) -> int | float:
    return float(text) if "." in text else int(text)


@dataclass(frozen=True)
class PolicySpec:
    label: str
    policy: str
    ecg_mode: str | None = None
    charge_popt_overhead: bool = False

    @property
    def safe_label(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.label)


def parse_policy_spec(text: str) -> PolicySpec:
    raw = text.strip()
    upper = raw.upper().replace("-", "_")
    charge_popt = False
    if upper.endswith("_CHARGED"):
        upper = upper[: -len("_CHARGED")]
        charge_popt = True
    elif upper.endswith(":CHARGED"):
        upper = upper[: -len(":CHARGED")]
        charge_popt = True

    if upper.startswith("ECG:"):
        mode = upper.split(":", 1)[1]
        label = f"ECG_{mode}" + ("_CHARGED" if charge_popt else "")
        return PolicySpec(label=label, policy="ECG", ecg_mode=mode,
                          charge_popt_overhead=charge_popt)
    if upper.startswith("ECG_") and upper != "ECG":
        mode = upper.split("ECG_", 1)[1]
        label = f"ECG_{mode}" + ("_CHARGED" if charge_popt else "")
        return PolicySpec(label=label, policy="ECG", ecg_mode=mode,
                          charge_popt_overhead=charge_popt)
    if upper in ("P_OPT", "P-OPT"):
        return PolicySpec(label="POPT_CHARGED" if charge_popt else "POPT",
                          policy="POPT", charge_popt_overhead=charge_popt)
    if upper == "POPT":
        return PolicySpec(label="POPT_CHARGED" if charge_popt else "POPT",
                          policy="POPT", charge_popt_overhead=charge_popt)
    return PolicySpec(label=upper, policy=upper, charge_popt_overhead=charge_popt)


def parse_size_bytes(size: str | int) -> int:
    if isinstance(size, int):
        return size
    text = str(size).strip()
    match = re.fullmatch(r"([0-9]+)\s*([A-Za-z]*)", text)
    if not match:
        raise ValueError(f"invalid size: {size!r}")
    value = int(match.group(1))
    suffix = match.group(2).lower()
    if suffix in ("", "b"):
        return value
    if suffix in ("k", "kb", "kib"):
        return value * 1024
    if suffix in ("m", "mb", "mib"):
        return value * 1024 * 1024
    if suffix in ("g", "gb", "gib"):
        return value * 1024 * 1024 * 1024
    raise ValueError(f"invalid size suffix in {size!r}")


def format_size_bytes(size_bytes: int) -> str:
    return f"{int(size_bytes)}B"


def graph_vertices_from_sg(path: Path) -> int | None:
    try:
        data = path.read_bytes()[:17]
    except OSError:
        return None
    if len(data) < 17:
        return None
    # GAPBS serialized graph header: bool directed, int64 edges, int64 vertices.
    return int(struct.unpack_from("<q", data, 1 + 8)[0])


def graph_vertices_from_mtx(path: Path) -> int | None:
    try:
        with path.open("r", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    return max(int(parts[0]), int(parts[1]))
    except (OSError, ValueError):
        return None
    return None


def estimate_num_vertices(options: str) -> int | None:
    parts = shlex.split(options)
    for index, part in enumerate(parts):
        if part == "-g" and index + 1 < len(parts):
            return 1 << int(parts[index + 1])
        match = re.fullmatch(r"-g([0-9]+)", part)
        if match:
            return 1 << int(match.group(1))
        if part == "-f" and index + 1 < len(parts):
            path = Path(parts[index + 1])
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            suffix = path.suffix.lower()
            if suffix in (".sg", ".wsg"):
                return graph_vertices_from_sg(path)
            if suffix == ".mtx":
                return graph_vertices_from_mtx(path)
    return None


def popt_charge_metadata(args: argparse.Namespace, spec: PolicySpec, l3_size: str) -> dict[str, Any]:
    requested_bytes = parse_size_bytes(l3_size)
    metadata: dict[str, Any] = {
        "popt_overhead_charged": int(spec.charge_popt_overhead),
        "popt_requested_l3_size": l3_size,
        "popt_effective_l3_size": l3_size,
        "popt_effective_l3_ways": args.l3_ways,
        "popt_reserved_ways": 0,
        "popt_reserved_bytes": 0,
        "popt_matrix_active_columns": 0,
        "popt_matrix_column_bytes": 0,
        "popt_matrix_stream_bytes": 0,
        "popt_matrix_stream_cache_lines": 0,
        "popt_estimated_vertices": "",
    }
    if not spec.charge_popt_overhead:
        return metadata

    vertices = estimate_num_vertices(args.options)
    if not vertices:
        metadata["popt_charge_warning"] = "could_not_estimate_vertices"
        return metadata

    line_size = parse_size_bytes(args.line_size)
    assoc = max(int(args.l3_ways), 1)
    property_bytes = max(int(args.popt_property_bytes), 1)
    active_columns = max(int(args.popt_active_columns), 1)
    num_epochs = max(int(args.popt_num_epochs), 1)
    min_data_ways = max(min(int(args.popt_min_data_ways), assoc), 1)

    num_cache_lines = (vertices * property_bytes + line_size - 1) // line_size
    column_bytes = num_cache_lines
    matrix_bytes = active_columns * column_bytes
    sets = max(requested_bytes // (assoc * line_size), 1)
    bytes_per_way = sets * line_size
    reserved_ways = (matrix_bytes + bytes_per_way - 1) // bytes_per_way
    reserved_ways = max(1, min(reserved_ways, assoc - min_data_ways))
    reserved_bytes = reserved_ways * bytes_per_way
    effective_ways = max(assoc - reserved_ways, min_data_ways)
    effective_bytes = sets * effective_ways * line_size
    stream_bytes = num_epochs * column_bytes
    stream_cache_lines = (stream_bytes + line_size - 1) // line_size

    metadata.update({
        "popt_effective_l3_size": format_size_bytes(effective_bytes),
        "popt_effective_l3_ways": str(effective_ways),
        "popt_reserved_ways": reserved_ways,
        "popt_reserved_bytes": reserved_bytes,
        "popt_matrix_active_columns": active_columns,
        "popt_matrix_column_bytes": column_bytes,
        "popt_matrix_stream_bytes": stream_bytes,
        "popt_matrix_stream_cache_lines": stream_cache_lines,
        "popt_estimated_vertices": vertices,
    })
    return metadata


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    timeout: int,
    stdout_path: Path,
    dry_run: bool,
) -> subprocess.CompletedProcess[str] | None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(shlex.quote(part) for part in cmd)
    stdout_path.with_suffix(stdout_path.suffix + ".cmd").write_text(command_text + "\n")

    if dry_run:
        print(f"[dry-run] {command_text}")
        return None

    start = time.time()
    with stdout_path.open("w") as out:
        out.write(f"$ {command_text}\n")
        out.flush()
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        out.write(f"\n[exit_code] {result.returncode}\n")
        out.write(f"[elapsed_s] {time.time() - start:.3f}\n")
    return result


def clear_runtime_sideband_files() -> None:
    for path in GEM5_RUNTIME_SIDEBAND_FILES:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def build_targets(args: argparse.Namespace) -> None:
    if args.no_build or args.dry_run:
        return

    targets = []
    if args.suite in ("cache-sim", "both"):
        targets.append(f"sim-{args.benchmark}")
    if args.suite in ("gem5", "both"):
        targets.append(f"gem5-m5ops-{args.benchmark}")

    for target in targets:
        print(f"[build] make {target}")
        subprocess.run(["make", target], cwd=str(PROJECT_ROOT), check=True)


def cache_size_env(size: str, ways: str) -> tuple[str, str]:
    return size, ways


def parse_ecg_log_stats(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    stats: dict[str, Any] = {}
    pattern = re.compile(r"(build_s|vertices|pfx_candidates|pfx_encoded|pfx_no_candidate|pfx_table_miss|pfx_dedup_skips|runtime_no_target|runtime_duplicate|runtime_issued)=([0-9.]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        if not line.startswith("ECG Mask Stats:"):
            continue
        for key, value in pattern.findall(line):
            out_key = f"ecg_{key}"
            stats[out_key] = float(value) if "." in value else int(value)
    return stats


def cache_sim_env(args: argparse.Namespace, spec: PolicySpec, effective_l3_size: str,
                  effective_l3_ways: str, json_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CACHE_ULTRAFAST": "0",
        "CACHE_POLICY": spec.policy,
        "CACHE_L1_POLICY": "LRU",
        "CACHE_L2_POLICY": "LRU",
        "CACHE_L3_POLICY": spec.policy,
        "CACHE_L1_SIZE": args.l1d_size,
        "CACHE_L1_WAYS": args.l1d_ways,
        "CACHE_L2_SIZE": args.l2_size,
        "CACHE_L2_WAYS": args.l2_ways,
        "CACHE_L3_SIZE": effective_l3_size,
        "CACHE_L3_WAYS": effective_l3_ways,
        "CACHE_LINE_SIZE": args.line_size,
        "CACHE_OUTPUT_JSON": str(json_path),
    })
    if spec.ecg_mode:
        env["ECG_MODE"] = spec.ecg_mode
    return env


def run_cache_sim(args: argparse.Namespace, out_dir: Path, spec: PolicySpec, l3_size: str) -> list[dict[str, Any]]:
    binary = PROJECT_ROOT / "bench" / "bin_sim" / args.benchmark
    label = f"cache_sim_{args.benchmark}_{spec.safe_label}_L3{sanitize(l3_size)}"
    json_path = out_dir / "cache_sim" / f"{label}.json"
    log_path = out_dir / "logs" / f"{label}.log"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary)] + shlex.split(args.options)
    charge = popt_charge_metadata(args, spec, l3_size)
    effective_l3_size = str(charge["popt_effective_l3_size"])
    effective_l3_ways = str(charge["popt_effective_l3_ways"])
    env = cache_sim_env(args, spec, effective_l3_size, effective_l3_ways, json_path)

    result = run_command(cmd, PROJECT_ROOT, env, args.timeout_cache, log_path, args.dry_run)
    if args.dry_run:
        return []

    row = base_row("cache_sim", args, spec, l3_size, charge)
    row.update({"section": 0, "log_path": str(log_path), "json_path": str(json_path)})
    if result is None or result.returncode != 0:
        row.update({"status": "error", "error": f"exit_code={result.returncode if result else 'unknown'}"})
        return [row]
    if not json_path.exists():
        row.update({"status": "error", "error": "missing cache_sim json"})
        return [row]

    data = json.loads(json_path.read_text())
    row.update({
        "status": "ok",
        "total_accesses": data.get("total_accesses"),
        "memory_accesses": data.get("memory_accesses"),
    })
    for key in (
        "prefetch_requests",
        "prefetch_cache_hits",
        "prefetch_fills",
        "prefetch_useful",
        "prefetch_evicted_before_use",
        "prefetch_pending",
        "total_memory_traffic",
    ):
        row[key] = data.get(key)
    fills = row.get("prefetch_fills") or 0
    requests = row.get("prefetch_requests") or 0
    if fills:
        row["prefetch_fill_useful_rate"] = (row.get("prefetch_useful") or 0) / fills
    if requests:
        row["prefetch_request_fill_rate"] = fills / requests
        row["prefetch_request_cache_hit_rate"] = (row.get("prefetch_cache_hits") or 0) / requests
    for level in ("L1", "L2", "L3"):
        stats = data.get(level, {})
        prefix = level.lower()
        hit_rate = stats.get("hit_rate")
        row[f"{prefix}_hit_rate"] = hit_rate
        row[f"{prefix}_miss_rate"] = None if hit_rate is None else 1.0 - float(hit_rate)
        row[f"{prefix}_hits"] = stats.get("hits")
        row[f"{prefix}_misses"] = stats.get("misses")
        row[f"{prefix}_policy"] = stats.get("policy")
    if spec.charge_popt_overhead:
        stream_lines = int(row.get("popt_matrix_stream_cache_lines") or 0)
        traffic = row.get("total_memory_traffic")
        if traffic not in (None, ""):
            row["popt_charged_total_memory_traffic"] = int(traffic) + stream_lines
    row.update(parse_ecg_log_stats(log_path))
    return [row]


def run_gem5(args: argparse.Namespace, out_dir: Path, spec: PolicySpec, l3_size: str) -> list[dict[str, Any]]:
    binary = PROJECT_ROOT / "bench" / "bin_gem5" / f"{args.benchmark}_m5ops"
    label = f"gem5_{args.benchmark}_{spec.safe_label}_L3{sanitize(l3_size)}"
    gem5_out = out_dir / "gem5" / label
    log_path = out_dir / "logs" / f"{label}.log"
    charge = popt_charge_metadata(args, spec, l3_size)
    effective_l3_size = str(charge["popt_effective_l3_size"])
    effective_l3_ways = str(charge["popt_effective_l3_ways"])

    cmd = [
        str(GEM5_OPT),
        f"--outdir={gem5_out}",
        str(GEM5_CONFIG),
        "--binary", str(binary),
        "--options", args.options,
        "--policy", spec.policy,
        "--prefetcher", args.prefetcher,
        "--prefetcher-level", args.prefetcher_level,
        "--l1d-size", args.l1d_size,
        "--l2-size", args.l2_size,
        "--l3-size", effective_l3_size,
        "--l3-ways", effective_l3_ways,
    ]
    if spec.ecg_mode:
        cmd.extend(["--ecg-mode", spec.ecg_mode])

    if not args.dry_run:
        clear_runtime_sideband_files()

    result = run_command(cmd, PROJECT_ROOT, None, args.timeout_gem5, log_path, args.dry_run)
    if args.dry_run:
        return []

    base = base_row("gem5", args, spec, l3_size, charge)
    base.update({"log_path": str(log_path), "gem5_out": str(gem5_out)})
    if result is None or result.returncode != 0:
        base.update({"section": 0, "status": "error", "error": f"exit_code={result.returncode if result else 'unknown'}"})
        return [base]

    stats_path = gem5_out / "stats.txt"
    if not stats_path.exists():
        base.update({"section": 0, "status": "error", "error": "missing stats.txt"})
        return [base]

    sections = parse_gem5_sections(stats_path)
    if not sections:
        base.update({"section": 0, "status": "error", "error": "no stats sections"})
        return [base]

    rows = []
    for section_id, stats in enumerate(sections, 1):
        row = dict(base)
        row.update({"section": section_id, "status": "ok", "stats_path": str(stats_path)})
        row.update(stats)
        if spec.charge_popt_overhead:
            stream_lines = int(row.get("popt_matrix_stream_cache_lines") or 0)
            l3_misses = row.get("l3_misses")
            if l3_misses not in (None, ""):
                row["popt_charged_l3_misses_plus_matrix_stream"] = int(l3_misses) + stream_lines
        rows.append(row)
    return rows


def parse_gem5_sections(stats_path: Path) -> list[dict[str, Any]]:
    text = stats_path.read_text(errors="replace")
    raw_sections = text.split("---------- Begin Simulation Statistics ----------")[1:]
    parsed = []
    for section in raw_sections:
        stats: dict[str, Any] = {}
        for out_key, gem5_key in GEM5_STAT_KEYS.items():
            match = re.search(rf"{re.escape(gem5_key)}\s+([0-9.]+)", section)
            if not match:
                continue
            value = match.group(1)
            stats[out_key] = parse_gem5_number(value)
        for out_key, stat_name in GEM5_PREFETCH_STAT_KEYS.items():
            match = re.search(rf"system\.(?:l2cache|cpu\.dcache)\.prefetcher\.{re.escape(stat_name)}\s+([0-9.]+)", section)
            if not match:
                continue
            value = match.group(1)
            stats[out_key] = parse_gem5_number(value)
        parsed.append(stats)
    return parsed


def base_row(simulator: str, args: argparse.Namespace, spec: PolicySpec, l3_size: str,
             charge: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "simulator": simulator,
        "benchmark": args.benchmark,
        "options": args.options,
        "prefetcher": args.prefetcher,
        "prefetcher_level": args.prefetcher_level,
        "ecg_prefetch_mode": os.environ.get("ECG_PREFETCH_MODE", ""),
        "ecg_prefetch_window": os.environ.get("ECG_PREFETCH_WINDOW", ""),
        "ecg_prefetch_lookahead": os.environ.get("ECG_PREFETCH_LOOKAHEAD", ""),
        "policy_label": spec.label,
        "policy": spec.policy,
        "ecg_mode": spec.ecg_mode or "",
        "l1d_size": args.l1d_size,
        "l2_size": args.l2_size,
        "l3_size": l3_size,
        "l3_ways": args.l3_ways,
        "l1_l2_policy": "LRU",
    }
    if charge:
        row.update(charge)
    return row


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "", text)


def write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "roi_matrix.json"
    csv_path = out_dir / "roi_matrix.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[write] {json_path}")
    print(f"[write] {csv_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched cache_sim/gem5 ROI policy matrix for ECG validation."
    )
    parser.add_argument("--suite", choices=["cache-sim", "gem5", "both"], default="both")
    parser.add_argument("--benchmark", default="pr")
    parser.add_argument("--options", default="-g 10 -k 16 -o 5 -n 1 -i 5")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--all-policies", action="store_true", help="Use the full ECG validation policy set.")
    parser.add_argument("--prefetcher", choices=["none", "DROPLET"], default="none",
                        help="gem5 prefetcher to attach; ignored by cache_sim.")
    parser.add_argument("--prefetcher-level", choices=["l1d", "l2"], default="l2",
                        help="gem5 cache level for --prefetcher; ignored by cache_sim.")
    parser.add_argument("--l1d-size", default="1kB")
    parser.add_argument("--l1d-ways", default="8")
    parser.add_argument("--l2-size", default="2kB")
    parser.add_argument("--l2-ways", default="4")
    parser.add_argument("--l3-sizes", nargs="+", default=["32kB"])
    parser.add_argument("--l3-ways", default="16")
    parser.add_argument("--line-size", default="64")
    parser.add_argument("--popt-property-bytes", default="4",
                        help="Vertex property bytes used to estimate P-OPT matrix column size for *_CHARGED policies.")
    parser.add_argument("--popt-active-columns", default="2",
                        help="Active rereference matrix columns charged for *_CHARGED policies (default: current+next).")
    parser.add_argument("--popt-num-epochs", default="256",
                        help="P-OPT epoch count used to estimate matrix streaming traffic for *_CHARGED policies.")
    parser.add_argument("--popt-min-data-ways", default="1",
                        help="Minimum LLC data ways kept after reserving P-OPT matrix ways.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--timeout-cache", type=int, default=600)
    parser.add_argument("--timeout-gem5", type=int, default=900)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    policies = [parse_policy_spec(p) for p in (ALL_POLICIES if args.all_policies else args.policies)]
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / now_tag()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    print(f"[roi-matrix] output: {out_dir}")
    print(f"[roi-matrix] suite={args.suite} benchmark={args.benchmark} options={args.options!r}")
    print(f"[roi-matrix] policies={', '.join(p.label for p in policies)}")
    print(f"[roi-matrix] l3_sizes={', '.join(args.l3_sizes)}")

    build_targets(args)

    rows: list[dict[str, Any]] = []
    for l3_size in args.l3_sizes:
        for spec in policies:
            if args.suite in ("cache-sim", "both"):
                print(f"[cache_sim] {spec.label} L3={l3_size}")
                rows.extend(run_cache_sim(args, out_dir, spec, l3_size))
            if args.suite in ("gem5", "both"):
                print(f"[gem5] {spec.label} L3={l3_size}")
                rows.extend(run_gem5(args, out_dir, spec, l3_size))

    if not args.dry_run:
        write_outputs(out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))