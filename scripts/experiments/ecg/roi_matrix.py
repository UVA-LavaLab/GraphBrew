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
import signal
import shlex
import shutil
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

GEM5_OPT = Path(os.environ.get(
    "GEM5_OPT",
    PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "gem5" / "build" / "X86" / "gem5.opt",
))
# S68 M5: override the kernel binary suffix to select RISCV via
# GEM5_KERNEL_SUFFIX=_riscv_m5ops for the ISA-delivery smoke. Default
# remains _m5ops (X86 build).
GEM5_KERNEL_SUFFIX = os.environ.get("GEM5_KERNEL_SUFFIX", "_m5ops")
GEM5_CONFIG = PROJECT_ROOT / "bench" / "include" / "gem5_sim" / "configs" / "graphbrew" / "graph_se.py"
GEM5_RUNTIME_SIDEBAND_FILES = (
    Path(os.environ.get("GEM5_GRAPHBREW_CTX", "/tmp/gem5_graphbrew_ctx.json")),
    Path(os.environ.get("GEM5_POPT_MATRIX", "/tmp/gem5_popt_matrix.bin")),
    Path(os.environ.get("GEM5_GRAPHBREW_OUT_EDGES", "/tmp/gem5_graphbrew_out_edges.bin")),
    Path(os.environ.get("GEM5_GRAPHBREW_IN_EDGES", "/tmp/gem5_graphbrew_in_edges.bin")),
)


def gem5_sideband_paths(gem5_out: Path) -> dict[str, Path]:
    sideband_dir = gem5_out / "graphbrew_sidebands"
    return {
        "context": sideband_dir / "gem5_graphbrew_ctx.json",
        "popt_matrix": sideband_dir / "gem5_popt_matrix.bin",
        "out_edges": sideband_dir / "gem5_graphbrew_out_edges.bin",
        "in_edges": sideband_dir / "gem5_graphbrew_in_edges.bin",
    }


DEFAULT_SNIPER_ROOT = Path("bench") / "include" / "sniper_sim" / "snipersim"
SNIPER_OVERLAY_STATUS = PROJECT_ROOT / "bench" / "include" / "sniper_sim" / ".sniper_overlays.json"
SNIPER_STATS_DIR = PROJECT_ROOT / "bench" / "include" / "sniper_sim" / "scripts"
SNIPER_RUNTIME_SIDEBAND_FILES = (
    Path(os.environ.get("SNIPER_GRAPHBREW_CTX", "/tmp/sniper_graphbrew_ctx.json")),
    Path(os.environ.get("SNIPER_POPT_MATRIX", "/tmp/sniper_popt_matrix.bin")),
    Path(os.environ.get("SNIPER_GRAPHBREW_OUT_EDGES", "/tmp/sniper_graphbrew_out_edges.bin")),
    Path(os.environ.get("SNIPER_GRAPHBREW_IN_EDGES", "/tmp/sniper_graphbrew_in_edges.bin")),
)
if str(SNIPER_STATS_DIR) not in sys.path:
    sys.path.insert(0, str(SNIPER_STATS_DIR))
from parse_stats import extract_graphbrew_metrics, read_sniper_stats


def project_relative_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sniper_root_path(args: argparse.Namespace) -> Path:
    return project_relative_path(args.sniper_root)


def sniper_runner_path(args: argparse.Namespace) -> Path:
    return sniper_root_path(args) / "run-sniper"

DEFAULT_POLICIES = [
    "LRU", "SRRIP", "GRASP", "POPT_CHARGED", "POPT",
    "ECG:DBG_ONLY", "ECG:DBG_PRIMARY_CHARGED", "ECG:DBG_PRIMARY",
    "ECG:POPT_PRIMARY",
]
SNIPER_DEFAULT_POLICIES = ["LRU", "SRRIP"]
SNIPER_POLICY_MAP = {
    "LRU": "lru",
    "SRRIP": "srrip",
}
SNIPER_GRAPH_POLICY_MAP = {
    "GRASP": "grasp",
    "POPT": "popt",
    "ECG": "ecg",
}
ALL_POLICIES = [
    "LRU",
    "SRRIP",
    "GRASP",
    "POPT_CHARGED",
    "POPT",
    "ECG:DBG_ONLY",
    "ECG:DBG_PRIMARY_CHARGED",
    "ECG:DBG_PRIMARY",
    "ECG:POPT_TIE",
    "ECG:POPT_PRIMARY",
    "ECG:ECG_EMBEDDED",
    "ECG:ECG_EPOCH_EMBEDDED",
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

ECG_PFX_MODE_VALUES = {
    "degree": "1",
    "popt": "2",
    "droplet": "3",  # DROPLET-style: sequential prefetch (no target selection)
    "far_future": "4",  # FAR-FUTURE: target from global hot_table (not v.s neighbors)
    "per_edge": "6",
    "cross_iter": "7",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "6": "6",
    "7": "7",
}


def ecg_pfx_env(args: argparse.Namespace) -> dict[str, str]:
    # For cache_sim, DROPLET maps to ECG_PREFETCH_MODE=3 (sequential
    # lookahead, no target selection — faithful comparator for the
    # ECG_PFX claim). The Sniper/gem5 DROPLET overlays use a separate
    # path (perf_model/.../prefetcher/droplet), so this only affects
    # cache_sim env.
    if args.prefetcher == "DROPLET":
        return {
            "ECG_PREFETCH_MODE": "3",
            "ECG_PREFETCH_WINDOW": str(args.ecg_pfx_window),
            "ECG_PREFETCH_LOOKAHEAD": str(args.ecg_pfx_lookahead),
        }
    if args.prefetcher != "ECG_PFX":
        return {}
    return {
        "ECG_PREFETCH_MODE": ECG_PFX_MODE_VALUES[str(args.ecg_pfx_mode)],
        "ECG_PREFETCH_WINDOW": str(args.ecg_pfx_window),
        "ECG_PREFETCH_LOOKAHEAD": str(args.ecg_pfx_lookahead),
    }


def effective_ecg_pfx_value(args: argparse.Namespace, name: str) -> str:
    return ecg_pfx_env(args).get(name, os.environ.get(name, ""))


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


def format_sniper_kb(size: str | int) -> int:
    size_bytes = parse_size_bytes(size)
    if size_bytes % 1024 != 0:
        raise ValueError(f"Sniper cache sizes must be whole KiB values, got {size!r}")
    return max(size_bytes // 1024, 1)


def sniper_l3_geometry(args: argparse.Namespace, l3_size: str, charge: dict[str, Any]) -> tuple[int, str, int]:
    line_size = parse_size_bytes(args.line_size)
    requested_bytes = parse_size_bytes(l3_size)
    requested_ways = max(int(args.l3_ways), 1)
    requested_sets = max(requested_bytes // (requested_ways * line_size), 1)
    desired_ways = max(int(charge["popt_effective_l3_ways"]), 1)

    # Sniper's cache_size is in integer KiB and its cache constructor requires
    # size == sets * ways * line_size. Charged P-OPT can produce fractional-KiB
    # effective sizes on tiny LLCs, so round data ways down to the nearest valid
    # geometry. This is conservative for charged policies and leaves uncharged
    # whole-KiB configurations unchanged.
    configured_ways = desired_ways
    while configured_ways > 1:
        configured_bytes = requested_sets * configured_ways * line_size
        if configured_bytes % 1024 == 0:
            return configured_bytes // 1024, str(configured_ways), configured_bytes
        configured_ways -= 1
    configured_bytes = requested_sets * configured_ways * line_size
    configured_kb = max(configured_bytes // 1024, 1)
    return configured_kb, str(configured_ways), configured_bytes


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def miss_rate(misses: Any, accesses: Any) -> float | None:
    miss_count = numeric(misses)
    access_count = numeric(accesses)
    if miss_count is None or not access_count:
        return None
    return miss_count / access_count


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
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            out.write(f"\n[timeout_s] {timeout}\n")
            out.write(f"[elapsed_s] {elapsed:.3f}\n")
            out.write("[timeout_action] SIGTERM process group\n")
            out.flush()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                out.write("[timeout_action] SIGKILL process group\n")
                out.flush()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            returncode = 124
        result = subprocess.CompletedProcess(cmd, returncode)
        out.write(f"\n[exit_code] {result.returncode}\n")
        out.write(f"[elapsed_s] {time.time() - start:.3f}\n")
    return result


def memory_limited_command(cmd: list[str], memory_limit_gb: float) -> list[str]:
    if memory_limit_gb <= 0.0:
        return cmd
    prlimit = shutil.which("prlimit")
    if not prlimit:
        raise RuntimeError("prlimit not found; cannot enforce Sniper unsafe workload memory limit")
    limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
    return [prlimit, f"--as={limit_bytes}", "--", *cmd]


def clear_runtime_sideband_files() -> None:
    for path in GEM5_RUNTIME_SIDEBAND_FILES:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def clear_sniper_runtime_sideband_files() -> None:
    for path in SNIPER_RUNTIME_SIDEBAND_FILES:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def clear_sideband_files(paths: dict[str, Path]) -> None:
    for path in paths.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def sniper_sideband_paths(sniper_out: Path) -> dict[str, Path]:
    sideband_dir = sniper_out / "graphbrew_sidebands"
    return {
        "context": sideband_dir / "sniper_graphbrew_ctx.json",
        "popt_matrix": sideband_dir / "sniper_popt_matrix.bin",
        "out_edges": sideband_dir / "sniper_graphbrew_out_edges.bin",
        "in_edges": sideband_dir / "sniper_graphbrew_in_edges.bin",
    }


def build_targets(args: argparse.Namespace) -> None:
    if args.no_build or args.dry_run:
        return

    targets = []
    if args.suite in ("cache-sim", "both"):
        targets.append(f"sim-{args.benchmark}")
    if args.suite in ("gem5", "both"):
        targets.append(f"gem5-m5ops-{args.benchmark}")
    if args.suite == "sniper":
        if args.sniper_workload == "pr_kernel_smoke":
            targets.append("sniper-pr_kernel_smoke")
        elif args.sniper_workload == "sg_kernel" and args.allow_sniper_sg_kernel_workload:
            targets.append("sniper-sg_kernel")
        elif args.allow_sniper_benchmark_workload:
            targets.append(f"sniper-{args.benchmark}")

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
        # cache_sim MUST run single-threaded for deterministic/reproducible
        # results: the OpenMP-parallel kernel records cache accesses in
        # nondeterministic interleaved order, so >1 thread yields
        # non-reproducible, thread-count-dependent miss counts.
        "OMP_NUM_THREADS": str(args.cache_sim_omp_threads),
    })
    env.update(ecg_pfx_env(args))
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
    binary = PROJECT_ROOT / "bench" / "bin_gem5" / f"{args.benchmark}{GEM5_KERNEL_SUFFIX}"
    label = f"gem5_{args.benchmark}_{spec.safe_label}_L3{sanitize(l3_size)}"
    gem5_out = out_dir / "gem5" / label
    log_path = out_dir / "logs" / f"{label}.log"
    sidebands = gem5_sideband_paths(gem5_out)
    charge = popt_charge_metadata(args, spec, l3_size)
    if args.prefetcher == "ECG_PFX" and not args.allow_gem5_ecg_pfx:
        row = base_row("gem5", args, spec, l3_size, charge)
        row.update({
            "section": 0,
            "log_path": str(log_path),
            "gem5_out": str(gem5_out),
            "status": "unsupported",
            "error": "ECG_PFX gem5 timing path is experimental; pass --allow-gem5-ecg-pfx only after rebuilding gem5 with the ECG_PFX SimObject scaffold.",
        })
        return [row]
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
    if args.prefetcher == "DROPLET":
        cmd.extend([
            "--droplet-prefetch-degree", str(args.droplet_prefetch_degree),
            "--droplet-indirect-degree", str(args.droplet_indirect_degree),
            "--droplet-stride-table-size", str(args.droplet_stride_table_size),
        ])
    if args.prefetcher == "ECG_PFX":
        cmd.extend([
            "--ecg-pfx-lookahead", str(args.ecg_pfx_lookahead),
            "--ecg-pfx-hint-filter", str(args.ecg_pfx_hint_filter),
            "--ecg-pfx-delivery", str(args.ecg_pfx_delivery),
        ])
    if spec.ecg_mode:
        cmd.extend(["--ecg-mode", spec.ecg_mode])

    if not args.dry_run:
        sidebands["context"].parent.mkdir(parents=True, exist_ok=True)
        clear_sideband_files(sidebands)

    env = dict(os.environ)
    env["GEM5_GRAPHBREW_CTX"] = str(sidebands["context"])
    env["GEM5_POPT_MATRIX"] = str(sidebands["popt_matrix"])
    env["GEM5_GRAPHBREW_OUT_EDGES"] = str(sidebands["out_edges"])
    env["GEM5_GRAPHBREW_IN_EDGES"] = str(sidebands["in_edges"])
    if args.prefetcher == "ECG_PFX":
        env.update(ecg_pfx_env(args))
        env["GEM5_ENABLE_ECG_PFX_HINTS"] = "1"
        env["GEM5_ECG_PFX_LOOKAHEAD"] = effective_ecg_pfx_value(args, "ECG_PREFETCH_LOOKAHEAD")

    result = run_command(cmd, PROJECT_ROOT, env, args.timeout_gem5, log_path, args.dry_run)
    if args.dry_run:
        return []

    base = base_row("gem5", args, spec, l3_size, charge)
    base.update({
        "log_path": str(log_path),
        "gem5_out": str(gem5_out),
        "gem5_sideband_dir": str(sidebands["context"].parent),
        "gem5_context_path": str(sidebands["context"]),
        "gem5_popt_matrix_path": str(sidebands["popt_matrix"]),
        "gem5_out_edges_path": str(sidebands["out_edges"]),
        "gem5_in_edges_path": str(sidebands["in_edges"]),
        "gem5_ecg_pfx_experimental": int(args.prefetcher == "ECG_PFX" and args.allow_gem5_ecg_pfx),
    })
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


def sniper_graph_policies_enabled(args: argparse.Namespace) -> bool:
    return bool(args.sniper_enable_graph_policies or SNIPER_OVERLAY_STATUS.exists())


def sniper_policy_name(args: argparse.Namespace, spec: PolicySpec) -> str | None:
    if spec.ecg_mode and spec.policy != "ECG":
        return None
    if spec.charge_popt_overhead and spec.policy not in ("POPT", "ECG"):
        return None
    if spec.policy in SNIPER_POLICY_MAP:
        return SNIPER_POLICY_MAP[spec.policy]
    if spec.policy == "ECG" and sniper_graph_policies_enabled(args):
        if spec.ecg_mode == "DBG_ONLY":
            return "grasp"
        if spec.ecg_mode == "POPT_PRIMARY":
            return "popt"
        return "ecg"
    if sniper_graph_policies_enabled(args):
        return SNIPER_GRAPH_POLICY_MAP.get(spec.policy)
    return None


def sniper_binary_and_options(args: argparse.Namespace) -> tuple[Path, list[str]]:
    if args.sniper_workload == "pr_kernel_smoke":
        if args.benchmark != "pr":
            raise SystemExit("--suite sniper --sniper-workload pr_kernel_smoke is only valid with --benchmark pr")
        return PROJECT_ROOT / "bench" / "bin_sniper" / "pr_kernel_smoke", []
    if args.sniper_workload == "kernel_smoke":
        supported = {"pr", "bfs", "sssp"}
        if args.benchmark not in supported:
            raise SystemExit(f"--suite sniper --sniper-workload kernel_smoke supports only {sorted(supported)}")
        return PROJECT_ROOT / "bench" / "bin_sniper" / f"{args.benchmark}_kernel_smoke", []
    if args.sniper_workload == "sg_kernel":
        options = shlex.split(args.options)
        if "-f" not in options:
            raise SystemExit("--suite sniper --sniper-workload sg_kernel requires --options with -f graph.sg")
        return PROJECT_ROOT / "bench" / "bin_sniper" / "sg_kernel", ["--benchmark", args.benchmark, *options]
    return PROJECT_ROOT / "bench" / "bin_sniper" / args.benchmark, shlex.split(args.options)


def run_sniper(args: argparse.Namespace, out_dir: Path, spec: PolicySpec, l3_size: str) -> list[dict[str, Any]]:
    label = f"sniper_{args.benchmark}_{spec.safe_label}_L3{sanitize(l3_size)}"
    if getattr(args, "_sniper_thread_sweep", False):
        label += f"_T{sanitize(str(args.sniper_cores))}"
    sniper_out = out_dir / "sniper" / label
    log_path = out_dir / "logs" / f"{label}.log"
    sidebands = sniper_sideband_paths(sniper_out)
    charge = popt_charge_metadata(args, spec, l3_size)
    row = base_row("sniper", args, spec, l3_size, charge)
    sniper_root = sniper_root_path(args)
    sniper_runner = sniper_runner_path(args)
    unsafe_sniper_workload = args.sniper_workload in ("benchmark", "sg_kernel")
    row.update({
        "section": 0,
        "log_path": str(log_path),
        "sniper_out": str(sniper_out),
        "sniper_root": str(sniper_root),
        "sniper_runner": str(sniper_runner),
        "sniper_sideband_dir": str(sidebands["context"].parent),
        "sniper_context_path": str(sidebands["context"]),
        "sniper_popt_matrix_path": str(sidebands["popt_matrix"]),
        "sniper_out_edges_path": str(sidebands["out_edges"]),
        "sniper_in_edges_path": str(sidebands["in_edges"]),
        "sniper_workload": args.sniper_workload,
        "sniper_cores": args.sniper_cores,
        "sniper_frontend": args.sniper_frontend,
        "sniper_omp_wait_policy": args.sniper_omp_wait_policy,
        "sniper_base_config": args.sniper_base_config,
        "sniper_extra_configs": " ".join(args.sniper_config),
        "sniper_address_domain": args.sniper_address_domain,
        "sniper_mimicos_memory_mb": args.sniper_mimicos_memory_mb,
        "sniper_mimicos_kernel_mb": args.sniper_mimicos_kernel_mb,
        "threads": args.sniper_cores,
        "sniper_metric_scope": "loads_only_cache_stats",
        "sniper_overlays_enabled": int(sniper_graph_policies_enabled(args)),
    })
    if unsafe_sniper_workload:
        row["sniper_memory_limit_gb"] = args.sniper_memory_limit_gb

    if spec.policy == "ECG" and spec.ecg_mode in ("DBG_ONLY", "POPT_PRIMARY"):
        row["sniper_policy_alias_for"] = spec.ecg_mode

    if args.sniper_workload == "benchmark" and not args.allow_sniper_benchmark_workload:
        row.update({
            "status": "unsupported",
            "error": "Full bench/bin_sniper wrappers are disabled by default after the tiny PR SDE/SIFT probe consumed about 53 GiB RSS; pass --allow-sniper-benchmark-workload only for bounded run-mode debugging.",
        })
        return [row]

    if args.prefetcher == "ECG_PFX":
        if not sniper_graph_policies_enabled(args):
            row.update({
                "status": "unsupported",
                "error": "Sniper ECG_PFX requires overlays from scripts/setup_sniper.py --apply-overlays.",
            })
            return [row]

    if args.sniper_workload == "sg_kernel" and not args.allow_sniper_sg_kernel_workload:
        row.update({
            "status": "unsupported",
            "error": "bench/bin_sniper/sg_kernel is native-clean for .sg load+ROI diagnostics, but under Sniper/SDE it repeated the ~50 GiB runaway child-process behavior; pass --allow-sniper-sg-kernel-workload only for tightly bounded run-mode debugging.",
        })
        return [row]

    if args.prefetcher == "DROPLET" and not sniper_graph_policies_enabled(args):
        row.update({
            "status": "unsupported",
            "error": "Sniper DROPLET requires overlays from scripts/setup_sniper.py --apply-overlays.",
        })
        return [row]

    policy_name = sniper_policy_name(args, spec)
    if policy_name is None:
        supported = "LRU/SRRIP"
        if not sniper_graph_policies_enabled(args):
            supported += "; apply overlays with scripts/setup_sniper.py --apply-overlays for GRASP/POPT"
        row.update({
            "status": "unsupported",
            "error": f"Sniper runner currently supports {supported}; POPT/ECG overlays are still Phase 3 work.",
        })
        return [row]

    binary, binary_options = sniper_binary_and_options(args)
    if not args.dry_run:
        if not sniper_runner.exists():
            row.update({"status": "error", "error": f"missing run-sniper: {sniper_runner}"})
            return [row]
        if not binary.exists():
            row.update({"status": "error", "error": f"missing Sniper benchmark binary: {binary}"})
            return [row]
        sidebands["context"].parent.mkdir(parents=True, exist_ok=True)
        clear_sideband_files(sidebands)

    l1_kb = format_sniper_kb(args.l1d_size)
    l2_kb = format_sniper_kb(args.l2_size)
    line_size = parse_size_bytes(args.line_size)
    l3_kb, sniper_l3_ways, sniper_l3_bytes = sniper_l3_geometry(args, l3_size, charge)
    row.update({
        "sniper_l3_config_kb": l3_kb,
        "sniper_l3_config_ways": sniper_l3_ways,
        "sniper_l3_config_bytes": sniper_l3_bytes,
    })

    cmd = [
        str(sniper_runner),
        "--roi",
        "--no-cache-warming",
    ]
    if args.sniper_frontend == "sift":
        cmd.append("--sift")
    cmd.extend([
        "-n", str(args.sniper_cores),
        "-d", str(sniper_out),
        "-c", args.sniper_base_config,
    ])
    for config_name in args.sniper_config:
        cmd.extend(["-c", config_name])
    sniper_config_values = {
        "general/total_cores": args.sniper_cores,
        "perf_model/l1_icache/cache_block_size": line_size,
        "perf_model/l1_dcache/cache_block_size": line_size,
        "perf_model/l2_cache/cache_block_size": line_size,
        "perf_model/l1_dcache/cache_size": l1_kb,
        "perf_model/l1_dcache/associativity": args.l1d_ways,
        "perf_model/l1_dcache/replacement_policy": "lru",
        "perf_model/l2_cache/cache_size": l2_kb,
        "perf_model/l2_cache/associativity": args.l2_ways,
        "perf_model/l2_cache/replacement_policy": "lru",
        "perf_model/nuca/cache_size": l3_kb,
        "perf_model/nuca/associativity": sniper_l3_ways,
        "perf_model/nuca/replacement_policy": policy_name,
        "perf_model/reserve_thp/memory_size": args.sniper_mimicos_memory_mb,
        "perf_model/reserve_thp/kernel_size": args.sniper_mimicos_kernel_mb,
    }
    sniper_config_values["general/translation_enabled"] = "false" if args.sniper_address_domain == "virtual" else "true"
    if args.prefetcher == "DROPLET":
        prefetch_config = "l1_dcache" if args.prefetcher_level == "l1d" else "l2_cache"
        sniper_config_values[f"perf_model/{prefetch_config}/prefetcher"] = "droplet"
        sniper_config_values[f"perf_model/{prefetch_config}/prefetcher/droplet/prefetch_degree"] = args.droplet_prefetch_degree
        sniper_config_values[f"perf_model/{prefetch_config}/prefetcher/droplet/indirect_degree"] = args.droplet_indirect_degree
        sniper_config_values[f"perf_model/{prefetch_config}/prefetcher/droplet/stride_table_size"] = args.droplet_stride_table_size
    elif args.prefetcher == "ECG_PFX":
        prefetch_config = "l1_dcache" if args.prefetcher_level == "l1d" else "l2_cache"
        sniper_config_values[f"perf_model/{prefetch_config}/prefetcher"] = "ecg_pfx"
    for key, value in sniper_config_values.items():
        cmd.extend(["-g", f"{key}={value}"])
    cmd.extend(["--", str(binary), *binary_options])

    if unsafe_sniper_workload:
        try:
            cmd = memory_limited_command(cmd, float(args.sniper_memory_limit_gb))
        except RuntimeError as exc:
            row.update({"status": "error", "error": str(exc)})
            return [row]

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.sniper_cores)
    if args.sniper_omp_wait_policy != "unset":
        env["OMP_WAIT_POLICY"] = args.sniper_omp_wait_policy
    else:
        env.pop("OMP_WAIT_POLICY", None)
    env["SNIPER_GRAPHBREW_CTX"] = str(sidebands["context"])
    env["SNIPER_POPT_MATRIX"] = str(sidebands["popt_matrix"])
    env["SNIPER_GRAPHBREW_OUT_EDGES"] = str(sidebands["out_edges"])
    env["SNIPER_GRAPHBREW_IN_EDGES"] = str(sidebands["in_edges"])
    if args.prefetcher == "ECG_PFX":
        env.update(ecg_pfx_env(args))
        env["SNIPER_ENABLE_ECG_PFX_HINTS"] = "1"
        env["SNIPER_ECG_PFX_LOOKAHEAD"] = effective_ecg_pfx_value(args, "ECG_PREFETCH_LOOKAHEAD")
        env["SNIPER_ECG_PFX_MODE"] = effective_ecg_pfx_value(args, "ECG_PREFETCH_MODE")
        env["SNIPER_ECG_PFX_HINT_FILTER"] = str(args.ecg_pfx_hint_filter)
        env["SNIPER_ECG_PFX_FILTER_ELEM_SIZE"] = "4"
        env["SNIPER_ECG_PFX_FILTER_LINE_SIZE"] = str(args.line_size)
    if spec.ecg_mode and policy_name == "ecg":
        env["SNIPER_ECG_MODE"] = spec.ecg_mode
    result = run_command(cmd, PROJECT_ROOT, env, args.timeout_sniper, log_path, args.dry_run)
    if args.dry_run:
        return []

    if result is None or result.returncode != 0:
        row.update({"status": "error", "error": f"exit_code={result.returncode if result else 'unknown'}"})
        return [row]

    raw_stats = read_sniper_stats(sniper_out)
    if not raw_stats.get("success"):
        row.update({"status": "error", "error": raw_stats.get("error", "missing Sniper stats")})
        return [row]

    metrics = extract_graphbrew_metrics(raw_stats)
    l1_accesses = metrics.get("l1d_loads", 0)
    l1_misses = metrics.get("l1d_load_misses", 0)
    l2_accesses = metrics.get("l2_loads", 0)
    l2_misses = metrics.get("l2_load_misses", 0)
    l3_accesses = metrics.get("llc_loads", 0)
    l3_misses = metrics.get("llc_load_misses", 0)
    row.update({
        "section": 1,
        "status": "ok",
        "stats_path": metrics.get("stats_path", ""),
        "sniper_policy_config": policy_name,
        "sim_ticks": metrics.get("cycles_or_time", 0),
        "instructions": metrics.get("instructions", 0),
        "ipc": metrics.get("ipc_raw", 0.0),
        "l1_accesses": l1_accesses,
        "l1_misses": l1_misses,
        "l1_miss_rate": miss_rate(l1_misses, l1_accesses),
        "l2_accesses": l2_accesses,
        "l2_misses": l2_misses,
        "l2_miss_rate": miss_rate(l2_misses, l2_accesses),
        "l3_accesses": l3_accesses,
        "l3_misses": l3_misses,
        "l3_miss_rate": miss_rate(l3_misses, l3_accesses),
        "l1_policy": "LRU",
        "l2_policy": "LRU",
        "l3_policy": policy_name.upper(),
    })
    for key in (
        "pf_issued",
        "pf_fillups",
        "pf_useful",
        "pf_evicted_before_use",
        "pf_invalidated_before_use",
        "droplet_sideband_loaded",
        "droplet_edge_accesses",
        "droplet_stride_issued",
        "droplet_indirect_issued",
        "droplet_duplicate_skips",
        "ecg_pfx_sideband_loaded",
        "ecg_pfx_target_hints_seen",
        "ecg_pfx_issued",
        "ecg_pfx_duplicate_skips",
        "ecg_pfx_no_sideband",
        "ecg_pfx_invalid_target",
        "sniper_cpi_base",
        "sniper_cpi_branch",
        "sniper_cpi_data_cache",
        "sniper_cpi_data_l1",
        "sniper_cpi_data_l2",
        "sniper_cpi_data_llc",
        "sniper_cpi_data_dram",
        "sniper_cpi_sync",
        "sniper_cpi_unknown",
        "sniper_nonidle_elapsed_time",
        "sniper_idle_elapsed_time",
        "sniper_elapsed_time",
    ):
        row[key] = metrics.get(key, 0)
    if args.prefetcher == "DROPLET":
        indirect_issued = int(row.get("droplet_indirect_issued") or 0)
        prefetch_issued = int(row.get("pf_issued") or 0)
        prefetch_useful = int(row.get("pf_useful") or 0)
        if indirect_issued == 0:
            error = "DROPLET sideband loaded but no edge accesses/prefetches issued."
            if args.sniper_address_domain == "translated":
                error += " Sniper cache addresses are translated while current GraphBrew sidebands are virtual."
            row.update({
                "status": "inactive",
                "droplet_activity": "inactive",
                "droplet_useful_activity": "inactive",
                "error": error,
            })
        elif prefetch_issued == 0:
            row.update({
                "status": "active_no_fill",
                "droplet_activity": "requested_no_fill",
                "droplet_useful_activity": "no_fill",
                "error": "DROPLET saw edge accesses and generated indirect requests, but Sniper did not enqueue cache prefetch fills.",
            })
        else:
            row["droplet_activity"] = "issued"
            row["droplet_useful_activity"] = "useful" if prefetch_useful > 0 else "issued_no_useful"
    if args.prefetcher == "ECG_PFX":
        hints_seen = int(row.get("ecg_pfx_target_hints_seen") or 0)
        pfx_issued = int(row.get("ecg_pfx_issued") or 0)
        prefetch_issued = int(row.get("pf_issued") or 0)
        if hints_seen == 0:
            row.update({
                "status": "inactive",
                "ecg_pfx_activity": "inactive",
                "error": "ECG_PFX prefetcher was configured but consumed no target hints.",
            })
        elif pfx_issued == 0:
            row.update({
                "status": "active_no_fill",
                "ecg_pfx_activity": "consumed_no_prefetch",
                "error": "ECG_PFX consumed target hints but issued no cache prefetch requests.",
            })
        elif prefetch_issued == 0:
            row.update({
                "status": "active_no_fill",
                "ecg_pfx_activity": "requested_no_fill",
                "error": "ECG_PFX consumed target hints and generated prefetch requests, but Sniper did not enqueue cache prefetch fills.",
            })
        else:
            row["ecg_pfx_activity"] = "issued"
    return [row]


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
    timing_model = "simulated_target_time"
    timing_valid_for_speedup = "1"
    timing_caveat = ""
    if args.prefetcher == "ECG_PFX" and simulator in ("gem5", "sniper"):
        timing_model = (
            "prototype_instruction_delivery"
            if simulator == "gem5" and args.ecg_pfx_delivery == "instruction"
            else "prototype_explicit_hint_delivery"
        )
        timing_valid_for_speedup = "0"
        timing_caveat = (
            "ECG_PFX timing includes prototype benchmark-emitted hint delivery; "
            "use cache and prefetch metrics for mechanism evidence until PFX is validated as instruction-carried metadata."
        )
    elif simulator == "cache-sim":
        timing_model = "cache_mechanism_model"

    row = {
        "simulator": simulator,
        "benchmark": args.benchmark,
        "options": args.options,
        "prefetcher": args.prefetcher,
        "prefetcher_level": args.prefetcher_level,
        "timing_model": timing_model,
        "timing_valid_for_speedup": timing_valid_for_speedup,
        "timing_caveat": timing_caveat,
        "droplet_prefetch_degree": args.droplet_prefetch_degree,
        "droplet_indirect_degree": args.droplet_indirect_degree,
        "droplet_stride_table_size": args.droplet_stride_table_size,
        "ecg_prefetch_mode": effective_ecg_pfx_value(args, "ECG_PREFETCH_MODE"),
        "ecg_prefetch_window": effective_ecg_pfx_value(args, "ECG_PREFETCH_WINDOW"),
        "ecg_prefetch_lookahead": effective_ecg_pfx_value(args, "ECG_PREFETCH_LOOKAHEAD"),
        "ecg_pfx_hint_filter": args.ecg_pfx_hint_filter,
        "ecg_pfx_delivery": args.ecg_pfx_delivery,
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
        description="Run matched cache_sim/gem5/Sniper ROI policy matrix for ECG validation."
    )
    parser.add_argument("--suite", choices=["cache-sim", "gem5", "sniper", "both"], default="both")
    parser.add_argument("--benchmark", default="pr")
    parser.add_argument("--options", default="-g 10 -k 16 -o 5 -n 1 -i 5")
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--all-policies", action="store_true", help="Use the full ECG validation policy set.")
    parser.add_argument("--prefetcher", choices=["none", "DROPLET", "ECG_PFX"], default="none",
                        help="Prefetcher to attach. ECG_PFX is supported by cache_sim and experimental gem5/Sniper hint paths.")
    parser.add_argument("--prefetcher-level", choices=["l1d", "l2"], default="l2",
                        help="gem5/Sniper cache level for --prefetcher; ignored by cache_sim ECG_PFX.")
    parser.add_argument("--droplet-prefetch-degree", type=int, default=1,
                        help="DROPLET edge-stream cache lines to prefetch per trigger (artifact default: 1).")
    parser.add_argument("--droplet-indirect-degree", type=int, default=16,
                        help="DROPLET neighbor IDs to translate into property prefetches per edge line (artifact default: one 64B line of 4B IDs).")
    parser.add_argument("--droplet-stride-table-size", type=int, default=64,
                        help="DROPLET stream table entries (artifact config streams default: 64).")
    parser.add_argument("--ecg-pfx-mode", choices=sorted(ECG_PFX_MODE_VALUES), default="popt",
                        help="ECG_PFX target selection: degree/hot-neighbor mode or P-OPT-ranked mode.")
    parser.add_argument("--ecg-pfx-window", default="16",
                        help="Runtime/construction dedup window for ECG_PFX.")
    parser.add_argument("--ecg-pfx-lookahead", default="4",
                        help="Algorithm lookahead distance for ECG_PFX temporal prefetch probes.")
    parser.add_argument("--ecg-pfx-hint-filter", default="16",
                        help="Recent-target filter capacity before emitting ECG_PFX hints; 0 disables filtering.")
    parser.add_argument("--ecg-pfx-delivery", choices=["explicit-hint", "instruction"], default="explicit-hint",
                        help="ECG_PFX detailed-sim delivery path. instruction uses gem5 RISC-V ecg.extract or x86 pseudo-op scaffolds.")
    parser.add_argument("--l1d-size", default="1kB")
    parser.add_argument("--l1d-ways", default="8")
    parser.add_argument("--l2-size", default="2kB")
    parser.add_argument("--l2-ways", default="4")
    parser.add_argument("--l3-sizes", nargs="+", default=["32kB"])
    parser.add_argument("--cache-sim-omp-threads", type=int, default=1,
                        help="OMP threads for cache_sim. MUST be 1 for deterministic/reproducible "
                             "results: the parallel kernel records cache accesses in nondeterministic "
                             "interleaved order, so >1 thread gives non-reproducible, "
                             "thread-count-dependent miss counts.")
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
    parser.add_argument("--timeout-sniper", type=int, default=600)
    parser.add_argument("--allow-gem5-ecg-pfx", action="store_true",
                        help="Run experimental gem5 ECG_PFX timing path. Requires rebuilt gem5 overlays; default is an explicit unsupported row.")
    parser.add_argument("--sniper-workload", choices=["pr_kernel_smoke", "kernel_smoke", "sg_kernel", "benchmark"], default="pr_kernel_smoke",
                        help="Use a fast fixed kernel smoke, file-backed .sg kernel, or the full bench/bin_sniper/<benchmark> wrapper.")
    parser.add_argument("--allow-sniper-benchmark-workload", action="store_true",
                        help="Allow full bench/bin_sniper/<benchmark> under Sniper. Unsafe until SDE/SIFT run mode is fixed; guarded by --sniper-memory-limit-gb.")
    parser.add_argument("--allow-sniper-sg-kernel-workload", action="store_true",
                        help="Allow file-backed bench/bin_sniper/sg_kernel under Sniper. Native .sg runs are clean, but Sniper/SDE sg_kernel repeated the high-memory runaway; use only for bounded run-mode debugging guarded by --sniper-memory-limit-gb.")
    parser.add_argument("--sniper-memory-limit-gb", type=float, default=16.0,
                        help="Address-space limit applied with prlimit to explicitly allowed unsafe Sniper benchmark/sg_kernel workloads. Set 0 to disable only for manual debugging.")
    parser.add_argument("--sniper-mimicos-memory-mb", default="4096",
                        help="Override perf_model/reserve_thp/memory_size for GraphBrew Sniper runs. The upstream baseline default is 131072 MB, which is excessive for these workloads.")
    parser.add_argument("--sniper-mimicos-kernel-mb", default="128",
                        help="Override perf_model/reserve_thp/kernel_size for GraphBrew Sniper runs. The upstream baseline default is 32768 MB, which is excessive for these workloads.")
    parser.add_argument("--sniper-enable-graph-policies", action="store_true",
                        help="Enable tracked Sniper graph-policy overlays even if .sniper_overlays.json is absent.")
    parser.add_argument("--sniper-cores", default="1", help="Core count passed to run-sniper -n and OMP_NUM_THREADS.")
    parser.add_argument("--threads", nargs="+", default=[],
                        help="Sniper thread/core counts to sweep. Alias for repeated --sniper-cores values.")
    parser.add_argument("--sniper-base-config", default="graphbrew/graph_sniper",
                        help="Base Sniper -c config for GraphBrew runs. Installed by scripts/setup_sniper.py from bench/include/sniper_sim/configs/.")
    parser.add_argument("--sniper-root", default=str(DEFAULT_SNIPER_ROOT),
                        help="Sniper checkout/install root containing run-sniper. Relative paths are resolved from the GraphBrew repository root.")
    parser.add_argument("--sniper-frontend", choices=["live", "sift"], default="live",
                        help="Sniper frontend mode. 'live' is the proven default; 'sift' inserts --sift for bounded trace-frontend probes.")
    parser.add_argument("--sniper-omp-wait-policy", choices=["passive", "active", "unset"], default="passive",
                        help="OMP_WAIT_POLICY for Sniper benchmark processes. Passive avoids SIFT/OpenMP barrier deadlocks observed with full wrappers.")
    parser.add_argument("--sniper-config", nargs="*", default=[], help="Additional Sniper -c config names after --sniper-base-config.")
    parser.add_argument("--sniper-address-domain", choices=["virtual", "translated"], default="virtual",
                        help="Address domain for Sniper cache-side GraphBrew sidebands. 'virtual' disables Sniper translation so exported virtual regions match cache callbacks; 'translated' keeps the baseline MMU path and requires translated/physical sidebands.")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.threads and args.suite != "sniper":
        raise SystemExit("--threads is currently supported only with --suite sniper")
    if args.all_policies:
        policy_texts = ALL_POLICIES
    elif args.policies is not None:
        policy_texts = args.policies
    elif args.suite == "sniper":
        policy_texts = SNIPER_DEFAULT_POLICIES
    else:
        policy_texts = DEFAULT_POLICIES
    policies = [parse_policy_spec(p) for p in policy_texts]
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
            if args.suite == "sniper":
                original_cores = str(args.sniper_cores)
                thread_values = [str(value) for value in (args.threads or [args.sniper_cores])]
                args._sniper_thread_sweep = bool(args.threads)
                for thread_count in thread_values:
                    args.sniper_cores = thread_count
                    print(f"[sniper] {spec.label} L3={l3_size} T={thread_count}")
                    rows.extend(run_sniper(args, out_dir, spec, l3_size))
                args.sniper_cores = original_cores
                args._sniper_thread_sweep = False

    if not args.dry_run:
        write_outputs(out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))