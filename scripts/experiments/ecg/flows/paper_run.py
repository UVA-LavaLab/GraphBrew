#!/usr/bin/env python3
"""Manifest-driven final-paper ECG run orchestrator.

This script does not replace ``roi_matrix.py`` or ``proof_matrix.py``. It wraps
them with the pieces needed for days-long paper runs:

- a checked-in JSON manifest,
- deterministic job expansion,
- dry-run/list modes,
- resumable output directories,
- per-job logs and status JSONL,
- a run manifest snapshot,
- a single-process lock around GraphBrew gem5 sideband users.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import functools
import hashlib
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from policy_specs import policy_output_label  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ECG_DIR = PROJECT_ROOT / "scripts" / "experiments" / "ecg"
ROI_MATRIX = ECG_DIR / "roi_matrix.py"
PROOF_MATRIX = ECG_DIR / "flows" / "proof_matrix.py"
DEFAULT_MANIFEST = ECG_DIR / "final_paper_manifest.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "ecg_experiments" / "final_paper_runs"
DEFAULT_LOCK = Path(os.environ.get("GRAPHBREW_FINAL_RUN_LOCK", "/tmp/graphbrew_final_paper_run.lock"))


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    kind: str
    command: list[str]
    out_dir: Path
    log_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def output_csv(self) -> Path:
        if self.kind == "proof_matrix":
            return self.out_dir / "proof_matrix.csv"
        return self.out_dir / "roi_matrix.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "job"


def normalize_filter_token(text: str) -> str:
    return sanitize(text.upper().replace("-", "_"))


def token_matches(text: str, filters: list[str]) -> bool:
    if not filters:
        return True
    normalized_text = normalize_filter_token(text)
    return any(normalize_filter_token(token) == normalized_text for token in filters)


def filter_policy_specs(policies: list[str], filters: list[str]) -> list[str]:
    if not filters:
        return policies
    requested_labels = {
        policy_output_label(policy) for policy in filters}
    return [
        policy for policy in policies
        if policy_output_label(policy) in requested_labels
    ]


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"manifest not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid manifest JSON in {path}: {exc}") from None


def resolve_path(path_text: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base / path


@functools.lru_cache(maxsize=None)
def path_fingerprint(path_text: str) -> str:
    path = Path(path_text)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(
            item for item in path.rglob("*")
            if item.is_file() and
            "__pycache__" not in item.parts and
            item.suffix not in {".pyc", ".log"}):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(path_fingerprint(str(child)).encode())
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def git_state_fingerprint() -> str:
    digest = hashlib.sha256()
    for command in (
        ["git", "rev-parse", "HEAD"],
        ["git", "diff", "--binary", "--no-ext-diff"],
        ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
    ):
        result = subprocess.run(
            command, cwd=str(PROJECT_ROOT), capture_output=True, check=False)
        digest.update(result.stdout)
        digest.update(result.stderr)
    return digest.hexdigest()


def roi_input_fingerprints(
        args: argparse.Namespace,
        settings: dict[str, Any],
        graph_path: Path | None,
        benchmark: str,
        effective_env: dict[str, str] | None = None) -> dict[str, str]:
    source_env = effective_env or os.environ
    paths = {
        "manifest": resolve_path(str(args.manifest)),
        "paper_run": Path(__file__).resolve(),
        "roi_matrix": ROI_MATRIX,
        "policy_specs": ECG_DIR / "lib" / "policy_specs.py",
    }
    if graph_path is not None:
        paths["graph"] = graph_path

    suite = str(settings.get("suite"))
    if suite == "sniper":
        root = resolve_path(str(settings.get(
            "sniper_root", "bench/include/sniper_sim/snipersim")))
        workload = str(settings.get("sniper_workload", "pr_kernel_smoke"))
        binary_name = (
            "sg_kernel" if workload == "sg_kernel"
            else "pr_kernel_smoke" if workload == "pr_kernel_smoke"
            else f"{benchmark}_kernel_smoke" if workload == "kernel_smoke"
            else benchmark
        )
        paths.update({
            "sniper_runner": root / "run-sniper",
            "sniper_record_trace": root / "record-trace",
            "sniper_binary": root / "lib" / "sniper",
            "sniper_config": root / "config",
            "sniper_runtime_scripts": root / "scripts",
            "sniper_tools": root / "tools",
            "sniper_sde": root / "sde_kit" / "sde64",
            "sniper_sift_recorder": root / "sift" / "recorder" /
            "obj-intel64" / "sde_sift_recorder.so",
            "benchmark_binary": PROJECT_ROOT / "bench" / "bin_sniper" / binary_name,
        })
        setarch = shutil.which("setarch")
        paths["setarch"] = (
            Path(setarch) if setarch else PROJECT_ROOT / ".missing-setarch")
    elif suite == "gem5":
        gem5_opt = Path(source_env.get(
            "GEM5_OPT",
            PROJECT_ROOT / "bench" / "include" / "gem5_sim" /
            "gem5" / "build" / "X86" / "gem5.opt"))
        suffix = source_env.get("GEM5_KERNEL_SUFFIX", "_m5ops")
        paths.update({
            "gem5_binary": gem5_opt,
            "gem5_config": PROJECT_ROOT / "bench" / "include" /
            "gem5_sim" / "configs" / "graphbrew",
            "gem5_benchmark_binary": PROJECT_ROOT / "bench" /
            "bin_gem5" / f"{benchmark}{suffix}",
        })
    elif suite == "cache-sim":
        paths["cache_sim_benchmark_binary"] = (
            PROJECT_ROOT / "bench" / "bin_sim" / benchmark)

    return {
        "git_state": git_state_fingerprint(),
        **{
            name: path_fingerprint(str(path.resolve()))
            for name, path in paths.items()
        },
    }


def find_graph_path(graph: dict[str, Any], graph_dir: Path, allow_missing: bool) -> Path | None:
    if "path" in graph:
        path = resolve_path(str(graph["path"]))
        if path.exists():
            return path
        if allow_missing:
            return path
        raise SystemExit(f"graph file missing for {graph.get('name')}: {path}")

    name = str(graph["name"])
    root = graph_dir / name
    patterns = [f"{name}.sg", "graph.sg", "*.sg", f"{name}.mtx", f"*/{name}.mtx", "*.el"]
    for pattern in patterns:
        matches = sorted(root.glob(pattern)) if root.exists() else []
        for match in matches:
            if match.is_file():
                return match
    if allow_missing:
        return root / f"{name}.sg"
    raise SystemExit(f"could not find graph file for {name} under {graph_dir}")


def graph_uses_synthetic_options(graph: dict[str, Any]) -> bool:
    return str(graph.get("options_key", "file_dbg")).startswith("synthetic_")


def options_for(
    manifest: dict[str, Any],
    graph: dict[str, Any],
    graph_path: Path | None,
    benchmark: str,
) -> str:
    option_key = str(graph.get("options_key", "file_dbg"))
    option_sets = manifest.get("benchmark_options", {})
    if option_key not in option_sets:
        raise SystemExit(f"unknown options_key={option_key!r} for graph {graph.get('name')}")
    options_by_benchmark = option_sets[option_key]
    if benchmark not in options_by_benchmark:
        raise SystemExit(f"missing benchmark options for {benchmark!r} in options_key={option_key!r}")
    template = str(options_by_benchmark[benchmark])
    return template.format(
        graph_name=graph.get("name", ""),
        graph_path=str(graph_path) if graph_path else "",
    )


def merged_defaults(manifest: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(manifest.get("defaults", {}))
    defaults.update(stage)
    return defaults


def scale_size(size_str: str, factor: int) -> str:
    """Multiply a cache size like '2MB' by an integer factor -> '8MB'. Used to hold
    per-core LLC constant across a Sniper multi-core sweep (shared L3 = per_core * cores)."""
    units = {"B": 1, "kB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
    m = re.match(r"^\s*(\d+)\s*([kKmMgG]?[bB])\s*$", str(size_str))
    if not m:
        return str(size_str)
    val = int(m.group(1))
    unit = {"kb": "kB", "mb": "MB", "gb": "GB", "b": "B"}.get(m.group(2).lower(), "B")
    total = val * units[unit] * int(factor)
    for u in ("GB", "MB", "kB", "B"):
        if total % units[u] == 0:
            return f"{total // units[u]}{u}"
    return f"{total}B"


def expand_jobs(args: argparse.Namespace, manifest: dict[str, Any], run_dir: Path) -> list[Job]:
    profiles = set(args.profile)
    stages = manifest.get("stages", [])
    graph_sets = manifest.get("graph_sets", {})
    jobs: list[Job] = []

    for stage in stages:
        stage_profiles = set(stage.get("profiles", []))
        if not profiles.intersection(stage_profiles):
            continue
        if args.only and not any(token in stage["name"] for token in args.only):
            continue
        if args.skip and any(token in stage["name"] for token in args.skip):
            continue
        settings = merged_defaults(manifest, stage)
        kind = str(stage["kind"])
        if kind == "proof_matrix":
            if args.policy:
                continue
            jobs.append(make_proof_job(args, run_dir, settings))
        elif kind == "roi_matrix":
            graph_set_name = str(settings["graph_set"])
            if graph_set_name not in graph_sets:
                raise SystemExit(f"unknown graph_set={graph_set_name!r} in stage {stage['name']}")
            for graph in graph_sets[graph_set_name]:
                graph_name = str(graph["name"])
                if not token_matches(graph_name, args.graph):
                    continue
                graph_path = None
                if not graph_uses_synthetic_options(graph):
                    graph_path = find_graph_path(graph, Path(args.graph_dir), True)
                for benchmark in settings.get("benchmarks", []):
                    if not token_matches(str(benchmark), args.benchmark):
                        continue
                    # Sniper multi-core LLC-constant sweep: hold per-core LLC fixed by
                    # scaling the shared L3 with the core count (per_core_l3 * cores). So
                    # {1,2,4,8} cores at 2MB/core -> {2,4,8,16}MB shared. gem5/cache_sim
                    # stay single-core (no sweep). This is how DROPLET(4c/8MB=2MB/core) and
                    # POPT(8c) core counts are matched at a constant per-core capacity.
                    core_sweep = settings.get("sniper_core_sweep")
                    if core_sweep and str(settings.get("suite")) == "sniper":
                        per_core = str(settings.get("per_core_l3", "2MB"))
                        for cores in core_sweep:
                            js = dict(settings)
                            js["sniper_cores"] = cores
                            js["l3_sizes"] = [scale_size(per_core, int(cores))]
                            js["_core_tag"] = f"c{cores}"
                            jobs.append(make_roi_job(args, manifest, run_dir, js, graph, graph_path, benchmark))
                    else:
                        jobs.append(make_roi_job(args, manifest, run_dir, settings, graph, graph_path, benchmark))
        else:
            raise SystemExit(f"unsupported stage kind={kind!r} in {stage['name']}")

    return jobs


def make_proof_job(args: argparse.Namespace, run_dir: Path, settings: dict[str, Any]) -> Job:
    out_dir = run_dir / "matrices" / str(settings.get("out_subdir", settings["name"]))
    log_path = run_dir / "logs" / f"{sanitize(settings['name'])}.log"
    command = [
        sys.executable,
        str(PROOF_MATRIX),
        "--benchmarks", *settings.get("benchmarks", ["pr", "bfs", "sssp"]),
        "--l1d-size", str(settings["l1d_size"]),
        "--l2-size", str(settings["l2_size"]),
        "--l3-sizes", *settings.get("l3_sizes", ["4kB"]),
        "--l3-ways", str(settings["l3_ways"]),
        "--line-size", str(settings["line_size"]),
        "--out-dir", str(out_dir),
        "--timeout-cache", str(settings["timeout_cache"]),
    ]
    if settings.get("no_build", True) or args.no_build:
        command.append("--no-build")
    if args.dry_run:
        command.append("--dry-run")
    proof_settings = {**settings, "suite": "cache-sim"}
    inputs: dict[str, str] = {}
    for benchmark in settings.get("benchmarks", ["pr", "bfs", "sssp"]):
        for name, value in roi_input_fingerprints(
                args, proof_settings, None, str(benchmark)).items():
            inputs[f"{benchmark}:{name}"] = value
    inputs["proof_matrix"] = path_fingerprint(str(PROOF_MATRIX.resolve()))
    material_env = {
        key: value for key, value in os.environ.items()
        if key.startswith((
            "CACHE_", "ECG_", "GEM5_", "SNIPER_", "OMP_"))
    }
    config_hash = hashlib.sha256(json.dumps(
        {"command": command, "env": material_env, "inputs": inputs},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    matrix_id = sanitize(str(settings["name"]))
    shard_group = os.environ.get(
        "GRAPHBREW_SHARD_GROUP", run_dir.name)
    env = {
        "GRAPHBREW_MATRIX_CONFIG_HASH": config_hash,
        "GRAPHBREW_MATRIX_ID": matrix_id,
        "GRAPHBREW_SHARD_GROUP": shard_group,
    }
    return Job(
        job_id=sanitize(str(settings["name"])),
        stage=str(settings["name"]),
        kind="proof_matrix",
        command=command,
        out_dir=out_dir,
        log_path=log_path,
        env=env,
        metadata={
            "stage": settings["name"],
            "matrix_id": matrix_id,
            "config_hash": config_hash,
            "matrix_config_hash": config_hash,
            "input_fingerprints": inputs,
        },
    )


def make_roi_job(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    run_dir: Path,
    settings: dict[str, Any],
    graph: dict[str, Any],
    graph_path: Path | None,
    benchmark: str,
) -> Job:
    graph_name = str(graph["name"])
    # Per-graph paper-faithful cell: a graph entry may carry its own cache cell
    # (l1d_size/l2_size/l3_ways/l3_sizes/line_size and structure_prefetch_degree)
    # so each real graph runs at a realistic hierarchy. These override the
    # stage/default. EXCEPTION: during a Sniper multi-core sweep the shared L3 is
    # computed from per_core_l3 * cores, so the graph's l3_sizes must NOT clobber it.
    settings = dict(settings)
    _skip = {"l3_sizes"} if settings.get("_core_tag") else set()
    for _cell_key in (
        "l1d_size", "l2_size", "l3_ways", "l3_sizes", "line_size",
        "structure_prefetch_degree",
    ):
        if _cell_key in graph and _cell_key not in _skip:
            settings[_cell_key] = graph[_cell_key]
    options = options_for(manifest, graph, graph_path, benchmark)
    core_tag = str(settings.get("_core_tag", ""))
    scaling_series_id = sanitize(
        f"{settings['name']}_{graph_name}_{benchmark}")
    matrix_id = sanitize(
        f"{settings['name']}_{graph_name}_{benchmark}" +
        (f"_{core_tag}" if core_tag else ""))
    out_dir = run_dir / "matrices" / str(settings.get("out_subdir", settings["name"])) / graph_name / benchmark
    if core_tag:
        out_dir = out_dir / core_tag
    job_id = sanitize(f"{settings['name']}_{graph_name}_{benchmark}" + (f"_{core_tag}" if core_tag else ""))
    all_policies = [
        str(policy) for policy in settings.get("policies", [])]
    policies = filter_policy_specs(all_policies, args.policy)
    if not policies:
        raise SystemExit(
            f"no policies selected for stage={settings['name']} graph={graph_name} benchmark={benchmark}; "
            f"requested filters={args.policy}"
        )
    if args.policy:
        job_id = sanitize(f"{job_id}_{'_'.join(policies)}")
    log_path = run_dir / "logs" / f"{job_id}.log"
    command = [
        sys.executable,
        str(ROI_MATRIX),
        "--suite", str(settings["suite"]),
        "--benchmark", benchmark,
        "--options", options,
        "--policies", *policies,
        "--prefetcher", str(settings.get("prefetcher", "none")),
        "--prefetcher-level", str(settings.get("prefetcher_level", "l2")),
        "--structure-prefetch-degree",
        str(settings.get("structure_prefetch_degree", 4)),
        "--l1d-size", str(settings["l1d_size"]),
        "--l2-size", str(settings["l2_size"]),
        "--l1d-ways", str(settings.get("l1d_ways", "8")),
        "--l2-ways", str(settings.get("l2_ways", "8")),
        "--l3-sizes", *settings.get("l3_sizes", ["4kB"]),
        "--l3-ways", str(settings["l3_ways"]),
        "--line-size", str(settings["line_size"]),
        "--out-dir", str(out_dir),
        "--timeout-cache", str(settings["timeout_cache"]),
        "--timeout-gem5", str(settings["timeout_gem5"]),
        "--timeout-sniper", str(settings.get("timeout_sniper", 600)),
        "--popt-reserve-model",
        str(settings.get("popt_reserve_model", "fixed_one")),
    ]
    if str(settings.get("prefetcher", "none")) == "ECG_PFX":
        command.extend(["--ecg-pfx-mode", str(settings.get("ecg_pfx_mode", "popt"))])
        command.extend(["--ecg-pfx-window", str(settings.get("ecg_pfx_window", 16))])
        command.extend(["--ecg-pfx-lookahead", str(settings.get("ecg_pfx_lookahead", 4))])
        command.extend(["--ecg-pfx-hint-filter", str(settings.get("ecg_pfx_hint_filter", 16))])
        command.extend(["--ecg-pfx-delivery", str(settings.get("ecg_pfx_delivery", "explicit-hint"))])
        if settings.get("allow_gem5_ecg_pfx"):
            command.append("--allow-gem5-ecg-pfx")
    if str(settings.get("prefetcher", "none")) == "DROPLET":
        # Sprint 6f-7 audit: DROPLET prefetcher in cache_sim mode 3 reads
        # `--ecg-pfx-lookahead` (since cache_sim shares the lookahead knob
        # between mode 3 / DROPLET-style and ECG_PFX modes). Without this
        # pass-through, DROPLET silently defaults to lookahead=4 even when
        # the manifest sets ecg_pfx_lookahead=8 or 16. This was caught
        # before the HPCA mode 6 go/no-go run (rubber-duck #1 blocker).
        if "ecg_pfx_lookahead" in settings:
            command.extend(["--ecg-pfx-lookahead", str(settings["ecg_pfx_lookahead"])])
        if "droplet_prefetch_degree" in settings:
            command.extend(["--droplet-prefetch-degree", str(settings["droplet_prefetch_degree"])])
        if "droplet_indirect_degree" in settings:
            command.extend(["--droplet-indirect-degree", str(settings["droplet_indirect_degree"])])
        if "droplet_stride_table_size" in settings:
            command.extend(["--droplet-stride-table-size", str(settings["droplet_stride_table_size"])])
    if str(settings.get("suite")) == "sniper":
        command.extend(["--sniper-workload", str(settings.get("sniper_workload", "pr_kernel_smoke"))])
        command.extend(["--sniper-cores", str(settings.get("sniper_cores", 1))])
        command.extend(["--sniper-root", str(settings.get("sniper_root", "bench/include/sniper_sim/snipersim"))])
        command.extend(["--sniper-frontend", str(settings.get("sniper_frontend", "live"))])
        command.extend(["--sniper-omp-wait-policy", str(settings.get("sniper_omp_wait_policy", "passive"))])
        if settings.get("sniper_roi_icount"):
            command.extend(["--sniper-roi-icount", str(settings.get("sniper_roi_icount"))])
        command.extend(["--sniper-base-config", str(settings.get("sniper_base_config", "graphbrew/graph_sniper"))])
        command.extend(["--sniper-address-domain", str(settings.get("sniper_address_domain", "virtual"))])
        if settings.get("require_sniper_aslr_disable"):
            command.append("--require-sniper-aslr-disable")
        command.extend(["--sniper-memory-limit-gb", str(settings.get("sniper_memory_limit_gb", 16))])
        command.extend(["--sniper-mimicos-memory-mb", str(settings.get("sniper_mimicos_memory_mb", 4096))])
        command.extend(["--sniper-mimicos-kernel-mb", str(settings.get("sniper_mimicos_kernel_mb", 128))])
        if settings.get("allow_sniper_benchmark_workload"):
            command.append("--allow-sniper-benchmark-workload")
        if settings.get("allow_sniper_sg_kernel_workload"):
            command.append("--allow-sniper-sg-kernel-workload")
        if settings.get("sniper_threads"):
            command.extend(["--threads", *[str(value) for value in settings.get("sniper_threads", [])]])
    if settings.get("no_build", True) or args.no_build:
        command.append("--no-build")
    if args.dry_run:
        command.append("--dry-run")
    env = {}
    if "omp_threads" in settings:
        env["OMP_NUM_THREADS"] = str(settings["omp_threads"])
    # Sprint 6f-7 / HPCA mode 6 evaluation: allow manifest stages to set
    # arbitrary env vars (e.g. ECG_EDGE_MASK_CHARGED, ECG_EDGE_MASK_AMPLIFY,
    # ECG_EDGE_MASK_LOOKAHEAD) which the cache_sim/Sniper binaries read
    # directly. These knobs are not first-class roi_matrix.py CLI args yet,
    # so manifest-level env propagation is the cleanest way to pass them
    # while preserving record-of-truth in the resolved manifest.
    stage_env = settings.get("env")
    if stage_env:
        if not isinstance(stage_env, dict):
            raise SystemExit(
                f"stage {settings.get('name')!r} 'env' must be a dict, "
                f"got {type(stage_env).__name__}"
            )
        for k, v in stage_env.items():
            env[str(k)] = str(v)
    material_env = {
        key: value for key, value in os.environ.items()
        if key.startswith((
            "CACHE_", "ECG_", "GEM5_", "SNIPER_", "OMP_"))
    }
    material_env.update(env)
    inputs = roi_input_fingerprints(
        args, settings, graph_path, benchmark, material_env)
    config_hash = hashlib.sha256(json.dumps(
        {"command": command, "env": material_env, "inputs": inputs},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected_policy_labels = [
        policy_output_label(policy) for policy in all_policies]
    matrix_command = list(command)
    policy_start = matrix_command.index("--policies") + 1
    policy_end = matrix_command.index("--prefetcher")
    matrix_command[policy_start:policy_end] = all_policies
    out_index = matrix_command.index("--out-dir") + 1
    matrix_command[out_index] = "<MATRIX_OUT_DIR>"
    matrix_config_hash = hashlib.sha256(json.dumps(
        {"command": matrix_command, "env": material_env, "inputs": inputs},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    shard_group = os.environ.get(
        "GRAPHBREW_SHARD_GROUP", run_dir.name)
    env.update({
        "GRAPHBREW_MATRIX_CONFIG_HASH": config_hash,
        "GRAPHBREW_MATRIX_GROUP_HASH": matrix_config_hash,
        "GRAPHBREW_MATRIX_ID": matrix_id,
        "GRAPHBREW_SHARD_GROUP": shard_group,
        "GRAPHBREW_EXPECTED_POLICY_LABELS": json.dumps(
            expected_policy_labels, separators=(",", ":")),
    })
    return Job(
        job_id=job_id,
        stage=str(settings["name"]),
        kind="roi_matrix",
        command=command,
        out_dir=out_dir,
        log_path=log_path,
        env=env,
        metadata={
            "stage": settings["name"],
            "suite": settings["suite"],
            "graph": graph_name,
            "graph_path": str(graph_path) if graph_path else "",
            "benchmark": benchmark,
            "matrix_id": matrix_id,
            "scaling_series_id": scaling_series_id,
            "per_core_l3_size": str(settings.get(
                "per_core_l3", settings.get("l3_sizes", [""])[0])),
            "l3_sizes": [str(size) for size in settings.get(
                "l3_sizes", ["4kB"])],
            "threads": [
                str(value) for value in (
                    settings.get("sniper_threads")
                    or [settings.get("sniper_cores", 1)])
            ] if str(settings.get("suite")) == "sniper" else [],
            "structure_prefetch_degree": int(
                settings.get("structure_prefetch_degree", 4)
                if settings.get("prefetcher") == "STRIDE" else 0),
            "options": options,
            "policies": policies,
            "expected_policy_labels": expected_policy_labels,
            "config_hash": config_hash,
            "matrix_config_hash": matrix_config_hash,
            "input_fingerprints": inputs,
            "prefetcher": settings.get("prefetcher", "none"),
            # Sprint 6f-7 / HPCA mode 6: record env knobs for reproducibility.
            # Stage env vars (e.g. ECG_EDGE_MASK_CHARGED, _AMPLIFY, _LOOKAHEAD)
            # are not visible in the subprocess command line, so without this
            # metadata field the jobs.csv would not record the actual
            # experimental knobs the binary ran with.
            "env": dict(env),
        },
    )


def csv_status(
        path: Path,
        expected_policies: list[str] | None = None) -> tuple[str, str]:
    if not path.exists():
        return "missing", "output CSV missing"
    try:
        rows = list(csv.DictReader(path.open(newline="")))
    except OSError as exc:
        return "error", str(exc)
    if not rows:
        return "error", "output CSV has no rows"
    statuses = {row.get("status", "") for row in rows}
    if statuses == {"ok"}:
        if expected_policies:
            expected = {
                policy_output_label(policy) for policy in expected_policies}
            actual = {
                row.get("policy_label", "") for row in rows
                if row.get("policy_label")}
            missing = sorted(expected - actual)
            if missing:
                return (
                    "partial",
                    f"missing policies={missing} rows={len(rows)}",
                )
        return "ok", f"{len(rows)} ok rows"
    return "failed", f"statuses={sorted(statuses)} rows={len(rows)}"


def job_csv_status(job: Job) -> tuple[str, str]:
    expected = [
        str(policy) for policy in job.metadata.get("policies", [])]
    status, detail = csv_status(job.output_csv, expected)
    if status != "ok":
        return status, detail
    if job.kind == "proof_matrix":
        marker = job.out_dir / "proof_matrix.complete.json"
        if not marker.exists():
            return "partial", "proof completion marker missing"
        try:
            payload = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return "error", f"invalid proof completion marker: {exc}"
        if (payload.get("complete") is not True or
                payload.get("all_rows_ok") is not True):
            return "partial", "proof completion marker is not successful"
        if payload.get("config_hash") != job.metadata.get("config_hash"):
            return "partial", "proof completion marker config mismatch"
        return status, detail
    if job.kind != "roi_matrix":
        return status, detail

    marker = job.out_dir / "roi_matrix.complete.json"
    if not marker.exists():
        return "partial", "completion marker missing"
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return "error", f"invalid completion marker: {exc}"
    if (payload.get("complete") is not True or
            payload.get("all_rows_ok") is not True):
        return "partial", "completion marker is not successful"

    expected_labels = [policy_output_label(policy) for policy in expected]
    checks = {
        "policy_labels": expected_labels,
        "l3_sizes": list(job.metadata.get("l3_sizes", [])),
        "threads": list(job.metadata.get("threads", [])),
        "structure_prefetch_degree": int(
            job.metadata.get("structure_prefetch_degree", 0)),
        "config_hash": str(job.metadata.get("config_hash", "")),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in checks.items()
        if payload.get(key) != value
    }
    if mismatches:
        return "partial", f"completion marker mismatch={mismatches}"
    return status, detail


def validate_job_graphs(run_dir: Path, jobs: list[Job], strict: bool) -> bool:
    records_by_path: dict[str, dict[str, Any]] = {}
    for job in jobs:
        graph_path = job.metadata.get("graph_path", "")
        if not graph_path:
            continue
        path = Path(str(graph_path))
        key = str(path)
        if key not in records_by_path:
            records_by_path[key] = {
                "path": key,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "jobs": [],
            }
        records_by_path[key]["jobs"].append(job.job_id)

    records = sorted(records_by_path.values(), key=lambda row: row["path"])
    graph_dir = run_dir / "preflight"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "graph_check.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    missing = [record for record in records if not record["exists"]]
    if not records:
        print("[graphs] no file-backed graph paths in selected jobs")
        return True
    if not missing:
        print(f"[graphs] all {len(records)} selected graph file(s) are present")
        return True

    print(f"[graphs] {len(missing)} selected graph file(s) are missing:")
    for record in missing:
        print(f"  - {record['path']} ({len(record['jobs'])} job(s))")
    if not strict:
        print("[graphs] continuing because graph check is non-strict for this mode")
    return False


def latest_run_dir() -> Path:
    if not RESULTS_ROOT.exists():
        raise SystemExit(f"no final-run directory exists under {RESULTS_ROOT}")
    candidates = [path for path in RESULTS_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise SystemExit(f"no final-run directory exists under {RESULTS_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_job_events(run_dir: Path) -> dict[str, dict[str, Any]]:
    status_path = run_dir / "run_status.jsonl"
    if not status_path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in status_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        job_id = str(record.get("job_id", ""))
        if job_id:
            latest[job_id] = record
    return latest


def display_status(row: dict[str, str], latest_events: dict[str, dict[str, Any]]) -> tuple[str, str]:
    status = row.get("status", "")
    detail = row.get("detail", "")
    latest = latest_events.get(row.get("job_id", ""), {})
    if status == "missing" and latest.get("event") == "start":
        return "running", f"started at {latest.get('utc', 'unknown time')} ({detail})"
    if latest.get("event") == "interrupted":
        return "interrupted", str(latest.get("detail", detail))
    if latest.get("event") == "finish":
        latest_status = str(latest.get("output_status", status))
        latest_detail = str(latest.get("detail", detail))
        if int(latest.get("exit_code", 0) or 0) != 0:
            return "failed", latest_detail
        return latest_status, latest_detail
    return status, detail


def print_run_status(run_dir: Path) -> int:
    jobs_path = run_dir / "jobs.csv"
    if not jobs_path.exists():
        print(f"[status] jobs.csv not found: {jobs_path}")
        return 1
    rows = list(csv.DictReader(jobs_path.open(newline="")))
    latest_events = latest_job_events(run_dir)
    counts: dict[str, int] = {}
    for row in rows:
        status, _detail = display_status(row, latest_events)
        counts[status] = counts.get(status, 0) + 1

    print(f"[status] run_dir={run_dir}")
    print(f"[status] jobs={len(rows)} counts={counts}")
    for name in ("combined_roi_matrix.csv", "combined_proof_matrix.csv"):
        path = run_dir / name
        if path.exists():
            with path.open(newline="") as fh:
                count = max(sum(1 for _ in fh) - 1, 0)
            print(f"[status] {name}: {count} row(s)")

    interesting = [row for row in rows if display_status(row, latest_events)[0] != "ok"]
    for row in interesting[:20]:
        status, detail = display_status(row, latest_events)
        print(f"  - {row.get('job_id')}: {status} ({detail})")
    if len(interesting) > 20:
        print(f"  ... {len(interesting) - 20} more non-ok job(s)")
    return 0


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_config_hash(
        args: argparse.Namespace, jobs: list[Job]) -> str:
    payload = {
        "profiles": args.profile,
        "filters": {
            "graph": args.graph,
            "benchmark": args.benchmark,
            "policy": args.policy,
            "job": args.job,
            "only": args.only,
            "skip": args.skip,
        },
        "jobs": [
            {
                "job_id": job.job_id,
                "config_hash": job.metadata.get("config_hash", ""),
                "matrix_config_hash": job.metadata.get(
                    "matrix_config_hash", ""),
            }
            for job in jobs
        ],
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_run_manifest(run_dir: Path, args: argparse.Namespace, manifest: dict[str, Any], jobs: list[Job]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    snapshot = {
        "created_utc": utc_now(),
        "run_config_hash": run_config_hash(args, jobs),
        "profiles": args.profile,
        "filters": {
            "graph": args.graph,
            "benchmark": args.benchmark,
            "policy": args.policy,
            "job": args.job,
            "only": args.only,
            "skip": args.skip,
        },
        "manifest": manifest,
        "jobs": [
            {
                "job_id": job.job_id,
                "stage": job.stage,
                "kind": job.kind,
                "command": job.command,
                "out_dir": str(job.out_dir),
                "log_path": str(job.log_path),
                "metadata": job.metadata,
            }
            for job in jobs
        ],
    }
    (run_dir / "resolved_manifest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    with (run_dir / "jobs.csv").open("w", newline="") as fh:
        fieldnames = ["job_id", "stage", "kind", "status", "detail", "out_dir", "log_path", "command"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            status, detail = job_csv_status(job)
            writer.writerow({
                "job_id": job.job_id,
                "stage": job.stage,
                "kind": job.kind,
                "status": status,
                "detail": detail,
                "out_dir": str(job.out_dir),
                "log_path": str(job.log_path),
                "command": command_text(job.command),
            })


def write_combined_outputs(run_dir: Path, jobs: list[Job]) -> None:
    rows_by_kind: dict[str, list[dict[str, Any]]] = {"roi_matrix": [], "proof_matrix": []}
    shard_group = os.environ.get(
        "GRAPHBREW_SHARD_GROUP", run_dir.name)
    for job in jobs:
        status, detail = job_csv_status(job)
        if not job.output_csv.exists():
            continue
        try:
            with job.output_csv.open(newline="") as fh:
                job_rows = list(csv.DictReader(fh))
        except OSError:
            continue
        if not job_rows:
            continue
        for row in job_rows:
            row.update({
                "final_job_id": job.job_id,
                "final_matrix_id": str(job.metadata.get(
                    "matrix_id", job.job_id)),
                "final_scaling_series_id": str(job.metadata.get(
                    "scaling_series_id",
                    job.metadata.get("matrix_id", job.job_id))),
                "per_core_l3_size": str(job.metadata.get(
                    "per_core_l3_size", "")),
                "final_expected_policy_labels": json.dumps(
                    job.metadata.get("expected_policy_labels", []),
                    separators=(",", ":")),
                "final_matrix_config_hash": str(job.metadata.get(
                    "matrix_config_hash", "")),
                "final_shard_group": shard_group,
                "final_stage": job.stage,
                "final_kind": job.kind,
                "final_output_csv": str(job.output_csv),
                "final_output_status": status,
                "final_output_detail": detail,
                "final_graph": str(job.metadata.get("graph", "")),
                "final_graph_path": str(job.metadata.get("graph_path", "")),
            })
            rows_by_kind[job.kind].append(row)

    for kind, rows in rows_by_kind.items():
        if not rows:
            continue
        out_path = run_dir / f"combined_{kind}.csv"
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[write] {out_path}")


def write_run_completion(run_dir: Path, jobs: list[Job]) -> None:
    statuses = {
        job.job_id: {
            "status": job_csv_status(job)[0],
            "config_hash": job.metadata.get("config_hash", ""),
        }
        for job in jobs
    }
    complete = bool(statuses) and all(
        value["status"] == "ok" for value in statuses.values())
    marker = run_dir / "run.complete.json"
    temp = run_dir / "run.complete.json.tmp"
    try:
        resolved = json.loads(
            (run_dir / "resolved_manifest.json").read_text())
        resolved_hash = str(resolved.get("run_config_hash", ""))
    except (OSError, json.JSONDecodeError):
        resolved_hash = ""
    temp.write_text(json.dumps({
        "complete": complete,
        "run_config_hash": resolved_hash,
        "jobs": statuses,
    }, indent=2, sort_keys=True) + "\n")
    temp.replace(marker)


def write_preflight(run_dir: Path, args: argparse.Namespace) -> None:
    preflight = run_dir / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    (preflight / "argv.txt").write_text(command_text([sys.executable, __file__, *sys.argv[1:]]) + "\n")
    for name, cmd in {
        "git_status.txt": ["git", "--no-pager", "status", "--short"],
        "git_diff_stat.txt": ["git", "--no-pager", "diff", "--stat"],
        "git_head.txt": ["git", "rev-parse", "HEAD"],
    }.items():
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
        (preflight / name).write_text(result.stdout + result.stderr)
    (preflight / "environment.json").write_text(json.dumps({
        "created_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "graph_dir": args.graph_dir,
        "lock_path": str(args.lock_path),
        "filters": {
            "graph": args.graph,
            "benchmark": args.benchmark,
            "policy": args.policy,
        },
    }, indent=2, sort_keys=True) + "\n")


@contextmanager
def run_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(f"pid={os.getpid()} created_utc={utc_now()}\n")
        fh.flush()
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def should_run(job: Job, args: argparse.Namespace) -> tuple[bool, str]:
    status, detail = job_csv_status(job)
    if args.force:
        return True, f"force ({status}: {detail})"
    if status == "ok" and args.resume:
        return False, detail
    if status == "failed" and args.skip_failed:
        return False, detail
    return True, f"{status}: {detail}"


def append_status(run_dir: Path, record: dict[str, Any]) -> None:
    record = {"utc": utc_now(), **record}
    with (run_dir / "run_status.jsonl").open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def terminate_process_group(process: subprocess.Popen[str], log: Any, timeout_s: float = 10.0) -> int:
    log.write("[interrupt_action] SIGTERM process group\n")
    log.flush()
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.write("[interrupt_action] SIGKILL process group\n")
        log.flush()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def run_job(job: Job, run_dir: Path, args: argparse.Namespace) -> int:
    do_run, reason = should_run(job, args)
    if not do_run:
        print(f"[skip] {job.job_id}: {reason}")
        append_status(run_dir, {"job_id": job.job_id, "event": "skip", "reason": reason})
        return 0

    print(f"[run] {job.job_id}: {reason}")
    print(f"      {command_text(job.command)}")
    append_status(run_dir, {"job_id": job.job_id, "event": "start", "reason": reason})
    if args.dry_run:
        append_status(run_dir, {"job_id": job.job_id, "event": "dry_run"})
        return 0

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with job.log_path.open("w") as log:
        log.write(f"$ {command_text(job.command)}\n")
        log.flush()
        process = subprocess.Popen(
            job.command,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, **job.env} if job.env else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            returncode = 130
            log.write("\n[final_paper_run_interrupted] KeyboardInterrupt\n")
            terminate_process_group(process, log)
            log.write(f"[final_paper_run_exit_code] {returncode}\n")
            log.write(f"[final_paper_run_elapsed_s] {time.time() - start:.3f}\n")
            status, detail = job_csv_status(job)
            append_status(run_dir, {
                "job_id": job.job_id,
                "event": "interrupted",
                "exit_code": returncode,
                "output_status": status,
                "detail": detail,
                "elapsed_s": round(time.time() - start, 3),
            })
            print(f"[interrupt] {job.job_id}: output={status} {detail}")
            return returncode
        log.write(f"\n[final_paper_run_exit_code] {returncode}\n")
        log.write(f"[final_paper_run_elapsed_s] {time.time() - start:.3f}\n")
    status, detail = job_csv_status(job)
    append_status(run_dir, {
        "job_id": job.job_id,
        "event": "finish",
        "exit_code": returncode,
        "output_status": status,
        "detail": detail,
        "elapsed_s": round(time.time() - start, 3),
    })
    if returncode != 0 or status != "ok":
        print(f"[fail] {job.job_id}: exit={returncode} output={status} {detail}")
        return returncode or 1
    print(f"[ok] {job.job_id}: {detail}")
    return 0


def print_job_list(jobs: list[Job]) -> None:
    for index, job in enumerate(jobs, 1):
        status, detail = job_csv_status(job)
        print(f"{index:03d} {job.job_id} [{status}] {detail}")
        print(f"    out: {job.out_dir}")
        print(f"    cmd: {command_text(job.command)}")


def filter_jobs(jobs: list[Job], args: argparse.Namespace) -> list[Job]:
    selected = jobs
    if args.job:
        selected = [job for job in selected if any(token in job.job_id for token in args.job)]
    if args.from_job:
        for index, job in enumerate(selected):
            if args.from_job in job.job_id:
                selected = selected[index:]
                break
        else:
            raise SystemExit(f"--from-job token did not match any expanded job: {args.from_job}")
    if args.limit:
        selected = selected[:args.limit]
    return selected


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ECG final-paper experiment profiles.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="JSON final-run manifest.")
    parser.add_argument("--profile", nargs="+", default=["ecg_smoke"], help="Manifest profile(s) to run.")
    parser.add_argument("--run-dir", default="", help="Run directory. Defaults to results/ecg_experiments/final_paper_runs/<profile>_<timestamp>.")
    parser.add_argument("--graph-dir", default=str(PROJECT_ROOT / "results" / "graphs"), help="Graph root for manifest graph names without explicit paths.")
    parser.add_argument("--only", nargs="+", default=[], help="Only stages whose name contains one of these tokens.")
    parser.add_argument("--skip", nargs="+", default=[], help="Skip stages whose name contains one of these tokens.")
    parser.add_argument("--graph", nargs="+", default=[], help="Only graph names matching these exact normalized tokens.")
    parser.add_argument("--benchmark", nargs="+", default=[], help="Only benchmarks matching these exact normalized tokens, e.g. pr bfs sssp.")
    parser.add_argument("--policy", nargs="+", default=[], help="Only ROI policies matching these exact normalized labels, e.g. LRU POPT_CHARGED ECG_DBG_PRIMARY.")
    parser.add_argument("--job", nargs="+", default=[], help="Only jobs whose job_id contains one of these tokens.")
    parser.add_argument("--from-job", default="", help="Start from the first expanded job whose job_id contains this token.")
    parser.add_argument("--limit", type=int, default=0, help="Run/list only the first N jobs after other filters.")
    parser.add_argument("--list", action="store_true", help="List expanded jobs and exit.")
    parser.add_argument("--status", action="store_true", help="Summarize an existing run directory and exit. Uses latest run if --run-dir is omitted.")
    parser.add_argument("--check-graphs", action="store_true", help="Check selected job graph paths and exit without running jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running jobs.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip jobs with all-ok output CSVs.")
    parser.add_argument("--force", action="store_true", help="Run jobs even when output already exists.")
    parser.add_argument("--skip-failed", action="store_true", help="Do not rerun jobs with existing failed CSV rows.")
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=True, help="Stop after first failed job.")
    parser.add_argument("--no-build", action="store_true", help="Pass --no-build to underlying runners.")
    parser.add_argument("--allow-missing-graphs", action="store_true", help="Allow dry-run/list expansion when large graph files are not present.")
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK, help="Advisory lock path for long gem5 runs.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.status:
        status_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
        if not status_dir.is_absolute():
            status_dir = PROJECT_ROOT / status_dir
        return print_run_status(status_dir)

    manifest_path = resolve_path(args.manifest)
    manifest = load_manifest(manifest_path)
    run_dir = Path(args.run_dir) if args.run_dir else RESULTS_ROOT / f"{'_'.join(args.profile)}_{now_tag()}"
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    jobs = filter_jobs(expand_jobs(args, manifest, run_dir), args)
    if not jobs:
        raise SystemExit(
            "no jobs selected; check --profile/--only/--skip/--graph/--benchmark/--policy/--job filters"
        )
    write_run_manifest(run_dir, args, manifest, jobs)
    write_preflight(run_dir, args)

    graph_strict = not (args.allow_missing_graphs or args.dry_run or args.list or args.check_graphs)
    graph_ok = validate_job_graphs(run_dir, jobs, strict=graph_strict)
    if args.check_graphs:
        return 0 if graph_ok else 4
    if not graph_ok and graph_strict:
        return 4

    print(f"[final-run] run_dir={run_dir}")
    print(f"[final-run] profiles={', '.join(args.profile)} jobs={len(jobs)}")
    if args.list:
        print_job_list(jobs)
        return 0

    if not args.dry_run:
        for name in (
            "combined_roi_matrix.csv",
            "combined_proof_matrix.csv",
            "run.complete.json",
        ):
            (run_dir / name).unlink(missing_ok=True)
    failures = 0
    try:
        lock_context = run_lock(args.lock_path) if not args.dry_run else nullcontext()
        with lock_context:
            for job in jobs:
                code = run_job(job, run_dir, args)
                if code != 0:
                    failures += 1
                    if args.stop_on_error:
                        break
    except BlockingIOError:
        print(f"[error] another final-paper run holds lock: {args.lock_path}", file=sys.stderr)
        return 2

    write_run_manifest(run_dir, args, manifest, jobs)
    write_combined_outputs(run_dir, jobs)
    write_run_completion(run_dir, jobs)
    if failures:
        print(f"[final-run] completed with {failures} failure(s)")
        return 1
    print("[final-run] completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))