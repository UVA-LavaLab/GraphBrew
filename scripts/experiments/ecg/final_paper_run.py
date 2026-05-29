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
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Allow `from literature_preflight import ...` whether this script is
# invoked directly (python final_paper_run.py) or loaded by an importer
# such as spec_from_file_location in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from literature_preflight import snapshot_preflight  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ECG_DIR = PROJECT_ROOT / "scripts" / "experiments" / "ecg"
ROI_MATRIX = ECG_DIR / "roi_matrix.py"
PROOF_MATRIX = ECG_DIR / "proof_matrix.py"
DEFAULT_MANIFEST = ECG_DIR / "final_paper_manifest.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "ecg_experiments" / "final_paper_runs"
DEFAULT_LOCK = Path(os.environ.get("GRAPHBREW_FINAL_RUN_LOCK", "/tmp/graphbrew_final_paper_run.lock"))
VALIDATION_GATE_CSVS = (
    PROJECT_ROOT / "results" / "ecg_experiments" / "proof_matrix" / "component_g12_l3_4kb_grasp_rrpv0" / "proof_matrix.csv",
    PROJECT_ROOT / "results" / "ecg_experiments" / "roi_matrix" / "gem5_grasp_dbg_parity_g10_post_insertion_fix" / "pr" / "roi_matrix.csv",
    PROJECT_ROOT / "results" / "ecg_experiments" / "roi_matrix" / "gem5_grasp_dbg_parity_g10_post_insertion_fix" / "bfs" / "roi_matrix.csv",
    PROJECT_ROOT / "results" / "ecg_experiments" / "roi_matrix" / "gem5_grasp_dbg_parity_g10_post_insertion_fix" / "sssp" / "roi_matrix.csv",
    PROJECT_ROOT / "results" / "ecg_experiments" / "roi_matrix" / "gem5_popt_current_vertex_hint_pr_g12_selected_v2" / "roi_matrix.csv",
    PROJECT_ROOT / "results" / "ecg_experiments" / "roi_matrix" / "gem5_popt_current_vertex_hint_sssp_g12_selected_v2" / "roi_matrix.csv",
    PROJECT_ROOT / "results" / "ecg_experiments" / "roi_matrix" / "pr_g12_l3_4kb_gem5_droplet_actual_edge_proof" / "roi_matrix.csv",
)


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
    return [policy for policy in policies if token_matches(policy, filters)]


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
    return Job(
        job_id=sanitize(str(settings["name"])),
        stage=str(settings["name"]),
        kind="proof_matrix",
        command=command,
        out_dir=out_dir,
        log_path=log_path,
        metadata={"stage": settings["name"]},
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
    options = options_for(manifest, graph, graph_path, benchmark)
    out_dir = run_dir / "matrices" / str(settings.get("out_subdir", settings["name"])) / graph_name / benchmark
    job_id = sanitize(f"{settings['name']}_{graph_name}_{benchmark}")
    policies = filter_policy_specs([str(policy) for policy in settings.get("policies", [])], args.policy)
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
        "--l1d-size", str(settings["l1d_size"]),
        "--l2-size", str(settings["l2_size"]),
        "--l3-sizes", *settings.get("l3_sizes", ["4kB"]),
        "--l3-ways", str(settings["l3_ways"]),
        "--line-size", str(settings["line_size"]),
        "--out-dir", str(out_dir),
        "--timeout-cache", str(settings["timeout_cache"]),
        "--timeout-gem5", str(settings["timeout_gem5"]),
        "--timeout-sniper", str(settings.get("timeout_sniper", 600)),
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
        command.extend(["--sniper-base-config", str(settings.get("sniper_base_config", "graphbrew/graph_sniper"))])
        command.extend(["--sniper-address-domain", str(settings.get("sniper_address_domain", "virtual"))])
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
            "options": options,
            "policies": policies,
            "prefetcher": settings.get("prefetcher", "none"),
        },
    )


def csv_status(path: Path) -> tuple[str, str]:
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
        return "ok", f"{len(rows)} ok rows"
    return "failed", f"statuses={sorted(statuses)} rows={len(rows)}"


def final_profiles_requested(profiles: list[str]) -> bool:
    return any(profile.startswith("final_") for profile in profiles)


def validate_gate(run_dir: Path, strict: bool) -> bool:
    records = []
    ok = True
    for path in VALIDATION_GATE_CSVS:
        status, detail = csv_status(path)
        record = {"path": str(path), "status": status, "detail": detail}
        records.append(record)
        if status != "ok":
            ok = False

    gate_dir = run_dir / "preflight"
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "validation_gate.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    if ok:
        print("[gate] existing faithfulness validation CSVs are present and ok")
        return True

    print("[gate] faithfulness validation gate failed:")
    for record in records:
        if record["status"] != "ok":
            print(f"  - {record['path']}: {record['status']} ({record['detail']})")
    if strict:
        return False
    print("[gate] continuing because gate is non-strict for this profile")
    return True


def validate_literature_gate(run_dir: Path, sweep_root: Path,
                              sweep_subdir: str, strict: bool) -> bool:
    """Run the literature_faithfulness comparator against *sweep_root*.

    Writes ``preflight/literature_gate.json`` summarising the verdicts and
    fails (returns False) if any unregistered ``disagree`` exists. When
    *strict* is False, prints the failures and returns True anyway.
    """

    if not sweep_root.exists():
        print(f"[lit-gate] sweep root {sweep_root} does not exist")
        if strict:
            print("[lit-gate] cannot run gate without sweep data; rerun the literature sweep or pass --skip-literature-gate")
            return False
        return True

    here = Path(__file__).resolve().parent
    lit_script = here / "literature_faithfulness.py"
    if not lit_script.exists():
        print(f"[lit-gate] literature_faithfulness.py missing at {lit_script}")
        return not strict

    gate_dir = run_dir / "preflight"
    gate_dir.mkdir(parents=True, exist_ok=True)
    json_out = gate_dir / "literature_gate.json"
    md_out = gate_dir / "literature_gate.md"

    cmd = [
        sys.executable, str(lit_script),
        "--sweep-root", str(sweep_root),
        "--sweep-subdir", sweep_subdir,
        "--json-out", str(json_out),
        "--md-out", str(md_out),
        "--no-exit-on-disagree",
    ]
    print(f"[lit-gate] running {' '.join(cmd)}")
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        print(f"[lit-gate] failed to launch comparator: {exc}")
        return not strict

    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    try:
        payload = json.loads(json_out.read_text()) if json_out.exists() else {}
    except json.JSONDecodeError as exc:
        print(f"[lit-gate] could not parse comparator JSON: {exc}")
        return not strict

    summary = payload.get("summary", {})
    disagreements = payload.get("disagreements", [])
    print(
        f"[lit-gate] verdicts: ok={summary.get('ok', 0)} "
        f"within_tol={summary.get('within_tolerance', 0)} "
        f"DISAGREE={summary.get('disagree', 0)} "
        f"known_deviation={summary.get('known_deviation', 0)} "
        f"missing={summary.get('missing', 0)} "
        f"insufficient={summary.get('insufficient_data', 0)}"
    )

    if disagreements:
        print("[lit-gate] DISAGREEMENTS:")
        for d in disagreements:
            delta = d.get("delta_pct")
            delta_s = f"{delta:+.3f}pp" if delta is not None else "n/a"
            print(
                f"  - {d['graph']}/{d['app']} L3={d['l3_size']} {d['policy']}: "
                f"Δ={delta_s}  ({d.get('citation', '')})"
            )

    if summary.get("disagree", 0) > 0:
        if strict:
            print("[lit-gate] gate FAILED — refusing to start jobs. Investigate "
                  "disagreements or register a known-deviation in "
                  "literature_baselines.KNOWN_DEVIATIONS before re-running.")
            return False
        print("[lit-gate] continuing because gate is non-strict for this profile")
    else:
        print("[lit-gate] gate PASSED")
    return True


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


def write_run_manifest(run_dir: Path, args: argparse.Namespace, manifest: dict[str, Any], jobs: list[Job]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    snapshot = {
        "created_utc": utc_now(),
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
            status, detail = csv_status(job.output_csv)
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
    for job in jobs:
        status, detail = csv_status(job.output_csv)
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
    status, detail = csv_status(job.output_csv)
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
            status, detail = csv_status(job.output_csv)
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
    status, detail = csv_status(job.output_csv)
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
        status, detail = csv_status(job.output_csv)
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
    parser.add_argument("--profile", nargs="+", default=["rehearsal"], help="Manifest profile(s) to run.")
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
    parser.add_argument("--skip-validation-gate", action="store_true", help="Do not require existing faithfulness-validation CSVs before final_* profiles.")
    parser.add_argument("--require-validation-gate", action="store_true", help="Require existing faithfulness-validation CSVs for any profile.")
    parser.add_argument(
        "--literature-gate-root", type=Path, default=None,
        help="Sweep root to run the literature_faithfulness comparator against "
             "before launching jobs. When a final_cache_sim profile is requested "
             "(or --require-literature-gate is set) and this points at a sweep "
             "tree, jobs are blocked unless every per-graph claim is ok/within "
             "tolerance/known-deviation.",
    )
    parser.add_argument(
        "--literature-gate-subdir", default="lit",
        help="Sub-directory under <graph>-<app>/ that holds roi_matrix.csv "
             "(default: lit).",
    )
    parser.add_argument(
        "--require-literature-gate", action="store_true",
        help="Force the literature-faithfulness gate even when no final_ "
             "profile is requested. On final_ profiles the snapshot gate "
             "runs by default; this flag is only needed to enforce it on "
             "other profiles.",
    )
    parser.add_argument(
        "--skip-literature-gate", action="store_true",
        help="Bypass both the snapshot and live literature-faithfulness "
             "gates even on final_ profiles.",
    )
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

    gate_required = args.require_validation_gate or final_profiles_requested(args.profile)
    if gate_required and not args.skip_validation_gate:
        if not validate_gate(run_dir, strict=True):
            return 3

    # Snapshot-based literature pre-flight: same opt-out semantics as
    # paper_pipeline.py. Skipped on inspection-only runs (dry-run, list,
    # check-graphs) because no real jobs will be dispatched.
    snapshot_gate_requested = (
        final_profiles_requested(args.profile)
        or args.require_literature_gate
    )
    inspection_only = args.dry_run or args.list or args.check_graphs
    if (
        snapshot_gate_requested
        and not args.skip_literature_gate
        and not inspection_only
    ):
        snap_rc = snapshot_preflight()
        if snap_rc != 0:
            return 5

    cache_sim_final_requested = any(
        prof == "final_cache_sim" or prof.startswith("final_cache_sim_")
        for prof in args.profile
    )
    lit_gate_required = (
        args.require_literature_gate
        or (cache_sim_final_requested and args.literature_gate_root is not None)
    )
    if lit_gate_required and not args.skip_literature_gate:
        if args.literature_gate_root is None:
            # The snapshot gate above already covered this case for
            # --require-literature-gate; here we only fall through when
            # a live-gate root was supplied.
            pass
        elif not validate_literature_gate(
            run_dir,
            args.literature_gate_root,
            args.literature_gate_subdir,
            strict=True,
        ):
            return 5

    print(f"[final-run] run_dir={run_dir}")
    print(f"[final-run] profiles={', '.join(args.profile)} jobs={len(jobs)}")
    if args.list:
        print_job_list(jobs)
        return 0

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
    if failures:
        print(f"[final-run] completed with {failures} failure(s)")
        return 1
    print("[final-run] completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))