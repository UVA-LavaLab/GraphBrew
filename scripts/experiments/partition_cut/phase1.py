#!/usr/bin/env python3
"""Run the frozen Phase 1 compact-shard cut-policy matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BFS_BINARY = PROJECT_ROOT / "bench" / "bin" / "bfs_p"


@dataclass(frozen=True)
class Policy:
    name: str
    options: tuple[str, ...]
    deterministic_required: bool


POLICIES = {
    "original": Policy("original", (), True),
    "rcm_bnf": Policy("rcm_bnf", ("-o", "11:bnf"), True),
    "gorder_csr": Policy("gorder_csr", ("-o", "9:csr"), True),
    "comm_cut_min": Policy(
        "comm_cut_min",
        (
            "-o",
            "12:leiden:compose:comm_cut_min:intra_hubsort",
        ),
        False,
    ),
}

FINGERPRINT_FIELDS = (
    "mapping_fingerprint",
    "source_topology_fingerprint",
    "shard_fingerprint",
    "ghost_fingerprint",
    "depth_fingerprint",
)


def _parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def _latest_record(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"unexpected benchmark database shape: {path}")
    return data[-1]


def _extract_record(
    *,
    policy: Policy,
    threads: int,
    repeat: int,
    wall_seconds: float,
    report: dict[str, Any],
) -> dict[str, Any]:
    trial = report["trial_details"][-1]
    answer = trial["answer"]
    if not trial.get("verified", False):
        raise RuntimeError(
            f"{policy.name} threads={threads} repeat={repeat} failed verification"
        )
    if "diagnostics_error" in answer:
        raise RuntimeError(
            f"{policy.name} diagnostics failed: {answer['diagnostics_error']}"
        )
    partition = answer["partition"]
    return {
        "policy": policy.name,
        "deterministic_required": policy.deterministic_required,
        "threads": threads,
        "repeat": repeat,
        "wall_seconds": wall_seconds,
        "bfs_seconds": trial["time_seconds"],
        "reorder_seconds": report["reorder_time"],
        "partition_seconds": partition["build_seconds"],
        "diagnostics_seconds": partition["diagnostics_seconds"],
        "remote_out_fraction": partition["remote_out_fraction"],
        "remote_in_fraction": partition["remote_in_fraction"],
        "max_remote_out_fraction": partition["max_remote_out_fraction"],
        "max_remote_in_fraction": partition["max_remote_in_fraction"],
        "ghost_count": partition["ghost_count"],
        "ghost_bytes": partition["ghost_bytes"],
        "ghost_byte_fraction": partition["ghost_byte_fraction"],
        "total_shard_bytes": partition["total_shard_bytes"],
        "max_shard_bytes": partition["max_shard_bytes"],
        "vertex_imbalance": partition["vertex_imbalance"],
        "balance_imbalance": partition["balance_imbalance"],
        "out_edge_imbalance": partition["out_edge_imbalance"],
        "in_edge_imbalance": partition["in_edge_imbalance"],
        "storage_imbalance": partition["storage_imbalance"],
        "mapping_fingerprint": partition["mapping_fingerprint"],
        "source_topology_fingerprint": partition["source_topology_fingerprint"],
        "shard_fingerprint": partition["shard_fingerprint"],
        "ghost_fingerprint": partition["ghost_fingerprint"],
        "source_id": answer["source_id"],
        "depth_fingerprint": answer["depth_fingerprint"],
        "reachable_vertices": answer["reachable_vertices"],
        "max_depth": answer["max_depth"],
    }


def _run_case(
    *,
    policy: Policy,
    graph_args: list[str],
    threads: int,
    repeat: int,
    partitions: int,
    balance: str,
    source: int,
    timeout: int,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = output_root / f"{policy.name}-t{threads}-r{repeat}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    command = [
        str(BFS_BINARY),
        *graph_args,
        "-n",
        "1",
        "-r",
        str(source),
        "-v",
        "-P",
        str(partitions),
        "-B",
        balance,
        "-D",
        str(run_dir) + "/",
        *policy.options,
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    wall_seconds = time.monotonic() - started
    (run_dir / "stdout.log").write_text(completed.stdout)
    (run_dir / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{policy.name} threads={threads} repeat={repeat} failed "
            f"with exit {completed.returncode}; see {run_dir}"
        )
    return _extract_record(
        policy=policy,
        threads=threads,
        repeat=repeat,
        wall_seconds=wall_seconds,
        report=_latest_record(run_dir / "benchmarks.json"),
    )


def policy_is_stable(records: list[dict[str, Any]]) -> bool:
    if not records:
        return False
    first = records[0]
    return all(
        all(record[field] == first[field] for field in FINGERPRINT_FIELDS)
        for record in records[1:]
    )


def validate_cross_policy(records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("partition cut matrix produced no records")
    topology = records[0]["source_topology_fingerprint"]
    depth = records[0]["depth_fingerprint"]
    source = records[0]["source_id"]
    for record in records[1:]:
        if record["source_topology_fingerprint"] != topology:
            raise RuntimeError(
                f"source topology changed under policy {record['policy']}"
            )
        if record["depth_fingerprint"] != depth:
            raise RuntimeError(
                f"BFS source-depth result changed under policy {record['policy']}"
            )
        if record["source_id"] != source:
            raise RuntimeError(
                f"source ID changed under policy {record['policy']}"
            )


def summarize_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty policy record set")

    def mean(field: str) -> float:
        return statistics.fmean(float(record[field]) for record in records)

    timing_by_threads: dict[str, dict[str, float | int]] = {}
    for threads in sorted({int(record["threads"]) for record in records}):
        selected = [
            record for record in records
            if int(record["threads"]) == threads
        ]

        def thread_mean(field: str) -> float:
            return statistics.fmean(
                float(record[field]) for record in selected
            )

        timing_by_threads[str(threads)] = {
            "runs": len(selected),
            "mean_wall_seconds": thread_mean("wall_seconds"),
            "mean_bfs_seconds": thread_mean("bfs_seconds"),
            "mean_reorder_seconds": thread_mean("reorder_seconds"),
            "mean_partition_seconds": thread_mean("partition_seconds"),
            "mean_diagnostics_seconds": thread_mean("diagnostics_seconds"),
        }
    primary_threads = max(int(value) for value in timing_by_threads)
    primary_timing = timing_by_threads[str(primary_threads)]
    first = records[0]
    return {
        "policy": first["policy"],
        "deterministic_required": first["deterministic_required"],
        "stable": policy_is_stable(records),
        "runs": len(records),
        "primary_threads": primary_threads,
        "timing_by_threads": timing_by_threads,
        "mean_wall_seconds": primary_timing["mean_wall_seconds"],
        "mean_bfs_seconds": primary_timing["mean_bfs_seconds"],
        "mean_reorder_seconds": primary_timing["mean_reorder_seconds"],
        "mean_partition_seconds": primary_timing["mean_partition_seconds"],
        "mean_diagnostics_seconds": primary_timing["mean_diagnostics_seconds"],
        "remote_fraction": max(
            mean("remote_out_fraction"),
            mean("remote_in_fraction"),
        ),
        "max_remote_fraction": max(
            mean("max_remote_out_fraction"),
            mean("max_remote_in_fraction"),
        ),
        "ghost_bytes": round(mean("ghost_bytes")),
        "ghost_byte_fraction": mean("ghost_byte_fraction"),
        "max_shard_bytes": round(mean("max_shard_bytes")),
        "max_edge_imbalance": max(
            mean("out_edge_imbalance"),
            mean("in_edge_imbalance"),
        ),
        "balance_imbalance": mean("balance_imbalance"),
        "storage_imbalance": mean("storage_imbalance"),
        "mapping_fingerprint": first["mapping_fingerprint"],
        "source_topology_fingerprint": first["source_topology_fingerprint"],
        "shard_fingerprint": first["shard_fingerprint"],
        "ghost_fingerprint": first["ghost_fingerprint"],
        "depth_fingerprint": first["depth_fingerprint"],
    }


def add_relative_metrics(summaries: list[dict[str, Any]]) -> None:
    baseline = next(
        (summary for summary in summaries if summary["policy"] == "original"),
        None,
    )
    if baseline is None:
        raise RuntimeError("Phase 1 matrix requires the ORIGINAL baseline")
    for summary in summaries:
        primary_threads = str(summary["primary_threads"])
        if primary_threads not in baseline["timing_by_threads"]:
            raise RuntimeError(
                f"ORIGINAL has no timing for OMP={primary_threads}"
            )
        baseline_timing = baseline["timing_by_threads"][primary_threads]
        timing = summary["timing_by_threads"][primary_threads]
        baseline_preprocess = (
            baseline_timing["mean_reorder_seconds"]
            + baseline_timing["mean_partition_seconds"]
            + baseline_timing["mean_diagnostics_seconds"]
        )
        preprocess = (
            timing["mean_reorder_seconds"]
            + timing["mean_partition_seconds"]
            + timing["mean_diagnostics_seconds"]
        )
        summary["remote_reduction"] = (
            baseline["remote_fraction"] / summary["remote_fraction"]
            if summary["remote_fraction"] > 0
            else float("inf")
        )
        summary["ghost_reduction"] = (
            baseline["ghost_bytes"] / summary["ghost_bytes"]
            if summary["ghost_bytes"] > 0
            else float("inf")
        )
        summary["max_shard_ratio"] = (
            summary["max_shard_bytes"] / baseline["max_shard_bytes"]
            if baseline["max_shard_bytes"] > 0
            else 1.0
        )
        summary["bfs_speedup"] = (
            baseline_timing["mean_bfs_seconds"] /
            timing["mean_bfs_seconds"]
            if timing["mean_bfs_seconds"] > 0
            else float("inf")
        )
        summary["preprocess_ratio"] = (
            preprocess / baseline_preprocess
            if baseline_preprocess > 0
            else float("inf")
        )
        summary["passes_widening_cut_gate"] = (
            summary["remote_reduction"] >= 1.25
            and summary["ghost_reduction"] >= 1.25
        )
        summary["passes_work_balance_gate"] = (
            summary["balance_imbalance"] <= 1.05
        )
        summary["passes_capacity_gate"] = (
            summary["max_shard_ratio"] <= 1.10
        )


def _print_summary(summaries: list[dict[str, Any]]) -> None:
    primary_threads = summaries[0]["primary_threads"] if summaries else 0
    print(
        "policy          stable remote    cut_gain ghost_gain work_imb "
        f"max_shard reorder_s@{primary_threads} bfs_s@{primary_threads}"
    )
    for summary in summaries:
        print(
            f"{summary['policy']:<15} "
            f"{str(summary['stable']):<6} "
            f"{summary['remote_fraction']:<9.5f} "
            f"{summary['remote_reduction']:<8.2f} "
            f"{summary['ghost_reduction']:<10.2f} "
            f"{summary['balance_imbalance']:<8.3f} "
            f"{summary['max_shard_ratio']:<9.3f} "
            f"{summary['mean_reorder_seconds']:<9.4f} "
            f"{summary['mean_bfs_seconds']:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    graph_group = parser.add_mutually_exclusive_group(required=True)
    graph_group.add_argument("--graph", type=Path)
    graph_group.add_argument("--scale", type=int)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(POLICIES),
        default=list(POLICIES),
    )
    parser.add_argument("--threads", type=_parse_ints, default=[1, os.cpu_count() or 1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--partitions", type=int, default=16)
    parser.add_argument(
        "--balance",
        choices=("vertices", "out", "total"),
        default="total",
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "partition_cut" / "phase1",
    )
    args = parser.parse_args()
    if args.repeats <= 0 or args.partitions <= 0:
        parser.error("--repeats and --partitions must be positive")
    if "original" not in args.policies:
        parser.error("--policies must include original as the relative baseline")

    if not BFS_BINARY.exists():
        raise SystemExit(f"missing {BFS_BINARY}; run `make bfs_p`")
    if args.graph is not None:
        graph = args.graph.resolve()
        if not graph.exists():
            raise SystemExit(f"missing graph: {graph}")
        graph_args = ["-f", str(graph)]
        graph_label = graph.stem
    else:
        if args.scale <= 0:
            parser.error("--scale must be positive")
        graph_args = ["-g", str(args.scale)]
        graph_label = f"kron-s{args.scale}"

    output_root = args.output.resolve() / graph_label
    output_root.mkdir(parents=True, exist_ok=True)
    selected = [POLICIES[name] for name in args.policies]
    records: list[dict[str, Any]] = []
    for policy in selected:
        for threads in args.threads:
            for repeat in range(args.repeats):
                print(
                    f"[partition-cut] graph={graph_label} policy={policy.name} "
                    f"threads={threads} repeat={repeat + 1}/{args.repeats}",
                    flush=True,
                )
                records.append(
                    _run_case(
                        policy=policy,
                        graph_args=graph_args,
                        threads=threads,
                        repeat=repeat,
                        partitions=args.partitions,
                        balance=args.balance,
                        source=args.source,
                        timeout=args.timeout,
                        output_root=output_root,
                    )
                )

    validate_cross_policy(records)
    summaries = [
        summarize_policy(
            [record for record in records if record["policy"] == policy.name]
        )
        for policy in selected
    ]
    add_relative_metrics(summaries)
    determinism_failures = [
        summary["policy"]
        for summary in summaries
        if summary["deterministic_required"] and not summary["stable"]
    ]
    result = {
        "schema": "graphbrew.partition_cut.phase1.v1",
        "graph": graph_label,
        "graph_args": graph_args,
        "partitions": args.partitions,
        "balance": args.balance,
        "source": args.source,
        "threads": args.threads,
        "repeats": args.repeats,
        "policies": [asdict(policy) for policy in selected],
        "valid": not determinism_failures,
        "determinism_failures": determinism_failures,
        "summaries": summaries,
        "records": records,
    }
    result_path = output_root / "phase1_summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print_summary(summaries)
    print(f"[partition-cut] wrote {result_path}")
    if determinism_failures:
        raise RuntimeError(
            "required deterministic policies are unstable: "
            + ", ".join(determinism_failures)
            + f"; see {result_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
