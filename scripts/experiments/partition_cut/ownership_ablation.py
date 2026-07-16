#!/usr/bin/env python3
"""Compare contiguous ranges with frozen-community owner_by_vertex."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.experiments.partition_cut.phase2 import (  # noqa: E402
    GRAPH_CASES,
    PROJECT_ROOT,
    geometric_mean,
    graph_identity,
)


ANALYZER = PROJECT_ROOT / "bench/bin/ownership_analysis"
MARKER = "[OWNERSHIP_ANALYSIS] "


def reduction(baseline: float, candidate: float) -> float | None:
    if candidate > 0:
        return baseline / candidate
    return None if baseline > 0 else 1.0


def reduction_value(value: float | None) -> float:
    return float("inf") if value is None else value


def json_reduction(value: float) -> float | None:
    return None if value == float("inf") else value


def format_reduction(value: float | None) -> str:
    return "elim" if value is None else f"{value:.2f}"


def validate_report(
    report: dict[str, Any],
    requested_partitions: int,
    balance: str,
) -> None:
    def exact_int(value: Any, name: str) -> int:
        if type(value) is not int or value < 0:
            raise RuntimeError(f"ownership {name} must be an integer")
        return value

    def exact_bool(value: Any, name: str) -> bool:
        if type(value) is not bool:
            raise RuntimeError(f"ownership {name} must be a boolean")
        return value

    def number(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise RuntimeError(f"ownership {name} must be numeric")
        return float(value)

    if report.get("schema") != \
            "graphbrew.partition_ownership_analysis.v1":
        raise RuntimeError("unknown ownership analysis schema")
    requested = exact_int(
        report.get("requested_partitions"),
        "requested partition count",
    )
    partitions = exact_int(
        report.get("partitions"), "partition count")
    vertices = exact_int(report.get("vertices"), "vertex count")
    exact_int(report.get("communities"), "community count")
    if (
        requested != requested_partitions
        or partitions != min(requested_partitions, vertices)
        or report["balance"] != balance
        or exact_bool(report.get("analysis_only"), "analysis_only")
            is not True
        or exact_bool(
            report.get("graph_shard_v1_compatible"),
            "graph_shard_v1_compatible",
        ) is not False
        or exact_bool(
            report.get("complete_per_bank_working_set_evaluated"),
            "complete_per_bank_working_set_evaluated",
        ) is not False
    ):
        raise RuntimeError("ownership analysis configuration mismatch")
    exact_bool(
        report.get("assignment_meets_work_balance_gate"),
        "assignment_meets_work_balance_gate",
    )
    for field in (
        "membership_fingerprint",
        "mapping_fingerprint",
    ):
        if not isinstance(report.get(field), str) or not report[field]:
            raise RuntimeError(
                f"ownership {field} must be a string")
    for arm in ("contiguous", "owner_by_vertex"):
        metrics = report[arm]
        if not isinstance(metrics, dict):
            raise RuntimeError("ownership metrics must be an object")
        metric_partitions = exact_int(
            metrics.get("partition_count"),
            f"{arm} partition count",
        )
        for field in (
            "ghost_slots",
            "ghost_bytes",
            "ownership_metadata_bytes",
            "bfs_bytes_per_superstep",
            "pr_bytes_per_iteration",
            "cc_bytes_per_iteration",
            "spmv_initial_bytes",
            "compact_total_storage_lower_bound_bytes",
            "compact_max_storage_lower_bound_bytes",
        ):
            exact_int(metrics.get(field), f"{arm}.{field}")
        for field in (
            "remote_out_fraction",
            "remote_in_fraction",
            "max_remote_out_fraction",
            "max_remote_in_fraction",
            "vertex_imbalance",
            "out_edge_imbalance",
            "in_edge_imbalance",
            "balance_imbalance",
            "compact_storage_lower_bound_imbalance",
        ):
            number(metrics.get(field), f"{arm}.{field}")
        if (
            not isinstance(metrics.get("owner_fingerprint"), str)
            or not metrics["owner_fingerprint"]
        ):
            raise RuntimeError(
                f"ownership {arm} fingerprint is invalid")
        per_shard = metrics.get("per_shard")
        if not isinstance(per_shard, dict):
            raise RuntimeError(
                f"ownership {arm} per_shard is invalid")
        for field in (
            "owned_vertices",
            "out_edges",
            "in_edges",
            "ghost_slots",
            "ownership_metadata_bytes",
            "compact_storage_lower_bound_bytes",
        ):
            values = per_shard.get(field)
            if (
                not isinstance(values, list)
                or len(values) != partitions
                or any(type(value) is not int or value < 0 for value in values)
            ):
                raise RuntimeError(
                    f"ownership {arm}.{field} is invalid")
        if metrics["remote_out_fraction"] != metrics["remote_in_fraction"]:
            raise RuntimeError("ownership remote totals disagree")
        if metric_partitions != partitions:
            raise RuntimeError("ownership partition count mismatch")


def validate_matrix_axes(
    graphs: list[str],
    threads: list[int],
    repeats: int,
) -> None:
    if (
        repeats < 2
        or len(set(threads)) < 2
        or len(threads) != len(set(threads))
        or any(value <= 0 for value in threads)
        or len(graphs) != len(set(graphs))
    ):
        raise ValueError(
            "ownership matrix requires unique graphs/threads "
            "and at least two repeats/thread counts")


def run_case(
    *,
    graph_name: str,
    graph_path: Path,
    threads: int,
    repeat: int,
    partitions: int,
    balance: str,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = output_root / graph_name / f"t{threads}-r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(ANALYZER),
        "-f",
        str(graph_path),
        "-P",
        str(partitions),
        "-B",
        balance,
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wall_seconds = time.monotonic() - started
    (run_dir / "stdout.log").write_text(completed.stdout)
    (run_dir / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ownership analysis failed for {graph_name} "
            f"OMP={threads} repeat={repeat}"
        )
    payloads = [
        line[len(MARKER):]
        for line in completed.stdout.splitlines()
        if line.startswith(MARKER)
    ]
    if len(payloads) != 1:
        raise RuntimeError(
            f"ownership analysis emitted {len(payloads)} records")
    report = json.loads(payloads[0])
    validate_report(report, partitions, balance)
    return {
        "threads": threads,
        "repeat": repeat,
        "wall_seconds": wall_seconds,
        "command": command,
        "graph": graph_identity(graph_path),
        "analyzer": graph_identity(ANALYZER),
        "report": report,
    }


def summarize_graph(records: list[dict[str, Any]]) -> dict[str, Any]:
    fingerprints = {
        (
            record["report"]["membership_fingerprint"],
            record["report"]["mapping_fingerprint"],
            record["report"]["contiguous"]["owner_fingerprint"],
            record["report"]["owner_by_vertex"]["owner_fingerprint"],
        )
        for record in records
    }
    if len(fingerprints) != 1:
        raise RuntimeError(
            "ownership analysis changed across repeats/thread counts")
    canonical_reports = {
        json.dumps(
            record["report"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    }
    if len(canonical_reports) != 1:
        raise RuntimeError(
            "ownership metrics changed across repeats/thread counts")
    report = records[0]["report"]
    contiguous = report["contiguous"]
    owner = report["owner_by_vertex"]
    baseline_remote = max(
        contiguous["remote_out_fraction"],
        contiguous["remote_in_fraction"],
    )
    owner_remote = max(
        owner["remote_out_fraction"],
        owner["remote_in_fraction"],
    )
    return {
        "stable": True,
        "assignment_meets_work_balance_gate":
            report["assignment_meets_work_balance_gate"],
        "communities": report["communities"],
        "membership_fingerprint": report["membership_fingerprint"],
        "mapping_fingerprint": report["mapping_fingerprint"],
        "contiguous": contiguous,
        "owner_by_vertex": owner,
        "remote_reduction": reduction(
            baseline_remote, owner_remote),
        "ghost_reduction": reduction(
            contiguous["ghost_bytes"], owner["ghost_bytes"]),
        "bfs_halo_reduction": reduction(
            contiguous["bfs_bytes_per_superstep"],
            owner["bfs_bytes_per_superstep"],
        ),
        "max_compact_storage_lower_bound_ratio": (
            owner["compact_max_storage_lower_bound_bytes"] /
            contiguous["compact_max_storage_lower_bound_bytes"]
        ),
        "assignment_meets_compact_storage_lower_bound_gate":
            owner["compact_storage_lower_bound_imbalance"] <= 1.10,
        "assignment_meets_edge_gate": max(
            owner["out_edge_imbalance"],
            owner["in_edge_imbalance"],
        ) <= 1.10,
        "assignment_meets_compact_capacity_ratio_lower_bound_gate": (
            owner["compact_max_storage_lower_bound_bytes"] /
            contiguous["compact_max_storage_lower_bound_bytes"]
        ) <= 1.10,
    }


def summarize_corpus(
    graph_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [graph["summary"] for graph in graph_results]
    combined = [
        min(
            reduction_value(summary["remote_reduction"]),
            reduction_value(summary["ghost_reduction"]),
        )
        for summary in summaries
    ]
    bfs_halo = [
        reduction_value(summary["bfs_halo_reduction"])
        for summary in summaries
    ]
    return {
        "graphs": len(summaries),
        "work_balance_gate_pass_graphs": sum(
            summary["assignment_meets_work_balance_gate"]
            for summary in summaries
        ),
        "stable": all(summary["stable"] for summary in summaries),
        "passes_work_balance_gate_on_all_graphs": all(
            summary["assignment_meets_work_balance_gate"]
            for summary in summaries
        ),
        "compact_storage_lower_bound_gate_pass_graphs": sum(
            summary["assignment_meets_compact_storage_lower_bound_gate"]
            for summary in summaries
        ),
        "edge_gate_pass_graphs": sum(
            summary["assignment_meets_edge_gate"]
            for summary in summaries
        ),
        "compact_capacity_ratio_lower_bound_gate_pass_graphs": sum(
            summary[
                "assignment_meets_compact_capacity_ratio_lower_bound_gate"]
            for summary in summaries
        ),
        "geomean_combined_reduction": json_reduction(
            geometric_mean(combined)),
        "worst_combined_reduction": json_reduction(min(combined)),
        "geomean_bfs_halo_reduction": json_reduction(
            geometric_mean(bfs_halo)),
        "worst_bfs_halo_reduction": json_reduction(min(bfs_halo)),
        "max_compact_storage_lower_bound_ratio": max(
            summary["max_compact_storage_lower_bound_ratio"]
            for summary in summaries),
        "max_balance_imbalance": max(
            summary["owner_by_vertex"]["balance_imbalance"]
            for summary in summaries
        ),
        "max_compact_storage_lower_bound_imbalance": max(
            summary["owner_by_vertex"][
                "compact_storage_lower_bound_imbalance"]
            for summary in summaries
        ),
        "max_edge_imbalance": max(
            max(
                summary["owner_by_vertex"]["out_edge_imbalance"],
                summary["owner_by_vertex"]["in_edge_imbalance"],
            )
            for summary in summaries
        ),
        "complete_per_bank_working_set_evaluated": False,
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=tuple(GRAPH_CASES),
        default=[
            "roadNet-PA",
            "delaunay_n17",
            "cit-HepPh",
            "soc-Slashdot0811",
            "roadNet-CA",
            "delaunay_n20",
            "cit-Patents",
            "soc-pokec",
        ],
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=PROJECT_ROOT / "results/graphs",
    )
    parser.add_argument("--threads", default="1,32")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--partitions", type=int, default=16)
    parser.add_argument(
        "--balance",
        choices=("vertices", "out", "total"),
        default="total",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT /
            "results/partition_cut/ownership-ablation"
        ),
    )
    args = parser.parse_args()
    threads = [int(value) for value in args.threads.split(",")]
    if not ANALYZER.is_file():
        parser.error("build ownership_analysis")
    try:
        validate_matrix_axes(args.graphs, threads, args.repeats)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    graph_results = []
    for graph_name in args.graphs:
        case = GRAPH_CASES[graph_name]
        graph_path = case.path(args.graph_root.resolve())
        records = [
            run_case(
                graph_name=graph_name,
                graph_path=graph_path,
                threads=thread_count,
                repeat=repeat,
                partitions=args.partitions,
                balance=args.balance,
                output_root=output,
            )
            for thread_count in threads
            for repeat in range(args.repeats)
        ]
        graph_results.append({
            "graph": asdict(case),
            "path": str(graph_path),
            "summary": summarize_graph(records),
            "records": records,
        })
        print(
            f"[ownership] {graph_name}: "
            f"remote={format_reduction(graph_results[-1]['summary']['remote_reduction'])} "
            f"ghost={format_reduction(graph_results[-1]['summary']['ghost_reduction'])} "
            f"max_lb={graph_results[-1]['summary']['max_compact_storage_lower_bound_ratio']:.3f}"
        )
    result = {
        "schema": "graphbrew.partition_ownership_corpus.v1",
        "graphs": args.graphs,
        "partitions": args.partitions,
        "balance": args.balance,
        "threads": threads,
        "repeats": args.repeats,
        "graph_results": graph_results,
        "aggregate": summarize_corpus(graph_results),
    }
    (output / "ownership_summary.json").write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
