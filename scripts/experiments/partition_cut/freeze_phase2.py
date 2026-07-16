#!/usr/bin/env python3
"""Freeze portable, auditable Phase 2 partition-cut evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.partition_cut.phase2 import (  # noqa: E402
    POLICIES,
    summarize_graph_records,
    summarize_corpus,
    validate_runtime_traffic,
)


DEFAULT_MATRICES = {
    "smoke": (
        PROJECT_ROOT
        / "results/partition_cut/phase2-smoke-native/"
        "phase2_summary.json"
    ),
    "scale_production": (
        PROJECT_ROOT
        / "results/partition_cut/phase2-scale-native-production/"
        "phase2_summary.json"
    ),
    "scale_research": (
        PROJECT_ROOT
        / "results/partition_cut/phase2-scale-native-determinism/"
        "phase2_summary.json"
    ),
    "web_anchor": (
        PROJECT_ROOT
        / "results/partition_cut/phase2-web-native/"
        "phase2_summary.json"
    ),
    "balance_vertices": (
        PROJECT_ROOT
        / "results/partition_cut/balance-vertices-intra-rcmpp/"
        "phase2_summary.json"
    ),
    "balance_out": (
        PROJECT_ROOT
        / "results/partition_cut/balance-out-intra-rcmpp/"
        "phase2_summary.json"
    ),
    "balance_total": (
        PROJECT_ROOT
        / "results/partition_cut/balance-total-intra-rcmpp/"
        "phase2_summary.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return value


def normalize_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_paths(item) for item in value]
    if isinstance(value, str):
        return _repo_relative(value)
    return value


def compact_runtime_traffic(
    traffic: dict[str, Any],
) -> dict[str, Any]:
    projection = traffic["graphblox_projection"]
    bfs = traffic["bfs"]
    phase_totals = {
        phase: {
            "supersteps": 0,
            "cpu_ghost_sync_bytes": 0,
            "remote_parent_messages": 0,
            "remote_parent_bytes": 0,
            "graphblox_halo_bytes": 0,
        }
        for phase in ("p-bsp-td", "p-bsp-bu")
    }
    shard_totals = {
        int(shard["shard_id"]): {
            "shard_id": int(shard["shard_id"]),
            "cpu_ghost_sync_bytes": 0,
            "remote_parent_messages": 0,
            "remote_parent_bytes": 0,
            "graphblox_halo_bytes": 0,
        }
        for shard in projection["shards"]
    }
    for step in bfs["steps"]:
        phase = phase_totals[str(step["phase"])]
        phase["supersteps"] += 1
        for field in (
            "cpu_ghost_sync_bytes",
            "remote_parent_messages",
            "remote_parent_bytes",
            "graphblox_halo_bytes",
        ):
            phase[field] += int(step[field])
        for shard in step["shards"]:
            aggregate = shard_totals[int(shard["shard_id"])]
            for field in (
                "cpu_ghost_sync_bytes",
                "remote_parent_messages",
                "remote_parent_bytes",
                "graphblox_halo_bytes",
            ):
                aggregate[field] += int(shard[field])
    step_payload = json.dumps(
        bfs["steps"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return {
        "schema": traffic["schema"],
        "ghost_slots": traffic["ghost_slots"],
        "graphblox_projection": projection,
        "bfs": {
            key: bfs[key]
            for key in (
                "supersteps",
                "cpu_ghost_sync_values",
                "cpu_ghost_sync_bytes",
                "remote_parent_messages",
                "remote_parent_bytes",
                "graphblox_halo_values",
                "graphblox_halo_bytes",
            )
        }
        | {
            "phase_totals": phase_totals,
            "shards": [
                shard_totals[index]
                for index in sorted(shard_totals)
            ],
            "steps_sha256": hashlib.sha256(step_payload).hexdigest(),
        },
    }


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _matrix_command(summary: dict[str, Any]) -> list[str]:
    command = [
        ".venv/bin/python",
        "scripts/experiments/partition_cut/phase2.py",
    ]
    graph_names = [
        str(graph["name"]) for graph in summary["graphs"]
    ]
    command.extend(["--graphs", *graph_names])
    policy_names = [
        str(policy["name"]) for policy in summary["policies"]
    ]
    command.extend(["--policies", *policy_names])
    command.extend([
        "--threads",
        ",".join(str(value) for value in summary["threads"]),
        "--repeats",
        str(summary["repeats"]),
        "--partitions",
        str(summary["partitions"]),
        "--balance",
        str(summary["balance"]),
        "--source",
        str(summary["source"]),
    ])
    if summary.get("max_shard_bytes") is not None:
        command.extend([
            "--max-shard-bytes",
            str(summary["max_shard_bytes"]),
        ])
    return command


def _freeze_records(
    summary_path: Path,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    record_fields = (
        "policy",
        "deterministic_required",
        "threads",
        "repeat",
        "wall_seconds",
        "bfs_seconds",
        "reorder_seconds",
        "partition_seconds",
        "diagnostics_seconds",
        "remote_out_fraction",
        "remote_in_fraction",
        "max_remote_out_fraction",
        "max_remote_in_fraction",
        "ghost_count",
        "ghost_bytes",
        "ghost_byte_fraction",
        "total_shard_bytes",
        "max_shard_bytes",
        "vertex_imbalance",
        "balance_imbalance",
        "out_edge_imbalance",
        "in_edge_imbalance",
        "storage_imbalance",
        "mapping_fingerprint",
        "source_topology_fingerprint",
        "shard_fingerprint",
        "ghost_fingerprint",
        "source_id",
        "depth_fingerprint",
        "reachable_vertices",
        "max_depth",
        "runtime_config",
    )
    frozen: list[dict[str, Any]] = []
    matrix_root = summary_path.parent
    for graph in summary["graph_results"]:
        graph_name = str(graph["graph"])
        for record in graph["records"]:
            validate_runtime_traffic(record)
            run_dir = (
                matrix_root
                / graph_name
                / (
                    f"{record['policy']}-"
                    f"t{record['threads']}-"
                    f"r{record['repeat']}"
                )
            )
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            benchmark_path = run_dir / "benchmarks.json"
            if (
                not stdout_path.is_file()
                or not stderr_path.is_file()
                or not benchmark_path.is_file()
            ):
                raise RuntimeError(
                    f"missing raw evidence for {run_dir}"
                )
            item = normalize_paths({
                field: record[field]
                for field in record_fields
            })
            item["runtime_traffic"] = compact_runtime_traffic(
                record["runtime_traffic"])
            item["raw_logs"] = {
                "stdout_sha256": sha256_file(stdout_path),
                "stderr_sha256": sha256_file(stderr_path),
                "benchmarks_sha256": sha256_file(benchmark_path),
            }
            frozen.append(item)
    return frozen


def freeze_matrix(label: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"missing Phase 2 summary for {label}: {path}"
        )
    summary = json.loads(path.read_text())
    if not summary.get("correct"):
        raise RuntimeError(f"{label} is not correctness-valid")
    if not summary.get("determinism_complete"):
        raise RuntimeError(
            f"{label} lacks complete determinism evidence"
        )
    policy_names = [
        str(policy["name"]) for policy in summary["policies"]
    ]
    policies = [POLICIES[name] for name in policy_names]
    validated_graphs: list[dict[str, Any]] = []
    for graph in summary["graph_results"]:
        recomputed = summarize_graph_records(
            graph["records"],
            policies,
            int(summary.get("max_shard_bytes") or 0),
        )
        if recomputed != graph["summaries"]:
            raise RuntimeError(
                f"{label}:{graph['graph']} has stale policy summaries"
            )
        item = dict(graph)
        item["summaries"] = recomputed
        item["summaries_by_policy"] = {
            policy["policy"]: policy for policy in recomputed
        }
        validated_graphs.append(item)
    recomputed_aggregates = summarize_corpus(
        validated_graphs, policy_names)
    if recomputed_aggregates != summary["aggregates"]:
        raise RuntimeError(f"{label} has stale corpus aggregates")
    first_record = summary["graph_results"][0]["records"][0]
    return {
        "label": label,
        "summary_path": path.resolve().relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "summary_sha256": sha256_file(path),
        "command": _matrix_command(summary),
        "bfs_binary": normalize_paths(
            first_record["run_metadata"]["bfs_binary"]
        ),
        "configuration": {
            key: normalize_paths(summary.get(key))
            for key in (
                "schema",
                "preset",
                "graphs",
                "partitions",
                "balance",
                "source",
                "threads",
                "repeats",
                "policies",
                "max_shard_bytes",
                "correct",
                "determinism_complete",
                "determinism_passed",
                "valid",
            )
        },
        "aggregates": normalize_paths(recomputed_aggregates),
        "graphs": [
            {
                "graph": graph["graph"],
                "category": graph["category"],
                "size_tier": graph["size_tier"],
                "graph_identity": normalize_paths(
                    graph["graph_identity"]),
                "summaries": normalize_paths(graph["summaries"]),
            }
            for graph in validated_graphs
        ],
        "records": _freeze_records(path, summary),
    }


def combined_aggregates(
    matrices: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw = {
        label: json.loads(
            (
                PROJECT_ROOT / matrix["summary_path"]
            ).read_text()
        )
        for label, matrix in matrices.items()
    }
    smoke = raw["smoke"]["graph_results"]
    production = raw["scale_production"]["graph_results"]
    research = raw["scale_research"]["graph_results"]
    return {
        "production": summarize_corpus(
            smoke + production,
            ["original", "rcm_bnf", "gorder_csr"],
        ),
        "research": summarize_corpus(
            smoke + research,
            ["original", "comm_cut_min", "intra_rcmpp"],
        ),
    }


def balance_sweep(
    matrices: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for balance in ("vertices", "out", "total"):
        matrix = matrices[f"balance_{balance}"]
        aggregate = next(
            item
            for item in matrix["aggregates"]
            if item["policy"] == "intra_rcmpp"
        )
        result[balance] = aggregate
    return result


def _semantic_payload(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    generated = dict(result.get("_generated", {}))
    generated.pop("generated_at", None)
    generated.pop("source_commit", None)
    generated.pop("source_tree", None)
    result["_generated"] = generated
    return result


def write_evidence(
    output: Path,
    matrices: dict[str, Path],
) -> None:
    frozen = {
        label: freeze_matrix(label, path.resolve())
        for label, path in matrices.items()
    }
    generator_inputs = [
        Path(__file__).resolve(),
        (
            PROJECT_ROOT
            / "scripts/experiments/partition_cut/phase2.py"
        ),
    ]
    payload = {
        "schema": "graphbrew.partition_cut.phase2.evidence.v1",
        "_generated": {
            "generator": (
                "scripts/experiments/partition_cut/"
                "freeze_phase2.py"
            ),
            "source_commit": _git_output("rev-parse", "HEAD"),
            "source_tree": (
                "dirty"
                if _git_output("status", "--porcelain")
                else "clean"
            ),
            "generated_at": datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": [
                {
                    "path": path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in generator_inputs
            ],
            "regenerate": (
                ".venv/bin/python "
                "scripts/experiments/partition_cut/"
                "freeze_phase2.py"
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "matrices": frozen,
        "combined_aggregates": combined_aggregates(frozen),
        "balance_sweep": balance_sweep(frozen),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        existing = json.loads(output.read_text())
        if _semantic_payload(existing) == _semantic_payload(payload):
            return
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "scripts/experiments/partition_cut/evidence/"
            "phase2_native.json"
        ),
    )
    args = parser.parse_args()
    matrices = dict(DEFAULT_MATRICES)
    for raw in args.matrix:
        label, separator, path = raw.partition("=")
        if not separator or not label or not path:
            parser.error("--matrix requires LABEL=PATH")
        matrices[label] = Path(path)
    missing = sorted(set(DEFAULT_MATRICES) - set(matrices))
    if missing:
        parser.error(
            "missing required matrices: " + ", ".join(missing)
        )
    write_evidence(args.output.resolve(), matrices)
    print(f"[partition-cut-p2] froze {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
