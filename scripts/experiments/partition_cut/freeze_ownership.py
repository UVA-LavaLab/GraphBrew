#!/usr/bin/env python3
"""Freeze portable ownership-ablation evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.partition_cut.freeze_phase2 import (  # noqa: E402
    _semantic_payload,
    freeze_preparation,
    normalize_paths,
    sha256_file,
)
from scripts.experiments.partition_cut.ownership_ablation import (  # noqa: E402
    ANALYZER,
    MARKER,
    summarize_corpus,
    summarize_graph,
    validate_matrix_axes,
    validate_report,
)
from scripts.experiments.partition_cut.phase2 import (  # noqa: E402
    GRAPH_CASES,
    graph_identity,
)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def type_strict_equal(lhs: Any, rhs: Any) -> bool:
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, dict):
        return (
            lhs.keys() == rhs.keys()
            and all(
                type_strict_equal(lhs[key], rhs[key])
                for key in lhs
            )
        )
    if isinstance(lhs, list):
        return (
            len(lhs) == len(rhs)
            and all(
                type_strict_equal(left, right)
                for left, right in zip(lhs, rhs)
            )
        )
    return lhs == rhs


def freeze_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    def exact_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"ownership {name} must be an integer")
        return value

    def exact_bool(value: Any, name: str) -> bool:
        if type(value) is not bool:
            raise RuntimeError(
                f"ownership {name} must be a boolean")
        return value

    aggregate = summary.get("aggregate")
    if (
        summary.get("schema") !=
            "graphbrew.partition_ownership_corpus.v1"
        or not isinstance(aggregate, dict)
        or exact_bool(aggregate.get("stable"), "aggregate stable")
            is not True
        or exact_bool(
            aggregate.get("promotion_eligible"),
            "aggregate promotion_eligible",
        ) is not False
        or exact_bool(
            aggregate.get("complete_per_bank_working_set_evaluated"),
            "aggregate complete_per_bank_working_set_evaluated",
        ) is not False
    ):
        raise RuntimeError("ownership summary contract is invalid")

    configured_graphs = summary.get("graphs")
    actual_graphs = [
        graph["graph"]["name"]
        for graph in summary.get("graph_results", [])
    ]
    if (
        not isinstance(configured_graphs, list)
        or len(configured_graphs) != len(set(configured_graphs))
        or actual_graphs != configured_graphs
        or len(actual_graphs) != len(set(actual_graphs))
    ):
        raise RuntimeError("ownership graph matrix is incomplete")
    try:
        validate_matrix_axes(
            configured_graphs,
            [
                exact_int(value, "thread count")
                for value in summary.get("threads", [])
            ],
            exact_int(summary.get("repeats"), "repeat count"),
        )
    except ValueError as error:
        raise RuntimeError(
            "ownership repeat/thread matrix is incomplete") from error
    frozen_graphs = []
    recomputed_graph_results = []
    root = path.parent
    expected_coordinates = {
        (exact_int(threads, "thread count"), repeat)
        for threads in summary["threads"]
        for repeat in range(
            exact_int(summary["repeats"], "repeat count"))
    }
    analyzer_identity = normalize_paths(graph_identity(ANALYZER))
    for graph in summary["graph_results"]:
        graph_name = graph["graph"]["name"]
        graph_path = Path(str(graph["path"]))
        if (
            graph_name not in GRAPH_CASES
            or graph["graph"] != asdict(GRAPH_CASES[graph_name])
            or graph_path != GRAPH_CASES[graph_name].path(
                PROJECT_ROOT / "results/graphs")
        ):
            raise RuntimeError(
                f"ownership graph configuration mismatch: {graph_name}")
        expected_graph_identity = normalize_paths(
            graph_identity(graph_path))
        expected_command = [
            str(ANALYZER),
            "-f",
            str(graph_path),
            "-P",
            str(exact_int(
                summary["partitions"], "partition count")),
            "-B",
            str(summary["balance"]),
        ]
        records = []
        fingerprints = set()
        coordinates = set()
        for record in graph["records"]:
            coordinate = (
                exact_int(record["threads"], "record thread count"),
                exact_int(record["repeat"], "record repeat"),
            )
            if coordinate in coordinates:
                raise RuntimeError(
                    f"duplicate ownership record: {graph_name}")
            coordinates.add(coordinate)
            report = record["report"]
            fingerprints.add((
                report["membership_fingerprint"],
                report["mapping_fingerprint"],
                report["contiguous"]["owner_fingerprint"],
                report["owner_by_vertex"]["owner_fingerprint"],
            ))
            run_dir = root / graph_name / (
                f"t{record['threads']}-r{record['repeat']}")
            stdout = run_dir / "stdout.log"
            stderr = run_dir / "stderr.log"
            if not stdout.is_file() or not stderr.is_file():
                raise RuntimeError(
                    f"missing ownership raw logs: {run_dir}")
            payloads = [
                line[len(MARKER):]
                for line in stdout.read_text().splitlines()
                if line.startswith(MARKER)
            ]
            if len(payloads) != 1:
                raise RuntimeError(
                    f"invalid ownership raw record: {run_dir}")
            raw_report = json.loads(payloads[0])
            if not type_strict_equal(raw_report, report):
                raise RuntimeError(
                    f"ownership raw/summary mismatch: {run_dir}")
            validate_report(
                report,
                exact_int(summary["partitions"], "partition count"),
                str(summary["balance"]),
            )
            if normalize_paths(record["analyzer"]) != analyzer_identity:
                raise RuntimeError(
                    f"ownership analyzer identity mismatch: {run_dir}")
            if record["command"] != expected_command:
                raise RuntimeError(
                    f"ownership command mismatch: {run_dir}")
            if normalize_paths(record["graph"]) != expected_graph_identity:
                raise RuntimeError(
                    f"ownership graph identity mismatch: {run_dir}")
            records.append({
                "threads": record["threads"],
                "repeat": record["repeat"],
                "wall_seconds": record["wall_seconds"],
                "command": normalize_paths(record["command"]),
                "graph": normalize_paths(record["graph"]),
                "analyzer": normalize_paths(record["analyzer"]),
                "report": report,
                "raw_logs": {
                    "stdout_sha256": sha256_file(stdout),
                    "stderr_sha256": sha256_file(stderr),
                },
            })
        if len(fingerprints) != 1:
            raise RuntimeError(
                f"ownership fingerprints are unstable: {graph_name}")
        if coordinates != expected_coordinates:
            raise RuntimeError(
                f"ownership record matrix is incomplete: {graph_name}")
        recomputed_summary = summarize_graph(graph["records"])
        if not type_strict_equal(
                recomputed_summary, graph["summary"]):
            raise RuntimeError(
                f"stale ownership graph summary: {graph_name}")
        recomputed_graph = {
            "graph": graph["graph"],
            "path": graph["path"],
            "summary": recomputed_summary,
            "records": graph["records"],
        }
        recomputed_graph_results.append(recomputed_graph)
        preparation_graph = {
            "graph": graph_name,
            "category": graph["graph"]["category"],
            "size_tier": graph["graph"]["size_tier"],
            "path": graph["path"],
            "graph_identity": expected_graph_identity,
        }
        frozen_graphs.append({
            "graph": graph["graph"],
            "path": normalize_paths(graph["path"]),
            "preparation": freeze_preparation(preparation_graph),
            "summary": recomputed_summary,
            "records": records,
        })
    recomputed_aggregate = summarize_corpus(
        recomputed_graph_results)
    if not type_strict_equal(
            recomputed_aggregate, summary["aggregate"]):
        raise RuntimeError("stale ownership corpus aggregate")
    normalized_summary = normalize_paths(summary)
    canonical_summary = json.dumps(
        normalized_summary,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return {
        "configuration": {
            key: summary[key]
            for key in (
                "graphs",
                "partitions",
                "balance",
                "threads",
                "repeats",
            )
        },
        "aggregate": recomputed_aggregate,
        "graphs": frozen_graphs,
        "summary_path": path.resolve().relative_to(
            PROJECT_ROOT).as_posix(),
        "canonical_summary_sha256":
            hashlib.sha256(canonical_summary).hexdigest(),
    }


def write_evidence(input_path: Path, output_path: Path) -> None:
    frozen = freeze_summary(input_path)
    generator_inputs = [
        Path(__file__).resolve(),
        PROJECT_ROOT /
            "scripts/experiments/partition_cut/ownership_ablation.py",
        PROJECT_ROOT /
            "scripts/experiments/partition_cut/freeze_phase2.py",
        PROJECT_ROOT /
            "scripts/experiments/partition_cut/phase2.py",
    ]
    payload = {
        "schema":
            "graphbrew.partition_ownership_analysis.evidence.v1",
        "_generated": {
            "generator":
                "scripts/experiments/partition_cut/freeze_ownership.py",
            "source_commit": git_output("rev-parse", "HEAD"),
            "source_tree":
                "dirty" if git_output("status", "--porcelain") else "clean",
            "generated_at": datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": [
                {
                    "path": source.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(source),
                }
                for source in generator_inputs
            ],
            "regenerate": (
                ".venv/bin/python "
                "scripts/experiments/partition_cut/freeze_ownership.py"
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "analysis": frozen,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        existing = json.loads(output_path.read_text())
        if _semantic_payload(existing) == _semantic_payload(payload):
            return
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> int:
    input_path = (
        PROJECT_ROOT /
        "results/partition_cut/ownership-ablation/"
        "ownership_summary.json"
    )
    output_path = (
        PROJECT_ROOT /
        "scripts/experiments/partition_cut/evidence/"
        "ownership_ablation.json"
    )
    write_evidence(input_path, output_path)
    print(f"[ownership] froze {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
