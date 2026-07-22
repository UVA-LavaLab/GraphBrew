#!/usr/bin/env python3
"""Validate structural work and parallel-ownership contracts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "bench" / "contracts" / "edge_gas_algorithms.json"
EDGE_STRUCTURE = {
    "bfs": (
        "bench/include/graphbrew/algorithms/bfs_edge.h",
        ["BFSSparsePush", "BFSDensePull", "schedule(dynamic, 1024)"],
        "direction_optimized",
    ),
    "bc": (
        "bench/include/graphbrew/algorithms/bc_edge.h",
        ["BCForwardEdge", "levels", "#pragma omp atomic update"],
        "brandes",
    ),
    "cc": (
        "bench/include/graphbrew/algorithms/connected_components_edge.h",
        ["AfforestEdge", "edge::AtomicLoad(labels[source]) == frequent",
         "const std::size_t sampled"],
        "afforest",
    ),
    "cc_sv": (
        "bench/include/graphbrew/algorithms/connected_components_edge.h",
        ["ShiloachVishkinEdge", "AtomicAssignIfEqualRelaxed",
         "source >= destination"],
        "hook_and_shortcut",
    ),
    "pr": (
        "bench/include/graphbrew/algorithms/pagerank_edge.h",
        ["PartitionSegments", "PageRankAsyncEdge"],
        "iterations",
    ),
    "pr_spmv": (
        "bench/include/graphbrew/algorithms/pagerank_edge.h",
        ["PartitionSegments", "PageRankJacobiEdge"],
        "iterations",
    ),
    "sssp": (
        "bench/include/graphbrew/algorithms/sssp_edge.h",
        ["thread_bins", "kSSSPBinFusionThreshold", "RelaxWeightedEdges"],
        "delta_stepping",
    ),
    "tc": (
        "bench/include/graphbrew/algorithms/tc_edge.h",
        ["CountTriangleIntersection", "schedule(dynamic, 64)",
         "source >= destination"],
        "intersections",
    ),
}
GAS_STRUCTURE = {
    "cc": (
        "bench/include/graphbrew/algorithms/cc_gas.h",
        "GatherSchedule::kActive",
        "active_weak_neighbor",
    ),
    "pr": (
        "bench/include/graphbrew/algorithms/pagerank_gas.h",
        "GatherSchedule::kDense",
        "dense_gather_apply_scatter",
    ),
    "sssp": (
        "bench/include/graphbrew/algorithms/sssp_gas.h",
        "GatherSchedule::kActive",
        "active_bellman_ford",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_patterns(path: str, patterns: list[str]) -> None:
    text = (ROOT / path).read_text()
    for pattern in patterns:
        require(pattern in text, f"{path}: missing structural marker {pattern}")


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    performance = contract["performance_contract"]
    require(
        set(EDGE_STRUCTURE) == set(performance),
        "edge structure/performance contract sets differ",
    )
    for name, (path, patterns, work_token) in EDGE_STRUCTURE.items():
        require_patterns(path, patterns)
        require(
            work_token in performance[name]["edge_work"].lower(),
            f"{name}: edge work contract is not bound to structure check",
        )

    require_patterns(
        "bench/include/graphbrew/gas/executor.h",
        ["GatherSchedule::kDense", "active.sparse()", "ApplyVertex"],
    )
    for name, (path, schedule, work_token) in GAS_STRUCTURE.items():
        require_patterns(path, [schedule])
        require(
            work_token in performance[name]["gas_work"].lower(),
            f"{name}: GAS work contract is not bound to schedule check",
        )

    audited_sources = list(
        (ROOT / "bench" / "include" / "graphbrew" / "algorithms").glob("*.h")
    )
    audited_sources += list(
        (ROOT / "bench" / "include" / "graphbrew" / "edge").glob("*.h")
    )
    audited_sources += list(
        (ROOT / "bench" / "include" / "graphbrew" / "gas").glob("*.h")
    )
    for algorithm in contract["algorithms"]:
        for source_field in ("edge_source", "gas_source"):
            source = algorithm.get(source_field)
            if source is not None:
                audited_sources.append(ROOT / source)
    allowed_critical = {
        (
            "bench/include/graphbrew/edge/frontier.h",
            "#pragma omp critical(graphbrew_frontier_builder_overflow)",
        ),
    }
    for source in audited_sources:
        relative = str(source.relative_to(ROOT))
        for line in source.read_text().splitlines():
            if "#pragma omp critical" not in line:
                continue
            require(
                (relative, line.strip()) in allowed_critical,
                f"{relative}: unapproved critical section",
            )

    for algorithm in contract["algorithms"]:
        for source_field in ("edge_source", "gas_source"):
            source = algorithm.get(source_field)
            if source is None:
                continue
            text = (ROOT / source).read_text()
            benchmark = text.find("BenchmarkKernel")
            require(
                benchmark >= 0,
                f"{source}: BenchmarkKernel missing",
            )
            conversion = max(
                text.rfind("Flatten", 0, benchmark),
                text.rfind("flattenGraphOut", 0, benchmark),
                text.rfind("RelabelByDegree", 0, benchmark),
            )
            require(
                conversion >= 0,
                f"{source}: graph-view construction not before timing",
            )

    print(
        "edge-gas-structure-check: PASS "
        "(8 edge work contracts; 3 GAS work contracts; "
        "only guarded frontier-overflow critical section)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
