#!/usr/bin/env python3
"""Widen compact-shard cut policies across graph classes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.partition_cut.phase1 import (  # noqa: E402
    BFS_BINARY,
    Policy,
    _parse_ints,
    _print_summary,
    _run_case,
    add_relative_metrics,
    summarize_policy,
    validate_cross_policy,
)


@dataclass(frozen=True)
class GraphCase:
    name: str
    category: str
    size_tier: str
    symmetric: bool

    def path(self, graph_root: Path) -> Path:
        return graph_root / self.name / f"{self.name}.sg"


GRAPH_CASES = {
    "roadNet-PA": GraphCase(
        "roadNet-PA", "road", "smoke", True
    ),
    "delaunay_n17": GraphCase(
        "delaunay_n17", "mesh", "smoke", True
    ),
    "cit-HepPh": GraphCase(
        "cit-HepPh", "citation", "smoke", False
    ),
    "soc-Slashdot0811": GraphCase(
        "soc-Slashdot0811", "social", "smoke", False
    ),
    "web-Google": GraphCase(
        "web-Google", "web", "anchor", False
    ),
    "roadNet-CA": GraphCase(
        "roadNet-CA", "road", "scale", True
    ),
    "delaunay_n20": GraphCase(
        "delaunay_n20", "mesh", "scale", True
    ),
    "cit-Patents": GraphCase(
        "cit-Patents", "citation", "scale", False
    ),
    "soc-pokec": GraphCase(
        "soc-pokec", "social", "scale", False
    ),
}

PRESETS = {
    "smoke": (
        "roadNet-PA",
        "delaunay_n17",
        "cit-HepPh",
        "soc-Slashdot0811",
    ),
    "scale": (
        "roadNet-CA",
        "delaunay_n20",
        "cit-Patents",
        "soc-pokec",
    ),
    "anchor": ("web-Google",),
}

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
        True,
    ),
    "sg_hilbert": Policy(
        "sg_hilbert",
        (
            "-o",
            "12:leiden:compose:sg_hilbert:comm_identity:intra_hubsort",
        ),
        True,
    ),
    "intra_hub2": Policy(
        "intra_hub2",
        (
            "-o",
            "12:leiden:compose:comm_degree_desc:intra_hub2",
        ),
        True,
    ),
    "intra_rcmpp": Policy(
        "intra_rcmpp",
        (
            "-o",
            "12:leiden:compose:comm_degree_desc:intra_rcmpp",
        ),
        True,
    ),
    "leiden_hubsort": Policy(
        "leiden_hubsort",
        (
            "-o",
            "12:leiden:compose:comm_degree_desc:intra_hubsort",
        ),
        True,
    ),
}

COMMUNITY_RUNTIME_EXPECTATIONS = {
    "comm_cut_min": {
        "super_graph": "none",
        "community": {
            "cut_min",
            "cut_min_fallback_degree_desc",
        },
        "intra": "hubsort",
    },
    "sg_hilbert": {
        "super_graph": "hilbert",
        "community": {"identity"},
        "intra": "hubsort",
    },
    "intra_hub2": {
        "super_graph": "none",
        "community": {"degree_desc"},
        "intra": "hub2",
    },
    "intra_rcmpp": {
        "super_graph": "none",
        "community": {"degree_desc"},
        "intra": "rcmpp",
    },
    "leiden_hubsort": {
        "super_graph": "none",
        "community": {"degree_desc"},
        "intra": "hubsort",
    },
}
BASE_RUNTIME_EXPECTATIONS = {
    "original": {
        "algorithm": None,
    },
    "rcm_bnf": {
        "algorithm": "RCMOrder",
        "rcm_variant": "bnf",
    },
    "gorder_csr": {
        "algorithm": "GOrder",
        "gorder_variant": "csr",
    },
}

ALGORITHM_RE = re.compile(r"^Algorithm:\s+(\S+)", re.MULTILINE)
GRAPHBREW_MODE_RE = re.compile(
    r"^GraphBrew: aggregation=(\S+), ordering=(\S+),",
    re.MULTILINE,
)
GRAPHBREW_M_RE = re.compile(
    r"^GraphBrew: mComputation=(\S+)",
    re.MULTILINE,
)
GRAPHBREW_COMMUNITY_THREADS_RE = re.compile(
    r"^GraphBrew: community-detection=(serial|parallel)$",
    re.MULTILINE,
)
GRAPHBREW_ORDERING_THREADS_RE = re.compile(
    r"^GraphBrew: ordering-threads=(\d+)$",
    re.MULTILINE,
)
GRAPHBREW_COMMUNITIES_RE = re.compile(
    r"^GraphBrew:\s+\d+ passes,\s+\d+ iters,\s+"
    r"(\d+) communities, time=",
    re.MULTILINE,
)
COMPOSE_RE = re.compile(
    r"^\s*compose: sg=(\S+) comm=(\S+) intra=(\S+) "
    r"refine=(\S+),",
    re.MULTILINE,
)


def parse_runtime_config(stdout: str) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    algorithm = ALGORITHM_RE.search(stdout)
    if algorithm:
        runtime["algorithm"] = algorithm.group(1)
    if "RCM_BNF" in stdout:
        runtime["rcm_variant"] = "bnf"
    if "GOrder_CSR" in stdout:
        runtime["gorder_variant"] = "csr"
    mode = GRAPHBREW_MODE_RE.search(stdout)
    if mode:
        runtime["aggregation"] = mode.group(1)
        runtime["ordering"] = mode.group(2)
    m_computation = GRAPHBREW_M_RE.search(stdout)
    if m_computation:
        runtime["m_computation"] = m_computation.group(1)
    community_threads = GRAPHBREW_COMMUNITY_THREADS_RE.search(stdout)
    if community_threads:
        runtime["community_detection"] = community_threads.group(1)
    ordering_threads = GRAPHBREW_ORDERING_THREADS_RE.search(stdout)
    if ordering_threads:
        runtime["ordering_threads"] = int(ordering_threads.group(1))
    communities = GRAPHBREW_COMMUNITIES_RE.search(stdout)
    if communities:
        runtime["communities"] = int(communities.group(1))
    compose = COMPOSE_RE.search(stdout)
    if compose:
        runtime.update({
            "super_graph": compose.group(1),
            "community": compose.group(2),
            "intra": compose.group(3),
            "refine": compose.group(4),
        })
        runtime["cut_min_fallback"] = (
            runtime["community"] ==
            "cut_min_fallback_degree_desc"
        )
    return runtime


def validate_runtime_config(
    policy: Policy,
    runtime: dict[str, Any],
    expected_threads: int | None = None,
) -> None:
    expected = COMMUNITY_RUNTIME_EXPECTATIONS.get(policy.name)
    if expected is None:
        if policy.name not in BASE_RUNTIME_EXPECTATIONS:
            return
        expected_runtime = BASE_RUNTIME_EXPECTATIONS[policy.name]
        mismatches = {
            key: (runtime.get(key), value)
            for key, value in expected_runtime.items()
            if runtime.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"{policy.name} runtime configuration mismatch: "
                f"{mismatches}"
            )
        return
    required = {
        "algorithm": "GraphBrewOrder",
        "aggregation": "gve-csr",
        "ordering": "compose",
        "m_computation": "total-edges",
        "community_detection": "serial",
        "super_graph": expected["super_graph"],
        "intra": expected["intra"],
        "refine": "none",
    }
    mismatches = {
        key: (runtime.get(key), value)
        for key, value in required.items()
        if runtime.get(key) != value
    }
    if runtime.get("community") not in expected["community"]:
        mismatches["community"] = (
            runtime.get("community"),
            sorted(expected["community"]),
        )
    if (
        expected_threads is not None
        and runtime.get("ordering_threads") != expected_threads
    ):
        mismatches["ordering_threads"] = (
            runtime.get("ordering_threads"),
            expected_threads,
        )
    if mismatches:
        raise RuntimeError(
            f"{policy.name} runtime configuration mismatch: "
            f"{mismatches}"
        )


def selected_graphs(
    preset: str,
    names: list[str] | None,
) -> list[GraphCase]:
    selected_names = names or list(PRESETS[preset])
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("Phase 2 graph selection contains duplicates")
    return [GRAPH_CASES[name] for name in selected_names]


def _convertible_input(graph_dir: Path, name: str) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("**/*.mtx", "**/*.el"):
        matches = sorted(graph_dir.glob(pattern))
        exact = [path for path in matches if path.stem == name]
        candidates.extend(exact or matches)
        if candidates:
            break
    return candidates[0] if candidates else None


def graph_needs_download(
    graph: GraphCase,
    graph_root: Path,
) -> bool:
    output = graph.path(graph_root)
    return (
        not output.is_file()
        or _convertible_input(output.parent, graph.name) is None
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def converter_command(
    converter: Path,
    graph: GraphCase,
    source: Path,
    output: Path,
) -> list[str]:
    command = [
        str(converter),
        "-f",
        str(source),
    ]
    if graph.symmetric:
        command.append("-s")
    command.extend(["-b", str(output)])
    return command


def expected_run_metadata(
    *,
    graph: GraphCase,
    graph_path: Path,
    policy: Policy,
    threads: int,
    repeat: int,
    partitions: int,
    balance: str,
    source: int,
) -> dict[str, Any]:
    return {
        "graph": asdict(graph),
        "graph_file": graph_identity(graph_path),
        "bfs_binary": graph_identity(BFS_BINARY),
        "policy": {
            "name": policy.name,
            "options": list(policy.options),
            "deterministic_required":
                policy.deterministic_required,
        },
        "threads": threads,
        "repeat": repeat,
        "partitions": partitions,
        "balance": balance,
        "source": source,
    }


def validate_record_matrix(
    records: list[dict[str, Any]],
    expected_metadata: dict[
        tuple[str, int, int],
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    records_by_key: dict[
        tuple[str, int, int],
        dict[str, Any],
    ] = {}
    for record in records:
        key = (
            str(record.get("policy", "")),
            int(record.get("threads", 0)),
            int(record.get("repeat", -1)),
        )
        if key in records_by_key:
            raise RuntimeError(
                f"duplicate Phase 2 record {key}"
            )
        records_by_key[key] = record
    expected_keys = set(expected_metadata)
    actual_keys = set(records_by_key)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "Phase 2 record matrix mismatch: "
            f"missing={sorted(expected_keys - actual_keys)} "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    for key, metadata in expected_metadata.items():
        if records_by_key[key].get("run_metadata") != metadata:
            raise RuntimeError(
                f"Phase 2 record metadata mismatch for {key}"
            )
    return [
        records_by_key[key] for key in sorted(expected_keys)
    ]


def prepare_graphs(
    graphs: list[GraphCase],
    graph_root: Path,
    *,
    workers: int,
) -> None:
    from scripts.lib.pipeline.download import download_graphs

    graph_root.mkdir(parents=True, exist_ok=True)
    missing = [
        graph.name
        for graph in graphs
        if graph_needs_download(graph, graph_root)
    ]
    if missing:
        downloaded = download_graphs(
            graphs=missing,
            dest_dir=graph_root,
            max_workers=min(workers, len(missing)),
        )
        if len(downloaded) != len(missing):
            available = {
                path.parent.name for path in downloaded
            }
            failed = sorted(set(missing) - available)
            raise RuntimeError(
                "failed to download Phase 2 graphs: "
                + ", ".join(failed)
            )

    converter = PROJECT_ROOT / "bench" / "bin" / "converter"
    if not converter.is_file():
        raise RuntimeError(
            f"missing {converter}; run `make converter`"
        )
    for graph in graphs:
        output = graph.path(graph_root)
        graph_dir = output.parent
        source = _convertible_input(graph_dir, graph.name)
        if source is None:
            raise RuntimeError(
                f"no .mtx/.el input found for {graph.name}"
            )
        logical_args = ["-f", source.name]
        if graph.symmetric:
            logical_args.append("-s")
        logical_args.extend(["-b", output.name])
        desired_manifest = {
            "schema": "graphbrew.partition_cut.prepare.v1",
            "graph": asdict(graph),
            "source": {
                "relative_path": source.relative_to(
                    graph_dir
                ).as_posix(),
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            },
            "converter": {
                "path": str(converter.resolve()),
                "size": converter.stat().st_size,
                "sha256": sha256_file(converter),
            },
            "arguments": logical_args,
        }
        manifest_path = graph_dir / "phase2_prepare.json"
        if (
            output.is_file()
            and output.stat().st_size > 0
            and manifest_path.is_file()
        ):
            existing = json.loads(manifest_path.read_text())
            if (
                existing.get("inputs") == desired_manifest
                and existing.get("output") ==
                graph_identity(output)
            ):
                continue
        command = converter_command(
            converter, graph, source, output)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,
        )
        (graph_dir / "phase2_prepare.stdout.log").write_text(
            completed.stdout
        )
        (graph_dir / "phase2_prepare.stderr.log").write_text(
            completed.stderr
        )
        if (
            completed.returncode != 0
            or not output.is_file()
            or output.stat().st_size == 0
        ):
            raise RuntimeError(
                f"failed to convert {graph.name}; see {graph_dir}"
            )
        manifest_path.write_text(
            json.dumps(
                {
                    "inputs": desired_manifest,
                    "output": graph_identity(output),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def geometric_mean(values: list[float]) -> float:
    if not values or any(value < 0 for value in values):
        raise ValueError(
            "geometric mean requires nonnegative values"
        )
    if any(value == 0 for value in values):
        return 0.0
    if any(math.isinf(value) for value in values):
        return float("inf")
    return math.exp(
        sum(math.log(value) for value in values) / len(values)
    )


def classify_determinism(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError(
            "cannot classify determinism without records"
        )
    fingerprint_fields = (
        "mapping_fingerprint",
        "source_topology_fingerprint",
        "shard_fingerprint",
        "ghost_fingerprint",
        "depth_fingerprint",
    )

    def fingerprint(record: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(record[field] for field in fingerprint_fields)

    by_threads: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_threads.setdefault(int(record["threads"]), []).append(
            record
        )
    repeat_stable_by_threads = {
        str(threads): all(
            fingerprint(record) ==
            fingerprint(selected[0])
            for record in selected[1:]
        )
        for threads, selected in sorted(by_threads.items())
    }
    repeat_stable = all(repeat_stable_by_threads.values())
    representatives = [
        selected[0]
        for _, selected in sorted(by_threads.items())
    ]
    cross_thread_stable = all(
        fingerprint(record) ==
        fingerprint(representatives[0])
        for record in representatives[1:]
    )
    evidence_complete = (
        len(by_threads) >= 2
        and all(len(selected) >= 2 for selected in by_threads.values())
    )
    if not repeat_stable:
        classification = "repeat_variant"
    elif not evidence_complete:
        classification = "insufficient_evidence"
    elif cross_thread_stable:
        classification = "deterministic"
    else:
        classification = "thread_variant"
    return {
        "classification": classification,
        "repeat_stable": repeat_stable,
        "cross_thread_stable": cross_thread_stable,
        "evidence_complete": evidence_complete,
        "repeat_stable_by_threads": repeat_stable_by_threads,
    }


def _range(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


RUNTIME_TRAFFIC_SCHEMA = "graphbrew.partition_runtime_traffic.v1"
UINT64_MAX = (1 << 64) - 1


def _traffic_dict(
    value: Any,
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Phase 2 runtime traffic {path} is not an object"
        )
    return value


def _traffic_list(
    value: Any,
    path: str,
) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(
            f"Phase 2 runtime traffic {path} is not an array"
        )
    return value


def _traffic_uint(
    value: Any,
    path: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > UINT64_MAX
    ):
        raise RuntimeError(
            f"Phase 2 runtime traffic {path} "
            "must be a nonnegative integer"
        )
    return value


def _traffic_product(
    lhs: int,
    rhs: int,
    path: str,
) -> int:
    result = lhs * rhs
    if result > UINT64_MAX:
        raise RuntimeError(
            f"Phase 2 runtime traffic {path} overflows uint64"
        )
    return result


def _traffic_add(
    lhs: int,
    rhs: int,
    path: str,
) -> int:
    result = lhs + rhs
    if result > UINT64_MAX:
        raise RuntimeError(
            f"Phase 2 runtime traffic {path} overflows uint64"
        )
    return result


def _require_traffic_equal(
    actual: int,
    expected: int,
    path: str,
) -> None:
    if actual != expected:
        raise RuntimeError(
            f"Phase 2 runtime traffic {path} mismatch: "
            f"{actual} != {expected}"
        )


def validate_runtime_traffic(
    record: dict[str, Any],
) -> dict[str, Any]:
    traffic = record.get("runtime_traffic")
    if not isinstance(traffic, dict):
        raise RuntimeError(
            "Phase 2 record lacks runtime traffic"
        )
    if traffic.get("schema") != RUNTIME_TRAFFIC_SCHEMA:
        raise RuntimeError(
            "Phase 2 record has unknown runtime traffic schema"
        )
    ghost_slots = _traffic_uint(
        traffic.get("ghost_slots"), "ghost_slots")
    if "ghost_count" in record:
        _require_traffic_equal(
            ghost_slots,
            _traffic_uint(record["ghost_count"], "record.ghost_count"),
            "ghost_slots",
        )

    projection = _traffic_dict(
        traffic.get("graphblox_projection"),
        "graphblox_projection",
    )
    projection_fields = {
        "bfs_bytes_per_superstep": _traffic_product(
            ghost_slots, 8, "graphblox_projection"),
        "pr_bytes_per_iteration": _traffic_product(
            ghost_slots, 4, "graphblox_projection"),
        "cc_bytes_per_iteration": _traffic_product(
            ghost_slots, 4, "graphblox_projection"),
        "spmv_initial_bytes": _traffic_product(
            ghost_slots, 4, "graphblox_projection"),
    }
    for field, expected in projection_fields.items():
        _require_traffic_equal(
            _traffic_uint(
                projection.get(field),
                f"graphblox_projection.{field}",
            ),
            expected,
            f"graphblox_projection.{field}",
        )

    projection_shards = _traffic_list(
        projection.get("shards"),
        "graphblox_projection.shards",
    )
    if not projection_shards:
        raise RuntimeError(
            "Phase 2 runtime traffic has no shard projections"
        )
    projection_by_shard: dict[int, dict[str, int]] = {}
    projection_sums = {
        "ghost_slots": 0,
        **{field: 0 for field in projection_fields},
    }
    for raw_shard in projection_shards:
        shard = _traffic_dict(
            raw_shard, "graphblox_projection.shards[]")
        shard_id = _traffic_uint(
            shard.get("shard_id"),
            "graphblox_projection.shards[].shard_id",
        )
        if shard_id in projection_by_shard:
            raise RuntimeError(
                "Phase 2 runtime traffic has duplicate shard projections"
            )
        shard_ghosts = _traffic_uint(
            shard.get("ghost_slots"),
            f"graphblox_projection.shards[{shard_id}].ghost_slots",
        )
        expected_shard = {
            "ghost_slots": shard_ghosts,
            "bfs_bytes_per_superstep": _traffic_product(
                shard_ghosts, 8, "shard projection"),
            "pr_bytes_per_iteration": _traffic_product(
                shard_ghosts, 4, "shard projection"),
            "cc_bytes_per_iteration": _traffic_product(
                shard_ghosts, 4, "shard projection"),
            "spmv_initial_bytes": _traffic_product(
                shard_ghosts, 4, "shard projection"),
        }
        for field, expected in expected_shard.items():
            actual = _traffic_uint(
                shard.get(field),
                (
                    "graphblox_projection."
                    f"shards[{shard_id}].{field}"
                ),
            )
            _require_traffic_equal(
                actual,
                expected,
                (
                    "graphblox_projection."
                    f"shards[{shard_id}].{field}"
                ),
            )
            projection_sums[field] = _traffic_add(
                projection_sums[field],
                actual,
                f"graphblox_projection.shards.{field}",
            )
        projection_by_shard[shard_id] = expected_shard
    if sorted(projection_by_shard) != \
            list(range(len(projection_by_shard))):
        raise RuntimeError(
            "Phase 2 runtime traffic shard IDs are not contiguous"
        )
    _require_traffic_equal(
        projection_sums["ghost_slots"],
        ghost_slots,
        "graphblox_projection.shards.ghost_slots",
    )
    for field, expected in projection_fields.items():
        _require_traffic_equal(
            projection_sums[field],
            expected,
            f"graphblox_projection.shards.{field}",
        )

    bfs = _traffic_dict(traffic.get("bfs"), "bfs")
    steps = _traffic_list(bfs.get("steps"), "bfs.steps")
    supersteps = _traffic_uint(
        bfs.get("supersteps"), "bfs.supersteps")
    _require_traffic_equal(
        supersteps, len(steps), "bfs.supersteps")
    if "max_depth" in record:
        _require_traffic_equal(
            supersteps,
            _traffic_add(
                _traffic_uint(
                    record["max_depth"], "record.max_depth"),
                1,
                "record.max_depth",
            ),
            "bfs.supersteps",
        )
    aggregate = {
        "cpu_ghost_sync_bytes": 0,
        "remote_parent_messages": 0,
        "remote_parent_bytes": 0,
        "graphblox_halo_bytes": 0,
    }
    for expected_step, raw_step in enumerate(steps):
        step = _traffic_dict(raw_step, f"bfs.steps[{expected_step}]")
        _require_traffic_equal(
            _traffic_uint(
                step.get("step"),
                f"bfs.steps[{expected_step}].step",
            ),
            expected_step,
            f"bfs.steps[{expected_step}].step",
        )
        phase = step.get("phase")
        if phase not in {"p-bsp-td", "p-bsp-bu"}:
            raise RuntimeError(
                f"Phase 2 runtime traffic has invalid BFS phase {phase!r}"
            )
        step_shards = _traffic_list(
            step.get("shards"),
            f"bfs.steps[{expected_step}].shards",
        )
        if len(step_shards) != len(projection_by_shard):
            raise RuntimeError(
                "Phase 2 runtime traffic BFS shard count mismatch"
            )
        step_totals = {
            "cpu_ghost_sync_bytes": 0,
            "remote_parent_messages": 0,
            "remote_parent_bytes": 0,
            "graphblox_halo_bytes": 0,
        }
        seen_shards: set[int] = set()
        for raw_shard in step_shards:
            shard = _traffic_dict(
                raw_shard,
                f"bfs.steps[{expected_step}].shards[]",
            )
            shard_id = _traffic_uint(
                shard.get("shard_id"),
                f"bfs.steps[{expected_step}].shards[].shard_id",
            )
            if (
                shard_id not in projection_by_shard
                or shard_id in seen_shards
            ):
                raise RuntimeError(
                    "Phase 2 runtime traffic has invalid BFS shard IDs"
                )
            seen_shards.add(shard_id)
            halo = projection_by_shard[shard_id]
            expected_sync = (
                halo["ghost_slots"]
                if phase == "p-bsp-bu"
                else 0
            )
            shard_values = {
                "cpu_ghost_sync_bytes": _traffic_uint(
                    shard.get("cpu_ghost_sync_bytes"),
                    (
                        f"bfs.steps[{expected_step}]."
                        f"shards[{shard_id}].cpu_ghost_sync_bytes"
                    ),
                ),
                "remote_parent_messages": _traffic_uint(
                    shard.get("remote_parent_messages"),
                    (
                        f"bfs.steps[{expected_step}]."
                        f"shards[{shard_id}].remote_parent_messages"
                    ),
                ),
                "remote_parent_bytes": _traffic_uint(
                    shard.get("remote_parent_bytes"),
                    (
                        f"bfs.steps[{expected_step}]."
                        f"shards[{shard_id}].remote_parent_bytes"
                    ),
                ),
                "graphblox_halo_bytes": _traffic_uint(
                    shard.get("graphblox_halo_bytes"),
                    (
                        f"bfs.steps[{expected_step}]."
                        f"shards[{shard_id}].graphblox_halo_bytes"
                    ),
                ),
            }
            _require_traffic_equal(
                shard_values["cpu_ghost_sync_bytes"],
                expected_sync,
                "BFS per-shard ghost synchronization",
            )
            _require_traffic_equal(
                shard_values["remote_parent_bytes"],
                _traffic_product(
                    shard_values["remote_parent_messages"],
                    4,
                    "BFS per-shard remote-parent bytes",
                ),
                "BFS per-shard remote-parent bytes",
            )
            if phase == "p-bsp-bu":
                _require_traffic_equal(
                    shard_values["remote_parent_messages"],
                    0,
                    "BFS bottom-up remote-parent messages",
                )
            _require_traffic_equal(
                shard_values["graphblox_halo_bytes"],
                halo["bfs_bytes_per_superstep"],
                "BFS per-shard GraphBlox halo bytes",
            )
            for field, value in shard_values.items():
                step_totals[field] = _traffic_add(
                    step_totals[field],
                    value,
                    f"bfs.steps[{expected_step}].{field}",
                )
        for field, expected in step_totals.items():
            actual = _traffic_uint(
                step.get(field),
                f"bfs.steps[{expected_step}].{field}",
            )
            _require_traffic_equal(
                actual,
                expected,
                f"bfs.steps[{expected_step}].{field}",
            )
            aggregate[field] = _traffic_add(
                aggregate[field],
                actual,
                f"bfs.{field}",
            )
    for field, expected in aggregate.items():
        _require_traffic_equal(
            _traffic_uint(bfs.get(field), f"bfs.{field}"),
            expected,
            f"bfs.{field}",
        )
    _require_traffic_equal(
        _traffic_uint(
            bfs.get("cpu_ghost_sync_values"),
            "bfs.cpu_ghost_sync_values",
        ),
        aggregate["cpu_ghost_sync_bytes"],
        "bfs.cpu_ghost_sync_values",
    )
    _require_traffic_equal(
        _traffic_product(
            _traffic_uint(
                bfs.get("graphblox_halo_values"),
                "bfs.graphblox_halo_values",
            ),
            4,
            "bfs.graphblox_halo_values",
        ),
        aggregate["graphblox_halo_bytes"],
        "bfs.graphblox_halo_values",
    )
    _require_traffic_equal(
        aggregate["graphblox_halo_bytes"],
        _traffic_product(
            projection_fields["bfs_bytes_per_superstep"],
            supersteps,
            "bfs.graphblox_halo_bytes",
        ),
        "bfs.graphblox_halo_bytes",
    )
    return traffic


def runtime_traffic(record: dict[str, Any]) -> dict[str, Any]:
    return validate_runtime_traffic(record)


def validate_runtime_traffic_consistency(
    records: list[dict[str, Any]],
) -> None:
    by_fingerprint: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ] = {}
    for record in records:
        key = (
            str(record["mapping_fingerprint"]),
            str(record["shard_fingerprint"]),
            str(record["ghost_fingerprint"]),
            str(record["depth_fingerprint"]),
        )
        traffic = runtime_traffic(record)
        previous = by_fingerprint.setdefault(key, traffic)
        if traffic != previous:
            raise RuntimeError(
                "Phase 2 runtime traffic changed for identical "
                "partition and BFS fingerprints"
            )


def summarize_phase2_policy(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_runtime_traffic_consistency(records)
    summary = summarize_policy(records)
    quality_by_threads: dict[str, dict[str, dict[str, float]]] = {}
    for threads in sorted({int(record["threads"]) for record in records}):
        selected = [
            record
            for record in records
            if int(record["threads"]) == threads
        ]
        quality_by_threads[str(threads)] = {
            "remote_fraction": _range([
                max(
                    float(record["remote_out_fraction"]),
                    float(record["remote_in_fraction"]),
                )
                for record in selected
            ]),
            "ghost_bytes": _range([
                float(record["ghost_bytes"])
                for record in selected
            ]),
            "max_shard_bytes": _range([
                float(record["max_shard_bytes"])
                for record in selected
            ]),
            "storage_imbalance": _range([
                float(record["storage_imbalance"])
                for record in selected
            ]),
            "max_edge_imbalance": _range([
                max(
                    float(record["out_edge_imbalance"]),
                    float(record["in_edge_imbalance"]),
                )
                for record in selected
            ]),
            "bfs_graphblox_halo_bytes": _range([
                float(
                    runtime_traffic(record)["bfs"]
                    ["graphblox_halo_bytes"]
                )
                for record in selected
            ]),
        }
    primary_threads = int(summary["primary_threads"])
    primary = [
        record
        for record in records
        if int(record["threads"]) == primary_threads
    ]
    first = primary[0]
    primary_traffic = [
        runtime_traffic(record) for record in primary
    ]
    summary.update({
        "quality_by_threads": quality_by_threads,
        "quality_aggregation": "worst_repeat_at_primary_threads",
        "remote_fraction": max(
            max(
                float(record["remote_out_fraction"]),
                float(record["remote_in_fraction"]),
            )
            for record in primary
        ),
        "max_remote_fraction": max(
            max(
                float(record["max_remote_out_fraction"]),
                float(record["max_remote_in_fraction"]),
            )
            for record in primary
        ),
        "ghost_bytes": max(
            int(record["ghost_bytes"]) for record in primary
        ),
        "ghost_byte_fraction": max(
            float(record["ghost_byte_fraction"])
            for record in primary
        ),
        "max_shard_bytes": max(
            int(record["max_shard_bytes"])
            for record in records
        ),
        "balance_imbalance": max(
            float(record["balance_imbalance"])
            for record in records
        ),
        "storage_imbalance": max(
            float(record["storage_imbalance"])
            for record in records
        ),
        "max_edge_imbalance": max(
            max(
                float(record["out_edge_imbalance"]),
                float(record["in_edge_imbalance"]),
            )
            for record in records
        ),
        "ghost_slots": max(
            int(traffic["ghost_slots"])
            for traffic in primary_traffic
        ),
        "bfs_supersteps": max(
            int(traffic["bfs"]["supersteps"])
            for traffic in primary_traffic
        ),
        "bfs_cpu_ghost_sync_bytes": max(
            int(traffic["bfs"]["cpu_ghost_sync_bytes"])
            for traffic in primary_traffic
        ),
        "bfs_remote_parent_bytes": max(
            int(traffic["bfs"]["remote_parent_bytes"])
            for traffic in primary_traffic
        ),
        "bfs_cpu_total_bytes": max(
            int(traffic["bfs"]["cpu_ghost_sync_bytes"])
            + int(traffic["bfs"]["remote_parent_bytes"])
            for traffic in primary_traffic
        ),
        "bfs_graphblox_halo_bytes": max(
            int(traffic["bfs"]["graphblox_halo_bytes"])
            for traffic in primary_traffic
        ),
        "bfs_halo_bytes_per_superstep": max(
            int(
                traffic["graphblox_projection"]
                ["bfs_bytes_per_superstep"]
            )
            for traffic in primary_traffic
        ),
        "pr_halo_bytes_per_iteration": max(
            int(
                traffic["graphblox_projection"]
                ["pr_bytes_per_iteration"]
            )
            for traffic in primary_traffic
        ),
        "cc_halo_bytes_per_iteration": max(
            int(
                traffic["graphblox_projection"]
                ["cc_bytes_per_iteration"]
            )
            for traffic in primary_traffic
        ),
        "spmv_initial_halo_bytes": max(
            int(
                traffic["graphblox_projection"]
                ["spmv_initial_bytes"]
            )
            for traffic in primary_traffic
        ),
        "mapping_fingerprint": first["mapping_fingerprint"],
        "source_topology_fingerprint":
            first["source_topology_fingerprint"],
        "shard_fingerprint": first["shard_fingerprint"],
        "ghost_fingerprint": first["ghost_fingerprint"],
        "depth_fingerprint": first["depth_fingerprint"],
    })
    return summary


def add_runtime_relative_metrics(
    summaries: list[dict[str, Any]],
) -> None:
    baseline = next(
        summary
        for summary in summaries
        if summary["policy"] == "original"
    )
    fields = {
        "bfs_cpu_ghost_sync_reduction":
            "bfs_cpu_ghost_sync_bytes",
        "bfs_remote_parent_reduction":
            "bfs_remote_parent_bytes",
        "bfs_cpu_total_reduction":
            "bfs_cpu_total_bytes",
        "bfs_halo_reduction":
            "bfs_graphblox_halo_bytes",
        "pr_halo_reduction":
            "pr_halo_bytes_per_iteration",
        "cc_halo_reduction":
            "cc_halo_bytes_per_iteration",
        "spmv_halo_reduction":
            "spmv_initial_halo_bytes",
    }

    def reduction(
        baseline_value: int,
        value: int,
    ) -> float | None:
        if value > 0:
            return baseline_value / value
        return None if baseline_value > 0 else 1.0

    for summary in summaries:
        summary["runtime_reduction_encoding"] = (
            "null means positive baseline traffic was eliminated"
        )
        for output_field, input_field in fields.items():
            summary[output_field] = reduction(
                int(baseline[input_field]),
                int(summary[input_field]),
            )


def _runtime_reduction_value(value: Any) -> float:
    return float("inf") if value is None else float(value)


def _json_reduction(value: float) -> float | None:
    return None if math.isinf(value) else value


def summarize_graph_records(
    records: list[dict[str, Any]],
    policies: list[Policy],
    max_shard_bytes: int,
) -> list[dict[str, Any]]:
    if not records:
        raise RuntimeError("Phase 2 graph has no records")
    policy_by_name = {policy.name: policy for policy in policies}
    for record in records:
        policy_name = str(record["policy"])
        if policy_name not in policy_by_name:
            raise RuntimeError(
                f"unknown Phase 2 policy record {policy_name}"
            )
        validate_runtime_config(
            policy_by_name[policy_name],
            record.get("runtime_config", {}),
            int(record["threads"]),
        )
    validate_cross_policy(records)
    summaries = [
        summarize_phase2_policy([
            record
            for record in records
            if record["policy"] == policy.name
        ])
        for policy in policies
    ]
    add_relative_metrics(summaries)
    add_runtime_relative_metrics(summaries)
    for summary in summaries:
        summary["absolute_capacity_limit"] = max_shard_bytes or None
        summary["passes_absolute_capacity_gate"] = (
            max_shard_bytes > 0
            and int(summary["max_shard_bytes"]) <= max_shard_bytes
        )
        summary["passes_storage_balance_gate"] = (
            float(summary["storage_imbalance"]) <= 1.10
        )
        summary["passes_direction_balance_gate"] = (
            float(summary["max_edge_imbalance"]) <= 1.10
        )
        summary["passes_capacity_gate"] = (
            bool(summary["passes_capacity_gate"])
            and summary["passes_absolute_capacity_gate"]
        )
        policy_records = [
            record
            for record in records
            if record["policy"] == summary["policy"]
        ]
        determinism = classify_determinism(policy_records)
        summary.update({
            "determinism_class": determinism["classification"],
            "repeat_stable": determinism["repeat_stable"],
            "cross_thread_stable":
                determinism["cross_thread_stable"],
            "determinism_evidence_complete":
                determinism["evidence_complete"],
            "repeat_stable_by_threads":
                determinism["repeat_stable_by_threads"],
        })
        runtime_configs = [
            record.get("runtime_config", {})
            for record in policy_records
        ]
        fallback_runs = sum(
            bool(config.get("cut_min_fallback"))
            for config in runtime_configs
        )
        community_counts = [
            int(config["communities"])
            for config in runtime_configs
            if "communities" in config
        ]
        summary.update({
            "runtime_fallback_runs": fallback_runs,
            "runtime_fallback_fraction":
                fallback_runs / len(runtime_configs),
            "runtime_policy_exact": fallback_runs == 0,
            "community_count_range": (
                {
                    "min": min(community_counts),
                    "max": max(community_counts),
                }
                if community_counts
                else None
            ),
        })
    return summaries


def summarize_corpus(
    graph_results: list[dict[str, Any]],
    policy_names: list[str],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for policy_name in policy_names:
        summaries = [
            graph["summaries_by_policy"][policy_name]
            for graph in graph_results
        ]
        combined_reductions = [
            min(
                float(summary["remote_reduction"]),
                float(summary["ghost_reduction"]),
            )
            for summary in summaries
        ]
        bfs_halo_reductions = [
            _runtime_reduction_value(
                summary["bfs_halo_reduction"])
            for summary in summaries
        ]
        bfs_cpu_reductions = [
            _runtime_reduction_value(
                summary["bfs_cpu_total_reduction"])
            for summary in summaries
        ]
        bfs_ghost_sync_reductions = [
            _runtime_reduction_value(
                summary["bfs_cpu_ghost_sync_reduction"])
            for summary in summaries
        ]
        bfs_remote_parent_reductions = [
            _runtime_reduction_value(
                summary["bfs_remote_parent_reduction"])
            for summary in summaries
        ]
        pr_halo_reductions = [
            _runtime_reduction_value(
                summary["pr_halo_reduction"])
            for summary in summaries
        ]
        cc_halo_reductions = [
            _runtime_reduction_value(
                summary["cc_halo_reduction"])
            for summary in summaries
        ]
        spmv_halo_reductions = [
            _runtime_reduction_value(
                summary["spmv_halo_reduction"])
            for summary in summaries
        ]
        runtime_reductions = [
            *bfs_halo_reductions,
            *bfs_cpu_reductions,
            *bfs_ghost_sync_reductions,
            *bfs_remote_parent_reductions,
            *pr_halo_reductions,
            *cc_halo_reductions,
            *spmv_halo_reductions,
        ]
        preprocess_ratios = [
            float(summary["preprocess_ratio"])
            for summary in summaries
        ]
        aggregate = {
            "policy": policy_name,
            "graphs": len(summaries),
            "categories": sorted(
                {str(graph["category"]) for graph in graph_results}
            ),
            "stable_on_all_graphs": all(
                summary.get(
                    "determinism_class",
                    (
                        "deterministic"
                        if summary.get("stable")
                        else "repeat_variant"
                    ),
                )
                == "deterministic"
                for summary in summaries
            ),
            "geomean_remote_reduction": geometric_mean([
                float(summary["remote_reduction"])
                for summary in summaries
            ]),
            "geomean_ghost_reduction": geometric_mean([
                float(summary["ghost_reduction"])
                for summary in summaries
            ]),
            "geomean_combined_reduction": geometric_mean(
                combined_reductions
            ),
            "runtime_reduction_encoding":
                "null means positive baseline traffic was eliminated",
            "geomean_bfs_halo_reduction": _json_reduction(
                geometric_mean(bfs_halo_reductions)),
            "geomean_bfs_cpu_total_reduction": _json_reduction(
                geometric_mean(bfs_cpu_reductions)),
            "geomean_bfs_cpu_ghost_sync_reduction": _json_reduction(
                geometric_mean(bfs_ghost_sync_reductions)),
            "geomean_bfs_remote_parent_reduction": _json_reduction(
                geometric_mean(bfs_remote_parent_reductions)),
            "geomean_pr_halo_reduction": _json_reduction(
                geometric_mean(pr_halo_reductions)),
            "geomean_cc_halo_reduction": _json_reduction(
                geometric_mean(cc_halo_reductions)),
            "geomean_spmv_halo_reduction": _json_reduction(
                geometric_mean(spmv_halo_reductions)),
            "worst_bfs_halo_reduction": _json_reduction(
                min(bfs_halo_reductions)),
            "worst_bfs_cpu_total_reduction": _json_reduction(
                min(bfs_cpu_reductions)),
            "worst_bfs_cpu_ghost_sync_reduction": _json_reduction(
                min(bfs_ghost_sync_reductions)),
            "worst_bfs_remote_parent_reduction": _json_reduction(
                min(bfs_remote_parent_reductions)),
            "worst_pr_halo_reduction": _json_reduction(
                min(pr_halo_reductions)),
            "worst_cc_halo_reduction": _json_reduction(
                min(cc_halo_reductions)),
            "worst_spmv_halo_reduction": _json_reduction(
                min(spmv_halo_reductions)),
            "worst_runtime_traffic_reduction": _json_reduction(
                min(runtime_reductions)),
            "worst_combined_reduction": min(
                combined_reductions
            ),
            "max_shard_ratio": max(
                float(summary["max_shard_ratio"])
                for summary in summaries
            ),
            "max_balance_imbalance": max(
                float(summary["balance_imbalance"])
                for summary in summaries
            ),
            "max_storage_imbalance": max(
                float(summary.get("storage_imbalance", 1.0))
                for summary in summaries
            ),
            "max_edge_imbalance": max(
                float(summary.get("max_edge_imbalance", 1.0))
                for summary in summaries
            ),
            "absolute_capacity_passed": all(
                bool(
                    summary.get(
                        "passes_absolute_capacity_gate",
                        True,
                    )
                )
                for summary in summaries
            ),
            "runtime_policy_exact_on_all": all(
                bool(summary.get("runtime_policy_exact", True))
                for summary in summaries
            ),
            "fallback_graphs": sum(
                int(summary.get("runtime_fallback_runs", 0)) > 0
                for summary in summaries
            ),
            "max_preprocess_ratio": max(preprocess_ratios),
            "widening_gate_wins": sum(
                bool(summary.get("passes_widening_cut_gate", False))
                for summary in summaries
            ),
        }
        aggregate["passes_universal_default_gate"] = (
            aggregate["stable_on_all_graphs"]
            and aggregate["geomean_combined_reduction"] >= 1.5
            and aggregate["worst_combined_reduction"] >= (1.0 / 1.10)
            and _runtime_reduction_value(
                aggregate["worst_runtime_traffic_reduction"]) >=
                (1.0 / 1.10)
            and aggregate["max_shard_ratio"] <= 1.10
            and aggregate["max_balance_imbalance"] <= 1.05
            and aggregate["max_storage_imbalance"] <= 1.10
            and aggregate["max_edge_imbalance"] <= 1.10
            and aggregate["absolute_capacity_passed"]
            and aggregate["runtime_policy_exact_on_all"]
            and aggregate["max_preprocess_ratio"] <= 10.0
        )
        aggregates.append(aggregate)
    return aggregates


def _print_corpus_summary(
    aggregates: list[dict[str, Any]],
) -> None:
    def runtime_value(value: Any, width: int) -> str:
        text = "elim" if value is None else f"{float(value):.2f}"
        return f"{text:<{width}}"

    print(
        "policy          stable geo_cut geo_ghost bfs_cpu bfs_halo "
        "worst_rt max_shard storage max_work max_prep exact universal"
    )
    for summary in aggregates:
        print(
            f"{summary['policy']:<15} "
            f"{str(summary['stable_on_all_graphs']):<6} "
            f"{summary['geomean_remote_reduction']:<7.2f} "
            f"{summary['geomean_ghost_reduction']:<9.2f} "
            f"{runtime_value(summary['geomean_bfs_cpu_total_reduction'], 7)} "
            f"{runtime_value(summary['geomean_bfs_halo_reduction'], 8)} "
            f"{runtime_value(summary['worst_runtime_traffic_reduction'], 8)} "
            f"{summary['max_shard_ratio']:<9.3f} "
            f"{summary['max_storage_imbalance']:<7.3f} "
            f"{summary['max_balance_imbalance']:<8.3f} "
            f"{summary['max_preprocess_ratio']:<8.2f} "
            f"{str(summary['runtime_policy_exact_on_all']):<5} "
            f"{summary['passes_universal_default_gate']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="smoke",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=tuple(GRAPH_CASES),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(POLICIES),
        default=list(POLICIES),
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "graphs",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="rebuild summaries from existing per-graph records",
    )
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument(
        "--threads",
        type=_parse_ints,
        default=[1, os.cpu_count() or 1],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--partitions", type=int, default=16)
    parser.add_argument(
        "--balance",
        choices=("vertices", "out", "total"),
        default="total",
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES,
        help="absolute per-shard storage budget",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "partition_cut"
            / "phase2"
        ),
    )
    args = parser.parse_args()
    if (
        args.repeats <= 0
        or args.partitions <= 0
        or args.download_workers <= 0
        or args.max_shard_bytes < 0
    ):
        parser.error(
            "--repeats, --partitions, and --download-workers "
            "must be positive"
        )
    if "original" not in args.policies:
        parser.error(
            "--policies must include original as the relative baseline"
        )

    graphs = selected_graphs(args.preset, args.graphs)
    graph_root = args.graph_root.resolve()
    if args.prepare:
        prepare_graphs(
            graphs,
            graph_root,
            workers=args.download_workers,
        )
    missing = [
        graph.path(graph_root)
        for graph in graphs
        if not graph.path(graph_root).is_file()
    ]
    if missing:
        raise SystemExit(
            "missing Phase 2 graphs; pass --prepare:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    selected_policies = [
        POLICIES[name] for name in args.policies
    ]
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    graph_results: list[dict[str, Any]] = []
    determinism_failures: list[str] = []
    determinism_incomplete: list[str] = []
    for graph in graphs:
        graph_output = output_root / graph.name
        graph_path = graph.path(graph_root)
        if args.summarize_existing:
            existing_path = (
                graph_output / "phase2_graph_summary.json"
            )
            if not existing_path.is_file():
                raise RuntimeError(
                    f"missing existing Phase 2 records: "
                    f"{existing_path}"
                )
            existing = json.loads(existing_path.read_text())
            records = existing.get("records")
            if not isinstance(records, list):
                raise RuntimeError(
                    f"existing Phase 2 records are invalid: "
                    f"{existing_path}"
                )
            for record in records:
                run_dir = graph_output / (
                    f"{record.get('policy')}-"
                    f"t{record.get('threads')}-"
                    f"r{record.get('repeat')}"
                )
                stdout_path = run_dir / "stdout.log"
                if not stdout_path.is_file():
                    raise RuntimeError(
                        f"missing existing runtime log: "
                        f"{stdout_path}"
                    )
                record["runtime_config"] = (
                    parse_runtime_config(
                        stdout_path.read_text())
                )
        else:
            records = []
            for policy in selected_policies:
                for threads in args.threads:
                    for repeat in range(args.repeats):
                        print(
                            f"[partition-cut-p2] graph={graph.name} "
                            f"class={graph.category} "
                            f"policy={policy.name} "
                            f"threads={threads} "
                            f"repeat={repeat + 1}/{args.repeats}",
                            flush=True,
                        )
                        record = _run_case(
                                policy=policy,
                                graph_args=[
                                    "-f",
                                    str(graph_path),
                                ],
                                threads=threads,
                                repeat=repeat,
                                partitions=args.partitions,
                                balance=args.balance,
                                source=args.source,
                                timeout=args.timeout,
                                output_root=graph_output,
                            )
                        run_dir = graph_output / (
                            f"{policy.name}-t{threads}-r{repeat}"
                        )
                        stdout = (
                            run_dir / "stdout.log"
                        ).read_text()
                        runtime_config = parse_runtime_config(
                            stdout
                        )
                        validate_runtime_config(
                            policy, runtime_config, threads
                        )
                        record["runtime_config"] = runtime_config
                        record["run_metadata"] = (
                            expected_run_metadata(
                                graph=graph,
                                graph_path=graph_path,
                                policy=policy,
                                threads=threads,
                                repeat=repeat,
                                partitions=args.partitions,
                                balance=args.balance,
                                source=args.source,
                            )
                        )
                        records.append(record)
        expected_metadata = {
            (policy.name, threads, repeat):
            expected_run_metadata(
                graph=graph,
                graph_path=graph_path,
                policy=policy,
                threads=threads,
                repeat=repeat,
                partitions=args.partitions,
                balance=args.balance,
                source=args.source,
            )
            for policy in selected_policies
            for threads in args.threads
            for repeat in range(args.repeats)
        }
        records = validate_record_matrix(
            records, expected_metadata)
        summaries = summarize_graph_records(
            records,
            selected_policies,
            args.max_shard_bytes,
        )
        for summary in summaries:
            if not summary["determinism_evidence_complete"]:
                determinism_incomplete.append(
                    f"{graph.name}:{summary['policy']}"
                )
            if (
                summary["deterministic_required"]
                and summary["determinism_evidence_complete"]
                and summary["determinism_class"] != "deterministic"
            ):
                determinism_failures.append(
                    f"{graph.name}:{summary['policy']}"
                )
        _print_summary(summaries)
        summaries_by_policy = {
            summary["policy"]: summary
            for summary in summaries
        }
        graph_result = {
            "graph": graph.name,
            "category": graph.category,
            "size_tier": graph.size_tier,
            "path": str(graph.path(graph_root)),
            "graph_identity": graph_identity(graph_path),
            "summaries": summaries,
            "summaries_by_policy": summaries_by_policy,
            "records": records,
        }
        graph_results.append(graph_result)
        (graph_output / "phase2_graph_summary.json").write_text(
            json.dumps(
                graph_result,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    aggregates = summarize_corpus(
        graph_results,
        [policy.name for policy in selected_policies],
    )
    result = {
        "schema": "graphbrew.partition_cut.phase2.v1",
        "preset": args.preset,
        "graphs": [asdict(graph) for graph in graphs],
        "partitions": args.partitions,
        "balance": args.balance,
        "source": args.source,
        "max_shard_bytes": args.max_shard_bytes or None,
        "threads": args.threads,
        "repeats": args.repeats,
        "policies": [
            asdict(policy) for policy in selected_policies
        ],
        "correct": True,
        "determinism_complete": not determinism_incomplete,
        "determinism_passed": not determinism_failures,
        "all_policies_deterministic": all(
            aggregate["stable_on_all_graphs"]
            for aggregate in aggregates
        ),
        "valid": (
            not determinism_failures
            and not determinism_incomplete
        ),
        "determinism_failures": determinism_failures,
        "determinism_incomplete": determinism_incomplete,
        "aggregates": aggregates,
        "graph_results": graph_results,
    }
    result_path = output_root / "phase2_summary.json"
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    _print_corpus_summary(aggregates)
    print(f"[partition-cut-p2] wrote {result_path}")
    if determinism_failures:
        raise RuntimeError(
            "required deterministic policies are unstable: "
            + ", ".join(determinism_failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
