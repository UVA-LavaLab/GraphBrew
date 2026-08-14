"""Timing-free forensic analysis of frozen GraphBrew SG/LO artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
import resource
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from scripts.lib.pipeline.benchmark import file_sha256
from scripts.lib.pipeline.benchmark import repository_scope_state

FORENSICS_SCHEMA = "graphbrew-mapping-forensics/v1"
FORENSICS_MODE = "diagnostic-forensic"
FORENSICS_SAMPLE_SEED = 0
FORENSICS_RSS_LIMIT_BYTES = 56 * 1024**3
FORENSICS_WALL_LIMIT_SECONDS = 4 * 60 * 60
FORENSICS_PLAN_SCHEMA = "graphbrew-mapping-forensics-plan/v1"
FORENSICS_RESULT_SCHEMA = "graphbrew-mapping-forensics-result/v1"
EDGE_CHUNK_TARGET = 4_000_000
M3_SAMPLE_LIMIT = 65_536
M3_BOOTSTRAP_BUCKETS = 256
M3_BOOTSTRAP_REPLICATES = 1024
GAP_THRESHOLDS = (8, 64, 4096, 262144)
CLASS_SUPPORT_MIN = 0.001
CLASS_GAP64_FRACTION_MIN = 0.25
CLASS_DIVERGENCE_MARGIN_MIN = 0.05
CLASS_HEADROOM_MIN = 0.05

DISCOVERY_GRAPHS = (
    "cit-Patents",
    "soc-pokec",
    "USA-road-d.USA",
    "soc-LiveJournal1",
    "delaunay_n24",
    "com-Orkut",
    "wikipedia_link_en",
    "Gong-gplus",
)
CONFIRMATION_GRAPHS = (
    "hollywood-2009",
    "webbase-2001",
    "twitter7",
)
GRAPH_TYPES = {
    "cit-Patents": "citation",
    "soc-pokec": "social",
    "USA-road-d.USA": "road",
    "soc-LiveJournal1": "social",
    "delaunay_n24": "mesh",
    "com-Orkut": "social",
    "wikipedia_link_en": "content",
    "Gong-gplus": "social",
    "hollywood-2009": "collaboration",
    "webbase-2001": "web",
    "twitter7": "social",
}
DEFAULT_GRAPH_ROOT = Path("/media/Data/00_GraphDatasets/GraphBrew")
DEFAULT_MAPPING_ROOT = (
    DEFAULT_GRAPH_ROOT / "artifacts" / "vldb_mappings"
)
DEFAULT_EQUIVALENCE_ROOT = (
    DEFAULT_GRAPH_ROOT
    / "artifacts" / "vldb_paper" / "exp3_overhead"
    / "equivalence_checks"
)
DEFAULT_FORENSICS_ROOT = (
    DEFAULT_GRAPH_ROOT / "artifacts" / "mapping_forensics"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORENSICS_IMPLEMENTATION_SCOPE = (
    "scripts/lib/analysis/mapping_forensics.py",
    "scripts/graphbrew_experiment.py",
    "scripts/lib/pipeline/benchmark.py",
)
FORENSICS_PROVENANCE_FILES = (
    "docs/RESEARCH_ROADMAP.md",
    "scripts/test/test_mapping_forensics.py",
)

LAYOUT_INPUT = "INPUT-SHUFFLED"
LAYOUT_SOURCE = "SOURCE-ID-DIAGNOSTIC"
LAYOUT_DBG = "5"
LAYOUT_RABBIT_DRAWS = (
    "8:csr#draw0",
    "8:csr#draw1",
    "8:csr#draw2",
)
LAYOUT_GORDER = "9:csr"
MEASURED_LAYOUTS = (
    LAYOUT_INPUT,
    LAYOUT_SOURCE,
    LAYOUT_DBG,
    *LAYOUT_RABBIT_DRAWS,
    LAYOUT_GORDER,
)

@dataclass(frozen=True)
class SGLayout:
    directed: bool
    nodes: int
    directed_edges: int
    header_bytes: int
    out_offsets_at: int
    out_neighbors_at: int
    in_offsets_at: int | None
    in_neighbors_at: int | None
    org_ids_at: int
    expected_bytes: int


@dataclass(frozen=True)
class MappingArtifact:
    label: str
    path: Path
    sidecar_path: Path | None = None
    draw: int | None = None


@dataclass(frozen=True)
class GraphArtifactSet:
    graph: str
    graph_type: str
    sg_path: Path
    dbg: MappingArtifact
    rabbit_draws: tuple[MappingArtifact, ...]
    rabbit_alias: Path
    rabbit_sidecar: Path
    gorder: MappingArtifact
    gorder_equivalence: Path | None


@dataclass(frozen=True)
class ClassPredicate:
    class_id: int
    name: str
    scheme: str
    code: int
    cardinality: int
    detector_work: str


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def discovery_decision_sha256(
    summary: Mapping[str, Any],
) -> str:
    return canonical_json_sha256({
        "schema": summary.get("schema"),
        "plan_sha256": summary.get("plan_sha256"),
        "thresholds_sha256": summary.get("thresholds_sha256"),
        "status": summary.get("status"),
        "nomination": summary.get("nomination"),
    })


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _repository_state(
    *,
    require_clean: bool,
) -> dict[str, Any]:
    state = repository_scope_state(
        PROJECT_ROOT, FORENSICS_IMPLEMENTATION_SCOPE)
    if require_clean and (
        state["relevant_untracked"]
        or state["relevant_diff_sha256"]
            != hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(
            "Commit the reviewed Route-F implementation before "
            "freezing or executing its plan"
        )
    return state


def _implementation_sha256s() -> dict[str, str]:
    return {
        relative: file_sha256(
            PROJECT_ROOT / relative, use_cache=False)
        for relative in FORENSICS_IMPLEMENTATION_SCOPE
    }


class SerializedGraphMMap:
    """Read-only mmap view of an unweighted GAP serialized graph."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path).resolve()
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(
            self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.layout = self._parse_layout()
        self.out_offsets = np.ndarray(
            shape=(self.layout.nodes + 1,),
            dtype="<i8",
            buffer=self._mmap,
            offset=self.layout.out_offsets_at,
        )
        self.out_neighbors = np.ndarray(
            shape=(self.layout.directed_edges,),
            dtype="<i4",
            buffer=self._mmap,
            offset=self.layout.out_neighbors_at,
        )
        self.org_ids = np.ndarray(
            shape=(self.layout.nodes,),
            dtype="<i4",
            buffer=self._mmap,
            offset=self.layout.org_ids_at,
        )
        if self.layout.directed:
            assert self.layout.in_offsets_at is not None
            assert self.layout.in_neighbors_at is not None
            self.in_offsets = np.ndarray(
                shape=(self.layout.nodes + 1,),
                dtype="<i8",
                buffer=self._mmap,
                offset=self.layout.in_offsets_at,
            )
            self.in_neighbors = np.ndarray(
                shape=(self.layout.directed_edges,),
                dtype="<i4",
                buffer=self._mmap,
                offset=self.layout.in_neighbors_at,
            )
        else:
            self.in_offsets = self.out_offsets
            self.in_neighbors = self.out_neighbors
        self._validate_offsets()

    def _parse_layout(self) -> SGLayout:
        file_size = self.path.stat().st_size
        header_bytes = struct.calcsize("<?qq")
        if file_size < header_bytes:
            raise ValueError("Serialized graph is smaller than its header")
        directed, directed_edges, nodes = struct.unpack_from(
            "<?qq", self._mmap, 0)
        if nodes < 0 or directed_edges < 0:
            raise ValueError(
                "Serialized graph has negative dimensions")
        if nodes > np.iinfo(np.int32).max:
            raise ValueError(
                "Serialized graph exceeds int32 vertex IDs")
        index_bytes = (nodes + 1) * np.dtype("<i8").itemsize
        neighbor_bytes = directed_edges * np.dtype("<i4").itemsize
        ids_bytes = nodes * np.dtype("<i4").itemsize
        offset = header_bytes
        out_offsets_at = offset
        offset += index_bytes
        out_neighbors_at = offset
        offset += neighbor_bytes
        in_offsets_at = None
        in_neighbors_at = None
        if directed:
            in_offsets_at = offset
            offset += index_bytes
            in_neighbors_at = offset
            offset += neighbor_bytes
        org_ids_at = offset
        offset += ids_bytes
        if offset != file_size:
            raise ValueError(
                "Serialized graph size does not match declared layout")
        if not directed and directed_edges % 2:
            raise ValueError(
                "Symmetric graph has an odd directed-edge count")
        return SGLayout(
            directed=bool(directed),
            nodes=int(nodes),
            directed_edges=int(directed_edges),
            header_bytes=header_bytes,
            out_offsets_at=out_offsets_at,
            out_neighbors_at=out_neighbors_at,
            in_offsets_at=in_offsets_at,
            in_neighbors_at=in_neighbors_at,
            org_ids_at=org_ids_at,
            expected_bytes=offset,
        )

    def _validate_offsets(self) -> None:
        if (
            int(self.out_offsets[0]) != 0
            or int(self.out_offsets[-1])
            != self.layout.directed_edges
        ):
            raise ValueError("Serialized graph offsets have invalid bounds")
        for start in range(0, self.layout.nodes, 4_000_000):
            stop = min(self.layout.nodes, start + 4_000_000)
            if np.any(
                self.out_offsets[start + 1:stop + 1]
                < self.out_offsets[start:stop]
            ):
                raise ValueError(
                    "Serialized graph offsets are not monotonic")

    @property
    def nodes(self) -> int:
        return self.layout.nodes

    @property
    def directed_edges(self) -> int:
        return self.layout.directed_edges

    @property
    def undirected_edges(self) -> int:
        if self.layout.directed:
            return self.layout.directed_edges
        return self.layout.directed_edges // 2

    def degrees(self) -> np.ndarray:
        result = np.empty(self.nodes, dtype=np.int32)
        for start in range(0, self.nodes, 4_000_000):
            stop = min(self.nodes, start + 4_000_000)
            values = (
                self.out_offsets[start + 1:stop + 1]
                - self.out_offsets[start:stop]
            )
            if np.any(values > np.iinfo(np.int32).max):
                raise ValueError("Vertex degree exceeds int32")
            result[start:stop] = values.astype(np.int32)
        return result

    def iter_edge_chunks(
        self,
        target_edges: int = EDGE_CHUNK_TARGET,
        *,
        undirected_once: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if target_edges <= 0:
            raise ValueError("Edge chunk target must be positive")
        start_vertex = 0
        while start_vertex < self.nodes:
            start_edge = int(self.out_offsets[start_vertex])
            target_edge = min(
                self.directed_edges, start_edge + target_edges)
            end_vertex = int(np.searchsorted(
                self.out_offsets,
                target_edge,
                side="right",
            ) - 1)
            end_vertex = max(start_vertex + 1, end_vertex)
            end_vertex = min(self.nodes, end_vertex)
            end_edge = int(self.out_offsets[end_vertex])
            local_degrees = (
                self.out_offsets[start_vertex + 1:end_vertex + 1]
                - self.out_offsets[start_vertex:end_vertex]
            ).astype(np.int64, copy=False)
            sources = np.repeat(
                np.arange(
                    start_vertex, end_vertex, dtype=np.int32),
                local_degrees,
            )
            destinations = np.asarray(
                self.out_neighbors[start_edge:end_edge],
                dtype=np.int32,
            )
            if (
                destinations.size != sources.size
                or np.any(destinations < 0)
                or np.any(destinations >= self.nodes)
            ):
                raise ValueError(
                    "Serialized graph neighbor region is invalid")
            if undirected_once and not self.layout.directed:
                keep = sources < destinations
                yield sources[keep], destinations[keep]
            else:
                yield sources, destinations
            start_vertex = end_vertex

    def close(self) -> None:
        for name in (
            "out_offsets",
            "out_neighbors",
            "in_offsets",
            "in_neighbors",
            "org_ids",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "SerializedGraphMMap":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.close()
        return False


def current_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def peak_rss_bytes() -> int:
    return int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ) * 1024


def enforce_rss_limit(
    limit_bytes: int = FORENSICS_RSS_LIMIT_BYTES,
) -> None:
    observed = current_rss_bytes()
    if observed > limit_bytes:
        raise MemoryError(
            f"Forensics RSS {observed} exceeds limit {limit_bytes}")


def validate_int32_permutation(
    values: np.ndarray,
    *,
    expected_size: int,
    label: str,
    chunk_size: int = 1_000_000,
) -> None:
    if values.ndim != 1 or values.size != expected_size:
        raise ValueError(
            f"{label} length changed: {values.size} != {expected_size}")
    seen = np.zeros(expected_size, dtype=np.uint8)
    observed = 0
    for start in range(0, expected_size, chunk_size):
        chunk = np.asarray(
            values[start:start + chunk_size], dtype=np.int64)
        if (
            np.any(chunk < 0)
            or np.any(chunk >= expected_size)
        ):
            raise ValueError(f"{label} contains an out-of-range ID")
        unique = np.unique(chunk)
        if unique.size != chunk.size or np.any(seen[unique]):
            raise ValueError(f"{label} contains duplicate IDs")
        seen[unique] = 1
        observed += unique.size
    if observed != expected_size:
        raise ValueError(f"{label} is not a complete permutation")


def array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(values)
    view = memoryview(contiguous).cast("B")
    block = 16 * 1024 * 1024
    for start in range(0, len(view), block):
        digest.update(view[start:start + block])
    return digest.hexdigest()


def composed_permutation_fingerprint(
    positions_by_sg: np.ndarray,
) -> str:
    return "forensic-int32-sha256:" + array_sha256(
        positions_by_sg.astype("<i4", copy=False))


def load_text_mapping_positions(
    mapping_path: str | os.PathLike,
    *,
    nodes: int,
    org_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(mapping_path).resolve()
    new_to_source = np.fromfile(path, dtype=np.int64, sep=" ")
    validate_int32_permutation(
        new_to_source,
        expected_size=nodes,
        label=f"mapping {path.name}",
    )
    source_to_new = np.empty(nodes, dtype=np.int32)
    for start in range(0, nodes, 4_000_000):
        stop = min(nodes, start + 4_000_000)
        source_ids = new_to_source[start:stop]
        source_to_new[source_ids] = np.arange(
            start, stop, dtype=np.int32)
    validate_int32_permutation(
        org_ids,
        expected_size=nodes,
        label="serialized graph org_ids",
    )
    positions = source_to_new[
        np.asarray(org_ids, dtype=np.int32)
    ]
    metadata = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path, use_cache=False),
        "source_to_new_fingerprint":
            composed_permutation_fingerprint(source_to_new),
        "composed_sg_to_new_fingerprint":
            composed_permutation_fingerprint(positions),
    }
    del new_to_source
    del source_to_new
    enforce_rss_limit()
    return positions, metadata


def dbg_bucket_codes(
    degrees: np.ndarray,
    *,
    average_degree: int | None = None,
) -> np.ndarray:
    if degrees.ndim != 1:
        raise ValueError("Degrees must be one-dimensional")
    nodes = degrees.size
    average = (
        int(average_degree)
        if average_degree is not None
        else int(np.sum(
            degrees, dtype=np.int64) // max(1, nodes))
    )
    thresholds = np.array([
        average // 2,
        average,
        average * 2,
        average * 4,
        average * 8,
        average * 16,
        average * 32,
    ], dtype=np.int64)
    return np.searchsorted(
        thresholds,
        degrees.astype(np.int64, copy=False),
        side="left",
    ).astype(np.uint8)


def validate_dbg_semantics(
    positions_by_sg: np.ndarray,
    degrees: np.ndarray,
) -> dict[str, Any]:
    if positions_by_sg.size != degrees.size:
        raise ValueError("DBG mapping and degree lengths differ")
    inverse = np.empty(positions_by_sg.size, dtype=np.int32)
    inverse[positions_by_sg] = np.arange(
        positions_by_sg.size, dtype=np.int32)
    directed_average = int(
        np.sum(degrees, dtype=np.int64)
        // max(1, degrees.size)
    )
    policies = (
        ("adjacency-degree-average", directed_average),
        ("legacy-half-edge-average", directed_average // 2),
    )
    matches = []
    for policy, average in policies:
        if policy == "legacy-half-edge-average" and average == 0:
            continue
        ordered_buckets = dbg_bucket_codes(
            degrees, average_degree=average)[inverse]
        if not np.any(
            ordered_buckets[1:] > ordered_buckets[:-1]
        ):
            counts = np.bincount(ordered_buckets, minlength=8)
            matches.append({
                "policy": policy,
                "average_degree": average,
                "bucket_counts_low_to_high": counts.tolist(),
                "ordered_buckets_high_to_low": [
                    int(code) for code in np.flatnonzero(counts)[::-1]
                ],
            })
    if matches:
        return {
            "schema": "dbg-bucket-validation/v3",
            "valid": True,
            "semantics": (
                matches[0]["policy"]
                if len(matches) == 1 else "ambiguous"
            ),
            "consistent_semantics": [
                match["policy"] for match in matches
            ],
            "matches": matches,
        }
    raise ValueError(
        "DBG mapping violates current and legacy bucket semantics")


def _quantile_codes(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    finite = np.asarray(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("Cannot quantile-bin an empty feature")
    thresholds = [
        float(value) for value in np.quantile(
            finite,
            (0.25, 0.5, 0.75),
            method="higher",
        )
    ]
    codes = np.searchsorted(
        np.asarray(thresholds),
        values,
        side="right",
    ).astype(np.uint8)
    return codes, thresholds


def _batched_sorted_row_contains(
    graph: SerializedGraphMMap,
    row_vertices: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if row_vertices.size != targets.size:
        raise ValueError("Batched adjacency query lengths differ")
    lower = _batched_sorted_row_bound(
        graph, row_vertices, targets, upper=False)
    upper = _batched_sorted_row_bound(
        graph, row_vertices, targets, upper=True)
    return upper > lower


def _batched_sorted_row_bound(
    graph: SerializedGraphMMap,
    row_vertices: np.ndarray,
    targets: np.ndarray,
    *,
    upper: bool,
) -> np.ndarray:
    low = np.asarray(
        graph.out_offsets[row_vertices], dtype=np.int64).copy()
    high = np.asarray(
        graph.out_offsets[row_vertices + 1], dtype=np.int64).copy()
    while np.any(low < high):
        middle = (low + high) // 2
        safe_middle = np.minimum(
            middle, graph.directed_edges - 1)
        values = graph.out_neighbors[safe_middle]
        move_low = (
            (low < high)
            & (values <= targets if upper else values < targets)
        )
        low = np.where(move_low, middle + 1, low)
        high = np.where(move_low, high, middle)
    return low


def _batched_sorted_row_count(
    graph: SerializedGraphMMap,
    row_vertices: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    lower = _batched_sorted_row_bound(
        graph, row_vertices, targets, upper=False)
    upper = _batched_sorted_row_bound(
        graph, row_vertices, targets, upper=True)
    return upper - lower


def _self_loop_counts(
    graph: SerializedGraphMMap,
) -> np.ndarray:
    counts = np.empty(graph.nodes, dtype=np.int32)
    for start in range(0, graph.nodes, 2_000_000):
        stop = min(graph.nodes, start + 2_000_000)
        vertices = np.arange(start, stop, dtype=np.int32)
        counts[start:stop] = _batched_sorted_row_count(
            graph, vertices, vertices).astype(np.int32)
    return counts


def _bounded_low_degree_clustering(
    graph: SerializedGraphMMap,
    start_vertex: int,
    end_vertex: int,
    local_degree: np.ndarray,
    local_loop_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_result = np.full(
        end_vertex - start_vertex, np.nan, dtype=np.float32)
    eligible = np.flatnonzero(
        (local_degree >= 2)
        & (local_degree <= 8)
        & (local_loop_count == 0)
    )
    if eligible.size == 0:
        return local_result, eligible
    vertices = (
        eligible + start_vertex).astype(np.int32)
    row_degree = local_degree[eligible].astype(np.int32)
    row_start = np.asarray(
        graph.out_offsets[vertices], dtype=np.int64)
    padded = np.full((eligible.size, 8), -1, dtype=np.int32)
    for rank in range(8):
        valid = row_degree > rank
        if np.any(valid):
            padded[valid, rank] = graph.out_neighbors[
                row_start[valid] + rank
            ]
    closed = np.zeros(eligible.size, dtype=np.int16)
    for left_rank in range(7):
        for right_rank in range(left_rank + 1, 8):
            valid = row_degree > right_rank
            if not np.any(valid):
                continue
            found = _batched_sorted_row_contains(
                graph,
                padded[valid, left_rank],
                padded[valid, right_rank],
            )
            closed[valid] += found.astype(np.int16)
    possible = (
        row_degree * (row_degree - 1) // 2
    ).astype(np.float32)
    local_result[eligible] = closed / possible
    return local_result, eligible


def _clustering_codes(
    values: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    finite = np.isfinite(values)
    codes = np.full(values.size, 4, dtype=np.uint8)
    if not np.any(finite):
        return codes, []
    finite_codes, thresholds = _quantile_codes(values[finite])
    codes[finite] = finite_codes
    return codes, thresholds


def compute_vertex_feature_codes(
    graph: SerializedGraphMMap,
    degrees: np.ndarray,
    *,
    retain_undirected_edges: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    list[tuple[np.ndarray, np.ndarray]] | None,
    np.ndarray,
]:
    if graph.layout.directed:
        raise ValueError(
            "Route F currently requires symmetric SG inputs")
    loop_count_by_vertex = _self_loop_counts(graph)
    feature_degrees = degrees - loop_count_by_vertex
    if np.any(feature_degrees < 0):
        raise ValueError("Self-loop count exceeds stored degree")
    neighbor_mean = np.zeros(graph.nodes, dtype=np.float32)
    core_proxy = np.zeros(graph.nodes, dtype=np.int32)
    clustering = np.full(
        graph.nodes, np.nan, dtype=np.float32)
    self_loops = 0
    forward_edges = 0
    retained_edges: list[
        tuple[np.ndarray, np.ndarray]
    ] | None = [] if retain_undirected_edges else None
    start_vertex = 0
    while start_vertex < graph.nodes:
        start_edge = int(graph.out_offsets[start_vertex])
        target_edge = min(
            graph.directed_edges,
            start_edge + EDGE_CHUNK_TARGET,
        )
        end_vertex = int(np.searchsorted(
            graph.out_offsets, target_edge, side="right") - 1)
        end_vertex = max(start_vertex + 1, end_vertex)
        end_vertex = min(graph.nodes, end_vertex)
        end_edge = int(graph.out_offsets[end_vertex])
        local_offsets = (
            graph.out_offsets[start_vertex:end_vertex + 1]
            - start_edge
        ).astype(np.int64, copy=False)
        raw_local_degree = degrees[start_vertex:end_vertex]
        local_degree = feature_degrees[start_vertex:end_vertex]
        destinations = np.asarray(
            graph.out_neighbors[start_edge:end_edge],
            dtype=np.int32,
        )
        sources = np.repeat(
            np.arange(
                start_vertex, end_vertex, dtype=np.int32),
            raw_local_degree.astype(np.int64),
        )
        if destinations.size != sources.size:
            raise ValueError("Feature edge chunk shape changed")
        if destinations.size > 1 and np.any(
            (sources[1:] == sources[:-1])
            & (destinations[1:] < destinations[:-1])
        ):
            raise ValueError("SG neighbor rows are not sorted")
        if destinations.size:
            run_start = np.empty(
                destinations.size, dtype=np.bool_)
            run_start[0] = True
            run_start[1:] = (
                (sources[1:] != sources[:-1])
                | (destinations[1:] != destinations[:-1])
            )
            starts = np.flatnonzero(run_start)
            counts = np.diff(np.append(
                starts, destinations.size))
            unique_sources = sources[starts]
            unique_destinations = destinations[starts]
            reciprocal = _batched_sorted_row_count(
                graph,
                unique_destinations,
                unique_sources,
            )
            if np.any(reciprocal != counts):
                raise ValueError(
                    "SG reciprocal edge multiplicities are invalid")
        self_loops += int(np.count_nonzero(
            sources == destinations))
        forward_mask = sources < destinations
        forward_edges += int(np.count_nonzero(forward_mask))
        if retained_edges is not None:
            retained_edges.append((
                sources[forward_mask].copy(),
                destinations[forward_mask].copy(),
            ))
        neighbor_degrees = feature_degrees[
            destinations].astype(np.float64)
        loop_mask = sources == destinations
        non_loop_weights = (~loop_mask).astype(np.int32)
        local_loop_count = loop_count_by_vertex[
            start_vertex:end_vertex]
        effective_degree = local_degree
        neighbor_degrees[loop_mask] = 0.0
        prefix = np.empty(neighbor_degrees.size + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(neighbor_degrees, out=prefix[1:])
        sums = prefix[local_offsets[1:]] - prefix[local_offsets[:-1]]
        np.divide(
            sums,
            np.maximum(effective_degree, 1),
            out=neighbor_mean[start_vertex:end_vertex],
            casting="unsafe",
        )
        supported = (
            neighbor_degrees
            >= feature_degrees[sources].astype(np.float64)
        ).astype(np.int32) * non_loop_weights
        support_prefix = np.empty(
            supported.size + 1, dtype=np.int64)
        support_prefix[0] = 0
        np.cumsum(supported, out=support_prefix[1:])
        core_proxy[start_vertex:end_vertex] = (
            support_prefix[local_offsets[1:]]
            - support_prefix[local_offsets[:-1]]
        ).astype(np.int32)
        local_clustering, _eligible = _bounded_low_degree_clustering(
            graph,
            start_vertex,
            end_vertex,
            effective_degree,
            local_loop_count,
        )
        clustering[start_vertex:end_vertex] = local_clustering
        start_vertex = end_vertex
        enforce_rss_limit()
    if 2 * forward_edges + self_loops != graph.directed_edges:
        raise ValueError(
            "Symmetric SG edge pairing is invalid: "
            f"2*{forward_edges}+{self_loops} != "
            f"{graph.directed_edges}"
        )
    degree_codes, degree_thresholds = _quantile_codes(
        feature_degrees)
    neighbor_codes, neighbor_thresholds = _quantile_codes(neighbor_mean)
    core_codes, core_thresholds = _quantile_codes(core_proxy)
    clustering_codes, clustering_thresholds = _clustering_codes(
        clustering)
    codes = {
        "degree": degree_codes,
        "neighbor": neighbor_codes,
        "core": core_codes,
        "clustering": clustering_codes,
        "degree_neighbor": (
            degree_codes * 4 + neighbor_codes).astype(np.uint8),
        "core_clustering": (
            core_codes * 5 + clustering_codes).astype(np.uint8),
    }
    metadata = {
        "schema": "forensic-feature-codes/v1",
        "degree_quantiles": degree_thresholds,
        "neighbor_degree_mean_quantiles": neighbor_thresholds,
        "core_proxy_quantiles": core_thresholds,
        "core_proxy_definition":
            "count(neighbor_degree >= vertex_degree)",
        "bounded_clustering_quantiles": clustering_thresholds,
        "bounded_clustering_definition":
            "exact coefficient for degree 2..8; unknown code 4 otherwise",
        "self_loop_entries_observed": self_loops,
        "undirected_edges_scanned": forward_edges,
        "directed_edge_identity":
            f"2*{forward_edges}+{self_loops}={graph.directed_edges}",
    }
    return codes, metadata, retained_edges, feature_degrees


def frozen_class_predicates() -> tuple[ClassPredicate, ...]:
    predicates = []
    class_id = 0
    for scheme in ("degree", "neighbor", "core"):
        for code in range(4):
            predicates.append(ClassPredicate(
                class_id=class_id,
                name=f"{scheme}:q{code}",
                scheme=scheme,
                code=code,
                cardinality=4,
                detector_work="O(m)",
            ))
            class_id += 1
    for code in range(5):
        predicates.append(ClassPredicate(
            class_id=class_id,
            name=(
                f"clustering:q{code}"
                if code < 4 else "clustering:unmeasured"
            ),
            scheme="clustering",
            code=code,
            cardinality=5,
            detector_work="O(n log max_degree) diagnostic",
        ))
        class_id += 1
    for scheme in ("degree_neighbor",):
        for code in range(16):
            left, right = divmod(code, 4)
            predicates.append(ClassPredicate(
                class_id=class_id,
                name=f"{scheme}:q{left}-q{right}",
                scheme=scheme,
                code=code,
                cardinality=16,
                detector_work="O(m)",
            ))
            class_id += 1
    for code in range(20):
        left, right = divmod(code, 5)
        predicates.append(ClassPredicate(
            class_id=class_id,
            name=f"core_clustering:q{left}-q{right}",
            scheme="core_clustering",
            code=code,
            cardinality=20,
            detector_work="O(n log max_degree) diagnostic",
        ))
        class_id += 1
    if len(predicates) != 53:
        raise RuntimeError("Forensic class bank must contain 53 predicates")
    return tuple(predicates)


CLASS_PREDICATES = frozen_class_predicates()
CLASS_SPEC = {
    "schema": "forensic-class-spec/v1",
    "predicates": [
        {
            "class_id": predicate.class_id,
            "name": predicate.name,
            "scheme": predicate.scheme,
            "code": predicate.code,
            "cardinality": predicate.cardinality,
            "detector_work": predicate.detector_work,
        }
        for predicate in CLASS_PREDICATES
    ],
    "support_min": CLASS_SUPPORT_MIN,
    "gap64_fraction_min": CLASS_GAP64_FRACTION_MIN,
    "divergence_margin_min": CLASS_DIVERGENCE_MARGIN_MIN,
    "headroom_min": CLASS_HEADROOM_MIN,
    "sample_seed": FORENSICS_SAMPLE_SEED,
    "m3_sample_limit": M3_SAMPLE_LIMIT,
    "m3_bootstrap_buckets": M3_BOOTSTRAP_BUCKETS,
    "m3_bootstrap_replicates": M3_BOOTSTRAP_REPLICATES,
    "quantile_method": "numpy-higher",
    "core_proxy": "count(neighbor_degree >= vertex_degree)",
    "bounded_clustering":
        "exact degree 2..8, unknown code 4 otherwise",
    "positive_bit_cost":
        "1+floor(log2(max(1, absolute_position_gap)))",
    "gap_thresholds": list(GAP_THRESHOLDS),
    "u64_definition":
        "class excess above b(64)=7 divided by global bit mass",
    "rabbit_gorder_disagreement":
        "minimum of three actual Rabbit-vs-Gorder class rates",
    "rabbit_draw_disagreement":
        "maximum of three separately aggregated Rabbit-pair class rates",
    "h0_rule": "7 of exactly 8 discovery graphs",
    "corpus_rule": "at least 3 distinct graphs and 2 graph types",
    "nomination_score":
        "mean(min(excess64_density_rabbit_min,"
        "excess64_density_gorder)*divergence_margin)",
    "nomination_tiebreak": "lowest class_id",
}
CLASS_BANK_SHA256 = hashlib.sha256(json.dumps(
    CLASS_SPEC,
    sort_keys=True,
    separators=(",", ":"),
).encode()).hexdigest()


def positive_bit_cost(gaps: np.ndarray) -> np.ndarray:
    safe = np.maximum(gaps, 1)
    return (
        np.floor(np.log2(safe)).astype(np.uint8) + 1
    )


def analytic_random_same_line_null(
    nodes: int,
    *,
    vertices_per_line: int = 8,
) -> float:
    if nodes < 2:
        return 0.0
    if vertices_per_line <= 0:
        raise ValueError("Vertices per line must be positive")
    full_blocks, remainder = divmod(nodes, vertices_per_line)
    within_pairs = (
        full_blocks * vertices_per_line * (vertices_per_line - 1) // 2
        + remainder * (remainder - 1) // 2
    )
    total_pairs = nodes * (nodes - 1) // 2
    return within_pairs / total_pairs


def sampled_distinct_lines_per_degree(
    graph: SerializedGraphMMap,
    degrees: np.ndarray,
    layouts: Mapping[str, np.ndarray],
    degree_codes: np.ndarray,
    *,
    sample_limit: int = M3_SAMPLE_LIMIT,
    seed: int = FORENSICS_SAMPLE_SEED,
) -> dict[str, Any]:
    if sample_limit <= 0:
        raise ValueError("M3 sample limit must be positive")
    if degree_codes.size != graph.nodes:
        raise ValueError("M3 degree-code length changed")
    random = np.random.default_rng(seed)
    sample_parts = []
    per_stratum = math.ceil(sample_limit / 4)
    for stratum in range(4):
        candidates = np.flatnonzero(degree_codes == stratum)
        if candidates.size <= per_stratum:
            selected = candidates
        else:
            ordinals = random.choice(
                candidates.size,
                size=per_stratum,
                replace=False,
            )
            selected = candidates[ordinals]
        sample_parts.append(selected.astype(np.int32, copy=False))
    sample = np.concatenate(sample_parts)
    if sample.size > sample_limit:
        sample = sample[:sample_limit]
    accumulators = {
        label: {
            "ratio_sum": np.zeros(4, dtype=np.float64),
            "count": np.zeros(4, dtype=np.int64),
            "bucket_sum": np.zeros(
                (4, M3_BOOTSTRAP_BUCKETS), dtype=np.float64),
            "bucket_count": np.zeros(
                (4, M3_BOOTSTRAP_BUCKETS), dtype=np.int64),
        }
        for label in layouts
    }
    for vertex in sample:
        vertex_id = int(vertex)
        start = int(graph.out_offsets[vertex_id])
        stop = int(graph.out_offsets[vertex_id + 1])
        neighbors = np.asarray(
            graph.out_neighbors[start:stop], dtype=np.int32)
        neighbors = neighbors[neighbors != vertex_id]
        degree = max(1, int(degrees[vertex_id]))
        stratum = int(degree_codes[vertex_id])
        mixed = (
            vertex_id + seed + 0x9E3779B97F4A7C15
        ) & ((1 << 64) - 1)
        mixed = (
            (mixed ^ (mixed >> 30))
            * 0xBF58476D1CE4E5B9
        ) & ((1 << 64) - 1)
        mixed = (
            (mixed ^ (mixed >> 27))
            * 0x94D049BB133111EB
        ) & ((1 << 64) - 1)
        mixed ^= mixed >> 31
        bucket = (
            mixed >> 56
        ) & (M3_BOOTSTRAP_BUCKETS - 1)
        for label, positions in layouts.items():
            distinct_lines = (
                np.unique(positions[neighbors] // 8).size
                if neighbors.size else 0
            )
            ratio = distinct_lines / degree
            accum = accumulators[label]
            accum["ratio_sum"][stratum] += ratio
            accum["count"][stratum] += 1
            accum["bucket_sum"][stratum, bucket] += ratio
            accum["bucket_count"][stratum, bucket] += 1
    result = {}
    for label_index, (label, accum) in enumerate(accumulators.items()):
        strata = []
        for stratum in range(4):
            count = int(accum["count"][stratum])
            valid = accum["bucket_count"][stratum] > 0
            bucket_means = np.divide(
                accum["bucket_sum"][stratum, valid],
                accum["bucket_count"][stratum, valid],
            )
            bootstrap_rng = np.random.default_rng(
                seed + label_index * 16 + stratum)
            selections = bootstrap_rng.integers(
                0,
                M3_BOOTSTRAP_BUCKETS,
                size=(
                    M3_BOOTSTRAP_REPLICATES,
                    M3_BOOTSTRAP_BUCKETS,
                ),
                dtype=np.int16,
            )
            bootstrap_sums = np.sum(
                accum["bucket_sum"][stratum][selections],
                axis=1,
            )
            bootstrap_counts = np.sum(
                accum["bucket_count"][stratum][selections],
                axis=1,
            )
            bootstrap_means = np.divide(
                bootstrap_sums,
                np.maximum(bootstrap_counts, 1),
            )
            strata.append({
                "degree_quantile": stratum,
                "sample_count": count,
                "mean_distinct_lines_per_degree": (
                    accum["ratio_sum"][stratum] / max(1, count)
                ),
                "bootstrap_95_interval": [
                    float(np.quantile(bootstrap_means, 0.025)),
                    float(np.quantile(bootstrap_means, 0.975)),
                ],
                "bucket_sums": accum[
                    "bucket_sum"][stratum].tolist(),
                "bucket_counts": accum[
                    "bucket_count"][stratum].tolist(),
                "cluster_bucket_means": bucket_means.tolist(),
            })
        result[label] = strata
    return {
        "schema": "forensic-m3-distinct-lines/v1",
        "diagnostic_only": True,
        "pre_rejected_objective":
            "hypergraph-connectivity-lambda-minus-one",
        "sample_seed": seed,
        "sample_limit": sample_limit,
        "sample_count": int(sample.size),
        "bootstrap_bucket_count": M3_BOOTSTRAP_BUCKETS,
        "bootstrap_replicates": M3_BOOTSTRAP_REPLICATES,
        "bootstrap_semantics":
            "resample 256 mixed-hash buckets with replacement",
        "layouts": result,
    }


def _incident_bincount(
    source_codes: np.ndarray,
    destination_codes: np.ndarray,
    weights: np.ndarray,
    cardinality: int,
) -> np.ndarray:
    source = np.bincount(
        source_codes,
        weights=weights,
        minlength=cardinality,
    )
    destination = np.bincount(
        destination_codes,
        weights=weights,
        minlength=cardinality,
    )
    same = source_codes == destination_codes
    duplicate = np.bincount(
        source_codes[same],
        weights=weights[same],
        minlength=cardinality,
    )
    return source + destination - duplicate


def _layout_metric_state() -> dict[str, Any]:
    return {
        "edges": 0,
        "bit_sum": 0.0,
        "same_line_edges": 0,
        "gap_exceed_count": {
            str(threshold): 0 for threshold in GAP_THRESHOLDS
        },
        "gap_excess_bit_sum": {
            str(threshold): 0.0 for threshold in GAP_THRESHOLDS
        },
    }


def scan_multi_layout_metrics(
    graph: SerializedGraphMMap,
    layouts: Mapping[str, np.ndarray],
    feature_codes: Mapping[str, np.ndarray],
    *,
    self_loops: int,
    expected_undirected_edges: int,
    edge_chunks: Iterable[
        tuple[np.ndarray, np.ndarray]
    ] | None = None,
) -> dict[str, Any]:
    required = set(MEASURED_LAYOUTS)
    if set(layouts) != required:
        raise ValueError(
            "Forensic layout set changed: "
            + " ".join(sorted(set(layouts))))
    for label, positions in layouts.items():
        if positions.dtype != np.int32 or positions.size != graph.nodes:
            raise ValueError(
                f"Layout {label} must be an int32 SG-position array")
    class_accumulators: dict[str, dict[str, np.ndarray]] = {}
    for scheme, codes in feature_codes.items():
        cardinalities = {
            predicate.cardinality
            for predicate in CLASS_PREDICATES
            if predicate.scheme == scheme
        }
        if len(cardinalities) != 1:
            raise ValueError(
                f"Forensic class scheme changed: {scheme}")
        cardinality = cardinalities.pop()
        class_accumulators[scheme] = {
            "support": np.zeros(cardinality, dtype=np.float64),
            "gorder_gap64": np.zeros(cardinality, dtype=np.float64),
            "gorder_excess64": np.zeros(cardinality, dtype=np.float64),
            "rabbit_gorder_disagreement_draws":
                np.zeros((3, cardinality), dtype=np.float64),
            "rabbit_pair_disagreement":
                np.zeros((3, cardinality), dtype=np.float64),
            "rabbit_gap64_draws":
                np.zeros((3, cardinality), dtype=np.float64),
            "rabbit_excess64_draws":
                np.zeros((3, cardinality), dtype=np.float64),
        }
        if codes.size != graph.nodes:
            raise ValueError(f"Feature code length changed: {scheme}")
    layout_state = {
        label: _layout_metric_state() for label in layouts
    }
    undirected_edges = 0
    rabbit_pair_disagreement_edges = np.zeros(3, dtype=np.int64)
    chunks = (
        edge_chunks
        if edge_chunks is not None
        else graph.iter_edge_chunks()
    )
    for sources, destinations in chunks:
        edges = sources.size
        if edges == 0:
            continue
        undirected_edges += edges
        bits: dict[str, np.ndarray] = {}
        gaps: dict[str, np.ndarray] = {}
        for label, positions in layouts.items():
            gap = np.abs(
                positions[sources].astype(np.int64)
                - positions[destinations].astype(np.int64)
            )
            bit = positive_bit_cost(gap)
            gaps[label] = gap
            bits[label] = bit
            state = layout_state[label]
            state["edges"] += edges
            state["bit_sum"] += float(np.sum(
                bit, dtype=np.float64))
            state["same_line_edges"] += int(np.count_nonzero(
                positions[sources] // 8
                == positions[destinations] // 8
            ))
            for threshold in GAP_THRESHOLDS:
                key = str(threshold)
                threshold_bit = (
                    1 + int(math.floor(math.log2(threshold))))
                state["gap_exceed_count"][key] += int(
                    np.count_nonzero(gap > threshold))
                state["gap_excess_bit_sum"][key] += float(np.sum(
                    np.maximum(
                        bit.astype(np.int16) - threshold_bit,
                        0,
                    ),
                    dtype=np.float64,
                ))
        gorder_bits = bits[LAYOUT_GORDER]
        rabbit_pair_arrays = (
            (
                bits[LAYOUT_RABBIT_DRAWS[0]]
                != bits[LAYOUT_RABBIT_DRAWS[1]]
            ).astype(np.float64),
            (
                bits[LAYOUT_RABBIT_DRAWS[0]]
                != bits[LAYOUT_RABBIT_DRAWS[2]]
            ).astype(np.float64),
            (
                bits[LAYOUT_RABBIT_DRAWS[1]]
                != bits[LAYOUT_RABBIT_DRAWS[2]]
            ).astype(np.float64),
        )
        for pair_index, pair_values in enumerate(rabbit_pair_arrays):
            rabbit_pair_disagreement_edges[pair_index] += int(
                np.count_nonzero(pair_values))
        gorder_gap64 = (gorder_bits > 7).astype(np.float64)
        gorder_excess64 = np.maximum(
            gorder_bits.astype(np.int16) - 7, 0
        ).astype(np.float64)
        ones = np.ones(edges, dtype=np.float64)
        for scheme, codes in feature_codes.items():
            cardinality = class_accumulators[scheme][
                "support"].size
            source_codes = codes[sources]
            destination_codes = codes[destinations]
            accum = class_accumulators[scheme]
            accum["support"] += _incident_bincount(
                source_codes, destination_codes, ones, cardinality)
            accum["gorder_gap64"] += _incident_bincount(
                source_codes,
                destination_codes,
                gorder_gap64,
                cardinality,
            )
            accum["gorder_excess64"] += _incident_bincount(
                source_codes,
                destination_codes,
                gorder_excess64,
                cardinality,
            )
            for draw, label in enumerate(LAYOUT_RABBIT_DRAWS):
                draw_gorder_disagreement = (
                    bits[label] != gorder_bits
                ).astype(np.float64)
                draw_gap64 = (
                    bits[label] > 7).astype(np.float64)
                draw_excess64 = np.maximum(
                    bits[label].astype(np.int16) - 7, 0
                ).astype(np.float64)
                accum["rabbit_gorder_disagreement_draws"][
                    draw
                ] += _incident_bincount(
                    source_codes,
                    destination_codes,
                    draw_gorder_disagreement,
                    cardinality,
                )
                accum["rabbit_gap64_draws"][draw] += _incident_bincount(
                    source_codes,
                    destination_codes,
                    draw_gap64,
                    cardinality,
                )
                accum["rabbit_excess64_draws"][draw] += _incident_bincount(
                    source_codes,
                    destination_codes,
                    draw_excess64,
                    cardinality,
                )
            for pair_index, pair_values in enumerate(rabbit_pair_arrays):
                accum["rabbit_pair_disagreement"][
                    pair_index
                ] += _incident_bincount(
                    source_codes,
                    destination_codes,
                    pair_values,
                    cardinality,
                )
        enforce_rss_limit()
    if (
        undirected_edges != expected_undirected_edges
        or 2 * undirected_edges + self_loops
            != graph.directed_edges
    ):
        raise ValueError(
            "Undirected/self-loop edge identity changed during scan")
    layout_metrics = {}
    for label, state in layout_state.items():
        edges = max(1, state["edges"])
        layout_metrics[label] = {
            "mean_positive_bit_mloga":
                state["bit_sum"] / edges,
            "same_line_fraction":
                state["same_line_edges"] / edges,
            "gap_exceed_fraction": {
                key: value / edges
                for key, value in state[
                    "gap_exceed_count"].items()
            },
            "gap_excess_bit_mass_per_edge": {
                key: value / edges
                for key, value in state[
                    "gap_excess_bit_sum"].items()
            },
            "bit_sum": state["bit_sum"],
            "edges": state["edges"],
        }
    rabbit_bit_totals = np.asarray([
        layout_metrics[label]["bit_sum"]
        for label in LAYOUT_RABBIT_DRAWS
    ], dtype=np.float64)
    gorder_bit_total = layout_metrics[LAYOUT_GORDER]["bit_sum"]
    class_metrics = []
    predicate_by_scheme_code = {
        (predicate.scheme, predicate.code): predicate
        for predicate in CLASS_PREDICATES
    }
    for scheme, accum in class_accumulators.items():
        for code in range(accum["support"].size):
            predicate = predicate_by_scheme_code[(scheme, code)]
            support = accum["support"][code]
            denominator = max(1.0, support)
            rabbit_gorder_draws = (
                accum["rabbit_gorder_disagreement_draws"][:, code]
                / denominator
            )
            rabbit_pair_rates = (
                accum["rabbit_pair_disagreement"][:, code]
                / denominator
            )
            rabbit_gap_draws = (
                accum["rabbit_gap64_draws"][:, code]
                / denominator
            )
            rabbit_excess_draws = (
                accum["rabbit_excess64_draws"][:, code]
                / np.maximum(rabbit_bit_totals, 1.0)
            )
            rabbit_excess_density_draws = (
                accum["rabbit_excess64_draws"][:, code]
                / denominator
            )
            gorder_excess_density = (
                accum["gorder_excess64"][code] / denominator
            )
            class_metrics.append({
                "class_id": predicate.class_id,
                "class_name": predicate.name,
                "scheme": scheme,
                "code": code,
                "incident_edges": float(support),
                "support_fraction":
                    float(support / max(1, undirected_edges)),
                "rabbit_gorder_disagreement_draws":
                    rabbit_gorder_draws.tolist(),
                "rabbit_gorder_disagreement_range": {
                    "min": float(np.min(rabbit_gorder_draws)),
                    "median": float(np.median(rabbit_gorder_draws)),
                    "max": float(np.max(rabbit_gorder_draws)),
                },
                "rabbit_pair_disagreement_rates":
                    rabbit_pair_rates.tolist(),
                "rabbit_pair_disagreement_max":
                    float(np.max(rabbit_pair_rates)),
                "rabbit_beyond_gap64_bit_fraction_draws":
                    rabbit_gap_draws.tolist(),
                "rabbit_beyond_gap64_bit_fraction_range": {
                    "min": float(np.min(rabbit_gap_draws)),
                    "median": float(np.median(rabbit_gap_draws)),
                    "max": float(np.max(rabbit_gap_draws)),
                },
                "gorder_beyond_gap64_bit_fraction":
                    float(
                        accum["gorder_gap64"][code] / denominator),
                "rabbit_u64_draws":
                    rabbit_excess_draws.tolist(),
                "rabbit_u64_range": {
                    "min": float(np.min(rabbit_excess_draws)),
                    "median": float(np.median(rabbit_excess_draws)),
                    "max": float(np.max(rabbit_excess_draws)),
                },
                "gorder_u64":
                    float(
                        accum["gorder_excess64"][code]
                        / max(1.0, gorder_bit_total)
                    ),
                "rabbit_excess64_per_class_edge_range": {
                    "min": float(np.min(
                        rabbit_excess_density_draws)),
                    "median": float(np.median(
                        rabbit_excess_density_draws)),
                    "max": float(np.max(
                        rabbit_excess_density_draws)),
                },
                "gorder_excess64_per_class_edge":
                    float(gorder_excess_density),
                "h4_detector_work":
                    predicate.detector_work,
            })
    return {
        "schema": "forensic-multi-layout-metrics/v1",
        "measurement_mode": FORENSICS_MODE,
        "claim_eligible": False,
        "nodes": graph.nodes,
        "undirected_edges": undirected_edges,
        "self_loops_excluded": self_loops,
        "rabbit_pair_disagreement_fractions": (
            rabbit_pair_disagreement_edges
            / max(1, undirected_edges)
        ).tolist(),
        "rabbit_pair_disagreement_max":
            float(np.max(
                rabbit_pair_disagreement_edges
                / max(1, undirected_edges)
            )),
        "layout_metrics": layout_metrics,
        "class_metrics": class_metrics,
        "metric_definitions": {
            "gap_exceed_fraction":
                "strict position gap > threshold",
            "gap_excess_bit_mass_per_edge":
                "max(0, b_sigma(edge) - b(threshold)) per edge",
            "beyond_gap64_bit_fraction":
                "b_sigma(edge) > b(64)=7, equivalent to gap >= 128",
            "u64":
                "class excess above b(64), divided by global bit mass",
            "rabbit_gorder_disagreement":
                "three actual Rabbit-vs-Gorder rates; no synthetic layout",
            "rabbit_pair_disagreement":
                "maximum of three separately aggregated draw-pair rates",
        },
    }


def evaluate_class_gates(
    class_row: Mapping[str, Any],
) -> dict[str, Any]:
    margin = (
        float(class_row[
            "rabbit_gorder_disagreement_range"]["min"])
        - float(class_row["rabbit_pair_disagreement_max"])
    )
    h1 = bool(
        float(class_row["support_fraction"]) >= CLASS_SUPPORT_MIN)
    h2 = bool(
        h1
        and float(class_row[
            "rabbit_beyond_gap64_bit_fraction_range"]["min"])
            >= CLASS_GAP64_FRACTION_MIN
        and float(class_row[
            "gorder_beyond_gap64_bit_fraction"])
            >= CLASS_GAP64_FRACTION_MIN
        and margin >= CLASS_DIVERGENCE_MARGIN_MIN
    )
    h3 = bool(
        h2
        and float(class_row["rabbit_u64_range"]["min"])
            >= CLASS_HEADROOM_MIN
        and float(class_row["gorder_u64"]) >= CLASS_HEADROOM_MIN
    )
    return {
        "h1_pass": h1,
        "h2_pass": h2,
        "h3_pass": h3,
        "divergence_margin": margin,
    }


def nominate_class(
    per_graph_metrics: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    graph_rows = list(per_graph_metrics)
    if not graph_rows:
        raise ValueError("No discovery metrics were provided")
    observed_graphs = {
        str(row.get("graph")) for row in graph_rows
    }
    if (
        observed_graphs != set(DISCOVERY_GRAPHS)
        or len(graph_rows) != len(DISCOVERY_GRAPHS)
    ):
        raise ValueError(
            "Forensic nomination requires exactly the discovery cohort")
    h0_passes = int(sum(
        min(
            np.median([
                row["layout_metrics"][label][
                    "mean_positive_bit_mloga"]
                for label in LAYOUT_RABBIT_DRAWS
            ]),
            row["layout_metrics"][LAYOUT_GORDER][
                "mean_positive_bit_mloga"],
        )
        < row["layout_metrics"][LAYOUT_INPUT][
            "mean_positive_bit_mloga"]
        for row in graph_rows
    ))
    if h0_passes < 7:
        return {
            "schema": "forensic-nomination/v1",
            "h0_passes": h0_passes,
            "h0_required": 7,
            "status": "stop-h0",
            "nominee": None,
        }
    gate_rows: dict[int, list[dict[str, Any]]] = {}
    for graph_row in graph_rows:
        for class_row in graph_row["class_metrics"]:
            gates = evaluate_class_gates(class_row)
            margin = gates["divergence_margin"]
            h1 = gates["h1_pass"]
            h2 = gates["h2_pass"]
            h3 = gates["h3_pass"]
            score_component = (
                min(
                    class_row[
                        "rabbit_excess64_per_class_edge_range"]["min"],
                    class_row["gorder_excess64_per_class_edge"],
                )
                * max(0.0, margin)
            )
            gate_rows.setdefault(
                int(class_row["class_id"]), []
            ).append({
                **class_row,
                "graph": graph_row["graph"],
                "graph_type": graph_row["graph_type"],
                "divergence_margin": float(margin),
                "rabbit_pair_disagreement_max":
                    class_row["rabbit_pair_disagreement_max"],
                "h1_pass": h1,
                "h2_pass": h2,
                "h3_pass": h3,
                "score_component": float(score_component),
            })
    summaries = []
    ranked = []
    for class_id, rows in gate_rows.items():
        h1_rows = [row for row in rows if row["h1_pass"]]
        h2_rows = [row for row in rows if row["h2_pass"]]
        h3_rows = [row for row in rows if row["h3_pass"]]
        predicate = CLASS_PREDICATES[class_id]
        h1_types = {row["graph_type"] for row in h1_rows}
        h2_types = {row["graph_type"] for row in h2_rows}
        h3_types = {row["graph_type"] for row in h3_rows}
        h1_corpus_pass = bool(
            len(h1_rows) >= 3 and len(h1_types) >= 2)
        h2_corpus_pass = bool(
            h1_corpus_pass
            and len(h2_rows) >= 3
            and len(h2_types) >= 2
        )
        h3_corpus_pass = bool(
            h2_corpus_pass
            and len(h3_rows) >= 3
            and len(h3_types) >= 2
        )
        h4_pass = bool(predicate.detector_work == "O(m)")
        summary = {
            "class_id": class_id,
            "class_name": predicate.name,
            "detector_work": predicate.detector_work,
            "h1_corpus_pass": h1_corpus_pass,
            "h2_corpus_pass": h2_corpus_pass,
            "h3_corpus_pass": h3_corpus_pass,
            "h4_pass": h4_pass,
            "h1_graphs": sorted(
                row["graph"] for row in h1_rows),
            "h2_graphs": sorted(
                row["graph"] for row in h2_rows),
            "h3_graphs": sorted(
                row["graph"] for row in h3_rows),
            "per_graph": rows,
        }
        summaries.append(summary)
        if not (h3_corpus_pass and h4_pass):
            continue
        ranked.append({
            **summary,
            "qualifying_graphs": sorted(
                row["graph"] for row in h3_rows),
            "topology_types": sorted(h3_types),
            "nomination_score": sum(
                row["score_component"] for row in h3_rows
            ) / len(h3_rows),
        })
    ranked.sort(
        key=lambda row: (
            row["nomination_score"],
            -row["class_id"],
        ),
        reverse=True,
    )
    if ranked:
        status = "nominee"
    elif not any(row["h1_corpus_pass"] for row in summaries):
        status = "stop-h1"
    elif not any(row["h2_corpus_pass"] for row in summaries):
        status = "stop-h2"
    elif not any(row["h3_corpus_pass"] for row in summaries):
        status = "stop-h3"
    else:
        status = "stop-h4"
    return {
        "schema": "forensic-nomination/v1",
        "h0_passes": h0_passes,
        "h0_required": 7,
        "status": status,
        "nominee": ranked[0] if ranked else None,
        "eligible_class_count": len(ranked),
        "class_bank_sha256": CLASS_BANK_SHA256,
        "class_gate_summaries": summaries,
    }


def build_artifact_set(
    graph: str,
    graph_root: Path,
    mapping_root: Path,
    equivalence_root: Path,
) -> GraphArtifactSet:
    if graph not in GRAPH_TYPES:
        raise ValueError(f"Unknown forensic graph: {graph}")
    graph_dir = Path(graph_root) / graph
    mapping_dir = Path(mapping_root) / graph
    equivalence = (
        None if graph == "twitter7"
        else Path(equivalence_root)
        / graph / "9_csr.equivalence.json"
    )
    return GraphArtifactSet(
        graph=graph,
        graph_type=GRAPH_TYPES[graph],
        sg_path=graph_dir / f"{graph}.sg",
        dbg=MappingArtifact(
            LAYOUT_DBG,
            mapping_dir / "5.lo",
            mapping_dir / "5.json",
            0,
        ),
        rabbit_draws=tuple(
            MappingArtifact(
                label,
                mapping_dir / f"8_csr.draw{draw}.lo",
                mapping_dir / "8_csr.json",
                draw,
            )
            for draw, label in enumerate(LAYOUT_RABBIT_DRAWS)
        ),
        rabbit_alias=mapping_dir / "8_csr.lo",
        rabbit_sidecar=mapping_dir / "8_csr.json",
        gorder=MappingArtifact(
            LAYOUT_GORDER,
            mapping_dir / "9_csr.lo",
            mapping_dir / "9_csr.json",
            0,
        ),
        gorder_equivalence=equivalence,
    )


def artifact_input_paths(
    artifacts: GraphArtifactSet,
) -> tuple[Path, ...]:
    paths = [
        artifacts.sg_path,
        artifacts.dbg.path,
        artifacts.dbg.sidecar_path,
        artifacts.rabbit_alias,
        artifacts.rabbit_sidecar,
        artifacts.gorder.path,
        artifacts.gorder.sidecar_path,
        *[draw.path for draw in artifacts.rabbit_draws],
    ]
    if artifacts.gorder_equivalence is not None:
        paths.append(artifacts.gorder_equivalence)
    resolved = []
    for path in paths:
        if path is None:
            raise ValueError("Forensic artifact path is missing")
        resolved.append(Path(path).resolve())
    return tuple(resolved)


def freeze_artifact_manifest(
    artifacts: GraphArtifactSet,
) -> dict[str, Any]:
    paths = artifact_input_paths(artifacts)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Forensic artifacts are missing: " + " ".join(missing))
    return {
        "schema": "forensic-input-manifest/v1",
        "graph": artifacts.graph,
        "graph_type": artifacts.graph_type,
        "created_before_parse": True,
        "inputs": {
            str(path): {
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": file_sha256(path, use_cache=False),
            }
            for path in paths
        },
    }


def verify_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    rehash: bool,
) -> None:
    if manifest.get("schema") != "forensic-input-manifest/v1":
        raise ValueError("Unsupported forensic input manifest")
    for path_text, expected in manifest.get("inputs", {}).items():
        path = Path(path_text)
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["bytes"])
            or path.stat().st_mtime_ns != int(expected["mtime_ns"])
            or (
                rehash
                and file_sha256(path, use_cache=False)
                    != expected["sha256"]
            )
        ):
            raise RuntimeError(
                f"Forensic input changed after freeze: {path}")


def _stat_lockbox(
    artifacts: GraphArtifactSet,
) -> dict[str, Any]:
    paths = artifact_input_paths(artifacts)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Confirmation lockbox artifacts are missing: "
            + " ".join(missing)
        )
    return {
        "schema": "forensic-confirmation-lockbox/v1",
        "graph": artifacts.graph,
        "graph_type": artifacts.graph_type,
        "contents_unopened": True,
        "inputs": {
            str(path): {
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        },
    }


def _verify_stat_lockbox(lockbox: Mapping[str, Any]) -> None:
    if lockbox.get("schema") != "forensic-confirmation-lockbox/v1":
        raise ValueError("Unsupported confirmation lockbox")
    inputs = lockbox.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Confirmation lockbox has no sealed inputs")
    for path_text, expected in inputs.items():
        path = Path(path_text)
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["bytes"])
            or path.stat().st_mtime_ns != int(expected["mtime_ns"])
        ):
            raise RuntimeError(
                f"Confirmation lockbox changed: {path}")


def _read_sg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(struct.calcsize("<?qq"))
    if len(header) != struct.calcsize("<?qq"):
        raise ValueError(f"SG header is truncated: {path}")
    directed, directed_edges, nodes = struct.unpack("<?qq", header)
    if directed:
        raise ValueError("Route F requires symmetric SG inputs")
    return int(nodes), int(directed_edges // 2)


def _project_graph_seconds(
    artifacts: GraphArtifactSet,
    manifest: Mapping[str, Any],
) -> dict[str, float]:
    nodes, undirected_edges = _read_sg_dimensions(artifacts.sg_path)
    total_bytes = sum(
        int(record["bytes"])
        for record in manifest["inputs"].values()
    )
    mapping_bytes = sum(
        int(manifest["inputs"][str(path.resolve())]["bytes"])
        for path in (
            artifacts.dbg.path,
            artifacts.gorder.path,
            *[draw.path for draw in artifacts.rabbit_draws],
        )
    )
    hash_seconds = 2 * total_bytes / (180 * 1024**2)
    mapping_parse_seconds = mapping_bytes / (18 * 1024**2)
    edge_scan_seconds = undirected_edges / 350_000
    feature_pass_seconds = (2 * undirected_edges) / 1_500_000
    sampled_m3_seconds = min(180.0, nodes / 100_000)
    projected = (
        hash_seconds
        + mapping_parse_seconds
        + edge_scan_seconds
        + feature_pass_seconds
        + sampled_m3_seconds
    )
    sg_bytes = int(
        manifest["inputs"][str(artifacts.sg_path.resolve())]["bytes"])
    return {
        "nodes": nodes,
        "undirected_edges": undirected_edges,
        "input_bytes": total_bytes,
        "mapping_bytes": mapping_bytes,
        "hash_seconds": hash_seconds,
        "mapping_parse_seconds": mapping_parse_seconds,
        "edge_scan_seconds": edge_scan_seconds,
        "feature_pass_seconds": feature_pass_seconds,
        "sampled_m3_seconds": sampled_m3_seconds,
        "projected_seconds": projected,
        "projected_peak_bytes":
            estimated_peak_bytes(nodes, undirected_edges) + sg_bytes,
    }


def build_forensics_plan(
    *,
    graph_root: Path = DEFAULT_GRAPH_ROOT,
    mapping_root: Path = DEFAULT_MAPPING_ROOT,
    equivalence_root: Path = DEFAULT_EQUIVALENCE_ROOT,
    artifact_root: Path = DEFAULT_FORENSICS_ROOT,
    discovery_graphs: Iterable[str] = DISCOVERY_GRAPHS,
    confirmation_graphs: Iterable[str] = CONFIRMATION_GRAPHS,
    require_full_cohorts: bool = True,
    require_clean_implementation: bool = True,
) -> dict[str, Any]:
    repository_state = _repository_state(
        require_clean=require_clean_implementation)
    resolved_artifact_root = Path(artifact_root).resolve()
    resolved_mapping_root = Path(mapping_root).resolve()
    resolved_equivalence_root = Path(equivalence_root).resolve()
    if (
        resolved_artifact_root.is_relative_to(resolved_mapping_root)
        or resolved_artifact_root.is_relative_to(
            resolved_equivalence_root)
    ):
        raise ValueError(
            "Forensic output root cannot be inside campaign artifacts")
    discovery_names = tuple(discovery_graphs)
    confirmation_names = tuple(confirmation_graphs)
    if require_full_cohorts and (
        discovery_names != DISCOVERY_GRAPHS
        or confirmation_names != CONFIRMATION_GRAPHS
    ):
        raise ValueError("Route-F production cohorts changed")
    discovery_records = []
    total_projection = 0.0
    max_peak = 0
    for graph in discovery_names:
        artifacts = build_artifact_set(
            graph, graph_root, mapping_root, equivalence_root)
        manifest = freeze_artifact_manifest(artifacts)
        projection = _project_graph_seconds(artifacts, manifest)
        total_projection += projection["projected_seconds"]
        max_peak = max(
            max_peak, int(projection["projected_peak_bytes"]))
        discovery_records.append({
            "graph": graph,
            "graph_type": GRAPH_TYPES[graph],
            "manifest": manifest,
            "manifest_sha256": canonical_json_sha256(manifest),
            "projection": projection,
        })
    confirmation_records = []
    for graph in confirmation_names:
        artifacts = build_artifact_set(
            graph, graph_root, mapping_root, equivalence_root)
        lockbox = _stat_lockbox(artifacts)
        confirmation_records.append({
            "graph": graph,
            "graph_type": GRAPH_TYPES[graph],
            "lockbox": lockbox,
            "lockbox_sha256": canonical_json_sha256(lockbox),
        })
    if total_projection > FORENSICS_WALL_LIMIT_SECONDS:
        raise RuntimeError(
            "Projected discovery runtime exceeds the four-hour cap")
    if max_peak > FORENSICS_RSS_LIMIT_BYTES:
        raise RuntimeError(
            "Projected forensic memory exceeds the 56-GiB cap")
    plan = {
        "schema": FORENSICS_PLAN_SCHEMA,
        "measurement_mode": FORENSICS_MODE,
        "claim_eligible": False,
        "repository_state": repository_state,
        "implementation_scope": list(FORENSICS_IMPLEMENTATION_SCOPE),
        "implementation_sha256s": _implementation_sha256s(),
        "provenance_files": {
            relative: file_sha256(
                PROJECT_ROOT / relative, use_cache=False)
            for relative in FORENSICS_PROVENANCE_FILES
        },
        "class_spec": CLASS_SPEC,
        "class_bank_sha256": CLASS_BANK_SHA256,
        "nomination": {
            "maximum_classes": len(CLASS_PREDICATES),
            "nomination_count": 1,
            "h0_required_graphs": 7,
            "h1_min_graphs": 3,
            "h1_min_graph_types": 2,
            "support_min": CLASS_SUPPORT_MIN,
            "beyond_gap64_fraction_min":
                CLASS_GAP64_FRACTION_MIN,
            "divergence_margin_min":
                CLASS_DIVERGENCE_MARGIN_MIN,
            "u64_min": CLASS_HEADROOM_MIN,
        },
        "metrics": {
            "m1": "exact positive-bit MLogA per undirected non-loop edge",
            "m2": {
                "gap_thresholds": list(GAP_THRESHOLDS),
                "property_bytes": 8,
                "cache_line_bytes": 64,
            },
            "m3": {
                "diagnostic_only": True,
                "sample_limit": M3_SAMPLE_LIMIT,
                "sample_seed": FORENSICS_SAMPLE_SEED,
                "bootstrap_buckets": M3_BOOTSTRAP_BUCKETS,
                "bootstrap_replicates": M3_BOOTSTRAP_REPLICATES,
            },
            "m4": "actual Rabbit-draw/Gorder bit-bin disagreement",
            "m5": "three Rabbit draws and three pairwise controls",
            "m6": "class excess above b(64), falsifier-only",
        },
        "resource_policy": {
            "wall_seconds": FORENSICS_WALL_LIMIT_SECONDS,
            "rss_bytes": FORENSICS_RSS_LIMIT_BYTES,
            "projected_discovery_seconds": total_projection,
            "projected_peak_bytes": max_peak,
            "projection_model":
                "2x SHA at 180MiB/s + LO parse at 18MiB/s + "
                "metrics at 350k edges/s + feature queries at "
                "1.5M directed edges/s + bounded M3",
        },
        "paths": {
            "graph_root": str(Path(graph_root).resolve()),
            "mapping_root": str(Path(mapping_root).resolve()),
            "equivalence_root": str(Path(equivalence_root).resolve()),
            "artifact_root": str(resolved_artifact_root),
        },
        "discovery": discovery_records,
        "confirmation_lockbox": confirmation_records,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return plan


def write_forensics_plan(
    plan: Mapping[str, Any],
    artifact_root: Path = DEFAULT_FORENSICS_ROOT,
    *,
    refreeze: bool = False,
) -> Path:
    path = Path(artifact_root) / "plan.json"
    if path.is_file() and not refreeze:
        existing = json.loads(path.read_text())
        if (
            existing.get("plan_sha256")
                != plan.get("plan_sha256")
            or canonical_json_sha256({
                key: value for key, value in existing.items()
                if key != "plan_sha256"
            }) != existing.get("plan_sha256")
        ):
            raise RuntimeError(
                "Frozen forensics plan changed; "
                "refreeze only after review"
            )
    _atomic_json(plan, path)
    return path


def _load_bound_plan(
    path: Path,
    *,
    require_clean_implementation: bool,
) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    recorded = plan.pop("plan_sha256", None)
    if (
        plan.get("schema") != FORENSICS_PLAN_SCHEMA
        or recorded != canonical_json_sha256(plan)
    ):
        raise ValueError("Forensics plan binding is invalid")
    plan["plan_sha256"] = recorded
    current_state = _repository_state(
        require_clean=require_clean_implementation)
    if (
        current_state["relevant_diff_sha256"]
            != plan["repository_state"]["relevant_diff_sha256"]
        or current_state["relevant_untracked"]
            != plan["repository_state"]["relevant_untracked"]
        or plan.get("implementation_sha256s")
            != _implementation_sha256s()
    ):
        raise RuntimeError(
            "Route-F implementation changed after plan review")
    if plan["class_bank_sha256"] != CLASS_BANK_SHA256:
        raise RuntimeError("Route-F class specification changed")
    for record in plan["confirmation_lockbox"]:
        if (
            record["lockbox_sha256"]
            != canonical_json_sha256(record["lockbox"])
        ):
            raise ValueError("Confirmation lockbox digest changed")
        _verify_stat_lockbox(record["lockbox"])
    return plan


def validate_artifact_identity(
    artifacts: GraphArtifactSet,
    graph: SerializedGraphMMap,
    input_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    verify_artifact_manifest(input_manifest, rehash=False)
    dbg_sidecar = json.loads(
        artifacts.dbg.sidecar_path.read_text())
    rabbit_sidecar = json.loads(
        artifacts.rabbit_sidecar.read_text())
    gorder_sidecar = json.loads(
        artifacts.gorder.sidecar_path.read_text())
    expected_dimensions = {
        "nodes": graph.nodes,
        "edges": graph.directed_edges,
        "directed": graph.layout.directed,
    }
    checks = (
        (dbg_sidecar, "5", 1),
        (rabbit_sidecar, "8:csr", 3),
        (gorder_sidecar, "9:csr", 1),
    )
    for sidecar, algorithm, draws in checks:
        if sidecar.get("schema") != "reorder_meta/v4":
            raise ValueError("Forensics require historical sidecar v4")
        if sidecar.get("algo_key") != algorithm:
            raise ValueError(
                f"Forensic sidecar algorithm mismatch: {algorithm}")
        if sidecar.get("graph") != artifacts.graph:
            raise ValueError("Forensic sidecar graph mismatch")
        if sidecar.get("graph_info") != expected_dimensions:
            raise ValueError("Forensic sidecar dimensions changed")
        if sidecar.get("mapping_draw_count") != draws:
            raise ValueError("Forensic sidecar draw count changed")
    manifest_inputs = input_manifest["inputs"]
    rabbit_alias_sha = manifest_inputs[
        str(artifacts.rabbit_alias.resolve())]["sha256"]
    rabbit_draw_sha256s = [
        manifest_inputs[str(draw.path.resolve())]["sha256"]
        for draw in artifacts.rabbit_draws
    ]
    rabbit_draw0_sha = rabbit_draw_sha256s[0]
    if rabbit_alias_sha != rabbit_draw0_sha:
        raise ValueError(
            "Rabbit selected alias differs from draw 0")
    rabbit_draws = rabbit_sidecar.get("mapping_draws")
    if (
        not isinstance(rabbit_draws, list)
        or len(rabbit_draws) != 3
        or rabbit_sidecar.get("selected_draw") != 0
        or rabbit_sidecar.get("lo_path") != "8_csr.lo"
    ):
        raise ValueError("Rabbit draw metadata changed")
    for draw, record in enumerate(rabbit_draws):
        if (
            record.get("draw") != draw
            or record.get("path") != f"8_csr.draw{draw}.lo"
        ):
            raise ValueError("Rabbit draw path identity changed")
    if (
        dbg_sidecar.get("mapping_draws", [{}])[0].get("path") != "5.lo"
        or gorder_sidecar.get("mapping_draws", [{}])[0].get("path")
            != "9_csr.lo"
        or dbg_sidecar.get("selected_draw") != 0
        or gorder_sidecar.get("selected_draw") != 0
    ):
        raise ValueError("Single-draw mapping identity changed")
    gorder_equivalence = None
    if artifacts.gorder_equivalence is not None:
        gorder_equivalence = json.loads(
            artifacts.gorder_equivalence.read_text())
        if (
            gorder_equivalence.get("schema")
                != "mapping_equivalence/v1"
            or gorder_equivalence.get("graph") != artifacts.graph
            or gorder_equivalence.get("algorithm") != "9:csr"
            or gorder_equivalence.get("equal") is not True
            or Path(gorder_equivalence.get("promoted_path", "")).resolve()
                != artifacts.gorder.path.resolve()
        ):
            raise ValueError(
                "Gorder equivalence evidence is invalid")
        promoted_path = artifacts.gorder.path.resolve()
        if (
            int(gorder_equivalence.get("promoted_bytes", -1))
                != manifest_inputs[str(promoted_path)]["bytes"]
            or int(gorder_equivalence.get("live_bytes", -1))
                != int(gorder_equivalence.get("promoted_bytes", -1))
            or gorder_sidecar.get("mapping_origin")
                != "promoted-mapping-equivalent-legacy-gorder"
        ):
            raise ValueError(
                "Gorder equivalence is not bound to current bytes")
        checked_at = datetime.fromisoformat(
            str(gorder_equivalence.get("checked_at")).replace("Z", "+00:00")
        ).timestamp()
        if promoted_path.stat().st_mtime > checked_at:
            raise ValueError(
                "Gorder promoted mapping is newer than its receipt")
    elif (
        artifacts.graph != "twitter7"
        or gorder_sidecar.get("mapping_origin")
            == "promoted-mapping-equivalent-legacy-gorder"
    ):
        raise ValueError(
            "Direct Gorder mapping lacks required identity evidence")
    return {
        "schema": "forensic-artifact-identity/v1",
        "graph": artifacts.graph,
        "graph_type": artifacts.graph_type,
        "legacy_forensic": True,
        "dimensions": expected_dimensions,
        "rabbit_alias_sha256": rabbit_alias_sha,
        "rabbit_draw0_sha256": rabbit_draw0_sha,
        "rabbit_draw_sha256s": rabbit_draw_sha256s,
        "rabbit_draw_unique_count":
            len(set(rabbit_draw_sha256s)),
        "rabbit_draws_distinct":
            len(set(rabbit_draw_sha256s)) == 3,
        "gorder_mapping_origin":
            gorder_sidecar.get("mapping_origin", "direct-generation"),
        "gorder_equivalence_evidence": (
            "direct-generation"
            if artifacts.gorder_equivalence is None
            else "receipt-only; live scratch deleted by campaign policy"
        ),
        "gorder_equivalence": gorder_equivalence,
        "inputs": dict(manifest_inputs),
    }


def estimated_peak_bytes(nodes: int, undirected_edges: int) -> int:
    position_arrays = 7 * 4 * nodes
    vertex_arrays = 24 * nodes
    retained_edges = 8 * undirected_edges
    mapping_parse_transient = 12 * nodes
    safety = 2 * 1024**3
    return (
        position_arrays
        + vertex_arrays
        + retained_edges
        + mapping_parse_transient
        + safety
    )


def _check_deadline(deadline_monotonic: float) -> None:
    if time.monotonic() >= deadline_monotonic:
        raise TimeoutError("Forensic wall-clock cap reached")


def _check_projected_memory(
    nodes: int,
    undirected_edges: int,
    rss_limit_bytes: int,
) -> None:
    projected = current_rss_bytes() + estimated_peak_bytes(
        nodes, undirected_edges)
    if projected > rss_limit_bytes:
        raise MemoryError(
            "Forensic projected memory exceeds the RSS ceiling: "
            f"{projected} > {rss_limit_bytes}"
        )


def analyze_graph_artifacts(
    artifacts: GraphArtifactSet,
    *,
    input_manifest: Mapping[str, Any] | None = None,
    deadline_monotonic: float | None = None,
    rss_limit_bytes: int = FORENSICS_RSS_LIMIT_BYTES,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = (
        deadline_monotonic
        if deadline_monotonic is not None
        else started + FORENSICS_WALL_LIMIT_SECONDS
    )
    manifest = (
        dict(input_manifest)
        if input_manifest is not None
        else freeze_artifact_manifest(artifacts)
    )
    verify_artifact_manifest(manifest, rehash=True)
    _check_deadline(deadline)
    with SerializedGraphMMap(artifacts.sg_path) as graph:
        if graph.layout.directed:
            raise ValueError("Route F requires symmetric SG inputs")
        _check_projected_memory(
            graph.nodes, graph.undirected_edges, rss_limit_bytes)
        identity = validate_artifact_identity(
            artifacts, graph, manifest)
        verify_artifact_manifest(manifest, rehash=False)
        _check_deadline(deadline)
        validate_int32_permutation(
            graph.org_ids,
            expected_size=graph.nodes,
            label="serialized graph org_ids",
        )
        degrees = graph.degrees()
        input_positions = np.arange(
            graph.nodes, dtype=np.int32)
        source_positions = np.asarray(
            graph.org_ids, dtype=np.int32).copy()
        layout_metadata = {
            LAYOUT_INPUT: {
                "path": str(artifacts.sg_path),
                "composed_sg_to_new_fingerprint":
                    composed_permutation_fingerprint(input_positions),
            },
            LAYOUT_SOURCE: {
                "path": str(artifacts.sg_path),
                "definition": "position_by_sg=org_ids[sg_id]",
                "composed_sg_to_new_fingerprint":
                    composed_permutation_fingerprint(source_positions),
            },
        }
        dbg_positions, dbg_metadata = load_text_mapping_positions(
            artifacts.dbg.path,
            nodes=graph.nodes,
            org_ids=graph.org_ids,
        )
        layout_metadata[LAYOUT_DBG] = dbg_metadata
        rabbit_positions = {}
        for artifact in artifacts.rabbit_draws:
            positions, metadata = load_text_mapping_positions(
                artifact.path,
                nodes=graph.nodes,
                org_ids=graph.org_ids,
            )
            rabbit_positions[artifact.label] = positions
            layout_metadata[artifact.label] = metadata
        gorder_positions, gorder_metadata = load_text_mapping_positions(
            artifacts.gorder.path,
            nodes=graph.nodes,
            org_ids=graph.org_ids,
        )
        layout_metadata[LAYOUT_GORDER] = gorder_metadata
        verify_artifact_manifest(manifest, rehash=False)
        _check_deadline(deadline)
        layouts = {
            LAYOUT_INPUT: input_positions,
            LAYOUT_SOURCE: source_positions,
            LAYOUT_DBG: dbg_positions,
            **rabbit_positions,
            LAYOUT_GORDER: gorder_positions,
        }
        dbg_validation = validate_dbg_semantics(
            layouts[LAYOUT_DBG], degrees)
        (
            feature_codes,
            feature_metadata,
            edge_chunks,
            feature_degrees,
        ) = (
            compute_vertex_feature_codes(
                graph,
                degrees,
                retain_undirected_edges=True,
            )
        )
        assert edge_chunks is not None
        verify_artifact_manifest(manifest, rehash=False)
        _check_deadline(deadline)
        metrics = scan_multi_layout_metrics(
            graph,
            layouts,
            feature_codes,
            self_loops=int(
                feature_metadata["self_loop_entries_observed"]),
            expected_undirected_edges=int(
                feature_metadata["undirected_edges_scanned"]),
            edge_chunks=edge_chunks,
        )
        del edge_chunks
        _check_deadline(deadline)
        m3 = sampled_distinct_lines_per_degree(
            graph,
            feature_degrees,
            layouts,
            feature_codes["degree"],
        )
        verify_artifact_manifest(manifest, rehash=False)
        _check_deadline(deadline)
        post_layout_fingerprints = {
            label: composed_permutation_fingerprint(positions)
            for label, positions in layouts.items()
        }
        for label, fingerprint in post_layout_fingerprints.items():
            if (
                layout_metadata[label][
                    "composed_sg_to_new_fingerprint"]
                != fingerprint
            ):
                raise RuntimeError(
                    f"In-memory forensic layout changed: {label}")
        result = {
            **metrics,
            "graph": artifacts.graph,
            "graph_type": artifacts.graph_type,
            "artifact_identity": identity,
            "layout_artifacts": layout_metadata,
            "post_layout_fingerprints":
                post_layout_fingerprints,
            "dbg_validation": dbg_validation,
            "feature_metadata": feature_metadata,
            "m3_distinct_lines": m3,
            "class_bank_sha256": CLASS_BANK_SHA256,
            "analytic_random_same_line_null":
                analytic_random_same_line_null(graph.nodes),
            "elapsed_seconds": time.monotonic() - started,
            "current_rss_bytes": current_rss_bytes(),
            "peak_rss_bytes": peak_rss_bytes(),
        }
        verify_artifact_manifest(manifest, rehash=True)
        result["post_input_verification"] = "pass"
        if result["peak_rss_bytes"] > rss_limit_bytes:
            raise MemoryError("Forensic graph analysis exceeded RSS cap")
        return result


def execute_forensics_discovery(
    plan_path: Path,
    *,
    resume: bool = True,
    require_clean_implementation: bool = True,
) -> Path:
    plan = _load_bound_plan(
        Path(plan_path),
        require_clean_implementation=require_clean_implementation,
    )
    started = time.monotonic()
    artifact_root = (
        Path(plan["paths"]["artifact_root"])
        / plan["plan_sha256"]
    )
    per_graph_dir = artifact_root / "per_graph"
    prior_consumed = 0.0
    if resume:
        for record in plan["discovery"]:
            path = per_graph_dir / f"{record['graph']}.json"
            if path.is_file():
                existing = json.loads(path.read_text())
                if (
                    existing.get("schema") == FORENSICS_RESULT_SCHEMA
                    and existing.get("plan_sha256")
                        == plan["plan_sha256"]
                ):
                    prior_consumed += float(
                        existing.get("elapsed_seconds", 0.0))
    remaining_budget = (
        int(plan["resource_policy"]["wall_seconds"])
        - prior_consumed
    )
    if remaining_budget <= 0:
        error = TimeoutError(
            "Cumulative forensic wall-clock cap is exhausted")
        _atomic_json({
            "schema": "graphbrew-mapping-forensics-discovery/v1",
            "plan": str(Path(plan_path).resolve()),
            "plan_sha256": plan["plan_sha256"],
            "measurement_mode": FORENSICS_MODE,
            "claim_eligible": False,
            "status": "negative-result",
            "negative_result": {
                "failed_gate": "wall-clock",
                "graph": None,
                "error": str(error),
                "statement": "Route F stopped at a resource gate.",
            },
            "consumed_seconds": prior_consumed,
            "peak_rss_bytes": peak_rss_bytes(),
            "completed_graphs": [],
            "result_paths": [],
            "confirmation_lockbox_unopened": True,
        }, artifact_root / "discovery_summary.json")
        raise error
    deadline = started + remaining_budget
    rows = []
    projected_completed = 0.0
    executed_projected = 0.0
    executed_elapsed = 0.0
    resumed_graphs = []
    projected_total = sum(
        float(record["projection"]["projected_seconds"])
        for record in plan["discovery"]
    )
    for index, record in enumerate(plan["discovery"]):
        graph = record["graph"]
        result_path = per_graph_dir / f"{graph}.json"
        try:
            _check_deadline(deadline)
            if result_path.is_file() and resume:
                existing = json.loads(result_path.read_text())
                if (
                    existing.get("schema") != FORENSICS_RESULT_SCHEMA
                    or existing.get("plan_sha256") != plan["plan_sha256"]
                    or existing.get("input_manifest_sha256")
                        != record["manifest_sha256"]
                    or existing.get("post_input_verification") != "pass"
                ):
                    raise RuntimeError(
                        f"Stale forensic result requires no-resume: {graph}")
                rows.append(existing)
                resumed_graphs.append(graph)
            else:
                if (
                    record["manifest_sha256"]
                    != canonical_json_sha256(record["manifest"])
                ):
                    raise ValueError(
                        f"Discovery manifest digest changed: {graph}")
                artifacts = build_artifact_set(
                    graph,
                    Path(plan["paths"]["graph_root"]),
                    Path(plan["paths"]["mapping_root"]),
                    Path(plan["paths"]["equivalence_root"]),
                )
                result = analyze_graph_artifacts(
                    artifacts,
                    input_manifest=record["manifest"],
                    deadline_monotonic=deadline,
                    rss_limit_bytes=int(
                        plan["resource_policy"]["rss_bytes"]),
                )
                result.update({
                    "schema": FORENSICS_RESULT_SCHEMA,
                    "plan_sha256": plan["plan_sha256"],
                    "input_manifest_sha256":
                        record["manifest_sha256"],
                })
                _atomic_json(result, result_path)
                rows.append(result)
                executed_projected += float(
                    record["projection"]["projected_seconds"])
                executed_elapsed += float(result["elapsed_seconds"])
            projected_completed += float(
                record["projection"]["projected_seconds"])
            elapsed = time.monotonic() - started
            if executed_projected > 0 and index + 1 < len(
                plan["discovery"]
            ):
                slowdown = max(
                    1.0, executed_elapsed / executed_projected)
                remaining = projected_total - projected_completed
                if elapsed + slowdown * remaining > int(
                    plan["resource_policy"]["wall_seconds"]
                ):
                    raise TimeoutError(
                        "Observed forensic throughput projects beyond "
                        "the four-hour cap"
                    )
        except (OSError, ValueError, RuntimeError, MemoryError, TimeoutError) as error:
            failed_gate = (
                "wall-clock" if isinstance(error, TimeoutError)
                else "rss" if isinstance(error, MemoryError)
                else "artifact"
            )
            failure = {
                "schema": "graphbrew-mapping-forensics-discovery/v1",
                "plan": str(Path(plan_path).resolve()),
                "plan_sha256": plan["plan_sha256"],
                "measurement_mode": FORENSICS_MODE,
                "claim_eligible": False,
                "status": "negative-result",
                "negative_result": {
                    "failed_gate": failed_gate,
                    "graph": graph,
                    "error": str(error),
                    "statement": (
                        "Route F stopped at an artifact or resource gate."
                    ),
                },
                "consumed_seconds": sum(
                    float(row.get("elapsed_seconds", 0.0))
                    for row in rows
                ),
                "peak_rss_bytes": peak_rss_bytes(),
                "completed_graphs": [
                    row["graph"] for row in rows
                ],
                "result_paths": [
                    str((
                        per_graph_dir / f"{row['graph']}.json"
                    ).resolve())
                    for row in rows
                ],
                "confirmation_lockbox_unopened": True,
            }
            _atomic_json(
                failure, artifact_root / "discovery_summary.json")
            raise
    nomination = nominate_class(rows)
    summary = {
        "schema": "graphbrew-mapping-forensics-discovery/v1",
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": plan["plan_sha256"],
        "measurement_mode": FORENSICS_MODE,
        "claim_eligible": False,
        "graphs": [row["graph"] for row in rows],
        "elapsed_seconds": time.monotonic() - started,
        "consumed_seconds": sum(
            float(row.get("elapsed_seconds", 0.0))
            for row in rows
        ),
        "resumed_graphs": resumed_graphs,
        "rabbit_draw_control": {
            row["graph"]: {
                "draws_distinct": row["artifact_identity"][
                    "rabbit_draws_distinct"],
                "draw_unique_count": row["artifact_identity"][
                    "rabbit_draw_unique_count"],
                "pair_disagreement_max": row[
                    "rabbit_pair_disagreement_max"],
            }
            for row in rows
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "nomination": nomination,
        "thresholds_sha256":
            canonical_json_sha256(plan["nomination"]),
        "result_paths": [
            str((per_graph_dir / f"{row['graph']}.json").resolve())
            for row in rows
        ],
        "confirmation_lockbox_unopened": True,
        "status": (
            "signature-pending-novelty-review"
            if nomination["status"] == "nominee"
            else "negative-result"
        ),
    }
    if summary["status"] == "negative-result":
        summary["negative_result"] = {
            "failed_gate": nomination["status"],
            "statement": (
                "No prevalent, recoverable, novelty-safe shared "
                "Rabbit/Gorder forensic signature qualified."
            ),
        }
    else:
        summary["signature_template"] = {
            "schema": "graphbrew-forensic-signature/v1",
            "plan_sha256": plan["plan_sha256"],
            "discovery_decision_sha256":
                "filled-from-discovery-summary",
            "class_bank_sha256": CLASS_BANK_SHA256,
            "thresholds_sha256":
                canonical_json_sha256(plan["nomination"]),
            "class_id": nomination["nominee"]["class_id"],
            "class_name": nomination["nominee"]["class_name"],
            "mechanism_spec_sha256": "required-after-novelty-review",
            "independent_review_1": "pending",
            "independent_review_2": "pending",
        }
    summary["discovery_decision_sha256"] = (
        discovery_decision_sha256(summary)
    )
    if "signature_template" in summary:
        summary["signature_template"]["discovery_decision_sha256"] = (
            summary["discovery_decision_sha256"]
        )
    output = artifact_root / "discovery_summary.json"
    _atomic_json(summary, output)
    return output


def _validate_frozen_signature(
    signature: Mapping[str, Any],
    plan: Mapping[str, Any],
    discovery_summary: Mapping[str, Any],
) -> int:
    if (
        signature.get("schema")
            != "graphbrew-forensic-signature/v1"
        or signature.get("plan_sha256") != plan["plan_sha256"]
        or signature.get("discovery_decision_sha256")
            != discovery_decision_sha256(discovery_summary)
        or signature.get("class_bank_sha256") != CLASS_BANK_SHA256
        or signature.get("thresholds_sha256")
            != canonical_json_sha256(plan["nomination"])
        or signature.get("independent_review_1") != "approved"
        or signature.get("independent_review_2") != "approved"
    ):
        raise ValueError("Frozen forensic signature is not authorized")
    if (
        discovery_summary.get("schema")
            != "graphbrew-mapping-forensics-discovery/v1"
        or discovery_summary.get("plan_sha256") != plan["plan_sha256"]
        or discovery_summary.get("status")
            != "signature-pending-novelty-review"
    ):
        raise ValueError(
            "Discovery did not authorize confirmation")
    class_id = int(signature.get("class_id", -1))
    if class_id < 0 or class_id >= len(CLASS_PREDICATES):
        raise ValueError("Frozen forensic class ID is invalid")
    if signature.get("class_name") != CLASS_PREDICATES[class_id].name:
        raise ValueError("Frozen forensic class name changed")
    nominee = discovery_summary.get("nomination", {}).get("nominee")
    if (
        not isinstance(nominee, dict)
        or int(nominee.get("class_id", -1)) != class_id
    ):
        raise ValueError(
            "Signature does not match the discovery nominee")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(signature.get("mechanism_spec_sha256", "")),
    ):
        raise ValueError("Frozen mechanism specification is missing")
    return class_id


def execute_forensics_confirmation(
    plan_path: Path,
    signature_path: Path,
    *,
    resume: bool = True,
    require_clean_implementation: bool = True,
) -> Path:
    plan = _load_bound_plan(
        Path(plan_path),
        require_clean_implementation=require_clean_implementation,
    )
    artifact_root = (
        Path(plan["paths"]["artifact_root"])
        / plan["plan_sha256"]
    )
    discovery_path = artifact_root / "discovery_summary.json"
    if not discovery_path.is_file():
        raise FileNotFoundError(
            "Confirmation requires a completed discovery summary")
    discovery_summary = json.loads(discovery_path.read_text())
    signature = json.loads(Path(signature_path).read_text())
    class_id = _validate_frozen_signature(
        signature, plan, discovery_summary)
    started = time.monotonic()
    per_graph_dir = artifact_root / "confirmation"
    signature_sha = file_sha256(
        signature_path, use_cache=False)
    prior_consumed = 0.0
    resumed_graphs = []
    if resume:
        for record in plan["confirmation_lockbox"]:
            path = per_graph_dir / f"{record['graph']}.json"
            if path.is_file():
                existing = json.loads(path.read_text())
                if (
                    existing.get("schema") == FORENSICS_RESULT_SCHEMA
                    and existing.get("plan_sha256")
                        == plan["plan_sha256"]
                    and existing.get("signature_sha256") == signature_sha
                ):
                    prior_consumed += float(
                        existing.get("elapsed_seconds", 0.0))
    remaining_budget = (
        int(plan["resource_policy"]["wall_seconds"])
        - prior_consumed
    )
    if remaining_budget <= 0:
        raise TimeoutError(
            "Cumulative confirmation wall-clock cap is exhausted")
    deadline = started + remaining_budget
    rows = []
    for record in plan["confirmation_lockbox"]:
        graph = record["graph"]
        result_path = per_graph_dir / f"{graph}.json"
        if result_path.is_file() and resume:
            existing = json.loads(result_path.read_text())
            if (
                existing.get("schema") != FORENSICS_RESULT_SCHEMA
                or existing.get("plan_sha256") != plan["plan_sha256"]
                or existing.get("signature_sha256")
                    != signature_sha
            ):
                raise RuntimeError(
                    f"Stale confirmation result: {graph}")
            rows.append(existing)
            resumed_graphs.append(graph)
            continue
        artifacts = build_artifact_set(
            graph,
            Path(plan["paths"]["graph_root"]),
            Path(plan["paths"]["mapping_root"]),
            Path(plan["paths"]["equivalence_root"]),
        )
        manifest = freeze_artifact_manifest(artifacts)
        result = analyze_graph_artifacts(
            artifacts,
            input_manifest=manifest,
            deadline_monotonic=deadline,
            rss_limit_bytes=int(
                plan["resource_policy"]["rss_bytes"]),
        )
        result.update({
            "schema": FORENSICS_RESULT_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "signature_sha256":
                signature_sha,
        })
        _atomic_json(result, result_path)
        rows.append(result)
    confirmations = []
    for row in rows:
        matches = [
            item for item in row["class_metrics"]
            if int(item["class_id"]) == class_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Confirmation class row missing or duplicated: "
                f"{row['graph']}/{class_id}"
            )
        class_row = matches[0]
        gates = evaluate_class_gates(class_row)
        confirmations.append({
            "graph": row["graph"],
            "graph_type": row["graph_type"],
            **gates,
            "h4_pass": (
                CLASS_PREDICATES[class_id].detector_work == "O(m)"
            ),
            "class_metrics": class_row,
        })
    pass_count = sum(
        item["h1_pass"]
        and item["h2_pass"]
        and item["h3_pass"]
        and item["h4_pass"]
        for item in confirmations
    )
    output_payload = {
        "schema": "graphbrew-mapping-forensics-confirmation/v1",
        "plan_sha256": plan["plan_sha256"],
        "signature": str(Path(signature_path).resolve()),
        "signature_sha256":
            signature_sha,
        "class_id": class_id,
        "class_name": CLASS_PREDICATES[class_id].name,
        "confirmation": confirmations,
        "required_passes": 2,
        "observed_passes": int(pass_count),
        "status": (
            "n4-replicated" if pass_count >= 2
            else "negative-result"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "consumed_seconds": sum(
            float(row.get("elapsed_seconds", 0.0))
            for row in rows
        ),
        "resumed_graphs": resumed_graphs,
        "peak_rss_bytes": peak_rss_bytes(),
        "measurement_mode": FORENSICS_MODE,
        "claim_eligible": False,
    }
    output = artifact_root / "confirmation_summary.json"
    _atomic_json(output_payload, output)
    return output


__all__ = [
    "CLASS_BANK_SHA256",
    "CLASS_PREDICATES",
    "CONFIRMATION_GRAPHS",
    "DISCOVERY_GRAPHS",
    "FORENSICS_MODE",
    "FORENSICS_PLAN_SCHEMA",
    "FORENSICS_RSS_LIMIT_BYTES",
    "FORENSICS_SCHEMA",
    "FORENSICS_WALL_LIMIT_SECONDS",
    "GRAPH_TYPES",
    "GraphArtifactSet",
    "MappingArtifact",
    "SGLayout",
    "SerializedGraphMMap",
    "analyze_graph_artifacts",
    "artifact_input_paths",
    "array_sha256",
    "build_artifact_set",
    "build_forensics_plan",
    "canonical_json_sha256",
    "composed_permutation_fingerprint",
    "compute_vertex_feature_codes",
    "dbg_bucket_codes",
    "enforce_rss_limit",
    "estimated_peak_bytes",
    "execute_forensics_confirmation",
    "execute_forensics_discovery",
    "freeze_artifact_manifest",
    "frozen_class_predicates",
    "load_text_mapping_positions",
    "nominate_class",
    "positive_bit_cost",
    "sampled_distinct_lines_per_degree",
    "scan_multi_layout_metrics",
    "validate_artifact_identity",
    "validate_dbg_semantics",
    "validate_int32_permutation",
    "verify_artifact_manifest",
    "write_forensics_plan",
]
