#!/usr/bin/env python3
"""Generate GraphBrew's public explanatory figures from one checked example."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
CATALOG = FIGURES / "reordering"
EDITABLE = FIGURES / "editable"
FIXTURE = FIGURES / "graphbrew-running-example.json"
CAPTURE = FIGURES / "catalog-capture.json"
EXAMPLE_EDGE_LIST = FIGURES / "data/graphbrew-running-example.el"
EXAMPLE_MAP = FIGURES / "data/graphbrew-running-example.lo"
LOW_REUSE_EVIDENCE = ROOT / "docs/allkernel-lowreuse-evidence.json"

INK = "#27313A"
PAGE = "#FFFFFF"
NEUTRAL = "#F8F6EC"
BLUE = "#EDF5FF"
GREEN = "#E7F7EA"
AMBER = "#FFF0D8"
ROSE = "#F7DEDC"
VIOLET = "#EEE9FF"
ACTION = "#1769C2"
MUTED = "#9AA3AD"
COMMUNITY_B = "#B45309"

GRAPH_POSITIONS = {
    0: (80, 155),
    3: (155, 90),
    5: (155, 220),
    8: (245, 155),
    1: (335, 90),
    2: (415, 155),
    4: (370, 235),
    6: (505, 220),
    7: (505, 90),
}

CATALOG_NAMES = {
    0: "ORIGINAL",
    1: "RANDOM",
    2: "SORT",
    3: "HUBSORT",
    4: "HUBCLUSTER",
    5: "DBG",
    6: "HUBSORTDBG",
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    9: "GORDER",
    10: "CORDER",
    11: "RCM",
    12: "GraphBrewOrder",
    13: "MAP",
    14: "AdaptiveOrder",
    15: "LeidenOrder",
    16: "GoGraphOrder",
}

CATALOG_SLUGS = {
    0: "original",
    1: "random",
    2: "sort",
    3: "hubsort",
    4: "hubcluster",
    5: "dbg",
    6: "hubsortdbg",
    7: "hubclusterdbg",
    8: "rabbitorder",
    9: "gorder",
    10: "corder",
    11: "rcm",
    12: "graphbreworder",
    13: "map",
    14: "adaptiveorder",
    15: "leidenorder",
    16: "gographorder",
}

CATALOG_FAMILIES = {
    0: "identity",
    1: "random",
    2: "degree",
    3: "degree",
    4: "degree",
    5: "buckets",
    6: "buckets",
    7: "buckets",
    8: "community",
    9: "window",
    10: "hotcold",
    11: "bandwidth",
    12: "composed",
    13: "map",
    14: "selector",
    15: "community",
    16: "directed",
}

CATALOG_COPY = {
    0: (
        "Identity permutation",
        ["new_id = old_id", "No analysis or movement"],
        "Topology and memory order are identical.",
    ),
    1: (
        "Fixed seed-0 shuffle",
        ["Control for label sensitivity", "Deterministic seed, arbitrary locality"],
        "Only labels change; the topology is fixed.",
    ),
    2: (
        "Global degree sort",
        ["Descending degree", "Ties follow implementation order"],
        "High-degree vertices move toward the first memory region.",
    ),
    3: (
        "Sort selected hubs",
        [
            "Hub: degree > integer average degree",
            "Preserve source IDs where possible",
        ],
        "Only the hub region receives a full degree sort.",
    ),
    4: (
        "Stable hub clustering",
        ["Same average-degree hub threshold", "Preserve source IDs where possible"],
        "The hub/non-hub split changes regions without sorting every vertex.",
    ),
    5: (
        "Degree buckets",
        ["Logarithmic degree buckets", "Stable order inside each bucket"],
        "Bucket boundaries, not one global sort, define the output.",
    ),
    6: (
        "HubSort + buckets",
        ["Sort the hub bucket", "Then emit the remaining buckets"],
        "A sorted hub bucket is followed by grouped non-hubs.",
    ),
    7: (
        "Stable hub/non-hub buckets",
        ["Compact hubs first", "Preserve encounter order"],
        "Both regions preserve encounter order.",
    ),
    8: (
        "Community merge + DFS",
        ["Build merge dendrogram", "Emit a dendrogram DFS"],
        "Community blocks become contiguous; DFS chooses hierarchy order.",
    ),
    9: (
        "Standalone GORDER_csr",
        ["Sliding-window neighbor overlap", "Faithful comparator implementation"],
        "This is distinct from GraphBrew's relaxed local intra_gorder.",
    ),
    10: (
        "Canonical hot/cold",
        ["Classify workload regions", "Pack hot properties first"],
        "The workload segmentation defines the memory regions.",
    ),
    11: (
        "BNF + RCM",
        ["Peripheral BFS", "Reverse the CM order"],
        "The output targets lower graph bandwidth.",
    ),
    12: (
        "Explicit three-axis compose",
        ["Partition -> block -> local layout", "Shown spec uses gordf4 for scale"],
        "GraphBrew emits one explicit composition, not a competitor fallback.",
    ),
    13: (
        "External label list",
        ["file[new_id] = original_id", "Load, validate, then apply"],
        "MAP materializes a supplied order; it does not discover one.",
    ),
    14: (
        "Policy-selected arm",
        ["Features choose an existing arm", "No fixed AdaptiveOrder permutation"],
        "The output shown is the selected GraphBrew arm for this illustration.",
    ),
    15: (
        "Leiden + post-layout",
        ["Detect communities", "Apply the selected layout"],
        "Community detection alone is not an ordering.",
    ),
    16: (
        "Directed forward-edge",
        [
            "Increase edges with new(src) < new(dst)",
            "Symmetric input is only a control",
        ],
        "The objective is directed; the shown order is measured on this edge list.",
    ),
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_fixture() -> dict:
    payload = json.loads(FIXTURE.read_text())
    validate_fixture(payload)
    return payload


def adjacency(payload: dict) -> dict[int, set[int]]:
    vertices = payload["graph"]["vertices"]
    result = {vertex: set() for vertex in vertices}
    for source, target in payload["graph"]["undirected_edges"]:
        result[source].add(target)
        result[target].add(source)
    return result


def csr_for_order(
    graph: dict[int, set[int]],
    order: list[int],
) -> tuple[list[int], list[int]]:
    forward = {old_id: new_id for new_id, old_id in enumerate(order)}
    offsets = [0]
    neighbors: list[int] = []
    for old_id in order:
        row = sorted(forward[neighbor] for neighbor in graph[old_id])
        neighbors.extend(row)
        offsets.append(len(neighbors))
    return offsets, neighbors


def reference_bfs_order(
    graph: dict[int, set[int]],
    vertices: list[int],
) -> tuple[int, list[list[int]], list[int]]:
    members = set(vertices)
    root = max(vertices, key=lambda vertex: (len(graph[vertex]), -vertex))
    visited = {root}
    levels = [[root]]
    order = [root]
    while len(visited) < len(vertices):
        next_level = []
        for source in levels[-1]:
            for target in sorted(graph[source]):
                if target in members and target not in visited:
                    visited.add(target)
                    next_level.append(target)
                    order.append(target)
        if not next_level:
            for vertex in vertices:
                if vertex not in visited:
                    visited.add(vertex)
                    next_level.append(vertex)
                    order.append(vertex)
        levels.append(next_level)
    return root, levels, order


def reference_relaxed_gorder(
    graph: dict[int, set[int]],
    vertices: list[int],
    window: int,
) -> list[int]:
    if len(vertices) <= 3:
        return list(vertices)
    members = set(vertices)
    root = max(vertices, key=lambda vertex: (len(graph[vertex]), -vertex))
    active = set(vertices)
    active.remove(root)
    keys = {vertex: 0 for vertex in vertices}
    recency = {vertex: -index for index, vertex in enumerate(vertices)}
    clock = 0

    def touch(vertex: int, delta: int) -> None:
        nonlocal clock
        if vertex not in active:
            return
        keys[vertex] += delta
        clock += 1
        recency[vertex] = clock

    placed = [root]
    for neighbor in sorted(graph[root]):
        if neighbor in members:
            touch(neighbor, 1)
    while active:
        best = max(active, key=lambda vertex: (keys[vertex], recency[vertex]))
        active.remove(best)
        placed.append(best)
        for neighbor in sorted(graph[best]):
            if neighbor in members:
                touch(neighbor, 1)
        if len(placed) > window:
            expired = placed[-1 - window]
            for neighbor in sorted(graph[expired]):
                if neighbor in members:
                    touch(neighbor, -1)
    return placed


def validate_fixture(payload: dict) -> None:
    vertices = payload["graph"]["vertices"]
    if vertices != list(range(len(vertices))):
        raise ValueError("running-example vertices must be dense IDs")
    graph = adjacency(payload)
    edge_list_edges = [
        tuple(map(int, line.split()))
        for line in EXAMPLE_EDGE_LIST.read_text().splitlines()
        if line.strip()
    ]
    if edge_list_edges != [
        tuple(edge) for edge in payload["graph"]["undirected_edges"]
    ]:
        raise ValueError("tracked edge-list file disagrees with fixture graph")
    original_offsets, original_neighbors = csr_for_order(graph, vertices)
    expected_original = payload["graph"]["original_csr"]
    if (
        original_offsets != expected_original["offsets"]
        or original_neighbors != expected_original["neighbors"]
    ):
        raise ValueError("original CSR does not match the edge list")

    composition = payload["composition"]
    communities = composition["communities"]
    members = [vertex for values in communities.values() for vertex in values]
    if sorted(members) != vertices or len(set(members)) != len(vertices):
        raise ValueError("communities must partition the graph")
    c0 = composition["local_orders"]["C0"]
    bfs_root, bfs_levels, bfs_order = reference_bfs_order(
        graph,
        communities["C0"],
    )
    if (
        c0["root"] != bfs_root
        or c0["levels"] != bfs_levels
        or c0["order"] != bfs_order
    ):
        raise ValueError("C0 BFS fixture disagrees with reference semantics")
    c1 = composition["local_orders"]["C1"]
    if c1["order"] != reference_relaxed_gorder(
        graph,
        communities["C1"],
        c1["window"],
    ):
        raise ValueError(
            "C1 relaxed-Gorder fixture disagrees with reference semantics"
        )
    final_order = composition["final_order"]
    if sorted(final_order) != vertices:
        raise ValueError("final order is not a permutation")
    expected_order = [
        vertex
        for community in composition["block_order"]
        for vertex in composition["local_orders"][community]["order"]
    ]
    if final_order != expected_order:
        raise ValueError("final order does not compose block/local orders")
    map_order = [
        int(value) for value in EXAMPLE_MAP.read_text().split()
    ]
    if map_order != final_order:
        raise ValueError("tracked MAP input disagrees with final order")
    block_only_order = composition["block_only_order"]
    expected_block_only = [
        vertex
        for community in composition["block_order"]
        for vertex in sorted(communities[community])
    ]
    if block_only_order != expected_block_only:
        raise ValueError("block-only order must preserve input order per block")
    block_only_forward = [0] * len(vertices)
    for new_id, old_id in enumerate(block_only_order):
        block_only_forward[old_id] = new_id
    if block_only_forward != composition["block_only_forward_mapping"]:
        raise ValueError("block-only forward mapping mismatch")
    forward = [0] * len(vertices)
    for new_id, old_id in enumerate(final_order):
        forward[old_id] = new_id
    if forward != composition["forward_mapping"]:
        raise ValueError("forward mapping does not invert final order")
    offsets, neighbors = csr_for_order(graph, final_order)
    expected = composition["relabeled_csr"]
    if offsets != expected["offsets"] or neighbors != expected["neighbors"]:
        raise ValueError("relabeled CSR does not match the final order")

    tracked = payload["tracked_vertex"]
    old_id = tracked["old_id"]
    new_id = forward[old_id]
    if new_id != tracked["new_id"]:
        raise ValueError("tracked new ID mismatch")
    old_neighbors = sorted(graph[old_id])
    new_neighbors = sorted(forward[value] for value in graph[old_id])
    if old_neighbors != tracked["old_neighbors"]:
        raise ValueError("tracked original neighbors mismatch")
    if new_neighbors != tracked["new_neighbor_ids"]:
        raise ValueError("tracked new neighbors mismatch")
    per_line = tracked["vertices_per_cache_line"]
    old_lines = len({value // per_line for value in old_neighbors})
    new_lines = len({value // per_line for value in new_neighbors})
    if (
        old_lines != tracked["old_cache_lines"]
        or new_lines != tracked["new_cache_lines"]
    ):
        raise ValueError("tracked cache-line count mismatch")
    if max(old_neighbors) - min(old_neighbors) != tracked["old_id_span"]:
        raise ValueError("tracked old span mismatch")
    if max(new_neighbors) - min(new_neighbors) != tracked["new_id_span"]:
        raise ValueError("tracked new span mismatch")

    for raw_id, record in payload["catalog"]["algorithms"].items():
        algorithm_id = int(raw_id)
        if algorithm_id not in CATALOG_NAMES:
            raise ValueError(f"unknown catalog ID {algorithm_id}")
        if sorted(record["order"]) != vertices:
            raise ValueError(f"catalog order {algorithm_id} is invalid")
    if payload["catalog"]["algorithms"]["13"]["order"] != final_order:
        raise ValueError("catalog MAP order must equal the tracked final order")

    evidence = json.loads(LOW_REUSE_EVIDENCE.read_text())
    rows = {row["graph"]: row for row in evidence["records"]}
    for key, graph_name in (
        ("selected", "amazon0601"),
        ("fallback", "soc-Slashdot0811"),
    ):
        stored = payload["policy_examples"][key]
        row = rows[graph_name]
        if stored["graph"] != graph_name:
            raise ValueError(f"policy example {key} graph mismatch")
        comparisons = {
            "avg_degree": row["features"]["avg_degree"],
            "degree_cv": row["features"]["degree_cv"],
            "property_wsr_llc": row["features"]["property_wsr_llc"],
            "selected_strategy": row["selected_strategy"],
            "boost_over_candidate_kernel":
                row["boost_over_candidate_kernel"],
            "boost_over_selected_end_to_end":
                row["selected_strategy_speedup_reuse2"],
        }
        if key == "selected":
            comparisons["candidate_over_boost_mapping"] = (
                row["candidate_over_boost_mapping"]
            )
        for field, expected_value in comparisons.items():
            if stored[field] != expected_value:
                raise ValueError(
                    f"policy example {key} field {field} drifted"
                )

    if CAPTURE.is_file():
        receipt = json.loads(CAPTURE.read_text())
        if receipt["converter_sha256"] != payload["catalog"]["converter_sha256"]:
            raise ValueError("catalog converter hash disagrees with receipt")
        if receipt["input_sha256"] != hashlib.sha256(
            EXAMPLE_EDGE_LIST.read_bytes()
        ).hexdigest():
            raise ValueError("catalog input hash disagrees with receipt")
        if receipt["map_sha256"] != hashlib.sha256(
            EXAMPLE_MAP.read_bytes()
        ).hexdigest():
            raise ValueError("catalog map hash disagrees with receipt")
        for algorithm_id, record in payload["catalog"]["algorithms"].items():
            if int(algorithm_id) == 14:
                continue
            if receipt["records"][algorithm_id]["order"] != record["order"]:
                raise ValueError(
                    f"catalog receipt drift for algorithm {algorithm_id}"
                )


def type_scale(width: int) -> dict[str, int]:
    if width == 1200:
        return {
            "title": 32,
            "subtitle": 18,
            "domain": 16,
            "heading": 22,
            "label": 18,
            "body": 17,
            "small": 16,
        }
    if width == 1400:
        return {
            "title": 32,
            "subtitle": 17,
            "domain": 15,
            "heading": 22,
            "label": 17,
            "body": 15,
            "small": 14,
        }
    return {
        "title": 36,
        "subtitle": 19,
        "domain": 15,
        "heading": 24,
        "label": 19,
        "body": 18,
        "small": 16,
    }


def approximate_text_width(value: str, font_size: int, *, bold: bool = False) -> float:
    width = 0.0
    for character in value:
        if character in "ilI1|.,:;!' ":
            factor = 0.28
        elif character in "MW@#%":
            factor = 0.88
        elif character.isupper():
            factor = 0.66
        else:
            factor = 0.53
        width += factor * font_size
    return width * (1.05 if bold else 1.0)


class SVG:
    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        description: str,
        deck: str,
    ) -> None:
        self.width = width
        self.height = height
        scale = type_scale(width)
        self.lines = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" '
                f'role="img" aria-labelledby="title desc" fill="{INK}">'
            ),
            f'  <title id="title">{esc(title)}</title>',
            f'  <desc id="desc">{esc(description)}</desc>',
            "  <defs>",
            (
                '    <marker id="arrow" viewBox="0 0 10 8" '
                'markerWidth="10" markerHeight="8" refX="10" refY="4" '
                'orient="auto" markerUnits="userSpaceOnUse">'
            ),
            f'      <path d="M0 0 L10 4 L0 8 Z" fill="{ACTION}"/>',
            "    </marker>",
            "    <style>",
            "      :root{color-scheme:light dark}",
            "      @media(prefers-color-scheme:dark){",
            f'        [fill="{INK}"]{{fill:#ECE7DD}}[stroke="{INK}"]{{stroke:#ECE7DD}}',
            f'        [fill="{PAGE}"]{{fill:#1E2327}}[fill="{NEUTRAL}"]{{fill:#252A2E}}',
            f'        [fill="{BLUE}"]{{fill:#273846}}[fill="{GREEN}"]{{fill:#24382A}}',
            f'        [fill="{AMBER}"]{{fill:#3B3122}}[fill="{ROSE}"]{{fill:#3D292A}}',
            f'        [fill="{VIOLET}"]{{fill:#302C3C}}[stroke="{MUTED}"]{{stroke:#747D86}}',
            f'        [stroke="{ACTION}"]{{stroke:#63A8FF}}[fill="{ACTION}"]{{fill:#63A8FF}}',
            "        .arrow{stroke:#63A8FF!important}.edge{stroke:#ECE7DD!important}",
            "      }",
            '      .sans{font-family:Arial,Helvetica,sans-serif}',
            '      .mono{font-family:"SFMono-Regular",Consolas,monospace}',
            (
                f'      .title{{font-size:{scale["title"]}px;font-weight:700}}'
                f'.subtitle{{font-size:{scale["subtitle"]}px}}'
            ),
            (
                f'      .domain{{font-size:{scale["domain"]}px;font-weight:700;letter-spacing:1.2px}}'
                f'.heading{{font-size:{scale["heading"]}px;font-weight:700}}'
            ),
            (
                f'      .label{{font-size:{scale["label"]}px;font-weight:700}}'
                f'.body{{font-size:{scale["body"]}px}}'
                f'.small{{font-size:{scale["small"]}px}}'
            ),
            f'      .arrow{{fill:none;stroke:{ACTION};stroke-width:3;marker-end:url(#arrow)}}',
            f'      .edge{{stroke:{INK};stroke-width:2.2}}',
            "    </style>",
            "  </defs>",
            f'  <rect width="{width}" height="{height}" fill="{PAGE}"/>',
            f'  <text x="42" y="47" class="sans title">{esc(title)}</text>',
            f'  <text x="42" y="75" class="sans subtitle">{esc(deck)}</text>',
            f'  <line x1="42" y1="99" x2="{width - 42}" y2="99" stroke="{INK}" stroke-width="2"/>',
        ]

    def add(self, line: str) -> None:
        self.lines.append("  " + line)

    def rect(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        fill: str,
        *,
        stroke: str = INK,
        stroke_width: float = 2,
        radius: int = 10,
        dash: str | None = None,
    ) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"{dashed}/>'
        )

    def text(
        self,
        x: int,
        y: int,
        value: object,
        css: str = "body",
        *,
        anchor: str | None = None,
        fill: str | None = None,
    ) -> None:
        anchored = f' text-anchor="{anchor}"' if anchor else ""
        colored = f' fill="{fill}"' if fill else ""
        self.add(
            f'<text x="{x}" y="{y}" class="sans {css}"'
            f'{anchored}{colored}>{esc(value)}</text>'
        )

    def mono(
        self,
        x: int,
        y: int,
        value: object,
        css: str = "small",
        *,
        anchor: str | None = None,
    ) -> None:
        anchored = f' text-anchor="{anchor}"' if anchor else ""
        self.add(
            f'<text x="{x}" y="{y}" class="mono {css}"'
            f'{anchored}>{esc(value)}</text>'
        )

    def line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        css: str = "edge",
        stroke: str | None = None,
        stroke_width: float | None = None,
    ) -> None:
        if stroke:
            width = stroke_width or 2
            self.add(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="{width}"/>'
            )
        else:
            self.add(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{css}"/>'
            )

    def arrow(self, path: str) -> None:
        self.add(f'<path d="{path}" class="arrow"/>')

    def circle(
        self,
        x: int,
        y: int,
        radius: int,
        fill: str,
        *,
        stroke: str = INK,
        stroke_width: float = 2,
        dash: str | None = None,
    ) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"{dashed}/>'
        )

    def badge(self, x: int, y: int, number: str) -> None:
        self.circle(x, y, 17, INK, stroke=INK, stroke_width=1)
        self.text(x, y + 6, number, "label", anchor="middle", fill=PAGE)

    def footer(self, y: int, value: str, *, height: int = 44) -> None:
        width = int(self.width * 0.76)
        x = (self.width - width) // 2
        self.rect(x, y, width, height, NEUTRAL, radius=9)
        self.text(
            self.width // 2,
            y + height // 2 + 5,
            value,
            "small",
            anchor="middle",
        )

    def finish(self) -> str:
        return "\n".join([*self.lines, "</svg>", ""])


def graph_fill(payload: dict, vertex: int) -> str:
    communities = payload["composition"]["communities"]
    if vertex in communities["C0"]:
        return BLUE
    return AMBER


def draw_graph(
    svg: SVG,
    payload: dict,
    origin_x: int,
    origin_y: int,
    *,
    labels: dict[int, int] | None = None,
    subset: set[int] | None = None,
    scale: float = 0.82,
    tracked: int | None = 2,
    color_communities: bool = True,
) -> None:
    positions = {
        vertex: (
            origin_x + int(x * scale),
            origin_y + int(y * scale),
        )
        for vertex, (x, y) in GRAPH_POSITIONS.items()
    }
    for source, target in payload["graph"]["undirected_edges"]:
        if subset and (source not in subset or target not in subset):
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        svg.line(x1, y1, x2, y2)
    for vertex in payload["graph"]["vertices"]:
        if subset and vertex not in subset:
            continue
        x, y = positions[vertex]
        is_tracked = vertex == tracked
        svg.circle(
            x,
            y,
            23 if is_tracked else 20,
            (
                GREEN
                if is_tracked
                else graph_fill(payload, vertex)
                if color_communities
                else NEUTRAL
            ),
            stroke=ACTION if is_tracked else INK,
            stroke_width=3 if is_tracked else 2,
        )
        label = labels[vertex] if labels else vertex
        svg.text(x, y + 5, label, "label", anchor="middle")


def array_strip(
    svg: SVG,
    x: int,
    y: int,
    values: Iterable[object],
    *,
    width: int,
    height: int = 42,
    fills: list[str] | None = None,
    tracked_index: int | None = None,
    prefix: str = "",
) -> None:
    values = list(values)
    cell = width / len(values)
    for index, value in enumerate(values):
        left = x + index * cell
        fill = fills[index] if fills else PAGE
        stroke = ACTION if index == tracked_index else INK
        stroke_width = 3 if index == tracked_index else 1.5
        svg.add(
            f'<rect x="{left:.2f}" y="{y}" width="{cell:.2f}" '
            f'height="{height}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )
        rendered = f"{prefix}{value}"
        svg.mono(
            int(left + cell / 2),
            y + height // 2 + 5,
            rendered,
            "small",
            anchor="middle",
        )


def stage_card(
    svg: SVG,
    x: int,
    y: int,
    width: int,
    height: int,
    number: str,
    title: str,
    lines: list[str],
    tracked: str,
    fill: str,
) -> None:
    svg.rect(x, y, width, height, fill, stroke_width=3, radius=11)
    svg.badge(x + 30, y + 30, number)
    svg.text(x + 58, y + 37, title, "heading")
    for index, line in enumerate(lines):
        svg.text(x + 25, y + 75 + index * 25, line, "body")
    svg.rect(x + 25, y + height - 42, width - 50, 28, PAGE, radius=6)
    svg.mono(x + width // 2, y + height - 23, tracked, "small", anchor="middle")


def generate_architecture(payload: dict) -> str:
    svg = SVG(
        1200,
        690,
        "GraphBrew: current system and Rabbit-free research direction",
        (
            "The validated low-reuse selector and proposed Rabbit-free generator "
            "share one six-stage composition pipeline and one tracked vertex."
        ),
        "Running example: follow v2 from input CSR to new ID 0 and 3 cache lines -> 2.",
    )
    svg.rect(35, 118, 545, 76, BLUE, stroke_width=3)
    svg.text(60, 147, "CURRENT VALIDATED PATH", "domain")
    svg.text(60, 174, "Explicit recipes + frozen low-reuse selector", "body")
    svg.rect(620, 118, 545, 76, VIOLET, stroke_width=3)
    svg.text(645, 147, "PROPOSED RESEARCH PATH", "domain")
    svg.text(645, 174, "Rabbit-free deterministic composition generator", "body")
    svg.rect(
        25,
        208,
        1150,
        380,
        "none",
        stroke=MUTED,
        stroke_width=2,
        radius=12,
        dash="8 7",
    )
    svg.text(45, 232, "SHARED SIX-STAGE PIPELINE", "domain")
    cards = [
        (35, 250, "1", "Load + profile", ["Read CSR", "Compute lightweight features"], "row v2 = [1,4,6,8]", BLUE),
        (430, 250, "2", "Partition", ["Assign C0 / C1", "Keep topology unchanged"], "v2 in C0", VIOLET),
        (825, 250, "3", "Block layout", ["SizeDesc: C0 first", "Reserve contiguous ranges"], "C0 -> IDs 0..4", AMBER),
        (35, 425, "4", "Vertex layout", ["BFS for C0", "relaxed Gorder for C1"], "v2 -> local ID 0", VIOLET),
        (430, 425, "5", "Emit relabeled CSR", ["Compose old -> new IDs", "Validate permutation"], "v2 -> new ID 0", GREEN),
        (825, 425, "6", "Kernel locality", ["Same neighbors", "Fewer property cache lines"], "v2: 3 lines -> 2", GREEN),
    ]
    for card in cards:
        stage_card(
            svg,
            card[0],
            card[1],
            340,
            155,
            *card[2:],
        )
    svg.arrow("M375 327 H426")
    svg.arrow("M770 327 H821")
    svg.arrow("M995 405 V413 H205 V421")
    svg.arrow("M375 502 H426")
    svg.arrow("M770 502 H821")
    svg.footer(625, "Invariant: graph topology and kernel semantics do not change; only vertex IDs and CSR order change.")
    return svg.finish()


def generate_graph_to_csr(payload: dict) -> str:
    tracked = payload["tracked_vertex"]
    original = payload["graph"]["original_csr"]
    vertex_count = len(payload["graph"]["vertices"])
    undirected_edges = len(payload["graph"]["undirected_edges"])
    directed_arcs = len(original["neighbors"])
    tracked_id = tracked["old_id"]
    row_start = original["offsets"][tracked_id]
    row_end = original["offsets"][tracked_id + 1]
    line_ids = [
        value // tracked["vertices_per_cache_line"]
        for value in tracked["old_neighbors"]
    ]
    svg = SVG(
        1200,
        560,
        "Stage 1: load the graph and expose the tracked CSR row",
        "The input graph, CSR arrays, and lightweight profile all refer to vertex v2.",
        "Illustrative 9-vertex graph; tracked object v2 is outlined in blue.",
    )
    svg.rect(
        25,
        112,
        1150,
        380,
        "none",
        stroke=MUTED,
        stroke_width=2,
        radius=12,
        dash="8 7",
    )
    svg.text(45, 136, "STAGE 1 · INPUT AND PROFILE", "domain")
    svg.rect(45, 150, 320, 325, PAGE, stroke=MUTED, stroke_width=2)
    svg.badge(75, 180, "1")
    svg.text(102, 187, "Input topology", "heading")
    draw_graph(
        svg,
        payload,
        38,
        180,
        scale=0.60,
        color_communities=False,
    )
    svg.text(
        65,
        445,
        f"{undirected_edges} undirected edges / {directed_arcs} CSR arcs",
        "small",
    )
    svg.rect(405, 150, 355, 325, AMBER, stroke_width=2)
    svg.text(430, 187, "CSR row for v2", "heading")
    svg.mono(
        430,
        224,
        (
            f"offsets[{tracked_id}:{tracked_id + 2}] = "
            f"{original['offsets'][tracked_id:tracked_id + 2]}"
        ),
        "body",
    )
    svg.text(
        430,
        260,
        f"col_idx positions {row_start}..{row_end - 1}",
        "body",
    )
    array_strip(
        svg,
        430,
        280,
        tracked["old_neighbors"],
        width=310,
        height=54,
        fills=[BLUE] * len(tracked["old_neighbors"]),
    )
    svg.text(
        430,
        358,
        (
            f"neighbor IDs span {min(tracked['old_neighbors'])}.."
            f"{max(tracked['old_neighbors'])}"
        ),
        "body",
    )
    svg.mono(430, 388, f"span = {tracked['old_id_span']}", "label")
    svg.text(430, 420, "Property IDs touch cache lines:", "body")
    svg.mono(
        430,
        450,
        f"{line_ids} -> {tracked['old_cache_lines']} distinct lines",
        "small",
    )
    svg.rect(800, 150, 355, 325, VIOLET, stroke_width=2)
    svg.text(825, 187, "Lightweight profile", "heading")
    degrees = [len(adjacency(payload)[vertex]) for vertex in payload["graph"]["vertices"]]
    svg.mono(825, 230, f"N = {vertex_count}", "body")
    svg.mono(825, 262, f"directed arcs = {directed_arcs}", "body")
    svg.mono(
        825,
        294,
        f"degree(v{tracked_id}) = {degrees[tracked_id]}",
        "body",
    )
    svg.mono(825, 326, f"max degree = {max(degrees)}", "body")
    svg.text(825, 365, "These values describe the input.", "body")
    svg.text(825, 395, "No kernel has run and no labels moved.", "body")
    svg.rect(825, 420, 305, 38, PAGE, radius=7)
    svg.mono(
        982,
        445,
        f"tracked = v{tracked_id}",
        "label",
        anchor="middle",
    )
    svg.arrow("M365 315 H401")
    svg.arrow("M760 315 H796")
    svg.footer(500, "Invariant: Stage 1 observes the graph; it does not change IDs, edges, or kernel work.")
    return svg.finish()


def generate_partition(payload: dict) -> str:
    communities = payload["composition"]["communities"]
    edge_count = len(payload["graph"]["undirected_edges"])
    svg = SVG(
        1200,
        590,
        "Stage 2: partition the same graph into C0 and C1",
        (
            "The example manually freezes one pedagogical membership so every "
            "later layout and CSR value can be checked exactly."
        ),
        "This is not the detector output for the tiny graph; v2 is tracked inside C0.",
    )
    svg.rect(
        25,
        112,
        1150,
        405,
        "none",
        stroke=MUTED,
        stroke_width=2,
        radius=12,
        dash="8 7",
    )
    svg.text(45, 136, "STAGE 2 · PARTITION", "domain")
    svg.rect(45, 150, 500, 345, PAGE, stroke=MUTED, stroke_width=2)
    svg.text(70, 185, "Before partition metadata", "heading")
    draw_graph(
        svg,
        payload,
        55,
        180,
        scale=0.72,
        color_communities=False,
    )
    svg.mono(70, 465, "membership = unknown", "small")
    svg.rect(655, 150, 500, 345, PAGE, stroke=MUTED, stroke_width=2)
    svg.badge(685, 180, "2")
    svg.text(712, 187, "Frozen example membership", "heading")
    draw_graph(svg, payload, 665, 180, scale=0.72)
    svg.rect(680, 430, 210, 42, BLUE, radius=7)
    svg.mono(
        785,
        457,
        f"C0 = {{{','.join(map(str, communities['C0']))}}}",
        "small",
        anchor="middle",
    )
    svg.rect(920, 430, 210, 42, AMBER, radius=7)
    svg.mono(
        1025,
        457,
        f"C1 = {{{','.join(map(str, communities['C1']))}}}",
        "small",
        anchor="middle",
    )
    svg.arrow("M545 320 H651")
    svg.footer(
        525,
        (
            "Invariant: partitioning adds C(v); vertex IDs and the "
            f"{edge_count}-edge topology remain unchanged."
        ),
    )
    return svg.finish()


def community_fills(payload: dict, order: list[int]) -> list[str]:
    c0 = set(payload["composition"]["communities"]["C0"])
    return [BLUE if vertex in c0 else AMBER for vertex in order]


def generate_size_desc(payload: dict) -> str:
    communities = payload["composition"]["communities"]
    block_order = payload["composition"]["block_only_order"]
    forward = payload["composition"]["block_only_forward_mapping"]
    tracked_id = payload["tracked_vertex"]["old_id"]
    c0_size = len(communities["C0"])
    c1_size = len(communities["C1"])
    svg = SVG(
        1200,
        550,
        "Stage 3: SizeDesc turns memberships into contiguous blocks",
        (
            f"C0 has {c0_size} vertices, so it receives IDs 0..{c0_size - 1}; "
            f"C1 receives IDs {c0_size}..{c0_size + c1_size - 1}."
        ),
        "Same C0/C1 membership from Stage 2; only the global block order changes.",
    )
    svg.rect(
        25,
        112,
        1150,
        350,
        "none",
        stroke=MUTED,
        stroke_width=2,
        radius=12,
        dash="8 7",
    )
    svg.text(45, 136, "STAGE 3 · BLOCK LAYOUT", "domain")
    svg.rect(45, 150, 500, 300, NEUTRAL, stroke=MUTED, stroke_width=2)
    svg.text(70, 185, "Input-ID memory order", "heading")
    array_strip(
        svg,
        70,
        210,
        payload["graph"]["vertices"],
        width=450,
        height=52,
        fills=community_fills(payload, payload["graph"]["vertices"]),
        tracked_index=tracked_id,
        prefix="v",
    )
    svg.text(70, 295, "C0 and C1 are interleaved.", "body")
    svg.mono(
        70,
        331,
        f"v{tracked_id} is at old ID {tracked_id}",
        "body",
    )
    svg.rect(70, 365, 450, 54, PAGE, radius=7)
    svg.mono(
        295,
        398,
        f"C0 size {c0_size} > C1 size {c1_size}",
        "label",
        anchor="middle",
    )
    svg.rect(655, 150, 500, 300, NEUTRAL, stroke=COMMUNITY_B, stroke_width=2)
    svg.badge(685, 180, "3")
    svg.text(712, 187, "SizeDesc block order", "heading")
    array_strip(
        svg,
        680,
        210,
        block_order,
        width=450,
        height=52,
        fills=community_fills(payload, block_order),
        tracked_index=block_order.index(tracked_id),
        prefix="v",
    )
    svg.text(
        680,
        295,
        f"C0 occupies new IDs 0..{c0_size - 1}.",
        "body",
    )
    svg.text(
        680,
        325,
        f"C1 occupies new IDs {c0_size}..{c0_size + c1_size - 1}.",
        "body",
    )
    svg.mono(
        680,
        365,
        f"old v{tracked_id} -> block ID {forward[tracked_id]}",
        "label",
    )
    svg.arrow("M545 300 H651")
    svg.footer(480, "Invariant: community membership and graph topology are unchanged; only block placement changes.")
    return svg.finish()


def generate_gordf(payload: dict) -> str:
    communities = payload["composition"]["communities"]
    local_orders = payload["composition"]["local_orders"]
    threshold = payload["composition"]["gorder_fallback_threshold"]
    c0_size = len(communities["C0"])
    c1_size = len(communities["C1"])
    svg = SVG(
        1200,
        510,
        "Stage 4 dispatch: tiny fast path, then one local-layout threshold",
        (
            f"For readability the example uses gordf{threshold}; "
            "the evaluated recipe uses gordf5000."
        ),
        f"Same blocks from Stage 3: C0 has {c0_size} vertices and C1 has {c1_size}.",
    )
    svg.rect(
        25,
        108,
        1150,
        322,
        "none",
        stroke=MUTED,
        stroke_width=2,
        radius=12,
        dash="8 7",
    )
    svg.text(45, 132, "STAGE 4 · PER-BLOCK DISPATCH", "domain")
    svg.rect(405, 118, 390, 112, VIOLET, stroke_width=3)
    svg.badge(435, 155, "4")
    svg.text(465, 157, f"Example: is |C| <= {threshold}?", "heading")
    svg.mono(600, 188, "|C| <= 3 keeps input order", "small", anchor="middle")
    svg.mono(
        600,
        214,
        f"4..{threshold}: Gorder; >{threshold}: BFS",
        "small",
        anchor="middle",
    )
    svg.rect(45, 260, 490, 160, NEUTRAL, stroke=COMMUNITY_B, stroke_width=2)
    svg.text(60, 296, f"YES: C1, size {c1_size}", "heading")
    svg.text(60, 330, "Use relaxed local Gorder (gw8).", "body")
    array_strip(
        svg,
        70,
        350,
        local_orders["C1"]["order"],
        width=440,
        fills=[AMBER] * c1_size,
        prefix="v",
    )
    svg.rect(665, 260, 490, 160, NEUTRAL, stroke=ACTION, stroke_width=2)
    svg.text(690, 296, f"NO: C0, size {c0_size}", "heading")
    svg.text(690, 330, "Use hub-rooted BFS fallback.", "body")
    array_strip(
        svg,
        690,
        350,
        local_orders["C0"]["order"],
        width=440,
        fills=[BLUE] * c0_size,
        tracked_index=0,
        prefix="v",
    )
    svg.arrow("M500 230 V242 H285 V256")
    svg.arrow("M700 230 V242 H915 V256")
    svg.footer(445, "Invariant: both local orders stay inside their assigned block ranges.")
    return svg.finish()


def subgraph_positions(vertices: list[int], center_x: int, center_y: int) -> dict[int, tuple[int, int]]:
    radius = 105
    result = {}
    for index, vertex in enumerate(vertices):
        angle = -math.pi / 2 + 2 * math.pi * index / len(vertices)
        result[vertex] = (
            center_x + int(radius * math.cos(angle)),
            center_y + int(radius * math.sin(angle)),
        )
    return result


def draw_subgraph(
    svg: SVG,
    payload: dict,
    vertices: list[int],
    center_x: int,
    center_y: int,
    *,
    tracked: int,
    labels: dict[int, int] | None = None,
    fill: str,
) -> None:
    positions = subgraph_positions(vertices, center_x, center_y)
    edge_set = {
        tuple(sorted(edge))
        for edge in payload["graph"]["undirected_edges"]
    }
    for index, source in enumerate(vertices):
        for target in vertices[index + 1:]:
            if tuple(sorted((source, target))) in edge_set:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                svg.line(x1, y1, x2, y2)
    for vertex in vertices:
        x, y = positions[vertex]
        is_tracked = vertex == tracked
        svg.circle(
            x,
            y,
            23 if is_tracked else 20,
            GREEN if is_tracked else fill,
            stroke=ACTION if is_tracked else INK,
            stroke_width=3 if is_tracked else 2,
        )
        svg.text(
            x,
            y + 5,
            labels[vertex] if labels else vertex,
            "label",
            anchor="middle",
        )


def generate_gorder(payload: dict) -> str:
    before = payload["composition"]["communities"]["C1"]
    after = payload["composition"]["local_orders"]["C1"]["order"]
    labels = {vertex: index for index, vertex in enumerate(after)}
    svg = SVG(
        1200,
        535,
        "Stage 4A: relaxed local Gorder orders the small C1 block",
        "The production intra_gorder is a direct-neighbor UnitHeap heuristic, not faithful standalone GORDER_csr.",
        (
            "Branch detail switches the highlight to v8 because tracked v2 is "
            "in C0; gw8 changes only C1 local IDs."
        ),
    )
    svg.rect(
        25,
        112,
        1150,
        350,
        "none",
        stroke=MUTED,
        stroke_width=2,
        radius=12,
        dash="8 7",
    )
    svg.text(45, 136, "STAGE 4A · SMALL-BLOCK LOCAL LAYOUT", "domain")
    svg.rect(45, 150, 490, 300, NEUTRAL, stroke=COMMUNITY_B, stroke_width=2)
    svg.text(70, 185, "Before local layout", "heading")
    draw_subgraph(svg, payload, before, 290, 265, tracked=8, fill=AMBER)
    array_strip(svg, 70, 390, before, width=440, fills=[AMBER] * 4, prefix="v")
    svg.rect(665, 150, 490, 300, NEUTRAL, stroke=ACTION, stroke_width=2)
    svg.badge(695, 180, "4A")
    svg.text(735, 187, "After relaxed intra_gorder", "heading")
    draw_subgraph(svg, payload, before, 910, 265, tracked=8, labels=labels, fill=AMBER)
    array_strip(svg, 690, 390, after, width=440, fills=[AMBER] * 4, tracked_index=0, prefix="v")
    svg.arrow("M535 300 H661")
    svg.footer(475, "Invariant: C1 remains IDs 5..8 globally; only its four local positions change.")
    return svg.finish()


def generate_bfs(payload: dict) -> str:
    values = payload["composition"]["communities"]["C0"]
    local = payload["composition"]["local_orders"]["C0"]
    labels = {vertex: index for index, vertex in enumerate(local["order"])}
    svg = SVG(
        1200,
        535,
        "Stage 4B: hub-rooted BFS orders the large C0 block",
        "The tracked vertex v2 is the highest-degree root and receives local ID 0.",
        "Same C0 = {1,2,4,6,7}; BFS levels are [2] -> [1,4,6] -> [7].",
    )
    svg.rect(
        25,
        112,
        1150,
        350,
        "none",
        stroke=MUTED,
        stroke_width=2,
        radius=12,
        dash="8 7",
    )
    svg.text(45, 136, "STAGE 4B · LARGE-BLOCK LOCAL LAYOUT", "domain")
    svg.rect(45, 150, 490, 300, NEUTRAL, stroke=ACTION, stroke_width=2)
    svg.text(70, 185, "Before local layout", "heading")
    draw_subgraph(svg, payload, values, 290, 265, tracked=2, fill=BLUE)
    array_strip(svg, 70, 390, values, width=440, fills=[BLUE] * 5, tracked_index=1, prefix="v")
    svg.rect(665, 150, 490, 300, NEUTRAL, stroke=ACTION, stroke_width=2)
    svg.badge(695, 180, "4B")
    svg.text(735, 187, "After BFS levels", "heading")
    draw_subgraph(svg, payload, values, 910, 265, tracked=2, labels=labels, fill=BLUE)
    array_strip(svg, 690, 390, local["order"], width=440, fills=[BLUE] * 5, tracked_index=0, prefix="v")
    svg.arrow("M535 300 H661")
    svg.footer(475, "Invariant: C0 remains the first five global IDs; BFS only assigns its local order.")
    return svg.finish()


def generate_relabel(payload: dict) -> str:
    composition = payload["composition"]
    order = composition["final_order"]
    forward = composition["forward_mapping"]
    svg = SVG(
        1200,
        545,
        "Stage 5: compose the block and local orders into relabeled CSR",
        "The final permutation maps tracked v2 to new ID 0 and rewrites CSR rows and destinations consistently.",
        "Final memory order: [v2,v1,v4,v6,v7 | v8,v5,v0,v3].",
    )
    svg.rect(
        25, 112, 1150, 365, "none",
        stroke=MUTED, stroke_width=2, radius=12, dash="8 7",
    )
    svg.text(45, 136, "STAGE 5 · RELABEL AND EMIT", "domain")
    svg.rect(45, 150, 520, 315, NEUTRAL, stroke=VIOLET, stroke_width=2)
    svg.badge(75, 180, "5")
    svg.text(105, 187, "Old ID -> new ID", "heading")
    array_strip(svg, 70, 210, payload["graph"]["vertices"], width=470, fills=[BLUE] * 9, prefix="v")
    array_strip(svg, 70, 262, forward, width=470, fills=[GREEN] * 9, tracked_index=2, prefix="n")
    svg.text(70, 340, "Composed memory order", "label")
    array_strip(
        svg,
        70,
        360,
        order,
        width=470,
        fills=community_fills(payload, order),
        tracked_index=0,
        prefix="v",
    )
    svg.rect(625, 150, 530, 315, NEUTRAL, stroke=GREEN, stroke_width=2)
    svg.text(650, 187, "Relabeled CSR row 0 (old v2)", "heading")
    relabeled = composition["relabeled_csr"]
    row_zero = relabeled["neighbors"][
        relabeled["offsets"][0]:relabeled["offsets"][1]
    ]
    svg.mono(650, 230, f"offsets[0:2] = {relabeled['offsets'][0:2]}", "body")
    svg.text(650, 266, "new neighbor IDs", "body")
    array_strip(
        svg,
        650,
        285,
        row_zero,
        width=480,
        fills=[GREEN] * len(row_zero),
    )
    svg.mono(650, 370, f"row 0 = {row_zero}", "label")
    svg.text(
        650,
        405,
        f"All {len(relabeled['neighbors'])} directed arcs are preserved.",
        "body",
    )
    svg.text(650, 435, "Only row and destination indices change.", "body")
    svg.arrow("M565 305 H621")
    svg.footer(490, "Invariant: the permutation is bijective and the relabeled CSR represents the same graph.")
    return svg.finish()


def cache_line_strip(
    svg: SVG,
    x: int,
    y: int,
    values: list[int],
    *,
    width: int,
    tracked_line_count: int,
) -> None:
    array_strip(svg, x, y, values, width=width, height=54, fills=[GREEN] * len(values))
    per_line = 4
    line_ids = [value // per_line for value in values]
    for line_id in sorted(set(line_ids)):
        indexes = [index for index, value in enumerate(line_ids) if value == line_id]
        left = x + min(indexes) * width / len(values)
        right = x + (max(indexes) + 1) * width / len(values)
        svg.add(
            f'<rect x="{left:.2f}" y="{y - 9}" width="{right - left:.2f}" '
            f'height="72" fill="none" stroke="{ACTION}" stroke-width="2" '
            f'stroke-dasharray="5 4"/>'
        )
        svg.mono(int((left + right) / 2), y + 82, f"line {line_id}", "small", anchor="middle")
    svg.text(
        x + width // 2,
        y + 116,
        f"{tracked_line_count} distinct property cache lines",
        "label",
        anchor="middle",
    )


def generate_locality(payload: dict) -> str:
    tracked = payload["tracked_vertex"]
    svg = SVG(
        1200,
        535,
        "Stage 6: the same v2 neighbors touch fewer property cache lines",
        "With four vertex properties per cache line, relabeling changes three touched lines into two.",
        "Same neighbor set; only the neighbor IDs and property addresses are relabeled.",
    )
    svg.rect(
        25, 112, 1150, 350, "none",
        stroke=MUTED, stroke_width=2, radius=12, dash="8 7",
    )
    svg.text(45, 136, "STAGE 6 · LOCALITY OUTCOME", "domain")
    svg.rect(45, 150, 490, 300, NEUTRAL, stroke=ROSE, stroke_width=2)
    svg.text(70, 185, "Before: old neighbor IDs", "heading")
    cache_line_strip(
        svg,
        70,
        225,
        tracked["old_neighbors"],
        width=440,
        tracked_line_count=tracked["old_cache_lines"],
    )
    svg.mono(290, 400, f"span = {tracked['old_id_span']}", "label", anchor="middle")
    svg.rect(665, 150, 490, 300, NEUTRAL, stroke=GREEN, stroke_width=2)
    svg.badge(695, 180, "6")
    svg.text(735, 187, "After: new neighbor IDs", "heading")
    cache_line_strip(
        svg,
        690,
        225,
        tracked["new_neighbor_ids"],
        width=440,
        tracked_line_count=tracked["new_cache_lines"],
    )
    svg.mono(910, 400, f"span = {tracked['new_id_span']}", "label", anchor="middle")
    svg.arrow("M535 300 H661")
    svg.footer(475, "Invariant: v2 still reads the same four neighbor properties; address locality is the only payoff shown.")
    return svg.finish()


def generate_cd_parallel(payload: dict) -> str:
    svg = SVG(
        1200,
        500,
        "Control: cd_serial versus cd_parallel",
        "Both modes use the same graph and recipe; parallel move scheduling may change the realized membership and mapping.",
        "Running example IDs v0..v8; fingerprint equality, not the branch name, proves byte identity.",
    )
    svg.rect(
        25, 112, 1150, 305, "none",
        stroke=MUTED, stroke_width=2, radius=12, dash="8 7",
    )
    svg.rect(35, 125, 520, 280, AMBER, stroke_width=3)
    svg.text(60, 160, "cd_serial", "heading")
    svg.text(60, 195, "One ordered move stream.", "body")
    array_strip(svg, 60, 225, range(9), width=470, fills=[AMBER] * 9, tracked_index=2, prefix="v")
    svg.mono(60, 310, "membership fingerprint = M0", "body")
    svg.mono(60, 342, "mapping fingerprint = P0", "body")
    svg.rect(645, 125, 520, 280, BLUE, stroke_width=3)
    svg.text(670, 160, "cd_parallel", "heading")
    svg.text(670, 195, "Workers evaluate vertices concurrently.", "body")
    svg.mono(670, 232, "T0: v0, v3, v6", "small")
    svg.mono(670, 262, "T1: v1, v4, v7", "small")
    svg.mono(670, 292, "T2: v2, v5, v8", "small")
    svg.mono(670, 332, "repeat draws -> compare M/P", "body")
    svg.arrow("M555 270 H641")
    svg.footer(430, "Contract: schedule-sensitive builds are repeated; differing fingerprints must not be pooled.")
    return svg.finish()


def generate_sgmb(payload: dict) -> str:
    svg = SVG(
        1200,
        520,
        "Control: sgmb4096 batches proposals but commits them in order",
        "The batch contains community super-nodes, not original graph vertices.",
        "Illustrative C0/C1 supergraph derived from the same running example.",
    )
    svg.rect(
        25, 112, 1150, 315, "none",
        stroke=MUTED, stroke_width=2, radius=12, dash="8 7",
    )
    svg.rect(35, 125, 1130, 105, VIOLET, stroke_width=3)
    svg.text(60, 160, "Community supergraph", "heading")
    svg.circle(310, 182, 25, BLUE)
    svg.text(310, 188, "C0", "label", anchor="middle")
    svg.circle(890, 182, 25, AMBER)
    svg.text(890, 188, "C1", "label", anchor="middle")
    svg.line(337, 182, 863, 182, stroke=ACTION, stroke_width=4)
    svg.mono(600, 174, "cross edges: (v8,v1), (v8,v2)", "small", anchor="middle")
    svg.rect(35, 270, 520, 145, BLUE, stroke_width=3)
    svg.text(60, 305, "1. Proposal phase", "heading")
    svg.mono(60, 340, "batch = [C0, C1]", "body")
    svg.text(60, 372, "Workers evaluate from one pre-commit state.", "body")
    svg.rect(645, 270, 520, 145, GREEN, stroke_width=3)
    svg.text(670, 305, "2. Ordered commit", "heading")
    array_strip(svg, 670, 330, ["C0", "C1"], width=470, fills=[BLUE, AMBER])
    svg.text(670, 395, "Commit order is deterministic within the batch.", "body")
    svg.arrow("M555 342 H641")
    svg.footer(445, "Invariant: sgmb changes proposal batching, not the requested partition/layout recipe.")
    return svg.finish()


def generate_norefine(payload: dict) -> str:
    svg = SVG(
        1200,
        500,
        "Control: norefine removes the constrained Leiden refinement phase",
        "Local moving and aggregation still run; the connectivity and subset-optimality guarantees no longer apply.",
        "Same running graph and C0/C1 notation; this figure compares phase structure, not measured quality.",
    )
    svg.rect(
        25, 112, 1150, 300, "none",
        stroke=MUTED, stroke_width=2, radius=12, dash="8 7",
    )
    svg.rect(35, 130, 300, 240, BLUE, stroke_width=3)
    svg.text(60, 165, "Local moving", "heading")
    svg.text(60, 202, "Move vertices between groups.", "body")
    svg.mono(60, 245, "v2 -> candidate C0", "body")
    svg.rect(450, 130, 300, 110, AMBER, stroke_width=3)
    svg.text(475, 165, "norefine path", "heading")
    svg.text(475, 202, "Aggregate directly.", "body")
    svg.rect(450, 285, 300, 110, VIOLET, stroke_width=3)
    svg.text(475, 320, "full Leiden path", "heading")
    svg.text(475, 357, "Refine within bounds, then aggregate.", "body")
    svg.rect(865, 130, 300, 265, ROSE, stroke_width=3)
    svg.text(890, 165, "Scientific consequence", "heading")
    svg.text(890, 205, "norefine lowers preprocessing.", "body")
    svg.text(890, 235, "It may change the partition.", "body")
    svg.text(890, 265, "Do not claim Leiden guarantees.", "body")
    svg.rect(890, 310, 250, 48, PAGE, radius=7)
    svg.mono(1015, 340, "bounded local-moving path", "small", anchor="middle")
    svg.arrow("M335 205 H446")
    svg.arrow("M335 300 H446")
    svg.arrow("M750 185 H861")
    svg.arrow("M750 340 H861")
    svg.footer(430, "Invariant: norefine changes community-detection semantics; vertex-layout stages still run afterward.")
    return svg.finish()


def generate_policy(payload: dict) -> str:
    selected = payload["policy_examples"]["selected"]
    fallback = payload["policy_examples"]["fallback"]
    svg = SVG(
        1200,
        680,
        "Validated low-reuse selector: one rule, two real holdout outcomes",
        "This is the current evidence-bound policy, separate from the proposed Rabbit-free generator.",
        "Reuse = 2 examples from the frozen rule2 holdout: amazon0601 and soc-Slashdot0811.",
    )
    svg.rect(35, 120, 300, 210, BLUE, stroke_width=3)
    svg.text(60, 155, "Runtime inputs", "heading")
    svg.text(60, 195, "Graph + supported kernel", "body")
    svg.text(60, 225, "Machine LLC", "body")
    svg.text(60, 255, "Declared reuse = 2", "body")
    svg.text(60, 285, "Tier-0 sampled features", "body")
    svg.rect(450, 120, 300, 210, VIOLET, stroke_width=3)
    svg.text(475, 155, "Frozen predicate", "heading")
    svg.mono(475, 194, "kernel supported; reuse <= 2", "small")
    svg.mono(475, 220, "N >= 1,000; WSR/LLC <= 3.2", "small")
    svg.mono(475, 246, "CV <= 2.68 and (deg <= 60", "small")
    svg.mono(475, 270, "or WSR/LLC <= 0.82), or CV > 8", "small")
    svg.text(475, 302, "No benchmarking before choice.", "small")
    svg.rect(865, 120, 300, 210, AMBER, stroke_width=3)
    svg.text(890, 155, "Selected arm", "heading")
    svg.text(890, 198, "FastLeiden-Gorder8", "body")
    svg.text(890, 228, "or Rabbit Boost fallback", "body")
    svg.text(890, 270, "This is a validated portfolio,", "small")
    svg.text(890, 296, "not the final Rabbit-free claim.", "small")
    svg.arrow("M335 225 H446")
    svg.arrow("M750 225 H861")
    svg.rect(35, 375, 545, 220, GREEN, stroke_width=3)
    svg.text(60, 410, f"SELECTED: {selected['graph']}", "heading")
    svg.mono(60, 448, f"degree CV = {selected['degree_cv']:.3f}", "body")
    svg.mono(60, 478, f"WSR/LLC = {selected['property_wsr_llc']:.3f}", "body")
    svg.mono(60, 508, f"candidate/Boost map = {selected['candidate_over_boost_mapping']:.3f}", "body")
    svg.mono(60, 538, f"Boost/candidate kernel = {selected['boost_over_candidate_kernel']:.3f}", "body")
    svg.mono(60, 568, f"Boost/selected E2E = {selected['boost_over_selected_end_to_end']:.3f}x", "label")
    svg.rect(620, 375, 545, 220, ROSE, stroke_width=3)
    svg.text(645, 410, f"FALLBACK: {fallback['graph']}", "heading")
    svg.mono(645, 448, f"degree CV = {fallback['degree_cv']:.3f}", "body")
    svg.mono(645, 478, f"WSR/LLC = {fallback['property_wsr_llc']:.3f}", "body")
    svg.mono(645, 508, f"Boost/candidate kernel = {fallback['boost_over_candidate_kernel']:.3f}", "body")
    svg.text(645, 542, "Outside the frozen GraphBrew region.", "body")
    svg.mono(
        645,
        572,
        (
            "Boost/selected E2E = "
            f"{fallback['boost_over_selected_end_to_end']:.3f}x"
        ),
        "label",
    )
    svg.footer(620, "Evidence scope: seven kernels, reuse 1-2, fixed platform; the selector is not a universal GraphBrew result.")
    return svg.finish()


def edge_span_metrics(payload: dict, order: list[int]) -> tuple[int, float]:
    forward = {old_id: new_id for new_id, old_id in enumerate(order)}
    spans = [
        abs(forward[source] - forward[target])
        for source, target in payload["graph"]["undirected_edges"]
    ]
    return max(spans), sum(spans) / len(spans)


def catalog_details(payload: dict, algorithm_id: int, order: list[int]) -> list[str]:
    graph = adjacency(payload)
    if algorithm_id == 2:
        return [
            "degrees in output order:",
            " ".join(str(len(graph[vertex])) for vertex in order),
        ]
    if algorithm_id == 11:
        before = edge_span_metrics(payload, list(range(9)))
        after = edge_span_metrics(payload, order)
        return [
            f"max span {before[0]} -> {after[0]}",
            f"mean span {before[1]:.2f} -> {after[1]:.2f}",
        ]
    if algorithm_id == 16:
        return [
            "symmetric input: objective is a control",
            "use a directed corpus for a GoGraph claim",
        ]
    if algorithm_id == 12:
        return [
            "measured end-to-end detector output",
            "different from frozen pedagogical C0/C1",
        ]
    if algorithm_id == 14:
        return ["features -> selected arm", "output equals selected arm"]
    return CATALOG_COPY[algorithm_id][1]


def catalog_display_spec(algorithm_id: int, spec: str) -> str:
    if algorithm_id == 12:
        return "12:leiden:compose:...:gordf4"
    if algorithm_id == 13:
        return "13:docs/figures/data/graphbrew-running-example.lo"
    return spec


def generate_catalog_input(payload: dict) -> str:
    svg = SVG(
        1200,
        430,
        "Shared input for all 17 reordering figures",
        "Every catalog strip below starts from this exact topology, original ID order, and measured converter build.",
        "The output strips are directly comparable because the input graph never changes.",
    )
    svg.rect(35, 120, 540, 240, BLUE, stroke_width=3)
    draw_graph(
        svg,
        payload,
        45,
        120,
        scale=0.78,
        color_communities=False,
    )
    svg.rect(610, 120, 555, 240, AMBER, stroke_width=3)
    svg.text(635, 155, "Input order and degree", "heading")
    array_strip(svg, 635, 185, range(9), width=505, fills=[BLUE] * 9, tracked_index=2, prefix="v")
    degrees = [len(adjacency(payload)[vertex]) for vertex in range(9)]
    array_strip(svg, 635, 245, degrees, width=505, fills=[NEUTRAL] * 9)
    svg.text(635, 325, "Top row: original IDs. Bottom row: degree.", "small")
    svg.footer(375, "Invariant for the catalog: same 9 vertices, same 12 undirected edges, same converter binary.")
    return svg.finish()


def generate_catalog_figure(payload: dict, algorithm_id: int) -> str:
    record = payload["catalog"]["algorithms"][str(algorithm_id)]
    order = record["order"]
    positions = {vertex: index for index, vertex in enumerate(order)}
    displacements = {
        vertex: abs(positions[vertex] - vertex)
        for vertex in range(9)
    }
    moved = max(displacements, key=displacements.get)
    has_movement = displacements[moved] > 0
    title = f"{algorithm_id}. {CATALOG_NAMES[algorithm_id]}"
    mechanism, _default_lines, footer = CATALOG_COPY[algorithm_id]
    details = catalog_details(payload, algorithm_id, order)
    svg = SVG(
        1200,
        430,
        title,
        f"{mechanism}. Output measured on the shared 9-vertex example.",
        (
            f"CLI {catalog_display_spec(algorithm_id, record['spec'])} | "
            + (
                f"moved example: v{moved} old {moved} -> new {positions[moved]}."
                if has_movement
                else "no vertex moves."
            )
        ),
    )
    svg.rect(
        25, 112, 1150, 255, "none",
        stroke=MUTED, stroke_width=2, radius=12, dash="8 7",
    )
    svg.rect(35, 125, 280, 230, PAGE, stroke=MUTED, stroke_width=2)
    svg.text(60, 158, "Same input graph", "heading")
    draw_graph(
        svg,
        payload,
        25,
        150,
        scale=0.48,
        tracked=moved if has_movement else None,
        color_communities=False,
    )
    svg.mono(
        60,
        330,
        (
            f"v{moved}: old {moved} -> new {positions[moved]}"
            if has_movement
            else "identity: no vertex moves"
        ),
        "small",
    )
    svg.rect(365, 125, 450, 230, NEUTRAL, stroke=ACTION, stroke_width=2)
    svg.text(390, 158, "Measured output order", "heading")
    fills = [
        GREEN
        if has_movement and vertex == moved
        else PAGE
        for vertex in order
    ]
    array_strip(
        svg,
        390,
        190,
        order,
        width=400,
        height=58,
        fills=fills,
        tracked_index=positions[moved] if has_movement else None,
        prefix="v",
    )
    svg.mono(390, 280, f"permutation = {order}", "small")
    svg.rect(865, 125, 300, 230, VIOLET, stroke_width=2)
    svg.badge(895, 155, str(algorithm_id))
    heading_css = (
        "label"
        if approximate_text_width(mechanism, 22, bold=True) > 250
        else "heading"
    )
    if approximate_text_width(
        mechanism,
        18 if heading_css == "label" else 22,
        bold=True,
    ) > 250:
        raise ValueError(
            f"catalog heading does not fit for algorithm {algorithm_id}"
        )
    svg.text(
        925,
        162,
        mechanism,
        heading_css,
    )
    for index, line in enumerate(details[:3]):
        css = "small" if len(line) > 36 else "body"
        svg.text(890, 215 + index * 32, line, css)
    svg.arrow("M315 240 H361")
    svg.arrow("M815 240 H861")
    svg.footer(380, footer, height=36)
    return svg.finish()


def mx_cell(
    cell_id: str,
    value: str,
    style: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    return (
        f'        <mxCell id="{esc(cell_id)}" value="{esc(value)}" '
        f'style="{esc(style)}" vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x:g}" y="{y:g}" width="{width:g}" '
        f'height="{height:g}" as="geometry" />\n'
        "        </mxCell>"
    )


def mx_edge(
    cell_id: str,
    source: str,
    target: str,
    *,
    stroke: str = INK,
    stroke_width: float = 2.2,
) -> str:
    return (
        f'        <mxCell id="{esc(cell_id)}" '
        f'style="endArrow=none;html=1;rounded=0;strokeColor={stroke};'
        f'strokeWidth={stroke_width:g};" edge="1" parent="1" '
        f'source="{esc(source)}" target="{esc(target)}">\n'
        '          <mxGeometry relative="1" as="geometry" />\n'
        "        </mxCell>"
    )


def drawio_catalog_page(payload: dict, algorithm_id: int) -> str:
    record = payload["catalog"]["algorithms"][str(algorithm_id)]
    order = record["order"]
    positions = {vertex: index for index, vertex in enumerate(order)}
    displacements = {
        vertex: abs(positions[vertex] - vertex)
        for vertex in range(9)
    }
    moved = max(displacements, key=displacements.get)
    has_movement = displacements[moved] > 0
    mechanism, _lines, footer = CATALOG_COPY[algorithm_id]
    details = catalog_details(payload, algorithm_id, order)
    heading_css = (
        "label"
        if approximate_text_width(mechanism, 22, bold=True) > 250
        else "heading"
    )
    cells = [
        '        <mxCell id="0" />',
        '        <mxCell id="1" parent="0" />',
        mx_cell("title", f"{algorithm_id}. {CATALOG_NAMES[algorithm_id]}", "text;html=1;strokeColor=none;fillColor=none;fontSize=30;fontStyle=1;fontColor=#27313A;", 42, 15, 700, 42),
        mx_cell("subtitle", mechanism, "text;html=1;strokeColor=none;fillColor=none;fontSize=16;fontColor=#27313A;", 42, 58, 1050, 28),
        mx_cell("graph-card", "", "rounded=1;html=1;fillColor=#FFFFFF;strokeColor=#9AA3AD;strokeWidth=2;", 35, 125, 280, 230),
        mx_cell("order-card", "", "rounded=1;html=1;fillColor=#F8F6EC;strokeColor=#1769C2;strokeWidth=2;", 365, 125, 450, 230),
        mx_cell("mechanism-card", "", "rounded=1;html=1;fillColor=#EEE9FF;strokeColor=#27313A;strokeWidth=2;", 865, 125, 300, 230),
        mx_cell("graph-heading", "Same input graph", "text;html=1;strokeColor=none;fillColor=none;fontSize=20;fontStyle=1;fontColor=#27313A;", 60, 140, 220, 35),
        mx_cell("order-heading", "Measured output order", "text;html=1;strokeColor=none;fillColor=none;fontSize=20;fontStyle=1;fontColor=#27313A;", 390, 140, 300, 35),
        mx_cell(
            "mechanism-heading",
            mechanism,
            (
                "text;html=1;strokeColor=none;fillColor=none;"
                f"fontSize={18 if heading_css == 'label' else 20};"
                "fontStyle=1;fontColor=#27313A;"
            ),
            920,
            140,
            220,
            35,
        ),
        mx_cell("footer", footer, "rounded=1;html=1;fillColor=#F8F6EC;strokeColor=#27313A;strokeWidth=2;fontSize=14;fontColor=#27313A;align=center;", 144, 380, 912, 36),
    ]
    graph_positions = {
        vertex: (
            25 + int(x * 0.48),
            150 + int(y * 0.48),
        )
        for vertex, (x, y) in GRAPH_POSITIONS.items()
    }
    for index, (source, target) in enumerate(
        payload["graph"]["undirected_edges"]
    ):
        cells.append(
            mx_edge(
                f"edge-{index}",
                f"node-{source}",
                f"node-{target}",
            )
        )
    for vertex in payload["graph"]["vertices"]:
        x, y = graph_positions[vertex]
        highlighted = has_movement and vertex == moved
        diameter = 46 if highlighted else 40
        cells.append(
            mx_cell(
                f"node-{vertex}",
                str(vertex),
                (
                    "ellipse;html=1;aspect=fixed;align=center;"
                    f"fillColor={GREEN if highlighted else NEUTRAL};"
                    f"strokeColor={ACTION if highlighted else INK};"
                    f"strokeWidth={3 if highlighted else 2};"
                    "fontSize=18;fontStyle=1;fontColor=#27313A;"
                ),
                x - diameter / 2,
                y - diameter / 2,
                diameter,
                diameter,
            )
        )
    cells.append(
        mx_cell(
            "movement",
            (
                f"v{moved}: old {moved} -> new {positions[moved]}"
                if has_movement
                else "identity: no vertex moves"
            ),
            "text;html=1;strokeColor=none;fillColor=none;fontSize=14;fontFamily=Consolas;fontColor=#27313A;",
            60,
            315,
            230,
            28,
        )
    )
    cell_width = 400 / 9
    for index, vertex in enumerate(order):
        highlighted = has_movement and vertex == moved
        fill = GREEN if highlighted else PAGE
        stroke = ACTION if highlighted else INK
        cells.append(
            mx_cell(
                f"order-{index}",
                f"v{vertex}",
                (
                    "rounded=0;html=1;align=center;verticalAlign=middle;"
                    f"fillColor={fill};strokeColor={stroke};"
                    f"strokeWidth={'3' if highlighted else '1.5'};"
                    "fontSize=16;fontColor=#27313A;"
                ),
                390 + index * cell_width,
                190,
                cell_width,
                58,
            )
        )
    cells.append(
        mx_cell(
            "permutation",
            f"permutation = {order}",
            "text;html=1;strokeColor=none;fillColor=none;fontSize=14;fontFamily=Consolas;fontColor=#27313A;",
            390,
            260,
            400,
            28,
        )
    )
    for index, line in enumerate(details[:3]):
        cells.append(
            mx_cell(
                f"detail-{index}",
                line,
                "text;html=1;strokeColor=none;fillColor=none;fontSize=15;fontColor=#27313A;",
                890,
                200 + index * 32,
                250,
                28,
            )
        )
    return (
        f'  <diagram id="graphbrew-{algorithm_id}" '
        f'name="{esc(CATALOG_NAMES[algorithm_id])}">\n'
        '    <mxGraphModel dx="1200" dy="430" grid="1" gridSize="10" '
        'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        'pageScale="1" pageWidth="1200" pageHeight="430" math="0" shadow="0">\n'
        "      <root>\n"
        + "\n".join(cells)
        + "\n      </root>\n    </mxGraphModel>\n  </diagram>"
    )


def drawio_catalog_input(payload: dict) -> str:
    cells = [
        '        <mxCell id="0" />',
        '        <mxCell id="1" parent="0" />',
        mx_cell("title", "Shared catalog input", "text;html=1;strokeColor=none;fillColor=none;fontSize=30;fontStyle=1;fontColor=#27313A;", 42, 15, 700, 42),
        mx_cell("subtitle", "Same 9 vertices, 12 edges, original order, and degree row.", "text;html=1;strokeColor=none;fillColor=none;fontSize=16;fontColor=#27313A;", 42, 58, 1050, 28),
        mx_cell("graph-card", "", "rounded=1;html=1;fillColor=#EDF5FF;strokeColor=#27313A;strokeWidth=3;", 35, 120, 540, 240),
        mx_cell("order-card", "", "rounded=1;html=1;fillColor=#FFF0D8;strokeColor=#27313A;strokeWidth=3;", 610, 120, 555, 240),
    ]
    positions = {
        vertex: (
            45 + int(x * 0.78),
            120 + int(y * 0.78),
        )
        for vertex, (x, y) in GRAPH_POSITIONS.items()
    }
    for index, (source, target) in enumerate(
        payload["graph"]["undirected_edges"]
    ):
        cells.append(
            mx_edge(
                f"edge-{index}",
                f"node-{source}",
                f"node-{target}",
            )
        )
    for vertex in payload["graph"]["vertices"]:
        x, y = positions[vertex]
        tracked = vertex == payload["tracked_vertex"]["old_id"]
        diameter = 46 if tracked else 40
        cells.append(
            mx_cell(
                f"node-{vertex}",
                str(vertex),
                (
                    "ellipse;html=1;aspect=fixed;align=center;"
                    f"fillColor={GREEN if tracked else NEUTRAL};"
                    f"strokeColor={ACTION if tracked else INK};"
                    f"strokeWidth={3 if tracked else 2};"
                    "fontSize=18;fontStyle=1;fontColor=#27313A;"
                ),
                x - diameter / 2,
                y - diameter / 2,
                diameter,
                diameter,
            )
        )
    degrees = [len(adjacency(payload)[vertex]) for vertex in range(9)]
    for row, values in enumerate((list(range(9)), degrees)):
        for index, value in enumerate(values):
            cells.append(
                mx_cell(
                    f"row-{row}-{index}",
                    f"v{value}" if row == 0 else str(value),
                    "rounded=0;html=1;fillColor=#FFFFFF;strokeColor=#27313A;strokeWidth=1.5;fontSize=16;fontColor=#27313A;align=center;",
                    635 + index * (505 / 9),
                    185 + row * 62,
                    505 / 9,
                    52,
                )
            )
    return (
        '  <diagram id="graphbrew-input" name="Shared input">\n'
        '    <mxGraphModel dx="1200" dy="430" grid="1" gridSize="10" '
        'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        'pageScale="1" pageWidth="1200" pageHeight="430" math="0" shadow="0">\n'
        "      <root>\n"
        + "\n".join(cells)
        + "\n      </root>\n    </mxGraphModel>\n  </diagram>"
    )


def generate_catalog_markdown(payload: dict) -> str:
    groups = [
        ("Baselines", [0, 1]),
        ("Degree and bucket layouts", list(range(2, 8))),
        ("Community and locality layouts", list(range(8, 13))),
        ("External and selected layouts", [13, 14, 15]),
        ("Directed layout", [16]),
    ]
    lines = [
        "# Reordering Figure Catalog",
        "",
        "Every strip uses the same measured 9-vertex input and the same converter",
        "binary. The shared input is shown once; each algorithm then shows only its",
        "measured output order and the mechanism that produced it.",
        "",
        "[![Shared catalog input](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/example-input.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/example-input.svg)",
        "",
        "**Shared-example contract.** All 17 outputs preserve the same nine vertices",
        "and twelve undirected edges. The blue-outlined cell is the vertex with the",
        "largest displacement in that measured output.",
        "",
        f"Converter SHA256: `{payload['catalog']['converter_sha256']}`.",
        "",
        (
            "[Capture receipt](https://github.com/UVA-LavaLab/GraphBrew/blob/"
            "main/docs/figures/catalog-capture.json) with commands, mapping "
            "fingerprints, and raw stdout tails."
        ),
        "",
        "Re-capture and regenerate:",
        "",
        "```bash",
        "python3 scripts/generate_public_figures.py --capture-catalog",
        "python3 scripts/generate_public_figures.py --check",
        "```",
        "",
        "[Download the generated 18-page draw.io bundle](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/GraphBrew-reordering-figures.drawio).",
        "",
        "On a narrow screen, select any figure to open the full-resolution SVG.",
        "",
    ]
    for group_name, algorithm_ids in groups:
        lines.extend([f"## {group_name}", ""])
        for algorithm_id in algorithm_ids:
            record = payload["catalog"]["algorithms"][str(algorithm_id)]
            mechanism, _details, footer = CATALOG_COPY[algorithm_id]
            name = CATALOG_NAMES[algorithm_id]
            slug = CATALOG_SLUGS[algorithm_id]
            lines.extend(
                [
                    f"### {algorithm_id}. {name}",
                    "",
                    f"- **CLI:** `{record['spec']}`",
                    f"- **Mechanism:** {mechanism}.",
                    (
                        "- **Evidence:** output order captured from the shared "
                        "example"
                        if algorithm_id != 14
                        else "- **Evidence:** selected-arm illustration; AdaptiveOrder has no fixed permutation"
                    ),
                    "",
                    (
                        f"[![{name} measured output](https://raw.githubusercontent.com/"
                        f"UVA-LavaLab/GraphBrew/main/docs/figures/reordering/"
                        f"{algorithm_id:02d}-{slug}.svg)](https://raw.githubusercontent.com/"
                        f"UVA-LavaLab/GraphBrew/main/docs/figures/reordering/"
                        f"{algorithm_id:02d}-{slug}.svg)"
                    ),
                    "",
                    f"**Figure.** {footer}",
                    "",
                    (
                        f"[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/"
                        f"docs/figures/editable/{algorithm_id:02d}-{slug}.drawio)."
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def generate_catalog_manifest(payload: dict) -> str:
    techniques = []
    for algorithm_id in range(17):
        record = payload["catalog"]["algorithms"][str(algorithm_id)]
        slug = CATALOG_SLUGS[algorithm_id]
        mechanism = CATALOG_COPY[algorithm_id][0]
        techniques.append(
            {
                "id": algorithm_id,
                "slug": slug,
                "name": CATALOG_NAMES[algorithm_id],
                "flag": record["spec"],
                "family": CATALOG_FAMILIES[algorithm_id],
                "mechanism": mechanism,
                "measured_order": record["order"],
                "svg": (
                    "docs/figures/reordering/"
                    f"{algorithm_id:02d}-{slug}.svg"
                ),
                "drawio": (
                    "docs/figures/editable/"
                    f"{algorithm_id:02d}-{slug}.drawio"
                ),
            }
        )
    payload_out = {
        "schema": "graphbrew-reordering-figures/v2",
        "generator": "scripts/generate_public_figures.py",
        "fixture": "docs/figures/graphbrew-running-example.json",
        "shared_input": {
            "svg": "docs/figures/reordering/example-input.svg",
            "drawio": "docs/figures/editable/example-input.drawio",
        },
        "converter_sha256": payload["catalog"]["converter_sha256"],
        "techniques": techniques,
    }
    return json.dumps(payload_out, indent=2) + "\n"


def write_outputs(payload: dict, target_root: Path) -> dict[Path, str]:
    outputs: dict[Path, str] = {
        target_root / "docs/figures/graphbrew-architecture.svg": generate_architecture(payload),
        target_root / "docs/figures/graphbrew-graph-to-csr.svg": generate_graph_to_csr(payload),
        target_root / "docs/figures/graphbrew-leiden-transform.svg": generate_partition(payload),
        target_root / "docs/figures/graphbrew-sizedesc-transform.svg": generate_size_desc(payload),
        target_root / "docs/figures/graphbrew-gordf5000.svg": generate_gordf(payload),
        target_root / "docs/figures/graphbrew-gorder-transform.svg": generate_gorder(payload),
        target_root / "docs/figures/graphbrew-bfs-transform.svg": generate_bfs(payload),
        target_root / "docs/figures/graphbrew-relabel-emit.svg": generate_relabel(payload),
        target_root / "docs/figures/graphbrew-locality-outcome.svg": generate_locality(payload),
        target_root / "docs/figures/graphbrew-cd-parallel.svg": generate_cd_parallel(payload),
        target_root / "docs/figures/graphbrew-sgmb4096.svg": generate_sgmb(payload),
        target_root / "docs/figures/graphbrew-norefine.svg": generate_norefine(payload),
        target_root / "docs/figures/graphbrew-lowreuse-policy.svg": generate_policy(payload),
        target_root / "docs/figures/reordering/example-input.svg": generate_catalog_input(payload),
    }
    for algorithm_id in range(17):
        name = f"{algorithm_id:02d}-{CATALOG_SLUGS[algorithm_id]}"
        outputs[
            target_root / f"docs/figures/reordering/{name}.svg"
        ] = generate_catalog_figure(payload, algorithm_id)
        page = drawio_catalog_page(payload, algorithm_id)
        outputs[
            target_root / f"docs/figures/editable/{name}.drawio"
        ] = (
            "<?xml version='1.0' encoding='utf-8'?>\n"
            '<mxfile host="app.diagrams.net" agent="GraphBrew" '
            'version="24.7.17" type="device">\n'
            f"{page}\n</mxfile>\n"
        )
    input_page = drawio_catalog_input(payload)
    outputs[target_root / "docs/figures/editable/example-input.drawio"] = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<mxfile host="app.diagrams.net" agent="GraphBrew" '
        'version="24.7.17" type="device">\n'
        f"{input_page}\n</mxfile>\n"
    )
    pages = [input_page] + [
        drawio_catalog_page(payload, algorithm_id)
        for algorithm_id in range(17)
    ]
    outputs[
        target_root / "docs/figures/editable/GraphBrew-reordering-figures.drawio"
    ] = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<mxfile host="app.diagrams.net" agent="GraphBrew" '
        'version="24.7.17" type="device">\n'
        + "\n".join(pages)
        + "\n</mxfile>\n"
    )
    outputs[
        target_root / "wiki/Reordering-Figure-Catalog.md"
    ] = generate_catalog_markdown(payload)
    outputs[
        target_root / "docs/figures/reordering/manifest.json"
    ] = generate_catalog_manifest(payload)
    return outputs


def public_manifest(payload: dict, outputs: dict[Path, str]) -> dict:
    records = []
    for path, content in sorted(outputs.items(), key=lambda item: str(item[0])):
        if path.suffix not in {".svg", ".drawio", ".json", ".md"}:
            continue
        relative = path.relative_to(ROOT)
        records.append(
            {
                "path": str(relative),
                "kind": path.suffix[1:],
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "generator": "scripts/generate_public_figures.py",
                "fixture": str(FIXTURE.relative_to(ROOT)),
            }
        )
    return {
        "schema": "graphbrew-public-figures/v1",
        "running_example_schema": payload["schema"],
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (
                FIXTURE,
                CAPTURE,
                EXAMPLE_EDGE_LIST,
                EXAMPLE_MAP,
                ROOT / "scripts/generate_public_figures.py",
            )
        ],
        "records": records,
    }


def materialize(outputs: dict[Path, str]) -> None:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def check_outputs(outputs: dict[Path, str]) -> list[str]:
    mismatches = []
    for path, expected in outputs.items():
        if not path.is_file():
            mismatches.append(f"missing {path.relative_to(ROOT)}")
        elif path.read_text() != expected:
            mismatches.append(f"stale {path.relative_to(ROOT)}")
    return mismatches


def capture_catalog(payload: dict) -> dict:
    converter = ROOT / "bench/bin/converter"
    if not converter.is_file():
        raise RuntimeError("build bench/bin/converter before catalog capture")
    converter_sha = hashlib.sha256(converter.read_bytes()).hexdigest()
    input_sha = hashlib.sha256(EXAMPLE_EDGE_LIST.read_bytes()).hexdigest()
    map_sha = hashlib.sha256(EXAMPLE_MAP.read_bytes()).hexdigest()
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "FALSE",
    })
    records = {}
    with tempfile.TemporaryDirectory(prefix="graphbrew-catalog-") as raw_tmp:
        tmp = Path(raw_tmp)
        for algorithm_id in range(17):
            if algorithm_id == 14:
                continue
            stored_spec = payload["catalog"]["algorithms"][
                str(algorithm_id)
            ]["spec"]
            runtime_spec = (
                f"13:{EXAMPLE_MAP}"
                if algorithm_id == 13
                else stored_spec
            )
            output_path = tmp / f"{algorithm_id}.lo"
            command = [
                str(converter),
                "-f",
                str(EXAMPLE_EDGE_LIST),
                "-s",
                "-o",
                runtime_spec,
                "-q",
                str(output_path),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
                timeout=300,
            )
            order = [int(value) for value in output_path.read_text().split()]
            fingerprint_match = re.findall(
                r"Composed Mapping Fingerprint:([0-9a-f]+)",
                completed.stdout,
            )
            if not fingerprint_match:
                fingerprint_match = re.findall(
                    r"Mapping Fingerprint:\s*([0-9a-f]+)",
                    completed.stdout,
                )
            if not fingerprint_match:
                raise RuntimeError(
                    f"mapping fingerprint missing for algorithm {algorithm_id}"
                )
            payload["catalog"]["algorithms"][str(algorithm_id)][
                "order"
            ] = order
            records[str(algorithm_id)] = {
                "spec": stored_spec,
                "runtime_spec": runtime_spec,
                "order": order,
                "mapping_fingerprint": fingerprint_match[-1],
                "command": command,
                "command_template": [
                    "bench/bin/converter",
                    "-f",
                    str(EXAMPLE_EDGE_LIST.relative_to(ROOT)),
                    "-s",
                    "-o",
                    (
                        "13:docs/figures/data/graphbrew-running-example.lo"
                        if algorithm_id == 13
                        else stored_spec
                    ),
                    "-q",
                    f"<output>/{algorithm_id}.lo",
                ],
                "stdout_tail": completed.stdout.splitlines()[-20:],
            }
    payload["catalog"]["algorithms"]["14"]["order"] = list(
        payload["catalog"]["algorithms"]["12"]["order"]
    )
    payload["catalog"]["converter_sha256"] = converter_sha
    payload["catalog"]["omp_threads"] = 1
    receipt = {
        "schema": "graphbrew-catalog-capture/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "relevant_worktree_diff_sha256": hashlib.sha256(
            subprocess.check_output(
                [
                    "git",
                    "diff",
                    "--",
                    "Makefile",
                    "bench/include",
                    "bench/src/converter.cc",
                ],
                cwd=ROOT,
            )
        ).hexdigest(),
        "converter": str(converter.relative_to(ROOT)),
        "converter_sha256": converter_sha,
        "input": str(EXAMPLE_EDGE_LIST.relative_to(ROOT)),
        "input_sha256": input_sha,
        "map_input": str(EXAMPLE_MAP.relative_to(ROOT)),
        "map_sha256": map_sha,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OMP_DYNAMIC": "FALSE",
        },
        "records": records,
        "adaptive_order": {
            "status": "selected-arm illustration",
            "order_source_algorithm": 12,
            "order": payload["catalog"]["algorithms"]["14"]["order"],
        },
    }
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n")
    CAPTURE.write_text(json.dumps(receipt, indent=2) + "\n")
    validate_fixture(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--capture-catalog", action="store_true")
    args = parser.parse_args()
    payload = load_fixture()
    if args.capture_catalog:
        payload = capture_catalog(payload)
    outputs = write_outputs(payload, ROOT)
    manifest_path = ROOT / "docs/figures/public-manifest.json"
    manifest = public_manifest(payload, outputs)
    outputs[manifest_path] = json.dumps(manifest, indent=2) + "\n"
    if args.check:
        mismatches = check_outputs(outputs)
        if mismatches:
            raise SystemExit("\n".join(mismatches))
        print(f"{len(outputs)} public figure artifacts are current")
        return
    materialize(outputs)
    print(f"generated {len(outputs)} public figure artifacts")


if __name__ == "__main__":
    main()
