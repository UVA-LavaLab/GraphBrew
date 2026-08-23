"""Bind public figures, editable pages, and wiki prose to one example fixture."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from scripts.lib.core.utils import ALGORITHMS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES = PROJECT_ROOT / "docs/figures"
MANIFEST = FIGURES / "reordering/manifest.json"
PUBLIC_MANIFEST = FIGURES / "public-manifest.json"
FIXTURE = FIGURES / "graphbrew-running-example.json"
CAPTURE = FIGURES / "catalog-capture.json"
CATALOG = PROJECT_ROOT / "wiki/Reordering-Figure-Catalog.md"
RUNNING_EXAMPLE = PROJECT_ROOT / "wiki/GraphBrew-Running-Example.md"
EDITABLE = FIGURES / "editable"
EDITABLE_BUNDLE = EDITABLE / "GraphBrew-reordering-figures.drawio"
GENERATOR = PROJECT_ROOT / "scripts/generate_public_figures.py"

TOP_LEVEL_FIGURES = {
    "graphbrew-architecture.svg",
    "graphbrew-graph-to-csr.svg",
    "graphbrew-leiden-transform.svg",
    "graphbrew-sizedesc-transform.svg",
    "graphbrew-gordf5000.svg",
    "graphbrew-gorder-transform.svg",
    "graphbrew-bfs-transform.svg",
    "graphbrew-relabel-emit.svg",
    "graphbrew-locality-outcome.svg",
    "graphbrew-cd-parallel.svg",
    "graphbrew-sgmb4096.svg",
    "graphbrew-norefine.svg",
    "graphbrew-lowreuse-policy.svg",
}


def test_public_figure_generator_is_current():
    subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_catalog_uses_one_measured_input_and_output_strips():
    payload = json.loads(MANIFEST.read_text())
    assert payload["schema"] == "graphbrew-reordering-figures/v2"
    techniques = payload["techniques"]
    assert [row["id"] for row in techniques] == list(range(17))
    assert {row["id"]: row["name"] for row in techniques} == ALGORITHMS
    assert Path(payload["shared_input"]["svg"]).name == "example-input.svg"

    fixture = json.loads(FIXTURE.read_text())
    catalog = CATALOG.read_text()
    assert "reordering/example-input.svg" in catalog
    assert payload["converter_sha256"] == fixture["catalog"]["converter_sha256"]

    for row in techniques:
        svg = PROJECT_ROOT / row["svg"]
        drawio = PROJECT_ROOT / row["drawio"]
        root = ET.parse(svg).getroot()
        assert root.get("width") == "1200"
        assert root.get("height") == "360"
        source = svg.read_text()
        assert 'role="img"' in source
        assert "prefers-color-scheme:dark" in source
        assert ".arrow{stroke:#63A8FF!important}" in source
        order_match = re.search(r"permutation = \[([0-9, ]+)\]", source)
        assert order_match
        order = [int(value) for value in order_match.group(1).split(",")]
        assert order == row["measured_order"]
        assert sorted(order) == list(range(9))

        drawio_root = ET.parse(drawio).getroot()
        diagram = drawio_root.find("diagram")
        assert diagram is not None
        cells = diagram.findall("./mxGraphModel/root/mxCell")
        for cell in cells:
            geometry = cell.find("mxGeometry")
            if geometry is None:
                continue
            for key in ("x", "y", "width", "height"):
                if key in geometry.attrib:
                    float(geometry.attrib[key])
        output_cells = sorted(
            (
                cell for cell in cells
                if re.fullmatch(r"order-\d+", cell.get("id", ""))
            ),
            key=lambda cell: int(cell.get("id", "").split("-")[1]),
        )
        assert [cell.get("value") for cell in output_cells] == [
            f"v{vertex}" for vertex in order
        ]

        assert row["name"] in catalog
        assert Path(row["svg"]).name in catalog
        assert Path(row["drawio"]).name in catalog


def test_editable_bundle_is_generated_from_all_pages():
    bundle = ET.parse(EDITABLE_BUNDLE).getroot()
    bundle_diagrams = bundle.findall("diagram")
    assert len(bundle_diagrams) == 18

    standalone = [EDITABLE / "example-input.drawio"] + sorted(
        EDITABLE.glob("[0-9][0-9]-*.drawio")
    )
    assert len(standalone) == 18
    input_diagram = ET.parse(standalone[0]).getroot().find("diagram")
    assert input_diagram is not None
    input_cells = input_diagram.findall("./mxGraphModel/root/mxCell")
    assert sum(
        re.fullmatch(r"node-\d+", cell.get("id", "")) is not None
        for cell in input_cells
    ) == 9
    assert sum(
        re.fullmatch(r"edge-\d+", cell.get("id", "")) is not None
        for cell in input_cells
    ) == 12

    def normalized(element):
        return (
            element.tag,
            tuple(sorted(element.attrib.items())),
            (element.text or "").strip(),
            tuple(normalized(child) for child in element),
        )

    for path, bundled in zip(standalone, bundle_diagrams):
        diagram = ET.parse(path).getroot().find("diagram")
        assert diagram is not None
        assert normalized(diagram) == normalized(bundled)


def test_running_example_drives_every_stage():
    payload = json.loads(FIXTURE.read_text())
    assert payload["schema"] == "graphbrew-running-example/v1"
    assert payload["composition"]["final_order"] == [
        2, 1, 4, 6, 7, 8, 5, 0, 3,
    ]
    assert payload["composition"]["forward_mapping"] == [
        7, 1, 0, 8, 2, 6, 3, 4, 5,
    ]
    assert payload["composition"]["block_only_order"] == [
        1, 2, 4, 6, 7, 0, 3, 5, 8,
    ]
    assert payload["composition"]["block_only_forward_mapping"] == [
        5, 0, 1, 6, 2, 7, 3, 4, 8,
    ]
    assert payload["tracked_vertex"]["old_cache_lines"] == 3
    assert payload["tracked_vertex"]["new_cache_lines"] == 2

    source = RUNNING_EXAMPLE.read_text()
    for stage in range(1, 7):
        assert f"## {stage}." in source
    for filename in (
        "graphbrew-graph-to-csr.svg",
        "graphbrew-leiden-transform.svg",
        "graphbrew-sizedesc-transform.svg",
        "graphbrew-gordf5000.svg",
        "graphbrew-gorder-transform.svg",
        "graphbrew-bfs-transform.svg",
        "graphbrew-relabel-emit.svg",
        "graphbrew-locality-outcome.svg",
    ):
        assert filename in source
    assert "`v2`" in source
    assert "manually frozen after Stage 2" in source
    assert "`[v2,v1,v4,v6,v7,v8,v5,v0,v3]`" in source


def test_top_level_figures_share_the_visual_contract():
    paths = sorted(FIGURES.glob("graphbrew-*.svg"))
    assert {path.name for path in paths} == TOP_LEVEL_FIGURES
    for path in paths:
        root = ET.parse(path).getroot()
        source = path.read_text()
        assert root.get("width") == "1200"
        assert float(root.get("width", "0")) / float(root.get("height", "1")) >= 1.4
        assert 'role="img"' in source
        assert "<title" in source and "<desc" in source
        assert "prefers-color-scheme:dark" in source
        assert ".arrow{stroke:#63A8FF!important}" in source
        font_sizes = [
            int(value)
            for value in re.findall(r"font-size:(\d+)px", source)
        ]
        assert font_sizes and min(font_sizes) >= 14
        assert max(font_sizes) <= 32


def test_public_manifest_binds_generated_outputs():
    payload = json.loads(PUBLIC_MANIFEST.read_text())
    assert payload["schema"] == "graphbrew-public-figures/v1"
    paths = {record["path"] for record in payload["records"]}
    assert "wiki/Reordering-Figure-Catalog.md" in paths
    assert "docs/figures/reordering/manifest.json" in paths
    assert "docs/figures/reordering/example-input.svg" in paths
    assert "docs/figures/editable/example-input.drawio" in paths
    source_paths = {record["path"] for record in payload["sources"]}
    assert source_paths == {
        "docs/figures/graphbrew-running-example.json",
        "docs/figures/catalog-capture.json",
        "docs/figures/data/graphbrew-running-example.el",
        "docs/figures/data/graphbrew-running-example.lo",
        "scripts/generate_public_figures.py",
    }


def test_catalog_capture_binds_binary_input_map_and_orders():
    fixture = json.loads(FIXTURE.read_text())
    receipt = json.loads(CAPTURE.read_text())
    assert receipt["schema"] == "graphbrew-catalog-capture/v1"
    assert receipt["converter_sha256"] == fixture["catalog"]["converter_sha256"]
    assert receipt["environment"]["OMP_NUM_THREADS"] == "1"
    assert len(receipt["records"]) == 16
    for algorithm_id, record in fixture["catalog"]["algorithms"].items():
        if algorithm_id == "14":
            assert record["order"] == receipt["adaptive_order"]["order"]
            continue
        captured = receipt["records"][algorithm_id]
        assert captured["order"] == record["order"]
        assert re.fullmatch(
            r"[0-9a-f]{16}",
            captured["mapping_fingerprint"],
        )
