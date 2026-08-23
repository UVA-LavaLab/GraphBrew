"""Keep the SVG and Lucidchart-editable reordering catalog complete."""

import json
import html
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from scripts.lib.core.utils import ALGORITHMS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "docs/figures/reordering/manifest.json"
CATALOG = PROJECT_ROOT / "wiki/Reordering-Figure-Catalog.md"
EDITABLE_README = PROJECT_ROOT / "docs/figures/editable/README.md"
EDITABLE_BUNDLE = (
    PROJECT_ROOT
    / "docs/figures/editable/GraphBrew-reordering-figures.drawio"
)
GRAPHBREW_FIGURES = PROJECT_ROOT / "docs/figures"
GRAPH_TRANSFORMS = {
    "graphbrew-bfs-transform.svg",
    "graphbrew-gorder-transform.svg",
    "graphbrew-leiden-transform.svg",
    "graphbrew-sizedesc-transform.svg",
}


def test_reordering_figure_catalog_covers_every_algorithm_id():
    payload = json.loads(MANIFEST.read_text())
    techniques = payload["techniques"]

    assert [row["id"] for row in techniques] == list(range(17))
    assert {
        row["id"]: row["name"]
        for row in techniques
    } == ALGORITHMS

    catalog = CATALOG.read_text()
    for row in techniques:
        svg = PROJECT_ROOT / row["svg"]
        drawio = PROJECT_ROOT / row["drawio"]

        svg_root = ET.parse(svg).getroot()
        assert svg_root.tag.endswith("svg")
        source = svg.read_text()
        font_sizes = [
            int(value)
            for value in re.findall(r"font-size:(\d+)px", source)
        ]
        assert font_sizes and min(font_sizes) >= 21
        svg_rects = [
            element
            for element in svg_root
            if element.tag.endswith("rect")
        ]
        assert any(
            rect.get("width") == svg_root.get("width")
            and rect.get("height") == svg_root.get("height")
            and rect.get("fill") == "#FFFFFF"
            for rect in svg_rects
        )
        graph_panels = [
            rect for rect in svg_rects
            if rect.get("class") == "graph-panel"
        ]
        assert len(graph_panels) == 2
        node_radii = [
            int(value)
            for value in re.findall(
                r'<circle[^>]+r="(\d+)"[^>]+stroke="#27313A"',
                source,
            )
        ]
        assert node_radii and min(node_radii) >= 27
        cards = ((55, 205, 590, 365), (755, 205, 590, 365))
        for cx, cy, radius in re.findall(
            r'<circle[^>]+cx="(\d+)" cy="(\d+)" r="(\d+)"'
            r'[^>]+stroke="#27313A"',
            source,
        ):
            x, y, r = map(int, (cx, cy, radius))
            assert any(
                left <= x - r
                and x + r <= left + width
                and top <= y - r
                and y + r <= top + height
                for left, top, width, height in cards
            )
        for cx, cy, rx, ry in re.findall(
            r'<ellipse[^>]+cx="(\d+)" cy="(\d+)" '
            r'rx="(\d+)" ry="(\d+)"',
            source,
        ):
            x, y, x_radius, y_radius = map(
                int, (cx, cy, rx, ry),
            )
            assert any(
                left <= x - x_radius
                and x + x_radius <= left + width
                and top <= y - y_radius
                and y + y_radius <= top + height
                for left, top, width, height in cards
            )
        assert 'stroke="#9AA3AD"' in source
        assert "#1769C2" in source
        assert "prefers-color-scheme:dark" in source

        drawio_root = ET.parse(drawio).getroot()
        assert drawio_root.tag == "mxfile"
        graph_root = drawio_root.find("./diagram/mxGraphModel/root")
        assert graph_root is not None
        cells = graph_root.findall("mxCell")
        assert sum(cell.get("vertex") == "1" for cell in cells) >= 30
        assert sum(cell.get("edge") == "1" for cell in cells) >= 20
        cells_by_id = {cell.get("id"): cell for cell in cells}
        required_ids = {
            "left-card",
            "right-card",
            "before-order-heading",
            "after-order-heading",
            "note",
            "note-main",
            "footer-detail",
        }
        assert required_ids <= set(cells_by_id)
        assert cells_by_id["before-order-heading"].get("value") == (
            "old vertices in input-ID order"
        )
        assert cells_by_id["after-order-heading"].get("value") == (
            "old vertices in new-ID order"
        )
        for card_id, x in (("left-card", "55"), ("right-card", "755")):
            geometry = cells_by_id[card_id].find("mxGeometry")
            assert geometry is not None
            assert geometry.attrib == {
                "x": x,
                "y": "205",
                "width": "590",
                "height": "365",
                "as": "geometry",
            }
        graph_nodes = [
            cell for cell in cells
            if re.fullmatch(r"[ab]n\d+", cell.get("id", ""))
        ]
        assert graph_nodes
        for node in graph_nodes:
            geometry = node.find("mxGeometry")
            assert geometry is not None
            assert float(geometry.get("width", "0")) >= 54
            assert "fontSize=24" in node.get("style", "")
            assert node.get("style", "").startswith("ellipse;")
            center = (
                round(
                    float(geometry.get("x", "0"))
                    + float(geometry.get("width", "0")) / 2
                ),
                round(
                    float(geometry.get("y", "0"))
                    + float(geometry.get("height", "0")) / 2
                ),
            )
            svg_nodes = {
                (int(cx), int(cy)): int(radius) * 2
                for cx, cy, radius in re.findall(
                    r'<circle[^>]+cx="(\d+)" cy="(\d+)" r="(\d+)"'
                    r'[^>]+stroke="#27313A"',
                    source,
                )
            }
            assert center in svg_nodes
            assert float(geometry.get("width", "0")) == svg_nodes[center]
        role_fonts = {
            "title": "fontSize=44",
            "subtitle": "fontSize=24",
            "domain": "fontSize=21",
            "before-heading": "fontSize=31",
            "after-heading": "fontSize=31",
            "arrow-label": "fontSize=22",
            "note-main": "fontSize=24",
            "footer-detail": "fontSize=21",
        }
        for cell_id, font in role_fonts.items():
            assert font in cells_by_id[cell_id].get("style", "")
            if cell_id != "arrow-label":
                assert cells_by_id[cell_id].get(
                    "style", ""
                ).startswith("text;")
        main_caption = re.search(
            r'<text x="700" y="607"[^>]*>(.*?)</text>',
            source,
        )
        detail_caption = re.search(
            r'<text x="700" y="638"[^>]*>(.*?)</text>',
            source,
        )
        assert main_caption and detail_caption
        assert len(html.unescape(main_caption.group(1))) <= 80
        assert cells_by_id["note-main"].get("value") == html.unescape(
            main_caption.group(1)
        )
        assert cells_by_id["footer-detail"].get("value") == html.unescape(
            detail_caption.group(1)
        )

        assert row["name"] in catalog
        assert Path(row["svg"]).name in catalog
        assert Path(row["drawio"]).name in catalog


def test_lucidchart_bundle_contains_every_figure():
    bundle = ET.parse(EDITABLE_BUNDLE).getroot()
    assert bundle.tag == "mxfile"
    assert len(bundle.findall("diagram")) == len(ALGORITHMS)
    bundle_diagrams = bundle.findall("diagram")
    for diagram in bundle_diagrams:
        graph_root = diagram.find("./mxGraphModel/root")
        assert graph_root is not None
        ids = {
            cell.get("id")
            for cell in graph_root.findall("mxCell")
        }
        assert {"left-card", "right-card"} <= ids
    for standalone_path, bundle_diagram in zip(
        sorted((PROJECT_ROOT / "docs/figures/editable").glob("[0-9][0-9]-*.drawio")),
        bundle_diagrams,
    ):
        standalone = ET.parse(standalone_path).getroot().find("diagram")
        assert standalone is not None
        assert ET.tostring(standalone) == ET.tostring(bundle_diagram)

    instructions = EDITABLE_README.read_text()
    assert "Import documents" in instructions
    assert "16389149809428-Import-files-into-Lucidchart" in instructions
    assert "GraphBrew-reordering-figures.drawio" in instructions
    assert "54 px regular" in instructions
    assert "60 px emphasized-hub" in instructions


def test_public_graphbrew_figures_follow_visual_contract():
    figures = sorted(GRAPHBREW_FIGURES.glob("graphbrew-*.svg"))
    assert len(figures) == 10
    for svg in figures:
        source = svg.read_text()
        root = ET.parse(svg).getroot()
        assert root.tag.endswith("svg")
        width = root.get("width")
        height = root.get("height")
        canvas = next(
            (
                element for element in root
                if element.tag.endswith("rect")
                and element.get("width") == width
                and element.get("height") == height
                and element.get("fill") == "#FFFFFF"
            ),
            None,
        )
        assert canvas is not None
        font_sizes = [
            int(value)
            for value in re.findall(r"font-size:(\d+)px", source)
        ]
        assert font_sizes
        minimum = 19 if svg.name == "graphbrew-architecture.svg" else 21
        assert min(font_sizes) >= minimum
        assert "prefers-color-scheme:dark" in source
        if svg.name in GRAPH_TRANSFORMS:
            assert source.count('class="graph-panel"') == 2
            node_radii = [
                int(value)
                for value in re.findall(
                    r'<circle[^>]+r="(\d+)"[^>]+class="node"',
                    source,
                )
            ]
            assert node_radii and min(node_radii) >= 27
