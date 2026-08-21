"""Keep the SVG and Lucidchart-editable reordering catalog complete."""

import json
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
        assert font_sizes and min(font_sizes) >= 20
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

        assert row["name"] in catalog
        assert Path(row["svg"]).name in catalog
        assert Path(row["drawio"]).name in catalog


def test_lucidchart_bundle_contains_every_figure():
    bundle = ET.parse(EDITABLE_BUNDLE).getroot()
    assert bundle.tag == "mxfile"
    assert len(bundle.findall("diagram")) == len(ALGORITHMS)

    instructions = EDITABLE_README.read_text()
    assert "Import documents" in instructions
    assert "16389149809428-Import-files-into-Lucidchart" in instructions
    assert "GraphBrew-reordering-figures.drawio" in instructions
