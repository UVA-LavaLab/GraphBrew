"""Bind the low-reuse visual explanation to public evidence."""

import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/allkernel-lowreuse-evidence.json"
FIGURE_DIR = PROJECT_ROOT / "docs/figures"
PUBLIC_FIGURE_MANIFEST = FIGURE_DIR / "public-manifest.json"
FIGURES = (
    "graphbrew-architecture.svg",
    "graphbrew-lowreuse-policy.svg",
    "graphbrew-leiden-transform.svg",
    "graphbrew-sizedesc-transform.svg",
    "graphbrew-gorder-transform.svg",
    "graphbrew-bfs-transform.svg",
    "graphbrew-cd-parallel.svg",
    "graphbrew-sgmb4096.svg",
    "graphbrew-gordf5000.svg",
    "graphbrew-norefine.svg",
)
MECHANISM_FIGURES = (
    "graphbrew-leiden-transform.svg",
    "graphbrew-sizedesc-transform.svg",
    "graphbrew-gorder-transform.svg",
    "graphbrew-bfs-transform.svg",
    "graphbrew-cd-parallel.svg",
    "graphbrew-sgmb4096.svg",
    "graphbrew-gordf5000.svg",
    "graphbrew-norefine.svg",
)
README = PROJECT_ROOT / "README.md"
WIKI_HOME = PROJECT_ROOT / "wiki/Home.md"
WIKI_SIDEBAR = PROJECT_ROOT / "wiki/_Sidebar.md"
SELECTOR_PAGE = PROJECT_ROOT / "wiki/All-Kernel-Low-Reuse-Selector.md"
GRAPHBREW_PAGE = PROJECT_ROOT / "wiki/GraphBrewOrder.md"
RUNNING_EXAMPLE_PAGE = PROJECT_ROOT / "wiki/GraphBrew-Running-Example.md"
ADAPTIVE_PAGE = PROJECT_ROOT / "wiki/AdaptiveOrder.md"
LEGACY_PAGE = PROJECT_ROOT / "wiki/AdaptiveOrder-ML.md"
REMOVED_FIGURES = (
    "allkernel-selector-feature-map.svg",
    "allkernel-selector-holdout-speedup.svg",
    "allkernel-selector-cost-breakdown.svg",
)


def _rule2_records(payload):
    return [
        row
        for row in payload["records"]
        if row["phase"] == "rule2_holdout"
    ]


def test_allkernel_selector_evidence_gates():
    payload = json.loads(EVIDENCE.read_text())
    records = payload["records"]

    assert len(records) == 30
    assert {
        phase: sum(row["phase"] == phase for row in records)
        for phase in {"derivation", "rule1_holdout", "rule2_holdout"}
    } == {
        "derivation": 11,
        "rule1_holdout": 7,
        "rule2_holdout": 12,
    }

    final_selected = [
        row
        for row in _rule2_records(payload)
        if row["selected_strategy"] == "FastLeiden-Gorder8"
    ]
    final_fallback = [
        row
        for row in _rule2_records(payload)
        if row["selected_strategy"] == "Rabbit Boost"
    ]
    assert len(final_selected) == 7
    assert len(final_fallback) == 5
    assert all(
        row["boost_over_candidate_reuse1"] > 1.0
        and row["boost_over_candidate_reuse2"] > 1.0
        for row in final_selected
    )
    assert all(
        row["selected_strategy_speedup_reuse1"] == 1.0
        and row["selected_strategy_speedup_reuse2"] == 1.0
        for row in final_fallback
    )


def test_selector_documentation_uses_evidence_bound_figures():
    payload = json.loads(EVIDENCE.read_text())
    selector = SELECTOR_PAGE.read_text()
    graphbrew = GRAPHBREW_PAGE.read_text()
    running_example = RUNNING_EXAMPLE_PAGE.read_text()
    adaptive = ADAPTIVE_PAGE.read_text()
    readme = README.read_text()
    home = WIKI_HOME.read_text()
    sidebar = WIKI_SIDEBAR.read_text()
    legacy = LEGACY_PAGE.read_text()
    adaptive_flat = " ".join(adaptive.split())

    for filename in FIGURES:
        path = FIGURE_DIR / filename
        assert path.stat().st_size > 3000
        assert ET.parse(path).getroot().tag.endswith("svg")

    architecture = FIGURE_DIR / "graphbrew-architecture.svg"
    public_manifest = json.loads(PUBLIC_FIGURE_MANIFEST.read_text())
    manifest_hashes = {
        record["path"]: record["sha256"]
        for record in public_manifest["records"]
    }
    assert hashlib.sha256(architecture.read_bytes()).hexdigest() == (
        manifest_hashes["docs/figures/graphbrew-architecture.svg"]
    )

    for filename in MECHANISM_FIGURES:
        path = FIGURE_DIR / filename
        source = path.read_text()
        font_sizes = [
            int(value)
            for value in re.findall(
                r"font-size:(\d+)px",
                source,
            )
        ]
        assert font_sizes and min(font_sizes) >= 14
        assert max(font_sizes) <= 32
        assert 'stroke="#9AA3AD"' in source
        assert any(
            fill in source
            for fill in (
                'fill="#FFF0D8"',
                'fill="#EDF5FF"',
                'fill="#E7F7EA"',
                'fill="#F7DEDC"',
                'fill="#EEE9FF"',
            )
        )
        assert "#1769C2" in source
        assert "prefers-color-scheme:dark" in source

    assert "graphbrew-lowreuse-policy.svg" in selector
    assert "graphbrew-leiden-transform.svg" in running_example
    assert "graphbrew-sizedesc-transform.svg" in running_example
    assert "graphbrew-gorder-transform.svg" in running_example
    assert "graphbrew-bfs-transform.svg" in running_example
    assert "graphbrew-relabel-emit.svg" in running_example
    assert "graphbrew-locality-outcome.svg" in running_example
    assert "graphbrew-cd-parallel.svg" in graphbrew
    assert "graphbrew-sgmb4096.svg" in graphbrew
    assert "graphbrew-gordf5000.svg" in graphbrew
    assert "graphbrew-norefine.svg" in graphbrew
    assert "graphbrew-architecture.svg" in graphbrew
    assert "graphbrew-graph-to-csr.svg" in graphbrew
    assert "graphbrew-leiden-transform.svg" in graphbrew
    assert payload["candidate"] in selector.replace("\n", "")
    assert payload["candidate"] in graphbrew.replace("\n", "")
    for threshold in ("3.2", "2.68", "60", "0.82", "8"):
        assert threshold in selector

    for row in _rule2_records(payload):
        assert row["graph"] in selector
    assert "does not run the kernel" in selector
    assert "not Rabbit-free" in selector
    assert "Adaptive Feature Time" in selector
    assert "community super-nodes" in graphbrew
    assert "whole-graph cutoff" in graphbrew

    assert "AdaptiveOrder](AdaptiveOrder)" in home
    assert "GraphBrewOrder](GraphBrewOrder)" in sidebar
    assert "Low-Reuse Selector](All-Kernel-Low-Reuse-Selector)" in sidebar
    assert "/wiki/AdaptiveOrder)" in readme
    assert "not a machine-learning model" in adaptive_flat
    assert "runtime ML selector" not in adaptive
    assert "AdaptiveOrder](AdaptiveOrder)" in legacy
    assert len(legacy.splitlines()) < 10

    public_story = "\n".join(
        (readme, home, selector, graphbrew, running_example, adaptive)
    )
    for figure in REMOVED_FIGURES:
        assert figure not in public_story
