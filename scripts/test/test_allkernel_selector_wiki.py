"""Bind the low-reuse visual explanation to public evidence."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/allkernel-lowreuse-evidence.json"
FIGURE_DIR = PROJECT_ROOT / "docs/figures"
FIGURES = (
    "graphbrew-architecture.svg",
    "graphbrew-lowreuse-policy.svg",
    "graphbrew-relabeling-example.svg",
    "graphbrew-cost-controls.svg",
)
README = PROJECT_ROOT / "README.md"
WIKI_HOME = PROJECT_ROOT / "wiki/Home.md"
SELECTOR_PAGE = PROJECT_ROOT / "wiki/All-Kernel-Low-Reuse-Selector.md"
GRAPHBREW_PAGE = PROJECT_ROOT / "wiki/GraphBrewOrder.md"
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
    adaptive = ADAPTIVE_PAGE.read_text()
    readme = README.read_text()
    home = WIKI_HOME.read_text()
    legacy = LEGACY_PAGE.read_text()
    adaptive_flat = " ".join(adaptive.split())

    for filename in FIGURES:
        path = FIGURE_DIR / filename
        assert path.stat().st_size > 5000
        assert ET.parse(path).getroot().tag.endswith("svg")

    assert "graphbrew-lowreuse-policy.svg" in selector
    assert "graphbrew-relabeling-example.svg" in graphbrew
    assert "graphbrew-cost-controls.svg" in graphbrew
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
    assert "/wiki/AdaptiveOrder)" in readme
    assert "not a machine-learning model" in adaptive_flat
    assert "runtime ML selector" not in adaptive
    assert "AdaptiveOrder](AdaptiveOrder)" in legacy
    assert len(legacy.splitlines()) < 10

    public_story = "\n".join(
        (readme, home, selector, graphbrew, adaptive)
    )
    for figure in REMOVED_FIGURES:
        assert figure not in public_story
