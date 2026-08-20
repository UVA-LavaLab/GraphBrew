"""Keep the concise selector documentation bound to public evidence."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/allkernel-lowreuse-evidence.json"
ARCHITECTURE = PROJECT_ROOT / "docs/figures/graphbrew-architecture.svg"
README = PROJECT_ROOT / "README.md"
WIKI_HOME = PROJECT_ROOT / "wiki/Home.md"
SELECTOR_PAGE = PROJECT_ROOT / "wiki/All-Kernel-Low-Reuse-Selector.md"
ADAPTIVE_PAGE = PROJECT_ROOT / "wiki/AdaptiveOrder.md"
LEGACY_PAGE = PROJECT_ROOT / "wiki/AdaptiveOrder-ML.md"
REMOVED_FIGURES = (
    "allkernel-selector-feature-map.svg",
    "allkernel-selector-holdout-speedup.svg",
    "allkernel-selector-cost-breakdown.svg",
)


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
        for row in records
        if (
            row["phase"] == "rule2_holdout"
            and row["selected_strategy"] == "FastLeiden-Gorder8"
        )
    ]
    assert len(final_selected) == 7
    assert all(
        row["boost_over_candidate_reuse1"] > 1.0
        and row["boost_over_candidate_reuse2"] > 1.0
        for row in final_selected
    )


def test_selector_documentation_uses_canonical_story():
    payload = json.loads(EVIDENCE.read_text())
    selector = SELECTOR_PAGE.read_text()
    adaptive = ADAPTIVE_PAGE.read_text()
    readme = README.read_text()
    home = WIKI_HOME.read_text()
    legacy = LEGACY_PAGE.read_text()
    adaptive_flat = " ".join(adaptive.split())

    assert ARCHITECTURE.stat().st_size > 1000
    assert "<svg" in ARCHITECTURE.read_text()
    assert "graphbrew-architecture.svg" in selector
    assert payload["candidate"] in selector.replace("\n", "")
    assert payload["predicate"] in selector.replace("\n", " ")
    assert "allkernel-lowreuse-evidence.json" in selector

    assert "AdaptiveOrder](AdaptiveOrder)" in home
    assert "/wiki/AdaptiveOrder)" in readme
    assert "not a machine-learning model" in adaptive_flat
    assert "runtime ML selector" not in adaptive
    assert "AdaptiveOrder](AdaptiveOrder)" in legacy
    assert len(legacy.splitlines()) < 10

    public_story = "\n".join((readme, home, selector, adaptive))
    for figure in REMOVED_FIGURES:
        assert figure not in public_story
