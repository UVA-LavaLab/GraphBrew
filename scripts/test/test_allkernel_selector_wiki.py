"""Keep the selector wiki page synchronized with public evidence."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/allkernel-lowreuse-evidence.json"
WIKI_PAGE = PROJECT_ROOT / "wiki/All-Kernel-Low-Reuse-Selector.md"
FIGURES = (
    "allkernel-selector-feature-map.svg",
    "allkernel-selector-holdout-speedup.svg",
    "allkernel-selector-cost-breakdown.svg",
)


def test_allkernel_selector_wiki_is_evidence_bound():
    payload = json.loads(EVIDENCE.read_text())
    records = payload["records"]
    assert len(records) == 30
    assert {
        phase: sum(row["phase"] == phase for row in records)
        for phase in {
            "derivation",
            "rule1_holdout",
            "rule2_holdout",
        }
    } == {
        "derivation": 11,
        "rule1_holdout": 7,
        "rule2_holdout": 12,
    }

    final_selected = [
        row for row in records
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

    wiki = WIKI_PAGE.read_text()
    for row in records:
        assert row["graph"] in wiki
    for figure in FIGURES:
        path = PROJECT_ROOT / "docs/figures" / figure
        assert path.stat().st_size > 1000
        assert figure in wiki
