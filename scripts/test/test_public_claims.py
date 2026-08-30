"""Bind the public story and figures to the frozen paper evidence."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/recommendation-evidence.json"
FIGURE_DIR = PROJECT_ROOT / "docs/figures"
PUBLIC_MANIFEST = FIGURE_DIR / "public-manifest.json"
README = PROJECT_ROOT / "README.md"
HOME = PROJECT_ROOT / "wiki/Home.md"
CLAIMS = PROJECT_ROOT / "wiki/Evidence-and-Claims.md"
GRAPHBREW = PROJECT_ROOT / "wiki/GraphBrewOrder.md"
RUNNING_EXAMPLE = PROJECT_ROOT / "wiki/GraphBrew-Running-Example.md"
ADAPTIVE = PROJECT_ROOT / "wiki/AdaptiveOrder.md"
HISTORICAL_POLICY = (
    PROJECT_ROOT / "wiki/Historical-Low-Reuse-Policy.md"
)
SIDEBAR = PROJECT_ROOT / "wiki/_Sidebar.md"


def test_public_evidence_has_only_current_claims():
    payload = json.loads(EVIDENCE.read_text())
    assert payload["schema"] == "graphbrew-public-evidence/v2"
    claims = payload["confirmed_claims"]
    assert (
        claims["quality_arm"]["name"]
        == "LeidenGVE-SizeDesc-LocalGorder8"
    )
    assert claims["quality_arm"]["kernel_gm"] == {
        "rabbit_csr_over_graphbrew": 1.041969100132076,
        "rabbit_boost_over_graphbrew": 1.0443919202041194,
        "gorder_csr_over_graphbrew": 1.051836941035804,
    }
    assert (
        claims["quality_arm"]["mapping_gm"][
            "graphbrew_over_gorder_csr"
        ]
        < 1.0
    )
    assert (
        claims["compact_and_emit"]["mapping_only"][
            "five_graph_candidate_over_min_rabbit_mapping_gm"
        ]
        < 0.5
    )
    atlas = claims["composition_atlas"]
    assert atlas["in_sample_oracle_over_fastest_comparator_gm"] > 1.0
    assert atlas["best_fixed_over_fastest_comparator_gm"] < 1.0
    assert atlas["graph_held_out_over_fastest_comparator_gm"] < 1.0
    dependence = atlas["workload_dependence"]
    assert dependence["distinct_winning_compositions"] == 8
    assert dependence["cell_oracle_over_best_fixed_graphbrew_gm"] > 1.1
    assert (
        dependence[
            "postselected_graph_type_kernel_over_fastest_comparator_gm"
        ]
        > 1.0
    )
    assert (
        dependence[
            "held_out_graph_type_kernel_over_fastest_comparator_gm"
        ]
        < 1.0
    )
    sealed = claims["sealed_composability"]
    assert sealed["distinct_winning_compositions"] == 7
    assert sealed["candidate_compositions"] == 7
    assert sealed["cell_oracle_over_best_fixed_graphbrew_gm"] > 1.2
    assert sealed["fastest_comparator_over_cell_oracle_gm"] > 1.1
    assert sealed["frozen_family_kernel_over_fastest_comparator_gm"] < 1.0
    assert sealed["gorder_csr_cell_gm_crossover_reuse"] == 67
    assert not (PROJECT_ROOT / "docs/allkernel-lowreuse-evidence.json").exists()


def test_public_story_matches_claim_boundary():
    texts = {
        path.name: path.read_text()
        for path in (
            README,
            HOME,
            CLAIMS,
            GRAPHBREW,
            RUNNING_EXAMPLE,
            ADAPTIVE,
            HISTORICAL_POLICY,
            SIDEBAR,
        )
    }
    story = "\n".join(texts.values())

    for required in (
        "LeidenGVE-SizeDesc-LocalGorder8",
        "Compact-and-Emit",
        "1.042x",
        "1.044x",
        "1.052x",
        "0.752x",
        "1.229x",
        "0.896x",
    ):
        assert required in story

    assert "Evidence and Claims" in texts["Home.md"]
    assert "Evidence and Claims" in texts["_Sidebar.md"]
    assert "Low-Reuse Selector" not in texts["_Sidebar.md"]
    assert "not a paper contribution" in texts["AdaptiveOrder.md"]
    assert "Historical Low-Reuse Policy" in (
        texts["Historical-Low-Reuse-Policy.md"]
    )

    for stale in (
        "validated policy today",
        "proposed Rabbit-free composition generator",
        "graphbrew-public-v3",
        "graphbrew-lowreuse-policy.svg",
    ):
        assert stale not in story


def test_claim_figures_are_manifest_bound():
    figure_names = (
        "graphbrew-architecture.svg",
        "graphbrew-evidence-boundary.svg",
        "graphbrew-compact-emit.svg",
    )
    manifest = json.loads(PUBLIC_MANIFEST.read_text())
    hashes = {
        record["path"]: record["sha256"]
        for record in manifest["records"]
    }
    for filename in figure_names:
        path = FIGURE_DIR / filename
        assert path.stat().st_size > 3000
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        assert root.get("data-figure-schema") == "graphbrew-public/v4"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[
            f"docs/figures/{filename}"
        ]

    assert not (FIGURE_DIR / "graphbrew-lowreuse-policy.svg").exists()
