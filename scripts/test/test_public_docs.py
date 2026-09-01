#!/usr/bin/env python3
"""Public docs and orchestrator help must describe the generic workflow."""

import json
import subprocess
import sys
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_public_docs_exclude_unreleased_material():
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "INDEX.md",
        PROJECT_ROOT / "docs" / "figures" / "README.md",
        PROJECT_ROOT / "scripts" / "README.md",
        PROJECT_ROOT / "scripts" / "experiments" / "README.md",
        *sorted((PROJECT_ROOT / "wiki").glob("*.md")),
    ]
    text = "\n".join(path.read_text() for path in paths)
    lower = text.lower()

    for forbidden in (
        "c++ trains models at runtime",
        "worst-case baseline",
        "vldb_paper_experiments.py",
        "paper/main.tex",
        "logo cv on all ml models",
        "paper story",
        "what the paper",
        "evidence and claims",
        "recommendation-evidence.json",
        "reproducing the frozen study",
        "frozen study reproduction",
        "frozen publication workflow",
        "headline paper contribution",
        "current paper",
        "paper experiments",
        "--paper-preview",
        "--paper-graph-dir",
        "--paper-artifact-root",
        "graphbrew-evidence-boundary.svg",
        "research direction",
        "research agenda",
        "1.229x",
        "0.896x",
        "1.052x",
        "0.752x",
    ):
        assert forbidden not in lower


def test_public_tree_excludes_private_and_agent_process_markers():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    text_extensions = {
        ".cc", ".cpp", ".h", ".hpp", ".md", ".py", ".txt",
        ".yml", ".yaml", ".ps1", ".hxx", ".sbatch", ".json",
        ".def", ".sh",
    }
    suffixless_names = {"Makefile", ".gitignore", ".gitattributes"}
    content = []
    for relative in result.stdout.splitlines():
        path = PROJECT_ROOT / relative
        if (
            path.suffix in text_extensions
            or path.name in suffixless_names
        ) and path.is_file():
            content.append(path.read_text(errors="ignore"))
    public_text = "\n".join(content)
    forbidden = (
        "claude" + "-opus",
        "gpt" + "-5",
        "opus" + "_review",
        "sol" + "_review",
        "/Users/" + "amughrabi",
    )
    assert not any(
        re.search(pattern, public_text, flags=re.IGNORECASE)
        for pattern in forbidden
    )


def test_public_docs_defer_specialized_campaign_reproduction():
    getting_started = (
        PROJECT_ROOT / "wiki" / "Getting-Started.md"
    ).read_text()
    reproducibility = (
        PROJECT_ROOT / "wiki" / "Reproducible-Experiments.md"
    ).read_text()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/graphbrew_experiment.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    public_text = getting_started + reproducibility + result.stdout
    assert "scripts/graphbrew_experiment.py" in public_text
    for deferred in (
        "--vldb",
        "--paper-preview",
        "--paper-graph-dir",
        "--paper-artifact-root",
        "--mechanism-discovery",
        "--mapping-forensics",
    ):
        assert deferred not in public_text


def test_unreleased_publication_artifacts_are_absent():
    for relative in (
        "docs/recommendation-evidence.json",
        "docs/figures/graphbrew-evidence-boundary.svg",
        "wiki/Evidence-and-Claims.md",
        "wiki/Historical-Low-Reuse-Policy.md",
    ):
        assert not (PROJECT_ROOT / relative).exists()


def test_public_navigation_and_figure_manifest_are_generic():
    sidebar = (PROJECT_ROOT / "wiki/_Sidebar.md").read_text()
    assert "Evidence and Claims" not in sidebar
    assert "Historical Low-Reuse Policy" not in sidebar

    manifest = json.loads(
        (PROJECT_ROOT / "docs/figures/public-manifest.json").read_text()
    )
    source_paths = {record["path"] for record in manifest["sources"]}
    record_paths = {record["path"] for record in manifest["records"]}
    assert "docs/recommendation-evidence.json" not in source_paths
    assert "docs/figures/graphbrew-evidence-boundary.svg" not in record_paths

    architecture = (
        PROJECT_ROOT / "docs/figures/graphbrew-architecture.svg"
    ).read_text()
    assert "EXPLICIT OPERATORS" in architecture
    assert "MEASUREMENT OUTPUTS" in architecture


def test_one_graph_dry_run_has_no_retired_evaluation_stage():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/graphbrew_experiment.py",
            "--target-graphs",
            "1",
            "--dry-run",
            "--skip-cache",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Offline fit" in result.stdout
    assert "LOGO" not in result.stdout
