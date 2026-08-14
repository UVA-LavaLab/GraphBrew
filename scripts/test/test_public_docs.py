#!/usr/bin/env python3
"""Public docs and orchestrator help must describe the current workflow."""

import subprocess
import sys
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_public_docs_exclude_retired_workflow_claims():
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "wiki" / "Home.md",
        PROJECT_ROOT / "wiki" / "Python-Scripts.md",
        PROJECT_ROOT / "wiki" / "Running-Benchmarks.md",
    ]
    text = "\n".join(path.read_text() for path in paths)

    assert "C++ trains models at runtime" not in text
    assert "worst-case baseline" not in text
    assert "vldb_paper_experiments.py" not in text
    assert "paper/main.tex" not in text
    assert "LOGO CV on all ML models" not in text


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


def test_public_docs_use_orchestrator_for_normal_frozen_runs():
    getting_started = (
        PROJECT_ROOT / "wiki" / "Getting-Started.md"
    ).read_text()
    quick_start = (
        PROJECT_ROOT / "wiki" / "Reproducible-Experiments.md"
    ).read_text().split("## 2. Prerequisites", 1)[0]

    assert "scripts/graphbrew_experiment.py --vldb" in getting_started
    assert "scripts/graphbrew_experiment.py --vldb" in quick_start
    assert "scripts/experiments/vldb/runner.py --all" not in quick_start


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
