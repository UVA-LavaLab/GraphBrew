#!/usr/bin/env python3
"""The root Makefile is the verification-command SSOT."""

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_harness_delegates_to_make_check_without_github_workflows():
    harness = (
        PROJECT_ROOT / "scripts" / "graphbrew_experiment.py"
    ).read_text()

    assert not (PROJECT_ROOT / ".github").exists()
    assert 'cmd = ["make", "check"]' in harness
    assert 'cmd = ["python3", "-m", "pytest"' not in harness


def test_make_check_excludes_extended_reference_suites():
    makefile = (PROJECT_ROOT / "Makefile").read_text()

    assert (
        "check: lint-includes check-native-core "
        "\\"
    ) in makefile
    assert "$(addprefix $(BIN_DIR)/,pr bfs cc sssp converter)" in makefile
    assert "$(BIN_WORK_DIR)/bfs" in makefile
    assert "--max-skips=$(MAX_TEST_SKIPS)" in makefile
    assert "$(PYTHON) -m pytest scripts/test $(PYTEST_ARGS)" in makefile
    check_declaration = next(
        line for line in makefile.splitlines()
        if line.startswith("check: ")
    )
    assert "check-partition" not in check_declaration
    assert "check-edge" not in check_declaration
    assert "check-gas" not in check_declaration


def test_native_gate_propagates_first_failure(tmp_path):
    failing = tmp_path / "failing"
    passing = tmp_path / "passing"
    failing.write_text("#!/bin/sh\nexit 7\n")
    passing.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(failing, 0o755)
    os.chmod(passing, 0o755)

    result = subprocess.run(
        [
            "make",
            "check-native-core",
            f"CORE_UNIT_TESTS_BIN={failing} {passing}",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
