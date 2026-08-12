#!/usr/bin/env python3
"""The root Makefile is the verification-command SSOT."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ci_and_harness_delegate_to_make_check():
    travis = (PROJECT_ROOT / ".travis.yml").read_text()
    harness = (
        PROJECT_ROOT / "scripts" / "graphbrew_experiment.py"
    ).read_text()

    assert "make check -j2 RABBIT_ENABLE=0" in travis
    assert "pytest scripts/test -q" not in travis
    assert 'cmd = ["make", "check"]' in harness
    assert 'cmd = ["python3", "-m", "pytest"' not in harness


def test_make_check_excludes_extended_reference_suites():
    makefile = (PROJECT_ROOT / "Makefile").read_text()

    assert (
        "check: lint-includes check-native-core "
        "$(BIN_DIR)/pr $(BIN_SIM_DIR)/pr"
    ) in makefile
    assert "$(PYTHON) -m pytest scripts/test $(PYTEST_ARGS)" in makefile
    check_declaration = next(
        line for line in makefile.splitlines()
        if line.startswith("check: ")
    )
    assert "check-partition" not in check_declaration
    assert "check-edge" not in check_declaration
    assert "check-gas" not in check_declaration
