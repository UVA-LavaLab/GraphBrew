import json
from pathlib import Path

import pytest

from scripts.experiments.ecg.gem5_guest_receipt import (
    PROJECT_ROOT,
    validate_receipt,
    write_receipt,
)


def test_guest_receipt_binds_binary_dependencies_and_git_state(tmp_path):
    source = tmp_path / "pr.cc"
    header = tmp_path / "reorder_hub.h"
    library = tmp_path / "libm5.a"
    binary = tmp_path / "pr_riscv_m5ops"
    depfile = tmp_path / "pr_riscv_m5ops.d"
    receipt = tmp_path / "pr_riscv_m5ops.build.json"
    source.write_text("source\n")
    header.write_text("directed adjacency count\n")
    library.write_bytes(b"library")
    binary.write_bytes(b"binary")
    depfile.write_text(f"{binary}: {source} {header}\n")

    payload = write_receipt(
        receipt, binary, depfile, "g++", "-O1", "-Ibench/include",
        source, [library])

    assert payload["binary"]["sha256"]
    assert str(header) in payload["dependencies"]
    assert validate_receipt(receipt, binary) == []
    header.write_text("stale half-edge count\n")
    assert any(
        "dependency changed" in error
        for error in validate_receipt(receipt, binary))


def test_riscv_make_rule_emits_compiler_depfile_and_receipt():
    makefile = (PROJECT_ROOT / "Makefile").read_text()
    assert "-MMD -MF $@.d" in makefile
    assert "$(GEM5_GUEST_RECEIPT) write" in makefile
    assert "--receipt $@.build.json" in makefile
    assert "--compiler \"$(RISCV_CXX)\"" in makefile


def test_current_riscv_pr_receipt_when_binary_is_present():
    binary = PROJECT_ROOT / "bench/bin_gem5/pr_riscv_m5ops"
    if not binary.is_file():
        pytest.skip("RISC-V PageRank guest is not built")
    receipt = Path(str(binary) + ".build.json")
    if not receipt.is_file():
        pytest.skip("RISC-V PageRank guest predates build receipts")
    assert validate_receipt(receipt, binary) == []
    payload = json.loads(receipt.read_text())
    assert any(
        name.endswith("graphbrew/reorder/reorder_hub.h")
        for name in payload["dependencies"])
