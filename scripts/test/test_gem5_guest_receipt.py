import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess
import sys

import pytest

from scripts.experiments.ecg.flows import paper_run
from scripts.experiments.ecg.gem5_guest_receipt import (
    MATERIAL_COMPILER_ENV,
    PINNED_RISCV_CXX,
    PINNED_RISCV_CXX_SHA256,
    PROJECT_ROOT,
    build_guest,
    material_environment,
    open_sealed_guest,
    sha256,
    stage_validated_guest,
    validate_receipt,
    verify_staged_guest,
)


def write_build_config(
        path: Path, flags: str, includes: str,
        compiler: str = "riscv64-linux-gnu-g++") -> None:
    values = {
        "RISCV_CXX": compiler,
        "RISCV_CXX_RESOLVED": shutil.which(compiler) or "",
        "RISCV_CXX_SHA256": PINNED_RISCV_CXX_SHA256,
        "CXXFLAGS_GEM5_RISCV": flags,
        "INCLUDES": includes,
        **material_environment(),
    }
    assert set(MATERIAL_COMPILER_ENV) <= set(values)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()))


def test_guest_build_atomically_binds_target_dependencies_and_git(tmp_path):
    source = tmp_path / "pr.cc"
    header = tmp_path / "reorder_hub.h"
    binary = tmp_path / "pr_riscv_m5ops"
    depfile = Path(str(binary) + ".d")
    receipt = Path(str(binary) + ".build.json")
    build_config = tmp_path / ".riscv_build_config"
    header.write_text("#define DBG_AVG_DEGREE 2\n")
    source.write_text(
        '#include "reorder_hub.h"\n'
        "int main() { return DBG_AVG_DEGREE == 2 ? 0 : 1; }\n")
    flags = "-O0 -static"
    includes = f"-I{tmp_path}"
    write_build_config(build_config, flags, includes)

    payload = build_guest(
        receipt, binary, depfile, "riscv64-linux-gnu-g++", flags,
        includes, source, [], build_config, str(binary))

    assert payload["schema_version"] == 2
    assert payload["source"] == str(source)
    assert str(header) in payload["dependencies"]
    assert validate_receipt(
        receipt, binary, source, [], build_config) == []

    header.write_text("#define DBG_AVG_DEGREE 1\n")
    assert any(
        "dependency hashes" in error
        for error in validate_receipt(
            receipt, binary, source, [], build_config))


def test_guest_receipt_cannot_be_copied_to_another_kernel(tmp_path):
    source = tmp_path / "bfs.cc"
    binary = tmp_path / "bfs_riscv_m5ops"
    depfile = Path(str(binary) + ".d")
    receipt = Path(str(binary) + ".build.json")
    build_config = tmp_path / ".riscv_build_config"
    source.write_text("int main() { return 0; }\n")
    flags = "-O0 -static"
    write_build_config(build_config, flags, "")
    build_guest(
        receipt, binary, depfile, "riscv64-linux-gnu-g++", flags, "",
        source, [], build_config, str(binary))

    other_source = tmp_path / "pr.cc"
    other_source.write_text("int main() { return 0; }\n")
    other_binary = tmp_path / "pr_riscv_m5ops"
    other_depfile = Path(str(other_binary) + ".d")
    other_receipt = Path(str(other_binary) + ".build.json")
    shutil.copy2(binary, other_binary)
    shutil.copy2(depfile, other_depfile)
    shutil.copy2(receipt, other_receipt)

    errors = validate_receipt(
        other_receipt, other_binary, other_source, [], build_config)
    assert any("different binary target" in error for error in errors)
    assert any("different kernel source" in error for error in errors)


def test_validated_guest_is_staged_and_rechecked_per_execution(tmp_path):
    source = tmp_path / "pr.cc"
    binary = tmp_path / "pr_riscv_m5ops"
    depfile = Path(str(binary) + ".d")
    receipt = Path(str(binary) + ".build.json")
    build_config = tmp_path / ".riscv_build_config"
    flags = "-O0 -static"
    source.write_text("int main() { return 0; }\n")
    write_build_config(build_config, flags, "")
    build_guest(
        receipt, binary, depfile, "riscv64-linux-gnu-g++", flags, "",
        source, [], build_config, str(binary))

    staged, expected_hash = stage_validated_guest(
        receipt, binary, source, [], build_config, tmp_path / "staged")
    verify_staged_guest(staged, expected_hash)
    os.chmod(staged.parent, 0o700)
    os.chmod(staged, 0o755)
    staged.write_bytes(b"swapped")
    with pytest.raises(ValueError, match="staged guest changed"):
        verify_staged_guest(staged, expected_hash)


def test_sealed_memfd_executes_immutable_open_file():
    binary = Path("/bin/true")
    fd, path = open_sealed_guest(binary, sha256(binary))
    try:
        result = subprocess.run([path], pass_fds=(fd,))
        assert result.returncode == 0
        with pytest.raises(OSError):
            os.write(fd, b"changed")
    finally:
        os.close(fd)


def test_riscv_make_rule_models_all_outputs_and_command_signature():
    makefile = (PROJECT_ROOT / "Makefile").read_text()
    assert "_riscv_m5ops.build.json &:" in makefile
    assert "-include $(wildcard $(BIN_GEM5_DIR)/*_riscv_m5ops.d)" in makefile
    assert "$(GEM5_GUEST_RECEIPT) build" in makefile
    assert "--build-config $(GEM5_RISCV_BUILD_CONFIG)" in makefile
    assert "RISCV_CXX_SHA256=" in makefile
    assert ".PRECIOUS: $(RISCV_GUEST_BINARIES)" in makefile


def test_traced_inputs_cover_openmp_math_and_linker_plugin(tmp_path):
    source = tmp_path / "toolchain.cc"
    binary = tmp_path / "toolchain_riscv_m5ops"
    depfile = Path(str(binary) + ".d")
    receipt = Path(str(binary) + ".build.json")
    build_config = tmp_path / ".riscv_build_config"
    flags = "-O0 -static -fopenmp"
    source.write_text(
        "#include <cmath>\n#include <omp.h>\n"
        "int main(int argc, char**) {\n"
        "  return static_cast<int>(std::sin(argc)) + omp_get_max_threads();\n"
        "}\n")
    write_build_config(build_config, flags, "")
    payload = build_guest(
        receipt, binary, depfile, "riscv64-linux-gnu-g++", flags, "",
        source, [], build_config, str(binary))
    traced_names = {Path(name).name for name in payload["traced_inputs"]}
    assert "libgomp.spec" in traced_names
    assert "libgomp.a" in traced_names
    assert "libm.a" in traced_names
    assert "liblto_plugin.so" in traced_names


def test_wrapper_compiler_and_config_drift_are_rejected(tmp_path):
    wrapper = tmp_path / "riscv64-linux-gnu-g++"
    wrapper.write_text("#!/bin/sh\nexec /bin/true \"$@\"\n")
    wrapper.chmod(0o755)
    build_config = tmp_path / ".riscv_build_config"
    write_build_config(build_config, "-O0 -static", "")
    source = tmp_path / "pr.cc"
    source.write_text("int main() { return 0; }\n")
    binary = tmp_path / "pr_riscv_m5ops"
    with pytest.raises(ValueError, match="pinned RISC-V compiler"):
        build_guest(
            Path(str(binary) + ".build.json"), binary,
            Path(str(binary) + ".d"), str(wrapper), "-O0 -static", "",
            source, [], build_config, str(binary))

    write_build_config(build_config, "-O2 -static", "")
    with pytest.raises(ValueError, match="do not match"):
        build_guest(
            Path(str(binary) + ".build.json"), binary,
            Path(str(binary) + ".d"), "riscv64-linux-gnu-g++",
            "-O0 -static", "", source, [], build_config, str(binary))


def test_paper_run_fingerprints_both_backends_and_resolves_gem5(
        monkeypatch):
    monkeypatch.setattr(
        paper_run, "path_fingerprint", lambda path: path)
    monkeypatch.setattr(
        paper_run, "git_state_fingerprint", lambda: "git")
    relative_gem5 = (
        "bench/include/gem5_sim/gem5/build/RISCV/gem5.opt")
    inputs = paper_run.roi_input_fingerprints(
        SimpleNamespace(
            manifest=str(
                PROJECT_ROOT /
                "scripts/experiments/ecg/final_paper_manifest.json")),
        {"suite": "both"}, None, "pr", {
            "GEM5_OPT": relative_gem5,
            "GEM5_KERNEL_SUFFIX": "_riscv_m5ops",
        })
    assert "cache_sim_benchmark_binary" in inputs
    assert "gem5_benchmark_binary" in inputs
    assert "gem5_guest_build_receipt" in inputs
    assert inputs["gem5_binary"] == str(
        (PROJECT_ROOT / relative_gem5).resolve())


def test_inconsistent_gem5_isa_overrides_fail_closed():
    env = dict(os.environ)
    env.update({
        "GEM5_OPT": (
            "bench/include/gem5_sim/gem5/build/X86/gem5.opt"),
        "GEM5_KERNEL_SUFFIX": "_riscv_m5ops",
    })
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/experiments/ecg/roi_matrix.py"),
            "--suite", "gem5", "--benchmark", "pr",
            "--policies", "LRU", "--no-build", "--dry-run",
        ],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "inconsistent gem5 ISA selection" in (
        result.stdout + result.stderr)


def test_current_riscv_pr_receipt_when_binary_is_present():
    binary = PROJECT_ROOT / "bench/bin_gem5/pr_riscv_m5ops"
    if not binary.is_file():
        pytest.skip("RISC-V PageRank guest is not built")
    receipt = Path(str(binary) + ".build.json")
    if not receipt.is_file():
        pytest.skip("RISC-V PageRank guest predates build receipts")
    source = PROJECT_ROOT / "bench/src_gem5/pr.cc"
    build_config = PROJECT_ROOT / "bench/bin_gem5/.riscv_build_config"
    link_inputs = [
        PROJECT_ROOT / "bench/include/gem5_sim/gem5/util/m5/"
        "build/riscv/out/libm5.a",
    ]
    assert validate_receipt(
        receipt, binary, source, link_inputs, build_config) == []
    payload = json.loads(receipt.read_text())
    assert any(
        name.endswith("graphbrew/reorder/reorder_hub.h")
        for name in payload["dependencies"])
