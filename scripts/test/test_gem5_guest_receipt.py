import json
import os
from pathlib import Path
import shutil
from contextlib import contextmanager
from types import SimpleNamespace
import subprocess
import sys

import pytest

from scripts.experiments.ecg.flows import paper_run
from scripts.experiments.ecg import gem5_guest_receipt as receipt_module
from scripts.experiments.ecg.gem5_guest_receipt import (
    MATERIAL_COMPILER_ENV,
    PINNED_FUSERMOUNT,
    PINNED_FUSERMOUNT_SHA256,
    PINNED_FUSEPY,
    PINNED_FUSEPY_SHA256,
    PINNED_LIBFUSE,
    PINNED_LIBFUSE_SHA256,
    PINNED_PYTHON,
    PINNED_PYTHON_SHA256,
    PINNED_PROOT,
    PINNED_PROOT_LIBC,
    PINNED_PROOT_LIBC_SHA256,
    PINNED_PROOT_LOADER,
    PINNED_PROOT_LOADER_SHA256,
    PINNED_PROOT_SHA256,
    PINNED_PROOT_TALLOC,
    PINNED_PROOT_TALLOC_SHA256,
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
        "PROOT": str(PINNED_PROOT),
        "PROOT_SHA256": PINNED_PROOT_SHA256,
        "PROOT_LOADER": str(PINNED_PROOT_LOADER),
        "PROOT_LOADER_SHA256": PINNED_PROOT_LOADER_SHA256,
        "PROOT_LIBC": str(PINNED_PROOT_LIBC),
        "PROOT_LIBC_SHA256": PINNED_PROOT_LIBC_SHA256,
        "PROOT_TALLOC": str(PINNED_PROOT_TALLOC),
        "PROOT_TALLOC_SHA256": PINNED_PROOT_TALLOC_SHA256,
        "FUSEPY": str(PINNED_FUSEPY),
        "FUSEPY_SHA256": PINNED_FUSEPY_SHA256,
        "LIBFUSE": str(PINNED_LIBFUSE),
        "LIBFUSE_SHA256": PINNED_LIBFUSE_SHA256,
        "FUSERMOUNT": str(PINNED_FUSERMOUNT),
        "FUSERMOUNT_SHA256": PINNED_FUSERMOUNT_SHA256,
        "PYTHON": str(PINNED_PYTHON),
        "PYTHON_SHA256": PINNED_PYTHON_SHA256,
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
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/setup_gem5_guest_tools.py"),
            "--verify-only",
        ],
        cwd=PROJECT_ROOT)
    assert result.returncode == 0


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
    traced_names = {
        Path(row["virtual_path"]).name
        for row in payload["traced_inputs"]
    }
    assert "libgomp.spec" in traced_names
    assert "libgomp.a" in traced_names
    assert "libm.a" in traced_names
    assert "liblto_plugin.so" in traced_names
    depfile_text = depfile.read_text()
    assert "libgomp.spec" in depfile_text
    assert "liblto_plugin.so" in depfile_text


def test_final_compile_consumes_immutable_snapshot_during_swap_restore(
        tmp_path, monkeypatch):
    source = tmp_path / "pr.cc"
    binary = tmp_path / "pr_riscv_m5ops"
    depfile = Path(str(binary) + ".d")
    receipt = Path(str(binary) + ".build.json")
    build_config = tmp_path / ".riscv_build_config"
    flags = "-O0 -static"
    original = (
        '__attribute__((used)) const char marker[] = "ORIGINAL";\n'
        "int main() { return marker[0] == 'O' ? 0 : 1; }\n")
    source.write_text(original)
    write_build_config(build_config, flags, "")
    real_mount = receipt_module.immutable_fuse_files

    @contextmanager
    def swap_restore(files, mountpoint):
        with real_mount(files, mountpoint):
            source.write_text(original.replace("ORIGINAL", "MALICIOUS"))
            try:
                yield
            finally:
                source.write_text(original)

    monkeypatch.setattr(
        receipt_module, "immutable_fuse_files", swap_restore)
    build_guest(
        receipt, binary, depfile, "riscv64-linux-gnu-g++", flags, "",
        source, [], build_config, str(binary))
    strings = subprocess.run(
        ["strings", str(binary)], capture_output=True, text=True,
        check=True).stdout
    assert "ORIGINAL" in strings
    assert "MALICIOUS" not in strings


def test_injected_loader_and_proot_environment_is_ignored(
        tmp_path, monkeypatch):
    source = tmp_path / "pr.cc"
    binary = tmp_path / "pr_riscv_m5ops"
    depfile = Path(str(binary) + ".d")
    receipt = Path(str(binary) + ".build.json")
    build_config = tmp_path / ".riscv_build_config"
    flags = "-O0 -static"
    source.write_text("int main() { return 0; }\n")
    write_build_config(build_config, flags, "")
    monkeypatch.setenv("LD_PRELOAD", "/does/not/exist.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/does/not/exist")
    monkeypatch.setenv("PROOT_LOADER", "/bin/false")
    monkeypatch.setenv("PYTHONPATH", "/does/not/exist")
    monkeypatch.setenv("FUSE_LIBRARY_PATH", "/does/not/exist.so")
    build_guest(
        receipt, binary, depfile, "riscv64-linux-gnu-g++", flags, "",
        source, [], build_config, str(binary))
    assert validate_receipt(
        receipt, binary, source, [], build_config) == []


def test_virtual_alias_retarget_invalidates_receipt(tmp_path):
    first = tmp_path / "first.h"
    second = tmp_path / "second.h"
    alias = tmp_path / "selected.h"
    source = tmp_path / "pr.cc"
    binary = tmp_path / "pr_riscv_m5ops"
    depfile = Path(str(binary) + ".d")
    receipt = Path(str(binary) + ".build.json")
    build_config = tmp_path / ".riscv_build_config"
    flags = "-O0 -static"
    first.write_text("#define VALUE 1\n")
    second.write_text("#define VALUE 2\n")
    alias.symlink_to(first)
    source.write_text(
        '#include "selected.h"\nint main() { return VALUE - 1; }\n')
    write_build_config(build_config, flags, f"-I{tmp_path}")
    build_guest(
        receipt, binary, depfile, "riscv64-linux-gnu-g++", flags,
        f"-I{tmp_path}", source, [], build_config, str(binary))
    alias.unlink()
    alias.symlink_to(second)
    errors = validate_receipt(
        receipt, binary, source, [], build_config)
    assert any("virtual alias changed" in error for error in errors)


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


def test_paper_run_requires_one_guest_hash_across_jobs(tmp_path):
    jobs = []
    for index in range(2):
        out_dir = tmp_path / f"job-{index}"
        out_dir.mkdir()
        (out_dir / "roi_matrix.csv").write_text(
            "status,simulator,gem5_guest_expected_sha256,"
            "gem5_guest_staged_sha256\n"
            "ok,gem5,abc,abc\n")
        jobs.append(paper_run.Job(
            job_id=f"job-{index}", stage=f"stage-{index}",
            kind="roi_matrix", command=[], out_dir=out_dir,
            log_path=tmp_path / f"job-{index}.log",
            metadata={"expected_gem5_guest_sha256": "abc"}))
    assert paper_run.validate_cross_job_guest_hashes(jobs) == (
        True, "abc")
    (jobs[1].output_csv).write_text(
        "status,simulator,gem5_guest_expected_sha256,"
        "gem5_guest_staged_sha256\n"
        "ok,gem5,abc,changed\n")
    ok, detail = paper_run.validate_cross_job_guest_hashes(jobs)
    assert not ok
    assert "guest hash mismatch" in detail


def test_paper_and_roi_environments_strip_code_injection():
    clean = paper_run.clean_job_environment({
        "LD_PRELOAD": "/bad.so",
        "PROOT_LOADER": "/bin/false",
        "PYTHONPATH": "/bad",
        "FUSE_LIBRARY_PATH": "/bad.so",
        "GEM5_KERNEL_SUFFIX": "_riscv_m5ops",
    })
    assert "LD_PRELOAD" not in clean
    assert "PROOT_LOADER" not in clean
    assert "PYTHONPATH" not in clean
    assert "FUSE_LIBRARY_PATH" not in clean
    assert clean["GEM5_KERNEL_SUFFIX"] == "_riscv_m5ops"
    assert clean["PATH"] == "/usr/bin:/bin"


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
