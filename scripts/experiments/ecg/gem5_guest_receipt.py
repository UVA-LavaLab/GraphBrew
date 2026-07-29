#!/usr/bin/env python3
"""Atomically build and verify provenance-bound gem5 guest kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PINNED_RISCV_CXX = Path("/usr/bin/riscv64-linux-gnu-g++-13")
PINNED_RISCV_CXX_SHA256 = (
    "a675774e2afe01433771f6745de50870300833dc60ed5854662b414eff5fb7b6")
MATERIAL_COMPILER_ENV = (
    "PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path = PROJECT_ROOT) -> dict[str, str]:
    def output(command: list[str]) -> bytes:
        result = subprocess.run(
            command, cwd=root, capture_output=True, check=True)
        return result.stdout

    commit = output(["git", "rev-parse", "HEAD"]).decode().strip()
    diff = output(["git", "diff", "--binary", "--no-ext-diff"])
    cached = output(
        ["git", "diff", "--cached", "--binary", "--no-ext-diff"])
    return {
        "commit": commit,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "cached_diff_sha256": hashlib.sha256(cached).hexdigest(),
    }


def resolve_compiler(compiler_text: str) -> Path:
    parts = shlex.split(compiler_text)
    if len(parts) != 1:
        raise ValueError(
            "RISCV_CXX must name one compiler executable; wrappers and "
            "embedded arguments are not supported")
    driver = shutil.which(parts[0])
    if not driver:
        raise ValueError(f"compiler not found: {parts[0]}")
    driver_path = Path(driver).resolve()
    if driver_path != PINNED_RISCV_CXX or \
            sha256(driver_path) != PINNED_RISCV_CXX_SHA256:
        raise ValueError(
            "RISCV_CXX must resolve to the pinned RISC-V compiler")
    if driver_path.read_bytes()[:4] != b"\x7fELF":
        raise ValueError("RISCV_CXX must be an ELF compiler, not a wrapper")
    return driver_path


def compiler_component(driver: Path, *arguments: str) -> dict[str, str]:
    output = subprocess.run(
        [str(driver), *arguments], capture_output=True, text=True,
        check=True).stdout.strip()
    path = Path(output)
    if not path.is_absolute():
        candidate = shutil.which(output)
        path = Path(candidate) if candidate else path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(
            f"compiler component is missing: {' '.join(arguments)} -> {path}")
    return {"path": str(path), "sha256": sha256(path)}


def compiler_receipt(compiler_text: str) -> dict:
    driver = resolve_compiler(compiler_text)
    version = subprocess.run(
        [str(driver), "--version"], capture_output=True, text=True,
        check=True).stdout.splitlines()[0]
    dumpmachine = subprocess.run(
        [str(driver), "-dumpmachine"], capture_output=True, text=True,
        check=True).stdout.strip()
    if dumpmachine != "riscv64-linux-gnu":
        raise ValueError(f"unexpected RISC-V compiler target: {dumpmachine}")
    return {
        "invoked": compiler_text,
        "driver": str(driver),
        "driver_sha256": sha256(driver),
        "version": version,
        "dumpmachine": dumpmachine,
        "cc1plus": compiler_component(
            driver, "-print-prog-name=cc1plus"),
        "collect2": compiler_component(
            driver, "-print-prog-name=collect2"),
        "libgcc": compiler_component(
            driver, "-print-libgcc-file-name"),
        "libstdcxx": compiler_component(
            driver, "-print-file-name=libstdc++.a"),
        "assembler": compiler_component(
            driver, "-print-prog-name=as"),
        "linker": compiler_component(
            driver, "-print-prog-name=ld"),
        "crt1": compiler_component(
            driver, "-print-file-name=crt1.o"),
        "crti": compiler_component(
            driver, "-print-file-name=crti.o"),
        "crtn": compiler_component(
            driver, "-print-file-name=crtn.o"),
        "crtbegin": compiler_component(
            driver, "-print-file-name=crtbeginT.o"),
        "crtend": compiler_component(
            driver, "-print-file-name=crtend.o"),
        "libgcc_eh": compiler_component(
            driver, "-print-file-name=libgcc_eh.a"),
        "libgomp": compiler_component(
            driver, "-print-file-name=libgomp.a"),
        "libc": compiler_component(
            driver, "-print-file-name=libc.a"),
        "libpthread": compiler_component(
            driver, "-print-file-name=libpthread.a"),
        "specs_sha256": hashlib.sha256(subprocess.run(
            [str(driver), "-dumpspecs"], capture_output=True,
            check=True).stdout).hexdigest(),
        "search_dirs_sha256": hashlib.sha256(subprocess.run(
            [str(driver), "-print-search-dirs"], capture_output=True,
            check=True).stdout).hexdigest(),
    }


def material_environment() -> dict[str, str]:
    return {
        name: os.environ.get(name, "")
        for name in MATERIAL_COMPILER_ENV
    }


def parse_build_config(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if not line or "=" not in line:
            raise ValueError(f"invalid RISC-V build config line: {line!r}")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate RISC-V build config key: {key}")
        values[key] = value
    required = {
        "RISCV_CXX",
        "RISCV_CXX_RESOLVED",
        "RISCV_CXX_SHA256",
        "CXXFLAGS_GEM5_RISCV",
        "INCLUDES",
        *MATERIAL_COMPILER_ENV,
    }
    if set(values) != required:
        raise ValueError("RISC-V build config fields do not match schema")
    return values


def validate_build_config(
        path: Path, compiler: str, flags: str, includes: str) -> dict[str, str]:
    values = parse_build_config(path)
    driver = resolve_compiler(compiler)
    normalized = dict(values)
    normalized["RISCV_CXX_RESOLVED"] = str(
        Path(values["RISCV_CXX_RESOLVED"]).resolve())
    expected = {
        "RISCV_CXX": compiler,
        "RISCV_CXX_RESOLVED": str(driver),
        "RISCV_CXX_SHA256": sha256(driver),
        "CXXFLAGS_GEM5_RISCV": flags,
        "INCLUDES": includes,
        **material_environment(),
    }
    if normalized != expected:
        raise ValueError(
            "requested compiler, flags, includes, or environment do not "
            "match .riscv_build_config")
    return values


def parse_depfile_text(
        text: str, root: Path = PROJECT_ROOT) -> tuple[str, Path, list[Path]]:
    flattened = text.replace("\\\n", " ")
    if ":" not in flattened:
        raise ValueError("compiler depfile has no target")
    target_text, dependency_text = flattened.split(":", 1)
    targets = shlex.split(target_text)
    if len(targets) != 1:
        raise ValueError("compiler depfile must have exactly one target")
    target_text = targets[0]
    target = Path(target_text)
    if not target.is_absolute():
        target = root / target
    dependencies = []
    for token in shlex.split(dependency_text):
        dependency = Path(token)
        if not dependency.is_absolute():
            dependency = root / dependency
        dependency = dependency.resolve()
        if not dependency.is_file():
            raise ValueError(f"dependency is missing: {dependency}")
        dependencies.append(dependency)
    return target_text, target.resolve(), sorted(set(dependencies))


def parse_depfile(
        path: Path, root: Path = PROJECT_ROOT
        ) -> tuple[str, Path, list[Path]]:
    return parse_depfile_text(path.read_text(), root)


def dependency_key(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {
        dependency_key(path): sha256(path)
        for path in sorted(set(item.resolve() for item in paths))
    }


def compile_command(
        driver: Path, flags: str, includes: str, depfile: Path,
        dep_target: Path, source: Path, link_inputs: Iterable[Path],
        binary: Path) -> list[str]:
    return [
        str(driver),
        *shlex.split(flags),
        *shlex.split(includes),
        "-MD", "-MF", str(depfile), "-MT", str(dep_target),
        str(source),
        *(str(path) for path in link_inputs),
        "-o", str(binary),
    ]


def dependency_scan_command(
        driver: Path, flags: str, includes: str, depfile: Path,
        dep_target: Path, source: Path) -> list[str]:
    return [
        str(driver),
        *shlex.split(flags),
        *shlex.split(includes),
        "-M", "-MF", str(depfile), "-MT", str(dep_target),
        str(source),
    ]


def normalize_output_paths(command: list[str]) -> list[str]:
    normalized = list(command)
    for option, marker in (("-MF", "<DEPFILE>"), ("-o", "<BINARY>")):
        if option in normalized:
            normalized[normalized.index(option) + 1] = marker
    return normalized


def require_riscv_elf(path: Path) -> None:
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise ValueError("guest compiler output is not ELF")
    byteorder = "little" if header[5] == 1 else "big"
    if int.from_bytes(header[18:20], byteorder) != 243:
        raise ValueError("guest compiler output is not RISC-V ELF")


def build_guest(
        receipt_path: Path, binary: Path, depfile: Path, compiler: str,
        flags: str, includes: str, source: Path,
        link_inputs: list[Path], build_config: Path,
        make_target: str) -> dict:
    binary = binary.resolve()
    depfile = depfile.resolve()
    receipt_path = receipt_path.resolve()
    source = source.resolve()
    link_inputs = [path.resolve() for path in link_inputs]
    build_config = build_config.resolve()
    expected_receipt = Path(str(binary) + ".build.json")
    expected_depfile = Path(str(binary) + ".d")
    if receipt_path != expected_receipt or depfile != expected_depfile:
        raise ValueError("receipt and depfile must be adjacent to the guest")
    if make_target != dependency_key(binary):
        raise ValueError("Make target does not match requested guest binary")
    for path in (source, build_config, *link_inputs):
        if not path.is_file():
            raise ValueError(f"build input is missing: {path}")

    driver = resolve_compiler(compiler)
    build_config_values = validate_build_config(
        build_config, compiler, flags, includes)
    compiler_before = compiler_receipt(compiler)
    git_before = git_state()
    fixed_dependencies = {
        PROJECT_ROOT / "Makefile",
        Path(__file__).resolve(),
        build_config,
        *link_inputs,
    }
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".{binary.name}.build.", dir=binary.parent) as temp_text:
        temp = Path(temp_text)
        pre_depfile = temp / "pre.d"
        temp_depfile = temp / binary.name.replace("/", "_")
        temp_depfile = temp_depfile.with_suffix(".d")
        temp_binary = temp / binary.name
        scan_command = dependency_scan_command(
            driver, flags, includes, pre_depfile,
            Path(make_target), source)
        subprocess.run(scan_command, cwd=PROJECT_ROOT, check=True)
        pre_target_text, pre_target, pre_dependencies = parse_depfile(
            pre_depfile)
        if pre_target_text != make_target or pre_target != binary or \
                source not in pre_dependencies:
            raise ValueError("dependency scan does not describe requested guest")
        before_paths = set(pre_dependencies) | fixed_dependencies
        before_hashes = snapshot(before_paths)

        command = compile_command(
            driver, flags, includes, temp_depfile, Path(make_target),
            source, link_inputs, temp_binary)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        post_target_text, post_target, post_dependencies = parse_depfile(
            temp_depfile)
        if post_target_text != make_target or post_target != binary or \
                source not in post_dependencies:
            raise ValueError("compile depfile does not describe requested guest")
        after_paths = set(post_dependencies) | fixed_dependencies
        after_hashes = snapshot(after_paths)
        if set(pre_dependencies) != set(post_dependencies):
            raise ValueError("dependency set changed during guest compilation")
        if before_hashes != after_hashes:
            raise ValueError("build input changed during guest compilation")
        if compiler_before != compiler_receipt(compiler):
            raise ValueError("compiler changed during guest compilation")
        if git_before != git_state():
            raise ValueError("git state changed during guest compilation")
        if not temp_binary.is_file():
            raise ValueError("compiler produced no guest binary")
        require_riscv_elf(temp_binary)

        canonical_command = compile_command(
            driver, flags, includes, depfile, Path(make_target),
            source, link_inputs, binary)
        payload = {
            "schema_version": 2,
            "git": git_before,
            "compiler": compiler_before,
            "compiler_environment": material_environment(),
            "build_config_values": build_config_values,
            "flags": flags,
            "includes": includes,
            "make_target": make_target,
            "source": dependency_key(source),
            "link_inputs": [
                dependency_key(path) for path in link_inputs
            ],
            "build_config": dependency_key(build_config),
            "dependency_scan_command": scan_command,
            "compile_command": command,
            "canonical_command": canonical_command,
            "binary": {
                "path": dependency_key(binary),
                "sha256": sha256(temp_binary),
            },
            "depfile": {
                "path": dependency_key(depfile),
                "sha256": sha256(temp_depfile),
                "target": post_target_text,
            },
            "dependencies": after_hashes,
        }
        temp_receipt = temp / receipt_path.name
        temp_receipt.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temp_binary, binary)
        os.replace(temp_depfile, depfile)
        os.replace(temp_receipt, receipt_path)
    return payload


def validate_receipt(
        receipt_path: Path, binary: Path, source: Path,
        link_inputs: list[Path], build_config: Path,
        root: Path = PROJECT_ROOT) -> list[str]:
    errors = []
    binary = binary.resolve()
    source = source.resolve()
    link_inputs = [path.resolve() for path in link_inputs]
    build_config = build_config.resolve()
    expected_receipt = Path(str(binary) + ".build.json")
    expected_depfile = Path(str(binary) + ".d")
    make_target = dependency_key(binary, root)
    if receipt_path.resolve() != expected_receipt:
        errors.append("guest receipt path does not match requested binary")
    if not receipt_path.is_file():
        return errors + [f"guest build receipt is missing: {receipt_path}"]
    try:
        payload = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return errors + [f"guest build receipt is unreadable: {error}"]
    if payload.get("schema_version") != 2:
        errors.append("unsupported guest build receipt schema")
    target_rows = {
        "binary": dependency_key(binary, root),
        "source": dependency_key(source, root),
        "build_config": dependency_key(build_config, root),
    }
    binary_row = payload.get("binary", {})
    if binary_row.get("path") != target_rows["binary"]:
        errors.append("guest receipt names a different binary target")
    if payload.get("source") != target_rows["source"]:
        errors.append("guest receipt names a different kernel source")
    if payload.get("build_config") != target_rows["build_config"]:
        errors.append("guest receipt names a different build configuration")
    if payload.get("make_target") != make_target:
        errors.append("guest receipt names a different Make target")
    expected_links = [dependency_key(path, root) for path in link_inputs]
    if payload.get("link_inputs") != expected_links:
        errors.append("guest receipt names different link inputs")
    if not binary.is_file() or binary_row.get("sha256") != sha256(binary):
        errors.append("guest binary hash does not match build receipt")
    else:
        try:
            require_riscv_elf(binary)
        except ValueError as error:
            errors.append(str(error))
    depfile_row = payload.get("depfile", {})
    if depfile_row.get("path") != dependency_key(expected_depfile, root):
        errors.append("guest receipt names a different depfile")
    if not expected_depfile.is_file() or \
            depfile_row.get("sha256") != sha256(expected_depfile):
        errors.append("guest depfile hash does not match build receipt")
        dep_dependencies = []
    else:
        try:
            dep_target_text, dep_target, dep_dependencies = parse_depfile(
                expected_depfile, root)
            if dep_target_text != make_target or dep_target != binary or \
                    depfile_row.get("target") != make_target:
                errors.append("guest depfile target does not match binary")
        except ValueError as error:
            errors.append(str(error))
            dep_dependencies = []
    try:
        if payload.get("git") != git_state(root):
            errors.append("guest binary was built from a different git state")
    except subprocess.CalledProcessError as error:
        errors.append(f"cannot verify git state: {error}")
    try:
        build_values = parse_build_config(build_config)
        compiler_text = build_values["RISCV_CXX"]
        flags = build_values["CXXFLAGS_GEM5_RISCV"]
        includes = build_values["INCLUDES"]
        current_build_values = validate_build_config(
            build_config, compiler_text, flags, includes)
        if payload.get("build_config_values") != current_build_values:
            errors.append("guest receipt build configuration is inconsistent")
    except (OSError, ValueError) as error:
        errors.append(f"guest build configuration cannot be verified: {error}")
        compiler_text, flags, includes = "", "", ""
    compiler = payload.get("compiler", {})
    try:
        current_compiler = compiler_receipt(compiler_text)
        if compiler != current_compiler:
            errors.append("guest compiler does not match build receipt")
    except (ValueError, subprocess.CalledProcessError) as error:
        errors.append(f"guest compiler cannot be verified: {error}")
        current_compiler = {}
    fixed_dependencies = {
        root / "Makefile",
        Path(__file__).resolve(),
        build_config,
        *link_inputs,
    }
    expected_dependencies = snapshot(
        set(dep_dependencies) | fixed_dependencies)
    if payload.get("dependencies") != expected_dependencies:
        errors.append("guest dependency hashes do not match build receipt")
    if source not in dep_dependencies:
        errors.append("guest depfile does not include requested kernel source")
    if current_compiler:
        expected_command = compile_command(
            Path(current_compiler["driver"]),
            flags, includes, expected_depfile, Path(make_target),
            source, link_inputs, binary)
        if payload.get("flags") != flags or \
                payload.get("includes") != includes:
            errors.append("guest compiler options do not match build config")
        if payload.get("canonical_command") != expected_command:
            errors.append("guest canonical compiler command is inconsistent")
        compile_row = payload.get("compile_command")
        if not isinstance(compile_row, list) or \
                normalize_output_paths(compile_row) != \
                normalize_output_paths(expected_command):
            errors.append("guest executed compiler command is inconsistent")
    return errors


def stage_validated_guest(
        receipt_path: Path, binary: Path, source: Path,
        link_inputs: list[Path], build_config: Path,
        staging_dir: Path, root: Path = PROJECT_ROOT) -> tuple[Path, str]:
    errors = validate_receipt(
        receipt_path, binary, source, link_inputs, build_config, root)
    if errors:
        raise ValueError("\n".join(errors))
    payload = json.loads(receipt_path.read_text())
    expected_hash = str(payload["binary"]["sha256"])
    if sha256(binary) != expected_hash:
        raise ValueError("guest changed after receipt validation")
    staging_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(staging_dir, 0o700)
    destination = staging_dir / f"{binary.name}.{expected_hash}"
    if not destination.is_file():
        temporary = staging_dir / f".{destination.name}.tmp"
        temporary.unlink(missing_ok=True)
        with binary.open("rb") as source_handle, temporary.open("xb") as out:
            shutil.copyfileobj(source_handle, out)
            out.flush()
            os.fsync(out.fileno())
        if sha256(binary) != expected_hash or \
                sha256(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise ValueError("guest changed while staging validated input")
        os.chmod(temporary, 0o555)
        os.replace(temporary, destination)
    if sha256(destination) != expected_hash:
        raise ValueError("staged guest hash mismatch")
    os.chmod(destination, 0o555)
    os.chmod(staging_dir, 0o555)
    return destination, expected_hash


def verify_staged_guest(path: Path, expected_hash: str) -> None:
    if not path.is_file() or sha256(path) != expected_hash:
        raise ValueError("validated staged guest changed before execution")


def remove_outputs(binary: Path, depfile: Path, receipt: Path) -> None:
    for path in (binary, depfile, receipt):
        path.unlink(missing_ok=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--binary", type=Path, required=True)
    build.add_argument("--depfile", type=Path, required=True)
    build.add_argument("--compiler", required=True)
    build.add_argument("--flags", default="")
    build.add_argument("--includes", default="")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument(
        "--link-input", type=Path, action="append", default=[])
    build.add_argument("--build-config", type=Path, required=True)
    build.add_argument("--make-target", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--binary", type=Path, required=True)
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument(
        "--link-input", type=Path, action="append", default=[])
    verify.add_argument("--build-config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.action == "build":
        try:
            build_guest(
                args.receipt, args.binary, args.depfile, args.compiler,
                args.flags, args.includes, args.source, args.link_input,
                args.build_config, args.make_target)
        except Exception:
            remove_outputs(args.binary, args.depfile, args.receipt)
            raise
        return 0
    errors = validate_receipt(
        args.receipt, args.binary, args.source,
        args.link_input, args.build_config)
    for error in errors:
        print(f"[FAIL] {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
