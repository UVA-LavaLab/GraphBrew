#!/usr/bin/env python3
"""Write and verify provenance receipts for prebuilt gem5 guest kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def compiler_receipt(compiler_text: str) -> dict[str, str]:
    parts = shlex.split(compiler_text)
    if not parts:
        raise ValueError("compiler command is empty")
    driver = shutil.which(parts[0])
    if not driver:
        raise ValueError(f"compiler not found: {parts[0]}")
    driver_path = Path(driver).resolve()
    version = subprocess.run(
        [str(driver_path), "--version"], capture_output=True, text=True,
        check=True).stdout.splitlines()[0]
    return {
        "invoked": compiler_text,
        "driver": str(driver_path),
        "driver_sha256": sha256(driver_path),
        "version": version,
    }


def parse_depfile(path: Path, root: Path = PROJECT_ROOT) -> list[Path]:
    text = path.read_text().replace("\\\n", " ")
    if ":" not in text:
        raise ValueError(f"invalid compiler depfile: {path}")
    dependency_text = text.split(":", 1)[1]
    dependencies = []
    for token in shlex.split(dependency_text):
        dependency = Path(token)
        if not dependency.is_absolute():
            dependency = root / dependency
        dependency = dependency.resolve()
        if not dependency.is_file():
            raise ValueError(f"dependency is missing: {dependency}")
        dependencies.append(dependency)
    return sorted(set(dependencies))


def dependency_key(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def command_parts(
        compiler: str, flags: str, includes: str, depfile: Path,
        source: Path, link_inputs: Iterable[Path], binary: Path) -> list[str]:
    return [
        *shlex.split(compiler),
        *shlex.split(flags),
        *shlex.split(includes),
        "-MMD", "-MF", str(depfile),
        str(source),
        *(str(path) for path in link_inputs),
        "-o", str(binary),
    ]


def write_receipt(
        receipt_path: Path, binary: Path, depfile: Path, compiler: str,
        flags: str, includes: str, source: Path,
        link_inputs: list[Path]) -> dict:
    binary = binary.resolve()
    depfile = depfile.resolve()
    source = source.resolve()
    link_inputs = [path.resolve() for path in link_inputs]
    dependencies = parse_depfile(depfile)
    dependencies.extend((
        PROJECT_ROOT / "Makefile",
        Path(__file__).resolve(),
        *link_inputs,
    ))
    dependencies = sorted(set(path.resolve() for path in dependencies))
    payload = {
        "schema_version": 1,
        "git": git_state(),
        "compiler": compiler_receipt(compiler),
        "command": command_parts(
            compiler, flags, includes, depfile, source,
            link_inputs, binary),
        "binary": {
            "path": dependency_key(binary),
            "sha256": sha256(binary),
        },
        "depfile": {
            "path": dependency_key(depfile),
            "sha256": sha256(depfile),
        },
        "dependencies": {
            dependency_key(path): sha256(path)
            for path in dependencies
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt_path)
    return payload


def validate_receipt(
        receipt_path: Path, binary: Path,
        root: Path = PROJECT_ROOT) -> list[str]:
    errors = []
    if not receipt_path.is_file():
        return [f"guest build receipt is missing: {receipt_path}"]
    try:
        payload = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"guest build receipt is unreadable: {error}"]
    if payload.get("schema_version") != 1:
        errors.append("unsupported guest build receipt schema")
    binary = binary.resolve()
    binary_row = payload.get("binary", {})
    if not binary.is_file() or binary_row.get("sha256") != sha256(binary):
        errors.append("guest binary hash does not match build receipt")
    try:
        if payload.get("git") != git_state(root):
            errors.append("guest binary was built from a different git state")
    except subprocess.CalledProcessError as error:
        errors.append(f"cannot verify git state: {error}")
    compiler = payload.get("compiler", {})
    compiler_path = Path(str(compiler.get("driver", "")))
    if not compiler_path.is_file() or \
            compiler.get("driver_sha256") != sha256(compiler_path):
        errors.append("guest compiler does not match build receipt")
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        errors.append("guest build receipt has no dependency hashes")
    else:
        for name, expected in dependencies.items():
            path = Path(name)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                errors.append(f"guest build dependency is missing: {name}")
            elif sha256(path) != expected:
                errors.append(f"guest build dependency changed: {name}")
    command = payload.get("command")
    if not isinstance(command, list) or not command:
        errors.append("guest build receipt has no compiler command")
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--receipt", type=Path, required=True)
    write.add_argument("--binary", type=Path, required=True)
    write.add_argument("--depfile", type=Path, required=True)
    write.add_argument("--compiler", required=True)
    write.add_argument("--flags", default="")
    write.add_argument("--includes", default="")
    write.add_argument("--source", type=Path, required=True)
    write.add_argument(
        "--link-input", type=Path, action="append", default=[])
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--binary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.action == "write":
        write_receipt(
            args.receipt, args.binary, args.depfile, args.compiler,
            args.flags, args.includes, args.source, args.link_input)
        return 0
    errors = validate_receipt(args.receipt, args.binary)
    for error in errors:
        print(f"[FAIL] {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
