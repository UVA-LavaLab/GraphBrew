#!/usr/bin/env python3
"""Run paired canonical and dense edge verifier profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "bench" / "contracts" / "edge_gas_algorithms.json"
VERIFICATION_PASS = re.compile(r"Verification:\s+PASS")
COMPONENT_COUNT = re.compile(r"There are\s+(\d+)\s+components")
TOTAL_ERROR = re.compile(r"Total Error:\s+([0-9.eE+-]+)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def graph_args(profile: dict) -> list[str]:
    if "path" in profile:
        command = ["-f", profile["path"]]
    else:
        command = ["-g", str(profile["generator_scale"])]
    if profile.get("symmetrize", False):
        command.append("-s")
    return command


def run(
    binary: Path,
    profile: dict,
    algorithm: dict,
    profile_name: str,
    threads: int,
) -> str:
    command = [
        str(binary.resolve()),
        *graph_args(profile),
        "-n",
        "1",
        "-v",
        *algorithm.get("profile_args", {}).get(profile_name, []),
    ]
    if algorithm["name"] in {"cc", "cc_sv"}:
        command.append("-a")
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        completed.returncode == 0,
        f"{binary.name}@{profile_name}/OMP={threads}: "
        f"exit {completed.returncode}\n"
        f"{completed.stdout}\n{completed.stderr}",
    )
    require(
        VERIFICATION_PASS.search(completed.stdout) is not None,
        f"{binary.name}@{profile_name}/OMP={threads}: "
        f"verifier did not pass\n{completed.stdout}\n{completed.stderr}",
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=ROOT / "bench" / "bin",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    args = parser.parse_args()
    require(
        args.threads and all(thread > 0 for thread in args.threads),
        "--threads values must be positive",
    )

    contract = json.loads(CONTRACT.read_text())
    profiles = contract["test_profiles"]
    implemented = set(contract["implemented_edge_variants"])
    algorithms = {
        item["name"]: item
        for item in contract["algorithms"]
        if item["name"] in implemented
    }
    require(
        set(algorithms) == implemented,
        "implemented edge algorithm metadata differs",
    )

    cells = 0
    for name in sorted(algorithms):
        algorithm = algorithms[name]
        canonical = args.bin_dir / name
        edge = args.bin_dir / algorithm["edge_binary"]
        require(canonical.is_file(), f"missing canonical binary {canonical}")
        require(edge.is_file(), f"missing edge binary {edge}")
        for profile_name in algorithm["profiles"]:
            profile = profiles[profile_name]
            canonical_output = run(
                canonical, profile, algorithm, profile_name, 4)
            canonical_components = None
            if name in {"cc", "cc_sv"}:
                match = COMPONENT_COUNT.search(canonical_output)
                require(
                    match is not None,
                    f"{name}@{profile_name}: canonical component count missing",
                )
                canonical_components = int(match.group(1))
            else:
                require(
                    TOTAL_ERROR.search(canonical_output) is not None,
                    f"{name}@{profile_name}: canonical residual missing",
                )

            for threads in args.threads:
                edge_output = run(
                    edge, profile, algorithm, profile_name, threads)
                if canonical_components is not None:
                    match = COMPONENT_COUNT.search(edge_output)
                    require(
                        match is not None,
                        f"{edge.name}@{profile_name}: "
                        "component count missing",
                    )
                    require(
                        int(match.group(1)) == canonical_components,
                        f"{edge.name}@{profile_name}/OMP={threads}: "
                        "component count differs",
                    )
                else:
                    require(
                        TOTAL_ERROR.search(edge_output) is not None,
                        f"{edge.name}@{profile_name}/OMP={threads}: "
                        "residual missing",
                    )
                cells += 1
            print(
                f"PASS {name} canonical+edge @ {profile_name} "
                f"(OMP={','.join(map(str, args.threads))})"
            )

    print(
        "edge-dense-profiles: PASS "
        f"({cells} edge cells; {len(algorithms)} algorithms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
