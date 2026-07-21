#!/usr/bin/env python3
"""Run two verifier-backed trials for every edge and GAS binary."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=ROOT / "bench" / "bin",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    require(args.threads > 0, "--threads must be positive")
    require(args.trials >= 2, "--trials must be at least two")

    contract = json.loads(CONTRACT.read_text())
    profiles = contract["test_profiles"]
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(args.threads)
    binaries: list[tuple[str, str, dict]] = []
    for algorithm in contract["algorithms"]:
        if "edge_binary" in algorithm:
            binaries.append(
                ("edge", algorithm["edge_binary"], algorithm))
        if "gas_binary" in algorithm:
            binaries.append(
                ("gas", algorithm["gas_binary"], algorithm))

    for paradigm, name, algorithm in binaries:
        binary = args.bin_dir / name
        require(binary.is_file(), f"missing binary {binary}")
        profile_name = algorithm["smoke_profile"]
        profile = profiles[profile_name]
        command = [
            str(binary.resolve()),
            *graph_args(profile),
            "-n",
            str(args.trials),
            "-v",
            *algorithm.get("profile_args", {}).get(profile_name, []),
        ]
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
            f"{name}: exit {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}",
        )
        require(
            len(VERIFICATION_PASS.findall(completed.stdout)) ==
            args.trials,
            f"{name}: expected {args.trials} verifier passes\n"
            f"{completed.stdout}\n{completed.stderr}",
        )
        print(
            f"PASS {name} ({paradigm}; "
            f"{args.trials} trials; OMP={args.threads})"
        )

    print(
        "edge-gas-repeatability: PASS "
        f"({len(binaries)} binaries; "
        f"{len(binaries) * args.trials} verified trials)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
