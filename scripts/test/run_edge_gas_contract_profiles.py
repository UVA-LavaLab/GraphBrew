#!/usr/bin/env python3
"""Run one verifier-backed smoke profile for every canonical algorithm."""

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


def profile_args(profile: dict, trials: int) -> list[str]:
    command: list[str]
    if "path" in profile:
        command = ["-f", profile["path"]]
    else:
        command = ["-g", str(profile["generator_scale"])]
    if profile.get("symmetrize", False):
        command.append("-s")
    command.extend(["-n", str(trials), "-v"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=ROOT / "bench" / "bin",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require(args.threads > 0, "--threads must be positive")

    contract = json.loads(CONTRACT.read_text())
    profiles = contract["test_profiles"]
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(args.threads)

    for algorithm in contract["algorithms"]:
        name = algorithm["name"]
        binary = args.bin_dir.resolve() / name
        require(binary.is_file(), f"{name}: missing binary {binary}")
        profile_name = algorithm["smoke_profile"]
        profile = profiles[profile_name]
        command = [
            str(binary),
            *profile_args(profile, 1),
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
            f"{name}@{profile_name}: exit {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}",
        )
        require(
            VERIFICATION_PASS.search(completed.stdout) is not None,
            f"{name}@{profile_name}: verifier did not pass\n"
            f"{completed.stdout}\n{completed.stderr}",
        )
        print(f"PASS {name} @ {profile_name}")
        if algorithm["source_picker_pair"]:
            pair_profile_name = algorithm["source_pair_profile"]
            pair_profile = profiles[pair_profile_name]
            pair_trials = algorithm["source_pair_trials"]
            pair_command = [
                str(binary),
                *profile_args(pair_profile, pair_trials),
                *algorithm.get("source_pair_args", []),
            ]
            paired = subprocess.run(
                pair_command,
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            require(
                paired.returncode == 0,
                f"{name}@source_pair: exit {paired.returncode}\n"
                f"{paired.stdout}\n{paired.stderr}",
            )
            require(
                len(VERIFICATION_PASS.findall(paired.stdout)) == pair_trials,
                f"{name}@source_pair: expected {pair_trials} verifier passes\n"
                f"{paired.stdout}\n{paired.stderr}",
            )
            print(
                f"PASS {name} @ source_pair "
                f"({pair_trials} random trials)"
            )

    print(
        "edge-gas-contract-profiles: PASS "
        f"({len(contract['algorithms'])} canonical algorithms; "
        f"OMP={args.threads})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
