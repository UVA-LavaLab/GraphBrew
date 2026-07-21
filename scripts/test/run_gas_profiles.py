#!/usr/bin/env python3
"""Run paired canonical and natural GAS verifier profiles."""

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
    extra_args: list[str],
    trials: int,
    threads: int,
    analysis: bool,
) -> str:
    command = [
        str(binary.resolve()),
        *graph_args(profile),
        "-n",
        str(trials),
        "-v",
        *extra_args,
    ]
    if analysis:
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
        f"{binary.name}/OMP={threads}: exit {completed.returncode}\n"
        f"{completed.stdout}\n{completed.stderr}",
    )
    require(
        len(VERIFICATION_PASS.findall(completed.stdout)) == trials,
        f"{binary.name}/OMP={threads}: expected {trials} verifier passes\n"
        f"{completed.stdout}\n{completed.stderr}",
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
    gas_algorithms = set(contract["implemented_gas_variants"])
    algorithms = {
        item["name"]: item
        for item in contract["algorithms"]
        if item["name"] in gas_algorithms
    }
    require(
        set(algorithms) == gas_algorithms,
        "GAS algorithm metadata differs",
    )

    cells = 0
    for name in sorted(algorithms):
        algorithm = algorithms[name]
        canonical = args.bin_dir / name
        gas = args.bin_dir / algorithm["gas_binary"]
        require(canonical.is_file(), f"missing canonical binary {canonical}")
        require(gas.is_file(), f"missing GAS binary {gas}")
        for profile_name in algorithm["profiles"]:
            profile = profiles[profile_name]
            profile_args = algorithm.get("profile_args", {}).get(
                profile_name, [])
            canonical_output = run(
                canonical,
                profile,
                profile_args,
                1,
                4,
                name == "cc",
            )
            canonical_components = None
            if name == "cc":
                match = COMPONENT_COUNT.search(canonical_output)
                require(
                    match is not None,
                    f"cc@{profile_name}: canonical count missing",
                )
                canonical_components = int(match.group(1))
            elif name == "pr":
                require(
                    TOTAL_ERROR.search(canonical_output) is not None,
                    f"pr@{profile_name}: canonical residual missing",
                )

            for threads in args.threads:
                gas_output = run(
                    gas,
                    profile,
                    profile_args,
                    1,
                    threads,
                    name == "cc",
                )
                if canonical_components is not None:
                    match = COMPONENT_COUNT.search(gas_output)
                    require(
                        match is not None,
                        f"cc_gas@{profile_name}: count missing",
                    )
                    require(
                        int(match.group(1)) == canonical_components,
                        f"cc_gas@{profile_name}/OMP={threads}: "
                        "component count differs",
                    )
                elif name == "pr":
                    require(
                        TOTAL_ERROR.search(gas_output) is not None,
                        f"pr_gas@{profile_name}/OMP={threads}: "
                        "residual missing",
                    )
                cells += 1
            print(
                f"PASS {name} canonical+gas @ {profile_name} "
                f"(OMP={','.join(map(str, args.threads))})"
            )

        if algorithm["source_picker_pair"]:
            pair_profile = profiles[algorithm["source_pair_profile"]]
            pair_trials = algorithm["source_pair_trials"]
            pair_args = algorithm.get("source_pair_args", [])
            run(canonical, pair_profile, pair_args, pair_trials, 4, False)
            for threads in args.threads:
                run(gas, pair_profile, pair_args, pair_trials, threads, False)
                cells += pair_trials
            print(
                f"PASS {name} paired sources "
                f"({pair_trials} trials; "
                f"OMP={','.join(map(str, args.threads))})"
            )

    print(
        "gas-profiles: PASS "
        f"({cells} verified GAS trials; {len(algorithms)} algorithms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
