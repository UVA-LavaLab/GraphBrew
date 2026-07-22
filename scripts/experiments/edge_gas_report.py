#!/usr/bin/env python3
"""Emit a verified, machine-local edge/GAS timing report with no speed gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "bench" / "contracts" / "edge_gas_algorithms.json"
AVERAGE_TIME = re.compile(r"Average Time:\s+([0-9.eE+-]+)")
TRIAL_TIME = re.compile(r"Trial Time:\s+([0-9.eE+-]+)")
GRAPH_STATS = re.compile(
    r"Graph has\s+(\d+)\s+nodes and\s+(\d+)\s+(?:un)?directed edges"
)
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


def selected_profile(algorithm: dict) -> str:
    for profile in algorithm["profiles"]:
        if profile.startswith("synthetic_"):
            return profile
    return algorithm["smoke_profile"]

def performance_args(algorithm: dict, profile_name: str) -> list[str]:
    args = list(algorithm.get("profile_args", {}).get(profile_name, []))
    if not algorithm.get("source_picker_pair", False):
        return args
    filtered: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "-r" and index + 1 < len(args):
            index += 2
            continue
        filtered.append(args[index])
        index += 1
    return filtered


def run(
    binary: Path,
    profile: dict,
    extra_args: list[str],
    trials: int,
    threads: int,
) -> tuple[float, float, list[float], int, int]:
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    completed = subprocess.run(
        [
            str(binary.resolve()),
            *graph_args(profile),
            "-n",
            str(trials),
            "-v",
            *extra_args,
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        completed.returncode == 0,
        f"{binary.name}: exit {completed.returncode}\n"
        f"{completed.stdout}\n{completed.stderr}",
    )
    require(
        len(VERIFICATION_PASS.findall(completed.stdout)) == trials,
        f"{binary.name}: verifier did not pass every trial",
    )
    time_match = AVERAGE_TIME.search(completed.stdout)
    trial_times = [float(value) for value in TRIAL_TIME.findall(completed.stdout)]
    graph_match = GRAPH_STATS.search(completed.stdout)
    require(time_match is not None, f"{binary.name}: average time missing")
    require(
        len(trial_times) == trials,
        f"{binary.name}: expected {trials} trial timings",
    )
    require(graph_match is not None, f"{binary.name}: graph stats missing")
    return (
        float(time_match.group(1)),
        statistics.median(trial_times),
        trial_times,
        int(graph_match.group(1)),
        int(graph_match.group(2)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=ROOT / "bench" / "bin",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--synthetic-scale", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(args.threads > 0, "--threads must be positive")
    require(args.trials > 0, "--trials must be positive")
    require(args.synthetic_scale > 0, "--synthetic-scale must be positive")

    contract = json.loads(CONTRACT.read_text())
    profiles = contract["test_profiles"]
    work = contract["performance_contract"]
    records: list[dict] = []
    for algorithm in contract["algorithms"]:
        name = algorithm["name"]
        profile_name = selected_profile(algorithm)
        profile = dict(profiles[profile_name])
        profile_label = profile_name
        if "generator_scale" in profile:
            profile["generator_scale"] = args.synthetic_scale
            profile_label = f"{profile_name}@scale{args.synthetic_scale}"
        extra_args = performance_args(algorithm, profile_name)
        variants = [("canonical", name)]
        variants.append(("edge", algorithm["edge_binary"]))
        if "gas_binary" in algorithm:
            variants.append(("gas", algorithm["gas_binary"]))

        canonical_median = None
        pending: list[dict] = []
        for paradigm, binary_name in variants:
            average, median, trial_times, vertices, edges = run(
                args.bin_dir / binary_name,
                profile,
                extra_args,
                args.trials,
                args.threads,
            )
            if paradigm == "canonical":
                canonical_median = median
            pending.append(
                {
                    "algorithm": name,
                    "paradigm": paradigm,
                    "binary": binary_name,
                    "profile": profile_label,
                    "vertices": vertices,
                    "input_edges": edges,
                    "average_seconds": average,
                    "median_seconds": median,
                    "trial_seconds": trial_times,
                    "nominal_input_medges_per_second":
                        (edges / median / 1_000_000) if median > 0 else None,
                    "work_contract":
                        work[name].get(
                            "gas_work" if paradigm == "gas" else "edge_work"
                        ),
                    "balance_contract": work[name]["edge_balance"],
                    "ownership_contract": work[name]["ownership"],
                }
            )
        require(canonical_median is not None, f"{name}: canonical run missing")
        for record in pending:
            record["speedup_vs_canonical"] = (
                canonical_median / record["median_seconds"]
                if record["median_seconds"] > 0
                else None
            )
            records.append(record)

    report = {
        "schema": "graphbrew.edge-gas-performance-report",
        "threads": args.threads,
        "trials": args.trials,
        "synthetic_scale": args.synthetic_scale,
        "verified": True,
        "hard_speed_gate": False,
        "source_policy":
            "Paired SourcePicker algorithms use their deterministic source "
            "sequence rather than a fixed -r source.",
        "metric_note":
            "Nominal input-edge rate is only an input-size normalization and "
            "is not actual examined-edge work for any schedule.",
        "records": records,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
