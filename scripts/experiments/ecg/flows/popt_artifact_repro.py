#!/usr/bin/env python3
"""Reproduce the public P-OPT PageRank cache-miss direction.

This gate intentionally does NOT claim to reproduce the paper's 8-core Sniper
speedups or Figure 12(a)'s P-OPT-vs-GRASP result: the public artifact contains
neither the modified Sniper model nor GRASP/DBG. It validates the public
artifact's cache-only PageRank result on its four graphs:

  LRU, DRRIP, P-OPT, T-OPT; 24 MiB/16-way LLC; no prefetch; one PR sweep.

The exact artifact uses Pin 2.14. On hosts where that Pin cannot run, callers
may supply a compatibility Pin/tool root, but the output is labelled as a port.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


PINNED_ARTIFACT_COMMIT = "53b5021846690d0f3445428c6380e877ecf7a10e"
PINNED_GRASP_COMMIT = "6e3814430265fc4f2513c95ef131a6522bc9d389"
POLICIES = {
    "lru": ("baseline", "lru"),
    "drrip": ("baseline", "drrip"),
    "popt-8b": ("popt", "popt-8b"),
    "opt-ideal": ("opt-ideal", "opt-ideal"),
    "grasp": ("baseline", "grasp"),
}
PUBLIC_POLICIES = ("lru", "drrip", "popt-8b", "opt-ideal")
DEFAULT_GRAPHS = (
    "uk-2002",
    "hugebubbles-00020",
    "kron25-d4",
    "urand25-d4",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_total_llc_misses(text: str) -> int:
    values = []
    for line in text.splitlines():
        if "[LLC-STAT] Total Misses" not in line:
            continue
        values.append(int(float(line.rsplit(" ", 1)[-1])))
    if not values:
        raise ValueError("artifact output contains no LLC Total Misses")
    return sum(values)


def artifact_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root,
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


def expected_paths(
        root: Path, pin_root: Path, tool_root: Path,
        graphs: Iterable[str], policies: Iterable[str]) -> list[Path]:
    paths = [pin_root / "pin"]
    for graph in graphs:
        paths.append(root / "input-graphs" / f"{graph}.sg")
    for policy in policies:
        app_version, tool_name = POLICIES[policy]
        paths.extend((
            root / "applications" / app_version / "pr",
            tool_root / tool_name / "cache_pinsim.so",
        ))
    return paths


def build_command(
        root: Path, pin_root: Path, tool_root: Path,
        graph: str, policy: str) -> list[str]:
    app_version, tool_name = POLICIES[policy]
    return [
        str(pin_root / "pin"),
        "-t", str(tool_root / tool_name / "cache_pinsim.so"),
        "--",
        str(root / "applications" / app_version / "pr"),
        "-f", str(root / "input-graphs" / f"{graph}.sg"),
        "-n", "1", "-i", "1",
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pin-root", type=Path)
    parser.add_argument("--tool-root", type=Path)
    parser.add_argument("--grasp-source-root", type=Path)
    parser.add_argument("--port-label", default="pin2-exact")
    parser.add_argument(
        "--gate", choices=("public", "dbg-grasp"), default="public")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPHS))
    parser.add_argument(
        "--policies", nargs="+", choices=sorted(POLICIES),
        default=None)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.policies is None:
        args.policies = list(
            PUBLIC_POLICIES if args.gate == "public"
            else ("lru", "grasp", "popt-8b"))

    root = args.artifact_root.resolve()
    pin_root = (args.pin_root or root / "pin-2.14").resolve()
    tool_root = (args.tool_root or root / "simulators").resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    head = artifact_head(root)
    if head != PINNED_ARTIFACT_COMMIT:
        raise SystemExit(
            f"artifact commit mismatch: {head} != {PINNED_ARTIFACT_COMMIT}")

    inputs = expected_paths(
        root, pin_root, tool_root, args.graphs, args.policies)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing reproduction inputs:\n  " + "\n  ".join(missing))

    public_gate = args.gate == "public"
    grasp_source = (
        args.grasp_source_root.resolve()
        if args.grasp_source_root else None)
    if not public_gate:
        if grasp_source is None:
            raise SystemExit("--gate dbg-grasp requires --grasp-source-root")
        grasp_head = artifact_head(grasp_source)
        if grasp_head != PINNED_GRASP_COMMIT:
            raise SystemExit(
                f"GRASP commit mismatch: {grasp_head} != "
                f"{PINNED_GRASP_COMMIT}")
        grasp_inputs = (
            grasp_source / "trace-based-simulators/grasp.cpp",
            grasp_source / "trace-based-simulators/common.h",
        )
        missing_grasp = [
            str(path) for path in grasp_inputs if not path.is_file()]
        if missing_grasp:
            raise SystemExit(
                "missing GRASP sources:\n  " + "\n  ".join(missing_grasp))
    else:
        grasp_head = ""
        grasp_inputs = ()
    manifest = {
        "artifact_commit": head,
        "port_label": args.port_label,
        "gate": args.gate,
        "grasp_source_commit": grasp_head,
        "scope": (
            "public-artifact PageRank LLC-miss direction"
            if public_gate else
            "deterministic DBG P-OPT-vs-GRASP direction port"),
        "claimable": {
            "popt_vs_drrip": public_gate,
            "popt_vs_grasp_direction": not public_gate,
            "popt_vs_grasp_figure12_exact": False,
            "execution_time": False,
        },
        "graphs": list(args.graphs),
        "policies": list(args.policies),
        "inputs": {
            str(path): sha256(path)
            for path in (*inputs, *grasp_inputs)
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows = []
    results_path = out_dir / "results.csv"
    if args.resume and results_path.exists():
        with results_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["exit_code"] = int(row["exit_code"])
            row["llc_demand_misses"] = int(row["llc_demand_misses"])
    completed = {
        (row["graph"], row["policy"]) for row in rows
        if row.get("status") == "ok"
    }
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for graph in args.graphs:
        for policy in args.policies:
            if (graph, policy) in completed:
                print(f"[resume] {graph}/{policy}", flush=True)
                continue
            command = build_command(
                root, pin_root, tool_root, graph, policy)
            print("$", " ".join(command), flush=True)
            if args.dry_run:
                continue
            log = out_dir / f"{graph}__{policy}.stdout"
            err = out_dir / f"{graph}__{policy}.stderr"
            with log.open("w") as stdout, err.open("w") as stderr:
                result = subprocess.run(
                    command, cwd=root / "scripts", env=env,
                    stdout=stdout, stderr=stderr, timeout=args.timeout,
                    check=False)
            text = log.read_text(errors="ignore")
            row = {
                "graph": graph,
                "policy": policy,
                "exit_code": result.returncode,
                "llc_demand_misses": parse_total_llc_misses(text),
                "stdout_sha256": sha256(log),
                "stderr_sha256": sha256(err),
                "status": "ok" if result.returncode == 0 else "error",
            }
            rows.append(row)
            write_csv(results_path, rows)

    if args.dry_run:
        return 0

    by_graph = {
        graph: {
            row["policy"]: row for row in rows if row["graph"] == graph
        }
        for graph in args.graphs
    }
    direction = {}
    complete = all(
        set(by_graph[graph]) == set(args.policies) and
        all(row["status"] == "ok" for row in by_graph[graph].values())
        for graph in args.graphs)
    reference = "drrip" if public_gate else "grasp"
    for graph, policies in by_graph.items():
        if "popt-8b" in policies and reference in policies:
            direction[graph] = (
                policies["popt-8b"]["llc_demand_misses"] <
                policies[reference]["llc_demand_misses"])
    direction_evaluated = {"popt-8b", reference} <= set(args.policies)
    passed = (
        complete and direction_evaluated and
        len(direction) == len(args.graphs) and all(direction.values()))
    (out_dir / "complete.json").write_text(json.dumps({
        "complete": complete,
        "direction_evaluated": direction_evaluated,
        "reference_policy": reference,
        "passed_popt_better_than_reference_every_graph": passed,
        "popt_better_than_reference": direction,
        "popt_vs_grasp_figure12_exact": False,
    }, indent=2, sort_keys=True) + "\n")
    return 0 if complete and (not direction_evaluated or passed) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
