"""
Hardware performance counter collection via Linux ``perf stat``.

Wraps GraphBrew benchmark binaries with ``perf stat`` to collect L1/LLC
cache miss counts, instructions, and cycles.  Results are returned as
structured dicts suitable for JSON serialisation.

Requirements:
    - Linux with ``perf`` installed (``linux-tools-common`` package)
    - ``/proc/sys/kernel/perf_event_paranoid`` <= 2  (or run as root)

Usage::

    from scripts.lib.analysis.perf_counters import run_with_perf_counters

    result = run_with_perf_counters(
        binary="bench/bin/pr",
        graph_path="results/graphs/web-Google/web-Google.sg",
        reorder_flag="-o 12",
        extra_args="-s -n 3",
    )
    print(result)
    # {'L1-dcache-load-misses': 123456, 'LLC-load-misses': 7890, ...}
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Default hardware events to collect
DEFAULT_EVENTS = [
    "L1-dcache-load-misses",
    "L1-dcache-loads",
    "LLC-load-misses",
    "LLC-loads",
    "instructions",
    "cycles",
    "cache-misses",
    "cache-references",
]

# Project root (two levels up from this file → scripts/lib/analysis/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _check_perf_available() -> bool:
    """Return True if ``perf`` is on PATH and usable."""
    if not shutil.which("perf"):
        return False
    try:
        result = subprocess.run(
            ["perf", "stat", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _parse_perf_output(stderr: str) -> Dict[str, int]:
    """Parse ``perf stat`` stderr into {event: count} dict.

    ``perf stat`` prints lines like::

        1,234,567      L1-dcache-load-misses     # ...
            <not supported>  LLC-loads

    We extract the integer count and event name from each line.
    """
    counters: Dict[str, int] = {}
    for line in stderr.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Performance"):
            continue
        # Skip "<not supported>" / "<not counted>" lines
        if "<not" in line:
            continue
        # Match: count (with commas/dots as thousand sep)  event-name
        m = re.match(r"^([\d,.\s]+?)\s+([\w-]+)", line)
        if m:
            count_str = m.group(1).replace(",", "").replace(".", "").strip()
            event = m.group(2)
            try:
                counters[event] = int(count_str)
            except ValueError:
                pass
    return counters


def run_with_perf_counters(
    binary: str,
    graph_path: str,
    reorder_flag: str = "",
    extra_args: str = "-s -n 1",
    events: Optional[List[str]] = None,
    timeout: int = 600,
    cwd: Optional[str] = None,
) -> Dict[str, object]:
    """Run a GraphBrew benchmark binary under ``perf stat``.

    Args:
        binary: Path to benchmark binary (e.g. ``bench/bin/pr``).
        graph_path: Path to ``.sg`` graph file.
        reorder_flag: Reorder flag string (e.g. ``-o 12`` or ``-o 9``).
        extra_args: Additional CLI args for the binary.
        events: List of perf event names.  Defaults to :data:`DEFAULT_EVENTS`.
        timeout: Max seconds before killing the process.
        cwd: Working directory (defaults to project root).

    Returns:
        Dict with keys:
            - ``counters``: {event_name: count}
            - ``binary``: binary path used
            - ``graph``: graph path used
            - ``reorder``: reorder flag
            - ``stdout``: raw benchmark stdout (for parsing times)
            - ``success``: bool
            - ``error``: error message if failed
    """
    if not _check_perf_available():
        return {
            "counters": {},
            "binary": binary,
            "graph": graph_path,
            "reorder": reorder_flag,
            "stdout": "",
            "success": False,
            "error": "perf is not available. Install linux-tools-common.",
        }

    if events is None:
        events = DEFAULT_EVENTS

    work_dir = cwd or str(_PROJECT_ROOT)

    # Build command
    event_str = ",".join(events)
    bench_cmd = f"{binary} -f {graph_path} {reorder_flag} {extra_args}"
    cmd = f"perf stat -e {event_str} -- {bench_cmd}"

    log.info(f"perf: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return {
            "counters": {},
            "binary": binary,
            "graph": graph_path,
            "reorder": reorder_flag,
            "stdout": "",
            "success": False,
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "counters": {},
            "binary": binary,
            "graph": graph_path,
            "reorder": reorder_flag,
            "stdout": "",
            "success": False,
            "error": str(e),
        }

    counters = _parse_perf_output(result.stderr)

    return {
        "counters": counters,
        "binary": binary,
        "graph": graph_path,
        "reorder": reorder_flag,
        "stdout": result.stdout,
        "success": result.returncode == 0,
        "error": "" if result.returncode == 0 else f"exit code {result.returncode}",
    }


def run_perf_sweep(
    graph_path: str,
    algorithms: Dict[str, str],
    benchmarks: List[str] = None,
    num_trials: int = 3,
    events: Optional[List[str]] = None,
    timeout: int = 600,
    bin_dir: str = "bench/bin",
) -> List[Dict]:
    """Run perf counters across multiple algorithms and benchmarks.

    Args:
        graph_path: Path to ``.sg`` graph file.
        algorithms: Dict mapping name → reorder flag (e.g.
            ``{"ORIGINAL": "-o 0", "Gorder": "-o 9", "GraphBrew": "-o 12"}``).
        benchmarks: List of benchmark names (e.g. ``["pr", "bfs"]``).
            Defaults to ``["pr", "bfs", "sssp", "cc"]``.
        num_trials: Number of trials per configuration.
        events: Perf events to collect.
        timeout: Timeout per run in seconds.
        bin_dir: Directory containing benchmark binaries.

    Returns:
        List of result dicts with algorithm name, benchmark, and counters.
    """
    if benchmarks is None:
        benchmarks = ["pr", "bfs", "sssp", "cc"]

    results = []
    graph_name = Path(graph_path).stem

    for bench in benchmarks:
        binary = f"{bin_dir}/{bench}"
        for algo_name, reorder_flag in algorithms.items():
            log.info(f"perf sweep: {graph_name} × {bench} × {algo_name}")
            result = run_with_perf_counters(
                binary=binary,
                graph_path=graph_path,
                reorder_flag=reorder_flag,
                extra_args=f"-s -n {num_trials}",
                events=events,
                timeout=timeout,
            )
            result["algorithm"] = algo_name
            result["benchmark"] = bench
            result["graph_name"] = graph_name
            result["num_trials"] = num_trials
            results.append(result)

    return results


def save_perf_results(results: List[Dict], output_path: str) -> None:
    """Save perf counter results to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved perf results to {output_path}")


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Collect hardware perf counters for GraphBrew benchmarks"
    )
    parser.add_argument(
        "-f", "--graph", required=True, help="Path to .sg graph file"
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        default=["pr", "bfs", "sssp", "cc"],
        help="Benchmark kernels to run",
    )
    parser.add_argument(
        "-n", "--trials", type=int, default=3, help="Trials per config"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/perf_counters.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ORIGINAL:-o 0", "Gorder:-o 9", "GraphBrew:-o 12"],
        help='Algorithm specs as "Name:-o N" pairs',
    )
    args = parser.parse_args()

    # Parse algorithm specs
    algos = {}
    for spec in args.algorithms:
        if ":" not in spec:
            print(f"Invalid algorithm spec: {spec} (expected Name:-o N)")
            sys.exit(1)
        name, flag = spec.split(":", 1)
        algos[name] = flag

    results = run_perf_sweep(
        graph_path=args.graph,
        algorithms=algos,
        benchmarks=args.benchmarks,
        num_trials=args.trials,
    )

    save_perf_results(results, args.output)
    print(f"\nCollected {len(results)} perf measurements → {args.output}")

    # Print summary
    for r in results:
        if r["success"] and r["counters"]:
            c = r["counters"]
            l1_miss = c.get("L1-dcache-load-misses", "N/A")
            llc_miss = c.get("LLC-load-misses", "N/A")
            print(
                f"  {r['graph_name']:20s} {r['benchmark']:6s} {r['algorithm']:20s} "
                f"L1miss={l1_miss:>12}  LLCmiss={llc_miss:>12}"
            )
        else:
            print(
                f"  {r.get('graph_name','?'):20s} {r.get('benchmark','?'):6s} "
                f"{r.get('algorithm','?'):20s} FAILED: {r.get('error','')}"
            )
