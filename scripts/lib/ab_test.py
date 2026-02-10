#!/usr/bin/env python3
"""
A/B test: compare AdaptiveOrder (-o 14) vs Original (-o 0) across all graphs.

Replaces the legacy ab_test.sh shell script with a pure-Python implementation.

Usage (standalone):
    python -m scripts.lib.ab_test
    python -m scripts.lib.ab_test --trials 5 --benchmarks bfs pr cc

Usage (library):
    from scripts.lib.ab_test import run_ab_test
    summary = run_ab_test(trials=3)
    print(f"Overall speedup: {summary['overall_speedup']:.2f}x")
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import BIN_DIR, GRAPHS_DIR, Logger

log = Logger()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ABResult:
    """Result for a single (graph, benchmark) A/B comparison."""
    graph: str
    benchmark: str
    original_time: float
    adaptive_time: float

    @property
    def speedup(self) -> float:
        if self.adaptive_time > 0:
            return self.original_time / self.adaptive_time
        return float("inf")

    @property
    def winner(self) -> str:
        """ADAP if >5% faster, ORIG if >5% slower, TIE otherwise."""
        r = self.speedup
        if r > 1.05:
            return "ADAP"
        elif r < 0.95:
            return "ORIG"
        return "TIE"


@dataclass
class ABSummary:
    """Aggregate summary of an A/B test run."""
    results: List[ABResult] = field(default_factory=list)
    total_original: float = 0.0
    total_adaptive: float = 0.0

    @property
    def overall_speedup(self) -> float:
        if self.total_adaptive > 0:
            return self.total_original / self.total_adaptive
        return float("inf")

    @property
    def wins(self) -> int:
        return sum(1 for r in self.results if r.winner == "ADAP")

    @property
    def ties(self) -> int:
        return sum(1 for r in self.results if r.winner == "TIE")

    @property
    def losses(self) -> int:
        return sum(1 for r in self.results if r.winner == "ORIG")

    def to_dict(self) -> dict:
        return {
            "overall_speedup": round(self.overall_speedup, 4),
            "total_original": round(self.total_original, 6),
            "total_adaptive": round(self.total_adaptive, 6),
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "results": [
                {
                    "graph": r.graph,
                    "benchmark": r.benchmark,
                    "original_time": r.original_time,
                    "adaptive_time": r.adaptive_time,
                    "speedup": round(r.speedup, 2),
                    "winner": r.winner,
                }
                for r in self.results
            ],
        }


# ============================================================================
# Graph Discovery
# ============================================================================

# Default graph ordering (sorted by size for predictable output)
DEFAULT_GRAPHS = [
    "soc-Epinions1", "soc-Slashdot0902", "cnr-2000", "web-BerkStan",
    "web-Google", "com-Youtube", "as-Skitter", "roadNet-CA",
    "wiki-topcats", "cit-Patents", "soc-LiveJournal1",
]

DEFAULT_BENCHMARKS = ["bfs", "pr", "pr_spmv", "cc", "cc_sv"]


def discover_ab_graphs(
    graphs_dir: str = None,
    graph_names: List[str] = None,
) -> List[Tuple[str, str]]:
    """
    Discover graphs for A/B testing. Returns (name, sg_path) pairs.
    If graph_names given, uses that order; otherwise uses DEFAULT_GRAPHS
    filtered to what actually exists.
    """
    graphs_dir = Path(graphs_dir or GRAPHS_DIR)
    candidates = graph_names or DEFAULT_GRAPHS
    found = []
    for name in candidates:
        sg = graphs_dir / name / f"{name}.sg"
        if sg.is_file():
            found.append((name, str(sg)))
    if not graph_names:
        # Also pick up any .sg graphs not in DEFAULT_GRAPHS
        for sg in sorted(graphs_dir.glob("*/*.sg")):
            name = sg.parent.name
            if name not in [f[0] for f in found]:
                found.append((name, str(sg)))
    return found


# ============================================================================
# Single benchmark run helper
# ============================================================================

def _run_bench(
    binary: str,
    sg_path: str,
    algo_id: int,
    trials: int,
    timeout: int,
) -> Optional[float]:
    """Run a benchmark and return the average time, or None on failure."""
    cmd = [binary, "-f", sg_path, "-a", "0", "-o", str(algo_id), "-n", str(trials)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None
        m = re.search(r"Average Time:\s+([\d.]+)", result.stdout)
        return float(m.group(1)) if m else None
    except (subprocess.TimeoutExpired, Exception):
        return None


# ============================================================================
# Main Runner
# ============================================================================

def run_ab_test(
    graphs_dir: str = None,
    graph_names: List[str] = None,
    benchmarks: List[str] = None,
    trials: int = 3,
    timeout: int = 300,
    bin_dir: str = None,
) -> ABSummary:
    """
    Run A/B test comparing AdaptiveOrder (-o 14) vs Original (-o 0).

    For each (graph, benchmark), runs both configurations and records
    the average execution time. AdaptiveOrder total time includes the
    reorder cost (it's baked into the -o 14 run).

    Args:
        graphs_dir: Directory containing graph subdirectories with .sg files
        graph_names: Optional list of specific graph names
        benchmarks: Benchmark types (default: bfs, pr, pr_spmv, cc, cc_sv)
        trials: Trials per run
        timeout: Timeout per benchmark invocation
        bin_dir: Directory containing benchmark binaries

    Returns:
        ABSummary with per-(graph, bench) results and aggregate stats.
    """
    benchmarks = benchmarks or DEFAULT_BENCHMARKS
    bin_dir = str(bin_dir or BIN_DIR)

    graphs = discover_ab_graphs(graphs_dir, graph_names)
    if not graphs:
        log.error("No .sg graphs found for A/B testing")
        return ABSummary()

    # Header
    print(
        f"{'Graph':<25} {'Bench':<8} {'Original(s)':>12} {'Adaptive(s)':>12} {'Speedup':>8}"
    )
    print(
        f"{'-'*25} {'-'*8} {'-'*12} {'-'*12} {'-'*8}"
    )

    summary = ABSummary()

    for graph_name, sg_path in graphs:
        for bench in benchmarks:
            binary = os.path.join(bin_dir, bench)
            if not os.path.isfile(binary):
                continue

            orig_time = _run_bench(binary, sg_path, 0, trials, timeout)
            adap_time = _run_bench(binary, sg_path, 14, trials, timeout)

            if orig_time is not None and adap_time is not None and orig_time > 0 and adap_time > 0:
                r = ABResult(
                    graph=graph_name,
                    benchmark=bench,
                    original_time=orig_time,
                    adaptive_time=adap_time,
                )
                summary.results.append(r)
                summary.total_original += orig_time
                summary.total_adaptive += adap_time

                print(
                    f"{graph_name:<25} {bench:<8} {orig_time:>12.5f} "
                    f"{adap_time:>12.5f} {r.speedup:>7.2f}x"
                )

    # Summary
    print(f"\n=== Summary ===")
    print(
        f"Total Original: {summary.total_original:.4f}s  "
        f"Total Adaptive: {summary.total_adaptive:.4f}s"
    )
    print(f"Overall speedup: {summary.overall_speedup:.2f}x")
    print(f"Wins: {summary.wins}  Ties: {summary.ties}  Losses: {summary.losses}")

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="A/B test: AdaptiveOrder vs Original on all .sg graphs",
    )
    parser.add_argument("--graphs-dir", default=None,
                        help="Directory containing graph subdirectories")
    parser.add_argument("--graphs", nargs="+", default=None,
                        help="Specific graph names to test")
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        help=f"Benchmark types (default: {DEFAULT_BENCHMARKS})")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per benchmark (default: 3)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per run in seconds (default: 300)")
    parser.add_argument("--bin-dir", default=None,
                        help="Directory containing benchmark binaries")
    args = parser.parse_args()

    run_ab_test(
        graphs_dir=args.graphs_dir,
        graph_names=args.graphs,
        benchmarks=args.benchmarks,
        trials=args.trials,
        timeout=args.timeout,
        bin_dir=args.bin_dir,
    )


if __name__ == "__main__":
    main()
