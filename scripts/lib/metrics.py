"""
Ordering quality and amortization metrics for GraphBrew experiments.

Computes derived metrics from benchmark + reorder results:
  - Amortization iterations (break-even point)
  - Effective speedup at N iterations
  - Ordering quality deltas (packing factor, FEF)
  - End-to-end ROI curves

These are POST-HOC metrics computed from existing BenchmarkResult and
ReorderResult data. They do NOT require additional benchmark runs.

Usage:
    from lib.metrics import compute_amortization_report
    report = compute_amortization_report(benchmark_results, reorder_results)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import logging

log = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AmortizationMetrics:
    """Amortization analysis for a single (graph, algorithm, benchmark) triple."""
    graph: str
    algorithm: str
    benchmark: str
    
    # Raw inputs
    original_kernel_time: float     # Kernel time with ORIGINAL ordering (seconds)
    reordered_kernel_time: float    # Kernel time with this algorithm's ordering (seconds)
    reorder_time: float             # Time to compute the reordering (seconds)
    
    # Derived metrics
    kernel_speedup: float = 0.0           # original / reordered (>1 = faster)
    time_saved_per_iter: float = 0.0      # Seconds saved per iteration
    amortization_iters: float = float('inf')  # Iterations to break even
    e2e_speedup_at_1: float = 0.0         # End-to-end speedup at 1 iteration
    e2e_speedup_at_10: float = 0.0        # End-to-end speedup at 10 iterations
    e2e_speedup_at_100: float = 0.0       # End-to-end speedup at 100 iterations
    roi_per_iter: float = 0.0             # Return on investment per iteration after break-even
    
    def __post_init__(self):
        if self.original_kernel_time > 0 and self.reordered_kernel_time > 0:
            self.kernel_speedup = self.original_kernel_time / self.reordered_kernel_time
            self.time_saved_per_iter = self.original_kernel_time - self.reordered_kernel_time
            
            if self.time_saved_per_iter > 0:
                self.amortization_iters = self.reorder_time / self.time_saved_per_iter
                self.roi_per_iter = self.time_saved_per_iter
            else:
                self.amortization_iters = float('inf')
                self.roi_per_iter = self.time_saved_per_iter  # Negative = losing per iter
            
            for n in [1, 10, 100]:
                e2e_orig = n * self.original_kernel_time
                e2e_reord = self.reorder_time + n * self.reordered_kernel_time
                speedup = e2e_orig / e2e_reord if e2e_reord > 0 else 0.0
                if n == 1: self.e2e_speedup_at_1 = speedup
                elif n == 10: self.e2e_speedup_at_10 = speedup
                elif n == 100: self.e2e_speedup_at_100 = speedup

    def e2e_speedup_at(self, n: int) -> float:
        """End-to-end speedup at N iterations."""
        e2e_orig = n * self.original_kernel_time
        e2e_reord = self.reorder_time + n * self.reordered_kernel_time
        return e2e_orig / e2e_reord if e2e_reord > 0 else 0.0
    
    def is_profitable(self) -> bool:
        """True if reordering ever pays off (kernel speedup > 1)."""
        return self.kernel_speedup > 1.0
    
    def break_even_summary(self) -> str:
        """Human-readable break-even summary."""
        if not self.is_profitable():
            return f"NEVER (kernel {self.kernel_speedup:.2f}x = slower)"
        if self.amortization_iters < 1:
            return f"INSTANT ({self.amortization_iters:.2f} iters)"
        if self.amortization_iters < 10:
            return f"FAST ({self.amortization_iters:.1f} iters)"
        if self.amortization_iters < 100:
            return f"OK ({self.amortization_iters:.0f} iters)"
        return f"SLOW ({self.amortization_iters:.0f} iters)"


@dataclass
class VariantComparison:
    """Head-to-head comparison of two variants on a (graph, benchmark)."""
    graph: str
    benchmark: str
    variant_a: str           # e.g. "rabbit:csr"
    variant_b: str           # e.g. "graphbrew:hrab"
    
    # Kernel comparison
    kernel_a: float = 0.0
    kernel_b: float = 0.0
    kernel_winner: str = ""
    kernel_speedup: float = 0.0  # b_time / a_time (>1 = a is faster)
    
    # Reorder cost comparison
    reorder_a: float = 0.0
    reorder_b: float = 0.0
    reorder_ratio: float = 0.0  # b_reorder / a_reorder
    
    # End-to-end at various iteration counts
    e2e_winner_at_1: str = ""
    e2e_winner_at_10: str = ""
    e2e_winner_at_100: str = ""
    crossover_iters: float = float('inf')  # Iterations where slower-to-reorder variant wins


@dataclass
class AmortizationReport:
    """Full amortization report across all graphs/benchmarks."""
    entries: List[AmortizationMetrics] = field(default_factory=list)
    comparisons: List[VariantComparison] = field(default_factory=list)
    
    def profitable_count(self) -> int:
        return sum(1 for e in self.entries if e.is_profitable())
    
    def median_amortization(self) -> float:
        """Median iterations to amortize across profitable entries."""
        profitable = sorted(e.amortization_iters for e in self.entries 
                          if e.is_profitable() and e.amortization_iters < float('inf'))
        if not profitable:
            return float('inf')
        mid = len(profitable) // 2
        return profitable[mid]
    
    def geo_mean_kernel_speedup(self) -> float:
        """Geometric mean of kernel speedups (all entries)."""
        vals = [e.kernel_speedup for e in self.entries if e.kernel_speedup > 0]
        if not vals:
            return 0.0
        return math.exp(sum(math.log(v) for v in vals) / len(vals))
    
    def geo_mean_e2e_speedup(self, n: int = 10) -> float:
        """Geometric mean of end-to-end speedup at N iterations."""
        vals = [e.e2e_speedup_at(n) for e in self.entries if e.e2e_speedup_at(n) > 0]
        if not vals:
            return 0.0
        return math.exp(sum(math.log(v) for v in vals) / len(vals))


# =============================================================================
# Core Computation
# =============================================================================

def compute_amortization(
    benchmark_results: list,
    reorder_results: list,
    reference_algo: str = "ORIGINAL"
) -> AmortizationReport:
    """
    Compute amortization metrics from benchmark and reorder results.
    
    Args:
        benchmark_results: List of BenchmarkResult (or dicts with same keys)
        reorder_results: List of ReorderResult (or dicts with same keys)
        reference_algo: Algorithm to use as baseline (default: ORIGINAL)
    
    Returns:
        AmortizationReport with per-(graph, algo, benchmark) entries
    """
    report = AmortizationReport()
    
    # Normalise inputs to dicts
    def to_dict(x):
        if hasattr(x, 'to_dict'):
            return x.to_dict()
        if hasattr(x, '__dict__'):
            return vars(x)
        return x
    
    bench_dicts = [to_dict(r) for r in benchmark_results]
    reorder_dicts = [to_dict(r) for r in reorder_results]
    
    # Build lookup: (graph, benchmark) → {algo: time_seconds}
    kernel_times: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in bench_dicts:
        if not r.get('success', True) or r.get('time_seconds', 0) <= 0:
            continue
        key = (r['graph'], r['benchmark'])
        if key not in kernel_times:
            kernel_times[key] = {}
        kernel_times[key][r['algorithm']] = r['time_seconds']
    
    # Build lookup: (graph, algo) → reorder_time
    reorder_times: Dict[Tuple[str, str], float] = {}
    for r in reorder_dicts:
        key = (r.get('graph', ''), r.get('algorithm_name', ''))
        reorder_times[key] = r.get('reorder_time', 0.0)
    
    # Also try getting reorder_time from benchmark results (fallback)
    for r in bench_dicts:
        key = (r['graph'], r['algorithm'])
        if key not in reorder_times and r.get('reorder_time', 0) > 0:
            reorder_times[key] = r['reorder_time']
    
    # Compute amortization for each (graph, algo, benchmark) where baseline exists
    for (graph, bench), algos in kernel_times.items():
        orig_time = algos.get(reference_algo, 0)
        if orig_time <= 0:
            continue
        
        for algo, kernel_time in algos.items():
            if algo == reference_algo:
                continue
            
            reorder_time = reorder_times.get((graph, algo), 0.0)
            
            entry = AmortizationMetrics(
                graph=graph,
                algorithm=algo,
                benchmark=bench,
                original_kernel_time=orig_time,
                reordered_kernel_time=kernel_time,
                reorder_time=reorder_time,
            )
            report.entries.append(entry)
    
    return report


def compute_variant_comparison(
    benchmark_results: list,
    reorder_results: list,
    variant_a: str,
    variant_b: str,
) -> List[VariantComparison]:
    """
    Head-to-head comparison of two variants across all graphs/benchmarks.
    
    Args:
        variant_a, variant_b: Algorithm names to compare
    
    Returns:
        List of VariantComparison, one per (graph, benchmark)
    """
    def to_dict(x):
        if hasattr(x, '__dict__'):
            return vars(x)
        return x
    
    bench_dicts = [to_dict(r) for r in benchmark_results]
    reorder_dicts = [to_dict(r) for r in reorder_results]
    
    # Build lookups
    kernel: Dict[Tuple[str, str, str], float] = {}
    for r in bench_dicts:
        if r.get('time_seconds', 0) > 0:
            kernel[(r['graph'], r['benchmark'], r['algorithm'])] = r['time_seconds']
    
    reorder: Dict[Tuple[str, str], float] = {}
    for r in reorder_dicts:
        reorder[(r.get('graph', ''), r.get('algorithm_name', ''))] = r.get('reorder_time', 0.0)
    for r in bench_dicts:
        k = (r['graph'], r['algorithm'])
        if k not in reorder and r.get('reorder_time', 0) > 0:
            reorder[k] = r['reorder_time']
    
    comparisons = []
    seen = set()
    
    for (graph, bench, algo) in kernel:
        if algo != variant_a:
            continue
        if (graph, bench) in seen:
            continue
        
        ka = kernel.get((graph, bench, variant_a), 0)
        kb = kernel.get((graph, bench, variant_b), 0)
        if ka <= 0 or kb <= 0:
            continue
        
        seen.add((graph, bench))
        ra = reorder.get((graph, variant_a), 0)
        rb = reorder.get((graph, variant_b), 0)
        
        comp = VariantComparison(
            graph=graph, benchmark=bench,
            variant_a=variant_a, variant_b=variant_b,
            kernel_a=ka, kernel_b=kb,
            kernel_winner=variant_a if ka < kb else variant_b,
            kernel_speedup=kb / ka if ka > 0 else 0,
            reorder_a=ra, reorder_b=rb,
            reorder_ratio=rb / ra if ra > 0 else float('inf'),
        )
        
        # End-to-end winners at various iteration counts
        for n in [1, 10, 100]:
            e2e_a = ra + n * ka
            e2e_b = rb + n * kb
            winner = variant_a if e2e_a < e2e_b else variant_b
            if n == 1: comp.e2e_winner_at_1 = winner
            elif n == 10: comp.e2e_winner_at_10 = winner
            elif n == 100: comp.e2e_winner_at_100 = winner
        
        # Crossover: when does the slower-to-reorder variant win?
        # Solve: ra + n*ka = rb + n*kb  →  n = (rb - ra) / (ka - kb)
        if ka != kb:
            n_cross = (rb - ra) / (ka - kb)
            comp.crossover_iters = n_cross if n_cross > 0 else float('inf')
        
        comparisons.append(comp)
    
    return comparisons


# =============================================================================
# Reporting
# =============================================================================

def format_amortization_table(report: AmortizationReport, max_rows: int = 50) -> str:
    """Format amortization report as a readable table."""
    lines = []
    lines.append("=" * 110)
    lines.append(f"{'Graph':<20} {'Algorithm':<25} {'Bench':<6} {'Kernel':>8} "
                 f"{'Reorder':>8} {'Amort':>8} {'E2E@1':>7} {'E2E@10':>7} {'E2E@100':>7} {'Verdict'}")
    lines.append("-" * 110)
    
    # Sort by amortization iterations (best first)
    sorted_entries = sorted(report.entries, 
                           key=lambda e: (e.benchmark, e.amortization_iters))
    
    for entry in sorted_entries[:max_rows]:
        amort_str = (f"{entry.amortization_iters:>7.1f}" 
                    if entry.amortization_iters < float('inf') else "   NEVER")
        lines.append(
            f"{entry.graph:<20} {entry.algorithm:<25} {entry.benchmark:<6} "
            f"{entry.kernel_speedup:>7.2f}x "
            f"{entry.reorder_time:>7.4f}s "
            f"{amort_str} "
            f"{entry.e2e_speedup_at_1:>6.2f}x "
            f"{entry.e2e_speedup_at_10:>6.2f}x "
            f"{entry.e2e_speedup_at_100:>6.2f}x "
            f"{entry.break_even_summary()}"
        )
    
    lines.append("=" * 110)
    lines.append(f"Total entries: {len(report.entries)}")
    lines.append(f"Profitable: {report.profitable_count()} / {len(report.entries)}")
    lines.append(f"Median amortization: {report.median_amortization():.1f} iterations")
    lines.append(f"Geo-mean kernel speedup: {report.geo_mean_kernel_speedup():.3f}x")
    lines.append(f"Geo-mean E2E@10: {report.geo_mean_e2e_speedup(10):.3f}x")
    lines.append(f"Geo-mean E2E@100: {report.geo_mean_e2e_speedup(100):.3f}x")
    
    return "\n".join(lines)


def format_comparison_table(comparisons: List[VariantComparison]) -> str:
    """Format head-to-head comparison as a table."""
    if not comparisons:
        return "No comparisons to display."
    
    va = comparisons[0].variant_a
    vb = comparisons[0].variant_b
    
    lines = []
    lines.append(f"Head-to-head: {va} vs {vb}")
    lines.append("=" * 100)
    lines.append(f"{'Graph':<20} {'Bench':<6} {'KernelA':>8} {'KernelB':>8} "
                 f"{'Winner':>12} {'E2E@1':>8} {'E2E@10':>8} {'E2E@100':>8} {'Crossover':>10}")
    lines.append("-" * 100)
    
    a_wins = {1: 0, 10: 0, 100: 0}
    b_wins = {1: 0, 10: 0, 100: 0}
    
    for c in sorted(comparisons, key=lambda x: (x.benchmark, x.graph)):
        cross_str = (f"{c.crossover_iters:.1f}" 
                    if c.crossover_iters < float('inf') else "NEVER")
        lines.append(
            f"{c.graph:<20} {c.benchmark:<6} "
            f"{c.kernel_a:>7.5f}s {c.kernel_b:>7.5f}s "
            f"{c.kernel_winner:>12} "
            f"{c.e2e_winner_at_1:>8} "
            f"{c.e2e_winner_at_10:>8} "
            f"{c.e2e_winner_at_100:>8} "
            f"{cross_str:>10}"
        )
        for n, winner_attr in [(1, 'e2e_winner_at_1'), (10, 'e2e_winner_at_10'), (100, 'e2e_winner_at_100')]:
            w = getattr(c, winner_attr)
            if w == va: a_wins[n] += 1
            else: b_wins[n] += 1
    
    lines.append("=" * 100)
    total = len(comparisons)
    for n in [1, 10, 100]:
        lines.append(f"  E2E@{n:>3}: {va} wins {a_wins[n]}/{total}, {vb} wins {b_wins[n]}/{total}")
    
    return "\n".join(lines)


def print_amortization_report(report: AmortizationReport) -> None:
    """Print full amortization report to stdout."""
    print(format_amortization_table(report))


def print_comparison_report(comparisons: List[VariantComparison]) -> None:
    """Print head-to-head comparison to stdout."""
    print(format_comparison_table(comparisons))
