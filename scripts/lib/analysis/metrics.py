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
from typing import Dict, List, Tuple
import math
from ..core.utils import Logger

log = Logger()


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
    min_efficient_n: int = 0              # Smallest N where overhead < 5% of total cost
    
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
            
            # Minimum efficient N: smallest N where reorder overhead is < 5% of total
            # Total cost = reorder_time + N * reordered_kernel_time
            # Overhead fraction = reorder_time / total_cost < 0.05
            # => reorder_time < 0.05 * (reorder_time + N * reordered_kernel_time)
            # => 0.95 * reorder_time < 0.05 * N * reordered_kernel_time
            # => N > 19 * reorder_time / reordered_kernel_time
            if self.reordered_kernel_time > 0 and self.reorder_time > 0:
                self.min_efficient_n = math.ceil(19.0 * self.reorder_time / self.reordered_kernel_time)
            else:
                self.min_efficient_n = 0

    def e2e_speedup_at(self, n: int) -> float:
        """End-to-end speedup at N iterations (amortized: includes reorder cost)."""
        e2e_orig = n * self.original_kernel_time
        e2e_reord = self.reorder_time + n * self.reordered_kernel_time
        return e2e_orig / e2e_reord if e2e_reord > 0 else 0.0
    
    def amortized_cost_per_iter(self, n: int) -> float:
        """Per-iteration cost including amortized reorder overhead.
        
        Directly comparable to original_kernel_time.
        amortized = (reorder_time + N * reordered_kernel_time) / N
                   = reordered_kernel_time + reorder_time / N
        """
        if n <= 0:
            return float('inf')
        return self.reordered_kernel_time + self.reorder_time / n
    
    def net_time_saved(self, n: int) -> float:
        """Net time saved after N iterations (negative = net loss).
        
        net = N * original_time - (reorder_time + N * reordered_time)
            = N * time_saved_per_iter - reorder_time
        """
        return n * self.time_saved_per_iter - self.reorder_time
    
    def time_to_recoup_human(self) -> str:
        """Human-readable time to recoup reorder overhead.
        
        Assumes 1 iteration/second (use actual workload rate for real estimates).
        Shows the wall-clock time of the reorder overhead itself.
        """
        if self.reorder_time <= 0:
            return "0s"
        t = self.reorder_time
        if t < 1:
            return f"{t*1000:.0f}ms"
        if t < 60:
            return f"{t:.1f}s"
        if t < 3600:
            return f"{t/60:.1f}min"
        return f"{t/3600:.1f}h"
    
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
class AmortizationReport:
    """Full amortization report across all graphs/benchmarks."""
    entries: List[AmortizationMetrics] = field(default_factory=list)
    
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


# =============================================================================
# Reporting
# =============================================================================

def format_amortization_table(report: AmortizationReport, max_rows: int = 50) -> str:
    """Format amortization report as a readable table.
    
    Columns:
        Graph/Algorithm/Bench: identification
        Kernel: kernel-only speedup (no overhead)
        Reorder: reorder overhead (wall-clock)
        Break-even: iterations to amortize overhead (N* = overhead / time_saved_per_iter)
        E2E@10/100: end-to-end speedup INCLUDING reorder overhead, amortized over N iters
        MinN@95%: minimum N where reorder overhead < 5% of total cost
        Verdict: qualitative assessment
    """
    lines = []
    w = 125
    lines.append("=" * w)
    lines.append("AMORTIZATION ANALYSIS")
    lines.append(f"  E2E speedup = N × baseline_time / (reorder_overhead + N × reordered_time)")
    lines.append(f"  Break-even N* = reorder_overhead / time_saved_per_iteration")
    lines.append("-" * w)
    lines.append(f"{'Graph':<20} {'Algorithm':<22} {'Bench':<7} {'Kernel':>7} "
                 f"{'Reorder':>9} {'N*':>8} {'E2E@10':>7} {'E2E@100':>8} "
                 f"{'MinN@95%':>9} {'Verdict'}")
    lines.append("-" * w)
    
    # Sort by amortization iterations (best first)
    sorted_entries = sorted(report.entries, 
                           key=lambda e: (e.benchmark, e.amortization_iters))
    
    for entry in sorted_entries[:max_rows]:
        amort_str = (f"{entry.amortization_iters:>7.1f}" 
                    if entry.amortization_iters < float('inf') else "   NEVER")
        min_n_str = (f"{entry.min_efficient_n:>8d}" 
                    if entry.min_efficient_n > 0 else "       0")
        lines.append(
            f"{entry.graph:<20} {entry.algorithm:<22} {entry.benchmark:<7} "
            f"{entry.kernel_speedup:>6.2f}x "
            f"{entry.time_to_recoup_human():>9s} "
            f"{amort_str} "
            f"{entry.e2e_speedup_at_10:>6.2f}x "
            f"{entry.e2e_speedup_at_100:>7.2f}x "
            f"{min_n_str} "
            f"{entry.break_even_summary()}"
        )
    
    lines.append("=" * w)
    lines.append(f"Total entries: {len(report.entries)}")
    lines.append(f"Profitable: {report.profitable_count()} / {len(report.entries)}")
    lines.append(f"Median break-even: {report.median_amortization():.1f} iterations")
    lines.append(f"Geo-mean kernel speedup: {report.geo_mean_kernel_speedup():.3f}x")
    lines.append(f"Geo-mean E2E@10: {report.geo_mean_e2e_speedup(10):.3f}x")
    lines.append(f"Geo-mean E2E@100: {report.geo_mean_e2e_speedup(100):.3f}x")
    
    return "\n".join(lines)
