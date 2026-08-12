"""Shared ordered benchmark subsets and cache-capacity policies."""

from __future__ import annotations

from collections.abc import Iterable

MIB = 1024 * 1024

# Complete canonical benchmark inventory. Order is the generic harness order.
ALL_BENCHMARKS = (
    "pr",
    "pr_spmv",
    "bfs",
    "cc",
    "cc_sv",
    "sssp",
    "bc",
    "tc",
)

# Generic reordering studies exclude combinatorial triangle counting.
REORDER_BENCHMARKS = tuple(
    benchmark for benchmark in ALL_BENCHMARKS
    if benchmark != "tc"
)

# Frozen publication ordering. This order is part of figure/table provenance.
PAPER_BENCHMARK_ORDER = (
    "bfs",
    "pr",
    "pr_spmv",
    "sssp",
    "cc",
    "cc_sv",
    "bc",
)

PREVIEW_BENCHMARK_ORDER = ("pr", "bfs")
ADAPTIVE_CACHE_BENCHMARK_ORDER = ("pr", "bfs", "cc", "sssp")
MODEL_ABLATION_BENCHMARK_ORDER = (
    "bc",
    "bfs",
    "cc",
    "cc_sv",
    "pr",
    "pr_spmv",
    "sssp",
    "tc",
)

# Candidate capacity ladder in MiB. Convert to bytes only at runtime/API edges.
CACHE_CAPACITY_CANDIDATES_MIB = (
    2,
    8,
    22,
    32,
    64,
    128,
    256,
    512,
)
PAPER_CACHE_CAPACITIES_MIB = CACHE_CAPACITY_CANDIDATES_MIB[:5]
PAPER_CACHE_PREVIEW_CAPACITIES_MIB = (2, 8, 64)
CACHE_PR_ITERATIONS = 5
SHUFFLED_LABEL_SEED = 0

LEGACY_NON_NESTED_LOGO_ERROR = (
    "Legacy non-nested LOGO is retired for adaptive claims; use the "
    "nested leave-one-topology-out evaluator with fold-local portfolio "
    "selection, model fitting, and OOD calibration"
)


def reject_legacy_non_nested_logo(context: str = "") -> None:
    prefix = f"{context}: " if context else ""
    raise RuntimeError(prefix + LEGACY_NON_NESTED_LOGO_ERROR)


def retired_legacy_logo(function):
    """Decorator for public evaluators whose non-nested protocol is retired."""
    def rejected(*args, **kwargs):
        reject_legacy_non_nested_logo(function.__name__)

    rejected.__name__ = function.__name__
    rejected.__qualname__ = function.__qualname__
    rejected.__doc__ = function.__doc__
    rejected.__module__ = function.__module__
    return rejected


def mib_to_bytes(capacity_mib: int) -> int:
    if type(capacity_mib) is not int or capacity_mib <= 0:
        raise ValueError("Cache capacity must be a positive integer MiB value")
    return capacity_mib * MIB


def cache_capacities_bytes(
    capacities_mib: Iterable[int],
) -> tuple[int, ...]:
    return tuple(mib_to_bytes(capacity) for capacity in capacities_mib)


def validate_benchmark_subset(
    benchmarks: Iterable[str],
) -> tuple[str, ...]:
    values = tuple(benchmarks)
    if not values or len(values) != len(set(values)):
        raise ValueError("Benchmark subsets must be non-empty and unique")
    unknown = set(values) - set(ALL_BENCHMARKS)
    if unknown:
        raise ValueError(
            "Unknown benchmarks: " + ", ".join(sorted(unknown)))
    return values


for _subset in (
    REORDER_BENCHMARKS,
    PAPER_BENCHMARK_ORDER,
    PREVIEW_BENCHMARK_ORDER,
    ADAPTIVE_CACHE_BENCHMARK_ORDER,
    MODEL_ABLATION_BENCHMARK_ORDER,
):
    validate_benchmark_subset(_subset)
