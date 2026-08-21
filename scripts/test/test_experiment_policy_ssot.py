"""Regression tests for shared benchmark and cache policy SSOTs."""

from __future__ import annotations

import pytest

from scripts.experiments.vldb import config as vldb_config
from scripts.lib.core import utils
from scripts.lib.core.experiment_policy import (
    ALL_BENCHMARKS,
    CACHE_PR_ITERATIONS,
    END_TO_END_REUSE_COUNTS,
    MODEL_ABLATION_BENCHMARK_ORDER,
    PAPER_BENCHMARK_ORDER,
    PAPER_CACHE_CAPACITIES_MIB,
    PAPER_CACHE_PREVIEW_CAPACITIES_MIB,
    PREVIEW_BENCHMARK_ORDER,
    REORDER_BENCHMARKS,
    SHUFFLED_LABEL_SEED,
    cache_capacities_bytes,
    mib_to_bytes,
)


def test_named_benchmark_orders_are_exact_and_derived():
    assert ALL_BENCHMARKS == (
        "pr", "pr_spmv", "bfs", "cc",
        "cc_sv", "sssp", "bc", "tc",
    )
    assert REORDER_BENCHMARKS == tuple(
        benchmark for benchmark in ALL_BENCHMARKS
        if benchmark != "tc"
    )
    assert PAPER_BENCHMARK_ORDER == (
        "bfs", "pr", "pr_spmv", "sssp", "cc", "cc_sv", "bc",
    )
    assert PREVIEW_BENCHMARK_ORDER == ("pr", "bfs")
    assert set(MODEL_ABLATION_BENCHMARK_ORDER) == set(ALL_BENCHMARKS)


def test_compatibility_lists_resolve_to_shared_policies():
    assert utils.BENCHMARKS == list(ALL_BENCHMARKS)
    assert utils.EXPERIMENT_BENCHMARKS == list(REORDER_BENCHMARKS)
    assert vldb_config.BENCHMARKS == list(PAPER_BENCHMARK_ORDER)
    assert vldb_config.BENCHMARKS_PREVIEW == list(
        PREVIEW_BENCHMARK_ORDER)


def test_cache_policies_use_mib_ssot_and_byte_boundaries():
    assert PAPER_CACHE_CAPACITIES_MIB == (2, 8, 22, 32, 64)
    assert PAPER_CACHE_PREVIEW_CAPACITIES_MIB == (2, 8, 64)
    assert CACHE_PR_ITERATIONS == 5
    assert END_TO_END_REUSE_COUNTS == (1, 5, 10, 20, 50, 100)
    assert vldb_config.E2E_REUSE_COUNTS == list(
        END_TO_END_REUSE_COUNTS)
    assert vldb_config.CACHE_PR_ITERATIONS == CACHE_PR_ITERATIONS
    assert vldb_config.RANDOM_BASELINE_SEED == SHUFFLED_LABEL_SEED == 0
    assert vldb_config.CACHE_SIZES == list(
        cache_capacities_bytes(PAPER_CACHE_CAPACITIES_MIB))
    assert vldb_config.CACHE_SIZES_PREVIEW == list(
        cache_capacities_bytes(PAPER_CACHE_PREVIEW_CAPACITIES_MIB))


@pytest.mark.parametrize("value", [0, -1, 1.5, "22", None, True])
def test_cache_mib_conversion_fails_closed(value):
    with pytest.raises(ValueError, match="positive integer"):
        mib_to_bytes(value)
