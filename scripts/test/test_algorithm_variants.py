#!/usr/bin/env python3
"""
Algorithm Variant Integration Tests
====================================

Comprehensive parametrized tests covering ALL algorithm variant combinations.
Each test runs the binary with a given -o option and verifies it exits cleanly
(exit code 0, no crash, no assertion failure).

Tiers:
  1. Basic algorithms (0-11, 15) — no variants
  2. RabbitOrder (8) variants — csr, boost
  3. GraphBrewOrder (12) presets — leiden, rabbit, hubcluster
  4. GraphBrewOrder (12) preset + positional overrides
  5. GraphBrewOrder (12) token mode — ordering strategies
  6. GraphBrewOrder (12) token combinations — multi-token
  7. GraphBrewOrder (12) resolution & feature flags
  8. LeidenOrder (15) resolution variants
  9. Edge cases — legacy aliases, old format, boundary values

Usage:
    pytest scripts/test/test_algorithm_variants.py -v
    pytest scripts/test/test_algorithm_variants.py -k "tier1" -v
    pytest scripts/test/test_algorithm_variants.py -k "graphbrew" -v
    pytest scripts/test/test_algorithm_variants.py --timeout=120 -v

Requires:
    - bench/bin/pr binary (run `make pr` first)
    - A test graph (uses bundled tiny.el, or soc-Epinions1.sg if available)
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
PR_BINARY = BIN_DIR / "pr"

# Test graphs: prefer a real graph for community algos, fall back to tiny
TINY_GRAPH = PROJECT_ROOT / "scripts" / "test" / "graphs" / "tiny" / "tiny.el"
REAL_GRAPH = PROJECT_ROOT / "results" / "graphs" / "soc-Epinions1" / "soc-Epinions1.sg"

# Timeout for each binary invocation (seconds)
BINARY_TIMEOUT = 120

# Algorithms that need a real graph (community detection is degenerate on 4 nodes)
NEEDS_REAL_GRAPH = {8, 9, 10, 11, 12, 14, 15}

# Algorithms we skip entirely in CI (need special setup or are too slow)
SKIP_ALGOS = {
    13,  # MAP — needs a mapping file
    14,  # AdaptiveOrder — needs weights & may crash on tiny graphs
}


def get_graph_path(algo_id: int) -> str:
    """Return best available graph path for the given algorithm."""
    if algo_id in NEEDS_REAL_GRAPH and REAL_GRAPH.exists():
        return str(REAL_GRAPH)
    return str(TINY_GRAPH)


def run_pr(option: str, graph_path: str = None, timeout: int = BINARY_TIMEOUT) -> subprocess.CompletedProcess:
    """Run the pr binary with -o option and return the result."""
    if graph_path is None:
        # Extract algo ID from option to choose graph
        algo_id = int(option.split(":")[0])
        graph_path = get_graph_path(algo_id)

    cmd = [str(PR_BINARY), "-f", graph_path, "-s", "-o", option, "-n", "1"]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


@pytest.fixture(scope="session", autouse=True)
def check_prerequisites():
    """Verify pr binary exists before running any tests."""
    if not PR_BINARY.exists():
        pytest.skip(f"pr binary not found at {PR_BINARY}. Run 'make pr' first.")
    if not TINY_GRAPH.exists():
        pytest.skip(f"Tiny test graph not found at {TINY_GRAPH}.")


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Basic algorithms (0-11, 15) — no variants
# ═══════════════════════════════════════════════════════════════════════════

TIER1_BASIC = [
    ("0",  "ORIGINAL"),
    ("1",  "RANDOM"),
    ("2",  "SORT"),
    ("3",  "HUBSORT"),
    ("4",  "HUBCLUSTER"),
    ("5",  "DBG"),
    ("6",  "HUBSORTDBG"),
    ("7",  "HUBCLUSTERDBG"),
    ("8",  "RABBITORDER_default"),
    ("9",  "GORDER"),
    ("10", "CORDER"),
    ("11", "RCMORDER"),
    ("12", "GraphBrewOrder_default"),
    ("15", "LeidenOrder_default"),
]


@pytest.mark.parametrize("option,name", TIER1_BASIC, ids=[t[1] for t in TIER1_BASIC])
def test_tier1_basic(option, name):
    """Tier 1: Each basic algorithm runs and exits cleanly."""
    algo_id = int(option.split(":")[0])
    if algo_id in SKIP_ALGOS:
        pytest.skip(f"Algorithm {algo_id} skipped (needs special setup)")
    result = run_pr(option)
    assert result.returncode == 0, (
        f"Algorithm {name} (-o {option}) failed with exit code {result.returncode}.\n"
        f"stderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: RabbitOrder (8) variants
# ═══════════════════════════════════════════════════════════════════════════

TIER2_RABBIT = [
    ("8:csr",   "RABBITORDER_csr"),
    ("8:boost", "RABBITORDER_boost"),
]


@pytest.mark.parametrize("option,name", TIER2_RABBIT, ids=[t[1] for t in TIER2_RABBIT])
def test_tier2_rabbitorder_variants(option, name):
    """Tier 2: RabbitOrder csr/boost variants."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"RabbitOrder variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )



# ═══════════════════════════════════════════════════════════════════════════
# TIER 2b: RCM (11) variants
# ═══════════════════════════════════════════════════════════════════════════

TIER2B_RCM = [
    ("11",     "RCM_default"),
    ("11:bnf", "RCM_bnf"),
]


@pytest.mark.parametrize("option,name", TIER2B_RCM, ids=[t[1] for t in TIER2B_RCM])
def test_tier2b_rcm_variants(option, name):
    """Tier 2b: RCM default/bnf variants."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"RCM variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2c: GOrder (9) variants
# ═══════════════════════════════════════════════════════════════════════════

TIER2C_GORDER = [
    ("9",     "GORDER_default"),
    ("9:csr", "GORDER_csr"),
]


@pytest.mark.parametrize("option,name", TIER2C_GORDER, ids=[t[1] for t in TIER2C_GORDER])
def test_tier2c_gorder_variants(option, name):
    """Tier 2c: GOrder default/csr variants."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GOrder variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: GraphBrewOrder (12) presets
# ═══════════════════════════════════════════════════════════════════════════

TIER3_PRESETS = [
    ("12:leiden",      "GraphBrew_leiden"),
    ("12:rabbit",      "GraphBrew_rabbit"),
    ("12:hubcluster",  "GraphBrew_hubcluster"),
]


@pytest.mark.parametrize("option,name", TIER3_PRESETS, ids=[t[1] for t in TIER3_PRESETS])
def test_tier3_graphbrew_presets(option, name):
    """Tier 3: GraphBrewOrder named presets."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew preset {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 4: GraphBrewOrder (12) preset + positional overrides
# ═══════════════════════════════════════════════════════════════════════════

TIER4_POSITIONAL = [
    ("12:leiden:0",        "leiden_final_ORIGINAL"),
    ("12:leiden:6",        "leiden_final_HUBSORTDBG"),
    ("12:leiden:7",        "leiden_final_HUBCLUSTERDBG"),
    ("12:leiden:8",        "leiden_final_RABBITORDER"),
    ("12:leiden:8:0.75",   "leiden_res_0.75"),
    ("12:leiden:8:1.5",    "leiden_res_1.5"),
    ("12:leiden:8:auto",   "leiden_res_auto"),
    ("12:leiden:8:dynamic", "leiden_res_dynamic"),
    ("12:rabbit:7",        "rabbit_final_HUBCLUSTERDBG"),
    ("12:rabbit:8:0.5",    "rabbit_res_0.5"),
    ("12:hubcluster:6",    "hubcluster_final_HUBSORTDBG"),
]


@pytest.mark.parametrize("option,name", TIER4_POSITIONAL, ids=[t[1] for t in TIER4_POSITIONAL])
def test_tier4_preset_positional(option, name):
    """Tier 4: Preset + positional override (final_algo, resolution)."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew positional {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 5: GraphBrewOrder (12) token mode — ordering strategies
# ═══════════════════════════════════════════════════════════════════════════

TIER5_TOKENS = [
    ("12:hrab",         "token_hrab"),
    ("12:dfs",          "token_dfs"),
    ("12:bfs",          "token_bfs"),
    ("12:conn",         "token_conn"),
    ("12:dbg",          "token_dbg"),
    ("12:corder",       "token_corder"),
    ("12:dbg-global",   "token_dbg_global"),
    ("12:corder-global","token_corder_global"),
    ("12:community",    "token_community"),
    ("12:hierarchical", "token_hierarchical"),
    ("12:hcache",       "token_hcache"),
    ("12:tqr",          "token_tqr"),
]


@pytest.mark.parametrize("option,name", TIER5_TOKENS, ids=[t[1] for t in TIER5_TOKENS])
def test_tier5_token_ordering(option, name):
    """Tier 5: Token-mode ordering strategies."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew token {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 6: GraphBrewOrder (12) token combinations
# ═══════════════════════════════════════════════════════════════════════════

TIER6_COMBOS = [
    ("12:hrab:gvecsr",            "hrab_gvecsr"),
    ("12:hrab:gvecsr:totalm",     "hrab_gvecsr_totalm"),
    ("12:dfs:streaming",          "dfs_streaming"),
    ("12:dbg:streaming",          "dbg_streaming"),
    ("12:hrab:0.75",              "hrab_res_0.75"),
    ("12:graphbrew",              "graphbrew_mode"),
    ("12:graphbrew:final8",       "graphbrew_final8"),
    ("12:graphbrew:final6",       "graphbrew_final6"),
    ("12:graphbrew:depth2",       "graphbrew_depth2"),
    ("12:graphbrew:subauto",      "graphbrew_subauto"),
]


@pytest.mark.parametrize("option,name", TIER6_COMBOS, ids=[t[1] for t in TIER6_COMBOS])
def test_tier6_token_combos(option, name):
    """Tier 6: Multi-token combinations."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew combo {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 7: GraphBrewOrder (12) resolution & feature flags
# ═══════════════════════════════════════════════════════════════════════════

TIER7_FLAGS = [
    ("12:hrab:auto",              "auto_resolution"),
    ("12:hrab:dynamic",           "dynamic_resolution"),
    ("12:hrab:0.5",               "low_resolution"),
    ("12:hrab:2.0",               "high_resolution"),
    ("12:hrab:merge",             "community_merging"),
    ("12:hrab:verify",            "topology_verify"),
    ("12:hrab:norefine",          "no_refinement"),
    ("12:hrab:refine0",           "refine_pass0_only"),
    ("12:hrab:lazyupdate",        "lazy_updates"),
    ("12:hrab:gord",              "gorder_intra"),
    ("12:hrab:hsort",             "hub_sort_post"),
]


@pytest.mark.parametrize("option,name", TIER7_FLAGS, ids=[t[1] for t in TIER7_FLAGS])
def test_tier7_feature_flags(option, name):
    """Tier 7: Feature flags and resolution modes."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"GraphBrew flag {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 8: LeidenOrder (15) resolution variants
# ═══════════════════════════════════════════════════════════════════════════

TIER8_LEIDEN = [
    ("15:0.5",  "LeidenOrder_res_0.5"),
    ("15:0.75", "LeidenOrder_res_0.75"),
    ("15:1.0",  "LeidenOrder_res_1.0"),
    ("15:1.5",  "LeidenOrder_res_1.5"),
]


@pytest.mark.parametrize("option,name", TIER8_LEIDEN, ids=[t[1] for t in TIER8_LEIDEN])
def test_tier8_leiden_resolution(option, name):
    """Tier 8: LeidenOrder with different resolution values."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"LeidenOrder variant {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TIER 9: Edge cases — legacy aliases, old format, boundary values
# ═══════════════════════════════════════════════════════════════════════════

TIER9_EDGE = [
    # Token parser quality preset
    ("12:quality",                "token_quality_preset"),
    # graphbrew prefix (backward compat)
    ("12:graphbrew:hrab",         "graphbrew_prefix_hrab"),
    # Multiple feature flags combined
    ("12:hrab:gvecsr:totalm:refine0:0.75", "full_combo"),
]


@pytest.mark.parametrize("option,name", TIER9_EDGE, ids=[t[1] for t in TIER9_EDGE])
def test_tier9_edge_cases(option, name):
    """Tier 9: Edge cases — legacy aliases, old format, combined flags."""
    result = run_pr(option)
    assert result.returncode == 0, (
        f"Edge case {name} (-o {option}) failed.\nstderr: {result.stderr[-500:]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test: verify all tiers have expected count
# ═══════════════════════════════════════════════════════════════════════════

def test_variant_count():
    """Verify we have ~70+ test cases across all tiers."""
    total = (
        len(TIER1_BASIC) + len(TIER2_RABBIT) + len(TIER3_PRESETS) +
        len(TIER4_POSITIONAL) + len(TIER5_TOKENS) + len(TIER6_COMBOS) +
        len(TIER7_FLAGS) + len(TIER8_LEIDEN) + len(TIER9_EDGE)
    )
    assert total >= 70, f"Expected ≥70 test cases, got {total}"
