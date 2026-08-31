#!/usr/bin/env python3
"""
Shared configuration for the frozen GraphBrew evaluation.

Defines all algorithm IDs, GraphBrew variants, chained orderings,
graph datasets, benchmarks, and evaluation parameters used across
the experiment runner and figure generator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.lib.core.experiment_policy import (
    CACHE_PR_ITERATIONS,
    END_TO_END_REUSE_COUNTS,
    PAPER_BENCHMARK_ORDER,
    PAPER_CACHE_CAPACITIES_MIB,
    PAPER_CACHE_PREVIEW_CAPACITIES_MIB,
    PREVIEW_BENCHMARK_ORDER,
    REORDER_SEMANTICS_VERSION,
    SHUFFLED_LABEL_SEED,
    cache_capacities_bytes,
)

RANDOM_BASELINE_SEED = SHUFFLED_LABEL_SEED

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bench" / "bin"
BIN_WORK_DIR = PROJECT_ROOT / "bench" / "bin_work"
BIN_SIM_DIR = PROJECT_ROOT / "bench" / "bin_sim"
PAPER_GRAPH_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_PAPER_GRAPH_ROOT",
        "/media/NVMeData/00_GraphDatasets/GraphBrew",
    )
).resolve()
PAPER_ARTIFACT_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_VLDB_ROOT",
        PAPER_GRAPH_ROOT / "artifacts",
    )
).resolve()
VLDB_ROOT = Path(
    os.environ.get("GRAPHBREW_VLDB_ROOT", PROJECT_ROOT / "results")
).resolve()
RESULTS_DIR = VLDB_ROOT / "vldb_paper"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---------------------------------------------------------------------------
# Algorithm Definitions
# ---------------------------------------------------------------------------

# Baseline reorder algorithm IDs (no GraphBrew variants)
BASELINE_ALGORITHMS = {
    0: "SHUFFLED",
    1: "RANDOM",
    2: "SORT",
    3: "HUBSORT",
    4: "HUBCLUSTER",
    5: "DBG",
    6: "HUBSORTDBG",
    7: "HUBCLUSTERDBG",
    8: "RABBITORDER",
    9: "GORDER",
    11: "RCM",
    16: "GoGraph-Core",
}

# GraphBrew variants (all use algorithm ID 12)
GRAPHBREW_VARIANTS = [
    "leiden",            # GVE-Leiden + per-community RabbitOrder
    "rabbit",            # GraphBrew Rabbit aggregation + dendrogram DFS
    "hubcluster",        # Leiden + global hub/non-hub split
    "hrab",              # Leiden + RabbitOrder super-graph + RCM intra
    "hrab:bfs_intra",    # Controlled HRAB candidate with BFS-only intra-community order
    "tqr",               # Tile-Quantized RabbitOrder
    "hcache",            # Hierarchical cache-aware
    "streaming",         # Leiden + lazy aggregation
    "rabbit:dbg",        # Rabbit detection + DBG degree-grouping
    "rabbit:hubcluster", # Rabbit detection + global hub/non-hub split
]

# ---------------------------------------------------------------------------
# COMPOSE configurations — v5 paper headline configs (§15, §18, §19, §49).
# Each entry is (label, order_spec). The order_spec is passed verbatim to
# `-o <spec>`. These cover the SuperGraph × Community × Intra × Refinement
# design space and include the new HubSort / DegreeAsc / RCM++ primitives.
# ---------------------------------------------------------------------------
COMPOSE_VARIANTS = [
    # label,        order_spec
    ("FastLeiden-Gorder8",
     "12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:"
     "cd_parallel:sgmb4096:gordf5000:norefine:2:2"),
    ("Leiden-Gorder8",
     "12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8"),
    ("Leiden-HubSort",
     "12:leiden:compose:sg_none:comm_size_desc:intra_hubsort"),
    ("Leiden-DegreeAsc",
     "12:leiden:compose:sg_none:comm_size_desc:intra_deg_asc"),
    ("Leiden-CommDegree-HubSort",
     "12:leiden:compose:sg_none:comm_degree_desc:intra_hubsort"),
    ("Leiden-CommDegree-DegreeAsc",
     "12:leiden:compose:sg_none:comm_degree_desc:intra_deg_asc"),
    ("Rabbit-HubSort",
     "12:rabbit:compose:sg_none:comm_identity:intra_hubsort"),
    ("SuperRabbit-HubSort",
     "12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort"),
    ("Rabbit-Dendrogram",
     "12:rabbit:compose:sg_none:comm_identity:intra_dendrogram"),
    ("Rabbit-SizeDesc-Dendrogram",
     "12:rabbit:compose:sg_none:comm_size_desc:intra_dendrogram"),
    ("Rabbit-DegreeDesc-Dendrogram",
     "12:rabbit:compose:sg_none:comm_degree_desc:intra_dendrogram"),
    ("Rabbit-SuperRCM-Dendrogram",
     "12:rabbit:compose:sg_super_rcm:comm_identity:intra_dendrogram"),
    ("Rabbit-Hilbert-Dendrogram",
     "12:rabbit:compose:sg_hilbert:comm_identity:intra_dendrogram"),
    ("Rabbit-BoundaryLast",
     "12:rabbit:compose:sg_none:comm_identity:intra_boundary_last"),
    ("Rabbit-CoreOrder",
     "12:rabbit:compose:sg_none:comm_identity:intra_core"),
    ("Leiden-RCMpp",
     "12:leiden:compose:sg_none:comm_size_desc:intra_rcmpp"),
    ("Leiden-CommDegree-RCMpp",
     "12:leiden:compose:sg_none:comm_degree_desc:intra_rcmpp"),
]

# RabbitOrder implementation variants (both use algorithm ID 8)
RABBITORDER_VARIANTS = {
    "8:csr":   "RabbitOrder (CSR)",    # Standalone CSR implementation
    "8:boost": "RabbitOrder (Boost)",  # Original Boost-based implementation
}

# Chained orderings: list of (display_name, cli_flags) tuples
CHAINED_ORDERINGS = [
    ("GB-Leiden+DBG",        ["-o", "12:leiden", "-o", "5"]),
    ("GB-Leiden+HubCluster", ["-o", "12:leiden", "-o", "4"]),
    ("GB-HRAB+DBG",          ["-o", "12:hrab",   "-o", "5"]),
    ("GB-Leiden+GoGraph",    ["-o", "12:leiden", "-o", "16"]),
    ("RabbitOrder+DBG",      ["-o", "8:csr",     "-o", "5"]),
]

# Reviewer-facing names use Partitioner-BlockLayout-VertexLayout. Exact CLI
# strings remain the canonical artifact identity.
GRAPHBREW_DISPLAY_NAMES = {
    "12:leiden": "LeidenGVE-Blocks-Rabbit",
    "12:rabbit": "Rabbit-Dendrogram-DFS",
    "12:hubcluster": "Leiden-HubSplit-CommunityDegree",
    "12:hrab": "Leiden-SuperRabbit-RCM",
    "12:hrab:bfs_intra": "Leiden-SuperRabbit-BFS",
    "12:tqr": "Leiden-TileRabbit-BFS",
    "12:hcache": "Leiden-Hierarchy-BFS",
    "12:streaming": "LeidenLazy-Blocks-Rabbit",
    "12:rabbit:dbg": "Rabbit-Blocks-DBG",
    "12:rabbit:hubcluster": "Rabbit-HubSplit-CommunityDegree",
    "12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8":
        "LeidenGVE-SizeDesc-Gorder8",
    "12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:"
    "cd_parallel:sgmb4096:gordf5000:norefine:2:2":
        "FastLeiden-SizeDesc-Gorder8",
    "12:leiden:compose:sg_none:comm_size_desc:intra_hubsort":
        "LeidenGVE-SizeDesc-HubSort",
    "12:leiden:compose:sg_none:comm_size_desc:intra_deg_asc":
        "LeidenGVE-SizeDesc-DegreeAsc",
    "12:leiden:compose:sg_none:comm_degree_desc:intra_hubsort":
        "LeidenGVE-DegreeDesc-HubSort",
    "12:leiden:compose:sg_none:comm_degree_desc:intra_deg_asc":
        "LeidenGVE-DegreeDesc-DegreeAsc",
    "12:rabbit:compose:sg_none:comm_identity:intra_hubsort":
        "Rabbit-Identity-HubSort",
    "12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort":
        "Rabbit-SuperRabbit-HubSort",
    "12:leiden:compose:sg_none:comm_size_desc:intra_rcmpp":
        "LeidenGVE-SizeDesc-RCMpp",
    "12:leiden:compose:sg_none:comm_degree_desc:intra_rcmpp":
        "LeidenGVE-DegreeDesc-RCMpp",
    "12:rabbit:compose:sg_none:comm_identity:intra_dendrogram":
        "Rabbit-Identity-Dendrogram",
    "12:rabbit:compose:sg_none:comm_size_desc:intra_dendrogram":
        "Rabbit-SizeDesc-Dendrogram",
    "12:rabbit:compose:sg_none:comm_degree_desc:intra_dendrogram":
        "Rabbit-DegreeDesc-Dendrogram",
    "12:rabbit:compose:sg_super_rcm:comm_identity:intra_dendrogram":
        "Rabbit-SuperRCM-Dendrogram",
    "12:rabbit:compose:sg_hilbert:comm_identity:intra_dendrogram":
        "Rabbit-Hilbert-Dendrogram",
    "12:rabbit:compose:sg_none:comm_identity:intra_boundary_last":
        "Rabbit-Identity-BoundaryLast",
    "12:rabbit:compose:sg_none:comm_identity:intra_core":
        "Rabbit-Identity-CoreOrder",
    "12:rabbit:compose:sg_super_rabbit:comm_degree_desc:intra_hubsort":
        "Rabbit-SuperRabbit.DegreeDesc-HubSort",
    "12:leiden:compose:sg_none:comm_identity:intra_hubsort":
        "LeidenGVE-Identity-HubSort",
    "12:rabbit:compose:sg_none:comm_identity:intra_hubsort:refine_2swap":
        "Rabbit-Identity-HubSort+2Swap",
}

# Paper-facing baseline matrix. RabbitOrder implementations replace the
# ambiguous bare ID 8 so every experiment compares native CSR and Boost
# explicitly without also running a duplicate default-CSR cell.
EVALUATION_BASELINES = {
    **{
        str(aid): name
        for aid, name in BASELINE_ALGORITHMS.items()
        if aid not in {8, 9}
    },
    "9:csr": "GORDER",
    **RABBITORDER_VARIANTS,
}

ALL_ALGORITHMS = {
    **EVALUATION_BASELINES,
    **{
        f"12:{variant}": GRAPHBREW_DISPLAY_NAMES[f"12:{variant}"]
        for variant in GRAPHBREW_VARIANTS
    },
    **{
        spec: GRAPHBREW_DISPLAY_NAMES[spec]
        for _label, spec in COMPOSE_VARIANTS
    },
}

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# Compatibility lists derived from named shared ordered policies.
BENCHMARKS = list(PAPER_BENCHMARK_ORDER)
BENCHMARKS_PREVIEW = list(PREVIEW_BENCHMARK_ORDER)

# ---------------------------------------------------------------------------
# Graph Datasets
# ---------------------------------------------------------------------------

# Graphs for full evaluation (match paper Table 2)
EVAL_GRAPHS = [
    {"name": "cit-Patents", "short": "patents", "type": "citation", "nodes": 3_774_768, "undirected_edges": 16_518_947, "vertices_m": 3.77, "edges_m": 16.52},
    {"name": "soc-pokec", "short": "pokec", "type": "social", "nodes": 1_632_803, "undirected_edges": 22_301_964, "vertices_m": 1.63, "edges_m": 22.30},
    {"name": "USA-road-d.USA", "short": "road", "type": "road", "nodes": 23_947_347, "undirected_edges": 28_854_312, "vertices_m": 23.95, "edges_m": 28.85},
    {"name": "soc-LiveJournal1", "short": "journal", "type": "social", "nodes": 4_847_571, "undirected_edges": 42_851_237, "vertices_m": 4.85, "edges_m": 42.85},
    {"name": "delaunay_n24", "short": "delaunay", "type": "mesh", "nodes": 16_777_216, "undirected_edges": 50_331_601, "vertices_m": 16.78, "edges_m": 50.33},
    {"name": "com-Orkut", "short": "orkut", "type": "social", "nodes": 3_072_441, "undirected_edges": 117_185_083, "vertices_m": 3.07, "edges_m": 117.19},
    {"name": "hollywood-2009", "short": "hollywood", "type": "collaboration", "nodes": 1_139_905, "undirected_edges": 56_375_711, "vertices_m": 1.14, "edges_m": 56.38},
    {"name": "wikipedia_link_en", "short": "wikipedia", "type": "content", "nodes": 13_593_033, "undirected_edges": 334_591_525, "vertices_m": 13.59, "edges_m": 334.59},
    {"name": "Gong-gplus", "short": "gplus", "type": "social", "nodes": 28_943_748, "undirected_edges": 335_661_327, "vertices_m": 28.94, "edges_m": 335.66},
    {"name": "webbase-2001", "short": "webbase", "type": "web", "nodes": 118_142_143, "undirected_edges": 854_809_761, "vertices_m": 118.14, "edges_m": 854.81},
    {"name": "twitter7", "short": "twitter", "type": "social", "nodes": 61_578_415, "undirected_edges": 1_202_513_046, "vertices_m": 61.58, "edges_m": 1202.51},
]

# Representative cache-capacity cohort. The full kernel campaign remains on
# all evaluation graphs; cache simulation uses one graph from each distinct
# topology regime exercised by the final paper analysis.
CACHE_GRAPH_NAMES = [
    "cit-Patents",
    "com-Orkut",
    "hollywood-2009",
    "USA-road-d.USA",
]
_EVAL_GRAPH_BY_NAME = {
    graph["name"]: graph
    for graph in EVAL_GRAPHS
}


def _main_graph(name: str) -> dict:
    """Return a copy of frozen main-corpus metadata."""
    return dict(_EVAL_GRAPH_BY_NAME[name])

# Graphs for 64 GB machines (11 graphs, all auto-downloadable from SuiteSparse)
# Drops twitter7/webbase-2001 (>1B edges) and manual-download graphs;
# adds as-Skitter, kron_g500-logn21, indochina-2004, uk-2002 for type diversity.
# Numeric fields for those added graphs are resource-planning hints from their
# source catalogues, not frozen paper-corpus edge-count claims.
EVAL_GRAPHS_64GB = [
    {"name": "as-Skitter",         "short": "skitter",   "type": "infrastructure", "vertices_m": 1.70,   "edges_m": 11.10},
    _main_graph("cit-Patents"),
    _main_graph("soc-pokec"),
    _main_graph("USA-road-d.USA"),
    _main_graph("soc-LiveJournal1"),
    _main_graph("delaunay_n24"),
    _main_graph("hollywood-2009"),
    _main_graph("com-Orkut"),
    {"name": "kron_g500-logn21",   "short": "kron21",    "type": "synthetic",      "vertices_m": 2.10,   "edges_m": 182.08},
    {"name": "indochina-2004",     "short": "indochina", "type": "web",            "vertices_m": 7.41,   "edges_m": 194.11},
    {"name": "uk-2002",            "short": "uk02",      "type": "web",            "vertices_m": 18.52,  "edges_m": 298.11},
]

# Additive adaptive-selector corpus expansion. These graphs are not part of the
# frozen VLDB paper matrix; dimensions are source-catalog planning hints until
# graph_source/v2 conversion records the exact symmetrized graph.
ADAPTIVE_CPU_EXPANSION_GRAPHS = [
    {"name": "email-Enron",         "short": "enron",      "type": "communication", "vertices_m": 0.04, "edges_m": 0.18},
    {"name": "wiki-Talk",           "short": "wikitalk",   "type": "communication", "vertices_m": 2.39, "edges_m": 5.02},
    {"name": "wiki-topcats",        "short": "topcats",    "type": "content",       "vertices_m": 1.79, "edges_m": 28.51},
    {"name": "roadNet-CA",          "short": "roadca",     "type": "road",          "vertices_m": 1.97, "edges_m": 2.77},
    {"name": "roadNet-TX",          "short": "roadtx",     "type": "road",          "vertices_m": 1.39, "edges_m": 1.92},
    {"name": "amazon0601",          "short": "amazon",     "type": "commerce",      "vertices_m": 0.40, "edges_m": 3.39},
    {"name": "web-Google",          "short": "google",     "type": "web",           "vertices_m": 0.92, "edges_m": 5.11},
    {"name": "web-BerkStan",        "short": "berkstan",   "type": "web",           "vertices_m": 0.69, "edges_m": 7.60},
    {"name": "as-Skitter",          "short": "skitter",    "type": "infrastructure", "vertices_m": 1.70, "edges_m": 11.10},
    {"name": "coPapersDBLP",        "short": "copapers",   "type": "collaboration", "vertices_m": 0.54, "edges_m": 15.25},
    {"name": "coPapersCiteseer",    "short": "cociteseer", "type": "citation",      "vertices_m": 0.43, "edges_m": 16.04},
    {"name": "com-Youtube",         "short": "youtube",    "type": "social",        "vertices_m": 1.13, "edges_m": 2.99},
    {"name": "in-2004",             "short": "in04",       "type": "web",           "vertices_m": 1.38, "edges_m": 16.92},
    {"name": "rgg_n_2_20_s0",       "short": "rgg20",      "type": "mesh",          "vertices_m": 1.05, "edges_m": 5.83},
    {"name": "kron_g500-logn18",    "short": "kron18",     "type": "synthetic",     "vertices_m": 0.26, "edges_m": 21.17},
    {"name": "soc-Slashdot0811",    "short": "slashdot",   "type": "social",        "vertices_m": 0.08, "edges_m": 0.91},
    {"name": "cit-HepPh",           "short": "hepph",      "type": "citation",      "vertices_m": 0.03, "edges_m": 0.42},
    {"name": "cnr-2000",            "short": "cnr",        "type": "web",           "vertices_m": 0.33, "edges_m": 3.22},
    {"name": "dblp-2010",           "short": "dblp",       "type": "collaboration", "vertices_m": 0.33, "edges_m": 1.62},
]

# Local evaluation (fits 64GB RAM, covers all topology types from paper Table 5).
# Results use the same chart pipeline as EVAL_GRAPHS — just swap the list and
# re-run.  Full suite targets the lab machine with 256GB RAM.
EVAL_GRAPHS_LOCAL = [
    _main_graph("cit-Patents"),
    _main_graph("soc-pokec"),
    _main_graph("USA-road-d.USA"),
    _main_graph("soc-LiveJournal1"),
    _main_graph("hollywood-2009"),
    _main_graph("com-Orkut"),
]

# Small graphs for preview mode
PREVIEW_GRAPHS = [
    {"name": "email-Eu-core",      "short": "email",     "type": "social",   "vertices_m": 0.001,  "edges_m": 0.025},
    _main_graph("cit-Patents"),
]

# ---------------------------------------------------------------------------
# Experiment Parameters
# ---------------------------------------------------------------------------

# Number of algorithm trials per benchmark
TRIALS_FULL = 5
TRIALS_PREVIEW = 1
REORDER_TRIALS_FULL = 3
REORDER_TRIALS_PREVIEW = 1
CACHE_TRIALS = 1

# Weighted SSSP policy. Delta is tuned on SHUFFLED only, then frozen across
# every ordering for that graph. Populate this table only from the recorded
# tuner artifact after semantic consistency checks pass.
SSSP_WEIGHT_SCHEME = "hash"
SSSP_DELTA_CANDIDATES = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
]
SSSP_TUNING_SOURCES = 3
SSSP_TUNING_REPEATS = 1
SSSP_TUNING_REPLICATES = 3
SSSP_TUNING_TRIALS = (
    SSSP_TUNING_SOURCES
    * SSSP_TUNING_REPEATS
    * SSSP_TUNING_REPLICATES
)
SSSP_TUNING_PRACTICAL_TIE_RATIO = 1.02
SSSP_TUNING_T_CRITICAL_95_DF8 = 1.8595480375
SSSP_TUNING_ORDER_POLICY = "cyclic-shift/v1"
SSSP_SELECTION_RULE_ID = "pooled-block-median-tie/v2"
SSSP_POLICY_PATH = Path(
    os.environ.get(
        "GRAPHBREW_SSSP_POLICY_PATH",
        Path(__file__).resolve().parent / "sssp_policy.json",
    )
).resolve()
SSSP_TUNING_SNAPSHOT_PATH = Path(
    os.environ.get(
        "GRAPHBREW_SSSP_TUNING_SNAPSHOT_PATH",
        Path(__file__).resolve().parent / "sssp_delta_tuning.json",
    )
).resolve()
if SSSP_POLICY_PATH.is_file():
    _sssp_policy_payload = json.loads(SSSP_POLICY_PATH.read_text())
    if _sssp_policy_payload.get("schema") != "sssp_policy/v1":
        raise RuntimeError(
            f"Unsupported SSSP policy schema: {SSSP_POLICY_PATH}"
        )
    SSSP_POLICY: dict[str, dict] = dict(
        _sssp_policy_payload.get("policies", {})
    )
    SSSP_POLICY_SELECTION_RULE_ID = _sssp_policy_payload.get(
        "selection_rule_id"
    )
else:
    SSSP_POLICY = {}
    SSSP_POLICY_SELECTION_RULE_ID = None
RABBIT_MAPPING_DRAWS = 3
PR_FIXED_ITERATIONS = 20
PR_CONVERGENCE_MAX_ITERATIONS = 100
PR_TOLERANCE = 1e-4
BC_SOURCE_ITERATIONS = 1
VERIFICATION_TIMEOUT_MULTIPLIER = 4
E2E_REUSE_COUNTS = list(END_TO_END_REUSE_COUNTS)

# Timeout per command (seconds)
TIMEOUT_FULL = 3600
TIMEOUT_PREVIEW = 300
REORDER_TIMEOUT_FULL = 12 * 60 * 60
REORDER_TIMEOUT_PREVIEW = TIMEOUT_PREVIEW
OPTIONAL_REFINEMENT_TIMEOUT = 6 * 60 * 60
REORDER_TIMING_REUSE_GRAPHS = {
    "wikipedia_link_en",
    "Gong-gplus",
    "webbase-2001",
    "twitter7",
}
REORDER_TIMING_ANCHOR_ALGOS = {"1", "4", "5"}
PROMOTED_GORDER_GRAPHS = {
    "cit-Patents",
    "soc-pokec",
    "USA-road-d.USA",
    "soc-LiveJournal1",
    "delaunay_n24",
    "com-Orkut",
    "hollywood-2009",
    "wikipedia_link_en",
    "Gong-gplus",
    "webbase-2001",
}

# Final every-access cache-capacity sweep (bytes). Additional capacities remain
# available through --cache-sizes-kib for exploratory runs.
CACHE_SIZES = list(cache_capacities_bytes(
    PAPER_CACHE_CAPACITIES_MIB))

# Reduced sweep for rapid validation.
CACHE_SIZES_PREVIEW = list(cache_capacities_bytes(
    PAPER_CACHE_PREVIEW_CAPACITIES_MIB))

# Cache comparison focuses on the primary competition and controlled
# GraphBrew candidates. Use --cache-all-algorithms for the full matrix.
CACHE_ALGORITHM_KEYS = [
    "0",
    "5",
    "8:csr",
    "8:boost",
    "9:csr",
    "11",
    "12:leiden",
    "12:hrab:bfs_intra",
    "12:hrab",
    "12:tqr",
]
CACHE_ALGORITHM_KEYS_PREVIEW = [
    "0",
    "8:csr",
    "8:boost",
    "12:leiden",
    "12:hrab:bfs_intra",
    "12:hrab",
]

# Use one OpenMP thread per physical core for the publication scalability
# study. SMT can be evaluated separately without changing the primary matrix.
THREAD_COUNTS = [1, 2, 4, 8, 16]
SCALABILITY_REPEATS = 3
STABLE_BLOCK_ALGORITHM_KEY = (
    "12:rabbit:compose:"
    "sg_super_rabbit:comm_degree_desc:intra_hubsort"
)
SCALABILITY_ALGORITHM_KEYS = [
    "9:csr",
    "8:csr",
    "8:boost",
    "12:rabbit",
    "12:leiden",
    "12:hrab",
    STABLE_BLOCK_ALGORITHM_KEY,
]
SCALABILITY_GRAPH_NAMES = [
    "cit-Patents",
    "USA-road-d.USA",
    "com-Orkut",
]
SCALABILITY_TWITTER_GRAPH = "twitter7"
SCALABILITY_TWITTER_ALGORITHM_KEYS = [
    "8:csr",
    "8:boost",
    "12:rabbit",
    STABLE_BLOCK_ALGORITHM_KEY,
]

# Readable paper subset for Experiment 4. The JSON artifact retains every
# measured method; the paper table shows the primary conventional competitors
# and all ten core GraphBrew variants. Controlled compose candidates remain in
# Exp5/Exp7.
E2E_PAPER_ALGORITHM_KEYS = [
    "5",
    "8:csr",
    "8:boost",
    "9:csr",
    "11",
    "16",
    *[f"12:{variant}" for variant in GRAPHBREW_VARIANTS],
]

# ---------------------------------------------------------------------------
# Ablation study configurations (Experiment 5)
# ---------------------------------------------------------------------------

_ABLATION_RABBIT_DENDRO = (
    "12:rabbit:compose:sg_none:comm_identity:intra_dendrogram"
)
_ABLATION_RABBIT_HUBSORT = (
    "12:rabbit:compose:sg_none:comm_identity:intra_hubsort"
)
_ABLATION_SUPER_HUBSORT = (
    "12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort"
)
_ABLATION_SUPER_DEGREE_HUBSORT = STABLE_BLOCK_ALGORITHM_KEY
_ABLATION_LEIDEN_HUBSORT = (
    "12:leiden:compose:sg_none:comm_identity:intra_hubsort"
)
_ABLATION_REFINED_HUBSORT = (
    "12:rabbit:compose:"
    "sg_none:comm_identity:intra_hubsort:refine_2swap"
)
_BUDGETED_GORDER500 = (
    "12:leiden:compose:sg_none:comm_identity:"
    "intra_gorder:gw32:gordf500:cd_parallel:1:1"
)
_BUDGETED_BFS = (
    "12:leiden:compose:sg_none:comm_identity:"
    "intra_bfs:gw32:gordf500:cd_parallel:1:1"
)
_BUDGETED_GORDER_UNCAPPED = (
    "12:leiden:compose:sg_none:comm_identity:"
    "intra_gorder:gw32:gord:cd_parallel:1:1"
)
_BUDGETED_GORDER_SERIAL = (
    "12:leiden:compose:sg_none:comm_identity:"
    "intra_gorder:gw32:gordf500:cd_serial:1:1"
)
_BUDGETED_GORDER_SIZE_DESC = (
    "12:leiden:compose:sg_none:comm_size_desc:"
    "intra_gorder:gw32:gordf500:cd_parallel:1:1"
)
_FAST_LEIDEN_BFS = (
    "12:leiden:compose:sg_none:comm_identity:"
    "intra_bfs:cd_parallel:1:1"
)
_FULL_LEIDEN_SIZE_BFS = (
    "12:leiden:compose:sg_none:comm_size_desc:"
    "intra_bfs:gw8"
)
_ONEPASS_LEIDEN_SIZE_GORDER8 = (
    "12:leiden:compose:sg_none:comm_size_desc:"
    "intra_gorder:gw8:cd_parallel:1:1"
)
_ONEPASS_LEIDEN_IDENTITY_GORDER8 = (
    "12:leiden:compose:sg_none:comm_identity:"
    "intra_gorder:gw8:cd_parallel:1:1"
)
_ONEPASS_LEIDEN_SIZE_BFS = (
    "12:leiden:compose:sg_none:comm_size_desc:"
    "intra_bfs:gw8:cd_parallel:1:1"
)

ABLATION_CONTRASTS = [
    {
        "name": "Vertex layout",
        "base": _ABLATION_RABBIT_DENDRO,
        "variant": _ABLATION_RABBIT_HUBSORT,
        "effective_fields": ["intra_community_order"],
    },
    {
        "name": "Supergraph order",
        "base": _ABLATION_RABBIT_HUBSORT,
        "variant": _ABLATION_SUPER_HUBSORT,
        "effective_fields": ["super_graph"],
    },
    {
        "name": "Stable block sort",
        "base": _ABLATION_SUPER_HUBSORT,
        "variant": _ABLATION_SUPER_DEGREE_HUBSORT,
        "effective_fields": ["community_order"],
    },
    {
        "name": "Partitioner stack",
        "base": _ABLATION_RABBIT_HUBSORT,
        "variant": _ABLATION_LEIDEN_HUBSORT,
        "contrast_type": "bundled-substitution",
        "effective_fields": [
            "aggregation",
            "algorithm",
            "final_algo_id",
            "m_computation",
            "rabbit_degree_sort_preprocess",
            "refinement_depth",
            "resolution",
            "small_community_merging",
        ],
    },
    {
        "name": "Refinement pass",
        "base": _ABLATION_RABBIT_HUBSORT,
        "variant": _ABLATION_REFINED_HUBSORT,
        "effective_fields": ["refinement_pass"],
    },
    {
        "name": "Budgeted vertex layout",
        "base": _BUDGETED_GORDER500,
        "variant": _BUDGETED_BFS,
        "effective_fields": ["intra_community_order"],
    },
    {
        "name": "Gorder community budget",
        "base": _BUDGETED_GORDER500,
        "variant": _BUDGETED_GORDER_UNCAPPED,
        "effective_fields": ["gorder_fallback"],
    },
    {
        "name": "Community-detection schedule",
        "base": _BUDGETED_GORDER500,
        "variant": _BUDGETED_GORDER_SERIAL,
        "effective_fields": ["deterministic_community_detection"],
    },
    {
        "name": "Budgeted block order",
        "base": _BUDGETED_GORDER500,
        "variant": _BUDGETED_GORDER_SIZE_DESC,
        "effective_fields": ["community_order"],
    },
    {
        "name": "One-pass Gorder block order",
        "base": _ONEPASS_LEIDEN_SIZE_GORDER8,
        "variant": _ONEPASS_LEIDEN_IDENTITY_GORDER8,
        "effective_fields": ["community_order"],
    },
    {
        "name": "One-pass vertex layout",
        "base": _ONEPASS_LEIDEN_SIZE_GORDER8,
        "variant": _ONEPASS_LEIDEN_SIZE_BFS,
        "effective_fields": ["intra_community_order"],
    },
    {
        "name": "Leiden execution budget",
        "base": _FULL_LEIDEN_SIZE_BFS,
        "variant": _ONEPASS_LEIDEN_SIZE_BFS,
        "contrast_type": "bundled-budget-substitution",
        "effective_fields": [
            "deterministic_community_detection",
            "max_iterations",
            "max_passes",
        ],
    },
]

_BUDGETED_MECHANISM_NAMES = {
    _BUDGETED_GORDER500: "BudgetedLeiden-Identity-Gorder500",
    _BUDGETED_BFS: "BudgetedLeiden-Identity-BFS",
    _BUDGETED_GORDER_UNCAPPED: "BudgetedLeiden-Identity-GorderUncapped",
    _BUDGETED_GORDER_SERIAL: "BudgetedLeidenSerial-Identity-Gorder500",
    _BUDGETED_GORDER_SIZE_DESC: "BudgetedLeiden-SizeDesc-Gorder500",
    _FAST_LEIDEN_BFS: "FastLeiden-Identity-BFS",
    _FULL_LEIDEN_SIZE_BFS: "LeidenGVE-SizeDesc-BFS",
    _ONEPASS_LEIDEN_SIZE_GORDER8:
        "FastLeiden-SizeDesc-Gorder8",
    _ONEPASS_LEIDEN_IDENTITY_GORDER8:
        "FastLeiden-Identity-Gorder8",
    _ONEPASS_LEIDEN_SIZE_BFS:
        "FastLeiden-SizeDesc-BFS",
}

ABLATION_CONFIGS = [
    {"name": "Shuffled", "algo": "0", "desc": "Seeded shuffled input layout"},
    *[
        {
            "name": (
                _BUDGETED_MECHANISM_NAMES[spec]
                if spec in _BUDGETED_MECHANISM_NAMES
                else GRAPHBREW_DISPLAY_NAMES[spec]
            ),
            "algo": spec,
            "desc": "Controlled Experiment-5 configuration",
        }
        for spec in dict.fromkeys(
            item
            for contrast in ABLATION_CONTRASTS
            for item in (contrast["base"], contrast["variant"])
        )
    ],
]
for config in ABLATION_CONFIGS:
    if config["algo"] in _BUDGETED_MECHANISM_NAMES:
        config["name"] = _BUDGETED_MECHANISM_NAMES[config["algo"]]
        config["desc"] = (
            "Frozen budgeted Leiden-Gorder mechanism contrast"
        )
ABLATION_CONFIGS.append({
    "name": _BUDGETED_MECHANISM_NAMES[_FAST_LEIDEN_BFS],
    "algo": _FAST_LEIDEN_BFS,
    "desc": "One-pass parallel Leiden with identity blocks and BFS",
})

DIAGNOSTIC_CONFIGS = [
    {
        "name": f"FastLeiden-SizeDesc-Gorder8-{iterations}x{passes}",
        "algo": (
            "12:leiden:compose:sg_none:comm_size_desc:"
            f"intra_gorder:gw8:cd_parallel:{iterations}:{passes}"
        ),
        "desc": "Diagnostic-only parallel Leiden budget frontier",
    }
    for iterations, passes in (
        (2, 1),
        (4, 1),
        (2, 2),
        (4, 2),
        (6, 3),
    )
]
DIAGNOSTIC_CONFIGS.extend([
    {
        "name": "LeidenGVE-DegreeDesc-Gorder8",
        "algo": (
            "12:leiden:compose:sg_none:comm_degree_desc:"
            "intra_gorder:gw8"
        ),
        "desc": "Explicit-only fixed-membership mechanism factorial",
    },
    {
        "name": "RCM-MIND-Diagnostic",
        "algo": "11:mind",
        "desc": "Explicit-only single-pass MIND-start RCM upper bound",
    },
    {
        "name": "RCM-BNF-Diagnostic",
        "algo": "11:bnf",
        "desc": "Explicit-only CSR-native BNF RCM upper bound",
    },
])

DUAL_ARM_S0_CONFIGS = [
    {
        "name": "S0-FastLeiden-SizeDesc-Overlap5000-1x1",
        "algo": (
            "12:leiden:compose:sg_none:comm_size_desc:"
            "intra_gorder:gw8:cd_parallel:sgmb4096:"
            "gordf5000:norefine:1:1"
        ),
        "desc": "Development-only dual-arm budget ladder",
    },
    {
        "name": "S0-FastLeiden-SizeDesc-Overlap500-1x1",
        "algo": (
            "12:leiden:compose:sg_none:comm_size_desc:"
            "intra_gorder:gw8:cd_parallel:sgmb4096:"
            "gordf500:norefine:1:1"
        ),
        "desc": "Development-only dual-arm budget ladder",
    },
    {
        "name": "S0-FastLeiden-SizeDesc-BFS-1x1",
        "algo": (
            "12:leiden:compose:sg_none:comm_size_desc:"
            "intra_bfs:cd_parallel:sgmb4096:norefine:1:1"
        ),
        "desc": "Development-only dual-arm budget ladder",
    },
]

DUAL_ARM_S2_CONFIGS = [
    {
        "name": "S2-FastLeiden-Identity-BFS-1x1-B4096",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs:cd_parallel:sgmb4096:norefine:1:1"
        ),
        "desc": "Development-only sparse-graph mapping-cost screen",
    },
    {
        "name": "S2-FastLeiden-SizeAsc-BFS-1x1-B4096",
        "algo": (
            "12:leiden:compose:sg_none:comm_size_asc:"
            "intra_bfs:cd_parallel:sgmb4096:norefine:1:1"
        ),
        "desc": "Development-only sparse-graph mapping-cost screen",
    },
    {
        "name": "S2-FastLeiden-SizeDesc-BFS-1x1-B8192",
        "algo": (
            "12:leiden:compose:sg_none:comm_size_desc:"
            "intra_bfs:cd_parallel:sgmb8192:norefine:1:1"
        ),
        "desc": "Development-only sparse-graph mapping-cost screen",
    },
    {
        "name": "S2-FastLeiden-SizeDesc-BFS-1x1-B16384",
        "algo": (
            "12:leiden:compose:sg_none:comm_size_desc:"
            "intra_bfs:cd_parallel:sgmb16384:norefine:1:1"
        ),
        "desc": "Development-only sparse-graph mapping-cost screen",
    },
]

DUAL_ARM_V3_CONFIGS = [
    {
        "name": "V3-Serial-Identity-BFS-Control",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs:cd_serial:norefine:1:1"
        ),
        "desc": "Deterministic direct-emission equivalence control",
    },
    {
        "name": "V3-Serial-Identity-BFSDirect-Control",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs_direct:cd_serial:norefine:1:1"
        ),
        "desc": "Deterministic direct-emission equivalence control",
    },
    {
        "name": "V3-Serial-Identity-BFSCompact-Control",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs_compact:cd_serial:norefine:1:1"
        ),
        "desc": "Deterministic compact-emission equivalence control",
    },
    {
        "name": "V3-Serial-Identity-BFSCompactDirect-Control",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs_compact_direct:cd_serial:norefine:1:1"
        ),
        "desc": "Deterministic compact-and-emit equivalence control",
    },
    {
        "name": "V3-FastLeiden-Identity-BFS-1x1-B4096",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs:cd_parallel:sgmb4096:norefine:1:1"
        ),
        "desc": "Parallel sparse-ID scheduling baseline",
    },
    {
        "name": "V3-FastLeiden-Identity-BFSDirect-1x1-B4096",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs_direct:cd_parallel:sgmb4096:norefine:1:1"
        ),
        "desc": "Native fused BFS and final-ID emission",
    },
    {
        "name": "V3-FastLeiden-Identity-BFSCompact-1x1-B4096",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs_compact:cd_parallel:sgmb4096:norefine:1:1"
        ),
        "desc": "Native compact active-community emission",
    },
    {
        "name": "V3-FastLeiden-Identity-BFSCompactDirect-1x1-B4096",
        "algo": (
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_bfs_compact_direct:cd_parallel:sgmb4096:"
            "norefine:1:1"
        ),
        "desc": "Native compact-and-emit composition",
    },
]

DUAL_ARM_V5_CONFIGS = [
    {
        "name": f"V5-FastLeiden-{label}-BFSCompactDirect-1x1-B4096",
        "algo": (
            f"12:leiden:compose:sg_none:{token}:"
            "intra_bfs_compact_direct:cd_parallel:sgmb4096:"
            "norefine:1:1"
        ),
        "desc": "Development-only compact-and-emit community-order screen",
    }
    for label, token in (
        ("SizeDesc", "comm_size_desc"),
        ("SizeAsc", "comm_size_asc"),
        ("DegreeDesc", "comm_degree_desc"),
        ("DegreeAsc", "comm_degree_asc"),
    )
]

DUAL_ARM_V6_CONFIGS = [
    {
        "name": f"V6-FastLeiden-{community}-{intra}-1x1-B4096",
        "algo": (
            f"12:leiden:compose:sg_none:{community_token}:"
            f"{intra_token}:cd_parallel:sgmb4096:norefine:1:1"
        ),
        "desc": "Development-only fast intra-order quality-potential screen",
    }
    for community, community_token in (
        ("SizeDesc", "comm_size_desc"),
        ("DegreeDesc", "comm_degree_desc"),
    )
    for intra, intra_token in (
        ("HubSort", "intra_hubsort"),
        ("DegreeAsc", "intra_deg_asc"),
    )
]

COMPOSITION_P0_CONFIGS = [
    {
        "name": name,
        "algo": spec,
        "desc": "Explicit-only composition P0 treatment",
    }
    for name, spec in (
        (
            "P0-Identity-HubSort",
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_hubsort:cd_serial:refine_none",
        ),
        (
            "P0-Identity-Alternate",
            "12:leiden:compose:sg_none:comm_identity:"
            "intra_alternate:cd_serial:refine_none",
        ),
        (
            "P0-SuperRCM-Identity-HubSort",
            "12:leiden:compose:sg_super_rcm:comm_identity:"
            "intra_hubsort:cd_serial:refine_none",
        ),
        (
            "P0-SuperRCM-Identity-Alternate",
            "12:leiden:compose:sg_super_rcm:comm_identity:"
            "intra_alternate:cd_serial:refine_none",
        ),
    )
]
COMPOSITION_P0_ALGORITHM_KEYS = tuple(
    config["algo"] for config in COMPOSITION_P0_CONFIGS
)

ALGORITHM_GRAPH_EXCLUSIONS = {
    "twitter7": {
        _ABLATION_REFINED_HUBSORT: (
            f"two-swap refinement exceeded the deterministic "
            f"{OPTIONAL_REFINEMENT_TIMEOUT}-second optional-refinement "
            "applicability budget in both original and optimized "
            "implementations"
        ),
    },
}

ALGORITHM_GRAPH_EXCLUSION_EVIDENCE = {
    "twitter7": {
        _ABLATION_REFINED_HUBSORT: {
            "schema": "reorder_timeout_evidence/v1",
            "artifact": (
                "scripts/experiments/vldb/"
                "twitter_refine_2swap_timeout_evidence.json"
            ),
            "external_artifact": (
                "pre_stage3_validation/"
                "twitter-refine-2swap-timeout-evidence.json"
            ),
            "timeout_seconds": OPTIONAL_REFINEMENT_TIMEOUT,
        },
    },
}


def algorithm_exclusion_reason(
    graph_name: str,
    algorithm_key: str,
) -> str | None:
    return ALGORITHM_GRAPH_EXCLUSIONS.get(
        graph_name, {},
    ).get(algorithm_key)


# ---------------------------------------------------------------------------
# Graph type groupings (Experiment 6)
# ---------------------------------------------------------------------------

GRAPH_TYPE_GROUPS = {
    "social":        ["soc-pokec", "com-Orkut", "soc-LiveJournal1", "Gong-gplus", "twitter7"],
    "web":           ["webbase-2001"],
    "road":          ["USA-road-d.USA"],
    "citation":      ["cit-Patents"],
    "content":       ["wikipedia_link_en"],
    "collaboration": ["hollywood-2009"],
    "mesh":          ["delaunay_n24"],
}


def get_converter_flags(algo_key: str) -> list[str]:
    """Convert an algorithm key like '12:hrab' to converter CLI flags."""
    if ":" in algo_key:
        parts = algo_key.split(":", 1)
        return ["-o", f"{parts[0]}:{parts[1]}"]
    return ["-o", algo_key]


# ---------------------------------------------------------------------------
# VLDB Graph Download Sources
# ---------------------------------------------------------------------------
# Maps EVAL_GRAPHS name → download info.  Graphs marked source="suitesparse"
# are fetched automatically; others require manual download.

VLDB_GRAPH_SOURCES = {
    # ── Auto-download from SuiteSparse (already in catalog) ──
    "email-Eu-core":     {"source": "catalog"},
    "cit-Patents":       {"source": "catalog"},
    "soc-pokec":         {"source": "catalog"},
    "USA-road-d.USA":    {"source": "catalog"},
    "soc-LiveJournal1":  {"source": "catalog"},
    "delaunay_n24":      {"source": "catalog"},
    "com-Orkut":         {"source": "catalog"},
    "hollywood-2009":    {"source": "catalog"},
    "webbase-2001":      {"source": "catalog"},
    "twitter7":          {"source": "catalog"},
    "email-Enron":       {"source": "catalog"},
    "wiki-Talk":         {"source": "catalog"},
    "wiki-topcats":      {"source": "catalog"},
    "roadNet-CA":        {"source": "catalog"},
    "roadNet-TX":        {"source": "catalog"},
    "amazon0601":        {"source": "catalog"},
    "web-Google":        {"source": "catalog"},
    "web-BerkStan":      {"source": "catalog"},
    "coPapersDBLP":      {"source": "catalog"},
    "coPapersCiteseer":  {"source": "catalog"},
    "com-Youtube":       {"source": "catalog"},
    "in-2004":           {"source": "catalog"},
    "rgg_n_2_20_s0":     {"source": "catalog"},
    "kron_g500-logn18":  {"source": "catalog"},
    "soc-Slashdot0811":  {"source": "catalog"},
    "cit-HepPh":         {"source": "catalog"},
    "cnr-2000":          {"source": "catalog"},
    "dblp-2010":         {"source": "catalog"},

    # ── Additional graphs for EVAL_GRAPHS_64GB ──
    "as-Skitter":        {"source": "catalog"},
    "kron_g500-logn21":  {"source": "catalog"},
    "indochina-2004":    {"source": "catalog"},
    "uk-2002":           {"source": "catalog"},

    # ── Manual download required ──
    "wikipedia_link_en": {
        "source": "manual",
        "url": "http://konect.cc/networks/wikipedia_link_en/",
        "instructions": (
            "Download from KONECT: http://konect.cc/networks/wikipedia_link_en/\n"
            "Extract and convert to Matrix Market (.mtx) or edge-list (.el) format,\n"
            "then place as results/graphs/wikipedia_link_en/wikipedia_link_en.el"
        ),
    },
    "Gong-gplus": {
        "source": "manual",
        "url": "https://people.duke.edu/~zg70/gplus.html",
        "gdrive_id": "1HF8Q2N_hxsaQ26MarKYxZEQhqI66qAxV",
        "instructions": (
            "Download from https://people.duke.edu/~zg70/gplus.html\n"
            "  (Google Drive: https://drive.google.com/file/d/"
            "1HF8Q2N_hxsaQ26MarKYxZEQhqI66qAxV/view)\n"
            "Extract snapshot 4 (keep all edges with TimeID 0-3), convert to\n"
            "edge-list format, and place as results/graphs/Gong-gplus/Gong-gplus.el"
        ),
    },
}
