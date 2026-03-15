# ============================================================================
# gem5 SimObject definitions for GraphBrew graph-aware replacement policies
# ============================================================================
#
# Defines the Python-side SimObject classes that expose GRASP, P-OPT, and
# ECG replacement policies to gem5's configuration system.
#
# Usage in gem5 Python config:
#   from m5.objects import GraphGraspRP, GraphPoptRP, GraphEcgRP
#
#   l3_repl = GraphGraspRP(max_rrpv=7, num_buckets=11, hot_fraction=0.1)
#   l3_repl = GraphPoptRP(max_rrpv=7)
#   l3_repl = GraphEcgRP(rrpv_max=7, num_buckets=11, ecg_mode="DBG_PRIMARY")
# ============================================================================

from m5.params import *
from m5.proxy import *
from m5.SimObject import SimObject
from m5.objects.ReplacementPolicies import BaseReplacementPolicy


class GraphGraspRP(BaseReplacementPolicy):
    """GRASP: Graph-aware cache Replacement with Software Prefetching
    (Faldu et al., HPCA 2020)

    Extends SRRIP with degree-based 3-tier insertion and hit promotion.
    Requires DBG-reordered graph or GraphCacheContext for bucket classification.
    """
    type = 'GraphGraspRP'
    cxx_header = "mem/cache/replacement_policies/grasp_rp.hh"
    cxx_class = 'gem5::replacement_policy::GraphGraspRP'

    max_rrpv = Param.Int(7,
        "Maximum RRPV value (2^rrpv_bits - 1). Default 7 for 3-bit RRPV.")
    num_buckets = Param.Int(11,
        "Number of degree buckets for vertex classification (matching DBG).")
    hot_fraction = Param.Float(0.1,
        "Fraction of LLC capacity reserved for high-degree hub vertices.")


class GraphPoptRP(BaseReplacementPolicy):
    """P-OPT: Practical Optimal cache replacement for Graph Analytics
    (Balaji et al., HPCA 2021)

    Oracle baseline using pre-computed rereference distances from the graph
    transpose. 3-phase eviction: non-graph first, then max rereference
    distance, then RRIP tiebreaker.
    """
    type = 'GraphPoptRP'
    cxx_header = "mem/cache/replacement_policies/popt_rp.hh"
    cxx_class = 'gem5::replacement_policy::GraphPoptRP'

    max_rrpv = Param.Int(7,
        "Maximum RRPV value for tiebreaking. Default 7 (3-bit).")


class GraphEcgRP(BaseReplacementPolicy):
    """ECG: Expressing Locality and Prefetching for Optimal Caching in Graphs
    (Mughrabi et al., GrAPL @ IPDPS 2026)

    3-level layered eviction with mode-dependent tiebreaker:
      DBG_PRIMARY:  SRRIP → DBG tier → dynamic P-OPT (default)
      POPT_PRIMARY: SRRIP → dynamic P-OPT → DBG tier
      DBG_ONLY:     SRRIP → DBG tier (fast path, no P-OPT)

    Supports per-access mask hints via custom ECG instruction.
    """
    type = 'GraphEcgRP'
    cxx_header = "mem/cache/replacement_policies/ecg_rp.hh"
    cxx_class = 'gem5::replacement_policy::GraphEcgRP'

    rrpv_max = Param.Int(7,
        "Maximum RRPV value. 7 for 3-bit, 255 for 8-bit.")
    num_buckets = Param.Int(11,
        "Number of degree buckets for DBG classification.")
    ecg_mode = Param.String("DBG_PRIMARY",
        "Eviction mode: DBG_PRIMARY, POPT_PRIMARY, or DBG_ONLY.")
