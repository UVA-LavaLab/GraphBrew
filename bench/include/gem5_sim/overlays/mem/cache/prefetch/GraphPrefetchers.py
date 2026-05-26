# ============================================================================
# gem5 SimObject definition for DROPLET graph prefetcher
# ============================================================================

from m5.params import *
from m5.proxy import *
from m5.SimObject import SimObject
from m5.objects.Prefetcher import QueuedPrefetcher


class GraphDropletPrefetcher(QueuedPrefetcher):
    """DROPLET: Data-awaRe decOuPLed prEfeTcher for Graphs
    (Basak et al., HPCA 2019)

    Separated prefetch engines for edge-list (stride) and property data
    (indirect). The indirect chain: edge_list[i] → neighbor_id →
    property[neighbor_id] decouples from the core's dependency chain.

    Performance: 1.37x average speedup, 15-45% LLC miss reduction.
    Complementary with ECG replacement policy.
    """
    type = 'GraphDropletPrefetcher'
    cxx_header = "mem/cache/prefetch/droplet.hh"
    cxx_class = 'gem5::prefetch::GraphDropletPrefetcher'

    prefetch_degree = Param.Int(4,
        "Number of edge-list cache lines to prefetch ahead.")
    indirect_degree = Param.Int(8,
        "Number of indirect property prefetches per edge access.")
    stride_table_size = Param.Int(16,
        "Number of entries in the stride detector table.")


class GraphEcgPfxPrefetcher(QueuedPrefetcher):
    """ECG_PFX: consumes GraphBrew ECG prefetch target hints.

    The target arrives through a GraphBrew m5ops work item emitted by benchmark
    code after decoding ECG/fat-ID metadata. This differs from DROPLET, which
    infers property targets from CSR edge streams.
    """
    type = 'GraphEcgPfxPrefetcher'
    cxx_header = "mem/cache/prefetch/ecg_pfx.hh"
    cxx_class = 'gem5::prefetch::GraphEcgPfxPrefetcher'

    recent_filter_size = Param.Int(256,
        "Number of recently issued property-line prefetches to suppress.")
