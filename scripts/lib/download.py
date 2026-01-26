#!/usr/bin/env python3
"""
Graph downloading utilities for GraphBrew.

Downloads benchmark graphs from SuiteSparse Matrix Collection.
Can be used standalone or as a library.

Standalone usage:
    python -m scripts.lib.download --size SMALL
    python -m scripts.lib.download --list
    python -m scripts.lib.download --graph email-Enron

Library usage:
    from scripts.lib.download import download_graphs, GRAPH_CATALOG
    from scripts.lib.download import DOWNLOAD_GRAPHS_SMALL, DOWNLOAD_GRAPHS_MEDIUM
    
    download_graphs(size="SMALL")
    download_graphs(graphs=["email-Enron", "web-Google"])
"""

import os
import sys
import tarfile
import gzip
import shutil
import urllib.request
import urllib.error
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from .utils import GRAPHS_DIR, RESULTS_DIR, log, ensure_directories


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DownloadableGraph:
    """Information about a downloadable graph from SuiteSparse."""
    name: str
    url: str
    size_mb: int
    nodes: int
    edges: int
    symmetric: bool  # Is the graph undirected/symmetric
    category: str    # Graph type: social, web, road, citation, etc.
    description: str = ""
    
    @property
    def avg_degree(self) -> float:
        """Average degree (edges/nodes)."""
        return self.edges / self.nodes if self.nodes > 0 else 0
    
    def estimated_memory_gb(self) -> float:
        """Estimate memory required to process this graph in GB.
        
        Based on CSR format: approximately 12 bytes per edge (offsets + neighbors)
        plus some overhead for node data.
        """
        bytes_per_edge = 12  # 4 bytes offset + 8 bytes for edge data
        bytes_per_node = 8   # Node data overhead
        edge_bytes = self.edges * bytes_per_edge
        node_bytes = self.nodes * bytes_per_node
        overhead = 1.5  # Safety factor for algorithm workspace
        return (edge_bytes + node_bytes) * overhead / (1024 ** 3)


# =============================================================================
# Graph Catalog - Complete Collection
# =============================================================================

# Small graphs (<20MB) - good for quick testing and development
DOWNLOAD_GRAPHS_SMALL = [
    # Communication networks
    DownloadableGraph("email-Enron", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-Enron.tar.gz",
                      5, 36692, 183831, True, "communication", "Enron email network"),
    DownloadableGraph("email-EuAll", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-EuAll.tar.gz",
                      4, 265214, 420045, False, "communication", "EU email network"),
    # Collaboration networks
    DownloadableGraph("ca-AstroPh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-AstroPh.tar.gz",
                      3, 18772, 198110, True, "collaboration", "Arxiv Astro Physics"),
    DownloadableGraph("ca-CondMat", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-CondMat.tar.gz",
                      2, 23133, 93497, True, "collaboration", "Condensed Matter"),
    DownloadableGraph("ca-GrQc", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-GrQc.tar.gz",
                      1, 5242, 14496, True, "collaboration", "General Relativity"),
    DownloadableGraph("ca-HepPh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-HepPh.tar.gz",
                      2, 12008, 118521, True, "collaboration", "High Energy Physics"),
    DownloadableGraph("ca-HepTh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-HepTh.tar.gz",
                      1, 9877, 25998, True, "collaboration", "High Energy Physics Theory"),
    # P2P networks
    DownloadableGraph("p2p-Gnutella31", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella31.tar.gz",
                      2, 62586, 147892, False, "p2p", "Gnutella P2P network"),
    DownloadableGraph("p2p-Gnutella30", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella30.tar.gz",
                      1, 36682, 88328, False, "p2p", "Gnutella P2P Aug 30"),
    DownloadableGraph("p2p-Gnutella25", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella25.tar.gz",
                      1, 22687, 54705, False, "p2p", "Gnutella P2P Aug 25"),
    DownloadableGraph("p2p-Gnutella24", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella24.tar.gz",
                      1, 26518, 65369, False, "p2p", "Gnutella P2P Aug 24"),
    # Social networks (small)
    DownloadableGraph("soc-Slashdot0811", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Slashdot0811.tar.gz",
                      8, 77360, 905468, False, "social", "Slashdot Nov 2008"),
    DownloadableGraph("soc-Slashdot0902", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Slashdot0902.tar.gz",
                      9, 82168, 948464, False, "social", "Slashdot Feb 2009"),
    DownloadableGraph("soc-sign-epinions", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-sign-epinions.tar.gz",
                      10, 131828, 841372, False, "social", "Epinions signed network"),
    # Citation networks (small)
    DownloadableGraph("cit-HepPh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-HepPh.tar.gz",
                      8, 34546, 421578, False, "citation", "HEP-PH citations"),
    DownloadableGraph("cit-HepTh", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-HepTh.tar.gz",
                      4, 27770, 352807, False, "citation", "HEP-TH citations"),
]

# Medium graphs (20MB - 200MB) - ~35 graphs
DOWNLOAD_GRAPHS_MEDIUM = [
    # Communication
    DownloadableGraph("wiki-Talk", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Talk.tar.gz",
                      80, 2394385, 5021410, False, "communication", "Wikipedia talk"),
    DownloadableGraph("wiki-topcats", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-topcats.tar.gz",
                      120, 1791489, 28511807, False, "web", "Wikipedia top categories"),
    # Citation networks
    DownloadableGraph("cit-Patents", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz",
                      262, 3774768, 16518948, False, "citation", "US Patent citations"),
    # Road networks
    DownloadableGraph("roadNet-PA", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz",
                      40, 1090920, 1541898, True, "road", "Pennsylvania roads"),
    DownloadableGraph("roadNet-CA", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz",
                      60, 1971281, 2766607, True, "road", "California roads"),
    DownloadableGraph("roadNet-TX", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-TX.tar.gz",
                      45, 1393383, 1921660, True, "road", "Texas roads"),
    # Social networks (medium)
    DownloadableGraph("soc-Epinions1", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Epinions1.tar.gz",
                      12, 75888, 508837, False, "social", "Epinions social"),
    # Commerce networks
    DownloadableGraph("amazon0302", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz",
                      15, 262111, 1234877, False, "commerce", "Amazon Mar 2003"),
    DownloadableGraph("amazon0312", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0312.tar.gz",
                      18, 400727, 3200440, False, "commerce", "Amazon Dec 2003"),
    DownloadableGraph("amazon0505", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0505.tar.gz",
                      22, 410236, 3356824, False, "commerce", "Amazon May 2005"),
    DownloadableGraph("amazon0601", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0601.tar.gz",
                      25, 403394, 3387388, False, "commerce", "Amazon Jun 2001"),
    # Web graphs
    DownloadableGraph("web-NotreDame", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-NotreDame.tar.gz",
                      15, 325729, 1497134, False, "web", "Notre Dame web"),
    DownloadableGraph("web-Stanford", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Stanford.tar.gz",
                      20, 281903, 2312497, False, "web", "Stanford web"),
    DownloadableGraph("web-BerkStan", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-BerkStan.tar.gz",
                      50, 685230, 7600595, False, "web", "Berkeley-Stanford web"),
    DownloadableGraph("web-Google", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz",
                      35, 916428, 5105039, False, "web", "Google web graph"),
    # Infrastructure
    DownloadableGraph("as-Skitter", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/as-Skitter.tar.gz",
                      90, 1696415, 11095298, True, "infrastructure", "Internet topology"),
    # Autonomous systems
    DownloadableGraph("Oregon-1", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/Oregon-1.tar.gz",
                      1, 11174, 23409, False, "infrastructure", "Oregon AS peering"),
    DownloadableGraph("Oregon-2", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/Oregon-2.tar.gz",
                      1, 11461, 32730, False, "infrastructure", "Oregon AS peering 2"),
    # DIMACS10 graphs (sparse)
    DownloadableGraph("delaunay_n17", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n17.tar.gz",
                      5, 131072, 393176, True, "mesh", "Delaunay triangulation n=17"),
    DownloadableGraph("delaunay_n18", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n18.tar.gz",
                      10, 262144, 786396, True, "mesh", "Delaunay triangulation n=18"),
    DownloadableGraph("delaunay_n19", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n19.tar.gz",
                      20, 524288, 1572823, True, "mesh", "Delaunay triangulation n=19"),
    DownloadableGraph("delaunay_n20", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n20.tar.gz",
                      40, 1048576, 3145686, True, "mesh", "Delaunay triangulation n=20"),
    DownloadableGraph("rgg_n_2_17_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_17_s0.tar.gz",
                      15, 131072, 728753, True, "mesh", "Random geometric graph n=17"),
    DownloadableGraph("rgg_n_2_18_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_18_s0.tar.gz",
                      30, 262144, 1457506, True, "mesh", "Random geometric graph n=18"),
    DownloadableGraph("rgg_n_2_19_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_19_s0.tar.gz",
                      60, 524288, 2915013, True, "mesh", "Random geometric graph n=19"),
    # Power-law and scale-free
    DownloadableGraph("preferentialAttachment", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/preferentialAttachment.tar.gz",
                      10, 100000, 499985, True, "synthetic", "Preferential attachment model"),
    DownloadableGraph("smallworld", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/smallworld.tar.gz",
                      10, 100000, 499998, True, "synthetic", "Small world model"),
    # Additional web graphs
    DownloadableGraph("cnr-2000", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/cnr-2000.tar.gz",
                      30, 325557, 3216152, False, "web", "Italian CNR web 2000"),
]

# Large graphs (200MB - 2GB) - ~40 graphs  
DOWNLOAD_GRAPHS_LARGE = [
    # Social networks (large)
    DownloadableGraph("soc-LiveJournal1", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz",
                      1024, 4847571, 68993773, False, "social", "LiveJournal social"),
    DownloadableGraph("com-Orkut", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Orkut.tar.gz",
                      800, 3072441, 117185083, True, "social", "Orkut social network"),
    DownloadableGraph("com-Youtube", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Youtube.tar.gz",
                      250, 1134890, 2987624, True, "social", "Youtube social network"),
    DownloadableGraph("com-Amazon", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Amazon.tar.gz",
                      220, 334863, 925872, True, "commerce", "Amazon product network"),
    DownloadableGraph("com-DBLP", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-DBLP.tar.gz",
                      200, 317080, 1049866, True, "collaboration", "DBLP collaboration"),
    # Collaboration
    DownloadableGraph("hollywood-2009", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz",
                      600, 1139905, 57515616, True, "collaboration", "Hollywood actors"),
    DownloadableGraph("dblp-2010", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/dblp-2010.tar.gz",
                      200, 326186, 1615400, True, "collaboration", "DBLP 2010"),
    # Web graphs (large)
    DownloadableGraph("in-2004", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/in-2004.tar.gz",
                      450, 1382908, 16917053, False, "web", "Indian web 2004"),
    DownloadableGraph("eu-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/eu-2005.tar.gz",
                      500, 862664, 19235140, False, "web", "European web 2005"),
    DownloadableGraph("uk-2002", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2002.tar.gz",
                      2500, 18520486, 298113762, False, "web", "UK web 2002"),
    DownloadableGraph("arabic-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/arabic-2005.tar.gz",
                      2200, 22744080, 639999458, False, "web", "Arabic web 2005"),
    DownloadableGraph("indochina-2004", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/indochina-2004.tar.gz",
                      600, 7414866, 194109311, False, "web", "Indochina web 2004"),
    DownloadableGraph("sk-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/sk-2005.tar.gz",
                      1100, 50636154, 1949412601, False, "web", "Slovakia web 2005"),
    # Road networks (large) - OSM graphs
    DownloadableGraph("europe-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/europe_osm.tar.gz",
                      1200, 50912018, 108109320, True, "road", "European OSM roads"),
    DownloadableGraph("asia-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/asia_osm.tar.gz",
                      600, 11950757, 25423206, True, "road", "Asian OSM roads"),
    DownloadableGraph("great-britain-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/great-britain_osm.tar.gz",
                      250, 7733822, 16313034, True, "road", "Great Britain OSM roads"),
    DownloadableGraph("germany-osm", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/germany_osm.tar.gz",
                      300, 11548845, 24738362, True, "road", "Germany OSM roads"),
    # DIMACS10 large meshes
    DownloadableGraph("delaunay_n21", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n21.tar.gz",
                      80, 2097152, 6291372, True, "mesh", "Delaunay triangulation n=21"),
    DownloadableGraph("delaunay_n22", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n22.tar.gz",
                      160, 4194304, 12582869, True, "mesh", "Delaunay triangulation n=22"),
    DownloadableGraph("delaunay_n23", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n23.tar.gz",
                      320, 8388608, 25165784, True, "mesh", "Delaunay triangulation n=23"),
    DownloadableGraph("delaunay_n24", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n24.tar.gz",
                      640, 16777216, 50331601, True, "mesh", "Delaunay triangulation n=24"),
    DownloadableGraph("rgg_n_2_20_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_20_s0.tar.gz",
                      120, 1048576, 5830030, True, "mesh", "Random geometric graph n=20"),
    DownloadableGraph("rgg_n_2_21_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_21_s0.tar.gz",
                      240, 2097152, 11660061, True, "mesh", "Random geometric graph n=21"),
    DownloadableGraph("rgg_n_2_22_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_22_s0.tar.gz",
                      480, 4194304, 23320130, True, "mesh", "Random geometric graph n=22"),
    DownloadableGraph("rgg_n_2_23_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_23_s0.tar.gz",
                      960, 8388608, 46640257, True, "mesh", "Random geometric graph n=23"),
    DownloadableGraph("rgg_n_2_24_s0", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/rgg_n_2_24_s0.tar.gz",
                      1920, 16777216, 93280513, True, "mesh", "Random geometric graph n=24"),
    # Clustering benchmarks
    DownloadableGraph("coPapersDBLP", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersDBLP.tar.gz",
                      400, 540486, 15245729, True, "collaboration", "DBLP co-author papers"),
    DownloadableGraph("coPapersCiteseer", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersCiteseer.tar.gz",
                      450, 434102, 16036720, True, "citation", "Citeseer co-papers"),
    DownloadableGraph("citationCiteseer", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/citationCiteseer.tar.gz",
                      350, 268495, 1156647, False, "citation", "Citeseer citations"),
    DownloadableGraph("coAuthorsDBLP", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coAuthorsDBLP.tar.gz",
                      200, 299067, 977676, True, "collaboration", "DBLP co-authors"),
    DownloadableGraph("coAuthorsCiteseer", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coAuthorsCiteseer.tar.gz",
                      160, 227320, 814134, True, "collaboration", "Citeseer co-authors"),
    # Wikipedia graphs
    DownloadableGraph("wiki-Vote", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Vote.tar.gz",
                      2, 7115, 103689, False, "social", "Wikipedia adminship votes"),
    # Kron graphs (synthetic power-law)
    DownloadableGraph("kron_g500-logn16", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn16.tar.gz",
                      200, 65536, 4912201, True, "synthetic", "Kronecker graph logn=16"),
    DownloadableGraph("kron_g500-logn17", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn17.tar.gz",
                      400, 131072, 10228360, True, "synthetic", "Kronecker graph logn=17"),
    DownloadableGraph("kron_g500-logn18", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn18.tar.gz",
                      800, 262144, 21165908, True, "synthetic", "Kronecker graph logn=18"),
    DownloadableGraph("kron_g500-logn19", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn19.tar.gz",
                      1600, 524288, 43561574, True, "synthetic", "Kronecker graph logn=19"),
    DownloadableGraph("kron_g500-logn20", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn20.tar.gz",
                      3200, 1048576, 89239674, True, "synthetic", "Kronecker graph logn=20"),
]

# Extra-large graphs (>2GB) - requires significant memory
DOWNLOAD_GRAPHS_XLARGE = [
    # Massive web graphs
    DownloadableGraph("uk-2005", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/uk-2005.tar.gz",
                      3200, 39459925, 936364282, False, "web", "UK web 2005"),
    DownloadableGraph("webbase-2001", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/webbase-2001.tar.gz",
                      8500, 118142155, 1019903190, False, "web", "WebBase 2001 crawl"),
    DownloadableGraph("it-2004", "https://suitesparse-collection-website.herokuapp.com/MM/LAW/it-2004.tar.gz",
                      3500, 41291594, 1150725436, False, "web", "Italian web 2004"),
    # Massive social
    DownloadableGraph("com-Friendster", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Friendster.tar.gz",
                      31000, 65608366, 1806067135, True, "social", "Friendster social network"),
    DownloadableGraph("twitter7", "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/twitter7.tar.gz",
                      12000, 41652230, 1468365182, False, "social", "Twitter follower network"),
    # Large meshes
    DownloadableGraph("kron_g500-logn21", "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz",
                      6400, 2097152, 182081864, True, "synthetic", "Kronecker graph logn=21"),
]

# Combined catalog by size
GRAPH_CATALOG = {
    "SMALL": {g.name: g for g in DOWNLOAD_GRAPHS_SMALL},
    "MEDIUM": {g.name: g for g in DOWNLOAD_GRAPHS_MEDIUM},
    "LARGE": {g.name: g for g in DOWNLOAD_GRAPHS_LARGE},
    "XLARGE": {g.name: g for g in DOWNLOAD_GRAPHS_XLARGE},
}

# All graphs combined
ALL_GRAPHS = (
    DOWNLOAD_GRAPHS_SMALL + 
    DOWNLOAD_GRAPHS_MEDIUM + 
    DOWNLOAD_GRAPHS_LARGE + 
    DOWNLOAD_GRAPHS_XLARGE
)


# =============================================================================
# Catalog Lookup Functions
# =============================================================================

def get_graph_info(name: str) -> Optional[DownloadableGraph]:
    """Look up graph info by name from catalog."""
    for size_cat in GRAPH_CATALOG.values():
        if name in size_cat:
            return size_cat[name]
    return None


def get_graphs_by_size(size: str) -> List[DownloadableGraph]:
    """Get list of graphs for a size category."""
    size = size.upper()
    if size == "ALL":
        return ALL_GRAPHS.copy()
    return list(GRAPH_CATALOG.get(size, {}).values())


def get_graphs_by_category(category: str) -> List[DownloadableGraph]:
    """Get all graphs of a specific category (social, web, road, etc.)."""
    return [g for g in ALL_GRAPHS if g.category == category]


def get_catalog_stats() -> Dict:
    """Get statistics about the graph catalog."""
    return {
        "SMALL": {
            "count": len(DOWNLOAD_GRAPHS_SMALL),
            "total_mb": sum(g.size_mb for g in DOWNLOAD_GRAPHS_SMALL),
            "categories": list(set(g.category for g in DOWNLOAD_GRAPHS_SMALL))
        },
        "MEDIUM": {
            "count": len(DOWNLOAD_GRAPHS_MEDIUM),
            "total_mb": sum(g.size_mb for g in DOWNLOAD_GRAPHS_MEDIUM),
            "categories": list(set(g.category for g in DOWNLOAD_GRAPHS_MEDIUM))
        },
        "LARGE": {
            "count": len(DOWNLOAD_GRAPHS_LARGE),
            "total_mb": sum(g.size_mb for g in DOWNLOAD_GRAPHS_LARGE),
            "categories": list(set(g.category for g in DOWNLOAD_GRAPHS_LARGE))
        },
        "XLARGE": {
            "count": len(DOWNLOAD_GRAPHS_XLARGE),
            "total_mb": sum(g.size_mb for g in DOWNLOAD_GRAPHS_XLARGE),
            "categories": list(set(g.category for g in DOWNLOAD_GRAPHS_XLARGE))
        },
    }


# =============================================================================
# Parallel Download Status Tracking
# =============================================================================

class DownloadStatus(Enum):
    """Status of a download."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class DownloadProgress:
    """Progress tracking for a single download."""
    graph: DownloadableGraph
    status: DownloadStatus = DownloadStatus.PENDING
    bytes_downloaded: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    error: str = ""
    result_path: Optional[Path] = None
    
    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time == 0:
            return 0.0
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time
    
    @property
    def speed_mbps(self) -> float:
        """Download speed in MB/s."""
        if self.elapsed <= 0 or self.bytes_downloaded <= 0:
            return 0.0
        return (self.bytes_downloaded / (1024 * 1024)) / self.elapsed
    
    @property
    def progress_pct(self) -> float:
        """Download progress percentage."""
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, (self.bytes_downloaded / self.total_bytes) * 100)


class ParallelDownloadManager:
    """
    Manages parallel downloads with live status reporting.
    
    Provides:
    - Parallel downloads to maximize bandwidth
    - Live progress display with per-download status
    - Blocks until all downloads complete
    - Detailed summary at the end
    """
    
    def __init__(self, max_workers: int = 4, show_progress: bool = True):
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.progress: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._stop_display = threading.Event()
    
    def _update_status(self, name: str, **kwargs):
        """Thread-safe status update."""
        with self._lock:
            if name in self.progress:
                for key, value in kwargs.items():
                    setattr(self.progress[name], key, value)
    
    def _download_with_progress(self, url: str, dest_path: Path, name: str) -> bool:
        """Download a file with progress tracking."""
        try:
            self._update_status(name, status=DownloadStatus.DOWNLOADING, start_time=time.time())
            
            # Open URL and get size
            req = urllib.request.Request(url, headers={'User-Agent': 'GraphBrew/1.0'})
            response = urllib.request.urlopen(req, timeout=300)
            total_size = int(response.headers.get('content-length', 0))
            self._update_status(name, total_bytes=total_size)
            
            # Download with progress tracking
            block_size = 8192
            bytes_downloaded = 0
            
            with open(dest_path, 'wb') as f:
                while True:
                    data = response.read(block_size)
                    if not data:
                        break
                    f.write(data)
                    bytes_downloaded += len(data)
                    self._update_status(name, bytes_downloaded=bytes_downloaded)
            
            return True
        except Exception as e:
            self._update_status(name, status=DownloadStatus.FAILED, error=str(e), end_time=time.time())
            return False
    
    def _download_single(self, graph: DownloadableGraph, dest_dir: Path, force: bool) -> Optional[Path]:
        """Download and extract a single graph with status tracking."""
        name = graph.name
        graph_dir = dest_dir / name
        
        # Check if already exists
        if graph_dir.exists() and not force:
            mtx_path = find_mtx_file(graph_dir)
            if mtx_path and mtx_path.exists():
                self._update_status(name, status=DownloadStatus.SKIPPED, 
                                  result_path=mtx_path, end_time=time.time())
                return mtx_path
        
        # Create directory
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        # Download
        tar_path = graph_dir / f"{name}.tar.gz"
        if not self._download_with_progress(graph.url, tar_path, name):
            return None
        
        # Extract
        self._update_status(name, status=DownloadStatus.EXTRACTING)
        if not extract_tarball(tar_path, graph_dir):
            self._update_status(name, status=DownloadStatus.FAILED, 
                              error="Extraction failed", end_time=time.time())
            return None
        
        # Clean up tarball
        try:
            tar_path.unlink()
        except:
            pass
        
        # Find extracted .mtx file
        mtx_path = find_mtx_file(graph_dir)
        if not mtx_path:
            self._update_status(name, status=DownloadStatus.FAILED, 
                              error="No .mtx file found", end_time=time.time())
            return None
        
        self._update_status(name, status=DownloadStatus.DONE, 
                          result_path=mtx_path, end_time=time.time())
        return mtx_path
    
    def _display_progress(self):
        """Display live progress for all downloads."""
        status_symbols = {
            DownloadStatus.PENDING: "⋯",
            DownloadStatus.DOWNLOADING: "↓",
            DownloadStatus.EXTRACTING: "⚙",
            DownloadStatus.DONE: "✓",
            DownloadStatus.SKIPPED: "○",
            DownloadStatus.FAILED: "✗",
        }
        status_colors = {
            DownloadStatus.PENDING: "",
            DownloadStatus.DOWNLOADING: "\033[94m",  # Blue
            DownloadStatus.EXTRACTING: "\033[93m",   # Yellow
            DownloadStatus.DONE: "\033[92m",         # Green
            DownloadStatus.SKIPPED: "\033[96m",      # Cyan
            DownloadStatus.FAILED: "\033[91m",       # Red
        }
        reset = "\033[0m"
        
        # Count completed
        with self._lock:
            items = list(self.progress.items())
        
        # Clear previous output and show current status
        done = sum(1 for _, p in items if p.status in (DownloadStatus.DONE, DownloadStatus.SKIPPED, DownloadStatus.FAILED))
        total = len(items)
        active = [p for _, p in items if p.status in (DownloadStatus.DOWNLOADING, DownloadStatus.EXTRACTING)]
        
        # Build status line
        lines = []
        lines.append(f"\r  Progress: {done}/{total} complete")
        
        # Show active downloads (up to 4)
        for prog in active[:4]:
            sym = status_symbols[prog.status]
            color = status_colors[prog.status]
            name = prog.graph.name[:20].ljust(20)
            if prog.status == DownloadStatus.DOWNLOADING:
                pct = prog.progress_pct
                speed = prog.speed_mbps
                line = f"    {color}{sym}{reset} {name} {pct:5.1f}% @ {speed:.1f} MB/s"
            else:
                line = f"    {color}{sym}{reset} {name} extracting..."
            lines.append(line)
        
        if len(active) > 4:
            lines.append(f"    ... and {len(active) - 4} more")
        
        # Print (with newlines for active items)
        output = lines[0]
        if active:
            output += " | " + " | ".join([l.strip() for l in lines[1:]])
        
        print(output.ljust(120), end='\r', flush=True)
    
    def download_all(
        self,
        graphs: List[DownloadableGraph],
        dest_dir: Path = None,
        force: bool = False,
    ) -> Tuple[List[Path], List[str]]:
        """
        Download all graphs in parallel, blocking until complete.
        
        Args:
            graphs: List of graphs to download
            dest_dir: Destination directory
            force: Force re-download
            
        Returns:
            Tuple of (successful_paths, failed_names)
        """
        if dest_dir is None:
            dest_dir = GRAPHS_DIR
        
        ensure_directories()
        
        # Initialize progress tracking
        for graph in graphs:
            self.progress[graph.name] = DownloadProgress(graph=graph)
        
        successful = []
        failed = []
        
        # Start progress display thread
        display_thread = None
        if self.show_progress and sys.stdout.isatty():
            def display_loop():
                while not self._stop_display.is_set():
                    self._display_progress()
                    time.sleep(0.5)
            display_thread = threading.Thread(target=display_loop, daemon=True)
            display_thread.start()
        
        try:
            # Download with thread pool
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._download_single, g, dest_dir, force): g.name
                    for g in graphs
                }
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        if result:
                            successful.append(result)
                        else:
                            failed.append(name)
                    except Exception as e:
                        self._update_status(name, status=DownloadStatus.FAILED, 
                                          error=str(e), end_time=time.time())
                        failed.append(name)
        finally:
            # Stop display thread
            self._stop_display.set()
            if display_thread:
                display_thread.join(timeout=1.0)
            
            # Clear progress line
            print(" " * 120, end='\r')
        
        return successful, failed
    
    def print_summary(self):
        """Print detailed summary of all downloads."""
        print("\n" + "─" * 70)
        print("  DOWNLOAD SUMMARY")
        print("─" * 70)
        
        done = []
        skipped = []
        failed = []
        
        for name, prog in self.progress.items():
            if prog.status == DownloadStatus.DONE:
                done.append(prog)
            elif prog.status == DownloadStatus.SKIPPED:
                skipped.append(prog)
            elif prog.status == DownloadStatus.FAILED:
                failed.append(prog)
        
        total_mb = sum(p.graph.size_mb for p in done)
        total_time = sum(p.elapsed for p in done)
        avg_speed = total_mb / total_time if total_time > 0 else 0
        
        print(f"\n  \033[92m✓ Downloaded:\033[0m {len(done)} graphs ({total_mb:.0f} MB)")
        if done and total_time > 0:
            print(f"    Average speed: {avg_speed:.1f} MB/s")
            print(f"    Total time: {total_time:.1f}s")
        
        if skipped:
            print(f"\n  \033[96m○ Skipped (already exist):\033[0m {len(skipped)} graphs")
            for prog in skipped[:5]:
                print(f"      {prog.graph.name}")
            if len(skipped) > 5:
                print(f"      ... and {len(skipped) - 5} more")
        
        if failed:
            print(f"\n  \033[91m✗ Failed:\033[0m {len(failed)} graphs")
            for prog in failed:
                print(f"      {prog.graph.name}: {prog.error}")
        
        print("─" * 70)


def download_graphs_parallel(
    size: str = "SMALL",
    graphs: List[str] = None,
    category: str = None,
    dest_dir: Path = None,
    max_workers: int = 4,
    force: bool = False,
    max_size_mb: int = None,
    max_count: int = None,
    show_progress: bool = True,
    wait_for_all: bool = True,
) -> Tuple[List[Path], List[str]]:
    """
    Download graphs in parallel for maximum bandwidth utilization.
    
    This function downloads multiple graphs concurrently to optimize internet
    bandwidth usage. It blocks until ALL downloads complete before returning,
    ensuring graphs are ready before running experiments.
    
    Args:
        size: Size category ("SMALL", "MEDIUM", "LARGE", "XLARGE", "ALL")
        graphs: Specific graph names to download (overrides size)
        category: Filter by category (e.g., "social", "web", "road")
        dest_dir: Destination directory
        max_workers: Number of parallel download threads (default: 4)
        force: Force re-download even if exists
        max_size_mb: Skip graphs larger than this
        max_count: Maximum number of graphs to download
        show_progress: Show live download progress
        wait_for_all: Always True - blocks until all downloads complete
        
    Returns:
        Tuple of (successful_paths, failed_names)
        
    Example:
        # Download small graphs in parallel
        paths, failed = download_graphs_parallel(size="SMALL", max_workers=4)
        
        # Experiment only starts after downloads complete
        for path in paths:
            run_experiment(path)
    """
    ensure_directories()
    
    if dest_dir is None:
        dest_dir = GRAPHS_DIR
    
    # Build download list
    if graphs:
        download_list = []
        for name in graphs:
            info = get_graph_info(name)
            if info:
                download_list.append(info)
            else:
                log.warning(f"Unknown graph: {name}")
    else:
        download_list = get_graphs_by_size(size)
    
    # Apply filters
    if category:
        download_list = [g for g in download_list if g.category == category]
    
    if max_size_mb:
        download_list = [g for g in download_list if g.size_mb <= max_size_mb]
    
    if max_count and len(download_list) > max_count:
        download_list = download_list[:max_count]
    
    if not download_list:
        log.warning("No graphs match the specified criteria")
        return [], []
    
    total_mb = sum(g.size_mb for g in download_list)
    
    print("\n" + "═" * 70)
    print("  PARALLEL GRAPH DOWNLOAD")
    print("═" * 70)
    print(f"  Graphs to download: {len(download_list)}")
    print(f"  Total size: ~{total_mb:,} MB")
    print(f"  Parallel workers: {max_workers}")
    print(f"  Destination: {dest_dir}")
    print("═" * 70 + "\n")
    
    # Create manager and download
    manager = ParallelDownloadManager(max_workers=max_workers, show_progress=show_progress)
    successful, failed = manager.download_all(download_list, dest_dir, force)
    
    # Print summary
    manager.print_summary()
    
    return successful, failed


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, dest_path: Path, timeout: int = 300) -> bool:
    """Download a file from URL to destination."""
    try:
        log.debug(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        log.error(f"Failed to download {url}: {e}")
        return False


def extract_tarball(tar_path: Path, dest_dir: Path) -> bool:
    """Extract tarball to destination directory."""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        return True
    except Exception as e:
        log.error(f"Failed to extract {tar_path}: {e}")
        return False


def find_mtx_file(directory: Path) -> Optional[Path]:
    """
    Find .mtx file in directory, handling various archive structures.
    SuiteSparse archives have inconsistent structures (some have subdirectories).
    """
    # Direct files
    mtx_files = list(directory.glob("*.mtx"))
    if mtx_files:
        return mtx_files[0]
    
    # Check subdirectories (common with SuiteSparse)
    for subdir in directory.iterdir():
        if subdir.is_dir():
            mtx_files = list(subdir.glob("*.mtx"))
            if mtx_files:
                # Move file up to main directory for consistency
                target = directory / mtx_files[0].name
                if not target.exists():
                    shutil.move(str(mtx_files[0]), str(target))
                return target
    
    # Deep search
    mtx_files = list(directory.rglob("*.mtx"))
    if mtx_files:
        target = directory / mtx_files[0].name
        if mtx_files[0] != target:
            shutil.move(str(mtx_files[0]), str(target))
        return target
    
    return None


def download_graph(
    graph: DownloadableGraph,
    dest_dir: Path = None,
    force: bool = False
) -> Optional[Path]:
    """
    Download and extract a single graph.
    
    Args:
        graph: DownloadableGraph object (or name string)
        dest_dir: Destination directory (default: GRAPHS_DIR)
        force: Force re-download even if exists
        
    Returns:
        Path to extracted graph file, or None if failed
    """
    # Handle name lookup
    if isinstance(graph, str):
        info = get_graph_info(graph)
        if info is None:
            log.error(f"Unknown graph: {graph}")
            return None
        graph = info
    
    if dest_dir is None:
        dest_dir = GRAPHS_DIR
    
    name = graph.name
    graph_dir = dest_dir / name
    
    # Check if already exists
    if graph_dir.exists() and not force:
        mtx_path = find_mtx_file(graph_dir)
        if mtx_path and mtx_path.exists():
            log.debug(f"Graph {name} already exists at {mtx_path}")
            return mtx_path
    
    # Create directory
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    tar_path = graph_dir / f"{name}.tar.gz"
    log.info(f"Downloading {name} ({graph.size_mb}MB)...")
    
    if not download_file(graph.url, tar_path):
        return None
    
    # Extract
    log.info(f"Extracting {name}...")
    if not extract_tarball(tar_path, graph_dir):
        return None
    
    # Clean up tarball
    try:
        tar_path.unlink()
    except:
        pass
    
    # Find extracted .mtx file
    mtx_path = find_mtx_file(graph_dir)
    if not mtx_path:
        log.error(f"No .mtx file found after extracting {name}")
        return None
    
    log.success(f"Downloaded {name} -> {mtx_path}")
    return mtx_path


def download_graphs(
    size: str = "SMALL",
    graphs: List[str] = None,
    category: str = None,
    dest_dir: Path = None,
    max_workers: int = 4,
    force: bool = False,
    max_size_mb: int = None,
    max_count: int = None,
) -> List[Path]:
    """
    Download multiple graphs.
    
    Args:
        size: Size category ("SMALL", "MEDIUM", "LARGE", "XLARGE", "ALL")
        graphs: Specific graph names to download (overrides size)
        category: Filter by category (e.g., "social", "web", "road")
        dest_dir: Destination directory
        max_workers: Parallel download threads
        force: Force re-download
        max_size_mb: Skip graphs larger than this
        max_count: Maximum number of graphs to download
        
    Returns:
        List of paths to downloaded graph files
    """
    ensure_directories()
    
    if dest_dir is None:
        dest_dir = GRAPHS_DIR
    
    # Build download list
    if graphs:
        # Specific graphs by name
        download_list = []
        for name in graphs:
            info = get_graph_info(name)
            if info:
                download_list.append(info)
            else:
                log.warning(f"Unknown graph: {name}")
    else:
        download_list = get_graphs_by_size(size)
    
    # Apply filters
    if category:
        download_list = [g for g in download_list if g.category == category]
    
    if max_size_mb:
        download_list = [g for g in download_list if g.size_mb <= max_size_mb]
    
    if max_count and len(download_list) > max_count:
        download_list = download_list[:max_count]
    
    if not download_list:
        log.warning("No graphs match the specified criteria")
        return []
    
    total_mb = sum(g.size_mb for g in download_list)
    log.info(f"Downloading {len(download_list)} graphs (~{total_mb}MB total)...")
    
    # Download with threading
    downloaded = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_graph, g, dest_dir, force): g.name
            for g in download_list
        }
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    downloaded.append(result)
            except Exception as e:
                log.error(f"Failed to download {name}: {e}")
    
    log.info(f"Downloaded {len(downloaded)}/{len(download_list)} graphs")
    return downloaded


# =============================================================================
# Listing Functions
# =============================================================================

def list_available_graphs(size: str = None) -> None:
    """Print available graphs in catalog."""
    print("\n" + "=" * 70)
    print("Available Graphs in Catalog")
    print("=" * 70)
    
    sizes = [size.upper()] if size else ["SMALL", "MEDIUM", "LARGE", "XLARGE"]
    
    for size_name in sizes:
        if size_name not in GRAPH_CATALOG:
            continue
        catalog = GRAPH_CATALOG[size_name]
        total_mb = sum(g.size_mb for g in catalog.values())
        print(f"\n{size_name} ({len(catalog)} graphs, {total_mb:,}MB total)")
        print("-" * 60)
        
        for name, info in sorted(catalog.items()):
            print(f"  {name:25s} {info.category:15s} {info.size_mb:5d}MB  {info.nodes:>10,} nodes")


def list_downloaded_graphs(dest_dir: Path = None) -> List[Dict]:
    """List downloaded graphs with their info."""
    if dest_dir is None:
        dest_dir = GRAPHS_DIR
    
    graphs = []
    if not dest_dir.exists():
        return graphs
    
    for graph_dir in dest_dir.iterdir():
        if not graph_dir.is_dir():
            continue
        
        mtx_path = find_mtx_file(graph_dir)
        if mtx_path and mtx_path.exists():
            size_mb = mtx_path.stat().st_size / (1024 * 1024)
            # Look up catalog info
            info = get_graph_info(graph_dir.name)
            graphs.append({
                "name": graph_dir.name,
                "path": str(mtx_path),
                "size_mb": size_mb,
                "nodes": info.nodes if info else 0,
                "edges": info.edges if info else 0,
                "category": info.category if info else "unknown",
            })
    
    return graphs


# =============================================================================
# Standalone CLI
# =============================================================================

def main():
    """CLI for graph downloading."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download benchmark graphs for GraphBrew",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available graphs
    python -m scripts.lib.download --list
    python -m scripts.lib.download --list --size MEDIUM
    
    # Download by size category
    python -m scripts.lib.download --size SMALL
    python -m scripts.lib.download --size ALL --max-count 10
    
    # Download specific graphs
    python -m scripts.lib.download --graph email-Enron web-Google
    
    # Download by category
    python -m scripts.lib.download --size ALL --category social
    
    # Check what's already downloaded
    python -m scripts.lib.download --list-downloaded
        """
    )
    
    parser.add_argument("--size", choices=["SMALL", "MEDIUM", "LARGE", "XLARGE", "ALL"],
                       default="SMALL", help="Size category to download")
    parser.add_argument("--graph", nargs="+", dest="graphs",
                       help="Specific graphs to download")
    parser.add_argument("--category", help="Filter by category (social, web, road, etc.)")
    parser.add_argument("--list", action="store_true", help="List available graphs")
    parser.add_argument("--list-downloaded", action="store_true",
                       help="List already downloaded graphs")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--workers", type=int, default=4,
                       help="Parallel download workers")
    parser.add_argument("--max-size", type=int, dest="max_size_mb",
                       help="Skip graphs larger than this (MB)")
    parser.add_argument("--max-count", type=int,
                       help="Maximum number of graphs to download")
    parser.add_argument("--dest", type=Path,
                       help="Destination directory (default: results/graphs/)")
    parser.add_argument("--stats", action="store_true", help="Show catalog statistics")
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_catalog_stats()
        print("\nGraph Catalog Statistics:")
        print("=" * 50)
        total_graphs = 0
        total_size = 0
        for size, info in stats.items():
            print(f"\n{size}:")
            print(f"  Graphs: {info['count']}")
            print(f"  Total Size: {info['total_mb']:,}MB")
            print(f"  Categories: {', '.join(info['categories'])}")
            total_graphs += info['count']
            total_size += info['total_mb']
        print(f"\nTotal: {total_graphs} graphs, {total_size:,}MB")
        return
    
    if args.list:
        list_available_graphs(args.size if args.size != "SMALL" else None)
        return
    
    if args.list_downloaded:
        graphs = list_downloaded_graphs(args.dest)
        print(f"\nDownloaded graphs ({len(graphs)}):")
        print("-" * 60)
        for g in sorted(graphs, key=lambda x: x['name']):
            print(f"  {g['name']:30s} {g['size_mb']:8.1f}MB  {g['category']:15s}")
        return
    
    # Download
    downloaded = download_graphs(
        size=args.size,
        graphs=args.graphs,
        category=args.category,
        dest_dir=args.dest,
        max_workers=args.workers,
        force=args.force,
        max_size_mb=args.max_size_mb,
        max_count=args.max_count,
    )
    
    print(f"\nSuccessfully downloaded {len(downloaded)} graphs")
    for path in downloaded:
        print(f"  {path}")


if __name__ == "__main__":
    main()
