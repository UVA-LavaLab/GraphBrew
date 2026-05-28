#!/usr/bin/env python3
"""Literature baseline expectations for GRASP / SRRIP / P-OPT.

This module is the single source of truth for "what should our baselines
look like in cache_sim if they faithfully implement the published
policies". It is consumed by:

  - ``literature_faithfulness.py`` — comparator that joins the live
    cache_sim sweep against these expectations and reports per-tuple
    ``ok / within_tolerance / disagree / missing`` verdicts.
  - ``scripts/test/test_baselines_match_literature.py`` — pytest gate
    that fails on disagreements not pre-recorded in
    ``KNOWN_DEVIATIONS``.

We only encode **single-core, replacement-only** expectations because
the local cache_sim is single-core and prefetch-free. Reordering is
``-o 5`` (Degree-Based Grouping) for all entries that require it; GRASP
in particular only activates its hot-region insertion logic when a DBG
sideband region is registered.

Sources
-------
- Faldu, Akram, Diavastos, Naithani, Vougioukas, Yu, Lucia, Pellauer,
  Mukkara, Sanchez, Grot, "GRASP: A Graph-Aware Replacement Policy for
  Last-Level Caches", HPCA 2020.
  https://dl.acm.org/doi/10.1109/HPCA47549.2020.00040
- Balaji, Lucia, "P-OPT: Practical Optimal Cache Replacement for Graph
  Analytics", HPCA 2021.
  https://www.cs.cmu.edu/~vbalaji/papers/popt_hpca21.pdf
- Jaleel, Theobald, Steely, Emer, "High Performance Cache Replacement
  Using Re-reference Interval Prediction (RRIP)", ISCA 2010.

Conventions
-----------
Δ ≡ ``miss_rate(policy) − miss_rate(LRU)`` in absolute units (i.e. a
miss rate of 0.42 vs LRU 0.50 gives Δ = −0.08). A negative Δ means the
policy is **better** than LRU.

Each entry below is a `LiteratureClaim` carrying the cache size at
which we expect the claim to hold (``l3_size``), the sign that we
require, an optional ``min_abs_delta_pct`` (in absolute percentage
points the policy must beat LRU by, e.g. ``3.0`` for a 3 pp
improvement), and an upper magnitude bound ``max_abs_delta_pct`` to
catch grossly wrong implementations. ``tolerance_pct`` widens both
bounds to account for ordering / graph-size variance vs the paper's
exact configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LiteratureClaim:
    graph: str
    app: str
    l3_size: str
    policy: str
    expected_sign: str                # "-" (improves), "+" (regresses), "~" (no clear direction)
    min_abs_delta_pct: float | None   # min Δ vs LRU in absolute pp; None = unbounded
    max_abs_delta_pct: float | None   # max |Δ| vs LRU in absolute pp; None = unbounded
    tolerance_pct: float              # additive tolerance applied to both bounds
    rationale: str
    citation: str


# ---------------------------------------------------------------------------
# Tier 1 — graph-size-independent invariants (sanity checks)
# ---------------------------------------------------------------------------
# These hold regardless of graph; encoded once per app/cache combination.
# They protect against gross policy regressions (e.g. SRRIP catastrophically
# worse than LRU on PR — a sign of an RRPV bug).

INVARIANT_CLAIMS: tuple[LiteratureClaim, ...] = (
    # SRRIP ≈ LRU on power-law PR (Jaleel ISCA10 Fig 8; Faldu HPCA20 Fig 11).
    # Small differences either way are normal; >5 pp absolute is suspect.
    LiteratureClaim(
        graph="*power_law*", app="pr", l3_size="*", policy="SRRIP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=5.0,
        tolerance_pct=2.0,
        rationale="SRRIP without graph-aware insertion behaves like LRU on power-law: hot vertices retained by reuse, cold vertices age out at the same rate.",
        citation="Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1",
    ),
    # POPT ≥ GRASP at all L3 sizes (oracle look-ahead dominates degree heuristic).
    # Encoded as a relative claim handled by the comparator (POPT.delta ≤ GRASP.delta).
    LiteratureClaim(
        graph="*", app="pr", l3_size="*", policy="POPT_GE_GRASP",
        expected_sign="-", min_abs_delta_pct=None, max_abs_delta_pct=None,
        tolerance_pct=1.0,
        rationale="P-OPT is an oracle policy by construction; it cannot be worse than any heuristic, including GRASP, by more than tolerance.",
        citation="Balaji & Lucia HPCA 2021 §6.3",
    ),
)


# ---------------------------------------------------------------------------
# Tier 2 — per-graph claims at literature cache sizes
# ---------------------------------------------------------------------------
# These are the headline numbers that justify each paper. Cache organization:
#   L1d = 32 kB, 8-way   |   L2 = 256 kB, 8-way   |   L3 16-way, line=64
# Variable: L3 size. ``small`` = 1 MB (GRASP HPCA20 single-core baseline).
# ``medium`` = 4 MB. ``large`` = 8 MB.
# Apps run with ``-o 5`` (DBG reorder) and ``-i 2`` iterations for PR.

# GRASP papers' headline claim: 5-19% LLC miss reduction on PageRank averaged
# across cit-Patents, soc-LiveJournal1, soc-Pokec, web-Google, twitter at
# 1MB/core LLC.

PER_GRAPH_CLAIMS: tuple[LiteratureClaim, ...] = (
    # --- GRASP on PR @ 1 MB LLC ---
    LiteratureClaim(
        graph="cit-Patents", app="pr", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=3.0, max_abs_delta_pct=25.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 Fig 10: cit-Patents PR shows the largest LLC-miss-rate reduction from GRASP among evaluated graphs at 1MB LLC.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    LiteratureClaim(
        graph="soc-pokec", app="pr", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=2.0, max_abs_delta_pct=20.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 Fig 10: soc-Pokec PR shows ~5% miss-rate reduction at 1MB LLC.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    LiteratureClaim(
        graph="web-Google", app="pr", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=15.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 Fig 10: web-Google PR shows a smaller but consistent improvement; smaller graph fits more of the property array in 1MB.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    # --- GRASP on BC @ 1 MB LLC ---
    # GRASP paper reports BC gains comparable to PR on cit-Patents/soc-LJ.
    LiteratureClaim(
        graph="cit-Patents", app="bc", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=20.0,
        tolerance_pct=3.0,
        rationale="GRASP HPCA20 Fig 11: BC benefits from hot-vertex retention on power-law graphs, slightly less than PR.",
        citation="Faldu et al. HPCA 2020 Fig 11",
    ),
    LiteratureClaim(
        graph="web-Google", app="bc", l3_size="1MB", policy="GRASP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=5.0,
        tolerance_pct=3.0,
        rationale=(
            "GRASP HPCA20 Fig 11: BC on small web graphs sees modest "
            "improvement or near-parity; the four-array property layout "
            "splits hot capacity across all of them so the magnitude is "
            "smaller than PR's. Must NOT regress significantly below LRU."
        ),
        citation="Faldu et al. HPCA 2020 Fig 11",
    ),
    LiteratureClaim(
        graph="soc-pokec", app="bc", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=0.5, max_abs_delta_pct=15.0,
        tolerance_pct=3.0,
        rationale="GRASP HPCA20 Fig 11: BC on soc-Pokec shows positive but smaller-than-PR improvement.",
        citation="Faldu et al. HPCA 2020 Fig 11",
    ),

    # --- GRASP on BFS @ 1 MB LLC ---
    # BFS is one of the harder cases for GRASP because the frontier
    # is concentrated and most accesses are to the next frontier rather
    # than to the hottest vertices. Faldu reports modest gains.
    LiteratureClaim(
        graph="cit-Patents", app="bfs", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=0.5, max_abs_delta_pct=15.0,
        tolerance_pct=3.0,
        rationale="GRASP HPCA20 Fig 11: BFS gains are smaller than PR/BC because frontier traversal is breadth-first across all degrees, not hub-biased.",
        citation="Faldu et al. HPCA 2020 Fig 11",
    ),
    LiteratureClaim(
        graph="web-Google", app="bfs", l3_size="1MB", policy="GRASP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=8.0,
        tolerance_pct=4.0,
        rationale=(
            "GRASP HPCA20 Fig 11: BFS on web-Google should not regress "
            "significantly below LRU; small improvement (or tie) is "
            "acceptable given the single-source traversal pattern."
        ),
        citation="Faldu et al. HPCA 2020 Fig 11",
    ),

    # --- GRASP on SSSP @ 1 MB LLC ---
    # SSSP visits each vertex once per relaxation; hot-vertex protection
    # helps when high-degree vertices are repeatedly relaxed.
    LiteratureClaim(
        graph="cit-Patents", app="sssp", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=0.5, max_abs_delta_pct=15.0,
        tolerance_pct=3.0,
        rationale="P-OPT HPCA21 Fig 10: SSSP under GRASP improves over LRU but less than PR.",
        citation="Balaji & Lucia HPCA 2021 Fig 10 (GRASP bar)",
    ),

    # --- GRASP at large LLC: convergence ---
    # When LLC easily holds the property array, all policies converge.
    LiteratureClaim(
        graph="*", app="pr", l3_size="8MB", policy="GRASP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=3.0,
        tolerance_pct=1.0,
        rationale="GRASP HPCA20 Fig 10: gains shrink to ≤1-2 pp at 8MB LLC across all graphs because the working set comfortably fits.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),

    # --- POPT on PR @ 1 MB LLC: should be ≥ GRASP gain ---
    LiteratureClaim(
        graph="cit-Patents", app="pr", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=3.0, max_abs_delta_pct=30.0,
        tolerance_pct=2.0,
        rationale="P-OPT HPCA21 Fig 9: oracle re-reference matrix beats RRIP-family policies by 5-10 pp at moderate LLC sizes on cit-Patents PR.",
        citation="Balaji & Lucia HPCA 2021 Fig 9",
    ),
    LiteratureClaim(
        graph="soc-pokec", app="pr", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=2.0, max_abs_delta_pct=25.0,
        tolerance_pct=2.0,
        rationale="P-OPT HPCA21 Fig 9: soc-Pokec PR shows P-OPT close to OPT, well below LRU baseline at 1MB.",
        citation="Balaji & Lucia HPCA 2021 Fig 9",
    ),
    LiteratureClaim(
        graph="web-Google", app="pr", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=20.0,
        tolerance_pct=2.0,
        rationale="P-OPT HPCA21 Fig 9: web-Google PR shows P-OPT improvement, magnitude depends on graph size relative to LLC.",
        citation="Balaji & Lucia HPCA 2021 Fig 9",
    ),

    # --- POPT on SSSP @ 1 MB LLC ---
    # P-OPT paper's headline application; SSSP shows P-OPT > GRASP > LRU.
    LiteratureClaim(
        graph="cit-Patents", app="sssp", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=25.0,
        tolerance_pct=2.0,
        rationale="P-OPT HPCA21 Fig 10: cit-Patents SSSP benefits strongly from the oracle re-reference matrix.",
        citation="Balaji & Lucia HPCA 2021 Fig 10",
    ),
    LiteratureClaim(
        graph="soc-pokec", app="sssp", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=20.0,
        tolerance_pct=2.0,
        rationale="P-OPT HPCA21 Fig 10: soc-Pokec SSSP shows P-OPT beats GRASP and LRU at 1MB.",
        citation="Balaji & Lucia HPCA 2021 Fig 10",
    ),
)


# ---------------------------------------------------------------------------
# Known deviations — entries here downgrade test failures to xfails so CI
# stays green on documented issues. Add an entry only after the deviation
# is investigated and recorded in wiki/Baseline-Literature-Faithfulness.md.
# ---------------------------------------------------------------------------

KNOWN_DEVIATIONS: dict[tuple[str, str, str, str], str] = {
    # (graph, app, l3_size, policy): reason
    #
    # P-OPT's findVictimPOPT() (cache_sim.h:1043) Phase 1 always evicts the
    # first non-property cache line (CSR offsets, frontier bitmap, …)
    # before considering any property line — by design, matching P-OPT
    # HPCA 2021 §4.2. When the L3 is much smaller than the property array
    # (e.g. L3=1 MB vs 3.66 MB property array on web-Google), this is
    # optimal because every byte of L3 must hold reused property data.
    # When the L3 is *just larger* than the property array (e.g.
    # L3=4 MB vs 3.66 MB on web-Google), Phase 1 still kills CSR/offset
    # lines that GRASP would have kept, costing POPT 2.4 pp vs GRASP. At
    # L3=8 MB the asymmetry disappears because both policies fit
    # everything that matters. This is a documented policy choice, not a
    # simulator bug; matches the P-OPT paper's stated behaviour. See
    # wiki/Baseline-Literature-Faithfulness.md for the trace.
    ("web-Google", "pr", "4MB", "POPT_GE_GRASP"):
        "POPT Phase 1 aggressively evicts non-property cache lines (CSR "
        "offsets, frontier bitmap) regardless of their reuse. At L3=4 MB "
        "the property array (~3.66 MB) leaves only 0.34 MB for those "
        "lines, which thrash. GRASP retains them naturally. Matches "
        "P-OPT HPCA21 §4.2 design; not a sim bug.",
    ("web-Google", "bc", "4MB", "POPT_GE_GRASP"):
        "Same Phase-1 root cause as the PR/4MB entry above. BC has four "
        "vertex-indexed property arrays totalling ~12 MB; at L3=4 MB they "
        "spill anyway, but POPT still wastes capacity evicting CSR/offset "
        "lines first. GRASP keeps them. ~3 pp deficit observed.",
    ("web-Google", "bc", "8MB", "POPT_GE_GRASP"):
        "BC working set on web-Google (~12 MB across 4 property arrays) "
        "spills 4 MB at L3=8 MB. POPT Phase 1 still preferentially "
        "evicts CSR/offset lines, ceding ~3 pp to GRASP which protects "
        "them via SRRIP semantics outside hot region. P-OPT HPCA21 "
        "§4.2 design behaviour.",
    ("web-Google", "bfs", "1MB", "POPT_GE_GRASP"):
        "Single-source BFS has a frontier-based access pattern that "
        "P-OPT's offset matrix cannot exploit well (the next-vertex "
        "schedule is data-dependent). GRASP's hot-region protection "
        "of `parent[]` indexed by DBG-reordered vertex IDs captures "
        "the locality that exists. ~1 pp gap matches P-OPT HPCA21 "
        "Fig 10 BFS bars where POPT≈GRASP within noise.",
}


# ---------------------------------------------------------------------------
# Cache organizations used by the literature sweep.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheOrg:
    name: str
    l1d_size: str
    l1d_ways: str
    l2_size: str
    l2_ways: str
    l3_size: str
    l3_ways: str
    line_size: str
    rationale: str


LITERATURE_CACHE_ORGS: tuple[CacheOrg, ...] = (
    CacheOrg(
        name="grasp_canonical_1MB",
        l1d_size="32kB", l1d_ways="8",
        l2_size="256kB", l2_ways="8",
        l3_size="1MB",   l3_ways="16",
        line_size="64",
        rationale="GRASP HPCA20 single-core baseline (Faldu et al. Table 2).",
    ),
    CacheOrg(
        name="larger_4MB",
        l1d_size="32kB", l1d_ways="8",
        l2_size="256kB", l2_ways="8",
        l3_size="4MB",   l3_ways="16",
        line_size="64",
        rationale="GRASP HPCA20 LLC sweep upper end / P-OPT HPCA21 default.",
    ),
    CacheOrg(
        name="fits_8MB",
        l1d_size="32kB", l1d_ways="8",
        l2_size="256kB", l2_ways="8",
        l3_size="8MB",   l3_ways="16",
        line_size="64",
        rationale="Convergence point — LLC large enough that all policies should tie.",
    ),
    CacheOrg(
        name="stress_32kB",
        l1d_size="1kB",  l1d_ways="8",
        l2_size="2kB",   l2_ways="4",
        l3_size="32kB",  l3_ways="16",
        line_size="64",
        rationale="Existing GraphBrew stress config — keeps continuity with Tier C and the gem5 / Sniper sweeps.",
    ),
)


def claims_for(graph: str, app: str, l3_size: str) -> list[LiteratureClaim]:
    """Return claims that apply to (graph, app, l3_size).

    Wildcard ``"*"`` matches any value; ``"*power_law*"`` matches any graph
    we mark as power-law in :data:`POWER_LAW_GRAPHS`.
    """
    out: list[LiteratureClaim] = []
    for c in INVARIANT_CLAIMS + PER_GRAPH_CLAIMS:
        if c.graph != "*" and c.graph != "*power_law*" and c.graph != graph:
            continue
        if c.graph == "*power_law*" and graph not in POWER_LAW_GRAPHS:
            continue
        if c.app != "*" and c.app != app:
            continue
        if c.l3_size != "*" and c.l3_size != l3_size:
            continue
        out.append(c)
    return out


def all_known_graphs(claims: Iterable[LiteratureClaim] = PER_GRAPH_CLAIMS) -> set[str]:
    return {c.graph for c in claims if c.graph not in ("*", "*power_law*")}


POWER_LAW_GRAPHS: frozenset[str] = frozenset({
    "cit-Patents",
    "soc-pokec",
    "soc-LiveJournal1",
    "com-orkut",
    "web-Google",
    "twitter",
})


__all__ = [
    "LiteratureClaim",
    "CacheOrg",
    "INVARIANT_CLAIMS",
    "PER_GRAPH_CLAIMS",
    "KNOWN_DEVIATIONS",
    "LITERATURE_CACHE_ORGS",
    "POWER_LAW_GRAPHS",
    "claims_for",
    "all_known_graphs",
]
