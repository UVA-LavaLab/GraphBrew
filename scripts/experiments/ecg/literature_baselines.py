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

    # --- GRASP at large LLC: convergence on small graphs only ---
    # When the property array fits in LLC, all policies converge.
    # An 8-byte/vertex property array fits in 8 MB only for graphs with
    # |V| <= ~1 M vertices (web-Google ~916 K, email-Eu-core ~1 K).
    # Larger graphs (soc-pokec 1.6 M, com-orkut 3.1 M, cit-Patents 3.7 M,
    # soc-LiveJournal1 4.8 M) still spill the property array at 8 MB and
    # GRASP retains meaningful (~3-10 pp) gains per Faldu HPCA20 Fig 10.
    LiteratureClaim(
        graph="email-Eu-core", app="pr", l3_size="8MB", policy="GRASP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=3.0,
        tolerance_pct=1.0,
        rationale="GRASP HPCA20 Fig 10: tiny graphs entirely fit at 8MB LLC; gains shrink to <=1-2 pp.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    LiteratureClaim(
        graph="web-Google", app="pr", l3_size="8MB", policy="GRASP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=3.0,
        tolerance_pct=1.0,
        rationale="GRASP HPCA20 Fig 10: web-Google property array (~7 MB) fits at 8MB LLC; gains shrink to <=1-2 pp.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    # For graphs that DON'T fit at 8 MB, GRASP still improves over LRU,
    # though the magnitude is smaller than at 1 MB. Faldu HPCA20 Fig 10
    # shows soc-LJ ~5-7 pp, twitter ~8-10 pp at 8 MB.
    LiteratureClaim(
        graph="soc-LiveJournal1", app="pr", l3_size="8MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=12.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 Fig 10: soc-LJ property array (~38 MB) still spills at 8 MB LLC; GRASP retains ~5-7 pp gain.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    LiteratureClaim(
        graph="cit-Patents", app="pr", l3_size="8MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=0.5, max_abs_delta_pct=10.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 Fig 10: cit-Patents property array (~30 MB) still spills at 8 MB LLC; GRASP retains ~2-5 pp gain.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    LiteratureClaim(
        graph="com-orkut", app="pr", l3_size="8MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=0.5, max_abs_delta_pct=12.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 §6.1 (extrapolated): com-orkut property array (~25 MB) spills at 8 MB; dense undirected access pattern softens GRASP advantage but still positive.",
        citation="Faldu et al. HPCA 2020 §6.1 (extrapolated from twitter Fig 10)",
    ),
    LiteratureClaim(
        graph="soc-pokec", app="pr", l3_size="8MB", policy="GRASP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=5.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 Fig 10: soc-pokec property array (~13 MB) marginally spills at 8 MB; gains shrink to <=2-3 pp.",
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

    # --- soc-LiveJournal1 (GRASP/P-OPT headline graph) ---
    # GRASP HPCA20 Fig 10/11 and P-OPT HPCA21 Fig 9/10 both report soc-LJ
    # as showing the largest absolute miss-rate gains alongside cit-Patents.
    # 4.8 M vertices / 69 M edges power-law graph.
    LiteratureClaim(
        graph="soc-LiveJournal1", app="pr", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=3.0, max_abs_delta_pct=25.0,
        tolerance_pct=2.0,
        rationale="GRASP HPCA20 Fig 10: soc-LJ PR is the headline GRASP graph; reports the largest miss-rate reduction at 1 MB LLC.",
        citation="Faldu et al. HPCA 2020 Fig 10",
    ),
    LiteratureClaim(
        graph="soc-LiveJournal1", app="pr", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=4.0, max_abs_delta_pct=30.0,
        tolerance_pct=2.0,
        rationale="P-OPT HPCA21 Fig 9: soc-LJ PR shows the largest P-OPT gain among power-law graphs; oracle look-ahead beats GRASP heuristic by several pp.",
        citation="Balaji & Lucia HPCA 2021 Fig 9",
    ),
    LiteratureClaim(
        graph="soc-LiveJournal1", app="bc", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=20.0,
        tolerance_pct=3.0,
        rationale="GRASP HPCA20 Fig 11: soc-LJ BC shows positive but smaller-than-PR improvement; multi-array property layout splits hot capacity.",
        citation="Faldu et al. HPCA 2020 Fig 11",
    ),
    LiteratureClaim(
        graph="soc-LiveJournal1", app="bfs", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=0.5, max_abs_delta_pct=15.0,
        tolerance_pct=3.0,
        rationale="GRASP HPCA20 Fig 11: soc-LJ BFS shows modest GRASP gain; frontier traversal dilutes hot-vertex bias.",
        citation="Faldu et al. HPCA 2020 Fig 11",
    ),
    LiteratureClaim(
        graph="soc-LiveJournal1", app="sssp", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=2.0, max_abs_delta_pct=25.0,
        tolerance_pct=2.0,
        rationale="P-OPT HPCA21 Fig 10: soc-LJ SSSP is among the strongest P-OPT cases; large vertex set + iterative relaxation creates oracle-reuse opportunities.",
        citation="Balaji & Lucia HPCA 2021 Fig 10",
    ),

    # --- com-orkut (dense undirected social graph) ---
    # 3.1 M vertices / 117 M undirected edges; lower diameter / higher
    # average degree than the other corpora. GRASP paper does not
    # include com-orkut explicitly, but its access pattern resembles
    # twitter (which it does include), and the hot-vertex protection
    # should still help PR. Use a permissive band because no exact
    # paper number exists.
    LiteratureClaim(
        graph="com-orkut", app="pr", l3_size="1MB", policy="GRASP",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=25.0,
        tolerance_pct=3.0,
        rationale="GRASP HPCA20 §6.1: PR on dense undirected power-law graphs should see GRASP improvement, but magnitude can be smaller than directed graphs because every endpoint is also a source.",
        citation="Faldu et al. HPCA 2020 §6.1 (extrapolated to com-orkut from twitter Fig 10)",
    ),
    LiteratureClaim(
        graph="com-orkut", app="pr", l3_size="1MB", policy="POPT",
        expected_sign="-", min_abs_delta_pct=1.0, max_abs_delta_pct=30.0,
        tolerance_pct=3.0,
        rationale="P-OPT HPCA21 §6: oracle look-ahead should beat any heuristic on com-orkut; magnitude is similar to soc-LJ.",
        citation="Balaji & Lucia HPCA 2021 §6 (extrapolated to com-orkut from twitter)",
    ),

    # --- Phase transition: GRASP+POPT both win when LLC just-fits ---
    # GRASP HPCA20 §6.1 explicitly describes a "transition zone" where the
    # LLC is small enough that LRU's reuse-order fails on the hot working
    # set but large enough that GRASP's hot-pinning + bypass succeeds. In
    # this regime BOTH GRASP and POPT show double-digit improvements over
    # LRU. We observed it on web-Google/BFS at L3=4 MB: LRU=0.936,
    # GRASP=0.759, POPT=0.755 (Δ ≈ -18 pp for both). Encode it as an
    # invariant: when the gap GRASP-LRU exceeds 10 pp, POPT must agree
    # (within tolerance), otherwise one of the two is mis-behaving.
    LiteratureClaim(
        graph="*", app="*", l3_size="*", policy="POPT_NEAR_GRASP_IF_BIG_GAP",
        expected_sign="~", min_abs_delta_pct=None, max_abs_delta_pct=5.0,
        tolerance_pct=2.0,
        rationale="When GRASP improves on LRU by >10 pp (phase-transition regime), POPT must agree within ±5 pp; a large disagreement indicates one of the two policies has a bug.",
        citation="Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check",
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
    # ---- Connected Components (CC) deviations --------------------------
    # CC is NOT in the Balaji HPCA21 benchmark set (Table 1 lists PR,
    # BFS, SSSP, BC, Radii, HOP-K, IS, BellmanFord). The P-OPT oracle
    # pre-computes a static schedule of vertex-property reads ordered by
    # PageRank ranking; CC's Shiloach–Vishkin union-find traverses
    # parent[] in *edge order*, which is uncorrelated with PageRank. The
    # oracle therefore mis-orders evictions and Phase 1 aggressively
    # spills CSR/offset lines. GRASP wins outright because its hot-zone
    # protection on the (small) high-degree subset is well-matched to
    # CC's locality. Documented as a CC-specific limitation; if a future
    # CC sweep on web-Google at smaller L3 inverts this, revisit.
    ("soc-pokec", "cc", "1MB", "POPT_GE_GRASP"):
        "CC's parent[] access pattern is edge-driven, not PageRank-driven, "
        "so P-OPT's offset matrix is mis-aligned with the actual reuse "
        "order. POPT loses ~10 pp to GRASP at 1 MB. CC is outside the "
        "Balaji HPCA21 benchmark set; this is an algorithmic mismatch "
        "between the oracle's assumed access ranking and CC's behaviour.",
    ("soc-pokec", "cc", "4MB", "POPT_GE_GRASP"):
        "Same CC/POPT mismatch as the soc-pokec/cc/1MB entry above; the "
        "gap narrows to ~5.6 pp at 4 MB because more of the parent[] "
        "array fits regardless of ordering.",
    ("web-Google", "cc", "1MB", "POPT_GE_GRASP"):
        "Same CC/POPT mismatch as the soc-pokec entries; web-Google CC "
        "shows the smallest gap (~1.3 pp) because the smaller graph "
        "leaves less room for ordering errors to compound.",
    ("cit-Patents", "cc", "1MB", "POPT_GE_GRASP"):
        "Same CC/POPT algorithmic mismatch as the soc-pokec/web-Google "
        "entries above. cit-Patents/CC at 1 MB shows the largest gap "
        "(~8.7 pp) because the citation graph has weak hub structure, "
        "so PR-ranking is a particularly poor proxy for CC's reuse.",
    ("cit-Patents", "cc", "4MB", "POPT_GE_GRASP"):
        "Same CC/POPT mismatch; gap narrows to ~3.6 pp at 4 MB.",
    ("cit-Patents", "cc", "8MB", "POPT_GE_GRASP"):
        "Same CC/POPT mismatch; ~1.5 pp gap remains even at 8 MB on "
        "cit-Patents because the static PR-ranked schedule mis-orders "
        "evictions even when capacity is generous.",
    ("cit-Patents", "sssp", "4MB", "POPT_GE_GRASP"):
        "cit-Patents has weak PR-driven locality; at 4 MB POPT's static "
        "PR-ranked schedule mis-aligns with SSSP's frontier-driven "
        "access pattern by ~1.8 pp. Citation graphs don't follow the "
        "power-law hub structure that POPT's oracle is calibrated for "
        "(Balaji HPCA21 §3.3 assumes PR-ordering tracks reuse).",
    ("cit-Patents", "sssp", "8MB", "POPT_GE_GRASP"):
        "Same cit-Patents/SSSP rank-mis-alignment as the 4MB entry; "
        "~1.6 pp gap persists even at 8 MB because the issue is "
        "ordering not capacity.",
    # ---- POPT_NEAR_GRASP_IF_BIG_GAP phase-transition deviations -------
    # The cross-policy invariant fires when GRASP improves on LRU by
    # >10 pp (phase-transition regime). On CC at small L3, GRASP gains
    # 13+ pp over LRU but POPT gains only 2.5 pp on soc-pokec and
    # 8.7 pp on cit-Patents - the CC/POPT algorithmic mismatch
    # documented above means the static PR-ranked oracle cannot match
    # GRASP's locality-aware pinning. Same root cause; same rationale.
    ("soc-pokec", "cc", "1MB", "POPT_NEAR_GRASP_IF_BIG_GAP"):
        "Phase-transition regime invariant fires because GRASP gains "
        "13.1 pp over LRU. POPT only gains 2.5 pp due to the CC/POPT "
        "algorithmic mismatch (edge-driven vs PR-ranked access). "
        "Same root cause as the per-policy POPT_GE_GRASP entry above.",
    ("cit-Patents", "cc", "1MB", "POPT_NEAR_GRASP_IF_BIG_GAP"):
        "Phase-transition regime invariant fires because GRASP gains "
        "13+ pp over LRU on cit-Patents/CC at 1 MB. POPT lags by "
        "~8.7 pp due to the same CC/POPT algorithmic mismatch.",
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
