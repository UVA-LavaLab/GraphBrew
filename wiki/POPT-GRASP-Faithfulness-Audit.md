# P-OPT + GRASP Source-Faithfulness Audit (No-Hallucination)

## Goal
Establish whether GraphBrew's GRASP and P-OPT implementations match canonical sources 1:1 for policy logic, and identify any deliberate deviations.

## Canonical Sources (Pinned)
1. GRASP paper:
   - Title: Domain-Specialized Cache Management for Graph Analytics
   - Venue: HPCA 2020
   - DOI: 10.1109/HPCA47549.2020.00028
    - Official repository found in `research/README.md` and `research/caching/grasp.md`:
       https://github.com/faldupriyank/grasp
    - GitHub API description: "Source code for the evaluated benchmarks and proposed cache management technique, GRASP, in  [Faldu et al., HPCA'20]."
    - License: Apache-2.0
   - Open-access copy (via Semantic Scholar metadata):
     https://www.pure.ed.ac.uk/ws/files/131011069/Domain_Specialized_Cache_FALDU_DOA06112019_AFV.pdf
   - ArXiv ID in metadata: 2001.09783

2. P-OPT paper:
   - Title: P-OPT: Practical Optimal Cache Replacement for Graph Analytics
   - Venue: HPCA 2021
   - DOI: 10.1109/HPCA51647.2021.00062
   - Official repository found in `research/README.md`, `research/caching/popt.md`, and the paper text under `research/POPT_HPCA21_CameraReady.txt`:
     https://github.com/CMUAbstract/POPT-CacheSim-HPCA21
   - License: MIT

## Artifact / GitHub Status
- Official public artifact/source repositories are available for both papers.
- The initial exact-title GitHub metadata search missed them; the links were already recorded in ignored `research/` notes and, for P-OPT, in the local camera-ready paper text.
- Current conclusion: use paper text + DOI source + official repositories as the canonical baseline for faithfulness checks.

## GraphBrew Implementation Anchors
### cache_sim (primary faithfulness reference)
- GRASP victim logic: bench/include/cache_sim/cache_sim.h (findVictimGRASP)
- P-OPT 3-phase victim logic: bench/include/cache_sim/cache_sim.h (findVictimPOPT)
- GRASP classification and dynamic rereference hooks:
  - bench/include/cache_sim/graph_cache_context.h (classifyGRASP, findNextRef)
- Current-vertex hint flow used by P-OPT:
  - bench/include/cache_sim/graph_sim.h
  - bench/src_sim/* (SIM_SET_VERTEX + makeOffsetMatrix calls)

### gem5 overlays (secondary faithfulness reference)
- GRASP policy overlay:
  - bench/include/gem5_sim/overlays/mem/cache/replacement_policies/grasp_rp.hh
  - bench/include/gem5_sim/overlays/mem/cache/replacement_policies/grasp_rp.cc
- P-OPT policy overlay:
  - bench/include/gem5_sim/overlays/mem/cache/replacement_policies/popt_rp.hh
  - bench/include/gem5_sim/overlays/mem/cache/replacement_policies/popt_rp.cc

## Audit Result: 2026-05-27
Pinned upstream commits used for this audit:
- GRASP `faldupriyank/grasp`: `6e3814430265fc4f2513c95ef131a6522bc9d389`
- P-OPT `CMUAbstract/POPT-CacheSim-HPCA21`: `53b5021846690d0f3445428c6380e877ecf7a10e`

Resolved implementation differences:
- GRASP upstream uses separate priority insertion and hit promotion values:
   `P_RRIP=1` on hot-line insert and `H_RRIP=0` on hot-line hit. GraphBrew had
   used `0` for both. cache_sim, gem5 overlays, and Sniper overlays now use
   hot insertion `1` and hot hit `0`.
- P-OPT upstream Phase 1 evicts non-irregular/non-property data before applying
   rereference distance. GraphBrew gem5/Sniper overlays had used a mixed-set
   far-rereference boost heuristic (`dist > 64`). That heuristic has been removed;
   gem5 and Sniper P-OPT paths now evict non-property data first, then apply max
   rereference distance and RRIP tiebreaking.

Validation completed:
- `python3 -m pytest -q scripts/test/test_popt_grasp_faithfulness_sources.py scripts/test/test_gem5_popt_matrix.py scripts/test/test_popt_charged_policy.py` -> 16 passed.
- `make sim-pr` rebuilt `bench/bin_sim/pr`.
- Tiny cache_sim smoke at `/tmp/graphbrew-popt-grasp-faithfulness-smoke` produced 5/5 ok rows for LRU, GRASP, POPT, ECG_DBG_ONLY, and ECG_POPT_PRIMARY.
- Refreshed email-Eu-core PR/BFS/BC cache-size screens after the fix:
   `/tmp/graphbrew-faithful-email-core-pr-cache-size-refresh`,
   `/tmp/graphbrew-faithful-email-core-bfs-cache-size-refresh`, and
   `/tmp/graphbrew-faithful-email-core-bc-cache-size-refresh` produced 63/63 ok
   cache_sim rows.
- Refreshed email-Eu-core PR-SPMV/TC cache-size screens after the fix:
   `/tmp/graphbrew-faithful-email-core-prspmv-cache-size-refresh` and
   `/tmp/graphbrew-faithful-email-core-tc-cache-size-refresh` produced 42/42 ok
   cache_sim rows.
- `USE_SDE=1 make -C bench/include/sniper_sim/snipersim/common -j1` rebuilt the touched Sniper cache-set objects.
- `scons build/X86/gem5.opt -j2` rebuilt gem5 X86 successfully after overlay refresh.

Resulting caution:
- PR/BFS/BC/PR-SPMV/TC local cache-size screens have been refreshed; other older
   local cache_sim screens recorded before this audit, especially CC/CC-SV/SSSP
   and broad all-kernel summaries, still need refresh before final GRASP/DBG-mode
   claims.

## 1:1 Faithfulness Checklist
1. Paper-to-code mapping table
   - For each paper pseudocode block, map to exact function and code region in GraphBrew.
   - Mark each mapping as Exact / Equivalent / Heuristic / Missing.

2. GRASP invariants (cache_sim first)
   - Confirm insertion and hit behavior for high/moderate/low reuse tiers.
   - Confirm victim selection is plain SRRIP max-RRPV aging.
   - Confirm behavior under DBG and non-DBG ordering matches paper expectations.

3. P-OPT invariants (cache_sim first)
   - Confirm non-graph-data-first eviction phase.
   - Confirm max next-reference-distance selection among graph data.
   - Confirm RRIP tie-break on equal max-distance lines only.
   - Confirm current-vertex signal path is active for PR/BFS/SSSP kernels.

4. gem5 parity against cache_sim
   - For tightly bounded synthetic runs, compare POPT and ECG_POPT_PRIMARY parity rows.
   - Ensure any heuristic branches are either disabled in faithful mode or explicitly reported.

5. Reproduce paper-style experiment surface
   - PR-focused matrix first (as paper-aligned baseline).
   - Use paper-like cache-size points and report miss-rate deltas against LRU/SRRIP.
   - Separate uncharged oracle POPT from charged-overhead POPT in claims.

## Immediate Execution Plan
1. Refresh cache_sim + gem5/Sniper parity checks for GRASP and P-OPT under the strict upstream-faithful behavior.
2. Generate fresh component-proof rows after the GRASP hot-insert correction.
3. Generate a source-faithfulness report CSV with columns:
   - source_block, graphbrew_file, graphbrew_function, status, evidence_run, notes
4. Promote only rows passing strict-faithful checks into paper-facing claims.

## Claiming Rules
- Do not claim "matches paper" unless status is Exact or Equivalent with explicit justification.
- Any heuristic or fallback path must be labeled non-faithful in figures/tables.
- Keep POPT_CHARGED (overhead-aware) separate from POPT (oracle) in conclusions.
