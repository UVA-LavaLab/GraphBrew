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
- Direct upstream trace replay found two additional GRASP source-faithfulness
   details in the official trace simulator: the trace-header `propertyA/B-f`
   percentage, not a hard-coded 10%, defines the high-reuse boundary; and the
   simulator has no valid-line fast path, so GRASP victim selection applies even
   during cold fill. cache_sim now carries per-region GRASP `f` and uses the
   upstream cold-fill victim behavior for GRASP and `ECG_DBG_ONLY` parity.
- Follow-up live-benchmark audit found that direct trace replay was correct, but
   GraphBrew's live `registerPropertyArray()` fallback still used `f=10`. The
   upstream app instrumentation defaults `frontier_frac=50`, so PR/BC/Radii
   traces carry `propertyA/B-f=50`; BellmanFord-style traces can use
   `propertyA-f=100`. cache_sim, gem5, and Sniper live sideband contexts now
   default to `f=50` when no explicit trace/header value is available.
- Follow-up region-scope audit found that DBG ordering itself is correct (`-o 5`),
   but live GraphBrew had used every registered property array as a GRASP region.
   Upstream GRASP protects named `propertyA/B` arrays only. GraphCacheContext and
   gem5/Sniper sidebands now carry an explicit `grasp_region` flag: PR/PR-SPMV
   protect the contribution/source-value array, and the current BC default
   protects the backward dependency/delta array.
- P-OPT upstream Phase 1 evicts non-irregular/non-property data before applying
   rereference distance. GraphBrew gem5/Sniper overlays had used a mixed-set
   far-rereference boost heuristic (`dist > 64`). That heuristic has been removed;
   gem5 and Sniper P-OPT paths now evict non-property data first, then apply max
   rereference distance and RRIP tiebreaking.
- Direct source comparison against the official P-OPT artifact found a second,
   subtler divergence: GraphBrew's rereference matrix used the opposite MSB
   polarity from `CMUAbstract/POPT-CacheSim-HPCA21`. The official artifact uses
   `MSB=0` for "referenced in this epoch" and `MSB=1` for "not referenced; low
   bits encode distance to next reference." GraphBrew's `makeOffsetMatrix()` and
   cache_sim/gem5/Sniper decoders have been aligned to that official convention.

Validation completed:
- Direct GRASP upstream artifact comparison:
   `/tmp/graphbrew-upstream-policy-compare-grasp-all/comparison.csv` compares the
   official `faldupriyank/grasp` trace simulator against GraphBrew trace replay on
   all five bundled web-Google LLC traces at 1MB LLC. Result: 10/10 LRU/GRASP
   rows have identical total accesses and total misses. The cloned upstream needed
   one portability patch (`return 0` in `common.h`) because modern GCC emits a
   trap for the original non-void function that fell off the end; this does not
   change replacement logic.
- P-OPT original runtime smoke is partially runnable with local compatibility
   patches. The official `download_pin.py` installs Pin 2.14 under the upstream
   checkout. The current host lacks `g++-4.9`, so a local shim to system `g++`
   plus Pin-header ABI bypass and `_GLIBCXX_USE_CXX11_ABI=0` were needed to build
   the official pintools. Dynamic target binaries still abort under Pin 2.14 on
   this Ubuntu 24.04 loader (`unknown section type 19`, `.relr.dyn` in
   `/lib64/ld-linux-x86-64.so.2`). Rebuilding the official PR app as a fully
   static binary avoids that loader failure and reaches P-OPT LLC stats before a
   Pin teardown segfault. Smoke outputs:
   `/tmp/graphbrew-popt-pin-pr-g6-static.out` and
   `/tmp/graphbrew-popt-pin-email-popt-8b.out`. Treat this as activation evidence
   for the original P-OPT pintool on this host, not as final clean parity.
- Modern Pin attempt: Pin 3.30 (`pin-modern`) was installed locally and the
   official P-OPT cache pintools were patched to build with Pin's modern
   `source/tools/Config` rules and system `g++`. Required source compatibility
   change: add explicit `using namespace std;` in `cache_pinsim.cpp` because Pin
   3's libc++ headers no longer leak `string`, `cerr`, and `endl` globally.
   Result: all four pintools (`lru`, `drrip`, `popt-8b`, `opt-ideal`) build as
   `cache_pinsim-modern.so`, and the dynamic official PR app on `email-Eu-core`
   reaches LLC stats under modern Pin. Remaining blocker: the run still exits
   abnormally after stats (`signal 11` for LRU and Pin's `Unexpected memory
   deallocation request of aligned memory` for POPT). Logs:
   `/tmp/graphbrew-popt-modernpin-email-lru.out`,
   `/tmp/graphbrew-popt-modernpin-email-popt-8b.out`, and follow-up variants
   under `/tmp/graphbrew-popt-modernpin-*`. Treat modern Pin as a successful
   build/stats-activation patch, but not yet a clean end-to-end artifact run.
- P-OPT source/runtime alignment after the official encoding fix:
   `/tmp/graphbrew-popt-official-encoding-proof-email` produced 54/54 ok
   file-backed proof rows and a 42-row gate report with 33 pass / 9 fail. The
   important parity gates pass: `ECG_POPT_PRIMARY` matches pure `POPT` within the
   configured parity tolerance on PR/BFS/SSSP, and `ECG_DBG_ONLY` matches GRASP.
   Key corrected rows on `email-Eu-core`, 4kB LLC: PR `POPT_only=5448` and
   `ECG_POPT_primary=5466` (0.33% worse, parity pass but strict best-ECG gate
   fail), BFS `POPT_only=ECG_POPT_primary=1186`, SSSP `POPT_only=ECG_POPT_primary=63`.
   The no-full-POPT adaptive selector still fails PR/BFS, so dynamic POPT-primary
   remains the stable ECG oracle baseline.
- Original artifact versus GraphBrew runtime counters should not be compared as
   exact miss-count parity: the original Pin pintool uses full application memory
   instrumentation, hashed S-NUCA set selection, a 24MB/16-way LLC, and app-level
   data-type registration, while GraphBrew cache_sim is a controlled graph-kernel
   simulator. The useful comparison is algorithmic/source alignment plus
   qualitative matched-geometry behavior. At 24MB/16-way LLC on `email-Eu-core`
   PR, GraphBrew rows in
   `/tmp/graphbrew-popt-geometry-match-email-pr-official-encoding` show LRU=2135
   misses and POPT/`ECG_POPT_PRIMARY`=2134 misses; original modern-Pin smoke rows
   report roughly 2325--2326 LLC misses for LRU/POPT and `Start Way = 1` for
   POPT. Both stacks show this tiny graph is saturated at the original large LLC
   geometry.
- gem5/Sniper matched-hierarchy check: gem5 cannot instantiate the exact original
   24MB/16-way/64B LLC because that implies 24,576 sets and gem5 requires a
   power-of-two set count. A bracket run at 16MB and 32MB LLC was completed at
   `/tmp/graphbrew-gem5-popt-pin-geometry-email-pr-bracket` after rebuilding X86
   gem5 with the official P-OPT encoding fix. All 12 gem5 rows are `ok`. For the
   later stats section, LRU has L2/L3 misses 2449/271, POPT has 2440/271, and
   `ECG_POPT_PRIMARY` has 2429/269 at both 16MB and 32MB; POPT and ECG-P-OPT are
   effectively tied with LRU at this large-cache point, with tiny improvements.
   A safe Sniper `pr_kernel_smoke` at the exact 24MB hierarchy completed at
   `/tmp/graphbrew-sniper-popt-pin-geometry-kernel-smoke`: LRU LLC misses=93,
   POPT LLC misses=92. The full same-graph Sniper wrapper was not launched
   because prior runs showed the same-graph SDE/SIFT path can consume ~50 GiB RSS.
- DBG-ordering and SRRIP control check for GRASP: `-o 5` is DBG in GraphBrew.
   Corrected `f=50` plus explicit-region rows at
   `/tmp/graphbrew-grasp-region-scope-recheck` compare original (`-o 0`) versus
   DBG (`-o 5`) ordering on the two available `.sg` graphs. On `email-Eu-core`
   PR, original-order GRASP is slightly better than SRRIP (11,784 versus 11,872
   misses) but worse than LRU (11,041); DBG GRASP beats LRU (6,202 versus 6,580)
   but trails SRRIP (5,974). On `cit-Patents`, DBG barely changes PR misses,
   SRRIP beats LRU by about 9%, and GRASP remains about 20--21% worse than LRU.
   Conclusion: the old `email-Eu-core` PR loss was a live-path `f`/region-model
   bug; the remaining `cit-Patents` PR loss is a negative/control workload, not
   by itself a source-faithfulness mismatch.
- BC reorder-vs-replacement decomposition on `email-Eu-core`:
   `/tmp/graphbrew-grasp-region-scope-recheck/email-bc` compares original and
   DBG ordering with the upstream-like default BC GRASP region. LRU improves
   from 39,557 misses to 31,211 with DBG alone (21.1% fewer misses), SRRIP
   improves further to 28,005, and GRASP under DBG is 30,643 misses. GRASP still
   helps original ordering (32,732 misses versus SRRIP 37,895), but under DBG
   this live BC path is mostly a layout/SRRIP win. `ECG_DBG_ONLY` remains a useful
   variant at 27,676 misses under DBG, but should not be cited as strict GRASP
   parity for this path until the remaining live-context differences are isolated.
- Focused GraphBrew parity after the GRASP trace-faithfulness fixes:
   `/tmp/graphbrew-grasp-ecg-dbg-parity-after-upstream` produced 6/6 ok rows;
   PR/BFS/SSSP all have `GRASP_DBG_only == ECG_DBG_only` on the file-backed
   `email-Eu-core` proof graph.
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
- Refreshed ECG proof matrix with embedded/combined rows:
   `/tmp/graphbrew-ecg-validation-proof-email-core` produced 42/42 ok file-backed
   proof rows. The gate report has 30 verdict rows and 27 pass / 3 fail:
   GRASP/P-OPT parity, best-available ECG replacement, and PFX gates pass;
   PR/BFS DBG-primary hybrid value and PR embedded-quality gates fail under this
   local 4kB `email-Eu-core` proof. Synthetic `-g 12` proof rows were observed to
   vary across invocations, so use file-backed proof gates for quantitative
   interpretation.
- Embedded variant probe `/tmp/graphbrew-ecg-validation-proof-email-core-epoch`
   added `ECG_EPOCH_EMBEDDED`. It improves PR static embedded from 9,332 to 6,793
   misses and BFS static embedded from 1,270 to 1,212 misses, but still trails
   dynamic POPT/`ECG_POPT_PRIMARY` on PR/BFS.
- POPT-tie variant probe `/tmp/graphbrew-ecg-validation-proof-email-core-popt-tie`
   added `ECG_POPT_TIE` and produced 48/48 ok proof rows plus a 36-row gate
   report with 31 pass / 5 fail. It matches epoch-embedded on PR/BFS, showing
   that SRRIP candidate filtering is a major source of the remaining gap to
   dynamic POPT.
- Adaptive selector probe `/tmp/graphbrew-ecg-validation-proof-email-core-adaptive`
   produced 54/54 ok proof rows plus a 42-row gate report with 35 pass / 7 fail.
   `ECG_ADAPTIVE_ORACLE` passes by choosing `ECG_POPT_PRIMARY` on PR/BFS, while
   `ECG_ADAPTIVE_NO_FULL_POPT` still fails PR and slightly fails BFS versus POPT.
   This supports a POPT-primary low-overhead delivery direction rather than a
   claim that the cheaper embedded/tie variants already replace dynamic POPT.
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

## Tier C — gem5 / Sniper GRASP-vs-LRU sign test (snapshot)

Tier C cross-checks the per-(graph, app, L3-size) sign of
`miss_rate(GRASP) − miss_rate(LRU)` between the `cache_sim` reference
sweep and the gem5 / Sniper sweeps. Mandatory agreement is required at
L3 ∈ {4kB, 32kB}; the 256kB and 2MB cases are warnings only because the
working set fits comfortably and the policies converge.

Tooling:
- `scripts/experiments/ecg/sign_consistency.py` — comparator CLI; takes
  the three sweep roots and a list of `graph/app` pairs and emits a
  human-readable table plus an optional JSON summary.
- `scripts/test/test_grasp_sign_consistency.py` — pytest gate. Skips when
  the cache_sim reference or a simulator sweep CSV is missing, fails on
  unrecorded mandatory disagreements, and xfails on entries in
  `KNOWN_DISAGREEMENTS`.
- Source sweeps live under `/tmp/graphbrew-grasp-{cache,gem5,sniper}-sweep`
  with the layout `<graph>-<app>/DBG/roi_matrix.csv`. The handoff
  one-liner in `wiki/HANDOFF-grasp-popt-validation.md` regenerates the
  cache_sim sweep; `--suite gem5`/`--suite sniper` regenerate the others.

Initial sweep ({PR, BC} on {email-Eu-core, cit-Patents}, L1d=1kB, L2=2kB,
L3-ways=16, line=64):

| graph         | app | L3   | cache_sim Δ | gem5 Δ    | sniper Δ | mandatory agreement |
| ------------- | --- | ---- | ----------- | --------- | -------- | ------------------- |
| email-Eu-core | pr  | 4kB  | −0.0212     | +0.2081   | −0.0376  | gem5 ✗ (known), sniper ✓ |
| email-Eu-core | pr  | 32kB | +0.0269     | +0.0093   | +0.0039  | gem5 ✓, sniper ✓    |
| email-Eu-core | pr  | 256kB| +0.0033     | −0.0214   | +0.0033  | gem5 warn, sniper ✓ |
| email-Eu-core | pr  | 2MB  | +0.0005     | +0.0003   | +0.0015  | gem5 ✓, sniper ✓    |
| email-Eu-core | bc  | 4kB  | −0.1225     | −0.0520   | n/a      | gem5 ✓, sniper n/a  |
| email-Eu-core | bc  | 32kB | +0.0180     | +0.0011   | n/a      | gem5 ✓, sniper n/a  |
| email-Eu-core | bc  | 256kB| +0.0047     | −0.0033   | n/a      | gem5 warn, sniper n/a |
| email-Eu-core | bc  | 2MB  | +0.00005    | −0.00006  | n/a      | gem5 warn, sniper n/a |
| cit-Patents   | pr  | 4kB  | +0.1654     | (pending) | +0.0851  | sniper ✓            |
| cit-Patents   | pr  | 32kB | +0.2545     | (pending) | +0.0002  | sniper ✓            |
| cit-Patents   | pr  | 256kB| +0.0757     | (pending) | −0.0032  | sniper warn         |
| cit-Patents   | pr  | 2MB  | −0.0071     | (pending) | −0.0012  | sniper ✓            |
| cit-Patents   | bc  | 4kB  | +0.0193     | (pending) | n/a      | sniper n/a          |
| cit-Patents   | bc  | 32kB | +0.0759     | (pending) | n/a      | sniper n/a          |
| cit-Patents   | bc  | 256kB| +0.0664     | (pending) | n/a      | sniper n/a          |
| cit-Patents   | bc  | 2MB  | +0.0490     | (pending) | n/a      | sniper n/a          |

(Δ = `miss_rate(GRASP) − miss_rate(LRU)`. Negative = GRASP improves miss
rate. Sniper currently runs only the PR kernel-smoke workload; BC sweeps
are skipped pending a non-SDE-heavy Sniper workload path. The gem5
`cit-Patents` rows are left as `(pending)` because that sweep takes
multiple hours and is intentionally backgrounded; the pytest auto-skips
those entries until the CSVs land and the audit table can be amended.)

### Findings

- **Sniper PR matches cache_sim sign at all mandatory L3 sizes** for both
  `email-Eu-core` and `cit-Patents`. The 256kB warning for `cit-Patents`
  is at the convergence boundary.
- **gem5 BC matches cache_sim sign at all mandatory L3 sizes** on
  `email-Eu-core` (4kB and 32kB both agree). The 256kB and 2MB
  disagreements are sub-permille deltas at the convergence boundary and
  fall under the existing warning policy.
- **gem5 disagrees with cache_sim at `email-Eu-core` / PR / L3=4kB**. The
  delta is large (+0.21 vs −0.02) and both SRRIP *and* GRASP miss-rates
  jump well above LRU in gem5 at this size, while cache_sim sees SRRIP
  and GRASP both slightly improving on LRU:

  ```
  cache_sim 4kB: LRU=0.387 SRRIP=0.363 GRASP=0.366
  gem5 4kB sec1: LRU=0.292 SRRIP=0.600 GRASP=0.500
  ```

  The fact that SRRIP also degrades against gem5's LRU points to a
  generic RRPV / hot-region masking problem in the gem5 overlay rather
  than a GRASP-specific defect. This is captured as a known disagreement
  in `KNOWN_DISAGREEMENTS` so that the pytest stays green on currently
  documented behavior but fires immediately when the underlying issue is
  fixed (XPASS) or a new disagreement appears (FAIL). The same root
  cause is not active for BC, which suggests the trigger is tied to PR's
  push-style edge traversal pattern at very small L3.

### Open follow-ups

1. Replay `email-Eu-core` / PR / L3=4kB under gem5 with the
   `[graphctx]` registration trace enabled (Tier A log line, suppressible
   via `GRAPHBREW_SIDEBAND_LOG=0`) and confirm the GRASP region's
   `base`/`upper` lie inside the LLC-accessed range.
2. Compare gem5's `system.l3cache.rrip*` stats between LRU, SRRIP, and
   GRASP at 4kB to see whether the policy is forcing 3-bit-RRPV inserts
   that get evicted before reuse.
3. Once the gem5 `cit-Patents` sweep completes (~hours), update the table
   above and append any new mandatory disagreements (or add them to
   `KNOWN_DISAGREEMENTS` and the audit follow-up list).
4. Enable a non-SDE-heavy Sniper workload (`kernel_smoke` covers PR/BFS/SSSP;
   BC needs a dedicated path) so the BC half of the matrix can be measured.
