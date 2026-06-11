# 3-Way Implementation Parity + Paper-Fidelity Audit (v2)

**Date:** 2026-06-09
**Trigger:** User request — "review cache_sim / Sniper / gem5 implementation parity
when we compare the same implementation across simulators. Check if we missed any
bugs and our evaluation implementation matches the source papers."
**Scope:** The 4 evaluated algorithms across all 3 simulators:
ECG_PFX (mode-6 mask prefetcher), DROPLET (Basak HPCA'19), GRASP (Faldu HPCA'20),
P-OPT (Balaji HPCA'21).
**Method:** Source-level extraction in each sim (3 parallel sub-agents on
claude-sonnet-4.5) + manual ground-truthing of every load-bearing claim +
web-verification of the GRASP paper spec.

---

## TL;DR verdict

| Algorithm | Cross-sim parity | Matches source paper | Verdict |
|---|---|---|---|
| **ECG_PFX (mode-6)** | ✅ identical target selection | ✅ (this work's design) | **CLEAN** (1 duplication risk) |
| **DROPLET** | ✅ in real runs | ⚠️ cache_sim = best-case oracle | **CLEAN + 1 hygiene + 1 footnote** |
| **GRASP** | ✅ all sims use 50% | ❌ **paper says 10%** | 🔴 **PAPER-FIDELITY BUG** |
| **P-OPT** | ✅ identical | ✅ Balaji Alg.2 | **CLEAN** (MSB polarity doc'd) |

**Update (later same day):** while re-running the corrected GRASP baselines, found a
**second, broader bug — cache_sim is non-deterministic under multithreading** (see
§6). It affects the reproducibility of *every* `roi_matrix` cache_sim stage, not just
GRASP.

**One real paper-integrity bug found: GRASP uses `hot_fraction = 0.50` in all three
simulators, but Faldu HPCA'20 specifies 10%.** Cross-sim parity is intact (all three
agree), so this is a uniform 5× deviation from the paper, not a cross-sim divergence.

---

## 1. ECG_PFX mode-6 (headline prefetcher) — CLEAN

**Target-selection algorithm is identical across all three sims:** for each edge
`(src, neighbor[i])`, scan the next-K in-neighbors `[i+1 .. i+K]` and pick the **single
candidate with the lowest POPT re-reference distance** (`avg_reref_by_line`); encode it
in mask bits `[33:64]`.

- Shared builder: `bench/include/ecg_mode6_builder.h:199-218` (used by Sniper + gem5).
- cache_sim keeps its **own copy** `graph_cache_context.h::buildInEdgeMasks_PR`
  (`:1830-1856`) — **algorithmically identical** (verified line-by-line: same scan,
  same `best_dist` init=128, same `if (d < best)` selection, same bit packing).
- Mask layout identical: `[0:24]dest [24:26]dbg [26:33]popt [33:64]pfx`.
- Prefetch address identical: `property_base + vertex * elem_size` (cache_sim
  `SIM_CACHE_PREFETCH_VERTEX`; Sniper `propertyAddress`; gem5 `propertyAddrForVertex`).
- Parity guarded by `scripts/test/test_ecg_mode6_cross_sim_parity.py` (asserts equal
  `(edges, encoded)` across all three on email-Eu-core).

**Risk (low): code duplication.** cache_sim does not `#include` the shared header — it
maintains a hand-synced duplicate. They match today but can drift silently. *Recommend:*
make cache_sim call `ecg_mode6::buildInEdgeMasks` (or add a parity unit test on the
builders themselves, not just end-to-end stats).

**Minor, documented divergences (do NOT affect target selection):**
- AMPLIFY (extra sequential prefetches/edge) wired in cache_sim + Sniper, **not gem5**.
- Kernel dedup: cache_sim none (functional); Sniper O(1) bitmap, window 256; gem5
  ring buffer, window 16. Different absolute prefetch counts — expected cycle-accurate
  scope difference, already documented in `sim_3way_parity_verdict_v1.md`.
- CHARGED (model mask-array load traffic) is a cache_sim runtime toggle; Sniper/gem5
  pre-build the mask before ROI (implicitly CHARGED=0 / ISA-delivered model).

---

## 2. DROPLET (Basak HPCA'19) — CLEAN in real runs + hygiene + footnote

### 2a. Sniper — paper-faithful ✅
`droplet_prefetcher.cc:37-39`: `prefetch_degree=1, indirect_degree=16,
stride_table_size=64` — exactly Basak's defaults. Stride confidence ≥2 (max 3),
property addr `base + v*elem`. Clean.

### 2b. gem5 SimObject defaults diverge — but DORMANT (hygiene fix)
`GraphPrefetchers.py:26-31` declares `prefetch_degree=4, indirect_degree=8,
stride_table_size=16` — **diverges** from paper/Sniper. **However these are overridden
in every real run**:
- `roi_matrix.py:1209-1214` argparse defaults = `1/16/64` (paper-faithful).
- `graph_cache_config.py:177-179` `make_droplet_prefetcher` defaults = `1/16/64`.
- `graph_se.py:204-219` passes `args.droplet_*` straight through to the SimObject.

So actual gem5 DROPLET runs use 1/16/64. The 4/8/16 in `GraphPrefetchers.py` is a
**dead default** never exercised by the harness. *Recommend:* change it to 1/16/64 so a
reviewer reading the SimObject isn't misled (release hygiene; no result impact).

The only run that sets non-paper DROPLET params is `final_paper_manifest.json` block
`09e_sniper_sift_file_droplet_smoke` (`2/4/16`) — a **smoke test** (LRU-only), not a
published number.

### 2c. cache_sim DROPLET (`ECG_PREFETCH_MODE=3`) — best-case oracle (footnote)
`pr.cc:214-226`: sweeps the next `K=8` in-neighbors sequentially, no stride detector,
no mis-prediction. This is the comparator behind the "ECG_PFX uses ~1/3 the bandwidth
of DROPLET" claim (`droplet_vs_ecg_pfx_algorithm.md`). Because it issues **K=8 per
trigger vs the paper's indirect_degree=16**, and has **zero stride mis-prediction
overhead**, cache_sim DROPLET is **cheaper than real DROPLET** → it **understates**
ECG_PFX's advantage (the comparison is *conservative for ECG_PFX*). Already honestly
documented as "best-case." *Recommend:* footnote the K=8-vs-16 + oracle caveat in the
paper's DROPLET table so reviewers don't read it as a rigged-weak DROPLET.

---

## 3. GRASP (Faldu HPCA'20) — 🔴 PAPER-FIDELITY BUG

**All three simulators classify GRASP tiers with `hot_fraction = 0.50` (50% of LLC).
The paper reserves 10%.**

Verified hardcoded constant (single chokepoint each sim routes through):
- Sniper `graph_cache_context_sniper.cc:525` → `constexpr double hot_fraction = 0.50;`
  (callers `cache_set_grasp.cc:87,182`).
- gem5 `graph_cache_context_gem5.hh:491` → `constexpr double hot_fraction = 0.50;`
  (callers `grasp_rp.cc:131,155`).
- cache_sim `graph_cache_context.h:1336` uses `r.grasp_hot_percent`, which **defaults
  to 50** (`:113,1021`); PR/BFS/SSSP call `registerPropertyArray(..., -1.0, ...)`
  (`pr.cc:61-62`, `bfs.cc:191`, `sssp.cc:129`) → manual_hot_fraction unset → **50**.

**Paper spec = 10%, confirmed three ways:**
1. Faldu HPCA'20 §5.2 (web-verified): "reserving 10% of the LLC as the hot region…
   achieves nearly all the potential benefit."
2. Repo's own notes `research/caching/grasp.md:31`: "`hot_fraction = 0.1` (10% of LLC…
   'f' parameter in paper)" — listed under parameters that **"must match the paper."**
3. `research/gem5/grasp-gem5.md:20,53,73`: "Top 10% by edges", `hot_fraction=0.1` "(paper
   default)".

**Aggravating factor (gem5): the configurable knob is silently ignored.**
`GraphGraspRP` exposes `hot_fraction = Param.Float(0.1)` (`GraphReplacementPolicies.py:37`)
and stores it in `hotFraction` (`grasp_rp.cc:20`, `grasp_rp.hh:85`), **but that member is
never read** — classification calls `ctx.classifyGRASP(addr, llcSize)` which hardcodes
0.50. A reviewer inspecting the gem5 config sees `hot_fraction=0.1` and reasonably
concludes GRASP runs at 10%; it actually runs at 50%.

**Origin:** code comments (`graph_cache_context_sniper.cc:407-410`,
`graph_cache_context_gem5.hh:562-565`) rationalize 0.50 as "frontier_frac=50, matching
upstream GRASP traces" — a misreading: GRASP's `f` parameter is 0.1 (10% of LLC), not
50%.

**Materiality:** the 50% vs 10% split only matters when the property region exceeds
~10% of the LLC. For the tiny smoke graph (email-Eu-core, ~8 KB property vs 1 MB LLC)
both clamp to "all-HIGH" → identical. For the **headline graphs** (cit-Patents,
soc-pokec, soc-LiveJournal1, com-orkut, kron_s17, uniform_n18 — property regions ≥ LLC)
the hot boundary differs materially → **GRASP baseline numbers on the large graphs are
affected.** A 50% hot region dilutes GRASP's selectivity (protects ~half the LLC),
which generally **weakens** GRASP vs its paper-optimal 10% — i.e., risks **overstating**
ECG's win over GRASP. The repo's own notes (`grasp.md:27`) flag exactly this:
"published comparisons are unsound… any ECG improvement over GRASP requires GRASP at
full strength."

**Fix (single-line per sim, plus wire the dead gem5 knob):**
- Sniper `:525` and gem5 `:491`: `0.50 → 0.10` (or read the configurable value).
- gem5: make `classifyGRASP` honor `hotFraction` instead of a hardcoded constant.
- cache_sim: default `grasp_hot_percent` `50 → 10`, OR pass `0.10` from PR/BFS/SSSP
  `registerPropertyArray`.
- **Re-run all GRASP (and ECG-vs-GRASP) cells on the large graphs** at 10%. email-Eu-core
  is unaffected. This invalidates existing large-graph GRASP baseline numbers.

---

## 4. P-OPT (Balaji HPCA'21) — CLEAN

All three sims: transpose-CSR rereference matrix, 256 epochs, 8-bit entries, identical
`findNextRef`, and the **3-phase victim selection** (evict non-property first → max
re-reference distance → RRIP tiebreak) matching Balaji Algorithm 2. Verified callers and
constants line up across `cache_sim.h`, `cache_set_popt.cc`, `popt_rp.cc`.

**Documented, internally-consistent quirk:** the rereference matrix MSB polarity is
inverted vs the paper's prose (MSB=1 → "not referenced"). The matrix generator and all
three consumers agree, so semantics are preserved; only raw matrix files aren't portable
to a paper-literal decoder. Worth a one-line artifact note.

---

## 5. Action checklist (priority order)

1. 🔴 **GRASP hot_fraction 0.50 → 0.10** in all 3 sims; wire gem5's ignored
   `hotFraction` param; re-run large-graph GRASP + ECG-vs-GRASP baselines. *(Paper-blocking.)*
2. 🟡 gem5 DROPLET `GraphPrefetchers.py` defaults `4/8/16 → 1/16/64` (hygiene; no rerun).
3. 🟡 Paper footnote: cache_sim DROPLET is best-case (K=8, no stride mis-prediction);
   real DROPLET would be worse → ECG_PFX advantage is conservative.
4. 🟢 De-duplicate the ECG mode-6 builder (cache_sim should use the shared header) or add
   a builder-level parity test.
5. 🟢 Artifact note: P-OPT matrix MSB polarity is inverted-but-consistent.

**Cross-sim parity itself is sound** for all four algorithms — the simulators agree with
each other. The single substantive issue is fidelity to the GRASP paper, uniform across
all three.

---

## 6. cache_sim non-determinism under multithreading (found 2026-06-09, post-fix)

**Bug:** the cache_sim kernels (`bench/src_sim/{pr,bfs,sssp}.cc`) run the GAPBS
compute loop under `#pragma omp parallel for`. The `SIM_CACHE_READ/WRITE` macros
record every access into a single shared simulated cache. With >1 OpenMP thread the
accesses interleave in **nondeterministic order**, so the simulated miss counts are
both **non-reproducible run-to-run** and **dependent on the thread count**.

**Evidence** (kron_s16_k4 / PR, L3=4kB, LRU L3 misses):

| OMP_NUM_THREADS | run A | run B |
|---|---:|---:|
| 1  | 141,783 | 141,783 (identical — deterministic) |
| 16 | 297,802 | 309,591 (differ — nondeterministic, ~2× the 1-thread count) |

**Why it matters:** `roi_matrix.py` did **not** pin `OMP_NUM_THREADS` for the
cache-sim suite (it only set it for Sniper). `final_paper_run.py:330` only exports
`OMP_NUM_THREADS` when a stage carries an `omp_threads` setting — and the
`final_paper_manifest.json` cache_sim stages (`10_cache_sim_large_replacement`,
`10b`, `11`, `11a`, `12`, `12a`, `12b`) **do not** set it. So those stages ran with
ambient thread count → non-reproducible numbers. (The `proof_matrix.py` path is fine:
it already defaults `--omp-threads 1`.)

Confirmed against the May-28 baseline: current single-thread cit-Patents/PR LRU =
72,025,276; 16-thread = 92,262,749; the committed May-28 baseline = 83,521,142 — **none
match**, i.e. the published cache_sim large-replacement matrix is not regenerable.

**Fix (committed):** `roi_matrix.py` now pins `OMP_NUM_THREADS` for the cache-sim
suite via a new `--cache-sim-omp-threads` arg (**default 1**). cache_sim is now
deterministic by default (verified: kron_s16_k4 LRU = 786,255 on two consecutive
harness runs).

**Consequence for the paper:** every cache_sim `roi_matrix` table (blocks 10–12, not
just GRASP) should be **regenerated single-threaded** for a reproducible, self-
consistent matrix. The corrected-GRASP re-run is being done at 1 thread and doubles as
the deterministic regeneration of the large-graph replacement matrix (block 10).
Cycle-accurate sims (gem5/Sniper) are unaffected — they already run single-core.

---

## 7. Corrected cache_sim baselines (single-thread, deterministic, GRASP=10%)

Regenerated block-10 large-graph replacement matrix at `OMP_NUM_THREADS=1` with the
corrected GRASP hot_fraction=0.10. L1=1kB/L2=2kB/L3=4kB (component-proof stress regime),
options `-s -o 5 -n 1 -i 2` (pr). **These are the functional component-proof reference;
the paper headline comes from Sniper/gem5.**

| graph | bench | LRU | GRASP(10%) | ECG:DBG_ONLY | GRASP≡DBG_ONLY |
|---|---|---:|---:|---:|:--:|
| cit-Patents | pr | 72,025,276 | 73,690,184 | 73,690,184 | ✅ |
| cit-Patents | bfs | 5,275,492 | 5,329,390 | 5,329,390 | ✅ |
| cit-Patents | sssp | 30,271,486 | 32,038,956 | 32,038,956 | ✅ |
| soc-pokec | pr | 90,780,932 | 92,828,982 | 92,828,982 | ✅ |
| soc-LiveJournal1 | pr | 132,720,980 | 139,938,639 | 139,938,639 | ✅ |
| soc-LiveJournal1 | bfs | 6,009,565 | 6,008,674 | 6,008,674 | ✅ |
| soc-LiveJournal1 | sssp | 61,042,727 | 64,029,341 | 64,029,341 | ✅ |
| com-orkut | pr | 438,883,396 | 446,284,692 | 446,284,692 | ✅ |

(soc-pokec/bfs+sssp and com-orkut/bfs+sssp produce ~0–1 L3 misses at 4kB — trivial-ROI
config artifacts, omitted.)

**Validation:** `GRASP == ECG:DBG_ONLY` exactly in **all 12 cells** — the paper's §A3
equivalence invariant (ECG in DBG-only mode reduces to GRASP) holds after the fix.
At the 4kB stress cache the DBG-tier policies trail LRU slightly (expected; degree
protection has little headroom in a 64-line LLC). The decisive ECG-vs-GRASP comparison
is on Sniper/gem5 at realistic LLCs.

## 8. Sniper large-cell wall-time limit (scope note)

The Sniper ECG_PFX overnight batch confirmed the mechanism on the tractable cells
(kron_s16_k4 = 99.998% useful, uniform_n17_k8 = 100% useful). The denser large cells
exceed Sniper's practical wall budget: **kron_s17 hit the inner `--timeout-sniper 25000`
(~6.9 h) at ~78%** (status=error/exit 124 = timeout, not a crash); uniform_n18_k8 (262K
verts) is expected to behave similarly. Sniper paper coverage is therefore the
≤131K-vertex tractable set; cache_sim/gem5 cover the larger graphs. (The batch ran the
pre-fix Sniper binary — `hot_pct=50` in its log — which only affects the incidental
ECG:DBG replacement tier, not the ECG_PFX prefetch-coverage metric being measured.)

---

## 9. Corrected cycle-accurate GRASP baselines (gem5 + Sniper, hot_fraction=0.10)

**These are the paper-headline numbers** (final paper = Sniper/gem5). Both overlays were
rebuilt with the fix and verified to log `hot_pct=10` (gem5's previously-dead
`hot_fraction` Param now takes effect). Single-thread, L3=4kB, affected policies only
(LRU as unchanged sanity). All 12 cells `status=ok`, all `hot_pct=10`.

| suite | graph | bench | LRU | GRASP(10%) | ECG:DBG_ONLY | DBG_ONLY≈GRASP |
|---|---|---|---:|---:|---:|:--:|
| gem5 | email | pr | 11,343 | 18,557 | 17,857 | ✅ |
| gem5 | email | bfs | 9,908 | 13,347 | 13,283 | ✅ |
| gem5 | email | sssp | 24,575 | 23,116 | 22,696 | ✅ |
| gem5 | g12 | pr | 78,522 | 74,343 | 73,704 | ✅ |
| gem5 | g12 | bfs | 44,608 | 45,730 | 45,644 | ✅ |
| gem5 | g12 | sssp | 75,387 | 67,376 | 66,307 | ✅ |
| sniper | email | pr | 658 | 666 | 667 | ✅ |
| sniper | email | bfs | 920 | 910 | 901 | ✅ |
| sniper | email | sssp | 1,005 | 984 | 967 | ✅ |
| sniper | g12 | pr | 675 | 658 | 673 | ✅ |
| sniper | g12 | bfs | 915 | 903 | 902 | ✅ |
| sniper | g12 | sssp | 1,025 | 993 | 969 | ✅ |

**Findings:**
- **§A3 invariant holds on both cycle-accurate sims:** `ECG:DBG_ONLY ≈ GRASP` (<5% gap)
  in all 12 cells — looser than cache_sim's exact equality (separate `grasp_rp`/`ecg_rp`
  code paths + timing/full-process scope), within the paper's tolerance.
- **ECG ≥ GRASP / DBG-tier helps on structured graphs:** on g12, GRASP and ECG:DBG beat
  LRU (gem5 g12/pr 74.3K/73.7K vs LRU 78.5K; g12/sssp 67.4K/66.3K vs 75.4K). On the tiny
  email-Eu-core working set the margin is small/mixed (expected — little cache pressure).
- gem5's `hot_fraction` Param is now wired (was silently ignored); both sims confirmed at
  the paper-faithful 10%.

**Remaining (optional, slower tier):** gem5 cit-Patents/large-graph replacement (block
15/20) and Sniper cit-Patents (block 09g/h) run at the same corrected code but take much
longer cycle-accurate wall time; not required to establish the corrected invariants,
which hold across all 12 email/g12 cells above plus the 12-cell deterministic cache_sim
matrix (§7).

---

## 10. Evaluation / metric flaw audit (2026-06-11, 3 parallel sub-agents + ground-truthing)

Second-pass hunt for implementation/evaluation flaws beyond the GRASP + determinism
bugs. Every agent claim was manually verified; several agent "CRITICAL" flags were
misreads of the ECG thesis and are downgraded below.

### REAL issues to address in the writeup (not code bugs)

1. **ECG_PFX mode-6 headline assumes FREE metadata (CHARGED=0).** The 8 B/edge fat mask
   is uncharged in the paper headline (`hpca_mode6` manifest sets
   `ECG_EDGE_MASK_CHARGED=0`). At CHARGED=1 (software-realistic) mode-6 is **1.22× DRAM
   traffic** (sprint_6f-7_mode6_charged_audit). Disclosed in docs, but the paper must
   frame mode-6 as an **ISA-extension target** and show the CHARGED=1 software cost.
   The *replacement* policies are charged symmetrically (POPT_CHARGED /
   ECG_DBG_PRIMARY_CHARGED both pay for the P-OPT matrix) — that part is fair.
2. **`useful_rate` is "hit-on-prefetch", not "miss-prevention".** Sniper maps
   `pf_useful=L2.hits-prefetch` / `pf_issued=L2.prefetches`. The 99.998% headline is on
   `pf_issued`, which already excludes ~742K in-cache-filtered prefetches (uniform_n17_k8);
   **per emitted hint precision is ~79%**. There is no timeliness (`pf_late`) metric, so a
   prefetch counts useful even if it hid ~0 latency. Frame honestly.
3. **No speedup is claimable from ECG_PFX cycle-accurate runs.** The harness sets
   `timing_valid_for_speedup=0` for all ECG_PFX gem5/Sniper runs (IPC≈1.3e-5, dominated by
   magic-trap hint delivery). ECG_PFX evidence = miss-reduction + bandwidth only. (The ECG
   *eviction* cells DO have valid timing — `timing_valid_for_speedup=1`, IPC≈0.13, and
   ECG:DBG_ONLY IPC 0.138 > GRASP 0.134 > LRU 0.131 — so eviction speedup is claimable.)
4. **4 kB L3 headline is a stress regime.** An L-curve sweep (4 kB→1 MB) exists; lead with
   realistic sizes / show the curve so the win isn't seen as cherry-picked at 4 kB.

### Reviewer-attack vectors that are DEFENSIBLE (verified)

- **`-i 2` iterations:** verified ECG's advantage + full policy ordering are **stable from
  -i 2 to -i 20** (kron_s16_k4/PR: ECG:DBG 0.7903→0.7885, LRU 0.9115→0.9107; identical
  ordering). PageRank's per-iteration access pattern is steady, so -i 2 is representative.
- **Reordering `-o 5` (DBG):** applied uniformly to ALL policies — no differential edge.
- **DROPLET knob asymmetry:** DROPLET gets MORE bandwidth (indirect_degree 16 vs ECG's
  best-1-of-8) → biases AGAINST ECG; ECG wins on efficiency anyway. Conservative.
- **Selection bias:** paper-table generators iterate ALL (graph,app,L3) cells; no
  cherry-pick logic found.

### Agent claims DOWNGRADED (misread the ECG design)

- **"POPT_PRIMARY / DBG_PRIMARY use info the GRASP/P-OPT baselines lack → unfair":** This
  is the ECG **contribution** (combine degree + rereference), not cheating. ECG:DBG_PRIMARY
  = degree-primary + rereference tiebreak vs pure GRASP; ECG:POPT_PRIMARY = rereference-
  primary + degree tiebreak vs pure P-OPT (which itself uses the matrix). Fair as long as
  metadata is charged (CHARGED variants exist). NOT a bug.

### Latent code bug (NOT paper-affecting — unused modes) — ✅ RESOLVED 2026-06-11

- **Sniper `ECG_EMBEDDED` silently aliased `DBG_PRIMARY`** (cache_set_ecg.cc fell through to
  `findDBGPrimaryVictim`). **Fixed** (commit 117ab885): added `findECGEmbeddedVictim`
  matching gem5/cache_sim (max-RRPV → highest stored P-OPT hint → highest DBG tier).
  Verified distinct: email-Eu-core/pr ECG_EMBEDDED 656 vs DBG_PRIMARY 675 (were identical).
- **Silent fallback for unknown/cache_sim-only modes** (POPT_TIE, ECG_EPOCH_EMBEDDED, typos):
  `stringToECGMode` returned DBG_PRIMARY silently, mislabeling result rows. **Fixed** (commit
  28bc7a0a): gem5 + Sniper now fail loudly (`FATAL: unsupported ECG mode ...` + abort) for
  unrecognized modes; empty/DBG_PRIMARY default preserved. gem5 also already rejected bad
  `--ecg-mode` via argparse; this is the C++/env-path defense (the one Sniper uses). Paper
  modes (DBG_ONLY/DBG_PRIMARY/POPT_PRIMARY) verified unchanged (status=ok).

### Verified-good
- DBG_ONLY ≡ GRASP (exact in cache_sim; ≈ in cycle-accurate) — §A3 invariant holds.
- P-OPT baseline uses the rereference matrix (its own mechanism); ECG doesn't get
  "secret" oracle info beyond what's charged.

---

## 11. Paper-writeup framing check (2026-06-11) — prose already honest; numbers need corrected-GRASP regen

Audited whether the paper PROSE overstates the §10 framing issues. **It does not** — all
four are already handled honestly (the §10 agent concerns came from manifest/config
defaults, not the paper text):

| §10 concern | Paper reality | Citation |
|---|---|---|
| "CHARGED=0 free-metadata headline" | CHARGED=1 is headline; CHARGED=0 reported as a sensitivity that is *worse* (cache-warming) | section5_results.tex:174-180; section4_methodology.tex:193-243 |
| "99.998% useful is misleading" | Framed as bandwidth-efficiency (Pareto); useful-rate failure mode explained; not led with | section5_results.tex:182-189,259-289 |
| "no speedup but claims it" | Metrics are demand-memory/bandwidth pp-deltas; L3-miss-rate explicitly NOT the headline; IPC/energy limited to Sniper | section4_methodology.tex:150-164; section6_discussion.tex:170 |
| "4 kB cherry-picked headline" | Headline L3 = **1 MB** (literature-standard); 4 kB is only the component-proof regime; sensitivity studied | section4_methodology.tex:73,79-80 |

**Numerical follow-up (the real remaining work):** the GRASP-dependent tables
(`paper_table_grasp_parity`, baseline tables) were generated at the old `hot_fraction=0.50`
and at the 1 MB headline config. They need regeneration at the corrected 0.10. **However the
§A3 parity conclusion is robust**: GRASP and ECG_DBG both route through the same
`classifyGRASP`, so the corrected fraction shifts BOTH columns together — Δ(ECG_DBG−GRASP)
stays ≈0. Only the absolute miss-rates move. (cache_sim verified ECG_DBG_ONLY ≡ GRASP
*exactly* at 10% in §7.) No paper conclusion changes; the numbers should be refreshed via a
corrected-GRASP literature-corpus sweep at 1 MB → regenerate JSON → `make
lit-paper-table-grasp-parity`.

### §11 follow-up: corrected GRASP-parity verified (16/16 exact, deterministic, 1 MB)

Reproduced the 16 `paper_table_grasp_parity` cells at the paper headline config
(L3=1 MB, L1=32 kB, L2=256 kB) with **corrected hot_fraction=0.10** and
**single-thread** (deterministic). Result: **ECG_DBG_ONLY ≡ GRASP exactly, Δ=0.000pp on
all 16/16 cells** (max |Δ|=0.000pp). This is *tighter* than the published table (mean
|Δ|=0.04pp, max 0.21pp) — the old non-zero Δ was multithread-nondeterminism noise (§6),
not a real ECG_DBG/GRASP difference. The §A3 equivalence claim is fully robust to the
GRASP fix and actually becomes exact under the deterministic config.

Corrected miss-rates (GRASP = ECG_DBG_ONLY, L3=1 MB, single-thread): email-Eu-core/pr
1.0000; web-Google pr/bfs/bc 0.8062/0.9851/0.9149; cit-Patents pr/bfs/sssp/bc
0.9290/0.9912/0.9559/0.9678; soc-pokec pr/sssp/bc 0.8452/1.0000/0.8856; soc-LiveJournal1
pr/bfs/sssp/bc 0.8976/0.9624/0.8846/0.8986; com-orkut/pr 0.8553. (Absolute values differ
from the published table because those were generated multithreaded/non-reproducibly;
these are deterministic. Raw CSVs: results/grasp_parity_1mb/.)

**Action to refresh the paper table:** regenerate the full lit-faith corpus single-thread
at 1 MB (all policies, for the LRU/SRRIP/POPT columns too), then `make
lit-paper-table-grasp-parity`. Conclusions (ECG_DBG≡GRASP, ECG≥GRASP, ECG≈POPT) are
unchanged; only absolute numbers refresh.

---

## 12. ⚠️ CORRECTION (2026-06-11): the GRASP hot_fraction "fix" was WRONG — reverted

§3 / commit ba92bcf2 changed GRASP `hot_fraction` 0.50→0.10, citing Faldu's paper text
("10% of LLC") and a repo note. **That was a regression and has been reverted to 0.50**
(commit reverting it). Root cause of my error: I trusted the paper *text* over GRASP's
*released code*.

**Authoritative evidence (GRASP source):** faldupriyank/grasp `ligra/ligra.h:66`:
`int frontier_frac = 50;` — GRASP's released default is **50**, not 10. GRASP marks the
protected region as `add_region("propertyB", frontierAddr, frontier_frac, n)` — a
percentage of the **vertex space n**, default 50%. The original code's `hot_fraction=0.50`
(comment "frontier_frac=50, matching upstream GRASP traces") was correct.

**Empirical confirmation (same cells):**
- web-Google/pr GRASP: 0.50 → 0.4524 (beats SRRIP by 8.8pp, matches Faldu Fig 10 + prior
  committed 0.4531); 0.10 → 0.8062 (26pp WORSE than SRRIP — GRASP broken).
- `lit-faith`: 0.50 → 0 disagreements; 0.10 → 6 GRASP disagreements (Faldu Fig 10/11) on
  cells that passed at 0.50.

**Implication for §7, §9, §11 of this doc:** those tables were generated at the wrong
0.10 ("corrected" was a misnomer). The PARITY findings still hold (ECG_DBG_ONLY ≡ GRASP
shifts both columns together, so Δ≈0 at any fraction — verified). But the absolute
miss-rates in §7/§9/§11 are at 0.10 and should be disregarded; the faithful numbers are
at 0.50 (= the prior committed values, now reproduced deterministically).

**Net of the GRASP saga:** the original 0.50 was correct all along. The genuinely valuable
fixes from this session stand: cache_sim determinism (OMP_NUM_THREADS=1 pin), Sniper
ECG_EMBEDDED implementation, unsupported-mode guards, DROPLET SimObject-default hygiene.
The lasting GRASP lesson: a known normalization approximation (LLC-bytes vs vertex-%) is
calibrated/validated at value 50 via lit-faith; reproduce-the-results is the authority,
not the paper's text parameter.

**Open refinement (optional, not blocking):** make `classifyGRASP` mark hot/cold as
`frontier_frac × n` (vertex space) to mechanically match GRASP rather than `× LLC bytes`.
Today it's value-matched (50) + results-validated; the normalization is a documented
approximation.
