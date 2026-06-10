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
