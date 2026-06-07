# HPCA Mode 6 — Rigorous Evaluation Plan (Checklist v1)

**Sprint:** post-6f-7 reset
**Date:** 2026-06-07
**Author:** Rubber-duck-vetted, after 3 critique passes
**Status:** Draft for execution; gated by phased kill switches

This is the cleaned, rigorous, **reproducible** evaluation plan for the ECG
mode 6 HPCA submission. It replaces the sprawl of `/tmp/` experiments
accumulated in sprints 6f-1 through 6f-7.

## Goals

1. **Faithful baselines**: compare against DROPLET (Basak HPCA'19), P-OPT
   (Balaji HPCA'21), GRASP (Faldu HPCA'20) using their reported cache configs
   and implementation knobs, OR explicitly document any divergence.
2. **Reproducible**: from `git rev-parse HEAD` + graphs/, a single command
   regenerates every paper table and figure.
3. **Cleanly phased**: kill switches gate expensive experiments so a dead
   claim is caught in <2h, not after days of compute.
4. **Honest framing**: claim only what the data supports; explicitly cite
   the implementation gaps where we use simplified versions of baselines.

## Baseline-config matrix (paper-faithful values)

| param | **DROPLET** | **P-OPT** | **GRASP** | **Our current** |
|---|---|---|---|---|
| Cores | 4-core 2.66GHz | 8-core 2.266GHz | 8-core 2.66GHz | **1-core** |
| L1d | 32KB/8w, 4c | 32KB/8w Bit-PLRU, 3c | 32KB/4-8w + 16-stream stride, 4c | 32KB/8w |
| L2 | 256KB/8w, 8c | 256KB/8w Bit-PLRU, 8c | 256KB/8w unified, 6c | 256KB/8w |
| LLC | **8MB shared, 16w, 30c** | **3MB/core × 8 = 24MB DRRIP NUCA** | **2MB/core × 8 = 16MB NUCA** | **1MB fixed** |
| LLC per-core | 2MB/core | 3MB/core | 2MB/core | **1MB (single-core)** |
| DRAM | DDR3, 45ns | unspecified | unspecified | DDR4-2400 |
| Prefetcher attach | **L2 streamer + MC property prefetcher (decoupled)** | None (replacement-only) | None (replacement-only) | L2 (Sniper) or all-level (cache_sim) |
| Simulator | Sniper | Sniper + Pin-cache-sim | Sniper | Sniper + cache_sim |

**Canonical config for our paper**: single-core, **L1=32KB/8w, L2=256KB/8w,
L3 = 2MB** (matches per-core LLC of DROPLET and GRASP).

Sensitivity sweep: `L3 ∈ {1MB, 2MB, 4MB, 8MB, 16MB}` to validate the
"working set vs cache" theory across the full regime.

## Our baseline implementation gaps (must document in paper)

| baseline | paper | our impl | gap | label as |
|---|---|---|---|---|
| DROPLET | Decoupled L2-streamer + MC-property-prefetcher | cache_sim mode 3 (next-K oracle) + Sniper L2-attached | Not decoupled; no MC-side prefetcher | **"DROPLET-style streamMPP1"** |
| P-OPT | Rereference matrix + Belady-MIN replacement, DRRIP NUCA | `popt.h` rereference matrix + `ECG_POPT_PRIMARY` policy | Need to verify replacement matches Belady-MIN | **"P-OPT-style"** (until verified) |
| GRASP | Pin-and-protect ABR (Application-Based Reuse) | `ECG_DBG_ONLY` policy + DBG ranks | Need to verify ABR semantics match | **"GRASP-style"** (until verified) |

The honest reframing: "We compare ECG mode 6 against ARCHITECTURAL-LEVEL
APPROXIMATIONS of DROPLET, P-OPT, and GRASP under matched per-core L3
configurations. Full paper-faithful artifact reproduction is left for
follow-up cycle-accurate validation."

## Phase 0 — Faithfulness + automation preflight (no compute)

- [ ] **0.1 Baseline implementation audit**
  - [ ] Read `bench/include/graphbrew/partition/cagra/popt.h` — does the P-OPT
        replacement policy match Belady-MIN derivation from rereference matrix?
  - [ ] Find our GRASP implementation (likely in `bench/include/cache_sim/cache_sim.h`
        or `graph_cache_context.h`); verify pin-and-protect on hot vertices
  - [ ] Re-confirm DROPLET-style: cache_sim mode 3 + Sniper droplet_prefetcher
  - [ ] Output: `docs/findings/baseline_faithfulness_audit_v1.md`

- [ ] **0.2 Orchestrator gap fix**
  - [ ] `roi_matrix.py` does NOT currently expose `ECG_EDGE_MASK_CHARGED`,
        `ECG_EDGE_MASK_AMPLIFY`, `ECG_EDGE_MASK_LOOKAHEAD`. Add CLI args:
        `--ecg-edge-mask-charged 0|1`, `--ecg-edge-mask-amplify N`,
        `--ecg-edge-mask-lookahead N`. Propagate to subprocess env.
  - [ ] Record these values in `roi_matrix.csv` columns so output is
        self-describing.
  - [ ] Verify env propagation works for both cache_sim and Sniper paths.

- [ ] **0.3 Output directory canonicalization**
  - [ ] Define: `results/ecg_experiments/hpca_mode6/<run_id>/`
  - [ ] `manifest.json`, `resolved_manifest.json`, `jobs.csv`,
        `run_status.jsonl`, `preflight/`, `matrices/<phase>/<arm>/<graph>/`,
        `aggregate/all_roi_matrix.csv`, `logs/`
  - [ ] No reference to `/tmp/` for new runs

- [ ] **0.4 Manifest scaffold**
  - [ ] Create `scripts/experiments/ecg/hpca_mode6_manifest.json` with the
        phase-by-phase job spec
  - [ ] Each job records: l3_size, cores, policy, prefetcher, ecg_pfx_mode,
        edge_mask_charged, edge_mask_amplify, edge_mask_lookahead,
        dedup_window, droplet_degree, droplet_lh, graph, algorithm

## Phase 1 — Canonical config freeze + smoke (~30min wall)

- [ ] **1.1 Adopt canonical config**: 1-core, L1=32KB/8w, L2=256KB/8w,
      L3=**2MB**/16w (per-core LLC matched to DROPLET/GRASP)
- [ ] **1.2 Document divergence from each baseline** in `audit_v1.md`:
      "Our 1-core L3=2MB matches DROPLET/GRASP per-core LLC at 2MB/core;
      we test scaled-down absolute cache sizes for single-thread evaluation."
- [ ] **1.3 SMOKE TEST**: `make hpca-evaluation-smoke`
  - [ ] Runs `email-Eu-core/pr` at L3=2MB
  - [ ] All policies: `LRU, GRASP, POPT, ECG:DBG_ONLY, ECG:POPT_PRIMARY`
  - [ ] Under each: `none`, `DROPLET-style LH=16`, `mode6 CHARGED=0 amp=1`
  - [ ] Success criteria:
    - all rows `status=ok`
    - non-empty CSV
    - L3 accesses/misses nonzero
    - POPT matrix active, GRASP regions registered
    - prefetch arms issue requests (`pf_issued > 0`)
    - manifest CSV records every knob value
  - [ ] Expected wall: <15 min

## Phase 1.5 — KILL SWITCHES (~1-2h wall)

These are GO/NO-GO for the headline claim. If any fail, STOP and re-plan.

- [ ] **KILL-1: Mode 6 viability at canonical config**
  - [ ] Cells: `com-orkut/pr`, `soc-LiveJournal1/pr` at L3=2MB and L3=8MB
  - [ ] Arms: ECG:DBG_ONLY + (DROPLET-style LH=8, DROPLET-style LH=16,
        mode 6 CHARGED=0 AMPLIFY=1)
  - [ ] PASS: mode 6 amp=1 beats DROPLET-style LH=8 on (a) absolute
        demand misses saved AND (b) total memory traffic on at least
        one cell (large graph + small L3)
  - [ ] FAIL: headline claim is dead; reframe paper

- [ ] **KILL-2: Baseline parity sanity**
  - [ ] Cells: `email-Eu-core/pr`, `web-Google/pr` at L3=2MB
  - [ ] Compare: `GRASP` vs `ECG:DBG_ONLY` (should match within ~5%);
        `POPT` vs `ECG:POPT_PRIMARY` (should match within ~5%)
  - [ ] PASS: tight tolerance + sideband/matrix active markers present
  - [ ] FAIL: our policy implementations are broken; fix before any sweep

- [ ] **KILL-3: DROPLET activity**
  - [ ] Sniper DROPLET-style on `email-Eu-core/pr` (and `delaunay_n19/pr`)
  - [ ] Require: sideband loaded, edge accesses > 0, pf_issued > 0
  - [ ] PASS: prefetcher is alive
  - [ ] FAIL: no cycle-accurate DROPLET sweeps

## Phase 2 — Minimal baseline stability (~4-6h wall)

- [ ] **2.1 PR on common graphs at canonical config (L3=2MB)**
  - Graphs: `email-Eu-core, web-Google, cit-Patents, soc-LiveJournal1, com-orkut`
  - Arms: baseline (no pfx LRU), GRASP, POPT, DROPLET-style LH=16
  - Algorithm: PR only at this phase
  - 5 graphs × 4 arms = 20 cache_sim runs

- [ ] **2.2 Stability check**: re-run 2-3 cells to confirm determinism

## Phase 3 — Incremental ECG build-up (~6-10h wall)

- [ ] **3.1 ECG eviction validation** (mode 0)
  - [ ] ECG:DBG_ONLY vs GRASP on Phase 2 graphs
  - [ ] PASS: ECG:DBG_ONLY matches/beats GRASP-style on demand misses

- [ ] **3.2 ECG eviction + ECG_PFX mode 2 (runtime POPT lookahead)**
  - [ ] vs DROPLET-style on Phase 2 graphs
  - [ ] PASS: comparable or better demand-saved at lower bandwidth

- [ ] **3.3 ECG eviction + ECG_PFX mode 6 amp=1 CHARGED=0**
  - [ ] vs DROPLET-style on Phase 2 graphs
  - [ ] This is the HEADLINE comparison

- [ ] **3.4 Mode 6 vs P-OPT replacement only**
  - [ ] Compare ECG eviction (DBG) + mode 6 prefetch vs P-OPT eviction only
  - [ ] Shows compositional value of ECG (eviction + prefetch) vs replacement-only

## Phase 4 — Sensitivity sweep (~4-6h wall)

- [ ] **4.1 L3 size sweep**: `{1, 2, 4, 8, 16 MB}` on `soc-LJ`, `com-orkut`
      with all 4 arms; tests working-set theory directly
- [ ] **4.2 AMPLIFY sweep**: `{0, 1, 2, 4, 7}` on 2 cells at L3=2MB
- [ ] **4.3 Dedup-window sweep**: `{0, 16, 64, 256, 4096}` on 1 cell
- [ ] **4.4 CHARGED=0 vs CHARGED=1**: shows the ISA-extension benefit
      (already mostly done in sprint 6f-7, but re-run with canonical config)

## Phase 5 — Cycle-accurate cross-validation (~6-12h, deferred)

- [ ] **5.1 Sniper validation** of Phase 3.3 headline on `delaunay_n19` and
      `web-Google` (or smaller cells) at canonical config
- [ ] **5.2 gem5 validation** if `pfIssued > 0` smoke passes
  - [ ] Smoke: gem5 `email-Eu-core/pr` with ECG_PFX mode 6 — must show
        `pfIssued > 0`. If fails, defer gem5 entirely.
- [ ] **5.3 Sniper magic instruction work** (multi-day; deferred to future
      session unless needed for reviewer rebuttal)

## Cleanup tasks (parallel to phases 1-3)

- [ ] Quarantine old `/tmp/` paths (note: they're still referenced by
      some old Makefile targets and tests)
- [ ] Update `lit-paper-table-mode6-corpus` to read from
      `results/ecg_experiments/hpca_mode6/<run_id>/aggregate/`
- [ ] Update existing test gates (`scripts/test/test_ecg_mode6_*`) to
      read from canonical paths
- [ ] Move final-run lock path away from `/tmp/`

## Reproducibility contract

After this plan executes, the following must be true:

1. `git rev-parse HEAD` + `results/graphs/` + `make hpca-evaluation` →
   regenerates every paper table and figure
2. Each result CSV row records: git hash, graph hash, all knob values,
   wall time, status, manifest run_id
3. `make hpca-evaluation-smoke` (~15 min) verifies the toolchain end-to-end
4. `make hpca-evaluation-killswitch` (~1-2h) verifies the claim is alive
5. `make hpca-evaluation` (~24-40h) runs the full evaluation

## Top blind spots to remain aware of

- **Single-core results don't answer contention questions.** Mitigation:
  add small 2-4 core sanity check (not full multicore sweep) in Phase 5
- **CHARGED=0 is still idealized in cache_sim** until Sniper or gem5
  ISA-delivery is working
- **cache_sim fills all levels on prefetch** → can't model L1-only vs
  L2-only attachment effects
- **Our DROPLET is streamMPP1-class** (~4-12.5% weaker than paper-full
  DROPLET per Basak HPCA'19 evaluation Table)
- **pp/Mreq is fragile.** Primary metrics: absolute demand misses, prefetch
  fills, total memory traffic, DRAM/base ratios

## Cross-references

- `docs/findings/sprint_6f-7_mode6_charged_audit.md` — sprint 6f-7 corrections
- `docs/findings/sniper_delaunay_n19_4arm_dram_finding.md` — Sniper validation
- `docs/findings/gem5_ecg_pfx_simobject_gap.md` — gem5 SimObject gap
- `scripts/experiments/ecg/final_paper_run.py` — manifest orchestrator
  (to be extended with hpca_mode6_manifest.json)
- `bench/include/ecg_mode6_builder.h` — mode 6 mask construction
- `bench/include/graphbrew/partition/cagra/popt.h` — P-OPT math foundation
