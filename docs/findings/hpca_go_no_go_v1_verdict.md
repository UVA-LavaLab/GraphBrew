# HPCA Mode 6 — Go/No-Go v1 Verdict

**Date:** 2026-06-07
**Run dir:** `results/ecg_experiments/hpca_mode6/go_no_go_v1/`
**Manifest:** `scripts/experiments/ecg/hpca_mode6_manifest.json` (profile `go_no_go`)
**Wall:** ~2h, 10 stages all `status=ok`

## Verdict: 🟢 GO

Both kill switches passed. Phase 2 (full baseline sweep) unblocked.

## KILL-1: PASS on all 4 cells

`popt_off__isa__k2` beats `seq__sw__k8` AND `seq__sw__k16` on
(demand_misses_saved AND total_memory_traffic) on EVERY tested
(graph, L3) cell — not just one as the gate required.

| graph | L3 | baseline DRAM | DROPLET k8 DRAM | **mode 6 DRAM** | DRAM saved vs k8 | demand saved vs k8 |
|---|---|---:|---:|---:|---:|---:|
| com-orkut | 2MB | 173.8M | 173.4M | **139.5M (0.80×)** | -34.0M (-20%) | +29.6M |
| com-orkut | 8MB | 46.8M | 46.8M | **14.1M (0.30×)** | **-32.8M (-70%)** | +29.4M |
| soc-LJ | 2MB | 56.2M | 56.2M | **44.5M (0.79×)** | -11.7M (-21%) | +10.8M |
| soc-LJ | 8MB | 21.5M | 21.5M | **10.1M (0.47×)** | -11.4M (-53%) | +10.8M |

### Saved-per-fill (selectivity advantage)

| cell | DROPLET k8 | mode 6 isa k2 | advantage |
|---|---:|---:|---:|
| com-orkut L3=2MB | 1.003× | 1.249× | +24% |
| com-orkut L3=8MB | 1.000× | **3.476×** | **+247%** |
| soc-LJ L3=2MB | 1.000× | 1.278× | +28% |
| soc-LJ L3=8MB | 1.001× | 2.367× | +136% |

### prefetch-useful rate

Both DROPLET and mode 6 hit ~99-100% useful (each prefetched line ends
up demanded). The advantage isn't in accuracy — it's in **reuse**: mode 6
picks lines that are demanded multiple times before eviction
(saved/fill > 1.0), while DROPLET picks mostly one-use lines
(saved/fill ≈ 1.0).

This validates the "hot-hub selection" mechanism: POPT-rank captures
high-reuse vertices (hubs in power-law graphs); offline encoding via
the per-edge fat-mask delivers them via ISA extension with no runtime
selection cost.

## KILL-2: PASS — policy parity within noise

GRASP vs ECG:DBG_ONLY on web-Google L3=2MB:
- GRASP: 3,046,368 demand misses
- ECG:DBG_ONLY: 3,046,777 demand misses
- delta: +0.013% (essentially identical)

POPT vs ECG:POPT_PRIMARY on web-Google L3=2MB:
- POPT: 2,967,915 demand misses
- ECG:POPT_PRIMARY: 2,965,011 demand misses
- delta: -0.098% (ECG:POPT_PRIMARY slightly BETTER due to DBG tiebreak)

All 5 policies (LRU, GRASP, POPT, ECG:DBG_ONLY, ECG:POPT_PRIMARY) loaded
and ran with `status=ok` on both email-Eu-core and web-Google. No
nan/inf. Numeric ranges sane. Baseline parity within 0.1%.

## What this enables

1. **Headline claim is alive**: mode 6 with ISA delivery (Model B per
   sprint 6f-7 audit) beats DROPLET-style sequential prefetching at the
   canonical config matching DROPLET/GRASP per-core LLC.

2. **Advantage scales with L3**: at L3=2MB, mode 6 saves 20% DRAM; at
   L3=8MB it saves 70%. This is the OPPOSITE of "small-cache artifact"
   — the gap grows with realistic cache sizes.

3. **Phase 2 (baseline sweep) unblocked**: ready to run the full 5
   graphs × 5 policies × multiple L3 sizes corpus.

## Caveats from the Phase 0 audit (still apply)

- Our DROPLET implementation is streamMPP1-class, not the full decoupled
  L2-streamer + MC-property-prefetcher architecture from Basak HPCA'19.
  Full DROPLET (per paper Section IV.B) beats streamMPP1 by 4-12.5%, so
  mode 6's 20-70% advantage might shrink against a faithful DROPLET.

- ISA-delivered mode 6 (CHARGED=0) is still an idealized cache_sim model.
  Cycle-accurate Sniper validation is deferred (todo s67-future-sniper-magic).

- Single-core results don't address shared-cache contention. Paper must
  acknowledge this (the canonical L3=2MB matches per-core paper LLC).

## What's next

Phase 2: full baseline sweep on common corpus.

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
    --manifest scripts/experiments/ecg/hpca_mode6_manifest.json \
    --profile baselines \
    --run-dir results/ecg_experiments/hpca_mode6/baselines_v1
```

Followed by Phase 3 (ECG build-up), Phase 4 (sensitivity), Phase 5 (cycle-accurate).
