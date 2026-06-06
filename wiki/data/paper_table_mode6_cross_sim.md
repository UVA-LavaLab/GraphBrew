# Paper Table 8 — Sniper cross-simulator mode 6 corroboration

**Status: 1/4 cells with valid Sniper ECG_PFX data.**

Sprint 6f-6 cross-sim audit: pairs Sniper mode 6 cells (from pfx_sniper_mode6_sweep.sh) with their cache_sim mode 6 counterparts. Cells in 'pending' state are still simulating or have not yet been launched. The DROPLET column is the Sniper-side baseline-stronger comparator (different prefetching mechanism than ECG_PFX mode 6).

## ⚠️ Metric caveat — l3_miss_rate is NOT a fair cross-arm comparator here

The L3 miss-rate metric below is reported because it is the raw counter the simulator emits, but it is **misleading across arms** for two independent reasons:

1. **Prefetcher placement**: in our config DROPLET and ECG_PFX both attach at L2 (`--prefetcher-level l2`); the original DROPLET paper (Basak HPCA'19) prefetches at L1. The 1,969 useful prefetches DROPLET issues land at L2 and hide L2 miss latency, but the L3 still sees the same demand-miss count, so DROPLET's l3_miss_rate equals the baseline even though DROPLET genuinely helped.

2. **Mask-charged denominator inflation**: ECG_PFX mode 6 with `ECG_EDGE_MASK_CHARGED=1` (default) reads the per-edge mask array on every edge access, inflating `l3_accesses` by ~10× (2,360 → 23,295 on email-Eu-core). The mask reads are mostly L3 hits, so `l3_misses / l3_accesses` drops purely by denominator growth, not by actual demand-miss reduction. This is the same metric trap §4.3 of the paper documents (sprint 6f-2 finding).

Until the demand-memory rate (`memory_accesses / total_accesses`) is wired into the Sniper postfix builder, the headline metric for these Sniper cells is `pf_useful` (the count of prefetches that hid a demand miss at the prefetcher's attachment level).

## Per-cell Sniper measurements (l3_miss_rate; pp = baseline minus arm)

| Cell | none | DROPLET | ECG_PFX | DROPLET Δ | ECG_PFX Δ | cache_sim mode 6 Δ |
|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core-pr | 1.0000 | 1.0000 | 0.3764 | +0.00pp | +62.36pp | pending |
| delaunay_n19-pr | 0.9619 | 0.9596 | — | +0.24pp | pending | pending |
| roadNet-CA-pr | — | — | — | pending | pending | pending |
| web-Google-pr | — | — | — | pending | pending | -3.84pp |

## Honest reading of the email-Eu-core row

Raw Sniper counters (not in the table above):

- baseline: l3_accesses=2360, l3_misses=2360, pf_useful=0
- DROPLET: l3_accesses=2359, l3_misses=2359, pf_issued=1969, **pf_useful=1969** (100% accuracy at L2)
- ECG_PFX: l3_accesses=23295, l3_misses=8768, pf_issued=38, pf_useful=38

DROPLET's 1,969 useful L2 prefetches do not reduce L3 miss rate (because L3 sees the same demand stream), but they do hide latency at L2 — the headline `+62pp` ECG_PFX number in the table is denominator-driven, not a real demand-miss reduction. ECG_PFX issued only 38 prefetches because the single-slot mailbox in `graph_cache_context_gem5.hh:109` loses ~99% of kernel hints to overwrites; this is a known issue documented in `docs/findings/gem5_ecg_pfx_simobject_gap.md`.
