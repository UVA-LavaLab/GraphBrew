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

## DRAM-level demand traffic (the §4.3-safe metric)

Aggregated `dram-bank-group-*.num-requests` counters across all DRAM banks. Ratio >1 means MORE DRAM traffic than baseline (worse); ratio <1 means LESS DRAM traffic (better).

| Cell | none (req) | DROPLET (req) | ECG_PFX (req) | DROPLET/base | ECG_PFX/base |
|---|---:|---:|---:|---:|---:|
| email-Eu-core-pr | 2,360 | 2,359 | 15,312 | 1.00× ≈ | 6.49× ✗ |
| delaunay_n19-pr | 1,192,536 | 1,190,929 | — | 1.00× ≈ | pending |
| roadNet-CA-pr | — | — | — | pending | pending |
| web-Google-pr | — | — | — | pending | pending |

## Honest reading of the email-Eu-core row

Raw Sniper counters (not in the table above):

- baseline: l3_accesses=2360, l3_misses=2360, DRAM requests=2360, pf_useful=0
- DROPLET: l3_accesses=2359, l3_misses=2359, DRAM requests=2359, **pf_useful=1969** (100% L2-hit accuracy, but doesn't reduce DRAM traffic)
- ECG_PFX: l3_accesses=23295, l3_misses=8768, **DRAM requests=15312 (6.5× MORE)**, pf_issued=38, pf_useful=38

Under the L3-miss-rate metric ECG_PFX appeared to win +62pp; under the §4.3-safe DRAM-traffic metric, ECG_PFX **increases** DRAM traffic 6.5× on email-Eu-core because the per-edge mask reads themselves miss to DRAM. DROPLET's 1,969 useful L2 prefetches do not reduce DRAM traffic (those lines were cold-misses regardless) but they do hide L2 miss latency. email-Eu-core is structurally too small to demonstrate prefetching at L3 boundary: the entire property array fits in L1d (4 KB << 32 KB L1d). Cells larger than L1d (delaunay_n19's 2 MB and above) are needed to measure prefetcher value cleanly.

This is **NOT** a refutation of the cache_sim mode 6 corpus finding (which uses million-vertex graphs where the property array exceeds L3). It IS a demonstration that the convergence story (§5.4 of the paper) holds: when the cache hierarchy already captures the working set, prefetcher state adds bandwidth without benefit.
