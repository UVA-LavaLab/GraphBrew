# Paper Table 8 — Sniper cross-simulator mode 6 corroboration

**Status: 1/4 cells with valid Sniper ECG_PFX data.**

Sprint 6f-6 cross-sim audit: pairs Sniper mode 6 cells (from pfx_sniper_mode6_sweep.sh) with their cache_sim mode 6 counterparts. Cells in 'pending' state are still simulating or have not yet been launched. The DROPLET column is the Sniper-side baseline-stronger comparator (different prefetching mechanism than ECG_PFX mode 6).

## Per-cell Sniper measurements (l3_miss_rate; pp = baseline minus arm)

| Cell | none | DROPLET | ECG_PFX | DROPLET Δ | ECG_PFX Δ | cache_sim mode 6 Δ |
|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core-pr | 1.0000 | 1.0000 | 0.3764 | +0.00pp | +62.36pp | pending |
| delaunay_n19-pr | 0.9619 | 0.9596 | — | +0.24pp | pending | pending |
| roadNet-CA-pr | — | — | — | pending | pending | pending |
| web-Google-pr | — | — | — | pending | pending | -3.84pp |
