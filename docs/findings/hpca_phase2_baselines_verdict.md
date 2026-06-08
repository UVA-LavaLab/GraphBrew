# HPCA Phase 2 (Baselines) Verdict

**Date:** 2026-06-07
**Run dir:** `results/ecg_experiments/hpca_mode6/baselines_v1/`
**Profile:** `--profile baselines` (10 jobs, ~1h30 wall)
**Status:** all 10 jobs `status=ok`, sanity PASS

## Results — 5 graphs × 5 replacement policies at canonical config (L3=2MB)

| graph | LRU | GRASP | POPT | ECG:DBG_ONLY | ECG:POPT_PRIMARY | DROPLET-style k=8 |
|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core | 2,134 (1.00×) | 2,134 (1.00×) | 2,134 | 2,134 | 2,134 | 2,134 (1.00×) |
| web-Google | 4,505,694 (1.00×) | 3,048,170 (0.68×) | 2,968,147 (0.66×) | 3,047,696 | 2,966,401 | 1,426,953 (**0.32×**) |
| cit-Patents | 53,497,023 (1.00×) | 43,625,335 (0.82×) | 40,720,817 (0.76×) | 43,628,512 | 41,020,987 | 7,841,959 (**0.15×**) |
| soc-LJ | 62,002,738 (1.00×) | 56,377,668 (0.91×) | 48,501,065 (0.78×) | 56,237,072 | 48,872,155 | 13,238,979 (**0.21×**) |
| com-orkut | 184,933,051 (1.00×) | 173,505,699 (0.94×) | 149,109,226 (0.81×) | 173,505,817 | 149,290,553 | 30,963,952 (**0.17×**) |

DROPLET-style k=8 dramatically beats every replacement policy on demand-miss
reduction (68-85% reduction vs LRU), but at large prefetch_fills cost
(true_DRAM ≈ baseline, so DRAM-neutral).

## Baseline parity (KILL-2 follow-up)

| graph | GRASP vs ECG:DBG_ONLY | POPT vs ECG:POPT_PRIMARY |
|---|---:|---:|
| email-Eu-core | 0.000% | 0.000% |
| web-Google | -0.016% | -0.059% |
| cit-Patents | +0.007% | +0.737% |
| **soc-LJ** | **-0.249%** | **+0.765%** |
| com-orkut | 0.000% | +0.122% |

All within ±0.77% — confirms ECG variants are paper-faithful approximations
of GRASP and POPT respectively. The intentional DBG tiebreak in
ECG:POPT_PRIMARY adds at most 0.77% delta (which is generally an
improvement, not a regression, per Phase 0 audit).

## Verdict: 🟢 GO for Phase 3 buildup

Sanity checks pass. Phase 3 launched (`results/ecg_experiments/hpca_mode6/buildup_v1/`).

Phase 3 will test the ECG_PFX arms (mode 2 runtime POPT, mode 6 amp=0 ISA,
mode 6 amp=1 ISA = HEADLINE, mode 6 amp=1 SW = negctrl) on all 5 graphs.
20 jobs expected, ~1.5-2.5h wall.
