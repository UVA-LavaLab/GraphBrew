# Paper Table 7 — ECG Mode 6 (per-edge mask) corpus efficiency

Sprint 6f-5 spike: the per-edge ECG mask is the paper's actual ECG
instruction design (each edge in CSR carries a packed 64-bit mask
`dest|DBG|POPT|prefetch_target`). This table reports the 4-cell
corpus comparison of mode 6 vs mode 2 (runtime POPT lookahead) vs
DROPLET, focused on per-request bandwidth efficiency.

## Per-cell comparison (demand-memory metric)

| Cell | Baseline | Mode 2 K=1 | Mode 6 per-edge | DROPLET | Δ Mode 6 - Mode 2 |
|---|---:|---:|---:|---:|---:|
| cit-Patents-pr | 0.3330 | -9.60pp (17.6M) | -12.39pp (17.6M) | -27.64pp (58.5M) | **-2.79pp** (mode 6 better) |
| soc-LiveJournal1-pr | 0.1911 | -5.47pp (51.3M) | -5.47pp (51.3M) | -15.55pp (159.1M) | **-0.00pp** (mode 6 better) |
| com-orkut-pr | 0.2593 | -6.44pp (127.5M) | -8.43pp (127.5M) | -22.68pp (462.3M) | **-2.00pp** (mode 6 better) |
| web-Google-pr | 0.1525 | -4.88pp (4.6M) | -3.84pp (4.6M) | -11.40pp (15.5M) | **+1.04pp** (mode 6 worse) |

## Corpus aggregate per-request efficiency

| Config | Total savings | Total requests | **pp/Mreq** |
|---|---:|---:|---:|
| Mode 2 K=1 LH=8 | 26.39 pp | 201,033,589 | **0.1312** |
| Mode 6 per-edge pure | 30.14 pp | 201,033,509 | **0.1499** |
| DROPLET LH=8 | 77.27 pp | 695,384,902 | **0.1111** |

**Mode 6 vs Mode 2 ratio: 1.142× (+14.2%)** ← Mode 6 is 14.2% more bandwidth-efficient than runtime mode 2 lookahead
**Mode 6 vs DROPLET ratio: 1.349× (+34.9%)** ← Mode 6 is 34.9% more bandwidth-efficient than DROPLET

## Honest framing

Mode 6 does NOT beat DROPLET on absolute miss-rate reduction:
DROPLET issues 2.6× the prefetch bandwidth (695M vs 201M reqs across
the corpus) and achieves 2.6× the total savings (77.3 pp vs 30.1 pp).
The per-edge advantage is in PER-REQUEST efficiency — useful in
bandwidth- or energy-constrained deployments.

This bandwidth-efficiency story is consistent with the sprint 6f-5
saturation finding (docs/findings/prefetcher_saturation_under_eviction.md):
graph-aware prefetchers saturate under good eviction. Mode 6's value
is moving the efficient operating point on the Pareto curve, not
breaking the saturation cap.
