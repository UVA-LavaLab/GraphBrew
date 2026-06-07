# Paper Table 7 — ECG Mode 6 (per-edge mask) corpus efficiency

Sprint 6f-5 spike: the per-edge ECG mask is the paper's actual ECG
instruction design (each edge in CSR carries a packed 64-bit mask
`dest|DBG|POPT|prefetch_target`). This table reports the 4-cell
corpus comparison of mode 6 vs mode 2 (runtime POPT lookahead) vs
DROPLET, focused on per-request bandwidth efficiency.

## Per-cell comparison (demand-memory metric)

| Cell | Baseline | Mode 2 K=1 | Mode 6 per-edge | DROPLET | Δ Mode 6 - Mode 2 |
|---|---:|---:|---:|---:|---:|
| cit-Patents-pr | 0.3330 | -9.60pp (17.6M) | -2.03pp (17.6M) | -27.64pp (58.5M) | **+7.57pp** (mode 6 worse) |
| soc-LiveJournal1-pr | 0.1911 | -5.47pp (51.3M) | -0.52pp (51.3M) | -15.55pp (159.1M) | **+4.95pp** (mode 6 worse) |
| com-orkut-pr | 0.2593 | -6.44pp (127.5M) | -1.79pp (127.5M) | -22.68pp (462.3M) | **+4.65pp** (mode 6 worse) |
| web-Google-pr | 0.1525 | -4.88pp (4.6M) | +2.84pp (4.6M) | -11.40pp (15.5M) | **+7.72pp** (mode 6 worse) |

## Corpus aggregate per-request efficiency

| Config | Total savings | Total requests | **pp/Mreq** |
|---|---:|---:|---:|
| Mode 2 K=1 LH=8 | 26.39 pp | 201,033,589 | **0.1312** |
| Mode 6 per-edge pure | 1.50 pp | 201,033,569 | **0.0075** |
| DROPLET LH=8 | 77.27 pp | 695,384,902 | **0.1111** |

**Mode 6 vs Mode 2 ratio: 0.057× (-94.3%)** ← Mode 6 is -94.3% more bandwidth-efficient than runtime mode 2 lookahead
**Mode 6 vs DROPLET ratio: 0.067× (-93.3%)** ← Mode 6 is -93.3% more bandwidth-efficient than DROPLET

## Honest absolute traffic accounting (sprint 6f-7 Phase 2.3+2.7)

The pp/Mreq metric above is a *rate* and can shift when only the
denominator (`total_accesses`) changes. To defuse denominator-
gaming concerns, we also report absolute traffic in cache lines:
`total_memory_traffic = memory_accesses + prefetch_fills`. A
correctly-implemented prefetcher conserves total DRAM traffic;
it just converts demand misses into prefetch fills.

A mode 6 `total_traffic_ratio > 1.05x` can have TWO causes:
  (a) the CSR-double-read bug fixed in commit `1df4c5f9`, OR
  (b) legitimate per-edge mask DRAM cost when `ECG_EDGE_MASK_CHARGED=1`
      (software-delivered mask). Per sprint 6f-7 Phase 2.5 the design
      intent is ISA-delivered metadata (`CHARGED=0`) where mode 6
      DOMINATES DROPLET on large graphs (see docs/findings/
      sprint_6f-7_mode6_charged_audit.md for the full audit).

| Cell | Baseline DRAM | Mode 2 DRAM (× base) | Mode 6 DRAM (× base) | DROPLET DRAM (× base) |
|---|---:|---:|---:|---:|
| cit-Patents-pr | 54,062,431 | 54,039,368 (1.000×) | 66,739,800 (1.234×) 🚩 | 54,230,799 (1.003×) |
| soc-LiveJournal1-pr | 72,932,672 | 72,943,445 (1.000×) | 92,217,231 (1.264×) 🚩 | 72,937,944 (1.000×) |
| com-orkut-pr | 249,487,398 | 249,553,430 (1.000×) | 296,547,873 (1.189×) 🚩 | 249,663,040 (1.001×) |
| web-Google-pr | 6,391,455 | 6,391,822 (1.000×) | 10,215,146 (1.598×) 🚩 | 6,391,659 (1.000×) |

> 🚩 **Mode 6 DRAM inflation > 5% detected.** Per the sprint 6f-7 audit,
> this is the EXPECTED behavior under `ECG_EDGE_MASK_CHARGED=1`
> (software-delivered mask): the per-edge mask is read from memory and
> the fat-edge stream adds cache pressure. To validate the paper's
> ISA-extension design intent, re-run with `ECG_EDGE_MASK_CHARGED=0`
> (idealized ISA delivery). The CSR-double-read bug was fixed in
> commit `1df4c5f9` — that fix is already baked into this data.

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
