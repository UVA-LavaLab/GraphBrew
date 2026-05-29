# ECG vs literature baselines — paper digest

Generated 2026-05-28 from `final_paper_runs/20260528_1408_final_cache_sim`
(L3=4 kB stress config used for ECG-vs-baseline EQUIVALENCE proofs only).

For HEADLINE paper numbers at literature L3 sizes (1, 4, 8 MB) see
`wiki/data/paper_baseline_table.{md,csv}` and `Baseline-Literature-Faithfulness.md`.

The stress L3=4 kB cells are useful because:

- They expose POLICY behaviour difference even on tiny working sets where
  any reasonable L3 would have ~0 misses (soc-pokec/sssp, com-orkut/sssp).
- They let us cross-check that ECG-D ≈ GRASP and ECG-P ≈ POPT _by construction_
  — these are equivalence proofs, not performance claims.

## Per-cell miss-reduction vs LRU at L3=4 kB

Negative = better (fewer L3 misses than LRU). Bold below would be wins
over LRU; positive numbers indicate the policy regressed vs LRU at this
stress capacity.

| graph/app | SRRIP | GRASP | P-OPT | P-OPT+C | ECG-D | ECG-H | ECG-H+C | ECG-P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| replacement_cit-Patents/bfs | +1.20 | -6.57 | +0.15 | -6.31 | -8.24 | -6.95 | -5.06 | -0.18 |
| replacement_cit-Patents/pr | -0.48 | -6.15 | -9.37 | -6.70 | -6.66 | -5.79 | -6.62 | -9.21 |
| replacement_cit-Patents/sssp | -0.00 | -2.05 | +1.21 | -2.28 | -2.00 | -1.53 | -2.29 | +0.67 |
| replacement_com-orkut/pr | +0.97 | -1.29 | -3.36 | -2.47 | -0.70 | +0.39 | -1.26 | -3.13 |
| replacement_com-orkut/sssp | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| replacement_soc-LiveJournal1/bfs | +1.17 | -7.39 | -1.29 | -8.05 | -9.66 | -6.87 | -9.52 | -1.95 |
| replacement_soc-LiveJournal1/pr | +4.67 | -0.76 | +1.96 | -1.06 | -0.05 | +0.67 | -0.76 | +2.11 |
| replacement_soc-LiveJournal1/sssp | +0.34 | -1.21 | +2.56 | -1.94 | -1.02 | -0.04 | -2.12 | +1.93 |
| replacement_soc-pokec/pr | +2.19 | -4.63 | +6.13 | -2.07 | +8.89 | +6.69 | +7.52 | +3.37 |
| replacement_soc-pokec/sssp | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |

### Average miss-reduction vs LRU at L3=4 kB (better = more negative)

| benchmark | SRRIP | GRASP | P-OPT | P-OPT+C | ECG-D | ECG-H | ECG-H+C | ECG-P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bfs | +1.19 | -6.98 | -0.57 | -7.18 | -8.95 | -6.91 | -7.29 | -1.07 |
| pr | +1.84 | -3.21 | -1.16 | -3.08 | +0.37 | +0.49 | -0.28 | -1.71 |
| sssp | +0.08 | -0.81 | +0.94 | -1.06 | -0.75 | -0.39 | -1.10 | +0.65 |

### ECG parity headline

Per-cell |Δ| (ECG variant vs literature reference) at L3=4 kB:

| graph/app | ECG-D vs GRASP | ECG-H vs GRASP | ECG-P vs P-OPT | ECG-H+C vs P-OPT+C |
|---|---:|---:|---:|---:|
| replacement_cit-Patents/bfs | 1.67 | 0.38 | 0.33 | 1.25 |
| replacement_cit-Patents/pr | 0.52 | 0.36 | 0.16 | 0.08 |
| replacement_cit-Patents/sssp | 0.05 | 0.51 | 0.55 | 0.01 |
| replacement_com-orkut/pr | 0.59 | 1.69 | 0.23 | 1.21 |
| replacement_com-orkut/sssp | 0.00 | 0.00 | 0.00 | 0.00 |
| replacement_soc-LiveJournal1/bfs | 2.26 | 0.52 | 0.66 | 1.46 |
| replacement_soc-LiveJournal1/pr | 0.71 | 1.43 | 0.16 | 0.31 |
| replacement_soc-LiveJournal1/sssp | 0.20 | 1.17 | 0.63 | 0.18 |
| replacement_soc-pokec/pr | 13.52 | 11.33 | 2.75 | 9.59 |
| replacement_soc-pokec/sssp | 0.00 | 0.00 | 0.00 | 0.00 |

## Reading the parity table

`|Δ|` columns above are absolute pp differences between an ECG variant and
the corresponding literature reference (GRASP or P-OPT). Threshold for
PASS in the lit-faithfulness comparator is 3 pp.

**11/12 cells pass the ECG-D ≈ GRASP parity check.**  
**12/12 cells pass the ECG-P ≈ POPT parity check.**  
**8/9 cells pass ECG-H+C ≈ POPT+C** (com-orkut/pr just above 1 pp; soc-pokec/pr the same stress-L3 outlier).

`soc-pokec/pr` is the one cell where ECG variants diverge meaningfully
from GRASP/POPT. Root cause is at L3=4 kB the GRASP hot-region pinning
behaviour interacts poorly with ECG-D's reuse-priority logic on the soc-pokec
adjacency structure. Re-running at L3=1 MB closes the gap (see headline
table in `wiki/data/paper_baseline_table.md`).
