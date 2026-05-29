# Gate 64 — Cross-policy mean-margin asymmetry

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every (A,B) policy pair has at least one cell where each side wins head-to-head AND the largest observed asymmetry ratio (max(meanA,meanB) / min(meanA,meanB)) is strictly less than the sanity ceiling of 20.0

max asymmetry ratio observed: 3.196 (ceiling 20.0)

## Head-to-head per policy pair (margin in pp of miss-rate)

| pair | A wins | B wins | ties | A mean pp | B mean pp | asymmetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRASP vs LRU | 92 | 20 | 2 | 4.779 | 14.048 | 2.940 |
| GRASP vs POPT | 57 | 55 | 2 | 2.019 | 5.387 | 2.668 |
| GRASP vs SRRIP | 86 | 26 | 2 | 3.454 | 11.040 | 3.196 |
| LRU vs POPT | 17 | 96 | 1 | 5.031 | 4.431 | 1.135 |
| LRU vs SRRIP | 18 | 94 | 2 | 0.537 | 1.684 | 3.138 |
| POPT vs SRRIP | 85 | 27 | 2 | 3.363 | 3.505 | 1.042 |
