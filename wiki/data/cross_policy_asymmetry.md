# Gate 64 — Cross-policy mean-margin asymmetry

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every (A,B) policy pair has at least one cell where each side wins head-to-head AND the largest observed asymmetry ratio (max(meanA,meanB) / min(meanA,meanB)) is strictly less than the sanity ceiling of 20.0

max asymmetry ratio observed: 3.6535 (ceiling 20.0)

## Head-to-head per policy pair (margin in pp of miss-rate)

| pair | A wins | B wins | ties | A mean pp | B mean pp | asymmetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRASP vs LRU | 92 | 17 | 5 | 5.010 | 16.499 | 3.293 |
| GRASP vs POPT | 58 | 51 | 5 | 1.973 | 5.720 | 2.899 |
| GRASP vs SRRIP | 87 | 22 | 5 | 3.554 | 12.985 | 3.654 |
| LRU vs POPT | 16 | 94 | 4 | 5.345 | 4.715 | 1.133 |
| LRU vs SRRIP | 17 | 92 | 5 | 0.568 | 1.810 | 3.186 |
| POPT vs SRRIP | 85 | 24 | 5 | 3.464 | 3.898 | 1.125 |
