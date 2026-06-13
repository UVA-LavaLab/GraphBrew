# Gate 66 — Per-policy capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every policy's median slope is strictly less than -5.0 pp/octave, (2) LRU has the steepest (most-negative) median slope, and (3) GRASP has a strictly shallower (less-negative) median slope than LRU.

cells scored: 112; L3 axis: 1MB, 4MB, 8MB

steepest median (most cache-hungry policy): **LRU**

shallowest median (least cache-hungry policy): **GRASP**

median steepness gap: 2.599 pp/octave

## Per-policy slope distribution (pp of miss-rate per log2(L3 MB))

| policy | n cells | median | mean | p10 | p90 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRASP | 28 | -13.012 | -11.768 | -19.561 | -0.618 | -20.927 | 0.000 |
| LRU | 28 | -15.611 | -13.101 | -20.893 | -0.106 | -23.192 | 0.000 |
| POPT | 28 | -13.275 | -12.544 | -19.274 | -0.683 | -23.683 | 0.000 |
| SRRIP | 28 | -15.600 | -13.035 | -20.538 | -0.103 | -22.596 | 0.000 |
