# Gate 66 — Per-policy capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every policy's median slope is strictly less than -5.0 pp/octave, (2) LRU has the steepest (most-negative) median slope, and (3) GRASP has a strictly shallower (less-negative) median slope than LRU.

cells scored: 112; L3 axis: 1MB, 4MB, 8MB

steepest median (most cache-hungry policy): **LRU**

shallowest median (least cache-hungry policy): **GRASP**

median steepness gap: 0.967 pp/octave

## Per-policy slope distribution (pp of miss-rate per log2(L3 MB))

| policy | n cells | median | mean | p10 | p90 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRASP | 28 | -14.651 | -13.064 | -18.954 | -4.648 | -20.858 | -0.000 |
| LRU | 28 | -15.618 | -12.949 | -19.995 | -0.104 | -21.902 | 0.001 |
| POPT | 28 | -14.753 | -12.941 | -19.575 | -1.397 | -20.412 | 0.001 |
| SRRIP | 28 | -15.594 | -12.779 | -19.808 | -0.104 | -20.871 | 0.001 |
