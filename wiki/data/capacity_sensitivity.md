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
| GRASP | 28 | -14.651 | -13.106 | -19.280 | -5.531 | -20.858 | -0.000 |
| LRU | 28 | -15.618 | -12.909 | -19.994 | -0.104 | -22.059 | 0.001 |
| POPT | 28 | -14.762 | -12.951 | -19.715 | -1.397 | -20.412 | 0.012 |
| SRRIP | 28 | -15.597 | -12.773 | -19.808 | -0.104 | -21.021 | 0.003 |
