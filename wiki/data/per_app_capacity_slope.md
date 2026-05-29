# Gate 68 — Per-app capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every (app, policy) median slope < 0, (2) no app outside the pinned set has GRASP more than 1.0 pp/octave steeper than LRU, and (3) at least one app has every policy median below -5.0 pp/octave.

apps observed: bc, bfs, cc, pr, sssp

most cache-hungry app:  **sssp** (median-of-medians -19.4709 pp/octave)

least cache-hungry app: **bfs** (median-of-medians -5.2283 pp/octave)

per-app median-of-medians range: 14.243 pp/octave

## Per-(app, policy) median slope (pp / log2(L3 MB))

| app | GRASP | POPT | LRU | SRRIP | n cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| bc | -14.263 | -13.925 | -14.292 | -14.533 | 6 |
| bfs | -6.405 | -6.471 | -3.993 | -4.051 | 6 |
| cc | -14.076 | -16.100 | -15.517 | -14.788 | 5 |
| pr | -16.187 | -15.636 | -18.033 | -16.882 | 6 |
| sssp | -19.411 | -19.575 | -19.267 | -19.531 | 5 |
