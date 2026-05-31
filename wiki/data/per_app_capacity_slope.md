# Gate 68 — Per-app capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every (app, policy) median slope < 0, (2) no app outside the pinned set has GRASP more than 1.0 pp/octave steeper than LRU, and (3) at least one app has every policy median below -5.0 pp/octave.

apps observed: bc, bfs, cc, pr, sssp

most cache-hungry app:  **sssp** (median-of-medians -19.4173 pp/octave)

least cache-hungry app: **bfs** (median-of-medians -4.824 pp/octave)

per-app median-of-medians range: 14.593 pp/octave

## Per-(app, policy) median slope (pp / log2(L3 MB))

| app | GRASP | POPT | LRU | SRRIP | n cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| bc | -14.261 | -13.839 | -14.270 | -14.406 | 6 |
| bfs | -6.408 | -5.995 | -3.563 | -3.653 | 6 |
| cc | -14.076 | -16.100 | -15.517 | -14.788 | 5 |
| pr | -16.191 | -15.614 | -17.983 | -16.806 | 6 |
| sssp | -19.280 | -19.555 | -19.267 | -19.596 | 5 |
