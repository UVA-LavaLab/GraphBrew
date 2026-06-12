# Gate 68 — Per-app capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every (app, policy) median slope < 0, (2) no app outside the pinned set has GRASP more than 1.0 pp/octave steeper than LRU, and (3) at least one app has every policy median below -5.0 pp/octave.

apps observed: bc, bfs, cc, pr, sssp

most cache-hungry app:  **sssp** (median-of-medians -20.9986 pp/octave)

least cache-hungry app: **bfs** (median-of-medians -3.3845 pp/octave)

per-app median-of-medians range: 17.614 pp/octave

## Per-(app, policy) median slope (pp / log2(L3 MB))

| app | GRASP | POPT | LRU | SRRIP | n cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| bc | -13.886 | -12.871 | -13.450 | -13.891 | 6 |
| bfs | -2.266 | -5.903 | -3.397 | -3.372 | 6 |
| cc | -14.464 | -13.503 | -15.752 | -15.105 | 5 |
| pr | -15.695 | -15.122 | -18.006 | -17.154 | 6 |
| sssp | -19.647 | -20.172 | -22.828 | -21.825 | 5 |
