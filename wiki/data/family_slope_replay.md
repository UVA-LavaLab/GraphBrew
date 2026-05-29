# Gate 67 — Per-family capacity-sensitivity slope replay

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff at least one family replays the pattern (LRU and SRRIP both strictly steeper than GRASP, every policy median < -5.0 pp/octave) AND no NEW deviating family appears beyond the pinned set.

qualifying families (full 1MB/4MB/8MB coverage, all 4 policies, at least one app): citation, social, web

replay count: **2 / 3**
deviating families (pinned): ['social']
deviating families (new):    []

## Per-family median slope by policy (pp / log2(L3 MB))

Smaller (more negative) = more cache-hungry. Help floor: -5.0 pp/octave.

| family | GRASP | POPT | LRU | SRRIP | n cells | replays? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| citation | -13.680 | -16.100 | -15.517 | -14.788 | 5 | yes |
| social | -14.651 | -13.996 | -14.048 | -14.258 | 18 | no |
| web | -16.025 | -16.478 | -18.296 | -17.018 | 5 | yes |
