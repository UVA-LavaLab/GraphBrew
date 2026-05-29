# Gate 69 — Saturation distance vs capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) >= 80 non-flat per-(app, graph, policy) cells matched, (2) Pearson r >= 0.4, (3) Spearman rho >= 0.4, (4) median (distance_pp / |slope_pp|) in [0.7, 1.3]. Cells with |slope_pp| < 0.05 are reported as flat_cells and excluded.

cells matched: **102** (min 80); flat cells excluded: 10 (|slope| < 0.05 pp/octave)

Pearson r  (distance vs |slope|): **0.5065** (floor 0.4)
Spearman ρ (distance vs |slope|): **0.4530** (floor 0.4)

median distance_pp:          13.197 pp
median |slope_pp|:           15.966 pp/octave
median ratio dist / |slope|: 0.9639 (band [0.7, 1.3])

## Per-cell pairs (top 30 by distance)

| app | graph | policy | distance pp | |slope| pp/oct | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| cc | soc-pokec | LRU | 33.838 | 20.502 | 1.651 |
| sssp | cit-Patents | SRRIP | 29.448 | 19.531 | 1.508 |
| sssp | cit-Patents | LRU | 29.414 | 18.755 | 1.568 |
| sssp | cit-Patents | POPT | 27.816 | 19.827 | 1.403 |
| sssp | cit-Patents | GRASP | 27.645 | 20.114 | 1.374 |
| cc | soc-pokec | SRRIP | 27.221 | 19.808 | 1.374 |
| pr | cit-Patents | SRRIP | 27.051 | 19.583 | 1.381 |
| pr | cit-Patents | LRU | 26.317 | 18.389 | 1.431 |
| bc | soc-pokec | LRU | 25.744 | 19.895 | 1.294 |
| bc | soc-pokec | SRRIP | 24.547 | 19.937 | 1.231 |
| pr | cit-Patents | GRASP | 23.972 | 18.692 | 1.282 |
| bc | soc-pokec | POPT | 23.541 | 19.522 | 1.206 |
| cc | soc-pokec | POPT | 22.480 | 20.412 | 1.101 |
| bc | soc-pokec | GRASP | 22.108 | 18.626 | 1.187 |
| bc | web-Google | LRU | 21.719 | 18.296 | 1.187 |
| bc | web-Google | SRRIP | 21.301 | 17.924 | 1.188 |
| bc | web-Google | GRASP | 20.610 | 18.948 | 1.088 |
| bc | web-Google | POPT | 20.462 | 17.686 | 1.157 |
| bc | com-orkut | LRU | 20.056 | 15.720 | 1.276 |
| bc | com-orkut | SRRIP | 19.714 | 15.872 | 1.242 |
| sssp | soc-LiveJournal1 | SRRIP | 19.217 | 17.803 | 1.079 |
| sssp | soc-LiveJournal1 | LRU | 19.097 | 17.315 | 1.103 |
| bc | com-orkut | GRASP | 18.831 | 15.227 | 1.237 |
| bc | com-orkut | POPT | 18.772 | 14.427 | 1.301 |
| cc | com-orkut | POPT | 18.735 | 13.334 | 1.405 |
| pr | cit-Patents | POPT | 18.561 | 19.715 | 0.942 |
| sssp | soc-LiveJournal1 | GRASP | 18.007 | 19.411 | 0.928 |
| sssp | soc-LiveJournal1 | POPT | 17.760 | 17.962 | 0.989 |
| cc | soc-pokec | GRASP | 16.909 | 17.045 | 0.992 |
| cc | com-orkut | SRRIP | 16.763 | 10.583 | 1.584 |
