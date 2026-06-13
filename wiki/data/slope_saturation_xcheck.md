# Gate 69 — Saturation distance vs capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) >= 80 non-flat per-(app, graph, policy) cells matched, (2) Pearson r >= 0.4, (3) Spearman rho >= 0.35, (4) median (distance_pp / |slope_pp|) in [0.7, 1.3]. Cells with |slope_pp| < 0.05 are reported as flat_cells and excluded.

cells matched: **100** (min 80); flat cells excluded: 12 (|slope| < 0.05 pp/octave)

Pearson r  (distance vs |slope|): **0.4833** (floor 0.4)
Spearman ρ (distance vs |slope|): **0.4190** (floor 0.35)

median distance_pp:          12.894 pp
median |slope_pp|:           15.623 pp/octave
median ratio dist / |slope|: 0.9495 (band [0.7, 1.3])

## Per-cell pairs (top 30 by distance)

| app | graph | policy | distance pp | |slope| pp/oct | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| cc | soc-pokec | LRU | 40.089 | 20.893 | 1.919 |
| cc | soc-pokec | SRRIP | 31.628 | 20.538 | 1.540 |
| cc | soc-pokec | POPT | 30.396 | 19.274 | 1.577 |
| sssp | cit-Patents | LRU | 29.726 | 19.518 | 1.523 |
| sssp | cit-Patents | SRRIP | 29.686 | 20.051 | 1.480 |
| bc | web-Google | LRU | 29.516 | 18.666 | 1.581 |
| bc | web-Google | SRRIP | 28.748 | 19.216 | 1.496 |
| pr | cit-Patents | SRRIP | 27.213 | 19.691 | 1.382 |
| sssp | cit-Patents | GRASP | 26.959 | 19.697 | 1.369 |
| sssp | cit-Patents | POPT | 26.803 | 18.559 | 1.444 |
| pr | cit-Patents | LRU | 26.385 | 18.433 | 1.431 |
| bc | soc-pokec | LRU | 26.233 | 19.269 | 1.361 |
| bc | soc-pokec | SRRIP | 25.523 | 19.711 | 1.295 |
| bc | web-Google | GRASP | 25.514 | 18.188 | 1.403 |
| bc | web-Google | POPT | 24.063 | 15.975 | 1.506 |
| bc | soc-pokec | POPT | 23.989 | 18.559 | 1.293 |
| bc | soc-pokec | GRASP | 23.520 | 19.322 | 1.217 |
| pr | cit-Patents | GRASP | 22.839 | 19.561 | 1.168 |
| sssp | soc-LiveJournal1 | LRU | 21.395 | 18.981 | 1.127 |
| sssp | soc-LiveJournal1 | SRRIP | 20.195 | 18.766 | 1.076 |
| bc | com-orkut | LRU | 19.748 | 15.049 | 1.312 |
| bc | com-orkut | SRRIP | 19.712 | 15.401 | 1.280 |
| sssp | soc-LiveJournal1 | POPT | 18.897 | 17.646 | 1.071 |
| pr | cit-Patents | POPT | 18.482 | 18.509 | 0.999 |
| bc | com-orkut | GRASP | 18.444 | 14.961 | 1.233 |
| sssp | soc-LiveJournal1 | GRASP | 17.847 | 18.088 | 0.987 |
| pr | soc-LiveJournal1 | LRU | 16.174 | 15.469 | 1.046 |
| pr | soc-LiveJournal1 | SRRIP | 16.071 | 15.800 | 1.017 |
| pr | com-orkut | LRU | 16.004 | 18.749 | 0.854 |
| bc | com-orkut | POPT | 16.003 | 12.898 | 1.241 |
