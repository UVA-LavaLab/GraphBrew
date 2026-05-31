# Gate 69 — Saturation distance vs capacity-sensitivity slope

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) >= 80 non-flat per-(app, graph, policy) cells matched, (2) Pearson r >= 0.4, (3) Spearman rho >= 0.4, (4) median (distance_pp / |slope_pp|) in [0.7, 1.3]. Cells with |slope_pp| < 0.05 are reported as flat_cells and excluded.

cells matched: **101** (min 80); flat cells excluded: 11 (|slope| < 0.05 pp/octave)

Pearson r  (distance vs |slope|): **0.4923** (floor 0.4)
Spearman ρ (distance vs |slope|): **0.4371** (floor 0.4)

median distance_pp:          13.348 pp
median |slope_pp|:           16.025 pp/octave
median ratio dist / |slope|: 0.9761 (band [0.7, 1.3])

## Per-cell pairs (top 30 by distance)

| app | graph | policy | distance pp | |slope| pp/oct | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| cc | soc-pokec | LRU | 33.838 | 20.502 | 1.651 |
| sssp | cit-Patents | LRU | 29.327 | 18.917 | 1.550 |
| sssp | cit-Patents | SRRIP | 29.223 | 19.596 | 1.491 |
| sssp | cit-Patents | GRASP | 27.489 | 20.179 | 1.362 |
| sssp | cit-Patents | POPT | 27.464 | 19.768 | 1.389 |
| cc | soc-pokec | SRRIP | 27.221 | 19.808 | 1.374 |
| pr | cit-Patents | SRRIP | 27.143 | 19.655 | 1.381 |
| pr | cit-Patents | LRU | 26.376 | 18.432 | 1.431 |
| bc | soc-pokec | LRU | 25.211 | 19.836 | 1.271 |
| bc | soc-pokec | SRRIP | 24.374 | 19.971 | 1.220 |
| pr | cit-Patents | GRASP | 23.982 | 18.696 | 1.283 |
| bc | soc-pokec | POPT | 23.539 | 19.550 | 1.204 |
| cc | soc-pokec | POPT | 22.480 | 20.412 | 1.101 |
| bc | soc-pokec | GRASP | 22.002 | 18.616 | 1.182 |
| bc | web-Google | LRU | 21.935 | 18.071 | 1.214 |
| bc | web-Google | SRRIP | 21.190 | 17.920 | 1.183 |
| bc | web-Google | POPT | 20.632 | 17.837 | 1.157 |
| bc | web-Google | GRASP | 20.626 | 18.954 | 1.088 |
| bc | com-orkut | LRU | 20.056 | 15.720 | 1.276 |
| bc | com-orkut | SRRIP | 19.714 | 15.872 | 1.242 |
| sssp | soc-LiveJournal1 | LRU | 19.402 | 17.797 | 1.090 |
| sssp | soc-LiveJournal1 | SRRIP | 19.104 | 17.622 | 1.084 |
| bc | com-orkut | GRASP | 18.831 | 15.227 | 1.237 |
| bc | com-orkut | POPT | 18.772 | 14.427 | 1.301 |
| cc | com-orkut | POPT | 18.735 | 13.334 | 1.405 |
| pr | cit-Patents | POPT | 18.441 | 19.694 | 0.936 |
| sssp | soc-LiveJournal1 | POPT | 17.701 | 17.548 | 1.009 |
| sssp | soc-LiveJournal1 | GRASP | 17.229 | 17.857 | 0.965 |
| cc | soc-pokec | GRASP | 16.909 | 17.045 | 0.992 |
| cc | com-orkut | SRRIP | 16.763 | 10.583 | 1.584 |
