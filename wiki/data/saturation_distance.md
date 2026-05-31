# Gate 65 — Per-app saturation distance at 4MB->8MB

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every cell with WSS > 4 MB has non-negative 4MB->8MB best-policy improvement, (2) email-Eu-core (pico-sentinel) is saturated for every app within 0.05 pp, and (3) per-app median saturation distance varies by at least 3.0 pp across apps.

cells measured: 28 across 5 apps

app-level diversity range: 12.5486 pp (threshold 3.0)

## Per-app saturation distance (pp of best-policy miss rate)

| app | n graphs | median pp | mean pp | p90 pp | max pp | min pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bc | 6 | 17.332 | 15.444 | 20.626 | 22.002 | 0.000 |
| bfs | 6 | 4.783 | 4.569 | 7.790 | 8.085 | 0.002 |
| cc | 5 | 14.868 | 12.123 | 16.909 | 16.909 | 0.004 |
| pr | 6 | 7.422 | 7.745 | 11.027 | 18.441 | 0.000 |
| sssp | 5 | 12.495 | 12.721 | 27.489 | 27.489 | 0.002 |

## All cells (4MB -> 8MB best-policy improvement)

| app | graph | best4 pp | best8 pp | distance pp | sentinel |
| --- | --- | ---: | ---: | ---: | :---: |
| bc | cit-Patents | 71.124 | 55.753 | 15.370 |  |
| bc | com-orkut | 52.422 | 33.590 | 18.831 |  |
| bc | email-Eu-core | 0.012 | 0.012 | 0.000 | pico |
| bc | soc-LiveJournal1 | 54.923 | 39.090 | 15.833 |  |
| bc | soc-pokec | 41.941 | 19.939 | 22.002 |  |
| bc | web-Google | 34.279 | 13.652 | 20.626 |  |
| bfs | cit-Patents | 88.463 | 81.529 | 6.934 |  |
| bfs | com-orkut | 97.451 | 95.479 | 1.972 |  |
| bfs | email-Eu-core | 4.585 | 4.583 | 0.002 | pico |
| bfs | soc-LiveJournal1 | 52.524 | 44.734 | 7.790 |  |
| bfs | soc-pokec | 67.361 | 59.276 | 8.085 |  |
| bfs | web-Google | 75.558 | 72.925 | 2.633 |  |
| cc | cit-Patents | 24.730 | 11.709 | 13.021 |  |
| cc | com-orkut | 36.984 | 22.116 | 14.868 |  |
| cc | soc-LiveJournal1 | 47.740 | 31.927 | 15.813 |  |
| cc | soc-pokec | 22.640 | 5.731 | 16.909 |  |
| cc | web-Google | 4.568 | 4.564 | 0.004 |  |
| pr | cit-Patents | 36.739 | 18.298 | 18.441 |  |
| pr | com-orkut | 21.238 | 10.827 | 10.411 |  |
| pr | email-Eu-core | 100.000 | 100.000 | 0.000 | pico |
| pr | soc-LiveJournal1 | 28.861 | 17.833 | 11.027 |  |
| pr | soc-pokec | 12.357 | 7.924 | 4.433 |  |
| pr | web-Google | 10.569 | 8.413 | 2.156 |  |
| sssp | cit-Patents | 47.291 | 19.801 | 27.489 |  |
| sssp | com-orkut | 14.727 | 2.232 | 12.495 |  |
| sssp | soc-LiveJournal1 | 29.629 | 12.401 | 17.229 |  |
| sssp | soc-pokec | 6.683 | 0.291 | 6.391 |  |
| sssp | web-Google | 1.628 | 1.626 | 0.002 |  |
