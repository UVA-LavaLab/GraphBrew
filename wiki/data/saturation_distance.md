# Gate 65 — Per-app saturation distance at 4MB->8MB

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every cell with WSS > 4 MB has non-negative 4MB->8MB best-policy improvement, (2) email-Eu-core (pico-sentinel) is saturated for every app within 0.05 pp, and (3) per-app median saturation distance varies by at least 3.0 pp across apps.

cells measured: 28 across 5 apps

app-level diversity range: 12.8058 pp (threshold 3.0)

## Per-app saturation distance (pp of best-policy miss rate)

| app | n graphs | median pp | mean pp | p90 pp | max pp | min pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bc | 6 | 17.450 | 15.552 | 20.610 | 22.108 | 0.000 |
| bfs | 6 | 4.644 | 4.627 | 8.085 | 8.418 | 0.002 |
| cc | 5 | 14.868 | 12.123 | 16.909 | 16.909 | 0.004 |
| pr | 6 | 7.544 | 7.833 | 11.204 | 18.561 | -0.003 |
| sssp | 5 | 12.495 | 12.927 | 27.645 | 27.645 | 0.002 |

## All cells (4MB -> 8MB best-policy improvement)

| app | graph | best4 pp | best8 pp | distance pp | sentinel |
| --- | --- | ---: | ---: | ---: | :---: |
| bc | cit-Patents | 72.087 | 56.395 | 15.693 |  |
| bc | com-orkut | 52.422 | 33.590 | 18.831 |  |
| bc | email-Eu-core | 0.012 | 0.012 | 0.000 | pico |
| bc | soc-LiveJournal1 | 55.107 | 39.039 | 16.068 |  |
| bc | soc-pokec | 42.046 | 19.938 | 22.108 |  |
| bc | web-Google | 34.180 | 13.570 | 20.610 |  |
| bfs | cit-Patents | 86.901 | 78.483 | 8.418 |  |
| bfs | com-orkut | 97.451 | 95.479 | 1.972 |  |
| bfs | email-Eu-core | 4.585 | 4.583 | 0.002 | pico |
| bfs | soc-LiveJournal1 | 58.858 | 52.147 | 6.711 |  |
| bfs | soc-pokec | 67.361 | 59.276 | 8.085 |  |
| bfs | web-Google | 75.518 | 72.942 | 2.576 |  |
| cc | cit-Patents | 24.730 | 11.709 | 13.021 |  |
| cc | com-orkut | 36.984 | 22.116 | 14.868 |  |
| cc | soc-LiveJournal1 | 47.740 | 31.927 | 15.813 |  |
| cc | soc-pokec | 22.640 | 5.731 | 16.909 |  |
| cc | web-Google | 4.568 | 4.564 | 0.004 |  |
| pr | cit-Patents | 37.178 | 18.618 | 18.561 |  |
| pr | com-orkut | 21.575 | 10.937 | 10.638 |  |
| pr | email-Eu-core | 0.756 | 0.759 | -0.003 | pico |
| pr | soc-LiveJournal1 | 29.169 | 17.965 | 11.204 |  |
| pr | soc-pokec | 12.380 | 7.929 | 4.451 |  |
| pr | web-Google | 10.559 | 8.413 | 2.146 |  |
| sssp | cit-Patents | 47.637 | 19.991 | 27.645 |  |
| sssp | com-orkut | 14.727 | 2.232 | 12.495 |  |
| sssp | soc-LiveJournal1 | 30.888 | 12.881 | 18.007 |  |
| sssp | soc-pokec | 6.782 | 0.294 | 6.487 |  |
| sssp | web-Google | 1.628 | 1.626 | 0.002 |  |
