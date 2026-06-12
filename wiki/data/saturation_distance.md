# Gate 65 — Per-app saturation distance at 4MB->8MB

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every cell with WSS > 4 MB has non-negative 4MB->8MB best-policy improvement, (2) email-Eu-core (pico-sentinel) is saturated for every app within 0.05 pp, and (3) per-app median saturation distance varies by at least 3.0 pp across apps.

cells measured: 28 across 5 apps

app-level diversity range: 12.7523 pp (threshold 3.0)

## Per-app saturation distance (pp of best-policy miss rate)

| app | n graphs | median pp | mean pp | p90 pp | max pp | min pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bc | 6 | 17.199 | 16.891 | 24.028 | 28.748 | 0.000 |
| bfs | 6 | 4.447 | 4.151 | 6.736 | 8.241 | 0.000 |
| cc | 5 | 15.343 | 12.836 | 18.757 | 18.757 | 0.000 |
| pr | 6 | 6.918 | 7.477 | 11.110 | 16.858 | 0.000 |
| sssp | 5 | 12.785 | 13.100 | 27.095 | 27.095 | 0.000 |

## All cells (4MB -> 8MB best-policy improvement)

| app | graph | best4 pp | best8 pp | distance pp | sentinel |
| --- | --- | ---: | ---: | ---: | :---: |
| bc | cit-Patents | 78.274 | 64.105 | 14.170 |  |
| bc | com-orkut | 52.504 | 34.060 | 18.444 |  |
| bc | email-Eu-core | 100.000 | 100.000 | 0.000 | pico |
| bc | soc-LiveJournal1 | 60.009 | 44.055 | 15.954 |  |
| bc | soc-pokec | 45.795 | 21.767 | 24.028 |  |
| bc | web-Google | 49.888 | 21.141 | 28.748 |  |
| bfs | cit-Patents | 90.266 | 84.029 | 6.236 |  |
| bfs | com-orkut | 98.119 | 97.083 | 1.036 |  |
| bfs | email-Eu-core | 100.000 | 100.000 | 0.000 | pico |
| bfs | soc-LiveJournal1 | 57.201 | 50.465 | 6.736 |  |
| bfs | soc-pokec | 70.809 | 62.568 | 8.241 |  |
| bfs | web-Google | 76.433 | 73.776 | 2.657 |  |
| cc | cit-Patents | 26.573 | 12.613 | 13.959 |  |
| cc | com-orkut | 41.063 | 24.941 | 16.122 |  |
| cc | soc-LiveJournal1 | 47.069 | 31.726 | 15.343 |  |
| cc | soc-pokec | 25.050 | 6.293 | 18.757 |  |
| cc | web-Google | 5.192 | 5.192 | 0.000 |  |
| pr | cit-Patents | 34.741 | 17.883 | 16.858 |  |
| pr | com-orkut | 22.455 | 11.973 | 10.482 |  |
| pr | email-Eu-core | 100.000 | 100.000 | 0.000 | pico |
| pr | soc-LiveJournal1 | 30.041 | 18.931 | 11.110 |  |
| pr | soc-pokec | 12.488 | 9.134 | 3.354 |  |
| pr | web-Google | 11.492 | 8.435 | 3.057 |  |
| sssp | cit-Patents | 49.781 | 22.686 | 27.095 |  |
| sssp | com-orkut | 14.306 | 1.522 | 12.785 |  |
| sssp | soc-LiveJournal1 | 31.834 | 13.540 | 18.294 |  |
| sssp | soc-pokec | 7.655 | 0.329 | 7.326 |  |
| sssp | web-Google | 2.348 | 2.348 | 0.000 |  |
