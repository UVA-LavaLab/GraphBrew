# Gate 65 — Per-app saturation distance at 4MB->8MB

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff (1) every cell with WSS > 4 MB has non-negative 4MB->8MB best-policy improvement, (2) email-Eu-core (pico-sentinel) is saturated for every app within 0.05 pp, and (3) per-app median saturation distance varies by at least 3.0 pp across apps.

cells measured: 28 across 5 apps

app-level diversity range: 12.2929 pp (threshold 3.0)

## Per-app saturation distance (pp of best-policy miss rate)

| app | n graphs | median pp | mean pp | p90 pp | max pp | min pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bc | 6 | 17.199 | 16.806 | 23.520 | 28.748 | 0.000 |
| bfs | 6 | 4.906 | 4.419 | 6.214 | 10.000 | 0.000 |
| cc | 5 | 11.060 | 11.324 | 30.396 | 30.396 | 0.000 |
| pr | 6 | 8.950 | 8.733 | 12.866 | 18.482 | 0.000 |
| sssp | 5 | 13.098 | 13.046 | 26.959 | 26.959 | 0.000 |

## All cells (4MB -> 8MB best-policy improvement)

| app | graph | best4 pp | best8 pp | distance pp | sentinel |
| --- | --- | ---: | ---: | ---: | :---: |
| bc | cit-Patents | 78.274 | 64.105 | 14.170 |  |
| bc | com-orkut | 52.504 | 34.060 | 18.444 |  |
| bc | email-Eu-core | 100.000 | 100.000 | 0.000 | pico |
| bc | soc-LiveJournal1 | 60.009 | 44.055 | 15.954 |  |
| bc | soc-pokec | 45.795 | 22.274 | 23.520 |  |
| bc | web-Google | 49.888 | 21.141 | 28.748 |  |
| bfs | cit-Patents | 90.266 | 85.853 | 4.412 |  |
| bfs | com-orkut | 98.119 | 97.630 | 0.489 |  |
| bfs | email-Eu-core | 100.000 | 100.000 | 0.000 | pico |
| bfs | soc-LiveJournal1 | 57.201 | 50.986 | 6.214 |  |
| bfs | soc-pokec | 72.599 | 62.599 | 10.000 |  |
| bfs | web-Google | 79.894 | 74.495 | 5.399 |  |
| cc | cit-Patents | 26.962 | 15.903 | 11.060 |  |
| cc | com-orkut | 42.049 | 39.584 | 2.465 |  |
| cc | soc-LiveJournal1 | 48.623 | 35.925 | 12.698 |  |
| cc | soc-pokec | 36.689 | 6.293 | 30.396 |  |
| cc | web-Google | 5.192 | 5.192 | 0.000 |  |
| pr | cit-Patents | 39.067 | 20.584 | 18.482 |  |
| pr | com-orkut | 26.131 | 13.264 | 12.866 |  |
| pr | email-Eu-core | 100.000 | 100.000 | 0.000 | pico |
| pr | soc-LiveJournal1 | 33.262 | 20.482 | 12.780 |  |
| pr | soc-pokec | 14.253 | 9.134 | 5.119 |  |
| pr | web-Google | 11.918 | 8.766 | 3.152 |  |
| sssp | cit-Patents | 49.781 | 22.822 | 26.959 |  |
| sssp | com-orkut | 14.620 | 1.522 | 13.098 |  |
| sssp | soc-LiveJournal1 | 31.834 | 13.987 | 17.847 |  |
| sssp | soc-pokec | 7.655 | 0.329 | 7.326 |  |
| sssp | web-Google | 2.348 | 2.348 | 0.000 |  |
