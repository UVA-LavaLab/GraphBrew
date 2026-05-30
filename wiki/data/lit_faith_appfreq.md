# Literature-faithfulness app-frequency audit (LIT-AppFreq)

Per-app axis-coverage check: each app must touch enough distinct graphs / L3 sizes / policies / observation rows for the downstream per-app comparators to remain well-defined, and the anchor app (pr) must cover the full corpus.

## Summary

| Metric | Value |
|---|---|
| Total per_observation rows | 456 |
| Distinct apps | 5 |
| Corpus graph count | 8 |
| Min graphs per app | 6 |
| Min L3 sizes per app | 3 |
| Min policies per app | 3 |
| Min L3 per (app, graph) | 3 |
| Min rows per app | 60 |
| Anchor app | pr |
| Violations | 0 |

## Per-app coverage

| app | graphs | L3 sizes | policies | rows |
|---|---|---|---|---:|
| bc | 7 (cit-Patents,com-orkut,email-Eu-core,roadNet-CA,soc-LiveJournal1,soc-pokec,web-Google) | 7 (16kB,1MB,256kB,4MB,4kB,64kB,8MB) | 4 (GRASP,LRU,POPT,SRRIP) | 92 |
| bfs | 7 (cit-Patents,com-orkut,email-Eu-core,roadNet-CA,soc-LiveJournal1,soc-pokec,web-Google) | 7 (16kB,1MB,256kB,4MB,4kB,64kB,8MB) | 4 (GRASP,LRU,POPT,SRRIP) | 92 |
| cc | 6 (cit-Patents,com-orkut,roadNet-CA,soc-LiveJournal1,soc-pokec,web-Google) | 7 (16kB,1MB,256kB,4MB,4kB,64kB,8MB) | 4 (GRASP,LRU,POPT,SRRIP) | 80 |
| pr | 8 (cit-Patents,com-orkut,delaunay_n19,email-Eu-core,roadNet-CA,soc-LiveJournal1,soc-pokec,web-Google) | 7 (16kB,1MB,256kB,4MB,4kB,64kB,8MB) | 4 (GRASP,LRU,POPT,SRRIP) | 112 |
| sssp | 6 (cit-Patents,com-orkut,roadNet-CA,soc-LiveJournal1,soc-pokec,web-Google) | 7 (16kB,1MB,256kB,4MB,4kB,64kB,8MB) | 4 (GRASP,LRU,POPT,SRRIP) | 80 |

## Per-(app, graph) L3 sweep

| app | graph | L3 count | L3 sizes |
|---|---|---:|---|
| bc | cit-Patents | 3 | 1MB,4MB,8MB |
| bc | com-orkut | 3 | 1MB,4MB,8MB |
| bc | email-Eu-core | 3 | 1MB,4MB,8MB |
| bc | roadNet-CA | 5 | 16kB,1MB,256kB,4kB,64kB |
| bc | soc-LiveJournal1 | 3 | 1MB,4MB,8MB |
| bc | soc-pokec | 3 | 1MB,4MB,8MB |
| bc | web-Google | 3 | 1MB,4MB,8MB |
| bfs | cit-Patents | 3 | 1MB,4MB,8MB |
| bfs | com-orkut | 3 | 1MB,4MB,8MB |
| bfs | email-Eu-core | 3 | 1MB,4MB,8MB |
| bfs | roadNet-CA | 5 | 16kB,1MB,256kB,4kB,64kB |
| bfs | soc-LiveJournal1 | 3 | 1MB,4MB,8MB |
| bfs | soc-pokec | 3 | 1MB,4MB,8MB |
| bfs | web-Google | 3 | 1MB,4MB,8MB |
| cc | cit-Patents | 3 | 1MB,4MB,8MB |
| cc | com-orkut | 3 | 1MB,4MB,8MB |
| cc | roadNet-CA | 5 | 16kB,1MB,256kB,4kB,64kB |
| cc | soc-LiveJournal1 | 3 | 1MB,4MB,8MB |
| cc | soc-pokec | 3 | 1MB,4MB,8MB |
| cc | web-Google | 3 | 1MB,4MB,8MB |
| pr | cit-Patents | 3 | 1MB,4MB,8MB |
| pr | com-orkut | 3 | 1MB,4MB,8MB |
| pr | delaunay_n19 | 5 | 16kB,1MB,256kB,4kB,64kB |
| pr | email-Eu-core | 3 | 1MB,4MB,8MB |
| pr | roadNet-CA | 5 | 16kB,1MB,256kB,4kB,64kB |
| pr | soc-LiveJournal1 | 3 | 1MB,4MB,8MB |
| pr | soc-pokec | 3 | 1MB,4MB,8MB |
| pr | web-Google | 3 | 1MB,4MB,8MB |
| sssp | cit-Patents | 3 | 1MB,4MB,8MB |
| sssp | com-orkut | 3 | 1MB,4MB,8MB |
| sssp | roadNet-CA | 5 | 16kB,1MB,256kB,4kB,64kB |
| sssp | soc-LiveJournal1 | 3 | 1MB,4MB,8MB |
| sssp | soc-pokec | 3 | 1MB,4MB,8MB |
| sssp | web-Google | 3 | 1MB,4MB,8MB |

_No violations._
