# Literature faithfulness — ECG variants companion

**Sweep root:** `/tmp/graphbrew-lit-baseline`
**Total ECG-variant rows:** 78
**Policies emitted:** ['ECG_DBG_PRIMARY', 'POPT_CHARGED']

## Rows-per-policy

| policy | rows |
|---|---:|
| `ECG_DBG_PRIMARY` | 48 |
| `POPT_CHARGED` | 30 |

## All rows

| graph | app | L3 | policy | miss_rate | ΔvsLRU(pp) |
|---|---|---|---|---:|---:|
| cit-Patents | bc | 1MB | `ECG_DBG_PRIMARY` | 0.894961 | +1.0635 |
| cit-Patents | bc | 4MB | `ECG_DBG_PRIMARY` | 0.706543 | -2.5310 |
| cit-Patents | bc | 8MB | `ECG_DBG_PRIMARY` | 0.556870 | -5.0253 |
| cit-Patents | bfs | 1MB | `ECG_DBG_PRIMARY` | 0.954985 | -1.5823 |
| cit-Patents | bfs | 4MB | `ECG_DBG_PRIMARY` | 0.878178 | -4.8741 |
| cit-Patents | bfs | 8MB | `ECG_DBG_PRIMARY` | 0.814111 | -8.7955 |
| cit-Patents | pr | 1MB | `ECG_DBG_PRIMARY` | 0.786886 | -10.7552 |
| cit-Patents | pr | 4MB | `ECG_DBG_PRIMARY` | 0.441418 | -14.7922 |
| cit-Patents | pr | 8MB | `ECG_DBG_PRIMARY` | 0.211821 | -11.3760 |
| cit-Patents | sssp | 1MB | `ECG_DBG_PRIMARY` | 0.816636 | -3.2005 |
| cit-Patents | sssp | 4MB | `ECG_DBG_PRIMARY` | 0.468532 | -8.5053 |
| cit-Patents | sssp | 8MB | `ECG_DBG_PRIMARY` | 0.197287 | -6.3025 |
| com-orkut | pr | 1MB | `ECG_DBG_PRIMARY` | 0.652825 | -5.1903 |
| com-orkut | pr | 4MB | `ECG_DBG_PRIMARY` | 0.266468 | -4.2009 |
| com-orkut | pr | 8MB | `ECG_DBG_PRIMARY` | 0.129734 | -3.3874 |
| delaunay_n19 | pr | 16kB | `POPT_CHARGED` | 0.999959 | +0.0069 |
| delaunay_n19 | pr | 1MB | `POPT_CHARGED` | 0.791013 | -16.2916 |
| delaunay_n19 | pr | 256kB | `POPT_CHARGED` | 0.957698 | -3.4987 |
| delaunay_n19 | pr | 4kB | `POPT_CHARGED` | 0.999975 | +0.0124 |
| delaunay_n19 | pr | 64kB | `POPT_CHARGED` | 0.999958 | +0.0296 |
| email-Eu-core | pr | 1MB | `ECG_DBG_PRIMARY` | 1.000000 | +0.0000 |
| email-Eu-core | pr | 4MB | `ECG_DBG_PRIMARY` | 0.999532 | -0.0468 |
| email-Eu-core | pr | 8MB | `ECG_DBG_PRIMARY` | 1.000000 | +0.0000 |
| roadNet-CA | bc | 16kB | `POPT_CHARGED` | 0.999911 | +0.0106 |
| roadNet-CA | bc | 1MB | `POPT_CHARGED` | 0.797922 | +24.9960 |
| roadNet-CA | bc | 256kB | `POPT_CHARGED` | 0.999861 | +7.7760 |
| roadNet-CA | bc | 4kB | `POPT_CHARGED` | 0.999927 | +0.0082 |
| roadNet-CA | bc | 64kB | `POPT_CHARGED` | 0.999876 | +0.3372 |
| roadNet-CA | bfs | 16kB | `POPT_CHARGED` | 0.999961 | +0.0062 |
| roadNet-CA | bfs | 1MB | `POPT_CHARGED` | 0.612629 | +35.7277 |
| roadNet-CA | bfs | 256kB | `POPT_CHARGED` | 0.999951 | +10.8940 |
| roadNet-CA | bfs | 4kB | `POPT_CHARGED` | 0.999984 | +0.0046 |
| roadNet-CA | bfs | 64kB | `POPT_CHARGED` | 0.999956 | +0.2556 |
| roadNet-CA | cc | 16kB | `POPT_CHARGED` | 0.999998 | +0.2858 |
| roadNet-CA | cc | 1MB | `POPT_CHARGED` | 0.836561 | +9.3784 |
| roadNet-CA | cc | 256kB | `POPT_CHARGED` | 0.999996 | +8.4027 |
| roadNet-CA | cc | 4kB | `POPT_CHARGED` | 0.999998 | +0.1151 |
| roadNet-CA | cc | 64kB | `POPT_CHARGED` | 0.999995 | +2.0269 |
| roadNet-CA | pr | 16kB | `POPT_CHARGED` | 0.999933 | +0.0032 |
| roadNet-CA | pr | 1MB | `POPT_CHARGED` | 0.907316 | -3.4402 |
| roadNet-CA | pr | 256kB | `POPT_CHARGED` | 0.999920 | +1.4274 |
| roadNet-CA | pr | 4kB | `POPT_CHARGED` | 0.999959 | +0.0026 |
| roadNet-CA | pr | 64kB | `POPT_CHARGED` | 0.999920 | +0.0666 |
| roadNet-CA | sssp | 16kB | `POPT_CHARGED` | 0.999893 | +0.2026 |
| roadNet-CA | sssp | 1MB | `POPT_CHARGED` | 0.324639 | +14.8380 |
| roadNet-CA | sssp | 256kB | `POPT_CHARGED` | 0.999845 | +55.8956 |
| roadNet-CA | sssp | 4kB | `POPT_CHARGED` | 0.999935 | +0.0065 |
| roadNet-CA | sssp | 64kB | `POPT_CHARGED` | 0.999860 | +6.8095 |
| soc-LiveJournal1 | bc | 1MB | `ECG_DBG_PRIMARY` | 0.794139 | -4.5426 |
| soc-LiveJournal1 | bc | 4MB | `ECG_DBG_PRIMARY` | 0.546570 | -5.5520 |
| soc-LiveJournal1 | bc | 8MB | `ECG_DBG_PRIMARY` | 0.390909 | -5.9333 |
| soc-LiveJournal1 | bfs | 1MB | `ECG_DBG_PRIMARY` | 0.731359 | -8.6199 |
| soc-LiveJournal1 | bfs | 4MB | `ECG_DBG_PRIMARY` | 0.520543 | -6.8273 |
| soc-LiveJournal1 | bfs | 8MB | `ECG_DBG_PRIMARY` | 0.446406 | -6.4188 |
| soc-LiveJournal1 | pr | 1MB | `ECG_DBG_PRIMARY` | 0.682611 | -4.9316 |
| soc-LiveJournal1 | pr | 4MB | `ECG_DBG_PRIMARY` | 0.344526 | -8.1218 |
| soc-LiveJournal1 | pr | 8MB | `ECG_DBG_PRIMARY` | 0.201912 | -7.3831 |
| soc-LiveJournal1 | sssp | 1MB | `ECG_DBG_PRIMARY` | 0.656085 | -4.9732 |
| soc-LiveJournal1 | sssp | 4MB | `ECG_DBG_PRIMARY` | 0.293924 | -6.8799 |
| soc-LiveJournal1 | sssp | 8MB | `ECG_DBG_PRIMARY` | 0.123136 | -4.5564 |
| soc-pokec | bc | 1MB | `ECG_DBG_PRIMARY` | 0.769320 | -8.1932 |
| soc-pokec | bc | 4MB | `ECG_DBG_PRIMARY` | 0.419784 | -7.7738 |
| soc-pokec | bc | 8MB | `ECG_DBG_PRIMARY` | 0.198352 | -4.7059 |
| soc-pokec | pr | 1MB | `ECG_DBG_PRIMARY` | 0.541432 | -13.8131 |
| soc-pokec | pr | 4MB | `ECG_DBG_PRIMARY` | 0.130915 | -6.5998 |
| soc-pokec | pr | 8MB | `ECG_DBG_PRIMARY` | 0.079362 | -2.1032 |
| soc-pokec | sssp | 1MB | `ECG_DBG_PRIMARY` | 0.528358 | -10.6779 |
| soc-pokec | sssp | 4MB | `ECG_DBG_PRIMARY` | 0.067013 | -3.0617 |
| soc-pokec | sssp | 8MB | `ECG_DBG_PRIMARY` | 0.002917 | -0.0014 |
| web-Google | bc | 1MB | `ECG_DBG_PRIMARY` | 0.698902 | -0.0925 |
| web-Google | bc | 4MB | `ECG_DBG_PRIMARY` | 0.342560 | -2.6752 |
| web-Google | bc | 8MB | `ECG_DBG_PRIMARY` | 0.135450 | -1.4508 |
| web-Google | bfs | 1MB | `ECG_DBG_PRIMARY` | 0.930991 | -3.9451 |
| web-Google | bfs | 4MB | `ECG_DBG_PRIMARY` | 0.755354 | -18.1760 |
| web-Google | bfs | 8MB | `ECG_DBG_PRIMARY` | 0.719385 | -8.9686 |
| web-Google | pr | 1MB | `ECG_DBG_PRIMARY` | 0.445604 | -15.5234 |
| web-Google | pr | 4MB | `ECG_DBG_PRIMARY` | 0.107114 | -3.4317 |
| web-Google | pr | 8MB | `ECG_DBG_PRIMARY` | 0.084288 | -1.2760 |
