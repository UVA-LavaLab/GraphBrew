# Headline coverage proof (gate 282)

**Scope:** `headline_1MB` — literature-derived from `literature_baselines.{INVARIANT,PER_GRAPH}_CLAIMS`.

## Totals

- Required cells: **225**
- Present (in scope): **75**
- Missing: **150**
- Coverage: **33.3 %**
- Extra rows on disk (out of scope): 575

## Per-simulator coverage

| sim | present | required | coverage |
|---|---:|---:|---:|
| `cache_sim` | 75 | 75 | 100.0 % |
| `gem5` | 0 | 75 | 0.0 % |
| `sniper` | 0 | 75 | 0.0 % |

## Missing cells by workstation tier

| tier | missing |
|---|---:|
| LOCAL | 40 |
| LOCAL_TIGHT | 60 |
| SLURM | 50 |

## Per (simulator, graph) coverage

| sim | graph | tier | present/req | coverage |
|---|---|---|---:|---:|
| `cache_sim` | `cit-Patents` | LOCAL | 20/20 | 100.0 % |
| `cache_sim` | `com-orkut` | SLURM | 5/5 | 100.0 % |
| `cache_sim` | `soc-LiveJournal1` | SLURM | 20/20 | 100.0 % |
| `cache_sim` | `soc-pokec` | LOCAL_TIGHT | 15/15 | 100.0 % |
| `cache_sim` | `web-Google` | LOCAL_TIGHT | 15/15 | 100.0 % |
| `gem5` | `cit-Patents` | LOCAL | 0/20 | 0.0 % |
| `gem5` | `com-orkut` | SLURM | 0/5 | 0.0 % |
| `gem5` | `soc-LiveJournal1` | SLURM | 0/20 | 0.0 % |
| `gem5` | `soc-pokec` | LOCAL_TIGHT | 0/15 | 0.0 % |
| `gem5` | `web-Google` | LOCAL_TIGHT | 0/15 | 0.0 % |
| `sniper` | `cit-Patents` | LOCAL | 0/20 | 0.0 % |
| `sniper` | `com-orkut` | SLURM | 0/5 | 0.0 % |
| `sniper` | `soc-LiveJournal1` | SLURM | 0/20 | 0.0 % |
| `sniper` | `soc-pokec` | LOCAL_TIGHT | 0/15 | 0.0 % |
| `sniper` | `web-Google` | LOCAL_TIGHT | 0/15 | 0.0 % |

## Workstation-runnable missing cells (100 cells)

| sim | graph | app | L3 | policy | prefetcher | tier |
|---|---|---|---|---|---|---|
| gem5 | cit-Patents | bc | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| gem5 | cit-Patents | bc | 1MB | GRASP | no_pfx | LOCAL |
| gem5 | cit-Patents | bc | 1MB | LRU | no_pfx | LOCAL |
| gem5 | cit-Patents | bc | 1MB | POPT | no_pfx | LOCAL |
| gem5 | cit-Patents | bc | 1MB | SRRIP | no_pfx | LOCAL |
| gem5 | cit-Patents | bfs | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| gem5 | cit-Patents | bfs | 1MB | GRASP | no_pfx | LOCAL |
| gem5 | cit-Patents | bfs | 1MB | LRU | no_pfx | LOCAL |
| gem5 | cit-Patents | bfs | 1MB | POPT | no_pfx | LOCAL |
| gem5 | cit-Patents | bfs | 1MB | SRRIP | no_pfx | LOCAL |
| gem5 | cit-Patents | pr | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| gem5 | cit-Patents | pr | 1MB | GRASP | no_pfx | LOCAL |
| gem5 | cit-Patents | pr | 1MB | LRU | no_pfx | LOCAL |
| gem5 | cit-Patents | pr | 1MB | POPT | no_pfx | LOCAL |
| gem5 | cit-Patents | pr | 1MB | SRRIP | no_pfx | LOCAL |
| gem5 | cit-Patents | sssp | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| gem5 | cit-Patents | sssp | 1MB | GRASP | no_pfx | LOCAL |
| gem5 | cit-Patents | sssp | 1MB | LRU | no_pfx | LOCAL |
| gem5 | cit-Patents | sssp | 1MB | POPT | no_pfx | LOCAL |
| gem5 | cit-Patents | sssp | 1MB | SRRIP | no_pfx | LOCAL |
| gem5 | soc-pokec | bc | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | bc | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | bc | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | bc | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | bc | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | pr | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | pr | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | pr | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | pr | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | pr | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | sssp | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | sssp | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | sssp | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | sssp | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| gem5 | soc-pokec | sssp | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bc | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bc | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bc | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bc | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bc | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bfs | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bfs | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bfs | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bfs | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | bfs | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | pr | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | pr | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | pr | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | pr | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| gem5 | web-Google | pr | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| sniper | cit-Patents | bc | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| sniper | cit-Patents | bc | 1MB | GRASP | no_pfx | LOCAL |
| sniper | cit-Patents | bc | 1MB | LRU | no_pfx | LOCAL |
| sniper | cit-Patents | bc | 1MB | POPT | no_pfx | LOCAL |
| sniper | cit-Patents | bc | 1MB | SRRIP | no_pfx | LOCAL |
| sniper | cit-Patents | bfs | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| sniper | cit-Patents | bfs | 1MB | GRASP | no_pfx | LOCAL |
| sniper | cit-Patents | bfs | 1MB | LRU | no_pfx | LOCAL |
| sniper | cit-Patents | bfs | 1MB | POPT | no_pfx | LOCAL |
| sniper | cit-Patents | bfs | 1MB | SRRIP | no_pfx | LOCAL |
| sniper | cit-Patents | pr | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| sniper | cit-Patents | pr | 1MB | GRASP | no_pfx | LOCAL |
| sniper | cit-Patents | pr | 1MB | LRU | no_pfx | LOCAL |
| sniper | cit-Patents | pr | 1MB | POPT | no_pfx | LOCAL |
| sniper | cit-Patents | pr | 1MB | SRRIP | no_pfx | LOCAL |
| sniper | cit-Patents | sssp | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL |
| sniper | cit-Patents | sssp | 1MB | GRASP | no_pfx | LOCAL |
| sniper | cit-Patents | sssp | 1MB | LRU | no_pfx | LOCAL |
| sniper | cit-Patents | sssp | 1MB | POPT | no_pfx | LOCAL |
| sniper | cit-Patents | sssp | 1MB | SRRIP | no_pfx | LOCAL |
| sniper | soc-pokec | bc | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | bc | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | bc | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | bc | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | bc | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | pr | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | pr | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | pr | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | pr | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | pr | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | sssp | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | sssp | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | sssp | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | sssp | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| sniper | soc-pokec | sssp | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bc | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bc | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bc | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bc | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bc | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bfs | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bfs | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bfs | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bfs | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | bfs | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | pr | 1MB | ECG_DBG_PRIMARY | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | pr | 1MB | GRASP | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | pr | 1MB | LRU | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | pr | 1MB | POPT | no_pfx | LOCAL_TIGHT |
| sniper | web-Google | pr | 1MB | SRRIP | no_pfx | LOCAL_TIGHT |
