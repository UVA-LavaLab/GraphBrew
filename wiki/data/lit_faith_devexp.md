# Literature-faithfulness deviation-explanation audit (LIT-DevExp)

Per `status == 'known_deviation'` row: the `known_deviation_reason` text must name at least one algorithmic mechanism, exceed a length floor, carry a non-empty citation, resolve any cross-references, and the same reason text may not cover more than half the rows.

## Summary

| Metric | Value |
|---|---|
| Known-deviation rows | 37 |
| Min reason length | 60 chars |
| Min mechanism hits | 1 |
| Reuse ceiling fraction | 0.5 |
| Median reason length | 396 chars |
| Median mechanism hits | 2 |
| Unique reason texts | 37 |
| Max reuse count (single text) | 1 |
| Rows with cross-reference | 0 |
| Violations | 0 |

## Per-row detail

| graph | app | l3 | policy | len | mech hits | xref | excerpt |
|---|---|---|---|---|---|---|---|
| cit-Patents | bc | 4MB | POPT_GE_GRASP | 397 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/bc@4MB (Δ=+... |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | 270 | 2 | N | cit-Patents/bc/8 MB: in P-OPT Phase 1, BC's dependency frontier is unusually bur... |
| cit-Patents | bfs | 4MB | POPT_GE_GRASP | 398 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/bfs@4MB (Δ=... |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | 397 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/cc@4MB (Δ=+... |
| cit-Patents | sssp | 1MB | POPT_GE_GRASP | 399 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/sssp@1MB (Δ... |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | 399 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/sssp@4MB (Δ... |
| cit-Patents | sssp | 8MB | POPT_GE_GRASP | 399 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/sssp@8MB (Δ... |
| com-orkut | bc | 1MB | POPT_GE_GRASP | 395 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on com-orkut/bc@1MB (Δ=+2.... |
| com-orkut | bc | 4MB | POPT_GE_GRASP | 250 | 4 | N | com-orkut/bc/4 MB: the BC frontier expands through very high-degree communities,... |
| com-orkut | bc | 8MB | POPT_GE_GRASP | 246 | 3 | N | com-orkut/bc/8 MB: at the larger LLC, BC's frontier rereference stream still jum... |
| com-orkut | cc | 1MB | POPT_GE_GRASP | 225 | 5 | N | com-orkut/cc/1 MB: the union-find working set is capacity-pinched, and edge-driv... |
| com-orkut | cc | 4MB | POPT_GE_GRASP | 239 | 3 | N | com-orkut/cc/4 MB: CC's edge-driven union-find accesses phase-shift away from P-... |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | 244 | 3 | N | com-orkut/cc/4 MB near-GRASP check: this GRASP-strong phase has edge-driven unio... |
| com-orkut | pr | 1MB | POPT_GE_GRASP | 395 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on com-orkut/pr@1MB (Δ=+3.... |
| com-orkut | sssp | 1MB | POPT_GE_GRASP | 254 | 3 | N | com-orkut/sssp/1 MB: delta-stepping frontier buckets revisit vertices irregularl... |
| com-orkut | sssp | 4MB | POPT_GE_GRASP | 397 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on com-orkut/sssp@4MB (Δ=+... |
| soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | 402 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bc@4MB... |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | 402 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bc@8MB... |
| soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | 403 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bfs@1M... |
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | 403 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bfs@4M... |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | 244 | 5 | N | soc-LiveJournal1/cc/4 MB: social-graph CC creates union-find rereference cluster... |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | 265 | 3 | N | soc-LiveJournal1/cc/8 MB: with more cache, union-find still produces edge-driven... |
| soc-LiveJournal1 | sssp | 1MB | POPT_GE_GRASP | 404 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/sssp@1... |
| soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | 404 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/sssp@4... |
| soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | 404 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/sssp@8... |
| soc-pokec | bc | 1MB | POPT_GE_GRASP | 395 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bc@1MB (Δ=+2.... |
| soc-pokec | bc | 4MB | POPT_GE_GRASP | 395 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bc@4MB (Δ=+4.... |
| soc-pokec | bc | 8MB | POPT_GE_GRASP | 395 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bc@8MB (Δ=+4.... |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | 396 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bfs@1MB (Δ=+2... |
| soc-pokec | pr | 1MB | POPT_GE_GRASP | 395 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/pr@1MB (Δ=+2.... |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | 231 | 2 | N | soc-pokec/sssp/1 MB: delta-stepping's frontier buckets thrash the small LLC, and... |
| soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | 397 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/sssp@1MB (Δ=+... |
| soc-pokec | sssp | 4MB | POPT_GE_GRASP | 397 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/sssp@4MB (Δ=+... |
| web-Google | bc | 4MB | POPT_GE_GRASP | 252 | 2 | N | web-Google/bc/4 MB: web-graph BC alternates frontier waves with sparse back-depe... |
| web-Google | bc | 8MB | POPT_GE_GRASP | 214 | 2 | N | web-Google/bc/8 MB: larger-cache BC still has frontier rereference gaps on the w... |
| web-Google | bfs | 1MB | POPT_GE_GRASP | 397 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on web-Google/bfs@1MB (Δ=+... |
| web-Google | sssp | 1MB | POPT_GE_GRASP | 398 | 2 | N | INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on web-Google/sssp@1MB (Δ=... |

_No violations._
