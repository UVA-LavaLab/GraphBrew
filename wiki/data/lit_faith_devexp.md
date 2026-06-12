# Literature-faithfulness deviation-explanation audit (LIT-DevExp)

Per `status == 'known_deviation'` row: the `known_deviation_reason` text must name at least one algorithmic mechanism, exceed a length floor, carry a non-empty citation, resolve any cross-references, and the same reason text may not cover more than half the rows.

## Summary

| Metric | Value |
|---|---|
| Known-deviation rows | 17 |
| Min reason length | 60 chars |
| Min mechanism hits | 1 |
| Reuse ceiling fraction | 0.5 |
| Median reason length | 246 chars |
| Median mechanism hits | 3 |
| Unique reason texts | 17 |
| Max reuse count (single text) | 1 |
| Rows with cross-reference | 0 |
| Violations | 0 |

## Per-row detail

| graph | app | l3 | policy | len | mech hits | xref | excerpt |
|---|---|---|---|---|---|---|---|
| cit-Patents | bc | 8MB | POPT_GE_GRASP | 270 | 2 | N | cit-Patents/bc/8 MB: in P-OPT Phase 1, BC's dependency frontier is unusually bur... |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | 252 | 3 | N | cit-Patents/cc/8 MB: CC's union-find probes are edge-driven rather than rank-sta... |
| com-orkut | bc | 4MB | POPT_GE_GRASP | 250 | 4 | N | com-orkut/bc/4 MB: the BC frontier expands through very high-degree communities,... |
| com-orkut | bc | 8MB | POPT_GE_GRASP | 246 | 3 | N | com-orkut/bc/8 MB: at the larger LLC, BC's frontier rereference stream still jum... |
| com-orkut | cc | 1MB | POPT_GE_GRASP | 225 | 5 | N | com-orkut/cc/1 MB: the union-find working set is capacity-pinched, and edge-driv... |
| com-orkut | cc | 4MB | POPT_GE_GRASP | 239 | 3 | N | com-orkut/cc/4 MB: CC's edge-driven union-find accesses phase-shift away from P-... |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | 244 | 3 | N | com-orkut/cc/4 MB near-GRASP check: this GRASP-strong phase has edge-driven unio... |
| com-orkut | cc | 8MB | POPT_GE_GRASP | 251 | 4 | N | com-orkut/cc/8 MB: even after capacity pressure eases, union-find rereference lo... |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | 248 | 5 | N | com-orkut/cc/8 MB near-GRASP check: the union-find phase transition leaves a mea... |
| com-orkut | sssp | 1MB | POPT_GE_GRASP | 254 | 3 | N | com-orkut/sssp/1 MB: delta-stepping frontier buckets revisit vertices irregularl... |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | 244 | 5 | N | soc-LiveJournal1/cc/4 MB: social-graph CC creates union-find rereference cluster... |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | 265 | 3 | N | soc-LiveJournal1/cc/8 MB: with more cache, union-find still produces edge-driven... |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | 246 | 3 | N | soc-pokec/cc/1 MB: the tight cache exposes CC's edge-driven union-find rereferen... |
| soc-pokec | cc | 4MB | POPT_GE_GRASP | 232 | 3 | N | soc-pokec/cc/4 MB: component merging creates a union-find rereference pattern th... |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | 231 | 2 | N | soc-pokec/sssp/1 MB: delta-stepping's frontier buckets thrash the small LLC, and... |
| web-Google | bc | 4MB | POPT_GE_GRASP | 252 | 2 | N | web-Google/bc/4 MB: web-graph BC alternates frontier waves with sparse back-depe... |
| web-Google | bc | 8MB | POPT_GE_GRASP | 214 | 2 | N | web-Google/bc/8 MB: larger-cache BC still has frontier rereference gaps on the w... |

_No violations._
