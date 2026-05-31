# Literature-faithfulness deviation-explanation audit (LIT-DevExp)

Per `status == 'known_deviation'` row: the `known_deviation_reason` text must name at least one algorithmic mechanism, exceed a length floor, carry a non-empty citation, resolve any cross-references, and the same reason text may not cover more than half the rows.

## Summary

| Metric | Value |
|---|---|
| Known-deviation rows | 24 |
| Min reason length | 60 chars |
| Min mechanism hits | 1 |
| Reuse ceiling fraction | 0.5 |
| Median reason length | 263 chars |
| Median mechanism hits | 5 |
| Unique reason texts | 24 |
| Max reuse count (single text) | 1 |
| Rows with cross-reference | 12 |
| Violations | 0 |

## Per-row detail

| graph | app | l3 | policy | len | mech hits | xref | excerpt |
|---|---|---|---|---|---|---|---|
| cit-Patents | cc | 1MB | POPT_GE_GRASP | 241 | 4 | Y | Same CC/POPT algorithmic mismatch as the soc-pokec/web-Google entries above. cit... |
| cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | 169 | 1 | N | Phase-transition regime invariant fires because GRASP gains 13+ pp over LRU on c... |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | 316 | 10 | Y | Same CC/POPT algorithmic mismatch as the 1MB entry above: CC's union-find parent... |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | 161 | 4 | N | Same CC/POPT mismatch; ~1.5 pp gap remains even at 8 MB on cit-Patents because t... |
| com-orkut | bc | 1MB | POPT_GE_GRASP | 178 | 2 | N | Same BC/PR-rank mismatch as soc-LJ; com-orkut has even higher clustering coeffic... |
| com-orkut | bc | 4MB | POPT_GE_GRASP | 328 | 5 | Y | Same com-orkut/BC PR-rank vs dependency-frontier mismatch as the 1MB entry; gap ... |
| com-orkut | bc | 8MB | POPT_GE_GRASP | 416 | 5 | Y | Same com-orkut/BC PR-rank vs dependency-frontier mismatch as the 1MB / 4MB entri... |
| com-orkut | cc | 1MB | POPT_GE_GRASP | 151 | 3 | N | Same CC/POPT mismatch as soc-pokec/cit-Patents CC entries; com-orkut shows the l... |
| com-orkut | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | 337 | 8 | N | Phase-transition regime invariant fires because GRASP gains 13+ pp over LRU on c... |
| com-orkut | cc | 4MB | POPT_GE_GRASP | 291 | 8 | Y | Same CC/POPT algorithmic mismatch as the com-orkut/cc/1MB entry above: union-fin... |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | 166 | 2 | Y | Same com-orkut/CC mismatch as the 1MB entry; gap persists at ~10 pp at 4MB becau... |
| com-orkut | cc | 8MB | POPT_GE_GRASP | 173 | 6 | N | Same CC/POPT mismatch; ~6 pp gap remains even at 8 MB on com-orkut because the s... |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | 421 | 5 | Y | Same soc-LJ/BC PR-rank vs dependency-frontier mismatch as the 1MB / 4MB entries:... |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | 279 | 8 | Y | Same CC/POPT algorithmic mismatch as the soc-LiveJournal1/cc/1MB entry above: un... |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | 176 | 5 | N | Same soc-LJ/CC mismatch; gap is widest (+3.1 pp) at 8MB where GRASP's locality p... |
| soc-pokec | bc | 1MB | POPT_GE_GRASP | 397 | 6 | N | BC's forward + backward sweeps from `-r 0` (highest-PR hub) expand a frontier wh... |
| soc-pokec | bc | 4MB | POPT_GE_GRASP | 145 | 7 | Y | Same source-rooted frontier vs PR-rank mis-alignment as the 1 MB entry; ~1.7 pp ... |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | 313 | 4 | N | CC's parent[] access pattern is edge-driven, not PageRank-driven, so P-OPT's off... |
| soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | 236 | 4 | Y | Phase-transition regime invariant fires because GRASP gains 13.1 pp over LRU. PO... |
| soc-pokec | cc | 4MB | POPT_GE_GRASP | 157 | 1 | Y | Same CC/POPT mismatch as the soc-pokec/cc/1MB entry above; the gap narrows to ~5... |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | 445 | 9 | N | Same frontier-vs-rank mis-alignment as cit-Patents/SSSP. The non-hub source (`-r... |
| web-Google | bc | 4MB | POPT_GE_GRASP | 247 | 2 | Y | Same Phase-1 root cause as the PR/4MB entry above. BC has four vertex-indexed pr... |
| web-Google | bc | 8MB | POPT_GE_GRASP | 263 | 1 | N | BC working set on web-Google (~12 MB across 4 property arrays) spills 4 MB at L3... |
| web-Google | pr | 4MB | POPT_GE_GRASP | 286 | 3 | N | POPT Phase 1 aggressively evicts non-property cache lines (CSR offsets, frontier... |

_No violations._
