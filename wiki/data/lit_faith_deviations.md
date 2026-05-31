# Literature-faithfulness known-deviation completeness audit

Every (graph, app, l3, policy) cell in `KNOWN_DEVIATIONS` must carry a complete, quantitative, and anchored explanation; every live `status="known_deviation"` row must resolve to a whitelist entry; and the whitelist must not accumulate orphan entries that no live cell exercises.

## Summary

- KNOWN_DEVIATIONS entries: **38**
- Live lit-faith known_deviation rows: **24**
- Orphan whitelist entries (no live cell): **0**
- Inactive whitelist entries (live cell present, status ≠ `known_deviation`): **14**
- Live KD rows without whitelist entry: **0**
- Well-formed reasons (≥ 80 chars + quantitative phrase + anchor + not orphan): **38** / 38
- Reasons missing quantitative phrase: **0**
- Reasons missing anchor: **0**
- Short reasons (< 80 chars): **0**
- Coverage: **2** policies × **5** graphs × **5** apps × **3** L3 sizes

## Coverage breakdown

**By policy:**
- `POPT_GE_GRASP` → 34
- `POPT_NEAR_GRASP_IF_BIG_GAP` → 4

**By graph:**
- `cit-Patents` → 8
- `com-orkut` → 8
- `soc-LiveJournal1` → 10
- `soc-pokec` → 7
- `web-Google` → 5

**By app:**
- `bc` → 12
- `bfs` → 5
- `cc` → 16
- `pr` → 1
- `sssp` → 4

**By L3 size:**
- `1MB` → 15
- `4MB` → 14
- `8MB` → 9

## Vocabulary fingerprint

| term | reason count |
|---|---:|
| `Phase 1` | 5 |
| `P-OPT` | 8 |
| `GRASP` | 20 |
| `SRRIP` | 1 |
| `frontier` | 15 |
| `MPKI` | 0 |
| `design` | 3 |
| `sim` | 1 |
| `single-core` | 0 |
| `property array` | 1 |
| `HPCA` | 0 |

## Per-entry reason status

| graph | app | L3 | policy | len | quant | anchor | orphan | inactive | well-formed |
|---|---|---|---|---:|---|---|---|---|---|
| cit-Patents | bc | 4MB | POPT_GE_GRASP | 312 | ✓ | ✓ | — | ⚠ | ✓ |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | 158 | ✓ | ✓ | — | ⚠ | ✓ |
| cit-Patents | cc | 1MB | POPT_GE_GRASP | 241 | ✓ | ✓ | — | — | ✓ |
| cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | 169 | ✓ | ✓ | — | — | ✓ |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | 316 | ✓ | ✓ | — | — | ✓ |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | 161 | ✓ | ✓ | — | — | ✓ |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | 296 | ✓ | ✓ | — | ⚠ | ✓ |
| cit-Patents | sssp | 8MB | POPT_GE_GRASP | 136 | ✓ | ✓ | — | ⚠ | ✓ |
| com-orkut | bc | 1MB | POPT_GE_GRASP | 178 | ✓ | ✓ | — | — | ✓ |
| com-orkut | bc | 4MB | POPT_GE_GRASP | 328 | ✓ | ✓ | — | — | ✓ |
| com-orkut | bc | 8MB | POPT_GE_GRASP | 416 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 1MB | POPT_GE_GRASP | 151 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | 337 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 4MB | POPT_GE_GRASP | 291 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | 166 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 8MB | POPT_GE_GRASP | 173 | ✓ | ✓ | — | — | ✓ |
| soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | 238 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | 95 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | 421 | ✓ | ✓ | — | — | ✓ |
| soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | 430 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | 351 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | 294 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | 234 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | 279 | ✓ | ✓ | — | — | ✓ |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | 176 | ✓ | ✓ | — | — | ✓ |
| soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | 362 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-pokec | bc | 1MB | POPT_GE_GRASP | 397 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | bc | 4MB | POPT_GE_GRASP | 145 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | 354 | ✓ | ✓ | — | ⚠ | ✓ |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | 313 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | 236 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | cc | 4MB | POPT_GE_GRASP | 157 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | 445 | ✓ | ✓ | — | — | ✓ |
| web-Google | bc | 4MB | POPT_GE_GRASP | 247 | ✓ | ✓ | — | — | ✓ |
| web-Google | bc | 8MB | POPT_GE_GRASP | 263 | ✓ | ✓ | — | — | ✓ |
| web-Google | bfs | 1MB | POPT_GE_GRASP | 339 | ✓ | ✓ | — | ⚠ | ✓ |
| web-Google | cc | 1MB | POPT_GE_GRASP | 293 | ✓ | ✓ | — | ⚠ | ✓ |
| web-Google | pr | 4MB | POPT_GE_GRASP | 286 | ✓ | ✓ | — | — | ✓ |
