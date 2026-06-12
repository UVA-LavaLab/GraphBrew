# Literature-faithfulness known-deviation completeness audit

Every (graph, app, l3, policy) cell in `KNOWN_DEVIATIONS` must carry a complete, quantitative, and anchored explanation; every live `status="known_deviation"` row must resolve to a whitelist entry; and the whitelist must not accumulate orphan entries that no live cell exercises.

## Summary

- KNOWN_DEVIATIONS entries: **17**
- Live lit-faith known_deviation rows: **17**
- Orphan whitelist entries (no live cell): **0**
- Inactive whitelist entries (live cell present, status ≠ `known_deviation`): **0**
- Live KD rows without whitelist entry: **0**
- Well-formed reasons (≥ 80 chars + quantitative phrase + anchor + not orphan): **17** / 17
- Reasons missing quantitative phrase: **0**
- Reasons missing anchor: **0**
- Short reasons (< 80 chars): **0**
- Coverage: **2** policies × **5** graphs × **3** apps × **3** L3 sizes

## Coverage breakdown

**By policy:**
- `POPT_GE_GRASP` → 15
- `POPT_NEAR_GRASP_IF_BIG_GAP` → 2

**By graph:**
- `cit-Patents` → 2
- `com-orkut` → 8
- `soc-LiveJournal1` → 2
- `soc-pokec` → 3
- `web-Google` → 2

**By app:**
- `bc` → 5
- `cc` → 10
- `sssp` → 2

**By L3 size:**
- `1MB` → 4
- `4MB` → 6
- `8MB` → 7

## Vocabulary fingerprint

| term | reason count |
|---|---:|
| `Phase 1` | 1 |
| `P-OPT` | 17 |
| `GRASP` | 17 |
| `SRRIP` | 0 |
| `frontier` | 7 |
| `MPKI` | 0 |
| `design` | 0 |
| `sim` | 0 |
| `single-core` | 0 |
| `property array` | 0 |
| `HPCA` | 17 |

## Per-entry reason status

| graph | app | L3 | policy | len | quant | anchor | orphan | inactive | well-formed |
|---|---|---|---|---:|---|---|---|---|---|
| cit-Patents | bc | 8MB | POPT_GE_GRASP | 270 | ✓ | ✓ | — | — | ✓ |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | 252 | ✓ | ✓ | — | — | ✓ |
| com-orkut | bc | 4MB | POPT_GE_GRASP | 250 | ✓ | ✓ | — | — | ✓ |
| com-orkut | bc | 8MB | POPT_GE_GRASP | 246 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 1MB | POPT_GE_GRASP | 225 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 4MB | POPT_GE_GRASP | 239 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | 244 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 8MB | POPT_GE_GRASP | 251 | ✓ | ✓ | — | — | ✓ |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | 248 | ✓ | ✓ | — | — | ✓ |
| com-orkut | sssp | 1MB | POPT_GE_GRASP | 254 | ✓ | ✓ | — | — | ✓ |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | 244 | ✓ | ✓ | — | — | ✓ |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | 265 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | 246 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | cc | 4MB | POPT_GE_GRASP | 232 | ✓ | ✓ | — | — | ✓ |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | 231 | ✓ | ✓ | — | — | ✓ |
| web-Google | bc | 4MB | POPT_GE_GRASP | 252 | ✓ | ✓ | — | — | ✓ |
| web-Google | bc | 8MB | POPT_GE_GRASP | 214 | ✓ | ✓ | — | — | ✓ |
