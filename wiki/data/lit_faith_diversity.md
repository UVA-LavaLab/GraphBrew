# Literature-faithfulness diversity audit

Audit of 279 per-claim entries from `wiki/data/literature_faithfulness_postfix.json`. The headline diversity numerics are produced by [`scripts/experiments/ecg/lit_faith_diversity.py`](../../scripts/experiments/ecg/lit_faith_diversity.py) and locked by the `LIT-Cov` confidence gate.

## Summary

| Field | Value |
|---|---|
| `claims_total` | 279 |
| `n_families` | 3 |
| `n_apps` | 5 |
| `n_l3_sizes` | 3 |
| `n_policies` | 5 |
| `n_graphs` | 6 |
| `n_papers` | 3 |
| `n_triangulated_cells` | 153 |
| `n_sign_inconsistent_cells` | 0 |
| `min_family_count` | 50 |
| `min_app_count` | 45 |
| `min_l3_count` | 84 |
| `min_paper_count` | 84 |
| `min_graph_count` | 28 |

## Claims by graph family

| Value | Count |
|---|---|
| `citation` | 52 |
| `social` | 177 |
| `web` | 50 |

## Claims by application

| Value | Count |
|---|---|
| `bc` | 58 |
| `bfs` | 57 |
| `cc` | 45 |
| `pr` | 70 |
| `sssp` | 49 |

## Claims by L3 size

| Value | Count |
|---|---|
| `1MB` | 105 |
| `4MB` | 84 |
| `8MB` | 90 |

## Claims by policy

| Value | Count |
|---|---|
| `GRASP` | 19 |
| `POPT` | 8 |
| `POPT_GE_GRASP` | 84 |
| `POPT_NEAR_GRASP_IF_BIG_GAP` | 84 |
| `SRRIP` | 84 |

## Claims by paper

| Value | Count |
|---|---|
| `Balaji HPCA21` | 192 |
| `Faldu HPCA20` | 156 |
| `Jaleel ISCA10` | 84 |

## Claims by status

| Value | Count |
|---|---|
| `insufficient_data` | 28 |
| `known_deviation` | 17 |
| `ok` | 234 |

## Claims by expected_sign

| Value | Count |
|---|---|
| `-` | 106 |
| `~` | 173 |

## Cross-paper triangulation cells

153 cells receive claims from ≥ 2 distinct papers; 0 of them carry inconsistent `expected_sign` (a paper-vs-paper disagreement the corpus surfaces).

| graph | app | l3_size | policy | n_papers | papers | signs | consistent |
|---|---|---|---|---|---|---|---|
| `cit-Patents` | `bc` | `1MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `bc` | `1MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `bc` | `4MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `bc` | `4MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `bc` | `8MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `bc` | `8MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `bfs` | `1MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `bfs` | `1MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `bfs` | `4MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `bfs` | `4MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `bfs` | `8MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `bfs` | `8MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `cc` | `1MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `cc` | `4MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `cc` | `8MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `pr` | `1MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `pr` | `1MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `pr` | `4MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `pr` | `4MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `pr` | `8MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `pr` | `8MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `sssp` | `1MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `sssp` | `1MB` | `SRRIP` | 2 | Balaji HPCA21, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `sssp` | `4MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `sssp` | `4MB` | `SRRIP` | 2 | Balaji HPCA21, Jaleel ISCA10 | ~ | ✅ |
| `cit-Patents` | `sssp` | `8MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `cit-Patents` | `sssp` | `8MB` | `SRRIP` | 2 | Balaji HPCA21, Jaleel ISCA10 | ~ | ✅ |
| `com-orkut` | `bc` | `1MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| `com-orkut` | `bc` | `1MB` | `SRRIP` | 2 | Faldu HPCA20, Jaleel ISCA10 | ~ | ✅ |
| `com-orkut` | `bc` | `4MB` | `POPT_NEAR_GRASP_IF_BIG_GAP` | 2 | Balaji HPCA21, Faldu HPCA20 | ~ | ✅ |
| _… 123 more_ |

## Family × app density

| family\app | bc | bfs | cc | pr | sssp |
|---|---|---|---|---|---|
| `citation` | 10 | 10 | 9 | 12 | 11 |
| `social` | 38 | 37 | 27 | 46 | 29 |
| `web` | 10 | 10 | 9 | 12 | 9 |
