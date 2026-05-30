# Literature-faithfulness diversity audit

Audit of 330 per-claim entries from `wiki/data/literature_faithfulness_postfix.json`. The headline diversity numerics are produced by [`scripts/experiments/ecg/lit_faith_diversity.py`](../../scripts/experiments/ecg/lit_faith_diversity.py) and locked by the `LIT-Cov` confidence gate.

## Summary

| Field | Value |
|---|---|
| `claims_total` | 330 |
| `n_families` | 5 |
| `n_apps` | 5 |
| `n_l3_sizes` | 7 |
| `n_policies` | 5 |
| `n_graphs` | 8 |
| `n_papers` | 3 |
| `n_triangulated_cells` | 174 |
| `n_sign_inconsistent_cells` | 0 |
| `min_family_count` | 10 |
| `min_app_count` | 55 |
| `min_l3_count` | 12 |
| `min_paper_count` | 75 |
| `min_graph_count` | 10 |

## Claims by graph family

| Value | Count |
|---|---|
| `citation` | 52 |
| `mesh` | 10 |
| `road` | 50 |
| `social` | 168 |
| `web` | 50 |

## Claims by application

| Value | Count |
|---|---|
| `bc` | 65 |
| `bfs` | 64 |
| `cc` | 55 |
| `pr` | 87 |
| `sssp` | 59 |

## Claims by L3 size

| Value | Count |
|---|---|
| `16kB` | 12 |
| `1MB` | 114 |
| `256kB` | 12 |
| `4MB` | 81 |
| `4kB` | 12 |
| `64kB` | 12 |
| `8MB` | 87 |

## Claims by policy

| Value | Count |
|---|---|
| `GRASP` | 19 |
| `POPT` | 8 |
| `POPT_GE_GRASP` | 114 |
| `POPT_NEAR_GRASP_IF_BIG_GAP` | 114 |
| `SRRIP` | 75 |

## Claims by paper

| Value | Count |
|---|---|
| `Balaji HPCA21` | 252 |
| `Faldu HPCA20` | 177 |
| `Jaleel ISCA10` | 75 |

## Claims by status

| Value | Count |
|---|---|
| `known_deviation` | 30 |
| `ok` | 298 |
| `within_tolerance` | 2 |

## Claims by expected_sign

| Value | Count |
|---|---|
| `-` | 136 |
| `~` | 194 |

## Cross-paper triangulation cells

174 cells receive claims from ≥ 2 distinct papers; 0 of them carry inconsistent `expected_sign` (a paper-vs-paper disagreement the corpus surfaces).

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
| _… 144 more_ |

## Family × app density

| family\app | bc | bfs | cc | pr | sssp |
|---|---|---|---|---|---|
| `citation` | 10 | 10 | 9 | 12 | 11 |
| `mesh` | 0 | 0 | 0 | 10 | 0 |
| `road` | 10 | 10 | 10 | 10 | 10 |
| `social` | 35 | 34 | 27 | 43 | 29 |
| `web` | 10 | 10 | 9 | 12 | 9 |
