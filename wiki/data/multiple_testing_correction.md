# Multiple-testing correction across confidence gates

Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections
applied jointly to every p-value emitted by the confidence-gate suite.

- α (per-test): **0.05**
- Number of tests in the family: **81**
- Naive significant (p ≤ α, uncorrected): **44**
- Holm-Bonferroni survivors (FWER ≤ α): **28**
- Benjamini-Hochberg survivors (FDR ≤ α): **38**
- Expected false positives if all nulls true: **4.05**

## Per-source breakdown

| Source | n tests | Naive sig | HB survives | BH survives |
| :----- | ------: | --------: | ----------: | ----------: |
| `bootstrap_paired_gap` | 30 | 14 | 10 | 13 |
| `mannwhitney_gap` | 30 | 18 | 9 | 14 |
| `popt_vs_grasp_family_app` | 21 | 12 | 9 | 11 |

## Holm-Bonferroni survivors (n=28)

These are the claims that survive **strong family-wise** correction;
they are the ones safe to state as 'statistically significant' in
the paper at α=0.05.

| Source | Scope | Comparison | p (two-sided) |
| :----- | :---- | :--------- | -----------: |
| `mannwhitney_gap` | app=pr | LRU vs POPT | 0 |
| `mannwhitney_gap` | app=pr | POPT vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=bc | LRU vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=cc | GRASP vs LRU | 0 |
| `bootstrap_paired_gap` | app=cc | GRASP vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=cc | LRU vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=pr | GRASP vs LRU | 0 |
| `bootstrap_paired_gap` | app=pr | GRASP vs POPT | 0 |
| `bootstrap_paired_gap` | app=pr | GRASP vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=pr | LRU vs POPT | 0 |
| `bootstrap_paired_gap` | app=pr | LRU vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=pr | POPT vs SRRIP | 0 |
| `popt_vs_grasp_family_app` | citation/bfs | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | citation/cc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | citation/pr | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | citation/sssp | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | road/bc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | road/sssp | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | social/bc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | social/cc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | web/cc | POPT vs GRASP | 0 |
| `mannwhitney_gap` | app=cc | GRASP vs POPT | 5.00e-06 |
| `mannwhitney_gap` | app=cc | GRASP vs SRRIP | 1.00e-05 |
| `mannwhitney_gap` | app=cc | GRASP vs LRU | 1.60e-05 |
| `mannwhitney_gap` | app=pr | GRASP vs POPT | 3.12e-04 |
| `mannwhitney_gap` | app=bc | GRASP vs LRU | 3.28e-04 |
| `mannwhitney_gap` | app=bfs | POPT vs SRRIP | 3.28e-04 |
| `mannwhitney_gap` | app=bfs | LRU vs POPT | 6.88e-04 |

## BH-only survivors (n=10)

These survive FDR control but NOT FWER — paper-honest framing is
'discoveries with FDR ≤ 5%', not 'significant'.

| Source | Scope | Comparison | p (two-sided) |
| :----- | :---- | :--------- | -----------: |
| `bootstrap_paired_gap` | app=cc | GRASP vs POPT | 1.00e-03 |
| `popt_vs_grasp_family_app` | social/pr | POPT vs GRASP | 1.00e-03 |
| `bootstrap_paired_gap` | app=cc | LRU vs POPT | 1.00e-03 |
| `mannwhitney_gap` | app=bfs | GRASP vs SRRIP | 2.71e-03 |
| `bootstrap_paired_gap` | app=sssp | LRU vs SRRIP | 3.00e-03 |
| `mannwhitney_gap` | app=bfs | GRASP vs LRU | 3.60e-03 |
| `mannwhitney_gap` | app=pr | GRASP vs LRU | 6.05e-03 |
| `mannwhitney_gap` | app=bc | GRASP vs POPT | 6.24e-03 |
| `mannwhitney_gap` | app=bc | GRASP vs SRRIP | 1.71e-02 |
| `popt_vs_grasp_family_app` | road/pr | POPT vs GRASP | 2.30e-02 |
