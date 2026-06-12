# Multiple-testing correction across confidence gates

Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections
applied jointly to every p-value emitted by the confidence-gate suite.

- α (per-test): **0.05**
- Number of tests in the family: **81**
- Naive significant (p ≤ α, uncorrected): **41**
- Holm-Bonferroni survivors (FWER ≤ α): **24**
- Benjamini-Hochberg survivors (FDR ≤ α): **33**
- Expected false positives if all nulls true: **4.05**

## Per-source breakdown

| Source | n tests | Naive sig | HB survives | BH survives |
| :----- | ------: | --------: | ----------: | ----------: |
| `bootstrap_paired_gap` | 30 | 17 | 14 | 16 |
| `mannwhitney_gap` | 30 | 17 | 6 | 11 |
| `popt_vs_grasp_family_app` | 21 | 7 | 4 | 6 |

## Holm-Bonferroni survivors (n=24)

These are the claims that survive **strong family-wise** correction;
they are the ones safe to state as 'statistically significant' in
the paper at α=0.05.

| Source | Scope | Comparison | p (two-sided) |
| :----- | :---- | :--------- | -----------: |
| `mannwhitney_gap` | app=pr | LRU vs POPT | 0 |
| `mannwhitney_gap` | app=pr | POPT vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=bc | GRASP vs POPT | 0 |
| `bootstrap_paired_gap` | app=bc | LRU vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=cc | GRASP vs LRU | 0 |
| `bootstrap_paired_gap` | app=cc | GRASP vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=cc | LRU vs POPT | 0 |
| `bootstrap_paired_gap` | app=cc | LRU vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=cc | POPT vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=pr | GRASP vs LRU | 0 |
| `bootstrap_paired_gap` | app=pr | GRASP vs POPT | 0 |
| `bootstrap_paired_gap` | app=pr | GRASP vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=pr | LRU vs POPT | 0 |
| `bootstrap_paired_gap` | app=pr | LRU vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=pr | POPT vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=sssp | LRU vs SRRIP | 0 |
| `popt_vs_grasp_family_app` | citation/pr | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | road/cc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | social/pr | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | web/pr | POPT vs GRASP | 0 |
| `mannwhitney_gap` | app=pr | GRASP vs POPT | 3.00e-06 |
| `mannwhitney_gap` | app=cc | GRASP vs LRU | 2.60e-05 |
| `mannwhitney_gap` | app=cc | GRASP vs SRRIP | 2.60e-05 |
| `mannwhitney_gap` | app=bfs | POPT vs SRRIP | 5.40e-04 |

## BH-only survivors (n=9)

These survive FDR control but NOT FWER — paper-honest framing is
'discoveries with FDR ≤ 5%', not 'significant'.

| Source | Scope | Comparison | p (two-sided) |
| :----- | :---- | :--------- | -----------: |
| `popt_vs_grasp_family_app` | mesh/pr | POPT vs GRASP | 1.00e-03 |
| `mannwhitney_gap` | app=bfs | LRU vs POPT | 1.24e-03 |
| `mannwhitney_gap` | app=cc | GRASP vs POPT | 2.68e-03 |
| `bootstrap_paired_gap` | app=cc | GRASP vs POPT | 5.00e-03 |
| `popt_vs_grasp_family_app` | social/cc | POPT vs GRASP | 5.00e-03 |
| `mannwhitney_gap` | app=bc | GRASP vs LRU | 5.45e-03 |
| `mannwhitney_gap` | app=cc | LRU vs POPT | 7.71e-03 |
| `mannwhitney_gap` | app=bc | GRASP vs SRRIP | 8.66e-03 |
| `bootstrap_paired_gap` | app=bfs | GRASP vs POPT | 1.20e-02 |
