# Multiple-testing correction across confidence gates

Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections
applied jointly to every p-value emitted by the confidence-gate suite.

- α (per-test): **0.05**
- Number of tests in the family: **81**
- Naive significant (p ≤ α, uncorrected): **43**
- Holm-Bonferroni survivors (FWER ≤ α): **25**
- Benjamini-Hochberg survivors (FDR ≤ α): **38**
- Expected false positives if all nulls true: **4.05**

## Per-source breakdown

| Source | n tests | Naive sig | HB survives | BH survives |
| :----- | ------: | --------: | ----------: | ----------: |
| `bootstrap_paired_gap` | 30 | 14 | 13 | 14 |
| `mannwhitney_gap` | 30 | 19 | 4 | 15 |
| `popt_vs_grasp_family_app` | 21 | 10 | 8 | 9 |

## Holm-Bonferroni survivors (n=25)

These are the claims that survive **strong family-wise** correction;
they are the ones safe to state as 'statistically significant' in
the paper at α=0.05.

| Source | Scope | Comparison | p (two-sided) |
| :----- | :---- | :--------- | -----------: |
| `bootstrap_paired_gap` | app=bc | GRASP vs POPT | 0 |
| `bootstrap_paired_gap` | app=bc | LRU vs SRRIP | 0 |
| `bootstrap_paired_gap` | app=bc | POPT vs SRRIP | 0 |
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
| `popt_vs_grasp_family_app` | citation/bc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | citation/pr | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | citation/sssp | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | social/bc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | social/sssp | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | web/bc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | web/cc | POPT vs GRASP | 0 |
| `popt_vs_grasp_family_app` | web/pr | POPT vs GRASP | 0 |
| `mannwhitney_gap` | app=pr | LRU vs POPT | 1.00e-06 |
| `mannwhitney_gap` | app=pr | POPT vs SRRIP | 4.00e-06 |
| `mannwhitney_gap` | app=pr | GRASP vs POPT | 4.01e-04 |
| `mannwhitney_gap` | app=bc | GRASP vs POPT | 6.10e-04 |

## BH-only survivors (n=13)

These survive FDR control but NOT FWER — paper-honest framing is
'discoveries with FDR ≤ 5%', not 'significant'.

| Source | Scope | Comparison | p (two-sided) |
| :----- | :---- | :--------- | -----------: |
| `popt_vs_grasp_family_app` | mesh/pr | POPT vs GRASP | 1.00e-03 |
| `mannwhitney_gap` | app=bc | GRASP vs LRU | 1.29e-03 |
| `mannwhitney_gap` | app=cc | LRU vs POPT | 2.14e-03 |
| `mannwhitney_gap` | app=bc | GRASP vs SRRIP | 2.52e-03 |
| `mannwhitney_gap` | app=sssp | GRASP vs POPT | 3.19e-03 |
| `mannwhitney_gap` | app=bfs | POPT vs SRRIP | 3.48e-03 |
| `mannwhitney_gap` | app=pr | GRASP vs LRU | 6.85e-03 |
| `mannwhitney_gap` | app=bfs | GRASP vs SRRIP | 7.36e-03 |
| `mannwhitney_gap` | app=bfs | LRU vs POPT | 7.85e-03 |
| `bootstrap_paired_gap` | app=sssp | POPT vs SRRIP | 8.00e-03 |
| `mannwhitney_gap` | app=cc | POPT vs SRRIP | 8.35e-03 |
| `mannwhitney_gap` | app=sssp | GRASP vs LRU | 9.41e-03 |
| `mannwhitney_gap` | app=bfs | GRASP vs LRU | 1.43e-02 |
