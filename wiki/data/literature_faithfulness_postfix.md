# Literature-faithfulness summary

- Sweep root: `/tmp/graphbrew-lit-baseline`
- Claims total: **46**
- Verdict mix: **27 ok**, 0 within-tolerance, **1 DISAGREE**, 1 known-deviation, 0 missing, 17 insufficient_data
- min_accesses threshold: 10000

## Observed L3 miss-rates

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | Δ GRASP-LRU | Δ POPT-LRU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core | bc | 1MB | 0.9906 | 1.0000 | 1.0000 | 0.9968 | +0.938pp | +0.622pp |
| email-Eu-core | bc | 4MB | 0.9968 | 1.0000 | 0.9968 | 1.0000 | +0.000pp | +0.317pp |
| email-Eu-core | bc | 8MB | 1.0000 | 0.9875 | 0.9906 | 1.0000 | -0.943pp | +0.000pp |
| email-Eu-core | bfs | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 1MB | 1.0000 | 1.0000 | 0.9995 | 1.0000 | -0.047pp | +0.000pp |
| email-Eu-core | pr | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| soc-pokec | cc | 1MB | 0.6994 | 0.6668 | 0.5684 | 0.6742 | -13.105pp | -2.525pp |
| soc-pokec | cc | 4MB | 0.3961 | 0.3299 | 0.2264 | 0.2825 | -16.968pp | -11.360pp |
| soc-pokec | cc | 8MB | 0.0577 | 0.0577 | 0.0573 | 0.0577 | -0.039pp | -0.003pp |
| soc-pokec | sssp | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| soc-pokec | sssp | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| soc-pokec | sssp | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| web-Google | bc | 1MB | 0.7073 | 0.6889 | 0.7075 | 0.7023 | +0.020pp | -0.493pp |
| web-Google | bc | 4MB | 0.3687 | 0.3574 | 0.3418 | 0.3708 | -2.691pp | +0.211pp |
| web-Google | bc | 8MB | 0.1515 | 0.1444 | 0.1357 | 0.1662 | -1.583pp | +1.467pp |
| web-Google | bfs | 1MB | 0.9700 | 0.9693 | 0.9363 | 0.9473 | -3.368pp | -2.270pp |
| web-Google | bfs | 4MB | 0.9359 | 0.9349 | 0.7590 | 0.7552 | -17.683pp | -18.069pp |
| web-Google | bfs | 8MB | 0.8086 | 0.8038 | 0.7542 | 0.7294 | -5.442pp | -7.923pp |
| web-Google | cc | 1MB | 0.5608 | 0.5222 | 0.4944 | 0.5071 | -6.636pp | -5.369pp |
| web-Google | cc | 4MB | 0.0457 | 0.0457 | 0.0457 | 0.0457 | -0.005pp | -0.002pp |
| web-Google | cc | 8MB | 0.0456 | 0.0457 | 0.0457 | 0.0457 | +0.005pp | +0.005pp |
| web-Google | pr | 1MB | 0.6009 | 0.5439 | 0.4532 | 0.4167 | -14.766pp | -18.413pp |
| web-Google | pr | 4MB | 0.1414 | 0.1190 | 0.1056 | 0.1163 | -3.583pp | -2.510pp |
| web-Google | pr | 8MB | 0.0970 | 0.0934 | 0.0845 | 0.0841 | -1.253pp | -1.292pp |
| web-Google | sssp | 1MB | 0.5558 | 0.5507 | 0.5561 | 0.5475 | +0.034pp | -0.824pp |
| web-Google | sssp | 4MB | 0.0163 | 0.0163 | 0.0163 | 0.0163 | +0.001pp | +0.001pp |
| web-Google | sssp | 8MB | 0.0163 | 0.0163 | 0.0163 | 0.0163 | -0.003pp | -0.003pp |

## Per-claim verdicts

| status | graph | app | L3 | policy | Δ | citation |
|---|---|---|---|---|---:|---|
| insufficient_data | email-Eu-core | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | -0.316pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.317pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.943pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 1MB | POPT_GE_GRASP | +0.047pp | Balaji & Lucia HPCA 2021 §6.3 |
| insufficient_data | email-Eu-core | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.047pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 4MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 |
| insufficient_data | email-Eu-core | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 8MB | GRASP | +0.000pp | Faldu et al. HPCA 2020 Fig 10 |
| insufficient_data | email-Eu-core | pr | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 |
| insufficient_data | email-Eu-core | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| disagree | soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +10.580pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.608pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.037pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | soc-pokec | sssp | 1MB | POPT | +0.000pp | Balaji & Lucia HPCA 2021 Fig 10 |
| insufficient_data | soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | soc-pokec | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | soc-pokec | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 1MB | GRASP | +0.020pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.512pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.902pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.050pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 1MB | GRASP | -3.368pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.098pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.386pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.481pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.268pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.004pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.001pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 1MB | SRRIP | -5.698pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 1MB | GRASP | -14.766pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 1MB | POPT | -18.413pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | web-Google | pr | 1MB | POPT_GE_GRASP | -3.647pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | web-Google | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.647pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 4MB | SRRIP | -2.238pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| known_deviation | web-Google | pr | 4MB | POPT_GE_GRASP | +1.072pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | web-Google | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.072pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 8MB | SRRIP | -0.369pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 8MB | GRASP | -1.253pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 8MB | POPT_GE_GRASP | -0.038pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | web-Google | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.038pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.858pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |

## Known deviations (registered)

- **web-Google/pr L3=4MB POPT_GE_GRASP** (+1.072pp): 

## ⚠ Disagreements (need investigation)

- **soc-pokec/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** Δ=+10.580pp — Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check
