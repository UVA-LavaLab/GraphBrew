# Literature-faithfulness summary

- Sweep root: `/tmp/graphbrew-lit-baseline`
- Claims total: **121**
- Verdict mix: **98 ok**, 0 within-tolerance, **0 DISAGREE**, 3 known-deviation, 0 missing, 20 insufficient_data
- min_accesses threshold: 10000

## Observed L3 miss-rates

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | Δ GRASP-LRU | Δ POPT-LRU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | bfs | 1MB | 0.9665 | 0.9625 | 0.9565 | 0.9576 | -0.998pp | -0.892pp |
| cit-Patents | bfs | 4MB | 0.9098 | 0.9030 | 0.8690 | 0.8728 | -4.081pp | -3.701pp |
| cit-Patents | bfs | 8MB | 0.8711 | 0.8679 | 0.7848 | 0.7917 | -8.624pp | -7.936pp |
| cit-Patents | cc | 1MB | 0.6431 | 0.6010 | 0.5262 | 0.6134 | -11.691pp | -2.965pp |
| cit-Patents | cc | 4MB | 0.3170 | 0.2940 | 0.2473 | 0.2838 | -6.967pp | -3.319pp |
| cit-Patents | cc | 8MB | 0.1815 | 0.1601 | 0.1171 | 0.1324 | -6.445pp | -4.917pp |
| cit-Patents | pr | 1MB | 0.8923 | 0.8780 | 0.7858 | 0.7753 | -10.647pp | -11.699pp |
| cit-Patents | pr | 4MB | 0.5879 | 0.5461 | 0.4542 | 0.3718 | -13.369pp | -21.615pp |
| cit-Patents | pr | 8MB | 0.3248 | 0.2756 | 0.2145 | 0.1862 | -11.025pp | -13.859pp |
| cit-Patents | sssp | 1MB | 0.8529 | 0.8497 | 0.8184 | 0.8270 | -3.452pp | -2.588pp |
| cit-Patents | sssp | 4MB | 0.5631 | 0.5385 | 0.4764 | 0.4944 | -8.673pp | -6.869pp |
| cit-Patents | sssp | 8MB | 0.2690 | 0.2440 | 0.1999 | 0.2162 | -6.904pp | -5.271pp |
| email-Eu-core | bc | 1MB | 0.9906 | 1.0000 | 1.0000 | 0.9968 | +0.938pp | +0.622pp |
| email-Eu-core | bc | 4MB | 0.9968 | 1.0000 | 0.9968 | 1.0000 | +0.000pp | +0.317pp |
| email-Eu-core | bc | 8MB | 1.0000 | 0.9875 | 0.9906 | 1.0000 | -0.943pp | +0.000pp |
| email-Eu-core | bfs | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 1MB | 1.0000 | 1.0000 | 0.9995 | 1.0000 | -0.047pp | +0.000pp |
| email-Eu-core | pr | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| soc-LiveJournal1 | pr | 1MB | 0.7322 | 0.6994 | 0.6843 | 0.6248 | -4.791pp | -10.742pp |
| soc-LiveJournal1 | pr | 4MB | 0.4261 | 0.3897 | 0.3514 | 0.2917 | -7.470pp | -13.446pp |
| soc-LiveJournal1 | pr | 8MB | 0.2756 | 0.2406 | 0.2018 | 0.1796 | -7.385pp | -9.597pp |
| soc-pokec | cc | 1MB | 0.6994 | 0.6668 | 0.5684 | 0.6742 | -13.105pp | -2.525pp |
| soc-pokec | cc | 4MB | 0.3961 | 0.3299 | 0.2264 | 0.2825 | -16.968pp | -11.360pp |
| soc-pokec | cc | 8MB | 0.0577 | 0.0577 | 0.0573 | 0.0577 | -0.039pp | -0.003pp |
| soc-pokec | pr | 1MB | 0.6796 | 0.6344 | 0.5434 | 0.5476 | -13.617pp | -13.196pp |
| soc-pokec | pr | 4MB | 0.1969 | 0.1616 | 0.1302 | 0.1238 | -6.672pp | -7.314pp |
| soc-pokec | pr | 8MB | 0.1004 | 0.0883 | 0.0793 | 0.0874 | -2.113pp | -1.298pp |
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
| ok | cit-Patents | bfs | 1MB | SRRIP | -0.397pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 1MB | GRASP | -0.998pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.106pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 4MB | SRRIP | -0.685pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.380pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 8MB | SRRIP | -0.313pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.689pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 1MB | SRRIP | -4.213pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.726pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 4MB | SRRIP | -2.299pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | cit-Patents | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.648pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 8MB | SRRIP | -2.142pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | cit-Patents | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.528pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 1MB | SRRIP | -1.427pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 1MB | GRASP | -10.647pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 1MB | POPT | -11.699pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | cit-Patents | pr | 1MB | POPT_GE_GRASP | -1.052pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | cit-Patents | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.052pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 4MB | SRRIP | -4.181pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 4MB | POPT_GE_GRASP | -8.246pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | cit-Patents | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.246pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 8MB | SRRIP | -4.915pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 8MB | GRASP | -11.025pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 8MB | POPT_GE_GRASP | -2.834pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | cit-Patents | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.834pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 1MB | SRRIP | -0.320pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 1MB | GRASP | -3.452pp | Balaji & Lucia HPCA 2021 Fig 10 (GRASP bar) |
| ok | cit-Patents | sssp | 1MB | POPT | -2.588pp | Balaji & Lucia HPCA 2021 Fig 10 |
| ok | cit-Patents | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.863pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 4MB | SRRIP | -2.465pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.804pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 8MB | SRRIP | -2.498pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.633pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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
| ok | soc-LiveJournal1 | pr | 1MB | SRRIP | -3.276pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 1MB | GRASP | -4.791pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT | -10.742pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_GE_GRASP | -5.950pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.950pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 4MB | SRRIP | -3.644pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_GE_GRASP | -5.975pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.975pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 8MB | SRRIP | -3.498pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 8MB | GRASP | -7.385pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_GE_GRASP | -2.212pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.212pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 1MB | SRRIP | -3.263pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +10.580pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 4MB | SRRIP | -6.614pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.608pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 8MB | SRRIP | +0.002pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-pokec | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.037pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 1MB | SRRIP | -4.517pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 1MB | GRASP | -13.617pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 1MB | POPT | -13.196pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | soc-pokec | pr | 1MB | POPT_GE_GRASP | +0.421pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-pokec | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.421pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 4MB | SRRIP | -3.534pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 4MB | POPT_GE_GRASP | -0.642pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-pokec | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.642pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 8MB | SRRIP | -1.215pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 8MB | GRASP | -2.113pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 8MB | POPT_GE_GRASP | +0.814pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-pokec | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.814pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | soc-pokec | sssp | 1MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| insufficient_data | soc-pokec | sssp | 1MB | POPT | +0.000pp | Balaji & Lucia HPCA 2021 Fig 10 |
| insufficient_data | soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | soc-pokec | sssp | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| insufficient_data | soc-pokec | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | soc-pokec | sssp | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| insufficient_data | soc-pokec | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 1MB | SRRIP | -1.838pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bc | 1MB | GRASP | +0.020pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.512pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 4MB | SRRIP | -1.128pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.902pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 8MB | SRRIP | -0.711pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.050pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 1MB | SRRIP | -0.068pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 1MB | GRASP | -3.368pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.098pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 4MB | SRRIP | -0.102pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.386pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 8MB | SRRIP | -0.481pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.481pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 1MB | SRRIP | -3.859pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.268pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 4MB | SRRIP | -0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.004pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 8MB | SRRIP | +0.002pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
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
| ok | web-Google | sssp | 1MB | SRRIP | -0.502pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.858pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 4MB | SRRIP | +0.002pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 8MB | SRRIP | -0.003pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |

## Known deviations (registered)

- **cit-Patents/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+8.726pp): 
- **soc-pokec/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+10.580pp): 
- **web-Google/pr L3=4MB POPT_GE_GRASP** (+1.072pp): 
