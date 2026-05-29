# Literature-faithfulness summary

- Sweep root: `/tmp/graphbrew-lit-baseline`
- Claims total: **280**
- Verdict mix: **248 ok**, 2 within-tolerance, **0 DISAGREE**, 30 known-deviation, 0 missing, 0 insufficient_data
- min_accesses threshold: 10000

## Observed L3 miss-rates

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | Δ GRASP-LRU | Δ POPT-LRU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | bc | 1MB | 0.8990 | 0.8927 | 0.9024 | 0.8976 | +0.337pp | -0.144pp |
| cit-Patents | bc | 4MB | 0.7414 | 0.7294 | 0.7209 | 0.7368 | -2.053pp | -0.456pp |
| cit-Patents | bc | 8MB | 0.6167 | 0.5927 | 0.5639 | 0.5892 | -5.271pp | -2.742pp |
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
| com-orkut | bc | 1MB | 0.8624 | 0.8370 | 0.7999 | 0.8247 | -6.245pp | -3.770pp |
| com-orkut | bc | 4MB | 0.5827 | 0.5502 | 0.5242 | 0.5709 | -5.844pp | -1.176pp |
| com-orkut | bc | 8MB | 0.3821 | 0.3531 | 0.3359 | 0.3832 | -4.620pp | +0.107pp |
| com-orkut | bfs | 1MB | 0.9973 | 0.9976 | 0.9949 | 0.9978 | -0.234pp | +0.056pp |
| com-orkut | bfs | 4MB | 0.9949 | 0.9956 | 0.9785 | 0.9745 | -1.648pp | -2.043pp |
| com-orkut | bfs | 8MB | 0.9942 | 0.9945 | 0.9621 | 0.9548 | -3.215pp | -3.944pp |
| com-orkut | cc | 1MB | 0.6889 | 0.6510 | 0.5837 | 0.6939 | -10.523pp | +0.494pp |
| com-orkut | cc | 4MB | 0.5033 | 0.4888 | 0.3698 | 0.4704 | -13.345pp | -3.290pp |
| com-orkut | cc | 8MB | 0.4306 | 0.3212 | 0.2212 | 0.2830 | -20.943pp | -14.756pp |
| com-orkut | pr | 1MB | 0.7104 | 0.6686 | 0.6867 | 0.6184 | -2.368pp | -9.198pp |
| com-orkut | pr | 4MB | 0.3108 | 0.2743 | 0.2679 | 0.2158 | -4.295pp | -9.508pp |
| com-orkut | pr | 8MB | 0.1646 | 0.1422 | 0.1296 | 0.1094 | -3.500pp | -5.521pp |
| com-orkut | sssp | 1MB | 0.6603 | 0.6204 | 0.6338 | 0.5968 | -2.657pp | -6.354pp |
| com-orkut | sssp | 4MB | 0.1697 | 0.1541 | 0.1589 | 0.1473 | -1.084pp | -2.245pp |
| com-orkut | sssp | 8MB | 0.0246 | 0.0223 | 0.0224 | 0.0241 | -0.215pp | -0.053pp |
| email-Eu-core | bc | 1MB | 0.0001 | 0.0001 | 0.0001 | 0.0001 | +0.000pp | +0.000pp |
| email-Eu-core | bc | 4MB | 0.0001 | 0.0001 | 0.0001 | 0.0002 | -0.000pp | +0.010pp |
| email-Eu-core | bc | 8MB | 0.0001 | 0.0001 | 0.0001 | 0.0001 | +0.000pp | -0.000pp |
| email-Eu-core | bfs | 1MB | 0.0458 | 0.0458 | 0.2234 | 0.0459 | +17.752pp | +0.000pp |
| email-Eu-core | bfs | 4MB | 0.0459 | 0.0459 | 0.0459 | 0.0458 | -0.000pp | -0.000pp |
| email-Eu-core | bfs | 8MB | 0.0459 | 0.0459 | 0.0458 | 0.0458 | -0.005pp | -0.003pp |
| email-Eu-core | pr | 1MB | 0.0081 | 0.0077 | 0.0113 | 0.0075 | +0.329pp | -0.059pp |
| email-Eu-core | pr | 4MB | 0.0077 | 0.0076 | 0.0076 | 0.0078 | -0.013pp | +0.012pp |
| email-Eu-core | pr | 8MB | 0.0080 | 0.0078 | 0.0076 | 0.0078 | -0.043pp | -0.021pp |
| roadNet-CA | pr | 4kB | 0.9999 | 0.9999 | 0.9997 | 0.9997 | -0.022pp | -0.023pp |
| roadNet-CA | pr | 16kB | 0.9999 | 0.9999 | 0.9989 | 0.9992 | -0.099pp | -0.068pp |
| roadNet-CA | pr | 64kB | 0.9993 | 0.9996 | 0.9956 | 0.9827 | -0.366pp | -1.660pp |
| roadNet-CA | pr | 256kB | 0.9856 | 0.9850 | 0.9885 | 0.9334 | +0.283pp | -5.228pp |
| roadNet-CA | pr | 1MB | 0.9417 | 0.9393 | 0.9564 | 0.8918 | +1.472pp | -4.995pp |
| soc-LiveJournal1 | bc | 1MB | 0.8432 | 0.8250 | 0.7949 | 0.8148 | -4.825pp | -2.840pp |
| soc-LiveJournal1 | bc | 4MB | 0.6038 | 0.5828 | 0.5511 | 0.5698 | -5.271pp | -3.394pp |
| soc-LiveJournal1 | bc | 8MB | 0.4528 | 0.4237 | 0.3904 | 0.4062 | -6.240pp | -4.657pp |
| soc-LiveJournal1 | bfs | 1MB | 0.7987 | 0.7829 | 0.7871 | 0.7716 | -1.154pp | -2.709pp |
| soc-LiveJournal1 | bfs | 4MB | 0.6075 | 0.6009 | 0.6007 | 0.5886 | -0.675pp | -1.888pp |
| soc-LiveJournal1 | bfs | 8MB | 0.5571 | 0.5521 | 0.5311 | 0.5215 | -2.603pp | -3.564pp |
| soc-LiveJournal1 | cc | 1MB | 0.7832 | 0.7722 | 0.7450 | 0.7590 | -3.818pp | -2.421pp |
| soc-LiveJournal1 | cc | 4MB | 0.5437 | 0.5324 | 0.4774 | 0.4955 | -6.630pp | -4.820pp |
| soc-LiveJournal1 | cc | 8MB | 0.4314 | 0.4099 | 0.3193 | 0.3501 | -11.211pp | -8.128pp |
| soc-LiveJournal1 | pr | 1MB | 0.7322 | 0.6994 | 0.6843 | 0.6248 | -4.791pp | -10.742pp |
| soc-LiveJournal1 | pr | 4MB | 0.4261 | 0.3897 | 0.3514 | 0.2917 | -7.470pp | -13.446pp |
| soc-LiveJournal1 | pr | 8MB | 0.2756 | 0.2406 | 0.2018 | 0.1796 | -7.385pp | -9.597pp |
| soc-LiveJournal1 | sssp | 1MB | 0.6877 | 0.6871 | 0.7083 | 0.6730 | +2.064pp | -1.466pp |
| soc-LiveJournal1 | sssp | 4MB | 0.3556 | 0.3423 | 0.3089 | 0.3122 | -4.676pp | -4.345pp |
| soc-LiveJournal1 | sssp | 8MB | 0.1647 | 0.1502 | 0.1288 | 0.1346 | -3.585pp | -3.008pp |
| soc-pokec | bc | 1MB | 0.8520 | 0.8288 | 0.7651 | 0.7959 | -8.691pp | -5.612pp |
| soc-pokec | bc | 4MB | 0.5009 | 0.4670 | 0.4205 | 0.4376 | -8.048pp | -6.331pp |
| soc-pokec | bc | 8MB | 0.2435 | 0.2215 | 0.1994 | 0.2022 | -4.412pp | -4.127pp |
| soc-pokec | bfs | 1MB | 0.8717 | 0.8620 | 0.8430 | 0.8556 | -2.872pp | -1.610pp |
| soc-pokec | bfs | 4MB | 0.7401 | 0.7387 | 0.6764 | 0.6736 | -6.369pp | -6.652pp |
| soc-pokec | bfs | 8MB | 0.7146 | 0.7136 | 0.6038 | 0.5928 | -11.079pp | -12.180pp |
| soc-pokec | cc | 1MB | 0.6994 | 0.6668 | 0.5684 | 0.6742 | -13.105pp | -2.525pp |
| soc-pokec | cc | 4MB | 0.3961 | 0.3299 | 0.2264 | 0.2825 | -16.968pp | -11.360pp |
| soc-pokec | cc | 8MB | 0.0577 | 0.0577 | 0.0573 | 0.0577 | -0.039pp | -0.003pp |
| soc-pokec | pr | 1MB | 0.6796 | 0.6344 | 0.5434 | 0.5476 | -13.617pp | -13.196pp |
| soc-pokec | pr | 4MB | 0.1969 | 0.1616 | 0.1302 | 0.1238 | -6.672pp | -7.314pp |
| soc-pokec | pr | 8MB | 0.1004 | 0.0883 | 0.0793 | 0.0874 | -2.113pp | -1.298pp |
| soc-pokec | sssp | 1MB | 0.6397 | 0.6080 | 0.5294 | 0.5703 | -11.039pp | -6.942pp |
| soc-pokec | sssp | 4MB | 0.0987 | 0.0854 | 0.0678 | 0.0765 | -3.091pp | -2.219pp |
| soc-pokec | sssp | 8MB | 0.0030 | 0.0030 | 0.0030 | 0.0029 | +0.000pp | -0.001pp |
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
| ok | cit-Patents | bc | 1MB | SRRIP | -0.631pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| within_tolerance | cit-Patents | bc | 1MB | GRASP | +0.337pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bc | 1MB | POPT_GE_GRASP | -0.481pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.481pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 4MB | SRRIP | -1.203pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | cit-Patents | bc | 4MB | POPT_GE_GRASP | +1.597pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.597pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 8MB | SRRIP | -2.393pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | cit-Patents | bc | 8MB | POPT_GE_GRASP | +2.529pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.529pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 1MB | SRRIP | -0.397pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 1MB | GRASP | -0.998pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bfs | 1MB | POPT_GE_GRASP | +0.106pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.106pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 4MB | SRRIP | -0.685pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 4MB | POPT_GE_GRASP | +0.380pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.380pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 8MB | SRRIP | -0.313pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 8MB | POPT_GE_GRASP | +0.689pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.689pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 1MB | SRRIP | -4.213pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 1MB | POPT_GE_GRASP | +8.726pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.726pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 4MB | SRRIP | -2.299pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 4MB | POPT_GE_GRASP | +3.648pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.648pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 8MB | SRRIP | -2.142pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 8MB | POPT_GE_GRASP | +1.528pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
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
| ok | cit-Patents | sssp | 1MB | POPT_GE_GRASP | +0.863pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.863pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 4MB | SRRIP | -2.465pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | cit-Patents | sssp | 4MB | POPT_GE_GRASP | +1.804pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.804pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 8MB | SRRIP | -2.498pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | cit-Patents | sssp | 8MB | POPT_GE_GRASP | +1.633pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.633pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 1MB | SRRIP | -2.541pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 1MB | POPT_GE_GRASP | +2.475pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.475pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 4MB | SRRIP | -3.241pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 4MB | POPT_GE_GRASP | +4.668pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.668pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 8MB | SRRIP | -2.899pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 8MB | POPT_GE_GRASP | +4.727pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.727pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 1MB | SRRIP | +0.034pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 1MB | POPT_GE_GRASP | +0.289pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.289pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 4MB | SRRIP | +0.065pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 4MB | POPT_GE_GRASP | -0.395pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.395pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 8MB | SRRIP | +0.027pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 8MB | POPT_GE_GRASP | -0.729pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.729pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 1MB | SRRIP | -3.791pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 1MB | POPT_GE_GRASP | +11.016pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | com-orkut | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +11.016pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 4MB | SRRIP | -1.449pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 4MB | POPT_GE_GRASP | +10.055pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +10.055pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 8MB | SRRIP | -10.943pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 8MB | POPT_GE_GRASP | +6.187pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.187pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 1MB | SRRIP | -4.177pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 1MB | GRASP | -2.368pp | Faldu et al. HPCA 2020 §6.1 (extrapolated to com-orkut from twitter Fig 10) |
| ok | com-orkut | pr | 1MB | POPT | -9.198pp | Balaji & Lucia HPCA 2021 §6 (extrapolated to com-orkut from twitter) |
| ok | com-orkut | pr | 1MB | POPT_GE_GRASP | -6.830pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | com-orkut | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.830pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 4MB | SRRIP | -3.651pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 4MB | POPT_GE_GRASP | -5.213pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | com-orkut | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.213pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 8MB | SRRIP | -2.240pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 8MB | GRASP | -3.500pp | Faldu et al. HPCA 2020 §6.1 (extrapolated from twitter Fig 10) |
| ok | com-orkut | pr | 8MB | POPT_GE_GRASP | -2.021pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | com-orkut | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.021pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 1MB | SRRIP | -3.990pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 1MB | POPT_GE_GRASP | -3.696pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.696pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 4MB | SRRIP | -1.559pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 4MB | POPT_GE_GRASP | -1.161pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.161pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 8MB | SRRIP | -0.228pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 8MB | POPT_GE_GRASP | +0.162pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.162pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | bc | 1MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | email-Eu-core | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | bc | 4MB | POPT_GE_GRASP | +0.010pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | email-Eu-core | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.010pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | bc | 8MB | POPT_GE_GRASP | -0.000pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | email-Eu-core | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | bfs | 1MB | POPT_GE_GRASP | -17.751pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | email-Eu-core | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +17.751pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | bfs | 4MB | POPT_GE_GRASP | -0.000pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | email-Eu-core | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | bfs | 8MB | POPT_GE_GRASP | +0.002pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | email-Eu-core | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.002pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | pr | 1MB | POPT_GE_GRASP | -0.388pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | email-Eu-core | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.388pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | pr | 4MB | POPT_GE_GRASP | +0.025pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | email-Eu-core | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.025pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | email-Eu-core | pr | 8MB | GRASP | -0.043pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | email-Eu-core | pr | 8MB | POPT_GE_GRASP | +0.022pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | email-Eu-core | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.022pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | pr | 16kB | POPT_GE_GRASP | +0.031pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | roadNet-CA | pr | 16kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.031pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | pr | 1MB | POPT_GE_GRASP | -6.467pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | roadNet-CA | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.467pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | pr | 256kB | POPT_GE_GRASP | -5.511pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | roadNet-CA | pr | 256kB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.511pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | pr | 4kB | POPT_GE_GRASP | -0.001pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | roadNet-CA | pr | 4kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.001pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | pr | 64kB | POPT_GE_GRASP | -1.294pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | roadNet-CA | pr | 64kB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.294pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 1MB | SRRIP | -1.822pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bc | 1MB | GRASP | -4.825pp | Faldu et al. HPCA 2020 Fig 11 |
| known_deviation | soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | +1.985pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.985pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 4MB | SRRIP | -2.097pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | +1.877pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.877pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 8MB | SRRIP | -2.911pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | +1.583pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.583pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 1MB | SRRIP | -1.581pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 1MB | GRASP | -1.154pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | -1.556pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.556pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 4MB | SRRIP | -0.661pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | -1.213pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.213pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 8MB | SRRIP | -0.496pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | -0.961pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.961pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 1MB | SRRIP | -1.101pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | +1.398pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.398pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 4MB | SRRIP | -1.130pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | +1.810pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.810pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 8MB | SRRIP | -2.152pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | +3.083pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.083pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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
| ok | soc-LiveJournal1 | sssp | 1MB | SRRIP | -0.060pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| within_tolerance | soc-LiveJournal1 | sssp | 1MB | POPT | -1.466pp | Balaji & Lucia HPCA 2021 Fig 10 |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT_GE_GRASP | -3.530pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.530pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 4MB | SRRIP | -1.329pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | +0.331pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.331pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 8MB | SRRIP | -1.450pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | +0.578pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.578pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 1MB | SRRIP | -2.319pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 1MB | GRASP | -8.691pp | Faldu et al. HPCA 2020 Fig 11 |
| known_deviation | soc-pokec | bc | 1MB | POPT_GE_GRASP | +3.079pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.079pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 4MB | SRRIP | -3.397pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-pokec | bc | 4MB | POPT_GE_GRASP | +1.717pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.717pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 8MB | SRRIP | -2.199pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 8MB | POPT_GE_GRASP | +0.285pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.285pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 1MB | SRRIP | -0.972pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 1MB | POPT_GE_GRASP | +1.262pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.262pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 4MB | SRRIP | -0.138pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 4MB | POPT_GE_GRASP | -0.283pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.283pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 8MB | SRRIP | -0.097pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 8MB | POPT_GE_GRASP | -1.102pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.102pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 1MB | SRRIP | -3.263pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-pokec | cc | 1MB | POPT_GE_GRASP | +10.580pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +10.580pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 4MB | SRRIP | -6.614pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-pokec | cc | 4MB | POPT_GE_GRASP | +5.608pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.608pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 8MB | SRRIP | +0.002pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-pokec | cc | 8MB | POPT_GE_GRASP | +0.037pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
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
| ok | soc-pokec | sssp | 1MB | SRRIP | -3.171pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 1MB | POPT | -6.942pp | Balaji & Lucia HPCA 2021 Fig 10 |
| known_deviation | soc-pokec | sssp | 1MB | POPT_GE_GRASP | +4.097pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.097pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 4MB | SRRIP | -1.335pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 4MB | POPT_GE_GRASP | +0.872pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.872pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 8MB | POPT_GE_GRASP | -0.001pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.001pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 1MB | SRRIP | -1.838pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bc | 1MB | GRASP | +0.020pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bc | 1MB | POPT_GE_GRASP | -0.512pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.512pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 4MB | SRRIP | -1.128pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 4MB | POPT_GE_GRASP | +2.902pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.902pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 8MB | SRRIP | -0.711pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 8MB | POPT_GE_GRASP | +3.050pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.050pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 1MB | SRRIP | -0.068pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 1MB | GRASP | -3.368pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bfs | 1MB | POPT_GE_GRASP | +1.098pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.098pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 4MB | SRRIP | -0.102pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 4MB | POPT_GE_GRASP | -0.386pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.386pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 8MB | SRRIP | -0.481pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 8MB | POPT_GE_GRASP | -2.481pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.481pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 1MB | SRRIP | -3.859pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 1MB | POPT_GE_GRASP | +1.268pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.268pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 4MB | SRRIP | -0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 4MB | POPT_GE_GRASP | +0.004pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.004pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 8MB | SRRIP | +0.002pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 8MB | POPT_GE_GRASP | -0.001pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
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
| ok | web-Google | sssp | 1MB | POPT_GE_GRASP | -0.858pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.858pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 4MB | SRRIP | +0.002pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 4MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 8MB | SRRIP | -0.003pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |

## Known deviations (registered)

- **cit-Patents/bc L3=4MB POPT_GE_GRASP** (+1.597pp): Same source-rooted BC frontier vs PR-rank mis-alignment as the soc-pokec/bc entries. cit-Patents (3.7M vertices) is highest-PR-hub rooted (`-r 0`); GRASP's hot-region pinning happens to track the directed BC sub-graph more closely than POPT's PR-ranked static schedule. Gap is ~1.1 pp at 4 MB and 1.6 pp at 8 MB.
- **cit-Patents/bc L3=8MB POPT_GE_GRASP** (+2.529pp): Same BC frontier vs PR-rank mis-alignment as the 4 MB entry; gap persists at ~1.6 pp at 8 MB because the issue is ordering, not capacity (Balaji HPCA21 §3.3).
- **cit-Patents/cc L3=1MB POPT_GE_GRASP** (+8.726pp): Same CC/POPT algorithmic mismatch as the soc-pokec/web-Google entries above. cit-Patents/CC at 1 MB shows the largest gap (~8.7 pp) because the citation graph has weak hub structure, so PR-ranking is a particularly poor proxy for CC's reuse.
- **cit-Patents/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+8.726pp): Phase-transition regime invariant fires because GRASP gains 13+ pp over LRU on cit-Patents/CC at 1 MB. POPT lags by ~8.7 pp due to the same CC/POPT algorithmic mismatch.
- **cit-Patents/cc L3=4MB POPT_GE_GRASP** (+3.648pp): Same CC/POPT algorithmic mismatch as the 1MB entry above: CC's union-find parent[] reuse is edge-driven, not PR-ranked, so POPT's static schedule mis-orders evictions. Gap narrows to ~3.6 pp at 4 MB because additional capacity masks some ordering errors but the schedule still spends evictions on the wrong vertices.
- **cit-Patents/cc L3=8MB POPT_GE_GRASP** (+1.528pp): Same CC/POPT mismatch; ~1.5 pp gap remains even at 8 MB on cit-Patents because the static PR-ranked schedule mis-orders evictions even when capacity is generous.
- **cit-Patents/sssp L3=4MB POPT_GE_GRASP** (+1.804pp): cit-Patents has weak PR-driven locality; at 4 MB POPT's static PR-ranked schedule mis-aligns with SSSP's frontier-driven access pattern by ~1.8 pp. Citation graphs don't follow the power-law hub structure that POPT's oracle is calibrated for (Balaji HPCA21 §3.3 assumes PR-ordering tracks reuse).
- **cit-Patents/sssp L3=8MB POPT_GE_GRASP** (+1.633pp): Same cit-Patents/SSSP rank-mis-alignment as the 4MB entry; ~1.6 pp gap persists even at 8 MB because the issue is ordering not capacity.
- **com-orkut/bc L3=1MB POPT_GE_GRASP** (+2.475pp): Same BC/PR-rank mismatch as soc-LJ; com-orkut has even higher clustering coefficient (~0.17) so the gap widens to +2.5 pp at 1MB. GRASP wins by pinning the dense-subgraph pivots.
- **com-orkut/bc L3=4MB POPT_GE_GRASP** (+4.668pp): Same com-orkut/BC PR-rank vs dependency-frontier mismatch as the 1MB entry; gap widens to +4.7 pp at 4MB because the larger cache amplifies POPT's mis-ordering penalty — more dense-subgraph pivots survive a single BC iteration under GRASP's pinning but are evicted in PR-rank order under POPT before the reverse pass needs them.
- **com-orkut/bc L3=8MB POPT_GE_GRASP** (+4.727pp): Same com-orkut/BC PR-rank vs dependency-frontier mismatch as the 1MB / 4MB entries; gap is +4.7 pp at 8MB. com-orkut's high clustering coefficient (~0.17) and 17:1 hub-edge concentration make the PR-rank schedule maximally mis-aligned with BC's reverse-BFS dependency accumulation; GRASP wins because its hot-vertex hot-region holds the same dense-subgraph pivots that BC repeatedly re-visits across source vertices.
- **com-orkut/cc L3=1MB POPT_GE_GRASP** (+11.016pp): Same CC/POPT mismatch as soc-pokec/cit-Patents CC entries; com-orkut shows the largest gap (~11 pp) due to maximal PR-rank vs edge-order mis-alignment.
- **com-orkut/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+11.016pp): Phase-transition regime invariant fires because GRASP gains 13+ pp over LRU on com-orkut/CC at 1 MB. POPT lags by ~11 pp - the largest CC/POPT gap observed - because Orkut is the highest-clustering corpus graph (CC ~0.17), so the static PR-ranked oracle is maximally mis-aligned with the union-find traversal's edge-driven reuse pattern.
- **com-orkut/cc L3=4MB POPT_GE_GRASP** (+10.055pp): Same CC/POPT algorithmic mismatch as the com-orkut/cc/1MB entry above: union-find's edge-driven parent[] reuse is mis-aligned with POPT's static PR-rank schedule. Gap persists at ~10 pp at 4MB because the mismatch is in ordering, not capacity (see Balaji HPCA21 §3.3 PR-ordering assumption).
- **com-orkut/cc L3=4MB POPT_NEAR_GRASP_IF_BIG_GAP** (+10.055pp): Same com-orkut/CC mismatch as the 1MB entry; gap persists at ~10 pp at 4MB because the issue is ordering not capacity (see Balaji HPCA21 §3.3 PR-ordering assumption).
- **com-orkut/cc L3=8MB POPT_GE_GRASP** (+6.187pp): Same CC/POPT mismatch; ~6 pp gap remains even at 8 MB on com-orkut because the static PR-ranked schedule mis-orders evictions of edge-driven CC reuse regardless of capacity.
- **soc-LiveJournal1/bc L3=1MB POPT_GE_GRASP** (+1.985pp): BC's reverse-BFS dependency accumulation traverses high-degree pivots in topologically-derived frontier order, not PR rank; GRASP pins the pivots while POPT's static PR-rank schedule evicts them. Gap is +2.0 pp at 1MB on soc-LiveJournal1.
- **soc-LiveJournal1/bc L3=4MB POPT_GE_GRASP** (+1.877pp): Same soc-LJ/BC PR-rank vs dependency-frontier mismatch as the 1MB entry; gap is +1.9 pp at 4MB.
- **soc-LiveJournal1/bc L3=8MB POPT_GE_GRASP** (+1.583pp): Same soc-LJ/BC PR-rank vs dependency-frontier mismatch as the 1MB / 4MB entries: BC's reverse-BFS dependency accumulation traverses high-degree pivots in frontier order, not PR rank, so POPT's static PR-rank schedule mis-orders the pivot working set. Gap is +1.6 pp at 8MB on soc-LiveJournal1 — narrower than 1MB/4MB because the larger cache absorbs more of the misordered evictions but the algorithmic mismatch persists.
- **soc-LiveJournal1/cc L3=4MB POPT_GE_GRASP** (+1.810pp): Same CC/POPT algorithmic mismatch as the soc-LiveJournal1/cc/1MB entry above: union-find's edge-driven parent[] reuse mis-aligns with POPT's static PR-rank schedule. Gap widens to +1.8 pp at 4MB because moderate capacity exposes more wasted evictions before the working set fits.
- **soc-LiveJournal1/cc L3=8MB POPT_GE_GRASP** (+3.083pp): Same soc-LJ/CC mismatch; gap is widest (+3.1 pp) at 8MB where GRASP's locality preservation pays off more than POPT's static PR-rank ordering of CC's union-find edge traversal.
- **soc-pokec/bc L3=1MB POPT_GE_GRASP** (+3.079pp): BC's forward + backward sweeps from `-r 0` (highest-PR hub) expand a frontier whose access pattern correlates with the directed sub-graph from the source rather than with global PR-rank order. POPT's static PR-ranked schedule mis-predicts reuse by ~3.1 pp at 1 MB. GRASP's hot-region pinning happens to track the BC frontier on soc-pokec. cit-Patents/bc and soc-LJ/bc do not exhibit this mismatch.
- **soc-pokec/bc L3=4MB POPT_GE_GRASP** (+1.717pp): Same source-rooted frontier vs PR-rank mis-alignment as the 1 MB entry; ~1.7 pp gap persists at 4 MB because the issue is ordering, not capacity.
- **soc-pokec/cc L3=1MB POPT_GE_GRASP** (+10.580pp): CC's parent[] access pattern is edge-driven, not PageRank-driven, so P-OPT's offset matrix is mis-aligned with the actual reuse order. POPT loses ~10 pp to GRASP at 1 MB. CC is outside the Balaji HPCA21 benchmark set; this is an algorithmic mismatch between the oracle's assumed access ranking and CC's behaviour.
- **soc-pokec/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+10.580pp): Phase-transition regime invariant fires because GRASP gains 13.1 pp over LRU. POPT only gains 2.5 pp due to the CC/POPT algorithmic mismatch (edge-driven vs PR-ranked access). Same root cause as the per-policy POPT_GE_GRASP entry above.
- **soc-pokec/cc L3=4MB POPT_GE_GRASP** (+5.608pp): Same CC/POPT mismatch as the soc-pokec/cc/1MB entry above; the gap narrows to ~5.6 pp at 4 MB because more of the parent[] array fits regardless of ordering.
- **soc-pokec/sssp L3=1MB POPT_GE_GRASP** (+4.097pp): Same frontier-vs-rank mis-alignment as cit-Patents/SSSP. The non-hub source (`-r 800000`) selected for soc-pokec/SSSP produces a BFS-like frontier that doesn't follow POPT's PR-rank ordering, so the static schedule mis-predicts reuse. ~4.1 pp gap at 1 MB closes to ~0.9 pp at 4 MB and ~0 at 8 MB. GRASP's hot-region pinning happens to align with the active frontier vertices in this regime. (Balaji HPCA21 §3.3 assumes PR-ordering tracks reuse.)
- **web-Google/bc L3=4MB POPT_GE_GRASP** (+2.902pp): Same Phase-1 root cause as the PR/4MB entry above. BC has four vertex-indexed property arrays totalling ~12 MB; at L3=4 MB they spill anyway, but POPT still wastes capacity evicting CSR/offset lines first. GRASP keeps them. ~3 pp deficit observed.
- **web-Google/bc L3=8MB POPT_GE_GRASP** (+3.050pp): BC working set on web-Google (~12 MB across 4 property arrays) spills 4 MB at L3=8 MB. POPT Phase 1 still preferentially evicts CSR/offset lines, ceding ~3 pp to GRASP which protects them via SRRIP semantics outside hot region. P-OPT HPCA21 §4.2 design behaviour.
- **web-Google/pr L3=4MB POPT_GE_GRASP** (+1.072pp): POPT Phase 1 aggressively evicts non-property cache lines (CSR offsets, frontier bitmap) regardless of their reuse. At L3=4 MB the property array (~3.66 MB) leaves only 0.34 MB for those lines, which thrash. GRASP retains them naturally. Matches P-OPT HPCA21 §4.2 design; not a sim bug.
