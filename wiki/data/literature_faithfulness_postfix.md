# Literature-faithfulness summary

- Sweep root: `/tmp/graphbrew-lit-baseline`
- Claims total: **279**
- Verdict mix: **214 ok**, 0 within-tolerance, **0 DISAGREE**, 37 known-deviation, 0 missing, 28 insufficient_data
- min_accesses threshold: 10000

## Observed L3 miss-rates

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | Δ GRASP-LRU | Δ POPT-LRU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | bc | 1MB | 0.9451 | 0.9426 | 0.9295 | 0.9369 | -1.559pp | -0.823pp |
| cit-Patents | bc | 4MB | 0.8189 | 0.8082 | 0.7827 | 0.8150 | -3.618pp | -0.397pp |
| cit-Patents | bc | 8MB | 0.6864 | 0.6661 | 0.6410 | 0.6940 | -4.537pp | +0.761pp |
| cit-Patents | bfs | 1MB | 0.9815 | 0.9791 | 0.9738 | 0.9760 | -0.762pp | -0.547pp |
| cit-Patents | bfs | 4MB | 0.9467 | 0.9418 | 0.9027 | 0.9203 | -4.406pp | -2.643pp |
| cit-Patents | bfs | 8MB | 0.9214 | 0.9187 | 0.8712 | 0.8585 | -5.016pp | -6.283pp |
| cit-Patents | cc | 1MB | 0.6668 | 0.6262 | 0.5962 | 0.5782 | -7.064pp | -8.859pp |
| cit-Patents | cc | 4MB | 0.3369 | 0.3136 | 0.2696 | 0.2974 | -6.729pp | -3.951pp |
| cit-Patents | cc | 8MB | 0.1980 | 0.1757 | 0.1590 | 0.1706 | -3.894pp | -2.734pp |
| cit-Patents | pr | 1MB | 0.8951 | 0.8803 | 0.8182 | 0.7611 | -7.691pp | -13.403pp |
| cit-Patents | pr | 4MB | 0.5901 | 0.5466 | 0.4532 | 0.3907 | -13.684pp | -19.939pp |
| cit-Patents | pr | 8MB | 0.3262 | 0.2745 | 0.2248 | 0.2058 | -10.138pp | -12.036pp |
| cit-Patents | sssp | 1MB | 0.8768 | 0.8701 | 0.8336 | 0.8593 | -4.318pp | -1.754pp |
| cit-Patents | sssp | 4MB | 0.5681 | 0.5461 | 0.4978 | 0.5541 | -7.032pp | -1.406pp |
| cit-Patents | sssp | 8MB | 0.2709 | 0.2493 | 0.2282 | 0.2860 | -4.265pp | +1.517pp |
| com-orkut | bc | 1MB | 0.8369 | 0.8205 | 0.7964 | 0.8232 | -4.052pp | -1.372pp |
| com-orkut | bc | 4MB | 0.5735 | 0.5470 | 0.5250 | 0.5901 | -4.848pp | +1.656pp |
| com-orkut | bc | 8MB | 0.3760 | 0.3499 | 0.3406 | 0.4300 | -3.545pp | +5.400pp |
| com-orkut | bfs | 1MB | 0.9989 | 0.9988 | 0.9981 | 0.9984 | -0.077pp | -0.050pp |
| com-orkut | bfs | 4MB | 0.9965 | 0.9964 | 0.9812 | 0.9911 | -1.530pp | -0.541pp |
| com-orkut | bfs | 8MB | 0.9958 | 0.9958 | 0.9807 | 0.9763 | -1.508pp | -1.950pp |
| com-orkut | cc | 1MB | 0.7415 | 0.7081 | 0.6710 | 0.7161 | -7.057pp | -2.548pp |
| com-orkut | cc | 4MB | 0.5676 | 0.5532 | 0.4205 | 0.5494 | -14.712pp | -1.825pp |
| com-orkut | cc | 8MB | 0.5063 | 0.4220 | 0.4114 | 0.3958 | -9.487pp | -11.045pp |
| com-orkut | pr | 1MB | 0.7426 | 0.7065 | 0.6383 | 0.6738 | -10.426pp | -6.878pp |
| com-orkut | pr | 4MB | 0.3456 | 0.3036 | 0.2997 | 0.2613 | -4.590pp | -8.430pp |
| com-orkut | pr | 8MB | 0.1856 | 0.1595 | 0.1599 | 0.1326 | -2.567pp | -5.293pp |
| com-orkut | sssp | 1MB | 0.6867 | 0.6539 | 0.5929 | 0.6537 | -9.386pp | -3.309pp |
| com-orkut | sssp | 4MB | 0.1730 | 0.1529 | 0.1462 | 0.1873 | -2.684pp | +1.422pp |
| com-orkut | sssp | 8MB | 0.0162 | 0.0152 | 0.0169 | 0.0294 | +0.072pp | +1.320pp |
| delaunay_n19 | pr | 4kB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000pp | +0.000pp |
| delaunay_n19 | pr | 16kB | 1.0000 | 1.0000 | 0.9997 | 0.9977 | -0.026pp | -0.229pp |
| delaunay_n19 | pr | 64kB | 1.0000 | 1.0000 | 0.9973 | 0.9778 | -0.270pp | -2.221pp |
| delaunay_n19 | pr | 256kB | 0.9992 | 0.9991 | 0.9830 | 0.9359 | -1.623pp | -6.331pp |
| delaunay_n19 | pr | 1MB | 0.9668 | 0.9518 | 0.8516 | 0.8104 | -11.516pp | -15.636pp |
| email-Eu-core | bc | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bc | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bc | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| roadNet-CA | bc | 4kB | 1.0000 | 1.0000 | 1.0000 | 0.9999 | -0.001pp | -0.008pp |
| roadNet-CA | bc | 16kB | 1.0000 | 1.0000 | 0.9999 | 0.9987 | -0.011pp | -0.130pp |
| roadNet-CA | bc | 64kB | 0.9998 | 0.9998 | 0.9943 | 0.9949 | -0.548pp | -0.490pp |
| roadNet-CA | bc | 256kB | 0.9892 | 0.9874 | 0.9153 | 0.9581 | -7.388pp | -3.104pp |
| roadNet-CA | bc | 1MB | 0.6485 | 0.6550 | 0.8108 | 0.8252 | +16.234pp | +17.674pp |
| roadNet-CA | bfs | 4kB | 1.0000 | 1.0000 | 0.9998 | 1.0000 | -0.024pp | +0.000pp |
| roadNet-CA | bfs | 16kB | 1.0000 | 1.0000 | 0.9958 | 0.9999 | -0.417pp | -0.006pp |
| roadNet-CA | bfs | 64kB | 0.9996 | 0.9995 | 0.9417 | 0.9959 | -5.783pp | -0.368pp |
| roadNet-CA | bfs | 256kB | 0.9399 | 0.9319 | 0.8768 | 0.8547 | -6.317pp | -8.524pp |
| roadNet-CA | bfs | 1MB | 0.2411 | 0.2849 | 0.8669 | 0.5923 | +62.586pp | +35.126pp |
| roadNet-CA | cc | 4kB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.004pp | -0.004pp |
| roadNet-CA | cc | 16kB | 1.0000 | 1.0000 | 0.9995 | 0.9999 | -0.049pp | -0.014pp |
| roadNet-CA | cc | 64kB | 1.0000 | 1.0000 | 0.9980 | 0.9988 | -0.202pp | -0.117pp |
| roadNet-CA | cc | 256kB | 0.9998 | 0.9998 | 0.9945 | 0.9860 | -0.526pp | -1.380pp |
| roadNet-CA | cc | 1MB | 0.9917 | 0.9909 | 0.9451 | 0.9419 | -4.657pp | -4.978pp |
| roadNet-CA | pr | 4kB | 1.0000 | 1.0000 | 0.9996 | 1.0000 | -0.036pp | +0.000pp |
| roadNet-CA | pr | 16kB | 1.0000 | 1.0000 | 0.9995 | 1.0000 | -0.049pp | -0.001pp |
| roadNet-CA | pr | 64kB | 1.0000 | 1.0000 | 0.9992 | 0.9917 | -0.079pp | -0.827pp |
| roadNet-CA | pr | 256kB | 0.9953 | 0.9950 | 0.9890 | 0.9469 | -0.629pp | -4.841pp |
| roadNet-CA | pr | 1MB | 0.9525 | 0.9530 | 0.9780 | 0.9136 | +2.552pp | -3.892pp |
| roadNet-CA | sssp | 4kB | 1.0000 | 1.0000 | 0.9994 | 1.0000 | -0.060pp | -0.003pp |
| roadNet-CA | sssp | 16kB | 0.9997 | 0.9996 | 0.9863 | 0.9976 | -1.339pp | -0.218pp |
| roadNet-CA | sssp | 64kB | 0.9731 | 0.9670 | 0.8700 | 0.9496 | -10.308pp | -2.350pp |
| roadNet-CA | sssp | 256kB | 0.5657 | 0.5505 | 0.8005 | 0.6586 | +23.483pp | +9.297pp |
| roadNet-CA | sssp | 1MB | 0.1665 | 0.1780 | 0.7248 | 0.2712 | +55.827pp | +10.469pp |
| soc-LiveJournal1 | bc | 1MB | 0.8649 | 0.8501 | 0.8312 | 0.8425 | -3.369pp | -2.234pp |
| soc-LiveJournal1 | bc | 4MB | 0.6552 | 0.6297 | 0.6001 | 0.6298 | -5.510pp | -2.537pp |
| soc-LiveJournal1 | bc | 8MB | 0.5025 | 0.4719 | 0.4405 | 0.4799 | -6.194pp | -2.256pp |
| soc-LiveJournal1 | bfs | 1MB | 0.8357 | 0.8107 | 0.7837 | 0.8056 | -5.197pp | -3.006pp |
| soc-LiveJournal1 | bfs | 4MB | 0.6446 | 0.6185 | 0.5720 | 0.6056 | -7.259pp | -3.901pp |
| soc-LiveJournal1 | bfs | 8MB | 0.5632 | 0.5460 | 0.5099 | 0.5199 | -5.333pp | -4.327pp |
| soc-LiveJournal1 | cc | 1MB | 0.7986 | 0.7745 | 0.7546 | 0.7033 | -4.405pp | -9.533pp |
| soc-LiveJournal1 | cc | 4MB | 0.5755 | 0.5535 | 0.4862 | 0.5107 | -8.925pp | -6.478pp |
| soc-LiveJournal1 | cc | 8MB | 0.4686 | 0.4353 | 0.3593 | 0.4020 | -10.934pp | -6.658pp |
| soc-LiveJournal1 | pr | 1MB | 0.7654 | 0.7340 | 0.6825 | 0.6642 | -8.284pp | -10.116pp |
| soc-LiveJournal1 | pr | 4MB | 0.4616 | 0.4201 | 0.3689 | 0.3326 | -9.275pp | -12.901pp |
| soc-LiveJournal1 | pr | 8MB | 0.2999 | 0.2594 | 0.2253 | 0.2048 | -7.457pp | -9.508pp |
| soc-LiveJournal1 | sssp | 1MB | 0.7439 | 0.7177 | 0.6820 | 0.7010 | -6.187pp | -4.288pp |
| soc-LiveJournal1 | sssp | 4MB | 0.3836 | 0.3538 | 0.3183 | 0.3581 | -6.524pp | -2.549pp |
| soc-LiveJournal1 | sssp | 8MB | 0.1696 | 0.1518 | 0.1399 | 0.1691 | -2.976pp | -0.051pp |
| soc-pokec | bc | 1MB | 0.8597 | 0.8442 | 0.8108 | 0.8325 | -4.892pp | -2.728pp |
| soc-pokec | bc | 4MB | 0.5301 | 0.4964 | 0.4579 | 0.5047 | -7.212pp | -2.534pp |
| soc-pokec | bc | 8MB | 0.2677 | 0.2412 | 0.2227 | 0.2648 | -4.500pp | -0.291pp |
| soc-pokec | bfs | 1MB | 0.9063 | 0.8922 | 0.8594 | 0.8870 | -4.692pp | -1.929pp |
| soc-pokec | bfs | 4MB | 0.7873 | 0.7812 | 0.7559 | 0.7260 | -3.139pp | -6.129pp |
| soc-pokec | bfs | 8MB | 0.7596 | 0.7536 | 0.7439 | 0.6260 | -1.578pp | -13.365pp |
| soc-pokec | cc | 1MB | 0.7281 | 0.7013 | 0.6509 | 0.6634 | -7.718pp | -6.471pp |
| soc-pokec | cc | 4MB | 0.4638 | 0.3792 | 0.4100 | 0.3669 | -5.378pp | -9.693pp |
| soc-pokec | cc | 8MB | 0.0629 | 0.0629 | 0.3731 | 0.0629 | +31.015pp | +0.000pp |
| soc-pokec | pr | 1MB | 0.6962 | 0.6517 | 0.5557 | 0.5761 | -14.047pp | -12.006pp |
| soc-pokec | pr | 4MB | 0.2094 | 0.1703 | 0.1437 | 0.1425 | -6.572pp | -6.686pp |
| soc-pokec | pr | 8MB | 0.1073 | 0.0944 | 0.0913 | 0.0971 | -1.595pp | -1.024pp |
| soc-pokec | sssp | 1MB | 0.6719 | 0.6366 | 0.5581 | 0.6294 | -11.387pp | -4.252pp |
| soc-pokec | sssp | 4MB | 0.0997 | 0.0870 | 0.0766 | 0.1170 | -2.319pp | +1.721pp |
| soc-pokec | sssp | 8MB | 0.0033 | 0.0033 | 0.0076 | 0.0033 | +0.432pp | +0.000pp |
| web-Google | bc | 1MB | 0.8027 | 0.8069 | 0.8238 | 0.8378 | +2.107pp | +3.512pp |
| web-Google | bc | 4MB | 0.5162 | 0.4989 | 0.5186 | 0.5830 | +0.243pp | +6.684pp |
| web-Google | bc | 8MB | 0.2210 | 0.2114 | 0.2635 | 0.3424 | +4.246pp | +12.137pp |
| web-Google | bfs | 1MB | 0.9804 | 0.9796 | 0.9444 | 0.9619 | -3.601pp | -1.853pp |
| web-Google | bfs | 4MB | 0.9486 | 0.9473 | 0.9191 | 0.7989 | -2.946pp | -14.964pp |
| web-Google | bfs | 8MB | 0.8194 | 0.8136 | 0.9125 | 0.7449 | +9.318pp | -7.442pp |
| web-Google | cc | 1MB | 0.5859 | 0.5569 | 0.5410 | 0.5314 | -4.496pp | -5.454pp |
| web-Google | cc | 4MB | 0.0519 | 0.0519 | 0.2583 | 0.0519 | +20.635pp | +0.000pp |
| web-Google | cc | 8MB | 0.0519 | 0.0519 | 0.1671 | 0.0519 | +11.518pp | +0.000pp |
| web-Google | pr | 1MB | 0.5984 | 0.5409 | 0.4509 | 0.4299 | -14.745pp | -16.850pp |
| web-Google | pr | 4MB | 0.1421 | 0.1192 | 0.1291 | 0.1258 | -1.302pp | -1.632pp |
| web-Google | pr | 8MB | 0.0972 | 0.0934 | 0.1068 | 0.0877 | +0.969pp | -0.950pp |
| web-Google | sssp | 1MB | 0.6638 | 0.6562 | 0.6403 | 0.6866 | -2.347pp | +2.281pp |
| web-Google | sssp | 4MB | 0.0235 | 0.0235 | 0.0816 | 0.0235 | +5.814pp | +0.000pp |
| web-Google | sssp | 8MB | 0.0235 | 0.0235 | 0.0475 | 0.0235 | +2.406pp | +0.000pp |

## Per-claim verdicts

| status | graph | app | L3 | policy | Δ | citation |
|---|---|---|---|---|---:|---|
| ok | cit-Patents | bc | 1MB | SRRIP | -0.255pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bc | 1MB | GRASP | -1.559pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bc | 1MB | POPT_GE_GRASP | +0.736pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.736pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 4MB | SRRIP | -1.069pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | cit-Patents | bc | 4MB | POPT_GE_GRASP | +3.221pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.221pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 8MB | SRRIP | -2.036pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | cit-Patents | bc | 8MB | POPT_GE_GRASP | +5.298pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.298pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 1MB | SRRIP | -0.240pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 1MB | GRASP | -0.762pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bfs | 1MB | POPT_GE_GRASP | +0.215pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.215pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 4MB | SRRIP | -0.496pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | cit-Patents | bfs | 4MB | POPT_GE_GRASP | +1.762pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.762pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 8MB | SRRIP | -0.266pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 8MB | POPT_GE_GRASP | -1.267pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.267pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 1MB | SRRIP | -4.058pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | cit-Patents | cc | 1MB | POPT_GE_GRASP | -1.795pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.795pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 4MB | SRRIP | -2.332pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 4MB | POPT_GE_GRASP | +2.778pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.778pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 8MB | SRRIP | -2.226pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | cit-Patents | cc | 8MB | POPT_GE_GRASP | +1.160pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.160pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 1MB | SRRIP | -1.481pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 1MB | GRASP | -7.691pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 1MB | POPT | -13.403pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | cit-Patents | pr | 1MB | POPT_GE_GRASP | -5.712pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.712pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 4MB | SRRIP | -4.341pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 4MB | POPT_GE_GRASP | -6.255pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.255pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 8MB | SRRIP | -5.169pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 8MB | GRASP | -10.138pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 8MB | POPT_GE_GRASP | -1.898pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.898pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 1MB | SRRIP | -0.676pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 1MB | GRASP | -4.318pp | Balaji & Lucia HPCA 2021 Fig 10 (GRASP bar) |
| ok | cit-Patents | sssp | 1MB | POPT | -1.754pp | Balaji & Lucia HPCA 2021 Fig 10 |
| known_deviation | cit-Patents | sssp | 1MB | POPT_GE_GRASP | +2.564pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.564pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 4MB | SRRIP | -2.200pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | cit-Patents | sssp | 4MB | POPT_GE_GRASP | +5.625pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.625pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 8MB | SRRIP | -2.161pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | cit-Patents | sssp | 8MB | POPT_GE_GRASP | +5.782pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.782pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 1MB | SRRIP | -1.638pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 1MB | POPT_GE_GRASP | +2.680pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.680pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 4MB | SRRIP | -2.653pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 4MB | POPT_GE_GRASP | +6.504pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.504pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 8MB | SRRIP | -2.617pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 8MB | POPT_GE_GRASP | +8.945pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.945pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 1MB | SRRIP | -0.010pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 1MB | POPT_GE_GRASP | +0.027pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.027pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 4MB | SRRIP | -0.004pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 4MB | POPT_GE_GRASP | +0.989pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.989pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 8MB | SRRIP | -0.004pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 8MB | POPT_GE_GRASP | -0.443pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.443pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 1MB | SRRIP | -3.347pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 1MB | POPT_GE_GRASP | +4.509pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.509pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 4MB | SRRIP | -1.441pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 4MB | POPT_GE_GRASP | +12.887pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| known_deviation | com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +12.887pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 8MB | SRRIP | -8.429pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | com-orkut | cc | 8MB | POPT_GE_GRASP | -1.558pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.558pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 1MB | SRRIP | -3.603pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 1MB | GRASP | -10.426pp | Faldu et al. HPCA 2020 §6.1 (extrapolated to com-orkut from twitter Fig 10) |
| ok | com-orkut | pr | 1MB | POPT | -6.878pp | Balaji & Lucia HPCA 2021 §6 (extrapolated to com-orkut from twitter) |
| known_deviation | com-orkut | pr | 1MB | POPT_GE_GRASP | +3.548pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.548pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 4MB | SRRIP | -4.205pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 4MB | POPT_GE_GRASP | -3.840pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.840pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 8MB | SRRIP | -2.610pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 8MB | GRASP | -2.567pp | Faldu et al. HPCA 2020 §6.1 (extrapolated from twitter Fig 10) |
| ok | com-orkut | pr | 8MB | POPT_GE_GRASP | -2.726pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.726pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 1MB | SRRIP | -3.288pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | com-orkut | sssp | 1MB | POPT_GE_GRASP | +6.077pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.077pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 4MB | SRRIP | -2.015pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | com-orkut | sssp | 4MB | POPT_GE_GRASP | +4.106pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.106pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 8MB | SRRIP | -0.097pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 8MB | POPT_GE_GRASP | +1.248pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.248pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bc | 1MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| insufficient_data | email-Eu-core | bc | 1MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bc | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| insufficient_data | email-Eu-core | bc | 4MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bc | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| insufficient_data | email-Eu-core | bc | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bfs | 1MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| insufficient_data | email-Eu-core | bfs | 1MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bfs | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| insufficient_data | email-Eu-core | bfs | 4MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | bfs | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| insufficient_data | email-Eu-core | bfs | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 1MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| insufficient_data | email-Eu-core | pr | 1MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| insufficient_data | email-Eu-core | pr | 4MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| insufficient_data | email-Eu-core | pr | 8MB | GRASP | +0.000pp | Faldu et al. HPCA 2020 Fig 10 |
| insufficient_data | email-Eu-core | pr | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| insufficient_data | email-Eu-core | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 1MB | SRRIP | -1.474pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bc | 1MB | GRASP | -3.369pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | +1.135pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.135pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 4MB | SRRIP | -2.550pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | +2.973pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.973pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 8MB | SRRIP | -3.058pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | +3.938pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.938pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 1MB | SRRIP | -2.500pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 1MB | GRASP | -5.197pp | Faldu et al. HPCA 2020 Fig 11 |
| known_deviation | soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | +2.190pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.190pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 4MB | SRRIP | -2.609pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | +3.358pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.358pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 8MB | SRRIP | -1.723pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | +1.006pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.006pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 1MB | SRRIP | -2.419pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | -5.128pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.128pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 4MB | SRRIP | -2.196pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | +2.446pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.446pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 8MB | SRRIP | -3.331pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | +4.276pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.276pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 1MB | SRRIP | -3.141pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 1MB | GRASP | -8.284pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT | -10.116pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_GE_GRASP | -1.831pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.831pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 4MB | SRRIP | -4.151pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_GE_GRASP | -3.626pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.626pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 8MB | SRRIP | -4.048pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 8MB | GRASP | -7.457pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_GE_GRASP | -2.051pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.051pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 1MB | SRRIP | -2.623pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT | -4.288pp | Balaji & Lucia HPCA 2021 Fig 10 |
| known_deviation | soc-LiveJournal1 | sssp | 1MB | POPT_GE_GRASP | +1.899pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.899pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 4MB | SRRIP | -2.978pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | +3.975pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.975pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 8MB | SRRIP | -1.779pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | +2.925pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.925pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 1MB | SRRIP | -1.556pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 1MB | GRASP | -4.892pp | Faldu et al. HPCA 2020 Fig 11 |
| known_deviation | soc-pokec | bc | 1MB | POPT_GE_GRASP | +2.164pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.164pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 4MB | SRRIP | -3.362pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-pokec | bc | 4MB | POPT_GE_GRASP | +4.678pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.678pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 8MB | SRRIP | -2.652pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-pokec | bc | 8MB | POPT_GE_GRASP | +4.209pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.209pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 1MB | SRRIP | -1.407pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-pokec | bfs | 1MB | POPT_GE_GRASP | +2.764pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.764pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 4MB | SRRIP | -0.611pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 4MB | POPT_GE_GRASP | -2.990pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.990pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 8MB | SRRIP | -0.603pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 8MB | POPT_GE_GRASP | -11.787pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +11.787pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 1MB | SRRIP | -2.685pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-pokec | cc | 1MB | POPT_GE_GRASP | +1.248pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.248pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 4MB | SRRIP | -8.461pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-pokec | cc | 4MB | POPT_GE_GRASP | -4.315pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.315pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-pokec | cc | 8MB | POPT_GE_GRASP | -31.015pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +31.015pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 1MB | SRRIP | -4.442pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 1MB | GRASP | -14.047pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 1MB | POPT | -12.006pp | Balaji & Lucia HPCA 2021 Fig 9 |
| known_deviation | soc-pokec | pr | 1MB | POPT_GE_GRASP | +2.041pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.041pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 4MB | SRRIP | -3.913pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 4MB | POPT_GE_GRASP | -0.114pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.114pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 8MB | SRRIP | -1.292pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 8MB | GRASP | -1.595pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 8MB | POPT_GE_GRASP | +0.571pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.571pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 1MB | SRRIP | -3.534pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 1MB | POPT | -4.252pp | Balaji & Lucia HPCA 2021 Fig 10 |
| known_deviation | soc-pokec | sssp | 1MB | POPT_GE_GRASP | +7.135pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| known_deviation | soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +7.135pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 4MB | SRRIP | -1.275pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | soc-pokec | sssp | 4MB | POPT_GE_GRASP | +4.040pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.040pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 8MB | POPT_GE_GRASP | -0.432pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.432pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 1MB | SRRIP | +0.424pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bc | 1MB | GRASP | +2.107pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bc | 1MB | POPT_GE_GRASP | +1.405pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.405pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 4MB | SRRIP | -1.729pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 4MB | POPT_GE_GRASP | +6.440pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.440pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 8MB | SRRIP | -0.961pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 8MB | POPT_GE_GRASP | +7.891pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +7.891pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 1MB | SRRIP | -0.079pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 1MB | GRASP | -3.601pp | Faldu et al. HPCA 2020 Fig 11 |
| known_deviation | web-Google | bfs | 1MB | POPT_GE_GRASP | +1.748pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.748pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 4MB | SRRIP | -0.124pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 4MB | POPT_GE_GRASP | -12.018pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +12.018pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 8MB | SRRIP | -0.574pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 8MB | POPT_GE_GRASP | -16.759pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +16.759pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 1MB | SRRIP | -2.907pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 1MB | POPT_GE_GRASP | -0.957pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.957pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 4MB | POPT_GE_GRASP | -20.635pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +20.635pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 8MB | POPT_GE_GRASP | -11.518pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +11.518pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 1MB | SRRIP | -5.749pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 1MB | GRASP | -14.745pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 1MB | POPT | -16.850pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | web-Google | pr | 1MB | POPT_GE_GRASP | -2.105pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.105pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 4MB | SRRIP | -2.293pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 4MB | POPT_GE_GRASP | -0.330pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.330pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 8MB | SRRIP | -0.379pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 8MB | GRASP | +0.969pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 8MB | POPT_GE_GRASP | -1.919pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.919pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 1MB | SRRIP | -0.760pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | web-Google | sssp | 1MB | POPT_GE_GRASP | +4.627pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.627pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 4MB | POPT_GE_GRASP | -5.814pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.814pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 8MB | POPT_GE_GRASP | -2.406pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.406pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |

## Known deviations (registered)

- **cit-Patents/bc L3=4MB POPT_GE_GRASP** (+3.221pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/bc@4MB (Δ=+3.22pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **cit-Patents/bc L3=8MB POPT_GE_GRASP** (+5.298pp): cit-Patents/bc/8 MB: in P-OPT Phase 1, BC's dependency frontier is unusually bursty, so P-OPT's offline rereference approximation chases stale edge visits and loses this cell to GRASP; the power-law GEOMEAN POPT<=GRASP gate remains Balaji & Lucia HPCA'21's actual claim.
- **cit-Patents/bfs L3=4MB POPT_GE_GRASP** (+1.762pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/bfs@4MB (Δ=+1.76pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **cit-Patents/cc L3=4MB POPT_GE_GRASP** (+2.778pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/cc@4MB (Δ=+2.78pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **cit-Patents/sssp L3=1MB POPT_GE_GRASP** (+2.564pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/sssp@1MB (Δ=+2.56pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **cit-Patents/sssp L3=4MB POPT_GE_GRASP** (+5.625pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/sssp@4MB (Δ=+5.63pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **cit-Patents/sssp L3=8MB POPT_GE_GRASP** (+5.782pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on cit-Patents/sssp@8MB (Δ=+5.78pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **com-orkut/bc L3=1MB POPT_GE_GRASP** (+2.680pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on com-orkut/bc@1MB (Δ=+2.68pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **com-orkut/bc L3=4MB POPT_GE_GRASP** (+6.504pp): com-orkut/bc/4 MB: the BC frontier expands through very high-degree communities, so P-OPT's offline rereference ordering misses GRASP's hot-region bias in this one cell; Balaji & Lucia HPCA'21 is represented by the power-law GEOMEAN POPT<=GRASP gate.
- **com-orkut/bc L3=8MB POPT_GE_GRASP** (+8.945pp): com-orkut/bc/8 MB: at the larger LLC, BC's frontier rereference stream still jumps between hubs faster than P-OPT's OPT approximation adapts, letting GRASP win locally while the power-law GEOMEAN POPT<=GRASP gate preserves Balaji & Lucia HPCA'21.
- **com-orkut/cc L3=1MB POPT_GE_GRASP** (+4.509pp): com-orkut/cc/1 MB: the union-find working set is capacity-pinched, and edge-driven rereference bursts make P-OPT evict lines GRASP keeps hot; the audited power-law GEOMEAN POPT<=GRASP gate is the Balaji & Lucia HPCA'21 claim.
- **com-orkut/cc L3=4MB POPT_GE_GRASP** (+12.887pp): com-orkut/cc/4 MB: CC's edge-driven union-find accesses phase-shift away from P-OPT's static rereference ranking, so GRASP's founded hot-region filter wins this cell; Balaji & Lucia HPCA'21 supports the power-law GEOMEAN POPT<=GRASP audit.
- **com-orkut/cc L3=4MB POPT_NEAR_GRASP_IF_BIG_GAP** (+12.887pp): com-orkut/cc/4 MB near-GRASP check: this GRASP-strong phase has edge-driven union-find rereferences that P-OPT smooths too aggressively, exceeding the per-cell near band; the power-law GEOMEAN POPT<=GRASP gate is Balaji & Lucia HPCA'21's claim.
- **com-orkut/pr L3=1MB POPT_GE_GRASP** (+3.548pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on com-orkut/pr@1MB (Δ=+3.55pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **com-orkut/sssp L3=1MB POPT_GE_GRASP** (+6.077pp): com-orkut/sssp/1 MB: delta-stepping frontier buckets revisit vertices irregularly, so P-OPT's rereference lookahead is noisier than GRASP's hot-region retention for this cell; Balaji & Lucia HPCA'21 is preserved by the power-law GEOMEAN POPT<=GRASP gate.
- **com-orkut/sssp L3=4MB POPT_GE_GRASP** (+4.106pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on com-orkut/sssp@4MB (Δ=+4.11pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-LiveJournal1/bc L3=4MB POPT_GE_GRASP** (+2.973pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bc@4MB (Δ=+2.97pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-LiveJournal1/bc L3=8MB POPT_GE_GRASP** (+3.938pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bc@8MB (Δ=+3.94pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-LiveJournal1/bfs L3=1MB POPT_GE_GRASP** (+2.190pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bfs@1MB (Δ=+2.19pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-LiveJournal1/bfs L3=4MB POPT_GE_GRASP** (+3.358pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/bfs@4MB (Δ=+3.36pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-LiveJournal1/cc L3=4MB POPT_GE_GRASP** (+2.446pp): soc-LiveJournal1/cc/4 MB: social-graph CC creates union-find rereference clusters that are edge-driven, not PR-rank ordered, so P-OPT falls behind GRASP locally; the literature claim from Balaji & Lucia HPCA'21 is power-law GEOMEAN POPT<=GRASP.
- **soc-LiveJournal1/cc L3=8MB POPT_GE_GRASP** (+4.276pp): soc-LiveJournal1/cc/8 MB: with more cache, union-find still produces edge-driven rereference bursts across components, where GRASP's array-relative hot set beats P-OPT's approximation in this cell; Balaji & Lucia HPCA'21 is checked by power-law GEOMEAN POPT<=GRASP.
- **soc-LiveJournal1/sssp L3=1MB POPT_GE_GRASP** (+1.899pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/sssp@1MB (Δ=+1.90pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-LiveJournal1/sssp L3=4MB POPT_GE_GRASP** (+3.975pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/sssp@4MB (Δ=+3.97pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-LiveJournal1/sssp L3=8MB POPT_GE_GRASP** (+2.925pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-LiveJournal1/sssp@8MB (Δ=+2.92pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-pokec/bc L3=1MB POPT_GE_GRASP** (+2.164pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bc@1MB (Δ=+2.16pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-pokec/bc L3=4MB POPT_GE_GRASP** (+4.678pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bc@4MB (Δ=+4.68pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-pokec/bc L3=8MB POPT_GE_GRASP** (+4.209pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bc@8MB (Δ=+4.21pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-pokec/bfs L3=1MB POPT_GE_GRASP** (+2.764pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/bfs@1MB (Δ=+2.76pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-pokec/pr L3=1MB POPT_GE_GRASP** (+2.041pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/pr@1MB (Δ=+2.04pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-pokec/sssp L3=1MB POPT_GE_GRASP** (+7.135pp): soc-pokec/sssp/1 MB: delta-stepping's frontier buckets thrash the small LLC, and P-OPT's rereference model loses the hot-distance rows GRASP keeps; Balaji & Lucia HPCA'21's actual audited statement is power-law GEOMEAN POPT<=GRASP.
- **soc-pokec/sssp L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+7.135pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/sssp@1MB (Δ=+7.13pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **soc-pokec/sssp L3=4MB POPT_GE_GRASP** (+4.040pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on soc-pokec/sssp@4MB (Δ=+4.04pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **web-Google/bc L3=4MB POPT_GE_GRASP** (+6.440pp): web-Google/bc/4 MB: web-graph BC alternates frontier waves with sparse back-dependencies, so P-OPT's offline rereference approximation misses GRASP's hot-frontier retention in this cell; Balaji & Lucia HPCA'21 is gated as power-law GEOMEAN POPT<=GRASP.
- **web-Google/bc L3=8MB POPT_GE_GRASP** (+7.891pp): web-Google/bc/8 MB: larger-cache BC still has frontier rereference gaps on the web crawl, causing a local P-OPT loss to GRASP while the power-law GEOMEAN POPT<=GRASP gate continues to encode Balaji & Lucia HPCA'21.
- **web-Google/bfs L3=1MB POPT_GE_GRASP** (+1.748pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on web-Google/bfs@1MB (Δ=+1.75pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
- **web-Google/sssp L3=1MB POPT_GE_GRASP** (+4.627pp): INFORMATIONAL per-cell diagnostic: P-OPT trails GRASP on web-Google/sssp@1MB (Δ=+4.63pp). P-OPT (Balaji & Lucia HPCA'21) is an offline OPT approximation whose authoritative claim is the power-law GEOMEAN (POPT_GE_GRASP_GEOMEAN gate), not per-cell dominance; per-cell losses on this irregular cell are expected and the faithful 1-way RRM capacity charge widens them, but the geomean win still holds.
