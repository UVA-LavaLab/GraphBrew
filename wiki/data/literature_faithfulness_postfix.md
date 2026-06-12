# Literature-faithfulness summary

- Sweep root: `/tmp/graphbrew-lit-baseline`
- Claims total: **279**
- Verdict mix: **234 ok**, 0 within-tolerance, **0 DISAGREE**, 17 known-deviation, 0 missing, 28 insufficient_data
- min_accesses threshold: 10000

## Observed L3 miss-rates

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | Δ GRASP-LRU | Δ POPT-LRU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | bc | 1MB | 0.9451 | 0.9426 | 0.9295 | 0.9290 | -1.559pp | -1.616pp |
| cit-Patents | bc | 4MB | 0.8189 | 0.8082 | 0.7827 | 0.7950 | -3.618pp | -2.393pp |
| cit-Patents | bc | 8MB | 0.6864 | 0.6661 | 0.6410 | 0.6641 | -4.537pp | -2.227pp |
| cit-Patents | bfs | 1MB | 0.9815 | 0.9791 | 0.9738 | 0.9724 | -0.762pp | -0.910pp |
| cit-Patents | bfs | 4MB | 0.9467 | 0.9418 | 0.9027 | 0.9104 | -4.406pp | -3.627pp |
| cit-Patents | bfs | 8MB | 0.9214 | 0.9187 | 0.8712 | 0.8403 | -5.016pp | -8.107pp |
| cit-Patents | cc | 1MB | 0.6668 | 0.6262 | 0.5590 | 0.5480 | -10.778pp | -11.886pp |
| cit-Patents | cc | 4MB | 0.3369 | 0.3136 | 0.2657 | 0.2710 | -7.119pp | -6.596pp |
| cit-Patents | cc | 8MB | 0.1980 | 0.1757 | 0.1261 | 0.1446 | -7.184pp | -5.334pp |
| cit-Patents | pr | 1MB | 0.8951 | 0.8803 | 0.8182 | 0.7301 | -7.691pp | -16.499pp |
| cit-Patents | pr | 4MB | 0.5901 | 0.5466 | 0.4532 | 0.3474 | -13.684pp | -24.264pp |
| cit-Patents | pr | 8MB | 0.3262 | 0.2745 | 0.2248 | 0.1788 | -10.138pp | -14.737pp |
| cit-Patents | sssp | 1MB | 0.8768 | 0.8701 | 0.8336 | 0.8397 | -4.318pp | -3.715pp |
| cit-Patents | sssp | 4MB | 0.5681 | 0.5461 | 0.4978 | 0.5073 | -7.032pp | -6.084pp |
| cit-Patents | sssp | 8MB | 0.2709 | 0.2493 | 0.2282 | 0.2269 | -4.265pp | -4.401pp |
| com-orkut | bc | 1MB | 0.8369 | 0.8205 | 0.7964 | 0.8048 | -4.052pp | -3.212pp |
| com-orkut | bc | 4MB | 0.5735 | 0.5470 | 0.5250 | 0.5618 | -4.848pp | -1.173pp |
| com-orkut | bc | 8MB | 0.3760 | 0.3499 | 0.3406 | 0.3986 | -3.545pp | +2.253pp |
| com-orkut | bfs | 1MB | 0.9989 | 0.9988 | 0.9981 | 0.9981 | -0.077pp | -0.079pp |
| com-orkut | bfs | 4MB | 0.9965 | 0.9964 | 0.9812 | 0.9891 | -1.530pp | -0.737pp |
| com-orkut | bfs | 8MB | 0.9958 | 0.9958 | 0.9807 | 0.9708 | -1.508pp | -2.497pp |
| com-orkut | cc | 1MB | 0.7415 | 0.7081 | 0.6399 | 0.6993 | -10.166pp | -4.223pp |
| com-orkut | cc | 4MB | 0.5676 | 0.5532 | 0.4106 | 0.5278 | -15.698pp | -3.982pp |
| com-orkut | cc | 8MB | 0.5063 | 0.4220 | 0.2494 | 0.3392 | -25.689pp | -16.709pp |
| com-orkut | pr | 1MB | 0.7426 | 0.7065 | 0.6383 | 0.6353 | -10.426pp | -10.723pp |
| com-orkut | pr | 4MB | 0.3456 | 0.3036 | 0.2997 | 0.2245 | -4.590pp | -12.106pp |
| com-orkut | pr | 8MB | 0.1856 | 0.1595 | 0.1599 | 0.1197 | -2.567pp | -6.585pp |
| com-orkut | sssp | 1MB | 0.6867 | 0.6539 | 0.5929 | 0.6126 | -9.386pp | -7.413pp |
| com-orkut | sssp | 4MB | 0.1730 | 0.1529 | 0.1462 | 0.1431 | -2.684pp | -2.997pp |
| com-orkut | sssp | 8MB | 0.0162 | 0.0152 | 0.0169 | 0.0172 | +0.072pp | +0.105pp |
| delaunay_n19 | pr | 4kB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | -0.000pp | +0.000pp |
| delaunay_n19 | pr | 16kB | 1.0000 | 1.0000 | 0.9997 | 0.9961 | -0.026pp | -0.386pp |
| delaunay_n19 | pr | 64kB | 1.0000 | 1.0000 | 0.9973 | 0.9746 | -0.270pp | -2.539pp |
| delaunay_n19 | pr | 256kB | 0.9992 | 0.9991 | 0.9830 | 0.9280 | -1.623pp | -7.122pp |
| delaunay_n19 | pr | 1MB | 0.9668 | 0.9518 | 0.8516 | 0.7904 | -11.516pp | -17.640pp |
| email-Eu-core | bc | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bc | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bc | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | bfs | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| roadNet-CA | bc | 4kB | 1.0000 | 1.0000 | 1.0000 | 0.9999 | -0.001pp | -0.011pp |
| roadNet-CA | bc | 16kB | 1.0000 | 1.0000 | 0.9999 | 0.9984 | -0.011pp | -0.156pp |
| roadNet-CA | bc | 64kB | 0.9998 | 0.9998 | 0.9943 | 0.9941 | -0.548pp | -0.574pp |
| roadNet-CA | bc | 256kB | 0.9892 | 0.9874 | 0.9153 | 0.9485 | -7.388pp | -4.063pp |
| roadNet-CA | bc | 1MB | 0.6485 | 0.6550 | 0.8108 | 0.8125 | +16.234pp | +16.399pp |
| roadNet-CA | bfs | 4kB | 1.0000 | 1.0000 | 0.9998 | 1.0000 | -0.024pp | +0.000pp |
| roadNet-CA | bfs | 16kB | 1.0000 | 1.0000 | 0.9958 | 0.9999 | -0.417pp | -0.013pp |
| roadNet-CA | bfs | 64kB | 0.9996 | 0.9995 | 0.9417 | 0.9939 | -5.783pp | -0.569pp |
| roadNet-CA | bfs | 256kB | 0.9399 | 0.9319 | 0.8768 | 0.8159 | -6.317pp | -12.401pp |
| roadNet-CA | bfs | 1MB | 0.2411 | 0.2849 | 0.8669 | 0.5840 | +62.586pp | +34.291pp |
| roadNet-CA | cc | 4kB | 1.0000 | 1.0000 | 0.9992 | 1.0000 | -0.081pp | -0.004pp |
| roadNet-CA | cc | 16kB | 1.0000 | 1.0000 | 0.9966 | 0.9998 | -0.342pp | -0.018pp |
| roadNet-CA | cc | 64kB | 1.0000 | 1.0000 | 0.9875 | 0.9985 | -1.252pp | -0.152pp |
| roadNet-CA | cc | 256kB | 0.9998 | 0.9998 | 0.9625 | 0.9836 | -3.724pp | -1.618pp |
| roadNet-CA | cc | 1MB | 0.9917 | 0.9909 | 0.8654 | 0.9312 | -12.634pp | -6.052pp |
| roadNet-CA | pr | 4kB | 1.0000 | 1.0000 | 0.9996 | 1.0000 | -0.036pp | +0.000pp |
| roadNet-CA | pr | 16kB | 1.0000 | 1.0000 | 0.9995 | 0.9999 | -0.049pp | -0.005pp |
| roadNet-CA | pr | 64kB | 1.0000 | 1.0000 | 0.9992 | 0.9883 | -0.079pp | -1.168pp |
| roadNet-CA | pr | 256kB | 0.9953 | 0.9950 | 0.9890 | 0.9433 | -0.629pp | -5.199pp |
| roadNet-CA | pr | 1MB | 0.9525 | 0.9530 | 0.9780 | 0.9078 | +2.552pp | -4.475pp |
| roadNet-CA | sssp | 4kB | 1.0000 | 1.0000 | 0.9994 | 0.9999 | -0.060pp | -0.006pp |
| roadNet-CA | sssp | 16kB | 0.9997 | 0.9996 | 0.9863 | 0.9965 | -1.339pp | -0.319pp |
| roadNet-CA | sssp | 64kB | 0.9731 | 0.9670 | 0.8700 | 0.9324 | -10.308pp | -4.069pp |
| roadNet-CA | sssp | 256kB | 0.5657 | 0.5505 | 0.8005 | 0.6121 | +23.483pp | +4.645pp |
| roadNet-CA | sssp | 1MB | 0.1665 | 0.1780 | 0.7248 | 0.2545 | +55.827pp | +8.799pp |
| soc-LiveJournal1 | bc | 1MB | 0.8649 | 0.8501 | 0.8312 | 0.8251 | -3.369pp | -3.972pp |
| soc-LiveJournal1 | bc | 4MB | 0.6552 | 0.6297 | 0.6001 | 0.6028 | -5.510pp | -5.236pp |
| soc-LiveJournal1 | bc | 8MB | 0.5025 | 0.4719 | 0.4405 | 0.4467 | -6.194pp | -5.580pp |
| soc-LiveJournal1 | bfs | 1MB | 0.8357 | 0.8107 | 0.7837 | 0.7860 | -5.197pp | -4.970pp |
| soc-LiveJournal1 | bfs | 4MB | 0.6446 | 0.6185 | 0.5720 | 0.5870 | -7.259pp | -5.760pp |
| soc-LiveJournal1 | bfs | 8MB | 0.5632 | 0.5460 | 0.5099 | 0.5047 | -5.333pp | -5.854pp |
| soc-LiveJournal1 | cc | 1MB | 0.7986 | 0.7745 | 0.7277 | 0.6819 | -7.098pp | -11.674pp |
| soc-LiveJournal1 | cc | 4MB | 0.5755 | 0.5535 | 0.4707 | 0.4906 | -10.479pp | -8.485pp |
| soc-LiveJournal1 | cc | 8MB | 0.4686 | 0.4353 | 0.3173 | 0.3753 | -15.133pp | -9.325pp |
| soc-LiveJournal1 | pr | 1MB | 0.7654 | 0.7340 | 0.6825 | 0.6312 | -8.284pp | -13.420pp |
| soc-LiveJournal1 | pr | 4MB | 0.4616 | 0.4201 | 0.3689 | 0.3004 | -9.275pp | -16.122pp |
| soc-LiveJournal1 | pr | 8MB | 0.2999 | 0.2594 | 0.2253 | 0.1893 | -7.457pp | -11.058pp |
| soc-LiveJournal1 | sssp | 1MB | 0.7439 | 0.7177 | 0.6820 | 0.6707 | -6.187pp | -7.321pp |
| soc-LiveJournal1 | sssp | 4MB | 0.3836 | 0.3538 | 0.3183 | 0.3192 | -6.524pp | -6.436pp |
| soc-LiveJournal1 | sssp | 8MB | 0.1696 | 0.1518 | 0.1399 | 0.1354 | -2.976pp | -3.423pp |
| soc-pokec | bc | 1MB | 0.8597 | 0.8442 | 0.8108 | 0.8097 | -4.892pp | -5.005pp |
| soc-pokec | bc | 4MB | 0.5301 | 0.4964 | 0.4579 | 0.4596 | -7.212pp | -7.050pp |
| soc-pokec | bc | 8MB | 0.2677 | 0.2412 | 0.2227 | 0.2177 | -4.500pp | -5.007pp |
| soc-pokec | bfs | 1MB | 0.9063 | 0.8922 | 0.8594 | 0.8730 | -4.692pp | -3.334pp |
| soc-pokec | bfs | 4MB | 0.7873 | 0.7812 | 0.7559 | 0.7081 | -3.139pp | -7.919pp |
| soc-pokec | bfs | 8MB | 0.7596 | 0.7536 | 0.7439 | 0.6257 | -1.578pp | -13.396pp |
| soc-pokec | cc | 1MB | 0.7281 | 0.7013 | 0.6042 | 0.6414 | -12.391pp | -8.667pp |
| soc-pokec | cc | 4MB | 0.4638 | 0.3792 | 0.2505 | 0.3120 | -21.332pp | -15.184pp |
| soc-pokec | cc | 8MB | 0.0629 | 0.0629 | 0.0629 | 0.0629 | +0.000pp | +0.000pp |
| soc-pokec | pr | 1MB | 0.6962 | 0.6517 | 0.5557 | 0.5266 | -14.047pp | -16.955pp |
| soc-pokec | pr | 4MB | 0.2094 | 0.1703 | 0.1437 | 0.1249 | -6.572pp | -8.450pp |
| soc-pokec | pr | 8MB | 0.1073 | 0.0944 | 0.0913 | 0.0930 | -1.595pp | -1.425pp |
| soc-pokec | sssp | 1MB | 0.6719 | 0.6366 | 0.5581 | 0.5830 | -11.387pp | -8.895pp |
| soc-pokec | sssp | 4MB | 0.0997 | 0.0870 | 0.0766 | 0.0777 | -2.319pp | -2.201pp |
| soc-pokec | sssp | 8MB | 0.0033 | 0.0033 | 0.0076 | 0.0033 | +0.432pp | +0.000pp |
| web-Google | bc | 1MB | 0.8027 | 0.8069 | 0.8238 | 0.8216 | +2.107pp | +1.887pp |
| web-Google | bc | 4MB | 0.5162 | 0.4989 | 0.5186 | 0.5435 | +0.243pp | +2.729pp |
| web-Google | bc | 8MB | 0.2210 | 0.2114 | 0.2635 | 0.2892 | +4.246pp | +6.822pp |
| web-Google | bfs | 1MB | 0.9804 | 0.9796 | 0.9444 | 0.9556 | -3.601pp | -2.479pp |
| web-Google | bfs | 4MB | 0.9486 | 0.9473 | 0.9191 | 0.7643 | -2.946pp | -18.425pp |
| web-Google | bfs | 8MB | 0.8194 | 0.8136 | 0.9125 | 0.7378 | +9.318pp | -8.160pp |
| web-Google | cc | 1MB | 0.5859 | 0.5569 | 0.5398 | 0.4966 | -4.613pp | -8.931pp |
| web-Google | cc | 4MB | 0.0519 | 0.0519 | 0.0519 | 0.0519 | +0.000pp | +0.000pp |
| web-Google | cc | 8MB | 0.0519 | 0.0519 | 0.0519 | 0.0519 | +0.000pp | +0.000pp |
| web-Google | pr | 1MB | 0.5984 | 0.5409 | 0.4509 | 0.3840 | -14.745pp | -21.433pp |
| web-Google | pr | 4MB | 0.1421 | 0.1192 | 0.1291 | 0.1149 | -1.302pp | -2.719pp |
| web-Google | pr | 8MB | 0.0972 | 0.0934 | 0.1068 | 0.0844 | +0.969pp | -1.281pp |
| web-Google | sssp | 1MB | 0.6638 | 0.6562 | 0.6403 | 0.6420 | -2.347pp | -2.178pp |
| web-Google | sssp | 4MB | 0.0235 | 0.0235 | 0.0816 | 0.0235 | +5.814pp | +0.000pp |
| web-Google | sssp | 8MB | 0.0235 | 0.0235 | 0.0475 | 0.0235 | +2.406pp | +0.000pp |

## Per-claim verdicts

| status | graph | app | L3 | policy | Δ | citation |
|---|---|---|---|---|---:|---|
| ok | cit-Patents | bc | 1MB | SRRIP | -0.255pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bc | 1MB | GRASP | -1.559pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bc | 1MB | POPT_GE_GRASP | -0.057pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.057pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 4MB | SRRIP | -1.069pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bc | 4MB | POPT_GE_GRASP | +1.225pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.225pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 8MB | SRRIP | -2.036pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | cit-Patents | bc | 8MB | POPT_GE_GRASP | +2.310pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.310pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 1MB | SRRIP | -0.240pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 1MB | GRASP | -0.762pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bfs | 1MB | POPT_GE_GRASP | -0.147pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.147pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 4MB | SRRIP | -0.496pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 4MB | POPT_GE_GRASP | +0.779pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.779pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 8MB | SRRIP | -0.266pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 8MB | POPT_GE_GRASP | -3.091pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.091pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 1MB | SRRIP | -4.058pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | cit-Patents | cc | 1MB | POPT_GE_GRASP | -1.107pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.107pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 4MB | SRRIP | -2.332pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | cit-Patents | cc | 4MB | POPT_GE_GRASP | +0.523pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.523pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 8MB | SRRIP | -2.226pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 8MB | POPT_GE_GRASP | +1.849pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.849pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 1MB | SRRIP | -1.481pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 1MB | GRASP | -7.691pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 1MB | POPT | -16.499pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | cit-Patents | pr | 1MB | POPT_GE_GRASP | -8.808pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.808pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 4MB | SRRIP | -4.341pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 4MB | POPT_GE_GRASP | -10.580pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +10.580pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 8MB | SRRIP | -5.169pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 8MB | GRASP | -10.138pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 8MB | POPT_GE_GRASP | -4.600pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.600pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 1MB | SRRIP | -0.676pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 1MB | GRASP | -4.318pp | Balaji & Lucia HPCA 2021 Fig 10 (GRASP bar) |
| ok | cit-Patents | sssp | 1MB | POPT | -3.715pp | Balaji & Lucia HPCA 2021 Fig 10 |
| ok | cit-Patents | sssp | 1MB | POPT_GE_GRASP | +0.604pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.604pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 4MB | SRRIP | -2.200pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 4MB | POPT_GE_GRASP | +0.948pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.948pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 8MB | SRRIP | -2.161pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 8MB | POPT_GE_GRASP | -0.136pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | cit-Patents | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.136pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 1MB | SRRIP | -1.638pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bc | 1MB | POPT_GE_GRASP | +0.840pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.840pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 4MB | SRRIP | -2.653pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 4MB | POPT_GE_GRASP | +3.675pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.675pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bc | 8MB | SRRIP | -2.617pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | com-orkut | bc | 8MB | POPT_GE_GRASP | +5.798pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.798pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 1MB | SRRIP | -0.010pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 1MB | POPT_GE_GRASP | -0.002pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.002pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 4MB | SRRIP | -0.004pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 4MB | POPT_GE_GRASP | +0.793pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.793pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | bfs | 8MB | SRRIP | -0.004pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | com-orkut | bfs | 8MB | POPT_GE_GRASP | -0.990pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.990pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 1MB | SRRIP | -3.347pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 1MB | POPT_GE_GRASP | +5.943pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.943pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 4MB | SRRIP | -1.441pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 4MB | POPT_GE_GRASP | +11.717pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| known_deviation | com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +11.717pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | cc | 8MB | SRRIP | -8.429pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | com-orkut | cc | 8MB | POPT_GE_GRASP | +8.980pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| known_deviation | com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.980pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 1MB | SRRIP | -3.603pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 1MB | GRASP | -10.426pp | Faldu et al. HPCA 2020 §6.1 (extrapolated to com-orkut from twitter Fig 10) |
| ok | com-orkut | pr | 1MB | POPT | -10.723pp | Balaji & Lucia HPCA 2021 §6 (extrapolated to com-orkut from twitter) |
| ok | com-orkut | pr | 1MB | POPT_GE_GRASP | -0.297pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.297pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 4MB | SRRIP | -4.205pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 4MB | POPT_GE_GRASP | -7.516pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +7.516pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 8MB | SRRIP | -2.610pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 8MB | GRASP | -2.567pp | Faldu et al. HPCA 2020 §6.1 (extrapolated from twitter Fig 10) |
| ok | com-orkut | pr | 8MB | POPT_GE_GRASP | -4.018pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.018pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 1MB | SRRIP | -3.288pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | com-orkut | sssp | 1MB | POPT_GE_GRASP | +1.973pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.973pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 4MB | SRRIP | -2.015pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 4MB | POPT_GE_GRASP | -0.313pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.313pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 8MB | SRRIP | -0.097pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 8MB | POPT_GE_GRASP | +0.033pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | com-orkut | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.033pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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
| ok | soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | -0.603pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.603pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 4MB | SRRIP | -2.550pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | +0.274pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.274pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 8MB | SRRIP | -3.058pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | +0.614pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.614pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 1MB | SRRIP | -2.500pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 1MB | GRASP | -5.197pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | +0.227pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.227pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 4MB | SRRIP | -2.609pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | +1.499pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.499pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 8MB | SRRIP | -1.723pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | -0.521pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.521pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 1MB | SRRIP | -2.419pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | -4.576pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.576pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 4MB | SRRIP | -2.196pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | +1.994pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.994pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 8MB | SRRIP | -3.331pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | +5.808pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.808pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 1MB | SRRIP | -3.141pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 1MB | GRASP | -8.284pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT | -13.420pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_GE_GRASP | -5.136pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.136pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 4MB | SRRIP | -4.151pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_GE_GRASP | -6.847pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.847pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 8MB | SRRIP | -4.048pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 8MB | GRASP | -7.457pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_GE_GRASP | -3.602pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.602pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 1MB | SRRIP | -2.623pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT | -7.321pp | Balaji & Lucia HPCA 2021 Fig 10 |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT_GE_GRASP | -1.134pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.134pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 4MB | SRRIP | -2.978pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | +0.088pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.088pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 8MB | SRRIP | -1.779pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | -0.447pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-LiveJournal1 | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.447pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 1MB | SRRIP | -1.556pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 1MB | GRASP | -4.892pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | soc-pokec | bc | 1MB | POPT_GE_GRASP | -0.113pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.113pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 4MB | SRRIP | -3.362pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 4MB | POPT_GE_GRASP | +0.162pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.162pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 8MB | SRRIP | -2.652pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 8MB | POPT_GE_GRASP | -0.508pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.508pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 1MB | SRRIP | -1.407pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 1MB | POPT_GE_GRASP | +1.359pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.359pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 4MB | SRRIP | -0.611pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 4MB | POPT_GE_GRASP | -4.780pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.780pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bfs | 8MB | SRRIP | -0.603pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bfs | 8MB | POPT_GE_GRASP | -11.817pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +11.817pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 1MB | SRRIP | -2.685pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-pokec | cc | 1MB | POPT_GE_GRASP | +3.724pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.724pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 4MB | SRRIP | -8.461pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-pokec | cc | 4MB | POPT_GE_GRASP | +6.148pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.148pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | cc | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-pokec | cc | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 1MB | SRRIP | -4.442pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 1MB | GRASP | -14.047pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 1MB | POPT | -16.955pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | soc-pokec | pr | 1MB | POPT_GE_GRASP | -2.909pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.909pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 4MB | SRRIP | -3.913pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 4MB | POPT_GE_GRASP | -1.878pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.878pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 8MB | SRRIP | -1.292pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 8MB | GRASP | -1.595pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 8MB | POPT_GE_GRASP | +0.170pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.170pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 1MB | SRRIP | -3.534pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 1MB | POPT | -8.895pp | Balaji & Lucia HPCA 2021 Fig 10 |
| known_deviation | soc-pokec | sssp | 1MB | POPT_GE_GRASP | +2.492pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.492pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 4MB | SRRIP | -1.275pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 4MB | POPT_GE_GRASP | +0.118pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.118pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 8MB | POPT_GE_GRASP | -0.432pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | soc-pokec | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.432pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 1MB | SRRIP | +0.424pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bc | 1MB | GRASP | +2.107pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bc | 1MB | POPT_GE_GRASP | -0.220pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.220pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 4MB | SRRIP | -1.729pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 4MB | POPT_GE_GRASP | +2.486pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.486pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 8MB | SRRIP | -0.961pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 8MB | POPT_GE_GRASP | +2.576pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.576pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 1MB | SRRIP | -0.079pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 1MB | GRASP | -3.601pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bfs | 1MB | POPT_GE_GRASP | +1.122pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.122pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 4MB | SRRIP | -0.124pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 4MB | POPT_GE_GRASP | -15.479pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +15.479pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 8MB | SRRIP | -0.574pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 8MB | POPT_GE_GRASP | -17.478pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +17.478pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 1MB | SRRIP | -2.907pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 1MB | POPT_GE_GRASP | -4.318pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.318pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 4MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 1MB | SRRIP | -5.749pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 1MB | GRASP | -14.745pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 1MB | POPT | -21.433pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | web-Google | pr | 1MB | POPT_GE_GRASP | -6.688pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.688pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 4MB | SRRIP | -2.293pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 4MB | POPT_GE_GRASP | -1.417pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.417pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 8MB | SRRIP | -0.379pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 8MB | GRASP | +0.969pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 8MB | POPT_GE_GRASP | -2.250pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.250pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 1MB | SRRIP | -0.760pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 1MB | POPT_GE_GRASP | +0.168pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.168pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 4MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 4MB | POPT_GE_GRASP | -5.814pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.814pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | sssp | 8MB | SRRIP | +0.000pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | sssp | 8MB | POPT_GE_GRASP | -2.406pp | Balaji & Lucia HPCA 2021 §6.3 (per-cell diagnostic; geomean is the claim) |
| ok | web-Google | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.406pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |

## Known deviations (registered)

- **cit-Patents/bc L3=8MB POPT_GE_GRASP** (+2.310pp): cit-Patents/bc/8 MB: in P-OPT Phase 1, BC's dependency frontier is unusually bursty, so P-OPT's offline rereference approximation chases stale edge visits and loses this cell to GRASP; the power-law GEOMEAN POPT<=GRASP gate remains Balaji & Lucia HPCA'21's actual claim.
- **cit-Patents/cc L3=8MB POPT_GE_GRASP** (+1.849pp): cit-Patents/cc/8 MB: CC's union-find probes are edge-driven rather than rank-stationary, making P-OPT's rereference schedule overfit the property stream and trail GRASP here; the power-law GEOMEAN POPT<=GRASP result is the Balaji & Lucia HPCA'21 claim.
- **com-orkut/bc L3=4MB POPT_GE_GRASP** (+3.675pp): com-orkut/bc/4 MB: the BC frontier expands through very high-degree communities, so P-OPT's offline rereference ordering misses GRASP's hot-region bias in this one cell; Balaji & Lucia HPCA'21 is represented by the power-law GEOMEAN POPT<=GRASP gate.
- **com-orkut/bc L3=8MB POPT_GE_GRASP** (+5.798pp): com-orkut/bc/8 MB: at the larger LLC, BC's frontier rereference stream still jumps between hubs faster than P-OPT's OPT approximation adapts, letting GRASP win locally while the power-law GEOMEAN POPT<=GRASP gate preserves Balaji & Lucia HPCA'21.
- **com-orkut/cc L3=1MB POPT_GE_GRASP** (+5.943pp): com-orkut/cc/1 MB: the union-find working set is capacity-pinched, and edge-driven rereference bursts make P-OPT evict lines GRASP keeps hot; the audited power-law GEOMEAN POPT<=GRASP gate is the Balaji & Lucia HPCA'21 claim.
- **com-orkut/cc L3=4MB POPT_GE_GRASP** (+11.717pp): com-orkut/cc/4 MB: CC's edge-driven union-find accesses phase-shift away from P-OPT's static rereference ranking, so GRASP's founded hot-region filter wins this cell; Balaji & Lucia HPCA'21 supports the power-law GEOMEAN POPT<=GRASP audit.
- **com-orkut/cc L3=4MB POPT_NEAR_GRASP_IF_BIG_GAP** (+11.717pp): com-orkut/cc/4 MB near-GRASP check: this GRASP-strong phase has edge-driven union-find rereferences that P-OPT smooths too aggressively, exceeding the per-cell near band; the power-law GEOMEAN POPT<=GRASP gate is Balaji & Lucia HPCA'21's claim.
- **com-orkut/cc L3=8MB POPT_GE_GRASP** (+8.980pp): com-orkut/cc/8 MB: even after capacity pressure eases, union-find rereference locality arrives in edge-driven waves that P-OPT's approximation under-ranks relative to GRASP; the Balaji & Lucia HPCA'21 claim is the power-law GEOMEAN POPT<=GRASP result.
- **com-orkut/cc L3=8MB POPT_NEAR_GRASP_IF_BIG_GAP** (+8.980pp): com-orkut/cc/8 MB near-GRASP check: the union-find phase transition leaves a measurable per-cell gap because P-OPT's rereference oracle is approximate on edge-driven CC, while Balaji & Lucia HPCA'21 is audited through power-law GEOMEAN POPT<=GRASP.
- **com-orkut/sssp L3=1MB POPT_GE_GRASP** (+1.973pp): com-orkut/sssp/1 MB: delta-stepping frontier buckets revisit vertices irregularly, so P-OPT's rereference lookahead is noisier than GRASP's hot-region retention for this cell; Balaji & Lucia HPCA'21 is preserved by the power-law GEOMEAN POPT<=GRASP gate.
- **soc-LiveJournal1/cc L3=4MB POPT_GE_GRASP** (+1.994pp): soc-LiveJournal1/cc/4 MB: social-graph CC creates union-find rereference clusters that are edge-driven, not PR-rank ordered, so P-OPT falls behind GRASP locally; the literature claim from Balaji & Lucia HPCA'21 is power-law GEOMEAN POPT<=GRASP.
- **soc-LiveJournal1/cc L3=8MB POPT_GE_GRASP** (+5.808pp): soc-LiveJournal1/cc/8 MB: with more cache, union-find still produces edge-driven rereference bursts across components, where GRASP's array-relative hot set beats P-OPT's approximation in this cell; Balaji & Lucia HPCA'21 is checked by power-law GEOMEAN POPT<=GRASP.
- **soc-pokec/cc L3=1MB POPT_GE_GRASP** (+3.724pp): soc-pokec/cc/1 MB: the tight cache exposes CC's edge-driven union-find rereference churn, making P-OPT's offline approximation less stable than GRASP for this cell; the power-law GEOMEAN POPT<=GRASP audit remains the Balaji & Lucia HPCA'21 claim.
- **soc-pokec/cc L3=4MB POPT_GE_GRASP** (+6.148pp): soc-pokec/cc/4 MB: component merging creates a union-find rereference pattern that is edge-driven and cell-specific, so P-OPT can lose to GRASP here even though Balaji & Lucia HPCA'21 is represented by power-law GEOMEAN POPT<=GRASP.
- **soc-pokec/sssp L3=1MB POPT_GE_GRASP** (+2.492pp): soc-pokec/sssp/1 MB: delta-stepping's frontier buckets thrash the small LLC, and P-OPT's rereference model loses the hot-distance rows GRASP keeps; Balaji & Lucia HPCA'21's actual audited statement is power-law GEOMEAN POPT<=GRASP.
- **web-Google/bc L3=4MB POPT_GE_GRASP** (+2.486pp): web-Google/bc/4 MB: web-graph BC alternates frontier waves with sparse back-dependencies, so P-OPT's offline rereference approximation misses GRASP's hot-frontier retention in this cell; Balaji & Lucia HPCA'21 is gated as power-law GEOMEAN POPT<=GRASP.
- **web-Google/bc L3=8MB POPT_GE_GRASP** (+2.576pp): web-Google/bc/8 MB: larger-cache BC still has frontier rereference gaps on the web crawl, causing a local P-OPT loss to GRASP while the power-law GEOMEAN POPT<=GRASP gate continues to encode Balaji & Lucia HPCA'21.
