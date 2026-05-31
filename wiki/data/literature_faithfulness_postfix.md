# Literature-faithfulness summary

- Sweep root: `/tmp/graphbrew-lit-baseline`
- Claims total: **330**
- Verdict mix: **298 ok**, 1 within-tolerance, **0 DISAGREE**, 24 known-deviation, 0 missing, 7 insufficient_data
- min_accesses threshold: 10000

## Observed L3 miss-rates

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | Δ GRASP-LRU | Δ POPT-LRU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | bc | 1MB | 0.8843 | 0.8796 | 0.8985 | 0.8825 | +1.420pp | -0.182pp |
| cit-Patents | bc | 4MB | 0.7319 | 0.7228 | 0.7112 | 0.7164 | -2.062pp | -1.544pp |
| cit-Patents | bc | 8MB | 0.6071 | 0.5892 | 0.5575 | 0.5709 | -4.959pp | -3.617pp |
| cit-Patents | bfs | 1MB | 0.9708 | 0.9667 | 0.9593 | 0.9603 | -1.149pp | -1.047pp |
| cit-Patents | bfs | 4MB | 0.9269 | 0.9220 | 0.8846 | 0.8936 | -4.229pp | -3.332pp |
| cit-Patents | bfs | 8MB | 0.9021 | 0.8995 | 0.8153 | 0.8226 | -8.678pp | -7.943pp |
| cit-Patents | cc | 1MB | 0.6431 | 0.6010 | 0.5262 | 0.6134 | -11.691pp | -2.965pp |
| cit-Patents | cc | 4MB | 0.3170 | 0.2940 | 0.2473 | 0.2838 | -6.967pp | -3.319pp |
| cit-Patents | cc | 8MB | 0.1815 | 0.1601 | 0.1171 | 0.1324 | -6.445pp | -4.917pp |
| cit-Patents | pr | 1MB | 0.8944 | 0.8798 | 0.7855 | 0.7713 | -10.890pp | -12.314pp |
| cit-Patents | pr | 4MB | 0.5893 | 0.5466 | 0.4539 | 0.3674 | -13.543pp | -22.195pp |
| cit-Patents | pr | 8MB | 0.3256 | 0.2751 | 0.2141 | 0.1830 | -11.149pp | -14.260pp |
| cit-Patents | sssp | 1MB | 0.8486 | 0.8444 | 0.8180 | 0.8208 | -3.062pp | -2.782pp |
| cit-Patents | sssp | 4MB | 0.5536 | 0.5295 | 0.4729 | 0.4870 | -8.068pp | -6.657pp |
| cit-Patents | sssp | 8MB | 0.2603 | 0.2373 | 0.1980 | 0.2124 | -6.230pp | -4.793pp |
| com-orkut | bc | 1MB | 0.8624 | 0.8370 | 0.7999 | 0.8247 | -6.245pp | -3.770pp |
| com-orkut | bc | 4MB | 0.5827 | 0.5502 | 0.5242 | 0.5709 | -5.844pp | -1.176pp |
| com-orkut | bc | 8MB | 0.3821 | 0.3531 | 0.3359 | 0.3832 | -4.620pp | +0.107pp |
| com-orkut | bfs | 1MB | 0.9973 | 0.9976 | 0.9949 | 0.9978 | -0.234pp | +0.056pp |
| com-orkut | bfs | 4MB | 0.9949 | 0.9956 | 0.9785 | 0.9745 | -1.648pp | -2.043pp |
| com-orkut | bfs | 8MB | 0.9942 | 0.9945 | 0.9621 | 0.9548 | -3.215pp | -3.944pp |
| com-orkut | cc | 1MB | 0.6889 | 0.6510 | 0.5837 | 0.6939 | -10.523pp | +0.494pp |
| com-orkut | cc | 4MB | 0.5033 | 0.4888 | 0.3698 | 0.4704 | -13.345pp | -3.290pp |
| com-orkut | cc | 8MB | 0.4306 | 0.3212 | 0.2212 | 0.2830 | -20.943pp | -14.756pp |
| com-orkut | pr | 1MB | 0.7047 | 0.6625 | 0.6850 | 0.6109 | -1.977pp | -9.379pp |
| com-orkut | pr | 4MB | 0.3085 | 0.2717 | 0.2653 | 0.2124 | -4.318pp | -9.610pp |
| com-orkut | pr | 8MB | 0.1636 | 0.1409 | 0.1284 | 0.1083 | -3.523pp | -5.534pp |
| com-orkut | sssp | 1MB | 0.6603 | 0.6204 | 0.6338 | 0.5968 | -2.657pp | -6.354pp |
| com-orkut | sssp | 4MB | 0.1697 | 0.1541 | 0.1589 | 0.1473 | -1.084pp | -2.245pp |
| com-orkut | sssp | 8MB | 0.0246 | 0.0223 | 0.0224 | 0.0241 | -0.215pp | -0.053pp |
| delaunay_n19 | pr | 4kB | 0.9999 | 0.9999 | 0.9987 | 0.9995 | -0.115pp | -0.036pp |
| delaunay_n19 | pr | 16kB | 0.9999 | 0.9998 | 0.9956 | 0.9989 | -0.433pp | -0.102pp |
| delaunay_n19 | pr | 64kB | 0.9997 | 0.9996 | 0.9855 | 0.9832 | -1.416pp | -1.650pp |
| delaunay_n19 | pr | 256kB | 0.9927 | 0.9928 | 0.9480 | 0.9418 | -4.470pp | -5.085pp |
| delaunay_n19 | pr | 1MB | 0.9539 | 0.9394 | 0.8166 | 0.7780 | -13.730pp | -17.597pp |
| email-Eu-core | bc | 1MB | 0.0001 | 0.0001 | 0.0001 | 0.0001 | +0.000pp | +0.000pp |
| email-Eu-core | bc | 4MB | 0.0001 | 0.0001 | 0.0001 | 0.0002 | -0.000pp | +0.010pp |
| email-Eu-core | bc | 8MB | 0.0001 | 0.0001 | 0.0001 | 0.0001 | +0.000pp | -0.000pp |
| email-Eu-core | bfs | 1MB | 0.0458 | 0.0458 | 0.2234 | 0.0459 | +17.752pp | +0.000pp |
| email-Eu-core | bfs | 4MB | 0.0459 | 0.0459 | 0.0459 | 0.0458 | -0.000pp | -0.000pp |
| email-Eu-core | bfs | 8MB | 0.0459 | 0.0459 | 0.0458 | 0.0458 | -0.005pp | -0.003pp |
| email-Eu-core | pr | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 4MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| email-Eu-core | pr | 8MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.000pp | +0.000pp |
| roadNet-CA | bc | 4kB | 0.9998 | 0.9999 | 0.9993 | 0.9992 | -0.053pp | -0.064pp |
| roadNet-CA | bc | 16kB | 0.9998 | 0.9998 | 0.9977 | 0.9966 | -0.210pp | -0.320pp |
| roadNet-CA | bc | 64kB | 0.9965 | 0.9960 | 0.9922 | 0.9806 | -0.433pp | -1.590pp |
| roadNet-CA | bc | 256kB | 0.9221 | 0.9092 | 0.9753 | 0.9125 | +5.323pp | -0.959pp |
| roadNet-CA | bc | 1MB | 0.5480 | 0.5270 | 0.9083 | 0.7524 | +36.032pp | +20.444pp |
| roadNet-CA | bfs | 4kB | 0.9999 | 0.9999 | 0.9994 | 0.9999 | -0.052pp | -0.007pp |
| roadNet-CA | bfs | 16kB | 0.9999 | 0.9999 | 0.9980 | 0.9992 | -0.190pp | -0.068pp |
| roadNet-CA | bfs | 64kB | 0.9974 | 0.9968 | 0.9929 | 0.9732 | -0.454pp | -2.421pp |
| roadNet-CA | bfs | 256kB | 0.8910 | 0.8661 | 0.9809 | 0.7786 | +8.988pp | -11.246pp |
| roadNet-CA | bfs | 1MB | 0.2554 | 0.3018 | 0.9394 | 0.5893 | +68.404pp | +33.391pp |
| roadNet-CA | cc | 4kB | 0.9988 | 0.9984 | 0.9983 | 0.9989 | -0.050pp | +0.010pp |
| roadNet-CA | cc | 16kB | 0.9971 | 0.9974 | 0.9939 | 0.9960 | -0.325pp | -0.110pp |
| roadNet-CA | cc | 64kB | 0.9797 | 0.9758 | 0.9753 | 0.9781 | -0.438pp | -0.167pp |
| roadNet-CA | cc | 256kB | 0.9160 | 0.9106 | 0.9473 | 0.9195 | +3.133pp | +0.349pp |
| roadNet-CA | cc | 1MB | 0.7428 | 0.7482 | 0.8340 | 0.7934 | +9.120pp | +5.058pp |
| roadNet-CA | pr | 4kB | 0.9999 | 0.9999 | 0.9997 | 0.9997 | -0.022pp | -0.023pp |
| roadNet-CA | pr | 16kB | 0.9999 | 0.9999 | 0.9989 | 0.9992 | -0.099pp | -0.068pp |
| roadNet-CA | pr | 64kB | 0.9993 | 0.9996 | 0.9956 | 0.9827 | -0.366pp | -1.660pp |
| roadNet-CA | pr | 256kB | 0.9856 | 0.9850 | 0.9885 | 0.9334 | +0.283pp | -5.228pp |
| roadNet-CA | pr | 1MB | 0.9417 | 0.9393 | 0.9564 | 0.8918 | +1.472pp | -4.995pp |
| roadNet-CA | sssp | 4kB | 0.9999 | 0.9999 | 0.9986 | 0.9984 | -0.125pp | -0.151pp |
| roadNet-CA | sssp | 16kB | 0.9979 | 0.9974 | 0.9959 | 0.9867 | -0.195pp | -1.118pp |
| roadNet-CA | sssp | 64kB | 0.9318 | 0.9149 | 0.9858 | 0.8893 | +5.401pp | -4.247pp |
| roadNet-CA | sssp | 256kB | 0.4409 | 0.4684 | 0.9623 | 0.5791 | +52.137pp | +13.820pp |
| roadNet-CA | sssp | 1MB | 0.1763 | 0.1913 | 0.8774 | 0.2772 | +70.117pp | +10.094pp |
| soc-LiveJournal1 | bc | 1MB | 0.8396 | 0.8175 | 0.7948 | 0.8096 | -4.474pp | -2.992pp |
| soc-LiveJournal1 | bc | 4MB | 0.6021 | 0.5769 | 0.5492 | 0.5637 | -5.286pp | -3.841pp |
| soc-LiveJournal1 | bc | 8MB | 0.4502 | 0.4248 | 0.3909 | 0.4074 | -5.934pp | -4.287pp |
| soc-LiveJournal1 | bfs | 1MB | 0.8176 | 0.7824 | 0.7413 | 0.7521 | -7.631pp | -6.541pp |
| soc-LiveJournal1 | bfs | 4MB | 0.5888 | 0.5642 | 0.5252 | 0.5376 | -6.357pp | -5.117pp |
| soc-LiveJournal1 | bfs | 8MB | 0.5106 | 0.4956 | 0.4473 | 0.4593 | -6.325pp | -5.125pp |
| soc-LiveJournal1 | cc | 1MB | 0.7832 | 0.7722 | 0.7450 | 0.7590 | -3.818pp | -2.421pp |
| soc-LiveJournal1 | cc | 4MB | 0.5437 | 0.5324 | 0.4774 | 0.4955 | -6.630pp | -4.820pp |
| soc-LiveJournal1 | cc | 8MB | 0.4314 | 0.4099 | 0.3193 | 0.3501 | -11.211pp | -8.128pp |
| soc-LiveJournal1 | pr | 1MB | 0.7319 | 0.6987 | 0.6839 | 0.6226 | -4.801pp | -10.930pp |
| soc-LiveJournal1 | pr | 4MB | 0.4257 | 0.3894 | 0.3506 | 0.2886 | -7.519pp | -13.714pp |
| soc-LiveJournal1 | pr | 8MB | 0.2757 | 0.2399 | 0.2012 | 0.1783 | -7.450pp | -9.741pp |
| soc-LiveJournal1 | sssp | 1MB | 0.7058 | 0.6799 | 0.6584 | 0.6582 | -4.737pp | -4.758pp |
| soc-LiveJournal1 | sssp | 4MB | 0.3627 | 0.3393 | 0.2963 | 0.3085 | -6.643pp | -5.423pp |
| soc-LiveJournal1 | sssp | 8MB | 0.1687 | 0.1483 | 0.1240 | 0.1315 | -4.469pp | -3.722pp |
| soc-pokec | bc | 1MB | 0.8513 | 0.8285 | 0.7646 | 0.7946 | -8.662pp | -5.661pp |
| soc-pokec | bc | 4MB | 0.4975 | 0.4643 | 0.4194 | 0.4355 | -7.812pp | -6.197pp |
| soc-pokec | bc | 8MB | 0.2454 | 0.2205 | 0.1994 | 0.2002 | -4.602pp | -4.525pp |
| soc-pokec | bfs | 1MB | 0.8717 | 0.8620 | 0.8430 | 0.8556 | -2.872pp | -1.610pp |
| soc-pokec | bfs | 4MB | 0.7401 | 0.7387 | 0.6764 | 0.6736 | -6.369pp | -6.652pp |
| soc-pokec | bfs | 8MB | 0.7146 | 0.7136 | 0.6038 | 0.5928 | -11.079pp | -12.180pp |
| soc-pokec | cc | 1MB | 0.6994 | 0.6668 | 0.5684 | 0.6742 | -13.105pp | -2.525pp |
| soc-pokec | cc | 4MB | 0.3961 | 0.3299 | 0.2264 | 0.2825 | -16.968pp | -11.360pp |
| soc-pokec | cc | 8MB | 0.0577 | 0.0577 | 0.0573 | 0.0577 | -0.039pp | -0.003pp |
| soc-pokec | pr | 1MB | 0.6796 | 0.6343 | 0.5433 | 0.5468 | -13.627pp | -13.279pp |
| soc-pokec | pr | 4MB | 0.1969 | 0.1615 | 0.1301 | 0.1236 | -6.682pp | -7.334pp |
| soc-pokec | pr | 8MB | 0.1004 | 0.0882 | 0.0792 | 0.0874 | -2.116pp | -1.297pp |
| soc-pokec | sssp | 1MB | 0.6351 | 0.6037 | 0.5269 | 0.5650 | -10.823pp | -7.014pp |
| soc-pokec | sssp | 4MB | 0.0976 | 0.0847 | 0.0668 | 0.0757 | -3.080pp | -2.197pp |
| soc-pokec | sssp | 8MB | 0.0029 | 0.0029 | 0.0029 | 0.0029 | -0.001pp | -0.002pp |
| web-Google | bc | 1MB | 0.6998 | 0.6882 | 0.7085 | 0.7036 | +0.867pp | +0.379pp |
| web-Google | bc | 4MB | 0.3693 | 0.3560 | 0.3428 | 0.3692 | -2.652pp | -0.007pp |
| web-Google | bc | 8MB | 0.1500 | 0.1441 | 0.1365 | 0.1629 | -1.343pp | +1.296pp |
| web-Google | bfs | 1MB | 0.9704 | 0.9699 | 0.9363 | 0.9467 | -3.418pp | -2.372pp |
| web-Google | bfs | 4MB | 0.9371 | 0.9348 | 0.7575 | 0.7556 | -17.965pp | -18.153pp |
| web-Google | bfs | 8MB | 0.8091 | 0.8013 | 0.7543 | 0.7293 | -5.474pp | -7.982pp |
| web-Google | cc | 1MB | 0.5608 | 0.5222 | 0.4944 | 0.5071 | -6.636pp | -5.369pp |
| web-Google | cc | 4MB | 0.0457 | 0.0457 | 0.0457 | 0.0457 | -0.005pp | -0.002pp |
| web-Google | cc | 8MB | 0.0456 | 0.0457 | 0.0457 | 0.0457 | +0.005pp | +0.005pp |
| web-Google | pr | 1MB | 0.6008 | 0.5442 | 0.4531 | 0.4168 | -14.770pp | -18.400pp |
| web-Google | pr | 4MB | 0.1414 | 0.1190 | 0.1057 | 0.1163 | -3.574pp | -2.511pp |
| web-Google | pr | 8MB | 0.0970 | 0.0934 | 0.0845 | 0.0841 | -1.256pp | -1.292pp |
| web-Google | sssp | 1MB | 0.5558 | 0.5507 | 0.5561 | 0.5475 | +0.034pp | -0.824pp |
| web-Google | sssp | 4MB | 0.0163 | 0.0163 | 0.0163 | 0.0163 | +0.001pp | +0.001pp |
| web-Google | sssp | 8MB | 0.0163 | 0.0163 | 0.0163 | 0.0163 | -0.003pp | -0.003pp |

## Per-claim verdicts

| status | graph | app | L3 | policy | Δ | citation |
|---|---|---|---|---|---:|---|
| ok | cit-Patents | bc | 1MB | SRRIP | -0.477pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| within_tolerance | cit-Patents | bc | 1MB | GRASP | +1.420pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bc | 1MB | POPT_GE_GRASP | -1.602pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.602pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 4MB | SRRIP | -0.908pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bc | 4MB | POPT_GE_GRASP | +0.518pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.518pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bc | 8MB | SRRIP | -1.796pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bc | 8MB | POPT_GE_GRASP | +1.342pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.342pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 1MB | SRRIP | -0.409pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 1MB | GRASP | -1.149pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | cit-Patents | bfs | 1MB | POPT_GE_GRASP | +0.102pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.102pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 4MB | SRRIP | -0.489pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 4MB | POPT_GE_GRASP | +0.897pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.897pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | bfs | 8MB | SRRIP | -0.255pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | cit-Patents | bfs | 8MB | POPT_GE_GRASP | +0.734pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.734pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 1MB | SRRIP | -4.213pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 1MB | POPT_GE_GRASP | +8.726pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| known_deviation | cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.726pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 4MB | SRRIP | -2.299pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 4MB | POPT_GE_GRASP | +3.648pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.648pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | cc | 8MB | SRRIP | -2.142pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | cit-Patents | cc | 8MB | POPT_GE_GRASP | +1.528pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.528pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 1MB | SRRIP | -1.468pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 1MB | GRASP | -10.890pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 1MB | POPT | -12.314pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | cit-Patents | pr | 1MB | POPT_GE_GRASP | -1.424pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | cit-Patents | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.424pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 4MB | SRRIP | -4.277pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 4MB | POPT_GE_GRASP | -8.652pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | cit-Patents | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +8.652pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | pr | 8MB | SRRIP | -5.045pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | cit-Patents | pr | 8MB | GRASP | -11.149pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | cit-Patents | pr | 8MB | POPT_GE_GRASP | -3.111pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | cit-Patents | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.111pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 1MB | SRRIP | -0.422pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 1MB | GRASP | -3.062pp | Balaji & Lucia HPCA 2021 Fig 10 (GRASP bar) |
| ok | cit-Patents | sssp | 1MB | POPT | -2.782pp | Balaji & Lucia HPCA 2021 Fig 10 |
| ok | cit-Patents | sssp | 1MB | POPT_GE_GRASP | +0.281pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.281pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 4MB | SRRIP | -2.406pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 4MB | POPT_GE_GRASP | +1.411pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.411pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | cit-Patents | sssp | 8MB | SRRIP | -2.301pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 8MB | POPT_GE_GRASP | +1.437pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | cit-Patents | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.437pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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
| ok | com-orkut | pr | 1MB | SRRIP | -4.225pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 1MB | GRASP | -1.977pp | Faldu et al. HPCA 2020 §6.1 (extrapolated to com-orkut from twitter Fig 10) |
| ok | com-orkut | pr | 1MB | POPT | -9.379pp | Balaji & Lucia HPCA 2021 §6 (extrapolated to com-orkut from twitter) |
| ok | com-orkut | pr | 1MB | POPT_GE_GRASP | -7.402pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | com-orkut | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +7.402pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 4MB | SRRIP | -3.681pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 4MB | POPT_GE_GRASP | -5.291pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | com-orkut | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +5.291pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | pr | 8MB | SRRIP | -2.272pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | com-orkut | pr | 8MB | GRASP | -3.523pp | Faldu et al. HPCA 2020 §6.1 (extrapolated from twitter Fig 10) |
| ok | com-orkut | pr | 8MB | POPT_GE_GRASP | -2.011pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | com-orkut | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.011pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 1MB | SRRIP | -3.990pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 1MB | POPT_GE_GRASP | -3.696pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.696pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 4MB | SRRIP | -1.559pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 4MB | POPT_GE_GRASP | -1.161pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.161pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | com-orkut | sssp | 8MB | SRRIP | -0.228pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 8MB | POPT_GE_GRASP | +0.162pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | com-orkut | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.162pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | delaunay_n19 | pr | 16kB | POPT_GE_GRASP | +0.332pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | delaunay_n19 | pr | 16kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.332pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | delaunay_n19 | pr | 1MB | POPT_GE_GRASP | -3.867pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | delaunay_n19 | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.867pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | delaunay_n19 | pr | 256kB | POPT_GE_GRASP | -0.615pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | delaunay_n19 | pr | 256kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.615pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | delaunay_n19 | pr | 4kB | POPT_GE_GRASP | +0.079pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | delaunay_n19 | pr | 4kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.079pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | delaunay_n19 | pr | 64kB | POPT_GE_GRASP | -0.234pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | delaunay_n19 | pr | 64kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.234pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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
| insufficient_data | email-Eu-core | pr | 1MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 |
| insufficient_data | email-Eu-core | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 4MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 |
| insufficient_data | email-Eu-core | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| insufficient_data | email-Eu-core | pr | 8MB | GRASP | +0.000pp | Faldu et al. HPCA 2020 Fig 10 |
| insufficient_data | email-Eu-core | pr | 8MB | POPT_GE_GRASP | +0.000pp | Balaji & Lucia HPCA 2021 §6.3 |
| insufficient_data | email-Eu-core | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bc | 16kB | POPT_GE_GRASP | -0.110pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bc | 16kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.110pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bc | 1MB | POPT_GE_GRASP | -15.588pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +15.588pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bc | 256kB | POPT_GE_GRASP | -6.282pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bc | 256kB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.282pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bc | 4kB | POPT_GE_GRASP | -0.011pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bc | 4kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.011pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bc | 64kB | POPT_GE_GRASP | -1.157pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bc | 64kB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.157pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bfs | 16kB | POPT_GE_GRASP | +0.122pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bfs | 16kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.122pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bfs | 1MB | POPT_GE_GRASP | -35.014pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +35.014pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bfs | 256kB | POPT_GE_GRASP | -20.233pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bfs | 256kB | POPT_NEAR_GRASP_IF_BIG_GAP | +20.233pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bfs | 4kB | POPT_GE_GRASP | +0.044pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bfs | 4kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.044pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | bfs | 64kB | POPT_GE_GRASP | -1.966pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | bfs | 64kB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.966pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | cc | 16kB | POPT_GE_GRASP | +0.215pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | cc | 16kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.215pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | cc | 1MB | POPT_GE_GRASP | -4.061pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +4.061pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | cc | 256kB | POPT_GE_GRASP | -2.785pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | cc | 256kB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.785pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | cc | 4kB | POPT_GE_GRASP | +0.060pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | cc | 4kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.060pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | cc | 64kB | POPT_GE_GRASP | +0.272pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | cc | 64kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.272pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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
| ok | roadNet-CA | sssp | 16kB | POPT_GE_GRASP | -0.923pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | sssp | 16kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.923pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | sssp | 1MB | POPT_GE_GRASP | -60.023pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +60.023pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | sssp | 256kB | POPT_GE_GRASP | -38.317pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | sssp | 256kB | POPT_NEAR_GRASP_IF_BIG_GAP | +38.317pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | sssp | 4kB | POPT_GE_GRASP | -0.026pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | sssp | 4kB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.026pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | roadNet-CA | sssp | 64kB | POPT_GE_GRASP | -9.648pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | roadNet-CA | sssp | 64kB | POPT_NEAR_GRASP_IF_BIG_GAP | +9.648pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 1MB | SRRIP | -2.205pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bc | 1MB | GRASP | -4.474pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | +1.482pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.482pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 4MB | SRRIP | -2.517pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | +1.445pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.445pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bc | 8MB | SRRIP | -2.547pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | +1.648pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.648pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 1MB | SRRIP | -3.518pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 1MB | GRASP | -7.631pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | +1.090pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.090pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 4MB | SRRIP | -2.464pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | +1.240pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.240pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | bfs | 8MB | SRRIP | -1.498pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | +1.200pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.200pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 1MB | SRRIP | -1.101pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | +1.398pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.398pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 4MB | SRRIP | -1.130pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | +1.810pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.810pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | cc | 8MB | SRRIP | -2.152pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| known_deviation | soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | +3.083pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.083pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 1MB | SRRIP | -3.325pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 1MB | GRASP | -4.801pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT | -10.930pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_GE_GRASP | -6.129pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-LiveJournal1 | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.129pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 4MB | SRRIP | -3.632pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_GE_GRASP | -6.195pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-LiveJournal1 | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +6.195pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | pr | 8MB | SRRIP | -3.583pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-LiveJournal1 | pr | 8MB | GRASP | -7.450pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_GE_GRASP | -2.291pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-LiveJournal1 | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.291pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 1MB | SRRIP | -2.595pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT | -4.758pp | Balaji & Lucia HPCA 2021 Fig 10 |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT_GE_GRASP | -0.021pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.021pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 4MB | SRRIP | -2.343pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | +1.220pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.220pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-LiveJournal1 | sssp | 8MB | SRRIP | -2.045pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | +0.747pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-LiveJournal1 | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.747pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 1MB | SRRIP | -2.276pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 1MB | GRASP | -8.662pp | Faldu et al. HPCA 2020 Fig 11 |
| known_deviation | soc-pokec | bc | 1MB | POPT_GE_GRASP | +3.001pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.001pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 4MB | SRRIP | -3.324pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | soc-pokec | bc | 4MB | POPT_GE_GRASP | +1.614pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.614pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | bc | 8MB | SRRIP | -2.486pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | soc-pokec | bc | 8MB | POPT_GE_GRASP | +0.077pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.077pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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
| ok | soc-pokec | pr | 1MB | SRRIP | -4.529pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 1MB | GRASP | -13.627pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 1MB | POPT | -13.279pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | soc-pokec | pr | 1MB | POPT_GE_GRASP | +0.347pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-pokec | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.347pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 4MB | SRRIP | -3.545pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 4MB | POPT_GE_GRASP | -0.652pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-pokec | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.652pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | pr | 8MB | SRRIP | -1.218pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | soc-pokec | pr | 8MB | GRASP | -2.116pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | soc-pokec | pr | 8MB | POPT_GE_GRASP | +0.819pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | soc-pokec | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.819pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 1MB | SRRIP | -3.146pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 1MB | POPT | -7.014pp | Balaji & Lucia HPCA 2021 Fig 10 |
| known_deviation | soc-pokec | sssp | 1MB | POPT_GE_GRASP | +3.810pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.810pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 4MB | SRRIP | -1.288pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 4MB | POPT_GE_GRASP | +0.884pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.884pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | soc-pokec | sssp | 8MB | SRRIP | -0.001pp | Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 8MB | POPT_GE_GRASP | -0.000pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | soc-pokec | sssp | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.000pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 1MB | SRRIP | -1.163pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bc | 1MB | GRASP | +0.867pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bc | 1MB | POPT_GE_GRASP | -0.488pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.488pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 4MB | SRRIP | -1.335pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 4MB | POPT_GE_GRASP | +2.646pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.646pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bc | 8MB | SRRIP | -0.590pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| known_deviation | web-Google | bc | 8MB | POPT_GE_GRASP | +2.640pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.640pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 1MB | SRRIP | -0.055pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 1MB | GRASP | -3.418pp | Faldu et al. HPCA 2020 Fig 11 |
| ok | web-Google | bfs | 1MB | POPT_GE_GRASP | +1.046pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bfs | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.046pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 4MB | SRRIP | -0.233pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 4MB | POPT_GE_GRASP | -0.188pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bfs | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.188pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | bfs | 8MB | SRRIP | -0.777pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended) |
| ok | web-Google | bfs | 8MB | POPT_GE_GRASP | -2.508pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | bfs | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +2.508pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 1MB | SRRIP | -3.859pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 1MB | POPT_GE_GRASP | +1.268pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.268pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 4MB | SRRIP | -0.000pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 4MB | POPT_GE_GRASP | +0.004pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.004pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | cc | 8MB | SRRIP | +0.002pp | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| ok | web-Google | cc | 8MB | POPT_GE_GRASP | -0.001pp | Balaji & Lucia HPCA 2021 §6.3 (extended) |
| ok | web-Google | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.001pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 1MB | SRRIP | -5.668pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 1MB | GRASP | -14.770pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 1MB | POPT | -18.400pp | Balaji & Lucia HPCA 2021 Fig 9 |
| ok | web-Google | pr | 1MB | POPT_GE_GRASP | -3.630pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | web-Google | pr | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | +3.630pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 4MB | SRRIP | -2.244pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| known_deviation | web-Google | pr | 4MB | POPT_GE_GRASP | +1.063pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | web-Google | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | +1.063pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
| ok | web-Google | pr | 8MB | SRRIP | -0.369pp | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| ok | web-Google | pr | 8MB | GRASP | -1.256pp | Faldu et al. HPCA 2020 Fig 10 |
| ok | web-Google | pr | 8MB | POPT_GE_GRASP | -0.036pp | Balaji & Lucia HPCA 2021 §6.3 |
| ok | web-Google | pr | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | +0.036pp | Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check |
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

- **cit-Patents/cc L3=1MB POPT_GE_GRASP** (+8.726pp): Same CC/POPT algorithmic mismatch as the soc-pokec/web-Google entries above. cit-Patents/CC at 1 MB shows the largest gap (~8.7 pp) because the citation graph has weak hub structure, so PR-ranking is a particularly poor proxy for CC's reuse.
- **cit-Patents/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+8.726pp): Phase-transition regime invariant fires because GRASP gains 13+ pp over LRU on cit-Patents/CC at 1 MB. POPT lags by ~8.7 pp due to the same CC/POPT algorithmic mismatch.
- **cit-Patents/cc L3=4MB POPT_GE_GRASP** (+3.648pp): Same CC/POPT algorithmic mismatch as the 1MB entry above: CC's union-find parent[] reuse is edge-driven, not PR-ranked, so POPT's static schedule mis-orders evictions. Gap narrows to ~3.6 pp at 4 MB because additional capacity masks some ordering errors but the schedule still spends evictions on the wrong vertices.
- **cit-Patents/cc L3=8MB POPT_GE_GRASP** (+1.528pp): Same CC/POPT mismatch; ~1.5 pp gap remains even at 8 MB on cit-Patents because the static PR-ranked schedule mis-orders evictions even when capacity is generous.
- **com-orkut/bc L3=1MB POPT_GE_GRASP** (+2.475pp): Same BC/PR-rank mismatch as soc-LJ; com-orkut has even higher clustering coefficient (~0.17) so the gap widens to +2.5 pp at 1MB. GRASP wins by pinning the dense-subgraph pivots.
- **com-orkut/bc L3=4MB POPT_GE_GRASP** (+4.668pp): Same com-orkut/BC PR-rank vs dependency-frontier mismatch as the 1MB entry; gap widens to +4.7 pp at 4MB because the larger cache amplifies POPT's mis-ordering penalty — more dense-subgraph pivots survive a single BC iteration under GRASP's pinning but are evicted in PR-rank order under POPT before the reverse pass needs them.
- **com-orkut/bc L3=8MB POPT_GE_GRASP** (+4.727pp): Same com-orkut/BC PR-rank vs dependency-frontier mismatch as the 1MB / 4MB entries; gap is +4.7 pp at 8MB. com-orkut's high clustering coefficient (~0.17) and 17:1 hub-edge concentration make the PR-rank schedule maximally mis-aligned with BC's reverse-BFS dependency accumulation; GRASP wins because its hot-vertex hot-region holds the same dense-subgraph pivots that BC repeatedly re-visits across source vertices.
- **com-orkut/cc L3=1MB POPT_GE_GRASP** (+11.016pp): Same CC/POPT mismatch as soc-pokec/cit-Patents CC entries; com-orkut shows the largest gap (~11 pp) due to maximal PR-rank vs edge-order mis-alignment.
- **com-orkut/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+11.016pp): Phase-transition regime invariant fires because GRASP gains 13+ pp over LRU on com-orkut/CC at 1 MB. POPT lags by ~11 pp - the largest CC/POPT gap observed - because Orkut is the highest-clustering corpus graph (CC ~0.17), so the static PR-ranked oracle is maximally mis-aligned with the union-find traversal's edge-driven reuse pattern.
- **com-orkut/cc L3=4MB POPT_GE_GRASP** (+10.055pp): Same CC/POPT algorithmic mismatch as the com-orkut/cc/1MB entry above: union-find's edge-driven parent[] reuse is mis-aligned with POPT's static PR-rank schedule. Gap persists at ~10 pp at 4MB because the mismatch is in ordering, not capacity (see Balaji HPCA21 §3.3 PR-ordering assumption).
- **com-orkut/cc L3=4MB POPT_NEAR_GRASP_IF_BIG_GAP** (+10.055pp): Same com-orkut/CC mismatch as the 1MB entry; gap persists at ~10 pp at 4MB because the issue is ordering not capacity (see Balaji HPCA21 §3.3 PR-ordering assumption).
- **com-orkut/cc L3=8MB POPT_GE_GRASP** (+6.187pp): Same CC/POPT mismatch; ~6 pp gap remains even at 8 MB on com-orkut because the static PR-ranked schedule mis-orders evictions of edge-driven CC reuse regardless of capacity.
- **soc-LiveJournal1/bc L3=8MB POPT_GE_GRASP** (+1.648pp): Same soc-LJ/BC PR-rank vs dependency-frontier mismatch as the 1MB / 4MB entries: BC's reverse-BFS dependency accumulation traverses high-degree pivots in frontier order, not PR rank, so POPT's static PR-rank schedule mis-orders the pivot working set. Gap is +1.6 pp at 8MB on soc-LiveJournal1 — narrower than 1MB/4MB because the larger cache absorbs more of the misordered evictions but the algorithmic mismatch persists.
- **soc-LiveJournal1/cc L3=4MB POPT_GE_GRASP** (+1.810pp): Same CC/POPT algorithmic mismatch as the soc-LiveJournal1/cc/1MB entry above: union-find's edge-driven parent[] reuse mis-aligns with POPT's static PR-rank schedule. Gap widens to +1.8 pp at 4MB because moderate capacity exposes more wasted evictions before the working set fits.
- **soc-LiveJournal1/cc L3=8MB POPT_GE_GRASP** (+3.083pp): Same soc-LJ/CC mismatch; gap is widest (+3.1 pp) at 8MB where GRASP's locality preservation pays off more than POPT's static PR-rank ordering of CC's union-find edge traversal.
- **soc-pokec/bc L3=1MB POPT_GE_GRASP** (+3.001pp): BC's forward + backward sweeps from `-r 0` (highest-PR hub) expand a frontier whose access pattern correlates with the directed sub-graph from the source rather than with global PR-rank order. POPT's static PR-ranked schedule mis-predicts reuse by ~3.1 pp at 1 MB. GRASP's hot-region pinning happens to track the BC frontier on soc-pokec. cit-Patents/bc and soc-LJ/bc do not exhibit this mismatch.
- **soc-pokec/bc L3=4MB POPT_GE_GRASP** (+1.614pp): Same source-rooted frontier vs PR-rank mis-alignment as the 1 MB entry; ~1.7 pp gap persists at 4 MB because the issue is ordering, not capacity.
- **soc-pokec/cc L3=1MB POPT_GE_GRASP** (+10.580pp): CC's parent[] access pattern is edge-driven, not PageRank-driven, so P-OPT's offset matrix is mis-aligned with the actual reuse order. POPT loses ~10 pp to GRASP at 1 MB. CC is outside the Balaji HPCA21 benchmark set; this is an algorithmic mismatch between the oracle's assumed access ranking and CC's behaviour.
- **soc-pokec/cc L3=1MB POPT_NEAR_GRASP_IF_BIG_GAP** (+10.580pp): Phase-transition regime invariant fires because GRASP gains 13.1 pp over LRU. POPT only gains 2.5 pp due to the CC/POPT algorithmic mismatch (edge-driven vs PR-ranked access). Same root cause as the per-policy POPT_GE_GRASP entry above.
- **soc-pokec/cc L3=4MB POPT_GE_GRASP** (+5.608pp): Same CC/POPT mismatch as the soc-pokec/cc/1MB entry above; the gap narrows to ~5.6 pp at 4 MB because more of the parent[] array fits regardless of ordering.
- **soc-pokec/sssp L3=1MB POPT_GE_GRASP** (+3.810pp): Same frontier-vs-rank mis-alignment as cit-Patents/SSSP. The non-hub source (`-r 800000`) selected for soc-pokec/SSSP produces a BFS-like frontier that doesn't follow POPT's PR-rank ordering, so the static schedule mis-predicts reuse. ~4.1 pp gap at 1 MB closes to ~0.9 pp at 4 MB and ~0 at 8 MB. GRASP's hot-region pinning happens to align with the active frontier vertices in this regime. (Balaji HPCA21 §3.3 assumes PR-ordering tracks reuse.)
- **web-Google/bc L3=4MB POPT_GE_GRASP** (+2.646pp): Same Phase-1 root cause as the PR/4MB entry above. BC has four vertex-indexed property arrays totalling ~12 MB; at L3=4 MB they spill anyway, but POPT still wastes capacity evicting CSR/offset lines first. GRASP keeps them. ~3 pp deficit observed.
- **web-Google/bc L3=8MB POPT_GE_GRASP** (+2.640pp): BC working set on web-Google (~12 MB across 4 property arrays) spills 4 MB at L3=8 MB. POPT Phase 1 still preferentially evicts CSR/offset lines, ceding ~3 pp to GRASP which protects them via SRRIP semantics outside hot region. P-OPT HPCA21 §4.2 design behaviour.
- **web-Google/pr L3=4MB POPT_GE_GRASP** (+1.063pp): POPT Phase 1 aggressively evicts non-property cache lines (CSR offsets, frontier bitmap) regardless of their reuse. At L3=4 MB the property array (~3.66 MB) leaves only 0.34 MB for those lines, which thrash. GRASP retains them naturally. Matches P-OPT HPCA21 §4.2 design; not a sim bug.
