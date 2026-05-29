# Paper baseline table

All Δ values are percentage-points of L3 miss-rate vs LRU at the same (graph, app, L3) tuple (negative = better).

Verdict suffix encodes the literature-claim outcome where applicable: `✓` ok, `~` within tolerance, `✗` DISAGREE, `?` insufficient data.


| Graph | App | L3 | LRU miss | SRRIP Δ | GRASP Δ | POPT Δ | GRASP claim | POPT claim |
|---|---|---|---:|---:|---:|---:|---|---|
| cit-Patents | bc | 1MB | 89.90% | -0.63 | +0.34 | -0.14 | ~ within_tol | — |
| cit-Patents | bc | 4MB | 74.14% | -1.20 | -2.05 | -0.46 | — | — |
| cit-Patents | bc | 8MB | 61.67% | -2.39 | -5.27 | -2.74 | — | — |
| cit-Patents | bfs | 1MB | 96.65% | -0.40 | -1.00 | -0.89 | ✓ ok | — |
| cit-Patents | bfs | 4MB | 90.98% | -0.68 | -4.08 | -3.70 | — | — |
| cit-Patents | bfs | 8MB | 87.11% | -0.31 | -8.62 | -7.94 | — | — |
| cit-Patents | cc | 1MB | 64.31% | -4.21 | -11.69 | -2.97 | — | — |
| cit-Patents | cc | 4MB | 31.70% | -2.30 | -6.97 | -3.32 | — | — |
| cit-Patents | cc | 8MB | 18.15% | -2.14 | -6.45 | -4.92 | — | — |
| cit-Patents | pr | 1MB | 89.23% | -1.43 | -10.65 | -11.70 | ✓ ok | ✓ ok |
| cit-Patents | pr | 4MB | 58.79% | -4.18 | -13.37 | -21.62 | — | — |
| cit-Patents | pr | 8MB | 32.48% | -4.92 | -11.02 | -13.86 | ~ within_tol | — |
| cit-Patents | sssp | 1MB | 85.29% | -0.32 | -3.45 | -2.59 | ✓ ok | ✓ ok |
| cit-Patents | sssp | 4MB | 56.31% | -2.46 | -8.67 | -6.87 | — | — |
| cit-Patents | sssp | 8MB | 26.90% | -2.50 | -6.90 | -5.27 | — | — |
| com-orkut | bc | 1MB | 86.24% | -2.54 | -6.24 | -3.77 | — | — |
| com-orkut | bc | 4MB | 58.27% | -3.24 | -5.84 | -1.18 | — | — |
| com-orkut | bc | 8MB | 38.21% | -2.90 | -4.62 | +0.11 | — | — |
| com-orkut | bfs | 1MB | 99.73% | +0.03 | -0.23 | +0.06 | — | — |
| com-orkut | bfs | 4MB | 99.49% | +0.07 | -1.65 | -2.04 | — | — |
| com-orkut | bfs | 8MB | 99.42% | +0.03 | -3.22 | -3.94 | — | — |
| com-orkut | cc | 1MB | 68.89% | -3.79 | -10.52 | +0.49 | — | — |
| com-orkut | cc | 4MB | 50.33% | -1.45 | -13.34 | -3.29 | — | — |
| com-orkut | cc | 8MB | 43.06% | -10.94 | -20.94 | -14.76 | — | — |
| com-orkut | pr | 1MB | 71.04% | -4.18 | -2.37 | -9.20 | ✓ ok | ✓ ok |
| com-orkut | pr | 4MB | 31.08% | -3.65 | -4.30 | -9.51 | — | — |
| com-orkut | pr | 8MB | 16.46% | -2.24 | -3.50 | -5.52 | ✓ ok | — |
| com-orkut | sssp | 1MB | 66.03% | -3.99 | -2.66 | -6.35 | — | — |
| com-orkut | sssp | 4MB | 16.97% | -1.56 | -1.08 | -2.24 | — | — |
| com-orkut | sssp | 8MB | 2.46% | -0.23 | -0.21 | -0.05 | — | — |
| email-Eu-core | bc | 1MB | 0.01% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bc | 4MB | 0.01% | -0.00 | -0.00 | +0.01 | — | — |
| email-Eu-core | bc | 8MB | 0.01% | -0.00 | +0.00 | -0.00 | — | — |
| email-Eu-core | bfs | 1MB | 4.58% | -0.00 | +17.75 | +0.00 | — | — |
| email-Eu-core | bfs | 4MB | 4.59% | +0.00 | -0.00 | -0.00 | — | — |
| email-Eu-core | bfs | 8MB | 4.59% | -0.00 | -0.00 | -0.00 | — | — |
| email-Eu-core | pr | 1MB | 0.81% | -0.04 | +0.33 | -0.06 | — | — |
| email-Eu-core | pr | 4MB | 0.77% | -0.01 | -0.01 | +0.01 | — | — |
| email-Eu-core | pr | 8MB | 0.80% | -0.02 | -0.04 | -0.02 | ✓ ok | — |
| roadNet-CA | pr | 1MB | 94.14% | +0.03 | +1.51 | -4.98 | — | — |
| soc-LiveJournal1 | bc | 1MB | 84.32% | -1.82 | -4.83 | -2.84 | ✓ ok | — |
| soc-LiveJournal1 | bc | 4MB | 60.38% | -2.10 | -5.27 | -3.39 | — | — |
| soc-LiveJournal1 | bc | 8MB | 45.28% | -2.91 | -6.24 | -4.66 | — | — |
| soc-LiveJournal1 | bfs | 1MB | 79.87% | -1.58 | -1.15 | -2.71 | ✓ ok | — |
| soc-LiveJournal1 | bfs | 4MB | 60.75% | -0.66 | -0.68 | -1.89 | — | — |
| soc-LiveJournal1 | bfs | 8MB | 55.71% | -0.50 | -2.60 | -3.56 | — | — |
| soc-LiveJournal1 | cc | 1MB | 78.32% | -1.10 | -3.82 | -2.42 | — | — |
| soc-LiveJournal1 | cc | 4MB | 54.37% | -1.13 | -6.63 | -4.82 | — | — |
| soc-LiveJournal1 | cc | 8MB | 43.14% | -2.15 | -11.21 | -8.13 | — | — |
| soc-LiveJournal1 | pr | 1MB | 73.22% | -3.28 | -4.79 | -10.74 | ✓ ok | ✓ ok |
| soc-LiveJournal1 | pr | 4MB | 42.61% | -3.64 | -7.47 | -13.45 | — | — |
| soc-LiveJournal1 | pr | 8MB | 27.56% | -3.50 | -7.38 | -9.60 | ✓ ok | — |
| soc-LiveJournal1 | sssp | 1MB | 68.77% | -0.06 | +2.06 | -1.47 | — | ~ within_tol |
| soc-LiveJournal1 | sssp | 4MB | 35.56% | -1.33 | -4.68 | -4.34 | — | — |
| soc-LiveJournal1 | sssp | 8MB | 16.47% | -1.45 | -3.59 | -3.01 | — | — |
| soc-pokec | bc | 1MB | 85.20% | -2.32 | -8.69 | -5.61 | ✓ ok | — |
| soc-pokec | bc | 4MB | 50.09% | -3.40 | -8.05 | -6.33 | — | — |
| soc-pokec | bc | 8MB | 24.35% | -2.20 | -4.41 | -4.13 | — | — |
| soc-pokec | bfs | 1MB | 87.17% | -0.97 | -2.87 | -1.61 | — | — |
| soc-pokec | bfs | 4MB | 74.01% | -0.14 | -6.37 | -6.65 | — | — |
| soc-pokec | bfs | 8MB | 71.46% | -0.10 | -11.08 | -12.18 | — | — |
| soc-pokec | cc | 1MB | 69.94% | -3.26 | -13.11 | -2.52 | — | — |
| soc-pokec | cc | 4MB | 39.61% | -6.61 | -16.97 | -11.36 | — | — |
| soc-pokec | cc | 8MB | 5.77% | +0.00 | -0.04 | -0.00 | — | — |
| soc-pokec | pr | 1MB | 67.96% | -4.52 | -13.62 | -13.20 | ✓ ok | ✓ ok |
| soc-pokec | pr | 4MB | 19.69% | -3.53 | -6.67 | -7.31 | — | — |
| soc-pokec | pr | 8MB | 10.04% | -1.21 | -2.11 | -1.30 | ✓ ok | — |
| soc-pokec | sssp | 1MB | 63.97% | -3.17 | -11.04 | -6.94 | — | ✓ ok |
| soc-pokec | sssp | 4MB | 9.87% | -1.33 | -3.09 | -2.22 | — | — |
| soc-pokec | sssp | 8MB | 0.30% | +0.00 | +0.00 | -0.00 | — | — |
| web-Google | bc | 1MB | 70.73% | -1.84 | +0.02 | -0.49 | ✓ ok | — |
| web-Google | bc | 4MB | 36.87% | -1.13 | -2.69 | +0.21 | — | — |
| web-Google | bc | 8MB | 15.15% | -0.71 | -1.58 | +1.47 | — | — |
| web-Google | bfs | 1MB | 97.00% | -0.07 | -3.37 | -2.27 | ✓ ok | — |
| web-Google | bfs | 4MB | 93.59% | -0.10 | -17.68 | -18.07 | — | — |
| web-Google | bfs | 8MB | 80.86% | -0.48 | -5.44 | -7.92 | — | — |
| web-Google | cc | 1MB | 56.08% | -3.86 | -6.64 | -5.37 | — | — |
| web-Google | cc | 4MB | 4.57% | -0.00 | -0.01 | -0.00 | — | — |
| web-Google | cc | 8MB | 4.56% | +0.00 | +0.01 | +0.00 | — | — |
| web-Google | pr | 1MB | 60.09% | -5.70 | -14.77 | -18.41 | ✓ ok | ✓ ok |
| web-Google | pr | 4MB | 14.14% | -2.24 | -3.58 | -2.51 | — | — |
| web-Google | pr | 8MB | 9.70% | -0.37 | -1.25 | -1.29 | ✓ ok | — |
| web-Google | sssp | 1MB | 55.58% | -0.50 | +0.03 | -0.82 | — | — |
| web-Google | sssp | 4MB | 1.63% | +0.00 | +0.00 | +0.00 | — | — |
| web-Google | sssp | 8MB | 1.63% | -0.00 | -0.00 | -0.00 | — | — |
