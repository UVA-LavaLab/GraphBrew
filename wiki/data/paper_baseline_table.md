# Paper baseline table

All Δ values are percentage-points of L3 miss-rate vs LRU at the same (graph, app, L3) tuple (negative = better).

Verdict suffix encodes the literature-claim outcome where applicable: `✓` ok, `~` within tolerance, `✗` DISAGREE, `?` insufficient data.

| Graph | App | L3 | LRU miss | SRRIP Δ | GRASP Δ | POPT Δ | GRASP claim | POPT claim |
|---|---|---|---:|---:|---:|---:|---|---|
| cit-Patents | bc | 1MB | 88.43% | -0.48 | +1.42 | -0.18 | ✓ ok | — |
| cit-Patents | bc | 4MB | 73.19% | -0.91 | -2.06 | -1.54 | — | — |
| cit-Patents | bc | 8MB | 60.71% | -1.80 | -4.96 | -3.62 | — | — |
| cit-Patents | bfs | 1MB | 97.08% | -0.41 | -1.15 | -1.05 | ✓ ok | — |
| cit-Patents | bfs | 4MB | 92.69% | -0.49 | -4.23 | -3.33 | — | — |
| cit-Patents | bfs | 8MB | 90.21% | -0.25 | -8.68 | -7.94 | — | — |
| cit-Patents | cc | 1MB | 64.31% | -4.21 | -11.69 | -2.97 | — | — |
| cit-Patents | cc | 4MB | 31.70% | -2.30 | -6.97 | -3.32 | — | — |
| cit-Patents | cc | 8MB | 18.15% | -2.14 | -6.45 | -4.92 | — | — |
| cit-Patents | pr | 1MB | 89.44% | -1.47 | -10.89 | -12.31 | ✓ ok | ✓ ok |
| cit-Patents | pr | 4MB | 58.93% | -4.28 | -13.54 | -22.19 | — | — |
| cit-Patents | pr | 8MB | 32.56% | -5.04 | -11.15 | -14.26 | ~ within_tol | — |
| cit-Patents | sssp | 1MB | 84.86% | -0.42 | -3.06 | -2.78 | ✓ ok | ✓ ok |
| cit-Patents | sssp | 4MB | 55.36% | -2.41 | -8.07 | -6.66 | — | — |
| cit-Patents | sssp | 8MB | 26.03% | -2.30 | -6.23 | -4.79 | — | — |
| com-orkut | bc | 1MB | 86.24% | -2.54 | -6.24 | -3.77 | — | — |
| com-orkut | bc | 4MB | 58.27% | -3.24 | -5.84 | -1.18 | — | — |
| com-orkut | bc | 8MB | 38.21% | -2.90 | -4.62 | +0.11 | — | — |
| com-orkut | bfs | 1MB | 99.73% | +0.03 | -0.23 | +0.06 | — | — |
| com-orkut | bfs | 4MB | 99.49% | +0.07 | -1.65 | -2.04 | — | — |
| com-orkut | bfs | 8MB | 99.42% | +0.03 | -3.22 | -3.94 | — | — |
| com-orkut | cc | 1MB | 68.89% | -3.79 | -10.52 | +0.49 | — | — |
| com-orkut | cc | 4MB | 50.33% | -1.45 | -13.34 | -3.29 | — | — |
| com-orkut | cc | 8MB | 43.06% | -10.94 | -20.94 | -14.76 | — | — |
| com-orkut | pr | 1MB | 70.47% | -4.22 | -1.98 | -9.38 | ✓ ok | ✓ ok |
| com-orkut | pr | 4MB | 30.85% | -3.68 | -4.32 | -9.61 | — | — |
| com-orkut | pr | 8MB | 16.36% | -2.27 | -3.52 | -5.53 | ✓ ok | — |
| com-orkut | sssp | 1MB | 66.03% | -3.99 | -2.66 | -6.35 | — | — |
| com-orkut | sssp | 4MB | 16.97% | -1.56 | -1.08 | -2.24 | — | — |
| com-orkut | sssp | 8MB | 2.46% | -0.23 | -0.21 | -0.05 | — | — |
| delaunay_n19 | pr | 16kB | 99.99% | -0.01 | -0.43 | -0.10 | — | — |
| delaunay_n19 | pr | 1MB | 95.39% | -1.46 | -13.73 | -17.60 | — | — |
| delaunay_n19 | pr | 256kB | 99.27% | +0.01 | -4.47 | -5.09 | — | — |
| delaunay_n19 | pr | 4kB | 99.99% | +0.00 | -0.12 | -0.04 | — | — |
| delaunay_n19 | pr | 64kB | 99.97% | -0.01 | -1.42 | -1.65 | — | — |
| email-Eu-core | bc | 1MB | 0.01% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bc | 4MB | 0.01% | -0.00 | -0.00 | +0.01 | — | — |
| email-Eu-core | bc | 8MB | 0.01% | -0.00 | +0.00 | -0.00 | — | — |
| email-Eu-core | bfs | 1MB | 4.58% | -0.00 | +17.75 | +0.00 | — | — |
| email-Eu-core | bfs | 4MB | 4.59% | +0.00 | -0.00 | -0.00 | — | — |
| email-Eu-core | bfs | 8MB | 4.59% | -0.00 | -0.00 | -0.00 | — | — |
| email-Eu-core | pr | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | ? insufficient | — |
| roadNet-CA | bc | 16kB | 99.98% | -0.00 | -0.21 | -0.32 | — | — |
| roadNet-CA | bc | 1MB | 54.80% | -2.10 | +36.03 | +20.44 | — | — |
| roadNet-CA | bc | 256kB | 92.21% | -1.29 | +5.32 | -0.96 | — | — |
| roadNet-CA | bc | 4kB | 99.98% | +0.00 | -0.05 | -0.06 | — | — |
| roadNet-CA | bc | 64kB | 99.65% | -0.05 | -0.43 | -1.59 | — | — |
| roadNet-CA | bfs | 16kB | 99.99% | -0.00 | -0.19 | -0.07 | — | — |
| roadNet-CA | bfs | 1MB | 25.54% | +4.64 | +68.40 | +33.39 | — | — |
| roadNet-CA | bfs | 256kB | 89.10% | -2.49 | +8.99 | -11.25 | — | — |
| roadNet-CA | bfs | 4kB | 99.99% | -0.00 | -0.05 | -0.01 | — | — |
| roadNet-CA | bfs | 64kB | 99.74% | -0.06 | -0.45 | -2.42 | — | — |
| roadNet-CA | cc | 16kB | 99.71% | +0.03 | -0.32 | -0.11 | — | — |
| roadNet-CA | cc | 1MB | 74.28% | +0.54 | +9.12 | +5.06 | — | — |
| roadNet-CA | cc | 256kB | 91.60% | -0.53 | +3.13 | +0.35 | — | — |
| roadNet-CA | cc | 4kB | 99.88% | -0.04 | -0.05 | +0.01 | — | — |
| roadNet-CA | cc | 64kB | 97.97% | -0.39 | -0.44 | -0.17 | — | — |
| roadNet-CA | pr | 16kB | 99.99% | +0.00 | -0.10 | -0.07 | — | — |
| roadNet-CA | pr | 1MB | 94.17% | -0.24 | +1.47 | -4.99 | — | — |
| roadNet-CA | pr | 256kB | 98.56% | -0.07 | +0.28 | -5.23 | — | — |
| roadNet-CA | pr | 4kB | 99.99% | -0.00 | -0.02 | -0.02 | — | — |
| roadNet-CA | pr | 64kB | 99.93% | +0.04 | -0.37 | -1.66 | — | — |
| roadNet-CA | sssp | 16kB | 99.79% | -0.04 | -0.20 | -1.12 | — | — |
| roadNet-CA | sssp | 1MB | 17.63% | +1.50 | +70.12 | +10.09 | — | — |
| roadNet-CA | sssp | 256kB | 44.09% | +2.76 | +52.14 | +13.82 | — | — |
| roadNet-CA | sssp | 4kB | 99.99% | +0.00 | -0.13 | -0.15 | — | — |
| roadNet-CA | sssp | 64kB | 93.18% | -1.69 | +5.40 | -4.25 | — | — |
| soc-LiveJournal1 | bc | 1MB | 83.96% | -2.21 | -4.47 | -2.99 | ✓ ok | — |
| soc-LiveJournal1 | bc | 4MB | 60.21% | -2.52 | -5.29 | -3.84 | — | — |
| soc-LiveJournal1 | bc | 8MB | 45.02% | -2.55 | -5.93 | -4.29 | — | — |
| soc-LiveJournal1 | bfs | 1MB | 81.76% | -3.52 | -7.63 | -6.54 | ✓ ok | — |
| soc-LiveJournal1 | bfs | 4MB | 58.88% | -2.46 | -6.36 | -5.12 | — | — |
| soc-LiveJournal1 | bfs | 8MB | 51.06% | -1.50 | -6.33 | -5.12 | — | — |
| soc-LiveJournal1 | cc | 1MB | 78.32% | -1.10 | -3.82 | -2.42 | — | — |
| soc-LiveJournal1 | cc | 4MB | 54.37% | -1.13 | -6.63 | -4.82 | — | — |
| soc-LiveJournal1 | cc | 8MB | 43.14% | -2.15 | -11.21 | -8.13 | — | — |
| soc-LiveJournal1 | pr | 1MB | 73.19% | -3.33 | -4.80 | -10.93 | ✓ ok | ✓ ok |
| soc-LiveJournal1 | pr | 4MB | 42.57% | -3.63 | -7.52 | -13.71 | — | — |
| soc-LiveJournal1 | pr | 8MB | 27.57% | -3.58 | -7.45 | -9.74 | ✓ ok | — |
| soc-LiveJournal1 | sssp | 1MB | 70.58% | -2.60 | -4.74 | -4.76 | — | ✓ ok |
| soc-LiveJournal1 | sssp | 4MB | 36.27% | -2.34 | -6.64 | -5.42 | — | — |
| soc-LiveJournal1 | sssp | 8MB | 16.87% | -2.04 | -4.47 | -3.72 | — | — |
| soc-pokec | bc | 1MB | 85.13% | -2.28 | -8.66 | -5.66 | ✓ ok | — |
| soc-pokec | bc | 4MB | 49.75% | -3.32 | -7.81 | -6.20 | — | — |
| soc-pokec | bc | 8MB | 24.54% | -2.49 | -4.60 | -4.53 | — | — |
| soc-pokec | bfs | 1MB | 87.17% | -0.97 | -2.87 | -1.61 | — | — |
| soc-pokec | bfs | 4MB | 74.01% | -0.14 | -6.37 | -6.65 | — | — |
| soc-pokec | bfs | 8MB | 71.46% | -0.10 | -11.08 | -12.18 | — | — |
| soc-pokec | cc | 1MB | 69.94% | -3.26 | -13.11 | -2.52 | — | — |
| soc-pokec | cc | 4MB | 39.61% | -6.61 | -16.97 | -11.36 | — | — |
| soc-pokec | cc | 8MB | 5.77% | +0.00 | -0.04 | -0.00 | — | — |
| soc-pokec | pr | 1MB | 67.96% | -4.53 | -13.63 | -13.28 | ✓ ok | ✓ ok |
| soc-pokec | pr | 4MB | 19.69% | -3.54 | -6.68 | -7.33 | — | — |
| soc-pokec | pr | 8MB | 10.04% | -1.22 | -2.12 | -1.30 | ✓ ok | — |
| soc-pokec | sssp | 1MB | 63.51% | -3.15 | -10.82 | -7.01 | — | ✓ ok |
| soc-pokec | sssp | 4MB | 9.76% | -1.29 | -3.08 | -2.20 | — | — |
| soc-pokec | sssp | 8MB | 0.29% | -0.00 | -0.00 | -0.00 | — | — |
| web-Google | bc | 1MB | 69.98% | -1.16 | +0.87 | +0.38 | ✓ ok | — |
| web-Google | bc | 4MB | 36.93% | -1.33 | -2.65 | -0.01 | — | — |
| web-Google | bc | 8MB | 15.00% | -0.59 | -1.34 | +1.30 | — | — |
| web-Google | bfs | 1MB | 97.04% | -0.06 | -3.42 | -2.37 | ✓ ok | — |
| web-Google | bfs | 4MB | 93.71% | -0.23 | -17.97 | -18.15 | — | — |
| web-Google | bfs | 8MB | 80.91% | -0.78 | -5.47 | -7.98 | — | — |
| web-Google | cc | 1MB | 56.08% | -3.86 | -6.64 | -5.37 | — | — |
| web-Google | cc | 4MB | 4.57% | -0.00 | -0.01 | -0.00 | — | — |
| web-Google | cc | 8MB | 4.56% | +0.00 | +0.01 | +0.00 | — | — |
| web-Google | pr | 1MB | 60.08% | -5.67 | -14.77 | -18.40 | ✓ ok | ✓ ok |
| web-Google | pr | 4MB | 14.14% | -2.24 | -3.57 | -2.51 | — | — |
| web-Google | pr | 8MB | 9.70% | -0.37 | -1.26 | -1.29 | ✓ ok | — |
| web-Google | sssp | 1MB | 55.58% | -0.50 | +0.03 | -0.82 | — | — |
| web-Google | sssp | 4MB | 1.63% | +0.00 | +0.00 | +0.00 | — | — |
| web-Google | sssp | 8MB | 1.63% | -0.00 | -0.00 | -0.00 | — | — |
