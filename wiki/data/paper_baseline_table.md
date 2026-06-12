# Paper baseline table

All Δ values are percentage-points of L3 miss-rate vs LRU at the same (graph, app, L3) tuple (negative = better).

Verdict suffix encodes the literature-claim outcome where applicable: `✓` ok, `~` within tolerance, `✗` DISAGREE, `?` insufficient data.

| Graph | App | L3 | LRU miss | SRRIP Δ | GRASP Δ | POPT Δ | GRASP claim | POPT claim |
|---|---|---|---:|---:|---:|---:|---|---|
| cit-Patents | bc | 1MB | 94.51% | -0.25 | -1.56 | -1.62 | ✓ ok | — |
| cit-Patents | bc | 4MB | 81.89% | -1.07 | -3.62 | -2.39 | — | — |
| cit-Patents | bc | 8MB | 68.64% | -2.04 | -4.54 | -2.23 | — | — |
| cit-Patents | bfs | 1MB | 98.15% | -0.24 | -0.76 | -0.91 | ✓ ok | — |
| cit-Patents | bfs | 4MB | 94.67% | -0.50 | -4.41 | -3.63 | — | — |
| cit-Patents | bfs | 8MB | 92.14% | -0.27 | -5.02 | -8.11 | — | — |
| cit-Patents | cc | 1MB | 66.68% | -4.06 | -10.78 | -11.89 | — | — |
| cit-Patents | cc | 4MB | 33.69% | -2.33 | -7.12 | -6.60 | — | — |
| cit-Patents | cc | 8MB | 19.80% | -2.23 | -7.18 | -5.33 | — | — |
| cit-Patents | pr | 1MB | 89.51% | -1.48 | -7.69 | -16.50 | ✓ ok | ✓ ok |
| cit-Patents | pr | 4MB | 59.01% | -4.34 | -13.68 | -24.26 | — | — |
| cit-Patents | pr | 8MB | 32.62% | -5.17 | -10.14 | -14.74 | ~ within_tol | — |
| cit-Patents | sssp | 1MB | 87.68% | -0.68 | -4.32 | -3.71 | ✓ ok | ✓ ok |
| cit-Patents | sssp | 4MB | 56.81% | -2.20 | -7.03 | -6.08 | — | — |
| cit-Patents | sssp | 8MB | 27.09% | -2.16 | -4.27 | -4.40 | — | — |
| com-orkut | bc | 1MB | 83.69% | -1.64 | -4.05 | -3.21 | — | — |
| com-orkut | bc | 4MB | 57.35% | -2.65 | -4.85 | -1.17 | — | — |
| com-orkut | bc | 8MB | 37.60% | -2.62 | -3.54 | +2.25 | — | — |
| com-orkut | bfs | 1MB | 99.89% | -0.01 | -0.08 | -0.08 | — | — |
| com-orkut | bfs | 4MB | 99.65% | -0.00 | -1.53 | -0.74 | — | — |
| com-orkut | bfs | 8MB | 99.58% | -0.00 | -1.51 | -2.50 | — | — |
| com-orkut | cc | 1MB | 74.15% | -3.35 | -10.17 | -4.22 | — | — |
| com-orkut | cc | 4MB | 56.76% | -1.44 | -15.70 | -3.98 | — | — |
| com-orkut | cc | 8MB | 50.63% | -8.43 | -25.69 | -16.71 | — | — |
| com-orkut | pr | 1MB | 74.26% | -3.60 | -10.43 | -10.72 | ✓ ok | ✓ ok |
| com-orkut | pr | 4MB | 34.56% | -4.21 | -4.59 | -12.11 | — | — |
| com-orkut | pr | 8MB | 18.56% | -2.61 | -2.57 | -6.58 | ✓ ok | — |
| com-orkut | sssp | 1MB | 68.67% | -3.29 | -9.39 | -7.41 | — | — |
| com-orkut | sssp | 4MB | 17.30% | -2.02 | -2.68 | -3.00 | — | — |
| com-orkut | sssp | 8MB | 1.62% | -0.10 | +0.07 | +0.10 | — | — |
| delaunay_n19 | pr | 16kB | 100.00% | +0.00 | -0.03 | -0.39 | — | — |
| delaunay_n19 | pr | 1MB | 96.68% | -1.49 | -11.52 | -17.64 | — | — |
| delaunay_n19 | pr | 256kB | 99.92% | -0.01 | -1.62 | -7.12 | — | — |
| delaunay_n19 | pr | 4kB | 100.00% | +0.00 | -0.00 | +0.00 | — | — |
| delaunay_n19 | pr | 64kB | 100.00% | +0.00 | -0.27 | -2.54 | — | — |
| email-Eu-core | bc | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bc | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bc | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | ? insufficient | — |
| roadNet-CA | bc | 16kB | 100.00% | +0.00 | -0.01 | -0.16 | — | — |
| roadNet-CA | bc | 1MB | 64.85% | +0.65 | +16.23 | +16.40 | — | — |
| roadNet-CA | bc | 256kB | 98.92% | -0.18 | -7.39 | -4.06 | — | — |
| roadNet-CA | bc | 4kB | 100.00% | +0.00 | -0.00 | -0.01 | — | — |
| roadNet-CA | bc | 64kB | 99.98% | -0.00 | -0.55 | -0.57 | — | — |
| roadNet-CA | bfs | 16kB | 100.00% | +0.00 | -0.42 | -0.01 | — | — |
| roadNet-CA | bfs | 1MB | 24.11% | +4.38 | +62.59 | +34.29 | — | — |
| roadNet-CA | bfs | 256kB | 93.99% | -0.80 | -6.32 | -12.40 | — | — |
| roadNet-CA | bfs | 4kB | 100.00% | +0.00 | -0.02 | +0.00 | — | — |
| roadNet-CA | bfs | 64kB | 99.96% | -0.01 | -5.78 | -0.57 | — | — |
| roadNet-CA | cc | 16kB | 100.00% | +0.00 | -0.34 | -0.02 | — | — |
| roadNet-CA | cc | 1MB | 99.17% | -0.08 | -12.63 | -6.05 | — | — |
| roadNet-CA | cc | 256kB | 99.98% | -0.00 | -3.72 | -1.62 | — | — |
| roadNet-CA | cc | 4kB | 100.00% | +0.00 | -0.08 | -0.00 | — | — |
| roadNet-CA | cc | 64kB | 100.00% | +0.00 | -1.25 | -0.15 | — | — |
| roadNet-CA | pr | 16kB | 100.00% | +0.00 | -0.05 | -0.01 | — | — |
| roadNet-CA | pr | 1MB | 95.25% | +0.05 | +2.55 | -4.47 | — | — |
| roadNet-CA | pr | 256kB | 99.53% | -0.02 | -0.63 | -5.20 | — | — |
| roadNet-CA | pr | 4kB | 100.00% | +0.00 | -0.04 | +0.00 | — | — |
| roadNet-CA | pr | 64kB | 100.00% | +0.00 | -0.08 | -1.17 | — | — |
| roadNet-CA | sssp | 16kB | 99.97% | -0.01 | -1.34 | -0.32 | — | — |
| roadNet-CA | sssp | 1MB | 16.65% | +1.14 | +55.83 | +8.80 | — | — |
| roadNet-CA | sssp | 256kB | 56.57% | -1.52 | +23.48 | +4.64 | — | — |
| roadNet-CA | sssp | 4kB | 100.00% | +0.00 | -0.06 | -0.01 | — | — |
| roadNet-CA | sssp | 64kB | 97.31% | -0.61 | -10.31 | -4.07 | — | — |
| soc-LiveJournal1 | bc | 1MB | 86.49% | -1.47 | -3.37 | -3.97 | ✓ ok | — |
| soc-LiveJournal1 | bc | 4MB | 65.52% | -2.55 | -5.51 | -5.24 | — | — |
| soc-LiveJournal1 | bc | 8MB | 50.25% | -3.06 | -6.19 | -5.58 | — | — |
| soc-LiveJournal1 | bfs | 1MB | 83.57% | -2.50 | -5.20 | -4.97 | ✓ ok | — |
| soc-LiveJournal1 | bfs | 4MB | 64.46% | -2.61 | -7.26 | -5.76 | — | — |
| soc-LiveJournal1 | bfs | 8MB | 56.32% | -1.72 | -5.33 | -5.85 | — | — |
| soc-LiveJournal1 | cc | 1MB | 79.86% | -2.42 | -7.10 | -11.67 | — | — |
| soc-LiveJournal1 | cc | 4MB | 57.55% | -2.20 | -10.48 | -8.49 | — | — |
| soc-LiveJournal1 | cc | 8MB | 46.86% | -3.33 | -15.13 | -9.32 | — | — |
| soc-LiveJournal1 | pr | 1MB | 76.54% | -3.14 | -8.28 | -13.42 | ✓ ok | ✓ ok |
| soc-LiveJournal1 | pr | 4MB | 46.16% | -4.15 | -9.28 | -16.12 | — | — |
| soc-LiveJournal1 | pr | 8MB | 29.99% | -4.05 | -7.46 | -11.06 | ✓ ok | — |
| soc-LiveJournal1 | sssp | 1MB | 74.39% | -2.62 | -6.19 | -7.32 | — | ✓ ok |
| soc-LiveJournal1 | sssp | 4MB | 38.36% | -2.98 | -6.52 | -6.44 | — | — |
| soc-LiveJournal1 | sssp | 8MB | 16.96% | -1.78 | -2.98 | -3.42 | — | — |
| soc-pokec | bc | 1MB | 85.97% | -1.56 | -4.89 | -5.00 | ✓ ok | — |
| soc-pokec | bc | 4MB | 53.01% | -3.36 | -7.21 | -7.05 | — | — |
| soc-pokec | bc | 8MB | 26.77% | -2.65 | -4.50 | -5.01 | — | — |
| soc-pokec | bfs | 1MB | 90.63% | -1.41 | -4.69 | -3.33 | — | — |
| soc-pokec | bfs | 4MB | 78.73% | -0.61 | -3.14 | -7.92 | — | — |
| soc-pokec | bfs | 8MB | 75.96% | -0.60 | -1.58 | -13.40 | — | — |
| soc-pokec | cc | 1MB | 72.81% | -2.69 | -12.39 | -8.67 | — | — |
| soc-pokec | cc | 4MB | 46.38% | -8.46 | -21.33 | -15.18 | — | — |
| soc-pokec | cc | 8MB | 6.29% | +0.00 | +0.00 | +0.00 | — | — |
| soc-pokec | pr | 1MB | 69.62% | -4.44 | -14.05 | -16.96 | ✓ ok | ✓ ok |
| soc-pokec | pr | 4MB | 20.94% | -3.91 | -6.57 | -8.45 | — | — |
| soc-pokec | pr | 8MB | 10.73% | -1.29 | -1.60 | -1.42 | ✓ ok | — |
| soc-pokec | sssp | 1MB | 67.19% | -3.53 | -11.39 | -8.90 | — | ✓ ok |
| soc-pokec | sssp | 4MB | 9.97% | -1.27 | -2.32 | -2.20 | — | — |
| soc-pokec | sssp | 8MB | 0.33% | +0.00 | +0.43 | +0.00 | — | — |
| web-Google | bc | 1MB | 80.27% | +0.42 | +2.11 | +1.89 | ✓ ok | — |
| web-Google | bc | 4MB | 51.62% | -1.73 | +0.24 | +2.73 | — | — |
| web-Google | bc | 8MB | 22.10% | -0.96 | +4.25 | +6.82 | — | — |
| web-Google | bfs | 1MB | 98.04% | -0.08 | -3.60 | -2.48 | ✓ ok | — |
| web-Google | bfs | 4MB | 94.86% | -0.12 | -2.95 | -18.42 | — | — |
| web-Google | bfs | 8MB | 81.94% | -0.57 | +9.32 | -8.16 | — | — |
| web-Google | cc | 1MB | 58.59% | -2.91 | -4.61 | -8.93 | — | — |
| web-Google | cc | 4MB | 5.19% | +0.00 | +0.00 | +0.00 | — | — |
| web-Google | cc | 8MB | 5.19% | +0.00 | +0.00 | +0.00 | — | — |
| web-Google | pr | 1MB | 59.84% | -5.75 | -14.75 | -21.43 | ✓ ok | ~ within_tol |
| web-Google | pr | 4MB | 14.21% | -2.29 | -1.30 | -2.72 | — | — |
| web-Google | pr | 8MB | 9.72% | -0.38 | +0.97 | -1.28 | ✓ ok | — |
| web-Google | sssp | 1MB | 66.38% | -0.76 | -2.35 | -2.18 | — | — |
| web-Google | sssp | 4MB | 2.35% | +0.00 | +5.81 | +0.00 | — | — |
| web-Google | sssp | 8MB | 2.35% | +0.00 | +2.41 | +0.00 | — | — |
