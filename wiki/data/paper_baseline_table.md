# Paper baseline table

All Δ values are percentage-points of L3 miss-rate vs LRU at the same (graph, app, L3) tuple (negative = better).

Verdict suffix encodes the literature-claim outcome where applicable: `✓` ok, `~` within tolerance, `✗` DISAGREE, `?` insufficient data.

| Graph | App | L3 | LRU miss | SRRIP Δ | GRASP Δ | POPT Δ | GRASP claim | POPT claim |
|---|---|---|---:|---:|---:|---:|---|---|
| cit-Patents | bc | 1MB | 94.51% | -0.25 | -1.56 | -0.82 | ✓ ok | — |
| cit-Patents | bc | 4MB | 81.89% | -1.07 | -3.62 | -0.40 | — | — |
| cit-Patents | bc | 8MB | 68.64% | -2.04 | -4.54 | +0.76 | — | — |
| cit-Patents | bfs | 1MB | 98.15% | -0.24 | -0.76 | -0.55 | ✓ ok | — |
| cit-Patents | bfs | 4MB | 94.67% | -0.50 | -4.41 | -2.64 | — | — |
| cit-Patents | bfs | 8MB | 92.14% | -0.27 | -5.02 | -6.28 | — | — |
| cit-Patents | cc | 1MB | 66.68% | -4.06 | -7.06 | -8.86 | — | — |
| cit-Patents | cc | 4MB | 33.69% | -2.33 | -6.73 | -3.95 | — | — |
| cit-Patents | cc | 8MB | 19.80% | -2.23 | -3.89 | -2.73 | — | — |
| cit-Patents | pr | 1MB | 89.51% | -1.48 | -7.69 | -13.40 | ✓ ok | ✓ ok |
| cit-Patents | pr | 4MB | 59.01% | -4.34 | -13.68 | -19.94 | — | — |
| cit-Patents | pr | 8MB | 32.62% | -5.17 | -10.14 | -12.04 | ~ within_tol | — |
| cit-Patents | sssp | 1MB | 87.68% | -0.68 | -4.32 | -1.75 | ✓ ok | ✓ ok |
| cit-Patents | sssp | 4MB | 56.81% | -2.20 | -7.03 | -1.41 | — | — |
| cit-Patents | sssp | 8MB | 27.09% | -2.16 | -4.27 | +1.52 | — | — |
| com-orkut | bc | 1MB | 83.69% | -1.64 | -4.05 | -1.37 | — | — |
| com-orkut | bc | 4MB | 57.35% | -2.65 | -4.85 | +1.66 | — | — |
| com-orkut | bc | 8MB | 37.60% | -2.62 | -3.54 | +5.40 | — | — |
| com-orkut | bfs | 1MB | 99.89% | -0.01 | -0.08 | -0.05 | — | — |
| com-orkut | bfs | 4MB | 99.65% | -0.00 | -1.53 | -0.54 | — | — |
| com-orkut | bfs | 8MB | 99.58% | -0.00 | -1.51 | -1.95 | — | — |
| com-orkut | cc | 1MB | 74.15% | -3.35 | -7.06 | -2.55 | — | — |
| com-orkut | cc | 4MB | 56.76% | -1.44 | -14.71 | -1.82 | — | — |
| com-orkut | cc | 8MB | 50.63% | -8.43 | -9.49 | -11.05 | — | — |
| com-orkut | pr | 1MB | 74.26% | -3.60 | -10.43 | -6.88 | ✓ ok | ✓ ok |
| com-orkut | pr | 4MB | 34.56% | -4.21 | -4.59 | -8.43 | — | — |
| com-orkut | pr | 8MB | 18.56% | -2.61 | -2.57 | -5.29 | ✓ ok | — |
| com-orkut | sssp | 1MB | 68.67% | -3.29 | -9.39 | -3.31 | — | — |
| com-orkut | sssp | 4MB | 17.30% | -2.02 | -2.68 | +1.42 | — | — |
| com-orkut | sssp | 8MB | 1.62% | -0.10 | +0.07 | +1.32 | — | — |
| delaunay_n19 | pr | 16kB | 100.00% | +0.00 | -0.03 | -0.23 | — | — |
| delaunay_n19 | pr | 1MB | 96.68% | -1.49 | -11.52 | -15.64 | — | — |
| delaunay_n19 | pr | 256kB | 99.92% | -0.01 | -1.62 | -6.33 | — | — |
| delaunay_n19 | pr | 4kB | 100.00% | +0.00 | -0.00 | +0.00 | — | — |
| delaunay_n19 | pr | 64kB | 100.00% | +0.00 | -0.27 | -2.22 | — | — |
| email-Eu-core | bc | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bc | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bc | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | ? insufficient | — |
| roadNet-CA | bc | 16kB | 100.00% | +0.00 | -0.01 | -0.13 | — | — |
| roadNet-CA | bc | 1MB | 64.85% | +0.65 | +16.23 | +17.67 | — | — |
| roadNet-CA | bc | 256kB | 98.92% | -0.18 | -7.39 | -3.10 | — | — |
| roadNet-CA | bc | 4kB | 100.00% | +0.00 | -0.00 | -0.01 | — | — |
| roadNet-CA | bc | 64kB | 99.98% | -0.00 | -0.55 | -0.49 | — | — |
| roadNet-CA | bfs | 16kB | 100.00% | +0.00 | -0.42 | -0.01 | — | — |
| roadNet-CA | bfs | 1MB | 24.11% | +4.38 | +62.59 | +35.13 | — | — |
| roadNet-CA | bfs | 256kB | 93.99% | -0.80 | -6.32 | -8.52 | — | — |
| roadNet-CA | bfs | 4kB | 100.00% | +0.00 | -0.02 | +0.00 | — | — |
| roadNet-CA | bfs | 64kB | 99.96% | -0.01 | -5.78 | -0.37 | — | — |
| roadNet-CA | cc | 16kB | 100.00% | +0.00 | -0.05 | -0.01 | — | — |
| roadNet-CA | cc | 1MB | 99.17% | -0.08 | -4.66 | -4.98 | — | — |
| roadNet-CA | cc | 256kB | 99.98% | -0.00 | -0.53 | -1.38 | — | — |
| roadNet-CA | cc | 4kB | 100.00% | +0.00 | -0.00 | -0.00 | — | — |
| roadNet-CA | cc | 64kB | 100.00% | +0.00 | -0.20 | -0.12 | — | — |
| roadNet-CA | pr | 16kB | 100.00% | +0.00 | -0.05 | -0.00 | — | — |
| roadNet-CA | pr | 1MB | 95.25% | +0.05 | +2.55 | -3.89 | — | — |
| roadNet-CA | pr | 256kB | 99.53% | -0.02 | -0.63 | -4.84 | — | — |
| roadNet-CA | pr | 4kB | 100.00% | +0.00 | -0.04 | +0.00 | — | — |
| roadNet-CA | pr | 64kB | 100.00% | +0.00 | -0.08 | -0.83 | — | — |
| roadNet-CA | sssp | 16kB | 99.97% | -0.01 | -1.34 | -0.22 | — | — |
| roadNet-CA | sssp | 1MB | 16.65% | +1.14 | +55.83 | +10.47 | — | — |
| roadNet-CA | sssp | 256kB | 56.57% | -1.52 | +23.48 | +9.30 | — | — |
| roadNet-CA | sssp | 4kB | 100.00% | +0.00 | -0.06 | -0.00 | — | — |
| roadNet-CA | sssp | 64kB | 97.31% | -0.61 | -10.31 | -2.35 | — | — |
| soc-LiveJournal1 | bc | 1MB | 86.49% | -1.47 | -3.37 | -2.23 | ✓ ok | — |
| soc-LiveJournal1 | bc | 4MB | 65.52% | -2.55 | -5.51 | -2.54 | — | — |
| soc-LiveJournal1 | bc | 8MB | 50.25% | -3.06 | -6.19 | -2.26 | — | — |
| soc-LiveJournal1 | bfs | 1MB | 83.57% | -2.50 | -5.20 | -3.01 | ✓ ok | — |
| soc-LiveJournal1 | bfs | 4MB | 64.46% | -2.61 | -7.26 | -3.90 | — | — |
| soc-LiveJournal1 | bfs | 8MB | 56.32% | -1.72 | -5.33 | -4.33 | — | — |
| soc-LiveJournal1 | cc | 1MB | 79.86% | -2.42 | -4.40 | -9.53 | — | — |
| soc-LiveJournal1 | cc | 4MB | 57.55% | -2.20 | -8.92 | -6.48 | — | — |
| soc-LiveJournal1 | cc | 8MB | 46.86% | -3.33 | -10.93 | -6.66 | — | — |
| soc-LiveJournal1 | pr | 1MB | 76.54% | -3.14 | -8.28 | -10.12 | ✓ ok | ✓ ok |
| soc-LiveJournal1 | pr | 4MB | 46.16% | -4.15 | -9.28 | -12.90 | — | — |
| soc-LiveJournal1 | pr | 8MB | 29.99% | -4.05 | -7.46 | -9.51 | ✓ ok | — |
| soc-LiveJournal1 | sssp | 1MB | 74.39% | -2.62 | -6.19 | -4.29 | — | ✓ ok |
| soc-LiveJournal1 | sssp | 4MB | 38.36% | -2.98 | -6.52 | -2.55 | — | — |
| soc-LiveJournal1 | sssp | 8MB | 16.96% | -1.78 | -2.98 | -0.05 | — | — |
| soc-pokec | bc | 1MB | 85.97% | -1.56 | -4.89 | -2.73 | ✓ ok | — |
| soc-pokec | bc | 4MB | 53.01% | -3.36 | -7.21 | -2.53 | — | — |
| soc-pokec | bc | 8MB | 26.77% | -2.65 | -4.50 | -0.29 | — | — |
| soc-pokec | bfs | 1MB | 90.63% | -1.41 | -4.69 | -1.93 | — | — |
| soc-pokec | bfs | 4MB | 78.73% | -0.61 | -3.14 | -6.13 | — | — |
| soc-pokec | bfs | 8MB | 75.96% | -0.60 | -1.58 | -13.37 | — | — |
| soc-pokec | cc | 1MB | 72.81% | -2.69 | -7.72 | -6.47 | — | — |
| soc-pokec | cc | 4MB | 46.38% | -8.46 | -5.38 | -9.69 | — | — |
| soc-pokec | cc | 8MB | 6.29% | +0.00 | +31.02 | +0.00 | — | — |
| soc-pokec | pr | 1MB | 69.62% | -4.44 | -14.05 | -12.01 | ✓ ok | ✓ ok |
| soc-pokec | pr | 4MB | 20.94% | -3.91 | -6.57 | -6.69 | — | — |
| soc-pokec | pr | 8MB | 10.73% | -1.29 | -1.60 | -1.02 | ✓ ok | — |
| soc-pokec | sssp | 1MB | 67.19% | -3.53 | -11.39 | -4.25 | — | ✓ ok |
| soc-pokec | sssp | 4MB | 9.97% | -1.27 | -2.32 | +1.72 | — | — |
| soc-pokec | sssp | 8MB | 0.33% | +0.00 | +0.43 | +0.00 | — | — |
| web-Google | bc | 1MB | 80.27% | +0.42 | +2.11 | +3.51 | ✓ ok | — |
| web-Google | bc | 4MB | 51.62% | -1.73 | +0.24 | +6.68 | — | — |
| web-Google | bc | 8MB | 22.10% | -0.96 | +4.25 | +12.14 | — | — |
| web-Google | bfs | 1MB | 98.04% | -0.08 | -3.60 | -1.85 | ✓ ok | — |
| web-Google | bfs | 4MB | 94.86% | -0.12 | -2.95 | -14.96 | — | — |
| web-Google | bfs | 8MB | 81.94% | -0.57 | +9.32 | -7.44 | — | — |
| web-Google | cc | 1MB | 58.59% | -2.91 | -4.50 | -5.45 | — | — |
| web-Google | cc | 4MB | 5.19% | +0.00 | +20.63 | +0.00 | — | — |
| web-Google | cc | 8MB | 5.19% | +0.00 | +11.52 | +0.00 | — | — |
| web-Google | pr | 1MB | 59.84% | -5.75 | -14.75 | -16.85 | ✓ ok | ✓ ok |
| web-Google | pr | 4MB | 14.21% | -2.29 | -1.30 | -1.63 | — | — |
| web-Google | pr | 8MB | 9.72% | -0.38 | +0.97 | -0.95 | ✓ ok | — |
| web-Google | sssp | 1MB | 66.38% | -0.76 | -2.35 | +2.28 | — | — |
| web-Google | sssp | 4MB | 2.35% | +0.00 | +5.81 | +0.00 | — | — |
| web-Google | sssp | 8MB | 2.35% | +0.00 | +2.41 | +0.00 | — | — |
