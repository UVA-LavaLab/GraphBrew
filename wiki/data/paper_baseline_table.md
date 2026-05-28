# Paper baseline table

All Δ values are percentage-points of L3 miss-rate vs LRU at the same (graph, app, L3) tuple (negative = better).

Verdict suffix encodes the literature-claim outcome where applicable: `✓` ok, `~` within tolerance, `✗` DISAGREE, `?` insufficient data.


| Graph | App | L3 | LRU miss | SRRIP Δ | GRASP Δ | POPT Δ | GRASP claim | POPT claim |
|---|---|---|---:|---:|---:|---:|---|---|
| email-Eu-core | bc | 1MB | 99.06% | +0.94 | +0.94 | +0.62 | — | — |
| email-Eu-core | bc | 4MB | 99.68% | +0.32 | +0.00 | +0.32 | — | — |
| email-Eu-core | bc | 8MB | 100.00% | -1.25 | -0.94 | +0.00 | — | — |
| email-Eu-core | bfs | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | bfs | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 1MB | 100.00% | +0.00 | -0.05 | +0.00 | — | — |
| email-Eu-core | pr | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| email-Eu-core | pr | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| soc-pokec | cc | 1MB | 69.94% | -3.26 | -13.11 | -2.52 | — | — |
| soc-pokec | cc | 4MB | 39.61% | -6.61 | -16.97 | -11.36 | — | — |
| soc-pokec | cc | 8MB | 5.77% | +0.00 | -0.04 | -0.00 | — | — |
| soc-pokec | sssp | 1MB | 100.00% | +0.00 | +0.00 | +0.00 | — | ? insufficient |
| soc-pokec | sssp | 4MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
| soc-pokec | sssp | 8MB | 100.00% | +0.00 | +0.00 | +0.00 | — | — |
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
| web-Google | pr | 8MB | 9.70% | -0.37 | -1.25 | -1.29 | — | — |
| web-Google | sssp | 1MB | 55.58% | -0.50 | +0.03 | -0.82 | — | — |
| web-Google | sssp | 4MB | 1.63% | +0.00 | +0.00 | +0.00 | — | — |
| web-Google | sssp | 8MB | 1.63% | -0.00 | -0.00 | -0.00 | — | — |
