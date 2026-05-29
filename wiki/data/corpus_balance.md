# Corpus tier / family balance audit

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

Corpus: **8 graphs** across **5 families** × **5 apps**; 360 paper-L3 cells in total.

## Dominance disclosures

- Dominant family by graph count: **social** with 4 graphs (50.0% of the corpus)
- Dominant family by paper-L3 cells: **social** with 216 cells (60.0% of paper-L3 cells)

## Diversity metrics (higher = more balanced)

| metric | value | max |
|---|---:|---:|
| Shannon H (graphs/family, bits) | 2.000 | 2.322 |
| Pielou evenness (graphs/family) | 0.861 | 1.000 |
| Simpson's D (graphs/family) | 0.688 | 0.800 |
| Pielou evenness (cells/app) | 0.997 | 1.000 |
| Simpson's D (cells/app) | 0.798 | 0.800 |

## Per-family composition

| family | n_graphs | n_paper_l3_cells | reaches_4MB | reaches_8MB | graphs |
|---|---:|---:|:---:|:---:|---|
| citation | 1 | 60 | ✅ | ✅ | cit-Patents |
| mesh | 1 | 4 | ❌ | ❌ | delaunay_n19 |
| road | 1 | 20 | ❌ | ❌ | roadNet-CA |
| social | 4 | 216 | ✅ | ✅ | com-orkut, email-Eu-core, soc-LiveJournal1, soc-pokec |
| web | 1 | 60 | ✅ | ✅ | web-Google |

## Per-(family, L3) cell counts

| family | 1MB | 4MB | 8MB |
|---|---:|---:|---:|
| citation | 20 | 20 | 20 |
| mesh | 4 | 0 | 0 |
| road | 20 | 0 | 0 |
| social | 72 | 72 | 72 |
| web | 20 | 20 | 20 |

## Per-(app, L3) cell counts

| app | 1MB | 4MB | 8MB |
|---|---:|---:|---:|
| bc | 28 | 24 | 24 |
| bfs | 28 | 24 | 24 |
| cc | 24 | 20 | 20 |
| pr | 32 | 24 | 24 |
| sssp | 24 | 20 | 20 |

## Honest disclosures

- Families lacking 4MB/8MB cells have graphs whose WSS-relative L3 classification lands them in 'over' regime before 4MB
- Reviewer comparisons that include 4MB+ L3 will exclude these families.
- Families capped below 4MB: ['mesh', 'road']
- Families capped below 8MB: ['mesh', 'road']
- Families reaching 8MB: ['citation', 'social', 'web']

