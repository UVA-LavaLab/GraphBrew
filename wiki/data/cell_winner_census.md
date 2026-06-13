# Cell-winner census: corpus decisiveness

Classification of every `(graph, app, l3_size)` cell by winner status.
Tied/no-winner cells must be excluded or qualified in any per-cell
win-rate claim — they are the corpus's 'unwinnable' cases.

- Total cells: **114**
- Cells with **unique winner**: **99** (86.84%)
- Cells with **tied winners**: **15** (13.16%)
- Cells with **no winner**: **0** (0.0%)

Tied cells by tie-count:

- 3-way tie: **6** cells
- 4-way tie: **9** cells

## Per-app census

| App | n cells | Unique | Tied | None | % decisive |
| :-- | ------: | -----: | ---: | ---: | ---------: |
| `bc` | 23 | 20 | 3 | 0 | 87.0% |
| `bfs` | 23 | 20 | 3 | 0 | 87.0% |
| `cc` | 20 | 17 | 3 | 0 | 85.0% |
| `pr` | 28 | 25 | 3 | 0 | 89.3% |
| `sssp` | 20 | 17 | 3 | 0 | 85.0% |

## Tied cells (n=15)

These cells have ≥2 policies tied at top of the cell.

| Graph | App | L3 | Tied policies |
| :---- | :-- | :- | :------------ |
| `email-Eu-core` | `bc` | `1MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `bc` | `4MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `bc` | `8MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `bfs` | `1MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `bfs` | `4MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `bfs` | `8MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `pr` | `1MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `pr` | `4MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `pr` | `8MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `soc-pokec` | `cc` | `8MB` | `LRU`, `POPT`, `SRRIP` |
| `soc-pokec` | `sssp` | `8MB` | `LRU`, `POPT`, `SRRIP` |
| `web-Google` | `cc` | `4MB` | `LRU`, `POPT`, `SRRIP` |
| `web-Google` | `cc` | `8MB` | `LRU`, `POPT`, `SRRIP` |
| `web-Google` | `sssp` | `4MB` | `LRU`, `POPT`, `SRRIP` |
| `web-Google` | `sssp` | `8MB` | `LRU`, `POPT`, `SRRIP` |
