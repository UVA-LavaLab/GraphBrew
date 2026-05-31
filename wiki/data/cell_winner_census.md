# Cell-winner census: corpus decisiveness

Classification of every `(graph, app, l3_size)` cell by winner status.
Tied/no-winner cells must be excluded or qualified in any per-cell
win-rate claim — they are the corpus's 'unwinnable' cases.

- Total cells: **114**
- Cells with **unique winner**: **108** (94.74%)
- Cells with **tied winners**: **6** (5.26%)
- Cells with **no winner**: **0** (0.0%)

Tied cells by tie-count:

- 2-way tie: **2** cells
- 4-way tie: **4** cells

## Per-app census

| App | n cells | Unique | Tied | None | % decisive |
| :-- | ------: | -----: | ---: | ---: | ---------: |
| `bc` | 23 | 20 | 3 | 0 | 87.0% |
| `bfs` | 23 | 23 | 0 | 0 | 100.0% |
| `cc` | 20 | 20 | 0 | 0 | 100.0% |
| `pr` | 28 | 25 | 3 | 0 | 89.3% |
| `sssp` | 20 | 20 | 0 | 0 | 100.0% |

## Tied cells (n=6)

These cells have ≥2 policies tied at top of the cell.

| Graph | App | L3 | Tied policies |
| :---- | :-- | :- | :------------ |
| `email-Eu-core` | `bc` | `1MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `bc` | `4MB` | `GRASP`, `SRRIP` |
| `email-Eu-core` | `bc` | `8MB` | `POPT`, `SRRIP` |
| `email-Eu-core` | `pr` | `1MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `pr` | `4MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
| `email-Eu-core` | `pr` | `8MB` | `GRASP`, `LRU`, `POPT`, `SRRIP` |
