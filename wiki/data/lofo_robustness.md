# Leave-one-family-out (LOFO) winner robustness

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

For each app, drop each of the 5 families (citation, mesh, road, social, web) in turn and re-rank policy winners by cell count. A LOFO-robust claim has the same top policy after every family drop — no single family is driving the headline.

**Headline:** 3/5 apps (60%) are LOFO-robust.  Robust apps: ['bc', 'pr', 'sssp'].  Fragile apps: ['bfs', 'cc'].

## Per-app verdict

| app | full top | full wins | LOFO-robust | fragile family drops |
|---|---|---:|---|---|
| bc | GRASP | 15 | ✅ | — |
| bfs | GRASP | 12 | ❌ | social |
| cc | POPT | 9 | ❌ | web |
| pr | POPT | 16 | ✅ | — |
| sssp | GRASP | 11 | ✅ | — |

## Per-app drop matrix

| app | full top | drop-citation | drop-mesh | drop-road | drop-social | drop-web |
|---|---|---|---|---|---|---|
| bc | GRASP (15) | GRASP (12) | GRASP (15) | GRASP (15) | GRASP (3) | GRASP (15) |
| bfs | GRASP (12) | GRASP (10) | GRASP (12) | GRASP (12) | GRASP (3) ⚠ | GRASP (11) |
| cc | POPT (9) | POPT (8) | POPT (9) | POPT (8) | POPT (5) | GRASP (7) ⚠ |
| pr | POPT (16) | POPT (13) | POPT (15) | POPT (15) | POPT (7) | POPT (14) |
| sssp | GRASP (11) | GRASP (8) | GRASP (11) | GRASP (11) | GRASP (4) | GRASP (10) |

## Interpretation

- LOFO is strictly stronger than LOGO (gate 41). Where LOGO drops one graph, LOFO drops 1–4 graphs (a whole family). Surviving LOFO is therefore a higher robustness bar.
- Fragile apps under LOFO are honestly disclosed: the paper must qualify those headline policies as 'family-sensitive' or report family-stratified winners instead.
