# Leave-one-family-out (LOFO) winner robustness

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

For each app, drop each of the 5 families (citation, mesh, road, social, web) in turn and re-rank policy winners by cell count. A LOFO-robust claim has the same top policy after every family drop — no single family is driving the headline.

**Headline:** 3/5 apps (60%) are LOFO-robust.  Robust apps: ['bc', 'cc', 'pr'].  Fragile apps: ['bfs', 'sssp'].

## Per-app verdict

| app | full top | full wins | LOFO-robust | fragile family drops |
|---|---|---:|---|---|
| bc | GRASP | 15 | ✅ | — |
| bfs | GRASP | 10 | ❌ | citation |
| cc | GRASP | 14 | ✅ | — |
| pr | POPT | 17 | ✅ | — |
| sssp | GRASP | 7 | ❌ | citation |

## Per-app drop matrix

| app | full top | drop-citation | drop-mesh | drop-road | drop-social | drop-web |
|---|---|---|---|---|---|---|
| bc | GRASP (15) | GRASP (13) | GRASP (15) | GRASP (15) | GRASP (4) | GRASP (13) |
| bfs | GRASP (10) | GRASP (7) ⚠ | GRASP (10) | GRASP (10) | GRASP (4) | GRASP (9) |
| cc | GRASP (14) | GRASP (11) | GRASP (14) | GRASP (14) | GRASP (5) | GRASP (12) |
| pr | POPT (17) | POPT (14) | POPT (16) | POPT (16) | POPT (7) | POPT (15) |
| sssp | GRASP (7) | POPT (5) ⚠ | GRASP (7) | GRASP (7) | GRASP (3) | GRASP (7) |

## Interpretation

- LOFO is strictly stronger than LOGO (gate 41). Where LOGO drops one graph, LOFO drops 1–4 graphs (a whole family). Surviving LOFO is therefore a higher robustness bar.
- Fragile apps under LOFO are honestly disclosed: the paper must qualify those headline policies as 'family-sensitive' or report family-stratified winners instead.
