# L3-sweep monotonicity universality (cache-sim)

**Verdict:** PASS  
**Source:** `wiki/data/oracle_gap.json`  
**L3 axis:** 4kB, 16kB, 64kB, 256kB, 1MB, 4MB, 8MB  
**Cells (>=2 L3 points):** 136  
**Total steps:** 320  
**Bumps:** 10 (3.12%) ceiling 10%  
**Largest bump:** 0.0096 pp (noise tolerance 0.50 pp)  
**Hard violations (>= 0.5 pp):** 0

## Verdict checks

| check | result |
|---|---|
| no_hard_violations | ✅ |
| bump_pct_under_ceiling | ✅ |
| largest_bump_within_noise | ✅ |

## Largest bump (worst-case cell)

`email-Eu-core` / `bc` / `POPT`: 1MB -> 4MB delta = +0.0096 pp

## All bumps (sorted by magnitude)

| graph | app | policy | from | to | delta pp |
|---|---|---|---|---|---:|
| email-Eu-core | bc | POPT | 1MB | 4MB | +0.0096 |
| delaunay_n19 | pr | LRU | 4kB | 16kB | +0.0039 |
| email-Eu-core | bfs | SRRIP | 1MB | 4MB | +0.0021 |
| email-Eu-core | bfs | LRU | 4MB | 8MB | +0.0020 |
| web-Google | cc | GRASP | 4MB | 8MB | +0.0016 |
| web-Google | sssp | LRU | 4MB | 8MB | +0.0014 |
| email-Eu-core | bfs | SRRIP | 4MB | 8MB | +0.0007 |
| roadNet-CA | pr | SRRIP | 4kB | 16kB | +0.0006 |
| email-Eu-core | bfs | LRU | 1MB | 4MB | +0.0004 |
| email-Eu-core | bc | GRASP | 4MB | 8MB | +0.0001 |

## Interpretation

Cache monotonicity (more cache cannot hurt) is a foundational soundness check that downstream slope/distance/sensitivity gates rely on. Small noise-level bumps (<0.5 pp) are expected from sampling and warmup; any larger bump would indicate a simulator bug, corrupted sweep, or pathological policy.
