# Paper Table 2 — ECG_DBG vs GRASP corpus parity (sec:results:eviction-equiv)

ECG_DBG uses the DBG-tier field of the per-vertex mask (the prefetch
field is unused); GRASP uses its native per-line tier tag. Both
policies target the same eviction-protection behavior. The corpus
equivalence below confirms that ECG's 2-bit DBG encoding
information-preservingly replicates GRASP's per-line tier tag.

## Summary (16 cells)

- Mean Δ (ECG_DBG - GRASP): **+0.009 pp** (statistical tie)
- Max |Δ|: **0.215 pp**
- TIE (|Δ| < 0.1 pp): **14 / 16** cells
- close (0.1 ≤ |Δ| < 0.5 pp): 2 cells
- DIFFER (|Δ| ≥ 0.5 pp): 0 cells

## Per-cell L3 miss-rates (lower = better)

| Graph | App | GRASP | ECG_DBG | Δ (pp) | Verdict |
|---|---|---:|---:|---:|---|
| cit-Patents | bc | 0.8985 | 0.8986 | +0.009 | TIE |
| cit-Patents | bfs | 0.9590 | 0.9596 | +0.056 | TIE |
| cit-Patents | pr | 0.7857 | 0.7857 | -0.004 | TIE |
| cit-Patents | sssp | 0.8177 | 0.8179 | +0.027 | TIE |
| com-orkut | pr | 0.6848 | 0.6849 | +0.002 | TIE |
| email-Eu-core | pr | 1.0000 | 1.0000 | +0.000 | TIE |
| soc-LiveJournal1 | bc | 0.7949 | 0.7948 | -0.018 | TIE |
| soc-LiveJournal1 | bfs | 0.7402 | 0.7397 | -0.055 | TIE |
| soc-LiveJournal1 | pr | 0.6839 | 0.6839 | -0.001 | TIE |
| soc-LiveJournal1 | sssp | 0.6586 | 0.6573 | -0.125 | close |
| soc-pokec | bc | 0.7648 | 0.7648 | -0.003 | TIE |
| soc-pokec | pr | 0.5433 | 0.5433 | +0.000 | TIE |
| soc-pokec | sssp | 0.5277 | 0.5282 | +0.048 | TIE |
| web-Google | bc | 0.7071 | 0.7092 | +0.215 | close |
| web-Google | bfs | 0.9357 | 0.9354 | -0.021 | TIE |
| web-Google | pr | 0.4531 | 0.4532 | +0.009 | TIE |
