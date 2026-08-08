# PageRank Screen: Current STOP Result

**Status:** current measured result; complete-design comparison; no GO claim.

Run directory:
`results/ecg_experiments/final_paper_runs/proposal_sota_v2_2db0a04b_20260802`

The run contains 12 complete cells and 84 `ok` rows:

- graphs: `web-Google-n16`, `soc-pokec-n16`,
  `cit-Patents-n18-sym`;
- PageRank iterations: 1, 2, 4, 8;
- policies: LRU, GRASP, optimistic charged and uncharged P-OPT,
  K2-LRU+StreamShield, static RRIP K2+StreamShield, and online
  K2+StreamShield.

The preregistered primary was static RRIP K2+StreamShield. The decision analyzer
reported:

| Baseline/control | Time ratio | Off-chip ratio |
|---|---:|---:|
| LRU | 0.9061 | 0.7227 |
| GRASP | 0.9835 | 0.9455 |
| optimistic charged P-OPT bound | 1.0235 | 0.9351 |
| uncharged P-OPT | 1.0503 | 1.1776 |
| K2-LRU+StreamShield | 0.9330 | 0.7855 |

The P-OPT row charges reserved capacity and cumulative matrix bytes, but
`popt_target_time_charged=0` omits matrix-stream latency. Its time is an
optimistic bound; realistic target-time P-OPT performance remains unresolved.

The screen is valid and the result is **STOP**. Static K2 misses the frozen
aggregate time threshold versus GRASP and the optimistic charged P-OPT bound.
Because matrix-stream target-time latency is omitted, this STOP does not
establish that a realistic P-OPT engine is faster. The decisive negative graph
is `cit-Patents-n18-sym`; at iteration 8:

| Baseline | Time ratio | Off-chip ratio |
|---|---:|---:|
| GRASP | 1.0613 | 1.1141 |
| optimistic charged P-OPT bound | 1.1373 | 1.0322 |

V1 used `epoch_first` as the static primary and was stopped after the first
cell. It remains a failed historical experiment and is not an active paper
configuration.

The current screen used DBG order because the P-OPT paper's direct GRASP
comparison is PageRank on DBG-ordered graphs. This screen does not establish
the separate claim that K2 avoids DBG preprocessing.
