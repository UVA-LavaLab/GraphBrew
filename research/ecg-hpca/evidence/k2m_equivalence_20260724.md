# Fresh K2-M Equivalence Evidence — 2026-07-24

## Three-simulator mechanism gate

- Commit: `86e407658c3b52b41e55f1389f88447b6abe9571`
- Command:
  `python3 scripts/experiments/ecg/verify/equiv_kernels.py --gem5 --sniper --kernels pr bfs sssp bc cc --schedule-k 2 --gem5-isa-variant mask`
- Graph: `email-Eu-core.sg`
- Result: **15/15 kernel x simulator cells passed**.
- Matrix: PR/BFS/SSSP/BC/CC are `ok` in cache_sim, gem5, and Sniper.
- All eviction traces obeyed the shared victim specification.
- All K2 distance mismatch counts were zero.
- gem5 executed the computed-address K2-M load in all five kernels.
- Sniper validated transport matching, exact governed-load binding, and
  epoch/context association in all five kernels.
- BC covered both governed arrays; compact SSSP provenance passed.

Frozen artifacts:

| Artifact | SHA-256 |
|---|---|
| `command.txt` | `93e4aea72ab11c133128dda0d4a09e6604f580763aae5a260501e9cf87ae7cc7` |
| `git_head.txt` | `d67f62275f02781ccc68e064b4acff18878282e357108196cf476e6f775fe33e` |
| `equivalence.log` | `d19df46de5b7a3417de1e8114c8fdbb59714a95a09bb2c582c4c2b576b03cb28` |
| `summary.json` | `0dc72d8e6192ae33d103643560a1a731d12df74588d925b2e17484af169f55ba` |

Run directory:
`results/ecg_experiments/final_paper_runs/ecg_3sim_k2m_equivalence_20260724`.

## Post-binding equal-semantic-work Sniper gate

- Commit: `30333aeaaf1ed1339238f589e66d335fd97aeabb`
- Profile: `ecg_sniper_semantic_gate`
- Result: **25/25 rows passed** across PR/BFS/SSSP/BC/CC and
  LRU/SRRIP/GRASP/charged P-OPT/K2-M.
- Every row executed exactly 4,096 static graph-edge visits and reported
  `truncated=1`.
- Every policy group had identical semantic output.
- LRU and K2-M had exactly 1.000x instruction ratio in all five kernels.
- All LRU/K2-M rows reported exact binding, epoch/context binding, matched
  transport, no instruction cap, and matrix-level semantic certification.

K2-M versus LRU on this synthetic semantic prefix:

| Kernel | Diagnostic speedup | L3 miss ratio | Instruction ratio |
|---|---:|---:|---:|
| PR | 1.041x | 0.767x | 1.000x |
| BFS | 0.987x | 1.160x | 1.000x |
| SSSP | 0.998x | 1.070x | 1.000x |
| BC | 0.929x | 0.862x | 1.000x |
| CC | 0.926x | 0.984x | 1.000x |
| Geomean | **0.975x** | **0.958x** | **1.000x** |

These timing values are diagnostic only: the workload is a truncated synthetic
prefix and every row has `timing_valid_for_speedup=0`. The result proves matched
work and mechanism behavior, not general or full-graph K2-M speedup.

Frozen artifacts:

| Artifact | SHA-256 |
|---|---|
| `combined_roi_matrix.csv` | `008a40ca7a2e10885bd0582c1906007415961c23c99fd3543f256645d6a19192` |
| `semantic_gate_certification.json` | `f9508e3f6660999cef0b9fbe5681f0578c4f5f587675a969dc00a3bc47cde31d` |
| `semantic_gate_summary.json` | `f74f705d3778fa7e621c5f0c779a7e1f8725d830221d9588a709bc2dea865af9` |

Run directory:
`results/ecg_experiments/final_paper_runs/ecg_sniper_semantic_gate_20260724`.
