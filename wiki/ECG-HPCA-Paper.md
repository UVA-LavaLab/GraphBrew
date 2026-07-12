# ECG Successor HPCA Paper

This is the only current wiki page for the ECG successor architecture. The
public paper name is intentionally still open; the implementation remains under
the `ECG_*` namespace.

## Important prior-publication boundary

The preliminary paper, *ECG: Expressing Locality and Prefetching for Optimal
Caching in Graph Structures*, appeared in the archival IEEE IPDPSW 2024
proceedings (pp. 520–525, DOI `10.1109/IPDPSW59749.2024.00094`).

HPCA does not permit a substantially similar archival-workshop submission. The
new paper therefore requires:

1. materially distinct contributions;
2. citation and disclosure of the workshop paper;
3. a contribution-delta statement;
4. written guidance from the HPCA PC chairs before registration.

A new name improves differentiation but does not replace this requirement.

## New architecture beyond the workshop paper

- **K2 records:** `dest32 | epoch1_16 | epoch2_16`.
- **Adaptive replacement:** epoch-first PR, degree-first BFS/SSSP, RRIP-first BC/CC.
- **StreamShield placement:** private-cache fills and LLC hits remain normal;
  bypassed LLC misses do not allocate.
- **Request-bound ISA:** `ecg.load2` and `ecg.stream.load2`.
- **Three-simulator artifact:** cache_sim, gem5, and Sniper exact delivery and
  victim-decision gates.
- **Full accounting:** demand misses, total traffic, reserved P-OPT capacity,
  simulated time, and instruction count.

## Current evidence

- Corrected real-graph cache_sim attribution: **K2 77.3%**, **StreamShield 22.7%**.
- gem5 synthetic mechanism cell: StreamShield improves fused K2 by **13.03%**.
- Sniper synthetic mechanism cell: StreamShield improves fused K2 by **0.65%**
  at identical instruction count.
- Overall detailed-simulator superiority over P-OPT is **not claimed** until the
  real-graph Sniper matrix completes.

## Required paper comparison

Every reported table includes:

```text
LRU  SRRIP  GRASP  charged P-OPT  K2  K2+StreamShield
```

The canonical policy labels are:

```text
ECG:K2
ECG:K2_STREAMSHIELD
```

## Canonical artifact

- Paper SSOT: [`research/ecg-hpca/`](https://github.com/UVA-LavaLab/GraphBrew/tree/graphbrew_ecg/research/ecg-hpca)
- Manifest: [`final_paper_manifest.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/graphbrew_ecg/scripts/experiments/ecg/final_paper_manifest.json)
- Runner: [`flows/paper_run.py`](https://github.com/UVA-LavaLab/GraphBrew/blob/graphbrew_ecg/scripts/experiments/ecg/flows/paper_run.py)
- Matrix engine: [`roi_matrix.py`](https://github.com/UVA-LavaLab/GraphBrew/blob/graphbrew_ecg/scripts/experiments/ecg/roi_matrix.py)
- Slurm shards: [`slurm/make_slurm_shards.py`](https://github.com/UVA-LavaLab/GraphBrew/blob/graphbrew_ecg/scripts/experiments/ecg/slurm/make_slurm_shards.py)

## Canonical run

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_successor_webgoogle \
  --no-build
```

Generated `results/` data is not committed.
