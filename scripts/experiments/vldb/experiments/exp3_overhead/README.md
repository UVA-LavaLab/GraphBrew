# exp3 — Reorder Overhead

**What it measures:** Wall-clock cost of the reorder pass itself
(amortizable cost analysis).

**Output:** `results/vldb_paper/exp3_overhead/overhead_results.json`

Live-measures all mappings on the seven smaller graphs in the final timing
state. It reuses structured Stage-02 timing only for wikipedia, Gong-gplus,
webbase, and twitter, whose single mapping sweep accounts for most of the
multi-day cost. Promoted Gorder mappings on wikipedia, Gong-gplus, and webbase
still run live and are byte-compared with the promoted mappings; all completed
within the final 12-hour budget. Small-graph live rows retain
their Stage-02 reference and calibration ratio. Every uncensored cell also
receives one final-state weighted MAP-application measurement for SSSP
amortization. Existing canonical `.lo` files are never regenerated.

Run the isolated cit-Patents check, isolated webbase check, and bulk command
with the exact same measurement generation, artifact root, thread count,
CPU list, timeout, and `--skip-build`. Do not rebuild
`bench/bin/converter` between phases.

## Run

```bash
python3 scripts/experiments/vldb/stages/03_cpu_perf.py \
  --exp 3 \
  --measurement-generation vldb-final-20260808 \
  --skip-build \
  --graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --threads 16 --cpu-list 0-15
```

## SLURM

```bash
sbatch --export=ALL,EXP=3 scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
