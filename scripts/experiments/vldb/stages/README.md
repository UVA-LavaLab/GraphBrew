# `stages/` — Independent VLDB experiment stages

Five stages, each runnable standalone. Pick only what you need.

| Stage | Script | What it does | Needs network | Needs fast CPU | Needs prior stage |
|---|---|---|---|---|---|
| 01 | `01_prep.py`     | Download + `.mtx`→`.sg` convert     | **Yes** | No   | — |
| 02 | `02_reorder.py`  | Pre-generate reorder mappings (`.lo`) | No  | Yes  | 01 |
| 03 | `03_cpu_perf.py` | Wall-clock kernel sweep on real CPU  | No  | **Yes** | 01 + 02 |
| 04 | `04_cache_sim.py`| Cache-simulator sweep (sim stats only) | No | No | 01 + 02 |
| 05 | `05_aggregate.py`| LaTeX tables + PNG figures from JSON | No  | No   | 03 and/or 04 |

> **Tip:** stages 03 and 04 are independent. Skip 04 if you don't care about
> cache stats. Run 04 on a slow/shared/other machine — the host's CPU speed
> doesn't affect simulation results.

## Quick smoke test (everything tiny, ~1 min total)

```bash
source .venv/bin/activate
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --preview   # optional
python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0             # optional
```

## Full local 6-graph run

```bash
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --local
```

## SLURM (UVA Rivanna / generic Slurm cluster)

The `slurm/` subfolder has one `.sbatch` per stage. Pass `EXP` and optionally
`GRAPHS` via `--export`:

```bash
# 1. Prep on a login node (needs internet)
python3 scripts/experiments/vldb/stages/01_prep.py --exp 2 --local

# 2. Pre-generate mappings
sbatch --export=ALL,EXP=2 scripts/experiments/vldb/stages/slurm/02_reorder.sbatch

# 3. CPU sweep — one job per graph (fan-out)
for g in cit-Patents com-Orkut hollywood-2009 soc-pokec soc-LiveJournal1; do
  sbatch --job-name="gbrew-exp2-$g" \
         --export=ALL,EXP=2,GRAPHS="$g" \
         scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
done

# 4. (optional) cache sim on a separate / slower node
sbatch --export=ALL,EXP=1 scripts/experiments/vldb/stages/slurm/04_cache_sim.sbatch

# 5. Aggregate when 03/04 complete
sbatch scripts/experiments/vldb/stages/slurm/05_aggregate.sbatch
```

### Large graphs (twitter7, webbase, kron, uk-2002, indochina, wikipedia)

Override partition + memory at submit time:

```bash
sbatch --partition=largemem --mem=512G --time=24:00:00 \
       --export=ALL,EXP=2,GRAPHS=twitter7 \
       scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```

## Common flags (all stages)

| Flag | Meaning |
|---|---|
| `--exp N`              | Experiment id 1..8 (required) |
| `--preview`            | Tiny 2-graph smoke run |
| `--local`              | 6-graph 64 GB-fit eval set |
| `--64gb`               | 11-graph auto-download eval set |
| `--graphs A B C`       | Override graph list by name |
| `--graph-dir PATH`     | Where graphs live (default `results/graphs/`) |
| `--dry-run`            | Print commands, don't execute |

## Outputs

| Stage | Output |
|---|---|
| 01 | `results/graphs/<name>/<name>.{sg,mtx,el}` |
| 02 | `results/vldb_mappings/<graph>/<algo_key>.{lo,time}` |
| 03 | `results/vldb_paper/exp<N>_*/...json` |
| 04 | `results/vldb_paper/exp1_cache/cache_results.json` |
| 05 | `paper/figures/`, `paper/dataCharts/` |

## Relationship to the legacy runner

`scripts/experiments/vldb/runner.py` still exists and runs every
stage end-to-end as before. The stage scripts here are thin wrappers around
the same internal functions, exposed as independent entry points so each
stage can be scheduled, resumed, or skipped independently.
