# `scripts/` — single source of truth

```
scripts/
├── experiments/               ← paper experiment runners (see experiments/README.md)
│   ├── vldb/                   VLDB 2026 paper — everything in one place
│   ├── ecg/                    ECG / GrAPL paper
│   ├── adaptive_ml/            ML-ordering model ablation
│   └── legacy/                 archived (no live imports)
├── lib/                       ← reusable Python modules (imported, not run)
│   ├── core/                   ResultsStore, parsing, run helpers
│   ├── pipeline/               download.py (catalog auto-download), build, convert
│   ├── analysis/               amortise, figures, cold-start sim
│   ├── ml/                     adaptive ordering model
│   └── tools/                  misc CLIs
├── test/                      ← pytest tests
├── graphbrew_experiment.py    ← legacy unified one-click pipeline
└── requirements.txt
```

## Canonical paths (single source of truth)

| Artifact | Path |
|---|---|
| Graphs (downloaded + converted)        | `results/graphs/<name>/<name>.{sg,mtx,el}` |
| Reorder mappings cache (`.lo` + `.time`) | `results/vldb_mappings/<graph>/<algo_key>.lo` |
| VLDB experiment JSON                  | `results/vldb_paper/exp<N>_*/` |
| Aggregated figures + tables           | `paper/figures/`, `paper/dataCharts/`, `results/vldb_paper/{figures,tables}/` |
| ECG experiments                       | `results/ecg_experiments/` |
| Generic logs                          | `results/logs/`, `results/slurm_logs/` |

Auto-download for the VLDB pipeline is driven by
[`experiments/vldb/config.py:VLDB_GRAPH_SOURCES`](experiments/vldb/config.py).

## Quick start (VLDB stage-based, recommended)

```bash
source .venv/bin/activate

# Smoke (~1 min, 2 tiny graphs)
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview

# Local 6-graph eval
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --local

# Cache stats only (host CPU speed doesn't matter)
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --local

# Figures
python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0
```

## Simulator setup helpers

```bash
# gem5 detailed backend
python3 scripts/setup_gem5.py --isa X86 --skip-build
python3 scripts/experiments/ecg/roi_matrix.py --suite gem5 --policies LRU --benchmark pr --l3-sizes 4kB --no-build

# Sniper scalable multicore backend scaffold
python3 scripts/setup_sniper.py --dry-run
python3 scripts/setup_sniper.py --skip-build
python3 scripts/setup_sniper.py --skip-build --apply-overlays
python3 scripts/experiments/ecg/roi_matrix.py --suite sniper --policies LRU SRRIP
python3 scripts/experiments/ecg/final_paper_run.py --profile sniper_kernel_smoke --run-dir /tmp/graphbrew-final-sniper-kernel-smoke --no-build --force
python3 scripts/experiments/ecg/final_paper_run.py --profile sniper_droplet_smoke --run-dir /tmp/graphbrew-final-sniper-droplet-smoke --no-build --force
python3 scripts/experiments/ecg/final_paper_run.py --profile sniper_sift_ecg_pfx_smoke --run-dir /tmp/graphbrew-sniper-ecg-pfx-profile --no-build --force
python3 scripts/experiments/ecg/final_paper_run.py --profile sniper_sift_file_ecg_pfx_smoke --run-dir /tmp/graphbrew-sniper-file-ecg-pfx-profile --no-build --force
python3 scripts/experiments/ecg/paper_pipeline.py --skip-run --input-run-dirs /tmp/graphbrew-final-sniper-kernel-smoke /tmp/graphbrew-final-sniper-thread-smoke /tmp/graphbrew-final-sniper-droplet-smoke --run-root /tmp/graphbrew-paper-pipeline-sniper-check-final
```

For ECG_PFX paper figures, aggregate the validated Sniper profiles with:

```bash
.venv/bin/python3 scripts/experiments/ecg/paper_pipeline.py \
	--skip-run \
	--input-run-dirs /tmp/graphbrew-sniper-ecg-pfx-profile /tmp/graphbrew-sniper-file-ecg-pfx-profile \
	--run-root /tmp/graphbrew-paper-pipeline-sniper-ecg-pfx
```

To measure ECG/ECG_PFX preprocessing overhead without running a cache simulator,
build and run the standalone utility. It reports graph-load time separately from
degree scan, optional P-OPT matrix construction, ECG mask/PFX construction, and
total preprocessing time.

```bash
make RABBIT_ENABLE=0 bench/bin_sim/ecg_preprocess
ECG_PREFETCH_MODE=2 \
ECG_PREPROCESS_REPEATS=5 \
ECG_PREPROCESS_OUTPUT_JSON=/tmp/graphbrew-ecg-preprocess.json \
OMP_NUM_THREADS=32 \
bench/bin_sim/ecg_preprocess -f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 0 -n 1
```

SLURM templates: `scripts/experiments/vldb/stages/slurm/*.sbatch`.

## Upstream GRASP/PIN parity validation

The GRASP, PIN, and BELADY policies in `bench/include/cache_sim/` are
validated against the upstream `faldupriyank/grasp` trace simulators via:

```bash
# Build both sides, replay web-Google traces, write comparison.csv
python3 scripts/experiments/ecg/upstream_policy_compare.py \
  --policies lru pin grasp belady \
  --traces BC.web-Google.cvgr.dbg.lru.llc.trace \
           BellmanFordOpt.web-Google.cintgr.dbg.lru.llc.trace \
           PageRankOpt.web-Google.cvgr.dbg.lru.llc.trace \
           PageRankDeltaOpt.web-Google.cvgr.dbg.lru.llc.trace \
           Radii.web-Google.cvgr.dbg.lru.llc.trace \
  --out-dir /tmp/graphbrew-upstream-policy-compare
```

LRU/PIN/GRASP route through `cache_sim::CacheLevel`; BELADY uses an
offline oracle inside [graphbrew_trace_replay.cc](experiments/ecg/graphbrew_trace_replay.cc)
because the live cache_sim API has no future-trace concept. Current
status: 20/20 zero-delta. Source-faithfulness pytest:
`pytest scripts/test/test_popt_grasp_faithfulness_sources.py`.

## Legacy / all-in-one entry points

- `scripts/experiments/vldb/runner.py --all --local` — monolithic VLDB runner
- `scripts/experiments/ecg/runner.py --all` — monolithic ECG runner
- `scripts/graphbrew_experiment.py --phase all` — original one-click pipeline
- `scripts/experiments/vldb/slurm/monolithic.sbatch` — monolithic SLURM template

## Tests

```bash
pytest scripts/test/
```
