# Build and Reproduction

Generated graphs, binaries, traces, and experiment output remain under
`results/` and are not tracked.

## 1. Prepare graph data

Download the three SNAP edge lists:

```bash
mkdir -p results/graphs/web-Google
curl -L https://snap.stanford.edu/data/web-Google.txt.gz |
  gzip -dc > results/graphs/web-Google/web-Google.el

mkdir -p results/graphs/soc-pokec
curl -L https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz |
  gzip -dc > results/graphs/soc-pokec/soc-pokec.el

mkdir -p results/graphs/cit-Patents
curl -L https://snap.stanford.edu/data/cit-Patents.txt.gz |
  gzip -dc > results/graphs/cit-Patents/cit-Patents.el
```

Create deterministic samples:

```bash
python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/web-Google/web-Google.el \
  --output results/graphs/web-Google-n16/web-Google-n16.el \
  --vertices results/graphs/web-Google-n16/web-Google-n16.vertices.tsv \
  --metadata results/graphs/web-Google-n16/web-Google-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/soc-pokec/soc-pokec.el \
  --output results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  --vertices results/graphs/soc-pokec-n16/soc-pokec-n16.vertices.tsv \
  --metadata results/graphs/soc-pokec-n16/soc-pokec-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/cit-Patents/cit-Patents.el \
  --output results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  --vertices results/graphs/cit-Patents-n18/cit-Patents-n18.vertices.tsv \
  --metadata results/graphs/cit-Patents-n18/cit-Patents-n18.sample.json \
  --target-vertices 262144
```

Convert the samples:

```bash
bench/bin/converter \
  -f results/graphs/web-Google-n16/web-Google-n16.el \
  -b results/graphs/web-Google-n16/web-Google-n16.sg

bench/bin/converter \
  -f results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  -b results/graphs/soc-pokec-n16/soc-pokec-n16.sg

bench/bin/converter \
  -f results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  -s -b results/graphs/cit-Patents-n18/cit-Patents-n18-sym.sg
```

## 2. Build

```bash
python3 -m pip install -r scripts/requirements.txt

make setup-gem5
make setup-gem5-guest-tools
make setup-sniper
make all-sim
make gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs \
  gem5-riscv-m5ops-sssp gem5-riscv-m5ops-bc gem5-riscv-m5ops-cc
make sniper-sg_kernel
```

## 3. Test

```bash
python3 -m pytest -q scripts/test
```

## 4. Inspect the PageRank study

```bash
python3 scripts/experiments/ecg/flows/experiment_run.py \
  --profile k2_pagerank_study \
  --run-dir results/ecg_experiments/runs/pagerank_dryrun \
  --list --dry-run --no-build --allow-missing-graphs
```

The profile expands to 12 whole cells: three graphs and four iteration counts.
Policy sharding is disabled so each comparison retains its matching baseline.

## 5. Run

```bash
python3 -I scripts/experiments/ecg/flows/experiment_run.py \
  --profile k2_pagerank_study \
  --run-dir results/ecg_experiments/runs/pagerank_final \
  --no-build --no-resume
```

For a provenance-locked rerun on the reference host, invoke
`/usr/bin/python3.12 -I` and add `--require-pinned-python`.

Summarize a complete run:

```bash
python3 scripts/experiments/ecg/analysis/pagerank_gate.py \
  --input results/ecg_experiments/runs/pagerank_final/combined_roi_matrix.csv \
  --config scripts/experiments/ecg/configs/pagerank_study.json \
  --output results/ecg_experiments/runs/pagerank_final/decision.json
```

## 6. Cross-simulator consistency

```bash
python3 -m pytest -q \
  scripts/test/test_grasp_sideband_registration.py \
  scripts/test/test_popt_permutation_equivalence.py

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc --schedule-k 2
```

## 7. Aggregate local output

```bash
python3 scripts/experiments/ecg/flows/aggregate_results.py \
  --skip-run \
  --input-run-dirs \
    results/ecg_experiments/runs/pagerank_final \
  --run-root results/ecg_experiments/aggregates/pagerank_final
```
