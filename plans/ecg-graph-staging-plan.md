# ECG Final Graph Staging Plan

## Purpose

Stage the large file-backed graphs required by ECG final-paper Slurm shards on
shared cluster storage. This plan is intentionally read-only on the local
workstation: generate/check commands locally, but run downloads and conversions
only on the cluster or another machine with enough disk, memory, and wall time.

## Required Graphs

The final profiles expect these serialized graph files:

```text
results/graphs/soc-pokec/soc-pokec.sg
results/graphs/soc-LiveJournal1/soc-LiveJournal1.sg
results/graphs/com-orkut/com-orkut.sg
results/graphs/cit-Patents/cit-Patents.sg
```

Current local status on 2026-05-26: `cit-Patents.sg` is staged, while
`soc-pokec.sg`, `soc-LiveJournal1.sg`, and `com-orkut.sg` are missing locally.
Cluster staging must verify the actual shared-storage state rather than assuming
the local workstation state matches the cluster.

## Cluster Setup

Run this from the shared GraphBrew checkout on a login or data-transfer node,
not from a Slurm worker that lacks outbound internet.

```bash
export GRAPHBREW_ROOT=$SCRATCH/GraphBrew
cd "$GRAPHBREW_ROOT"
source .venv/bin/activate

mkdir -p results/graphs results/ecg_experiments/slurm results/slurm_logs
make converter
```

Generate the authoritative staging checklist:

```bash
python3 scripts/experiments/ecg/ecg_graph_staging_status.py \
  --profile final_replacement final_droplet final_cache_sim final_cache_sim_ecg_pfx \
  --out results/ecg_experiments/slurm/final_graph_staging_status.csv

column -s, -t < results/ecg_experiments/slurm/final_graph_staging_status.csv
```

Important CSV fields:

```text
graph              manifest graph name and expected .sg directory
expected_path      final .sg file required by the manifest
status             ok or missing
source_url         SuiteSparse/SNAP archive URL when cataloged
catalog_graph      graph name accepted by the downloader
download_command   command to fetch and extract the Matrix Market archive
convert_command    command to convert the extracted .mtx to expected_path
```

`com-orkut` is the manifest name, but the download catalog name is
`com-Orkut`. Use the generated `catalog_graph`, `download_command`, and
`convert_command` fields rather than hand-normalizing names.

## Execute Missing Rows

Review the generated commands first:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

status = Path("results/ecg_experiments/slurm/final_graph_staging_status.csv")
for row in csv.DictReader(status.open()):
    if row["status"] != "missing":
        continue
    print(f"# {row['graph']} catalog={row['catalog_graph']} size={row['source_size_mb']}MB")
    print(f"# {row['source_url']}")
    print(row["download_command"])
    print(row["convert_command"])
    print()
PY
```

Then execute one missing graph at a time. The conversion command searches the
download directory for the extracted `.mtx`, so it tolerates SuiteSparse archive
filename casing such as `soc-Pokec.mtx`.

For unattended cluster staging after review:

```bash
python3 - <<'PY'
import csv
import subprocess
from pathlib import Path

status = Path("results/ecg_experiments/slurm/final_graph_staging_status.csv")
for row in csv.DictReader(status.open()):
    if row["status"] != "missing":
        continue
    print(f"[download] {row['graph']} from {row['source_url']}", flush=True)
    subprocess.run(row["download_command"], shell=True, check=True)
    print(f"[convert] {row['graph']} -> {row['expected_path']}", flush=True)
    subprocess.run(row["convert_command"], shell=True, check=True)
PY
```

Do not commit downloaded `.mtx`, `.tar.gz`, or `.sg` files. They belong under
ignored `results/graphs/` storage.

## Verify Before Submission

Regenerate the checklist after staging and require all rows to be present:

```bash
python3 scripts/experiments/ecg/ecg_graph_staging_status.py \
  --profile final_replacement final_droplet final_cache_sim final_cache_sim_ecg_pfx \
  --out results/ecg_experiments/slurm/final_graph_staging_status.csv \
  --fail-on-missing
```

Run strict preflight on the cluster before submitting any final array:

```bash
python3 scripts/experiments/ecg/ecg_cluster_preflight.py \
  --profile final_replacement final_droplet final_cache_sim final_cache_sim_ecg_pfx \
  --require-slurm \
  --shards results/ecg_experiments/slurm/final_repl_droplet_20260526_141547_shards.tsv \
           results/ecg_experiments/slurm/cache_sim_pfx_20260526_142555_shards.tsv \
  --scale-shards results/ecg_experiments/slurm/ecg_pfx_scale_20260526_133517_scale.tsv
```

Only after strict graph staging and preflight pass should the cluster smoke shard
or full final Slurm array be submitted.

## Definition Of Done

- `ecg_graph_staging_status.py --fail-on-missing` exits 0 on the cluster.
- `ecg_cluster_preflight.py --require-slurm` exits 0 without
  `--allow-missing-graphs`.
- Each required `expected_path` exists, is nonempty, and lives under shared
  cluster storage visible to all Slurm workers.
- The shard TSVs remain unchanged unless the final manifest changes.