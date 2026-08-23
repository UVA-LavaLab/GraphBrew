# Getting Started

Build GraphBrew and run your first benchmark.

## Requirements

| Component | Minimum | Notes |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) or macOS | tested on Ubuntu 22.04 LTS |
| Compiler | GCC 7+ | C++17 required |
| RAM | 8 GB | 16 GB+ for large graphs, 64 GB+ for the paper's full suite |
| Disk | 50 GB | benchmark graphs |
| Python | 3.8+ (optional) | analysis scripts use stdlib only |

Optional: `numpy`, `matplotlib`, `pandas` for plotting and analysis.

## Install dependencies

```bash
# Ubuntu / Debian
sudo apt-get install -y build-essential g++ libboost-all-dev \
    libnuma-dev google-perftools python3

# Fedora / RHEL
sudo dnf install -y gcc-c++ make boost-devel numactl-devel gperftools python3

# macOS
xcode-select --install
brew install gcc boost google-perftools
```

Verify with `python3 scripts/graphbrew_experiment.py --check-deps`.

## Build

```bash
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
make all        # production binaries → bench/bin/
make all-sim    # cache-sim binaries → bench/bin_sim/
```

Build flags:

| Flag | Effect |
|---|---|
| `RABBIT_ENABLE=0` | drop the Boost-backed Rabbit variant. The CSR variant stays enabled. |
| `DEBUG=1` | debug symbols |
| `SANITIZE=1` | address sanitizer |

Boost 1.58 is **only** needed for `-o 8:boost`; the default `-o 8` (CSR Rabbit) needs no Boost. If you do need it: `python3 scripts/graphbrew_experiment.py --install-boost` downloads and installs it to `/opt/boost_1_58_0`.

## Verify

```bash
ls bench/bin/   # expect: bc bfs cc cc_sv converter pr pr_spmv sssp tc tc_p
ls bench/bin_sim/   # expect: bc bfs cc cc_sv pr pr_spmv sssp tc

# Smoke test on the included tiny graph
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -n 3
```

You should see `Read Time`, `Build Time`, three `Trial Time` lines, and `Average Time`.

## First benchmark

PageRank on the tiny graph with three reordering choices:

```bash
GRAPH=scripts/test/graphs/tiny/tiny.el

# No reordering (baseline)
./bench/bin/pr -f $GRAPH -s -o 0 -n 3

# Cheap degree/bucket control
./bench/bin/pr -f $GRAPH -s -o 7 -n 3

# Explicit Rabbit-free GraphBrew composition
./bench/bin/pr -f $GRAPH -s \
  -o 12:leiden:compose:sg_none:comm_size_desc:intra_hubsort:cd_serial:refine_none \
  -n 3
```

`-o N` selects the reordering algorithm (see [Reordering-Algorithms](Reordering-Algorithms)). `-s` symmetrises a directed input. `-n N` runs N trials.

The HubSort recipe above is a simple starter composition. The
[running example](GraphBrew-Running-Example) uses the Gorder/BFS dispatch to
explain both local-layout branches, while the validated low-reuse arm uses
`gordf5000`.

## Run a real graph

```bash
# Download a small social network from SNAP
wget https://snap.stanford.edu/data/ego-Facebook.txt.gz
gunzip ego-Facebook.txt.gz
mv ego-Facebook.txt facebook.el

./bench/bin/pr -f facebook.el -s \
  -o 12:leiden:compose:sg_none:comm_size_desc:intra_hubsort:cd_serial:refine_none \
  -n 5
```

GraphBrew recipes explicitly select the partitioner, block layout, and local
vertex layout. Rabbit-dependent presets remain available as baselines, but
they are not part of the Rabbit-free research direction. Start with the
[GraphBrew Running Example](GraphBrew-Running-Example), then use
[GraphBrewOrder](GraphBrewOrder) as the token reference.

## Common flags

| Flag | Purpose |
|---|---|
| `-f <file>` | input graph |
| `-s` | symmetrise directed input |
| `-o <algo>` | reordering algorithm (or `-o 12:variant`) |
| `-n <N>` | number of trials |
| `-r <vertex>` | root for BFS / SSSP |
| `-i <iters>` | iterations per trial (PR) |

Full list: [Command-Line-Reference](Command-Line-Reference).

## Automated pipeline

For batch experiments (download → build → benchmark → analyse):

```bash
# Quick test (~10 min): small graphs only
python3 scripts/graphbrew_experiment.py --target-graphs 30 --size small

# Standard (~1 h): medium graphs
python3 scripts/graphbrew_experiment.py --target-graphs 80 --size medium

# Full (~4-8 h): all sizes, 150 graphs per bucket
python3 scripts/graphbrew_experiment.py --target-graphs 150

# Preview only — print commands without running
python3 scripts/graphbrew_experiment.py --target-graphs 150 --dry-run
```

`--target-graphs N` is shorthand for `--full --catalog-size N --auto --all-variants`. See [Benchmark-Suite](Benchmark-Suite) for size buckets.

## Reproducing the frozen study

```bash
# Preview: 2 small graphs, 1 trial — sanity check (~5 min)
python3 scripts/graphbrew_experiment.py --vldb --paper-preview \
  --paper-graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 4 --paper-cpu-list 24-27

# Full frozen matrix on the dedicated 16-core evaluation host
python3 scripts/graphbrew_experiment.py --vldb \
  --paper-graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 16 --paper-cpu-list 0-15
```

Use the independent stage runners only for restartable long runs. See
[Reproducible-Experiments](Reproducible-Experiments) for the frozen matrix and
stage guide.

## Common build issues

| Error | Fix |
|---|---|
| `fatal error: boost/range/algorithm.hpp` | `sudo apt-get install libboost-all-dev` |
| `-fopenmp not supported` | `sudo apt-get install libomp-dev` |
| `g++` too old (< 7) | install `g++-11`, `export CXX=g++-11`, rebuild |
| OOM during build | drop parallelism: `make -j2 all` |

More in [Troubleshooting](Troubleshooting).

## Next steps

- [Reordering-Algorithms](Reordering-Algorithms) — what each algorithm does and when to use it
- [GraphBrew Running Example](GraphBrew-Running-Example) — one graph carried through all six stages
- [GraphBrewOrder](GraphBrewOrder) — the explicit composition grammar and cost controls
- [Running-Benchmarks](Running-Benchmarks) — manual benchmark workflow
- [Reproducible-Experiments](Reproducible-Experiments) — reproducing the frozen study
