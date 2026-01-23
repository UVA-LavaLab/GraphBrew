# Installation Guide

This guide covers installing GraphBrew on Linux systems.

## 🚀 Quick Install (One-Click)

The fastest way to get started - builds automatically and downloads test graphs:

```bash
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
python3 scripts/graphbrew_experiment.py --full --download-size SMALL
```

This single command will:
- Build all binaries automatically
- Download benchmark graphs from SuiteSparse
- Run the complete experiment pipeline

---

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **Memory**: 8GB RAM (16GB+ recommended for large graphs)
- **Disk**: 50GB+ for benchmark graphs

### Required Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    g++-9 \
    libboost-all-dev \
    libnuma-dev \
    google-perftools \
    python3 \
    python3-pip

# Fedora/RHEL
sudo dnf install -y \
    gcc-c++ \
    boost-devel \
    numactl-devel \
    gperftools \
    python3 \
    python3-pip
```

### Optional Dependencies

```bash
# For Rabbit Order support (requires Boost 1.58+)
# Usually included in libboost-all-dev

# For visualization and analysis
pip3 install numpy matplotlib pandas
```

## Manual Build

### Clone the Repository

```bash
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew
```

### Basic Build

```bash
# Build all benchmarks
make all

# Build with cache simulation support
make sim

# Or build specific benchmarks
make pr    # PageRank
make bfs   # Breadth-First Search
make cc    # Connected Components
```

### Build Options

| Option | Description | Example |
|--------|-------------|---------|
| `RABBIT_ENABLE=1` | Enable Rabbit Order algorithm | `RABBIT_ENABLE=1 make all` |
| `DEBUG=1` | Build with debug symbols | `DEBUG=1 make all` |
| `SANITIZE=1` | Enable address sanitizer | `SANITIZE=1 make all` |

### Build with Rabbit Order

Rabbit Order requires Boost 1.58+:

```bash
# Check Boost version
cat /usr/include/boost/version.hpp | grep "BOOST_LIB_VERSION"

# Build with Rabbit Order
RABBIT_ENABLE=1 make all
```

### Verify Installation

```bash
# Check if binaries were created
ls -la bench/bin/

# Run a quick test
./bench/bin/pr -g 12 -o 0 -n 1
```

Expected output:
```
Generate Time:       0.02xxx
Build Time:          0.01xxx
...
Trial Time:          0.00xxx
```

## Python Environment Setup

### Install Python Dependencies

```bash
cd GraphBrew
pip3 install -r scripts/requirements.txt

# Or manually:
pip3 install numpy matplotlib pandas
```

### Verify Python Scripts

```bash
# Test the download script
python3 scripts/download/download_graphs.py --list --size SMALL

# Test correlation analysis
python3 scripts/analysis/correlation_analysis.py --quick
```

## Directory Structure After Build

```
GraphBrew/
├── bench/
│   ├── bin/           # Compiled binaries (pr, bfs, cc, etc.)
│   ├── bin_sim/       # Cache simulation binaries
│   ├── include/       # Header files
│   └── src/           # Source files
├── graphs/            # Downloaded benchmark graphs (or use results/graphs/)
├── results/           # Experiment outputs
│   ├── graphs/        # Downloaded graphs (when using --full)
│   └── training_*/    # Iterative training outputs
├── scripts/           # Python analysis tools
│   ├── graphbrew_experiment.py  # Main experiment script
│   ├── requirements.txt
│   ├── download/      # Graph download utilities
│   ├── analysis/      # Correlation and analysis tools
│   ├── utils/         # Shared utilities
│   └── weights/       # Auto-clustered type weights
│       ├── type_registry.json  # Graph → type mappings + centroids
│       └── type_N.json         # Per-cluster weights
└── wiki/              # This documentation
```

## Troubleshooting Build Issues

### Missing Boost Headers

```bash
# Error: fatal error: boost/range/algorithm.hpp: No such file or directory
sudo apt-get install libboost-all-dev
```

### Compiler Version Too Old

```bash
# Check GCC version
g++ --version

# Install newer GCC
sudo apt-get install g++-11
export CXX=g++-11
make clean && make all
```

### OpenMP Not Found

```bash
# Error: -fopenmp not supported
sudo apt-get install libomp-dev
```

### Out of Memory During Build

```bash
# Reduce parallel jobs
make -j2 all  # Instead of default parallel
```

## Next Steps

- [[Quick-Start]] - Run your first benchmark
- [[Running-Benchmarks]] - Learn all command-line options
- [[Reordering-Algorithms]] - Understand the algorithms

---

[← Back to Home](Home)
