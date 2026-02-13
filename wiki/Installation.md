# Installation Guide

This guide covers installing GraphBrew on Linux systems.

## 🚀 Quick Install (One-Click)

The fastest way to get started - automatically checks dependencies, builds, and downloads test graphs:

```bash
git clone https://github.com/UVA-LavaLab/GraphBrew.git
cd GraphBrew

# Check what dependencies are missing
python3 scripts/graphbrew_experiment.py --check-deps

# Auto-install missing dependencies (needs sudo)
python3 scripts/graphbrew_experiment.py --install-deps

# Run full pipeline
python3 scripts/graphbrew_experiment.py --full --size small
```

This will:
- Verify all system dependencies (Boost, g++, libnuma, etc.)
- Build all binaries automatically
- Download benchmark graphs from SuiteSparse
- Run the complete experiment pipeline

---

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Compiler**: GCC 7+ with C++17 support
- **Memory**: 8GB RAM (16GB+ recommended for large graphs)
- **Disk**: 50GB+ for benchmark graphs

### Automatic Dependency Check

GraphBrew can check and install dependencies automatically:

```bash
# Check all dependencies
python3 scripts/graphbrew_experiment.py --check-deps

# Sample output:
# GraphBrew Dependency Status:
# ==================================================
#   ✓ make             GNU Make 4.3
#   ✓ c++ compiler     g++ 11.4.0
#   ✓ boost            v1.74
#   ✓ numa             found
#   ✓ tcmalloc         found
#   ✓ python           v3.10.12
# ==================================================
#   All required dependencies satisfied!

# Install missing dependencies (Ubuntu/Debian/Fedora/macOS)
python3 scripts/graphbrew_experiment.py --install-deps
```

### Manual Dependency Installation

If automatic installation doesn't work, install manually:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    g++ \
    libboost-all-dev \
    libnuma-dev \
    google-perftools \
    python3 \
    python3-pip

# Fedora/RHEL
sudo dnf install -y \
    gcc-c++ \
    make \
    boost-devel \
    numactl-devel \
    gperftools \
    python3 \
    python3-pip

# Arch Linux
sudo pacman -S \
    base-devel \
    boost \
    numactl \
    gperftools

# macOS (with Homebrew)
xcode-select --install
brew install gcc boost google-perftools
```

### Optional Dependencies

```bash
# For visualization and analysis (optional)
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
| `RABBIT_ENABLE=0` | Disable Rabbit Order (enabled by default) | `RABBIT_ENABLE=0 make all` |
| `DEBUG=1` | Build with debug symbols | `DEBUG=1 make all` |
| `SANITIZE=1` | Enable address sanitizer | `SANITIZE=1 make all` |

### Rabbit Order and Boost 1.58

Rabbit Order (algorithm 8) has two variants:
- **`csr` (default)**: Native CSR implementation - faster, no external dependencies
- **`boost`**: Original Boost-based implementation - requires Boost 1.58.0

**Note:** The `csr` variant is used by default and does not require Boost. Only install Boost 1.58 if you need the original `boost` variant.

The `boost` variant requires **Boost 1.58.0** specifically for compatibility.
System package managers often install newer versions which may cause issues.

#### Automatic Installation (Recommended)

```bash
# Download, compile, and install Boost 1.58.0 to /opt/boost_1_58_0
python3 scripts/graphbrew_experiment.py --install-boost
```

This automatically:
1. Downloads Boost 1.58.0 source (~73MB)
2. Runs `bootstrap.sh` to configure
3. Compiles with `b2` using all CPU cores
4. Installs to `/opt/boost_1_58_0` with proper `include/` and `lib/` structure

**Note:** Compilation takes 10-30 minutes depending on your system.

#### Manual Installation

```bash
# Download Boost 1.58.0
wget https://archives.boost.io/release/1.58.0/source/boost_1_58_0.tar.gz
tar -xzf boost_1_58_0.tar.gz
cd boost_1_58_0

# Configure
./bootstrap.sh --prefix=/opt/boost_1_58_0

# Compile and install (uses all CPU cores)
cpuCores=$(nproc)
sudo ./b2 --with=all -j $cpuCores install

# Cleanup
cd .. && rm -rf boost_1_58_0 boost_1_58_0.tar.gz
```

#### Verify Boost Installation

```bash
# Check Boost version at GraphBrew expected path
cat /opt/boost_1_58_0/include/boost/version.hpp | grep "BOOST_LIB_VERSION"
# Should show: #define BOOST_LIB_VERSION "1_58"

# Or use the dependency checker
python3 scripts/graphbrew_experiment.py --check-deps
```

#### Building with Rabbit Order

```bash
# Build with RabbitOrder CSR variant (default, no Boost required)
make all

# Build with Boost support for RabbitOrder boost variant (requires Boost 1.58)
RABBIT_ENABLE=1 make all  # Boost at /opt/boost_1_58_0

# Or disable Rabbit Order boost support entirely
RABBIT_ENABLE=0 make all
```

**Note:** The `csr` variant works without Boost. `RABBIT_ENABLE` controls whether
the Boost-based `boost` variant is compiled.

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

## Python Environment

Core scripts require only Python 3.8+ standard library. Optional: `pip3 install numpy matplotlib pandas` for visualization.

```bash
python3 scripts/graphbrew_experiment.py --help  # Verify scripts work
```

## Directory Structure After Build

See [[Code-Architecture]] for full directory layout. Key paths:
- `bench/bin/` — Compiled binaries (pr, bfs, cc, etc.)
- `bench/bin_sim/` — Cache simulation binaries
- `scripts/lib/` — Core Python modules
- `results/weights/` — Perceptron weight files
- `results/` — Experiment outputs

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

[← Back to Home](Home) | [Quick Start →](Quick-Start)
