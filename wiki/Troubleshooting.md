# Troubleshooting Guide

Solutions for common issues when using GraphBrew.

---

## Build Issues

### Error: "g++: command not found"

**Solution**: Install GCC

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
brew install gcc
```

### Error: "unrecognized command line option '-std=c++17'"

**Cause**: GCC version is too old (need 7+)

**Solution**: Update GCC

```bash
# Ubuntu
sudo apt-get install g++-9
make CXX=g++-9

# macOS
brew install gcc@13
make CXX=g++-13
```

### Error: "omp.h: No such file or directory"

**Cause**: OpenMP not installed

**Solution**:

```bash
# Ubuntu
sudo apt-get install libomp-dev

# macOS
brew install libomp
```

### Error: "fatal error: bits/c++config.h: No such file"

**Solution**: Install multilib support

```bash
sudo apt-get install gcc-multilib g++-multilib
```

### Undefined reference to `omp_*`

**Cause**: Not linking OpenMP

**Solution**: Ensure `-fopenmp` is in both compile and link flags

```bash
make clean
make CXXFLAGS="-fopenmp -std=c++17 -O3"
```

### RabbitOrder Boost Variant Build Issues

**Error**: "boost/graph/adjacency_list.hpp: No such file"

**Cause**: Boost 1.58 not installed but trying to use RabbitOrder `boost` variant.

**Solutions**:

1. **Use the CSR variant (recommended)** - No Boost needed:
   ```bash
   # CSR variant is default, no build changes needed
   ./bench/bin/pr -f graph.el -o 8 -n 1      # Uses csr variant
   ./bench/bin/pr -f graph.el -o 8:csr -n 1  # Explicit csr
   ```

2. **Install Boost 1.58 for the boost variant**:
   ```bash
   python3 scripts/graphbrew_experiment.py --install-boost
   make clean && make all
   ./bench/bin/pr -f graph.el -o 8:boost -n 1
   ```

3. **Disable Boost support entirely**:
   ```bash
   RABBIT_ENABLE=0 make all
   ```

**Note**: The `csr` variant is faster and has no dependencies. The `boost` variant 
is the original implementation requiring Boost 1.58.0 specifically.

---

## Graph Loading Issues

### Error: "Cannot open file"

**Causes & Solutions**:

1. **File doesn't exist**
   ```bash
   ls -la graph.el
   # Use absolute path
   ./bench/bin/pr -f /full/path/to/graph.el -s
   ```

2. **Permission denied**
   ```bash
   chmod 644 graph.el
   ```

3. **Path has spaces**
   ```bash
   ./bench/bin/pr -f "path with spaces/graph.el" -s
   ```

### Error: "Invalid graph format"

**Diagnose**:
```bash
# Check file content
head -10 graph.el

# Check for hidden characters
cat -A graph.el | head -5

# Check line endings
file graph.el
```

**Common fixes**:

```bash
# Fix Windows line endings
dos2unix graph.el

# Remove header line
tail -n +2 graph.el > graph_clean.el

# Fix tabs
cat graph.el | tr '\t' ' ' > graph_clean.el
```

### Graph has 0 nodes/edges

**Causes**:
1. Empty file
2. Format mismatch (using .mtx options with .el file)
3. All edges filtered out

**Solution**:
```bash
# Verify file has content
wc -l graph.el

# Check format detection
head -5 graph.el
```

### SuiteSparse .mtx Conversion Fails

**Cause**: SuiteSparse archives often contain auxiliary `.mtx` files
(e.g., `*_nodename.mtx`, `*_coord.mtx`) alongside the actual graph matrix.
The converter picks the first `.mtx` file found, which may be a metadata
file (array format) rather than the sparse graph (coordinate format).

**Diagnose**:
```bash
# List all .mtx files in the graph directory
find results/graphs/GRAPH_NAME/ -name "*.mtx"
# The graph file has "coordinate" format in its header:
head -1 results/graphs/GRAPH_NAME/GRAPH_NAME/GRAPH_NAME.mtx
# Should show: %%MatrixMarket matrix coordinate ...
```

**Solution**:
```bash
# Convert manually using the correct .mtx file
./bench/bin/converter \
    -f results/graphs/GRAPH_NAME/GRAPH_NAME/GRAPH_NAME.mtx \
    -s -o 1 -b results/graphs/GRAPH_NAME/GRAPH_NAME.sg
```

### "Vertex index out of range"

**Cause**: Vertices not 0-indexed

**Solution**:
```bash
# Convert 1-indexed to 0-indexed
awk '{print $1-1, $2-1}' graph.el > graph_0indexed.el
```

---

## Runtime Issues

### Segmentation Fault

**Diagnose**:
```bash
# Build with debug symbols
make clean
make DEBUG=1

# Run with debugger
gdb ./bench/bin/pr
(gdb) run -f graph.el -s
(gdb) bt  # backtrace when it crashes
```

**Common causes**:

1. **Corrupted graph file**
   ```bash
   # Validate edges
   awk '{if($1<0 || $2<0) print "Bad line:", NR}' graph.el
   ```

2. **Out of memory** - See memory issues below

3. **Invalid algorithm ID**
   ```bash
   # Use valid IDs: 0-16 (13=MAP requires external .lo file)
   ./bench/bin/pr -f graph.el -s -o 7  # Valid
   ./bench/bin/pr -f graph.el -s -o 13 # Needs external .lo file
   ```

### Floating Point Exception (FPE)

**Symptoms**: "Floating point exception (core dumped)" error, especially with GraphBrewOrder (12) on Kronecker graphs or graphs with extreme community structure.

**Root Cause**: 
Integer division by zero can occur when processing communities with no internal edges. This happens on graphs with extreme structure (e.g., Kronecker graphs) where:
- Leiden creates many small communities
- Some communities have nodes only connected to OTHER communities
- The induced subgraph has 0 internal edges → empty graph → division by zero in `avgDegree = num_edges / num_nodes`

**Solution**: 
This issue was fixed in the codebase with guards in:
- `ReorderCommunitySubgraphStandalone` - Skips reordering for empty subgraphs
- `GenerateHubSortMapping`, `GenerateHubClusterMapping`, `GenerateDBGMapping`, etc. - Guard against `num_nodes == 0`
- `GenerateCOrderMapping`, `GenerateCOrderMapping_v2` - Guard against empty graphs
- `GVELeidenAdaptiveCSR` - Returns empty result for empty graphs

If you encounter this error, ensure you have the latest version:
```bash
git pull origin main
make clean && make all
```

**Graphs that may trigger this (now handled)**:
- Kronecker graphs (kron_g500-logn*)
- Synthetic power-law graphs with extreme degree distributions
- Graphs with highly disconnected community structure

### Out of Memory

**Symptoms**: Killed, OOM killer, very slow

**Solutions**:

1. **Check memory requirements**
   ```bash
   # Rough estimate: 16 bytes per edge
   edges=$(wc -l < graph.el)
   echo "Need approximately $((edges * 20 / 1024 / 1024)) MB"
   ```

2. **Reduce memory usage**
   ```bash
   # Use smaller graph for testing
   head -100000 graph.el > graph_small.el
   ```

3. **Increase swap** (temporary)
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Program Hangs

**Diagnose**:
```bash
# Check if still running
top -p $(pgrep -f "bench/bin")

# Check for deadlock
gdb -p $(pgrep -f "bench/bin")
(gdb) thread apply all bt
```

**Solutions**:

1. **Timeout protection**
   ```bash
   timeout 3600 ./bench/bin/pr -f graph.el -s
   ```

2. **Reduce threads**
   ```bash
   export OMP_NUM_THREADS=1
   ./bench/bin/pr -f graph.el -s
   ```

---

## Performance Issues

### Benchmark is Very Slow

**Check parallelism**:
```bash
echo $OMP_NUM_THREADS  # Should be set or use all cores
export OMP_NUM_THREADS=$(nproc)
```

**Check CPU frequency**:
```bash
# Disable power saving
sudo cpupower frequency-set -g performance
```

**Check memory bandwidth**:
```bash
# Use NUMA binding
numactl --cpunodebind=0 --membind=0 ./bench/bin/pr -f graph.el -s
```

### Inconsistent Timing Results

**Causes & Solutions**:

1. **Run more trials**
   ```bash
   ./bench/bin/pr -f graph.el -s -n 10
   ```

2. **Isolate system**
   ```bash
   # Use taskset to pin to specific CPUs
   taskset -c 0-7 ./bench/bin/pr -f graph.el -s
   ```

3. **Disable turbo boost**
   ```bash
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

### Reordering Doesn't Help

**Expected for**:
- Small graphs (< 10K vertices)
- Already well-ordered graphs
- Some road networks

**Try**:
1. Compare ORIGINAL, CSR Rabbit, and Boost Rabbit.
2. Measure mapping time separately from kernel time.
3. Use the frozen reuse-1/2 policy only within its documented kernel scope.

---

## Python Script Issues

### "ModuleNotFoundError: No module named 'numpy'"

```bash
pip install -r scripts/requirements.txt
```

### "Permission denied: ./bench/bin/pr"

```bash
chmod +x bench/bin/*
```

### Python script can't find binaries

```bash
# Check binaries exist
ls bench/bin/

# Build if missing
make all

# Use absolute path in scripts
--bin-dir /full/path/to/GraphBrew/bench/bin
```

### Reproducing the historical low-reuse policy

Use the complete algorithm-14 string and provide reuse 1 or 2:

```bash
./bench/bin/pr -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:1' -n 3
```

The policy can fall back when its historical predicate is false. It is a
compatibility experiment, not the current paper recommendation. For new
scientific runs, use an explicit Algorithm-12 composition and include
ORIGINAL. See [AdaptiveOrder](AdaptiveOrder).

### Historical offline-model artifact errors

`results/data/adaptive_models.json` is used only by retained offline-model
experiments. Validate it with:

```bash
python3 -m json.tool results/data/adaptive_models.json
```

The deterministic historical rule does not require this file.

---

## Verification Failures

### "Verification FAILED"

**For PageRank**:
- Check convergence: may need more iterations
- Check tolerance setting

**For BFS/SSSP**:
- Ensure graph is connected
- Check root vertex exists

**For CC**:
- Graph might have issues

**Debug**:
```bash
# Run with smaller graph
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -v -n 1
```

---

## Environment Issues

### Works on one machine, not another

**Check**:
1. Same GCC version
2. Same library versions
3. Same graph file (transfer with binary mode)

### Different results on different machines

**Expected causes**:
- Floating-point differences
- Thread scheduling differences
- Architecture differences

**For reproducibility**:
```bash
export OMP_NUM_THREADS=1
./bench/bin/pr -f graph.el -s -n 1
```

---

## Data Migration Issues

### Old Data Structure After Upgrade

**Symptom**: After updating GraphBrew, per-graph data is in `results/graphs/<graph>/benchmarks/` instead of the new `results/logs/<graph>/runs/<timestamp>/` structure.

**Solution**: Run the migration script:

```bash
# Migrate all graphs to new structure
python3 -m scripts.lib.core.graph_data --migrate

# Migrate a specific graph
python3 -m scripts.lib.core.graph_data --migrate-graph ca-GrQc

# Verify migration
python3 -m scripts.lib.core.graph_data --list-runs ca-GrQc
```

### Managing Multiple Experiment Runs

**List runs for a graph**:
```bash
python3 -m scripts.lib.core.graph_data --list-runs ca-GrQc
# Output: Runs for ca-GrQc (3 total):
#   20260127_152449: 80 benchmarks, 16 reorders
#   20260127_152437: 80 benchmarks, 16 reorders
#   migrated_20260127_151844: 80 benchmarks, 16 reorders
```

**Show specific run details**:
```bash
python3 -m scripts.lib.core.graph_data --show-run ca-GrQc 20260127_152449
```

**Clean up old runs** (keep most recent N):
```bash
python3 -m scripts.lib.core.graph_data --cleanup-runs --max-runs 5
```

---

## Quick Diagnostic Commands

```bash
# System info
uname -a
g++ --version
cat /proc/cpuinfo | grep "model name" | head -1
free -h

# GraphBrew check
ls -la bench/bin/
./bench/bin/pr --help
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -n 1

# Graph file check
file graph.el
wc -l graph.el
head -5 graph.el

# Per-graph data check
python3 -m scripts.lib.core.graph_data --list-graphs
python3 -m scripts.lib.core.graph_data --list-runs ca-GrQc

# Resource monitoring
top -d 1 -p $(pgrep -f "bench/bin")
vmstat 1
```

---

## Getting Help

If you've tried the above and still have issues:

1. **Search existing issues** on GitHub
2. **Create minimal reproducible example**
3. **Include diagnostic info**:
   - OS and version
   - GCC version
   - Command run
   - Error message
   - Graph file (or describe it)

4. **Open GitHub issue** with all info

---

[← Back to Home](Home) | [FAQ →](FAQ)
