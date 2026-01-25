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
   # Use valid IDs: 0-17 (13=MAP requires external .lo file)
   ./bench/bin/pr -f graph.el -s -o 7  # Valid
   ./bench/bin/pr -f graph.el -s -o 13 # Needs external .lo file
   ```

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
1. Different algorithms
2. AdaptiveOrder (14)
3. Check if graph has community structure

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

### JSON decode error in perceptron weights

```bash
# Validate JSON
python3 -c "import json; json.load(open('scripts/weights/active/type_0.json'))"

# Pretty-print to find error
python3 -m json.tool scripts/weights/active/type_0.json

# Check all type weight files
for f in scripts/weights/active/type_*.json; do
    echo "Checking $f..."
    python3 -c "import json; json.load(open('$f'))" && echo "OK" || echo "FAILED"
done
```

### Wrong type cluster selected

Graph type is selected via Euclidean distance to cluster centroids. If selection is wrong:

```bash
# Check what properties were detected
cat results/graph_properties_cache.json | python3 -m json.tool | grep -A 10 "your_graph_name"

# Check type registry centroids
cat scripts/weights/active/type_registry.json | python3 -m json.tool

# Re-run Phase 0 to recompute properties
python3 scripts/graphbrew_experiment.py --fill-weights --graphs small
```

**Auto-clustering system:**
Uses 9 features and Euclidean distance to match graphs to the nearest cluster centroid.

| Type | Typical Properties |
|------|----------|
| road | modularity < 0.1, degree_variance < 0.5, avg_degree < 10 |
| social | modularity > 0.3, degree_variance > 0.8 |
| web | hub_concentration > 0.5, degree_variance > 1.0 |
| powerlaw | degree_variance > 1.5, modularity < 0.3 |
| uniform | degree_variance < 0.5, hub_concentration < 0.3, modularity < 0.1 |

### Per-type weight files not generated

Ensure `--fill-weights` completes all phases:

```bash
# Check Phase 7 ran
grep "Phase 7" results/logs/*.log

# Manually generate per-type weights from existing results
python3 scripts/graphbrew_experiment.py --phase weights
```

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
./bench/bin/pr -f test/graphs/4.el -s -v -n 1
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
./bench/bin/pr -f test/graphs/4.el -s -n 1

# Graph file check
file graph.el
wc -l graph.el
head -5 graph.el

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
