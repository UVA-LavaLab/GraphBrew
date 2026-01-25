# Supported Graph Formats

GraphBrew supports multiple graph file formats for flexibility and compatibility with existing datasets.

## Quick Reference

| Format | Extension | Description | Weighted |
|--------|-----------|-------------|----------|
| Edge List | `.el` | Simple text format | No |
| Weighted Edge List | `.wel` | Text with weights | Yes |
| Matrix Market | `.mtx` | Standard sparse matrix | Optional |
| DIMACS | `.gr` | Challenge format | Yes |
| Serialized | `.sg`, `.graph` | Binary format | Optional |

---

## Edge List Format (.el)

### Description

The simplest format: one edge per line with source and destination vertices.

### Format

```
<source> <destination>
<source> <destination>
...
```

### Example

```
0 1
0 2
1 2
2 3
3 4
```

### Properties

- Vertices are 0-indexed
- Whitespace separated (space or tab)
- Directed by default (add `-s` to symmetrize)
- No header required

### Usage

```bash
./bench/bin/pr -f graph.el -s -n 3
```

### Converting From Other Formats

```bash
# From CSV
cat graph.csv | tr ',' ' ' > graph.el

# Remove headers
tail -n +2 graph_with_header.el > graph.el
```

---

## Weighted Edge List Format (.wel)

### Description

Edge list with weights for SSSP and weighted algorithms.

### Format

```
<source> <destination> <weight>
<source> <destination> <weight>
...
```

### Example

```
0 1 1.5
0 2 2.0
1 2 0.5
2 3 1.0
```

### Properties

- Weights are floating-point numbers
- Used for SSSP benchmark
- Weights can be integers or decimals

### Usage

```bash
./bench/bin/sssp -f graph.wel -s -r 0 -n 3
```

### Creating Weighted Graphs

```bash
# Add random weights to edge list
awk '{print $1, $2, rand()}' graph.el > graph.wel

# Add unit weights
awk '{print $1, $2, 1.0}' graph.el > graph.wel
```

---

## Matrix Market Format (.mtx)

### Description

Standard format from the Matrix Market collection. Widely used in scientific computing.

### Format

```
%%MatrixMarket matrix coordinate pattern general
<rows> <cols> <nnz>
<row> <col>
<row> <col>
...
```

Or with values:
```
%%MatrixMarket matrix coordinate real general
<rows> <cols> <nnz>
<row> <col> <value>
...
```

### Example

```
%%MatrixMarket matrix coordinate pattern general
5 5 6
1 2
1 3
2 3
3 4
4 5
5 1
```

### Properties

- **1-indexed** (vertices start at 1, not 0)
- Header line describes matrix type
- Dimensions line: rows, columns, non-zeros
- `pattern` = unweighted, `real` = weighted
- `symmetric` = edges only stored once

### Usage

```bash
./bench/bin/pr -f graph.mtx -s -n 3
```

### Common Matrix Market Types

| Type | Meaning |
|------|---------|
| `pattern` | Binary (0/1), unweighted |
| `real` | Floating-point weights |
| `integer` | Integer weights |
| `general` | Full matrix (not symmetric) |
| `symmetric` | Only lower triangle stored |

---

## DIMACS Format (.gr)

### Description

Format from DIMACS implementation challenges. Common for road networks.

### Format

```
c Comment line
p sp <nodes> <edges>
a <source> <destination> <weight>
a <source> <destination> <weight>
...
```

### Example

```
c Example graph
c This is a comment
p sp 5 6
a 1 2 10
a 1 3 20
a 2 3 5
a 3 4 15
a 4 5 10
a 5 1 25
```

### Properties

- **1-indexed** vertices
- `c` lines are comments
- `p sp` defines problem (shortest path)
- `a` lines are arcs (edges)
- Always weighted

### Usage

```bash
./bench/bin/sssp -f graph.gr -s -r 0 -n 3
```

### DIMACS Sources

- [DIMACS Challenge](http://www.diag.uniroma1.it/challenge9/)
- Road networks (USA, Europe)

---

## Serialized Binary Format (.sg, .graph)

### Description

Binary format for fast loading. Pre-computed CSR representation.

### Creating Serialized Graphs

```bash
# Convert edge list to serialized format
./bench/bin/converter -f graph.el -s -b graph.sg
```

### Usage

```bash
./bench/bin/pr -f graph.sg -n 3
```

### Properties

- Much faster loading for large graphs
- Larger file size than text
- Architecture-specific (endianness)
- Includes symmetrization if applied

### When to Use

| Graph Size | Recommendation |
|------------|----------------|
| < 1M edges | Use text format |
| 1M - 100M edges | Consider serialized |
| > 100M edges | Use serialized |

---

## Format Detection

GraphBrew automatically detects format by extension:

| Extension | Detected Format |
|-----------|-----------------|
| `.el` | Edge list |
| `.wel` | Weighted edge list |
| `.mtx` | Matrix Market |
| `.gr` | DIMACS |
| `.sg`, `.graph` | Serialized |

### Important Notes

- Format detection is based **solely on file extension**
- There are no flags to force a specific input format
- Rename files or convert them if extension doesn't match format

---

## Converting Between Formats

### Using the Converter

```bash
# Edge list to serialized
./bench/bin/converter -f graph.el -s -b graph.sg

# Edge list to MTX format
./bench/bin/converter -f graph.el -s -p graph.mtx

# Edge list to Ligra format
./bench/bin/converter -f graph.el -s -y graph.ligra

# MTX to edge list (manual)
tail -n +3 graph.mtx | awk '{print $1-1, $2-1}' > graph.el
```

### Python Conversion

```python
import networkx as nx

# Read various formats
G = nx.read_edgelist('graph.el', nodetype=int)
G = nx.read_weighted_edgelist('graph.wel', nodetype=int)

# Write to edge list
nx.write_edgelist(G, 'output.el', data=False)
```

### Common Conversions

```bash
# MTX to EL (1-indexed to 0-indexed)
grep -v "^%" graph.mtx | tail -n +2 | awk '{print $1-1, $2-1}' > graph.el

# GR to WEL
grep "^a" graph.gr | awk '{print $2-1, $3-1, $4}' > graph.wel

# CSV to EL
cat graph.csv | tr ',' ' ' | grep -v "source" > graph.el
```

---

## Downloading Graphs

### SNAP Dataset Collection

```bash
# Facebook social network
wget https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip facebook_combined.txt.gz
mv facebook_combined.txt facebook.el

# Twitter (large)
wget https://snap.stanford.edu/data/twitter_combined.txt.gz
```

### SuiteSparse Matrix Collection

```bash
# Download MTX format
wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-Enron.tar.gz
tar xzf email-Enron.tar.gz
```

### Network Repository

```bash
# Various formats
wget http://nrvis.com/download/data/soc/socfb-Caltech36.zip
unzip socfb-Caltech36.zip
```

---

## Graph Properties

### Required Properties

| Property | Requirement |
|----------|-------------|
| Vertices | Non-negative integers |
| Self-loops | Allowed (removed by default, use `-S` to keep) |
| Multi-edges | Allowed (may affect results) |
| Directed | Yes (use `-s` for undirected) |

### Recommended Properties

| Property | Recommendation |
|----------|----------------|
| Connected | Ideally connected or mostly connected |
| No isolates | Remove isolated vertices for cleaner results |
| Reasonable size | 100+ edges for meaningful benchmarks |

### Validating Graphs

```bash
# Check basic properties
head -5 graph.el
wc -l graph.el  # Edge count

# Check for issues
sort graph.el | uniq -c | sort -rn | head  # Duplicate edges
awk '$1==$2' graph.el  # Self-loops
```

---

## Troubleshooting

### "Invalid graph format"

```bash
# Check for headers
head -3 graph.el

# Check whitespace
cat -A graph.el | head -3  # Shows special chars

# Fix Windows line endings
dos2unix graph.el
```

### "Vertex out of range"

```bash
# Check indexing (should start at 0)
awk '{print $1, $2}' graph.el | sort -n | head -3

# Convert 1-indexed to 0-indexed
awk '{print $1-1, $2-1}' graph.el > graph_fixed.el
```

### "Graph has 0 edges"

```bash
# Check file is not empty
ls -la graph.el

# Check format matches extension
file graph.el
```

### Memory Issues with Large Graphs

```bash
# Check estimated memory (rough: 16 bytes per edge)
edges=$(wc -l < graph.el)
echo "Estimated memory: $((edges * 16 / 1024 / 1024)) MB"

# Use streaming for very large files
split -l 10000000 graph.el graph_part_
```

---

## Test Graphs

GraphBrew includes test graphs in `test/graphs/`:

| File | Nodes | Edges (undirected) | Description |
|------|-------|-------------------|-------------|
| `4.el` | 14 | 53 | Small test graph |
| `5.el` | 6 | 6 | Tiny test graph |
| `4.mtx` | 14 | 53 | MTX format |
| `4.gr` | 14 | 53 | DIMACS format |
| `4.wel` | 14 | 53 | Weighted |

Use these for quick testing:
```bash
./bench/bin/pr -f test/graphs/4.el -s -n 3
```

---

## Next Steps

- [[Running-Benchmarks]] - How to run benchmarks
- [[Quick-Start]] - Getting started guide
- [[Graph-Benchmarks]] - Available algorithms

---

[← Back to Home](Home) | [Running Benchmarks →](Running-Benchmarks)
