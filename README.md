[![Build Status](https://app.travis-ci.com/atmughrabi/GLay.svg?token=L3reAtGHdEVVPvzcVqQ6&branch=main)](https://app.travis-ci.com/atmughrabi/GLay)
[<p align="center"><img src="./docs/figures/logo.svg" width="200" ></p>](#graphbrew)

# GraphBrew

This repository contains the GAP Benchmarks Suite (GAPBS), modified to reorder graphs and test cache performance on various graph algorithms.

## Enhancements (Graph Brewing)

* **Multi-layered:** Graph reordering (multi-layered) for improved cache performance.
* **Rabbit Order:**  Community clustering order with incremental aggregation.
* **Leiden Order:**  Community clustering order with Louvian/refinement step.
* **Degree-Based Grouping:** Implementing degree-based grouping strategies to test benchmark performance.
* **Gorder:**  Window based ordering with reverse Cuthill-McKee (RCM) algorithm.
* **Corder:**  Workload Balancing via Graph Reordering on Multicore Systems.
* **P-OPT Segmentation:**  Exploring graph caching techniques for efficient handling of large-scale graphs.
* **GraphIt-DSL:** Integration of GraphIt-DSL segment graphs to improve locality.

## GAP Benchmarks

This project contains a collection of Graph Analytics for Performance (GAPBS) benchmarks implemented in C++. The benchmarks are designed to exercise the performance of graph algorithms on a CPU. 

**Key Algorithms**

* **bc:** Betweenness Centrality 
* **bfs:** Breadth-First Search (Direction Optimized) 
* **cc:** Connected Components (Afforest)
* **cc_sv:** Connected Components (ShiloachVishkin)
* **pr:** PageRank
* **pr_spmv:** PageRank (using sparse matrix-vector multiplication)
* **sssp:**  Single-Source Shortest Paths
* **tc:** Triangle Counting

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- **GCC**: The GNU Compiler Collection, specifically `g++` which supports C++11 or later.
- **Make**: The build utility to automate the compilation.
- **OpenMP**: Support for parallel programming in C++.
- **Boost** C++ library (1.58.0).
- **libnuma** (2.0.9).
- **libtcmalloc\_minimal** in google-perftools (2.1).

These tools are available on most Unix-like operating systems and can be installed via your package manager. For example, on Ubuntu, you can install them using:

```bash
sudo apt-get update
sudo apt-get install g++ make libomp-dev
```
### Installing Boost 1.58.0

1. **Remove Debian Package Installations**
   
   * If you installed Boost through Debian packages, use the following command to remove these installations:
```bash
sudo apt-get -y --purge remove libboost-all-dev libboost-doc libboost-dev
```
   * If Boost was installed from the source on your system, you can remove the installed library files with:
```bash
sudo rm -f /usr/lib/libboost_*
```

2. **Installing Boost 1.58.0**
   
   * First, navigate to your home directory and download the desired Boost version:
```bash
cd ~
wget http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz
tar -zxvf boost_1_58_0.tar.gz
cd boost_1_58_0
```
   * Determine the number of CPU cores available to optimize the compilation process:
```bash
cpuCores=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $NF}')
echo "Available CPU cores: $cpuCores"
```
   * Initialize the Boost installation script:
```bash
./bootstrap.sh  # This script prepares Boost for installation
```
   * Compile and install Boost using all available cores to speed up the process:
```bash
sudo ./b2 --with=all -j $cpuCores install
```

3. **Verify the Installation**
   
   * After installation, verify that Boost has been installed correctly by checking the installed version:
```bash
cat /usr/local/include/boost/version.hpp | grep "BOOST_LIB_VERSION"
```
   * The output should display the version of Boost you installed, like so:
```bash
//  BOOST_LIB_VERSION must be defined to be the same as BOOST_VERSION
#define BOOST_LIB_VERSION "1_58"
```

## Usage

### Example Usage
   * To compile, run, and then clean up the betweenness centrality benchmark:
```bash
make all
make run-bc
make clean
```
### Compiling the Benchmarks
   * To build all benchmarks:
```bash
make all
```

### Running the Benchmarks
   * To run a specific benchmark, use:
```bash
make run-<benchmark_name>
```
   * Where `<benchmark_name>` can be `bc`, `bfs`, etc.
```bash
make run-bfs
```

### Debugging
   * To run a benchmark with gdb:
```bash
make run-<benchmark_name>-gdb
```
   * To run a benchmark with memory checks (using valgrind):
```bash
make run-<benchmark_name>-mem
```

### Clean up
   * To clean up all compiled files:
```bash
make clean
```

### Help
   * To display help for a specific benchmark or for general usage:
```bash
make help-<benchmark_name>
make help
```

## Additional Commands

### Benchmark Parameter Sweeping
   * To sweep through different configurations (`-o` options from 1 to 13):
```bash
make run-<benchmark_name>-sweep
```

## Modifying the Makefile

* **Compiler and Flags:** Edit the `CXX`, `CXXFLAGS`, and `INCLUDES` variables to customize compilation settings.
* **Benchmark Targets:** Add or remove benchmark names from the `KERNELS` variable to control which ones are built.

**Project Structure**
- `bench/bin`: Executables are placed here.
- `bench/lib`: Library files can be stored here (not used by default).
- `bench/src`: Source code files (*.cc) for the benchmarks.
- `bench/include`: Header files for the benchmarks and various include files for libraries such as GAPBS, RABBIT, etc.
- `bench/obj`: Object files are stored here (directory creation is handled but not used by default).

**Compiler Flags**
- **CXXFLAGS**: Combines optimization and feature flags necessary for compiling the benchmarks.
- **LDLIBS**: Specifies additional linker flags needed for libraries such as `tcmalloc_minimal` and `numa`.

Graph Loading
-------------

All of the binaries use the same command-line options for loading graphs:
+ `-g 20` generates a Kronecker graph with 2^20 vertices (Graph500 specifications)
+ `-u 20` generates a uniform random graph with 2^20 vertices (degree 16)
+ `-f graph.el` loads graph from file graph.el
+ `-sf graph.el` symmetrizes graph loaded from file graph.el

The graph loading infrastructure understands the following formats:
+ `.el` plain-text edge-list with an edge per line as _node1_ _node2_
+ `.wel` plain-text weighted edge-list with an edge per line as _node1_ _node2_ _weight_
+ `.gr` [9th DIMACS Implementation Challenge](http://www.dis.uniroma1.it/challenge9/download.shtml) format
+ `.graph` Metis format (used in [10th DIMACS Implementation Challenge](http://www.cc.gatech.edu/dimacs10/index.shtml))
+ `.mtx` [Matrix Market](http://math.nist.gov/MatrixMarket/formats.html) format
+ `.sg` serialized pre-built graph (use `converter` to make)
+ `.wsg` weighted serialized pre-built graph (use `converter` to make)

GraphBrew Parameters
-------------
1. **GAP Parameters**
   * Reorder the graph, orders can bet layered.
   * Segment the graph for scalability, requires modifying the algorithm to iterate through segments.

```bash
-o <order>  : apply reordering strategy, optionally layer ordering 
               [example]-o 3 -o 2 -o 10:mapping.label [optional]

-j <segments>: number of segments for the graph [default:1]

Reordering Algorithms:
  - ORIGINAL      (0):  No reordering applied.
  - RANDOM        (1):  Apply random reordering.
  - SORT          (2):  Apply sort-based reordering.
  - HUBSORT       (3):  Apply hub-based sorting.
  - HUBCLUSTER    (4):  Apply clustering based on hub scores.
  - DBG           (5):  Apply degree-based grouping.
  - HUBSORTDBG    (6):  Combine hub sorting with degree-based grouping.
  - HUBCLUSTERDBG (7):  Combine hub clustering with degree-based grouping.
  - RABBITORDER   (8):  Apply community clustering with incremental aggregation.
  - GORDER        (9):  Apply dynamic programming BFS and windowing ordering.
  - CORDER        (10): Workload Balancing via Graph Reordering on Multicore Systems.
  - RCM           (11): RCM is ordered by the reverse Cuthill-McKee algorithm (BFS).
  - LeidenOrder   (12): Apply Leiden community clustering with louvain with refinement.
  - MAP           (13): Requires a file format for reordering. Use the -r 10:filename.label option.
```

2. **Makefile Flow**
```bash
available Make commands:
  all            - Builds all targets including GAP benchmarks (CPU)
  run-%          - Runs the specified GAP benchmark (bc bfs cc cc_sv pr pr_spmv sssp tc)
  help-%         - Print the specified Help (bc bfs cc cc_sv pr pr_spmv sssp tc)
  clean          - Removes all build artifacts
  help           - Displays this help message

Example Usage:
  make all - Compile the program.
  make clean - Clean build files.
  ./bench/bin/pr -g 15 -n 1 -r 10:mapping.label - Execute with MAP reordering using 'mapping.label'.

```

How to Cite
-----------

Please cite the following papers if you find this repository useful.
+ S. Beamer, K. Asanović, and D. Patterson, “The GAP Benchmark Suite,” arXiv:1508.03619 [cs], May 2017.
+ J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, and S. Iwamura.Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis.
+ P. Faldu, J. Diamond, and B. Grot, “A Closer Look at Lightweight Graph Reordering,” arXiv:2001.08448 [cs], Jan. 2020.
+ V. Balaji, N. Crago, A. Jaleel, and B. Lucia, “P-OPT: Practical Optimal Cache Replacement for Graph Analytics,” in 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), Feb. 2021, pp. 668–681. doi: 10.1109/HPCA51647.2021.00062.
+ Y. Zhang, V. Kiriansky, C. Mendis, S. Amarasinghe, and M. Zaharia, “Making caches work for graph analytics,” in 2017 IEEE International Conference on Big Data (Big Data), Dec. 2017, pp. 293–302. doi: 10.1109/BigData.2017.8257937.
+ Y. Zhang, M. Yang, R. Baghdadi, S. Kamil, J. Shun, and S. Amarasinghe, “GraphIt: a high-performance graph DSL,” Proc. ACM Program. Lang., vol. 2, no. OOPSLA, p. 121:1-121:30, Oct. 2018, doi: 10.1145/3276491.
+ H. Wei, J. X. Yu, C. Lu, and X. Lin, “Speedup Graph Processing by Graph Ordering,” New York, NY, USA, Jun. 2016, pp. 1813–1828. doi: 10.1145/2882903.2915220.
+ Y. Chen and Y.-C. Chung, “Workload Balancing via Graph Reordering on Multicore Systems,” IEEE Transactions on Parallel and Distributed Systems, 2021.
+ A. George and J. W. H. Liu, Computer Solution of Large Sparse Positive Definite Systems. Prentice-Hall, 1981
+ S. Sahu, “GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting.” arXiv, Mar. 28, 2024. doi: 10.48550/arXiv.2312.13936.
+ V. A. Traag, L. Waltman, and N. J. van Eck, “From Louvain to Leiden: guaranteeing well-connected communities,” Sci Rep, vol. 9, no. 1, p. 5233, Mar. 2019, doi: 10.1038/s41598-019-41695-z.
