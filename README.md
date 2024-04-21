[![Build Status](https://app.travis-ci.com/atmughrabi/GLay.svg?token=L3reAtGHdEVVPvzcVqQ6&branch=main)](https://app.travis-ci.com/atmughrabi/GLay)


# SuperGAP Benchmarks Suite

This repository contains the GAP (Graph Algorithms in Practice) benchmarks suite, designed to compile and run various graph algorithms using C++. The Makefile in this repository automates the process of compiling these benchmarks from source code.

## Enhancements

* **GraphIt-DSL:** Integration of GraphIt-DSL segment graphs to improve locality.
* **Degree-Based Grouping:** Implementing degree-based grouping strategies to test benchmark performance.
* **Rabbit Order:**  Community clustering order with incremental aggregation.
* **P-OPT Segmentation:**  Exploring graph caching techniques for efficient handling of large-scale graphs.

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

2. **Verify the Installation**
   
   * After installation, verify that Boost has been installed correctly by checking the installed version:
```bash
cat /usr/local/include/boost/version.hpp | grep "BOOST_LIB_VERSION"
```
   * The output should display the version of Boost you installed, like so:
```bash
//  BOOST_LIB_VERSION must be defined to be the same as BOOST_VERSION
#define BOOST_LIB_VERSION "1_54"
```

## GAP Benchmarks

This project contains a collection of Graph Analytics for Performance (GAP) benchmarks implemented in C++. The benchmarks are designed to exercise the performance of graph algorithms on a CPU. 

**Key Algorithms**

* **bc:** Betweenness Centrality 
* **bfs:** Breadth-First Search (Direction Optimized) 
* **cc:** Connected Components (Afforest)
* **cc_sv:** Connected Components (ShiloachVishkin)
* **pr:** PageRank
* **pr_spmv:** PageRank (using sparse matrix-vector multiplication)
* **sssp:**  Single-Source Shortest Paths
* **tc:** Triangle Counting

## Build Targets

The Makefile compiles and links the following executable targets:

* **Executable Benchmarks:** Located in the `bench/bin` directory.
* **Converter:** A utility for converting graphs into the project's input format (`bench/bin/converter`)

**Getting Started**

1. **Prerequisites**
   * A C++11 compliant compiler (e.g., g++)
   * OpenMP library for multi-threading support

2. **Building the Project**
   * Navigate to the project's root directory.
   * Run `make all` to compile the benchmarks and converter utility.

3. **Running the Benchmarks**
   * Run `make run-benchmark_name` (replace 'benchmark_name' with an algorithm name from the list above). Example: `make run-bc`  
   * The benchmark will use the following default parameters:
       * Random Graph (V): 2^10
       * Number of Runs: 1 

**Modifying the Makefile**

* **Compiler and Flags:** Edit the `CXX`, `CXXFLAGS`, and `INCLUDES` variables to customize compilation settings.
* **Benchmark Targets:** Add or remove benchmark names from the `KERNELS` variable to control which ones are built.

**Project Structure**

* `bench/src`: Contains the C++ source files for the benchmarks.
* `bench/include`: Contains header files.
* `bench/bin`: Stores the compiled executables.
* `bench/lib`: Used for intermediate build objects.

## Boost Installation Guide

This README outlines the steps for uninstalling Boost libraries installed via Debian packages or directly from source, and the subsequent installation of Boost version 1.54.0.

## Prerequisites

Before proceeding, update the package list and install necessary development tools and libraries:

```bash
sudo apt-get update
sudo apt-get -y install build-essential g++ python-dev autotools-dev libicu-dev libbz2-dev
```

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

How to Cite
-----------

Please cite the following papers if you find this repository useful.

+ Scott Beamer, Krste Asanović, David Patterson. "[*The GAP Benchmark Suite*](http://arxiv.org/abs/1508.03619)". arXiv:1508.03619 [cs.DC], 2015.
+ J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, and S. Iwamura.Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis.
     