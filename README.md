[![Build Status](https://app.travis-ci.com/atmughrabi/GraphBrew.svg?token=L3reAtGHdEVVPvzcVqQ6&branch=main)](https://app.travis-ci.com/atmughrabi/GraphBrew)
[<p align="center"><img src="./docs/figures/logo.svg" width="200" ></p>](#graphbrew)

# GraphBrew '>>' GAPBS

This repository contains the GAP Benchmarks Suite [(GAPBS)](https://github.com/sbeamer/gapbs), modified to reorder graphs and improve cache performance on various graph algorithms.

## Enhancements with Cache Friendly Graphs (Graph Brewing)

* **GraphBrew:** Graph reordering (multi-layered) for improved cache performance.
* **Leiden Order:** [link](https://github.com/puzzlef/leiden-communities-openmp) Community clustering order with Louvian/refinement step.
* **Rabbit Order:** [link](https://github.com/araij/rabbit_order) Community clustering order with incremental aggregation.
* **Degree-Based Grouping:** [link](https://github.com/ease-lab/dbg) Implementing degree-based grouping strategies to test benchmark performance.
* **Gorder:** [link](https://github.com/datourat/Gorder) Window based ordering with reverse Cuthill-McKee (RCM) algorithm.
* **Corder:** [link](https://github.com/yuang-chen/Corder-TPDS-21) Workload Balancing via Graph Reordering on Multicore Systems.
* **P-OPT Segmentation:** [link](https://github.com/CMUAbstract/POPT-CacheSim-HPCA21) Exploring graph caching techniques for efficient handling of large-scale graphs.
* **GraphIt-DSL:** [link](https://github.com/GraphIt-DSL/graphit) Integration of GraphIt-DSL segment graphs to improve locality.

## GAP Benchmarks

This project contains a collection of Graph Analytics for Performance [(GAPBS)](https://github.com/sbeamer/gapbs) benchmarks implemented in C++. The benchmarks are designed to exercise the performance of graph algorithms on a CPU. 

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

Before you begin, ensure you have the following installed on your system, [(section)](#installing-prerequisites):
- **Ubuntu**: All testing has been done on Ubuntu `22.04+` Operating System.
- **GCC**: The GNU Compiler Collection, specifically `g++9` which supports C++11 or later.
- **Make**: The build utility to automate the compilation.
- **OpenMP**: Support for parallel programming in C++.
### Compile with RabbitOrder
   * Go to Makefile <[line:8](https://github.com/atmughrabi/GraphBrew/blob/main/Makefile#L8)> make sure `RABBIT_ENABLE = 1`
```bash
<OR>
make RABBIT_ENABLE=1
```
   * **Boost** C++ library (1.58.0).
   * **libnuma** (2.0.9).
   * **libtcmalloc\_minimal** in google-perftools (2.1).

# GraphBrew Experiment Configuration

Graphbrew can explore the impact of graph reordering techniques on the performance of various graph algorithms. The configuration for these experiments is specified in the `scripts/<experiment-name>/run_experiment.py` file.  

### Experiment Execution

1. **Test Experiment:**
   * Use the `make exp-test` command. This will:
     * Execute the experiments as defined in the python script [(`scripts/test/run_experiment.py`)](./scripts/test/run_experiment.py), with smaller graphs.
     * Generate results (e.g., reorder time for each graph, average time for each algorithm) in the `bench/results` folder.
     * `make clean-results` will back up current results into `bench/backup` and delete `bench/results` for a new run.
     * Use this config for functional testing, to make sure all libraries are installed and GraphBrew is running -- **not for performance evaluation**.

## GraphBrew Experiment

Point the downloaded graphs into any directory by updating `BASE_DIR = "00_GraphDatasets/GBREW"` in [(`scripts/<experiment-name>/run_experiment.py`)](./scripts/test/run_experiment.py).

### Download Graphs

* **Twitter (TWTR | 31.4GB | .mtx):** A representation of the Twitter social network. [link](https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-twitter.tar.gz)
* **Road network (RD | 628.4MB | .mtx):** A road network from a specific geographic region. [link](https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-road.tar.gz)
* **LiveJournal (SLJ1 | 1GB | .mtx):** A social network from LiveJournal users. [link](https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz)
* **Patents (CPAT | 261.7MB | .mtx):** A citation network of patents. [link](https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz)
* **Orkut (CORKT | 1.8GB | .mtx):** A social network from Orkut. [link](https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Orkut.tar.gz)
* **Pokec (SPKC | 424MB | .mtx):** A social network from Pokec. [link](https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Pokec.tar.gz)
* **Web graph (WEB01 | 18.5GB | .mtx):** A crawl of a portion of the World Wide Web from 2001. [link](https://suitesparse-collection-website.herokuapp.com/MM/LAW/webbase-2001.tar.gz)
* **Google Plus (GPLUS | 7.3GB | .wel):** A social network from Google Plus (assume timestamps weights and filter out). [link](https://drive.google.com/file/d/1HF8Q2N_hxsaQ26MarKYxZEQhqI66qAxV/view?usp=sharing)
* **Wikipedia Links (WIKLE | 6.7GB | .el):** Links between Wikipedia pages in English. [link](http://konect.cc/files/download.tsv.wikipedia_link_en.tar.bz2)

### Rename the Graph

Rename each graph into the specified extension in the aforementioned list. For example, for the Twitter graph (Symbol: TWTR), rename the graph to `graph.mtx` and place it in the following directory structure:

```
00_GraphDatasets/GBREW/TWTR/graph.mtx
```

Follow the same pattern for each graph, using their respective symbols and extensions. Here is the directory structure for each graph:

* **Twitter:** `00_GraphDatasets/GBREW/TWTR/graph.mtx`
* **Road network:** `00_GraphDatasets/GBREW/RD/graph.mtx`
* **LiveJournal:** `00_GraphDatasets/GBREW/SLJ1/graph.mtx`
* **Patents:** `00_GraphDatasets/GBREW/CPAT/graph.mtx`
* **Orkut:** `00_GraphDatasets/GBREW/CORKT/graph.mtx`
* **Pokec:** `00_GraphDatasets/GBREW/SPKC/graph.mtx`
* **Web graph:** `00_GraphDatasets/GBREW/WEB01/graph.mtx`
* **Google Plus:** `00_GraphDatasets/GBREW/GPLUS/graph.wel`
* **Wikipedia Links:** `00_GraphDatasets/GBREW/WIKLE/graph.el`

Ensure that you download the graphs, extract them if necessary, and place them in the corresponding directories with the correct file names and extensions.

### Run GAPBS with GraphBrew:

* Use the `make exp-brew` command. This will:
  * Execute the experiments as defined in the configuration file [(`scripts/brew/run_experiment.py`)](scripts/brew/run_experiment.py).
  * Generate results (e.g., speedup graphs, overhead measurements) in the `bench/results` folder.

# GraphBrew Standalone

## Usage

### Example Usage
   * To compile, run, and then clean up the betweenness centrality benchmark:
```bash
make all
make run-bc
make clean
```
### Compiling a single Benchmarks
   * To build all benchmarks:
```bash
make bc
```
### Compiling the Benchmarks
   * To build all benchmarks:
```bash
make all
```

### Running a single Benchmarks
   * To run a specific benchmark, use:
```bash
make run-<benchmark_name>
```
   * Where `<benchmark_name>` can be `bc`, `bfs`, `converter`, etc.
```bash
make run-bfs
```
### Parameters
All parameters [(section)](#graphbrew-parameters) can be passed through the Make command via:
   * `RUN_PARAMS='-n1 -o11'`, for controlling aspects of the algorithm and reordering.
   * `GRAPH_BENCH ='-f ./test/graphs/4.el'`,`GRAPH_BENCH ='-g 4'`, for controlling the graph path, or kron/random generation.
All parameters [(section)](#graphbrew-parameters) can be passed through the binary command via:
   * `./bench/bin/<benchmark_name> -f ./test/graphs/4.el -n1 -o11`
   * `./bench/bin/<benchmark_name> -g 4 -n1 -o11`

### Relabeling the graph
   * `converter` is used to convert graphs and apply new labeling to them.
   * Please check converter parameters and pass them to `RUN_PARAMS='-p ./graph_8.mtx -o 8'`.
```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-p ./graph_8.mtx -o 8' 
<OR>
./bench/bin/converter -f ./graph.<mtx|el|sg> -p ./graph_8.mtx -o 8
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
   * To clean up all compiled including results (backed up automatically in `bench/backup`) files:
```bash
make clean-all
```

### Help
   * To display help for a specific benchmark or for general usage:
```bash
make help-<benchmark_name>
make help
```

## Generating Reordered Graphs

### Overview

Use the `make run-converter` command to generate reordered graphs from input graph files. The converter supports various output formats, including serialized graphs, edge lists, Matrix Market exchange format, Ligra adjacency graph format, and reordered labels.

### Command-Line Options

The `CLConvert` class provides several command-line options for generating different output formats. Here is a summary of the options:

- `-b file`: Output serialized graph to file (`.sg`).
- `-e file`: Output edge list to file (`.el`).
- `-p file`: Output Matrix Market exchange format to file (`.mtx`).
- `-y file`: Output in Ligra adjacency graph format to file (`.ligra`).
- `-w file`: Make output weighted (`.wel` | `.wsg`| `.wligra`).
- `-x file`: Output new reordered labels to file list (`.so`).
- `-q file`: Output new reordered labels to file serialized (`.lo`).
- `-o order`: Apply reordering strategy, optionally with a parameter (e.g., `-o 3`, `-o 2`, `-o 14:mapping.label`).

### Example Usage

#### Step 1: Prepare the Input Graph Files

Make sure you have the input graph files (`graph_1.<mtx|el|sg>`) while specifying their paths correctly.

#### Step 2: Run the Converter

Use the `make run-converter` command with the appropriate `GRAPH_BENCH` and `RUN_PARAMS` values to generate the reordered graphs. Here is an example command:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-p ./graph_8.mtx -o 8' 
```

#### Step 3: Specify Output Formats

You can specify multiple output formats by combining the command-line options. Here is an example that generates multiple output formats:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-b graph.sg -e graph.el -p graph.mtx -y graph.ligra -x labels.so -q labels.lo'
```

#### Step 4: Apply Reordering Strategy

To apply a reordering strategy on the newly generated graph, use the `-o` option. For example:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-p ./graph_3_2_14.mtx -o 3 -o 2 -o 14:mapping.<lo|so>'
```

### Combining Multiple Output Formats and Reordering

You can generate multiple output formats and apply reordering in a single command. Here is an example:

```bash
make run-converter GRAPH_BENCH='-f ./graph.<mtx|el|sg>' RUN_PARAMS='-b graph_3.sg -e graph_3.el -p graph_3.mtx -y graph_3.ligra -x labels_3.so -q labels_3.lo -o 3'
```


## Graph Loading

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

The GraphBrew loading infrastructure understands the following formats for reordering labels:
+ `-o 14:mapping.lo` loads new reodering labels from file `mapping.<lo|so>` is also supported
+ `.so` reordered serialized labels list (.so) (use `converter` to make), _node_id_ per line as _node_label_ 
+ `.lo` reordered plain-text labels list (.lo) (use `converter` to make), _node_id_ per line as _node_label_ 

## GraphBrew Parameters

All parameters can be passed through the make command via:
   * Reorder the graph, orders can be layered.
   * Segment the graph for scalability, requires modifying the algorithm to iterate through segments.
   * `RUN_PARAMS='-n1 -o11'`, for controlling aspects of the algorithm and reordering.
   * `GRAPH_BENCH ='-f ./test/graphs/4.el'`,`GRAPH_BENCH ='-g 4'`, for controlling the graph path, or kron/random generation.

### GAP Parameters (PageRank example)
```bash
make pr
--------------------------------------------------------------------------------
pagerank
 -h           : print this help message                                         
 -f <file>    : load graph from file                                            
 -s           : symmetrize input edge list                               [false]
 -g <scale>   : generate 2^scale kronecker graph                                
 -u <scale>   : generate 2^scale uniform-random graph                           
 -k <degree>  : average degree for synthetic graph                          [16]
 -m           : reduces memory usage during graph building               [false]
 -o <order>   : apply reordering strategy, optionally with a parameter 
               [example]-o 3 -o 2 -r 14:mapping.<lo|so>               [optional]
 -z <indegree>: use indegree for ordering [Degree Based Orderings]       [false]
 -j <segments>: number of segments for the graph 
               [type:n:m] <0:GRAPHIT/Cagra> <1:TRUST>                    [0:1:1]
 -a           : output analysis of last run                              [false]
 -n <n>       : perform n trials                                            [16]
 -r <node>    : start from node r                                         [rand]
 -v           : verify the output of each run                            [false]
 -l           : log performance within each trial                        [false]
 -i <i>       : perform at most i iterations                                [20]
 -t <t>       : use tolerance t                                       [0.000100]
--------------------------------------------------------------------------------
```
### Reorder Parameters
```bash
--------------------------------------------------------------------------------
-o <order>   : Apply reordering strategy, optionally layer ordering 
               [example]-o 3 -o 2 -o 14:mapping.<lo|so>               [optional]

-j <segments>: number of segments for the graph 
               [type:n:m] <0:GRAPHIT/Cagra> <1:TRUST>   [0:1:1]

-z <indegree>: use indegree for ordering [Degree Based Orderings]        [false]
--------------------------------------------------------------------------------
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
  - GraphBrewOrder(13): Leiden community clustering with rabbit order refinement..
  - MAP           (14): Requires a file format for reordering. Use the -r 10:filename.label option.
```
### Converter Parameters (Generate Optimized Graphs)
```bash
make help-converter
--------------------------------------------------------------------------------
converter
 -h          : print this help message                                         
 -f <file>   : load graph from file                                            
 -s          : symmetrize input edge list                                [false]
 -g <scale>  : generate 2^scale kronecker graph                                
 -u <scale>  : generate 2^scale uniform-random graph                           
 -k <degree> : average degree for synthetic graph                           [16]
 -m          : reduces memory usage during graph building                [false]
 -o <order>  : Apply reordering strategy, optionally layer ordering 
               [example]-o 3 -o 2 -o 14:mapping.<lo|so>               [optional]
 -z <indegree>: use indegree for ordering [Degree Based Orderings]       [false]
 -j <segments>: number of segments for the graph 
               [type:n:m] <0:GRAPHIT/Cagra> <1:TRUST>                    [0:1:1]
 --------------------------------------------------------------------------------
 -b <file>   : output serialized graph to file (.sg)                           
 -e <file>   : output edge list to file (.el)
 -p <file>   : output matrix market exchange format to file (.mtx)
 -y <file>   : output in Ligra adjacency graph format to file (.ligra)                                      
 -w <file>   : make output weighted (.wel|.wsg)                                
 -x <file>   : output new reordered labels to file list (.so)                  
 -q <file>   : output new reordered labels to file serialized (.lo)    
 --------------------------------------------------------------------------------
```

### Makefile Flow
```bash
available Make commands:
  all            - Builds all targets including GAP benchmarks (CPU)
  run-%          - Runs the specified GAP benchmark (bc bfs cc cc_sv pr pr_spmv sssp tc)
  help-%         - Print the specified Help (bc bfs cc cc_sv pr pr_spmv sssp tc)
  clean          - Removes all build artifacts
  help           - Displays this help message
 --------------------------------------------------------------------------------
Example Usage:
  make all - Compile the program.
  make clean - Clean build files.
  ./bench/bin/pr -g 15 -n 1 -o 14:mapping.lo - Execute with MAP reordering using 'mapping.<lo|so> '.

```

## Modifying the Makefile

### Compiler Setup
- **`CC`**: The C compiler to be used, checks for `gcc-9` first, if not found, falls back to `gcc`.
- **`CXX`**: The C++ compiler to be used, checks for `g++-9` first, if not found, falls back to `g++`.

### Directory Structure
- **`BIN_DIR`**: Directory for compiled binaries.
- **`LIB_DIR`**: Library directory.
- **`SRC_DIR`**: Source files directory.
- **`INC_DIR`**: Include directory for header files.
- **`OBJ_DIR`**: Object files directory.
- **`SCRIPT_DIR`**: Scripts used for operations like graph processing.
- **`BENCH_DIR`**: Benchmark directory.
- **`CONFIG_DIR`**: Configuration files for scripts and full expriments in congif.json format.
- **`RES_DIR`**: Directory where results are stored.
- **`BACKUP_DIR`**: Directory for backups of results `make clean-results`. backsup results then cleans them.

### Include Directories
- **`INCLUDE_<LIBRARY>`**: Each variable specifies the path to header files for various libraries or modules.
- **`INCLUDE_BOOST`**: Specifies the directory for Boost library headers.

### Compiler and Linker Flags
- **`CXXFLAGS`**: Compiler flags for C++ files, combining flags for different libraries and conditions.
- **`LDLIBS`**: Linker flags specifying libraries to link against.
- **`CXXFLAGS_<LIBRARY>`**: Specific compiler flags for various libraries/modules.
- **`LDLIBS_<LIBRARY>`**: Specific linker flags for various libraries/modules.

### Runtime and Execution
- **`PARALLEL`**: Number of parallel threads.
- **`FLUSH_CACHE`**: Whether or not to flush cache before running benchmarks.
- **`GRAPH_BENCH`**: Command line arguments for specifying graph benchmarks.
- **`RUN_PARAMS`**: General command line parameters for running benchmarks.

## Makefile Targets

### Primary Targets
- **`all`**: Compiles all benchmarks.
- **`clean`**: Removes binaries and intermediate files.
- **`clean-all`**: Removes binaries, results, and intermediate files.
- **`clean-results`**: Backs up and then cleans the results directory.
- **`exp-%`**: Runs a specific experiment by replacing `%` with the experiment.json name. E.g., `test.json`.
- **`run-%`**: Runs a specific benchmark by replacing `%` with the benchmark name. E.g., `run-bfs`.
- **`run-%-gdb`**: Runs a specific benchmark under GDB.
- **`run-%-mem`**: Runs a specific benchmark under Valgrind for memory leak checks.
- **`run-all`**: Runs all benchmarks.
- **`graph-%`**: Downloads necessary graphs for a specific benchmark at `CONFIG_DIR`.
- **`help`**: Displays help for all benchmarks.

### Compilation Rules
- **`$(BIN_DIR)/%`**: Compiles a `.cc` source file into a binary, taking dependencies into account.

### Directory Setup
- **`$(BIN_DIR)`**: Ensures the binary directory and required sub directories exist.

### Cleanup
- **`clean`**: Removes binaries and intermediate files.
- **`clean-all`**: Removes binaries, results, and intermediate files.

### Help
- **`help`**: Provides a generic help message about available commands.
- **`help-%`**: Provides specific help for each benchmark command, detailing reordering algorithms and usage examples.

## Project Structure
- `bench/bin`: Executable is placed here.
- `bench/lib`: Library files can be stored here (not used by default).
- `bench/src`: Source code files (*.cc) for the benchmarks.
- `bench/obj`    : Object files are stored here (directory creation is handled but not used by default).
- `bench/include`: Header files for the benchmarks and various include files for libraries such as GAPBS, RABBIT, etc.

## Project Experiments
- `bench/results`: experiment results from running `exp-%`.
- `bench/backups`: experiment backup results from running `clean-all` or `clean-results`.

## Installing Prerequisites (General)

* These tools are available on most Unix-like operating systems and can be installed via your package manager. For example, on Ubuntu, you can install them using:

```bash
sudo apt-get update
sudo apt-get install g++ make libomp-dev
```

## Installing Prerequisites (RabbitOrder)
* Go to Makefile <line:8> make sure `RABBIT_ENABLE = 1`
```bash
<OR>
make RABBIT_ENABLE=1
```

* These made optional if you don't need Rabbit Order or running on machines where you can't install these libraries
```bash
sudo apt-get install libgoogle-perftools-dev
sudo apt-get install python3 python3-pip python3-venv
```
### Installing Boost 1.58.0

1. First, navigate to your project directory


   * Download the desired Boost version `boost_1_58_0`:
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
./bootstrap.sh --prefix=/opt/boost_1_58_0 --with-python=python2.7 
```
   * Compile and install Boost using all available cores to speed up the process:
```bash
sudo ./b2 --with=all -j $cpuCores install
```

3. **Verify the Installation**

   
   * After installation, verify that Boost has been installed correctly by checking the installed version:
```bash
cat /opt/boost_1_58_0/include/boost/version.hpp | grep "BOOST_LIB_VERSION"
```
   * The output should display the version of Boost you installed, like so:
```bash
//  BOOST_LIB_VERSION must be defined to be the same as BOOST_VERSION
#define BOOST_LIB_VERSION "1_58"
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
