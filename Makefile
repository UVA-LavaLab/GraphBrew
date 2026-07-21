# =========================================================
# Compiler and Directories Setup
# =========================================================
# CXX = g++
# Attempt to find gcc-9, else use default gcc
CC  = $(shell which gcc-9 || which gcc)
CXX = $(shell which g++-9 || which g++)
RABBIT_ENABLE ?= 1
# =========================================================
PYTHON=@python3
PIP=@pip
# =========================================================
SCRIPT_DIR  = scripts
# Lint
LINT_INCLUDES = $(PYTHON) $(SCRIPT_DIR)/lib/tools/check_includes.py
BENCH_DIR   = bench
# =========================================================
BIN_DIR = $(BENCH_DIR)/bin
LIB_DIR = $(BENCH_DIR)/lib
SRC_DIR = $(BENCH_DIR)/src
INC_DIR = $(BENCH_DIR)/include
OBJ_DIR = $(BENCH_DIR)/obj
TEST_SRC_DIR = $(BENCH_DIR)/tests
TEST_BIN_DIR = $(BENCH_DIR)/test_bin

# =========================================================
# Include paths
INCLUDE_GAPBS     = $(INC_DIR)/external/gapbs
INCLUDE_GRAPHBREW = $(INC_DIR)/graphbrew
INCLUDE_EXTERNAL  = $(INC_DIR)/external
INCLUDE_CACHE     = $(INC_DIR)/cache_sim
# =========================================================
INCLUDE_BOOST  = /opt/boost_1_58_0/include  
# =========================================================
DEP_GAPBS     = $(wildcard $(INCLUDE_GAPBS)/*.h)
DEP_GRAPHBREW = $(wildcard $(INCLUDE_GRAPHBREW)/*.h) $(wildcard $(INCLUDE_GRAPHBREW)/algorithms/*.h) $(wildcard $(INCLUDE_GRAPHBREW)/edge/*.h) $(wildcard $(INCLUDE_GRAPHBREW)/gas/*.h) $(wildcard $(INCLUDE_GRAPHBREW)/reorder/*.h) $(wildcard $(INCLUDE_GRAPHBREW)/partition/*.h)
DEP_RABBIT = $(wildcard $(INCLUDE_EXTERNAL)/rabbit/*.hpp)
DEP_GORDER = $(wildcard $(INCLUDE_EXTERNAL)/gorder/*.h)
DEP_CORDER = $(wildcard $(INCLUDE_EXTERNAL)/corder/*.h)
DEP_LEIDEN = $(wildcard $(INCLUDE_EXTERNAL)/leiden/*.hxx)
# =========================================================

# =========================================================
#     CLI COMMANDS                           
# =========================================================
COMPILED_FILE = $(@F)
CREATE        = [$(BLUE)create!$(NC)]
SUCCESS       = [$(GREEN)success!$(NC)]
FAIL          = [$(RED)failure!$(NC)]
CREATE_MSG   = echo  "$(CREATE) $(COMPILED_FILE)"
SUCCESS_MSG   = echo  "$(SUCCESS) $(COMPILED_FILE)"
FAIL_MSG      = echo  "$(FAIL) $(COMPILED_FILE)"
EXIT_STATUS   = &&  $(SUCCESS_MSG) || ( $(FAIL_MSG) ; exit 1; )
CREATE_STATUS  = &&  $(CREATE_MSG) || ( $(FAIL_MSG) ; exit 1; )
# =========================================================
# Color coded messages                      
# =========================================================
YELLOW  =\033[0;33m
GREEN   =\033[0;32m
BLUE    =\033[0;34m
RED     =\033[0;31m
NC      =\033[0m
# =========================================================

# =========================================================
# Compiler Flags
# =========================================================
CXXFLAGS_GAP    = -std=c++17 -O3 -Wall -fopenmp -g -DNDEBUG
CXXFLAGS_RABBIT = -mcx16 -Wno-deprecated-declarations -Wno-parentheses -Wno-unused-local-typedefs
CXXFLAGS_GORDER = -m64 -march=native 
CXXFLAGS_GORDER += -DRelease -DGCC
CXXFLAGS_LEIDEN = -DTYPE=float -DMAX_THREADS=$(PARALLEL) -DREPEAT_METHOD=1
# =========================================================
LDLIBS_RABBIT   += -ltcmalloc_minimal -lnuma
# =========================================================
# Default library path for Boost libraries
BOOST_LIB_DIR := /opt/boost_1_58_0/lib
# Verify if the Boost library directory exists, otherwise use the fallback directory
ifeq ($(wildcard $(BOOST_LIB_DIR)/*),)
    BOOST_LIB_DIR := /usr/local/lib
endif
LDLIBS_BOOST    += -L$(BOOST_LIB_DIR)
# =========================================================
CXXFLAGS = $(CXXFLAGS_GAP) $(CXXFLAGS_GORDER) $(CXXFLAGS_LEIDEN) $(CXXFLAGS_RABBIT)
LDLIBS  = 
# =========================================================
INCLUDES = -I$(INCLUDE_BOOST) -I$(INCLUDE_GAPBS) -I$(INCLUDE_GRAPHBREW) -I$(INCLUDE_EXTERNAL) -I$(INC_DIR)
# =========================================================
# Optional RABBIT includes
ifeq ($(RABBIT_ENABLE), 1)
CXXFLAGS += -DRABBIT_ENABLE 
LDLIBS += $(LDLIBS_BOOST) $(LDLIBS_RABBIT)
endif
# =========================================================
# Targets
# =========================================================
KERNELS = bc bfs bfs_p cc cc_sv pr pr_spmv sssp tc tc_p
KERNELS_BIN = $(addprefix $(BIN_DIR)/,$(KERNELS))
DENSE_EDGE_KERNELS = cc_edge cc_sv_edge pr_edge pr_spmv_edge
DENSE_EDGE_KERNELS_BIN = $(addprefix $(BIN_DIR)/,$(DENSE_EDGE_KERNELS))
FRONTIER_EDGE_KERNELS = bfs_edge sssp_edge
FRONTIER_EDGE_KERNELS_BIN = $(addprefix $(BIN_DIR)/,$(FRONTIER_EDGE_KERNELS))
IRREGULAR_EDGE_KERNELS = bc_edge tc_edge
IRREGULAR_EDGE_KERNELS_BIN = $(addprefix $(BIN_DIR)/,$(IRREGULAR_EDGE_KERNELS))
EDGE_KERNELS = $(DENSE_EDGE_KERNELS) $(FRONTIER_EDGE_KERNELS) $(IRREGULAR_EDGE_KERNELS)
EDGE_KERNELS_BIN = $(addprefix $(BIN_DIR)/,$(EDGE_KERNELS))
GAS_KERNELS = cc_gas pr_gas sssp_gas
GAS_KERNELS_BIN = $(addprefix $(BIN_DIR)/,$(GAS_KERNELS))
SUITE = $(KERNELS_BIN) $(DENSE_EDGE_KERNELS_BIN) $(FRONTIER_EDGE_KERNELS_BIN) $(IRREGULAR_EDGE_KERNELS_BIN) $(GAS_KERNELS_BIN) $(BIN_DIR)/converter $(BIN_DIR)/graph_shard_export \
	$(BIN_DIR)/ownership_analysis $(BIN_DIR)/edge_view_benchmark
UNIT_TESTS = test_graph_partition test_partition_traffic test_shard_manifest \
	test_shard_stream test_ownership_analysis test_edge_primitives \
	test_gas_executor
UNIT_TESTS_BIN = $(addprefix $(TEST_BIN_DIR)/,$(UNIT_TESTS))
# =========================================================

.PHONY: $(KERNELS) $(DENSE_EDGE_KERNELS) $(FRONTIER_EDGE_KERNELS) $(IRREGULAR_EDGE_KERNELS) $(GAS_KERNELS) converter edge_view_benchmark edge-all edge-dense edge-frontier edge-irregular gas-all all check-partition check-edge-contracts check-edge-contract-profiles check-edge-primitives check-edge-dense check-edge-frontier check-edge-irregular check-edge check-gas-runtime check-gas check-edge-gas-repeatability check-edge-gas run-% exp-% graph-% help-% install-py-deps help clean clean-all clean-results run-%-gdb run-%-sweep $(BIN_DIR)/% scrub-all
ownership_analysis: $(BIN_DIR)/ownership_analysis
edge_view_benchmark: $(BIN_DIR)/edge_view_benchmark
edge-all: $(EDGE_KERNELS_BIN)
edge-dense: $(DENSE_EDGE_KERNELS_BIN)
edge-frontier: $(FRONTIER_EDGE_KERNELS_BIN)
edge-irregular: $(IRREGULAR_EDGE_KERNELS_BIN)
gas-all: $(GAS_KERNELS_BIN)
all: $(SUITE)

check-edge-contracts:
	$(PYTHON) scripts/test/check_edge_gas_contracts.py

check-edge-contract-profiles: $(addprefix $(BIN_DIR)/,$(filter-out bfs_p tc_p,$(KERNELS)))
	$(PYTHON) scripts/test/run_edge_gas_contract_profiles.py

check-edge-primitives: $(TEST_BIN_DIR)/test_edge_primitives $(BIN_DIR)/edge_view_benchmark
	@for threads in 1 2 4 8; do \
		OMP_NUM_THREADS=$$threads $(TEST_BIN_DIR)/test_edge_primitives; \
	done
	@OMP_NUM_THREADS=4 $(BIN_DIR)/edge_view_benchmark -g 8 >/dev/null

check-edge-dense: $(DENSE_EDGE_KERNELS_BIN) $(addprefix $(BIN_DIR)/,cc cc_sv pr pr_spmv)
	$(PYTHON) scripts/test/run_edge_dense_profiles.py

check-edge-frontier: $(FRONTIER_EDGE_KERNELS_BIN) $(addprefix $(BIN_DIR)/,bfs sssp)
	$(PYTHON) scripts/test/run_edge_frontier_profiles.py

check-edge-irregular: $(IRREGULAR_EDGE_KERNELS_BIN) $(addprefix $(BIN_DIR)/,bc tc)
	$(PYTHON) scripts/test/run_edge_irregular_profiles.py

check-edge: check-edge-contracts check-edge-contract-profiles check-edge-primitives check-edge-dense check-edge-frontier check-edge-irregular

check-gas-runtime: $(TEST_BIN_DIR)/test_gas_executor
	@for threads in 1 2 4 8; do \
		OMP_NUM_THREADS=$$threads $(TEST_BIN_DIR)/test_gas_executor; \
	done

check-gas: check-gas-runtime $(GAS_KERNELS_BIN) $(addprefix $(BIN_DIR)/,cc pr sssp)
	$(PYTHON) scripts/test/run_gas_profiles.py

check-edge-gas-repeatability: $(EDGE_KERNELS_BIN) $(GAS_KERNELS_BIN)
	$(PYTHON) scripts/test/run_edge_gas_repeatability.py

check-edge-gas: check-edge check-gas check-edge-gas-repeatability

check-partition: $(UNIT_TESTS_BIN) $(BIN_DIR)/bfs_p $(BIN_DIR)/converter $(BIN_DIR)/graph_shard_export
	@for test in $(UNIT_TESTS_BIN); do $$test; done
	$(PYTHON) scripts/test/check_partition_runtime_traffic.py \
		--bfs $(BIN_DIR)/bfs_p
	@for partitions in 1 2 4 16; do \
		output="$$(OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -g 10 -n 1 -r 0 -v -P $$partitions -B total)"; \
		if ! echo "$$output" | grep -q "Verification: *PASS"; then \
			echo " $(FAIL) Partitioned BFS P=$$partitions"; \
			echo "$$output"; \
			exit 1; \
		fi; \
	done; \
	echo " $(PASS) Partitioned BFS P=1/2/4/16"
	@output="$$(OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -f scripts/test/data/tiny.el -n 1 -r 0 -v -P 4 -B out)"; \
		if ! echo "$$output" | grep -q "Verification: *PASS"; then \
			echo " $(FAIL) Partitioned BFS directed graph"; \
			echo "$$output"; \
			exit 1; \
		fi; \
		echo " $(PASS) Partitioned BFS directed graph"
	@output="$$(OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -f scripts/test/data/tiny.el -s -n 1 -r 0 -v -P 4 -B total)"; \
		if ! echo "$$output" | grep -q "Verification: *PASS"; then \
			echo " $(FAIL) Partitioned BFS symmetric graph"; \
			echo "$$output"; \
			exit 1; \
		fi; \
		echo " $(PASS) Partitioned BFS symmetric graph"
	@one_root="$$(mktemp -d)"; \
		many_root="$$(mktemp -d)"; \
		trap 'rm -rf "$$one_root" "$$many_root"' EXIT; \
		one="$$(OMP_NUM_THREADS=1 $(BIN_DIR)/bfs_p -g 10 -n 1 -r 0 -v -o 11:bnf -P 4 -B total -E "$$one_root/package" | sed -n -e 's/^Partition fingerprints: //p' -e 's/^BFS source diagnostics: //p')"; \
		many="$$(OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -g 10 -n 1 -r 0 -v -o 11:bnf -P 4 -B total -E "$$many_root/package" | sed -n -e 's/^Partition fingerprints: //p' -e 's/^BFS source diagnostics: //p')"; \
		if test -z "$$one" || test "$$one" != "$$many"; then \
			echo " $(FAIL) Reordered partition/BFS fingerprints differ across thread counts"; \
			echo " OMP=1: $$one"; \
			echo " OMP=4: $$many"; \
			exit 1; \
		fi; \
		if ! diff -qr "$$one_root/package" "$$many_root/package" >/dev/null; then \
			echo " $(FAIL) graph.shard.v1 package differs across thread counts"; \
			exit 1; \
		fi; \
		echo " $(PASS) RCM:bnf fingerprints/package OMP=1/4"
	@for policy in \
		"comm_cut_min:12:leiden:compose:comm_cut_min:intra_hubsort" \
		"sg_hilbert:12:leiden:compose:sg_hilbert:comm_identity:intra_hubsort" \
		"intra_hub2:12:leiden:compose:comm_degree_desc:intra_hub2" \
		"intra_rcmpp:12:leiden:compose:comm_degree_desc:intra_rcmpp" \
		"leiden_hubsort:12:leiden:compose:comm_degree_desc:intra_hubsort"; do \
		name="$${policy%%:*}"; option="$${policy#*:}"; \
		one="$$(OMP_NUM_THREADS=1 $(BIN_DIR)/bfs_p -g 10 -n 1 -r 0 -v -o "$$option" -P 4 -B total | sed -n -e 's/^Partition fingerprints: //p' -e 's/^BFS source diagnostics: //p')"; \
		many="$$(OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -g 10 -n 1 -r 0 -v -o "$$option" -P 4 -B total | sed -n -e 's/^Partition fingerprints: //p' -e 's/^BFS source diagnostics: //p')"; \
		if test -z "$$one" || test "$$one" != "$$many"; then \
			echo " $(FAIL) $$name fingerprints differ across thread counts"; \
			echo " OMP=1: $$one"; \
			echo " OMP=4: $$many"; \
			exit 1; \
		fi; \
	done; \
	echo " $(PASS) community fingerprints OMP=1/4"
	@serial="$$(OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -g 8 -n 1 -r 0 -o 12:leiden:compose -P 2 -B total)"; \
		parallel="$$(OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -g 8 -n 1 -r 0 -o 12:leiden:compose:cd_parallel -P 2 -B total)"; \
		if ! echo "$$serial" | grep -q "community-detection=serial" || \
		   ! echo "$$serial" | grep -q "ordering-threads=4" || \
		   ! echo "$$parallel" | grep -q "community-detection=parallel"; then \
			echo " $(FAIL) community thread-mode contract"; \
			exit 1; \
		fi; \
		echo " $(PASS) community serial/parallel thread modes"
	@sg_root="$$(mktemp -d)"; \
		legacy_root="$$(mktemp -d)"; \
		stream_root="$$(mktemp -d)"; \
		trap 'rm -rf "$$sg_root" "$$legacy_root" "$$stream_root"' EXIT; \
		$(BIN_DIR)/converter -g 10 -b "$$sg_root/graph.sg" >/dev/null 2>&1; \
		OMP_NUM_THREADS=4 $(BIN_DIR)/bfs_p -f "$$sg_root/graph.sg" -n 1 -r 0 -P 3 -B total -E "$$legacy_root/package" >/dev/null 2>&1; \
		$(BIN_DIR)/graph_shard_export -f "$$sg_root/graph.sg" -P 3 -B total -E "$$stream_root/package" >/dev/null; \
		if ! diff -qr "$$legacy_root/package" "$$stream_root/package" >/dev/null; then \
			echo " $(FAIL) streaming graph.shard.v1 package differs from legacy"; \
			exit 1; \
		fi; \
		echo " $(PASS) streaming .sg->graph.shard.v1 matches legacy"

# =========================================================
# Runtime Flags OMP_NUM_THREADS
# =========================================================
PARALLEL=$(shell grep -c ^processor /proc/cpuinfo)
FLUSH_CACHE=1
# =========================================================

# =========================================================
# Running Benchmarks
# =========================================================
# GRAPH_BENCH = -f ./scripts/test/graphs/tiny/tiny.el
GRAPH_BENCH = -g 12 
RUN_PARAMS =  -o5 -n 2 -l
# =========================================================
run-%: $(BIN_DIR)/%
	@if [ "$(FLUSH_CACHE)" = "1" ]; then \
		echo "Attempting to mitigate cache effects by busy-looping..."; \
		dd if=/dev/zero of=/dev/null bs=1M count=1024; \
	fi; \
	OMP_NUM_THREADS=$(PARALLEL) ./$<  $(GRAPH_BENCH) $(RUN_PARAMS) $(EXIT_STATUS)

run-%-gdb: $(BIN_DIR)/%
	@OMP_NUM_THREADS=$(PARALLEL) gdb -ex=r --args ./$< $(GRAPH_BENCH) $(RUN_PARAMS)

run-%-mem: $(BIN_DIR)/%
	@OMP_NUM_THREADS=$(PARALLEL) valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes -v ./$< $(GRAPH_BENCH) $(RUN_PARAMS)

run-all: $(addprefix run-, $(KERNELS))

# Define a rule that sweeps through -o 1 to 7
run-%-sweep: $(BIN_DIR)/%
	@for o in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		echo "========================================================="; \
		if [ "$(FLUSH_CACHE)" = "1" ]; then \
		    echo "Attempting to mitigate cache effects by busy-looping..."; \
		    dd if=/dev/zero of=/dev/null bs=1M count=1024; \
		fi; \
		OMP_NUM_THREADS=$(PARALLEL) ./$(BIN_DIR)/$* $(GRAPH_BENCH) -s -n 1 -o $$o; \
	done
	
# =========================================================
# Rule to install Python dependencies
install-py-deps: ./$(SCRIPT_DIR)/requirements.txt
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r ./$(SCRIPT_DIR)/requirements.txt


# =========================================================
# Compilation Rules
# =========================================================
$(BIN_DIR)/%: $(SRC_DIR)/%.cc $(DEP_GAPBS) $(DEP_GRAPHBREW) $(DEP_RABBIT) $(DEP_GORDER) $(DEP_CORDER) $(DEP_LEIDEN) | $(BIN_DIR)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDLIBS) -o $@ $(EXIT_STATUS)

# The streaming exporter only depends on the gapbs reader and the graphbrew
# partition headers, so it builds with the lightweight GAP flags and links no
# reorder/boost/tcmalloc dependencies.
$(BIN_DIR)/graph_shard_export: $(SRC_DIR)/graph_shard_export.cc $(DEP_GAPBS) $(DEP_GRAPHBREW) | $(BIN_DIR)
	@$(CXX) $(CXXFLAGS_GAP) $(INCLUDES) $< -o $@ $(EXIT_STATUS)

$(TEST_BIN_DIR)/%: $(TEST_SRC_DIR)/%.cc $(DEP_GAPBS) $(DEP_GRAPHBREW) | $(TEST_BIN_DIR)
	@$(CXX) $(CXXFLAGS_GAP) $(INCLUDES) $< -o $@ $(EXIT_STATUS)

# =========================================================
# Directory Setup
# =========================================================
$(BIN_DIR):
	@mkdir -p $@ $(CREATE_STATUS)

$(TEST_BIN_DIR):
	@mkdir -p $@ $(CREATE_STATUS)

# =========================================================
# Cleanup
# =========================================================
clean:
	@rm -rf $(BIN_DIR) $(BIN_SIM_DIR) $(TEST_BIN_DIR) $(EXIT_STATUS)

clean-all: clean

# =========================================================
# Testing
# =========================================================
scrub-all:
	@rm -rf $(BIN_DIR) $(BIN_SIM_DIR) $(TEST_BIN_DIR) 00_* $(EXIT_STATUS)

# =========================================================
# Cache Simulation Builds
# =========================================================
SRC_SIM_DIR = $(BENCH_DIR)/src_sim
BIN_SIM_DIR = $(BENCH_DIR)/bin_sim
DEP_CACHE = $(wildcard $(INCLUDE_CACHE)/*.h)

# Simulation kernels (algorithms with cache instrumentation)
KERNELS_SIM = pr pr_spmv bfs bc cc cc_sv sssp tc

# Create bin_sim directory
$(BIN_SIM_DIR):
	mkdir -p $@

# Build simulation versions
$(BIN_SIM_DIR)/%: $(SRC_SIM_DIR)/%.cc $(DEP_GAPBS) $(DEP_CACHE) | $(BIN_SIM_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(INCLUDE_CACHE) $< $(LDLIBS) -o $@

# Convenience targets for simulation builds
.PHONY: sim-% all-sim clean-sim run-sim-% run-sim-multicore-%

sim-%: $(BIN_SIM_DIR)/%
	@echo "Built simulation version: $<"

all-sim: $(addprefix $(BIN_SIM_DIR)/, $(KERNELS_SIM))
	@echo "Built all simulation binaries"

clean-sim:
	rm -rf $(BIN_SIM_DIR)

.PHONY: lint-includes
lint-includes:
	@$(LINT_INCLUDES)

# Run simulation with default parameters (single-core mode)
run-sim-%: $(BIN_SIM_DIR)/%
	@echo "Running cache simulation (single-core): $<"
	@./$< -g 10 -n 1

# Run multi-core simulation (8 cores with private L1/L2, shared L3)
run-sim-multicore-%: $(BIN_SIM_DIR)/%
	@echo "Running cache simulation (multi-core, 8 cores): $<"
	@CACHE_MULTICORE=1 CACHE_NUM_CORES=8 ./$< -g 10 -n 1

# =========================================================
# Help
# =========================================================
help: help-pr
	@echo "Available Make commands:"	
	@echo "  all            - Builds all targets including GAP benchmarks (CPU)"
	@echo "  run-%          - Runs the specified GAP benchmark (bc bfs cc cc_sv pr pr_spmv sssp tc)"
	@echo "  help-%         - Print the specified Help (bc bfs cc cc_sv pr pr_spmv sssp tc)"
	@echo "  clean          - Removes all build artifacts"
	@echo "  lint-includes  - Check for legacy include paths"
	@echo ""
	@echo "Cache Simulation:"
	@echo "  all-sim          - Builds all cache simulation binaries (pr bfs bc cc sssp tc)"
	@echo "  sim-%            - Build simulation version of specified algorithm"
	@echo "  run-sim-%        - Run cache simulation (single-core mode)"
	@echo "  run-sim-multicore-% - Run multi-core simulation (8 cores, private L1/L2, shared L3)"
	@echo "  clean-sim        - Remove simulation build artifacts"
	@echo ""
	@echo "Cache Simulation Environment Variables:"
	@echo "  CACHE_L1_SIZE=32768       - L1 cache size in bytes (default: 32KB)"
	@echo "  CACHE_L1_WAYS=8           - L1 associativity (default: 8-way)"
	@echo "  CACHE_L2_SIZE=262144      - L2 cache size in bytes (default: 256KB)"
	@echo "  CACHE_L2_WAYS=8           - L2 associativity (default: 8-way)"
	@echo "  CACHE_L3_SIZE=8388608     - L3 cache size in bytes (default: 8MB)"
	@echo "  CACHE_L3_WAYS=16          - L3 associativity (default: 16-way)"
	@echo "  CACHE_LINE_SIZE=64        - Cache line size in bytes (default: 64)"
	@echo "  CACHE_POLICY=LRU          - Eviction policy: LRU, FIFO, RANDOM, LFU, PLRU, SRRIP"
	@echo "  CACHE_MULTICORE=1         - Enable multi-core mode (private L1/L2, shared L3)"
	@echo "  CACHE_NUM_CORES=8         - Number of cores for multi-core mode"
	@echo "  CACHE_OUTPUT_JSON=file    - Export stats to JSON file"
	@echo ""
	@echo "Other:"
	@echo "  publish-wiki   - Publish wiki/ to GitHub wiki"
	@echo "  wiki-status    - Show wiki files status"
	@echo "  help           - Displays this help message"

# Alias each kernel target to its corresponding help target
$(KERNELS) converter: %: help-%

help-%: $(BIN_DIR)/%
	@./$< -h 
	@echo ""
	@echo "Reordering Algorithms:"
	@echo "  ┌─────────────────────────────────────────────────────────────────────────────┐"
	@echo "  │ Basic Algorithms                                                            │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ ORIGINAL       (0):  No reordering applied                                  │"
	@echo "  │ RANDOM         (1):  Apply random reordering                                │"
	@echo "  │ SORT           (2):  Apply sort-based reordering                            │"
	@echo "  │ HUBSORT        (3):  Apply hub-based sorting                                │"
	@echo "  │ HUBCLUSTER     (4):  Apply clustering based on hub scores                   │"
	@echo "  │ DBG            (5):  Apply degree-based grouping                            │"
	@echo "  │ HUBSORTDBG     (6):  Combine hub sorting with degree-based grouping         │"
	@echo "  │ HUBCLUSTERDBG  (7):  Combine hub clustering with degree-based grouping      │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ Community-Based Algorithms                                                  │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ RABBITORDER    (8):  Community clustering (format: 8:variant)               │"
	@echo "  │                      Variants: csr (default), boost                         │"
	@echo "  │ GORDER         (9):  Dynamic programming BFS and windowing ordering         │"
	@echo "  │ CORDER        (10):  Workload balancing via graph reordering                │"
	@echo "  │ RCM           (11):  Reverse Cuthill-McKee algorithm (BFS-based)            │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ Advanced Hybrid Algorithms                                                  │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ GraphBrewOrder(12):  Leiden clustering + per-community ordering             │"
	@echo "  │                      Format: -o 12:<freq>:<intra_algo>:<resolution>         │"
	@echo "  │ MAP           (13):  Load reordering from file (-o 13:mapping.<lo|so>)      │"
	@echo "  │ AdaptiveOrder (14):  ML-based algorithm selector (perceptron/DT/hybrid)    │"
	@echo "  │                      Format: -o 14[:_[:_[:_[:selection_mode[:graph_name]]]]]│"
	@echo "  │                      Positions 0–2 are reserved (unused)                    │"
	@echo "  │                      selection_mode(pos 3): 0=fastest-reorder,              │"
	@echo "  │                        1=fastest-execution(default), 2=best-endtoend,       │"
	@echo "  │                        3=best-amortization                                  │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ Leiden Algorithms                                                           │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ LeidenOrder   (15):  Leiden via GVE-Leiden library (baseline reference)     │"
	@echo "  └─────────────────────────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "Example Usage:"
	@echo "  make all - Compile the program."
	@echo "  make clean - Clean build files."
	@echo "  ./$< -g 15 -n 1 -o 14           - Execute with AdaptiveOrder (auto-select best)"
	@echo "  ./$< -g 15 -n 1 -o 14::::0       - AdaptiveOrder with fastest-reorder mode"
	@echo "  ./$< -g 15 -n 1 -o 14::::1:web-Google - fastest-execution with graph name hint"
	@echo "  ./$< -g 15 -n 1 -o 12:hrab              - Execute GraphBrewOrder with hybrid Leiden+Rabbit"
	@echo "  ./$< -g 15 -n 1 -o 12:community         - Execute GraphBrew with community sort"
	@echo "  ./$< -f graph.mtx -o 13:map.lo  - Execute with MAP reordering from file"

help-all: $(addprefix help-, $(KERNELS))

# =========================================================
# Wiki Publishing
# =========================================================
WIKI_REPO = https://github.com/UVA-LavaLab/GraphBrew.wiki.git
WIKI_DIR = wiki
WIKI_CLONE_DIR = .wiki_publish

.PHONY: publish-wiki wiki-status

publish-wiki:
	@echo "$(BLUE)Publishing wiki to GitHub...$(NC)"
	@if [ ! -d "$(WIKI_DIR)" ]; then \
		echo "$(RED)Error: wiki/ directory not found$(NC)"; \
		exit 1; \
	fi
	@rm -rf $(WIKI_CLONE_DIR)
	@echo "Cloning wiki repository..."
	@git clone $(WIKI_REPO) $(WIKI_CLONE_DIR) || { \
		echo "$(YELLOW)Wiki repo not initialized. Creating first commit...$(NC)"; \
		mkdir -p $(WIKI_CLONE_DIR); \
		cd $(WIKI_CLONE_DIR) && git init && git remote add origin $(WIKI_REPO); \
	}
	@echo "Copying wiki files..."
	@cp -r $(WIKI_DIR)/*.md $(WIKI_CLONE_DIR)/
	@cd $(WIKI_CLONE_DIR) && \
		git add -A && \
		git commit -m "Update wiki documentation - $$(date '+%Y-%m-%d %H:%M:%S')" && \
		git push origin master || git push origin main || { \
			echo "$(YELLOW)Trying to push to new branch...$(NC)"; \
			git push -u origin master || git push -u origin main; \
		}
	@rm -rf $(WIKI_CLONE_DIR)
	@echo "$(GREEN)Wiki published successfully!$(NC)"
	@echo "View at: https://github.com/UVA-LavaLab/GraphBrew/wiki"

wiki-status:
	@echo "Wiki files in $(WIKI_DIR)/:"
	@ls -la $(WIKI_DIR)/*.md 2>/dev/null | wc -l | xargs -I {} echo "  {} markdown files"
	@ls $(WIKI_DIR)/*.md 2>/dev/null | xargs -I {} basename {} | sed 's/^/  - /'