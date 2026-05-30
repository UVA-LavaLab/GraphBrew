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
DEP_GRAPHBREW = $(wildcard $(INCLUDE_GRAPHBREW)/reorder/*.h) $(wildcard $(INCLUDE_GRAPHBREW)/partition/*.h)
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
KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc tc_p
KERNELS_BIN = $(addprefix $(BIN_DIR)/,$(KERNELS))
SUITE = $(KERNELS_BIN) $(BIN_DIR)/converter
# =========================================================

.PHONY: $(KERNELS) converter all run-% exp-% graph-% help-% install-py-deps help clean clean-all clean-results run-%-gdb run-%-sweep $(BIN_DIR)/% scrub-all
all: $(SUITE)

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

# =========================================================
# Directory Setup
# =========================================================
$(BIN_DIR):
	@mkdir -p $@ $(CREATE_STATUS)

# =========================================================
# Cleanup
# =========================================================
clean:
	@rm -rf $(BIN_DIR) $(BIN_SIM_DIR) $(EXIT_STATUS)

clean-all: clean-results clean-sim
	@rm -rf $(BIN_DIR) $(BIN_SIM_DIR) $(EXIT_STATUS)

# =========================================================
# Testing
# =========================================================
scrub-all:
	@rm -rf $(BIN_DIR) $(BIN_SIM_DIR) 00_* $(EXIT_STATUS)

# =========================================================
# Cache Simulation Builds
# =========================================================
SRC_SIM_DIR = $(BENCH_DIR)/src_sim
BIN_SIM_DIR = $(BENCH_DIR)/bin_sim
DEP_CACHE = $(wildcard $(INCLUDE_CACHE)/*.h)

# Simulation kernels (algorithms with cache instrumentation)
KERNELS_SIM = pr pr_spmv bfs bc cc cc_sv sssp tc ecg_preprocess

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

# =========================================================
# gem5 Benchmark Builds (static, single-threaded, for SE mode)
# =========================================================
SRC_GEM5_DIR = $(BENCH_DIR)/src_gem5
BIN_GEM5_DIR = $(BENCH_DIR)/bin_gem5
GEM5_HARNESS_INC = $(INC_DIR)/gem5_sim
GEM5_SIM_DIR = $(INC_DIR)/gem5_sim
GEM5_DIR = $(GEM5_SIM_DIR)/gem5
GEM5_M5_DIR = $(GEM5_DIR)/util/m5
GEM5_M5_LIB = $(GEM5_M5_DIR)/build/x86/out/libm5.a
GEM5_M5_RISCV_LIB = $(GEM5_M5_DIR)/build/riscv/out/libm5.a
RISCV_CXX ?= riscv64-linux-gnu-g++
RISCV_CROSS_COMPILE ?= riscv64-linux-gnu-

# gem5 benchmarks: single-threaded at runtime but headers need OpenMP.
# Static linking for gem5 SE mode. -O1 to avoid unsupported instructions.
CXXFLAGS_GEM5 = -std=c++17 -O1 -Wall -g -DNDEBUG -DNO_M5OPS -fopenmp
CXXFLAGS_GEM5_M5OPS = $(filter-out -DNO_M5OPS,$(CXXFLAGS_GEM5)) -I$(GEM5_DIR)/include
CXXFLAGS_GEM5_RISCV_M5OPS = $(CXXFLAGS_GEM5_M5OPS) -static
KERNELS_GEM5 = pr pr_spmv bfs sssp cc cc_sv bc tc

$(BIN_GEM5_DIR):
	@mkdir -p $@ $(CREATE_STATUS)

$(BIN_GEM5_DIR)/%: $(SRC_GEM5_DIR)/%.cc $(DEP_GAPBS) | $(BIN_GEM5_DIR)
	@$(CXX) $(CXXFLAGS_GEM5) $(CXXFLAGS_LEIDEN) $(INCLUDES) $< $(LDLIBS) -o $@ $(EXIT_STATUS)

$(GEM5_M5_LIB):
	@cd $(GEM5_M5_DIR) && scons -j$(PARALLEL) build/x86/out/m5

$(GEM5_M5_RISCV_LIB):
	@cd $(GEM5_M5_DIR) && scons -j$(PARALLEL) build/riscv/out/m5 riscv.CROSS_COMPILE=$(RISCV_CROSS_COMPILE)

$(BIN_GEM5_DIR)/%_m5ops: $(SRC_GEM5_DIR)/%.cc $(DEP_GAPBS) $(GEM5_M5_LIB) | $(BIN_GEM5_DIR)
	@$(CXX) $(CXXFLAGS_GEM5_M5OPS) $(CXXFLAGS_LEIDEN) $(INCLUDES) $< $(GEM5_M5_LIB) $(LDLIBS) -o $@ $(EXIT_STATUS)

$(BIN_GEM5_DIR)/%_riscv_m5ops: $(SRC_GEM5_DIR)/%.cc $(DEP_GAPBS) $(GEM5_M5_RISCV_LIB) | $(BIN_GEM5_DIR)
	@$(RISCV_CXX) $(CXXFLAGS_GEM5_RISCV_M5OPS) $(CXXFLAGS_LEIDEN) $(INCLUDES) $< $(GEM5_M5_RISCV_LIB) -o $@ $(EXIT_STATUS)

.PRECIOUS: $(BIN_GEM5_DIR)/%_m5ops
.PRECIOUS: $(BIN_GEM5_DIR)/%_riscv_m5ops

.PHONY: gem5-% gem5-m5ops-% gem5-riscv-m5ops-% all-gem5 clean-gem5-bin run-gem5-%

gem5-%: $(BIN_GEM5_DIR)/%
	@echo "Built gem5 version: $<"

gem5-m5ops-%: $(BIN_GEM5_DIR)/%_m5ops
	@echo "Built gem5 ROI/m5ops version: $<"

gem5-riscv-m5ops-%: $(BIN_GEM5_DIR)/%_riscv_m5ops
	@echo "Built RISC-V gem5 ROI/m5ops version: $<"

all-gem5: $(addprefix $(BIN_GEM5_DIR)/, $(KERNELS_GEM5))
	@echo "Built all gem5 benchmarks"

clean-gem5-bin:
	rm -rf $(BIN_GEM5_DIR)

# Run gem5 benchmark natively (for testing without gem5)
run-gem5-%: $(BIN_GEM5_DIR)/%
	@echo "Running gem5 benchmark natively: $<"
	@./$< -g 10 -n 1

# =========================================================
# gem5 Simulation Setup
# =========================================================
.PHONY: setup-gem5 clean-gem5

setup-gem5:
	@echo "Setting up gem5 for GraphBrew..."
	$(PYTHON) $(SCRIPT_DIR)/setup_gem5.py --isa X86 --jobs $(PARALLEL)

clean-gem5:
	@echo "Removing cloned gem5..."
	$(PYTHON) $(SCRIPT_DIR)/setup_gem5.py --clean

# =========================================================
# Sniper Simulation Setup
# =========================================================
SRC_SNIPER_DIR = $(BENCH_DIR)/src_sniper
BIN_SNIPER_DIR = $(BENCH_DIR)/bin_sniper
SNIPER_HARNESS_INC = $(INC_DIR)/sniper_sim
SNIPER_SIM_DIR = $(INC_DIR)/sniper_sim
SNIPER_DIR = $(SNIPER_SIM_DIR)/snipersim
SNIPER_INCLUDE = $(SNIPER_DIR)/include
CXXFLAGS_SNIPER = -std=c++17 -O2 -Wall -g -DNDEBUG -fopenmp -I$(INC_DIR) -I$(SNIPER_INCLUDE)

$(BIN_SNIPER_DIR):
	@mkdir -p $@ $(CREATE_STATUS)

$(BIN_SNIPER_DIR)/%: $(SRC_SNIPER_DIR)/%.cc $(DEP_GAPBS) | $(BIN_SNIPER_DIR)
	@$(CXX) $(CXXFLAGS_SNIPER) $(CXXFLAGS_LEIDEN) $(INCLUDES) $< $(LDLIBS) -o $@ $(EXIT_STATUS)

.PRECIOUS: $(BIN_SNIPER_DIR)/%

.PHONY: setup-sniper clean-sniper sniper-% run-sniper-%

setup-sniper:
	@echo "Setting up Sniper for GraphBrew..."
	$(PYTHON) $(SCRIPT_DIR)/setup_sniper.py --jobs $(PARALLEL)

clean-sniper:
	@echo "Removing cloned Sniper..."
	$(PYTHON) $(SCRIPT_DIR)/setup_sniper.py --clean

sniper-%: $(BIN_SNIPER_DIR)/%
	@echo "Built Sniper benchmark: $<"

run-sniper-%: $(BIN_SNIPER_DIR)/%
	@echo "Running Sniper benchmark natively: $<"
	@./$<

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
	@echo "gem5 Simulation:"
	@echo "  setup-gem5       - Clone, patch, and build gem5 with GraphBrew policies"
	@echo "  clean-gem5       - Remove cloned gem5 directory"
	@echo ""
	@echo "gem5 Benchmarks (static, single-threaded for SE mode):"
	@echo "  all-gem5         - Build all gem5 benchmark binaries"
	@echo "  gem5-%           - Build gem5 version of specified algorithm"
	@echo "  gem5-m5ops-%     - Build gem5 benchmark with ROI m5ops markers"
	@echo "  gem5-riscv-m5ops-% - Build RISC-V gem5 benchmark with ROI m5ops markers"
	@echo "  run-gem5-%       - Run gem5 benchmark natively (for testing)"
	@echo "  clean-gem5-bin   - Remove gem5 benchmark binaries"
	@echo ""
	@echo "Sniper Simulation:"
	@echo "  setup-sniper     - Clone and build Sniper for GraphBrew"
	@echo "  clean-sniper     - Remove cloned Sniper directory"
	@echo "  sniper-%         - Build Sniper-oriented benchmark (initial: hello_roi)"
	@echo "  run-sniper-%     - Run Sniper-oriented benchmark natively"
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
	@echo "  CACHE_L1_POLICY=LRU       - Optional L1 policy override (defaults to CACHE_POLICY)"
	@echo "  CACHE_L2_POLICY=LRU       - Optional L2 policy override (defaults to CACHE_POLICY)"
	@echo "  CACHE_L3_POLICY=LRU       - Optional L3 policy override (defaults to CACHE_POLICY)"
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

# =========================================================
# Confidence / literature-faithfulness convenience targets.
#
# These wrap the ECG comparator + dashboard so contributors can
# get a single-screen go/no-go without remembering the long
# python -m invocations.
# =========================================================
LIT_SWEEP_ROOT  ?= /tmp/graphbrew-lit-baseline
LIT_SWEEP_SUBDIR ?= lit
GEM5_ANCHOR_ROOT ?= /tmp/graphbrew-grasp-gem5-sweep
GEM5_ANCHOR_SUBDIR ?= DBG
GEM5_ANCHOR_GRAPHS ?= email-Eu-core
SNIPER_ANCHOR_ROOT ?= /tmp/graphbrew-grasp-sniper-sweep
SNIPER_ANCHOR_SUBDIR ?= DBG
SNIPER_ANCHOR_GRAPHS ?= email-Eu-core cit-Patents
# BFS deferred: small_cache_divergence fails on both graphs (working-set vs 4kB)
# and the email-Eu-core headline shows GRASP +1.49pp over LRU (insufficient reuse).
SNIPER_ANCHOR_APPS ?= pr sssp
WIKI_DATA       := $(WIKI_DIR)/data

.PHONY: lit-faith lit-repro lit-budget lit-table lit-winner lit-thrash lit-cross-tool lit-cross-tool-winners lit-density lit-popt-vs-grasp lit-popt-vs-grasp-by-family-app lit-wilson-wins lit-cohens-h lit-gap-effect-size lit-l3-stability lit-mt-correction lit-logo-robust lit-cell-census lit-family-geomean lit-per-graph-app-stability lit-corpus-balance lit-distribution-diagnostics lit-lofo-robustness lit-winner-margin-gradient lit-oracle-gap-auc lit-policy-auc-correlation lit-policy-stability lit-cache-sensitivity-slope lit-per-graph-cache-slope lit-cross-generator-gap-parity lit-cache-saturation-onset lit-gap-distribution-shape lit-family-policy-auc-clustering lit-oracle-gap-curvature lit-policy-rank-kendall lit-wss-knee-location lit-family-curvature-replay lit-winner-margin-by-regime lit-family-margin-replay lit-cross-policy-asymmetry lit-saturation-distance lit-capacity-sensitivity lit-family-slope-replay lit-per-app-capacity-slope lit-slope-saturation-xcheck lit-gem5-slope-replay lit-sniper-slope-replay lit-cross-tool-slope-ordering lit-per-app-srrip-vs-grasp lit-cross-tool-lru-regime lit-saturation-slope-extremum lit-cross-tool-slope-universality lit-monotonicity-universality lit-anchor-cell-census lit-family-saturation-distance lit-anchor-monotonicity-replay lit-policy-steepness-ranking lit-anchor-cross-tool-agreement lit-deviations lit-diversity lit-margin lit-signmass lit-citations lit-knowndev lit-tolerance lit-accesses lit-citexapp lit-monotonicity lit-stat lit-polyord lit-devexp lit-ratgrid lit-cellcomp lit-appfreq lit-regimesign lit-citdate lit-ecg-parity lit-regime-taxonomy lit-oracle-gap lit-oracle-gap-by-app lit-oracle-by-app-bootstrap lit-wss-relative-l3 lit-bootstrap-ci lit-family-sensitivity lit-catalog lit-reproduce-smoke lit-claims gem5-anchor sniper-anchor confidence confidence-fast

lit-faith:
	@echo "$(BLUE)Regenerating literature faithfulness report...$(NC)"
	@python3 scripts/experiments/ecg/literature_faithfulness.py \
		--sweep-root $(LIT_SWEEP_ROOT) \
		--sweep-subdir $(LIT_SWEEP_SUBDIR) \
		--json-out $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--md-out   $(WIKI_DATA)/literature_faithfulness_postfix.md \
		--csv-out  $(WIKI_DATA)/literature_faithfulness_postfix.csv

lit-repro:
	@echo "$(BLUE)Regenerating literature reproduction summary...$(NC)"
	@python3 -m scripts.experiments.ecg.literature_reproduction_summary \
		--lit-faith-json $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--markdown $(WIKI_DATA)/literature_reproduction_summary.md \
		--csv      $(WIKI_DATA)/literature_reproduction_summary.csv

lit-budget:
	@echo "$(BLUE)Regenerating regression budget...$(NC)"
	@python3 scripts/experiments/ecg/regression_budget.py \
		--lit-faith-json $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out $(WIKI_DATA)/regression_budget.json \
		--md-out   $(WIKI_DATA)/regression_budget.md \
		--csv-out  $(WIKI_DATA)/regression_budget.csv

lit-table:
	@echo "$(BLUE)Regenerating paper baseline table...$(NC)"
	@python3 -m scripts.experiments.ecg.paper_baseline_table \
		--sweep-root $(LIT_SWEEP_ROOT) \
		--sweep-subdir $(LIT_SWEEP_SUBDIR) \
		--markdown $(WIKI_DATA)/paper_baseline_table.md \
		--csv      $(WIKI_DATA)/paper_baseline_table.csv \
		--json     $(WIKI_DATA)/paper_baseline_table.json

# Project the lit-faith CSV onto a winner-per-cell view and emit
# paper-grade per-policy / per-family / per-regime win counts. Depends
# only on the lit-faith CSV being current.
lit-winner: lit-faith
	@echo "$(BLUE)Regenerating policy winner table...$(NC)"
	@python3 -m scripts.experiments.ecg.policy_winner_table \
		--lit-faith-csv $(WIKI_DATA)/literature_faithfulness_postfix.csv \
		--corpus-json   $(WIKI_DATA)/corpus_diversity.json \
		--csv-out  $(WIKI_DATA)/policy_winner_table.csv \
		--json-out $(WIKI_DATA)/policy_winner_table.json \
		--md-out   $(WIKI_DATA)/policy_winner_table.md

# Build the small-L3 (4 kB) "thrash" sub-report from the standalone
# final_cache_sim sweep. This regime intentionally overflows the hot
# working set so LRU/SRRIP can beat GRASP/POPT and the four ECG
# variants get exercised. Falls back gracefully if no sweep is on disk.
lit-thrash:
	@echo "$(BLUE)Regenerating small-L3 thrash report...$(NC)"
	@if ls results/ecg_experiments/paper_pipeline/*/final_cache_sim/combined_roi_matrix.csv >/dev/null 2>&1; then \
		python3 -m scripts.experiments.ecg.small_l3_thrash_report \
			--csv-out  $(WIKI_DATA)/small_l3_thrash.csv \
			--json-out $(WIKI_DATA)/small_l3_thrash.json \
			--md-out   $(WIKI_DATA)/small_l3_thrash.md; \
	else \
		echo "$(BLUE)  no final_cache_sim sweep on disk; reusing on-disk snapshot.$(NC)"; \
	fi

# Regenerate the gem5 literature anchor (small graph, fast).
# Scoped to email-Eu-core by default; expand GEM5_ANCHOR_GRAPHS once
# additional gem5 sweeps land. A missing sweep root is non-fatal so
# the gate degrades to "missing" rather than blocking the build.
gem5-anchor:
	@echo "$(BLUE)Regenerating gem5 literature anchor...$(NC)"
	@if [ -d "$(GEM5_ANCHOR_ROOT)" ]; then \
		python3 scripts/experiments/ecg/gem5_anchor_summary.py \
			--sweep-root $(GEM5_ANCHOR_ROOT) \
			--sweep-subdir $(GEM5_ANCHOR_SUBDIR) \
			--graphs $(GEM5_ANCHOR_GRAPHS) \
			--title "gem5 literature anchor" \
			--json-out $(WIKI_DATA)/gem5_anchor.json \
			--md-out   $(WIKI_DATA)/gem5_anchor.md \
			--exit-on-disagree; \
	else \
		echo "$(BLUE)  gem5 sweep dir $(GEM5_ANCHOR_ROOT) not present; reusing on-disk snapshot.$(NC)"; \
	fi

# Regenerate the Sniper literature anchor. PR + SSSP are validated on
# both graphs; BFS is deferred (4 kB small_cache_divergence under the
# 2 pp floor on both graphs, and email-Eu-core headline shows
# GRASP +1.49 pp over LRU — insufficient reuse for the L-shape).
sniper-anchor:
	@echo "$(BLUE)Regenerating Sniper literature anchor...$(NC)"
	@if [ -d "$(SNIPER_ANCHOR_ROOT)" ]; then \
		python3 scripts/experiments/ecg/gem5_anchor_summary.py \
			--sweep-root $(SNIPER_ANCHOR_ROOT) \
			--sweep-subdir $(SNIPER_ANCHOR_SUBDIR) \
			--graphs $(SNIPER_ANCHOR_GRAPHS) \
			--apps $(SNIPER_ANCHOR_APPS) \
			--title "Sniper literature anchor" \
			--json-out $(WIKI_DATA)/sniper_anchor.json \
			--md-out   $(WIKI_DATA)/sniper_anchor.md \
			--exit-on-disagree; \
	else \
		echo "$(BLUE)  Sniper sweep dir $(SNIPER_ANCHOR_ROOT) not present; reusing on-disk snapshot.$(NC)"; \
	fi

# Pair each cache_sim lit-faith cell with the matching gem5/Sniper
# anchor cell and verify cross-tool agreement at saturation. Depends
# on the three input artifacts already being on disk; degrades to a
# refresh of the report from existing inputs if anchors are stale.
lit-cross-tool:
	@echo "$(BLUE)Regenerating cross-tool saturation report...$(NC)"
	@python3 -m scripts.experiments.ecg.cross_tool_saturation_report \
		--lit-faith-csv $(WIKI_DATA)/literature_faithfulness_postfix.csv \
		--gem5-anchor   $(WIKI_DATA)/gem5_anchor.json \
		--sniper-anchor $(WIKI_DATA)/sniper_anchor.json \
		--csv-out  $(WIKI_DATA)/cross_tool_saturation.csv \
		--json-out $(WIKI_DATA)/cross_tool_saturation.json \
		--md-out   $(WIKI_DATA)/cross_tool_saturation.md

# Per-graph literature claim density mini-report. Reads the
# reproduction summary and tallies claims/cells/status mix per graph.
# Depends only on lit-repro being current.
lit-density: lit-repro
	@echo "$(BLUE)Regenerating per-graph claim density report...$(NC)"
	@python3 -m scripts.experiments.ecg.claim_density_report \
		--repro-csv $(WIKI_DATA)/literature_reproduction_summary.csv \
		--csv-out  $(WIKI_DATA)/claim_density.csv \
		--json-out $(WIKI_DATA)/claim_density.json \
		--md-out   $(WIKI_DATA)/claim_density.md

# POPT-vs-GRASP head-to-head delta report. For each (graph, app, L3)
# cell with both GRASP and POPT data this projects miss_rate(POPT) -
# miss_rate(GRASP) in percentage points, broken down by graph family
# and L3 regime. Answers "when does POPT actually improve on GRASP?"
lit-popt-vs-grasp: lit-faith
	@echo "$(BLUE)Regenerating POPT vs GRASP delta report...$(NC)"
	@python3 -m scripts.experiments.ecg.popt_vs_grasp_report \
		--lit-faith-csv $(WIKI_DATA)/literature_faithfulness_postfix.csv \
		--corpus-json   $(WIKI_DATA)/corpus_diversity.json \
		--csv-out       $(WIKI_DATA)/popt_vs_grasp_delta.csv \
		--json-out      $(WIKI_DATA)/popt_vs_grasp_delta.json \
		--md-out        $(WIKI_DATA)/popt_vs_grasp_delta.md

# Inventory + mechanism classification of every known_deviation row in
# the reproduction summary. Each deviation gets a categorical label so
# the paper's KNOWN_DEVIATIONS table is point-by-point explainable.
lit-deviations: lit-repro lit-faith
	@echo "$(BLUE)Regenerating literature deviations inventory...$(NC)"
	@python3 -m scripts.experiments.ecg.literature_deviations_report \
		--repro-csv     $(WIKI_DATA)/literature_reproduction_summary.csv \
		--lit-faith-csv $(WIKI_DATA)/literature_faithfulness_postfix.csv \
		--csv-out       $(WIKI_DATA)/literature_deviations.csv \
		--json-out      $(WIKI_DATA)/literature_deviations.json \
		--md-out        $(WIKI_DATA)/literature_deviations.md

# Literature-faithfulness diversity audit: family × app × L3 × paper
# coverage matrix + cross-paper triangulation cells (where multiple
# papers issue a claim on the same cell). The LIT-Cov confidence gate
# locks per-axis floors so future regens cannot silently drop a graph
# family or a cited paper.
lit-diversity: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness diversity audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_diversity \
		--lit-faith-json $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out       $(WIKI_DATA)/lit_faith_diversity.json \
		--md-out         $(WIKI_DATA)/lit_faith_diversity.md \
		--csv-out        $(WIKI_DATA)/lit_faith_diversity.csv

# Literature-faithfulness margin audit: per-claim distance-to-disagree
# distribution. LIT-Mar gate locks the median and per-family medians
# above a comfortable floor and caps the count of "fragile" cells
# (< 1 pp from flipping into disagree) so corpus drift can't silently
# erode confidence in the lit-faith verdicts.
lit-margin: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness margin audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_margin \
		--lit-faith-json $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out       $(WIKI_DATA)/lit_faith_margin.json \
		--md-out         $(WIKI_DATA)/lit_faith_margin.md \
		--csv-out        $(WIKI_DATA)/lit_faith_margin.csv

# Literature-faithfulness sign-mass concentration: for each
# (expected_sign × policy) bucket, report how often observed delta_pct
# carries the literature's claimed sign. LIT-Sig locks per-bucket
# Wilson 95 % lower bounds and binomial sign-test p-values so a
# regression that erases the sign signal (e.g. a policy that newly
# loses to LRU on most cells but stays inside the magnitude envelope)
# trips the gate.
lit-signmass: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness sign-mass audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_signmass \
		--lit-faith-json $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out       $(WIKI_DATA)/lit_faith_signmass.json \
		--md-out         $(WIKI_DATA)/lit_faith_signmass.md \
		--csv-out        $(WIKI_DATA)/lit_faith_signmass.csv

# Literature-faithfulness citation locator integrity audit: bijection
# between citations in literature_baselines.py and lit-faith corpus,
# anchor-paper inventory, citation well-formedness (venue + year +
# locator + known anchor). LIT-Cite trips on a dropped paper, a
# placeholder citation, or a stripped DOI in the source-of-truth module
# docstring.
lit-citations: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness citation integrity audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_citations \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--baselines-module scripts/experiments/ecg/literature_baselines.py \
		--json-out         $(WIKI_DATA)/lit_faith_citations.json \
		--md-out           $(WIKI_DATA)/lit_faith_citations.md \
		--csv-out          $(WIKI_DATA)/lit_faith_citations.csv

# Lit-faith known-deviation completeness audit (LIT-Dev, gate 225): for
# each entry in literature_baselines.KNOWN_DEVIATIONS, verify the reason
# is long, quantitative, and anchored (paper § / design term / algorithmic
# root-cause vocabulary). Also enforces bijection with the live faith
# corpus -- no orphan whitelist entries, no live `known_deviation` rows
# that lack a documented explanation.
lit-knowndev: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness known-deviation completeness audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_deviations \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--baselines-module scripts/experiments/ecg/literature_baselines.py \
		--json-out         $(WIKI_DATA)/lit_faith_deviations.json \
		--md-out           $(WIKI_DATA)/lit_faith_deviations.md \
		--csv-out          $(WIKI_DATA)/lit_faith_deviations.csv

# Lit-faith tolerance-calibration audit (LIT-Tol, gate 226): for every
# literature claim whose comparator-asserted bound actually fires, compute
# the slack (pp the observed |delta_pct| could move toward the disagree
# boundary). Surfaces fragile cells (1pp from flipping) and over-permissive
# bounds (slack much wider than typical regen noise). Gates corpus-wide
# median + per-policy minimum + strict-policy zero-fragile invariants.
lit-tolerance: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness tolerance calibration audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_tolerance \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_tolerance.json \
		--md-out           $(WIKI_DATA)/lit_faith_tolerance.md \
		--csv-out          $(WIKI_DATA)/lit_faith_tolerance.csv

# Lit-faith accesses-floor audit (LIT-Acc, gate 227): warmup-noise
# guard for the lit-faith corpus. Reads `accesses` per row and floors
# both production graphs and the email-Eu-core dev-smoke; tracks
# per-axis distribution and warmup buckets so silently-truncated
# traces cannot creep back in.
lit-accesses: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness accesses-floor audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_accesses \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_accesses.json \
		--md-out           $(WIKI_DATA)/lit_faith_accesses.md \
		--csv-out          $(WIKI_DATA)/lit_faith_accesses.csv

# Lit-faith cross-app rationale coherence audit (LIT-CXApp, gate 228):
# for every (citation, expected_sign) group, audit the per-cell
# rationales for contradictions, sign-vocabulary alignment, common
# anchor kernel, and length-span ratio. Zero contradictions + zero
# sign misses are the invariant.
lit-citexapp: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness cross-app rationale coherence audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_citexapp \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_citexapp.json \
		--md-out           $(WIKI_DATA)/lit_faith_citexapp.md \
		--csv-out          $(WIKI_DATA)/lit_faith_citexapp.csv

# Lit-faith cache-size monotonicity audit (LIT-Mono, gate 229): for
# every (graph, app, policy) triple with >=2 L3 sizes, enforce the
# physical invariant that miss rate is non-increasing in L3 size
# (tolerance 0.5 pp). Surfaces slope-per-doubling and saturated
# triples so corpus pressure can be reasoned about.
lit-monotonicity: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness cache-size monotonicity audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_monotonicity \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_monotonicity.json \
		--md-out           $(WIKI_DATA)/lit_faith_monotonicity.md \
		--csv-out          $(WIKI_DATA)/lit_faith_monotonicity.csv

# Lit-faith statistical-sanity audit (LIT-Stat, gate 230): re-derives
# delta_pct from the two miss-rate columns each per_claim row compares
# (LRU-vs-policy, POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP) and locks
# rounding drift, sign flips, NaN/inf, miss-rate bounds, status-label
# vocabulary, and status-vs-delta consistency.
lit-stat: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness statistical-sanity audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_stat \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_stat.json \
		--md-out           $(WIKI_DATA)/lit_faith_stat.md \
		--csv-out          $(WIKI_DATA)/lit_faith_stat.csv

# LIT-PolyOrd (gate 231): per (graph_family x app) policy-ordering audit.
# Locks the literature-faithful invariant POPT/GRASP <= LRU within
# tolerance on hub-bearing families (social/citation/web) while allowing
# documented hub-less regressions (road/mesh). Also enforces per-app
# global improve-frac floor so a corpus shrink toward weakly-improving
# cells cannot pass silently.
lit-polyord: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness policy-ordering audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_polyord \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_polyord.json \
		--md-out           $(WIKI_DATA)/lit_faith_polyord.md \
		--csv-out          $(WIKI_DATA)/lit_faith_polyord.csv

# LIT-DevExp (gate 232): deviation-explanation depth audit. Every
# known_deviation row's reason text must name an algorithmic mechanism
# (not just paraphrase the magnitude), exceed a length floor, carry a
# citation, resolve cross-references, and the same reason text may not
# cover more than half the rows.
lit-devexp: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness deviation-explanation audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_devexp \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_devexp.json \
		--md-out           $(WIKI_DATA)/lit_faith_devexp.md \
		--csv-out          $(WIKI_DATA)/lit_faith_devexp.csv

# LIT-RatGrid (gate 233): per (policy, graph, app) rationale uniqueness.
# Theorem-class policies (POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP,
# SRRIP) must carry exactly 1 rationale per (policy, app); point
# policies (GRASP, POPT, LRU) may carry <= 2 rationales per (policy,
# graph, app) to accommodate L3-regime variants. Point-policy
# rationales must contain a citation token (HPCA/MICRO/Fig/§).
lit-ratgrid: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness rationale-grid audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_ratgrid \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_ratgrid.json \
		--md-out           $(WIKI_DATA)/lit_faith_ratgrid.md \
		--csv-out          $(WIKI_DATA)/lit_faith_ratgrid.csv

# Per-cell cell-completeness audit on per_observation: canonical roster
# {LRU, GRASP, POPT} present per (graph, app, l3), LRU baseline carried,
# delta_vs_lru_pct arithmetic matches the underlying miss rates within
# 0.001 pp, L3 sweep covers >= 3 sizes per non-LRU policy, and every
# present policy shares the same L3 axis within (graph, app).
lit-cellcomp: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness cell-completeness audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_cellcomp \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_cellcomp.json \
		--md-out           $(WIKI_DATA)/lit_faith_cellcomp.md \
		--csv-out          $(WIKI_DATA)/lit_faith_cellcomp.csv

# Per-app axis-coverage audit on per_observation: every app must touch
# >= 6 distinct graphs, >= 3 L3 sizes, >= 3 policies (canonical roster
# {LRU, GRASP, POPT} present per app), every (app, graph) must cover
# >= 3 L3 sizes, every app must contribute >= 60 rows, and the anchor
# app (pr) must cover the full corpus.
lit-appfreq: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness app-frequency audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_appfreq \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_appfreq.json \
		--md-out           $(WIKI_DATA)/lit_faith_appfreq.md \
		--csv-out          $(WIKI_DATA)/lit_faith_appfreq.csv

# Citation/date parseability + per-policy origin match on per_claim:
# every citation must parse to (author, venue, year), name a top-tier
# architecture venue, and reference the originator publication for its
# policy (GRASP→Faldu HPCA 2020, POPT→Balaji HPCA 2021,
# SRRIP→Jaleel ISCA 2010), or be an explicit cross-attribution where the
# policy name appears in the citation string.
lit-citdate: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness citation/date audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_citdate \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_citdate.json \
		--md-out           $(WIKI_DATA)/lit_faith_citdate.md \
		--csv-out          $(WIKI_DATA)/lit_faith_citdate.csv

# ECG substrate-parity audit (gate 238 — ECG-Parity): cache_sim component-
# proof matrix's load-bearing invariants. ECG_DBG_only must match
# GRASP_DBG_only and ECG_POPT_primary must match POPT_only on L3
# miss-rate to within 5e-4. Every PFX-bearing ablation must show
# ecg_runtime_issued >= 1 per benchmark and nonzero useful prefetches
# on the PR anchor. Encoding hygiene: ecg_pfx_encoded <= ecg_pfx_candidates
# and every PFX counter non-negative. This is the proposal-side
# confidence floor before any cluster-scale ECG sweep can be launched.
lit-ecg-parity:
	@echo "$(BLUE)Regenerating ECG substrate-parity audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_ecg_parity \
		--postfix-json $(WIKI_DATA)/ecg_substrate_parity_postfix.json \
		--json-out     $(WIKI_DATA)/lit_faith_ecg_parity.json \
		--md-out       $(WIKI_DATA)/lit_faith_ecg_parity.md \
		--csv-out      $(WIKI_DATA)/lit_faith_ecg_parity.csv

# Regime-aware sign-tally + extreme magnitude ceiling on per_observation:
# hub families {social, citation, web} must not show majority regression
# (pos_cells > neg_cells AND bucket median > +0.5 pp simultaneously)
# and hub bucket median <= +0.5 pp; no-hub families {road, mesh} may
# exhibit L-curve sign-flipping but bucket median must remain within
# ±8 pp; no individual cell may exceed |delta_vs_lru| > 80 pp.
lit-regimesign: lit-faith
	@echo "$(BLUE)Regenerating literature-faithfulness regime-sign audit...$(NC)"
	@python3 -m scripts.experiments.ecg.lit_faith_regimesign \
		--lit-faith-json   $(WIKI_DATA)/literature_faithfulness_postfix.json \
		--json-out         $(WIKI_DATA)/lit_faith_regimesign.json \
		--md-out           $(WIKI_DATA)/lit_faith_regimesign.md \
		--csv-out          $(WIKI_DATA)/lit_faith_regimesign.csv

# Cross-tool *winner* agreement: at each tool's largest-L3 operating
# point per (graph, app), do the simulators pick the same winning
# policy? Different tools sweep different L3 ranges, so disagreement
# is common and the saturation report (lit-cross-tool) is the proper
# headline test — this one surfaces the negative cases so reviewers
# can see what should NOT be claimed without per-cell L3 context.
lit-cross-tool-winners:
	@echo "$(BLUE)Regenerating cross-tool winner agreement report...$(NC)"
	@python3 -m scripts.experiments.ecg.cross_tool_winners_report \
		--lit-faith-csv $(WIKI_DATA)/literature_faithfulness_postfix.csv \
		--gem5-json     $(WIKI_DATA)/gem5_anchor.json \
		--sniper-json   $(WIKI_DATA)/sniper_anchor.json \
		--csv-out  $(WIKI_DATA)/cross_tool_winners.csv \
		--json-out $(WIKI_DATA)/cross_tool_winners.json \
		--md-out   $(WIKI_DATA)/cross_tool_winners.md

# Winning-regime taxonomy: paper headline figure projecting the
# winner table onto (graph_family × L3 regime) bins and extracting
# minimal-rule implications. Depends on the winner table being fresh.
lit-regime-taxonomy: lit-winner
	@echo "$(BLUE)Regenerating winning-regime taxonomy...$(NC)"
	@python3 -m scripts.experiments.ecg.winning_regime_taxonomy \
		--winners-json $(WIKI_DATA)/policy_winner_table.json \
		--corpus-json  $(WIKI_DATA)/corpus_diversity.json \
		--csv-out      $(WIKI_DATA)/winning_regime_taxonomy.csv \
		--json-out     $(WIKI_DATA)/winning_regime_taxonomy.json \
		--md-out       $(WIKI_DATA)/winning_regime_taxonomy.md

# Per-policy gap to the empirical oracle (= min miss rate any of the
# four production policies achieved on each cell). Quantifies the
# performance headroom remaining for a new policy on our corpus.
lit-oracle-gap: lit-faith
	@echo "$(BLUE)Regenerating per-policy oracle-gap report...$(NC)"
	@python3 -m scripts.experiments.ecg.oracle_gap_report \
		--lit-faith-csv $(WIKI_DATA)/literature_faithfulness_postfix.csv \
		--csv-out  $(WIKI_DATA)/oracle_gap.csv \
		--json-out $(WIKI_DATA)/oracle_gap.json \
		--md-out   $(WIKI_DATA)/oracle_gap.md

# Per-kernel oracle-gap breakdown. Reuses oracle_gap.json and projects
# per (policy, app) so the paper has a per-kernel winner table
# (POPT on PR; GRASP on CC; etc.).
lit-oracle-gap-by-app: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-app oracle-gap breakdown...$(NC)"
	@python3 -m scripts.experiments.ecg.oracle_gap_by_app \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--csv-out     $(WIKI_DATA)/oracle_gap_by_app.csv \
		--json-out    $(WIKI_DATA)/oracle_gap_by_app.json \
		--md-out      $(WIKI_DATA)/oracle_gap_by_app.md

# Per-(family x app) POPT-vs-GRASP bootstrap CIs. Defends the core
# claim "POPT beats GRASP on road graphs" against the reviewer
# question "is this carried by every kernel or driven by one?". Key
# finding: road is POPT-favored on ALL 5 kernels (sssp -21.8 pp);
# cc-counter-narrative confirmed cell-by-cell on social/cc and
# citation/cc (both P=0.000); social/pr is also CI-strict POPT.
lit-popt-vs-grasp-by-family-app: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-(family x app) POPT-vs-GRASP bootstrap CIs...$(NC)"
	@python3 -m scripts.experiments.ecg.popt_vs_grasp_by_family_app \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/popt_vs_grasp_by_family_app.json \
		--md-out      $(WIKI_DATA)/popt_vs_grasp_by_family_app.md

# Wilson 95% CIs on per-(scope, policy) win-counts. Turns "wins
# X of N cells" point estimates into CI-backed sign claims:
# pr/POPT strict majority (CI [0.529, 0.848]); cc/GRASP strict
# majority AND above 25%-baseline (CI [0.640, 0.948]); cc/POPT
# strict below-chance; sssp POPT vs GRASP not CI-distinguishable.
lit-wilson-wins: lit-oracle-gap
	@echo "$(BLUE)Regenerating Wilson CIs on policy win-counts...$(NC)"
	@python3 -m scripts.experiments.ecg.wilson_win_rates \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/wilson_win_rates.json \
		--md-out      $(WIKI_DATA)/wilson_win_rates.md

# Cohen's h effect sizes on policy win-rate gaps. Complements Wilson
# CIs by quantifying how *big* each separable gap is. cc/GRASP vs
# POPT h=2.346 (largest in corpus); pr/POPT vs LRU/SRRIP h=2.014;
# sssp has no large-effect dominance pair (correctly pinned as a
# kernel where the policy ordering signal is weaker than elsewhere).
lit-cohens-h: lit-oracle-gap
	@echo "$(BLUE)Regenerating Cohen's h effect sizes on win-rates...$(NC)"
	@python3 -m scripts.experiments.ecg.cohens_h_win_rates \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/cohens_h_win_rates.json \
		--md-out      $(WIKI_DATA)/cohens_h_win_rates.md

# Cliff's delta + Mann-Whitney U on the RAW oracle-gap distributions.
# Wilson/Cohen defend win-COUNT claims; this gate defends the
# gap-MAGNITUDE claims with nonparametric, outlier-robust tests:
# pr/POPT vs LRU d=-0.911 p=0; cc/GRASP dominates all 3 at |d|≥0.474
# with MW p<1e-4; bfs/POPT vs LRU/SRRIP d≤-0.66 p<1e-4. sssp again
# has no large-effect dominance pair (correctly pinned).
lit-gap-effect-size: lit-oracle-gap
	@echo "$(BLUE)Regenerating Cliff's delta + Mann-Whitney on gap distributions...$(NC)"
	@python3 -m scripts.experiments.ecg.oracle_gap_effect_size \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/oracle_gap_effect_size.json \
		--md-out      $(WIKI_DATA)/oracle_gap_effect_size.md

# Per-L3-size policy stability: does the winner persist as L3 grows?
# cc/GRASP stable single winner at 1MB+4MB+8MB; pr/POPT stable single
# winner at 1MB+4MB+8MB; bfs is the canonical regime-change kernel
# (GRASP@1MB → POPT@≥4MB); sssp has no stable winner; bc is gray-zone
# (single top policy 4MB+8MB but tied at 1MB).
lit-l3-stability: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-L3-size policy stability...$(NC)"
	@python3 -m scripts.experiments.ecg.l3_policy_stability \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/l3_policy_stability.json \
		--md-out      $(WIKI_DATA)/l3_policy_stability.md

# Multiple-testing correction: pull every p-value emitted across gates
# 34, 38, and the per-(family,app) test into one family, then apply
# Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) at α=0.05. Pins
# which claims survive multiple-testing correction — the only ones
# that can honestly be called 'significant' in the paper.
lit-mt-correction: lit-popt-vs-grasp-by-family-app lit-gap-effect-size lit-oracle-by-app-bootstrap
	@echo "$(BLUE)Regenerating multiple-testing correction...$(NC)"
	@python3 -m scripts.experiments.ecg.multiple_testing_correction \
		--effect-size-json $(WIKI_DATA)/oracle_gap_effect_size.json \
		--bootstrap-json   $(WIKI_DATA)/oracle_gap_by_app_bootstrap.json \
		--family-app-json  $(WIKI_DATA)/popt_vs_grasp_by_family_app.json \
		--json-out         $(WIKI_DATA)/multiple_testing_correction.json \
		--md-out           $(WIKI_DATA)/multiple_testing_correction.md

# Leave-one-graph-out (LOGO) robustness: drop each graph in turn and
# re-rank winners. Robust = same headline across all drops; fragile =
# at least one drop changes the winner. pr/POPT, cc/GRASP, bc/GRASP
# are LOGO-robust; bfs and sssp are LOGO-fragile (consistent with
# gates 36-39 honest negative pins).
lit-logo-robust: lit-oracle-gap
	@echo "$(BLUE)Regenerating leave-one-graph-out robustness...$(NC)"
	@python3 -m scripts.experiments.ecg.leave_one_graph_out \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/leave_one_graph_out.json \
		--md-out      $(WIKI_DATA)/leave_one_graph_out.md

# Cell-winner census: how decisive is the corpus? 97% unique winners,
# 3% tied (all in bc/email-Eu-core), 0% no-winner. Pins the corpus
# decisiveness so any silent regression in winner clarity is caught.
lit-cell-census: lit-oracle-gap
	@echo "$(BLUE)Regenerating cell-winner census...$(NC)"
	@python3 -m scripts.experiments.ecg.cell_winner_census \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/cell_winner_census.json \
		--md-out      $(WIKI_DATA)/cell_winner_census.md

# Family geomean improvement vs LRU. Per (family, app, policy != LRU)
# the bootstrap CI on the geomean (miss_rate / miss_rate_LRU) over
# (graph, L3) cells. Quantifies the SIZE of the improvement that the
# significance gates only show direction for. 34/63 records CI-strict
# improvements, 0 CI-strict regressions. Marquee: citation/pr/POPT
# -32% miss-rate (CI [-43, -13]).
lit-family-geomean: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-(family,app,policy) geomean improvement...$(NC)"
	@python3 -m scripts.experiments.ecg.family_geomean_improvement \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/family_geomean_improvement.json \
		--md-out      $(WIKI_DATA)/family_geomean_improvement.md

# Per-(graph, app) winner stability across paper L3 sizes. Drills into
# the cells gate 39 averages over: 13/34 cells have a single winner
# stable across every L3 size present, 14/34 exhibit regime change.
# web-Google is maximally volatile (all 5 apps flip); soc-LiveJournal1
# + cit-Patents are the most reliable (4/5 stable each).
lit-per-graph-app-stability: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-(graph,app) L3 stability...$(NC)"
	@python3 -m scripts.experiments.ecg.per_graph_app_stability \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/per_graph_app_stability.json \
		--md-out      $(WIKI_DATA)/per_graph_app_stability.md

# Corpus tier/family balance audit. Pins composition: 8 graphs across
# 5 families with social dominance (4/8 = 50% by graph count, 60% by
# paper-L3 cells). Defends against 'unbalanced corpus' reviewer pushback
# by publishing the exact numbers and Pielou evenness (0.86).
lit-corpus-balance: lit-oracle-gap
	@echo "$(BLUE)Regenerating corpus balance audit...$(NC)"
	@python3 -m scripts.experiments.ecg.corpus_balance \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/corpus_balance.json \
		--md-out      $(WIKI_DATA)/corpus_balance.md

# Per-policy miss-rate distribution diagnostics. Defends bootstrap CI
# validity (gates 35 + 43) against reviewer pushback on tail behavior.
# Computes skewness + excess kurtosis per (app, policy) at paper L3 and
# pins them inside Hesterberg's published bootstrap-CI envelope
# (|skew| < 2, |excess kurt| < 7).
lit-distribution-diagnostics: lit-oracle-gap
	@echo "$(BLUE)Regenerating distribution diagnostics...$(NC)"
	@python3 -m scripts.experiments.ecg.distribution_diagnostics \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/distribution_diagnostics.json \
		--md-out      $(WIKI_DATA)/distribution_diagnostics.md

# Leave-one-family-out (LOFO) robustness. Strictly stronger sibling of
# gate 41 (LOGO): drops an entire family at a time and re-ranks per app.
# Pins which apps are LOFO-robust (bc/cc/pr) and which are honestly
# family-sensitive (bfs sensitive to social, sssp sensitive to citation).
lit-lofo-robustness: lit-oracle-gap
	@echo "$(BLUE)Regenerating LOFO robustness...$(NC)"
	@python3 -m scripts.experiments.ecg.lofo_robustness \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/lofo_robustness.json \
		--md-out      $(WIKI_DATA)/lofo_robustness.md

# Per-(app, L3) winner-margin gradient. Classifies every (app, L3) cell
# as decisive / moderate / weak / tied based on top-wins minus runner-up
# wins. Defends against 'your winner is one cell from flipping' reviewer
# pushback by surfacing the exact margin per cell.
lit-winner-margin-gradient: lit-oracle-gap
	@echo "$(BLUE)Regenerating winner margin gradient...$(NC)"
	@python3 -m scripts.experiments.ecg.winner_margin_gradient \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/winner_margin_gradient.json \
		--md-out      $(WIKI_DATA)/winner_margin_gradient.md

# Per-(app, policy) oracle-gap area-under-curve across L3 sweep.
# Collapses each policy trajectory across 1MB->4MB->8MB into one
# trapezoidal AUC score (gap_pp x log2(MB)). Smaller AUC = closer to
# offline oracle on average; gives a single-number per-policy ranking
# that complements the cell-vote view (gate 47/48).
lit-oracle-gap-auc: lit-oracle-gap
	@echo "$(BLUE)Regenerating oracle-gap AUC...$(NC)"
	@python3 -m scripts.experiments.ecg.oracle_gap_auc \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/oracle_gap_auc.json \
		--md-out      $(WIKI_DATA)/oracle_gap_auc.md

# Cross-app policy-AUC correlation matrix. Reads gate 49's AUC vectors,
# z-normalizes within each app, and runs pairwise Pearson correlation
# across the 4 policy dimensions. Shows the apps cluster into two
# AUC-winner groups (POPT for bfs/pr/sssp, GRASP for bc/cc) with
# strong intra-cluster correlation, defending the paper's two-class
# headline framing.
lit-policy-auc-correlation: lit-oracle-gap-auc
	@echo "$(BLUE)Regenerating policy AUC correlation matrix...$(NC)"
	@python3 -m scripts.experiments.ecg.policy_auc_correlation \
		--auc-json $(WIKI_DATA)/oracle_gap_auc.json \
		--json-out $(WIKI_DATA)/policy_auc_correlation.json \
		--md-out   $(WIKI_DATA)/policy_auc_correlation.md

# Per-policy stability index across apps. Reads gate 49's AUC vectors
# and computes coefficient of variation across the 5 paper apps per
# policy. Surfaces which policy is the 'safest all-rounder' (lowest
# CV, even if mediocre) versus the 'high-variance specialist' (lowest
# mean AUC but biggest swings). Also tracks rank per app and the
# 'always-in-top-2' safe-default flag.
lit-policy-stability: lit-oracle-gap-auc
	@echo "$(BLUE)Regenerating policy stability index...$(NC)"
	@python3 -m scripts.experiments.ecg.policy_stability \
		--auc-json $(WIKI_DATA)/oracle_gap_auc.json \
		--json-out $(WIKI_DATA)/policy_stability.json \
		--md-out   $(WIKI_DATA)/policy_stability.md

# Per-(app, policy) cache-sensitivity slope across L3 octaves. Reads
# gate 49's trajectories and computes per-octave slope (gap_pp
# shrinkage per log2(MB) step) + monotonicity flag. Surfaces the
# paper-grade finding that 'significant anti-scaling' (gap GROWS by
# >=1.0 pp at any octave) is exclusively confined to LRU/SRRIP — the
# oracle-aware policies GRASP and POPT never regress as L3 grows.
lit-cache-sensitivity-slope: lit-oracle-gap-auc
	@echo "$(BLUE)Regenerating cache-sensitivity slope...$(NC)"
	@python3 -m scripts.experiments.ecg.cache_sensitivity_slope \
		--auc-json $(WIKI_DATA)/oracle_gap_auc.json \
		--json-out $(WIKI_DATA)/cache_sensitivity_slope.json \
		--md-out   $(WIKI_DATA)/cache_sensitivity_slope.md

# Per-graph oracle-gap cache-sensitivity slope (gate 53). Refines
# gate 52 by re-running the anti-scaling analysis at the per-graph
# level instead of the corpus-averaged level. Surfaces 7 specific
# (graph, app) cells where GRASP/POPT do regress — exceptions to
# the corpus-averaged "oracle-aware never regress" story that the
# paper should disclose transparently.
lit-per-graph-cache-slope: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-graph cache-sensitivity slope...$(NC)"
	@python3 -m scripts.experiments.ecg.per_graph_cache_slope \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/per_graph_cache_slope.json \
		--md-out      $(WIKI_DATA)/per_graph_cache_slope.md

# Cross-generator gap_pp parity (gate 54). Reconciles three load-bearing
# aggregators (oracle_gap, oracle_gap_auc, cache_sensitivity_slope) to
# verify they all report identical gap_pp values (within 1e-3 pp) for
# every shared (app, policy, L3) triple. Catches silent staleness or
# aggregation drift that would invisibly invalidate paper narrative.
lit-cross-generator-gap-parity: lit-oracle-gap lit-oracle-gap-auc lit-cache-sensitivity-slope
	@echo "$(BLUE)Regenerating cross-generator gap_pp parity...$(NC)"
	@python3 -m scripts.experiments.ecg.cross_generator_gap_parity \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--auc-json    $(WIKI_DATA)/oracle_gap_auc.json \
		--slope-json  $(WIKI_DATA)/cache_sensitivity_slope.json \
		--json-out    $(WIKI_DATA)/cross_generator_gap_parity.json \
		--md-out      $(WIKI_DATA)/cross_generator_gap_parity.md

# Cache-saturation onset detection (gate 55). For each (app, policy)
# trajectory, identifies the L3 size beyond which additional cache
# buys negligible gap improvement. Paper-grade mechanism: POPT
# saturates earliest (already near-oracle from small caches); LRU
# rarely saturates within paper L3 (always benefits from doubling).
lit-cache-saturation-onset: lit-oracle-gap-auc
	@echo "$(BLUE)Regenerating cache-saturation onset detection...$(NC)"
	@python3 -m scripts.experiments.ecg.cache_saturation_onset \
		--auc-json $(WIKI_DATA)/oracle_gap_auc.json \
		--json-out $(WIKI_DATA)/cache_saturation_onset.json \
		--md-out   $(WIKI_DATA)/cache_saturation_onset.md

# Per-(app, L3, policy) gap-distribution shape envelope. Extends gate
# 46 from pooled marginals to every cell of the paper grid. Pins the
# 14-cell pinned-exception set where the textbook envelope is exceeded
# due to discrete near-zero+single-outlier patterns rather than
# heavy-tailed continuous data. A new cell leaving the pin = regression.
lit-gap-distribution-shape: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-cell gap-distribution shape envelope...$(NC)"
	@python3 -m scripts.experiments.ecg.gap_distribution_shape \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/gap_distribution_shape.json \
		--md-out      $(WIKI_DATA)/gap_distribution_shape.md

# Per-family policy-AUC clustering replay. Re-derives the AUC winner
# per app using only graphs in each family and checks whether the
# global POPT-friendly / GRASP-friendly clustering is intrinsic or a
# composition artefact. Today 3 qualifying families (citation, social,
# web) replay the global clustering with at most 2 pinned deviations
# (citation/bfs, citation/sssp — cit-Patents lower out-degree skew).
lit-family-policy-auc-clustering: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-family policy-AUC clustering replay...$(NC)"
	@python3 -m scripts.experiments.ecg.family_policy_auc_clustering \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/family_policy_auc_clustering.json \
		--md-out      $(WIKI_DATA)/family_policy_auc_clustering.md

# Per-(app, policy) discrete second derivative of the oracle-gap
# trajectory at the 4MB midpoint. Pins that oracle-aware policies
# (GRASP, POPT) have strictly more knees than non-oracle (LRU, SRRIP).
# Complements gate 55 saturation onset with the curvature signal.
lit-oracle-gap-curvature: lit-oracle-gap-auc lit-cache-saturation-onset
	@echo "$(BLUE)Regenerating oracle-gap trajectory curvature...$(NC)"
	@python3 -m scripts.experiments.ecg.oracle_gap_curvature \
		--auc-json $(WIKI_DATA)/oracle_gap_auc.json \
		--json-out $(WIKI_DATA)/oracle_gap_curvature.json \
		--md-out   $(WIKI_DATA)/oracle_gap_curvature.md

# Per-(app, graph) Kendall-tau rank correlation across the L3 octave.
# Asks "does the policy ranking at 1MB predict the ranking at 8MB?".
# Pins six cells where rank flips: three GRASP thrash@1MB→winner@4MB,
# two large-cache fit-in-WSS counter-productivity, one pico-corpus.
lit-policy-rank-kendall: lit-oracle-gap
	@echo "$(BLUE)Regenerating policy-rank Kendall-tau across L3 octave...$(NC)"
	@python3 -m scripts.experiments.ecg.policy_rank_kendall \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/policy_rank_kendall.json \
		--md-out      $(WIKI_DATA)/policy_rank_kendall.md

# Per-policy plateau location in WSS-relative L3 capacity. Walks the
# regime ladder (under_wss → near_wss → over_wss) and pins the first
# regime where each policy's median gap-to-oracle falls below 0.5pp.
# Verdict PASS iff oracle-aware policies plateau STRICTLY earlier than
# non-oracle. Today: GRASP/POPT plateau at under_wss; LRU/SRRIP only at
# over_wss — a full two ladder steps of separation.
lit-wss-knee-location: lit-wss-relative-l3
	@echo "$(BLUE)Regenerating WSS-relative knee location...$(NC)"
	@python3 -m scripts.experiments.ecg.wss_knee_location \
		--wss-json $(WIKI_DATA)/wss_relative_l3.json \
		--json-out $(WIKI_DATA)/wss_knee_location.json \
		--md-out   $(WIKI_DATA)/wss_knee_location.md

# Per-family replay of the gate 58 curvature signal. Each qualifying
# family must independently exhibit the global pattern that
# oracle-aware policies bend toward a plateau while non-oracle policies
# keep accelerating.
lit-family-curvature-replay: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-family curvature replay...$(NC)"
	@python3 -m scripts.experiments.ecg.family_curvature_replay \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/family_curvature_replay.json \
		--md-out      $(WIKI_DATA)/family_curvature_replay.md

# Per-(policy, WSS regime) distribution of winner margin (in pp of
# miss-rate) over second-best. Records the paper's central claim that
# oracle-aware policies' winning margins SHRINK as capacity loosens —
# proof that the payoff is biggest under pressure.
lit-winner-margin-by-regime: lit-oracle-gap lit-wss-relative-l3
	@echo "$(BLUE)Regenerating winner-margin distribution by WSS regime...$(NC)"
	@python3 -m scripts.experiments.ecg.winner_margin_by_regime \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--wss-json    $(WIKI_DATA)/wss_relative_l3.json \
		--json-out    $(WIKI_DATA)/winner_margin_by_regime.json \
		--md-out      $(WIKI_DATA)/winner_margin_by_regime.md

# Per-family replay of gate 62: does each graph family that has the
# diversity to test it independently reproduce the global margin-shrink
# pattern (oracle-aware policies win bigger under pressure)? Mirror of
# gates 57<->50 and 61<->58 (family-level analog of a corpus signal).
lit-family-margin-replay: lit-oracle-gap lit-wss-relative-l3
	@echo "$(BLUE)Regenerating per-family winner-margin replay...$(NC)"
	@python3 -m scripts.experiments.ecg.family_margin_replay \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--wss-json    $(WIKI_DATA)/wss_relative_l3.json \
		--json-out    $(WIKI_DATA)/family_margin_replay.json \
		--md-out      $(WIKI_DATA)/family_margin_replay.md

# Head-to-head asymmetry between every (A, B) policy pair: when A
# wins H2H against B, by how much? Pins how lopsided the loser-side
# magnitudes can grow; today the worst pair is 3.20x and the ceiling
# is a generous 20x sanity bound.
lit-cross-policy-asymmetry: lit-oracle-gap
	@echo "$(BLUE)Regenerating cross-policy mean-margin asymmetry...$(NC)"
	@python3 -m scripts.experiments.ecg.cross_policy_asymmetry \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/cross_policy_asymmetry.json \
		--md-out      $(WIKI_DATA)/cross_policy_asymmetry.md

# Per-app distance from saturation at 4MB->8MB: how much best-policy
# miss-rate remains recoverable by doubling the cache from 4 to 8 MB
# for each application. Records app-level memory-pressure diversity
# (bc 17.5pp median vs bfs 4.6pp median today).
lit-saturation-distance: lit-oracle-gap lit-wss-relative-l3
	@echo "$(BLUE)Regenerating per-app saturation distance...$(NC)"
	@python3 -m scripts.experiments.ecg.saturation_distance \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--wss-json    $(WIKI_DATA)/wss_relative_l3.json \
		--json-out    $(WIKI_DATA)/saturation_distance.json \
		--md-out      $(WIKI_DATA)/saturation_distance.md

# Per-policy capacity sensitivity: OLS slope of miss-rate (pp) vs
# log2(L3 MB) across {1MB, 4MB, 8MB} for every (app, graph, policy)
# cell. Pins LRU as the most-capacity-hungry policy and GRASP as the
# least (oracle-aware extracts more at small caches).
lit-capacity-sensitivity: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-policy capacity-sensitivity slope...$(NC)"
	@python3 -m scripts.experiments.ecg.capacity_sensitivity \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/capacity_sensitivity.json \
		--md-out      $(WIKI_DATA)/capacity_sensitivity.md

# Per-family replay of the gate 66 slope ordering. For each graph
# family with full 1MB/4MB/8MB coverage, recompute per-policy median
# slope and confirm LRU/SRRIP both strictly steeper than GRASP. The
# social family is pinned as a known deviation because email-Eu-core
# saturates at every L3, washing out its slope contribution.
lit-family-slope-replay: lit-oracle-gap lit-capacity-sensitivity
	@echo "$(BLUE)Regenerating per-family capacity-sensitivity slope replay...$(NC)"
	@python3 -m scripts.experiments.ecg.family_slope_replay \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/family_slope_replay.json \
		--md-out      $(WIKI_DATA)/family_slope_replay.md

# Per-app capacity-sensitivity slope. Breaks gate 66 out per kernel
# and ranks apps by cache-hungriness via the median-of-medians slope.
# sssp most cache-hungry; bfs least (saturates early). bfs pinned
# because its frontier-driven access pattern inverts the GRASP-vs-LRU
# slope ordering.
lit-per-app-capacity-slope: lit-oracle-gap lit-capacity-sensitivity
	@echo "$(BLUE)Regenerating per-app capacity-sensitivity slope...$(NC)"
	@python3 -m scripts.experiments.ecg.per_app_capacity_slope \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/per_app_capacity_slope.json \
		--md-out      $(WIKI_DATA)/per_app_capacity_slope.md

# Saturation distance vs capacity-sensitivity slope cross-check.
# Both metrics derive from the same per-(app, graph, policy) miss
# curve; this gate validates that they are positively correlated
# (Pearson r ~0.51, Spearman ~0.45) and on the same per-octave scale
# (median ratio ~0.96). It is a regression test for the gate 65 and
# gate 66 generators.
lit-slope-saturation-xcheck: lit-oracle-gap
	@echo "$(BLUE)Regenerating slope-vs-saturation cross-check...$(NC)"
	@python3 -m scripts.experiments.ecg.slope_saturation_xcheck \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/slope_saturation_xcheck.json \
		--md-out      $(WIKI_DATA)/slope_saturation_xcheck.md

# Gem5 anchor slope sanity gate. Computes OLS slope across the four
# gem5 anchor L3 sizes (4kB/32kB/256kB/2MB), verifies cache
# monotonicity, that every per-policy median is negative, that SRRIP
# is at least as steep as GRASP, and that GRASP is below the
# help-floor. The LRU-vs-GRASP delta is reported as INFORMATIONAL —
# sub-WSS scales (4kB < email-Eu-core WSS ~4.5kB) can invert the
# slope ordering relative to the cache-sim 1-8MB sweep.
lit-gem5-slope-replay: gem5-anchor
	@echo "$(BLUE)Regenerating gem5 anchor slope sanity gate...$(NC)"
	@python3 -m scripts.experiments.ecg.gem5_slope_replay \
		--anchor-json $(WIKI_DATA)/gem5_anchor.json \
		--json-out    $(WIKI_DATA)/gem5_slope_replay.json \
		--md-out      $(WIKI_DATA)/gem5_slope_replay.md

# Sniper anchor slope sanity gate. Mirror of gate 70 for Sniper.
# Six (app, graph) cells (bfs/pr/sssp at cit-Patents and
# email-Eu-core); same verdict checks: cache monotonicity, all
# per-policy medians negative, SRRIP at-least-as-steep-as GRASP,
# GRASP below help-floor.
lit-sniper-slope-replay: sniper-anchor
	@echo "$(BLUE)Regenerating sniper anchor slope sanity gate...$(NC)"
	@python3 -m scripts.experiments.ecg.sniper_slope_replay \
		--anchor-json $(WIKI_DATA)/sniper_anchor.json \
		--json-out    $(WIKI_DATA)/sniper_slope_replay.json \
		--md-out      $(WIKI_DATA)/sniper_slope_replay.md

# Cross-tool SRRIP-vs-GRASP slope ordering invariant. Reads gate 66
# (cache-sim), gate 70 (gem5 anchor), gate 71 (sniper anchor)
# per-policy medians and verifies SRRIP <= GRASP in EVERY tool, with
# at least 2 of 3 showing a strict gap >= 0.05 pp/oct.
lit-cross-tool-slope-ordering: lit-capacity-sensitivity lit-gem5-slope-replay lit-sniper-slope-replay
	@echo "$(BLUE)Regenerating cross-tool slope ordering gate...$(NC)"
	@python3 -m scripts.experiments.ecg.cross_tool_slope_ordering \
		--cache-sim-json $(WIKI_DATA)/capacity_sensitivity.json \
		--gem5-json      $(WIKI_DATA)/gem5_slope_replay.json \
		--sniper-json    $(WIKI_DATA)/sniper_slope_replay.json \
		--json-out       $(WIKI_DATA)/cross_tool_slope_ordering.json \
		--md-out         $(WIKI_DATA)/cross_tool_slope_ordering.md

# Per-app SRRIP-vs-GRASP slope ordering. Reads gate 68's per_app
# artifact and verifies that for every app NOT in PINNED_DEVIATING_APPS,
# SRRIP is at least as steep as GRASP (slack 1.0 pp/oct). bfs remains
# pinned (frontier-driven streaming pathology — gate 65 most-saturated).
lit-per-app-srrip-vs-grasp: lit-per-app-capacity-slope
	@echo "$(BLUE)Regenerating per-app SRRIP-vs-GRASP gate...$(NC)"
	@python3 -m scripts.experiments.ecg.per_app_srrip_vs_grasp \
		--per-app-json $(WIKI_DATA)/per_app_capacity_slope.json \
		--json-out     $(WIKI_DATA)/per_app_srrip_vs_grasp.json \
		--md-out       $(WIKI_DATA)/per_app_srrip_vs_grasp.md

# Cross-tool LRU-vs-GRASP regime inversion. Formalizes the regime-
# dependent finding from gates 70/71/72 INFORMATIONAL fields:
# cache-sim post-WSS shows LRU strictly steeper than GRASP, while
# both anchor tools sub-WSS show the opposite ordering. PASS confirms
# the regime physics is consistent and reproducible across tools.
lit-cross-tool-lru-regime: lit-capacity-sensitivity lit-gem5-slope-replay lit-sniper-slope-replay
	@echo "$(BLUE)Regenerating cross-tool LRU regime gate...$(NC)"
	@python3 -m scripts.experiments.ecg.cross_tool_lru_regime \
		--cache-sim-json $(WIKI_DATA)/capacity_sensitivity.json \
		--gem5-json      $(WIKI_DATA)/gem5_slope_replay.json \
		--sniper-json    $(WIKI_DATA)/sniper_slope_replay.json \
		--json-out       $(WIKI_DATA)/cross_tool_lru_regime.json \
		--md-out         $(WIKI_DATA)/cross_tool_lru_regime.md

# Per-app saturation-vs-slope extremum corroboration. Reads gate 65
# (saturation_distance per_app) and gate 68 (per-app slope) and
# verifies that bfs is the unique extremum on BOTH metrics (least
# cache-sensitive by distance AND by slope). Most-cache-hungry
# disagreement (sssp by slope, bc by distance) is reported as
# INFORMATIONAL — the regime-vs-aggregate distinction in action.
lit-saturation-slope-extremum: lit-saturation-distance lit-per-app-capacity-slope
	@echo "$(BLUE)Regenerating saturation-vs-slope extremum gate...$(NC)"
	@python3 -m scripts.experiments.ecg.saturation_slope_extremum \
		--distance-json $(WIKI_DATA)/saturation_distance.json \
		--slope-json    $(WIKI_DATA)/per_app_capacity_slope.json \
		--json-out      $(WIKI_DATA)/saturation_slope_extremum.json \
		--md-out        $(WIKI_DATA)/saturation_slope_extremum.md

# Cross-tool slope-sign universality roll-up: every (tool, policy)
# median capacity-sensitivity slope must be negative, in the [-25,
# -0.5] pp/oct band, and the per-tool steepness span must not exceed
# 5 pp/oct. Bundles gates 66 + 70 + 71 into a single roll-up that
# also catches partial regressions where one policy collapses while
# siblings stay healthy.
lit-cross-tool-slope-universality: lit-capacity-sensitivity lit-gem5-slope-replay lit-sniper-slope-replay
	@echo "$(BLUE)Regenerating cross-tool slope-sign universality gate...$(NC)"
	@python3 -m scripts.experiments.ecg.cross_tool_slope_universality \
		--cache-sim-json $(WIKI_DATA)/capacity_sensitivity.json \
		--gem5-json      $(WIKI_DATA)/gem5_slope_replay.json \
		--sniper-json    $(WIKI_DATA)/sniper_slope_replay.json \
		--json-out       $(WIKI_DATA)/cross_tool_slope_universality.json \
		--md-out         $(WIKI_DATA)/cross_tool_slope_universality.md

# Cell-level L3-sweep monotonicity universality (cache-sim): every
# (graph, app, policy) cell's miss_rate must be non-increasing as L3
# grows, within a documented measurement-noise tolerance (0.5 pp).
# Foundational soundness gate that downstream slope/distance/
# sensitivity gates depend on.
lit-monotonicity-universality: lit-oracle-gap
	@echo "$(BLUE)Regenerating L3 sweep monotonicity gate...$(NC)"
	@python3 -m scripts.experiments.ecg.monotonicity_universality \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/monotonicity_universality.json \
		--md-out      $(WIKI_DATA)/monotonicity_universality.md

# Anchor cell-pair census: pins gem5 (2 cells) and Sniper (6 cells)
# coverage, L3 axis, and policy set against silent shrinkage that
# would break downstream cross-tool gates (70/71/72/74/76).
lit-anchor-cell-census: lit-gem5-slope-replay lit-sniper-slope-replay
	@echo "$(BLUE)Regenerating anchor cell-pair census...$(NC)"
	@python3 -m scripts.experiments.ecg.anchor_cell_census \
		--gem5-json   $(WIKI_DATA)/gem5_slope_replay.json \
		--sniper-json $(WIKI_DATA)/sniper_slope_replay.json \
		--json-out    $(WIKI_DATA)/anchor_cell_census.json \
		--md-out      $(WIKI_DATA)/anchor_cell_census.md

# Per-family saturation-distance replay: mirrors gate 67 for the
# distance metric. Verifies each family's 4MB->8MB miss-rate drop
# median is non-negative; citation/social meet a 5 pp floor; web is
# pinned as the low-headroom outlier; ordering citation >= social >=
# web holds within a 1 pp slack.
lit-family-saturation-distance: lit-saturation-distance
	@echo "$(BLUE)Regenerating per-family saturation-distance replay...$(NC)"
	@python3 -m scripts.experiments.ecg.family_saturation_distance \
		--distance-json $(WIKI_DATA)/saturation_distance.json \
		--json-out      $(WIKI_DATA)/family_saturation_distance.json \
		--md-out        $(WIKI_DATA)/family_saturation_distance.md

# Anchor cell-level L3-sweep monotonicity replay (gate 80). Walks every
# (graph, app, policy) anchor cell in the gem5 and Sniper slope replays
# and asserts tier-aware monotonicity: gem5 is strictly monotone, sniper
# is permitted bounded bumps under per-tool ceilings, and no tool may
# exhibit a catastrophic (>=3 pp) regression at any L3 step.
lit-anchor-monotonicity-replay: lit-gem5-slope-replay lit-sniper-slope-replay
	@echo "$(BLUE)Regenerating anchor monotonicity replay...$(NC)"
	@python3 -m scripts.experiments.ecg.anchor_monotonicity_replay \
		--gem5-json   $(WIKI_DATA)/gem5_slope_replay.json \
		--sniper-json $(WIKI_DATA)/sniper_slope_replay.json \
		--json-out    $(WIKI_DATA)/anchor_monotonicity_replay.json \
		--md-out      $(WIKI_DATA)/anchor_monotonicity_replay.md

# Per-policy final-octave steepness ranking (gate 81). Aggregates the
# 4MB->8MB |slope| per policy across apps from cache_saturation_onset.json
# and locks the saturation-rank inversion: POPT/GRASP medians stay below
# 0.5 pp/oct, LRU/SRRIP medians stay above 0.5 pp/oct, and oracle-aware
# median must be < half non-oracle median (currently 0.16 vs 1.07 = 0.15x).
lit-policy-steepness-ranking: lit-cache-saturation-onset
	@echo "$(BLUE)Regenerating per-policy steepness ranking...$(NC)"
	@python3 -m scripts.experiments.ecg.policy_steepness_ranking \
		--onset-json $(WIKI_DATA)/cache_saturation_onset.json \
		--json-out   $(WIKI_DATA)/policy_steepness_ranking.json \
		--md-out     $(WIKI_DATA)/policy_steepness_ranking.md

# Cross-tool shared-anchor slope-sign agreement (gate 82). For every
# (graph, app, policy) cell present in BOTH gem5 and sniper anchors,
# asserts 100% sign-match, 100% both-negative, 100% sniper-steeper, and
# |sniper-gem5| <= 8 pp/oct. Locks physical replication on the shared
# (email-Eu-core, pr) x {GRASP, LRU, SRRIP} cells.
lit-anchor-cross-tool-agreement: lit-gem5-slope-replay lit-sniper-slope-replay
	@echo "$(BLUE)Regenerating cross-tool shared-anchor agreement...$(NC)"
	@python3 -m scripts.experiments.ecg.anchor_cross_tool_agreement \
		--gem5-json   $(WIKI_DATA)/gem5_slope_replay.json \
		--sniper-json $(WIKI_DATA)/sniper_slope_replay.json \
		--json-out    $(WIKI_DATA)/anchor_cross_tool_agreement.json \
		--md-out      $(WIKI_DATA)/anchor_cross_tool_agreement.md

# Per-kernel oracle-gap bootstrap CIs. Turns the per-app point
# estimates from lit-oracle-gap-by-app into CI-backed sign claims:
# pr→POPT < all (P=1.0), cc→GRASP < POPT (P=0.9995), bfs→POPT <
# GRASP (P=0.999), sssp→POPT < GRASP (P=0.971). Defends per-kernel
# narrative against 'is that difference real?' reviewer pushback.
lit-oracle-by-app-bootstrap: lit-oracle-gap
	@echo "$(BLUE)Regenerating per-kernel oracle-gap bootstrap CIs...$(NC)"
	@python3 -m scripts.experiments.ecg.oracle_gap_by_app_bootstrap \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/oracle_gap_by_app_bootstrap.json \
		--md-out      $(WIKI_DATA)/oracle_gap_by_app_bootstrap.md

# WSS-relative L3 axis. Bins each (graph, app, L3) cell by L3 / WSS
# ratio (WSS proxy from corpus_diversity working_set_ratio × 1 MB).
# Defends against "absolute L3 bytes obscure cross-graph comparisons"
# reviewer pushback by re-aggregating winners across under/near/over
# WSS regimes.
lit-wss-relative-l3: lit-oracle-gap
	@echo "$(BLUE)Regenerating WSS-relative L3 axis...$(NC)"
	@python3 -m scripts.experiments.ecg.wss_relative_l3 \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--corpus-json $(WIKI_DATA)/corpus_diversity.json \
		--json-out    $(WIKI_DATA)/wss_relative_l3.json \
		--md-out      $(WIKI_DATA)/wss_relative_l3.md

# Bootstrap confidence intervals on the load-bearing claims (POPT-vs-
# GRASP per family, sign-stability of mean orderings). Depends on the
# oracle_gap + popt_vs_grasp inputs being current.
lit-bootstrap-ci: lit-oracle-gap lit-popt-vs-grasp
	@echo "$(BLUE)Regenerating bootstrap CIs on load-bearing claims...$(NC)"
	@python3 -m scripts.experiments.ecg.bootstrap_ci \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--delta-json  $(WIKI_DATA)/popt_vs_grasp_delta.json \
		--json-out    $(WIKI_DATA)/bootstrap_ci.json \
		--md-out      $(WIKI_DATA)/bootstrap_ci.md

# Family-classification sensitivity sweep. For each (graph,
# alternative family) reassignment, rerun the bootstrap sign-
# stability claims and report which paper claims survive every
# relabeling vs which depend on a specific taxonomy choice.
# Defends the paper against the obvious reviewer challenge
# "what if you relabeled graph X as family Y?".
lit-family-sensitivity: lit-oracle-gap
	@echo "$(BLUE)Regenerating family-classification sensitivity...$(NC)"
	@python3 -m scripts.experiments.ecg.family_sensitivity \
		--oracle-json $(WIKI_DATA)/oracle_gap.json \
		--json-out    $(WIKI_DATA)/family_sensitivity.json \
		--md-out      $(WIKI_DATA)/family_sensitivity.md

# Paper-artifact catalog: single canonical index of every aggregator
# + its generator + governing gate + JSON artifact + headline finding.
# No data dependencies — just metadata + on-disk file existence audit.
lit-catalog:
	@echo "$(BLUE)Regenerating paper-artifact catalog...$(NC)"
	@python3 -m scripts.experiments.ecg.artifact_catalog \
		--json-out $(WIKI_DATA)/artifact_catalog.json \
		--md-out   $(WIKI_DATA)/artifact_catalog.md

# Reproducibility smoke: SHA-256-snapshot all tracked wiki/data
# aggregator artifacts, re-run the deterministic generator chain,
# and assert byte-identity. Catches the failure mode where a
# committed file silently goes stale relative to its generator.
# Allowed volatility (e.g. dashboard runtime_s) is masked before
# hashing — every other byte must be reproducible from inputs.
lit-reproduce-smoke:
	@echo "$(BLUE)Running reproducibility smoke...$(NC)"
	@python3 -m scripts.experiments.ecg.reproduce_smoke \
		--json-out $(WIKI_DATA)/reproduce_smoke.json \
		--md-out   $(WIKI_DATA)/reproduce_smoke.md

# Single source of truth for the paper's numerical claims. Reads all
# the paper-grade aggregator JSONs and emits a consolidated registry
# linking each claim to value + source + governing gate. Must run
# AFTER all other aggregators so the values are current.
lit-claims: lit-faith lit-repro lit-winner lit-thrash lit-cross-tool lit-cross-tool-winners lit-density lit-popt-vs-grasp lit-popt-vs-grasp-by-family-app lit-wilson-wins lit-cohens-h lit-gap-effect-size lit-l3-stability lit-mt-correction lit-logo-robust lit-cell-census lit-family-geomean lit-per-graph-app-stability lit-corpus-balance lit-distribution-diagnostics lit-lofo-robustness lit-winner-margin-gradient lit-oracle-gap-auc lit-policy-auc-correlation lit-policy-stability lit-cache-sensitivity-slope lit-per-graph-cache-slope lit-cross-generator-gap-parity lit-cache-saturation-onset lit-gap-distribution-shape lit-family-policy-auc-clustering lit-oracle-gap-curvature lit-policy-rank-kendall lit-wss-knee-location lit-family-curvature-replay lit-winner-margin-by-regime lit-family-margin-replay lit-cross-policy-asymmetry lit-saturation-distance lit-capacity-sensitivity lit-family-slope-replay lit-per-app-capacity-slope lit-slope-saturation-xcheck lit-gem5-slope-replay lit-sniper-slope-replay lit-cross-tool-slope-ordering lit-per-app-srrip-vs-grasp lit-cross-tool-lru-regime lit-saturation-slope-extremum lit-cross-tool-slope-universality lit-monotonicity-universality lit-anchor-cell-census lit-family-saturation-distance lit-anchor-monotonicity-replay lit-policy-steepness-ranking lit-anchor-cross-tool-agreement lit-deviations lit-diversity lit-margin lit-signmass lit-citations lit-knowndev lit-tolerance lit-accesses lit-citexapp lit-monotonicity lit-stat lit-polyord lit-devexp lit-ratgrid lit-cellcomp lit-appfreq lit-regimesign lit-citdate lit-ecg-parity lit-regime-taxonomy lit-oracle-gap lit-oracle-gap-by-app lit-oracle-by-app-bootstrap lit-wss-relative-l3 lit-bootstrap-ci lit-family-sensitivity lit-catalog
	@echo "$(BLUE)Regenerating paper claims registry...$(NC)"
	@python3 -m scripts.experiments.ecg.paper_claims_registry \
		--json-out $(WIKI_DATA)/paper_claims.json \
		--md-out   $(WIKI_DATA)/paper_claims.md

confidence: lit-faith lit-repro lit-budget lit-table lit-winner lit-thrash gem5-anchor sniper-anchor lit-cross-tool lit-cross-tool-winners lit-density lit-popt-vs-grasp lit-popt-vs-grasp-by-family-app lit-wilson-wins lit-cohens-h lit-gap-effect-size lit-l3-stability lit-mt-correction lit-logo-robust lit-cell-census lit-family-geomean lit-per-graph-app-stability lit-corpus-balance lit-distribution-diagnostics lit-lofo-robustness lit-winner-margin-gradient lit-oracle-gap-auc lit-policy-auc-correlation lit-policy-stability lit-cache-sensitivity-slope lit-per-graph-cache-slope lit-cross-generator-gap-parity lit-cache-saturation-onset lit-gap-distribution-shape lit-family-policy-auc-clustering lit-oracle-gap-curvature lit-policy-rank-kendall lit-wss-knee-location lit-family-curvature-replay lit-winner-margin-by-regime lit-family-margin-replay lit-cross-policy-asymmetry lit-saturation-distance lit-capacity-sensitivity lit-family-slope-replay lit-per-app-capacity-slope lit-slope-saturation-xcheck lit-gem5-slope-replay lit-sniper-slope-replay lit-cross-tool-slope-ordering lit-per-app-srrip-vs-grasp lit-cross-tool-lru-regime lit-saturation-slope-extremum lit-cross-tool-slope-universality lit-monotonicity-universality lit-anchor-cell-census lit-family-saturation-distance lit-anchor-monotonicity-replay lit-policy-steepness-ranking lit-anchor-cross-tool-agreement lit-deviations lit-diversity lit-margin lit-signmass lit-citations lit-knowndev lit-tolerance lit-accesses lit-citexapp lit-monotonicity lit-stat lit-polyord lit-devexp lit-ratgrid lit-cellcomp lit-appfreq lit-regimesign lit-citdate lit-ecg-parity lit-regime-taxonomy lit-oracle-gap lit-oracle-gap-by-app lit-oracle-by-app-bootstrap lit-wss-relative-l3 lit-bootstrap-ci lit-family-sensitivity lit-catalog lit-claims lit-reproduce-smoke
	@echo "$(BLUE)Rebuilding confidence dashboard...$(NC)"
	@python3 -m scripts.experiments.ecg.confidence_dashboard \
		--markdown $(WIKI_DATA)/confidence_dashboard.md \
		--json-out $(WIKI_DATA)/confidence_dashboard.json
	@echo "$(GREEN)See $(WIKI_DATA)/confidence_dashboard.md$(NC)"

# Quick variant: skip the lit-faith / lit-repro regen (assumes
# the artifacts on disk are already current) and just rerun the
# pytest tier gates + headline verdict. Useful in tight edit loops.
confidence-fast:
	@python3 -m scripts.experiments.ecg.confidence_dashboard \
		--markdown $(WIKI_DATA)/confidence_dashboard.md \
		--json-out $(WIKI_DATA)/confidence_dashboard.json
