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

.PHONY: lit-faith lit-repro lit-budget lit-table lit-winner gem5-anchor sniper-anchor confidence confidence-fast

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

confidence: lit-faith lit-repro lit-budget lit-table lit-winner gem5-anchor sniper-anchor
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
