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
BENCH_DIR   = bench
CONFIG_DIR  = $(SCRIPT_DIR)/config
# =========================================================
RES_DIR    = $(BENCH_DIR)/results
BACKUP_DIR = $(BENCH_DIR)/backups
# =========================================================
BIN_DIR = $(BENCH_DIR)/bin
LIB_DIR = $(BENCH_DIR)/lib
SRC_DIR = $(BENCH_DIR)/src
INC_DIR = $(BENCH_DIR)/include
OBJ_DIR = $(BENCH_DIR)/obj

# =========================================================
INCLUDE_GAPBS  = $(INC_DIR)/gapbs 
INCLUDE_RABBIT = $(INC_DIR)/rabbit
INCLUDE_GORDER = $(INC_DIR)/gorder
INCLUDE_CORDER = $(INC_DIR)/corder
INCLUDE_LEIDEN = $(INC_DIR)/leiden
# =========================================================
INCLUDE_BOOST  = /opt/boost_1_58_0/include  
# =========================================================
DEP_GAPBS  = $(wildcard $(INC_DIR)/gapbs/*.h)
DEP_RABBIT = $(wildcard $(INC_DIR)/rabbit/*.hpp)
DEP_GORDER = $(wildcard $(INC_DIR)/gorder/*.h)
DEP_CORDER = $(wildcard $(INC_DIR)/corder/*.h)
DEP_LEIDEN = $(wildcard $(INC_DIR)/leiden/*.hxx)
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
# TEST PASS OR FAIL
PASS = \033[92mPASS\033[0m
FAIL = \033[91mFAIL\033[0m
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
CXXFLAGS_GAP    = -std=c++17 -O3 -Wall -fopenmp -g
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
INCLUDES = -I$(INCLUDE_GAPBS) -I$(INCLUDE_GORDER) -I$(INCLUDE_CORDER) -I$(INCLUDE_LEIDEN) -I$(INCLUDE_BOOST)
# =========================================================
# Optional RABBIT includes
ifeq ($(RABBIT_ENABLE), 1)
CXXFLAGS += -DRABBIT_ENABLE 
LDLIBS += $(LDLIBS_BOOST) $(LDLIBS_RABBIT)
INCLUDES += -I$(INCLUDE_RABBIT)
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
# GRAPH_BENCH = -f ./test/graphs/4.el
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
	@for o in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do \
		echo "========================================================="; \
		if [ "$(FLUSH_CACHE)" = "1" ]; then
		    echo "Attempting to mitigate cache effects by busy-looping..."
		    dd if=/dev/zero of=/dev/null bs=1M count=1024
		fi;
		OMP_NUM_THREADS=$(PARALLEL) ./$(BIN_DIR)/$* $(GRAPH_BENCH) -s -n 1 -o $$o; \
	done
	
# =========================================================
# Rule to install Python dependencies
install-py-deps: ./$(SCRIPT_DIR)/requirements.txt
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r ./$(SCRIPT_DIR)/requirements.txt

exp-%: install-py-deps all $(BIN_DIR)/converter
	$(PYTHON) ./$(SCRIPT_DIR)/$*/run_experiment.py

graph-%: install-py-deps $(BIN_DIR)/converter
	$(PYTHON) ./$(SCRIPT_DIR)/graph_brew.py $(CONFIG_DIR)/$*/convert.json

# =========================================================
# Compilation Rules
# =========================================================
$(BIN_DIR)/%: $(SRC_DIR)/%.cc $(DEP_GAPBS) $(DEP_RABBIT) $(DEP_GORDER) $(DEP_CORDER) $(DEP_LEIDEN) | $(BIN_DIR)
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
test-topology: $(BIN_DIR)/bfs
	@echo "Running topology verification tests..."
	@python3 $(SCRIPT_DIR)/test_topology.py --graph="-g 12" --quick

test-topology-full: $(BIN_DIR)/bfs
	@echo "Running full topology verification tests..."
	@python3 $(SCRIPT_DIR)/test_topology.py --graph="-g 12"

test-topology-large: $(BIN_DIR)/bfs
	@echo "Running topology verification on larger graph..."
	@python3 $(SCRIPT_DIR)/test_topology.py --graph="-g 16" --quick
scrub-all:
	@rm -rf $(BIN_DIR) $(BACKUP_DIR) $(RES_DIR) 00_* $(EXIT_STATUS) 

clean-results: 
	@mkdir -p $(BACKUP_DIR)
	@if [ -d "$(RES_DIR)" ]; then \
		echo "Backing up results directory..."; \
		tar -czf $(BACKUP_DIR)/result_`date +"%Y%m%d_%H%M%S"`.tar.gz $(RES_DIR); \
		echo "Cleaning results directory..."; \
		rm -rf $(RES_DIR); \
		echo "Backup and clean completed."; \
	fi

# =========================================================
# Cache Simulation Builds
# =========================================================
SRC_SIM_DIR = $(BENCH_DIR)/src_sim
BIN_SIM_DIR = $(BENCH_DIR)/bin_sim
INCLUDE_CACHE = $(INC_DIR)/cache
DEP_CACHE = $(wildcard $(INCLUDE_CACHE)/*.h)

# Simulation kernels (algorithms with cache instrumentation)
KERNELS_SIM = pr bfs bc cc sssp tc

# Create bin_sim directory
$(BIN_SIM_DIR):
	mkdir -p $@

# Build simulation versions
$(BIN_SIM_DIR)/%: $(SRC_SIM_DIR)/%.cc $(DEP_GAPBS) $(DEP_CACHE) | $(BIN_SIM_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(INCLUDE_CACHE) $< $(LDLIBS) -o $@

# Convenience targets for simulation builds
.PHONY: sim-% all-sim clean-sim run-sim-%

sim-%: $(BIN_SIM_DIR)/%
	@echo "Built simulation version: $<"

all-sim: $(addprefix $(BIN_SIM_DIR)/, $(KERNELS_SIM))
	@echo "Built all simulation binaries"

clean-sim:
	rm -rf $(BIN_SIM_DIR)

# Run simulation with default parameters
run-sim-%: $(BIN_SIM_DIR)/%
	@echo "Running cache simulation: $<"
	@./$< -g 10 -n 1

# =========================================================
# Help
# =========================================================
help: help-pr
	@echo "Available Make commands:"	
	@echo "  all            - Builds all targets including GAP benchmarks (CPU)"
	@echo "  run-%          - Runs the specified GAP benchmark (bc bfs cc cc_sv pr pr_spmv sssp tc)"
	@echo "  help-%         - Print the specified Help (bc bfs cc cc_sv pr pr_spmv sssp tc)"
	@echo "  clean          - Removes all build artifacts"
	@echo ""
	@echo "Cache Simulation:"
	@echo "  all-sim        - Builds all cache simulation binaries (pr bfs bc cc sssp tc)"
	@echo "  sim-%          - Build simulation version of specified algorithm"
	@echo "  run-sim-%      - Run cache simulation with default graph"
	@echo "  clean-sim      - Remove simulation build artifacts"
	@echo ""
	@echo "Cache Simulation Environment Variables:"
	@echo "  CACHE_L1_SIZE=32768       - L1 cache size in bytes (default: 32KB)"
	@echo "  CACHE_L1_WAYS=8           - L1 associativity (default: 8-way)"
	@echo "  CACHE_L2_SIZE=262144      - L2 cache size in bytes (default: 256KB)"
	@echo "  CACHE_L2_WAYS=4           - L2 associativity (default: 4-way)"
	@echo "  CACHE_L3_SIZE=8388608     - L3 cache size in bytes (default: 8MB)"
	@echo "  CACHE_L3_WAYS=16          - L3 associativity (default: 16-way)"
	@echo "  CACHE_LINE_SIZE=64        - Cache line size in bytes (default: 64)"
	@echo "  CACHE_POLICY=LRU          - Eviction policy: LRU, FIFO, RANDOM, LFU, PLRU, SRRIP"
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
	@echo "  │ AdaptiveOrder (14):  ML-based perceptron selector for optimal algorithm     │"
	@echo "  │                      Format: -o 14:<depth>:<res>:<min_size>:<mode>          │"
	@echo "  │                      depth=0: per-community, 1+: multi-level recursion      │"
	@echo "  │                      mode=0: per-community, 1: full-graph adaptive          │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ Leiden Algorithms (15-17) - Parameter-based variant selection               │"
	@echo "  ├─────────────────────────────────────────────────────────────────────────────┤"
	@echo "  │ LeidenOrder   (15):  Leiden via igraph (format: 15:resolution)              │"
	@echo "  │ LeidenDendrogram(16): Dendrogram traversal (format: 16:variant:res)         │"
	@echo "  │                      Variants: dfs, dfshub, dfssize, bfs, hybrid (default)  │"
	@echo "  │ LeidenCSR     (17):  GVE-Leiden CSR-native (format: 17:variant:res:iter:pass)│"
	@echo "  │                      Variants: gve (default), gveopt, dfs, bfs, hubsort, fast│"
	@echo "  └─────────────────────────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "Example Usage:"
	@echo "  make all - Compile the program."
	@echo "  make clean - Clean build files."
	@echo "  ./$< -g 15 -n 1 -o 14           - Execute with AdaptiveOrder (auto-select best)"
	@echo "  ./$< -g 15 -n 1 -o 14:2         - Execute with multi-level AdaptiveOrder (depth=2)"
	@echo "  ./$< -g 15 -n 1 -o 14:0:0.75:50000:1 - Full-graph mode (pick single best algo)"
	@echo "  ./$< -g 15 -n 1 -o 16:hybrid:1.0 - Execute with LeidenDendrogram (hybrid)"
	@echo "  ./$< -g 15 -n 1 -o 17:gve:1.0:20:5 - Execute with LeidenCSR (GVE-Leiden)"
	@echo "  ./$< -g 15 -n 1 -o 12:10:16     - Execute GraphBrew with LeidenDendrogram"
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