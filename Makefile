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
RUN_PARAMS =  -o5 -n 1
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
	@rm -rf $(BIN_DIR) $(EXIT_STATUS)

clean-all: clean-results
	@rm -rf $(BIN_DIR) $(EXIT_STATUS)

scrub-all:
	@rm -rf $(BIN_DIR) $(BACKUP_DIR) $(RES_DIR) 00_* $(EXIT_STATUS) 

clean-results:
	@mkdir -p $(BACKUP_DIR)
	@echo "Backing up results directory..."
	@tar -czf $(BACKUP_DIR)/result_`date +"%Y%m%d_%H%M%S"`.tar.gz $(RES_DIR)
	@echo "Cleaning results directory..."
	@rm -rf $(RES_DIR)/*/data_charts
	@rm -rf $(RES_DIR)/*/data_csv
	@echo "Backup and clean completed."

# =========================================================
# Help
# =========================================================
help: help-pr
	@echo "Available Make commands:"	
	@echo "  all            - Builds all targets including GAP benchmarks (CPU)"
	@echo "  run-%          - Runs the specified GAP benchmark (bc bfs cc cc_sv pr pr_spmv sssp tc)"
	@echo "  help-%         - Print the specified Help (bc bfs cc cc_sv pr pr_spmv sssp tc)"
	@echo "  clean          - Removes all build artifacts"
	@echo "  help           - Displays this help message"

# Alias each kernel target to its corresponding help target
$(KERNELS) converter: %: help-%

help-%: $(BIN_DIR)/%
	@./$< -h 
	@echo ""
	@echo "Reordering Algorithms:"
	@echo "  - ORIGINAL      (0):  No reordering applied."
	@echo "  - RANDOM        (1):  Apply random reordering."
	@echo "  - SORT          (2):  Apply sort-based reordering."
	@echo "  - HUBSORT       (3):  Apply hub-based sorting."
	@echo "  - HUBCLUSTER    (4):  Apply clustering based on hub scores."
	@echo "  - DBG           (5):  Apply degree-based grouping."
	@echo "  - HUBSORTDBG    (6):  Combine hub sorting with degree-based grouping."
	@echo "  - HUBCLUSTERDBG (7):  Combine hub clustering with degree-based grouping."
	@echo "  - RABBITORDER   (8):  Apply community clustering with incremental aggregation."
	@echo "  - GORDER        (9):  Apply dynamic programming BFS and windowing ordering."
	@echo "  - CORDER        (10): Workload Balancing via Graph Reordering on Multicore Systems."
	@echo "  - RCM           (11): RCM is ordered by the reverse Cuthill-McKee algorithm (BFS)."
	@echo "  - LeidenOrder   (12): Apply Leiden community clustering with louvain with refinement."
	@echo "  - GraphBrewOrder(13): Leiden community clustering with rabbit order refinement."
	@echo "  - MAP           (14): Requires a file format for reordering. Use the -r 10:filename.label option."
	@echo ""
	@echo "Example Usage:"
	@echo "  make all - Compile the program."
	@echo "  make clean - Clean build files."
	@echo "  ./$< -g 15 -n 1 -o 10:mapping.label - Execute with MAP reordering using 'mapping.label'."

help-all: $(addprefix help-, $(KERNELS))