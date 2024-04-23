# =========================================================
# Compiler and Directories Setup
# =========================================================
# CXX = g++
# Attempt to find gcc-9, else use default gcc
CC  = $(shell which gcc-9 || which gcc)
CXX = $(shell which g++-9 || which g++)
# =========================================================
BENCH_DIR = bench
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
CREATE        = [${BLUE}create!${NC}]
SUCCESS       = [${GREEN}success!${NC}]
FAIL          = [${RED}failure!${NC}]
CREATE_MSG   = echo  "$(CREATE) $(COMPILED_FILE)"
SUCCESS_MSG   = echo  "$(SUCCESS) $(COMPILED_FILE)"
FAIL_MSG      = echo  "$(FAIL) $(COMPILED_FILE)"
EXIT_STATUS   = &&  $(SUCCESS_MSG) || { $(FAIL_MSG) ; exit 1; }
CREATE_STATUS  = &&  $(CREATE_MSG) || { $(FAIL_MSG) ; exit 1; }
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
CXXFLAGS_GAP    = -std=c++17 -O3 -Wall -fopenmp
CXXFLAGS_RABBIT = -mcx16 
CXXFLAGS_GORDER = -m64 -march=native 
CXXFLAGS_GORDER += -DRelease -DGCC
CXXFLAGS_LEIDEN = -DTYPE=float -DMAX_THREADS=$(PARALLEL) -DREPEAT_METHOD=1 -DOPENMP
# =========================================================
LDLIBS_RABBIT   += -ltcmalloc_minimal -lnuma
# =========================================================
CXXFLAGS = $(CXXFLAGS_GAP) $(CXXFLAGS_RABBIT) $(CXXFLAGS_GORDER) 
LDLIBS   = $(LDLIBS_RABBIT)
# =========================================================
# CXXFLAGS += -D_DEBUG
INCLUDES = -I$(INCLUDE_GAPBS) -I$(INCLUDE_RABBIT) -I$(INCLUDE_GORDER) -I$(INCLUDE_CORDER) -I$(INCLUDE_LEIDEN)

# =========================================================
# Targets
# =========================================================
KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc
SUITE = $(addprefix $(BIN_DIR)/,$(KERNELS)) $(BIN_DIR)/converter
# =========================================================

.PHONY: all run-% help-% help clean run-%-gdb run-%-sweep $(BIN_DIR)/%
all: $(SUITE)

# =========================================================
# Runtime Flags OMP_NUM_THREADS
# =========================================================
PARALLEL=16
# =========================================================

# =========================================================
# Running Benchmarks
# =========================================================
# GRAPH_BENCH = -f /home/ab/Documents/00_github_repos/00_GraphDatasets/SNAP/soc-LiveJournal1/graph.el
GRAPH_BENCH = -g 20
RUN_PARAMS = $(GRAPH_BENCH) -n 1  -i 100 -o 5
# =========================================================
run-%: $(BIN_DIR)/%
	@OMP_NUM_THREADS=$(PARALLEL) ./$< $(RUN_PARAMS) $(EXIT_STATUS)

run-%-gdb: $(BIN_DIR)/%
	@OMP_NUM_THREADS=1 gdb -ex=r --args ./$< $(RUN_PARAMS)

run-%-mem: $(BIN_DIR)/%
	@OMP_NUM_THREADS=1 valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes -v ./$< $(RUN_PARAMS)

run-all: $(addprefix run-, $(KERNELS))

# Define a rule that sweeps through -o 1 to 7
run-%-sweep: $(BIN_DIR)/%
	@for o in 1 2 3 4 5 6 7 8 9 10 11; do \
		echo "========================================================="; \
		OMP_NUM_THREADS=$(PARALLEL) ./$(BIN_DIR)/$* $(GRAPH_BENCH) -n 1 -o $$o; \
	done

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
	@echo "  - MAP           (12): Requires a file format for reordering. Use the -r 10:filename.label option."
	@echo ""
	@echo "Example Usage:"
	@echo "  make all - Compile the program."
	@echo "  make clean - Clean build files."
	@echo "  ./$< -g 15 -n 1 -o 10:mapping.label - Execute with MAP reordering using 'mapping.label'."

help-all: $(addprefix help-, $(KERNELS))