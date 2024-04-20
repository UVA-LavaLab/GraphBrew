# =========================================================
# Compiler and Directories Setup
# =========================================================
CXX = g++
# =========================================================
BENCH_DIR = bench
BIN_DIR = $(BENCH_DIR)/bin
LIB_DIR = $(BENCH_DIR)/lib
SRC_DIR = $(BENCH_DIR)/src
INCLUDE_DIR = $(BENCH_DIR)/include

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
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp 
INCLUDES = -I$(INCLUDE_DIR)

# =========================================================
# Targets
# =========================================================
KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc
SUITE = $(addprefix $(BIN_DIR)/,$(KERNELS)) $(BIN_DIR)/converter
# =========================================================

.PHONY: all run-% help-% help clean
all: $(SUITE)

# =========================================================
# Compilation Rules
# =========================================================
$(BIN_DIR)/%: $(SRC_DIR)/%.cc $(INCLUDE_DIR)/*.h | $(BIN_DIR) $(LIB_DIR)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(EXIT_STATUS)

# =========================================================
# Running Benchmarks
# =========================================================
run-%: $(BIN_DIR)/%
	@./$< -g 15 -n 1 $(EXIT_STATUS)

run-all: $(addprefix run-, $(KERNELS))

# =========================================================
# Directory Setup
# =========================================================
$(BIN_DIR) $(LIB_DIR):
	@mkdir -p $@ $(CREATE_STATUS)

# =========================================================
# Cleanup
# =========================================================
clean:
	@rm -rf $(BIN_DIR) $(LIB_DIR) $(EXIT_STATUS)

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
	@echo "  - ORIGINAL (0): No reordering applied."
	@echo "  - RANDOM (1): Apply random reordering."
	@echo "  - SORT (2): Apply sort-based reordering."
	@echo "  - HUBSORT (3): Apply hub-based sorting."
	@echo "  - HUBCLUSTER (4): Apply clustering based on hub scores."
	@echo "  - DBG (5): Apply degree-based grouping."
	@echo "  - HUBSORTDBG (6): Combine hub sorting with degree-based grouping."
	@echo "  - HUBCLUSTERDBG (7): Combine hub clustering with degree-based grouping."
	@echo "  - MAP (10): Requires a file format for reordering. Use the -r 10:filename.label option."
	@echo ""
	@echo "Example Usage:"
	@echo "  make all - Compile the program."
	@echo "  make clean - Clean build files."
	@echo "  ./$< -g 15 -n 1 -r 10:mapping.label - Execute with MAP reordering using 'mapping.label'."

help-all: $(addprefix help-, $(KERNELS))