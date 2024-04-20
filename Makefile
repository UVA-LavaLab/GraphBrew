# Compiler and Directories Setup
CXX = g++
BENCH_DIR = bench
BIN_DIR = $(BENCH_DIR)/bin
LIB_DIR = $(BENCH_DIR)/lib
SRC_DIR = $(BENCH_DIR)/src
INCLUDE_DIR = $(BENCH_DIR)/include

# Compiler Flags
CXXFLAGS = -std=c++11 -O3 -Wall -fopenmp 
INCLUDES = -I$(INCLUDE_DIR)

# Targets
KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc
SUITE = $(addprefix $(BIN_DIR)/,$(KERNELS)) $(BIN_DIR)/converter

.PHONY: all run-% clean help
all: $(SUITE)

# Compilation Rules
$(BIN_DIR)/%: $(SRC_DIR)/%.cc $(INCLUDE_DIR)/*.h | $(BIN_DIR) $(LIB_DIR)
	@echo "Compiling GAP $< CPU..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# Running Benchmarks
run-%: $(BIN_DIR)/%
	@echo "Running GAP $@ CPU..."
	./$< -g 10 -n 1

# Directory Setup
$(BIN_DIR) $(LIB_DIR):
	@mkdir -p $@

# Cleanup
clean:
	@echo "Cleaning up..."
	@rm -rf $(BIN_DIR) $(LIB_DIR)

# Help
help:
	@echo "Available commands:"
	@echo "  all            - Builds all targets including GAP benchmarks (CPU)"
	@echo "  run-%          - Runs the specified GAP benchmark"
	@echo "  clean          - Removes all build artifacts"
	@echo "  help           - Displays this help message"
