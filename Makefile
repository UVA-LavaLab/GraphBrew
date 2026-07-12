CC  := $(shell which gcc-9 || which gcc)
CXX := $(shell which g++-9 || which g++)
PYTHON ?= python3
PARALLEL ?= $(shell grep -c ^processor /proc/cpuinfo)
RABBIT_ENABLE ?= 0

BENCH_DIR := bench
INC_DIR := $(BENCH_DIR)/include
BIN_DIR := $(BENCH_DIR)/bin
BIN_SIM_DIR := $(BENCH_DIR)/bin_sim
BIN_GEM5_DIR := $(BENCH_DIR)/bin_gem5
BIN_SNIPER_DIR := $(BENCH_DIR)/bin_sniper

INCLUDE_GAPBS := $(INC_DIR)/external/gapbs
INCLUDE_GRAPHBREW := $(INC_DIR)/graphbrew
INCLUDE_EXTERNAL := $(INC_DIR)/external
INCLUDE_CACHE := $(INC_DIR)/cache_sim

INCLUDES := -I$(INCLUDE_GAPBS) -I$(INCLUDE_GRAPHBREW) \
	-I$(INCLUDE_EXTERNAL) -I$(INC_DIR)

CXXFLAGS := -std=c++17 -O3 -Wall -fopenmp -g -DNDEBUG -m64 -march=native \
	-DTYPE=float -DMAX_THREADS=$(PARALLEL) -DREPEAT_METHOD=1
LDLIBS :=

ifeq ($(RABBIT_ENABLE),1)
CXXFLAGS += -DRABBIT_ENABLE -mcx16 -Wno-deprecated-declarations \
	-Wno-parentheses -Wno-unused-local-typedefs
LDLIBS += -ltcmalloc_minimal -lnuma
endif

DEP_GAPBS := $(wildcard $(INCLUDE_GAPBS)/*.h)
DEP_GRAPH := $(wildcard $(INCLUDE_GRAPHBREW)/reorder/*.h) \
	$(wildcard $(INCLUDE_GRAPHBREW)/partition/*.h)
DEP_EXTERNAL := $(wildcard $(INCLUDE_EXTERNAL)/rabbit/*.hpp) \
	$(wildcard $(INCLUDE_EXTERNAL)/gorder/*.h) \
	$(wildcard $(INCLUDE_EXTERNAL)/corder/*.h) \
	$(wildcard $(INCLUDE_EXTERNAL)/leiden/*.hxx)
DEP_CACHE := $(wildcard $(INCLUDE_CACHE)/*.h) \
	$(INC_DIR)/ecg_victim_policy.h \
	$(INC_DIR)/ecg_mode6_builder.h \
	$(INC_DIR)/ecg_epoch_builder.h

KERNELS_SIM := pr pr_spmv bfs bc cc cc_sv sssp tc ecg_preprocess
KERNELS_GEM5 := pr pr_spmv bfs sssp cc cc_sv bc tc
KERNELS_SNIPER := sg_kernel pr bfs sssp bc cc cc_sv \
	pr_kernel_smoke bfs_kernel_smoke sssp_kernel_smoke hello_roi

.PHONY: all artifact converter all-sim all-gem5 all-sniper \
	sim-% gem5-% gem5-m5ops-% gem5-riscv-m5ops-% sniper-% \
	setup-gem5 setup-sniper test verify clean clean-sim clean-gem5-bin \
	clean-sniper-bin help

all: all-sim

artifact: all-sim
	@echo "cache_sim ECG artifact built"

help:
	@echo "ECG artifact targets:"
	@echo "  make all-sim                  Build cache_sim kernels"
	@echo "  make converter                Build .el -> .sg converter"
	@echo "  make setup-gem5               Install/build gem5 overlays"
	@echo "  make gem5-riscv-m5ops-pr      Build RISC-V ECG PageRank"
	@echo "  make setup-sniper             Install/build Sniper overlays"
	@echo "  make sniper-sg_kernel         Build canonical Sniper workload"
	@echo "  make test                     Run Python artifact tests"

$(BIN_DIR) $(BIN_SIM_DIR) $(BIN_GEM5_DIR) $(BIN_SNIPER_DIR):
	mkdir -p $@

converter: $(BIN_DIR)/converter

$(BIN_DIR)/converter: $(BENCH_DIR)/src/converter.cc $(DEP_GAPBS) \
	$(DEP_GRAPH) $(DEP_EXTERNAL) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LDLIBS) -o $@

$(BIN_SIM_DIR)/%: $(BENCH_DIR)/src_sim/%.cc $(DEP_GAPBS) \
	$(DEP_GRAPH) $(DEP_EXTERNAL) $(DEP_CACHE) | $(BIN_SIM_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(INCLUDE_CACHE) $< $(LDLIBS) -o $@

sim-%: $(BIN_SIM_DIR)/%
	@echo "Built cache_sim kernel: $<"

all-sim: $(addprefix $(BIN_SIM_DIR)/,$(KERNELS_SIM))
	@echo "Built all cache_sim ECG kernels"

GEM5_SIM_DIR := $(INC_DIR)/gem5_sim
GEM5_DIR := $(GEM5_SIM_DIR)/gem5
GEM5_M5_DIR := $(GEM5_DIR)/util/m5
GEM5_M5_LIB := $(GEM5_M5_DIR)/build/x86/out/libm5.a
GEM5_M5_RISCV_LIB := $(GEM5_M5_DIR)/build/riscv/out/libm5.a
RISCV_CXX ?= riscv64-linux-gnu-g++
RISCV_CROSS_COMPILE ?= riscv64-linux-gnu-

CXXFLAGS_GEM5 := -std=c++17 -O1 -Wall -g -DNDEBUG -DNO_M5OPS -fopenmp
CXXFLAGS_GEM5_M5OPS := $(filter-out -DNO_M5OPS,$(CXXFLAGS_GEM5)) \
	-I$(GEM5_DIR)/include
CXXFLAGS_GEM5_RISCV := $(CXXFLAGS_GEM5_M5OPS) -static -mno-relax

$(BIN_GEM5_DIR)/%: $(BENCH_DIR)/src_gem5/%.cc $(DEP_GAPBS) \
	$(DEP_GRAPH) $(DEP_EXTERNAL) | $(BIN_GEM5_DIR)
	$(CXX) $(CXXFLAGS_GEM5) $(INCLUDES) $< $(LDLIBS) -o $@

$(GEM5_M5_LIB):
	cd $(GEM5_M5_DIR) && scons -j$(PARALLEL) build/x86/out/m5

$(GEM5_M5_RISCV_LIB):
	cd $(GEM5_M5_DIR) && scons -j$(PARALLEL) build/riscv/out/m5 \
		riscv.CROSS_COMPILE=$(RISCV_CROSS_COMPILE)

$(BIN_GEM5_DIR)/%_m5ops: $(BENCH_DIR)/src_gem5/%.cc $(DEP_GAPBS) \
	$(DEP_GRAPH) $(DEP_EXTERNAL) $(GEM5_M5_LIB) | $(BIN_GEM5_DIR)
	$(CXX) $(CXXFLAGS_GEM5_M5OPS) $(INCLUDES) $< \
		$(GEM5_M5_LIB) $(LDLIBS) -o $@

$(BIN_GEM5_DIR)/%_riscv_m5ops: $(BENCH_DIR)/src_gem5/%.cc $(DEP_GAPBS) \
	$(DEP_GRAPH) $(DEP_EXTERNAL) $(GEM5_M5_RISCV_LIB) | $(BIN_GEM5_DIR)
	$(RISCV_CXX) $(CXXFLAGS_GEM5_RISCV) $(INCLUDES) $< \
		$(GEM5_M5_RISCV_LIB) -o $@

gem5-%: $(BIN_GEM5_DIR)/%
	@echo "Built gem5 kernel: $<"

gem5-m5ops-%: $(BIN_GEM5_DIR)/%_m5ops
	@echo "Built gem5 m5ops kernel: $<"

gem5-riscv-m5ops-%: $(BIN_GEM5_DIR)/%_riscv_m5ops
	@echo "Built RISC-V gem5 kernel: $<"

all-gem5: $(addprefix $(BIN_GEM5_DIR)/,$(KERNELS_GEM5))
	@echo "Built all native gem5 ECG kernels"

SNIPER_DIR := $(INC_DIR)/sniper_sim/snipersim
SNIPER_INCLUDE := $(SNIPER_DIR)/include
CXXFLAGS_SNIPER := -std=c++17 -O2 -Wall -g -DNDEBUG -fopenmp \
	-I$(INC_DIR) -I$(SNIPER_INCLUDE)

$(BIN_SNIPER_DIR)/%: $(BENCH_DIR)/src_sniper/%.cc $(DEP_GAPBS) \
	$(DEP_GRAPH) $(DEP_EXTERNAL) | $(BIN_SNIPER_DIR)
	$(CXX) $(CXXFLAGS_SNIPER) $(INCLUDES) $< $(LDLIBS) -o $@

sniper-%: $(BIN_SNIPER_DIR)/%
	@echo "Built Sniper kernel: $<"

all-sniper: $(addprefix $(BIN_SNIPER_DIR)/,$(KERNELS_SNIPER))
	@echo "Built all Sniper ECG kernels"

setup-gem5:
	$(PYTHON) scripts/setup_gem5.py --isa X86 RISCV --jobs $(PARALLEL)

setup-sniper:
	$(PYTHON) scripts/setup_sniper.py --jobs $(PARALLEL) --apply-overlays

test:
	pytest -q scripts/test

verify:
	$(PYTHON) scripts/experiments/ecg/verify/equiv_kernels.py \
		--gem5 --sniper --kernels pr bfs --schedule-k 2

clean-sim:
	rm -rf $(BIN_SIM_DIR)

clean-gem5-bin:
	rm -rf $(BIN_GEM5_DIR)

clean-sniper-bin:
	rm -rf $(BIN_SNIPER_DIR)

clean: clean-sim clean-gem5-bin clean-sniper-bin
