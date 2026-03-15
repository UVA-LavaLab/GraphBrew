#!/bin/bash
# ============================================================================
# gem5 Policy Sweep: Run L3-stressing PageRank workload across all policies
# ============================================================================
# Usage: bash bench/include/gem5_sim/tests/run_sweep.sh
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GEM5="$PROJECT_ROOT/bench/include/gem5_sim/gem5/build/X86/gem5.opt"
CONFIG="$SCRIPT_DIR/policy_sweep.py"
TEST_SRC="$SCRIPT_DIR/gem5_pr_large.c"
TEST_BIN="/tmp/gem5_pr_large"
RESULTS_DIR="/tmp/gem5_sweep_results"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check prerequisites
if [ ! -f "$GEM5" ]; then
    echo "Error: gem5 binary not found at $GEM5"
    echo "Run: make setup-gem5"
    exit 1
fi

# Compile test binary (static, no OpenMP)
echo -e "${BLUE}Compiling test binary...${NC}"
gcc -O1 -static -o "$TEST_BIN" "$TEST_SRC" -lm
echo "  Binary: $TEST_BIN"
echo "  Working set: ~11MB (overflows 8MB L3)"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run sweep
POLICIES="LRU SRRIP GRASP ECG"
echo -e "${BLUE}Running policy sweep: $POLICIES${NC}"
echo "============================================"

for policy in $POLICIES; do
    OUTDIR="$RESULTS_DIR/$policy"
    echo -e "\n${GREEN}=== $policy ===${NC}"

    $GEM5 --outdir="$OUTDIR" "$CONFIG" "$policy" --binary="$TEST_BIN" 2>&1 \
        | grep -E "Policy Test|Binary:|Policy:|Done @|PR sum"

    # Extract key stats
    echo "  Cache Stats:"
    grep "overallMissRate::total" "$OUTDIR/stats.txt" \
        | grep -E "dcache|l2cache|l3cache" \
        | while read line; do
            name=$(echo "$line" | awk -F'.' '{print $3}' | awk -F'.' '{print $1}')
            rate=$(echo "$line" | awk '{print $2}')
            printf "    %-10s miss rate: %s\n" "$name" "$rate"
        done

    # L3 specific
    l3_misses=$(grep "system.l3cache.overallMisses::total" "$OUTDIR/stats.txt" | awk '{print $2}')
    l3_accesses=$(grep "system.l3cache.overallAccesses::total" "$OUTDIR/stats.txt" | awk '{print $2}')
    l3_evictions=$(grep "system.l3cache.replacements" "$OUTDIR/stats.txt" | awk '{print $2}' | head -1)
    echo "    L3 misses:    $l3_misses"
    echo "    L3 accesses:  $l3_accesses"
    echo "    L3 evictions: $l3_evictions"
done

echo ""
echo "============================================"
echo -e "${GREEN}Sweep complete. Results in $RESULTS_DIR${NC}"
echo ""

# Summary comparison
echo "Policy Comparison (L3 miss rate):"
echo "---"
for policy in $POLICIES; do
    rate=$(grep "system.l3cache.overallMissRate::total" "$RESULTS_DIR/$policy/stats.txt" 2>/dev/null | awk '{print $2}')
    printf "  %-8s %s\n" "$policy" "${rate:-N/A}"
done
