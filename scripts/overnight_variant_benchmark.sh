#!/usr/bin/env bash
# =============================================================================
# Overnight Variant Benchmark — GOrder & RabbitOrder Variant Comparison
# =============================================================================
#
# Three experiments run sequentially through graphbrew_experiment.py:
#
#   Experiment 1: GOrder Default vs GOrder CSR vs GOrder Fast
#     → Verify CSR/Fast variants produce comparable or better performance
#     → Uses --gorder-variants default csr fast with --all-variants
#
#   Experiment 2: RabbitOrder CSR vs RabbitOrder Boost
#     → Verify native CSR variant can beat the Boost implementation
#     → Uses --rabbit-variants csr boost with --all-variants
#
#   Experiment 3: Original (no reorder) baseline
#     → Absolute reference: does reordering actually help?
#
# Graph sets: medium + large (downloaded fresh if missing)
# Results: saved to results/ with --isolate-run timestamps
# Duration estimate: 8-16 hours depending on graph count and machine
#
# Usage:
#   chmod +x scripts/overnight_variant_benchmark.sh
#   nohup ./scripts/overnight_variant_benchmark.sh > overnight_run.log 2>&1 &
#   # or
#   screen -S overnight ./scripts/overnight_variant_benchmark.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"
EXPERIMENT="$PYTHON scripts/graphbrew_experiment.py"
LOG_FILE="overnight_run_$(date +%Y%m%d_%H%M%S).log"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

separator() {
    log "================================================================="
}

run_experiment() {
    local description="$1"
    shift
    log "$description"
    if $EXPERIMENT "$@" 2>&1 | tee -a "$LOG_FILE"; then
        log "PASS: $description"
    else
        log "FAIL: $description (exit code $?)"
    fi
    separator
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared settings
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARKS="pr bfs cc sssp"
TRIALS=3
TIMEOUT_REORDER=14400   # 4 hours per reorder
TIMEOUT_BENCH=1800      # 30 min per benchmark run
TIMEOUT_SIM=3600        # 1 hour for cache sim
MEMORY_FLAGS="--auto-memory"

# Common flags for all experiment runs
COMMON_FLAGS=(
    --skip-download --skip-build
    --benchmarks $BENCHMARKS
    --trials $TRIALS
    --timeout-reorder $TIMEOUT_REORDER
    --timeout-benchmark $TIMEOUT_BENCH
    --timeout-sim $TIMEOUT_SIM
    --phase all
    --force-reorder
    --isolate-run
    $MEMORY_FLAGS
)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────
log "Starting Overnight Variant Benchmark"
log "Project: $PROJECT_DIR"
log "Python:  $PYTHON"
log "Log:     $LOG_FILE"
separator

log "Checking binaries..."
if [ ! -f bench/bin/converter ] || [ ! -f bench/bin/pr ]; then
    log "Building binaries..."
    make -j"$(nproc)" 2>&1 | tee -a "$LOG_FILE"
fi
log "Binaries OK: $(ls bench/bin/ | tr '\n' ' ')"
separator

# ─────────────────────────────────────────────────────────────────────────────
# Download graphs (medium + large, skip if already present)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 0: Downloading medium graphs..."
$EXPERIMENT --download-only --size medium $MEMORY_FLAGS 2>&1 | tee -a "$LOG_FILE"
separator

log "Phase 0: Downloading large graphs..."
$EXPERIMENT --download-only --size large $MEMORY_FLAGS 2>&1 | tee -a "$LOG_FILE"
separator

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: GOrder Variant Comparison
# ─────────────────────────────────────────────────────────────────────────────
# GOrder (algo 9) has three implementations:
#   default (-o 9)     : GoGraph baseline, converts to GoGraph adjacency format
#   csr     (-o 9:csr) : CSR-native, direct iterator access, lighter BFS-RCM
#   fast    (-o 9:fast): Parallel batch, atomic score updates, fan-out cap
#
# All three produce EQUIVALENT orderings — only implementation speed differs.
# --all-variants + --gorder-variants generates separate .lo files:
#   GORDER_default.lo, GORDER_csr.lo, GORDER_fast.lo
# Benchmarks run on each reordering to verify equivalent quality.
# ─────────────────────────────────────────────────────────────────────────────
separator
log "╔══════════════════════════════════════════════════════════════╗"
log "║   EXPERIMENT 1: GOrder Default vs CSR vs Fast              ║"
log "╚══════════════════════════════════════════════════════════════╝"
separator

run_experiment "Exp 1 (medium): GOrder default vs csr vs fast" \
    --size medium \
    --algo-list GORDER \
    --all-variants --gorder-variants default csr fast \
    "${COMMON_FLAGS[@]}"

run_experiment "Exp 1 (large): GOrder default vs csr vs fast" \
    --size large \
    --algo-list GORDER \
    --all-variants --gorder-variants default csr fast \
    --skip-slow \
    "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: RabbitOrder CSR vs Boost
# ─────────────────────────────────────────────────────────────────────────────
# RabbitOrder (algo 8) has two variants:
#   csr   : Native CSR implementation (no external dependencies)
#   boost : Original Boost-based implementation (requires Boost 1.58)
#
# --all-variants + --rabbit-variants generates separate .lo files:
#   RABBITORDER_csr.lo, RABBITORDER_boost.lo
# CSR should produce comparable or better results vs Boost.
# ─────────────────────────────────────────────────────────────────────────────
separator
log "╔══════════════════════════════════════════════════════════════╗"
log "║   EXPERIMENT 2: RabbitOrder CSR vs Boost                   ║"
log "╚══════════════════════════════════════════════════════════════╝"
separator

run_experiment "Exp 2 (medium): RabbitOrder csr vs boost" \
    --size medium \
    --algo-list RABBITORDER \
    --all-variants --rabbit-variants csr boost \
    "${COMMON_FLAGS[@]}"

run_experiment "Exp 2 (large): RabbitOrder csr vs boost" \
    --size large \
    --algo-list RABBITORDER \
    --all-variants --rabbit-variants csr boost \
    --skip-slow \
    "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Absolute Baseline Comparison
# ─────────────────────────────────────────────────────────────────────────────
# Run ORIGINAL (no reorder) as baseline reference.
# This verifies that reordering actually helps vs no reordering.
# ─────────────────────────────────────────────────────────────────────────────
separator
log "╔══════════════════════════════════════════════════════════════╗"
log "║   EXPERIMENT 3: Original (No Reorder) Baseline             ║"
log "╚══════════════════════════════════════════════════════════════╝"
separator

run_experiment "Exp 3 (medium): Original baseline" \
    --size medium \
    --algo-list ORIGINAL \
    "${COMMON_FLAGS[@]}"

run_experiment "Exp 3 (large): Original baseline" \
    --size large \
    --algo-list ORIGINAL \
    --skip-slow \
    "${COMMON_FLAGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
log "╔══════════════════════════════════════════════════════════════╗"
log "║   ALL EXPERIMENTS COMPLETE                                 ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""
log "Results location: results/"
log "Isolated runs:    results/logs/"
log "Log file:         $LOG_FILE"
log ""
log "To list all runs:"
log "  $EXPERIMENT --list-runs"
log ""
log "Expected comparisons:"
log "  1. GOrder: default vs csr vs fast  (reorder speed, same orderings)"
log "  2. RabbitOrder: csr vs boost       (reorder speed + quality)"
log "  3. All vs Original                 (reordering benefit)"
log ""
log "Finished at $(date)"
