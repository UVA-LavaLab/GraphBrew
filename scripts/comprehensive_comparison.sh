#!/bin/bash
# ===========================================================================
# Comprehensive GraphBrew Comparison: Old Standalone vs VIBE-Powered
# ===========================================================================
# Tests all matching GraphBrew variants on every available .sg graph.
# Old data from results/benchmark_20260208_032723.json (pre-VIBE).
# New data generated live with current VIBE-powered build.
#
# Goal: justify full deprecation of old standalone GraphBrew code.
# ===========================================================================

set -e

BIN="./bench/bin/pr"
TRIALS=3
THREADS=4
export OMP_NUM_THREADS=$THREADS

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="results/comprehensive_graphbrew_comparison_${TIMESTAMP}.csv"

# All available .sg graphs
GRAPHS=(
    "results/graphs/soc-Slashdot0902/soc-Slashdot0902.sg"
    "results/graphs/soc-Epinions1/soc-Epinions1.sg"
    "results/graphs/web-Google/web-Google.sg"
    "results/graphs/web-BerkStan/web-BerkStan.sg"
    "results/graphs/roadNet-CA/roadNet-CA.sg"
    "results/graphs/cit-Patents/cit-Patents.sg"
    "results/graphs/as-Skitter/as-Skitter.sg"
    "results/graphs/cnr-2000/cnr-2000.sg"
    "results/graphs/com-Youtube/com-Youtube.sg"
    "results/graphs/soc-LiveJournal1/soc-LiveJournal1.sg"
    "results/graphs/wiki-topcats/wiki-topcats.sg"
    "results/graphs/hollywood-2009/hollywood-2009.sg"
    "results/graphs/delaunay_n24/delaunay_n24.sg"
    "results/graphs/kron_g500-logn20/kron_g500-logn20.sg"
    "results/graphs/rgg_n_2_24_s0/rgg_n_2_24_s0.sg"
)

# New VIBE-powered GraphBrew variants to test
# Format: "label|cli_option"
ALGOS=(
    # GraphBrew (algo 12) variants — VIBE-powered
    "GB-default|12"
    "GB-leiden:rabbit8|12:leiden"
    "GB-gve:rabbit8|12:gve"
    "GB-gveopt:rabbit8|12:gveopt"
    "GB-leiden:hubclusterdbg7|12:leiden:7"
    "GB-gve:hubclusterdbg7|12:gve:7"
    "GB-rabbit|12:rabbit"
    "GB-hubcluster|12:hubcluster"
    # VIBE standalone (algo 16) — for cross-reference
    "VIBE-default|16:vibe"
    "VIBE-quality|16:vibe:quality"
    "VIBE-rabbit|16:vibe:rabbit"
    "VIBE-hrab|16:vibe:hrab"
    "VIBE-conn|16:vibe:conn"
)

echo "graph,algorithm,label,reorder_time,avg_pr_time,trials,threads,timestamp" > "$OUTFILE"

printf "\n"
printf "╔══════════════════════════════════════════════════════════════════════════════════╗\n"
printf "║   Comprehensive GraphBrew Comparison: Old Standalone vs VIBE-Powered           ║\n"
printf "║   Threads: %-4d  Trials: %-4d  Graphs: %-3d  Algorithms: %-3d                   ║\n" \
       "$THREADS" "$TRIALS" "${#GRAPHS[@]}" "${#ALGOS[@]}"
printf "╚══════════════════════════════════════════════════════════════════════════════════╝\n"
printf "\n"

GRAPH_NUM=0
TOTAL_GRAPHS=${#GRAPHS[@]}

for GRAPH in "${GRAPHS[@]}"; do
    GRAPH_NUM=$((GRAPH_NUM + 1))
    GNAME=$(basename "$GRAPH" .sg)
    
    if [ ! -f "$GRAPH" ]; then
        printf "  [SKIP] %s - file not found\n" "$GNAME"
        continue
    fi
    
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  [%d/%d] Graph: %-40s\n" "$GRAPH_NUM" "$TOTAL_GRAPHS" "$GNAME"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  %-30s %12s %12s\n" "Algorithm" "Reorder(s)" "PR-Avg(s)"
    printf "  %-30s %12s %12s\n" "─────────" "──────────" "─────────"

    for ALGO_ENTRY in "${ALGOS[@]}"; do
        IFS='|' read -r LABEL CLI_OPT <<< "$ALGO_ENTRY"
        
        OUTPUT=$($BIN -f "$GRAPH" -o "$CLI_OPT" -n "$TRIALS" -l 2>&1) || true
        
        # Extract reorder time
        REORDER_TIME=$(echo "$OUTPUT" | grep -oP '(?:Reorder|Relabel)\s+Map\s+Time:\s+\K[\d.]+' | tail -1)
        if [ -z "$REORDER_TIME" ]; then
            REORDER_TIME=$(echo "$OUTPUT" | grep -oP 'Total Reorder Time:\s+\K[\d.]+' | tail -1)
        fi
        [ -z "$REORDER_TIME" ] && REORDER_TIME="N/A"
        
        # Extract average PR time
        AVG_TIME=$(echo "$OUTPUT" | grep -oP 'Average Time:\s+\K[\d.]+' | tail -1)
        [ -z "$AVG_TIME" ] && AVG_TIME="N/A"
        
        printf "  %-30s %12s %12s\n" "$LABEL" "$REORDER_TIME" "$AVG_TIME"
        echo "$GNAME,$CLI_OPT,$LABEL,$REORDER_TIME,$AVG_TIME,$TRIALS,$THREADS,$TIMESTAMP" >> "$OUTFILE"
    done
    printf "\n"
done

printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
printf "Results saved: %s\n" "$OUTFILE"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
