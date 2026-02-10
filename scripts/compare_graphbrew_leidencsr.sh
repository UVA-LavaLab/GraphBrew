#!/bin/bash
# ===========================================================================
# GraphBrew Original vs GraphBrew-Powered Comparison
# ===========================================================================
# Compares the new GraphBrew-powered GraphBrew (algo 12) against the best
# GraphBrew LeidenCSR variants (algo 16) to verify performance parity.
#
# We use the GraphBrew-powered GraphBrew (current code) and compare against
# GraphBrew standalone (algo 16) to verify the dispatch doesn't add overhead.
#
# Metrics: reorder time + PageRank execution time (3 trials)
# ===========================================================================

set -e

BIN="./bench/bin/pr"
TRIALS=3
THREADS=4
export OMP_NUM_THREADS=$THREADS

# Diverse graph set: social, web, road, citation, internet
GRAPHS=(
    "results/graphs/soc-Slashdot0902/soc-Slashdot0902.sg"
    "results/graphs/web-Google/web-Google.sg"
    "results/graphs/roadNet-CA/roadNet-CA.sg"
    "results/graphs/cit-Patents/cit-Patents.sg"
    "results/graphs/as-Skitter/as-Skitter.sg"
    "results/graphs/cnr-2000/cnr-2000.sg"
)

# Algorithms to compare
# Format: "label|cli_option"
ALGOS=(
    "GB-default(leiden:rabbit8)|12"
    "GB-gve:rabbit8|12:gve"
    "GB-gve:hubclusterdbg7|12:gve:7"
    "GB-rabbit|12:rabbit"
    "GB-hubcluster|12:hubcluster"
    "GraphBrew-default|16:graphbrew"
    "GraphBrew-rabbit|16:graphbrew:rabbit"
    "GraphBrew-hrab|16:graphbrew:hrab"
    "GraphBrew-conn|16:graphbrew:conn"
    "GraphBrew-quality|16:graphbrew:quality"
)

OUTFILE="results/comparison_graphbrew_vs_leidencsr_$(date +%Y%m%d_%H%M%S).csv"

echo "graph,algorithm,label,reorder_time,avg_pr_time,trials" > "$OUTFILE"

printf "\n"
printf "╔══════════════════════════════════════════════════════════════════════════╗\n"
printf "║          GraphBrew (GraphBrew-Powered) vs GraphBrew +Standalone Comparison        ║\n"
printf "║          Threads: %-4d  Trials: %-4d                                   ║\n" "$THREADS" "$TRIALS"
printf "╚══════════════════════════════════════════════════════════════════════════╝\n"
printf "\n"

for GRAPH in "${GRAPHS[@]}"; do
    GNAME=$(basename "$GRAPH" .sg)
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  Graph: %-40s\n" "$GNAME"
    printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    printf "  %-35s %12s %12s\n" "Algorithm" "Reorder(s)" "PR-Avg(s)"
    printf "  %-35s %12s %12s\n" "---------" "----------" "---------"

    for ALGO_ENTRY in "${ALGOS[@]}"; do
        IFS='|' read -r LABEL CLI_OPT <<< "$ALGO_ENTRY"
        
        # Run benchmark
        OUTPUT=$($BIN -f "$GRAPH" -o "$CLI_OPT" -n "$TRIALS" -l 2>&1) || true
        
        # Extract reorder time (look for "Reorder Map Time:" or "Relabel Map Time:")
        REORDER_TIME=$(echo "$OUTPUT" | grep -oP '(?:Reorder|Relabel)\s+Map\s+Time:\s+\K[\d.]+' | tail -1)
        if [ -z "$REORDER_TIME" ]; then
            REORDER_TIME=$(echo "$OUTPUT" | grep -oP 'Total Reorder Time:\s+\K[\d.]+' | tail -1)
        fi
        if [ -z "$REORDER_TIME" ]; then
            REORDER_TIME="N/A"
        fi
        
        # Extract average PR time
        AVG_TIME=$(echo "$OUTPUT" | grep -oP 'Average Time:\s+\K[\d.]+' | tail -1)
        if [ -z "$AVG_TIME" ]; then
            AVG_TIME="N/A"
        fi
        
        printf "  %-35s %12s %12s\n" "$LABEL" "$REORDER_TIME" "$AVG_TIME"
        echo "$GNAME,$CLI_OPT,$LABEL,$REORDER_TIME,$AVG_TIME,$TRIALS" >> "$OUTFILE"
    done
    printf "\n"
done

printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
printf "Results saved to: %s\n" "$OUTFILE"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
