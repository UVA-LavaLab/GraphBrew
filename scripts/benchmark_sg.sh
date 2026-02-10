#!/usr/bin/env bash
# Quick benchmark: run all AdaptiveOrder-eligible algorithms on .sg files
# Collects training data that matches C++ runtime features

set -e
cd "$(dirname "$0")/.."

BIN=bench/bin
GRAPHS_DIR=results/graphs
OUTFILE="results/benchmark_sg_$(date +%Y%m%d_%H%M%S).json"
TRIALS=3

# AdaptiveOrder-eligible algorithms (by -o ID)
# 0=ORIGINAL 1=SORT 2=RANDOM 3=DBG 4=HUBSORT 5=HUBSORTDBG
# 6=HUBCLUSTER 7=HUBCLUSTERDBG 8=CORDER 9=RCM 10=GORDER
# 11=RABBITORDER 13=LeidenOrder 15=GraphBrewOrder
ALGOS="0 1 2 3 4 5 6 7 8 9 11 13 15"
# Skip GOrder (10) for large graphs - too slow

ALGO_NAMES=("ORIGINAL" "SORT" "RANDOM" "DBG" "HUBSORT" "HUBSORTDBG" 
            "HUBCLUSTER" "HUBCLUSTERDBG" "CORDER" "RCM" "GORDER"
            "RABBITORDER_csr" "" "LeidenOrder" "" "GraphBrewOrder_leiden")

# Focus on graphs that have .sg files AND are large enough to matter
GRAPHS=(
    web-BerkStan
    cnr-2000
    roadNet-CA
    as-Skitter
    com-Youtube
    soc-Epinions1
    soc-Slashdot0902
    web-Google
)

BENCHMARKS=(bfs pr cc)

echo "["  > "$OUTFILE"
first=true

for graph in "${GRAPHS[@]}"; do
    sg_file="$GRAPHS_DIR/$graph/$graph.sg"
    [ -f "$sg_file" ] || { echo "SKIP: $sg_file not found"; continue; }
    
    echo "=== $graph ==="
    
    for bench in "${BENCHMARKS[@]}"; do
        for algo_id in $ALGOS; do
            algo_name="${ALGO_NAMES[$algo_id]}"
            [ -z "$algo_name" ] && continue
            
            # Skip GOrder for graphs > 500K nodes (too slow)
            if [ "$algo_id" = "10" ]; then
                nodes=$(grep -oP '"nodes":\s*\K\d+' "$GRAPHS_DIR/$graph/features.json" 2>/dev/null || echo 0)
                [ "$nodes" -gt 500000 ] && continue
            fi
            
            echo -n "  $bench/$algo_name... "
            
            output=$($BIN/$bench -f "$sg_file" -o "$algo_id" -n "$TRIALS" 2>&1)
            avg_time=$(echo "$output" | grep "Average Time:" | head -1 | grep -oP '[\d.]+' || echo "")
            reorder_time=$(echo "$output" | grep "Reorder Time:" | head -1 | grep -oP '[\d.]+' || echo "0")
            
            if [ -n "$avg_time" ]; then
                echo "${avg_time}s"
                $first || echo "," >> "$OUTFILE"
                first=false
                cat >> "$OUTFILE" <<EOF
  {"graph": "$graph", "algorithm": "$algo_name", "algorithm_id": $algo_id, "benchmark": "$bench", "time_seconds": $avg_time, "reorder_time": ${reorder_time:-0}, "trials": $TRIALS, "success": true, "error": "", "extra": "sg_benchmark"}
EOF
            else
                echo "FAILED"
            fi
        done
    done
done

echo "]" >> "$OUTFILE"
echo ""
echo "Results saved to: $OUTFILE"
echo "Entries: $(grep -c '"graph"' "$OUTFILE")"
