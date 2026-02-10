#!/usr/bin/env bash
# Fresh benchmark: run all AdaptiveOrder-eligible algorithms on .sg files
# Includes all 5 benchmark variants: bfs, pr, pr_spmv, cc, cc_sv
#
# Usage: bash scripts/benchmark_fresh.sh

set -e
cd "$(dirname "$0")/.."

BIN=bench/bin
GRAPHS_DIR=results/graphs
OUTFILE="results/benchmark_fresh.json"
TRIALS=3

# AdaptiveOrder-eligible algorithms (by -o ID)
# Skip GOrder (9) — too slow on large graphs
# Skip GraphBrewOrder (12) and LeidenOrder (15) — they crash on some .sg files
ALGOS=(0 1 2 3 4 5 6 7 8 10 11)
ALGO_NAMES=(
    "ORIGINAL"          # 0
    "Random"            # 1
    "Sort"              # 2
    "HubSort"           # 3
    "HubCluster"        # 4
    "DBG"               # 5
    "HubSortDBG"        # 6
    "HubClusterDBG"     # 7
    "RabbitOrder"       # 8
    "GOrder"            # 9
    "COrder"            # 10
    "RCMOrder"          # 11
)

# All graphs with .sg files — sorted by size for predictable order
GRAPHS=(
    soc-Epinions1
    soc-Slashdot0902
    cnr-2000
    web-BerkStan
    web-Google
    com-Youtube
    as-Skitter
    roadNet-CA
    wiki-topcats
    cit-Patents
    soc-LiveJournal1
)

echo "Found ${#GRAPHS[@]} graphs with .sg files"
echo "Algorithms: ${ALGOS[*]}"

# All 5 benchmarks  
BENCHMARKS=(bfs pr pr_spmv cc cc_sv)

echo "[" > "$OUTFILE"
first=true
total=0
failed=0

for graph in "${GRAPHS[@]}"; do
    sg_file="$GRAPHS_DIR/$graph/$graph.sg"
    [ -f "$sg_file" ] || continue
    
    # Get node count for GOrder guard
    nodes=$(python3 -c "
import json, os
fp = os.path.join('$GRAPHS_DIR', '$graph', 'features.json')
if os.path.isfile(fp):
    d = json.load(open(fp))
    print(d.get('nodes', 0))
else:
    print(0)
" 2>/dev/null || echo 0)
    
    echo ""
    echo "================================================================"  
    echo "=== $graph (${nodes} nodes) ==="
    echo "================================================================"
    
    for bench in "${BENCHMARKS[@]}"; do
        for algo_id in "${ALGOS[@]}"; do
            algo_name="${ALGO_NAMES[$algo_id]}"
            [ -z "$algo_name" ] && continue
            
            # Skip GOrder for graphs > 500K nodes
            if [ "$algo_id" = "9" ] && [ "$nodes" -gt 500000 ]; then
                continue
            fi
            
            # Skip COrder for graphs > 2M nodes
            if [ "$algo_id" = "10" ] && [ "$nodes" -gt 2000000 ]; then
                continue
            fi
            
            echo -n "  $bench/$algo_name... "
            
            output=$(timeout 120 "$BIN/$bench" -f "$sg_file" -o "$algo_id" -n "$TRIALS" 2>&1) || {
                echo "TIMEOUT/ERROR"
                ((failed++)) || true
                continue
            }
            
            avg_time=$(echo "$output" | grep "Average Time:" | head -1 | grep -oP '[\d.]+' || echo "")
            reorder_time=$(echo "$output" | grep "Reorder Time:" | head -1 | grep -oP '[\d.]+' || echo "0")
            
            if [ -n "$avg_time" ]; then
                echo "${avg_time}s (reorder: ${reorder_time}s)"
                $first || echo "," >> "$OUTFILE"
                first=false
                cat >> "$OUTFILE" <<EOF
  {"graph": "$graph", "algorithm": "$algo_name", "algorithm_id": $algo_id, "benchmark": "$bench", "time_seconds": $avg_time, "reorder_time": ${reorder_time:-0}, "trials": $TRIALS, "success": true, "error": "", "extra": "sg_benchmark"}
EOF
                ((total++)) || true
            else
                echo "FAILED (no avg time)"
                ((failed++)) || true
            fi
        done
    done
done

echo "" >> "$OUTFILE"
echo "]" >> "$OUTFILE"

echo ""
echo "================================================================"
echo "Done. $total entries in $OUTFILE ($failed failed)"
echo "================================================================"
