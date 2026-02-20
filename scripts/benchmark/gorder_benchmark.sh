#!/usr/bin/env bash
# gorder_benchmark.sh — Comprehensive GOrder variant benchmark
# Measures reorder time + algorithm quality across all graphs & benchmarks
#
# Usage: bash scripts/benchmark/gorder_benchmark.sh [timeout_seconds]
#
# Output: results/gorder_benchmark_TIMESTAMP.csv
set -o pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TIMEOUT="${1:-600}"  # Default 10min timeout per run (matches TIMEOUT_BENCHMARK)
TRIALS=3
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTFILE="${REPO}/results/gorder_benchmark_${TIMESTAMP}.csv"

# Variants: baseline (no reorder) + 3 GOrder variants
VARIANTS=("0" "9" "9:csr" "9:fast")
VARIANT_NAMES=("ORIGINAL" "GOrder_default" "GOrder_csr" "GOrder_fast")

# 5 core benchmarks (tc excluded — all graphs are directed, tc requires undirected)
BENCHMARKS=("pr" "bfs" "sssp" "bc" "cc")

# All available graphs sorted by size
GRAPHS=(
    "ca-GrQc"
    "web-Google"
    "roadNet-CA"
    "cit-Patents"
    "as-Skitter"
    "soc-LiveJournal1"
    "indochina-2004"
    "com-Orkut"
)

# CSV header
echo "graph,variant,benchmark,reorder_time_s,avg_trial_time_s,status" > "$OUTFILE"
echo "=== GOrder Variant Benchmark ==="
echo "Output: $OUTFILE"
echo "Timeout: ${TIMEOUT}s per run, Trials: $TRIALS"
echo "Variants: ${VARIANT_NAMES[*]}"
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "Graphs: ${GRAPHS[*]}"
echo ""

total=$((${#GRAPHS[@]} * ${#VARIANTS[@]} * ${#BENCHMARKS[@]}))
done_count=0

for graph in "${GRAPHS[@]}"; do
    sg_file="${REPO}/results/graphs/${graph}/${graph}.sg"
    if [ ! -f "$sg_file" ]; then
        echo "SKIP: $sg_file not found"
        continue
    fi
    
    sg_size=$(stat -c%s "$sg_file" 2>/dev/null || echo 0)
    sg_mb=$((sg_size / 1048576))
    echo "=== Graph: $graph (${sg_mb}MB) ==="
    
    for vi in "${!VARIANTS[@]}"; do
        variant="${VARIANTS[$vi]}"
        vname="${VARIANT_NAMES[$vi]}"
        
        for bench in "${BENCHMARKS[@]}"; do
            done_count=$((done_count + 1))
            pct=$((done_count * 100 / total))
            
            bin="${REPO}/bench/bin/${bench}"
            if [ ! -x "$bin" ]; then
                echo "  [$pct%] $vname/$bench: binary not found"
                echo "$graph,$vname,$bench,,,missing_binary" >> "$OUTFILE"
                continue
            fi
            
            # Build command — ORIGINAL has no -o flag
            if [ "$variant" = "0" ]; then
                cmd="$bin -f $sg_file -n $TRIALS"
            else
                cmd="$bin -f $sg_file -o $variant -n $TRIALS"
            fi
            
            printf "  [%3d%%] %-18s %-4s ... " "$pct" "$vname" "$bench"
            
            # Run with timeout, capture output
            exit_code=0
            output=$(timeout "$TIMEOUT" $cmd 2>&1) || exit_code=$?
            
            if [ "$exit_code" -eq 124 ] || [ "$exit_code" -eq 143 ]; then
                echo "TIMEOUT (${TIMEOUT}s)"
                echo "$graph,$vname,$bench,,,timeout" >> "$OUTFILE"
                continue
            elif [ "$exit_code" -ne 0 ]; then
                echo "FAIL (exit=$exit_code)"
                echo "$graph,$vname,$bench,,,error_$exit_code" >> "$OUTFILE"
                continue
            fi
            
            # Parse reorder time
            reorder_time=$(echo "$output" | grep -oP 'Reorder Time:\s+\K[0-9.]+' | tail -1 || echo "")
            if [ "$variant" = "0" ]; then
                reorder_time="0"
            fi
            
            # Parse average trial time
            avg_time=$(echo "$output" | grep -oP 'Average Time:\s+\K[0-9.]+' | tail -1 || echo "")
            
            if [ -n "$avg_time" ]; then
                printf "reorder=%-8s avg=%-8s\n" "${reorder_time:-n/a}" "$avg_time"
                echo "$graph,$vname,$bench,${reorder_time:-},${avg_time},ok" >> "$OUTFILE"
            else
                echo "PARSE_FAIL"
                echo "$graph,$vname,$bench,,,parse_fail" >> "$OUTFILE"
            fi
        done
    done
done

echo ""
echo "=== Benchmark complete ==="
echo "Results: $OUTFILE"
echo "Total runs: $done_count"
wc -l < "$OUTFILE" | xargs -I{} echo "CSV rows (inc header): {}"
