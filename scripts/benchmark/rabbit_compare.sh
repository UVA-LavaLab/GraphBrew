#!/usr/bin/env bash
# RabbitOrder variant comparison: boost vs csr
# Usage: bash scripts/benchmark/rabbit_compare.sh [tag] [trials]
set -euo pipefail

TAG="${1:-rabbit_iter}"
TRIALS="${2:-10}"
GRAPHS=(
    "soc-LiveJournal1"
    "indochina-2004"
    "com-Orkut"
)
ALGOS=("pr" "cc" "bfs")
VARIANTS=("boost" "csr")
OUTDIR="results"
CSV="$OUTDIR/rabbit_${TAG}.csv"

echo "graph,algo,variant,trial,time_ms" > "$CSV"
echo "=== RabbitOrder Comparison: $TAG ==="
echo "Trials=$TRIALS, Graphs=${#GRAPHS[@]}, Algos=${#ALGOS[@]}, Variants=${#VARIANTS[@]}"
echo "Total runs: $(( ${#GRAPHS[@]} * ${#ALGOS[@]} * (${#VARIANTS[@]} + 1) * TRIALS ))"
echo ""

for graph in "${GRAPHS[@]}"; do
    SG="results/graphs/$graph/$graph.sg"
    if [[ ! -f "$SG" ]]; then
        echo "SKIP $graph (no .sg file)"
        continue
    fi

    for algo in "${ALGOS[@]}"; do
        BIN="bench/bin/$algo"
        
        # ORIGINAL (no reorder)
        echo -n "$graph $algo ORIGINAL: "
        for t in $(seq 1 "$TRIALS"); do
            ms=$("$BIN" -sf "$SG" -o 0 -n 1 2>&1 | grep "Trial Time:" | awk '{printf "%.1f", $NF * 1000}')
            echo "$graph,$algo,ORIGINAL,$t,$ms" >> "$CSV"
            echo -n "${ms} "
        done
        echo ""

        for variant in "${VARIANTS[@]}"; do
            echo -n "$graph $algo $variant: "
            for t in $(seq 1 "$TRIALS"); do
                ms=$("$BIN" -sf "$SG" -o "8:$variant" -n 1 2>&1 | grep "Trial Time:" | awk '{printf "%.1f", $NF * 1000}')
                echo "$graph,$algo,$variant,$t,$ms" >> "$CSV"
                echo -n "${ms} "
            done
            echo ""
        done
    done
    echo "---"
done

echo ""
echo "=== Summary ==="
echo "Results saved to $CSV"

# Print summary table
python3 -c "
import csv, statistics
data = {}
with open('$CSV') as f:
    for r in csv.DictReader(f):
        key = (r['graph'], r['algo'], r['variant'])
        data.setdefault(key, []).append(float(r['time_ms']))

print(f'{'Graph':<25} {'Algo':<6} {'Variant':<10} {'Mean':>8} {'Stdev':>8} {'csr/boost':>10}')
print('-' * 75)
for graph in sorted(set(k[0] for k in data)):
    for algo in sorted(set(k[1] for k in data)):
        boost_mean = None
        for variant in ['ORIGINAL', 'boost', 'csr']:
            key = (graph, algo, variant)
            if key not in data:
                continue
            vals = data[key]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            if variant == 'boost':
                boost_mean = m
            ratio = ''
            if variant == 'csr' and boost_mean and boost_mean > 0:
                ratio = f'{m/boost_mean:.3f}'
            print(f'{graph:<25} {algo:<6} {variant:<10} {m:>8.1f} {s:>8.1f} {ratio:>10}')
        print()
"
