#!/bin/bash
# Iterative comparison: ORIGINAL vs GOrder default vs GOrder CSR
# BFS + PR on 3 large graphs, 10 trials each
set -uo pipefail

REPO="/home/ab/Documents/00_github_repos/02_GraphBrew"
BFS="$REPO/bench/bin/bfs"
PR="$REPO/bench/bin/pr"
GRAPHS_DIR="$REPO/results/graphs"
TRIALS=10

GRAPHS=("soc-LiveJournal1" "indochina-2004" "com-Orkut")
ALGOS=("bfs" "pr" "cc")
VARIANTS=("original" "default" "csr")

ITER="${1:-iter0}"
OUTFILE="$REPO/results/gorder_iter_${ITER}.csv"

echo "graph,algo,variant,trial,time_ms" > "$OUTFILE"
echo "=== GOrder iteration: $ITER ==="
echo "Output: $OUTFILE"

for graph in "${GRAPHS[@]}"; do
    SG="$GRAPHS_DIR/$graph/${graph}.sg"
    if [[ ! -f "$SG" ]]; then
        echo "SKIP: $SG not found"
        continue
    fi

    for algo in "${ALGOS[@]}"; do
        if [[ "$algo" == "bfs" ]]; then
            BIN="$BFS"
        elif [[ "$algo" == "cc" ]]; then
            BIN="$REPO/bench/bin/cc"
        else
            BIN="$PR"
        fi

        for variant in "${VARIANTS[@]}"; do
            echo -n "  $graph/$algo/$variant: "

            if [[ "$variant" == "original" ]]; then
                ORDER_FLAG="-o 0"
            elif [[ "$variant" == "default" ]]; then
                ORDER_FLAG="-o 9"
            elif [[ "$variant" == "csr" ]]; then
                ORDER_FLAG="-o 9:csr"
            fi

            for trial in $(seq 1 $TRIALS); do
                t=$($BIN -sf "$SG" $ORDER_FLAG -n 1 2>/dev/null | grep 'Trial Time:' | awk '{print $NF}')
                if [[ -n "$t" ]]; then
                    ms=$(echo "$t * 1000" | bc -l)
                    echo "$graph,$algo,$variant,$trial,$ms" >> "$OUTFILE"
                    printf "%.1f " "$ms"
                else
                    echo "$graph,$algo,$variant,$trial,ERR" >> "$OUTFILE"
                    printf "ERR "
                fi
            done
            echo ""
        done
    done
done

echo ""
echo "=== Summary ==="
echo ""

# Compute medians and speedups (using awk)
awk -F',' 'NR>1 && $5!="ERR" {
    key = $1 "," $2 "," $3
    vals[key] = vals[key] " " $5
    count[key]++
}
END {
    for (key in vals) {
        n = split(vals[key], a, " ")
        # Sort values for median
        for (i=1; i<=n; i++) for (j=i+1; j<=n; j++)
            if (a[i]+0 > a[j]+0) { tmp=a[i]; a[i]=a[j]; a[j]=tmp }
        med = (n%2==1) ? a[int(n/2)+1] : (a[n/2]+a[n/2+1])/2
        medians[key] = med
    }
    # Print results grouped by graph,algo
    for (key in medians) {
        split(key, parts, ",")
        ga = parts[1] "," parts[2]
        vars[ga] = vars[ga] "|" parts[3] ":" medians[key]
    }
    for (ga in vars) {
        split(ga, parts, ",")
        graph = parts[1]; algo = parts[2]
        # Extract medians for each variant
        orig = 0; def = 0; csr = 0
        n2 = split(vars[ga], items, "|")
        for (i=1; i<=n2; i++) {
            if (items[i] == "") continue
            split(items[i], kv, ":")
            if (kv[1] == "original") orig = kv[2]+0
            else if (kv[1] == "default") def = kv[2]+0
            else if (kv[1] == "csr") csr = kv[2]+0
        }
        if (orig > 0) {
            printf "%s %s: ORIG=%.1fms  default=%.1fms (%.2fx)  csr=%.1fms (%.2fx)  csr/default=%.3f\n",
                graph, algo, orig, def, orig/def, csr, orig/csr, (def>0 ? csr/def : 0)
        }
    }
}' "$OUTFILE"

echo ""
echo "Done. Results in $OUTFILE"
