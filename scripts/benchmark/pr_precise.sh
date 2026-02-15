#!/bin/bash
# Precise PR-only comparison: 20 trials × 3 graphs × 3 variants
set -uo pipefail

REPO="/home/ab/Documents/00_github_repos/02_GraphBrew"
PR="$REPO/bench/bin/pr"
GRAPHS_DIR="$REPO/results/graphs"
TRIALS=20

GRAPHS=("soc-LiveJournal1" "indochina-2004" "com-Orkut")
VARIANTS=("original" "default" "csr")

ITER="${1:-iter_pr}"
OUTFILE="$REPO/results/gorder_${ITER}.csv"
echo "graph,variant,trial,time_ms" > "$OUTFILE"
echo "=== PR Precision Test: $ITER ($TRIALS trials) ==="

for graph in "${GRAPHS[@]}"; do
    SG="$GRAPHS_DIR/$graph/${graph}.sg"
    [[ ! -f "$SG" ]] && { echo "SKIP: $SG"; continue; }

    for variant in "${VARIANTS[@]}"; do
        case "$variant" in
            original) FLAG="-o 0" ;;
            default)  FLAG="-o 9" ;;
            csr)      FLAG="-o 9:csr" ;;
        esac

        printf "  %s/pr/%s: " "$graph" "$variant"
        for trial in $(seq 1 $TRIALS); do
            t=$($PR -sf "$SG" $FLAG -n 1 2>/dev/null | grep 'Trial Time:' | awk '{print $NF}')
            if [[ -n "$t" ]]; then
                ms=$(echo "$t * 1000" | bc -l)
                echo "$graph,$variant,$trial,$ms" >> "$OUTFILE"
                printf "%.0f " "$ms"
            else
                echo "$graph,$variant,$trial,ERR" >> "$OUTFILE"
                printf "E "
            fi
        done
        echo ""
    done
done

echo ""
echo "=== Summary (median ± IQR) ==="
awk -F',' 'NR>1 && $4!="ERR" {
    key = $1 "," $2
    times[key][++count[key]] = $4+0
}
END {
    for (key in count) {
        n = count[key]
        # Sort
        for (i=1; i<=n; i++) for (j=i+1; j<=n; j++)
            if (times[key][i] > times[key][j]) {
                tmp=times[key][i]; times[key][i]=times[key][j]; times[key][j]=tmp
            }
        q1 = times[key][int(n*0.25)+1]
        med = (n%2==1) ? times[key][int(n/2)+1] : (times[key][n/2]+times[key][n/2+1])/2
        q3 = times[key][int(n*0.75)+1]
        medians[key] = med
        printf "%s: median=%.1f  IQR=[%.1f, %.1f]  n=%d\n", key, med, q1, q3, n
    }
    # Compute speedups
    printf "\n--- Speedups ---\n"
    for (key in medians) {
        split(key, parts, ",")
        orig_key = parts[1] ",original"
        def_key = parts[1] ",default"
        if (parts[2] == "csr" && orig_key in medians && def_key in medians) {
            orig = medians[orig_key]
            def = medians[def_key]
            csr = medians[key]
            printf "%s: orig/csr=%.3fx  orig/def=%.3fx  csr/def=%.3f  delta=%.1f%%\n",
                parts[1], orig/csr, orig/def, csr/def, (1-csr/def)*100
        }
    }
}' "$OUTFILE"

echo ""
echo "Done. $OUTFILE"
