#!/usr/bin/env bash
# gorder_iterate.sh — Iterative GOrder CSR quality benchmark agent
#
# Rebuilds binaries, runs BFS/PR/CC on large graphs (10 trials),
# compares CSR vs default, and reports delta.
#
# Usage: bash scripts/benchmark/gorder_iterate.sh [iteration_label]
#
# Output: results/gorder_iter_LABEL.csv  +  console summary
set -o pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="${1:-$(date +%Y%m%d_%H%M%S)}"
TRIALS=10
OUTFILE="${REPO}/results/gorder_iter_${LABEL}.csv"

# Stable benchmarks only (source-independent or consistent-source)
BENCHMARKS=("bfs" "pr" "cc")

# Large graphs only — small graphs are too noisy
GRAPHS=("soc-LiveJournal1" "indochina-2004" "com-Orkut")

# Variants to compare
VARIANTS=("0" "9" "9:csr")
VARIANT_NAMES=("ORIGINAL" "GOrder_default" "GOrder_csr")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          GOrder CSR Quality Iteration: ${LABEL}            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# --- Step 1: Rebuild ---
echo ""
echo ">>> Step 1: Rebuilding binaries..."
cd "$REPO"
make clean -j >/dev/null 2>&1
if ! make -j$(nproc) 2>&1 | tail -3; then
    echo "BUILD FAILED — aborting"
    exit 1
fi
echo "    Build OK"

# --- Step 2: Benchmark ---
echo ""
echo ">>> Step 2: Benchmarking (${#GRAPHS[@]} graphs × ${#VARIANTS[@]} variants × ${#BENCHMARKS[@]} benchmarks × ${TRIALS} trials)"
echo "graph,variant,benchmark,reorder_time_s,avg_trial_time_s,status" > "$OUTFILE"

total=$((${#GRAPHS[@]} * ${#VARIANTS[@]} * ${#BENCHMARKS[@]}))
done_count=0

for graph in "${GRAPHS[@]}"; do
    sg_file="${REPO}/results/graphs/${graph}/${graph}.sg"
    if [ ! -f "$sg_file" ]; then
        echo "SKIP: $sg_file not found"
        continue
    fi
    echo ""
    echo "--- $graph ---"

    for vi in "${!VARIANTS[@]}"; do
        variant="${VARIANTS[$vi]}"
        vname="${VARIANT_NAMES[$vi]}"

        for bench in "${BENCHMARKS[@]}"; do
            done_count=$((done_count + 1))
            pct=$((done_count * 100 / total))

            bin="${REPO}/bench/bin/${bench}"

            if [ "$variant" = "0" ]; then
                cmd="$bin -f $sg_file -n $TRIALS"
            else
                cmd="$bin -f $sg_file -o $variant -n $TRIALS"
            fi

            printf "  [%3d%%] %-18s %-4s ... " "$pct" "$vname" "$bench"

            exit_code=0
            output=$(timeout 600 $cmd 2>&1) || exit_code=$?

            if [ "$exit_code" -eq 124 ] || [ "$exit_code" -eq 143 ]; then
                echo "TIMEOUT"
                echo "$graph,$vname,$bench,,,timeout" >> "$OUTFILE"
                continue
            elif [ "$exit_code" -ne 0 ]; then
                echo "FAIL (exit=$exit_code)"
                echo "$graph,$vname,$bench,,,error_$exit_code" >> "$OUTFILE"
                continue
            fi

            reorder_time=$(echo "$output" | grep -oP 'Reorder Time:\s+\K[0-9.]+' | tail -1 || echo "")
            [ "$variant" = "0" ] && reorder_time="0"
            avg_time=$(echo "$output" | grep -oP 'Average Time:\s+\K[0-9.]+' | tail -1 || echo "")

            if [ -n "$avg_time" ]; then
                printf "reorder=%-8s avg=%-10s\n" "${reorder_time:-n/a}" "$avg_time"
                echo "$graph,$vname,$bench,${reorder_time:-},${avg_time},ok" >> "$OUTFILE"
            else
                echo "PARSE_FAIL"
                echo "$graph,$vname,$bench,,,parse_fail" >> "$OUTFILE"
            fi
        done
    done
done

# --- Step 3: Analysis ---
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      RESULTS ANALYSIS                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Use awk to compute speedups and comparison
awk -F',' '
NR==1 { next }
$6 != "ok" { next }
{
    key = $1 "," $3
    time[key "," $2] = $5
    reorder[key "," $2] = $4
}
END {
    # Print per-benchmark comparison table
    split("bfs,pr,cc", benches, ",")
    split("soc-LiveJournal1,indochina-2004,com-Orkut", graphs, ",")

    printf "\n%-20s %-6s %12s %12s %12s   %8s %8s\n", \
        "Graph", "Bench", "ORIGINAL", "default", "CSR", "def_spd", "csr_spd"
    printf "%-20s %-6s %12s %12s %12s   %8s %8s\n", \
        "--------------------", "------", "------------", "------------", "------------", "--------", "--------"

    total_def_wins = 0; total_csr_wins = 0; total_ties = 0
    sum_log_def = 0; sum_log_csr = 0; n_valid = 0

    for (gi = 1; gi <= 3; gi++) {
        g = graphs[gi]
        for (bi = 1; bi <= 3; bi++) {
            b = benches[bi]
            k = g "," b
            t_o = time[k ",ORIGINAL"] + 0
            t_d = time[k ",GOrder_default"] + 0
            t_c = time[k ",GOrder_csr"] + 0

            if (t_o > 0 && t_d > 0 && t_c > 0) {
                spd_d = t_o / t_d
                spd_c = t_o / t_c
                marker = ""
                if (spd_c > spd_d * 1.02) { marker = " ✓CSR"; total_csr_wins++ }
                else if (spd_d > spd_c * 1.02) { marker = " ✗CSR"; total_def_wins++ }
                else { marker = " ≈"; total_ties++ }

                printf "%-20s %-6s %12.5f %12.5f %12.5f   %7.3fx %7.3fx%s\n", \
                    g, b, t_o, t_d, t_c, spd_d, spd_c, marker
                sum_log_def += log(spd_d)
                sum_log_csr += log(spd_c)
                n_valid++
            }
        }
    }

    if (n_valid > 0) {
        geo_def = exp(sum_log_def / n_valid)
        geo_csr = exp(sum_log_csr / n_valid)
        printf "\n"
        printf "Geometric Mean Speedup vs ORIGINAL:\n"
        printf "  GOrder_default:  %.4fx\n", geo_def
        printf "  GOrder_csr:      %.4fx\n", geo_csr
        printf "  CSR/default:     %.4fx  (>1 = CSR better)\n", geo_csr / geo_def
        printf "\nScore: default wins %d, CSR wins %d, ties %d\n", total_def_wins, total_csr_wins, total_ties
        printf "\n"
        if (geo_csr >= geo_def * 0.98) {
            printf ">>> STATUS: CSR MATCHES or BEATS default (within 2%%)\n"
        } else {
            printf ">>> STATUS: CSR still BEHIND default — ratio %.4f (need >= 0.98)\n", geo_csr / geo_def
        }
    }

    # Reorder time comparison
    printf "\nReorder Time Comparison (from first benchmark per graph):\n"
    printf "%-20s %12s %12s %8s\n", "Graph", "default(s)", "CSR(s)", "speedup"
    for (gi = 1; gi <= 3; gi++) {
        g = graphs[gi]
        b = benches[1]
        k = g "," b
        r_d = reorder[k ",GOrder_default"] + 0
        r_c = reorder[k ",GOrder_csr"] + 0
        if (r_c > 0) sp = r_d / r_c; else sp = 0
        printf "%-20s %12.3f %12.3f %7.2fx\n", g, r_d, r_c, sp
    }
}
' "$OUTFILE"

echo ""
echo "Results saved: $OUTFILE"
echo "Re-run after code changes: bash scripts/benchmark/gorder_iterate.sh iter2"
