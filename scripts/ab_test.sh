#!/usr/bin/env bash
# A/B test: Adaptive (-o 14) vs Original (-o 0) across all graphs/benchmarks
set -uo pipefail
cd "$(dirname "$0")/.."

GRAPHS=(
  soc-Epinions1 soc-Slashdot0902 cnr-2000 web-BerkStan web-Google
  com-Youtube as-Skitter roadNet-CA wiki-topcats cit-Patents soc-LiveJournal1
)
BENCHMARKS=(bfs pr pr_spmv cc cc_sv)
TRIALS=3

printf "%-25s %-8s %12s %12s %8s\n" "Graph" "Bench" "Original(s)" "Adaptive(s)" "Speedup"
printf "%-25s %-8s %12s %12s %8s\n" "-------------------------" "--------" "------------" "------------" "--------"

total_orig=0
total_adap=0
wins=0
losses=0
ties=0

for graph in "${GRAPHS[@]}"; do
  sg="results/graphs/${graph}/${graph}.sg"
  [[ -f "$sg" ]] || continue
  for bench in "${BENCHMARKS[@]}"; do
    bin="bench/bin/${bench}"
    [[ -x "$bin" ]] || continue
    
    # Original
    orig_time=$("$bin" -f "$sg" -a 0 -o 0 -n $TRIALS 2>&1 | grep -oP 'Average Time:\s+\K[\d.]+' || echo "0")
    # Adaptive  
    adap_time=$("$bin" -f "$sg" -a 0 -o 14 -n $TRIALS 2>&1 | grep -oP 'Average Time:\s+\K[\d.]+' || echo "0")
    
    if [[ "$orig_time" != "0" && "$adap_time" != "0" ]]; then
      speedup=$(python3 -c "o=$orig_time; a=$adap_time; print(f'{o/a:.2f}x' if a>0 else 'inf')")
      ratio=$(python3 -c "o=$orig_time; a=$adap_time; print(f'{o/a:.4f}')")
      total_orig=$(python3 -c "print($total_orig + $orig_time)")
      total_adap=$(python3 -c "print($total_adap + $adap_time)")
      if python3 -c "exit(0 if $ratio > 1.05 else 1)"; then
        wins=$((wins+1))
      elif python3 -c "exit(0 if $ratio < 0.95 else 1)"; then
        losses=$((losses+1))
      else
        ties=$((ties+1))
      fi
      printf "%-25s %-8s %12.5f %12.5f %8s\n" "$graph" "$bench" "$orig_time" "$adap_time" "$speedup"
    fi
  done
done

echo ""
echo "=== Summary ==="
echo "Total Original: ${total_orig}s  Total Adaptive: ${total_adap}s"
overall=$(python3 -c "print(f'{$total_orig/$total_adap:.2f}x' if $total_adap > 0 else 'N/A')")
echo "Overall speedup: ${overall}"
echo "Wins: $wins  Ties: $ties  Losses: $losses"
