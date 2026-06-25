#!/usr/bin/env bash
# pfx_sniper_mode6_status.sh — quick status check for the running
# Sniper mode 6 cross-sim sweep launched in sprint 6f-6.
#
# Reports: cells completed/in-progress, runner log tail, active Sniper
# procs, and a count of csv files emitted so far.

set -u

OUT_ROOT="${1:-/tmp/graphbrew-pfx-sniper-mode6}"

if [ ! -d "$OUT_ROOT" ]; then
  echo "OUT_ROOT not found: $OUT_ROOT"
  exit 1
fi

LOG="$OUT_ROOT/runner_pfx_sniper_mode6.log"
echo "=== Runner log tail ==="
[ -f "$LOG" ] && tail -10 "$LOG" || echo "(no log)"

echo
echo "=== Cells with results ==="
COMPLETED=$(find "$OUT_ROOT" -name "roi_matrix.csv" -size +0 2>/dev/null | sort)
if [ -z "$COMPLETED" ]; then
  echo "(none)"
else
  echo "$COMPLETED" | while read csv; do
    cell="${csv#$OUT_ROOT/}"
    cell="${cell%/roi_matrix.csv}"
    status=$(awk -F, '
      NR==1 { for (i=1;i<=NF;i++) if($i=="status") si=i; next }
      si { print $si; exit }
    ' "$csv" 2>/dev/null)
    rate=$(awk -F, '
      NR==1 { for (i=1;i<=NF;i++) if($i=="l3_miss_rate") li=i; next }
      li { print $li; exit }
    ' "$csv" 2>/dev/null)
    echo "  $cell  status=$status l3_miss_rate=$rate"
  done
fi

echo
echo "=== Active Sniper processes ==="
ps -eo pid,pcpu,rss,etime,comm | grep -E "sg_kernel|^[[:space:]]*[0-9]+[[:space:]]+[0-9.]+[[:space:]]+[0-9]+[[:space:]]+[^ ]+[[:space:]]+sniper$" | grep -v grep || echo "(none)"

echo
echo "=== Output files (recent first) ==="
find "$OUT_ROOT" -type f -mmin -60 2>/dev/null | sort -r | head -8
