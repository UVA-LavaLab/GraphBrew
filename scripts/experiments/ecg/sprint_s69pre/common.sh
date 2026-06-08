#!/usr/bin/env bash
# common.sh — shared helpers for sprint S68 milestone scripts.
#
# Every milestone script sources this file. Common helpers:
#   - REPO / SPRINT_DIR / RESULTS_DIR path constants
#   - log() / die() / step()
#   - csv_field() / csv_int_field() — CSV column extraction
#   - milestone_done / milestone_fail — write a JSON status file the
#     driver + rubber-duck can read
#   - require_clean_workdir / require_no_running_gem5
#
# Convention: each milestone produces:
#   ${RESULTS_DIR}/<m_id>/status.json   { "status": "ok|fail|blocked", ... }
#   ${RESULTS_DIR}/<m_id>/log.txt       human-readable trace

set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
SPRINT_DIR="${REPO}/scripts/experiments/ecg/sprint_s69pre"
RESULTS_DIR="${REPO}/results/sprint_s69pre"
mkdir -p "${RESULTS_DIR}"

log() { printf '[s68] %s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE:-/dev/stderr}"; }
die() { log "FATAL: $*"; exit 99; }
step() { log "── $* ──"; }

# csv_field <csv_path> <field_name> — print the value from the FIRST data row
csv_field() {
  local csv="$1"; local field="$2"
  [ -f "$csv" ] || { echo ""; return; }
  python3 -c "
import csv, sys
with open('$csv') as f:
    r = csv.DictReader(f)
    for row in r:
        print(row.get('$field', ''))
        break
"
}

# csv_int_field <csv_path> <field_name> — same but coerce to int (default 0)
csv_int_field() {
  local v
  v="$(csv_field "$1" "$2")"
  if [ -z "$v" ] || ! [[ "$v" =~ ^[0-9]+$ ]]; then echo 0; else echo "$v"; fi
}

# milestone_done <m_id> <key=value>...
milestone_done() {
  local m_id="$1"; shift
  local mdir="${RESULTS_DIR}/${m_id}"
  mkdir -p "$mdir"
  python3 - "$m_id" "$mdir" "$@" <<'PY'
import json, sys, datetime
m_id, mdir = sys.argv[1], sys.argv[2]
data = {"milestone": m_id,
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "evidence": {}}
for kv in sys.argv[3:]:
    if "=" in kv:
        k, v = kv.split("=", 1)
        data["evidence"][k] = v
with open(f"{mdir}/status.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"[s68] milestone {m_id} → OK")
PY
}

# milestone_fail <m_id> <reason> [<key=value>...]
milestone_fail() {
  local m_id="$1"; local reason="$2"; shift 2
  local mdir="${RESULTS_DIR}/${m_id}"
  mkdir -p "$mdir"
  python3 - "$m_id" "$mdir" "$reason" "$@" <<'PY'
import json, sys, datetime
m_id, mdir, reason = sys.argv[1], sys.argv[2], sys.argv[3]
data = {"milestone": m_id,
        "status": "blocked",
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(),
        "evidence": {}}
for kv in sys.argv[4:]:
    if "=" in kv:
        k, v = kv.split("=", 1)
        data["evidence"][k] = v
with open(f"{mdir}/status.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"[s68] milestone {m_id} → BLOCKED: {reason}")
PY
}

# SQL helpers — update the session todo store
todo_set_status() {
  local id="$1"; local status="$2"
  # Use the agent-side SQL via sqlite if available; for shell execution we
  # leverage a tiny helper file the driver passes via env.
  if [ -n "${SQL_TODO_HELPER:-}" ] && [ -x "${SQL_TODO_HELPER}" ]; then
    "${SQL_TODO_HELPER}" "$id" "$status"
  else
    # fallback: just log; the driver will reconcile via session SQL
    log "TODO ${id} → ${status}  (no SQL helper wired, driver will reconcile)"
  fi
}

require_no_running_gem5() {
  if pgrep -af "gem5\.opt" >/dev/null 2>&1; then
    die "another gem5 instance is running; refusing to start a conflicting sweep"
  fi
}

# Standard smoke runner for ECG_PFX / DROPLET / none on a graph
run_gem5_smoke() {
  local arm="$1"; local graph_path="$2"; local out_dir="$3"
  local timeout_s="${4:-540}"
  mkdir -p "$out_dir"
  local pfx_arg=() mode_arg=()
  case "$arm" in
    DROPLET) pfx_arg=(--prefetcher DROPLET --prefetcher-level l2) ;;
    ECG_PFX) pfx_arg=(--prefetcher ECG_PFX --prefetcher-level l2 --allow-gem5-ecg-pfx)
             mode_arg=(--ecg-pfx-mode per_edge) ;;
    none)    ;;
    *) die "unknown arm: $arm" ;;
  esac
  ECG_CONTAINER_BITS=64 timeout "$timeout_s" python3 "${REPO}/scripts/experiments/ecg/roi_matrix.py" \
      --suite gem5 --no-build \
      --benchmark pr \
      --options "-f ${graph_path} -s -o 5 -n 1 -i 2" \
      --policies LRU \
      "${pfx_arg[@]}" "${mode_arg[@]}" \
      --l1d-size 32kB --l1d-ways 8 \
      --l2-size 256kB --l2-ways 8 \
      --l3-sizes 1MB --l3-ways 16 \
      --line-size 64 \
      --timeout-gem5 "$((timeout_s - 60))" \
      --out-dir "$out_dir"
}
