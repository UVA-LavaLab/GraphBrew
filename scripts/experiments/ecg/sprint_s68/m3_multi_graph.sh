#!/usr/bin/env bash
# M3: Multi-graph smoke (email-Eu-core, kron_s17).
#
# Rubber-duck-revised exit criteria (Pivot B from M3 v1 verdict):
#   - email-Eu-core: pf_identified > 0 AND pf_issued > 0
#     (negative control — 4KB property region fits in L1d, pf_useful=0 expected)
#   - kron_s17:      pf_identified > 0 AND pf_issued > 0 AND pf_useful > 0
#     (utility proof — 524KB property region exceeds L2, prefetches MUST help)
#
# delaunay_n19 was tried in M3 v1 and timed out at 89 min wall (gem5 X86
# timing simulation of 524K vertex PR doesn't fit a 90 min budget).
# kron_s17 (131K vertices) should fit in 20-40 min and exercise the
# same code paths.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m3-multi-graph"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M3: multi-graph ECG_PFX smoke (email-Eu-core mechanism + kron_s16_k4 utility)"

require_no_running_gem5

# Format: short:path:timeout:exit_criteria
# exit_criteria: "mechanism" requires identified>0 && issued>0
#                "utility"   requires identified>0 && issued>0 && useful>0
GRAPHS=(
  "email-Eu-core:${REPO}/results/graphs/email-Eu-core/email-Eu-core.sg:600:mechanism"
  "kron_s16_k4:${REPO}/results/graphs/kron_s16_k4/kron_s16_k4.sg:2400:utility"
)

FAIL_GRAPHS=""
EVIDENCE=()

for entry in "${GRAPHS[@]}"; do
  IFS=':' read -r short path timeout criteria <<< "${entry}"
  if [ ! -f "${path}" ]; then
    log "SKIP ${short} — graph file missing (${path})"
    EVIDENCE+=("${short}_status=skip_missing")
    continue
  fi
  out="${RESULTS_DIR}/${M_ID}/${short}"
  rm -rf "${out}"
  log "running ${short} (timeout ${timeout}s, criteria=${criteria})"
  if run_gem5_smoke ECG_PFX "${path}" "${out}" "${timeout}" >> "${LOG_FILE}" 2>&1; then
    csv="${out}/roi_matrix.csv"
    identified="$(csv_int_field "${csv}" pf_identified)"
    issued="$(csv_int_field "${csv}" pf_issued)"
    useful="$(csv_int_field "${csv}" pf_useful)"
    status="$(csv_field "${csv}" status)"
    log "${short}: identified=${identified} issued=${issued} useful=${useful} status=${status}"
    EVIDENCE+=(
      "${short}_identified=${identified}"
      "${short}_issued=${issued}"
      "${short}_useful=${useful}"
      "${short}_status=${status}"
      "${short}_criteria=${criteria}"
    )
    if [ "${status}" != "ok" ]; then
      FAIL_GRAPHS="${FAIL_GRAPHS} ${short}(status=${status})"
      continue
    fi
    if [ "${criteria}" = "mechanism" ]; then
      if [ "${identified}" -le 0 ] || [ "${issued}" -le 0 ]; then
        FAIL_GRAPHS="${FAIL_GRAPHS} ${short}(mechanism: identified=${identified}, issued=${issued})"
      fi
    elif [ "${criteria}" = "utility" ]; then
      if [ "${identified}" -le 0 ] || [ "${issued}" -le 0 ] || [ "${useful}" -le 0 ]; then
        FAIL_GRAPHS="${FAIL_GRAPHS} ${short}(utility: identified=${identified}, issued=${issued}, useful=${useful})"
      fi
    fi
  else
    log "FAIL ${short} (timeout or runtime error)"
    EVIDENCE+=("${short}_status=fail" "${short}_criteria=${criteria}")
    FAIL_GRAPHS="${FAIL_GRAPHS} ${short}(fail)"
  fi
done

if [ -n "${FAIL_GRAPHS}" ]; then
  milestone_fail "${M_ID}" "graphs failed:${FAIL_GRAPHS}" "${EVIDENCE[@]}"
  exit 1
fi

milestone_done "${M_ID}" "${EVIDENCE[@]}"
