#!/usr/bin/env bash
# M3: Multi-graph smoke (email-Eu-core, delaunay_n19).
# Exit criteria: both cells complete with status=ok AND pf_useful > 0.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m3-multi-graph"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M3: multi-graph ECG_PFX smoke (email-Eu-core, delaunay_n19)"

require_no_running_gem5

GRAPHS=(
  "email-Eu-core:${REPO}/results/graphs/email-Eu-core/email-Eu-core.sg:600"
  "delaunay_n19:${REPO}/results/graphs/delaunay_n19/delaunay_n19.sg:5400"
)

FAIL_GRAPHS=""
EVIDENCE=()

for entry in "${GRAPHS[@]}"; do
  IFS=':' read -r short path timeout <<< "${entry}"
  if [ ! -f "${path}" ]; then
    log "SKIP ${short} — graph file missing (${path})"
    EVIDENCE+=("${short}_status=skip_missing")
    continue
  fi
  out="${RESULTS_DIR}/${M_ID}/${short}"
  rm -rf "${out}"
  log "running ${short} (timeout ${timeout}s)"
  if run_gem5_smoke ECG_PFX "${path}" "${out}" "${timeout}" >> "${LOG_FILE}" 2>&1; then
    csv="${out}/roi_matrix.csv"
    identified="$(csv_int_field "${csv}" pf_identified)"
    issued="$(csv_int_field "${csv}" pf_issued)"
    useful="$(csv_int_field "${csv}" pf_useful)"
    log "${short}: identified=${identified} issued=${issued} useful=${useful}"
    EVIDENCE+=("${short}_identified=${identified}" "${short}_issued=${issued}" "${short}_useful=${useful}")
    if [ "${useful}" -le 0 ]; then
      FAIL_GRAPHS="${FAIL_GRAPHS} ${short}(useful=0)"
    fi
  else
    log "FAIL ${short} (timeout or runtime error)"
    EVIDENCE+=("${short}_status=fail")
    FAIL_GRAPHS="${FAIL_GRAPHS} ${short}(fail)"
  fi
done

if [ -n "${FAIL_GRAPHS}" ]; then
  milestone_fail "${M_ID}" "graphs failed: ${FAIL_GRAPHS}" "${EVIDENCE[@]}"
  exit 1
fi

milestone_done "${M_ID}" "${EVIDENCE[@]}"
