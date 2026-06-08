#!/usr/bin/env bash
# M2: Pre/post-fix smoke differential on email-Eu-core/pr.
#
# Exit criteria (machine-checkable):
#   pf_identified > 0  AND  pf_issued > 0
#
# Compare against the pre-fix audit baseline:
#   pf_identified = 0, pf_issued = 0, pf_span_page = 63
#
# If pf_issued is still 0 after the M1 patch, mark BLOCKED. The
# diagnosis was either wrong or incomplete and the user must be
# surfaced.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m2-smoke-diff"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M2: post-fix ECG_PFX smoke on email-Eu-core/pr"

require_no_running_gem5

# Require M1 done
M1_STATUS="${RESULTS_DIR}/s68-m1-mmu-patch/status.json"
if [ ! -f "${M1_STATUS}" ]; then
  milestone_fail "${M_ID}" "M1 prerequisite not done — run m1_mmu_patch.sh first"
  exit 1
fi
m1_ok="$(python3 -c "import json; print(json.load(open('${M1_STATUS}'))['status'])")"
if [ "${m1_ok}" != "ok" ]; then
  milestone_fail "${M_ID}" "M1 prerequisite did not finish OK (status=${m1_ok})"
  exit 1
fi

GRAPH="${REPO}/results/graphs/email-Eu-core/email-Eu-core.sg"
[ -f "${GRAPH}" ] || die "graph missing: ${GRAPH}"

OUT_PFX="${RESULTS_DIR}/${M_ID}/ECG_PFX"
OUT_DROP="${RESULTS_DIR}/${M_ID}/DROPLET"
rm -rf "${OUT_PFX}" "${OUT_DROP}"

log "running ECG_PFX smoke (post-fix) — should now show pf_issued > 0"
run_gem5_smoke ECG_PFX "${GRAPH}" "${OUT_PFX}" 540 2>&1 | tee -a "${LOG_FILE}" | tail -20

log "running DROPLET control smoke (regression check)"
run_gem5_smoke DROPLET "${GRAPH}" "${OUT_DROP}" 540 2>&1 | tee -a "${LOG_FILE}" | tail -20

PFX_CSV="${OUT_PFX}/roi_matrix.csv"
DRP_CSV="${OUT_DROP}/roi_matrix.csv"
[ -f "${PFX_CSV}" ] || { milestone_fail "${M_ID}" "ECG_PFX CSV not produced"; exit 1; }
[ -f "${DRP_CSV}" ] || { milestone_fail "${M_ID}" "DROPLET CSV not produced"; exit 1; }

pfx_identified="$(csv_int_field "${PFX_CSV}" pf_identified)"
pfx_issued="$(csv_int_field "${PFX_CSV}" pf_issued)"
pfx_useful="$(csv_int_field "${PFX_CSV}" pf_useful)"
pfx_span="$(csv_int_field "${PFX_CSV}" pf_span_page)"
pfx_l3miss="$(csv_field "${PFX_CSV}" l3_misses)"

drp_identified="$(csv_int_field "${DRP_CSV}" pf_identified)"
drp_issued="$(csv_int_field "${DRP_CSV}" pf_issued)"
drp_useful="$(csv_int_field "${DRP_CSV}" pf_useful)"

log ""
log "Results: ECG_PFX  identified=${pfx_identified} issued=${pfx_issued} useful=${pfx_useful} span_page=${pfx_span} l3_misses=${pfx_l3miss}"
log "Results: DROPLET  identified=${drp_identified} issued=${drp_issued} useful=${drp_useful}  (control)"
log "Baseline (pre-fix): identified=0 issued=0 useful=0 span_page=63"
log ""

EVIDENCE=(
  "ecg_pfx_identified=${pfx_identified}"
  "ecg_pfx_issued=${pfx_issued}"
  "ecg_pfx_useful=${pfx_useful}"
  "ecg_pfx_span_page=${pfx_span}"
  "droplet_identified=${drp_identified}"
  "droplet_issued=${drp_issued}"
  "droplet_useful=${drp_useful}"
  "baseline_identified=0"
  "baseline_issued=0"
)

if [ "${pfx_identified}" -le 0 ] || [ "${pfx_issued}" -le 0 ]; then
  milestone_fail "${M_ID}" \
    "ECG_PFX pf_issued still 0 after M1 patch — page-cross diagnosis incomplete; rubber-duck required" \
    "${EVIDENCE[@]}"
  exit 1
fi

# Sanity: DROPLET should NOT have regressed
if [ "${drp_issued}" -le 0 ]; then
  milestone_fail "${M_ID}" \
    "DROPLET control regressed (pf_issued=0) — M1 patch may have broken DROPLET" \
    "${EVIDENCE[@]}"
  exit 1
fi

milestone_done "${M_ID}" "${EVIDENCE[@]}"
