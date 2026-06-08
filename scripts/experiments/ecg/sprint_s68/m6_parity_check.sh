#!/usr/bin/env bash
# M6: gem5 vs cache_sim DRAM-traffic parity on mode-6.
#
# Pick up the same email-Eu-core/pr cell from cache_sim's HPCA
# popt_off__isa__k2 result (buildup_v1) and compare DRAM traffic.
#
# Exit criteria: relative delta within ±25%.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m6-parity-check"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M6: gem5 vs cache_sim DRAM-traffic parity"

# gem5 number from M5 (preferred) or M2 (fallback)
GEM5_CSV="${RESULTS_DIR}/s68-m5-isa-smoke/ECG_PFX_isa/roi_matrix.csv"
if [ ! -f "${GEM5_CSV}" ]; then
  GEM5_CSV="${RESULTS_DIR}/s68-m2-smoke-diff/ECG_PFX/roi_matrix.csv"
  log "M5 not present, falling back to M2 gem5 CSV"
fi
[ -f "${GEM5_CSV}" ] || { milestone_fail "${M_ID}" "no gem5 CSV available (M2/M5 missing)"; exit 1; }

# cache_sim number from buildup_v1
CACHE_SIM_DIR="${REPO}/results/ecg_experiments/hpca_mode6/buildup_v1"
CACHE_SIM_CSV="$(find "${CACHE_SIM_DIR}" -path '*email-Eu-core*popt_off__isa__k2*' -name '*.csv' -print -quit 2>/dev/null)"
if [ -z "${CACHE_SIM_CSV}" ]; then
  CACHE_SIM_CSV="$(find "${CACHE_SIM_DIR}" -path '*email-Eu-core*' -name '*.csv' -print -quit 2>/dev/null)"
  log "headline arm CSV not found, using any email-Eu-core CSV: ${CACHE_SIM_CSV:-NONE}"
fi
[ -n "${CACHE_SIM_CSV}" ] && [ -f "${CACHE_SIM_CSV}" ] || \
  { milestone_fail "${M_ID}" "no cache_sim CSV available for parity comparison"; exit 1; }

log "gem5     CSV: ${GEM5_CSV}"
log "cache_sim CSV: ${CACHE_SIM_CSV}"

# Extract DRAM traffic proxies
gem5_l3miss="$(csv_int_field "${GEM5_CSV}" l3_misses)"
gem5_l3acc="$(csv_int_field "${GEM5_CSV}" l3_accesses)"
gem5_pfissued="$(csv_int_field "${GEM5_CSV}" pf_issued)"
gem5_total="$(( gem5_l3miss + gem5_pfissued ))"

# cache_sim has different column names; try common variants
cs_mem="$(csv_int_field "${CACHE_SIM_CSV}" mem_acc)"
cs_pf="$(csv_int_field "${CACHE_SIM_CSV}" pf_fills)"
cs_total="$(( cs_mem + cs_pf ))"

log "gem5      l3_misses=${gem5_l3miss} pf_issued=${gem5_pfissued} → total_proxy=${gem5_total}"
log "cache_sim mem_acc=${cs_mem} pf_fills=${cs_pf} → total=${cs_total}"

if [ "${gem5_total}" -le 0 ] || [ "${cs_total}" -le 0 ]; then
  milestone_fail "${M_ID}" "one of gem5/cache_sim totals is zero — cannot compute parity" \
    "gem5_total=${gem5_total}" "cache_sim_total=${cs_total}"
  exit 1
fi

# Compute relative delta = |gem5 - cs| / cs
delta_pct="$(python3 -c "print(round(abs(${gem5_total} - ${cs_total}) * 100.0 / ${cs_total}, 2))")"
log "relative delta = ${delta_pct}% (target: ≤25%)"

EVIDENCE=(
  "gem5_l3_misses=${gem5_l3miss}"
  "gem5_pf_issued=${gem5_pfissued}"
  "gem5_total_proxy=${gem5_total}"
  "cache_sim_mem_acc=${cs_mem}"
  "cache_sim_pf_fills=${cs_pf}"
  "cache_sim_total=${cs_total}"
  "relative_delta_pct=${delta_pct}"
)

# Compare to 25.0 using awk for float
within="$(awk -v d="${delta_pct}" 'BEGIN{print (d<=25.0)?"yes":"no"}')"

if [ "${within}" = "no" ]; then
  milestone_fail "${M_ID}" "parity delta ${delta_pct}% > 25% threshold; rubber-duck required" \
    "${EVIDENCE[@]}"
  exit 1
fi

milestone_done "${M_ID}" "${EVIDENCE[@]}"
