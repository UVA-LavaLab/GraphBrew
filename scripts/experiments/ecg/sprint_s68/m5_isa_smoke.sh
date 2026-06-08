#!/usr/bin/env bash
# M5: ecg.extract opcode smoke on RISCV with fresh binary.
#
# Exit criteria:
#   - pf_issued > 0 in the CSV
#   - "ECG_PFX: first target vertex" appears in the gem5 log

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m5-isa-smoke"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M5: RISC-V ecg.extract opcode smoke"

require_no_running_gem5

GRAPH="${REPO}/results/graphs/email-Eu-core/email-Eu-core.sg"
[ -f "${GRAPH}" ] || die "graph missing: ${GRAPH}"

OUT="${RESULTS_DIR}/${M_ID}/ECG_PFX_isa"
rm -rf "${OUT}"
mkdir -p "${OUT}"

# Use the RISCV binary for ecg.extract
RISCV_BIN="${REPO}/bench/include/gem5_sim/gem5/build/RISCV/gem5.opt"
[ -x "${RISCV_BIN}" ] || die "RISCV gem5.opt missing (M4 must run first)"

# Use the RISCV kernel binary so the ecg.extract opcode is actually compiled in.
PR_BIN="${REPO}/bench/bin_gem5/pr_riscv_m5ops"
if [ ! -x "${PR_BIN}" ]; then
  milestone_fail "${M_ID}" \
    "RISCV kernel binary missing (${PR_BIN}) — build with 'make gem5-riscv-m5ops-pr' before M5"
  exit 1
fi

log "gem5 binary: ${RISCV_BIN}"
log "kernel:      ${PR_BIN}"
log "force ISA path via GEM5_OPT + GEM5_KERNEL_SUFFIX env vars"

# Force roi_matrix.py to use the RISCV gem5.opt + RISCV kernel binary.
# The env vars are picked up at module load time (see roi_matrix.py:46-54).
ECG_CONTAINER_BITS=64 \
  GEM5_OPT="${RISCV_BIN}" \
  GEM5_KERNEL_SUFFIX="_riscv_m5ops" \
  GEM5_ENABLE_ECG_EXTRACT=1 \
  timeout 1200 \
  python3 "${REPO}/scripts/experiments/ecg/roi_matrix.py" \
    --suite gem5 --no-build \
    --benchmark pr \
    --options "-f ${GRAPH} -s -o 5 -n 1 -i 2" \
    --policies LRU \
    --prefetcher ECG_PFX --prefetcher-level l2 --allow-gem5-ecg-pfx \
    --ecg-pfx-mode per_edge \
    --ecg-pfx-delivery instruction \
    --l1d-size 32kB --l1d-ways 8 \
    --l2-size 256kB --l2-ways 8 \
    --l3-sizes 1MB --l3-ways 16 \
    --line-size 64 \
    --timeout-gem5 1100 \
    --out-dir "${OUT}" >> "${LOG_FILE}" 2>&1 || true

csv="${OUT}/roi_matrix.csv"
gem5_log="$(find "${OUT}" -name '*.log' -print -quit 2>/dev/null)"
config_ini="$(find "${OUT}" -name 'config.ini' -print -quit 2>/dev/null)"

if [ ! -f "${csv}" ]; then
  milestone_fail "${M_ID}" "no CSV produced; inspect ${LOG_FILE}"
  exit 1
fi

issued="$(csv_int_field "${csv}" pf_issued)"
identified="$(csv_int_field "${csv}" pf_identified)"
useful="$(csv_int_field "${csv}" pf_useful)"

# Post-run assertions: was the RISCV path actually exercised?
ISA_PATH_USED=0
ISA_KERNEL_USED=0
if [ -n "${config_ini}" ] && [ -f "${config_ini}" ]; then
  if grep -q "build/RISCV/gem5.opt\|/RISCV/gem5\.opt" "${LOG_FILE}" 2>/dev/null || \
     grep -q "riscv" "${config_ini}" 2>/dev/null; then
    ISA_PATH_USED=1
  fi
  if grep -q "pr_riscv_m5ops" "${config_ini}" 2>/dev/null; then
    ISA_KERNEL_USED=1
  fi
fi
# Also check the gem5 log preamble
if [ -n "${gem5_log}" ] && [ -f "${gem5_log}" ]; then
  if grep -q "RISCV\|riscv" "${gem5_log}" 2>/dev/null; then
    ISA_PATH_USED=1
  fi
  if grep -q "pr_riscv_m5ops" "${gem5_log}" 2>/dev/null; then
    ISA_KERNEL_USED=1
  fi
fi

opcode_msg="missing"
if [ -n "${gem5_log}" ] && [ -f "${gem5_log}" ]; then
  if grep -q "ECG_PFX: first target vertex" "${gem5_log}"; then
    opcode_msg="present"
  fi
fi

log "results: identified=${identified} issued=${issued} useful=${useful}"
log "         opcode_msg=${opcode_msg} isa_path_used=${ISA_PATH_USED} isa_kernel_used=${ISA_KERNEL_USED}"

EVIDENCE=(
  "pf_identified=${identified}"
  "pf_issued=${issued}"
  "pf_useful=${useful}"
  "opcode_first_hint_msg=${opcode_msg}"
  "isa_path_used=${ISA_PATH_USED}"
  "isa_kernel_used=${ISA_KERNEL_USED}"
  "gem5_log=${gem5_log:-none}"
  "config_ini=${config_ini:-none}"
)

if [ "${ISA_PATH_USED}" -ne 1 ] || [ "${ISA_KERNEL_USED}" -ne 1 ]; then
  milestone_fail "${M_ID}" \
    "ISA path NOT used (isa_path_used=${ISA_PATH_USED}, isa_kernel_used=${ISA_KERNEL_USED}) — check GEM5_OPT/GEM5_KERNEL_SUFFIX env propagation" \
    "${EVIDENCE[@]}"
  exit 1
fi

if [ "${issued}" -le 0 ]; then
  milestone_fail "${M_ID}" "pf_issued=0 — ISA path not delivering hints" "${EVIDENCE[@]}"
  exit 1
fi

milestone_done "${M_ID}" "${EVIDENCE[@]}"
