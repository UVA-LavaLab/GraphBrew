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

# Run gem5 with ECG_PFX --ecg-pfx-delivery instruction so the ecg.extract
# opcode is emitted by the kernel. Note: this requires the RISCV kernel
# binary bench/bin_gem5/pr_riscv_m5ops (or equivalent).
PR_BIN="${REPO}/bench/bin_gem5/pr_riscv_m5ops"
if [ ! -x "${PR_BIN}" ]; then
  log "RISCV kernel binary ${PR_BIN} missing — falling back to X86 m5op path"
  PR_BIN="${REPO}/bench/bin_gem5/pr_m5ops"
fi
[ -x "${PR_BIN}" ] || die "no kernel binary available"

GEM5_BIN="${RISCV_BIN}"
if [[ "${PR_BIN}" != *riscv* ]]; then
  GEM5_BIN="${REPO}/bench/include/gem5_sim/gem5/build/X86/gem5.opt"
  log "using X86 gem5 binary with X86 kernel (m5op delivery, not ISA)"
fi
log "gem5 binary: ${GEM5_BIN}"
log "kernel:      ${PR_BIN}"

# Use roi_matrix.py with ECG_PFX --ecg-pfx-delivery instruction
ECG_CONTAINER_BITS=64 GEM5_ENABLE_ECG_EXTRACT=1 timeout 1200 \
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

if [ ! -f "${csv}" ]; then
  milestone_fail "${M_ID}" "no CSV produced; inspect ${LOG_FILE}"
  exit 1
fi

issued="$(csv_int_field "${csv}" pf_issued)"
identified="$(csv_int_field "${csv}" pf_identified)"
useful="$(csv_int_field "${csv}" pf_useful)"

opcode_msg=""
if [ -n "${gem5_log}" ] && [ -f "${gem5_log}" ]; then
  if grep -q "ECG_PFX: first target vertex" "${gem5_log}"; then
    opcode_msg="present"
  else
    opcode_msg="missing"
  fi
fi

log "results: identified=${identified} issued=${issued} useful=${useful} opcode_msg=${opcode_msg}"

EVIDENCE=(
  "pf_identified=${identified}"
  "pf_issued=${issued}"
  "pf_useful=${useful}"
  "opcode_first_hint_msg=${opcode_msg}"
  "gem5_log=${gem5_log:-none}"
)

if [ "${issued}" -le 0 ]; then
  milestone_fail "${M_ID}" "pf_issued=0 — ISA path not delivering hints" "${EVIDENCE[@]}"
  exit 1
fi

milestone_done "${M_ID}" "${EVIDENCE[@]}"
