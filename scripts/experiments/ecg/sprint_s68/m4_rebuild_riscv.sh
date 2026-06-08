#!/usr/bin/env bash
# M4: Rebuild RISCV gem5.opt with all sprint 6f-5/6f-6 fixes.
#
# Exit criteria:
#   bench/include/gem5_sim/gem5/build/RISCV/gem5.opt exists AND
#   mtime newer than the latest commit touching the overlay
#   directory, AND symbol GEM5_ECG_PFX_RECENT_FILTER_SIZE present
#   in strings.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m4-rebuild-riscv"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M4: rebuild RISCV gem5.opt"

RISCV_BIN="${REPO}/bench/include/gem5_sim/gem5/build/RISCV/gem5.opt"

OVERLAY_LATEST_TS="$(cd "${REPO}" && git log -1 --format=%ct -- bench/include/gem5_sim/overlays/)"
CURRENT_BIN_TS="$(stat -c %Y "${RISCV_BIN}" 2>/dev/null || echo 0)"

log "overlay latest commit ts=${OVERLAY_LATEST_TS}"
log "current RISCV binary mtime=${CURRENT_BIN_TS}"

if [ "${CURRENT_BIN_TS}" -gt "${OVERLAY_LATEST_TS}" ]; then
  if strings "${RISCV_BIN}" 2>/dev/null | grep -q "GEM5_ECG_PFX_RECENT_FILTER_SIZE"; then
    log "RISCV binary already fresh AND has post-6f-6 symbols — idempotent skip"
    milestone_done "${M_ID}" \
      "rebuild_skipped=true" \
      "bin_mtime=${CURRENT_BIN_TS}" \
      "overlay_latest_ts=${OVERLAY_LATEST_TS}"
    exit 0
  fi
  log "binary mtime newer but symbols missing — rebuild needed"
fi

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
[ "${JOBS}" -gt 2 ] && JOBS=$((JOBS - 2))
log "starting RISCV build with --jobs ${JOBS} (estimated 30-60 min)"

cd "${REPO}"
if ! python3 scripts/setup_gem5.py --isa RISCV --rebuild --jobs "${JOBS}" >> "${LOG_FILE}" 2>&1; then
  milestone_fail "${M_ID}" "setup_gem5.py --isa RISCV failed; inspect ${LOG_FILE}"
  exit 1
fi

NEW_TS="$(stat -c %Y "${RISCV_BIN}" 2>/dev/null || echo 0)"
if [ "${NEW_TS}" -le "${OVERLAY_LATEST_TS}" ]; then
  milestone_fail "${M_ID}" "build claimed success but RISCV binary mtime not updated" \
    "bin_mtime=${NEW_TS}" "overlay_latest_ts=${OVERLAY_LATEST_TS}"
  exit 1
fi

if ! strings "${RISCV_BIN}" 2>/dev/null | grep -q "GEM5_ECG_PFX_RECENT_FILTER_SIZE"; then
  milestone_fail "${M_ID}" "post-6f-6 symbol GEM5_ECG_PFX_RECENT_FILTER_SIZE missing from rebuilt binary" \
    "bin_mtime=${NEW_TS}"
  exit 1
fi

bin_size="$(stat -c %s "${RISCV_BIN}")"
log "rebuild OK — bin_size=${bin_size} bytes mtime=${NEW_TS}"
milestone_done "${M_ID}" \
  "rebuild_skipped=false" \
  "bin_mtime=${NEW_TS}" \
  "bin_size_bytes=${bin_size}" \
  "overlay_latest_ts=${OVERLAY_LATEST_TS}" \
  "post6f6_symbol_present=true"
