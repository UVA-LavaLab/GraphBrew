#!/usr/bin/env bash
# M5b: Add Queued::getPacket() latency-readiness guard.
#
# Flagged by M1b rubber-duck. Without this guard, the M1b queue-servicing
# patch lets getPacket() issue a prefetch on the SAME tick its translation
# completed — bypassing the prefetcher's `latency` cycles. This skews
# cycle-accurate measurements before M6.
#
# Patch script: scripts/experiments/ecg/sprint_s68/_patch_queued_cc_latency.py
# (idempotent, marker-guarded).
#
# After patching, do an X86 incremental rebuild. M4 (RISCV rebuild)
# will pick up this fix automatically when it rebuilds the RISCV binary.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m5b-latency-guard"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M5b: queued.cc getPacket() latency guard + X86 incremental rebuild"

python3 "${HERE}/_patch_queued_cc_latency.py" "${REPO}" 2>&1 | tee -a "${LOG_FILE}"
patch_rc="${PIPESTATUS[0]}"
if [ "${patch_rc}" -ne 0 ]; then
  milestone_fail "${M_ID}" "queued.cc patcher exited with code ${patch_rc}"
  exit 1
fi

TARGET="${REPO}/bench/include/gem5_sim/gem5/src/mem/cache/prefetch/queued.cc"
marker_count="$(grep -c "S68-LATENCY-GUARD-PATCH" "${TARGET}" || true)"
log "S68-LATENCY-GUARD-PATCH marker count: ${marker_count}"
if [ "${marker_count}" -lt 1 ]; then
  milestone_fail "${M_ID}" "patch marker missing from queued.cc"
  exit 1
fi

# Save the patch to the overlay tree for durability
PATCH_DEST="${REPO}/bench/include/gem5_sim/overlays/mem/cache/prefetch/queued_cc_latency.patch"
if [ ! -f "${PATCH_DEST}" ]; then
  log "exporting patch to overlay tree (durable across setup_gem5.py re-runs)"
  cd "${REPO}/bench/include/gem5_sim/gem5"
  git diff src/mem/cache/prefetch/queued.cc > "${PATCH_DEST}"
  log "patch saved: ${PATCH_DEST}"
fi

# Rebuild X86 (incremental — only queued.cc + dependents)
GEM5_BIN="${REPO}/bench/include/gem5_sim/gem5/build/X86/gem5.opt"
BIN_TS_BEFORE="$(stat -c %Y "${GEM5_BIN}" 2>/dev/null || echo 0)"
SRC_TS="$(stat -c %Y "${TARGET}")"

if [ "${BIN_TS_BEFORE}" -gt "${SRC_TS}" ]; then
  log "X86 binary already newer — idempotent skip"
  milestone_done "${M_ID}" \
    "rebuild_skipped=true" \
    "bin_mtime=${BIN_TS_BEFORE}" \
    "src_mtime=${SRC_TS}" \
    "marker_count=${marker_count}"
  exit 0
fi

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
[ "${JOBS}" -gt 2 ] && JOBS=$((JOBS - 2))
log "rebuilding X86 gem5 incremental scons -j ${JOBS}"

cd "${REPO}/bench/include/gem5_sim/gem5"
if ! scons build/X86/gem5.opt -j "${JOBS}" >> "${LOG_FILE}" 2>&1; then
  milestone_fail "${M_ID}" "X86 scons rebuild failed; inspect ${LOG_FILE}"
  exit 1
fi

NEW_TS="$(stat -c %Y "${GEM5_BIN}" 2>/dev/null || echo 0)"
if [ "${NEW_TS}" -le "${SRC_TS}" ]; then
  milestone_fail "${M_ID}" "rebuild claimed success but binary mtime not updated" \
    "bin_mtime=${NEW_TS}" "src_mtime=${SRC_TS}"
  exit 1
fi

milestone_done "${M_ID}" \
  "rebuild_skipped=false" \
  "bin_mtime=${NEW_TS}" \
  "src_mtime=${SRC_TS}" \
  "marker_count=${marker_count}" \
  "overlay_patch=${PATCH_DEST}"
