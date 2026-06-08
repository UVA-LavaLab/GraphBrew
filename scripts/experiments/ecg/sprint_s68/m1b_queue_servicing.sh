#!/usr/bin/env bash
# M1b: Queue-servicing scheduling fix in gem5 Queued::nextPrefetchReadyTime
# + X86 gem5 rebuild.
#
# Root cause discovered by M2 evidence + rubber-duck:
#   With the M1 MMU patch, ECG_PFX prefetches now pass the page-cross
#   filter and enter pfqMissingTranslation. But Queued's
#   nextPrefetchReadyTime() returns MaxTick when pfq is empty even if
#   pfqMissingTranslation is non-empty. The cache uses that return
#   value to schedule the next prefetch event; if MaxTick, the cache
#   never wakes up to call getPacket(), which is the ONLY caller of
#   processMissingTranslations(). Net effect: prefetchers that produce
#   only cross-page candidates (ECG_PFX) are silently never serviced.
#
#   DROPLET escapes because it pushes many same-page edge-stream
#   prefetches that go straight to pfq, keeping the cache busy enough
#   to drain pfqMissingTranslation as a side effect.
#
# Fix: patch nextPrefetchReadyTime() to return curTick() when
# pfqMissingTranslation has entries even if pfq is empty. Patch script:
# scripts/experiments/ecg/sprint_s68/_patch_queued_hh.py (idempotent).
#
# Rebuild: X86 gem5 incremental scons build. Header-only change → many
# .o files recompile. Estimated 5-15 min wall.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m1b-queue-servicing"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M1b: queue-servicing scheduling fix + X86 gem5 rebuild"

# Apply the patch (idempotent)
python3 "${HERE}/_patch_queued_hh.py" "${REPO}" 2>&1 | tee -a "${LOG_FILE}"
patch_rc="${PIPESTATUS[0]}"
if [ "${patch_rc}" -ne 0 ]; then
  milestone_fail "${M_ID}" "queued.hh patcher exited with code ${patch_rc}"
  exit 1
fi

TARGET="${REPO}/bench/include/gem5_sim/gem5/src/mem/cache/prefetch/queued.hh"
marker_count="$(grep -c "S68-QUEUE-SERVICING-PATCH" "${TARGET}" || true)"
log "S68-QUEUE-SERVICING-PATCH marker count in queued.hh: ${marker_count}"
if [ "${marker_count}" -lt 1 ]; then
  milestone_fail "${M_ID}" "patch marker missing from queued.hh after patching"
  exit 1
fi

# Rebuild X86 gem5 (incremental, scons direct)
GEM5_BIN="${REPO}/bench/include/gem5_sim/gem5/build/X86/gem5.opt"
BIN_TS_BEFORE="$(stat -c %Y "${GEM5_BIN}" 2>/dev/null || echo 0)"
SRC_TS="$(stat -c %Y "${TARGET}")"

if [ "${BIN_TS_BEFORE}" -gt "${SRC_TS}" ]; then
  log "X86 binary is already newer than patched queued.hh — rebuild already done"
  milestone_done "${M_ID}" \
    "rebuild_skipped=true" \
    "bin_mtime=${BIN_TS_BEFORE}" \
    "src_mtime=${SRC_TS}" \
    "marker_count=${marker_count}"
  exit 0
fi

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
[ "${JOBS}" -gt 2 ] && JOBS=$((JOBS - 2))
log "rebuilding X86 gem5 (incremental scons) with --jobs ${JOBS}"
log "  this will take 5-15 minutes; output streams to ${LOG_FILE}"

cd "${REPO}/bench/include/gem5_sim/gem5"
if ! scons build/X86/gem5.opt -j "${JOBS}" >> "${LOG_FILE}" 2>&1; then
  milestone_fail "${M_ID}" "X86 scons rebuild failed; inspect ${LOG_FILE}"
  exit 1
fi

NEW_TS="$(stat -c %Y "${GEM5_BIN}" 2>/dev/null || echo 0)"
if [ "${NEW_TS}" -le "${SRC_TS}" ]; then
  milestone_fail "${M_ID}" "rebuild claimed success but X86 binary mtime not updated" \
    "bin_mtime=${NEW_TS}" "src_mtime=${SRC_TS}"
  exit 1
fi

bin_size="$(stat -c %s "${GEM5_BIN}")"
log "X86 rebuild OK — bin_size=${bin_size} bytes mtime=${NEW_TS}"
milestone_done "${M_ID}" \
  "rebuild_skipped=false" \
  "bin_mtime=${NEW_TS}" \
  "bin_size_bytes=${bin_size}" \
  "src_mtime=${SRC_TS}" \
  "marker_count=${marker_count}"
