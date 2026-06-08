#!/usr/bin/env bash
# M1: Apply MMU plumbing patch to graph_se.py.
#
# Root cause (audit v1): gem5 Queued::notify drops every prefetch whose
# target line crosses a page boundary unless the prefetcher has mmu
# set via BasePrefetcher.registerMMU(simObj). ECG_PFX prefetches are
# essentially always cross-page (random property[v] targets), so the
# filter kills them all (pf_identified=0).
#
# This script idempotently inserts the registerMMU calls after each
# prefetcher attachment block in graph_se.py.

set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${HERE}/common.sh"

M_ID="s68-m1-mmu-patch"
LOG_FILE="${RESULTS_DIR}/${M_ID}/log.txt"
mkdir -p "${RESULTS_DIR}/${M_ID}"
: > "${LOG_FILE}"

step "M1: MMU plumbing patch"

TARGET="${REPO}/bench/include/gem5_sim/configs/graphbrew/graph_se.py"
MARKER="# S68-MMU-PATCH"

if grep -q "${MARKER}" "${TARGET}"; then
  log "patch already present (marker '${MARKER}' found) — idempotent skip"
  matches="$(grep -c "registerMMU" "${TARGET}" || true)"
  milestone_done "${M_ID}" \
    "patch_already_present=true" \
    "register_mmu_call_count=${matches}" \
    "target_file=${TARGET}"
  exit 0
fi

# Apply with a python rewriter (safer than sed for indent matters).
python3 - "${TARGET}" "${MARKER}" <<'PY'
import sys, re, io

path, marker = sys.argv[1], sys.argv[2]
with open(path, "r") as f:
    src = f.read()

# After "system.l2cache.prefetcher = make_*_prefetcher(...)" or
# "system.cpu.dcache.prefetcher = make_*_prefetcher(...)" we want to
# emit:
#     <indent>cache_obj.prefetcher.registerMMU(system.cpu.mmu)   # S68-MMU-PATCH
#
# We use a regex pass that captures the indentation, target attribute,
# and inserts the registerMMU call directly below.

pattern = re.compile(
    r"^(?P<indent>[ \t]+)(?P<target>system(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\.prefetcher = make_(?:droplet|ecg_pfx)_prefetcher\([^)]*\)[ \t]*$",
    re.MULTILINE,
)

def repl(m):
    indent = m.group("indent")
    target = m.group("target")
    original = m.group(0)
    addition = (
        f"\n{indent}# S68-MMU-PATCH: gem5 Queued::notify drops cross-page\n"
        f"{indent}# prefetches unless prefetcher.mmu is set. See\n"
        f"{indent}# docs/findings/gem5_implementation_audit_v1.md.\n"
        f"{indent}_pf = {target}.prefetcher\n"
        f"{indent}if hasattr(system.cpu, 'mmu'):\n"
        f"{indent}    _pf.registerMMU(system.cpu.mmu)\n"
        f"{indent}elif hasattr(system.cpu, 'dtb'):\n"
        f"{indent}    _pf.registerMMU(system.cpu.dtb)"
    )
    return original + addition

new_src, n = pattern.subn(repl, src)
if n == 0:
    print("ERROR: patch pattern did not match any prefetcher attachment site",
          file=sys.stderr)
    sys.exit(2)

with open(path, "w") as f:
    f.write(new_src)
print(f"patched {n} prefetcher attachment site(s)")
PY

matches="$(grep -c "registerMMU" "${TARGET}" || true)"
log "registerMMU now appears ${matches} times in graph_se.py"

# Python syntax sanity check
if ! python3 -c "import ast; ast.parse(open('${TARGET}').read())"; then
  milestone_fail "${M_ID}" "graph_se.py syntax check failed after patch" \
    "target_file=${TARGET}"
  exit 1
fi
log "syntax check OK"

milestone_done "${M_ID}" \
  "register_mmu_call_count=${matches}" \
  "target_file=${TARGET}" \
  "marker=${MARKER}"
