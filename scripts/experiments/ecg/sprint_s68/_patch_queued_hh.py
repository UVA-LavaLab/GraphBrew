#!/usr/bin/env python3
"""
M1b helper: patch gem5 queued.hh to fix the queue-servicing scheduling
gap discovered in M2.

Root cause: `Queued::nextPrefetchReadyTime()` returns MaxTick when pfq
is empty, even if pfqMissingTranslation has pending entries. The cache
uses that return value to schedule the next prefetch event. If MaxTick,
the cache never wakes up to call getPacket(), which is the only caller
of processMissingTranslations(). Result: prefetchers that produce
only cross-page candidates (ECG_PFX) are never serviced.

Fix: nextPrefetchReadyTime() returns curTick() when only
pfqMissingTranslation has entries.

This script is idempotent — re-runs detect the marker and skip.
"""
import sys, os, re

REPO = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

TARGET = os.path.join(
    REPO,
    "bench/include/gem5_sim/gem5/src/mem/cache/prefetch/queued.hh",
)

MARKER = "// S68-QUEUE-SERVICING-PATCH"

with open(TARGET, "r") as f:
    src = f.read()

if MARKER in src:
    print(f"[m1b] {TARGET}: already patched (marker present) — skip")
    print(f"[m1b] register_marker_count={src.count(MARKER)}")
    sys.exit(0)

OLD = (
    "    Tick nextPrefetchReadyTime() const override\n"
    "    {\n"
    "        return pfq.empty() ? MaxTick : pfq.front().tick;\n"
    "    }"
)

NEW = (
    "    Tick nextPrefetchReadyTime() const override\n"
    "    {\n"
    "        " + MARKER + ":\n"
    "        // If pfq is empty but pfqMissingTranslation has pending\n"
    "        // entries, wake the cache up immediately so getPacket()\n"
    "        // runs and processMissingTranslations() drains the queue.\n"
    "        // Without this, prefetchers that produce ONLY cross-page\n"
    "        // candidates (e.g., ECG_PFX targeting property[random_v])\n"
    "        // never get serviced and pf_issued stays 0. See\n"
    "        // docs/findings/gem5_implementation_audit_v1.md and the\n"
    "        // M2 evidence in sprint S68.\n"
    "        if (!pfq.empty()) return pfq.front().tick;\n"
    "        if (!pfqMissingTranslation.empty()) return curTick();\n"
    "        return MaxTick;\n"
    "    }"
)

if OLD not in src:
    print(f"[m1b] ERROR: original nextPrefetchReadyTime body not found verbatim",
          file=sys.stderr)
    sys.exit(2)

src = src.replace(OLD, NEW, 1)

# Ensure curTick() is visible. base/types.hh is already in the includes
# block; curTick lives in sim/cur_tick.hh (gem5 v22+) or sim/eventq.hh
# (older). Check + insert if needed.
need_include = '#include "sim/cur_tick.hh"'
if need_include not in src and 'sim/eventq.hh' not in src:
    src = src.replace(
        '#include "base/types.hh"\n',
        '#include "base/types.hh"\n' + need_include + "\n",
        1,
    )
    print(f"[m1b] added {need_include}")

with open(TARGET, "w") as f:
    f.write(src)

print(f"[m1b] patched {TARGET}")
print(f"[m1b] marker_count={src.count(MARKER)}")
