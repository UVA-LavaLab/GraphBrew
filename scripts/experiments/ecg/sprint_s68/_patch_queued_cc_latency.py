#!/usr/bin/env python3
"""
M5b helper: add a latency-readiness guard to gem5 Queued::getPacket().

Recommended by M1b rubber-duck before M6 cycle-accurate parity comparison.

Without this guard, the M1b queue-servicing patch allows getPacket()
to issue a prefetch on the SAME tick its translation completes,
bypassing the prefetcher's `latency` cycles. This skews cycle-accurate
measurements: prefetches appear to arrive faster than they really
would on hardware.

Patch: in Queued::getPacket() (queued.cc), after the
`processMissingTranslations(queueSize)` call but BEFORE popping pfq,
add:

```cpp
if (pfq.front().tick > curTick()) {
    return nullptr;
}
```

This preserves the latency contract: a prefetch ready at tick T won't
be issued before T.

Idempotent: marker-guarded.
"""
import sys, os

REPO = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

TARGET = os.path.join(
    REPO,
    "bench/include/gem5_sim/gem5/src/mem/cache/prefetch/queued.cc",
)

MARKER = "// S68-LATENCY-GUARD-PATCH"

with open(TARGET, "r") as f:
    src = f.read()

if MARKER in src:
    print(f"[m5b] {TARGET}: already patched (marker present) — skip")
    sys.exit(0)

# Target the getPacket() function — locate the second `if (pfq.empty())`
# check which fires AFTER processMissingTranslations.
OLD = (
    "    if (pfq.empty()) {\n"
    "        DPRINTF(HWPrefetch, \"No hardware prefetches available.\\n\");\n"
    "        return nullptr;\n"
    "    }\n"
    "\n"
    "    PacketPtr pkt = pfq.front().pkt;\n"
)

NEW = (
    "    if (pfq.empty()) {\n"
    "        DPRINTF(HWPrefetch, \"No hardware prefetches available.\\n\");\n"
    "        return nullptr;\n"
    "    }\n"
    "\n"
    "    " + MARKER + ": preserve prefetcher latency contract.\n"
    "    // After processMissingTranslations() above, a freshly translated\n"
    "    // packet may have pf_time == curTick() + latency, i.e. NOT ready\n"
    "    // yet. Without this guard, getPacket() would issue it on the same\n"
    "    // tick the translation completed, skipping the latency cycle.\n"
    "    // This is critical for M6 cycle-accurate parity comparisons.\n"
    "    if (pfq.front().tick > curTick()) {\n"
    "        DPRINTF(HWPrefetch, \"Prefetch front not yet ready "
    "(tick=%llu > curTick=%llu).\\n\",\n"
    "                pfq.front().tick, curTick());\n"
    "        return nullptr;\n"
    "    }\n"
    "\n"
    "    PacketPtr pkt = pfq.front().pkt;\n"
)

if OLD not in src:
    print(f"[m5b] ERROR: getPacket() body did not match expected pattern",
          file=sys.stderr)
    print(f"[m5b] check {TARGET} around line 249-254", file=sys.stderr)
    sys.exit(2)

# Ensure curTick is available; queued.cc already includes queued.hh which
# now has cur_tick.hh per M1b.
src = src.replace(OLD, NEW, 1)

with open(TARGET, "w") as f:
    f.write(src)

print(f"[m5b] patched {TARGET}")
print(f"[m5b] marker_count={src.count(MARKER)}")
