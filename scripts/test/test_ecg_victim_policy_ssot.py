"""ECG eviction-decision single-source-of-truth (SSOT) gate.

The ECG_GRASP_POPT victim-selection logic lives in one header,
``bench/include/ecg_victim_policy.h``, which cache_sim, gem5 and Sniper all call.
To keep "nothing is ported/mirrored" true, every simulator's co-located copy of
that header must be byte-identical to the canonical one. If they ever drift, the
decision logic could differ between backends — this test fails loudly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "bench" / "include" / "ecg_victim_policy.h"
# Tracked co-located copies (gem5 uses the .hh convention; content is identical).
COPIES = [
    ROOT / "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_victim_policy.hh",
    ROOT / "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/ecg_victim_policy.h",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_canonical_exists():
    assert CANONICAL.is_file(), f"canonical ECG policy header missing: {CANONICAL}"


def test_all_copies_byte_identical():
    want = _sha(CANONICAL)
    for c in COPIES:
        assert c.is_file(), f"overlay ECG policy copy missing: {c}"
        assert _sha(c) == want, (
            f"ECG policy header drift: {c} differs from canonical {CANONICAL}.\n"
            f"All simulators must share the identical eviction decision; re-copy "
            f"bench/include/ecg_victim_policy.h into the overlay trees."
        )


def test_calls_present_in_each_simulator():
    """Each simulator's policy source must actually call the shared function."""
    callers = {
        "bench/include/cache_sim/cache_sim.h": 'ecg_policy::selectVictim',
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_rp.cc": 'ecg_policy::selectVictim',
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_ecg.cc": 'ecg_policy::selectVictim',
    }
    for rel, token in callers.items():
        text = (ROOT / rel).read_text(errors="ignore")
        assert token in text, f"{rel} does not call the shared {token}"


def test_prefetch_target_is_shared():
    """ECG prefetch-target selection is a single shared header
    (bench/include/ecg_mode6_builder.h, compiled into every kernel). The cache_sim
    mask builder must call it rather than duplicate the lookahead logic, so the
    one prefetch-target unit test covers all three simulators."""
    builder = ROOT / "bench/include/ecg_mode6_builder.h"
    assert builder.is_file(), f"shared mask builder missing: {builder}"
    assert "selectPrefetchTarget" in builder.read_text(errors="ignore")
    cc_ctx = ROOT / "bench/include/cache_sim/graph_cache_context.h"
    assert "ecg_mode6::selectPrefetchTarget" in cc_ctx.read_text(errors="ignore"), (
        "cache_sim graph_cache_context.h must call the shared "
        "ecg_mode6::selectPrefetchTarget, not duplicate the lookahead logic"
    )
