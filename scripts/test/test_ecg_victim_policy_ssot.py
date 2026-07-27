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
        assert "ecg_policy::parseVariant" in text, (
            f"{rel} does not use the shared fail-closed variant parser")


def test_grasp_insertion_classifier_is_shared():
    """The GRASP insertion tier (the INSERTION half of the policy) is also SSOT:
    each simulator's graph context must call ecg_policy::classifyGraspTier rather
    than duplicate the per-region boundary math (which previously drifted — e.g.
    cache_sim classified [upper,upper+8) as MODERATE while gem5/Sniper did not)."""
    callers = {
        "bench/include/cache_sim/graph_cache_context.h": 'ecg_policy::classifyGraspTier',
        "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh": 'ecg_policy::classifyGraspTier',
        "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc": 'ecg_policy::classifyGraspTier',
    }
    for rel, token in callers.items():
        text = (ROOT / rel).read_text(errors="ignore")
        assert token in text, f"{rel} does not call the shared {token} (GRASP tier drift risk)"


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


# The overlays are the tracked home; bench/include/gem5_sim/gem5 and the Sniper
# checkout are GENERATED and gitignored. Nothing previously noticed when a change
# was made directly in the generated tree, which is easy to do because that is
# where the build reads from and where a compiler error points you. Such a change
# builds, runs, and measures correctly on this machine and does not exist at all
# on any other -- the worst possible failure, because every local check passes.
GEM5_APPLIED = ROOT / "bench/include/gem5_sim/gem5/src"
GEM5_OVERLAY = ROOT / "bench/include/gem5_sim/overlays"


def test_applied_gem5_tree_matches_the_tracked_sources():
    """Files the build copies in must be byte-identical where it reads them.

    The pair list is derived from setup_gem5.OVERLAY_FILE_MAP rather than from a
    directory walk, because the map is the authority on what gets copied and it
    includes sources from OUTSIDE the overlays directory. A walk missed
    ../../hawkeye_policy.h, leaving a BASELINE replacement policy -- one that
    every comparison is measured against -- with no drift guard at all.
    """
    if not GEM5_APPLIED.is_dir():
        import pytest
        pytest.skip("gem5 checkout not present")
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "setup_gem5_for_test", ROOT / "scripts/setup_gem5.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["setup_gem5_for_test"] = mod
    spec.loader.exec_module(mod)

    checked, drifted, missing = 0, [], []
    for src_rel, dst_rel in mod.OVERLAY_FILE_MAP.items():
        src = (GEM5_OVERLAY / src_rel).resolve()
        dst = (GEM5_APPLIED / dst_rel).resolve()
        if not src.is_file():
            missing.append(f"source {src_rel}")
            continue
        if not dst.is_file():
            missing.append(f"installed {dst_rel}")
            continue
        checked += 1
        if _sha(src) != _sha(dst):
            drifted.append(dst_rel)
    assert checked > 0, "no copied pairs compared; the check is vacuous"
    assert not missing, (
        f"copy map entries with no file on one side: {missing}")
    assert not drifted, (
        f"{drifted} differ from their tracked sources. The gem5 checkout is "
        "generated and gitignored, so these edits exist only on this machine; "
        "move them into the tracked source and re-apply.")


def test_every_ecg_instruction_in_the_built_decoder_is_tracked():
    """An opcode added straight into the generated decoder would be lost.

    This is not hypothetical: ecg_extract2c was added to the gem5 checkout,
    built, and measured, while the tracked overlay knew nothing about it. A
    fresh clone would have produced a guest that emits the instruction and a
    simulator that cannot decode it.
    """
    applied = GEM5_APPLIED / "arch/riscv/isa/decoder.isa"
    overlay = (GEM5_OVERLAY / "arch/riscv/isa/decoder_ecg_extract.isa")
    if not applied.exists():
        import pytest
        pytest.skip("gem5 checkout not present")
    import re
    names = lambda t: set(re.findall(r"\b(ecg_[a-z0-9_]+)\(\{\{", t))
    built, tracked = names(applied.read_text()), names(overlay.read_text())
    untracked = sorted(built - tracked)
    assert not untracked, (
        f"{untracked} exist only in the generated gem5 decoder, so they are "
        "not in version control and will vanish on a fresh checkout; add them "
        f"to {overlay.relative_to(ROOT)}")
    # The other direction matters too: a tracked instruction absent from the
    # build means the overlay was edited without reinstalling, so the simulator
    # being measured is not the one in version control.
    uninstalled = sorted(tracked - built)
    assert not uninstalled, (
        f"{uninstalled} are tracked in the overlay but absent from the built "
        "decoder; re-apply the overlays, or the measured simulator is not the "
        "one under review")
