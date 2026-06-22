"""Gate 270 — gem5 overlay-file MD5/SHA-256 byte-content registry.

Locks the BYTE CONTENT of every shipped overlay file under
``bench/include/gem5_sim/overlays/`` against silent edits. Pairs with
gate 267 (gem5 overlay-installation tracker) — gate 267 ensures the
installation map is well-formed and exhaustive; gate 270 ensures the
files themselves haven't been silently rewritten between paper runs.

Catches the silent-edit cases that gate 267 misses:

* Someone fixes a "small bug" in ``grasp_rp.cc`` between paper runs
  and forgets to declare it — every subsequent measurement uses a
  policy that doesn't match the paper text.
* Someone adds a debug ``printf`` to ``ecg_pfx.cc`` that quietly
  affects timing in the gem5 build.
* Someone reformats ``GraphReplacementPolicies.py`` and the
  reformatter inadvertently removes a SimObject class.
* Someone edits ``decoder_ecg_extract.isa`` to change the opcode
  bit-pattern, breaking the riscv-toolchain pseudo-instruction.

Rules M1-M7:

* **M1** — every file in the registry hashes to its registered SHA-256.
* **M2** — every registered file's byte length is within sane bounds
  (50 < size < 500_000).
* **M3** — required content markers per file are present (per
  ``OVERLAY_REQUIRED_MARKERS`` mapping below).
* **M4** — registry is exhaustive: every regular file on disk under
  ``overlays/`` with a tracked extension is in the registry
  (modulo ``OVERLAY_HASH_EXTRA_ALLOW``).
* **M5** — only tracked extensions appear under ``overlays/``
  (``.cc``, ``.hh``, ``.py``, ``.isa``, ``.patch``).
* **M6** — SimObject declarations are present in the registered
  ``.py`` files (every policy/prefetcher token has a matching
  ``class <Token>(...)`` declaration).
* **M7** — both expected ``SConscript.patch`` files exist (one under
  replacement_policies/, one under prefetch/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAYS_ROOT = REPO_ROOT / "bench" / "include" / "gem5_sim" / "overlays"

# Canonical byte-content registry (path → SHA-256). Updated by running
# the generator with ``--update`` after a deliberate overlay edit.
OVERLAY_HASH_REGISTRY: dict[str, str] = {
    "arch/riscv/isa/decoder_ecg_extract.isa":
        "814b94e5f60629ade9b8396ec6661caeb98dbca2c72fc2c7f53262ff64bb4c39",
    "arch/riscv/isa/formats/ecg.isa":
        "a5751728427c5c024acb7e0ea3a0b414d6444586dff3bff2b87f2c167f4269e7",
    "mem/cache/prefetch/GraphPrefetchers.py":
        "1bf575d23bdb464288c749c486d9895d95212efac82ccdb44be6fbbe24b59868",
    "mem/cache/prefetch/SConscript.patch":
        "c5d46f5fcabe70266587c5146bfb7f415efc50cbbaee806a9a3123e3db7a6f70",
    "mem/cache/prefetch/droplet.cc":
        "fab5d0ae1b60bcf551d75ab0870a06736ee54550b1c746269d57d8e6d6acfd26",
    "mem/cache/prefetch/droplet.hh":
        "72ff12b8f33ffdea8c96708673aa161f3a372850c3c70d7156e8c86357f0deb9",
    "mem/cache/prefetch/ecg_pfx.cc":
        "abc3d84fbb2cfe7c7ac7b859ee227993212139c73d38a1a655d4984417346f87",
    "mem/cache/prefetch/ecg_pfx.hh":
        "07a5c4548b29bd42b7d69727a9e1c56ee6fbfd65fc8b343885e1be7f930e2732",
    "mem/cache/prefetch/queued_cc_latency.patch":
        "49ac0ec08659fcb454f9ab782156c1c57f7c87577926c714e882798333260488",
    "mem/cache/prefetch/queued_hh.patch":
        "0289b71df1815f64c7a27807556664e23f383bf6f69e0669ccfaa903014e608c",
    "mem/cache/replacement_policies/GraphReplacementPolicies.py":
        "eb5322513478bee3afa66d291a18a7dff80842ebb2c121527c291137ea75ece0",
    "mem/cache/replacement_policies/SConscript.patch":
        "6c860d12c30d3ae18d65ccf2fb9da466291cb8b2727d47e0d4b7da2c97293f69",
    "mem/cache/replacement_policies/ecg_rp.cc":
        "f495c4ac9dd55e9b5813f61d360a45b8395292ae3bff77a36196e67aa37fbd1c",
    "mem/cache/replacement_policies/ecg_rp.hh":
        "c529a51502fc3ff34614eed9feb0c486662e0f4f4417be975967e0b5d173d8b3",
    "mem/cache/replacement_policies/ecg_victim_policy.hh":
        "a231b2e241c842d39de28c81b4f0783e42d876227247fc56980b858d6ced5ab9",
    "mem/cache/replacement_policies/graph_cache_context_gem5.hh":
        "7e20b7b301885665a5983b200277ebeef1d344accc6311c76f0b83547dd533c9",
    "mem/cache/replacement_policies/grasp_rp.cc":
        "101d17b1c77b6f1c702754a411dd24ce1a26d584fdb95cee8326e47f23704c9e",
    "mem/cache/replacement_policies/grasp_rp.hh":
        "be84299387a61148f1cd234f517a9ec1c6c1e231c465da877b0028de26410466",
    "mem/cache/replacement_policies/popt_rp.cc":
        "d49b62bd7076d43beba112b6718ba5697f76c436eaeec4022281cdf79b8e0943",
    "mem/cache/replacement_policies/popt_rp.hh":
        "deeb4914d45aea570515a1acf6a59a189d45b76b7252fefe601ef420791ef93b",
}

OVERLAY_HASH_EXTRA_ALLOW: set[str] = set()

OVERLAY_TRACKED_EXTS: tuple[str, ...] = (".cc", ".hh", ".py", ".isa", ".patch")
OVERLAY_IGNORED_EXTS: tuple[str, ...] = (".md",)

OVERLAY_MIN_SIZE = 50
OVERLAY_MAX_SIZE = 500_000

OVERLAY_REQUIRED_MARKERS: dict[str, tuple[str, ...]] = {
    "mem/cache/replacement_policies/grasp_rp.cc": ("GraphGraspRP",),
    "mem/cache/replacement_policies/grasp_rp.hh": ("GraphGraspRP",),
    "mem/cache/replacement_policies/popt_rp.cc": ("GraphPoptRP",),
    "mem/cache/replacement_policies/popt_rp.hh": ("GraphPoptRP",),
    "mem/cache/replacement_policies/ecg_rp.cc": ("GraphEcgRP",),
    "mem/cache/replacement_policies/ecg_rp.hh": ("GraphEcgRP",),
    "mem/cache/replacement_policies/graph_cache_context_gem5.hh":
        ("graph_cache_context", "register"),
    "mem/cache/prefetch/droplet.cc": ("Droplet",),
    "mem/cache/prefetch/droplet.hh": ("Droplet",),
    "mem/cache/prefetch/ecg_pfx.cc": ("EcgPfx",),
    "mem/cache/prefetch/ecg_pfx.hh": ("EcgPfx",),
    "mem/cache/replacement_policies/GraphReplacementPolicies.py":
        ("class ", "SimObject"),
    "mem/cache/prefetch/GraphPrefetchers.py":
        ("class ", "SimObject"),
}

# .py files that should declare SimObject classes for each policy/prefetcher.
SIMOBJECT_PY_CLASSES: dict[str, tuple[str, ...]] = {
    "mem/cache/replacement_policies/GraphReplacementPolicies.py":
        ("GraphGrasp", "GraphPopt", "GraphEcg"),
    "mem/cache/prefetch/GraphPrefetchers.py":
        ("Droplet", "EcgPfx"),
}

EXPECTED_PATCHES = (
    "mem/cache/replacement_policies/SConscript.patch",
    "mem/cache/prefetch/SConscript.patch",
)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _walk_overlays() -> list[Path]:
    if not OVERLAYS_ROOT.is_dir():
        return []
    return sorted(
        p for p in OVERLAYS_ROOT.rglob("*")
        if p.is_file()
    )


def _rel(p: Path) -> str:
    return str(p.relative_to(OVERLAYS_ROOT))


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_m1_hash_parity() -> list[dict]:
    """Every registered file matches its SHA-256."""
    out = []
    for rel, want in OVERLAY_HASH_REGISTRY.items():
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            out.append({
                "rule": "M1",
                "path": rel,
                "issue": "registered file missing on disk",
            })
            continue
        got = _sha256(p)
        if got != want:
            out.append({
                "rule": "M1",
                "path": rel,
                "want_sha256": want,
                "got_sha256": got,
            })
    return out


def _check_m2_size_bounds() -> list[dict]:
    """Registered files have sensible byte sizes."""
    out = []
    for rel in OVERLAY_HASH_REGISTRY:
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            continue
        sz = p.stat().st_size
        if sz < OVERLAY_MIN_SIZE or sz > OVERLAY_MAX_SIZE:
            out.append({
                "rule": "M2",
                "path": rel,
                "size": sz,
                "bounds": [OVERLAY_MIN_SIZE, OVERLAY_MAX_SIZE],
            })
    return out


def _check_m3_required_markers() -> list[dict]:
    """Required tokens appear inside each marker-bearing file."""
    out = []
    for rel, markers in OVERLAY_REQUIRED_MARKERS.items():
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text("utf-8", errors="replace")
        for m in markers:
            if m not in text:
                out.append({
                    "rule": "M3",
                    "path": rel,
                    "missing_marker": m,
                })
    return out


def _check_m4_exhaustive() -> list[dict]:
    """Registry covers every regular tracked file under overlays/."""
    out = []
    on_disk = {
        _rel(p) for p in _walk_overlays()
        if p.suffix in OVERLAY_TRACKED_EXTS
    }
    registered = set(OVERLAY_HASH_REGISTRY) | OVERLAY_HASH_EXTRA_ALLOW
    for missing in sorted(on_disk - registered):
        out.append({
            "rule": "M4",
            "path": missing,
            "issue": "on disk but not in registry",
        })
    for extra in sorted(registered - on_disk):
        out.append({
            "rule": "M4",
            "path": extra,
            "issue": "in registry but missing on disk",
        })
    return out


def _check_m5_extension_whitelist() -> list[dict]:
    """No surprise extensions in overlays/ (ignores .md docs)."""
    out = []
    for p in _walk_overlays():
        if p.suffix in OVERLAY_IGNORED_EXTS:
            continue
        if p.suffix not in OVERLAY_TRACKED_EXTS:
            out.append({
                "rule": "M5",
                "path": _rel(p),
                "extension": p.suffix,
            })
    return out


def _check_m6_simobject_classes() -> list[dict]:
    """SimObject .py files declare classes for every policy/prefetcher token."""
    out = []
    for rel, tokens in SIMOBJECT_PY_CLASSES.items():
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text("utf-8", errors="replace")
        for t in tokens:
            rx = re.compile(rf"^class\s+\w*{re.escape(t)}\w*\s*\(", re.M)
            if not rx.search(text):
                out.append({
                    "rule": "M6",
                    "path": rel,
                    "missing_token": t,
                })
    return out


def _check_m7_patches_present() -> list[dict]:
    """Both expected SConscript.patch files exist."""
    out = []
    for rel in EXPECTED_PATCHES:
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            out.append({
                "rule": "M7",
                "path": rel,
                "issue": "expected SConscript.patch missing",
            })
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_m1_hash_parity()
    viols += _check_m2_size_bounds()
    viols += _check_m3_required_markers()
    viols += _check_m4_exhaustive()
    viols += _check_m5_extension_whitelist()
    viols += _check_m6_simobject_classes()
    viols += _check_m7_patches_present()

    counts = {
        "registered": len(OVERLAY_HASH_REGISTRY),
        "tracked_exts": list(OVERLAY_TRACKED_EXTS),
        "on_disk": sum(1 for _ in _walk_overlays()),
        "marker_files": len(OVERLAY_REQUIRED_MARKERS),
        "simobject_files": len(SIMOBJECT_PY_CLASSES),
        "expected_patches": len(EXPECTED_PATCHES),
    }
    return {
        "schema": "lit-faith-gem5-overlay-hash-registry/1",
        "status": "active",
        "counts": counts,
        "registry": {k: v for k, v in OVERLAY_HASH_REGISTRY.items()},
        "rules": {
            "M1": "registered file matches SHA-256",
            "M2": "registered file size in [50, 500_000]",
            "M3": "required content markers present per file",
            "M4": "registry exhaustive over overlays/ tree",
            "M5": "no surprise extensions (whitelist: .cc/.hh/.py/.isa/.patch)",
            "M6": "SimObject .py declares class per policy/prefetcher token",
            "M7": "SConscript.patch present under replacement_policies/ + prefetch/",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# gem5 overlay-file hash registry (gate 270)",
        "",
        "_Auto-generated by `lit_faith_gem5_overlay_hash_registry.py`._",
        "",
        f"- registered files: **{doc['counts']['registered']}**",
        f"- on-disk files: **{doc['counts']['on_disk']}**",
        f"- tracked extensions: **{', '.join(doc['counts']['tracked_exts'])}**",
        f"- files with required markers: **{doc['counts']['marker_files']}**",
        f"- SimObject .py files: **{doc['counts']['simobject_files']}**",
        f"- expected patches: **{doc['counts']['expected_patches']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
        "## Registry (path → SHA-256)",
        "",
        "| path | SHA-256 |",
        "| --- | --- |",
    ]
    for rel, h in sorted(doc["registry"].items()):
        lines.append(f"| `{rel}` | `{h}` |")
    lines.append("")
    if doc["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in doc["violations"]:
            lines.append(f"- {json.dumps(v, sort_keys=True)}")
    else:
        lines.append("## ✅ No violations")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", "utf-8")


def write_csv(doc: dict, path: Path) -> None:
    rows = ["kind,path,sha256_or_meta"]
    for rel, h in sorted(doc["registry"].items()):
        rows.append(f"hash,{rel},{h}")
    for v in doc["violations"]:
        rows.append(f"violation,{v.get('path','')},{v.get('rule','')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", "utf-8")


def _update_hashes() -> None:
    """Helper: rewrite OVERLAY_HASH_REGISTRY in this file with current shas.

    Prints a Python literal that a maintainer can paste in. We never
    self-modify on disk — drift must be reviewed.
    """
    print("OVERLAY_HASH_REGISTRY: dict[str, str] = {")
    for p in _walk_overlays():
        if p.suffix in OVERLAY_TRACKED_EXTS:
            print(f'    "{_rel(p)}":')
            print(f'        "{_sha256(p)}",')
    print("}")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", required=False, type=Path)
    ap.add_argument("--md-out", required=False, type=Path)
    ap.add_argument("--csv-out", required=False, type=Path)
    ap.add_argument("--update", action="store_true",
                    help="Print refreshed OVERLAY_HASH_REGISTRY literal.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.update:
        _update_hashes()
        return 0

    doc = audit()
    if args.json_out:
        write_json(doc, args.json_out)
    if args.md_out:
        write_md(doc, args.md_out)
    if args.csv_out:
        write_csv(doc, args.csv_out)

    print(
        f"[lit-faith-gem5-overlay-hash-registry] status={doc['status']} "
        f"registered={doc['counts']['registered']} "
        f"on_disk={doc['counts']['on_disk']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
