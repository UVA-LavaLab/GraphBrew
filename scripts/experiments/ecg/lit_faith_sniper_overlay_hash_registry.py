"""Gate 271 — Sniper overlay-file SHA-256 byte-content registry.

Mirror of gate 270 for Sniper. Locks the BYTE CONTENT of every shipped
overlay source under ``bench/include/sniper_sim/overlays/`` against
silent edits. Pairs with gates 266 (Sniper overlay-installation
tracker) and 268 (setup-script invariants) — those lock the
installation contract; gate 271 locks the actual file bytes the paper
depends on.

Catches the silent-edit cases that the installation-contract gates
miss: a developer fixes a bug in ``cache_set_grasp.cc`` between paper
runs and forgets to declare it; a debug ``printf`` slips into
``ecg_pfx_prefetcher.cc`` and quietly affects timing in the Sniper
build; a reformatter removes a class declaration in
``cache_set_popt.h``.

Rules N1-N6:

* **N1** — every file in the registry hashes to its registered SHA-256.
* **N2** — every registered file's byte length is within sane bounds.
* **N3** — required content markers per file are present.
* **N4** — registry exhaustive over the overlays/ tree.
* **N5** — only tracked extensions (.cc/.h; .md docs ignored).
* **N6** — every prefetcher/cache-set class declaration is present
  (regex ``^class <Token>\\b``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAYS_ROOT = REPO_ROOT / "bench" / "include" / "sniper_sim" / "overlays"

SNIPER_OVERLAY_HASH_REGISTRY: dict[str, str] = {
    "common/core/memory_subsystem/cache/cache_set_ecg.cc":
        "3fbe355af7646ec34fe495198f49d125a7bd09b552d6392f4bb23bd914c762af",
    "common/core/memory_subsystem/cache/cache_set_ecg.h":
        "234f564403be19f927463316a091b77af975a305a3d45cbd7813bf7afd00ea8f",
    "common/core/memory_subsystem/cache/cache_set_grasp.cc":
        "0ac38b7531157f8b345920b21a7ce27cceda95d028a6c7375e0e72be8a535434",
    "common/core/memory_subsystem/cache/cache_set_grasp.h":
        "e6ee05c0310b16c316e59d5d7db58d6e9c0b201ff620f8ebaf7b25313259e2c9",
    "common/core/memory_subsystem/cache/cache_set_popt.cc":
        "0cf1848be1fc326919ad6071a45d4cd2f23f96990df6d47cdf20f984536014e3",
    "common/core/memory_subsystem/cache/cache_set_popt.h":
        "9358c1d96a7cb3d74ab8b3aaf7184562df3033a5ee5954cef79872a3676caa6b",
    "common/core/memory_subsystem/cache/ecg_victim_policy.h":
        "cad8b7b8d57f48680208bb10b536421fcc8ec624c94aa83389429f56dbe8ad6b",
    "common/core/memory_subsystem/cache/graph_cache_context_sniper.cc":
        "51acd61e9de3e094b3ef3d6c2482f333255d0995165e43f4a74cd7583f3fe805",
    "common/core/memory_subsystem/cache/graph_cache_context_sniper.h":
        "936f824123fde9f9cfff8362e81c34895c02e83c287fd412e8556d96121e1744",
    "common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.cc":
        "a16892d1b66024ac028d324f41097f4e352c2d404f407cfb8c2afd4f464d698b",
    "common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.h":
        "4e6556065c6043282a808fcf1d3471629fe6eceda274189d3ac20998d7589ac8",
    "common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.cc":
        "f474306a613bcf4d3e2b1a8115fef0ec611aaabfd59eff1a6606c2e8cd7b073c",
    "common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.h":
        "31446f6e66ccfa2bccea5550f074f423d3fa9b1e803e6740c71292e09159ca89",
}

SNIPER_OVERLAY_HASH_EXTRA_ALLOW: set[str] = set()

SNIPER_OVERLAY_TRACKED_EXTS: tuple[str, ...] = (".cc", ".h")
SNIPER_OVERLAY_IGNORED_EXTS: tuple[str, ...] = (".md",)

SNIPER_OVERLAY_MIN_SIZE = 50
SNIPER_OVERLAY_MAX_SIZE = 500_000

SNIPER_OVERLAY_REQUIRED_MARKERS: dict[str, tuple[str, ...]] = {
    "common/core/memory_subsystem/cache/cache_set_grasp.cc": ("CacheSetGRASP",),
    "common/core/memory_subsystem/cache/cache_set_grasp.h": ("CacheSetGRASP",),
    "common/core/memory_subsystem/cache/cache_set_popt.cc": ("CacheSetPOPT",),
    "common/core/memory_subsystem/cache/cache_set_popt.h": ("CacheSetPOPT",),
    "common/core/memory_subsystem/cache/cache_set_ecg.cc": ("CacheSetECG",),
    "common/core/memory_subsystem/cache/cache_set_ecg.h": ("CacheSetECG",),
    "common/core/memory_subsystem/cache/graph_cache_context_sniper.cc":
        ("GraphCacheContext", "graphbrew"),
    "common/core/memory_subsystem/cache/graph_cache_context_sniper.h":
        ("GraphCacheContext", "graphbrew"),
    "common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.cc":
        ("DropletPrefetcher",),
    "common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.h":
        ("DropletPrefetcher",),
    "common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.cc":
        ("EcgPfxPrefetcher",),
    "common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.h":
        ("EcgPfxPrefetcher",),
}

SNIPER_OVERLAY_CLASS_DECLARATIONS: dict[str, str] = {
    "common/core/memory_subsystem/cache/cache_set_grasp.h": "CacheSetGRASP",
    "common/core/memory_subsystem/cache/cache_set_popt.h": "CacheSetPOPT",
    "common/core/memory_subsystem/cache/cache_set_ecg.h": "CacheSetECG",
    "common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.h":
        "DropletPrefetcher",
    "common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.h":
        "EcgPfxPrefetcher",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _walk_overlays() -> list[Path]:
    if not OVERLAYS_ROOT.is_dir():
        return []
    return sorted(p for p in OVERLAYS_ROOT.rglob("*") if p.is_file())


def _rel(p: Path) -> str:
    return str(p.relative_to(OVERLAYS_ROOT))


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_n1_hash_parity() -> list[dict]:
    out = []
    for rel, want in SNIPER_OVERLAY_HASH_REGISTRY.items():
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            out.append({"rule": "N1", "path": rel, "issue": "registered file missing"})
            continue
        got = _sha256(p)
        if got != want:
            out.append({"rule": "N1", "path": rel,
                        "want_sha256": want, "got_sha256": got})
    return out


def _check_n2_size_bounds() -> list[dict]:
    out = []
    for rel in SNIPER_OVERLAY_HASH_REGISTRY:
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            continue
        sz = p.stat().st_size
        if sz < SNIPER_OVERLAY_MIN_SIZE or sz > SNIPER_OVERLAY_MAX_SIZE:
            out.append({"rule": "N2", "path": rel, "size": sz,
                        "bounds": [SNIPER_OVERLAY_MIN_SIZE,
                                   SNIPER_OVERLAY_MAX_SIZE]})
    return out


def _check_n3_required_markers() -> list[dict]:
    out = []
    for rel, markers in SNIPER_OVERLAY_REQUIRED_MARKERS.items():
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text("utf-8", errors="replace")
        for m in markers:
            if m not in text:
                out.append({"rule": "N3", "path": rel, "missing_marker": m})
    return out


def _check_n4_exhaustive() -> list[dict]:
    out = []
    on_disk = {_rel(p) for p in _walk_overlays()
               if p.suffix in SNIPER_OVERLAY_TRACKED_EXTS}
    registered = set(SNIPER_OVERLAY_HASH_REGISTRY) | SNIPER_OVERLAY_HASH_EXTRA_ALLOW
    for missing in sorted(on_disk - registered):
        out.append({"rule": "N4", "path": missing,
                    "issue": "on disk but not in registry"})
    for extra in sorted(registered - on_disk):
        out.append({"rule": "N4", "path": extra,
                    "issue": "in registry but missing on disk"})
    return out


def _check_n5_extension_whitelist() -> list[dict]:
    out = []
    for p in _walk_overlays():
        if p.suffix in SNIPER_OVERLAY_IGNORED_EXTS:
            continue
        if p.suffix not in SNIPER_OVERLAY_TRACKED_EXTS:
            out.append({"rule": "N5", "path": _rel(p), "extension": p.suffix})
    return out


def _check_n6_class_declarations() -> list[dict]:
    out = []
    for rel, cls in SNIPER_OVERLAY_CLASS_DECLARATIONS.items():
        p = OVERLAYS_ROOT / rel
        if not p.is_file():
            continue
        text = p.read_text("utf-8", errors="replace")
        rx = re.compile(rf"^class\s+{re.escape(cls)}\b", re.M)
        if not rx.search(text):
            out.append({"rule": "N6", "path": rel, "missing_class": cls})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_n1_hash_parity()
    viols += _check_n2_size_bounds()
    viols += _check_n3_required_markers()
    viols += _check_n4_exhaustive()
    viols += _check_n5_extension_whitelist()
    viols += _check_n6_class_declarations()

    counts = {
        "registered": len(SNIPER_OVERLAY_HASH_REGISTRY),
        "tracked_exts": list(SNIPER_OVERLAY_TRACKED_EXTS),
        "on_disk": sum(1 for _ in _walk_overlays()),
        "marker_files": len(SNIPER_OVERLAY_REQUIRED_MARKERS),
        "class_files": len(SNIPER_OVERLAY_CLASS_DECLARATIONS),
    }
    return {
        "schema": "lit-faith-sniper-overlay-hash-registry/1",
        "status": "active",
        "counts": counts,
        "registry": dict(SNIPER_OVERLAY_HASH_REGISTRY),
        "rules": {
            "N1": "registered file matches SHA-256",
            "N2": "registered file size in [50, 500_000]",
            "N3": "required content markers present per file",
            "N4": "registry exhaustive over overlays/ tree",
            "N5": "no surprise extensions (whitelist: .cc/.h)",
            "N6": "expected class declaration present in header",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# Sniper overlay-file hash registry (gate 271)",
        "",
        "_Auto-generated by `lit_faith_sniper_overlay_hash_registry.py`._",
        "",
        f"- registered files: **{doc['counts']['registered']}**",
        f"- on-disk files: **{doc['counts']['on_disk']}**",
        f"- tracked extensions: **{', '.join(doc['counts']['tracked_exts'])}**",
        f"- files with required markers: **{doc['counts']['marker_files']}**",
        f"- class-declaration headers: **{doc['counts']['class_files']}**",
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
    print("SNIPER_OVERLAY_HASH_REGISTRY: dict[str, str] = {")
    for p in _walk_overlays():
        if p.suffix in SNIPER_OVERLAY_TRACKED_EXTS:
            print(f'    "{_rel(p)}":')
            print(f'        "{_sha256(p)}",')
    print("}")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", required=False, type=Path)
    ap.add_argument("--md-out", required=False, type=Path)
    ap.add_argument("--csv-out", required=False, type=Path)
    ap.add_argument("--update", action="store_true")
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
        f"[lit-faith-sniper-overlay-hash-registry] status={doc['status']} "
        f"registered={doc['counts']['registered']} "
        f"on_disk={doc['counts']['on_disk']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
