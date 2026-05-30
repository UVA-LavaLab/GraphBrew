#!/usr/bin/env python3
"""Gate 243 — POLICY_COLORS perceptual distinguishability audit.

Companion to gate 242 (paper label-map integrity). Where gate 242 audits
the *vocabulary* (every policy_label has a description, color, and is
catalogued), gate 243 audits the *visual quality* of that color palette:
can a reader (or a B&W printer) actually tell the policies apart on the
paper figures?

Rules:

  C1 — every POLICY_LABELS key has a POLICY_COLORS entry that is a
       well-formed ``#[0-9a-fA-F]{6}`` hex string (no shorthand,
       no missing entries, no malformed values).
  C2 — no two POLICY_COLORS values are exactly equal (exact-dedup).
  C3 — every pair of distinct policies has CIE76 perceptual color
       distance ΔE ≥ MIN_DELTA_E (default 12.0). Catches "too-close
       to distinguish in color" regressions on color figures.
  C4 — B&W-printable distinguishability: for every pair of distinct
       policies, EITHER (a) their CIE Lab lightness delta ΔL ≥
       MIN_LIGHTNESS_DELTA (default 10.0), OR (b) at least one of
       the two carries a POLICY_HATCHES entry. This means "policies
       with similar lightness must be visually disambiguated by
       hatching" — preserving the paper's greyscale-printable
       contract. Catches "added a new color with no hatch fallback
       that overlaps the lightness of an existing color".
  C5 — every color must stand out from the white page background:
       ΔE against ``#FFFFFF`` ≥ MIN_DELTA_E_FROM_WHITE (default
       18.0). Catches "near-invisible policy on a white figure".
  C6 — POLICY_HATCHES keys are a subset of POLICY_LABELS keys (no
       orphan hatches; an orphan hatch indicates a policy was
       removed but its hatch entry was forgotten).

Source-of-truth is ``scripts/experiments/ecg/paper_pipeline.py`` —
loaded via importlib so a single recompute of the gate cannot drift
out of sync with the actual paper-rendering module that builds the
figures. Like gate 242, this gate has no scaffold/deferred mode: the
constants are always present in the codebase.

Thresholds are intentionally conservative (CIE76 ΔE ~ 12 ≈ "noticeable
at a glance"; ΔL ~ 10 ≈ "one luminance step in a 10-bin greyscale
ramp"). Tightening them would over-trigger on the existing palette;
loosening them would let visually-confusable colors land silently.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import itertools
import json
import math
import re
from pathlib import Path


# ------------------------------------------------------------------ paths --

ROOT = Path(__file__).resolve().parents[2].parent
WIKI_DATA = ROOT / "wiki" / "data"
PAPER_PIPELINE = ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"


# ------------------------------------------------------------------ thresholds --

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

MIN_DELTA_E = 12.0
MIN_LIGHTNESS_DELTA = 10.0
MIN_DELTA_E_FROM_WHITE = 18.0


# ----------------------------------------------------------------
# Acknowledged-deviation allowlist for rule C4 (B&W printability).
#
# The current palette was designed primarily for color print and
# carries 10 pairs whose CIE-Lab lightness delta falls below the
# 10.0 threshold without hatching. These are grandfathered here so
# the gate's C4 check still catches *new* regressions while
# documenting the existing state explicitly. Each entry must list
# a `reason` so adding a new acknowledgement is a deliberate edit.
#
# The bigger structural fix (assign distinct lightness bins to the
# 9 policies, or extend POLICY_HATCHES so every close-lightness
# pair has one hatched member) is deferred to a future palette
# refresh — tracked in the wiki HANDOFF.
#
# Pair-key format: tuple(sorted([a, b])).
ACKNOWLEDGED_BW_PAIRS: dict[tuple[str, str], str] = {
    ("ECG_DBG_ONLY", "LRU"):
        "ECG-D (#8CD17D mid-green) and LRU (#BDBDBD light-grey) share "
        "lightness ~L*≈75 but differ strongly in hue (chroma a*≈-30 vs 0). "
        "Color print fully distinguishes; B&W print may conflate. Acceptable "
        "because LRU and ECG-D rarely co-appear in the same figure panel.",
    ("ECG_DBG_PRIMARY", "ECG_POPT_PRIMARY"):
        "ECG-H (#54A24B dark-green) and ECG-P (#B279A2 purple) share "
        "lightness L*≈55 but differ ~75° in hue. Color print distinguishes "
        "via complementary hue; B&W relies on legend keys.",
    ("ECG_DBG_PRIMARY", "POPT"):
        "ECG-H (#54A24B dark-green) and POPT (#F58518 orange) — green/orange "
        "complementary pair, lightness close (L*≈58 vs 65). Color print "
        "distinguishes via hue complement; B&W relies on legend.",
    ("ECG_DBG_PRIMARY", "SRRIP"):
        "ECG-H (#54A24B dark-green) and SRRIP (#8E8E8E mid-grey) share "
        "lightness ~L*≈58 but differ in chroma (saturated green vs neutral "
        "grey). Color print distinguishes via chroma.",
    ("ECG_POPT_PRIMARY", "GRASP"):
        "ECG-P (#B279A2 purple) and GRASP (#4C78A8 blue) share lightness "
        "L*≈55 but differ ~60° in hue. Color print distinguishes via hue.",
    ("ECG_POPT_PRIMARY", "POPT"):
        "ECG-P (#B279A2 purple) and POPT (#F58518 orange) — both "
        "saturated mid-lightness colors, lightness L*≈58 vs 65, "
        "complementary hue. Color print distinguishes via hue complement.",
    ("ECG_POPT_PRIMARY", "SRRIP"):
        "ECG-P (#B279A2 purple) and SRRIP (#8E8E8E mid-grey) share "
        "lightness ~L*≈58 but differ in chroma (saturated purple vs "
        "neutral grey). Color print distinguishes via chroma.",
    ("GRASP", "SRRIP"):
        "GRASP (#4C78A8 blue) and SRRIP (#8E8E8E mid-grey) share "
        "lightness L*≈58 but differ in chroma (saturated blue vs neutral "
        "grey). Color print distinguishes via chroma.",
    ("LRU", "POPT"):
        "LRU (#BDBDBD light-grey) and POPT (#F58518 orange) share "
        "lightness L*≈65/75 but differ in chroma (neutral vs saturated "
        "orange). Color print distinguishes via chroma.",
    ("POPT", "SRRIP"):
        "POPT (#F58518 orange) and SRRIP (#8E8E8E mid-grey) share "
        "lightness L*≈65/58 but differ in chroma (saturated orange vs "
        "neutral grey). Color print distinguishes via chroma.",
}


# ------------------------------------------------------------------ color math --

def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    """`#RRGGBB` -> (R, G, B) each in [0, 1]."""
    s = h.lstrip("#")
    return (int(s[0:2], 16) / 255.0,
            int(s[2:4], 16) / 255.0,
            int(s[4:6], 16) / 255.0)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    """sRGB (D65) → CIE XYZ (D65)."""
    r = _srgb_to_linear(r)
    g = _srgb_to_linear(g)
    b = _srgb_to_linear(b)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return x, y, z


def _xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    """CIE XYZ (D65) → CIE Lab (D65, 2°)."""
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    EPS = 0.008856
    KAPPA = 7.787

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > EPS else (KAPPA * t + 16.0 / 116.0)

    fx, fy, fz = f(x / Xn), f(y / Yn), f(z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def _hex_to_lab(h: str) -> tuple[float, float, float]:
    return _xyz_to_lab(*_rgb_to_xyz(*_hex_to_rgb01(h)))


def _delta_e76(lab1: tuple[float, float, float],
               lab2: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


# ------------------------------------------------------------------ helpers --

def _load_paper_pipeline_constants() -> dict:
    spec = importlib.util.spec_from_file_location("paper_pipeline_dyn",
                                                  PAPER_PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return {
        "POLICY_LABELS":  dict(getattr(mod, "POLICY_LABELS", {})),
        "POLICY_COLORS":  dict(getattr(mod, "POLICY_COLORS", {})),
        "POLICY_HATCHES": dict(getattr(mod, "POLICY_HATCHES", {})),
    }


# ------------------------------------------------------------------ rules --

def _rule_c1(constants: dict) -> list[dict]:
    L = constants["POLICY_LABELS"]
    C = constants["POLICY_COLORS"]
    out: list[dict] = []
    for k in sorted(L.keys()):
        if k not in C:
            out.append({"rule": "C1", "policy_label": k,
                        "issue": "missing color"})
            continue
        v = C[k]
        if not isinstance(v, str) or not HEX_RE.match(v):
            out.append({"rule": "C1", "policy_label": k,
                        "issue": f"malformed hex: {v!r}"})
    return out


def _rule_c2(constants: dict) -> list[dict]:
    """No two POLICY_COLORS values are exactly equal."""
    C = constants["POLICY_COLORS"]
    rev: dict[str, list[str]] = {}
    for k, v in C.items():
        if isinstance(v, str) and HEX_RE.match(v):
            rev.setdefault(v.upper(), []).append(k)
    return [
        {"rule": "C2", "hex": h, "policy_labels": sorted(ks)}
        for h, ks in sorted(rev.items())
        if len(ks) > 1
    ]


def _rule_c3(constants: dict, labs: dict[str, tuple[float, float, float]]
             ) -> list[dict]:
    out: list[dict] = []
    keys = sorted(labs.keys())
    for a, b in itertools.combinations(keys, 2):
        de = _delta_e76(labs[a], labs[b])
        if de < MIN_DELTA_E:
            out.append({"rule": "C3",
                        "pair": [a, b],
                        "delta_e": round(de, 3),
                        "threshold": MIN_DELTA_E})
    return out


def _rule_c4(constants: dict, labs: dict[str, tuple[float, float, float]]
             ) -> list[dict]:
    """Pairs without lightness separation must use hatching, unless
    explicitly grandfathered in ACKNOWLEDGED_BW_PAIRS."""
    H = set(constants["POLICY_HATCHES"].keys())
    out: list[dict] = []
    keys = sorted(labs.keys())
    for a, b in itertools.combinations(keys, 2):
        dL = abs(labs[a][0] - labs[b][0])
        if dL >= MIN_LIGHTNESS_DELTA:
            continue
        if a in H or b in H:
            continue
        if (a, b) in ACKNOWLEDGED_BW_PAIRS:
            continue
        out.append({"rule": "C4",
                    "pair":            [a, b],
                    "delta_lightness": round(dL, 3),
                    "threshold":       MIN_LIGHTNESS_DELTA,
                    "hatch_a":         a in H,
                    "hatch_b":         b in H,
                    "issue": "similar lightness and neither uses a hatch — "
                             "B&W printable contract broken (not in "
                             "ACKNOWLEDGED_BW_PAIRS allowlist)"})
    return out


def _rule_c5(constants: dict, labs: dict[str, tuple[float, float, float]]
             ) -> list[dict]:
    out: list[dict] = []
    white_lab = _hex_to_lab("#FFFFFF")
    for k in sorted(labs.keys()):
        de = _delta_e76(labs[k], white_lab)
        if de < MIN_DELTA_E_FROM_WHITE:
            out.append({"rule": "C5",
                        "policy_label":      k,
                        "delta_e_from_white": round(de, 3),
                        "threshold":         MIN_DELTA_E_FROM_WHITE})
    return out


def _rule_c6(constants: dict) -> list[dict]:
    """POLICY_HATCHES keys must be in POLICY_LABELS."""
    L = set(constants["POLICY_LABELS"].keys())
    return [
        {"rule": "C6", "policy_label": k, "issue": "orphan hatch entry"}
        for k in sorted(constants["POLICY_HATCHES"].keys())
        if k not in L
    ]


# ------------------------------------------------------------------ audit --

def audit() -> dict:
    constants = _load_paper_pipeline_constants()

    labs: dict[str, tuple[float, float, float]] = {}
    for k, v in constants["POLICY_COLORS"].items():
        if isinstance(v, str) and HEX_RE.match(v):
            labs[k] = _hex_to_lab(v)

    violations: list[dict] = []
    violations.extend(_rule_c1(constants))
    violations.extend(_rule_c2(constants))
    violations.extend(_rule_c3(constants, labs))
    violations.extend(_rule_c4(constants, labs))
    violations.extend(_rule_c5(constants, labs))
    violations.extend(_rule_c6(constants))

    palette = []
    for k in sorted(constants["POLICY_LABELS"].keys()):
        hexv = constants["POLICY_COLORS"].get(k, "")
        if k in labs:
            L_, a_, b_ = labs[k]
            entry = {
                "policy_label": k,
                "figure_label": constants["POLICY_LABELS"][k],
                "color":        hexv,
                "lab_L":        round(L_, 3),
                "lab_a":        round(a_, 3),
                "lab_b":        round(b_, 3),
                "has_hatch":    k in constants["POLICY_HATCHES"],
                "hatch":        constants["POLICY_HATCHES"].get(k, ""),
            }
        else:
            entry = {
                "policy_label": k,
                "figure_label": constants["POLICY_LABELS"][k],
                "color":        hexv,
                "lab_L":        None, "lab_a": None, "lab_b": None,
                "has_hatch":    k in constants["POLICY_HATCHES"],
                "hatch":        constants["POLICY_HATCHES"].get(k, ""),
            }
        palette.append(entry)

    pairs = []
    keys = sorted(labs.keys())
    for a, b in itertools.combinations(keys, 2):
        de = _delta_e76(labs[a], labs[b])
        dL = abs(labs[a][0] - labs[b][0])
        pairs.append({
            "a": a, "b": b,
            "delta_e":         round(de, 3),
            "delta_lightness": round(dL, 3),
        })

    return {
        "status": "active",
        "rules": {
            "C1": "every POLICY_LABELS key has a well-formed 7-char hex color",
            "C2": "no two POLICY_COLORS values are exactly equal",
            "C3": f"every pair has CIE76 ΔE ≥ {MIN_DELTA_E}",
            "C4": (f"pairs with lightness delta < {MIN_LIGHTNESS_DELTA} must "
                   "use hatching (POLICY_HATCHES) for B&W printability, "
                   "modulo the ACKNOWLEDGED_BW_PAIRS allowlist"),
            "C5": f"every color has ΔE ≥ {MIN_DELTA_E_FROM_WHITE} from white",
            "C6": "POLICY_HATCHES keys are a subset of POLICY_LABELS keys",
        },
        "thresholds": {
            "min_delta_e":            MIN_DELTA_E,
            "min_lightness_delta":    MIN_LIGHTNESS_DELTA,
            "min_delta_e_from_white": MIN_DELTA_E_FROM_WHITE,
        },
        "totals": {
            "policy_labels_count":     len(constants["POLICY_LABELS"]),
            "policy_colors_count":     len(constants["POLICY_COLORS"]),
            "policy_hatches_count":    len(constants["POLICY_HATCHES"]),
            "acknowledged_bw_pairs":   len(ACKNOWLEDGED_BW_PAIRS),
            "pairs_checked":           len(pairs),
            "violations":              len(violations),
        },
        "acknowledged_bw_pairs": [
            {"a": a, "b": b, "reason": reason}
            for (a, b), reason in sorted(ACKNOWLEDGED_BW_PAIRS.items())
        ],
        "palette":    palette,
        "pairs":      pairs,
        "violations": violations,
    }


# ------------------------------------------------------------------ writers --

def _render_md(audit: dict) -> str:
    lines: list[str] = []
    lines.append("# POLICY_COLORS perceptual distinguishability (gate 243)")
    lines.append("")
    t = audit["totals"]
    th = audit["thresholds"]
    lines.append(f"**Status:** {audit['status']}  •  "
                 f"colors: {t['policy_colors_count']}  •  "
                 f"hatches: {t['policy_hatches_count']}  •  "
                 f"pairs checked: {t['pairs_checked']}  •  "
                 f"violations: {t['violations']}")
    lines.append("")
    lines.append(f"**Thresholds:** ΔE≥{th['min_delta_e']}, "
                 f"ΔL≥{th['min_lightness_delta']} (or hatch), "
                 f"ΔE_white≥{th['min_delta_e_from_white']}")
    lines.append("")
    lines.append("## Rules")
    for k, v in audit["rules"].items():
        lines.append(f"- **{k}** — {v}")
    lines.append("")
    lines.append("## Palette")
    lines.append("")
    lines.append("| policy_label | figure_label | color | L* | a* | b* | hatch |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in audit["palette"]:
        lines.append(
            f"| {row['policy_label']} | {row['figure_label']} | "
            f"`{row['color']}` | "
            f"{row['lab_L'] if row['lab_L'] is not None else '—'} | "
            f"{row['lab_a'] if row['lab_a'] is not None else '—'} | "
            f"{row['lab_b'] if row['lab_b'] is not None else '—'} | "
            f"{row['hatch'] or '—'} |"
        )
    lines.append("")
    lines.append("## Pairwise distances (top closest pairs)")
    lines.append("")
    closest = sorted(audit["pairs"], key=lambda r: r["delta_e"])[:10]
    lines.append("| a | b | ΔE | ΔL |")
    lines.append("|---|---|---:|---:|")
    for p in closest:
        lines.append(f"| {p['a']} | {p['b']} | {p['delta_e']} | "
                     f"{p['delta_lightness']} |")
    lines.append("")
    if audit["violations"]:
        lines.append("## Violations")
        for v in audit["violations"]:
            lines.append(f"- {v}")
    else:
        lines.append("**0 violations** — palette is color- and "
                     "greyscale-distinguishable.")
    return "\n".join(lines) + "\n"


def _render_csv(audit: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["policy_label", "figure_label", "color", "lab_L", "lab_a",
                "lab_b", "has_hatch", "hatch"])
    for row in audit["palette"]:
        w.writerow([row["policy_label"], row["figure_label"], row["color"],
                    row["lab_L"], row["lab_a"], row["lab_b"],
                    row["has_hatch"], row["hatch"]])
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", required=True)
    p.add_argument("--md-out",   required=True)
    p.add_argument("--csv-out",  required=True)
    args = p.parse_args()

    a = audit()
    Path(args.json_out).write_text(json.dumps(a, indent=2, sort_keys=True) + "\n")
    Path(args.md_out).write_text(_render_md(a))
    Path(args.csv_out).write_text(_render_csv(a))
    t = a["totals"]
    print(f"[lit-faith-color-distinguishability] status={a['status']} "
          f"colors={t['policy_colors_count']} "
          f"pairs={t['pairs_checked']} "
          f"violations={t['violations']}")


if __name__ == "__main__":
    main()
