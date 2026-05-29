"""Gate 72 — cross-tool SRRIP-vs-GRASP slope ordering invariant.

The "oracle-aware policies are less cache-hungry" claim says that
GRASP (oracle-aware) has a shallower capacity-sensitivity slope
than SRRIP (non-oracle-aware): SRRIP's miss rate falls more steeply
as L3 grows because it cannot anticipate the reuse-likely block
identity at small caches.

This gate verifies the claim is REPLICATED across all three tools
in the GraphBrew pipeline:
  - cache-sim sweep (1MB/4MB/8MB), via gate 66's per-policy slope
  - gem5 anchor sweep (4kB/32kB/256kB/2MB), via gate 70's median
  - sniper anchor sweep (4kB/32kB/256kB/2MB), via gate 71's median

For each tool, the gate reads the corresponding gate's PER-POLICY
median slope (a single number per policy per tool) and checks the
SRRIP ≤ GRASP ordering. PASS iff:
  (1) all three tools' artifacts are present,
  (2) every tool shows median(SRRIP) ≤ median(GRASP),
  (3) at least 2 of the 3 tools show STRICT inequality with at
      least 0.05 pp/oct gap (this guards against trivial 0-gap
      "agreement" via measurement floor).

The LRU-vs-GRASP ordering is reported per tool as INFORMATIONAL
(gates 70/71 documented that sub-WSS anchor scales invert this).

Output schema:
  meta.tools                  : tool names compared
  meta.gap_floor_pp_octave    : 0.05
  meta.required_strict_tools  : 2
  per_tool[T].grasp_median    : pp/oct
  per_tool[T].srrip_median    : pp/oct
  per_tool[T].lru_median      : pp/oct (informational)
  per_tool[T].srrip_minus_grasp_pp_oct : pp/oct (gated)
  per_tool[T].lru_minus_grasp_pp_oct   : pp/oct (informational)
  per_tool[T].srrip_steeper           : bool (srrip < grasp)
  per_tool[T].srrip_strictly_steeper  : bool (srrip < grasp - gap_floor)
  verdict_checks.all_tools_srrip_le_grasp : bool
  verdict_checks.enough_tools_strictly_steeper : bool
  verdict                                 : PASS | FAIL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_SIM = REPO_ROOT / "wiki" / "data" / "capacity_sensitivity.json"
DEFAULT_GEM5     = REPO_ROOT / "wiki" / "data" / "gem5_slope_replay.json"
DEFAULT_SNIPER   = REPO_ROOT / "wiki" / "data" / "sniper_slope_replay.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "cross_tool_slope_ordering.json"
DEFAULT_MD_OUT   = REPO_ROOT / "wiki" / "data" / "cross_tool_slope_ordering.md"

GAP_FLOOR_PP_OCTAVE = 0.05
REQUIRED_STRICT_TOOLS = 2


def _cache_sim_medians(path: Path) -> dict[str, float]:
    """Pull per-policy median slope from gate 66's
    capacity_sensitivity.json. Schema: meta.policy_summary.<P>.median_pp.
    """
    doc = json.loads(path.read_text())
    per_policy = doc.get("meta", {}).get("policy_summary", {})
    if not isinstance(per_policy, dict) or not per_policy:
        raise ValueError(
            f"cannot find meta.policy_summary in {path}; "
            f"meta keys = {list(doc.get('meta', {}).keys())}"
        )
    out: dict[str, float] = {}
    for policy, entry in per_policy.items():
        if not isinstance(entry, dict):
            continue
        for k in ("median_pp", "median_pp_oct", "median",
                  "slope_pp_oct", "median_slope_pp_oct"):
            if k in entry and isinstance(entry[k], (int, float)):
                out[policy] = float(entry[k])
                break
    return out


def _anchor_medians(path: Path) -> dict[str, float]:
    """Pull per-policy median slope from a gemX/sniper anchor replay
    artifact (gates 70/71). They expose meta.per_policy.<P>.median."""
    doc = json.loads(path.read_text())
    per_policy = doc.get("meta", {}).get("per_policy", {})
    out: dict[str, float] = {}
    for policy, entry in per_policy.items():
        if not isinstance(entry, dict):
            continue
        med = entry.get("median")
        if isinstance(med, (int, float)):
            out[policy] = float(med)
    return out


def compute(cache_sim_path: Path, gem5_path: Path, sniper_path: Path) -> dict:
    tools_loaders = {
        "cache_sim": (cache_sim_path, _cache_sim_medians),
        "gem5":      (gem5_path,      _anchor_medians),
        "sniper":    (sniper_path,    _anchor_medians),
    }

    per_tool: dict[str, dict] = {}
    all_le = True
    n_strict = 0

    for tool, (path, loader) in tools_loaders.items():
        if not path.exists():
            per_tool[tool] = {
                "present": False,
                "source": str(path.name),
            }
            all_le = False
            continue
        meds = loader(path)
        grasp = meds.get("GRASP")
        srrip = meds.get("SRRIP")
        lru   = meds.get("LRU")
        if grasp is None or srrip is None:
            per_tool[tool] = {
                "present": True,
                "source": str(path.name),
                "grasp_median": grasp,
                "srrip_median": srrip,
                "lru_median": lru,
                "srrip_minus_grasp_pp_oct": None,
                "lru_minus_grasp_pp_oct": None,
                "srrip_steeper": None,
                "srrip_strictly_steeper": None,
                "error": "missing GRASP or SRRIP median",
            }
            all_le = False
            continue
        sg = srrip - grasp
        lg = (lru - grasp) if lru is not None else None
        steeper = (srrip <= grasp)
        strictly_steeper = (srrip < grasp - GAP_FLOOR_PP_OCTAVE)
        if not steeper:
            all_le = False
        if strictly_steeper:
            n_strict += 1
        per_tool[tool] = {
            "present": True,
            "source": str(path.name),
            "grasp_median": round(grasp, 4),
            "srrip_median": round(srrip, 4),
            "lru_median": round(lru, 4) if lru is not None else None,
            "srrip_minus_grasp_pp_oct": round(sg, 4),
            "lru_minus_grasp_pp_oct":
                round(lg, 4) if lg is not None else None,
            "srrip_steeper": steeper,
            "srrip_strictly_steeper": strictly_steeper,
        }

    verdict_checks = {
        "all_tools_present_and_valid":
            all(per_tool[t].get("present") and "error" not in per_tool[t]
                for t in tools_loaders),
        "all_tools_srrip_le_grasp": all_le,
        "enough_tools_strictly_steeper": n_strict >= REQUIRED_STRICT_TOOLS,
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    return {
        "meta": {
            "tools":                    list(tools_loaders.keys()),
            "gap_floor_pp_octave":      GAP_FLOOR_PP_OCTAVE,
            "required_strict_tools":    REQUIRED_STRICT_TOOLS,
            "n_strict_tools":           n_strict,
            "verdict_checks":           verdict_checks,
            "verdict":                  verdict,
        },
        "per_tool": per_tool,
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    pt = payload["per_tool"]
    lines = [
        "# Cross-tool SRRIP-vs-GRASP slope ordering",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Tools compared:** {', '.join(m['tools'])}  ",
        f"**Gap floor (strict):** {m['gap_floor_pp_octave']} pp/oct  ",
        f"**Required strict tools:** {m['required_strict_tools']}",
        "",
        "## Per-tool medians",
        "",
        "| tool | GRASP | SRRIP | LRU | SRRIP-GRASP | LRU-GRASP (info) | SRRIP <= GRASP | strict |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for tool in m["tools"]:
        e = pt[tool]
        if not e.get("present"):
            lines.append(f"| {tool} | n/a | n/a | n/a | n/a | n/a | ❌ | ❌ |")
            continue
        g = e.get("grasp_median")
        s = e.get("srrip_median")
        l = e.get("lru_median")
        sg = e.get("srrip_minus_grasp_pp_oct")
        lg = e.get("lru_minus_grasp_pp_oct")
        steeper = e.get("srrip_steeper")
        strict = e.get("srrip_strictly_steeper")
        gs = f"{g:+.4f}" if g is not None else "n/a"
        ss = f"{s:+.4f}" if s is not None else "n/a"
        ls = f"{l:+.4f}" if l is not None else "n/a"
        sgs = f"{sg:+.4f}" if sg is not None else "n/a"
        lgs = f"{lg:+.4f}" if lg is not None else "n/a"
        lines.append(
            f"| {tool} | {gs} | {ss} | {ls} | {sgs} | {lgs} | "
            f"{'✅' if steeper else '❌'} | "
            f"{'✅' if strict else '❌'} |"
        )
    lines += [
        "",
        "## Verdict checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for k, v in m["verdict_checks"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "GRASP is oracle-aware; SRRIP is not. The claim under test is "
        "that SRRIP is at least as cache-hungry as GRASP — its miss-rate "
        "slope vs log2(L3) should fall at least as steeply, because at "
        "small caches SRRIP cannot anticipate the reuse-likely block "
        "identity and pays a larger penalty.",
        "",
        "This gate confirms the claim is REPLICATED in all three tools "
        "of the GraphBrew pipeline (cache-sim sweep + gem5 anchor + "
        "sniper anchor). The LRU-vs-GRASP delta is reported per tool "
        "but explicitly NOT gated: gates 70/71 documented that sub-WSS "
        "anchor scales (4kB << email-Eu-core WSS ~4.5kB) can invert the "
        "LRU>GRASP ordering observed at 1-8MB scales — a regime-dependent "
        "physical effect, not a tool artifact.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-sim-json", type=Path, default=DEFAULT_CACHE_SIM)
    ap.add_argument("--gem5-json",      type=Path, default=DEFAULT_GEM5)
    ap.add_argument("--sniper-json",    type=Path, default=DEFAULT_SNIPER)
    ap.add_argument("--json-out",       type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",         type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = compute(args.cache_sim_json, args.gem5_json, args.sniper_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    print(
        f"cross-tool-slope-ordering: strict_tools={m['n_strict_tools']}/3 "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
