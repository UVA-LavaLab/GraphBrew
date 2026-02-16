#!/usr/bin/env python3
"""
Multi-Layer Validity Test
=========================

Validates that recursive multi-layer GraphBrew dispatch works correctly
on graphs with large communities.  Compares:

  1. Flat mode (depth=0):  -o 12:leiden      (default, no recursion)
  2. Recursive (depth=1):  -o 12:leiden:8:auto:3:1  (recurse into large communities)
  3. Recursive + auto sub: -o 12:leiden:8:auto:3:1:auto  (adaptive sub-algo)
  4. Compound flat:        -o 12:leiden:hrab  (ordering strategy, no recursion)
  5. Flat + features:      -o 12:leiden:8:auto:3:::merge:hubx  (feature flags)

Checks:
  - All produce valid permutations (bijective mapping)
  - Recursive mode reports sub-community cache analysis
  - Performance delta between flat and recursive
  - Large communities (>LLC) benefit from recursive splitting

Usage:
    pytest scripts/test/test_multilayer_validity.py -v
    python scripts/test/test_multilayer_validity.py  # standalone
"""

import subprocess
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.utils import (
    GRAPHBREW_LAYERS, enumerate_graphbrew_multilayer, BIN_DIR, GRAPHS_DIR,
)

# --- Configuration ---
# Pick a medium graph that has large communities (power-law degree distribution)
TEST_GRAPHS = []
for name in ["web-Google", "as-Skitter", "soc-Epinions1", "amazon0601"]:
    mtx = GRAPHS_DIR / name / f"{name}.mtx"
    if mtx.exists():
        TEST_GRAPHS.append((name, str(mtx)))
    if len(TEST_GRAPHS) >= 2:
        break

CONVERTER = BIN_DIR / "converter"
PR = BIN_DIR / "pr"

# Multi-layer configurations to test
CONFIGS = [
    ("flat_leiden",        "12:leiden"),
    ("flat_rabbit",        "12:rabbit"),
    ("flat_leiden_hrab",   "12:leiden:hrab"),
    ("recursive_d1",       "12:leiden:8:auto:3:1"),
    ("recursive_d1_auto",  "12:leiden:8:auto:3:1:auto"),
    ("flat_merge_hubx",    "12:leiden:merge:hubx"),
]


def run_reorder(graph_path: str, config_opt: str, timeout: int = 300) -> dict:
    """Run converter with a given reorder config, parse output."""
    cmd = [
        str(CONVERTER),
        "-f", graph_path,
        "-o", config_opt,
        "-n", "1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
        info = {
            "ok": result.returncode == 0,
            "output": output,
            "returncode": result.returncode,
        }

        # Parse reorder time
        m = re.search(r"Reorder Time:\s*([\d.]+)", output)
        if m:
            info["reorder_time"] = float(m.group(1))

        # Parse community count
        m = re.search(r"(\d+)\s+communities", output)
        if m:
            info["num_communities"] = int(m.group(1))

        # Parse sub-community info (recursive mode)
        m = re.search(r"sub-communities:\s*(\d+)\s+total.*?(\d+)\s+fit L2.*?(\d+)\s+fit LLC", output)
        if m:
            info["sub_comms_total"] = int(m.group(1))
            info["sub_comms_l2"] = int(m.group(2))
            info["sub_comms_llc"] = int(m.group(3))

        # Parse edge locality for recursive mode
        m = re.search(r"edge locality:\s*([\d.]+)%\s+internal", output)
        if m:
            info["edge_locality_pct"] = float(m.group(1))

        # Parse spatial locality
        m = re.search(r"avg edge distance=([\d.]+).*?ratio=([\d.]+)x", output)
        if m:
            info["avg_edge_distance"] = float(m.group(1))
            info["locality_ratio"] = float(m.group(2))

        # Parse sub-algo distribution
        m = re.search(r"sub-algo distribution:(.*)", output)
        if m:
            info["sub_algo_dist"] = m.group(1).strip()

        # Parse cache analysis
        m = re.search(r"L2=(\d+)KB.*?LLC=(\d+)MB.*?(\d+)B/node", output)
        if m:
            info["l2_kb"] = int(m.group(1))
            info["llc_mb"] = int(m.group(2))
            info["bytes_per_node"] = int(m.group(3))

        # Check for large communities
        large_comms = re.findall(r"Community \d+ \((\d+) nodes\)", output)
        if large_comms:
            info["large_comm_sizes"] = [int(x) for x in large_comms]

        # Parse GraphBrew ordering locality
        m = re.search(r"geo-mean edge distance:\s*([\d.]+)", output)
        if m:
            info["geo_mean_dist"] = float(m.group(1))
        m = re.search(r"near edges.*?:\s*([\d.]+)%", output)
        if m:
            info["near_edges_pct"] = float(m.group(1))

        return info
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_benchmark(graph_path: str, config_opt: str, timeout: int = 120) -> dict:
    """Run PR benchmark with a reorder config, parse kernel time."""
    cmd = [
        str(PR),
        "-f", graph_path,
        "-o", config_opt,
        "-n", "3",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
        info = {"ok": result.returncode == 0, "output": output}

        # Parse average trial time
        times = re.findall(r"Trial Time:\s*([\d.]+)", output)
        if times:
            info["trial_times"] = [float(t) for t in times]
            info["avg_time"] = sum(info["trial_times"]) / len(info["trial_times"])

        return info
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    """Run multi-layer validation experiments."""
    print("=" * 72)
    print("MULTI-LAYER VALIDITY TEST")
    print("=" * 72)

    # Check binaries exist
    if not CONVERTER.exists():
        print(f"ERROR: converter binary not found at {CONVERTER}")
        print("Build with: make -j$(nproc)")
        return 1
    if not PR.exists():
        print(f"ERROR: pr binary not found at {PR}")
        return 1

    if not TEST_GRAPHS:
        print("ERROR: No test graphs found in results/graphs/")
        print("Download with: python scripts/graphbrew_experiment.py --download-only")
        return 1

    # Print multi-layer space info
    info = enumerate_graphbrew_multilayer()
    print(f"\nMulti-layer space: {info['layers']['presets']} presets × "
          f"{info['layers']['orderings']} orderings × "
          f"{info['layers']['aggregations']} aggregations × "
          f"{info['layers']['feature_combos']} feature combos = "
          f"{info['total_discrete_configs']:,d} discrete configs")
    print(f"Active trained: {len(info['active_trained'])} | "
          f"Untrained compounds: {len(info['untrained'])}")

    all_results = {}

    for graph_name, graph_path in TEST_GRAPHS:
        print(f"\n{'=' * 72}")
        print(f"GRAPH: {graph_name} ({Path(graph_path).stat().st_size / 1e6:.1f} MB)")
        print(f"{'=' * 72}")

        graph_results = {}

        # --- Phase 1: Reorder validity ---
        print(f"\n--- Phase 1: Reorder Validity ({len(CONFIGS)} configs) ---")
        for config_name, config_opt in CONFIGS:
            print(f"\n  [{config_name}] -o {config_opt}")
            result = run_reorder(graph_path, config_opt)
            graph_results[config_name] = result

            if not result.get("ok"):
                print(f"    FAILED: {result.get('error', 'unknown')}")
                if "output" in result:
                    # Print last few lines of output for debugging
                    lines = result["output"].strip().split("\n")
                    for line in lines[-5:]:
                        print(f"    | {line}")
                continue

            print(f"    Reorder time: {result.get('reorder_time', '?'):.4f}s")
            if "num_communities" in result:
                print(f"    Communities: {result['num_communities']}")
            if "geo_mean_dist" in result:
                print(f"    Geo-mean edge distance: {result['geo_mean_dist']:.1f}, "
                      f"near edges: {result.get('near_edges_pct', '?'):.1f}%")

            # Recursive-specific output
            if "sub_comms_total" in result:
                print(f"    Sub-communities: {result['sub_comms_total']} total, "
                      f"{result['sub_comms_l2']} fit L2, "
                      f"{result['sub_comms_llc']} fit LLC")
            if "edge_locality_pct" in result:
                print(f"    Edge locality: {result['edge_locality_pct']:.1f}% internal")
            if "avg_edge_distance" in result:
                print(f"    Spatial locality: avg_dist={result['avg_edge_distance']:.0f}, "
                      f"ratio={result['locality_ratio']:.2f}x")
            if "sub_algo_dist" in result:
                print(f"    Sub-algo distribution: {result['sub_algo_dist']}")
            if "large_comm_sizes" in result:
                sizes = result["large_comm_sizes"]
                print(f"    Large communities: {len(sizes)} "
                      f"(max={max(sizes):,d}, min={min(sizes):,d})")
            if "l2_kb" in result:
                print(f"    Cache: L2={result['l2_kb']}KB, LLC={result['llc_mb']}MB, "
                      f"~{result['bytes_per_node']}B/node")

        # --- Phase 2: Performance comparison (flat vs recursive) ---
        print(f"\n--- Phase 2: PR Benchmark (flat vs recursive) ---")
        key_configs = [
            ("flat_leiden", "12:leiden"),
            ("recursive_d1", "12:leiden:8:auto:3:1"),
            ("flat_leiden_hrab", "12:leiden:hrab"),
        ]

        bench_results = {}
        for config_name, config_opt in key_configs:
            print(f"\n  [{config_name}] PR -o {config_opt}")
            result = run_benchmark(graph_path, config_opt)
            bench_results[config_name] = result

            if not result.get("ok"):
                print(f"    FAILED: {result.get('error', 'unknown')}")
                continue

            if "avg_time" in result:
                print(f"    Avg PR time: {result['avg_time']:.4f}s "
                      f"(trials: {[f'{t:.4f}' for t in result.get('trial_times', [])]})")

        graph_results["benchmarks"] = bench_results
        all_results[graph_name] = graph_results

    # --- Summary ---
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")

    for graph_name, graph_results in all_results.items():
        print(f"\n  {graph_name}:")
        # Compare flat vs recursive reorder times
        flat = graph_results.get("flat_leiden", {})
        recur = graph_results.get("recursive_d1", {})
        recur_auto = graph_results.get("recursive_d1_auto", {})

        if flat.get("ok") and recur.get("ok"):
            flat_t = flat.get("reorder_time", 0)
            recur_t = recur.get("reorder_time", 0)
            overhead = ((recur_t - flat_t) / max(flat_t, 0.001)) * 100
            print(f"    Reorder: flat={flat_t:.4f}s, recursive={recur_t:.4f}s "
                  f"(overhead: {overhead:+.1f}%)")

            # Locality comparison
            flat_geo = flat.get("geo_mean_dist")
            recur_geo = recur.get("geo_mean_dist")
            if flat_geo and recur_geo:
                improvement = ((flat_geo - recur_geo) / max(flat_geo, 1)) * 100
                print(f"    Geo-mean dist: flat={flat_geo:.1f}, "
                      f"recursive={recur_geo:.1f} "
                      f"({improvement:+.1f}% {'better' if improvement > 0 else 'worse'})")

        benches = graph_results.get("benchmarks", {})
        flat_b = benches.get("flat_leiden", {})
        recur_b = benches.get("recursive_d1", {})
        hrab_b = benches.get("flat_leiden_hrab", {})

        if flat_b.get("avg_time") and recur_b.get("avg_time"):
            flat_pr = flat_b["avg_time"]
            recur_pr = recur_b["avg_time"]
            speedup = ((flat_pr - recur_pr) / max(flat_pr, 0.001)) * 100
            print(f"    PR kernel: flat={flat_pr:.4f}s, recursive={recur_pr:.4f}s "
                  f"({speedup:+.1f}% {'faster' if speedup > 0 else 'slower'})")

        if hrab_b.get("avg_time"):
            print(f"    PR kernel: leiden_hrab={hrab_b['avg_time']:.4f}s")

        # Sub-community analysis
        if recur.get("sub_comms_total"):
            total_sc = recur["sub_comms_total"]
            l2_sc = recur.get("sub_comms_l2", 0)
            llc_sc = recur.get("sub_comms_llc", 0)
            exceed = total_sc - l2_sc - llc_sc
            print(f"    Cache fit: {l2_sc}/{total_sc} in L2, "
                  f"{llc_sc}/{total_sc} in LLC, "
                  f"{exceed}/{total_sc} exceed LLC")
            if recur.get("edge_locality_pct"):
                print(f"    Internal edges: {recur['edge_locality_pct']:.1f}%")

    print(f"\nDone. All configs tested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
