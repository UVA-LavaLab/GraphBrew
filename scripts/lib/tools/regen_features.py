#!/usr/bin/env python3
"""
Regenerate features.json for all .sg graph files by running the C++ binary
in analysis mode (-a 0). This ensures features match what C++ computes at
runtime (from directed .sg graphs) rather than from symmetrized .mtx files.

Usage:
    python3 scripts/graphbrew_experiment.py --regen-features
    python3 -m scripts.lib.regen_features
"""
import json, os, re, subprocess, sys, glob
from datetime import datetime

from ..core.utils import GRAPHS_DIR as _GRAPHS_DIR, BIN_DIR, RESULTS_DIR, Logger

GRAPHS_DIR = str(_GRAPHS_DIR)
BINARY = str(BIN_DIR / "pr")

log = Logger()

# Regex patterns matching C++ PrintTime output
PATTERNS = {
    "degree_variance":       r"Degree Variance[:\s]+([\d.]+)",
    "hub_concentration":     r"Hub Concentration[:\s]+([\d.]+)",
    "avg_degree":            r"Avg Degree[:\s]+([\d.]+)",
    "clustering_coefficient":r"Clustering Coefficient[:\s]+([\d.]+)",
    "avg_path_length":       r"Avg Path Length[:\s]+([\d.]+)",
    "diameter_estimate":     r"Diameter Estimate[:\s]+([\d.]+)",
    "community_count":       r"Community Count[:\s]+([\d.]+)",
    "packing_factor":        r"Packing Factor[:\s]+([\d.]+)",
    "forward_edge_fraction": r"Forward Edge Fraction[:\s]+([\d.]+)",
    "working_set_ratio":     r"Working Set Ratio[:\s]+([\d.]+)",
    "modularity":            r"[Mm]odularity[:\s]+([\d.]+)",
    "graph_density":         r"Graph Density[:\s]+([\d.]+e?[+-]?\d*)",
}
NODE_EDGE_PATTERNS = {
    "nodes": r"Nodes:\s+(\d+)",
    "edges": r"Edges:\s+(\d+)",
}

def parse_features(output: str) -> dict:
    features = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, output)
        if m:
            features[key] = float(m.group(1))
    for key, pattern in NODE_EDGE_PATTERNS.items():
        m = re.search(pattern, output)
        if m:
            features[key] = int(m.group(1))
    return features

def main():
    if not os.path.isfile(BINARY):
        log.error(f"Binary not found: {BINARY}"); sys.exit(1)

    from scripts.lib.core.datastore import get_props_store
    store = get_props_store()

    sg_files = sorted(glob.glob(os.path.join(GRAPHS_DIR, "*", "*.sg")))
    log.info(f"Found {len(sg_files)} .sg files")

    updated = 0
    for sg_path in sg_files:
        graph_dir = os.path.dirname(sg_path)
        graph_name = os.path.basename(graph_dir)

        log.info(f"\n{'='*60}")
        log.info(f"Processing: {graph_name}")
        log.info(f"  .sg file: {sg_path}")

        # Run C++ binary in analysis mode
        cmd = [BINARY, "-f", sg_path, "-a", "0", "-n", "1"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            log.warning("  TIMEOUT — skipping")
            continue
        except Exception as e:
            log.error(f"  ERROR: {e}")
            continue

        features = parse_features(output)
        if not features:
            log.warning("  No features parsed from output!")
            log.info(f"  stdout: {result.stdout[:200]}")
            continue

        # Show diff for key fields
        old_features = store.get(graph_name) or {}
        for key in ["degree_variance", "hub_concentration", "avg_degree", "packing_factor", "forward_edge_fraction", "working_set_ratio"]:
            old_val = old_features.get(key, "N/A")
            new_val = features.get(key, "N/A")
            changed = ""
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)) and abs(old_val - new_val) > 0.001:
                changed = " ← CHANGED"
            log.info(f"  {key:30s} old={old_val!s:>12s}  new={new_val!s:>12s}{changed}")

        # Merge into central store
        features["graph_name"] = graph_name
        features["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        store.update(graph_name, features)
        updated += 1

    # Persist once
    store.save()

    log.info(f"\n{'='*60}")
    log.info(f"Updated {updated}/{len(sg_files)} graphs")
    log.info(f"Store has {len(store.all())} entries")

if __name__ == "__main__":
    main()
