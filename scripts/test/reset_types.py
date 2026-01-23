#!/usr/bin/env python3
"""
Reset and recreate type clusters with lower threshold.
"""

import json
import os
import sys
import math
import shutil
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

WEIGHTS_DIR = Path(__file__).parent / "weights"
GRAPH_DATASETS = Path("/home/ab/Documents/00_github_repos/00_GraphDataSets")

# New threshold - create more distinct clusters
CLUSTER_DISTANCE_THRESHOLD = 0.15

# Diverse graphs with their approximate features
# These are computed from the actual edge lists
GRAPH_FEATURES = {
    "soc-LiveJournal1": {
        "nodes": 4847571, "edges": 68993773,
        "avg_degree": 28.47, "hub_concentration": 0.177,
        "modularity": 0.75, "degree_variance": 0.8,
    },
    "com-Youtube": {
        "nodes": 1134890, "edges": 2987624,
        "avg_degree": 5.27, "hub_concentration": 0.365,
        "modularity": 0.7, "degree_variance": 0.6,
    },
    "roadNet-CA": {
        "nodes": 1965206, "edges": 5533214,
        "avg_degree": 5.63, "hub_concentration": 0.017,
        "modularity": 0.3, "degree_variance": 0.1,  # Grid-like, uniform
    },
    "roadNet-TX": {
        "nodes": 1379917, "edges": 3843320,
        "avg_degree": 5.57, "hub_concentration": 0.016,
        "modularity": 0.3, "degree_variance": 0.1,
    },
    "web-Google": {
        "nodes": 875713, "edges": 5105039,
        "avg_degree": 11.66, "hub_concentration": 0.158,
        "modularity": 0.65, "degree_variance": 0.75,
    },
    "web-BerkStan": {
        "nodes": 685230, "edges": 7600595,
        "avg_degree": 22.18, "hub_concentration": 0.301,
        "modularity": 0.6, "degree_variance": 0.85,
    },
    "cit-Patents": {
        "nodes": 3774768, "edges": 16518948,
        "avg_degree": 8.75, "hub_concentration": 0.079,
        "modularity": 0.5, "degree_variance": 0.5,
    },
    "com-DBLP": {
        "nodes": 317080, "edges": 1049866,
        "avg_degree": 6.62, "hub_concentration": 0.113,
        "modularity": 0.8, "degree_variance": 0.4,
    },
    "com-Amazon": {
        "nodes": 334863, "edges": 925872,
        "avg_degree": 5.53, "hub_concentration": 0.08,
        "modularity": 0.85, "degree_variance": 0.35,
    },
}


def normalize_features(features):
    """Normalize features to [0,1] range."""
    ranges = {
        'modularity': (0, 1),
        'degree_variance': (0, 1),
        'hub_concentration': (0, 1),
        'avg_degree': (0, 100),
        'clustering_coefficient': (0, 1),
        'log_nodes': (3, 10),
        'log_edges': (3, 12),
    }
    
    log_nodes = math.log10(features.get('nodes', 1000) + 1)
    log_edges = math.log10(features.get('edges', 1000) + 1)
    
    normalized = []
    for key, (lo, hi) in ranges.items():
        if key == 'log_nodes':
            val = log_nodes
        elif key == 'log_edges':
            val = log_edges
        else:
            val = features.get(key, (lo + hi) / 2)
        normalized.append(max(0, min(1, (val - lo) / (hi - lo) if hi > lo else 0.5)))
    
    return normalized


def compute_distance(f1, f2):
    """Euclidean distance."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))


def main():
    print("="*60)
    print("Resetting Type Clusters with Lower Threshold")
    print("="*60)
    print(f"Threshold: {CLUSTER_DISTANCE_THRESHOLD}")
    print(f"Weights dir: {WEIGHTS_DIR}")
    
    # Backup existing
    backup_dir = WEIGHTS_DIR / "backup_before_reset"
    if WEIGHTS_DIR.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in WEIGHTS_DIR.glob("*.json"):
            if "backup" not in str(f):
                shutil.copy(f, backup_dir / f.name)
        print(f"\nBacked up existing weights to {backup_dir}")
    
    # Start fresh registry
    registry = {}
    type_counter = 0
    
    # Process graphs in order
    graph_assignments = {}
    
    print("\nAssigning graphs to types:")
    print("-" * 50)
    
    for graph_name, features in GRAPH_FEATURES.items():
        norm_feat = normalize_features(features)
        
        # Find closest type
        min_dist = float('inf')
        closest_type = None
        
        for type_name, type_info in registry.items():
            dist = compute_distance(norm_feat, type_info['centroid'])
            if dist < min_dist:
                min_dist = dist
                closest_type = type_name
        
        if closest_type and min_dist < CLUSTER_DISTANCE_THRESHOLD:
            # Join existing cluster
            type_info = registry[closest_type]
            count = type_info['sample_count']
            old_centroid = type_info['centroid']
            new_centroid = [
                old + (new - old) / (count + 1)
                for old, new in zip(old_centroid, norm_feat)
            ]
            registry[closest_type]['centroid'] = new_centroid
            registry[closest_type]['sample_count'] = count + 1
            registry[closest_type]['graphs'].append(graph_name)
            assigned_type = closest_type
            print(f"  {graph_name} -> {closest_type} (dist={min_dist:.4f}, joined)")
        else:
            # Create new cluster
            new_type = f"type_{type_counter}"
            type_counter += 1
            registry[new_type] = {
                'centroid': norm_feat,
                'sample_count': 1,
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'representative_features': {
                    'modularity': features.get('modularity', 0.5),
                    'degree_variance': features.get('degree_variance', 0.5),
                    'hub_concentration': features.get('hub_concentration', 0.3),
                    'avg_degree': features.get('avg_degree', 10),
                },
                'graphs': [graph_name]
            }
            assigned_type = new_type
            if closest_type:
                print(f"  {graph_name} -> {new_type} (dist={min_dist:.4f} > {CLUSTER_DISTANCE_THRESHOLD}, new cluster)")
            else:
                print(f"  {graph_name} -> {new_type} (first cluster)")
        
        graph_assignments[graph_name] = assigned_type
    
    # Clean up 'graphs' key for saving (not needed in registry)
    registry_to_save = {}
    for type_name, type_info in registry.items():
        info_copy = dict(type_info)
        del info_copy['graphs']
        registry_to_save[type_name] = info_copy
    
    # Save new registry
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_file = WEIGHTS_DIR / "type_registry.json"
    with open(registry_file, 'w') as f:
        json.dump(registry_to_save, f, indent=2)
    print(f"\nSaved registry to {registry_file}")
    
    # Create weight files for new types (copy from backup or create defaults)
    backup_type0 = backup_dir / "type_0.json"
    default_weights = None
    if backup_type0.exists():
        with open(backup_type0) as f:
            default_weights = json.load(f)
    
    for type_name in registry_to_save.keys():
        weights_file = WEIGHTS_DIR / f"{type_name}.json"
        if not weights_file.exists() and default_weights:
            with open(weights_file, 'w') as f:
                json.dump(default_weights, f, indent=2)
            print(f"  Created {weights_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Created {len(registry)} type clusters:")
    
    for type_name, type_info in registry.items():
        graphs = type_info['graphs']
        rep = type_info['representative_features']
        print(f"\n  {type_name}: {len(graphs)} graphs")
        print(f"    Graphs: {graphs}")
        print(f"    Features: mod={rep['modularity']:.2f}, var={rep['degree_variance']:.2f}, "
              f"hub={rep['hub_concentration']:.3f}")
    
    return len(registry)


if __name__ == "__main__":
    num = main()
    print(f"\n✓ Generated {num} type clusters")
