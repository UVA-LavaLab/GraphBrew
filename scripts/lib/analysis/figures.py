#!/usr/bin/env python3
"""
Generate reordering algorithm visualization figures for GraphBrew wiki.

Creates SVG figures showing:
1. Adjacency matrix spy plots (before/after reordering)
2. Algorithm pipeline diagrams
3. Algorithm category overview

Each figure shows how a specific reordering algorithm transforms
the graph's adjacency matrix pattern, visually demonstrating
cache locality improvements.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Output directory
OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

# Color palette
COLORS = {
    'original': '#6c757d',
    'hub': '#e63946',
    'community': '#457b9d',
    'locality': '#2a9d8f',
    'hybrid': '#e9c46a',
    'adaptive': '#f4a261',
    'edge': '#264653',
    'bg_light': '#f8f9fa',
    'grid': '#dee2e6',
}


def generate_sample_graph(n=40, graph_type='social'):
    """Generate a sample graph adjacency matrix with community structure."""
    np.random.seed(42)
    adj = np.zeros((n, n), dtype=int)

    if graph_type == 'social':
        # 4 communities with hub structure
        communities = [range(0, 10), range(10, 20), range(20, 30), range(30, 40)]
        # Dense intra-community edges
        for comm in communities:
            for i in comm:
                for j in comm:
                    if i != j and np.random.random() < 0.4:
                        adj[i][j] = 1
                        adj[j][i] = 1
        # Sparse inter-community edges
        for c1 in range(len(communities)):
            for c2 in range(c1 + 1, len(communities)):
                for _ in range(3):
                    i = np.random.choice(list(communities[c1]))
                    j = np.random.choice(list(communities[c2]))
                    adj[i][j] = 1
                    adj[j][i] = 1
        # Add hubs (high-degree vertices)
        hubs = [2, 13, 25, 35]
        for h in hubs:
            for v in range(n):
                if v != h and np.random.random() < 0.25:
                    adj[h][v] = 1
                    adj[v][h] = 1
    return adj


def scramble_matrix(adj):
    """Randomly permute the adjacency matrix to simulate unordered input."""
    n = adj.shape[0]
    perm = np.random.permutation(n)
    scrambled = adj[np.ix_(perm, perm)]
    return scrambled, perm


def reorder_hubsort(adj):
    """Reorder by degree (hubs first)."""
    degrees = adj.sum(axis=1)
    perm = np.argsort(-degrees)
    return adj[np.ix_(perm, perm)], perm


def reorder_hubcluster(adj):
    """Reorder hubs together with their neighbors."""
    n = adj.shape[0]
    degrees = adj.sum(axis=1)
    hub_threshold = np.percentile(degrees, 80)
    hubs = np.where(degrees >= hub_threshold)[0]
    non_hubs = np.where(degrees < hub_threshold)[0]
    perm = np.concatenate([hubs, non_hubs])
    return adj[np.ix_(perm, perm)], perm


def reorder_community(adj, n_communities=4):
    """Reorder by community detection (simulated Leiden)."""
    n = adj.shape[0]
    # Simple greedy community detection
    comm = np.zeros(n, dtype=int)
    comm_size = n // n_communities
    # Use adjacency clustering
    visited = set()
    current_comm = 0
    order = []

    for start in range(n):
        if start in visited:
            continue
        # BFS within community
        queue = [start]
        count = 0
        while queue and count < comm_size:
            v = queue.pop(0)
            if v in visited:
                continue
            visited.add(v)
            comm[v] = current_comm
            order.append(v)
            count += 1
            neighbors = np.where(adj[v] > 0)[0]
            for nb in neighbors:
                if nb not in visited:
                    queue.append(nb)
        current_comm += 1

    # Add any remaining
    for v in range(n):
        if v not in visited:
            order.append(v)

    perm = np.array(order)
    return adj[np.ix_(perm, perm)], perm


def reorder_graphbrew(adj):
    """Simulate GraphBrew: community detection + per-community RabbitOrder."""
    n = adj.shape[0]
    # First: community detection
    _, comm_perm = reorder_community(adj)
    adj_comm = adj[np.ix_(comm_perm, comm_perm)]

    # Then: within each community, sort by degree (simulating RabbitOrder)
    degrees = adj_comm.sum(axis=1)
    n_comm = 4
    comm_size = n // n_comm
    final_perm = []
    for c in range(n_comm):
        start = c * comm_size
        end = min(start + comm_size, n)
        comm_indices = list(range(start, end))
        # Sort within community by degree descending
        comm_indices.sort(key=lambda x: -degrees[x])
        final_perm.extend(comm_indices)
    # Remaining vertices
    for v in range(n):
        if v not in final_perm:
            final_perm.append(v)

    final_perm = np.array(final_perm)
    return adj_comm[np.ix_(final_perm, final_perm)], final_perm


def reorder_rcm(adj):
    """Reverse Cuthill-McKee for bandwidth reduction."""
    n = adj.shape[0]
    degrees = adj.sum(axis=1)
    # Start from minimum degree vertex
    start = np.argmin(degrees)
    visited = set()
    order = []
    queue = [start]

    while queue:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)
        order.append(v)
        neighbors = np.where(adj[v] > 0)[0]
        # Sort neighbors by degree (ascending)
        neighbors = sorted(neighbors, key=lambda x: degrees[x])
        for nb in neighbors:
            if nb not in visited:
                queue.append(nb)

    for v in range(n):
        if v not in visited:
            order.append(v)

    perm = np.array(order[::-1])  # Reverse
    return adj[np.ix_(perm, perm)], perm


def plot_spy_comparison(adj_before, adj_after, title_before, title_after,
                        filename, accent_color='#457b9d', subtitle=''):
    """Plot side-by-side spy plots showing before/after reordering."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Before
    ax1.spy(adj_before, markersize=2.5, color=COLORS['original'], alpha=0.7)
    ax1.set_title(title_before, fontsize=12, color=COLORS['original'])
    ax1.set_xlabel('Column (vertex ID)', fontsize=9)
    ax1.set_ylabel('Row (vertex ID)', fontsize=9)
    ax1.tick_params(labelsize=8)

    # After
    ax2.spy(adj_after, markersize=2.5, color=accent_color, alpha=0.7)
    ax2.set_title(title_after, fontsize=12, color=accent_color)
    ax2.set_xlabel('Column (vertex ID)', fontsize=9)
    ax2.set_ylabel('Row (vertex ID)', fontsize=9)
    ax2.tick_params(labelsize=8)

    # Bandwidth annotation
    n = adj_before.shape[0]
    bw_before = compute_bandwidth(adj_before)
    bw_after = compute_bandwidth(adj_after)
    nnz = int(adj_before.sum())

    # Add metrics as text below
    fig.text(0.27, 0.01, f'Bandwidth: {bw_before}', ha='center', fontsize=9, color=COLORS['original'])
    fig.text(0.73, 0.01, f'Bandwidth: {bw_after}  ({bw_after/bw_before:.0%})',
             ha='center', fontsize=9, color=accent_color)

    if subtitle:
        fig.suptitle(subtitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    filepath = OUT_DIR / filename
    plt.savefig(filepath, format='svg', bbox_inches='tight', pad_inches=0.15)
    plt.savefig(filepath.with_suffix('.png'), format='png', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  ✓ {filepath.name}")


def compute_bandwidth(adj):
    """Compute matrix bandwidth (max |i-j| for non-zero entries)."""
    rows, cols = np.where(adj > 0)
    if len(rows) == 0:
        return 0
    return int(np.max(np.abs(rows - cols)))


def plot_algorithm_overview(filename='reorder_overview.svg'):
    """Create algorithm category overview figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    categories = [
        {'name': 'Basic (0–2)', 'x': 0.5, 'y': 5.5, 'w': 2.5, 'h': 1.2,
         'color': '#adb5bd', 'algos': 'ORIGINAL\nRANDOM\nSORT'},
        {'name': 'Hub-Based (3–4)', 'x': 3.5, 'y': 5.5, 'w': 2.5, 'h': 1.2,
         'color': COLORS['hub'], 'algos': 'HUBSORT\nHUBCLUSTER'},
        {'name': 'DBG-Based (5–7)', 'x': 6.5, 'y': 5.5, 'w': 2.5, 'h': 1.2,
         'color': '#e76f51', 'algos': 'DBG\nHUBSORTDBG\nHUBCLUSTERDBG'},
        {'name': 'Community (8)', 'x': 0.5, 'y': 3.5, 'w': 2.5, 'h': 1.2,
         'color': COLORS['community'], 'algos': 'RABBITORDER'},
        {'name': 'Classic (9–11)', 'x': 3.5, 'y': 3.5, 'w': 2.5, 'h': 1.2,
         'color': COLORS['locality'], 'algos': 'GORDER\nCORDER\nRCM'},
        {'name': 'GraphBrew (12)', 'x': 6.5, 'y': 3.5, 'w': 2.5, 'h': 1.2,
         'color': COLORS['hybrid'], 'algos': 'GraphBrewOrder\n(Leiden+RabbitOrder)'},
        {'name': 'Adaptive (14)', 'x': 1.5, 'y': 1.5, 'w': 3.0, 'h': 1.2,
         'color': COLORS['adaptive'], 'algos': 'AdaptiveOrder\n(ML perceptron)'},
        {'name': 'Leiden (15)', 'x': 5.5, 'y': 1.5, 'w': 3.0, 'h': 1.2,
         'color': '#a8dadc', 'algos': 'LeidenOrder (15)'},
    ]

    for cat in categories:
        rect = mpatches.FancyBboxPatch(
            (cat['x'], cat['y']), cat['w'], cat['h'],
            boxstyle='round,pad=0.1', facecolor=cat['color'],
            edgecolor='#264653', linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(cat['x'] + cat['w']/2, cat['y'] + cat['h'] - 0.2,
                cat['name'], ha='center', va='top',
                fontweight='bold', fontsize=10, color='white')
        ax.text(cat['x'] + cat['w']/2, cat['y'] + 0.15,
                cat['algos'], ha='center', va='bottom',
                fontsize=8, color='white', linespacing=1.3)

    # Arrow from categories to "Better Cache Performance"
    ax.annotate('', xy=(5.5, 0.7), xytext=(1.5, 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['edge']))
    ax.text(3.5, 0.55, 'More Sophisticated → Better Cache Locality',
            ha='center', fontsize=10, style='italic', color=COLORS['edge'])

    fig.suptitle('GraphBrew: Algorithm Categories', fontsize=16, fontweight='bold', y=0.98)
    filepath = OUT_DIR / filename
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filepath.name}")


def plot_graphbrew_pipeline(filename='graphbrew_pipeline.svg'):
    """Create GraphBrew pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    steps = [
        {'name': '1. Input\nGraph', 'x': 0.3, 'color': '#adb5bd'},
        {'name': '2. Topology\nAnalysis', 'x': 2.3, 'color': COLORS['locality']},
        {'name': '3. Leiden\nCommunity\nDetection', 'x': 4.5, 'color': COLORS['community']},
        {'name': '4. Community\nClassification\n(small/large)', 'x': 6.9, 'color': COLORS['adaptive']},
        {'name': '5. Per-Community\nReordering\n(RabbitOrder)', 'x': 9.3, 'color': COLORS['hub']},
        {'name': '6. Final\nPermutation', 'x': 11.7, 'color': COLORS['locality']},
    ]

    bw = 1.8
    bh = 2.2
    for step in steps:
        rect = mpatches.FancyBboxPatch(
            (step['x'], 0.6), bw, bh,
            boxstyle='round,pad=0.12', facecolor=step['color'],
            edgecolor=COLORS['edge'], linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(step['x'] + bw/2, 0.6 + bh/2, step['name'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', linespacing=1.3)

    # Arrows between steps
    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i+1]['x'] - 0.05, 1.7),
                    xytext=(steps[i]['x'] + bw + 0.05, 1.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['edge']))

    fig.suptitle('GraphBrewOrder (Algorithm 12) Pipeline',
                 fontsize=14, fontweight='bold', y=0.98)
    filepath = OUT_DIR / filename
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filepath.name}")


def plot_adaptive_pipeline(filename='adaptive_pipeline.svg'):
    """Create AdaptiveOrder pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    # Main pipeline
    steps = [
        {'name': '1. Extract\nGraph\nFeatures', 'x': 0.3, 'y': 2.0, 'color': COLORS['locality']},
        {'name': '2. ML\nPerceptron\nScore', 'x': 3.0, 'y': 2.0, 'color': COLORS['adaptive']},
        {'name': '3. Select\nBest\nAlgorithm', 'x': 5.7, 'y': 2.0, 'color': COLORS['hub']},
        {'name': '4. Execute\nSelected\nAlgorithm', 'x': 8.4, 'y': 2.0, 'color': COLORS['community']},
        {'name': '5. Reordered\nGraph', 'x': 11.1, 'y': 2.0, 'color': COLORS['locality']},
    ]

    bw = 2.2
    bh = 2.0
    for step in steps:
        rect = mpatches.FancyBboxPatch(
            (step['x'], step['y']), bw, bh,
            boxstyle='round,pad=0.12', facecolor=step['color'],
            edgecolor=COLORS['edge'], linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(step['x'] + bw/2, step['y'] + bh/2, step['name'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', linespacing=1.3)

    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i+1]['x'] - 0.05, 3.0),
                    xytext=(steps[i]['x'] + bw + 0.05, 3.0),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['edge']))

    # Feature list
    features = 'Features: avg_degree, clustering_coeff, community_count,\nhub_concentration, degree_variance, packing_factor'
    ax.text(1.4, 0.8, features, fontsize=8, style='italic', color='#555',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))

    # Algorithm candidates
    candidates = 'Candidates: GraphBrewOrder (12), RabbitOrder (8),\nHubClusterDBG (7), Gorder (9), RCM (11), LeidenOrder (15)'
    ax.text(8.4 + bw/2, 0.8, candidates, fontsize=8, style='italic', color='#555',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))

    fig.suptitle('AdaptiveOrder (Algorithm 14) Pipeline — ML-Guided Selection',
                 fontsize=14, fontweight='bold', y=0.98)
    filepath = OUT_DIR / filename
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.png'), format='png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filepath.name}")


def main():
    print("Generating GraphBrew reorder visualization figures...")
    print(f"Output: {OUT_DIR}/\n")

    # Generate sample graph and scramble it
    adj_original = generate_sample_graph(n=40, graph_type='social')
    adj_scrambled, scramble_perm = scramble_matrix(adj_original)

    # 1. Algorithm Overview
    plot_algorithm_overview()

    # 2. Pipeline diagrams
    plot_graphbrew_pipeline()
    plot_adaptive_pipeline()

    # 3. Before/After spy plots for each algorithm category
    print("\nAdjacency matrix transformations:")

    # Original (scrambled) — the "before"
    adj_input = adj_scrambled

    # HubSort
    adj_hub, _ = reorder_hubsort(adj_input)
    plot_spy_comparison(adj_input, adj_hub,
                        'Before (Original Order)', 'After HubSort',
                        'reorder_hubsort.svg', COLORS['hub'],
                        'HubSort: High-degree vertices placed first')

    # HubCluster
    adj_hc, _ = reorder_hubcluster(adj_input)
    plot_spy_comparison(adj_input, adj_hc,
                        'Before (Original Order)', 'After HubCluster',
                        'reorder_hubcluster.svg', COLORS['hub'],
                        'HubCluster: Hubs grouped with their neighbors')

    # Community (Leiden-like)
    adj_comm, _ = reorder_community(adj_input)
    plot_spy_comparison(adj_input, adj_comm,
                        'Before (Original Order)', 'After Community Detection',
                        'reorder_community.svg', COLORS['community'],
                        'Community Reordering: Vertices grouped by community')

    # GraphBrew (Community + per-community RabbitOrder)
    adj_gb, _ = reorder_graphbrew(adj_input)
    plot_spy_comparison(adj_input, adj_gb,
                        'Before (Original Order)', 'After GraphBrewOrder',
                        'reorder_graphbrew.svg', COLORS['hybrid'],
                        'GraphBrewOrder: Community detection + per-community reordering')

    # RCM
    adj_rcm, _ = reorder_rcm(adj_input)
    plot_spy_comparison(adj_input, adj_rcm,
                        'Before (Original Order)', 'After RCM',
                        'reorder_rcm.svg', COLORS['locality'],
                        'Reverse Cuthill-McKee: Minimize bandwidth')

    # Combined comparison: Original vs HubSort vs Community vs GraphBrew
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    matrices = [
        (adj_input, 'Original (Scrambled)', COLORS['original']),
        (adj_hub, 'HubSort', COLORS['hub']),
        (adj_comm, 'Community (Leiden)', COLORS['community']),
        (adj_gb, 'GraphBrewOrder', COLORS['hybrid']),
    ]
    for ax, (mat, title, color) in zip(axes, matrices):
        ax.spy(mat, markersize=2, color=color, alpha=0.7)
        ax.set_title(title, fontsize=11, color=color, fontweight='bold')
        bw = compute_bandwidth(mat)
        ax.set_xlabel(f'Bandwidth: {bw}', fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle('Reordering Algorithm Comparison — Adjacency Matrix Patterns',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = OUT_DIR / 'reorder_comparison.svg'
    plt.savefig(filepath, format='svg', bbox_inches='tight', pad_inches=0.15)
    plt.savefig(filepath.with_suffix('.png'), format='png', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("  ✓ reorder_comparison.svg")

    print(f"\nDone! Generated figures in {OUT_DIR}/")


if __name__ == '__main__':
    main()
