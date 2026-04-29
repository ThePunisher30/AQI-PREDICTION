"""Graph Builder — Construct haversine-based adjacency matrix for city graph."""

import os, json
import numpy as np
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

LOCATIONS = CFG["locations"]
THRESHOLD_KM = CFG["graph"]["distance_threshold_km"]
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", CFG["paths"]["graph_data"])


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_adjacency_matrix(city_list=None):
    """Build weighted adjacency matrix using inverse haversine distance."""
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Use cities from data if available, otherwise from config
    if city_list is None:
        meta_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            city_list = meta["city_names"]
        else:
            city_list = sorted(LOCATIONS.keys())

    cities = sorted(city_list)
    n = len(cities)
    coords = [LOCATIONS[c] for c in cities]

    # Distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])

    # Adjacency: inverse distance, thresholded
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] <= THRESHOLD_KM:
                adj[i, j] = 1.0 / dist[i, j]

    # Add self-loops
    adj_with_self = adj + np.eye(n)

    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    D = np.diag(adj_with_self.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    adj_norm = D_inv_sqrt @ adj_with_self @ D_inv_sqrt

    np.save(os.path.join(GRAPH_DIR, "adj_matrix.npy"), adj_norm.astype(np.float32))
    np.save(os.path.join(GRAPH_DIR, "distance_matrix.npy"), dist.astype(np.float32))

    print(f"Graph built: {n} cities, threshold={THRESHOLD_KM}km")
    print(f"Cities: {cities}")
    print(f"Adjacency matrix shape: {adj_norm.shape}")
    print(f"Non-zero edges (excl self-loops): {(adj > 0).sum()}")
    return adj_norm, cities


if __name__ == "__main__":
    build_adjacency_matrix()
    print("\n✅ Graph construction complete.")
