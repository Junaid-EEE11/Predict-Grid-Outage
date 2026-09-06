import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

class CountyGraphBuilder:
    """Constructs, analyzes, and serializes spatial county adjacency graphs."""

    def __init__(self, region_config: Dict[str, Any]):
        self.region_config = region_config
        self.counties = region_config.get("counties", [])
        self.fips_list = [c["fips"] for c in self.counties]
        self.fips_to_idx = {fips: i for i, fips in enumerate(self.fips_list)}

    def build_graph_from_coordinates(self, max_distance_km: float = 120.0) -> nx.Graph:
        """Build distance-based / k-nearest / contiguity graph from county centroids."""
        G = nx.Graph()
        for c in self.counties:
            G.add_node(
                c["fips"],
                name=c["name"],
                state=c["state"],
                lat=c["lat"],
                lon=c["lon"],
                modeled_customers=c.get("modeled_customers", 50000)
            )

        # Connect counties within geographical threshold
        for i, c1 in enumerate(self.counties):
            for j, c2 in enumerate(self.counties):
                if i < j:
                    # Haversine distance approximation
                    dlat = np.radians(c2["lat"] - c1["lat"])
                    dlon = np.radians(c2["lon"] - c1["lon"])
                    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(c1["lat"])) * np.cos(np.radians(c2["lat"])) * np.sin(dlon/2.0)**2
                    c = 2.0 * np.arcsin(np.sqrt(a))
                    dist_km = 6371.0 * c
                    
                    # Connect if distance <= max_distance_km or ensure at least 2 nearest neighbors
                    if dist_km <= max_distance_km:
                        weight = float(np.exp(-dist_km / 50.0))
                        G.add_edge(c1["fips"], c2["fips"], weight=weight, distance_km=dist_km)

        # Ensure no isolated nodes
        for c1 in self.counties:
            f1 = c1["fips"]
            if G.degree(f1) == 0:
                # Connect to nearest neighbor
                distances = []
                for c2 in self.counties:
                    if c2["fips"] != f1:
                        dist = np.hypot(c2["lat"] - c1["lat"], c2["lon"] - c1["lon"])
                        distances.append((dist, c2["fips"]))
                distances.sort()
                G.add_edge(f1, distances[0][1], weight=0.5, distance_km=50.0)

        logger.info(f"County graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G

    def get_adjacency_matrix(self, G: nx.Graph, normalize: bool = True) -> np.ndarray:
        """Extract dense or sparse normalized adjacency matrix with self-loops."""
        N = len(self.fips_list)
        A = np.zeros((N, N), dtype=np.float32)

        for u, v, data in G.edges(data=True):
            if u in self.fips_to_idx and v in self.fips_to_idx:
                i, j = self.fips_to_idx[u], self.fips_to_idx[v]
                w = data.get("weight", 1.0)
                A[i, j] = w
                A[j, i] = w

        # Add self-loops
        np.fill_diagonal(A, 1.0)

        if normalize:
            # Symmetric degree normalization: D^{-1/2} A D^{-1/2}
            deg = np.sum(A, axis=1)
            deg_inv_sqrt = np.zeros_like(deg)
            mask = deg > 0
            deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
            D_inv_sqrt = np.diag(deg_inv_sqrt)
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt
            return A_norm

        return A

    def get_permuted_graph(self, G: nx.Graph, seed: int = 42) -> nx.Graph:
        """Ablation A11: Randomly permute node associations while preserving degree distribution."""
        rng = np.random.default_rng(seed)
        G_perm = nx.Graph()
        G_perm.add_nodes_from(G.nodes(data=True))
        
        nodes = list(G.nodes())
        perm_nodes = rng.permutation(nodes)
        mapping = dict(zip(nodes, perm_nodes))

        for u, v, data in G.edges(data=True):
            G_perm.add_edge(mapping[u], mapping[v], **data)

        return G_perm
