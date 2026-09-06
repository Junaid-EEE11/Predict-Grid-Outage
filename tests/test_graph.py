import pytest
import networkx as nx
import numpy as np
from grid_outage_research.geography.county_graph import CountyGraphBuilder

def test_county_graph_construction(sample_region_config):
    builder = CountyGraphBuilder(sample_region_config)
    G = builder.build_graph_from_coordinates(max_distance_km=150.0)
    assert G.number_of_nodes() == len(sample_region_config["counties"])
    assert G.number_of_edges() > 0
    # No isolated nodes
    for node in G.nodes():
        assert G.degree(node) > 0

    # Adjacency matrix
    A_norm = builder.get_adjacency_matrix(G, normalize=True)
    assert A_norm.shape == (len(sample_region_config["counties"]), len(sample_region_config["counties"]))
    # Symmetric
    assert np.allclose(A_norm, A_norm.T)

def test_permuted_graph_ablation(sample_region_config):
    builder = CountyGraphBuilder(sample_region_config)
    G = builder.build_graph_from_coordinates()
    G_perm = builder.get_permuted_graph(G, seed=42)
    assert G_perm.number_of_nodes() == G.number_of_nodes()
    assert G_perm.number_of_edges() == G.number_of_edges()
