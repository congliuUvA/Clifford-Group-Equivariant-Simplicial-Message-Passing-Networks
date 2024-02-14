import random

import gudhi
import torch
from itertools import product
from torch import tensor, Tensor
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.linalg import vector_norm
from torch_geometric.data import Data
from collections import defaultdict
from typing import Tuple, Dict, Set, FrozenSet
from gudhi.simplex_tree import SimplexTree
from torch_geometric.transforms import BaseTransform
import itertools
import math
import networkx as nx


def esmpn_rips_lift(graph: Data, dim: int, dis: float, simplex_tree=None) -> Tuple[Dict[int, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Generates simplicial complex based on Rips complex generated from point cloud or geometric graph. Returns a dictionary
    for the simplice and their features (x_dict), a dictionary for the different adjacencies (adj) and a dictionary with
    the different E(n) invariant geometric information as described in the paper.
    """
    if simplex_tree is None:
        # create simplicial complex
        if hasattr(graph, 'init_pos'):
            pos = graph.init_pos
        elif hasattr(graph, 'loc'):
            pos = graph.loc
        elif hasattr(graph, 'pos'):
            pos = graph.pos

        points = [pos[i].tolist() for i in range(pos.shape[0])]
        rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim)

    # generate dictionaries
    simplices = generate_simplices(simplex_tree)
    indices = generate_indices(simplex_tree)
    adj, inv = generate_adjacencies_and_invariants(indices, simplex_tree)
    x_dict = generate_features(simplices, indices)

    return x_dict, adj, inv


def generate_simplices(simplex_tree: SimplexTree) -> Dict[int, Set[FrozenSet]]:
    """
    Generates dictionary of simplices. For each dimensions"""
    sim = defaultdict(set)

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        sim[dim].add(frozenset(simplex))

    return sim


def generate_indices(simplex_tree: SimplexTree) -> Dict[int, Dict[FrozenSet, int]]:
    """
    Generates a dictionary which assigns to each simplex a unique index used for reference when finding the different
    adjacency types and invariants.
    """
    indices = defaultdict(dict)

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        simplex_set = frozenset(simplex)

        if simplex_set not in indices[dim]:
            indices[dim][simplex_set] = len(indices[dim])

    return indices


def generate_adjacencies_and_invariants(indices: Dict, simplex_tree: SimplexTree) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """todo: add"""
    adj = defaultdict(list)
    inv = defaultdict(list)

    for simplex, _ in simplex_tree.get_simplices():
        simplex_dim = len(simplex)-1
        simplex_index = indices[simplex_dim][frozenset(simplex)]

        boundaries = simplex_tree.get_boundaries(simplex)
        cofaces = simplex_tree.get_cofaces(simplex, 1)

        for coface, _ in cofaces:
            coface_boundaries = simplex_tree.get_boundaries(coface)

            for coface_boundary, _ in coface_boundaries:
                # check if coface is distinct from the simplex
                if frozenset(coface_boundary) != frozenset(simplex):
                    coface_boundary_index = indices[simplex_dim][frozenset(coface_boundary)]
                    # save adjacency
                    adj[f'{simplex_dim}_{simplex_dim}'].append(tensor([coface_boundary_index, simplex_index]))

                    # calculate the upper adjacent invariants and save
                    shared = [vertex for vertex in simplex if vertex in coface_boundary]
                    a, b = [vertex for vertex in simplex if vertex not in shared], [vertex for vertex in coface_boundary if vertex not in shared]

                    inv[f'{simplex_dim}_{simplex_dim}'].append(tensor([p for p in shared] + [a[0], b[0]]))

        for boundary, _ in boundaries:
            boundary_index = indices[simplex_dim-1][frozenset(boundary)]
            # save adjacency
            adj[f'{simplex_dim-1}_{simplex_dim}'].append(tensor([boundary_index, simplex_index]))

            # calculate the boundary invariants and save
            shared = [vertex for vertex in simplex if vertex in boundary]
            b = [vertex for vertex in simplex if vertex not in shared]

            inv[f'{simplex_dim-1}_{simplex_dim}'].append(tensor([p for p in shared] + [b[0]]))

    # Add fully connected 0-0 connections
    nodes = [i for i in range(len(indices[0]))]
    edges_present = [sim[0] for sim in simplex_tree.get_simplices() if len(sim[0]) == 2]

    for i, j in product(nodes, nodes):
        if [i, j] not in edges_present and i != j:
            adj['0_0'].append(tensor([i, j]))
            inv['0_0'].append(tensor([i, j]))

    for k, v in adj.items():
        adj[k] = torch.stack(v, dim=1)

    for k, v in inv.items():
        inv[k] = torch.stack(v, dim=1)

    return adj, inv


def generate_features(simplices: Dict[int, Set[FrozenSet]], indices: Dict[int, Dict[FrozenSet, int]]) -> Dict[int, Tensor]:
    x_dict = {}
    for i in range(len(simplices)):
        # num_dim_perms = math.factorial(i + 1) * (i + 1)
        x = torch.zeros((len(simplices[i]), i+1))
        for k, v in indices[i].items():
            x[v] = tensor(list(k))
            # x[v] = tensor(list(itertools.permutations(list(k)))).flatten()
        x_dict[i] = x.long()

    return x_dict


def simplicial_lift_hulls(graph: Data):
    edge_index = graph.edge_index  # Your edge index data
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    # triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
    # edges = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 2]
    # create simplex tree

    from scipy.spatial import ConvexHull
    from itertools import combinations
    hull = ConvexHull(graph.input.numpy())

    def extract_k_simplices(simplices, k):
        simplices_set = set()
        for simplex in simplices:
            for subset in combinations(simplex, k+1):
                simplices_set.add(tuple(sorted(subset)))
        return list(simplices_set)

    simplex_tree = gudhi.SimplexTree()

    for v in range(len(graph.input)):
        simplex_tree.insert([v])

    for k in range(1, 3):
        k_simplex = extract_k_simplices(hull.simplices, k)

        # print(f"{k}-simplex: {len(k_simplex)}")

        for simplex in k_simplex:
            simplex_tree.insert(simplex)
    return simplex_tree
