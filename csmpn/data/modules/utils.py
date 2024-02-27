# import graph_tool as gt
# import graph_tool.topology as top
from builtins import hasattr
import numpy as np
import torch
import itertools
import networkx as nx
import gudhi
from gudhi.simplex_tree import SimplexTree
from itertools import product
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from joblib import delayed

import torch
from torch import tensor, Tensor
from torch_geometric.data import Data
from collections import defaultdict
from typing import Tuple, Dict, Set, FrozenSet
import math

def generate_simplicies_single(simplex_tree: SimplexTree) -> Dict[int, Set[FrozenSet]]:
    """
    Generates dictionary of simplices. For each dimensions"""
    sim = defaultdict(set)

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        sim[dim].add(frozenset(simplex))

    return sim


def generate_indices_single(simplex_tree: SimplexTree) -> Dict[int, Dict[FrozenSet, int]]:
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

def generate_features_single(simplices: Dict[int, Set[FrozenSet]], indices: Dict[int, Dict[FrozenSet, int]]) -> Dict[int, Tensor]:
    x_dict = {}
    for i in range(len(simplices)):
        x = torch.zeros((len(simplices[i]), i + 1))
        for k, v in indices[i].items():
            x[v] = tensor(list(k))
        x_dict[i] = x.long()

    return x_dict

def generate_adjacencies_single(indices: Dict, simplex_tree: SimplexTree) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """todo: add"""
    adj = defaultdict(list)

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


        for boundary, _ in boundaries:
            boundary_index = indices[simplex_dim-1][frozenset(boundary)]
            # save adjacency
            adj[f'{simplex_dim-1}_{simplex_dim}'].append(tensor([boundary_index, simplex_index]))

    # Add fully connected 0-0 connections
    nodes = [i for i in range(len(indices[0]))]
    edges_present = [sim[0] for sim in simplex_tree.get_simplices() if len(sim[0]) == 2]

    for i, j in product(nodes, nodes):
        if [i, j] not in edges_present and i != j:
            adj['0_0'].append(tensor([i, j]))


    for k, v in adj.items():
        adj[k] = torch.stack(v, dim=1)


    return adj


def rips_lift(
    graph: Data, dim: int, dis: float
) -> Tuple[Dict[int, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Generates simplicial complex based on Rips complex generated from point cloud or geometric graph. Returns a dictionary
    for the simplice and their features (x_dict), a dictionary for the different adjacencies (adj) and a dictionary with
    the different E(n) invariant geometric information as described in the paper.
    """

    # create simplicial complex
    if hasattr(graph, "init_pos"):
        loc = graph.init_pos
    elif hasattr(graph, "loc"):
        loc = graph.loc
    elif hasattr(graph, "pos"):
        loc = graph.pos
    else:
        raise Exception(
            "Graphs in datasets have to be specified with locations for constructing simplicial complexes."
        )

    points = [loc[i].tolist() for i in range(loc.shape[0])]
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=dis)

    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim)
    # generate dictionaries
    simplices = generate_simplicies_single(simplex_tree)
    indices = generate_indices_single(simplex_tree)
    adj = generate_adjacencies_single(indices, simplex_tree)
    x_dict = generate_features_single(simplices, indices)
    return x_dict, adj  # type: ignore


def triangle_area(vertex1, vertex2, vertex3):
    # Calculate vectors between vertices
    vector1 = vertex2 - vertex1
    vector2 = vertex3 - vertex1

    # Calculate cross product and its magnitude
    cross_product = torch.cross(vector1, vector2)
    area = 0.5 * torch.linalg.norm(cross_product, dim=1)

    return area


def simplicial_lift(graph: Data, edge_th=10000, tri_th=10000):
    """
    Generates simplicial complex based on Rips complex generated from point cloud or geometric graph. Returns a dictionary
    for the simplice and their features (x_dict), a dictionary for the different adjacencies (adj) and a dictionary with
    the different E(n) invariant geometric information as described in the paper.
    """

    # create simplicial complex
    if hasattr(graph, "init_pos"):
        loc = graph.init_pos
    elif hasattr(graph, "loc"):
        loc = graph.loc
    elif hasattr(graph, "pos"):
        loc = graph.pos
    else:
        raise Exception(
            "Graphs in datasets have to be specified with locations for constructing simplicial complexes."
        )

    edge_index = graph.edge_index  # Your edge index data
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
    edges = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 2]
    # create simplex tree

    # insert nodes
    simplex_tree = gudhi.SimplexTree()
    for node in range(len(loc)):
        simplex_tree.insert([node])
    # insert edges
    edges = torch.tensor(edges)[
        torch.where(
            torch.norm(
                loc[torch.tensor(edges)][:, 0, :] - loc[torch.tensor(edges)][:, 1, :],
                dim=-1,
            )
            <= edge_th
        )
    ]
    for edge in edges:
        simplex_tree.insert(edge.numpy().tolist())
    # insert triangles
    if triangles != []:
        vertexs = loc[torch.tensor(triangles)]
        areas = triangle_area(vertexs[:, 0, :], vertexs[:, 1, :], vertexs[:, 2, :])
        triangles = torch.tensor(triangles)[torch.where(areas <= tri_th)]
        for triangle in triangles:
            simplex_tree.insert(triangle)

    # generate dictionaries
    simplices = generate_simplices(simplex_tree, simplex_tree, simplex_tree)
    indices = generate_indices(simplex_tree, simplex_tree, simplex_tree, simplices)
    adj = generate_adjacencies(indices, simplex_tree, simplex_tree, simplex_tree)
    x_dict = generate_features(simplices, indices)
    return x_dict, adj  # type: ignore


def simplicial_lift_hulls(graph: Data, dim: int):
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

    for k in range(1, dim+1):
        k_simplex = extract_k_simplices(hull.simplices, k)

        # print(f"{k}-simplex: {len(k_simplex)}")

        for simplex in k_simplex:
            simplex_tree.insert(simplex)

    # generate dictionaries
    simplices = generate_simplicies_single(simplex_tree)
    indices = generate_indices_single(simplex_tree)
    adj = generate_adjacencies_single(indices, simplex_tree)
    x_dict = generate_features_single(simplices, indices)
    return x_dict, adj  # type: ignore

def generate_simplices(
    simplex_tree: SimplexTree, simplex_tree_edge, simplex_tree_tri
) -> Dict[int, Set[FrozenSet]]:
    """
    Generates dictionary of simplices. For each dimensions"""
    sim = defaultdict(set)

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            sim[dim].add(frozenset(simplex))

    # add simplex with dimension higher than 0 in a low level simplex tree
    for simplex, _ in simplex_tree_edge.get_simplices():
        dim = len(simplex) - 1
        if dim == 1:
            sim[dim].add(frozenset(simplex))

    # only add triangles that have boundaries in recorded edges
    for simplex, _ in simplex_tree_tri.get_simplices():
        dim = len(simplex) - 1
        if dim == 2 and test_boundary(simplex, sim, simplex_tree_tri):
            sim[dim].add(frozenset(simplex))

    return sim


def test_boundary(simplex, sim, simplex_tree):
    boundaries = simplex_tree.get_boundaries(simplex)
    for boundary, _ in boundaries:
        if set(boundary) in sim[1]:
            return True
    return False


def generate_indices(
    simplex_tree: SimplexTree, simplex_tree_edge, simplex_tree_tri, simplices
) -> Dict[int, Dict[FrozenSet, int]]:
    """
    Generates a dictionary which assigns to each simplex a unique index used for reference when finding the different
    adjacency types and invariants.
    """
    indices = defaultdict(dict)

    # add node (dim=0 simplexes) in the original tree
    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            simplex_set = frozenset(simplex)

            if simplex_set not in indices[dim]:
                indices[dim][simplex_set] = len(indices[dim])

    # add edge and triangle in smaller radius tree
    for simplex, _ in simplex_tree_edge.get_simplices():
        dim = len(simplex) - 1
        if dim == 1:
            simplex_set = frozenset(simplex)

            if simplex_set not in indices[dim]:
                indices[dim][simplex_set] = len(indices[dim])

    for simplex, _ in simplex_tree_tri.get_simplices():
        dim = len(simplex) - 1
        if dim == 2 and test_boundary(simplex, simplices, simplex_tree_tri):
            simplex_set = frozenset(simplex)

            if simplex_set not in indices[dim]:
                indices[dim][simplex_set] = len(indices[dim])
    return indices


def generate_adjacencies(
    indices: Dict, simplex_tree: SimplexTree, simplex_tree_edge, simplex_tree_tri
) -> Dict[str, Tensor]:
    """
    Generate boundary and upper adjacency, coboundary adjacencies are automatically generated by boundary adjacencies.
    """
    adj = defaultdict(list)

    # iterate through all simplexes in original tree
    for simplex, _ in simplex_tree.get_simplices():
        simplex_dim = len(simplex) - 1
        if frozenset(simplex) in indices[simplex_dim]:
            simplex_index = indices[simplex_dim][frozenset(simplex)]
        else:
            continue

        if simplex_dim == 0:
            used_simplex_tree = simplex_tree
        elif simplex_dim == 1:
            used_simplex_tree = simplex_tree_edge
        else:
            used_simplex_tree = simplex_tree_tri
        boundaries = used_simplex_tree.get_boundaries(simplex)
        cofaces = used_simplex_tree.get_cofaces(simplex, 1)

        # upper adjacency
        for coface, _ in cofaces:
            coface_boundaries = simplex_tree.get_boundaries(coface)

            for coface_boundary, _ in coface_boundaries:
                # check if coface is distinct from the simplex
                if frozenset(coface_boundary) != frozenset(simplex):
                    coface_boundary_index = indices[simplex_dim][
                        frozenset(coface_boundary)
                    ]
                    # save adjacency
                    adj[f"{simplex_dim}_{simplex_dim}"].append(
                        tensor([coface_boundary_index, simplex_index])
                    )

        # boundary
        for boundary, _ in boundaries:
            if frozenset(boundary) in indices[simplex_dim - 1]:
                boundary_index = indices[simplex_dim - 1][frozenset(boundary)]
            else:
                continue
            # save adjacency
            adj[f"{simplex_dim-1}_{simplex_dim}"].append(
                tensor([boundary_index, simplex_index])
            )

    for k, v in adj.items():
        adj[k] = torch.stack(v, dim=1)
    return adj


def generate_features(
    simplices: Dict[int, Set[FrozenSet]], indices: Dict[int, Dict[FrozenSet, int]]
) -> Dict[int, Tensor]:
    x_dict = {}
    for i in range(len(simplices)):
        x = torch.zeros((len(simplices[i]), i + 1))
        for k, v in indices[i].items():
            x[v] = tensor(list(k))
        x_dict[i] = x.long()

    return x_dict
