from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.data import Data
from typing import Any
from csmpn.data.ESMPN.rips_lift import esmpn_rips_lift, generate_adjacencies_and_invariants, generate_features, generate_indices, generate_simplices, simplicial_lift_hulls
import gudhi
from collections import defaultdict

class SimplicialComplexData(Data):
    """
    Abstract simplicial complex class that generalises the pytorch geometric graph (Data). Adjacency tensors are stacked
    in the same fashion as the standard edge_index.
    """
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'adj' in key:
            i, j = key[4], key[6]
            return torch.tensor([[getattr(self, f'x_{i}').size(0)], [getattr(self, f'x_{j}').size(0)]])
        elif key == 'inv_0_0':
            return torch.tensor([[getattr(self, f'x_0').size(0)], [getattr(self, f'x_0').size(0)]])
        elif key == 'inv_0_1':
            return torch.tensor([[getattr(self, f'x_0').size(0)], [getattr(self, f'x_0').size(0)]])
        elif key == 'inv_1_1':
            return torch.tensor([[getattr(self, f'x_0').size(0)], [getattr(self, f'x_0').size(0)], [getattr(self, f'x_0').size(0)]])
        elif key == 'inv_1_2':
            return torch.tensor([[getattr(self, f'x_0').size(0)], [getattr(self, f'x_0').size(0)], [getattr(self, f'x_0').size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'adj' in key or 'inv' in key:
            return 1
        else:
            return 0

class ESMPN_SimplicialTransform(BaseTransform):
    """Todo: add
    The adjacency types (adj) are saved as properties, e.g. object.adj_1_2 gives the edge index from 1-simplices to
    2-simplices."""
    def __init__(self, dim=2, dis:float=3, label=None):
        self.dim = dim
        self.dis = dis
        self.label = label

    def __call__(self, graph: Data) -> SimplicialComplexData:
        if self.label is None:
            simplex_tree = None
        elif self.label == "hulls":
            simplex_tree = simplicial_lift_hulls(graph)
        else:
            simplex_tree = None
            AssertionError(f"Unrecoginzed label for simplicial lift: {self.label}")

        # get relevant dictionaries using the Rips complex based on the geometric graph or point cloud
        x_dict, adj_dict, inv_dict = esmpn_rips_lift(graph, self.dim, self.dis, simplex_tree)

        sim_com_data = SimplicialComplexData()
        sim_com_data = sim_com_data.from_dict(graph.to_dict())

        for k, v in x_dict.items():
            sim_com_data[f'x_{k}'] = v

        for k, v in adj_dict.items():
            sim_com_data[f'adj_{k}'] = v

        for k, v in inv_dict.items():
            sim_com_data[f'inv_{k}'] = v
        # remove old properties
        for att in ['edge_index']:
            if hasattr(sim_com_data, att):
                sim_com_data.pop(att)

        return sim_com_data


class ESMPN_ManualTransform(BaseTransform):
    """Todo: add
    The adjacency types (adj) are saved as properties, e.g. object.adj_1_2 gives the edge index from 1-simplices to
    2-simplices."""
    def __init__(self):
        self.num_edges = 9
        self.num_tris = 5
        self.num_nodes = 31

    def __call__(self, graph: Data):
        x_0 = [[node] for node in range(self.num_nodes)]
        x_1 = [
            [6,7], [7,8], [6,8], [1,2], [2,3], [1,3], [0,8], [0,3], [3,8], [24,25], [25,26], [24,26], [22,23], [21,22], [21,23]
        ]
        x_2 = [
            [0,3,8], [6,7,8], [1,2,3], [24,25,26], [21,22,23]
        ]

        simplex_tree = self.generate_simplex_tree(x_0, x_1, x_2)
        simplices = generate_simplices(simplex_tree)
        indices = generate_indices(simplex_tree)
        # indices = self.generate_index(x_0, x_1, x_2)
        adj_dict, inv_dict = generate_adjacencies_and_invariants(indices, simplex_tree)
        adj_dict['0_0'] = graph.edge_index
        x_dict = generate_features(simplices, indices)
        sim_com_data = SimplicialComplexData()
        sim_com_data = sim_com_data.from_dict(graph.to_dict())

        for k, v in x_dict.items():
            sim_com_data[f'x_{k}'] = v

        for k, v in adj_dict.items():
            sim_com_data[f'adj_{k}'] = v

        for k, v in inv_dict.items():
            sim_com_data[f'inv_{k}'] = v
        # remove old properties
        for att in ['edge_index']:
            if hasattr(sim_com_data, att):
                sim_com_data.pop(att)

        return sim_com_data

    def generate_simplex_tree(self, x_0, x_1, x_2):
        # insert nodes
        simplex_tree = gudhi.SimplexTree()  
        for node in x_0:
            simplex_tree.insert(node)
        # insert edges
        for edge in x_1:
            simplex_tree.insert(edge)
        # insert triangles
        for triangle in x_2:
            simplex_tree.insert(triangle)
        return simplex_tree
    
    def generate_index(self, x_0, x_1, x_2):
        indices = defaultdict(dict)
        sim = defaultdict(set)
        node_index = 0
        for node in x_0:
            indices[0][frozenset(node)] = node_index
            node_index += 1

        edge_index = 0
        for simplex in x_1:
            indices[1][frozenset(simplex)] = edge_index
            edge_index += 1
        

        tri_index = 0
        for simplex in x_2:
            indices[2][frozenset(simplex)] = tri_index
            tri_index += 1
        return indices