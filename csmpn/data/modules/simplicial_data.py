from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.data import Data
from typing import Any
from csmpn.data.modules.utils import rips_lift, simplicial_lift, simplicial_lift_hulls
from csmpn.algebra.cliffordalgebra import CliffordAlgebra
from collections import defaultdict
from torch_geometric.nn import radius_graph
import math

class SimplicialComplexData(Data):
    """
    Abstract simplicial complex class that generalises the pytorch geometric graph (Data). Adjacency tensors are stacked
    in the same fashion as the standard edge_index.
    """
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'adj' in key:
            i, j = key[4], key[6]
            return torch.tensor([[getattr(self, f'x_{i}').size(0)], [getattr(self, f'x_{j}').size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'adj' in key or 'edge_index' in key:
            return 1
        else:
            return 0

class SimplicialTransform(BaseTransform):
    """Todo: add
    The adjacency types (adj) are saved as properties, e.g. object.adj_1_2 gives the edge index from 1-simplices to
    2-simplices."""
    def __init__(self, dim=2, dis:float=2.0, use_for_loop=False, label="nbody", edge_th=10000., tri_th=10000., molecule_type=None):
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.dim = dim
        self.dis = dis
        self.label = label
        self.use_for_loop = use_for_loop
        self.edge_th = edge_th
        self.tri_th = tri_th
        self.molecule_type = molecule_type

    def __call__(self, graph: Data):
        # for MD17
        # if self.label in {"md17", "gravity"}:
        if self.label in {"gravity"}:
            x_dict, adj_dict = simplicial_lift(graph, self.edge_th, self.tri_th)
        elif self.label=="hulls":
            x_dict, adj_dict = simplicial_lift_hulls(graph, self.dim)
        else:
            if self.molecule_type == "aspirin":
                x_dict, adj_dict = simplicial_lift(graph, self.edge_th, self.tri_th)
            else:
                # get relevant dictionaries using the Rips complex based on the geometric graph or point cloud
                x_dict, adj_dict = rips_lift(graph, self.dim, self.dis)

        # vertices = len(x_dict[0])
        # edges = len(x_dict[1]) if 1 in x_dict else 0
        # triangles = len(x_dict[2]) if 2 in x_dict else 0
        # print(f"Vertices {vertices}, Edges {edges}, Triangles {triangles}")
        sim_com_data = SimplicialComplexData()
        sim_com_data = sim_com_data.from_dict(graph.to_dict())

        for k, v in x_dict.items():
            sim_com_data[f'x_{k}'] = v

        for k, v in adj_dict.items():
            sim_com_data[f'adj_{k}'] = v
        # # keep the original node-node connections
        # sim_com_data.adj_0_0 = graph.edge_index
        num_per_dim = self.get_num_simplicies(x_dict)

        sim_com_data = self.add_missing_adj(sim_com_data)
        sim_com_data = self.get_edge(x_dict, sim_com_data, num_per_dim)

        # sim_com_data = self.get_nbody_feat(sim_com_data, num_per_dim)
        # sim_com_data = self.gen_qm9_feat(sim_com_data, num_per_dim)
        # sim_com_data = self.gen_motion_feat(sim_com_data, num_per_dim)
        if self.label == "md17":
            sim_com_data = self.gen_md17_feat(sim_com_data, num_per_dim, self.dim)
        elif self.label == "nba":
            sim_com_data = self.gen_nba_feat(sim_com_data, num_per_dim)
        elif self.label == "gravity":
            sim_com_data = self.gen_gravity_feat(sim_com_data, num_per_dim)
        elif self.label == "nbody":
            sim_com_data = self.get_nbody_feat(sim_com_data, num_per_dim)
        elif self.label == "hulls":
            sim_com_data = self.gen_hulls_feat(sim_com_data, num_per_dim, self.dim)
        else:
            raise ValueError(f"Unknown dataset {self.label}.")
        if self.use_for_loop:
            return sim_com_data
        # if not using for-loop model, then construct the big graph.
        graphdata = Data(
            edge_index=sim_com_data.edge_index,
            node_types=sim_com_data.node_types,
            y=sim_com_data.y,
            x_ind=sim_com_data.x_ind,
        )
        if any(k.startswith('target_') for k in sim_com_data.keys):
            target_attrs = [attr for attr in sim_com_data.keys if attr.startswith("target")]
            assert len(target_attrs) == 1, "Got multiple or no targets."
            target_attr = target_attrs[0]
            setattr(graphdata, target_attr, getattr(sim_com_data, target_attr))
        if hasattr(sim_com_data, "target"):
            graphdata.target = sim_com_data.target
        if hasattr(sim_com_data, "pos"):
            graphdata.pos = sim_com_data.pos
        if hasattr(sim_com_data, "loc"):
            graphdata.loc = sim_com_data.loc
        if hasattr(sim_com_data, "edge_attr"):
            graphdata.edge_attr = sim_com_data.edge_attr
        if hasattr(sim_com_data, "x"):
            graphdata.x = sim_com_data.x
        if hasattr(sim_com_data, "name"):
            graphdata.name = sim_com_data.name
        if hasattr(sim_com_data, "vel"):
            graphdata.vel = sim_com_data.vel
        if hasattr(sim_com_data, "charges"):
            graphdata.charges = sim_com_data.charges
        if hasattr(sim_com_data, "mass"):
            graphdata.mass = sim_com_data.mass
        if hasattr(sim_com_data, "input"):
            graphdata.input = sim_com_data.input
        return graphdata
    
    def add_missing_adj(self, sim_com_data):
        # add in missing adjacencies (i.e. downward communication)
        for dim in range(self.dim):
            if hasattr(sim_com_data, f'adj_{dim}_{dim + 1}'):
                sim_com_data[f'adj_{dim + 1}_{dim}'] = sim_com_data[f'adj_{dim}_{dim + 1}'][[1, 0]].clone()
        return sim_com_data

    def get_num_simplicies(self, x_dict):
         # count the accumulated number of simplicies in each dimension.
        num = 0
        num_per_dim = [0]
        for dim in x_dict.keys():
            num += len(x_dict[dim])
            num_per_dim.append(num)
        return num_per_dim
    
    def get_edge(self, x_dict, sim_com_data, num_per_dim):
        """This function generates the edge index, with each simplex represented one object in the big graph, index starting from 0.
        
        args:
            x_dict: {dim: [node_index]} a dictionary, each key represents the dimension of the simplex, each value is a list of list containing len=dim+1 elements.
            sim_com_data: simplicial complex data transformed for for-loop model.
            num_per_dim: List, containing accumulated number of simplicies. E.g. [0, 5, 10, 15] means 5 nodes, 5 edges and 5 triangles.
        returns:
            edge_index: normal PyG edge index [2, n] of the big graph.
            edge_attr: normal PyG edge attr [n, 2], with each row specifying the type of the node. E.g. [0, 1] represents the edge is composed of a node and a edge in the big graph.
        """
        edge_index = []
        edge_attr = []
        comm_type_record = []

        for dim_source in x_dict.keys():
            for dim_target in x_dict.keys():
                if hasattr(sim_com_data, f"adj_{dim_source}_{dim_target}") and [dim_source, dim_target] not in comm_type_record:
                    s_t_edge_index = torch.zeros_like(sim_com_data[f"adj_{dim_source}_{dim_target}"])
                    s_t_edge_index[0] = sim_com_data[f"adj_{dim_source}_{dim_target}"][0] + num_per_dim[dim_source]
                    s_t_edge_index[1] = sim_com_data[f"adj_{dim_source}_{dim_target}"][1] + num_per_dim[dim_target]
                    edge_index.append(s_t_edge_index)
                    num_edges = sim_com_data[f"adj_{dim_source}_{dim_target}"].shape[1]
                    for _ in range(num_edges):
                        edge_attr.append([dim_source, dim_target])
                    sim_com_data[f"edge_attr_{dim_source}_{dim_target}"] = torch.tensor(
                        [[dim_source]*num_edges, [dim_target]*num_edges]
                    ).T
                    comm_type_record.append([dim_source, dim_target])

        edge_attr = torch.tensor(edge_attr).T
        edge_index = torch.cat(edge_index, dim=-1)

        sim_com_data.edge_index = edge_index
        sim_com_data.edge_attr = edge_attr.T

        return sim_com_data
    

    def gen_clifford_feat(self, num_node_list, graph, name):
        # I wanted to use geometric product to initialize the multivector of different simplex. 
        # E.g. node simplicies and edge simplicies are filled with the vector part, triangle simplicies are filled with the bivector part.
        # TODO: generalize this initialization to any dimension.
        assert self.dim <= 2
        feature_matrix = torch.zeros((num_node_list[-1], graph[name].shape[1]))
        for dim in range(self.dim+1):
            if hasattr(graph, f'x_{dim}'):
                if dim ==0:
                    feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = graph[name][graph[f'x_{dim}']].mean(dim=1)
                if dim == 1:
                    feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = graph[name][graph[f'x_{dim}']][:, 0, :] - graph[name][graph[f'x_{dim}']][:, 1, :]
                if dim == 2:
                    edge_1 = self.algebra.embed((graph[name][graph[f'x_{dim}']][:, 0, :] - graph[name][graph[f'x_{dim}']][:, 1, :]), (1,2,3))
                    edge_2 = self.algebra.embed((graph[name][graph[f'x_{dim}']][:, 0, :] - graph[name][graph[f'x_{dim}']][:, 2, :]), (1,2,3))
                    feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = self.algebra.geometric_product(edge_1, edge_2)[:, 4:7]
        graph[f"feat_{name}"] = feature_matrix
        return graph

    def gen_type_feat(self, num_node_list, graph, name):
        # x feature in QM9 datasets represent the atom types, 
        # each simplex should have three elements representing the types of nodes composing the simplex, 
        # with all the other node types filled as zero (represents null atom) if dimension of simplex is lower than the max dim.
        # E.g. For dim0 simplex with type = 0, atom type should [1, 0, 0]. [type+1, 0, 0]
        feature_matrix = torch.zeros((num_node_list[-1], self.dim+1))
        for dim in range(self.dim+1):
            if hasattr(graph, f'x_{dim}'):
                num_zero_col_to_fill = self.dim+1 - (dim+1)
                num_simplicies = graph[f'x_{dim}'].shape[0]
                feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = torch.cat((graph[name][graph[f'x_{dim}']], torch.zeros((num_simplicies, num_zero_col_to_fill))), dim=-1)
        graph[f"feat_{name}"] = feature_matrix
        return graph
    
    def gen_mean_feat(self, num_node_list, graph, name):
        # the other features of the simplicies are initialized by simply taking the mean value, or even can be initialized with zeros.
        feature_matrix = torch.zeros((num_node_list[-1], graph[name].shape[1]))
        for dim in range(self.dim+1):
            if hasattr(graph, f'x_{dim}'):
                feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = graph[name][graph[f'x_{dim}']].mean(dim=1)
        graph[f"feat_{name}"] = feature_matrix
        return graph
    
    def gen_simplicial_type_feat(self, num_node_list, graph):
        # we need also assign the simplicial dimensions to each simplicies in our graph. 
        # Sometimes we need to predict the features (positions) of the dim0 simplicies in the graph (E.g. NBody)
        node_types = torch.zeros((num_node_list[-1])).type(torch.long)
        for dim in range(len(num_node_list)-1):
            node_types[num_node_list[dim]: num_node_list[dim+1]] = dim
        graph[f"node_types"] = node_types
        return graph

    def get_nbody_feat(self, graph, num_node_list):

        # for name in feat_names:
        #     if name == "x":
        #         graph = self.gen_type_feat(num_node_list, graph, name)
        #         continue

        #     if name in ['pos', 'loc']:
        #         graph = self.gen_clifford_feat(num_node_list, graph, name)
        #         continue

        #     graph = self.gen_mean_feat(num_node_list, graph, name)

        # graph = self.gen_simplicial_type_feat(num_node_list, graph)

        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], 3))
        pos_feature_matrix[: num_node_list[1]] = graph.loc
        graph.loc = pos_feature_matrix

        # velocity features
        vel_feature_matrix = torch.zeros((num_node_list[-1], 3))
        vel_feature_matrix[: num_node_list[1]] = graph.vel
        graph.vel = vel_feature_matrix

        # charge features
        charge_feature_matrix = torch.zeros((num_node_list[-1], 1))
        charge_feature_matrix[: num_node_list[1]] = graph.charges
        graph.charges = charge_feature_matrix
        # index
        num_dim_perms = math.factorial(len(num_node_list) - 1) * (len(num_node_list) - 1)
        index_matrix = torch.zeros((num_node_list[-1], num_dim_perms))
        for i in range(self.dim + 1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i] : num_node_list[i + 1], : math.factorial(i + 1) * (i + 1)] = graph[
                    f"x_{i}"
                ]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)

        return graph


    def gen_hulls_feat(self, graph, num_node_list, dim):
        factor = len(num_node_list) - 1
        input_feature_matrix = torch.zeros((num_node_list[-1], 5))
        input_feature_matrix[: num_node_list[1]] = graph.input
        graph.input = input_feature_matrix
        num_dim_perms = math.factorial(factor) * factor
        index_matrix = torch.zeros((num_node_list[-1], num_dim_perms))
        for i in range(self.dim + 1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i] : num_node_list[i + 1], : math.factorial(i + 1) * (i + 1)] = graph[
                    f"x_{i}"
                ]
        graph.x_ind = index_matrix
        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)

        return graph

    def gen_gravity_feat(self, graph, num_node_list):
        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], 3))
        pos_feature_matrix[: num_node_list[1]] = graph.pos
        graph.pos = pos_feature_matrix

        # velocity features
        vel_feature_matrix = torch.zeros((num_node_list[-1], 3))
        vel_feature_matrix[: num_node_list[1]] = graph.vel
        graph.vel = vel_feature_matrix

        # mass features
        mass_feature_matrix = torch.zeros((num_node_list[-1], 1))
        mass_feature_matrix[: num_node_list[1]] = graph.mass[:, None]
        graph.mass = mass_feature_matrix

        # index
        index_matrix = torch.zeros((num_node_list[-1], 3))
        for i in range(self.dim + 1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i] : num_node_list[i + 1], : i + 1] = graph[
                    f"x_{i}"
                ]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)

        return graph
    
    def gen_qm9_feat(self, graph, num_node_list):
        # como empty features
        feat_dim = graph.x.shape[-1]
        feature_matrix = torch.zeros((num_node_list[-1], feat_dim))
        feature_matrix[: num_node_list[1]] = graph.x
        graph.x = feature_matrix

        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], 3))
        pos_feature_matrix[: num_node_list[1]] = graph.pos
        graph.pos = pos_feature_matrix

        # index
        index_matrix = torch.zeros((num_node_list[-1], 3))
        for i in range(self.dim+1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i]: num_node_list[i+1], :i+1] = graph[f"x_{i}"]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)
        return graph


    def gen_motion_feat(self, graph, num_node_list):

        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], 3))
        pos_feature_matrix[: num_node_list[1]] = graph.loc
        graph.pos = pos_feature_matrix

        vel_feature_matrix = torch.zeros((num_node_list[-1], 3))
        vel_feature_matrix[: num_node_list[1]] = graph.vel
        graph.vel = vel_feature_matrix

        # index
        index_matrix = torch.zeros((num_node_list[-1], 3))
        for i in range(self.dim+1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i]: num_node_list[i+1], :i+1] = graph[f"x_{i}"]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)
        return graph


    def gen_md17_feat(self, graph, num_node_list, dim):
        num_frames = graph.y.shape[1]
        graph.charges = graph.charges.unsqueeze(-1).repeat(1, num_frames).unsqueeze(-1)
        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], num_frames, 3))
        pos_feature_matrix[: num_node_list[1]] = graph.loc
        graph.loc = pos_feature_matrix

        # vel features
        vel_feature_matrix = torch.zeros((num_node_list[-1], num_frames, 3))
        vel_feature_matrix[: num_node_list[1]] = graph.vel
        graph.vel = vel_feature_matrix

        # charge features
        charge_feature_matrix = torch.zeros((num_node_list[-1], num_frames, 1))
        charge_feature_matrix[: num_node_list[1]] = graph.charges
        graph.charges = charge_feature_matrix

        # index
        index_matrix = torch.zeros((num_node_list[-1], dim+1))
        for i in range(self.dim+1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i]: num_node_list[i+1], :i+1] = graph[f"x_{i}"]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)
        return graph


    def gen_nba_feat(self, graph, num_node_list):
        num_frames = graph.pos.shape[1]

        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], num_frames, 2))
        pos_feature_matrix[: num_node_list[1]] = graph.pos
        graph.pos = pos_feature_matrix

        # vel features
        vel_feature_matrix = torch.zeros((num_node_list[-1], num_frames, 2))
        vel_feature_matrix[: num_node_list[1]] = graph.vel
        graph.vel = vel_feature_matrix

        # index
        index_matrix = torch.zeros((num_node_list[-1], 3))
        for i in range(self.dim+1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i]: num_node_list[i+1], :i+1] = graph[f"x_{i}"]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)
        return graph
    

class ManualTransform(BaseTransform):
    def __init__(self):
        self.name = "Manually adding triangles and edges to the graphs."
        self.num_edges = 12
        self.num_tris = 4
        self.num_nodes = 31

    def __call__(self, graph: Data):
        dim1_dim2 = torch.tensor([
            [43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46]
        ])
        dim1_dim0 = torch.tensor([
                    [31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 6, 7, 7, 8, 6, 8, 1, 2, 2, 3, 1, 3, 24, 25, 25, 26, 24, 26, 22, 23, 21, 22, 21, 23],
                    [6, 7, 7, 8, 6, 8, 1, 2, 2, 3, 1, 3, 24, 25, 25, 26, 24, 26, 22, 23, 21, 22, 21, 23, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42]
                ])
        dim1_dim1 = torch.tensor([
            [31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42],
            [32, 33, 31, 33, 31, 32, 35, 36, 34, 36, 34, 35, 38, 39, 37, 39, 37, 38, 41, 42, 40, 42, 40, 41]
        ])
        added_edge_index = torch.cat((dim1_dim2, dim1_dim0, dim1_dim1), dim=-1)
        edge_attr = torch.cat(
            (
                torch.zeros((130, 2)), # 0-0
                torch.stack((torch.ones(12)+1, torch.ones(12))).T, # 2-1
                torch.stack((torch.ones(12), torch.ones(12)+1)).T, # 1-2
                torch.stack((torch.ones(24), torch.zeros(24))).T, # 1-0
                torch.stack((torch.zeros(24), torch.ones(24))).T, # 0-1,
                torch.stack((torch.ones(24), torch.ones(24))).T, # 1-1,
            ), dim=0
        ).type(torch.LongTensor)
            
        x_0 = torch.arange(31).unsqueeze(1)
        x_1 = torch.tensor([
            [6,7], [7,8], [6,8], [1,2], [2,3], [1,3], [24,25], [25,26], [24,26], [22,23], [21,22], [21,23]
        ])
        x_2 = torch.tensor([
            [6,7,8], [1,2,3], [24,25,26], [21,22,23]
        ])
        edge_index = torch.cat((graph.edge_index[:, :-6], added_edge_index), dim=-1)

        # pos features
        pos_feature_matrix = torch.zeros((self.num_nodes + self.num_edges + self.num_tris, 3))
        pos_feature_matrix[: self.num_nodes] = graph.loc
        graph['pos'] = pos_feature_matrix

        # vel features
        vel_feature_matrix = torch.zeros((self.num_nodes + self.num_edges + self.num_tris, 3))
        vel_feature_matrix[: self.num_nodes] = graph.vel
        graph.vel = vel_feature_matrix

        # index 
        index_matrix = torch.zeros((self.num_nodes + self.num_edges + self.num_tris, 3))
        index_matrix[:self.num_nodes, :1] = x_0
        index_matrix[self.num_nodes: self.num_nodes + self.num_edges, :2] = x_1
        index_matrix[self.num_nodes + self.num_edges: self.num_nodes + self.num_edges + self.num_tris, :3] = x_2
        graph.x_ind = index_matrix

        node_types = torch.zeros((self.num_nodes + self.num_edges + self.num_tris)).type(torch.long)
        node_types[:self.num_nodes] = 0
        node_types[self.num_nodes: self.num_nodes + self.num_edges] = 1
        node_types[self.num_nodes + self.num_edges:] = 2

        graph['edge_index'] = edge_index
        graph['node_types'] = node_types
        graph['edge_attr'] = edge_attr
        return graph
    
    def add_missing_adj(self, sim_com_data):
        # add in missing adjacencies (i.e. downward communication)
        for dim in range(self.dim):
            if hasattr(sim_com_data, f'adj_{dim}_{dim + 1}'):
                sim_com_data[f'adj_{dim + 1}_{dim}'] = sim_com_data[f'adj_{dim}_{dim + 1}'][[1, 0]].clone()
        return sim_com_data

    def get_num_simplicies(self, x_dict):
         # count the accumulated number of simplicies in each dimension.
        num = 0
        num_per_dim = [0]
        for dim in x_dict.keys():
            num += len(x_dict[dim])
            num_per_dim.append(num)
        return num_per_dim
    
    def get_edge(self, x_dict, sim_com_data, num_per_dim):
        """This function generates the edge index, with each simplex represented one object in the big graph, index starting from 0.
        
        args:
            x_dict: {dim: [node_index]} a dictionary, each key represents the dimension of the simplex, each value is a list of list containing len=dim+1 elements.
            sim_com_data: simplicial complex data transformed for for-loop model.
            num_per_dim: List, containing accumulated number of simplicies. E.g. [0, 5, 10, 15] means 5 nodes, 5 edges and 5 triangles.
        returns:
            edge_index: normal PyG edge index [2, n] of the big graph.
            edge_attr: normal PyG edge attr [n, 2], with each row specifying the type of the node. E.g. [0, 1] represents the edge is composed of a node and a edge in the big graph.
        """
        edge_index = []
        edge_attr = []
        comm_type_record = []

        for dim_source in x_dict.keys():
            for dim_target in x_dict.keys():
                if hasattr(sim_com_data, f"adj_{dim_source}_{dim_target}") and [dim_source, dim_target] not in comm_type_record:
                    s_t_edge_index = torch.zeros_like(sim_com_data[f"adj_{dim_source}_{dim_target}"])
                    s_t_edge_index[0] = sim_com_data[f"adj_{dim_source}_{dim_target}"][0] + num_per_dim[dim_source]
                    s_t_edge_index[1] = sim_com_data[f"adj_{dim_source}_{dim_target}"][1] + num_per_dim[dim_target]
                    edge_index.append(s_t_edge_index)
                    num_edges = sim_com_data[f"adj_{dim_source}_{dim_target}"].shape[1]
                    for _ in range(num_edges):
                        edge_attr.append([dim_source, dim_target])
                    sim_com_data[f"edge_attr_{dim_source}_{dim_target}"] = torch.tensor(
                        [[dim_source]*num_edges, [dim_target]*num_edges]
                    ).T
                    comm_type_record.append([dim_source, dim_target])

        edge_attr = torch.tensor(edge_attr).T
        edge_index = torch.cat(edge_index, dim=-1)

        sim_com_data.edge_index = edge_index
        sim_com_data.edge_attr = edge_attr.T

        return sim_com_data
    

    def gen_clifford_feat(self, num_node_list, graph, name):
        # I wanted to use geometric product to initialize the multivector of different simplex. 
        # E.g. node simplicies and edge simplicies are filled with the vector part, triangle simplicies are filled with the bivector part.
        # TODO: generalize this initialization to any dimension.
        assert self.dim <= 2
        feature_matrix = torch.zeros((num_node_list[-1], graph[name].shape[1]))
        for dim in range(self.dim+1):
            if hasattr(graph, f'x_{dim}'):
                if dim ==0:
                    feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = graph[name][graph[f'x_{dim}']].mean(dim=1)
                if dim == 1:
                    feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = graph[name][graph[f'x_{dim}']][:, 0, :] - graph[name][graph[f'x_{dim}']][:, 1, :]
                if dim == 2:
                    edge_1 = self.algebra.embed((graph[name][graph[f'x_{dim}']][:, 0, :] - graph[name][graph[f'x_{dim}']][:, 1, :]), (1,2,3))
                    edge_2 = self.algebra.embed((graph[name][graph[f'x_{dim}']][:, 0, :] - graph[name][graph[f'x_{dim}']][:, 2, :]), (1,2,3))
                    feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = self.algebra.geometric_product(edge_1, edge_2)[:, 4:7]
        graph[f"feat_{name}"] = feature_matrix
        return graph

    def gen_type_feat(self, num_node_list, graph, name):
        # x feature in QM9 datasets represent the atom types, 
        # each simplex should have three elements representing the types of nodes composing the simplex, 
        # with all the other node types filled as zero (represents null atom) if dimension of simplex is lower than the max dim.
        # E.g. For dim0 simplex with type = 0, atom type should [1, 0, 0]. [type+1, 0, 0]
        feature_matrix = torch.zeros((num_node_list[-1], self.dim+1))
        for dim in range(self.dim+1):
            if hasattr(graph, f'x_{dim}'):
                num_zero_col_to_fill = self.dim+1 - (dim+1)
                num_simplicies = graph[f'x_{dim}'].shape[0]
                feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = torch.cat((graph[name][graph[f'x_{dim}']], torch.zeros((num_simplicies, num_zero_col_to_fill))), dim=-1)
        graph[f"feat_{name}"] = feature_matrix
        return graph
    
    def gen_mean_feat(self, num_node_list, graph, name):
        # the other features of the simplicies are initialized by simply taking the mean value, or even can be initialized with zeros.
        feature_matrix = torch.zeros((num_node_list[-1], graph[name].shape[1]))
        for dim in range(self.dim+1):
            if hasattr(graph, f'x_{dim}'):
                feature_matrix[num_node_list[dim]: num_node_list[dim+1]] = graph[name][graph[f'x_{dim}']].mean(dim=1)
        graph[f"feat_{name}"] = feature_matrix
        return graph
    
    def gen_simplicial_type_feat(self, num_node_list, graph):
        # we need also assign the simplicial dimensions to each simplicies in our graph. 
        # Sometimes we need to predict the features (positions) of the dim0 simplicies in the graph (E.g. NBody)
        node_types = torch.zeros((num_node_list[-1])).type(torch.long)
        for dim in range(len(num_node_list)-1):
            node_types[num_node_list[dim]: num_node_list[dim+1]] = dim
        graph[f"node_types"] = node_types
        return graph

    def get_feat(self, graph, num_node_list, feat_names):
        # generate the features from the original graph to the simplicial graph.
        if feat_names is None:
            AssertionError("feature names have to be declared before calculating the average features!")

        for name in feat_names:
            if name == "x":
                graph = self.gen_type_feat(num_node_list, graph, name)
                continue

            if name in ['pos', 'loc']:
                graph = self.gen_clifford_feat(num_node_list, graph, name)
                continue

            graph = self.gen_mean_feat(num_node_list, graph, name)

        graph = self.gen_simplicial_type_feat(num_node_list, graph)

        return graph
    
    def gen_qm9_feat(self, graph, num_node_list):
        # como empty features
        feat_dim = graph.x.shape[-1]
        feature_matrix = torch.zeros((num_node_list[-1], feat_dim))
        feature_matrix[: num_node_list[1]] = graph.x
        graph.x = feature_matrix

        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], 3))
        pos_feature_matrix[: num_node_list[1]] = graph.pos
        graph.pos = pos_feature_matrix

        # index
        index_matrix = torch.zeros((num_node_list[-1], 3))
        for i in range(self.dim+1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i]: num_node_list[i+1], :i+1] = graph[f"x_{i}"]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)
        return graph


    # def gen_motion_feat(self, graph, num_node_list):

    #     # pos features
    #     pos_feature_matrix = torch.zeros((num_node_list[-1], 3))
    #     pos_feature_matrix[: num_node_list[1]] = graph.loc
    #     graph.pos = pos_feature_matrix

    #     vel_feature_matrix = torch.zeros((num_node_list[-1], 3))
    #     vel_feature_matrix[: num_node_list[1]] = graph.vel
    #     graph.vel = vel_feature_matrix

    #     # index
    #     index_matrix = torch.zeros((num_node_list[-1], 3))
    #     for i in range(self.dim+1):
    #         if hasattr(graph, f"x_{i}"):
    #             index_matrix[num_node_list[i]: num_node_list[i+1], :i+1] = graph[f"x_{i}"]
    #     graph.x_ind = index_matrix

    #     # node_type features
    #     graph = self.gen_simplicial_type_feat(num_node_list, graph)
    #     return graph
    
    def get_motion_feat(self, graph, num_node_list):

        # pos features
        pos_feature_matrix = torch.zeros((num_node_list[-1], 3))
        pos_feature_matrix[: num_node_list[1]] = graph.pos
        graph.loc = pos_feature_matrix

        # velocity features
        vel_feature_matrix = torch.zeros((num_node_list[-1], 3))
        vel_feature_matrix[: num_node_list[1]] = graph.vel
        graph.vel = vel_feature_matrix

        # index
        num_dim_perms = math.factorial(len(num_node_list) - 1) * (len(num_node_list) - 1)
        index_matrix = torch.zeros((num_node_list[-1], num_dim_perms))
        for i in range(self.dim + 1):
            if hasattr(graph, f"x_{i}"):
                index_matrix[num_node_list[i] : num_node_list[i + 1], : math.factorial(i + 1) * (i + 1)] = graph[
                    f"x_{i}"
                ]
        graph.x_ind = index_matrix

        # node_type features
        graph = self.gen_simplicial_type_feat(num_node_list, graph)

        return graph

class NodeSampler:
    def __init__(self, sample_rate=0.4):
        self.sample_rate = sample_rate

    def __call__(self, graph: Data):
        dim1_index = torch.where(graph.node_types == 1)[0]
        dim2_index = torch.where(graph.node_types == 2)[0]

        sampled_dim1_index = self.sample_with_probability(dim1_index)
        sampled_dim2_index = self.sample_with_probability(dim2_index)

        remove_node_indices = torch.cat((sampled_dim1_index, sampled_dim2_index), dim=-1)
        graph = self.sample_graph(graph, remove_node_indices)
        return graph

    def sample_with_probability(self, index):
        prob = torch.ones_like(index) * self.sample_rate + 0.0
        sampled_index = torch.multinomial(prob, num_samples=int(len(index) * self.sample_rate), replacement=False)
        return index[sampled_index]

    def get_new_indices_after_removal(self, total_nodes, remove_node_indices):
        # Start by creating a tensor of -1s
        new_indices = torch.full((total_nodes,), -1, dtype=torch.long)

        # Create a mask where only the nodes we want to keep are marked as True
        keep_mask = torch.ones(total_nodes, dtype=torch.bool)
        keep_mask[remove_node_indices] = False

        # Create new indices for the nodes we want to keep
        new_values = torch.arange(total_nodes)[keep_mask].long()

        # Fill the new_indices tensor at the positions where we want to keep the nodes
        new_indices[keep_mask] = torch.arange(new_values.size(0))

        return new_indices


    def update_edge_index(self, edge_index, new_indices):
        # Remap the old node indices to new node indices
        edge_index[0] = new_indices[edge_index[0]]
        edge_index[1] = new_indices[edge_index[1]]
        
        # Remove edges where either source or target node is one of the removed nodes
        valid_edges = (edge_index[0] != -1) & (edge_index[1] != -1)
        
        return edge_index[:, valid_edges], valid_edges

    def sample_graph(self, data, remove_node_indices):
        num_nodes = data.num_nodes
        if isinstance(remove_node_indices, list):
            remove_node_indices = torch.tensor(remove_node_indices)

        node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        node_mask[remove_node_indices] = 0

        node_feat_names = ['node_types', 'x_ind', 'loc', 'vel', 'charges']
        for feat in node_feat_names:
            masked_feat = data[f'{feat}'][node_mask]
            data[f'{feat}'] = masked_feat

        new_indices = self.get_new_indices_after_removal(num_nodes, remove_node_indices)
        data.edge_index, valid_edges = self.update_edge_index(data.edge_index, new_indices)
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[valid_edges]
        return data