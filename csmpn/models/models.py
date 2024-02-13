import torch
import torch.nn.functional as F

from torch.nn import Embedding
from torch_geometric.nn import global_add_pool, global_mean_pool
from csmpn.models.layers import (
    EGNNLayer, EGCL, CEMLP, ESMPNLayer, cl_flatten, cl_split, MPNNLayer, EGNNBlock
    )
from csmpn.data.ESMPN.utils import compute_invariants_3d, TwoLayerMLP

from engineer.metrics.metrics import Loss, MetricCollection
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor

from csmpn.algebra.cliffordalgebra import CliffordAlgebra
from csmpn.models.cegnn_utils import MVLinear
from torch_scatter import scatter_mean


cls_criterion = torch.nn.CrossEntropyLoss()
bicls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.L1Loss(reduce=False)
msereg_criterion = torch.nn.MSELoss(reduce=False)


def get_loss_func(task_type: str):
    if task_type == 'classification':
        loss_fn = cls_criterion
    elif task_type == 'bin_classification':
        loss_fn = bicls_criterion
    elif task_type == 'regression':
        loss_fn = reg_criterion
    elif task_type == 'mse_regression':
        loss_fn = msereg_criterion
    else:
        raise NotImplementedError('Training on task type {} not yet supported.'.format(task_type))
    return loss_fn

########################### N-body ###########################
class EGNN(nn.Module):
    """ E(n)-equivariant Message Passing Network """
    def __init__(self, invariant_features=2, covariant_features=2, num_hidden=64, out_features=3, num_layers=4, k=None, aggr="mean", act=nn.SiLU, pool=global_add_pool, task_type="mse_regression"):
        super().__init__()
        edge_features = 2
        self.k = k

        self.embedder = nn.Sequential(nn.Linear(invariant_features, num_hidden),
                                      act(),
                                      nn.Linear(num_hidden, num_hidden))
        self.edge_mlp = nn.Sequential(nn.Linear(edge_features, num_hidden),
                                      act(),
                                      nn.Linear(num_hidden, num_hidden))
    
        layers = []
        for i in range(num_layers):
            layers.append(EGNNLayer(num_hidden, covariant_features, edge_features, num_hidden, num_hidden, aggr))
        self.layers = nn.ModuleList(layers)

        self.pooler = pool

        self.head = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                                  act(),
                                  nn.Linear(num_hidden, out_features))
        self.loss_func = get_loss_func(task_type=task_type)
        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )

    def forward(self, batch, step, mode):
        batch_size = batch.ptr.shape[0] - 1
        pos = batch.loc.reshape(batch_size, -1, 3)
        vel = batch.vel
    
        pos = pos.reshape(-1, 3)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        h = torch.cat((torch.norm(vel, dim=-1).unsqueeze(-1), batch.charges), dim=-1)
        # h = torch.norm(vel, dim=-1, keepdim=True)
        h = self.embedder(h)

        for layer in self.layers:
            h, covariants = layer(pos, vel, h, edge_index, edge_attr) 
            pos = covariants[:, 0, :]
            vel = covariants[:, 1, :]

            # # Update graph
            # if self.k:
            #     edge_index = knn_graph(pos, self.k, batch.batch)
            #     dist = torch.sum((pos[edge_index[1]] - pos[edge_index[0]]).pow(2), dim=-1, keepdim=True).sqrt()
        
        # h = self.head(h)
        x = (pos + batch.loc).reshape(-1, 3)

        targets = batch.y.view(-1, 3)
        mask = ~torch.isnan(targets)
        loss = self.loss_func(x[mask].squeeze(), targets[mask].squeeze()) # type: ignore
        backprop_loss = loss.mean()  # []
        return backprop_loss, {"loss": loss}
    

class CLGNN(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_features=28,
        out_features=1,
        edge_features_in=1,
        num_layers=3,
        normalization_init=0,
        residual=True,
        task_type="mse_regression",
        aggr="mean",
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.hidden_features = hidden_features
        self.n_layers = num_layers

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        layers = []

        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                    aggr=aggr,
                )
            )

        self.projection = nn.Sequential(
            MVLinear(self.algebra, hidden_features, out_features),
        )

        self.layers = nn.Sequential(*layers)

        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def _forward(self, h, edges, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)

        h = self.projection(h)
        return h

    def forward(self, batch, step):
        batch_size = batch.ptr.shape[0] - 1
        loc = batch.loc.reshape(batch_size, -1, 3)
        loc_mean = (loc - loc.mean(dim=1, keepdim=True)).reshape(-1, 3)
        vel = batch.vel
    
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        edge_attr = self.algebra.embed(edge_attr[..., None], (0,)) # type: ignore

        invariants = batch.charges
        invariants = self.algebra.embed(invariants, (0,)) # type: ignore

        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.algebra.embed(xv, (1, 2, 3)) # type: ignore

        input = torch.cat([invariants[:, None], covariants], dim=1)

        loc_pred = self._forward(input, edge_index, edge_attr)
        loc_pred = loc_pred[..., 0, 1:4]
        loc = loc.reshape(-1, 3)
        loc_pred = loc + loc_pred

        targets = batch.y.view(-1, 3)
        mask = ~torch.isnan(targets)
        loss = self.loss_func(loc_pred[mask].squeeze(), targets[mask].squeeze()) # type: ignore

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}
    
        #  [loss_batch1 [B], loss_batch2 [B]] -> all_loss (torch.cat) --> reduce (mean)


class SimplicialMPNN(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=2, num_hidden: int=64, num_out: int=3, num_layers: int=7, task_type="mse_regression") -> None:
        super().__init__()
        # embedding layers
        self.feature_embedding = nn.Linear(num_input, num_hidden)
        self.max_dim = max_dim

        # compute adjacency types up to dim
        adj_types = [[f'{dim}_{dim}', f'{dim+1}_{dim}', f'{dim}_{dim+1}'] for dim in range(self.max_dim)]
        adj_types = [adj for sub_list in adj_types for adj in sub_list]
        self.adj_types = adj_types

        self.layers = nn.ModuleList(
            [SimplicialMPLayer(adj_types, self.max_dim, num_hidden) for _ in range(num_layers)]
        )

        # pooling layers
        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_hidden)
            )
        # self.post_pool = nn.Sequential(
        #         nn.Linear((max_dim + 1) * num_hidden, num_hidden),
        #         nn.SiLU(),
        #         nn.Linear(num_hidden, num_out)
        #     )
        self.post_pool = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_out)
            )
        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def forward(self, graph: Data, step):
        # load graph
        loc = (graph.loc.view(-1, 5, 3) - graph.loc.reshape(-1, 5, 3).mean(dim=1).unsqueeze(1)).view(-1, 3)
        feat_in = torch.cat((torch.norm(graph.vel, dim=-1).unsqueeze(-1), graph.charges), dim=-1)

        x = {
            '0': feat_in[(graph.x_0 + graph.x_0_batch.view(-1, 1)*5)[:, 0]],
            '1': (feat_in[(graph.x_1 + graph.x_1_batch.view(-1, 1)*5)[:, 0]] + feat_in[(graph.x_1 + graph.x_1_batch.view(-1, 1)*5)[:, 1]]) / 2,
            '2': (feat_in[(graph.x_2 + graph.x_2_batch.view(-1, 1)*5)[:, 0]] + 
                  feat_in[(graph.x_2 + graph.x_2_batch.view(-1, 1)*5)[:, 1]] + 
                  feat_in[(graph.x_2 + graph.x_2_batch.view(-1, 1)*5)[:, 2]]) / 3,
        }
        # breakpoint()
        # # create norm vec feat combined with averaged charges
        # for dim, feat in x.items():
        #     new_feat = torch.zeros((feat.shape[0], 2)).to(self.feature_embedding.weight.device)
        #     new_feat[:, 0] = torch.norm(feat[:, :3], dim=-1)
        #     new_feat[:, 1] = feat[:, 3]
        #     x[dim] = new_feat

        x_batch = {str(dim): getattr(graph, f'x_{dim}_batch') for dim in range(self.max_dim + 1)}
        adj = {adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adj_types if
               hasattr(graph, f'adj_{adj_type}')}
        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}

        # add in missing adjacencies (i.e. downward communication)
        for dim in range(self.max_dim):
            adj[f'{dim + 1}_{dim}'] = adj[f'{dim}_{dim + 1}'][[1, 0]].clone()

        # message passing
        for layer in self.layers:
            x = layer(x, adj)

        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        # x = {dim: global_mean_pool(x[dim], x_batch[dim]) for dim, feature in x.items()}
        # state = torch.cat(tuple([feature for dim, feature in x.items()]), dim=1)
        state = x['0']
        out = self.post_pool(state)
        out = torch.squeeze(out)
        out += graph.loc.reshape(-1, 3)
        targets = graph.y.view(-1, 3)
        mask = ~torch.isnan(targets)
        loss = self.loss_func(out[mask].squeeze(), targets[mask].squeeze()) # type: ignore

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}

    def __str__(self):
        return f"Simplicial MPNN"


class CliffordSimplicialMPNN(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=3, num_hidden: int=28, num_out: int=1, num_layers: int=3, task_type="mse_regression") -> None:
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.max_dim = max_dim
        self.feature_embedding = CEMLP(algebra=self.algebra, in_features=num_input+self.max_dim+1, hidden_features=num_hidden, out_features=num_hidden, n_layers=1, normalization_init=0,)
        self.sim_type_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=max_dim+1, embedding_dim=num_hidden),
            nn.Linear(num_hidden, max_dim+1)
        )


        # compute adjacency types up to dim
        adj_types = [[f'{dim}_{dim}', f'{dim+1}_{dim}', f'{dim}_{dim+1}'] for dim in range(self.max_dim)]
        adj_types = [adj for sub_list in adj_types for adj in sub_list]
        self.adj_types = adj_types

        self.layers = nn.ModuleList(
            [CliffordSimplicialMPLayer(adj_types, self.max_dim, num_hidden, edge_attr=len(adj_types), node_attr=self.max_dim+1) for _ in range(num_layers)]
        )
        self.out_layer = CEMLP(algebra=self.algebra, in_features=num_hidden, hidden_features=num_hidden, out_features=num_out, n_layers=2, normalization_init=0,)

        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def forward(self, graph: Data, step):
        batch_size = graph.ptr.shape[0] - 1
        loc = graph.loc.reshape(batch_size, -1, 3)
        loc_mean = (loc - loc.mean(dim=1, keepdim=True)).reshape(-1, 3)
        vel = graph.vel

        invariants = graph.charges
        invariants = self.algebra.embed(invariants, (0,)) # type: ignore

        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.algebra.embed(xv, (1, 2, 3)) # type: ignore

        feat_in = torch.cat([invariants[:, None], covariants], dim=1)
        # load graph
        x = {
            '0': feat_in[(graph.x_0 + graph.x_0_batch.view(-1, 1)*5)[:, 0]],
            '1': (feat_in[(graph.x_1 + graph.x_1_batch.view(-1, 1)*5)[:, 0]] + feat_in[(graph.x_1 + graph.x_1_batch.view(-1, 1)*5)[:, 1]]) / 2,
            '2': (feat_in[(graph.x_2 + graph.x_2_batch.view(-1, 1)*5)[:, 0]] + 
                  feat_in[(graph.x_2 + graph.x_2_batch.view(-1, 1)*5)[:, 1]] + 
                  feat_in[(graph.x_2 + graph.x_2_batch.view(-1, 1)*5)[:, 2]]) / 3,
        }
        # x = {
        #     '0': feat_in[(graph.x_0 + graph.x_0_batch.view(-1, 1)*5)[:, 0]],
        #     '1': torch.zeros((graph.x_1.shape[0], 3, 8)).to('cuda'),
        #     '2': torch.zeros((graph.x_2.shape[0], 3, 8)).to('cuda'),
        # }

        # record num of simplicial complexes per dimension.
        num_per_dim = [x[dim].shape[0] for dim in x.keys()]

        # node type
        node_type = {}
        for dim in range(self.max_dim + 1):
            node_type[str(dim)] = torch.tensor([dim]*num_per_dim[dim])

        sim_embedding_dict = []
        for dim in x.keys():
            sim_embedding = self.sim_type_embedding(torch.tensor(int(dim)).to("cuda")).unsqueeze(1)
            sim_invariants = self.algebra.embed(sim_embedding, (0,))
            sim_embedding_dict.append(sim_invariants.unsqueeze(0))
            sim_invariants = sim_invariants.unsqueeze(0).repeat(x[dim].shape[0], 1, 1)
            x[dim] = torch.cat((sim_invariants, x[dim]), dim=1)
        sim_embedding_dict = torch.cat(sim_embedding_dict, dim=0)

        # derive feature matrix, node_attr and edge_attr
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}
        
        node_attr, edge_attr = {}, {}
        for dim in x.keys():
            node_attr[dim] = sim_embedding_dict[node_type[dim]]

        adj = {adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adj_types if
               hasattr(graph, f'adj_{adj_type}')}
        edge_attr_adj = {adj_type: getattr(graph, f'edge_attr_{adj_type}') for adj_type in self.adj_types if
               hasattr(graph, f'edge_attr_{adj_type}')}

        # get edge attribute by concatenating node type embeddings
        for adj_type in edge_attr_adj.keys():
            edge_attr[adj_type] = torch.cat(
                (sim_embedding_dict[edge_attr_adj[adj_type][:, 0]], sim_embedding_dict[edge_attr_adj[adj_type][:, 1]]), 
                dim=1
                )

        # message passing
        for layer in self.layers:
            x = layer(x, adj, edge_attr, node_attr)

        # read out
        out = self.out_layer(x['0'])
        
        loc_pred = out[..., 0, 1:4]
        loc = loc.reshape(-1, 3)
        loc_pred = loc + loc_pred

        targets = graph.y.view(-1, 3)
        mask = ~torch.isnan(targets)
        loss = self.loss_func(loc_pred[mask].squeeze(), targets[mask].squeeze()) # type: ignore

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}

    def __str__(self):
        return f"Clifford Simplicial MPNN for NBody"


class CliffordSharedSimplicialMPNN(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=3, num_hidden: int=28, num_out: int=1, num_layers: int=3, task_type="mse_regression") -> None:
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.max_dim = max_dim
        self.num_node_type = self.max_dim + 1
        self.feature_embedding = CEMLP(algebra=self.algebra, in_features=num_input + self.num_node_type, hidden_features=num_hidden, out_features=num_hidden, n_layers=1, normalization_init=0,)
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.cl_feature_embedding = nn.Sequential(
            MVLinear(self.algebra, num_input, num_input, subspaces=False),
            CEMLP(algebra=self.algebra, in_features=2*num_input, hidden_features=num_hidden, out_features=num_input, n_layers=1, normalization_init=0,),
            CEMLP(algebra=self.algebra, in_features=3*num_input, hidden_features=num_hidden, out_features=num_input, n_layers=1, normalization_init=0,)
        )
        self.feature_embedding = nn.Sequential(
            MVLinear(self.algebra, num_input+self.num_node_type, num_hidden, subspaces=False)
        )
        self.sim_type_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=max_dim+1, embedding_dim=max_dim+1)
        )

        # MPNN layers
        layers = []
        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    num_hidden,
                    num_hidden,
                    num_hidden,
                    edge_attr_features=2*(self.num_node_type),
                    node_attr_features=self.num_node_type,
                )
            )
        self.layers = nn.Sequential(*layers)
        
        # # GIN layers
        # self.layers = nn.ModuleList(
        #     [CliffordSharedSimplicialMPLayer(
        #     num_hidden, edge_attr=2*self.num_node_type, node_attr=self.num_node_type
        #     ) for _ in range(num_layers)]
        # )

        # self.projection = CEMLP(algebra=self.algebra, in_features=num_hidden, hidden_features=num_hidden, out_features=num_out, n_layers=1, normalization_init=0,)
        self.projection = nn.Sequential(
            CEMLP(self.algebra, num_hidden, num_hidden, num_hidden, n_layers=1),
            MVLinear(self.algebra, num_hidden, num_out),
        )
        self.train_metrics = MetricCollection(
            { 
                "loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def forward(self, graph: Data, mode, step):
        batch_size = graph.ptr.shape[0] - 1
        node_label = torch.cat([torch.tensor([graph.x_ind_ptr[i]]*(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i])) for i in range(batch_size)], dim=0).unsqueeze(-1).to(self.sim_type_embedding[0].weight.device).type(torch.long)
        x_ind = (graph.x_ind + node_label).to(torch.long)

        # pos = graph.loc.reshape(batch_size, -1, 3)
        # loc_mean = (pos - graph.loc[torch.where(graph.node_types==0)].reshape(batch_size, -1, 3).mean(dim=1, keepdim=True)).reshape(-1, 3)

        loc_node = graph.loc[torch.where(graph.node_types==0)]
        loc_mean_per_graph = global_mean_pool(loc_node.reshape(-1, 3), graph.batch[torch.where(graph.node_types==0)]).reshape(batch_size, 3) # [B, 3]
        all_loc = torch.cat([loc_mean_per_graph[i].unsqueeze(0).repeat(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i], 1) for i in range(batch_size)], dim=0)
        loc_mean = graph.loc - all_loc

        sim_embedding = self.sim_type_embedding(graph.node_types).unsqueeze(-1)
        sim_invariants = self.algebra.embed(sim_embedding, (0,))
        x = torch.zeros((graph.loc.shape[0], self.num_input, 8)).to(self.sim_type_embedding[0].weight.device)
        for i in range(self.max_dim+1):
            ind = torch.where(graph.node_types == i)[0]
            features = torch.cat(
                [self.algebra.embed(loc_mean[x_ind[ind][:, :i+1]], (1,2,3)),
                 self.algebra.embed(graph.vel[x_ind[ind][:, :i+1]], (1,2,3)),
                 self.algebra.embed(graph.charges[x_ind[ind][:, :i+1]], (0,)),
                 ], dim=1
            )
            x[torch.where(graph.node_types==i)] = self.cl_feature_embedding[i](features)

        node_attr = sim_invariants
        edge_attr = torch.cat((
            self.algebra.embed(self.sim_type_embedding(graph.edge_attr[:, 0]).unsqueeze(-1), (0,)),
            self.algebra.embed(self.sim_type_embedding(graph.edge_attr[:, 1]).unsqueeze(-1), (0,))
        ), dim=1)
         
        x = torch.cat((x, sim_invariants), dim=1)
        x = self.feature_embedding(x)
        # message passing
        for layer in self.layers:
            x = layer(x, graph.edge_index, edge_attr, node_attr)

        # read out
        out = self.projection(x[torch.where(graph.node_types==0)])
        loc_pred = out[..., 0, 1:4]
        loc = graph.loc[torch.where(graph.node_types==0)].reshape(-1, 3)
        loc_pred = loc + loc_pred

        targets = graph.y.view(-1, 3)
        mask = ~torch.isnan(targets)
        loss = self.loss_func(loc_pred[mask].squeeze(), targets[mask].squeeze()) # type: ignore

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}

    def __str__(self):
        return f"Clifford Shared Simplicial MPNN for NBody"


########################### CMU Motion ###########################
class MotionCLGNN(nn.Module):
    def __init__(
        self,
        in_features=2,
        num_hidden=28,
        out_features=1,
        edge_features_in=0,
        num_layers=3,
        normalization_init=0,
        residual=True,
        task_type="mse_regression",
        aggr="mean",
    ):
        super().__init__()
        hidden_features = num_hidden
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.hidden_features = hidden_features
        self.n_layers = num_layers

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        layers = []

        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                    aggr=aggr,
                )
            )

        self.projection = nn.Sequential(
            MVLinear(self.algebra, hidden_features, out_features),
        )

        self.layers = nn.Sequential(*layers)

        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def _forward(self, h, edges):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges)

        h = self.projection(h)
        return h

    def forward(self, batch, step, mode):
        batch_size = batch.ptr.shape[0] - 1
        loc = batch.loc.reshape(batch_size, -1, 3)
        loc_mean = (loc - loc.mean(dim=1, keepdim=True)).reshape(-1, 3)
        vel = batch.vel
    
        edge_index = batch.edge_index

        xv = torch.stack([loc_mean, vel], dim=1)
        inputs = self.algebra.embed(xv, (1, 2, 3)) # type: ignore

        loc_pred = self._forward(inputs, edge_index)
        loc_pred = loc_pred[..., 0, 1:4]
        loc = loc.reshape(-1, 3)
        loc_pred = loc + loc_pred

        targets = batch.y.view(-1, 3)

        loss = F.mse_loss(loc_pred, targets, reduction="none").mean(dim=1)  # [B]

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}
    
        #  [loss_batch1 [B], loss_batch2 [B]] -> all_loss (torch.cat) --> reduce (mean)


class EGNN_motion(nn.Module):
    """ E(n)-equivariant Message Passing Network """
    def __init__(self, invariant_features=1, covariant_features=2, num_hidden=64, out_features=3, num_layers=4, k=None, aggr="mean", act=nn.SiLU, pool=global_add_pool, task_type="mse_regression"):
        super().__init__()
        edge_features = 1
        self.k = k

        self.embedder = nn.Sequential(nn.Linear(invariant_features, num_hidden),
                                      act(),
                                      nn.Linear(num_hidden, num_hidden))
    
        layers = []
        for i in range(num_layers):
            layers.append(EGNNBlock(num_hidden, 1, num_hidden, num_hidden, dim=1, aggr=aggr, act=act))
        self.layers = nn.ModuleList(layers)

        self.pooler = pool

        self.head = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                                  act(),
                                  nn.Linear(num_hidden, out_features))
        self.loss_func = get_loss_func(task_type=task_type)
        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )

    def forward(self, batch, step, mode):
        batch_size = batch.ptr.shape[0] - 1
        pos = batch.loc
        vel = batch.vel
    
        edge_index = batch.edge_index
        h = torch.norm(vel, dim=-1, keepdim=True)
        h = self.embedder(h)

        for layer in self.layers:
            h, pos, vel = layer(h, pos, vel, edge_index, split=True) 

        loc_pred = pos

        targets = batch.y.view(-1, 3)

        loss = F.mse_loss(loc_pred, targets, reduction="none").mean(dim=1)  # [B]

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}
class CliffordSharedSimplicialMPNN_motion(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=2, num_hidden: int=16, num_out: int=1, num_layers: int=4, task_type="mse_regression", stats=None, condition=True) -> None:
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.max_dim = max_dim
        self.condition = condition
        self.num_node_type = self.max_dim + 1 if condition else 0
        self.num_hidden = num_hidden
        self.num_input = num_input
        # self.feature_embedding = CEMLP(algebra=self.algebra, in_features=num_input, hidden_features=num_hidden, out_features=num_hidden, n_layers=1, normalization_init=0,)
        self.feature_embedding = MVLinear(
            self.algebra, num_input + self.num_node_type, num_hidden, subspaces=False
        )
        self.cl_feature_embedding = nn.Sequential(
            MVLinear(self.algebra, num_input, num_input, subspaces=False),
            CEMLP(algebra=self.algebra, in_features=2*num_input, hidden_features=num_hidden, out_features=num_input, n_layers=1, normalization_init=0,),
            CEMLP(algebra=self.algebra, in_features=3*num_input, hidden_features=num_hidden, out_features=num_input, n_layers=2, normalization_init=0,)
        )
        self.sim_type_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=max_dim+1, embedding_dim=max_dim+1),
        )
        self.stats = stats

        # MPNN layers
        layers = []
        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    num_hidden,
                    num_hidden,
                    num_hidden,
                    edge_attr_features=2*(self.num_node_type),
                    node_attr_features=self.num_node_type,
                    aggr="mean",
                    normalization_init=0,
                )
            )
        self.layers = nn.Sequential(*layers)
        
        # # GIN layers
        # self.layers = nn.ModuleList(
        #     [CliffordSharedSimplicialMPLayer(
        #     num_hidden, edge_attr=2*self.num_node_type, node_attr=self.num_node_type
        #     ) for _ in range(num_layers)]
        # )

        # TODO: Now the maximum dimension can only be 2
        self.projection = nn.Sequential(
            MVLinear(self.algebra, num_hidden, num_out),
        )
        self.projection = nn.Sequential(
            CEMLP(self.algebra, num_hidden, num_hidden, num_hidden, n_layers=1),
            MVLinear(self.algebra, num_hidden, num_out),
        )

        self.train_metrics = MetricCollection(
            { 
                "loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def forward(self, graph, step, mode):
        batch_size = graph.ptr.shape[0] - 1
        node_label = torch.cat([torch.tensor([graph.x_ind_ptr[i]]*(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i])) for i in range(batch_size)], dim=0).unsqueeze(-1).to(self.sim_type_embedding[0].weight.device).type(torch.long)
        x_ind = (graph.x_ind + node_label).to(torch.long)

        ##################################### create mean features ####################################
        # # create mean features
        # for i in range(1, self.max_dim+1):
        #     ind = torch.where(graph.node_types == i)[0]
        #     graph.pos[ind] = torch.mean(graph.pos[x_ind[ind][:, :i+1]], dim=1)
        #     graph.vel[ind] = torch.mean(graph.vel[x_ind[ind][:, :i+1]], dim=1)

        # pos = graph.pos.reshape(batch_size, -1, 3)
        # pos_mean = (pos - pos.mean(dim=1, keepdim=True)).reshape(-1, 3)
        # xv = torch.stack((pos_mean, graph.vel), dim=1)
        # # embed feats in clifford space
        # x = self.algebra.embed(xv, (1, 2, 3)) # type: ignore

        # sim_embedding = self.sim_type_embedding(graph.node_types).unsqueeze(-1)
        # sim_invariants = self.algebra.embed(sim_embedding, (0,))
        # x = torch.cat((sim_invariants, x), dim=1) if self.condition else x
        # x = self.feature_embedding(x)
        ##################################### create mean features ####################################

        ##################################### create clifford features ####################################
        pos = graph.pos.reshape(batch_size, -1, 3)
        loc_mean = (pos - graph.pos[torch.where(graph.node_types==0)].reshape(batch_size, -1, 3).mean(dim=1, keepdim=True)).reshape(-1, 3)
        sim_embedding = self.sim_type_embedding(graph.node_types).unsqueeze(-1)
        sim_invariants = self.algebra.embed(sim_embedding, (0,))
        x = torch.zeros((graph.pos.shape[0], self.num_input, 8)).to(self.sim_type_embedding[0].weight.device)
        for i in range(self.max_dim+1):
            ind = torch.where(graph.node_types == i)[0]
            features = torch.cat(
                [self.algebra.embed(loc_mean[x_ind[ind][:, :i+1]], (1,2,3)),
                 self.algebra.embed(graph.vel[x_ind[ind][:, :i+1]], (1,2,3))], dim=1
            )
            x[torch.where(graph.node_types==i)] = self.cl_feature_embedding[i](features)
        ##################################### create clifford features ####################################
        x = torch.cat((x, sim_invariants), dim=1)
        node_attr = sim_invariants if self.condition else None
        edge_attr = torch.cat((
            self.algebra.embed(self.sim_type_embedding(graph.edge_attr[:, 0]).unsqueeze(-1), (0,)),
            self.algebra.embed(self.sim_type_embedding(graph.edge_attr[:, 1]).unsqueeze(-1), (0,))
        ), dim=1) if self.condition else None
        x = self.feature_embedding(x)
        # message passing
        for layer in self.layers:
            x = layer(x, graph.edge_index, edge_attr, node_attr)

        # # message passing with GIN
        # for layer in self.layers:
        #     x = layer(x, graph.edge_index, edge_attr=edge_attr, node_attr=node_attr)

        out = x[torch.where(graph.node_types==0)]
        # batch_size, num_simplex_dim*num_hidden, 8
        out = self.projection(out)
        pred = out[..., 0, 1:4]

        pos = graph.pos.reshape(-1, 3)[torch.where(graph.node_types==0)]
        pred = pos + pred

        targets = graph.y.view(-1, 3)
        mask = ~torch.isnan(targets)
        # loss = self.loss_func(pred[mask].squeeze(), targets[mask].squeeze()) # type: ignore
        loss = F.mse_loss(pred, targets, reduction="none").mean(dim=1)  # [B]

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}

    def __str__(self):
        return f"Clifford Shared Simplicial MPNN for Motion Dataset"
    
class EMPSN_motion(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, max_com: str="1_2", invariant_features=1, covariant_features=2, num_hidden: int=128, out_features=3, num_layers: int=7, dropout_rate: float=0.0, act_fn: str='silu', task_type="mse_regression") -> None:
        super().__init__()

        # compute adjacencies
        adjacencies = []
        max_dim = int(max_com[2])  # max_com = 1_2 --> max_dim = 2
        self.max_dim = max_dim
        inc_final = max_com[0] == max_com[2]

        for dim in range(max_dim + 1):
            if dim < max_dim or inc_final:
                adjacencies.append(f'{dim}_{dim}')

            if dim > 0:
                adjacencies.append(f'{dim-1}_{dim}')
                adjacencies.append(f'{dim}_{dim-1}')

        self.adjacencies = adjacencies

        # layers
        self.feature_embedding = nn.Linear(invariant_features, num_hidden)

        self.layers = nn.ModuleList(
            [ESMPNLayer(adjacencies, self.max_dim, num_hidden, act_fn) for _ in range(num_layers)]
        )
        # self.layers = nn.ModuleList(
        #     [EMPSNLayer_(adjacencies, max_dim, num_hidden, 
        #          covariant_features, num_hidden,
        #          num_hidden, act_fn) for _ in range(num_layers)]
        # )
        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = TwoLayerMLP(num_hidden, num_hidden, num_hidden, act_fn)
        self.post_pool = nn.Sequential(
            nn.Dropout(dropout_rate),
            TwoLayerMLP(num_hidden, num_hidden, 1, act_fn)
        )
        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def forward(self, graph: Data, step, mode):
        batch_size = graph.x_0_ptr.shape[0] - 1
        num_nodes = graph.loc.reshape(graph.x_0_ptr.shape[0]-1, -1, 3).shape[1]
        x_ind = {
            str(i): getattr(graph, f'x_{i}')+num_nodes*graph[f'x_{i}_batch'].unsqueeze(1) for i in range(self.max_dim + 1)
        }
        # compute initial features
        pos = {
            str(i): torch.sum(torch.stack([
                graph.loc[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }

        vel = {
            str(i): torch.sum(torch.stack([
                graph.vel[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }
        x = {
            str(i): torch.norm(vel[f'{i}'], dim=-1, keepdim=True) for i in range(self.max_dim + 1)
        }
        
        adj = {
            adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'adj_{adj_type}')
        }

        inv_ind = {
            adj_type: getattr(graph, f'inv_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'inv_{adj_type}')
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}

        # message passing
        for layer in self.layers:
            inv = compute_invariants_3d(x_ind, pos['0'], adj, inv_ind, graph.loc.device)
            for key, value in inv.items():
                inv[key] = value.reshape(value.shape[0], -1)

            # add in missing adjacencies (i.e. downward communication)
            for i in range(self.max_dim):
                adj[f'{i+1}_{i}'] = adj[f'{i}_{i + 1}'][[1, 0]].clone()
                inv[f'{i+1}_{i}'] = inv[f'{i}_{i + 1}'].clone()
            x, pos, vel = layer(x, adj, inv, pos, vel)
        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        x = self.post_pool(x['0'])
        loc_pred = x*pos['0'] + graph.loc

        targets = graph.y.view(-1, 3)
        loss = F.mse_loss(loc_pred, targets, reduction="none").mean(dim=1)  # [B]
        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}

    def __str__(self):
        return f"EMPSN for Motion ({self.type})"

########################### MD17 ###########################

class EGNN_md17(nn.Module):
    """ E(n)-equivariant Message Passing Network """
    def __init__(self, invariant_features=10, covariant_features=20, num_hidden=64, out_features=30, num_layers=4, aggr="mean", act=nn.SiLU, pool=global_add_pool, task_type="mse_regression"):
        super().__init__()
        edge_features = 10

        self.embedder = nn.Sequential(nn.Linear(invariant_features, num_hidden),
                                      act(),
                                      nn.Linear(num_hidden, num_hidden))
        self.edge_mlp = nn.Sequential(nn.Linear(edge_features, num_hidden),
                                      act(),
                                      nn.Linear(num_hidden, num_hidden))
    
        layers = []
        for i in range(num_layers):
            layers.append(EGNNLayer(num_hidden, covariant_features, edge_features, num_hidden, num_hidden, aggr))
        self.layers = nn.ModuleList(layers)

        self.pooler = pool

        self.head = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                                  act(),
                                  nn.Linear(num_hidden, out_features))
        self.loss_func = get_loss_func(task_type=task_type)
        self.train_metrics = MetricCollection(
            {   
                "loss": Loss(),
                "fde_loss": Loss()
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss()
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss()
            }
        )

    def forward(self, batch, step, mode):
        num_frames = batch.loc.shape[-2]
        batch_size = batch.ptr.shape[0] - 1
        pos = batch.loc.reshape(len(batch.loc), -1)
        vel = batch.vel.reshape(len(batch.vel), -1)

        edge_index = batch.edge_index
        h = torch.norm(batch.vel, dim=-1)
        h = self.embedder(h)

        for layer in self.layers:
            h, covariants = layer(pos, vel, h, edge_index, num_pos=10) 
            covariants = covariants.reshape(len(covariants), 20, -1)
            pos = covariants[:, :10, :]
            vel = covariants[:, 10:20, :]
            pos = pos.reshape(len(pos), -1)
            vel = vel.reshape(len(vel), -1)
        
        pos = pos.reshape(len(pos), -1, 3)
        loc_pred = pos + batch.loc

        targets = batch.y
        ade_loss = torch.sqrt(F.mse_loss(loc_pred.reshape(-1, 3), targets.view(-1, 3), reduction="none").sum(dim=-1)).reshape(batch_size, -1, num_frames).mean(dim=-1).mean(dim=-1)  # [B]
        fde_loss = torch.sqrt(F.mse_loss(loc_pred[:, -1, :], targets[:, -1, :], reduction="none").sum(dim=-1)).reshape(batch_size, -1).mean(dim=-1)

        backprop_loss = ade_loss.mean()  # []

        return backprop_loss, {"loss": ade_loss, "fde_loss": fde_loss}
    

class MD17CLGNN(nn.Module):
    def __init__(
        self,
        in_features=30,
        num_hidden=28,
        out_features=10,
        edge_features_in=0,
        num_layers=3,
        normalization_init=0,
        residual=True,
        task_type="mse_regression",
        aggr="mean",
    ):
        super().__init__()
        hidden_features = num_hidden
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_layers = num_layers

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        layers = []

        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                    aggr=aggr,
                )
            )

        # self.projection = nn.Sequential(
        #     MVLinear(self.algebra, hidden_features, out_features),
        # )
        self.projection = nn.Sequential(
            CEMLP(self.algebra, hidden_features, hidden_features, hidden_features, n_layers=1),
            MVLinear(self.algebra, hidden_features, out_features)
        )

        self.layers = nn.Sequential(*layers)

        self.train_metrics = MetricCollection(
            {   
                "loss": Loss(),
                "ade_loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
                "ade_loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
                "ade_loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def _forward(self, h, edges):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges)

        h = self.projection(h)
        return h

    def forward(self, batch, step, mode):
        num_frames = batch.loc.shape[-2]
        batch_size = batch.ptr.shape[0] - 1
        loc = batch.loc.reshape(batch_size, -1, num_frames, 3)
        loc_mean = loc - (loc.mean(dim=2, keepdim=True)).mean(dim=1, keepdim=True)
        vel = batch.vel.reshape(batch_size, -1, num_frames, 3)
    
        edge_index = batch.edge_index
        invariants = batch.charges.reshape(batch_size, -1).unsqueeze(-1).repeat(1, 1, num_frames).unsqueeze(-1)
        invariants = self.algebra.embed(invariants, (0,)).reshape(batch.loc.shape[0], -1, 8)

        xv = torch.cat([loc_mean, vel], dim=-2)
        xv = self.algebra.embed(xv, (1, 2, 3)).reshape(batch.loc.shape[0], -1, 8)

        inputs = torch.cat([invariants, xv], dim=1)
        loc_pred = self._forward(inputs, edge_index)
        loc_pred = loc_pred[..., 1:4]
        loc = loc.reshape(-1, num_frames, 3)
        loc_pred = loc + loc_pred
        targets = batch.y
        ade_loss = torch.sqrt(F.mse_loss(loc_pred.reshape(-1, 3), targets.view(-1, 3), reduction="none").sum(dim=-1)).reshape(batch_size, -1, num_frames).mean(dim=-1).mean(dim=-1)  # [B]
        fde_loss = torch.sqrt(F.mse_loss(loc_pred[:, -1, :], targets[:, -1, :], reduction="none").sum(dim=-1)).reshape(batch_size, -1).mean(dim=-1)
        loss = F.mse_loss(loc_pred.reshape(-1, 3), targets.view(-1, 3), reduction="none").reshape(batch_size, -1, 3).sum(-1).mean(-1)
        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss, "ade_loss": ade_loss, "fde_loss": fde_loss}
    
        #  [loss_batch1 [B], loss_batch2 [B]] -> all_loss (torch.cat) --> reduce (mean)


class CliffordSharedSimplicialMPNN_md17(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=30, num_hidden: int=28, num_out: int=10, num_layers: int=4, task_type="mse_regression", stats=None, condition=True) -> None:
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.max_dim = max_dim
        self.condition = condition
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_node_type = self.max_dim + 1 if condition else 0
        self.feature_embedding = nn.Sequential(
            MVLinear(self.algebra, num_input + self.num_node_type, num_hidden, subspaces=False)
        )
        self.cl_feature_embedding = nn.Sequential(
            MVLinear(self.algebra, num_input, num_input, subspaces=False), 
            # nn.Identity(),
            CEMLP(algebra=self.algebra, in_features=2*num_input, hidden_features=num_hidden, out_features=num_input, n_layers=1, normalization_init=0,),
            nn.Sequential(
                CEMLP(algebra=self.algebra, in_features=3*num_input, hidden_features=num_hidden, out_features=num_hidden, n_layers=1, normalization_init=0,),
                CEMLP(algebra=self.algebra, in_features=num_hidden, hidden_features=num_hidden, out_features=num_input, n_layers=1, normalization_init=0,)
            )
        )
        self.sim_type_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=max_dim+1, embedding_dim=max_dim+1),
            # nn.Linear(num_hidden, max_dim+1)
        )
        self.stats = stats

        # MPNN layers
        layers = []
        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    num_hidden,
                    num_hidden,
                    num_hidden,
                    edge_attr_features=2*(self.num_node_type),
                    node_attr_features=self.num_node_type,
                    aggr="sum",
                    normalization_init=0,
                )
            )
        self.layers = nn.Sequential(*layers)
        
        # # GIN layers
        # self.layers = nn.ModuleList(
        #     [CliffordSharedSimplicialMPLayer(
        #     num_hidden, 
        #     edge_attr=2*self.num_node_type, 
        #     node_attr=self.num_node_type
        #     ) for _ in range(num_layers)]
        # )

        # TODO: Now the maximum dimension can only be 2
        # self.projection = nn.Sequential(
        #     MVLinear(self.algebra, num_hidden, num_out),
        # )
        self.projection = nn.Sequential(
            CEMLP(self.algebra, num_hidden, num_hidden, num_hidden, n_layers=1),
            MVLinear(self.algebra, num_hidden, num_out),
            # MVSiLU(self.algebra, num_out),
            # MVLayerNorm(self.algebra, num_out),
        )

        self.train_metrics = MetricCollection(
            {   
                "loss": Loss(),
                "ade_loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
                "ade_loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
                "ade_loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def forward(self, graph, step, mode):
        num_frames = graph.loc.shape[1]
        batch_size = graph.ptr.shape[0] - 1
        node_label = torch.cat([torch.tensor([graph.x_ind_ptr[i]]*(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i])) for i in range(batch_size)], dim=0).unsqueeze(-1).to(self.sim_type_embedding[0].weight.device).type(torch.long)
        x_ind = (graph.x_ind + node_label).to(torch.long)

        # ##################################### create mean features ####################################
        # for i in range(1, self.max_dim+1):
        #     ind = torch.where(graph.node_types == i)[0]
        #     graph.loc[ind] = torch.mean(graph.loc[x_ind[ind][:, :i+1]], dim=1)
        #     graph.vel[ind] = torch.mean(graph.vel[x_ind[ind][:, :i+1]], dim=1)
        # loc_mean_value = global_mean_pool(graph.loc.reshape(-1, num_frames*3), graph.batch).reshape(batch_size, num_frames, 3).mean(dim=1).unsqueeze(1).repeat(1, num_frames, 1)  # [B]
        # loc_mean_value_shaped = torch.cat([loc_mean_value[i].unsqueeze(0).repeat(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i], 1, 1) for i in range(batch_size)], dim=0)
        
        # loc_mean = graph.loc - loc_mean_value_shaped
        # xv = torch.cat((loc_mean, graph.vel), dim=-2)
        # xv = loc_mean
        # # embed feats in clifford space
        # x = self.algebra.embed(xv, (1, 2, 3)) # type: ignore

        # sim_embedding = self.sim_type_embedding(graph.node_types).unsqueeze(-1)
        # sim_invariants = self.algebra.embed(sim_embedding, (0,))
        # x = torch.cat((sim_invariants, x), dim=1) if self.condition else x
        # x = self.feature_embedding(x)
        # ##################################### create mean features ####################################

        #################################### create clifford features ####################################
        loc_node = graph.loc[torch.where(graph.node_types==0)]
        loc_mean_per_graph = global_mean_pool(loc_node.reshape(-1, num_frames*3), graph.batch[torch.where(graph.node_types==0)]).reshape(batch_size, num_frames, 3).mean(dim=1, keepdim=True).repeat(1, num_frames, 1)  # [B, 10, 3]
        all_loc = torch.cat([loc_mean_per_graph[i].unsqueeze(0).repeat(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i], 1, 1) for i in range(batch_size)], dim=0)
        loc_mean = graph.loc - all_loc

        sim_embedding = self.sim_type_embedding(graph.node_types).unsqueeze(-1)
        sim_invariants = self.algebra.embed(sim_embedding, (0,))
        # x = torch.zeros((graph.loc.shape[0], num_frames*self.num_input, 8)).to(self.sim_type_embedding[0].weight.device)
        # for i in range(self.max_dim+1):
        #     ind = torch.where(graph.node_types == i)[0]
        #     idx = x_ind[ind][:, :i+1]
        #     loc_feat = self.algebra.embed(loc_mean[idx], (1,2,3)).permute(0, 2, 1, 3).reshape(loc_mean[idx].shape[0]*num_frames, i+1, 8)
        #     vel_feat = self.algebra.embed(graph.vel[idx], (1,2,3)).permute(0, 2, 1, 3).reshape(graph.vel[idx].shape[0]*num_frames, i+1, 8)
        #     charge_feat = self.algebra.embed(graph.charges[idx], (0,)).permute(0, 2, 1, 3).reshape(graph.charges[idx].shape[0]*num_frames, i+1, 8)
        #     features = torch.cat((charge_feat, loc_feat, vel_feat), dim=1)
        #     x[torch.where(graph.node_types==i)] = self.cl_feature_embedding[i](features).reshape(loc_mean[idx].shape[0], num_frames*self.num_input, 8)
        #################################### create clifford features ####################################

        x = torch.zeros((graph.loc.shape[0], self.num_input, 8)).to(self.sim_type_embedding[0].weight.device)
        for i in range(self.max_dim+1):
            ind = torch.where(graph.node_types == i)[0]
            idx = x_ind[ind][:, :i+1]
            if len(idx) != 0:
                features = torch.cat(
                    [self.algebra.embed(loc_mean[idx].reshape(idx.shape[0], -1, 3), (1,2,3)),
                    self.algebra.embed(graph.vel[idx].reshape(idx.shape[0], -1, 3), (1,2,3)),
                    self.algebra.embed(graph.charges[idx].reshape(idx.shape[0], -1, 1), (0,)),
                    ], dim=1
                )
                x[torch.where(graph.node_types==i)] = self.cl_feature_embedding[i](features)

        node_attr = sim_invariants if self.condition else None
        edge_attr = torch.cat((
            self.algebra.embed(self.sim_type_embedding(graph.edge_attr[:, 0]).unsqueeze(-1), (0,)),
            self.algebra.embed(self.sim_type_embedding(graph.edge_attr[:, 1]).unsqueeze(-1), (0,))
        ), dim=1) if self.condition else None

        x = torch.cat((x, sim_invariants), dim=1)
        x = self.feature_embedding(x)

        # message passing
        for layer in self.layers:
            x = layer(x, graph.edge_index, edge_attr, node_attr)

        out = x[torch.where(graph.node_types==0)]
        # batch_size, num_simplex_dim*num_hidden, 8
        out = self.projection(out)
        pred = out[..., 1:4]
        loc = graph.loc[torch.where(graph.node_types==0)]
        loc_pred = loc + pred

        # edges = x[torch.where(graph.node_types==1)]
        # edges = self.projection(edges)
        # edges_loc = edges[..., 1:4]

        # triangles = x[torch.where(graph.node_types==2)]
        # triangles = self.projection(triangles)
        # triangles_loc = triangles[..., 1:4]

        # all_pred_loc = torch.cat((pred, edges_loc, triangles_loc), dim=0)

        # j = 1
        # ind = torch.where(graph.node_types == j)[0]
        # coord_div = all_pred_loc[ind].unsqueeze(1) -  all_pred_loc[x_ind[ind][:, :j+1]]
        # vec_0, vel_1 = coord_div[:, 0, ...], coord_div[:, 1, ...]
        # cosine = torch.sum(vec_0 * vel_1, dim=-1) / (torch.norm(vec_0, dim=-1) * torch.norm(vel_1, dim=-1))
        # cosine_loss = ((cosine.flatten().reshape(batch_size, -1) + 1)**2).mean(dim=-1)
        targets = graph.y
        ade_loss = torch.sqrt(F.mse_loss(loc_pred.reshape(-1, 3), targets.view(-1, 3), reduction="none").sum(dim=-1)).reshape(batch_size, -1, num_frames).mean(dim=-1).mean(dim=-1)  # [B]
        fde_loss = torch.sqrt(F.mse_loss(loc_pred[:, -1, :], targets[:, -1, :], reduction="none").sum(dim=-1)).reshape(batch_size, -1).mean(dim=-1)
        loss = F.mse_loss(loc_pred.reshape(-1, 3), targets.view(-1, 3), reduction="none").reshape(batch_size, -1, 3).sum(-1).mean(-1)
        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss, "ade_loss": ade_loss, "fde_loss": fde_loss}

    def __str__(self):
        return f"Clifford Shared Simplicial MPNN for MD17 Dataset"
    

class EMPSN_md17(nn.Module):
    """
    E(n) Equivariant Message Passing Simplicial Networks (EMPSN)
    """
    def __init__(self, max_com: str="1_2", invariant_features=100, covariant_features=2, num_hidden: int=128, out_features=3, num_layers: int=7, dropout_rate: float=0.0, act_fn: str='silu', task_type="mse_regression") -> None:
        super().__init__()
        self.num_pos = 10
        # compute adjacencies
        adjacencies = []
        max_dim = int(max_com[2])  # max_com = 1_2 --> max_dim = 2
        self.max_dim = max_dim
        inc_final = max_com[0] == max_com[2]

        for dim in range(max_dim + 1):
            if dim < max_dim or inc_final:
                adjacencies.append(f'{dim}_{dim}')

            if dim > 0:
                adjacencies.append(f'{dim-1}_{dim}')
                adjacencies.append(f'{dim}_{dim-1}')

        self.adjacencies = adjacencies

        dict_temp = {
            '0_0': 30,
            '0_1': 30,
            '1_0': 30,
            '1_1': 60,
            '1_2': 60,
            '2_1': 60
        }

        # layers
        self.feature_embedding = nn.Linear(invariant_features, num_hidden)

        self.layers = nn.ModuleList(
            [ESMPNLayer(adjacencies, self.max_dim, num_hidden, act_fn, given_dict_temp=dict_temp, num_pos=10) for _ in range(num_layers)]
        )
        # self.layers = nn.ModuleList(
        #     [EMPSNLayer_(adjacencies, max_dim, num_hidden, 
        #          covariant_features, num_hidden,
        #          num_hidden, act_fn, num_pos=self.num_pos, given_dict_temp=dict_temp) for _ in range(num_layers)]
        # )

        self.pre_pool = nn.ModuleDict()
        for dim in range(self.max_dim+1):
            self.pre_pool[str(dim)] = TwoLayerMLP(num_hidden, num_hidden, num_hidden, act_fn)
        self.post_pool = nn.Sequential(
            nn.Dropout(dropout_rate),
            TwoLayerMLP(num_hidden, num_hidden, out_features*self.num_pos, act_fn)
        )
        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.loss_func = get_loss_func(task_type=task_type)

    def forward(self, graph: Data, step, mode):
        batch_size = graph.x_0_ptr.shape[0] - 1
        num_frames = graph.loc.shape[1]
        num_nodes = graph.loc.shape[0] // batch_size
        x_ind = {
            str(i): getattr(graph, f'x_{i}')+num_nodes*graph[f'x_{i}_batch'].unsqueeze(1) for i in range(self.max_dim + 1)
        }
        # compute initial features
        pos = {
            str(i): torch.sum(torch.stack([
                graph.loc[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }

        vel = {
            str(i): torch.sum(torch.stack([
                graph.vel[x_ind[str(i)][:, k]] for k in range(i+1)], dim=2), 2) / (i+1)
            for i in range(self.max_dim + 1)
        }

        x = {str(i): torch.cdist(pos[f'{i}'] - pos[f'{i}'].mean(dim=1, keepdim=True), pos[f'{i}'] - pos[f'{i}'].mean(dim=1, keepdim=True)) for i in range(self.max_dim + 1)}
        x = {str(i): x[str(i)].reshape(x[str(i)].shape[0], -1) for i in range(self.max_dim+1)}
        adj = {
            adj_type: getattr(graph, f'adj_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'adj_{adj_type}')
        }

        inv_ind = {
            adj_type: getattr(graph, f'inv_{adj_type}') for adj_type in self.adjacencies if hasattr(graph, f'inv_{adj_type}')
        }

        # embed features and E(n) invariant information
        x = {dim: self.feature_embedding(feature) for dim, feature in x.items()}

        for layer in self.layers:
            inv = compute_invariants_3d(x_ind, pos['0'], adj, inv_ind, graph.loc.device)
            for key, value in inv.items():
                inv[key] = value.reshape(value.shape[0], -1)

            # add in missing adjacencies (i.e. downward communication)
            for i in range(self.max_dim):
                adj[f'{i+1}_{i}'] = adj[f'{i}_{i + 1}'][[1, 0]].clone()
                inv[f'{i+1}_{i}'] = inv[f'{i}_{i + 1}'].clone()
            x, pos, vel = layer(x, adj, inv, pos, vel)
        # read out
        x = {dim: self.pre_pool[dim](feature) for dim, feature in x.items()}
        x = self.post_pool(x['0']).reshape(x['0'].shape[0], num_frames, 3)
        loc_pred = x*pos['0'] + graph.loc

        targets = graph.y
        ade_loss = torch.sqrt(F.mse_loss(loc_pred.reshape(-1, 3), targets.view(-1, 3), reduction="none").sum(dim=-1)).reshape(batch_size, -1, num_frames).mean(dim=-1).mean(dim=-1)  # [B]
        fde_loss = torch.sqrt(F.mse_loss(loc_pred[:, -1, :], targets[:, -1, :], reduction="none").sum(dim=-1)).reshape(batch_size, -1).mean(dim=-1)

        backprop_loss = ade_loss.mean()  # []

        return backprop_loss, {"loss": ade_loss, "fde_loss": fde_loss}

    def __str__(self):
        return f"EMPSN for MD17 ({self.type})"
    

class MPNN_md17(nn.Module):
    """ Message Passing Neural Network """
    def __init__(self, node_features=60, edge_features=0, num_hidden=128, out_features=30, num_layers=7, aggr="add", act=nn.ReLU, task_type="mse_regression"):
        """
        Here we choose global_add_pool as our default graph pooling methods,
        but with the other type of tasks, make sure to try also pooling methods like [global_max_pool, global_mean_pool]
        to make your network have specific features.
        """
        super().__init__()
        hidden_features = num_hidden
        self.embedder = nn.Sequential(nn.Linear(node_features, hidden_features),
                                      act(),
                                      nn.Linear(hidden_features, hidden_features))
        self.num_hidden = hidden_features

        layers = []
        for i in range(num_layers):
            layer = MPNNLayer(node_features=hidden_features,
                            hidden_features=hidden_features,
                            edge_features=edge_features,
                            out_features=hidden_features,
                            aggr=aggr,
                            act=act,
                              )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))
        self.train_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
                "fde_loss": Loss(),
            }
        )

    def _forward(self, x, edge_index, edge_attr=None):
        x = self.embedder(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr) 
        x = self.head(x)

        return x
    
    def forward(self, graph, step, mode):
        batch_size = graph.ptr.shape[0] - 1
        num_frames = 10
        x = torch.cat([graph.loc, graph.vel], dim=-1)
        edge_index = graph.edge_index
        edge_attr=None
        x = x.reshape(x.shape[0], -1)
        x = self._forward( x, edge_index, edge_attr)
        x = x.reshape(x.shape[0], -1, 3)
        loc_pred = x + graph.loc

        targets = graph.y
        ade_loss = torch.sqrt(F.mse_loss(loc_pred.reshape(-1, 3), targets.view(-1, 3), reduction="none").sum(dim=-1)).reshape(batch_size, -1, num_frames).mean(dim=-1).mean(dim=-1)  # [B]
        fde_loss = torch.sqrt(F.mse_loss(loc_pred[:, -1, :], targets[:, -1, :], reduction="none").sum(dim=-1)).reshape(batch_size, -1).mean(dim=-1)

        backprop_loss = ade_loss.mean()  # []

        return backprop_loss, {"loss": ade_loss, "fde_loss": fde_loss}
    

########################### GWL test ###########################
class CliffordSharedSimplicialMPNN_GWL(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=1, num_hidden: int=28, num_layers: int=3, task_type="GWL") -> None:
        super().__init__()
        if task_type == "GWL":
            self.num_out = 2
        if task_type == "GEO":
            self.num_out = 1
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.max_dim = max_dim
        self.num_node_type = self.max_dim + 1
        self.feature_embedding = CEMLP(algebra=self.algebra, in_features=num_input + self.num_node_type, hidden_features=num_hidden, out_features=num_hidden, n_layers=1, normalization_init=0,)
        self.num_hidden = num_hidden
        self.cl_feature_embedding = nn.Sequential(
            MVLinear(self.algebra, num_input, num_hidden, subspaces=False),
            CEMLP(algebra=self.algebra, in_features=2*num_input, hidden_features=num_hidden, out_features=num_hidden, n_layers=1, normalization_init=0,),
            CEMLP(algebra=self.algebra, in_features=3*num_input, hidden_features=num_hidden, out_features=num_hidden, n_layers=2, normalization_init=0,)
        )

        self.sim_type_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=max_dim+1, embedding_dim=num_hidden),
            nn.Linear(num_hidden, max_dim+1)
        )

        # MPNN layers
        layers = []
        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    num_hidden,
                    num_hidden,
                    num_hidden,
                    aggr="add",
                )
            )
        self.layers = nn.Sequential(*layers)
        

        self.projection = CEMLP(algebra=self.algebra, in_features=num_hidden, hidden_features=num_hidden, out_features=self.num_out, n_layers=1, normalization_init=0,)

        if self.num_out == 2:
            self.train_metrics = MetricCollection(
                { 
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.test_metrics = MetricCollection(
                {
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.loss_func = nn.CrossEntropyLoss(reduce=False)  
        if self.num_out == 1:
            self.train_metrics = MetricCollection(
                { 
                    "loss": Loss(),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "loss": Loss(),
                }
            )
            self.test_metrics = MetricCollection(
                {
                    "loss": Loss(),
                }
            )
            self.loss_func = torch.nn.MSELoss()

    def forward(self, graph: Data, mode, step):
        
        batch_size = graph.ptr.shape[0] - 1
        node_label = torch.cat([torch.tensor([graph.x_ind_ptr[i]]*(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i])) for i in range(batch_size)], dim=0).unsqueeze(-1).to(self.sim_type_embedding[0].weight.device).type(torch.long)
        x_ind = (graph.x_ind + node_label).to(torch.long)
        # noise = torch.rand((graph.pos.shape[0], 3)).to("cuda")*1000
        # graph.pos = graph.pos + noise

        ##################################### create clifford features ####################################
        graph.pos[torch.where(graph.node_types == 0)].reshape(batch_size, 6, 3)[:, :2, :] = torch.rand((batch_size, 1, 3)).repeat(1, 2, 1)
        graph.pos[torch.where(graph.node_types == 0)].reshape(batch_size, 6, 3)[:, 2:, :] = torch.rand((batch_size, 1, 3)).repeat(1, 4, 1)
        loc_mean_value = global_mean_pool(graph.pos, graph.batch).reshape(batch_size, 3)
        loc_mean_value_shaped = torch.cat([loc_mean_value[i].unsqueeze(0).repeat(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i], 1) for i in range(batch_size)], dim=0)
        
        loc_mean = graph.pos - loc_mean_value_shaped

        x = torch.zeros((graph.pos.shape[0], self.num_hidden, 8)).to(self.sim_type_embedding[0].weight.device)
        for i in range(self.max_dim+1):
            ind = torch.where(graph.node_types == i)[0]
            features = self.algebra.embed(loc_mean[x_ind[ind][:, :i+1]], (1,2,3))
            x[torch.where(graph.node_types==i)] = self.cl_feature_embedding[i](features)
        ##################################### create clifford features ####################################

        # # message passing
        for layer in self.layers:
            x = layer(x, graph.edge_index)

        # read out
        out = cl_split(global_mean_pool(cl_flatten(x), graph.batch))
        out = self.projection(out)
        if self.num_out == 2:
            preds = out[:, :, 0]
            targets = graph.y.reshape(batch_size, -1)
            loss = self.loss_func(preds, targets) # type: ignore

            backprop_loss = loss.mean()  # []
            pred_labels = torch.max(preds, dim=-1)[1]
            true_labels = torch.max(targets, dim=-1)[1]
            acc = (pred_labels == true_labels) + 0.

            return backprop_loss, {"loss": loss, "acc": acc}
        if self.num_out == 1:
            preds = out[:, 0, 1:4]
            targets = graph.y.reshape(batch_size, -1)
            loss = F.mse_loss(preds, targets, reduction="none").mean(dim=1)  # [B]

            backprop_loss = loss.mean()  # []

            return backprop_loss, {"loss": loss}

    def __str__(self):
        return f"Clifford Shared Simplicial MPNN for GWL"
    
class SharedSimplicialMPNN_GWL(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=3, num_hidden: int=28, num_layers: int=3, task_type="GWL") -> None:
        super().__init__()
        if task_type == "GWL":
            self.num_out = 2
        if task_type == "GEO":
            self.num_out = 1
        self.max_dim = max_dim
        self.num_node_type = self.max_dim + 1
        self.feature_embedding = nn.Linear(num_input + self.num_node_type, num_hidden)
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.cl_feature_embedding = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.Linear(2*num_input, num_hidden),
            nn.Linear(3*num_input, num_hidden)
        )

        layers = []
        for i in range(num_layers):
            layer = MPNNLayer(
                node_features=num_hidden, 
                edge_features=0, 
                hidden_features=num_hidden, 
                out_features=num_hidden, 
                aggr="add"
                )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        
        self.projection = nn.Linear(num_hidden, self.num_out)

        if self.num_out == 2:
            self.train_metrics = MetricCollection(
                { 
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.test_metrics = MetricCollection(
                {
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.loss_func = nn.CrossEntropyLoss(reduce=False)  
        if self.num_out == 1:
            self.train_metrics = MetricCollection(
                { 
                    "loss": Loss(),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "loss": Loss(),
                }
            )
            self.test_metrics = MetricCollection(
                {
                    "loss": Loss(),
                }
            )
            self.loss_func = torch.nn.MSELoss()

    def forward(self, graph: Data, mode, step):
        batch_size = graph.ptr.shape[0] - 1
        node_label = torch.cat([torch.tensor([graph.x_ind_ptr[i]]*(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i])) for i in range(batch_size)], dim=0).unsqueeze(-1).to("cuda").type(torch.long)
        x_ind = (graph.x_ind + node_label).to(torch.long)
        # noise = torch.rand((graph.pos.shape[0], 3)).to("cuda")*1000
        # graph.pos = graph.pos + noise

        graph.pos[torch.where(graph.node_types == 0)].reshape(batch_size, 6, 3)[:, :2, :] = torch.rand((batch_size, 1, 3)).repeat(1, 2, 1)
        graph.pos[torch.where(graph.node_types == 0)].reshape(batch_size, 6, 3)[:, 2:, :] = torch.rand((batch_size, 1, 3)).repeat(1, 4, 1)
        loc_mean_value = global_mean_pool(graph.pos, graph.batch).reshape(batch_size, 3)
        loc_mean_value_shaped = torch.cat([loc_mean_value[i].unsqueeze(0).repeat(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i], 1) for i in range(batch_size)], dim=0)
        loc_mean = graph.pos - loc_mean_value_shaped

        x = torch.zeros((graph.pos.shape[0], self.num_hidden)).to("cuda")
        for i in range(self.max_dim+1):
            ind = torch.where(graph.node_types == i)[0]
            features = loc_mean[x_ind[ind][:, :i+1]]
            if len(features) != 0:
                features = features.reshape(features.shape[0], -1)
                x[torch.where(graph.node_types==i)] = self.cl_feature_embedding[i](features)

        # # message passing
        for layer in self.layers:
            x = layer(x, graph.edge_index)

        # read out
        out = global_mean_pool(x, graph.batch)
        out = self.projection(out)
        if self.num_out == 2:
            preds = out
            targets = graph.y.reshape(batch_size, -1)
            loss = self.loss_func(preds, targets) # type: ignore

            backprop_loss = loss.mean()  # []
            pred_labels = torch.max(preds, dim=-1)[1]
            true_labels = torch.max(targets, dim=-1)[1]
            acc = (pred_labels == true_labels) + 0.

            return backprop_loss, {"loss": loss, "acc": acc}
        if self.num_out == 1:
            preds = out
            targets = graph.y.reshape(batch_size, -1)
            loss = F.mse_loss(preds, targets, reduction="none").mean(dim=1)  # [B]

            backprop_loss = loss.mean()  # []

            return backprop_loss, {"loss": loss}

class CLGNN_GWL(nn.Module):
    def __init__(
        self,
        in_features=1,
        num_hidden=28,
        edge_features_in=0,
        num_layers=3,
        normalization_init=0,
        residual=True,
        task_type="GWL",
        aggr="add",
    ):
        super().__init__()
        if task_type == "GWL":
            self.out_features = 2
        if task_type == "GEO":
            self.out_features = 1
        hidden_features = num_hidden
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.hidden_features = hidden_features
        self.n_layers = num_layers

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        layers = []

        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                    aggr=aggr,
                )
            )

        self.projection = nn.Sequential(
            MVLinear(self.algebra, hidden_features, self.out_features),
        )

        self.layers = nn.Sequential(*layers)


        if self.out_features == 2:
            self.train_metrics = MetricCollection(
                { 
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.test_metrics = MetricCollection(
                {
                    "loss": Loss(),
                    "acc": Loss(),
                }
            )
            self.loss_func = nn.CrossEntropyLoss(reduce=False)  
        if self.out_features == 1:
            self.train_metrics = MetricCollection(
                { 
                    "loss": Loss(),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "loss": Loss(),
                }
            )
            self.test_metrics = MetricCollection(
                {
                    "loss": Loss(),
                }
            )
            self.loss_func = torch.nn.MSELoss()

    def _forward(self, h, edges):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges)
        return h

    def forward(self, batch, step, mode):
        batch_size = batch.ptr.shape[0] - 1
        loc = batch.pos.reshape(batch_size, -1, 3)
        loc_mean = (loc - loc.mean(dim=1, keepdim=True)).reshape(-1, 3)
        rand_0 = torch.rand((batch_size, 1, 3)).to("cuda").repeat(1, 2, 1)
        rand_1 = torch.rand((batch_size, 1, 3)).to("cuda").repeat(1, 4, 1)
        loc[:, :2, :] = rand_0
        loc[:, 2:, :] = rand_1
        edge_index = batch.edge_index

        covariants = self.algebra.embed(loc_mean, (1, 2, 3)).unsqueeze(1) # type: ignore

        x = self._forward(covariants, edge_index)
        out = cl_split(global_mean_pool(cl_flatten(x), batch.batch))
        out = self.projection(out)
        if self.out_features == 2:
            preds = out[:, :, 0]
            targets = batch.y.reshape(batch_size, -1)
            loss = self.loss_func(preds, targets) # type: ignore

            backprop_loss = loss.mean()  # []
            pred_labels = torch.max(preds, dim=-1)[1]
            true_labels = torch.max(targets, dim=-1)[1]
            acc = (pred_labels == true_labels) + 0.

            return backprop_loss, {"loss": loss, "acc": acc}
        if self.out_features == 1:
            preds = out[:, 0, 1:4]
            targets = batch.y.reshape(batch_size, -1)
            loss = F.mse_loss(preds, targets, reduction="none").mean(dim=1)  # [B]

            backprop_loss = loss.mean()  # []

            return backprop_loss, {"loss": loss}

    def __str__(self):
        return f"Clifford Shared Simplicial MPNN for GWL"
    
    
        #  [loss_batch1 [B], loss_batch2 [B]] -> all_loss (torch.cat) --> reduce (mean)


class MPNN_GWL(nn.Module):
    """ Message Passing Neural Network """
    def __init__(self, node_features=3, edge_features=0, num_hidden=128, out_features=2, num_layers=7, aggr="add", act=nn.SiLU, task_type="mse_regression"):
        """
        Here we choose global_add_pool as our default graph pooling methods,
        but with the other type of tasks, make sure to try also pooling methods like [global_max_pool, global_mean_pool]
        to make your network have specific features.
        """
        super().__init__()
        hidden_features = num_hidden
        self.embedder = nn.Sequential(nn.Linear(node_features, hidden_features),
                                      act(),
                                      nn.Linear(hidden_features, hidden_features))
        self.num_hidden = hidden_features

        layers = []
        for i in range(num_layers):
            layer = MPNNLayer(node_features=hidden_features,
                            hidden_features=hidden_features,
                            edge_features=edge_features,
                            out_features=hidden_features,
                            aggr=aggr,
                            act=act,
                              )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                  act(),
                                  nn.Linear(hidden_features, out_features))
        self.train_metrics = MetricCollection(
            { 
                "loss": Loss(),
                "acc": Loss(),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "loss": Loss(),
                "acc": Loss(),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "loss": Loss(),
                "acc": Loss(),
            }
        )
        self.loss_func = nn.CrossEntropyLoss(reduce=False)

    def _forward(self, x, edge_index, edge_attr=None):
        x = self.embedder(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr) 


        return x
    
    def forward(self, graph, step, mode):
        batch_size = graph.ptr.shape[0] - 1
        x = graph.pos
        edge_index = graph.edge_index

        x = self._forward( x, edge_index)
        x = global_mean_pool(x, graph.batch)
        preds = self.head(x)

        targets = graph.y
        targets = graph.y.reshape(batch_size, -1)
        loss = self.loss_func(preds, targets) # type: ignore

        backprop_loss = loss.mean()  # []
        pred_labels = torch.max(preds, dim=-1)[1]
        true_labels = torch.max(targets, dim=-1)[1]
        acc = (pred_labels == true_labels) + 0.

        return backprop_loss, {"loss": loss, "acc": acc}