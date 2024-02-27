import torch
from torch import nn
import torch.nn.functional as F
from csmpn.algebra.cliffordalgebra import CliffordAlgebra
from csmpn.models.cegnn_utils import MVLinear, EGCL, CEMLP
from engineer.metrics.metrics import MetricCollection, Loss
from torch_geometric.nn import global_mean_pool
import math
import itertools


class HullsCliffordSharedSimplicialMPNN(nn.Module):
    def __init__(
        self,
        in_features=1,
        hidden_features=28,
        out_features=1,
        edge_features_in=0,
        num_layers=3,
        normalization_init=0,
        residual=True,
        aggr="mean",
        condition=True,
        max_dim: int = 2,
    ):
        super().__init__()
        self.max_dim = max_dim
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0, 1.0, 1.0))
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.n_layers = num_layers
        self.num_node_type = self.max_dim + 1 if condition else 0

        self.cl_feature_embedding = nn.ModuleList(
            [
                MVLinear(
                    self.algebra,
                    in_features,
                    hidden_features,
                    subspaces=False,
                )
            ] + [
                CEMLP(
                    algebra=self.algebra,
                    in_features=(i+1) * in_features,
                    hidden_features=hidden_features,
                    out_features=hidden_features,
                    n_layers=i,
                    normalization_init=0,
                ) for i in range(1, self.max_dim+1)
            ]
        )

        layers = []

        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_attr_features=2*(self.num_node_type),
                    node_attr_features=self.num_node_type,
                    residual=residual,
                    normalization_init=normalization_init,
                    aggr=aggr,
                )
            )

        self.projection = nn.Sequential(
            MVLinear(self.algebra, hidden_features, out_features),
        )

        self.readout = nn.Linear(3, 1)

        self.layers = nn.Sequential(*layers)

        self.train_metrics = self._setup_metrics()
        self.val_metrics = self._setup_metrics()
        self.test_metrics = self._setup_metrics()
        self.loss_func = nn.MSELoss(reduction="none")

    def _setup_metrics(self):
        return MetricCollection(
            {"loss": Loss()}
        )

    def _forward(self, h, edges, node_attr=None, edge_attr=None):
        for layer in self.layers:
            h = layer(h, edges, node_attr=node_attr, edge_attr=edge_attr)

        h = self.projection(h)
        return h

    def embed_simplicial_complex(self, graph):
        # For each node, indicates the starting index of the graph it belongs to.
        graph_start_idx = graph.x_ind_ptr[:-1][graph.x_ind_batch]

        # Gets the vertex indices for each simplex in the graph.
        simplex_indices = graph.x_ind.long() + graph_start_idx.unsqueeze(-1)

        input = torch.zeros(
            (len(graph.x_ind), self.hidden_features, 2**self.algebra.dim),
            device=graph.batch.device,
        )
        for d in range(self.max_dim + 1):
            
            d_simplices = simplex_indices[graph.node_types == d, : d + 1]
            pos = graph.input[d_simplices]

            perm_dim = d + 1

            index_list = list(range(perm_dim))
            index_permutations = torch.tensor(list(itertools.permutations(index_list)), device=graph.x_ind.device)
            pos = pos[:, index_permutations, ...].reshape(d_simplices.shape[0]*math.factorial(d + 1), d+1, -1)
            # Clifford embedding
            pos = self.algebra.embed_grade(pos, 1)

            # Concatenate
            features = pos
            embedding = self.cl_feature_embedding[d](features).reshape(d_simplices.shape[0], math.factorial(d + 1), -1, 2**self.algebra.dim).sum(dim=1)
            input[graph.node_types == d] = embedding

        return input

    def embed_simplex_types(self, batch):
        node_attr = (
            F.one_hot(batch.node_types, self.num_node_type).float().unsqueeze(-1)
        )
        node_attr = self.algebra.embed_grade(node_attr, 0)

        edge_attr = torch.cat(
            (
                node_attr[batch.edge_index[0]],
                node_attr[batch.edge_index[1]],
            ),
            dim=1,
        )
        return node_attr, edge_attr
    
    def forward(self, batch, step, mode):
        batch_size = batch.ptr.shape[0] - 1

        node_pos = batch.input[batch.node_types == 0].reshape(batch_size, -1, self.algebra.dim)
        node_pos_mean = node_pos.mean(dim=1, keepdim=True)
        node_pos_sub = node_pos - node_pos_mean
        batch.input[batch.node_types == 0] = node_pos_sub.reshape(-1, self.algebra.dim)

        x = self.embed_simplicial_complex(batch)
        node_attr, edge_attr = self.embed_simplex_types(batch)

        pred = self._forward(x, batch.edge_index, node_attr, edge_attr)

        pred = pred[:, :, 0]

        # all aggregated at once
        pred = global_mean_pool(pred, batch.x_ind_batch)

        loss = F.mse_loss(pred.squeeze(-1), batch.target, reduction='none')

        backprop_loss = loss.mean(0)  # []

        return backprop_loss, {'loss': loss}
