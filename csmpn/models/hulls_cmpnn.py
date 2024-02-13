import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch.nn.functional as F

from csmpn.algebra.cliffordalgebra import CliffordAlgebra
from csmpn.models.models import MVLinear, CEMLP, cl_flatten, cl_split
from engineer.metrics.metrics import Loss, MetricCollection

    

# class CEBlock(nn.Module):
#     def __init__(self, algebra, in_features, out_features, normalization_init=0):
#         super().__init__()

#         self.algebra = algebra
#         self.in_features = in_features
#         self.out_features = out_features

#         if in_features != out_features:
#             self.qkv = MVLinear(algebra, in_features, out_features * 3)
#         else:
#             self.qkv = MVLinear(algebra, in_features, out_features * 2)
#         self.gp = MVGeometricProduct(
#             algebra,
#             out_features,
#             normalization_init=normalization_init,
#         )
#         self.silu = MVSiLU(algebra, out_features)
#         self.norm = MVLayerNorm(algebra, out_features)

#     def forward(self, input):
#         if self.in_features == self.out_features:
#             q, v = self.qkv(input).chunk(2, dim=1)
#             k = input

#         else:
#             q, k, v = self.qkv(input).chunk(3, dim=1)
#         q = self.silu(q)
#         kv = self.norm(self.gp(k, v))
#         return q + kv


# class CEBlock(nn.Module):
#     def __init__(self, algebra, in_features, out_features, normalization_init=0):
#         super().__init__()

#         self.algebra = algebra
#         self.in_features = in_features
#         self.out_features = out_features

#         self.block = nn.Sequential(
#             MVLinear(self.algebra, in_features, out_features),
#             MVSiLU(self.algebra, out_features),
#             SteerableGeometricProductLayer(
#                 self.algebra,
#                 out_features,
#                 normalization_init=normalization_init,
#             ),
#             MVLayerNorm(self.algebra, out_features),
#             MVSiLU(self.algebra, out_features),
#         )

#     def forward(self, input):
#         return self.block(input)


# class CEMLP(nn.Module):
#     def __init__(
#         self,
#         algebra,
#         in_features,
#         hidden_features,
#         out_features,
#         n_layers=2,
#         normalization_init=0,
#     ):
#         super().__init__()
#         self.algebra = algebra
#         self.in_features = in_features
#         self.hidden_features = hidden_features
#         self.out_features = out_features
#         self.n_layers = n_layers

#         layers = []

#         # Add geometric product layers.
#         for i in range(n_layers - 1):
#             layers.append(
#                 CEBlock(
#                     self.algebra,
#                     in_features,
#                     hidden_features,
#                     normalization_init=normalization_init,
#                 )
#             )
#             in_features = hidden_features

#         # Add final layer.
#         layers.append(
#             CEBlock(
#                 self.algebra,
#                 in_features,
#                 out_features,
#                 normalization_init=normalization_init,
#             )
#         )
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


class EGCL(MessagePassing):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        edge_attr_features=0,
        node_attr_features=0,
        residual=True,
        normalization_init=0,
        aggr="mean",
    ):
        super().__init__(aggr=aggr)
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.edge_model = CEMLP(
            algebra,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

        self.node_model = CEMLP(
            algebra,
            self.in_features + self.out_features + node_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )
        self.algebra = algebra

    def message(self, h_i, h_j, edge_attr=None):
        # h_i, h_j = cl_split(h_i), cl_split(h_j)
        h_i, h_j = h_i.reshape(len(h_i), -1, 32), h_j.reshape(len(h_j), -1, 32)
        if edge_attr is None:
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)
        h_msg = self.edge_model(input)
        h_msg = cl_flatten(h_msg)
        return h_msg

    def update(self, h_agg, h, node_attr):
        # h_agg, h = cl_split(h_agg), cl_split(h)
        h_agg, h = h_agg.reshape(len(h_agg), -1, 32), h.reshape(len(h), -1, 32)
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)
        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h
        out_h = cl_flatten(out_h)
        return out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        h = cl_flatten(h)
        x = self.propagate(
            h=h, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr
        )
        # x = cl_split(x)
        x = x.reshape(len(x), -1, 32)

        return x


class HullsCMPNN(nn.Module):
    def __init__(
        self,
        in_features=1,
        hidden_features=32,
        out_features=1,
        edge_features_in=0,
        num_layers=3,
        normalization_init=0,
        residual=True,
        aggr="mean",
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0, 1.0, 1.0))
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

        self.train_metrics = self._setup_metrics()
        self.test_metrics = self._setup_metrics()
        self.loss_func = nn.MSELoss(reduction="none")

    def _setup_metrics(self):
        return MetricCollection({"loss": Loss()})

    def _forward(self, h, edges):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges)

        h = self.projection(h)
        return h

    def forward(self, batch, step, mode):

        batch_size = batch.ptr.shape[0] - 1

        input = batch.input.reshape(batch_size, -1, 5)
        mean_input = input.mean(dim=1, keepdims=True)
        input = input - mean_input
        input = input.reshape(-1, 1, 5)

        input = self.algebra.embed_grade(input, 1)

        pred = self._forward(input, batch.edge_index)

        pred = pred[:, :, 0]


        pred = global_mean_pool(pred, batch.batch)

        loss = F.mse_loss(pred.squeeze(-1), batch.target, reduction='none')

        backprop_loss = loss.mean(0)  # []

        return backprop_loss, {'loss': loss}
