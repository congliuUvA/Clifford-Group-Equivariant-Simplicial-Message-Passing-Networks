import torch

from typing import Callable, Optional, Tuple, Dict, List
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN, Identity
from torch_scatter import scatter
from abc import ABC, abstractmethod
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from csmpn.data.ESMPN.utils import TwoLayerMLP

from torch_geometric.nn import global_mean_pool, MessagePassing, GINConv
from torch_scatter import scatter_add, scatter_mean
from csmpn.algebra.cliffordalgebra import CliffordAlgebra
# import torch_sparse
import math

algebra = CliffordAlgebra((1, 1, 1))

def cl_flatten(h):
    batch_size = h.shape[0]
    return h.reshape(batch_size, -1)

def cl_split(h, algebra=CliffordAlgebra((1, 1, 1))):
    num_bases = 2 ** algebra.dim
    batch_size = h.shape[0]
    return h.reshape(batch_size, -1, num_bases)

def get_invariants(input, algebra):
    norms = algebra.qs(input, grades=algebra.grades[1:])
    return torch.cat([input[..., :1], *norms], dim=-1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class EMLPLayer(nn.Module):
    def __init__(
        self,
        x_features_in,
        x_features_out,
        h_features_in,
        h_features_out,
        hidden_features,
        vector_act=None,
    ):

        super().__init__()

        layers = []

        self.h_features_in = h_features_in
        self.h_features_out = h_features_out
        self.x_features_in = x_features_in
        self.x_features_out = x_features_out

        self.layers = nn.ModuleList(layers)

        self.coord_mlp = nn.Sequential(
            nn.Linear(h_features_out, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, x_features_in * x_features_out),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(h_features_in, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, h_features_out),
            nn.SiLU(),
        )

        if vector_act is None:
            self.vector_act = nn.Identity()
        else:
            self.vector_act = vector_act()

    def forward(self, x, h):

        x = x.view(len(x), self.x_features_in, -1)

        h = self.edge_mlp(h)
        phi = 1e-2 * self.coord_mlp(h).view(len(x), self.x_features_out, self.x_features_in)

        x = torch.einsum("bid, boi->bod", x, phi)
        # x = torch.clamp(x, -64, 64)
        x = self.vector_act(x)

        x = x.reshape(len(x), -1)
        return x, h
    

class MPNNLayer(MessagePassing):
    """ Message Passing Layer """
    def __init__(self, node_features, edge_features, hidden_features, out_features, aggr="add", act=nn.SiLU):
        super().__init__(aggr=aggr)
        self.node_features = node_features
        self.message_net = nn.Sequential(nn.Linear(2 * node_features + edge_features, hidden_features),
                                         act(),
                                         nn.Linear(hidden_features, hidden_features))
    
        self.update_net = nn.Sequential(nn.Linear(node_features + hidden_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, out_features))
    
        self.num_hidden = hidden_features
    def forward(self, x, edge_index, edge_attr=None):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j, edge_attr):
        # x_i = x_i.reshape(x_i.shape[0], -1, self.num_hidden)
        # x_j = x_j.reshape(x_j.shape[0], -1, self.num_hidden)
        """ Construct messages between nodes """
        input = [x_i, x_j] if edge_attr is None else [x_i, x_j, edge_attr]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)
        message = message.reshape(message.shape[0], -1)
        return message

    def update(self, message, x):
        """ Update node features """
        message = message.reshape(message.shape[0], -1, self.num_hidden)
        x = x.reshape(x.shape[0], -1, self.node_features)
        input = torch.cat((x, message), dim=-1)
        update = self.update_net(input)
        update = update.reshape(update.shape[0], -1)
        return update
    

class CliffordEquivariantGatingBlock(nn.Module):
    def __init__(self, algebra, in_features, out_features):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features

        self.probe = MVLinear(self.algebra, in_features, out_features)
        self.match = MVLinear(self.algebra, in_features, out_features)

        bilinear_form = self.algebra.cayley[
            range(2**self.algebra.dim), 0, range(2**self.algebra.dim)
        ]
        self.register_buffer("bilinear_form", bilinear_form)

    def _smooth_abs_sqrt(self, input, eps=1e-16):
        return (input**2 + eps) ** 0.25

    def forward(self, x):
        probe = self.probe(x)
        match = self.match(x)

        # b = self.algebra.b(probe, match).squeeze(-1)
        b = torch.einsum(
            "...i, i, ...i->...", self.algebra.beta(probe), self.bilinear_form, match
        )

        norm = self._smooth_abs_sqrt(
            torch.einsum("...i, i, ...i->...", match, self.bilinear_form, match)
        )

        match = match / norm[..., None]

        # k = kernel(probe, (1, 1, 1))
        # result = torch.einsum("...ij, ...j->...i", k, match)
        # result = (k @ match[..., None]).squeeze(-1)
        result = self.algebra.geometric_product(probe, match)

        # result = probe - self.algebra.geometric_product(probe, match)

        # result = probe
        # result[b < 0] = result[b < 0] - self.algebra.geometric_product(
        #     probe[b < 0], match[b < 0]
        # )

        result[b > 0] = probe[b > 0]

        return result / math.sqrt(2)


class EGNNLayer(MessagePassing):
    """ E(n)-equivariant Message Passing Layer """
    def __init__(self, invariant_features, covariant_features, edge_features, hidden_features, out_features, aggr):
        super().__init__(aggr=aggr)
        self.node_f = invariant_features
        self.edge_f = edge_features
        self.cov_f = covariant_features
        self.hidden_f = hidden_features
        self.message_net = EMLPLayer(x_features_in=covariant_features*2, x_features_out=covariant_features, 
                                     h_features_in=invariant_features*2 + edge_features, h_features_out=out_features, hidden_features=hidden_features)
    
        self.update_net = EMLPLayer(x_features_in=covariant_features*2, x_features_out=covariant_features, 
                                    h_features_in=hidden_features, h_features_out=out_features, hidden_features=hidden_features)


    def forward(self, pos, vel, h, edge_index, edge_attr=None, num_pos=1):
        invariants, covariants = self.propagate(edge_index, h=h, pos=pos, vel=vel, edge_attr=edge_attr, num_pos=num_pos)
        return invariants, covariants

    def message(self, h_i, h_j, pos_i, pos_j, vel_i, vel_j, edge_attr, num_pos):
        if num_pos > 1:
            pos_i = pos_i.reshape(len(pos_i), num_pos, -1)
            pos_j = pos_j.reshape(len(pos_j), num_pos, -1)
            vel_i = vel_i.reshape(len(vel_i), num_pos, -1)
            vel_j = vel_j.reshape(len(vel_j), num_pos, -1)
            dist = torch.norm((pos_i - pos_j), dim=-1)
        else:
            """ Create messages """
            dist = torch.norm((pos_i - pos_j), dim=-1).unsqueeze(1)
        edge_attr = torch.cat((edge_attr, dist), dim=-1) if edge_attr else dist
        invariants = [h_i, h_j] if edge_attr is None else [h_i, h_j, edge_attr]
        invariants = torch.cat(invariants, dim=-1)
        covariants = torch.cat([pos_i, pos_j, vel_i, vel_j], dim=1)
        covariants, invariants = self.message_net(covariants, invariants)
        message = torch.cat((invariants, covariants), dim=-1)
        return message

    def update(self, message, pos, vel, h, num_pos):
        if num_pos > 1:
            pos = pos.reshape(len(pos), num_pos, -1)
            vel = vel.reshape(len(vel), num_pos, -1)
        # split the features out
        invariants = message[:, :self.hidden_f]
        covariants = message[:, self.hidden_f:]

        # concat the covariant features and update again
        covariants = covariants.reshape(len(covariants), self.cov_f, -1)
        if num_pos > 1:
            covariants = torch.cat([covariants, pos, vel], dim=1)
        else:
            covariants = torch.cat([covariants, pos.unsqueeze(1), vel.unsqueeze(1)], dim=1)
        covariants = covariants.reshape(len(covariants), -1)
        covariants, invariants = self.update_net(covariants, invariants)
        covariants = covariants.reshape(len(covariants), self.cov_f, -1)

        # residue connection
        invariants = invariants + h
        if num_pos > 1:
            covariants = covariants + torch.cat((pos, vel), dim=1)
            covariants = covariants.reshape(len(covariants), -1)
        else:
            covariants = covariants + torch.stack((pos, vel), dim=1)
        return invariants, covariants
    

class EGNNBlock(MessagePassing):
    """ E(n)-equivariant Message Passing Layer """
    def __init__(self, node_features, edge_features, hidden_features, out_features, dim, aggr, act, num_pos=1):
        super().__init__(aggr=aggr)
        self.dim = dim
        self.node_f = node_features
        self.edge_f = edge_features
        self.hidden_f = hidden_features
        self.message_net = nn.Sequential(nn.Linear(2 * node_features + hidden_features, hidden_features),
                                         act(),
                                         nn.Linear(hidden_features, hidden_features))
        
        self.edge_net = nn.Sequential(nn.Linear(edge_features, hidden_features),
                                         act(),
                                         nn.Linear(hidden_features, hidden_features))
    
        self.update_net = nn.Sequential(nn.Linear(node_features + hidden_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, out_features))
        
        self.pos_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                     act(),
                                     nn.Linear(hidden_features, num_pos))

        self.vel_net = nn.Sequential(nn.Linear(node_features, hidden_features),
                                     act(),
                                     nn.Linear(hidden_features, num_pos))
        
        nn.init.xavier_uniform_(self.pos_net[-1].weight, gain=0.001)
        self.num_pos = num_pos

    

    def forward(self, x, pos, vel, edge_index, edge_attr=None, split=False):
        x, pos, vel = self.propagate(edge_index, x=x, pos=pos, vel=vel, edge_attr=edge_attr, split=split)
        return x, pos, vel

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr, split):
        """ Create messages """
        if edge_attr is None:
            edge_dist = torch.norm(pos_i - pos_j, dim=-1).unsqueeze(-1)
            edge_attr = self.edge_net(edge_dist)

        input = [x_i, x_j] if edge_attr is None else [x_i, x_j, edge_attr]
        input = torch.cat(input, dim=-1)
        if split:
            self.dim=self.num_pos*3
            pos_i = pos_i.reshape(pos_i.shape[0], -1, 3)
            pos_j = pos_j.reshape(pos_j.shape[0], -1, 3)
        message = self.message_net(input)
        pos_message = self.pos_net(message)

        pos_message = (pos_i - pos_j)*pos_message if not split else ((pos_i - pos_j)*pos_message.unsqueeze(-1)).reshape(x_i.shape[0], -1)
        message = torch.cat((message, pos_message), dim=-1)
        return message

    def update(self, message, x, pos, vel, split):
        if split:
            pos = pos.reshape(pos.shape[0], -1, 3)
            vel = vel.reshape(vel.shape[0], -1, 3)
        """ Update node features and positions """
        node_message, pos_message = message[..., :-self.dim], message[..., -self.dim:]
        # update velocity
        vel = self.vel_net(x) * vel + pos_message if not split else self.vel_net(x).unsqueeze(-1) * vel + pos_message.reshape(x.shape[0], -1, 3)
        # Update node features
        input = torch.cat((x, node_message), dim=-1)
        update = x + self.update_net(input)
        # Update positions
        pos += vel
        if split:
            pos = pos.reshape(pos.shape[0], -1)
            vel = vel.reshape(vel.shape[0], -1)
        return update, pos, vel


class CEBlock(nn.Module):
    def __init__(self, algebra, in_features, hidden_features, out_features, n_layers=1, normalization_init=0):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features

        self.qkv = MVLinear(algebra, in_features, out_features * 3)
        self.gp = MVGeometricProduct(
            algebra,
            out_features,
            normalization_init=normalization_init,
        )
        self.silu = MVSiLU(algebra, out_features)
        self.norm = MVLayerNorm(algebra, out_features)

    def forward(self, input):
        q, k, v = self.qkv(input).chunk(3, dim=1)
        q = self.silu(q)
        kv = self.norm(self.gp(k, v))
        return (q + kv) / math.sqrt(2)
    

class CEMLP(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
        residual=False
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.residual = residual

        layers = []

        # Add geometric product layers.
        for i in range(n_layers - 1):
            layers.append(
                nn.Sequential(
                    MVLinear(self.algebra, in_features, hidden_features),
                    MVSiLU(self.algebra, hidden_features),
                    SteerableGeometricProductLayer(
                        self.algebra,
                        hidden_features,
                        normalization_init=normalization_init,
                    ),
                    MVLayerNorm(self.algebra, hidden_features),
                )
            )
            in_features = hidden_features

        # Add final layer.
        layers.append(
            nn.Sequential(
                MVLinear(self.algebra, in_features, out_features),
                MVSiLU(self.algebra, out_features),
                SteerableGeometricProductLayer(
                    self.algebra,
                    out_features,
                    normalization_init=normalization_init,
                ),
                MVLayerNorm(self.algebra, out_features),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            if not self.residual:
                x = layer(x)
            else:
                x_out = layer(x)
                x = x + x_out
        return x


    
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
        # self.norm = MVLayerNorm(self.algebra, self.hidden_features)
        

    def message(self, h_i, h_j, edge_attr=None, algebra=CliffordAlgebra((1.0, 1.0, 1.0))):
        h_i, h_j = cl_split(h_i, algebra=algebra), cl_split(h_j, algebra=algebra)
        if edge_attr is None:  # Unused.
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)
        h_msg = self.edge_model(input)
        h_msg = cl_flatten(h_msg)
        return h_msg

    def update(self, h_agg, h, node_attr, algebra=CliffordAlgebra((1.0, 1.0, 1.0))):
        h_agg, h = cl_split(h_agg, algebra=algebra), cl_split(h, algebra=algebra)
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)
        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h
        # out_h = self.norm(out_h)
        out_h = cl_flatten(out_h)
        return out_h
    
    def forward(self, h, edge_index, edge_attr=None, node_attr=None, algebra=CliffordAlgebra((1.0, 1.0, 1.0))):
        h = cl_flatten(h)
        x = self.propagate(h=h, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr, algebra=algebra)
        x = cl_split(x, algebra=algebra)

        return x


class SimplicialMPLayer(nn.Module):
    """
    Layer of E(n) Equivariant Message Passing Simplicial Network.

    A message passing layer is added for each type of adjacency to the message_passing dict. For each simplex, a state is
    found by concatening the messages sent to that simplex, e.g. we update an edge by concatenating the messages from
    nodes, edges, and triangles. The simplex is update by passing this state through an MLP as found in the update dict.
    """
    def __init__(self, adj_types: List[str], max_dim: int, num_hidden: int) -> None:
        super().__init__()
        self.adj_types = adj_types

        # messages
        self.message_passing = nn.ModuleDict({
            adj: GINConv(
                nn.Sequential(
                    nn.Linear(num_hidden, num_hidden),
                    nn.SiLU(),
                    nn.Linear(num_hidden, num_hidden)
                )
            )
            for adj in self.adj_types
        })

        # updates
        self.update = nn.ModuleDict()
        for dim in range(max_dim + 1):
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in self.adj_types])
            self.update[str(dim)] = nn.Sequential(
                nn.Linear(factor * num_hidden, num_hidden),
                nn.SiLU(),
                nn.Linear(num_hidden, num_hidden)
            )

    def forward(self, x: Dict[str, Tensor], adj: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # pass the different messages of all adjacency types
        mes = {
            adj_type:
                self.message_passing[adj_type](
                    x=(x[adj_type[0]], x[adj_type[2]]),
                    edge_index=index,
                    size=(x[adj_type[0]].shape[0], x[adj_type[2]].shape[0])
                )
            for adj_type, index in adj.items()
        }

        # find update states through concatenation, update and add residual connection
        h = {dim: torch.cat(
            [feature] + [adj_mes for adj_type, adj_mes in mes.items() if adj_type[2] == dim], dim=1
        ) for dim, feature in x.items()}
        h = {dim: self.update[dim](feature) for dim, feature in h.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}
        return x
    

# class CLGINConv(MessagePassing):
#     """
#         PyG GINConv class.
#     """
    
#     def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False, hidden: int = 0,
#                  **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#         self.nn = nn
#         self.initial_eps = eps
#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.Tensor([eps]))
#         else:
#             self.register_buffer('eps', torch.Tensor([eps]))
#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.nn)
#         self.eps.data.fill_(self.initial_eps)

#     def forward(self, x, edge_index, size = None, edge_attr=None):
#         """"""
#         if isinstance(x, Tensor):
#             x = (x, x)
#         x = tuple([cl_flatten(x_i) for x_i in x])
#         # propagate_type: (x: OptPairTensor)
#         if edge_attr is not None:
#             edge_attr = cl_flatten(edge_attr)
#         out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
#         out = cl_split(out)
#         x_r = cl_split(x[1])
#         if x_r is not None:
#             out += (1 + self.eps) * x_r

#         return out

#     def message(self, x_i, x_j: Tensor, edge_attr) -> Tensor:
#         x_i, x_j = cl_split(x_i), cl_split(x_j)
#         if edge_attr is not None:
#             edge_attr = cl_split(edge_attr)
#             mes = torch.cat((x_i - x_j, edge_attr), dim=1)
#         else:
#             mes = x_i - x_j
#         h_msg = cl_flatten(self.nn(mes))

#         return h_msg

#     def message_and_aggregate(self, adj_t, x) -> Tensor:
#         x = tuple([cl_split(x_i) for x_i in x])
#         adj_t = adj_t.set_value(None, layout=None)
#         out = torch_sparse.matmul(adj_t, x[0], reduce=self.aggr) # type: ignore
#         return cl_flatten(out)
    
#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(nn={self.nn})'
    
class SimplicialEGNNLayer(nn.Module):
    def __init__(self, num_hidden, num_inv):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )
        self.message_update = nn.Sequential(
            nn.Linear(2 * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )

        self.edge_inf_mlp = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x, index, edge_attr):
        index_i, index_j = index
        x_i, x_j = x[0][index_i], x[1][index_j]
        messages, edge_weights = self.message(x_i, x_j, edge_attr)
        return messages, edge_weights

    def message(self, h_i, h_j, edge_attr=None):
        state = torch.cat((h_i, h_j, edge_attr), dim=-1) # type: ignore
        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        return messages, edge_weights
    

class SimplicialEGNNLayer_(nn.Module):
    """ E(n)-equivariant Message Passing Layer """
    def __init__(self, invariant_features, covariant_features, edge_features, hidden_features, out_features):
        super().__init__()
        self.node_f = invariant_features
        self.edge_f = edge_features
        self.cov_f = covariant_features
        self.hidden_f = hidden_features
        self.message_net = EMLPLayer(x_features_in=covariant_features*2, x_features_out=covariant_features, 
                                     h_features_in=invariant_features*2 + edge_features, h_features_out=out_features, hidden_features=hidden_features)
    
        self.update_net = EMLPLayer(x_features_in=covariant_features*2, x_features_out=covariant_features, 
                                    h_features_in=hidden_features, h_features_out=out_features, hidden_features=hidden_features)


    def forward(self, pos, vel, h, index, edge_attr, num_pos=1):
        index_i, index_j = index
        pos_i, pos_j = pos[0][index_i], pos[1][index_j]
        vel_i, vel_j = vel[0][index_i], vel[1][index_j]
        h_i, h_j = h[0][index_i], h[1][index_j]
        messages = self.message(h_i, h_j, pos_i, pos_j, vel_i, vel_j, edge_attr, num_pos)
        messages = scatter_mean(messages, index_j, dim=0)
        invariants, covariants = self.update(messages, pos[1], vel[1], h[1], num_pos)
        return invariants, covariants


    def message(self, h_i, h_j, pos_i, pos_j, vel_i, vel_j, edge_attr=None, num_pos=1):
        pos_i = pos_i.reshape(len(pos_i), num_pos, -1)
        pos_j = pos_j.reshape(len(pos_j), num_pos, -1)
        vel_i = vel_i.reshape(len(vel_i), num_pos, -1)
        vel_j = vel_j.reshape(len(vel_j), num_pos, -1)
        invariants = [h_i, h_j] if edge_attr is None else [h_i, h_j, edge_attr]
        invariants = torch.cat(invariants, dim=-1)
        covariants = torch.cat([pos_i, pos_j, vel_i, vel_j], dim=1)
        covariants, invariants = self.message_net(covariants, invariants)
        message = torch.cat((invariants, covariants), dim=-1)
        return message

    def update(self, message, pos, vel, h, num_pos):
        pos = pos.reshape(len(pos), num_pos, -1)
        vel = vel.reshape(len(vel), num_pos, -1)
        # split the features out
        invariants = message[:, :self.hidden_f]
        covariants = message[:, self.hidden_f:]

        # concat the covariant features and update again
        covariants = covariants.reshape(len(covariants), self.cov_f, -1)
        covariants = torch.cat([covariants, pos, vel], dim=1)
        covariants = covariants.reshape(len(covariants), -1)
        covariants, invariants = self.update_net(covariants, invariants)
        covariants = covariants.reshape(len(covariants), self.cov_f, -1)

        # residue connection
        invariants = invariants + h
        covariants = covariants + torch.cat((pos, vel), dim=1)
        return invariants, covariants


class PosMLP(nn.Module):
  def __init__(self, num_hidden, act_fn, num_pos):
    super().__init__()
    self.num_pos = num_pos
    self.vel_mlp = TwoLayerMLP(num_hidden, num_hidden, num_pos, act_fn)
    self.pos_mlp = TwoLayerMLP(num_hidden, num_hidden, num_pos, act_fn)

  def forward(self, x, vel, loc, mes, adj):
    send, rec = adj['0_0']
    loc_div = loc[send] - loc[rec]
    pos_message = self.pos_mlp(mes['0_0'])
    if self.num_pos > 1:
        pos_message = pos_message.unsqueeze(-1).repeat(1, 1, 3)
    weight_mes = loc_div * pos_message
    agg_mes = scatter_add(weight_mes, adj['0_0'][1], 0)
    loc += agg_mes
    vel_message = self.vel_mlp(x['0'])
    if self.num_pos > 1:
        vel_message = vel_message.unsqueeze(-1).repeat(1, 1, 3)
    loc += vel_message * vel
    return loc
  

class ESMPNLayer(nn.Module):
    """
    Layer of E(n) Equivariant Message Passing Simplicial Network.

    A message passing layer is added for each type of adjacency to the message_passing dict. For each simplex, a state is
    found by concatening the messages sent to that simplex, e.g. we update an edge by concatenating the messages from
    nodes, edges, and triangles. The simplex is update by passing this state through an MLP as found in the update dict.
    """
    def __init__(self, adjacencies: List[str], max_dim: int, num_hidden: int, act_fn: str, given_dict_temp=None, num_pos=1) -> None:
        super().__init__()
        self.adjacencies = adjacencies

        dict_temp = {
            '0_0': 3,
            '0_1': 3,
            '1_0': 3,
            '1_1': 6,
            '1_2': 6,
            '2_1': 6
        } if not given_dict_temp else given_dict_temp

        # messages
        self.message_passing = nn.ModuleDict({
            adj: SimplicialEGNNLayer(
                num_hidden, dict_temp[adj]
            ) for adj in adjacencies
        })

        # updates
        self.update = nn.ModuleDict()
        for dim in range(max_dim + 1):
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            self.update[str(dim)] = TwoLayerMLP(
                factor * num_hidden, 
                num_hidden, 
                num_hidden, 
                act_fn
            )

        self.norm = nn.ModuleDict(
                {
                    str(dim): nn.LayerNorm(num_hidden) 
                    for dim in range(max_dim + 1)
                }
            )
        
        self.pos_mlp = PosMLP(
            num_hidden, 
            act_fn, 
            num_pos
        )

    def forward(self, x: Dict[str, Tensor], adj: Dict[str, Tensor], inv: Dict[str, Tensor], pos, vel):
        # pass the different messages of all adjacency types
        mes = {}
        edge_weights = {}
        for adj_type, index in adj.items():
            mes[adj_type], edge_weights[adj_type] = self.message_passing[adj_type](
                x=(x[adj_type[0]], x[adj_type[2]]),
                index=index,
                edge_attr=inv[adj_type]
            )
        
        # aggregation
        aggs = {
                adj_type: scatter_add(
                    mes[adj_type]*edge_weights[adj_type],
                    adj[adj_type][1],
                    dim=0,
                    dim_size=x[adj_type[2]].shape[0]   
                ) for adj_type, _ in adj.items()
            }
        # find update states through concatenation, update and add residual connection
        h = {
            dim: torch.cat(
                [feature] + [adj_mes for adj_type, adj_mes in aggs.items() if adj_type[2] == dim], 
                dim=-1
            ) for dim, feature in x.items()
        }
        h = {dim: self.update[dim](feature) for dim, feature in h.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}
        x = {dim: self.norm[dim](feature) for dim, feature in x.items()}

        # update pos
        pos['0'] = self.pos_mlp(x, vel['0'], pos['0'], mes, adj)
            
        return x, pos, vel


class EMPSNLayer_(nn.Module):
    """ E(n)-equivariant Message Passing Layer for simplicial complexes """
    def __init__(self, adjacencies: List[str], max_dim: int, invariant_features: int, 
                 covariant_features: int, hidden_features: int, 
                 out_features: int, act_fn: str, num_pos=1, given_dict_temp=None):
        super().__init__()
        self.adjacencies = adjacencies
        self.max_dim = max_dim
        dict_temp = {
            '0_0': 3,
            '0_1': 3,
            '1_0': 3,
            '1_1': 6,
            '1_2': 6,
            '2_1': 6
        } if not given_dict_temp else given_dict_temp
        # messages
        self.message_passing = nn.ModuleDict({
            adj: SimplicialEGNNLayer_(invariant_features, covariant_features*num_pos, dict_temp[adj], hidden_features, out_features) for adj in adjacencies
        })
        self.num_pos = num_pos
    
        # updates
        self.update_net = nn.ModuleDict()
        for dim in range(max_dim + 1):
            factor = 1 + sum([adj_type[2] == str(dim) for adj_type in adjacencies])
            self.update_net[str(dim)] = TwoLayerMLP(factor * hidden_features, hidden_features, hidden_features, act_fn)

        self.norm = nn.ModuleDict({str(dim): nn.LayerNorm(hidden_features) for dim in range(max_dim + 1)})

    def forward(self, x: Dict[str, Tensor], adj: Dict[str, Tensor], inv: Dict[str, Tensor], pos, vel):
        updated_pos, updated_vel, mes = {}, {}, {}
        # pass the different messages of all adjacency types
        for adj_type, index in adj.items():
            invariants, covariants = self.message_passing[adj_type](pos=(pos[adj_type[0]], pos[adj_type[2]]), 
                                                                    vel=(vel[adj_type[0]], vel[adj_type[2]]),
                                                                    h=(x[adj_type[0]], x[adj_type[2]]), 
                                                                    index=index, edge_attr=inv[adj_type], num_pos=self.num_pos)
            mes[adj_type] = invariants
            covariants_pos, covariants_vel = covariants[:, :self.num_pos, :], covariants[:, self.num_pos:2*self.num_pos, :]
            for dim in range(self.max_dim + 1):
                if str(dim) == adj_type[2]:
                    if dim not in updated_pos:
                        updated_pos[str(dim)] = covariants_pos.squeeze()
                    else:
                        updated_pos[str(dim)] = updated_pos[str(dim)] + covariants_pos.squeeze()
                    if str(dim) not in updated_vel:
                        updated_vel[str(dim)] = covariants_vel.squeeze()
                    else:
                        updated_vel[str(dim)] = updated_vel[str(dim)] + covariants_vel.squeeze()
        # find update states through concatenation, update and add residual connection
        h = {
            dim: torch.cat(
                [feature] + [adj_mes for adj_type, adj_mes in mes.items() if adj_type[2] == dim], dim=-1
            ) for dim, feature in x.items()
        }
        h = {dim: self.update_net[dim](feature) for dim, feature in h.items()}
        x = {dim: feature + h[dim] for dim, feature in x.items()}
        x = {dim: self.norm[dim](feature) for dim, feature in x.items()}

        return x, updated_pos, updated_vel


class EGCLP(MessagePassing):
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
        aggr="sum",
    ):
        super().__init__(aggr=aggr)
        self.algebra = algebra
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.prod1 = CEMLP(
            algebra,
            self.hidden_features,
            self.hidden_features,
            self.hidden_features,
            n_layers=1
        )
        self.prod2 = CEMLP(
            algebra,
            self.hidden_features,
            self.hidden_features,
            self.hidden_features,
            n_layers=2
        )

        self.edge_model = CEMLP(
            algebra,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

        self.node_model = CEMLP(
            algebra,
            4 * self.hidden_features + self.node_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )
        self.algebra = algebra
        # self.norm = MVLayerNorm(self.algebra, self.hidden_features)
        

    def message(self, h_i, h_j, edge_attr=None, algebra=CliffordAlgebra((1.0, 1.0, 1.0))):
        h_i, h_j = cl_split(h_i, algebra=algebra), cl_split(h_j, algebra=algebra)
        if edge_attr is None:  # Unused.
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)
        h_msg = self.edge_model(input) # first order and second order
        h_msg = cl_flatten(h_msg)
        return h_msg

    def update(self, h_agg, h, node_attr, algebra=CliffordAlgebra((1.0, 1.0, 1.0))):
        h_agg, h = cl_split(h_agg, algebra=algebra), cl_split(h, algebra=algebra)
        h1 = self.prod1(h_agg) # second order, third order and forth order
        h2 = self.prod2(h_agg) # forth to eighth order

        if node_attr is not None:
            input_h = torch.cat([h, h_agg, h1, h2, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg, h1, h2], dim=1)
        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h
        # out_h = self.norm(out_h)
        out_h = cl_flatten(out_h)
        return out_h
    
    def forward(self, h, edge_index, edge_attr=None, node_attr=None, algebra=CliffordAlgebra((1.0, 1.0, 1.0))):
        h = cl_flatten(h)
        x = self.propagate(h=h, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr, algebra=algebra)
        x = cl_split(x, algebra=algebra)

        return x