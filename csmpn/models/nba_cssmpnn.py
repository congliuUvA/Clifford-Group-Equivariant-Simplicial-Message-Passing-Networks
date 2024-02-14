import torch
import torch.nn.functional as F
import torch.nn as nn
 
from csmpn.algebra.cliffordalgebra import CliffordAlgebra
from csmpn.models.cegnn_utils import MVLinear, EGCL, CEMLP

from engineer.metrics.metrics import Loss, MetricCollection
import itertools
import math

class NBACliffordSharedSimplicialMPNN(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=20, num_hidden: int=40, num_out: int=40, num_layers: int=4, stats=None, condition=True) -> None:
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1))
        self.max_dim = max_dim
        self.condition = condition
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.num_node_type = self.max_dim + 1 if condition else 0
        self.feature_embedding = MVLinear(
                self.algebra, 
                num_input + self.num_node_type, 
                num_hidden, 
                subspaces=False
            )
        self.cl_feature_embedding = nn.Sequential(
            MVLinear(
                self.algebra, 
                num_input, 
                num_input, 
                subspaces=False
            ), 
            CEMLP(
                algebra=self.algebra, 
                in_features=2*num_input, 
                hidden_features=num_hidden, 
                out_features=num_input, 
                n_layers=1, 
                normalization_init=0,
            ),
            nn.Sequential(
                CEMLP(
                    algebra=self.algebra, 
                    in_features=3*num_input, 
                    hidden_features=num_hidden, 
                    out_features=num_hidden, 
                    n_layers=1, 
                    normalization_init=0,
                ),
                CEMLP(
                    algebra=self.algebra, 
                    in_features=num_hidden, 
                    hidden_features=num_hidden, 
                    out_features=num_input, 
                    n_layers=1, 
                    normalization_init=0,
                )
            )
        )
        self.sim_type_embedding = nn.Embedding(num_embeddings=max_dim+1, embedding_dim=max_dim+1)
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
        
        self.projection = MVLinear(self.algebra, num_hidden, num_out)

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
        self.train_metrics = self._setup_metrics()
        self.val_metrics = self._setup_metrics()
        self.test_metrics = self._setup_metrics()
        self.loss_func = nn.MSELoss(reduction="none")

    def _setup_metrics(self):
        return MetricCollection({"loss": Loss(), "ade_loss": Loss(), "fde_loss": Loss(),})

    def embed_simplex_types(self, graph):
        node_attr = self.sim_type_embedding(graph.node_types).unsqueeze(-1)
        node_attr = self.algebra.embed_grade(node_attr, 0)

        edge_attr = torch.cat(
            (
                node_attr[graph.edge_index[0]],
                node_attr[graph.edge_index[1]],
            ),
            dim=1,
        )
        return node_attr, edge_attr

    def embed_simplicial_complex(self, graph):
        # For each node, indicates the starting index of the graph it belongs to.
        graph_start_idx = graph.x_ind_ptr[:-1][graph.x_ind_batch]

        # Gets the vertex indices for each simplex in the graph.
        simplex_indices = graph.x_ind.long() + graph_start_idx.unsqueeze(-1)

        input = torch.zeros(
            (len(graph.x_ind), self.num_input, 2**self.algebra.dim),
            device=graph.batch.device,
        )

        for d in range(self.max_dim + 1):
            d_simplices = simplex_indices[graph.node_types == d, : d + 1]
            if len(d_simplices) != 0:
                pos = graph.pos[d_simplices]
                vel = graph.vel[d_simplices]
                perm_dim = d + 1
                index_list = list(range(perm_dim))
                index_permutations = torch.tensor(list(itertools.permutations(index_list)), device=graph.x_ind.device)
                pos = pos[:,  index_permutations, ...].reshape(d_simplices.shape[0]*math.factorial(d + 1), pos.shape[2]*(d+1), -1)
                vel = vel[:,  index_permutations, ...].reshape(d_simplices.shape[0]*math.factorial(d + 1), vel.shape[2]*(d+1), -1)
                # Clifford embedding
                pos = self.algebra.embed_grade(pos, 1)
                vel = self.algebra.embed_grade(vel, 1)

                # Concatenate
                features = torch.cat((pos, vel), dim=1)
                embedding = self.cl_feature_embedding[d](features).reshape(d_simplices.shape[0], math.factorial(d + 1), -1, 2**self.algebra.dim).sum(dim=1)
                input[graph.node_types == d] = embedding

        return input

    def featurization(self, x, node_attr):
        x = torch.cat((x, node_attr), dim=1)
        x = self.feature_embedding(x)
        return x
    
    def forward(self, graph, step, mode):
        batch_size = graph.ptr.shape[0] - 1
        num_frames = graph.pos.shape[1]

        node_attr, edge_attr = self.embed_simplex_types(graph)
        x = self.embed_simplicial_complex(graph)
        x = self.featurization(x, node_attr)

        # message passing
        for layer in self.layers:
            x = layer(x, graph.edge_index, edge_attr, node_attr, self.algebra)

        out = x[torch.where(graph.node_types==0)]

        # batch_size, num_simplex_dim*num_hidden, 8
        out = self.projection(out)
        pred = out[..., 1:3]
        loc_pred = pred
        loc_pred = loc_pred.reshape(batch_size, 6, num_frames*4, -1)[:, :-1, ...]
        loc_pred = loc_pred.reshape(-1, self.num_out, self.algebra.dim)

        targets = graph.y
        ade_loss = torch.sqrt(F.mse_loss(loc_pred.reshape(-1, self.algebra.dim), targets.view(-1, self.algebra.dim), reduction="none").sum(dim=-1)).reshape(batch_size, -1, num_frames).mean(dim=-1).mean(dim=-1)  # [B]
        fde_loss = torch.sqrt(F.mse_loss(loc_pred[:, -1, :], targets[:, -1, :], reduction="none").sum(dim=-1)).reshape(batch_size, -1).mean(dim=-1)
        loss = ade_loss
        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss, "ade_loss": ade_loss, "fde_loss": fde_loss}

    def __str__(self):
        return f"Clifford Shared Simplicial MPNN for NBA Dataset"