import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from csmpn.models.layers import  EGCL, CEMLP
from csmpnn.algebra.cliffordalgebra import CliffordAlgebra
from cgen.models.modules.linear import MVLinear

from engineer.metrics.metrics import Loss, MetricCollection


class NBACliffordSharedSimplicialMPNN(nn.Module):
    def __init__(self, max_dim: int=2, num_input: int=20, num_hidden: int=28, num_out: int=40, num_layers: int=4, task_type="mse_regression", stats=None, condition=True) -> None:
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1))
        self.max_dim = max_dim
        self.condition = condition
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_out = num_out
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
        
        self.projection = nn.Sequential(
            CEMLP(self.algebra, num_hidden, num_hidden, num_hidden, n_layers=1),
            MVLinear(self.algebra, num_hidden, num_out),
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
        self.train_metrics = self._setup_metrics()
        self.val_metrics = self._setup_metrics()
        self.test_metrics = self._setup_metrics()
        self.loss_func = nn.MSELoss(reduction="none")

    def _setup_metrics(self):
        return MetricCollection({"loss": Loss(), "ade_loss": Loss(), "fde_loss": Loss(),})

    def forward(self, graph, step, mode):
        num_frames = graph.pos.shape[1]
        batch_size = graph.ptr.shape[0] - 1
        node_label = torch.cat([torch.tensor([graph.x_ind_ptr[i]]*(graph.x_ind_ptr[i+1]-graph.x_ind_ptr[i])) for i in range(batch_size)], dim=0).unsqueeze(-1).to(self.sim_type_embedding[0].weight.device).type(torch.long)
        x_ind = (graph.x_ind + node_label).to(torch.long)

        sim_embedding = self.sim_type_embedding(graph.node_types).unsqueeze(-1)
        sim_invariants = self.algebra.embed_grade(sim_embedding, 0)

        x = torch.zeros((graph.pos.shape[0], self.num_input, 2**self.algebra.dim)).to(self.sim_type_embedding[0].weight.device)
        for i in range(self.max_dim+1):
            ind = torch.where(graph.node_types == i)[0]
            idx = x_ind[ind][:, :i+1]
            if len(idx) != 0:
                features = torch.cat(
                    [self.algebra.embed_grade(graph.pos[idx].reshape(idx.shape[0], -1, self.algebra.dim), 1),
                    self.algebra.embed_grade(graph.vel[idx].reshape(idx.shape[0], -1, self.algebra.dim), 1),
                    ], dim=1
                )
                x[torch.where(graph.node_types==i)] = self.cl_feature_embedding[i](features)

        node_attr = sim_invariants if self.condition else None
        edge_attr = torch.cat((
            self.algebra.embed_grade(self.sim_type_embedding(graph.edge_attr[:, 0]).unsqueeze(-1), 0),
            self.algebra.embed_grade(self.sim_type_embedding(graph.edge_attr[:, 1]).unsqueeze(-1), 0)
        ), dim=1) if self.condition else None

        x = torch.cat((x, sim_invariants), dim=1)
        x = self.feature_embedding(x)

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
        # loss = F.mse_loss(loc_pred.reshape(-1, self.algebra.dim), targets.view(-1, self.algebra.dim), reduction="none").reshape(batch_size, -1, self.algebra.dim).sum(-1).mean(-1)
        loss = ade_loss
        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss, "ade_loss": ade_loss, "fde_loss": fde_loss}

    def __str__(self):
        return f"Clifford Shared Simplicial MPNN for MD17 Dataset"