import torch.nn as nn
import torch
import math
from torch_geometric.nn import MessagePassing
EPS = 1e-6

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
        dim: int: starting dim, default: 0.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]


class NormalizationLayer(nn.Module):
    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features

        self.a = nn.Parameter(torch.zeros(self.in_features, algebra.n_subspaces) + init)

    def forward(self, input):
        assert input.shape[1] == self.in_features

        norms = torch.cat(self.algebra.norms(input), dim=-1)
        s_a = torch.sigmoid(self.a)
        norms = s_a * (norms - 1) + 1  # Interpolates between 1 and the norm.
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        normalized = input / (norms + EPS)

        return normalized

class MVSiLU(nn.Module):
    def __init__(self, algebra, channels, invariant="mag2", exclude_dual=False):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.exclude_dual = exclude_dual
        self.invariant = invariant
        self.a = nn.Parameter(torch.ones(1, channels, algebra.dim + 1))
        self.b = nn.Parameter(torch.zeros(1, channels, algebra.dim + 1))

        if invariant == "norm":
            self._get_invariants = self._norms_except_scalar
        elif invariant == "mag2":
            self._get_invariants = self._mag2s_except_scalar
        else:
            raise ValueError(f"Invariant {invariant} not recognized.")

    def _norms_except_scalar(self, input):
        return self.algebra.norms(input, grades=self.algebra.grades[1:])

    def _mag2s_except_scalar(self, input):
        return self.algebra.qs(input, grades=self.algebra.grades[1:])

    def forward(self, input):
        norms = self._get_invariants(input)
        norms = torch.cat([input[..., :1], *norms], dim=-1)
        a = unsqueeze_like(self.a, norms, dim=2)
        b = unsqueeze_like(self.b, norms, dim=2)
        norms = a * norms + b
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.sigmoid(norms) * input


class MVLayerNorm(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))

    def forward(self, input):
        norm = self.algebra.norm(input)[..., :1].mean(dim=1, keepdim=True) + EPS
        a = unsqueeze_like(self.a, norm, dim=2)
        return a * input / norm

class SteerableGeometricProductLayer(nn.Module):
    def __init__(
        self, algebra, features, include_first_order=True, normalization_init=0
    ):
        super().__init__()

        self.algebra = algebra
        self.features = features
        self.include_first_order = include_first_order

        if normalization_init is not None:
            self.normalization = NormalizationLayer(
                algebra, features, normalization_init
            )
        else:
            self.normalization = nn.Identity()
        self.linear_right = MVLinear(algebra, features, features, bias=False)
        if include_first_order:
            self.linear_left = MVLinear(algebra, features, features, bias=True)

        self.product_paths = algebra.geometric_product_paths
        self.weight = nn.Parameter(torch.empty(features, self.product_paths.sum()))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / (math.sqrt(self.algebra.dim + 1)))

    def _get_weight(self):
        weight = torch.zeros(
            self.features,
            *self.product_paths.size(),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        weight[:, self.product_paths] = self.weight
        subspaces = self.algebra.subspaces
        weight_repeated = (
            weight.repeat_interleave(subspaces, dim=-3)
            .repeat_interleave(subspaces, dim=-2)
            .repeat_interleave(subspaces, dim=-1)
        )
        return self.algebra.cayley * weight_repeated

    def forward(self, input):
        input_right = self.linear_right(input)
        input_right = self.normalization(input_right)

        weight = self._get_weight()

        if self.include_first_order:
            return (
                self.linear_left(input)
                + torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)
            ) / math.sqrt(2)

        else:
            return torch.einsum("bni, nijk, bnk -> bnj", input, weight, input_right)




class CEMLP(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers

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
            x = layer(x)
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

    def message(self, h_i, h_j, edge_attr=None):
        h_i, h_j = self.algebra.split(h_i), self.algebra.split(h_j)
        if edge_attr is None:
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)
        h_msg = self.edge_model(input)
        h_msg = self.algebra.flatten(h_msg)
        return h_msg

    def update(self, h_agg, h, node_attr):
        h_agg, h = self.algebra.split(h_agg), self.algebra.split(h)
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)
        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h
        out_h = self.algebra.flatten(out_h)
        return out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        h = self.algebra.flatten(h)
        x = self.propagate(
            h=h, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr
        )
        x = self.algebra.split(x)

        return x


class MVLinear(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        out_features,
        subspaces=True,
        bias=True,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces

        if subspaces:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, algebra.n_subspaces)
            )
            self._forward = self._forward_subspaces
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter("bias", None)
            self.b_dims = ()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _forward(self, input):
        return torch.einsum("bm...i, nm->bn...i", input, self.weight)

    def _forward_subspaces(self, input):
        weight = self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)
        return torch.einsum("bm...i, nmi->bn...i", input, weight)

    def forward(self, input):
        result = self._forward(input)

        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
        return result