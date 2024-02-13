import torch
import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np


class TwoLayerMLP(nn.Module):
    """Basic two layer perceptron."""

    def __init__(self, num_input: int, num_hidden: int, num_out: int, act_fn: str) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            get_act_fn(act_fn),
            nn.Linear(num_hidden, num_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def get_act_fn(act_fn: str) -> nn.Module:
    """Return activation function based on name."""
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Activation function {act_fn} not recognized.')


class RFF(nn.Module):
    def __init__(self, in_features, out_features, sigma=1.0):
        super().__init__()

        sigma = [sigma] * in_features


        self.in_features = in_features
        self.out_features = out_features

        if out_features % 2 != 0:
            self.compensation = 1
        else:
            self.compensation = 0

        B = torch.randn(int(out_features / 2) + self.compensation, in_features)
        for i in range(in_features):
            B[:,i] = B[:,i] * sigma[i]

        B /= np.sqrt(2)
        self.register_buffer("B", B)

    def forward(self, x):
        x = F.linear(x, self.B)
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        if self.compensation:
            x = x[..., :-1]
        return x
    
def compute_invariants_3d(feat_ind, pos, adj, inv_ind, device):
    # angles
    angle = {}

    vecs = pos[feat_ind['1'][:, 0]] - pos[feat_ind['1'][:, 1]]
    send_vec, rec_vec = vecs[adj['1_1'][0]], vecs[adj['1_1'][1]]
    send_norm, rec_norm = torch.linalg.norm(send_vec, ord=2, dim=-1), torch.linalg.norm(rec_vec, ord=2, dim=-1)

    dot = torch.sum(send_vec * rec_vec, dim=-1)
    cos_angle = dot / (send_norm * rec_norm)
    eps = 1e-6
    angle['1_1'] = torch.arccos(cos_angle.clamp(-1 + eps, 1 - eps)).unsqueeze(-1)

    p1, p2, a = pos[inv_ind['1_2'][0]], pos[inv_ind['1_2'][1]], pos[inv_ind['1_2'][2]]
    v1, v2, b = p1 - a, p2 - a, p1 - p2
    v1_n, v2_n, b_n = torch.linalg.norm(v1, dim=-1), torch.linalg.norm(v2, dim=-1), torch.linalg.norm(b, dim=-1)
    v1_a = (torch.arccos((torch.sum(v1 * b, dim=-1) / (v1_n * b_n)).clamp(-1 + eps, 1 - eps))).unsqueeze(-1)
    v2_a = (torch.arccos((torch.sum(v2 * b, dim=-1) / (v2_n * b_n)).clamp(-1 + eps, 1 - eps))).unsqueeze(-1)
    b_a = (torch.arccos((torch.sum(v1 * v2, dim=-1) / (v1_n * v2_n)).clamp(-1 + eps, 1 - eps))).unsqueeze(-1)

    angle['1_2'] = torch.cat((v1_a + v2_a, b_a), dim=-1)

    # areas
    area = {}
    
    area['0'] = torch.zeros(pos.shape[:-1]).unsqueeze(-1)
    area['1'] = torch.norm(pos[feat_ind['1'][:, 0]] - pos[feat_ind['1'][:, 1]], dim=-1).unsqueeze(-1)
    vec_1 = pos[feat_ind['2'][:, 0]] - pos[feat_ind['2'][:, 1]]
    vec_2 = pos[feat_ind['2'][:, 0]] - pos[feat_ind['2'][:, 2]]
    if pos[feat_ind['2'][:, 0]].shape[-1] == 3 :
        area['2'] = (torch.norm(torch.cross(vec_1, vec_2, dim=-1), dim=-1) / 2).unsqueeze(-1) 
    else:
        norm_1 = torch.norm(vec_1, dim=-1)
        norm_2 = torch.norm(vec_2, dim=-1)
        cos_theta = torch.sum(vec_1*vec_2, dim=-1) / (norm_1 * norm_2)
        sin_theta = torch.sqrt(1 - cos_theta**2)
        area['2'] = (norm_1 * norm_2 * sin_theta).unsqueeze(-1) 
                            

    area = {k: v.to(feat_ind['0'].device) for k, v in area.items()}


    inv = {
        '0_0': torch.linalg.norm(pos[adj['0_0'][0]] - pos[adj['0_0'][1]], dim=-1).unsqueeze(-1),
        '0_1': torch.linalg.norm(pos[inv_ind['0_1'][0]] - pos[inv_ind['0_1'][1]], dim=-1).unsqueeze(-1),
        '1_1': torch.stack([
            torch.linalg.norm(pos[inv_ind['1_1'][0]] - pos[inv_ind['1_1'][1]], dim=-1),
            torch.linalg.norm(pos[inv_ind['1_1'][0]] - pos[inv_ind['1_1'][2]], dim=-1),
            torch.linalg.norm(pos[inv_ind['1_1'][1]] - pos[inv_ind['1_1'][2]], dim=-1),
        ], dim=-1),
        '1_2': torch.stack([
            torch.linalg.norm(pos[inv_ind['1_2'][0]] - pos[inv_ind['1_2'][2]], dim=-1)
            + torch.linalg.norm(pos[inv_ind['1_2'][1]] - pos[inv_ind['1_2'][2]], dim=-1),
            torch.linalg.norm(pos[inv_ind['1_2'][1]] - pos[inv_ind['1_2'][2]], dim=-1)
        ], dim=-1),
    }

    for k, v in inv.items():
        area_send, area_rec = area[k[0]], area[k[2]]
        send, rec = adj[k]
        area_send, area_rec = area_send[send], area_rec[rec]
        inv[k] = torch.cat((v, area_send, area_rec), dim=-1)
    inv['1_1'] = torch.cat((inv['1_1'], angle['1_1'].to(feat_ind['0'].device)), dim=-1)
    inv['1_2'] = torch.cat((inv['1_2'], angle['1_2'].to(feat_ind['0'].device)), dim=-1)

    return inv