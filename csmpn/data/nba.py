import os
import torch
import pickle 
import numpy as np
from csmpn.data.modules.simplicial_data import ManualTransform
from csmpn.data.ESMPN.simplicial_data import ESMPN_ManualTransform
from csmpn.data.modules.simplicial_data import SimplicialTransform
from torch_geometric.data import Data, InMemoryDataset, DataLoader as PyGDataLoader
import torch_geometric
import os
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.nn import knn_graph
from torch.utils.data.distributed import DistributedSampler


dataroot = os.environ['DATAROOT']

class NBA:
    """Dataloder for the Basketball trajectories datasets"""

    def __init__(self, obs_len=10, pred_len=40, mode="atk", split="train"):
        self.split = split
        self.data_dir = dataroot + "nba/" + mode + f"/trajectories_{self.split}.npy"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        assert self.seq_len == 50

        self.traj_abs = np.load(self.data_dir)
        self.traj_abs = self.traj_abs.swapaxes(1, 2)
        self.traj_vel = np.zeros(self.traj_abs.shape)
        self.traj_vel[:, :, 1:, :] = self.traj_abs[:, :, 1:, :] - self.traj_abs[:, :, :-1, :]

    def __len__(self):
        return self.traj_abs.shape[0]

    def __getitem__(self, idx):

        
        pos = torch.from_numpy(self.traj_abs[idx]).float()
        vel = torch.from_numpy(self.traj_vel[idx]).float()
        _, num_frames, dims = pos.shape
        reference_point = torch.ones((1, num_frames, dims))
        pos = torch.cat((pos, reference_point), dim=0)
        vel = torch.cat((vel, reference_point), dim=0)

        frame_0 = self.obs_len
        frame_T = self.pred_len + self.obs_len
        loc = pos[:, 0]
        edge_index = knn_graph(loc, k=10000)
        graph = Data(
            pos=pos[:,0:frame_0],
            vel=vel[:,0:frame_0],
            edge_index=edge_index,
            y=pos[:-1,frame_0:frame_T],
            init_pos=loc,
        )
        return graph
          

class NBASimplicialData(InMemoryDataset):
    """Motion Simplicial Dataset."""

    def __init__(self, root=dataroot, transform=None, pre_transform=None, pre_filter=None, partition="train", mode='atk'):
        self.partition = partition
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.partition}_data.pt']

    def process(self):
        self.dataset = NBA(mode=self.mode, split=self.partition)
        # Read data into huge `Data` list.
        data_list = [graph for graph in self.dataset]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 


class NBADataset:
    def __init__(self, batch_size=100, dim=2, simplicial=False, mode='atk', dis=10000):
        self.dim = dim
        self.batch_size = batch_size
        self.simplicial = simplicial
        self.label = "regular" if not self.simplicial else "simplicial"
        if not simplicial:
            self.train_dataset =  NBA(mode=mode, split="train")
            self.test_dataset =  NBA(mode=mode, split="test")
            self.valid_dataset =  NBA(mode=mode, split="val")
        else:
            self.pre_transform = SimplicialTransform(dis=dis, label="nba")
            self.train_dataset = NBASimplicialData(
                root=f'{dataroot}nba_{mode}_{self.label}_{dis}',
                pre_transform=self.pre_transform, partition="train", mode=mode
            )
            self.test_dataset = NBASimplicialData(
                root=f'{dataroot}nba_{mode}_{self.label}_{dis}',
                pre_transform=self.pre_transform, partition="test", mode=mode
            )
            self.valid_dataset = NBASimplicialData(
                root=f'{dataroot}nba_{mode}_{self.label}_{dis}',
                pre_transform=self.pre_transform, partition="val", mode=mode
            )
        if self.simplicial:
            self.follow = ["node_types", "x_ind"]  
        else:
            self.follow = None


    def train_loader(self):
        split = "train"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.train_dataset) if distributed else None)
        shuffle = True if split == 'train' and not distributed else False
        drop_last = split == 'train'
        if not self.simplicial:
            return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle)
        else:
            return PyGDataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, follow_batch=self.follow)

    def val_loader(self):
        split = "valid"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.valid_dataset) if distributed else None)
        shuffle = True if split == 'valid' and not distributed else False
        drop_last = split == 'valid'
        if not self.simplicial:
            return PyGDataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle)
        else:
            return PyGDataLoader(self.valid_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, follow_batch=self.follow)

    def test_loader(self):
        split = "test"
        distributed = torch.distributed.is_initialized()
        sampler = (DistributedSampler(self.test_dataset) if distributed else None)
        shuffle = True if split == 'train' and not distributed else False
        drop_last = split == 'train'
        if not self.simplicial:
            return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle)
        else:
            return PyGDataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=drop_last, sampler=sampler, shuffle=shuffle, follow_batch=self.follow)