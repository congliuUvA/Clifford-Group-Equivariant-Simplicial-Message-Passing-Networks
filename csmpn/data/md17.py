import os
import torch
import numpy as np
from csmpn.data.modules.simplicial_data import SimplicialTransform
from torch_geometric.data import Data, InMemoryDataset, DataLoader as PyGDataLoader
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.nn import knn_graph
from torch.utils.data.distributed import DistributedSampler
dataroot = os.environ["DATAROOT"]
class MD17:
    def __init__(self, partition, max_samples, data_dir, molecule_type, past_length=10,future_length=10, r=2.5):
        data_dir = os.path.join(data_dir, 'md17')
        full_dir = os.path.join(data_dir, molecule_type + '_' + partition + '.npy')

        self.max_samples = int(max_samples)
        self.data = self.load(data_dir,molecule_type,partition)
        self.past_length = past_length
        self.future_length = future_length
        self.partition = partition
        self.r = r

    def load(self,data_dir,molecule_type,partition):

        loc_full_dir = os.path.join(data_dir, molecule_type + '_' + partition + '.npy')
        edge_full_dir = os.path.join(data_dir, molecule_type + '_' + 'structure.npy')
        charges_dir = os.path.join(data_dir, molecule_type + '_' + 'charges.npy')
        
        loc = np.load(loc_full_dir) # (B,T,N,3)
        edge_attr = np.load(edge_full_dir)
        charges = torch.from_numpy(np.load(charges_dir))

        loc = loc[:self.max_samples]

        loc = torch.Tensor(loc)
        edge_attr = torch.Tensor(edge_attr)

        loc = loc.transpose(1,2)
        vel = torch.zeros_like(loc)
        vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
        vel[:,:,0] = vel[:,:,1]

        edge_attr = edge_attr[None,:,:].repeat(loc.shape[0],1,1)
        # edge_attr = self.get_molecule_structure(loc[0])
        return (loc, vel, edge_attr, charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr = loc[i], vel[i], edge_attr[i]
        frame_0 = self.past_length
        frame_T = self.past_length + self.future_length
        pos = loc[:, 0]
        # edge_index = radius_graph(pos, r=self.r, loop=False)
        edge_index = knn_graph(pos, k=int(self.r))
        graph = Data(
            loc=loc[:,0:frame_0],
            vel=vel[:,0:frame_0],
            edge_index=edge_index,
            y=loc[:,frame_0:frame_T],
            init_pos=pos,
            charges=charges,
        )

        return graph

    def __len__(self):
        return len(self.data[0])

class MD17SimplicialData(InMemoryDataset):
    """Motion Simplicial Dataset."""

    def __init__(self, root=dataroot, transform=None, pre_transform=None, pre_filter=None, num_samples=int(1e8), partition="train", molecule_type='aspirin', r=2.5):
        self.num_samples = num_samples
        self.partition = partition
        self.molecule_type = molecule_type
        self.r=r
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.partition}_data.pt']

    def process(self):
        self.dataset = MD17(partition=self.partition, max_samples=self.num_samples, data_dir=dataroot, molecule_type=self.molecule_type, r=self.r)
        # Read data into huge `Data` list.
        data_list = [graph for graph in self.dataset]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 

class MD17Dataset:
    def __init__(self, batch_size=100, dim=2, dis: float=2.5, simplicial=False, molecule_type='aspirin', dropout_rate=0.4, use_for_loop=False):
        self.dim = dim
        self.dis = dis
        self.batch_size = batch_size
        self.simplicial = simplicial
        self.label = "regular" if not self.simplicial else "simplicial"
        if not simplicial:
            self.train_dataset = MD17(partition='train', max_samples=5000, data_dir=dataroot, molecule_type=molecule_type, r=dis)
            self.test_dataset = MD17(partition='test', max_samples=2000, data_dir=dataroot, molecule_type=molecule_type, r=dis)
            self.valid_dataset = MD17(partition='val', max_samples=2000, data_dir=dataroot, molecule_type=molecule_type, r=dis)
        else:
            rootdir = f'{dataroot}md17_{molecule_type}_{dis}_{dim}_{self.label}'
            self.pre_transform = SimplicialTransform(label="md17", dim=dim, dis=dis, molecule_type=molecule_type)

            self.train_dataset = MD17SimplicialData(
                root=rootdir,
                pre_transform=self.pre_transform, partition="train", num_samples=5000, molecule_type=molecule_type, r=dis,
            )
            self.valid_dataset = MD17SimplicialData(
                root=rootdir,
                pre_transform=self.pre_transform, partition="val", num_samples=2000, molecule_type=molecule_type, r=dis,
            )
            self.test_dataset = MD17SimplicialData(
                root=rootdir,
                pre_transform=self.pre_transform, partition="test", num_samples=2000, molecule_type=molecule_type, r=dis,
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