import torch
import os

import numpy as np
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import trange
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from csmpn.data.modules.simplicial_data import SimplicialTransform
from tqdm import tqdm

DATAROOT = os.environ["DATAROOT"]


class ConvexHull(Dataset):
    def __init__(self, n_samples, split="train", n_particles=8):
        dataroot = os.path.join(os.environ["DATAROOT"], "hulls")
        self.input = np.load(os.path.join(dataroot, f"hulls_{split}_input.npy"))
        self.target = np.load(os.path.join(dataroot, f"hulls_{split}_target.npy"))

        self.input = self.input[:n_samples]
        self.target = self.target[:n_samples]

        assert len(self.input) == n_samples and len(self.target) == n_samples
        assert self.input.shape[1] == n_particles
        self.num_samples = n_samples

        self.edge_index = torch.tensor(
            [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
        ).T
        self.n_particles = n_particles

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return Data(
            **{
                "input": torch.tensor(self.input[idx]),
                "target": torch.tensor(self.target[idx]),
            },
            edge_index=self.edge_index,
            num_nodes=self.n_particles
        )


class HullsSimplicial(InMemoryDataset):
    def __init__(
        self,
        root=DATAROOT,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.dataset = ConvexHull(**kwargs)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data, self.slices = self.process_data()

    @property
    def processed_file_names(self):
        kwarg_string = "_".join(f"{key}={value}" for key, value in self.kwargs.items())
        return [f"hulls_{kwarg_string}.pt"]

    def process(self):
        data_list = [graph for graph in self.dataset]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ConvexHullDataset:
    def __init__(self, num_samples=16384, batch_size=8, simplicial=False, dim=2) -> None:
        super().__init__()
        self.simplicial = simplicial

        if not simplicial:
            self.train_dataset = ConvexHull(num_samples, split="train")
            self.val_dataset = ConvexHull(num_samples, split="val")
            self.test_dataset = ConvexHull(num_samples, split="test")
        else:
            self.transform = SimplicialTransform(
                dim=dim, label="hulls", edge_th=float("inf"), tri_th=float("inf")
            )
            self.train_dataset = HullsSimplicial(
                root=f'{DATAROOT}Hulls_dim{dim}',
                pre_transform=self.transform, n_samples=num_samples, split="train"
            )
            self.val_dataset = HullsSimplicial(
                root=f'{DATAROOT}Hulls_dim{dim}',
                pre_transform=self.transform, n_samples=num_samples, split="val"
            )
            self.test_dataset = HullsSimplicial(
                root=f'{DATAROOT}Hulls_dim{dim}',
                pre_transform=self.transform, n_samples=num_samples, split="test"
            )

        self.batch_size = batch_size

        if simplicial:
            self.follow = ["node_types", "x_ind"]
        else:
            self.follow = None

    def train_loader(self):
        if self.simplicial:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                follow_batch=self.follow,
            )
        else:
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=36
            )

    def val_loader(self):
        if self.simplicial:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                follow_batch=self.follow,
            )
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=36)

    def test_loader(self):
        if self.simplicial:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                follow_batch=self.follow,
            )
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=36)


def signed_volume_of_simplex(simplex):
    matrix = np.column_stack(simplex)
    return np.linalg.det(matrix) / np.math.factorial(matrix.shape[0])


def get_signed_hull_volume(hull):
    volume = 0
    ref_point = hull.points[0]
    for simplex_indices in hull.simplices:
        simplex_points = [hull.points[index] - ref_point for index in simplex_indices]
        volume += signed_volume_of_simplex(simplex_points)
    return volume


if __name__ == "__main__":
    dataroot = os.path.join(os.environ["DATAROOT"], "hulls")

    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    from scipy.spatial import ConvexHull

    def _generate_meshes(n_points):
        meshes = []
        for i in trange(n_points):
            points = np.random.randn(8, 5)
            hull = ConvexHull(points)
            volume = hull.volume
            meshes.append((points, volume))

        meshes, volumes = zip(*meshes)
        meshes = np.stack(meshes).astype(np.float32)
        volumes = np.stack(volumes).astype(np.float32)
        return meshes, volumes

    train_meshes, train_volumes = _generate_meshes(16384)
    val_meshes, val_volumes = _generate_meshes(16384)
    test_meshes, test_volumes = _generate_meshes(16384)
    np.save(os.path.join(dataroot, "hulls_train_input.npy"), train_meshes)
    np.save(os.path.join(dataroot, "hulls_train_target.npy"), train_volumes)
    np.save(os.path.join(dataroot, "hulls_val_input.npy"), val_meshes)
    np.save(os.path.join(dataroot, "hulls_val_target.npy"), val_volumes)
    np.save(os.path.join(dataroot, "hulls_test_input.npy"), test_meshes)
    np.save(os.path.join(dataroot, "hulls_test_target.npy"), test_volumes)
