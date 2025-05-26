import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import plyfile
import pyvista
import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

PROJECT_ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(PROJECT_ROOT_DIR)

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.dataset_utils import Statistics
from utils import PlaceHolder


def load_graph_file(graph_filename, directed):
    file_ext = os.path.splitext(graph_filename)[1].lower()

    if file_ext == '.vtp':
        vtk_data = pyvista.read(graph_filename)
        vtk_points = vtk_data.GetPoints().GetData()
        pos = torch.tensor(np.float32(np.asarray(vtk_points)), dtype=torch.float)
        edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
        edge_attr = torch.tensor(np.asarray(vtk_data.cell_data['edge_attr']), dtype=torch.long) + 1
        # edge_attr = torch.tensor(np.asarray(vtk_data.cell_data['label']), dtype=torch.long)
        edges = edges.T

    elif file_ext == '.ply':
        ply_data = plyfile.PlyData.read(graph_filename)
        vertices = ply_data['vertex']
        pos = torch.tensor(np.vstack([vertices['x'], vertices['y'], vertices['z']]).T,
                           dtype=torch.float)

        edges = torch.tensor(np.vstack([ply_data['edge']['vertex1'],
                                        ply_data['edge']['vertex2']]),
                             dtype=torch.long)
        if 'label' in ply_data['edge'].properties:
            edge_attr = torch.tensor(ply_data['edge']['label'], dtype=torch.long) + 1
        else:
            edge_attr = torch.zeros(edges.size(1), dtype=torch.long)

    pos = pos - torch.mean(pos, dim=0, keepdim=True)
    max_distance = torch.max(torch.norm(pos, dim=1))
    pos = pos / max_distance
    x = torch.ones((pos.size(0), 1))

    if not directed:
        edges, edge_attr = torch_geometric.utils.to_undirected(
            edge_index=edges, edge_attr=edge_attr, reduce="min")

    graph_data = Data(x=x, edge_index=edges, edge_attr=edge_attr, pos=pos)
    return graph_data


class CoWDataset(Dataset):
    def __init__(self, dataset_name, split, data_path, config):
        super(Dataset, self).__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.raw_data_path = data_path
        self.config = config

        self.is_directed = False
        self.num_edge_categories = 14
        self.num_node_categories = 14
        # self.num_edge_categories = 7
        self.num_degree_categories = 1
        self.data_path = os.path.join(PROJECT_ROOT_DIR, 'data', self.dataset_name)

        if not os.path.exists(os.path.join(self.data_path, f"{self.split}.pt")):
            os.makedirs(self.data_path, exist_ok=True)
            self.download()

        self.data = torch.load(os.path.join(self.data_path, f"{self.split}.pt"))
        self.statistics = self.compute_statistics()

    def download(self):
        all_graph_filenames = []
        for root, dirs, files in os.walk(self.raw_data_path):
            for file in files:
                if file.endswith('.vtp') or file.endswith('.ply'):
                    all_graph_filenames.append(os.path.join(root, file))

        max_edge_count = 0
        for path in all_graph_filenames:
            graph_data = load_graph_file(path, directed=self.is_directed)
            max_edge_count = max(max_edge_count, graph_data.edge_index.size(1))

        num_graphs = len(all_graph_filenames)
        test_len = int(round(num_graphs * 0.1))
        train_len = int(round((num_graphs - test_len) * 0.9))
        indices = torch.randperm(num_graphs)
        train_indices = indices[:train_len]
        test_indices = indices[train_len:]

        train_graphs = []
        test_graphs = []

        for i, graph_filename in enumerate(all_graph_filenames):
            graph_data = load_graph_file(graph_filename, directed=self.is_directed)
            graph_data.charges = torch.ones_like(graph_data.x)
            edge_count = graph_data.edge_index.size(1)
            graph_data.y = torch.tensor([[edge_count / max_edge_count]], dtype=torch.float)
            # graph_data.y = torch.zeros((1, 0), dtype=torch.float)
            if i in train_indices:
                train_graphs.append(graph_data)
            elif i in test_indices:
                test_graphs.append(graph_data)
        torch.save(train_graphs, os.path.join(self.data_path, "train.pt"))
        torch.save(test_graphs, os.path.join(self.data_path, "test.pt"))

        print(f"Saved {len(train_graphs)} training graphs and {len(test_graphs)} testing graphs.")

    def compute_statistics(self):
        num_nodes = Counter()
        edge_types = torch.zeros(self.num_edge_categories)
        atom_types = torch.zeros(self.num_degree_categories)
        bond_lengths = Counter()
        bond_angles = Counter()
        betti_vals = None

        for graph in self.data:
            # 统计节点数量分布
            num_nodes[graph.num_nodes] += 1
            # 统计边类型分布
            edge_types += torch.bincount(graph.edge_attr, minlength=self.num_edge_categories).float()
            # 统计节点类型分布
            atom_types += graph.x.sum(dim=0)
            # 统计边长度分布
            edge_vectors = graph.pos[graph.edge_index[0]] - graph.pos[graph.edge_index[1]]
            edge_lengths = torch.norm(edge_vectors, dim=1).tolist()
            for length in edge_lengths:
                bond_lengths[round(length, 2)] += 1

            for i in range(graph.edge_index.size(1)):
                for j in range(i + 1, graph.edge_index.size(1)):
                    if graph.edge_index[1, i] == graph.edge_index[1, j]:  # 共用一个节点
                        vec1 = edge_vectors[i]
                        vec2 = edge_vectors[j]
                        angle = torch.acos(torch.clamp(torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm()), -1.0, 1.0))
                        bond_angles[round(angle.item(), 2)] += 1

        atom_types = atom_types / atom_types.sum()
        edge_types = edge_types / edge_types.sum()

        return Statistics(num_nodes=num_nodes, atom_types=atom_types, bond_types=edge_types, bond_lengths=bond_lengths,
                          bond_angles=bond_angles, betti_vals=betti_vals, charge_types=torch.ones(1, ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def update_edge_length_info(self, graph_data, all_edge_lengths):
        cdists = torch.cdist(graph_data.pos, graph_data.pos)
        edge_distances = cdists[graph_data.edge_index[0], graph_data.edge_index[1]]
        rounded_distances = torch.round(edge_distances, decimals=2)
        for edge_type in range(self.num_edge_categories):
            mask = graph_data.edge_attr == edge_type
            for d in rounded_distances[mask]:
                all_edge_lengths[edge_type][d.item()] += 1


class CoWGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datapath = cfg.dataset.datapath
        self.train_dataset = CoWDataset(dataset_name=self.cfg.dataset.name,
                                        split='train', data_path=cfg.dataset.datapath, config=cfg)
        self.test_dataset = CoWDataset(dataset_name=self.cfg.dataset.name,
                                       split='test', data_path=cfg.dataset.datapath, config=cfg)
        self.val_dataset = CoWDataset(dataset_name=self.cfg.dataset.name,
                                       split='test', data_path=cfg.dataset.datapath, config=cfg)
        self.statistics = {'train': self.train_dataset.statistics, 'test': self.test_dataset.statistics, 'val': self.val_dataset.statistics}
        super().__init__(cfg, train_dataset=self.train_dataset, test_dataset=self.test_dataset, val_dataset=self.val_dataset)


class CoWDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.datamodule = datamodule
        self.statistics = datamodule.statistics
        self.name = cfg.dataset.name
        self.directed = cfg.dataset.get("is_directed", False)
        self.num_edge_categories = cfg.dataset.get("num_edge_categories", 14)
        self.num_degree_categories = cfg.dataset.get("num_degree_categories", 1)

        super().complete_infos(datamodule.statistics)

        y_out = 1
        y_in = 2
        self.input_dims = PlaceHolder(X=len(self.atom_types), E=self.edge_types.size(0), y=y_in, pos=3,
                                      directed=self.directed, charges=self.charges_types.size(0))
        self.output_dims = PlaceHolder(X=len(self.atom_types), E=self.edge_types.size(0), y=y_out, pos=3,
                                       directed=self.directed, charges=self.charges_types.size(0))
        self.collapse_charges = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).int()

    def to_one_hot(self, X, charges, E, node_mask):
        E = F.one_hot(E, num_classes=self.edge_types.size(0)).float()
        # X = F.one_hot(X.to(torch.long), num_classes=len(self.atom_types) + 1).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E, y=None, pos=None, directed=self.directed)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E


def normalize_betti_vals(betti_val_dict):
    for betti_number, components in betti_val_dict.items():
        total = sum(components.values())
        if total > 0:
            for component in components:
                components[component] /= total
    return betti_val_dict


def normalize_edge_lengths(all_edge_lengths):
    for bond_type, distances in all_edge_lengths.items():
        total = sum(distances.values())
        if total > 0:
            for d in distances:
                distances[d] /= total
    return all_edge_lengths


def update_edge_angle_info(graph_data, all_edge_angles):
    node_types = torch.argmax(graph_data.x, dim=1)
    for i in range(graph_data.x.size(0)):
        neighbors, _, _, _ = k_hop_subgraph(i, num_hops=1, edge_index=graph_data.edge_index, relabel_nodes=False)
        for j, k in [(j, k) for j in neighbors for k in neighbors if j < k]:
            a, b = graph_data.pos[j] - graph_data.pos[i], graph_data.pos[k] - graph_data.pos[i]
            angle = torch.acos(torch.clamp(torch.dot(a, b) / (a.norm() * b.norm() + 1e-6), -1.0, 1.0))
            bin_idx = int(torch.round(angle * 180 / math.pi, decimals=1) * 10)
            all_edge_angles[node_types[i].item(), bin_idx] += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'march'
    args.raw_data_path = r'E:\PycharmProjects\Data\march_discrete'
    args.datapath = r'./data'

    train_dataset = CoWDataset(dataset_name="march", split='train', data_path=args.raw_data_path, config=args)
    print("train_dataset:", train_dataset)