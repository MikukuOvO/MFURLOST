"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import dgl
import scipy.sparse as sp
import torch
import numpy as np
import json
import os
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, Reddit2, Flickr
from torch_geometric.utils import to_undirected, add_remaining_self_loops, from_scipy_sparse_matrix
from dgl.data import (
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
}
DATA_PATH = "dataset"

class InductiveDataset(Data):
    def __init__(self, data, num_classes):
        super(InductiveDataset, self).__init__()
        self.data = data
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(transform, pre_transform)
        self.data = data

    def len(self):
        return 1

    def get(self, idx):
        return self.data

def get_inductive_dataset(dataset_name):
    adj_full = sp.load_npz(f'dataset/{dataset_name}/adj_full.npz')
    edge_index1, edge_index2 = adj_full.nonzero()
    g = dgl.graph((edge_index1, edge_index2))
    raw_feat = np.load(f'dataset/{dataset_name}/feats.npy')
    g.ndata["feat"] = torch.from_numpy(raw_feat).float()
    with open(f'dataset/{dataset_name}/class_map.json', 'r') as f:
        class_label = json.load(f)
    node_labels = torch.tensor([class_label[str(i)] for i in range(g.number_of_nodes())])
    g.ndata["label"] = node_labels

    with open(f'dataset/{dataset_name}/role.json', 'r') as f:
        role = json.load(f)
    train_indices = role['tr']
    val_indices = role['va']
    test_indices = role['te']

    train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(train_indices), True)
    val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(val_indices), True)
    test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).scatter_(0, torch.tensor(test_indices), True)

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    g.num_classes = node_labels.max().item() + 1

    return g

def missing_edges(graph, ratio):
    num_edges = graph.number_of_edges()
    num_edges_to_remove = int(num_edges * ratio)
    all_edges = graph.edges()
    edges_to_remove = np.random.choice(num_edges, num_edges_to_remove, replace=False)
    u, v = all_edges[0][edges_to_remove], all_edges[1][edges_to_remove]
    graph.remove_edges(edges_to_remove)
    return graph

# train_g 和 g 的 mask
def get_dataset(args, name: str):
    if name in ["cora", "citeseer", "pubmed"]:
        dataset = GRAPH_DICT[name]()
        graph = dataset[0]
        num_classes = dataset.num_classes

        ### Adding edge missing
        graph = missing_edges(graph, args.missing_edge_rate)

        train_graph = graph
        missing_mask = torch.ones(len(graph.ndata["feat"]), dtype=torch.bool)
    elif name == "ogbn-arxiv":
        dataset = GRAPH_DICT[name](name)
        graph, node_labels = dataset[0]
        graph = dgl.add_reverse_edges(graph)

        ### Adding edge missing
        graph = missing_edges(graph, args.missing_edge_rate)

        num_nodes = graph.num_nodes()

        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']

        graph.ndata["train_mask"] = torch.full((num_nodes,), False).index_fill_(0, train_nids, True)
        graph.ndata["val_mask"] = torch.full((num_nodes,), False).index_fill_(0, valid_nids, True)
        graph.ndata["test_mask"] = torch.full((num_nodes,), False).index_fill_(0, test_nids, True)
        graph.ndata["label"] = node_labels[:, 0]
        num_classes = (node_labels.max() + 1).item()
        train_graph = graph
        missing_mask = torch.ones(len(graph.ndata["feat"]), dtype=torch.bool)
    elif name in ["reddit", "flickr", "sailing"]:
        graph = get_inductive_dataset(name)

        ### Adding edge missing
        graph = missing_edges(graph, args.missing_edge_rate)

        num_classes = graph.num_classes
        train_mask = graph.ndata['train_mask']
        train_graph = dgl.node_subgraph(graph, train_mask)
        missing_mask = graph.ndata["train_mask"]
    else:
        raise Exception("Unknown dataset.")
    
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    train_graph = train_graph.remove_self_loop()
    train_graph = train_graph.add_self_loop()

    num_features = graph.ndata["feat"].shape[1]

    return graph, train_graph, (num_features, num_classes), missing_mask

def get_link(dataset_name):
    # Load adjacency matrix
    adj_full = sp.load_npz(f'dataset/{dataset_name}/adj_full.npz')
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_full)
    
    # Load node features
    raw_feat = np.load(f'dataset/{dataset_name}/feats.npy')
    x = torch.from_numpy(raw_feat).float()
    
    # Load class labels
    with open(f'dataset/{dataset_name}/class_map.json', 'r') as f:
        class_label = json.load(f)
    y = torch.tensor([class_label[str(i)] for i in range(x.size(0))])
    
    # Load train/val/test splits
    with open(f'dataset/{dataset_name}/role.json', 'r') as f:
        role = json.load(f)
    train_indices = torch.tensor(role['tr'])
    val_indices = torch.tensor(role['va'])
    test_indices = torch.tensor(role['te'])
    
    # Create masks
    train_mask = torch.zeros(x.size(0), dtype=torch.bool).scatter_(0, train_indices, True)
    val_mask = torch.zeros(x.size(0), dtype=torch.bool).scatter_(0, val_indices, True)
    test_mask = torch.zeros(x.size(0), dtype=torch.bool).scatter_(0, test_indices, True)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_classes = y.max().item() + 1
    
    return data

def get_link_dataset(name: str):
    path = os.path.join(DATA_PATH, name)
    evaluator = None

    if name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(path, name)
    elif name in ["flickr", "reddit", "sailing"]:
        data = get_link(name)
        dataset = CustomDataset(data)
    else:
        raise Exception("Unknown dataset.")

    # Make graph undirected so that we have edges for both directions and add self loops
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index, num_nodes=dataset.data.x.shape[0])

    return dataset, evaluator