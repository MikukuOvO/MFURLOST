"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch

from utils import get_symmetrically_normalized_adjacency


class PaGNNConv(torch.nn.Module):
    def __init__(self, in_features, out_features, mask, edge_index, train_mask):
        super(PaGNNConv, self).__init__()
        self.lin = torch.nn.Linear(in_features, out_features)
        self.mask = mask.float()
        self.train_mask = train_mask.float()
        # edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, mask.shape[0])
        # self.adj = torch.sparse.FloatTensor(edge_index, values=edge_weight).to(edge_index.device)

    def forward(self, x, edge_index):
        x[x.isnan()] = 0
        if len(x) == len(self.train_mask):
            edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, self.train_mask.shape[0])
            newadj = torch.sparse.FloatTensor(edge_index, values=edge_weight).to(edge_index.device)
            numerator = torch.sparse.mm(newadj, torch.ones_like(x)) * torch.sparse.mm(newadj, self.train_mask * x)
            denominator = torch.sparse.mm(newadj, self.train_mask)
        else:
            edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, self.mask.shape[0])
            newadj = torch.sparse.FloatTensor(edge_index, values=edge_weight).to(edge_index.device)
            numerator = torch.sparse.mm(newadj, torch.ones_like(x)) * torch.sparse.mm(newadj, self.mask * x)
            denominator = torch.sparse.mm(newadj, self.mask)
        ratio = torch.nan_to_num(numerator / denominator)
        x = self.lin(ratio)

        return x
