"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import logging

logger = logging.getLogger(__name__)


def train(model, x, graph, edge_index, optimizer, critereon):
    model.train()

    return train_full_batch(model, x, graph, edge_index, optimizer, critereon)


def train_full_batch(model, x, graph, edge_index, optimizer, critereon):
    model.train()

    optimizer.zero_grad()
    y_pred = model(x, edge_index)[graph.ndata["train_mask"]]
    y_true = graph.ndata["label"][graph.ndata["train_mask"]].squeeze()

    loss = critereon(y_pred, y_true)
    loss.backward()
    optimizer.step()

    return loss