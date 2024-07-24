"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch

@torch.no_grad()
def test(model, x, graph, edge_index, logits=None):
    if logits is None:
        model.eval()
        logits = inference_full_batch(model, x, edge_index)

    accs = []

    for mask in [graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"]]:
        pred = logits[mask].max(dim=1)[1]  # Get the predicted classes
        acc = pred.eq(graph.ndata["label"][mask]).sum().item() / mask.sum().item()  # Compute accuracy
        accs.append(acc)
    
    return accs, logits


def inference_full_batch(model, x, edge_index):
    out = model(x, edge_index)
    return out
