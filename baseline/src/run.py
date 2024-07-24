"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import numpy as np
from tqdm import tqdm
import argparse
import logging
import time
import yaml

import torch

from data_loading import get_dataset
from utils import get_missing_feature_mask, get_edge_index
from models import get_model
from seeds import seeds
from filling_strategies import filling
from evaluation import test
from train import train
from pcfi import pcfi

parser = argparse.ArgumentParser("GNN-Missing-Features")
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of dataset",
    default="cora",
    choices=[
        "cora",
        "citeseer",
        "pubmed",
        "ogbn-arxiv",
        "reddit",
        "flickr",
        "sailing"
    ],
)
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="uniform", choices=["uniform", "structural"],
)
parser.add_argument(
    "--filling_method",
    type=str,
    help="Method to solve the missing feature problem",
    default="feature_propagation",
    choices=["random", "zero", "mean", "neighborhood_mean", "feature_propagation",],
)
parser.add_argument(
    "--model",
    type=str,
    help="Type of model to make a prediction on the downstream task",
    default="gat",
    choices=["mlp", "sgc", "sage", "gcn", "gat", "gcnmf", "pagnn", "lp", "pcfi"],
)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.9)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.005)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=5)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=2)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=40,
)
parser.add_argument("--lp_alpha", type=float, help="Alpha parameter of label propagation", default=0.9)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
parser.add_argument("--jk", action="store_true", help="Whether to use the jumping knowledge scheme")
parser.add_argument(
    "--batch_size", type=int, help="Batch size for models trained with neighborhood sampling", default=1024,
)
parser.add_argument(
    "--graph_sampling",
    help="Set if you want to use graph sampling (always true for large graphs)",
    action="store_true",
)
parser.add_argument(
    "--homophily", type=float, help="Level of homophily for synthetic datasets", default=None,
)
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument(
    "--log", type=str, help="Log Level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
)

parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=1.0)

parser.add_argument("--zero_fill", type=bool, default=False)
parser.add_argument("--task_name", type=str, default="transductive", choices=["transductive", "inductive"])

parser.add_argument("--missing_edge_rate", type=float, default=0.0)

def run(args):
    logger.info(args)

    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() and not (args.dataset_name == "OGBN-Products" and args.model == "lp")
        else "cpu"
    )
    graph, train_graph, (num_features, num_classes), missing_mask = get_dataset(args, name=args.dataset_name)

    train_edge_index = get_edge_index(train_graph)
    edge_index = get_edge_index(graph)

    train_graph = train_graph.to(device)
    graph = graph.to(device)
    train_edge_index = train_edge_index.to(device)
    edge_index = edge_index.to(device)
    
    test_accs, best_val_accs, train_times = [], [], []

    # for seed in tqdm(seeds[: args.n_runs]):
    for seed in tqdm(range(0, args.n_runs)):
        # print(seed)
        train_n_nodes = len(train_graph.ndata["feat"])
        n_nodes = len(graph.ndata["feat"])

        train_start = time.time()
        if args.model == "lp":
            model = get_model(
                model_name=args.model,
                num_features=num_features,
                num_classes=num_classes,
                edge_index=edge_index,
                x=None,
                args=args,
            ).to(device)
            logger.info("Starting Label Propagation")
            logits = model(y=graph.ndata["label"], edge_index=edge_index, mask=graph.ndata["train_mask"])
            (_, val_acc, test_acc), _ = test(model=None, x=None, graph=graph, edge_index=edge_index, logits=logits)
        else:
            if args.dataset_name == "sailing":
                x = graph.ndata["feat"].clone()
                missing_feature_mask = ~torch.isnan(x)

                train_x = train_graph.ndata["feat"].clone()
                train_missing_feature_mask = missing_feature_mask[missing_mask]
            else:     
                train_missing_feature_mask = get_missing_feature_mask(
                    rate=args.missing_rate, n_nodes=train_n_nodes, n_features=num_features, seed=seed, type=args.mask_type,
                ).to(device)
                train_x = train_graph.ndata["feat"].clone()
                train_x[~train_missing_feature_mask] = float("nan")

                missing_feature_mask = get_missing_feature_mask(
                    rate=args.missing_rate, n_nodes=n_nodes, n_features=num_features, seed=seed, type=args.mask_type,
                ).to(device)
                x = graph.ndata["feat"].clone()
                x[~missing_feature_mask] = float("nan")

            logger.debug("Starting feature filling")

            start = time.time()

            if args.model == "pcfi":
                if args.zero_fill == False:
                    train_filled_features = pcfi(train_edge_index, train_x, train_missing_feature_mask, args.num_iterations, args.mask_type, args.alpha, args.beta)
                    filled_features = pcfi(edge_index, x, missing_feature_mask, args.num_iterations, args.mask_type, args.alpha, args.beta)
                else:
                    train_filled_features = torch.zeros_like(train_x)
                    filled_features = torch.zeros_like(x)
            else:
                if args.zero_fill == False:
                    train_filled_features = (
                        filling(args.filling_method, train_edge_index, train_x, train_missing_feature_mask,
                                args.num_iterations, )
                        if args.model not in ["gcnmf", "pagnn"]
                        else torch.full_like(train_x, float("nan"))
                    )

                    filled_features = (
                        filling(args.filling_method, edge_index, x, missing_feature_mask,
                                args.num_iterations, )
                        if args.model not in ["gcnmf", "pagnn"]
                        else torch.full_like(x, float("nan"))
                    )
                else:
                    train_filled_features = torch.zeros_like(train_x)
                    filled_features = torch.zeros_like(x)
            logger.debug(f"Feature filling completed. It took: {time.time() - start:.2f}s")

            if args.task_name == "transductive":
                train_graph = graph
                train_edge_index = edge_index
                train_missing_feature_mask = missing_feature_mask
                train_x = x
                train_filled_features = filled_features     

            train_graph = train_graph.to(device)
            graph = graph.to(device)
            train_edge_index = train_edge_index.to(device)
            edge_index = edge_index.to(device)
            train_missing_feature_mask = train_missing_feature_mask.to(device)
            missing_feature_mask = missing_feature_mask.to(device)
            train_x = train_x.to(device)
            x = x.to(device)
            train_filled_features = train_filled_features.to(device)
            filled_features = filled_features.to(device)

            model = get_model(
                model_name=args.model,
                num_features=num_features,
                num_classes=num_classes,
                edge_index=edge_index,
                x=x,
                mask=missing_feature_mask,
                args=args,
                train_mask=train_missing_feature_mask
            ).to(device)
            params = list(model.parameters())

            optimizer = torch.optim.Adam(params, lr=args.lr)
            critereon = torch.nn.NLLLoss()

            test_acc = 0
            val_accs = []

            for epoch in tqdm(range(0, args.epochs)):
                start = time.time()
                train_x = torch.where(train_missing_feature_mask, train_x, train_filled_features)
                x = torch.where(missing_feature_mask, x, filled_features)
                train(model, train_x, train_graph, train_edge_index, optimizer, critereon)

                (train_acc, val_acc, tmp_test_acc), out = test(
                    model, x=x, graph=graph, edge_index=edge_index,
                )
                if epoch == 0 or val_acc > max(val_accs):
                    test_acc = tmp_test_acc
                    y_soft = out.softmax(dim=-1)

                val_accs.append(val_acc)
                if epoch > args.patience and max(val_accs[-args.patience :]) <= max(val_accs[: -args.patience]):
                    break
                logger.debug(
                    f"Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}. It took {time.time() - start:.2f}s"
                )

            (_, val_acc, test_acc), _ = test(model, x=x, graph=graph, edge_index=edge_index, logits=y_soft)
        best_val_accs.append(val_acc)
        test_accs.append(test_acc)
        print(f"Get test_acc {test_acc} for one epoch!")
        train_times.append(time.time() - train_start)

    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    print(f"Test Accuracy: {test_acc_mean * 100:.2f}% +- {test_acc_std * 100:.2f}")


if __name__ == "__main__":
    args = parser.parse_args()
    with open("hyperparameters.yaml", "r") as f:
        hyperparams = yaml.safe_load(f)
        dataset = args.dataset_name
        if dataset in hyperparams:
            for k, v in hyperparams[dataset].items():
                setattr(args, k, v)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=getattr(logging, args.log.upper(), None))

    run(args)
