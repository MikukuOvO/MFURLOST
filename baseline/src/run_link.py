import numpy as np
from tqdm import tqdm
import argparse
import torch
from data_loading import get_link_dataset
from utils import get_missing_feature_mask
from filling_strategies import filling
from seeds import seeds
from utils_link import train, test
import random
from torch_geometric.utils import train_test_split_edges
from pcfi import pcfi

from torch_geometric.nn import GAE
from models import GCNEncoder, GCNmfEncoder

parser = argparse.ArgumentParser("Setting for graphs with partially known features")
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
    "--model",
    type=str,
    help="Type of model to make a prediction on the downstream task",
    default="gat",
    choices=["mlp", "sgc", "sage", "gcn", "gat", "gcnmf", "pagnn", "lp", "pcfi"],
)
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="structural", choices=["uniform", "structural"],
)
parser.add_argument(
    "--filling_method",
    type=str,
    help="Method to solve the missing feature problem",
    default="feature_propagation",
    choices=["random", "zero", "mean", "neighborhood_mean", "feature_propagation",],
)
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.995)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=100,
)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=5)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--jk", action="store_true", help="Whether to use the jumping knowledge scheme")

parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)

def run(args):
    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() and not (args.dataset_name == "OGBN-Products" and args.model == "lp")
        else "cpu"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    aucs, aps = [], []

    for seed in tqdm(seeds[: args.n_runs]):

        dataset, evaluator = get_link_dataset(name=args.dataset_name)
        n_nodes, n_features = dataset.data.x.shape
        data = dataset.data
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1).to(device)

        if args.dataset_name == "sailing":
            x = data.x.clone()
            missing_feature_mask = ~torch.isnan(x)
        else:
            missing_feature_mask = get_missing_feature_mask(
                rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, seed = seed, type=args.mask_type,
            ).to(device)
            x = data.x.clone()
            x[~missing_feature_mask] = float("nan")

        if args.model == "pcfi":
            filled_features = pcfi(data.train_pos_edge_index, x, missing_feature_mask, args.num_iterations, args.mask_type, args.alpha, args.beta).to(device)
        elif args.model == "gat":
            filled_features = filling(args.filling_method, data.train_pos_edge_index, x, missing_feature_mask,
                        args.num_iterations, )
        elif args.model == "gcn":
            filled_features = torch.full_like(x, 0)
        elif args.model == "gcnmf":
            filled_features = torch.full_like(x, float("nan"))
        else:
            raise NotImplementedError
            
        data = data.to(device)
        x = torch.where(missing_feature_mask, data.x, filled_features)
        data.train_mask = data.val_mask = data.test_mask = None

        if args.model in ["pcfi", "gat"]:
            model = GAE(GCNEncoder(dataset.num_features, out_channels=16))
        elif args.model == "gcn":
            model = GAE(GCNEncoder(dataset.num_features, out_channels=16))
        elif args.model == "gcnmf":
            model = GAE(GCNmfEncoder(dataset.num_features, 16, args.hidden_dim, x, data.train_pos_edge_index))
        else:
            raise NotImplementedError

        model.to(device)
        train_pos_edge_index = data.train_pos_edge_index
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        epochs = 200
        for epoch in range(1, epochs+1):
            loss = train(model, x, train_pos_edge_index, optimizer)
            auc, ap = test(model, x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
            if epoch % 50 == 0:
                print('epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

        aucs.append(auc)
        aps.append(ap)
    test_auc_mean, test_auc_std = np.mean(aucs), np.std(aucs)
    test_ap_mean, test_ap_std = np.mean(aps), np.std(aps)

    print(f"AUC Accuracy: {test_auc_mean * 100:.2f}% +- {test_auc_std * 100:.2f}")
    print(f"AP Accuracy: {test_ap_mean * 100:.2f}% +- {test_ap_std * 100:.2f}")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    run(args)