import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from constants import feature_dict, task_type, dataset_choices
from explainers import *
from gnns import *
from utils.dataset import get_datasets
import torch_geometric.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--root", type=str, default="results/", help="Result directory.")
    parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv'])
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--task", type=str, default="nc")

    parser.add_argument("--train_batchsize", type=int, default=32)
    parser.add_argument("--test_batchsize", type=int, default=32)
    parser.add_argument("--sigma_length", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--feature_in", type=int)
    parser.add_argument("--data_size", type=int, default=-1)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha_cf", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prob_low", type=float, default=0.0)
    parser.add_argument("--prob_high", type=float, default=0.4)
    parser.add_argument("--sparsity_level", type=float, default=2.5)

    parser.add_argument("--normalization", type=str, default="instance")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--layers_per_conv", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--cat_output", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False)
    parser.add_argument("--noise_mlp", type=bool, default=True)
    parser.add_argument("--simplified", type=bool, default=True)

    return parser.parse_args()


args = parse_args()
args.noise_list = None

# 设置可见的 CUDA 设备
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定使用的 GPU
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

args.feature_in = feature_dict[args.dataset]
args.task = task_type[args.dataset]
train_dataset, val_dataset, clean_test_dataset, atk_test_dataset = get_datasets(name=args.dataset)

train_dataset = train_dataset[: args.data_size]
gnn_path = f"param/gnns/{args.dataset}_{args.gnn_type}.pt"
explainer = DiffExplainer(args.device, gnn_path)

# Train DMGNN over train_dataset and evaluate
explainer.explain_graph_task(args, train_dataset, val_dataset)

# Test DMGNN over clean_test_dataset and atk_test_dataset
explainer.test_model(args, clean_test_dataset)
explainer.test_model(args, atk_test_dataset)



